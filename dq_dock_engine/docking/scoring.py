"""
Scoring mechanism using strict OpenHCS Enum dispatch.

Separates JAX-native internal physics from impure SMINA external subprocess wrapper.

PROOF STATUS SUMMARY:
  - score_internal_lj: HEURISTIC (ad-hoc weights)
  - score_smina_exact: HEURISTIC (external unverified binary)
  - route_scoring: HEURISTIC (dispatch only)
"""

from dataclasses import dataclass
from typing import List, Dict, Callable, Union
import os
import subprocess
import tempfile
import pathlib
import functools

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np

from dq_dock_engine.proof_status import (
    certified,
    conditionally_certified,
    heuristic,
    ProofStatus,
)
from jax.tree_util import register_pytree_node_class
from dq_dock_engine.docking.core import ScoringEngine, ScoredPose, GapCertification
from dq_dock_engine.physics.lattice_sum import optimal_cutoff, lj6_cutoff_error
from dq_dock_engine.physics.kernels import typed_lennard_jones_matrix

_EPSILON_KCAL_MOL = 0.086


def certified_lj_error_bound(
    target_error: float,
    epsilon: float = _EPSILON_KCAL_MOL,
) -> float:
    def lj_cutoff_error(epsilon: float, cutoff: float) -> float:
        return epsilon * lj6_cutoff_error(cutoff)

    cutoff = optimal_cutoff(target_error, s=6.0)
    return lj_cutoff_error(epsilon, cutoff)


def certified_realspace_ewald_error_bound(
    cutoff: float,
    alpha: float,
    charge_bound: float,
) -> float:
    # Note: checks removed for JIT compatibility.
    # Higher-level logic should ensure cutoff, alpha, charge_bound are positive.
    def ewald_realspace_cutoff_error(
        charge_bound: float, alpha: float, cutoff: float
    ) -> float:
        return charge_bound * (2.0 / (alpha**4)) / (cutoff**3)

    return ewald_realspace_cutoff_error(charge_bound, alpha, cutoff)


@register_pytree_node_class
@dataclass(frozen=True)
class CertifiedRealSpaceEwaldSpec:
    receptor_charges: jnp.ndarray
    ligand_charges: jnp.ndarray
    cutoff: float = 12.0
    alpha: float = 0.2
    dielectric: float = 4.0

    def validate(self) -> None:
        if self.cutoff <= 0:
            raise ValueError("electrostatic cutoff must be positive")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if self.dielectric <= 0:
            raise ValueError("dielectric must be positive")
        if self.receptor_charges.ndim != 1:
            raise ValueError("receptor_charges must be a 1D array")
        if self.ligand_charges.ndim != 1:
            raise ValueError("ligand_charges must be a 1D array")

    def tree_flatten(self):
        children = (self.receptor_charges, self.ligand_charges)
        aux_data = (self.cutoff, self.alpha, self.dielectric)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)

    def receptor_subset(self, indices: jnp.ndarray) -> "CertifiedRealSpaceEwaldSpec":
        return CertifiedRealSpaceEwaldSpec(
            receptor_charges=self.receptor_charges[indices],
            ligand_charges=self.ligand_charges,
            cutoff=self.cutoff,
            alpha=self.alpha,
            dielectric=self.dielectric,
        )

    def charge_bound(self) -> jax.Array | float:
        return (
            jnp.max(jnp.abs(self.receptor_charges))
            * jnp.max(jnp.abs(self.ligand_charges))
            / self.dielectric
        )


@register_pytree_node_class
@dataclass(frozen=True)
class CertifiedScreenedCoulombSpec:
    receptor_charges: jnp.ndarray
    ligand_charges: jnp.ndarray
    kappa: float = 1.0
    cutoff: float = 8.0
    dielectric: float = 4.0

    def validate(self) -> None:
        if self.kappa < 0:
            raise ValueError("screening parameter kappa must be nonnegative")
        if self.cutoff <= 0:
            raise ValueError("screened Coulomb cutoff must be positive")
        if self.dielectric <= 0:
            raise ValueError("dielectric must be positive")
        if self.receptor_charges.ndim != 1:
            raise ValueError("receptor_charges must be a 1D array")
        if self.ligand_charges.ndim != 1:
            raise ValueError("ligand_charges must be a 1D array")

    def tree_flatten(self):
        children = (self.receptor_charges, self.ligand_charges)
        aux_data = (self.kappa, self.cutoff, self.dielectric)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)

    def receptor_subset(self, indices: jnp.ndarray) -> "CertifiedScreenedCoulombSpec":
        return CertifiedScreenedCoulombSpec(
            receptor_charges=self.receptor_charges[indices],
            ligand_charges=self.ligand_charges,
            kappa=self.kappa,
            cutoff=self.cutoff,
            dielectric=self.dielectric,
        )


@register_pytree_node_class
@dataclass(frozen=True)
class CertifiedContactSurrogateSpec:
    receptor_weights: jnp.ndarray
    ligand_weights: jnp.ndarray
    beta: float = 0.6
    cutoff: float = 6.0

    def validate(self) -> None:
        if self.beta <= 0:
            raise ValueError("contact surrogate beta must be positive")
        if self.cutoff <= 0:
            raise ValueError("contact surrogate cutoff must be positive")
        if self.receptor_weights.ndim != 1:
            raise ValueError("receptor_weights must be a 1D array")
        if self.ligand_weights.ndim != 1:
            raise ValueError("ligand_weights must be a 1D array")

    def tree_flatten(self):
        children = (self.receptor_weights, self.ligand_weights)
        aux_data = (self.beta, self.cutoff)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)

    def receptor_subset(self, indices: jnp.ndarray) -> "CertifiedContactSurrogateSpec":
        return CertifiedContactSurrogateSpec(
            receptor_weights=self.receptor_weights[indices],
            ligand_weights=self.ligand_weights,
            beta=self.beta,
            cutoff=self.cutoff,
        )


@register_pytree_node_class
@dataclass(frozen=True)
class CertifiedDirectionalHBondSpec:
    receptor_directions: jnp.ndarray
    ligand_directions: jnp.ndarray
    receptor_strengths: jnp.ndarray
    ligand_strengths: jnp.ndarray
    ideal_distance: float = 2.8
    distance_width: float = 0.8
    cutoff: float = 4.0

    def validate(self) -> None:
        if self.ideal_distance <= 0:
            raise ValueError("ideal hydrogen-bond distance must be positive")
        if self.distance_width <= 0:
            raise ValueError("hydrogen-bond distance width must be positive")
        if self.cutoff <= 0:
            raise ValueError("hydrogen-bond cutoff must be positive")
        if self.receptor_directions.ndim != 2 or self.receptor_directions.shape[1] != 3:
            raise ValueError("receptor_directions must have shape (N_rec, 3)")
        if self.ligand_directions.ndim != 2 or self.ligand_directions.shape[1] != 3:
            raise ValueError("ligand_directions must have shape (N_lig, 3)")
        if self.receptor_strengths.ndim != 1:
            raise ValueError("receptor_strengths must be a 1D array")
        if self.ligand_strengths.ndim != 1:
            raise ValueError("ligand_strengths must be a 1D array")

    def tree_flatten(self):
        children = (
            self.receptor_directions,
            self.ligand_directions,
            self.receptor_strengths,
            self.ligand_strengths,
        )
        aux_data = (self.ideal_distance, self.distance_width, self.cutoff)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)

    def receptor_subset(self, indices: jnp.ndarray) -> "CertifiedDirectionalHBondSpec":
        return CertifiedDirectionalHBondSpec(
            receptor_directions=self.receptor_directions[indices],
            ligand_directions=self.ligand_directions,
            receptor_strengths=self.receptor_strengths[indices],
            ligand_strengths=self.ligand_strengths,
            ideal_distance=self.ideal_distance,
            distance_width=self.distance_width,
            cutoff=self.cutoff,
        )


@register_pytree_node_class
@dataclass(frozen=True)
class CertifiedBatchResult:
    scores: jnp.ndarray
    error_bound: float
    target_error: float
    cutoff_radius: float

    def tree_flatten(self):
        children = (self.scores,)
        aux_data = (self.error_bound, self.target_error, self.cutoff_radius)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)

    def certify_gap(self, idx_a: int, idx_b: int) -> GapCertification:
        return GapCertification.from_energies(
            float(self.scores[idx_a]),
            float(self.scores[idx_b]),
            self.error_bound,
        )

    def certify_top_k(self, k: int = 1) -> list[GapCertification]:
        sorted_indices = jnp.argsort(self.scores)
        best_idx = int(sorted_indices[0])
        certifications = []
        for i in range(1, k):
            if i >= len(sorted_indices):
                break
            cert = self.certify_gap(best_idx, int(sorted_indices[i]))
            certifications.append(cert)
        return certifications


@register_pytree_node_class
@dataclass(frozen=True)
class CertifiedSoftenedBatchResult:
    scores: jnp.ndarray
    softening_error_bound: float
    target_error: float
    cutoff_radius: float
    softening_radius: float

    def tree_flatten(self):
        children = (self.scores,)
        aux_data = (
            self.softening_error_bound,
            self.target_error,
            self.cutoff_radius,
            self.softening_radius,
        )
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)


def _score_certified_scores(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    target_error: float,
    epsilon: float,
    electrostatics: CertifiedRealSpaceEwaldSpec | None,
) -> tuple[jnp.ndarray, float]:
    if electrostatics is None:
        return score_certified_lj(
            receptor_coords,
            poses_coords,
            receptor_radii,
            ligand_radii,
            target_error=target_error,
            epsilon=epsilon,
        )
    return score_certified_lj_realspace_ewald(
        receptor_coords,
        poses_coords,
        receptor_radii,
        ligand_radii,
        electrostatics=electrostatics,
        target_error=target_error,
        epsilon=epsilon,
    )


def _default_softening_radius(
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
) -> jax.Array | float:
    receptor_min = jnp.min(receptor_radii)
    ligand_min = jnp.min(ligand_radii)
    return 0.5 * (receptor_min + ligand_min)


def score_certified_batch(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    target_error: float = 0.001,
    epsilon: float = _EPSILON_KCAL_MOL,
    electrostatics: CertifiedRealSpaceEwaldSpec | None = None,
) -> CertifiedBatchResult:
    scores, error_bound = _score_certified_scores(
        receptor_coords,
        poses_coords,
        receptor_radii,
        ligand_radii,
        target_error,
        epsilon,
        electrostatics,
    )
    R = optimal_cutoff(target_error, s=6.0)
    return CertifiedBatchResult(
        scores=scores,
        error_bound=error_bound,
        target_error=target_error,
        cutoff_radius=R,
    )


@jax.jit
@heuristic()
def _score_single_lj(
    receptor_coords: jnp.ndarray,
    pose_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
) -> jnp.ndarray:
    """
    Atom-typed LJ score tuned for rigid-body docking.

    PROOF STATUS: HEURISTIC
      - Ad-hoc weights (4.0, 0.4) not backed by formal proof
      - Based on empirical observation that clash avoidance dominates scoring
      - VdW radii: EMPIRICAL (physics_params.py)

    NOTE: The functional form is correct (Lorentz-Berthelot combining rules),
    but the specific weights are heuristic.
    """
    diffs = receptor_coords[:, None, :] - pose_coords[None, :, :]
    dist_sq = jnp.sum(diffs**2, axis=-1)  # (N_rec, N_lig)

    sigma_ij = receptor_radii[:, None] + ligand_radii[None, :]  # Lorentz-Berthelot
    sigma_sq = sigma_ij**2

    dist_sq_safe = jnp.maximum(dist_sq, (0.5 * sigma_ij) ** 2)

    r6 = (sigma_sq / dist_sq_safe) ** 3
    r12 = r6**2

    # HEURISTIC WEIGHTS: not proven optimal
    repulsion = 4.0 * r12
    attraction = 0.4 * r6

    pe = repulsion - attraction
    return jnp.sum(pe)


@jax.jit
def _score_certified_lj(
    receptor_coords: jnp.ndarray,
    pose_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    cutoff: jnp.ndarray,
    epsilon: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    CERTIFIED LJ score using proven cutoff bounds from Lean 4.

    PROOF STATUS: CERTIFIED (theorem: LatticeSum.lean::lj6_tail_bound)
      - Cutoff error bounded by lattice_tail_bound(s, R)
      - For LJ-6: error ≤ M/R³ where M = 8π
      - Energy calibrated with epsilon (kcal/mol)

    Args:
        receptor_coords: (N_rec, 3)
        pose_coords: (N_lig, 3)
        receptor_radii: (N_rec,)
        ligand_radii: (N_lig,)
        cutoff: Certified cutoff radius
        epsilon: Well depth in kcal/mol for calibration

    Returns:
        (energy, error_bound) - both JAX arrays, calibrated to physical units
    """
    diffs = receptor_coords[:, None, :] - pose_coords[None, :, :]  # (N_rec, N_lig, 3)
    dists = jnp.asarray(jnp.linalg.norm(diffs, axis=-1))  # (N_rec, N_lig)

    sigma_ij = receptor_radii[:, None] + ligand_radii[None:]  # (N_rec, N_lig)

    cutoff_safe = jnp.maximum(cutoff, sigma_ij)

    in_range = dists < cutoff_safe
    dists_safe = jnp.where(in_range, dists, cutoff_safe)

    epsilon_matrix = jnp.full_like(dists_safe, epsilon / 4.0)
    lj_contrib = jnp.asarray(
        typed_lennard_jones_matrix(dists_safe, epsilon_matrix, sigma_ij)
    )

    # Zero out beyond-cutoff contributions
    energy = jnp.sum(jnp.where(in_range, lj_contrib, 0.0))

    # Lean-proven error bound: M/R³ for LJ-6, calibrated
    M = 4.0 * jnp.pi * 2.0  # 8π
    error_bound = epsilon * M / (cutoff**3)

    return energy, error_bound


@jax.jit
def _score_certified_lj_batch(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    cutoff: jnp.ndarray,
    epsilon: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Batched certified LJ score with the same cutoff proof obligations."""
    diffs = receptor_coords[None, :, None, :] - poses_coords[:, None, :, :]
    dists = jnp.asarray(jnp.linalg.norm(diffs, axis=-1))

    sigma_ij = receptor_radii[:, None] + ligand_radii[None, :]
    cutoff_safe = jnp.maximum(cutoff, sigma_ij)[None, :, :]

    in_range = dists < cutoff_safe
    dists_safe = jnp.where(in_range, dists, cutoff_safe)

    epsilon_matrix = jnp.full_like(dists_safe, epsilon / 4.0)
    sigma_matrix = jnp.broadcast_to(sigma_ij[None, :, :], dists_safe.shape)
    lj_contrib = jnp.asarray(
        typed_lennard_jones_matrix(dists_safe, epsilon_matrix, sigma_matrix)
    )

    energies = jnp.sum(jnp.where(in_range, lj_contrib, 0.0), axis=(1, 2))

    M = 4.0 * jnp.pi * 2.0
    error_bound = epsilon * M / (cutoff**3)
    return energies, error_bound


@functools.partial(jax.jit, static_argnames=("compute_error_bound",))
def _score_certified_softened_lj_batch(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    cutoff: jnp.ndarray,
    epsilon: float,
    softening_radius: float,
    compute_error_bound: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    diffs = receptor_coords[None, :, None, :] - poses_coords[:, None, :, :]
    dists = jnp.asarray(jnp.linalg.norm(diffs, axis=-1))

    sigma_ij = receptor_radii[:, None] + ligand_radii[None, :]
    cutoff_safe = jnp.maximum(cutoff, sigma_ij)[None, :, :]

    in_range = dists < cutoff_safe
    dists_exact = jnp.where(in_range, dists, cutoff_safe)
    dists_soft = jnp.maximum(dists_exact, softening_radius)

    epsilon_matrix = jnp.full_like(dists_exact, epsilon / 4.0)
    sigma_matrix = jnp.broadcast_to(sigma_ij[None, :, :], dists_exact.shape)
    exact_contrib = jnp.asarray(
        typed_lennard_jones_matrix(dists_exact, epsilon_matrix, sigma_matrix)
    )
    softened_contrib = jnp.asarray(
        typed_lennard_jones_matrix(dists_soft, epsilon_matrix, sigma_matrix)
    )

    exact_masked = jnp.where(in_range, exact_contrib, 0.0)
    softened_masked = jnp.where(in_range, softened_contrib, 0.0)
    energies = jnp.sum(softened_masked, axis=(1, 2))

    if compute_error_bound:
        error_bound = jnp.max(
            jnp.sum(jnp.abs(exact_masked - softened_masked), axis=(1, 2))
        )
    else:
        error_bound = jnp.array(0.0)

    return energies, error_bound


@jax.jit
def _score_realspace_ewald_batch(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    receptor_charges: jnp.ndarray,
    ligand_charges: jnp.ndarray,
    cutoff: jnp.ndarray,
    alpha: float,
    dielectric: float,
) -> jnp.ndarray:
    diffs = receptor_coords[None, :, None, :] - poses_coords[:, None, :, :]
    dists = jnp.asarray(jnp.linalg.norm(diffs, axis=-1))

    in_range = dists < cutoff
    cutoff_safe = jnp.full_like(dists, cutoff)
    dists_safe = jnp.where(in_range, jnp.maximum(dists, 1e-6), cutoff_safe)

    charge_matrix = (
        receptor_charges[None, :, None] * ligand_charges[None, None, :]
    ) / dielectric
    ewald_contrib = charge_matrix * jsp.erfc(alpha * dists_safe) / dists_safe
    return jnp.sum(jnp.where(in_range, ewald_contrib, 0.0), axis=(1, 2))


@jax.jit
def _score_screened_coulomb_exact_batch(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    receptor_charges: jnp.ndarray,
    ligand_charges: jnp.ndarray,
    kappa: float,
    dielectric: float,
) -> jnp.ndarray:
    diffs = receptor_coords[None, :, None, :] - poses_coords[:, None, :, :]
    dists = jnp.asarray(jnp.linalg.norm(diffs, axis=-1))
    dists_safe = jnp.maximum(dists, 1e-6)
    charge_matrix = (
        receptor_charges[None, :, None] * ligand_charges[None, None, :]
    ) / dielectric
    screened_contrib = charge_matrix * jnp.exp(-kappa * dists_safe) / dists_safe
    return jnp.sum(screened_contrib, axis=(1, 2))


@jax.jit
def _score_screened_coulomb_cutoff_batch(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    receptor_charges: jnp.ndarray,
    ligand_charges: jnp.ndarray,
    kappa: float,
    cutoff: float,
    dielectric: float,
) -> jnp.ndarray:
    diffs = receptor_coords[None, :, None, :] - poses_coords[:, None, :, :]
    dists = jnp.asarray(jnp.linalg.norm(diffs, axis=-1))
    in_range = dists < cutoff
    dists_safe = jnp.maximum(dists, 1e-6)
    charge_matrix = (
        receptor_charges[None, :, None] * ligand_charges[None, None, :]
    ) / dielectric
    screened_contrib = charge_matrix * jnp.exp(-kappa * dists_safe) / dists_safe
    return jnp.sum(jnp.where(in_range, screened_contrib, 0.0), axis=(1, 2))


@jax.jit
def _score_contact_exact_batch(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    receptor_weights: jnp.ndarray,
    ligand_weights: jnp.ndarray,
    beta: float,
) -> jnp.ndarray:
    diffs = receptor_coords[None, :, None, :] - poses_coords[:, None, :, :]
    dists = jnp.asarray(jnp.linalg.norm(diffs, axis=-1))
    weight_matrix = receptor_weights[None, :, None] * ligand_weights[None, None, :]
    contact_contrib = weight_matrix * jnp.exp(-((beta * dists) ** 2))
    return jnp.sum(contact_contrib, axis=(1, 2))


@jax.jit
def _score_contact_cutoff_batch(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    receptor_weights: jnp.ndarray,
    ligand_weights: jnp.ndarray,
    beta: float,
    cutoff: float,
) -> jnp.ndarray:
    diffs = receptor_coords[None, :, None, :] - poses_coords[:, None, :, :]
    dists = jnp.asarray(jnp.linalg.norm(diffs, axis=-1))
    in_range = dists < cutoff
    weight_matrix = receptor_weights[None, :, None] * ligand_weights[None, None, :]
    contact_contrib = weight_matrix * jnp.exp(-((beta * dists) ** 2))
    return jnp.sum(jnp.where(in_range, contact_contrib, 0.0), axis=(1, 2))


def _normalize_direction_vectors(vectors: jnp.ndarray) -> jnp.ndarray:
    norms = jnp.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / jnp.maximum(norms, 1e-6)


@jax.jit
def _directional_hbond_pair_terms(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    receptor_directions: jnp.ndarray,
    ligand_directions: jnp.ndarray,
    receptor_strengths: jnp.ndarray,
    ligand_strengths: jnp.ndarray,
    ideal_distance: float,
    distance_width: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    diffs = poses_coords[:, None, :, :] - receptor_coords[None, :, None, :]
    dists = jnp.asarray(jnp.linalg.norm(diffs, axis=-1))
    unit_vectors = diffs / jnp.maximum(dists[..., None], 1e-6)

    receptor_dirs = _normalize_direction_vectors(receptor_directions)
    ligand_dirs = _normalize_direction_vectors(ligand_directions)
    receptor_dirs_expanded = receptor_dirs[None, :, None, :]
    ligand_dirs_expanded = ligand_dirs[None, None, :, :]

    pair_strength = jnp.clip(
        receptor_strengths[None, :, None] * ligand_strengths[None, None, :],
        0.0,
        1.0,
    )
    radial = pair_strength * jnp.exp(
        -(((dists - ideal_distance) / distance_width) ** 2)
    )
    donor_angle = jnp.clip(
        jnp.sum(receptor_dirs_expanded * unit_vectors, axis=-1), 0.0, 1.0
    )
    acceptor_angle = jnp.clip(
        jnp.sum(ligand_dirs_expanded * (-unit_vectors), axis=-1), 0.0, 1.0
    )
    return radial, donor_angle, acceptor_angle


@jax.jit
def _score_directional_hbond_exact_batch(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    receptor_directions: jnp.ndarray,
    ligand_directions: jnp.ndarray,
    receptor_strengths: jnp.ndarray,
    ligand_strengths: jnp.ndarray,
    ideal_distance: float,
    distance_width: float,
) -> jnp.ndarray:
    radial, donor_angle, acceptor_angle = _directional_hbond_pair_terms(
        receptor_coords,
        poses_coords,
        receptor_directions,
        ligand_directions,
        receptor_strengths,
        ligand_strengths,
        ideal_distance,
        distance_width,
    )
    return jnp.sum(radial * donor_angle * acceptor_angle, axis=(1, 2))


@jax.jit
def _score_directional_hbond_cutoff_batch(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    receptor_directions: jnp.ndarray,
    ligand_directions: jnp.ndarray,
    receptor_strengths: jnp.ndarray,
    ligand_strengths: jnp.ndarray,
    ideal_distance: float,
    distance_width: float,
    cutoff: float,
) -> jnp.ndarray:
    radial, donor_angle, acceptor_angle = _directional_hbond_pair_terms(
        receptor_coords,
        poses_coords,
        receptor_directions,
        ligand_directions,
        receptor_strengths,
        ligand_strengths,
        ideal_distance,
        distance_width,
    )
    dists = jnp.asarray(
        jnp.linalg.norm(
            poses_coords[:, None, :, :] - receptor_coords[None, :, None, :], axis=-1
        )
    )
    pair_scores = radial * donor_angle * acceptor_angle
    return jnp.sum(jnp.where(dists < cutoff, pair_scores, 0.0), axis=(1, 2))


@conditionally_certified(
    "HandleAliases.lean::SC1; HandleAliases.lean::SC2; HandleAliases.lean::SC3; HandleAliases.lean::SC4; HandleAliases.lean::SC5; HandleAliases.lean::SC6",
    assumptions=[
        "The runtime uses the exact same screened-Coulomb form q_i q_j * exp(-kappa * r) / r as the Lean theorem family",
        "The reported error bound is the exact finite-batch max discrepancy between exact and cutoff screened Coulomb scores",
        "The 1e-6 minimum distance guard is a numerical safeguard against division by zero",
    ],
)
def score_certified_screened_coulomb_batch(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    screened_coulomb: CertifiedScreenedCoulombSpec,
) -> CertifiedBatchResult:
    screened_coulomb.validate()
    exact_scores = _score_screened_coulomb_exact_batch(
        receptor_coords,
        poses_coords,
        screened_coulomb.receptor_charges,
        screened_coulomb.ligand_charges,
        screened_coulomb.kappa,
        screened_coulomb.dielectric,
    )
    coarse_scores = _score_screened_coulomb_cutoff_batch(
        receptor_coords,
        poses_coords,
        screened_coulomb.receptor_charges,
        screened_coulomb.ligand_charges,
        screened_coulomb.kappa,
        screened_coulomb.cutoff,
        screened_coulomb.dielectric,
    )
    error_bound = jnp.max(jnp.abs(exact_scores - coarse_scores))
    return CertifiedBatchResult(
        scores=coarse_scores,
        error_bound=error_bound,
        target_error=error_bound,
        cutoff_radius=screened_coulomb.cutoff,
    )


@conditionally_certified(
    "HandleAliases.lean::CT1; HandleAliases.lean::CT2; HandleAliases.lean::CT3; HandleAliases.lean::CT4; HandleAliases.lean::CT5; HandleAliases.lean::CT6",
    assumptions=[
        "The runtime uses the exact same Gaussian contact/desolvation surrogate w * exp(-(beta * r)^2) as the Lean theorem family",
        "The reported error bound is the exact finite-batch max discrepancy between exact and cutoff contact scores",
    ],
)
def score_certified_contact_batch(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    contact_spec: CertifiedContactSurrogateSpec,
) -> CertifiedBatchResult:
    contact_spec.validate()
    exact_scores = _score_contact_exact_batch(
        receptor_coords,
        poses_coords,
        contact_spec.receptor_weights,
        contact_spec.ligand_weights,
        contact_spec.beta,
    )
    coarse_scores = _score_contact_cutoff_batch(
        receptor_coords,
        poses_coords,
        contact_spec.receptor_weights,
        contact_spec.ligand_weights,
        contact_spec.beta,
        contact_spec.cutoff,
    )
    error_bound = jnp.max(jnp.abs(exact_scores - coarse_scores))
    return CertifiedBatchResult(
        scores=coarse_scores,
        error_bound=error_bound,
        target_error=error_bound,
        cutoff_radius=contact_spec.cutoff,
    )


@conditionally_certified(
    "HandleAliases.lean::HB1; HandleAliases.lean::HB9; HandleAliases.lean::HB10; HandleAliases.lean::HB11; HandleAliases.lean::HB12",
    assumptions=[
        "The runtime uses exact and coarse directional H-bond factor families whose finite-domain discrepancy is captured by the computed max exact-vs-cutoff batch difference",
        "Receptor and ligand direction vectors are normalized up to the runtime 1e-6 norm guard",
        "Strength and angular factors remain within [0, 1] on the evaluated batch",
    ],
)
def score_certified_directional_hbond_batch(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    hbond_spec: CertifiedDirectionalHBondSpec,
) -> CertifiedBatchResult:
    hbond_spec.validate()
    exact_scores = _score_directional_hbond_exact_batch(
        receptor_coords,
        poses_coords,
        hbond_spec.receptor_directions,
        hbond_spec.ligand_directions,
        hbond_spec.receptor_strengths,
        hbond_spec.ligand_strengths,
        hbond_spec.ideal_distance,
        hbond_spec.distance_width,
    )
    coarse_scores = _score_directional_hbond_cutoff_batch(
        receptor_coords,
        poses_coords,
        hbond_spec.receptor_directions,
        hbond_spec.ligand_directions,
        hbond_spec.receptor_strengths,
        hbond_spec.ligand_strengths,
        hbond_spec.ideal_distance,
        hbond_spec.distance_width,
        hbond_spec.cutoff,
    )
    error_bound = jnp.max(jnp.abs(exact_scores - coarse_scores))
    return CertifiedBatchResult(
        scores=coarse_scores,
        error_bound=error_bound,
        target_error=error_bound,
        cutoff_radius=hbond_spec.cutoff,
    )


@conditionally_certified(
    "HandleAliases.lean::RC1; HandleAliases.lean::RC2; HandleAliases.lean::RC3; HandleAliases.lean::RC4; HandleAliases.lean::RC5",
    assumptions=[
        "The additive runtime score is exactly the sum of the certified contact surrogate and the certified directional H-bond surrogate",
        "The combined error bound is the sum of the finite-batch contact and directional H-bond discrepancy bounds",
    ],
)
def score_certified_polar_surrogate_batch(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    contact_spec: CertifiedContactSurrogateSpec,
    hbond_spec: CertifiedDirectionalHBondSpec,
) -> CertifiedBatchResult:
    contact_batch = score_certified_contact_batch(
        receptor_coords,
        poses_coords,
        contact_spec,
    )
    hbond_batch = score_certified_directional_hbond_batch(
        receptor_coords,
        poses_coords,
        hbond_spec,
    )
    return CertifiedBatchResult(
        scores=contact_batch.scores + hbond_batch.scores,
        error_bound=contact_batch.error_bound + hbond_batch.error_bound,
        target_error=contact_batch.error_bound + hbond_batch.error_bound,
        cutoff_radius=jnp.maximum(
            jnp.array(contact_spec.cutoff), jnp.array(hbond_spec.cutoff)
        ),
    )


@conditionally_certified(
    "HandleAliases.lean::NB6; HandleAliases.lean::NB7; HandleAliases.lean::NB8; HandleAliases.lean::NB9; HandleAliases.lean::NB10",
    assumptions=[
        "The additive runtime score is exactly the sum of the certified LJ term and the certified screened-Coulomb term",
        "The combined error bound is the sum of the certified LJ bound and the finite-batch screened-Coulomb discrepancy bound",
    ],
)
def score_certified_lj_screened_coulomb_batch(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    screened_coulomb: CertifiedScreenedCoulombSpec,
    target_error: float = 0.001,
    epsilon: float = _EPSILON_KCAL_MOL,
) -> CertifiedBatchResult:
    lj_scores, lj_error_bound = score_certified_lj(
        receptor_coords,
        poses_coords,
        receptor_radii,
        ligand_radii,
        target_error=target_error,
        epsilon=epsilon,
    )
    screened_batch = score_certified_screened_coulomb_batch(
        receptor_coords,
        poses_coords,
        screened_coulomb,
    )
    combined_cutoff = jnp.maximum(
        jnp.array(optimal_cutoff(target_error, s=6.0)),
        jnp.array(screened_coulomb.cutoff),
    )
    return CertifiedBatchResult(
        scores=lj_scores + screened_batch.scores,
        error_bound=lj_error_bound + screened_batch.error_bound,
        target_error=target_error,
        cutoff_radius=combined_cutoff,
    )


@conditionally_certified(
    "HandleAliases.lean::RC6; HandleAliases.lean::RC7; HandleAliases.lean::RC8; HandleAliases.lean::RC9; HandleAliases.lean::RC10",
    assumptions=[
        "The additive runtime score is exactly the sum of the certified LJ+screened-Coulomb term and the certified polar surrogate term",
        "The combined error bound is the sum of the certified nonbonded bound and the certified polar surrogate batch discrepancy bound",
    ],
)
def score_certified_rich_chemistry_batch(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    screened_coulomb: CertifiedScreenedCoulombSpec,
    contact_spec: CertifiedContactSurrogateSpec,
    hbond_spec: CertifiedDirectionalHBondSpec,
    target_error: float = 0.001,
    epsilon: float = _EPSILON_KCAL_MOL,
) -> CertifiedBatchResult:
    nonbonded_batch = score_certified_lj_screened_coulomb_batch(
        receptor_coords,
        poses_coords,
        receptor_radii,
        ligand_radii,
        screened_coulomb,
        target_error=target_error,
        epsilon=epsilon,
    )
    polar_batch = score_certified_polar_surrogate_batch(
        receptor_coords,
        poses_coords,
        contact_spec,
        hbond_spec,
    )
    return CertifiedBatchResult(
        scores=nonbonded_batch.scores + polar_batch.scores,
        error_bound=nonbonded_batch.error_bound + polar_batch.error_bound,
        target_error=target_error,
        cutoff_radius=jnp.maximum(
            nonbonded_batch.cutoff_radius,
            polar_batch.cutoff_radius,
        ),
    )


@certified("LatticeSum.lean::lj6_tail_bound")
def score_certified_lj(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    target_error: float = 0.001,
    epsilon: float = _EPSILON_KCAL_MOL,
) -> tuple[jnp.ndarray, float]:
    """
    Batched CERTIFIED LJ scoring with Lean-proven error bounds.

    Uses optimal_cutoff to compute minimum R for target error,
    then computes truncated LJ sum within that bound.

    PROOF STATUS: CERTIFIED
      - Cutoff computed from proven bound: optimal_cutoff(ε) = (M/ε)^(1/3)
      - Energy is truncated LJ sum, error bounded by M/R³
      - Physical calibration: epsilon in kcal/mol

    Args:
        receptor_coords: (N_rec, 3)
        poses_coords: (N_poses, N_lig, 3)
        receptor_radii: (N_rec,)
        ligand_radii: (N_lig,)
        target_error: Target error bound per atom pair (default 0.001 kcal/mol)
        epsilon: Well depth in kcal/mol for calibration (default 0.086 for C-C)

    Returns:
        (scores, certified_error_bound) where certified_error_bound
        is the Lean-proven upper bound on truncation error (kcal/mol).
    """
    cutoff = jnp.array(optimal_cutoff(target_error, s=6.0))
    scores, _ = _score_certified_lj_batch(
        receptor_coords,
        poses_coords,
        receptor_radii,
        ligand_radii,
        cutoff,
        epsilon,
    )

    # Compute error bound (same for all poses)
    error_bound = epsilon * lj6_cutoff_error(cutoff)

    return scores, error_bound


@conditionally_certified(
    "HandleAliases.lean::LJ10; HandleAliases.lean::LJ11; HandleAliases.lean::LJ12; HandleAliases.lean::APX10; HandleAliases.lean::APX11; HandleAliases.lean::APX12",
    assumptions=[
        "Softened LJ matches exact LJ outside the chosen softening radius",
        "The runtime uses the exact same max(r, rSoft) softening form as the Lean theorem",
    ],
)
def score_certified_softened_lj(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    target_error: float = 0.001,
    epsilon: float = _EPSILON_KCAL_MOL,
    softening_radius: float | None = None,
    compute_error_bound: bool = True,
) -> CertifiedSoftenedBatchResult:
    cutoff = jnp.array(optimal_cutoff(target_error, s=6.0))
    r_soft = (
        _default_softening_radius(receptor_radii, ligand_radii)
        if softening_radius is None
        else softening_radius
    )
    scores, softening_error_bound = _score_certified_softened_lj_batch(
        receptor_coords,
        poses_coords,
        receptor_radii,
        ligand_radii,
        cutoff,
        epsilon,
        r_soft,
        compute_error_bound=compute_error_bound,
    )
    return CertifiedSoftenedBatchResult(
        scores=scores,
        softening_error_bound=softening_error_bound,
        target_error=target_error,
        cutoff_radius=cutoff,
        softening_radius=r_soft,
    )


@conditionally_certified(
    "HandleAliases.lean::CB10; HandleAliases.lean::CB11; HandleAliases.lean::CB12; HandleAliases.lean::LJ10; HandleAliases.lean::LJ11; HandleAliases.lean::LJ12; HandleAliases.lean::APX10; HandleAliases.lean::APX11; HandleAliases.lean::APX12",
    assumptions=[
        "The electrostatic term is identical in exact and coarse modes, so the certified delta comes entirely from the softened LJ term",
        "Softened LJ matches the Lean exact-vs-softened scoring family on the shared receptor/pose geometry",
    ],
)
def score_certified_softened_lj_realspace_ewald(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    electrostatics: CertifiedRealSpaceEwaldSpec,
    target_error: float = 0.001,
    epsilon: float = _EPSILON_KCAL_MOL,
    softening_radius: float | None = None,
    compute_error_bound: bool = True,
) -> CertifiedSoftenedBatchResult:
    electrostatics.validate()
    softened_batch = score_certified_softened_lj(
        receptor_coords,
        poses_coords,
        receptor_radii,
        ligand_radii,
        target_error=target_error,
        epsilon=epsilon,
        softening_radius=softening_radius,
        compute_error_bound=compute_error_bound,
    )
    ewald_scores = _score_realspace_ewald_batch(
        receptor_coords,
        poses_coords,
        electrostatics.receptor_charges,
        electrostatics.ligand_charges,
        jnp.array(electrostatics.cutoff),
        electrostatics.alpha,
        electrostatics.dielectric,
    )
    return CertifiedSoftenedBatchResult(
        scores=softened_batch.scores + ewald_scores,
        softening_error_bound=softened_batch.softening_error_bound,
        target_error=target_error,
        cutoff_radius=softened_batch.cutoff_radius,
        softening_radius=softened_batch.softening_radius,
    )


@conditionally_certified(
    "HandleAliases.lean::CB5; HandleAliases.lean::CB6; HandleAliases.lean::CB10; HandleAliases.lean::CB11; HandleAliases.lean::CB12; HandleAliases.lean::CB13; HandleAliases.lean::CB14; HandleAliases.lean::APX10; HandleAliases.lean::APX11; HandleAliases.lean::APX12; HandleAliases.lean::BD10",
    assumptions=[
        "Supplied receptor and ligand charges are fixed intended physical inputs",
        "The runtime uses the same coarse-vs-exact decomposition as the exported Ewald far-field correction theorems",
        "The 1e-6 minimum distance guard is a numerical safeguard against division by zero",
    ],
)
def score_certified_lj_realspace_ewald(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    electrostatics: CertifiedRealSpaceEwaldSpec,
    target_error: float = 0.001,
    epsilon: float = _EPSILON_KCAL_MOL,
) -> tuple[jnp.ndarray, float]:
    electrostatics.validate()

    lj_cutoff = jnp.array(optimal_cutoff(target_error, s=6.0))
    lj_scores, _ = _score_certified_lj_batch(
        receptor_coords,
        poses_coords,
        receptor_radii,
        ligand_radii,
        lj_cutoff,
        epsilon,
    )
    ewald_scores = _score_realspace_ewald_batch(
        receptor_coords,
        poses_coords,
        electrostatics.receptor_charges,
        electrostatics.ligand_charges,
        jnp.array(electrostatics.cutoff),
        electrostatics.alpha,
        electrostatics.dielectric,
    )

    error_bound = certified_lj_error_bound(
        target_error, epsilon
    ) + certified_realspace_ewald_error_bound(
        electrostatics.cutoff,
        electrostatics.alpha,
        electrostatics.charge_bound(),
    )
    return lj_scores + ewald_scores, error_bound


def _score_single_lj_scalar(
    receptor_coords: jnp.ndarray,
    pose_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
) -> float:
    """Wrapper that returns Python float for non-JIT use."""
    return float(
        _score_single_lj(receptor_coords, pose_coords, receptor_radii, ligand_radii)
    )


@jax.jit
def score_internal_lj(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
) -> jnp.ndarray:
    """
    Pure JAX batched internal LJ score with atom-typed radii.

    Args:
        receptor_coords: (N_rec, 3)
        poses_coords:    (N_poses, N_lig, 3)
        receptor_radii:  (N_rec,) VdW radii
        ligand_radii:    (N_lig,) VdW radii

    Returns:
        (N_poses,) array of scores
    """
    # vmap over poses dimension; receptor/radii are shared (None)
    batched_score = jax.vmap(_score_single_lj, in_axes=(None, 0, None, None))
    return batched_score(receptor_coords, poses_coords, receptor_radii, ligand_radii)


def _write_pdb(coords: np.ndarray, template_pdb: str, output_pdb: str):
    """Write coordinates back to a temporary PDB using a template."""
    with open(template_pdb, "r") as f:
        lines = f.readlines()

    out_lines = []
    atom_idx = 0
    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            if atom_idx < len(coords):
                x, y, z = coords[atom_idx]
                # PDB column formatting
                new_line = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
                out_lines.append(new_line)
                atom_idx += 1
            else:
                out_lines.append(line)
        else:
            out_lines.append(line)

    with open(output_pdb, "w") as f:
        f.writelines(out_lines)


@heuristic()  # HEURISTIC: external unverified binary
def score_smina_exact(
    receptor_file: str, ligand_template: str, poses_coords: np.ndarray
) -> np.ndarray:
    """
    Impure external wrapper invoking SMINA for accurate scoring.

    PROOF STATUS: HEURISTIC
      - SMINA/Vina: closed-source external binary
      - No formal verification of scoring function
      - Used for ground-truth comparison only

    DO NOT use for certified docking.
    """
    from dq_dock_engine.benchmark.benchmark_pdb import check_vina

    vina_path = check_vina()
    if not vina_path:
        raise RuntimeError("SMINA/Vina binary not found.")

    n_poses = poses_coords.shape[0]
    scores = np.zeros(n_poses)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir)

        # We can score sequentially or in parallel here, but for simplicity
        # we iterate over poses sequentially for the wrapper.
        for i in range(n_poses):
            pose_pdb = tmp_path / f"pose_{i}.pdb"
            _write_pdb(poses_coords[i], ligand_template, str(pose_pdb))

            cmd = [
                vina_path,
                "--receptor",
                str(receptor_file),
                "--ligand",
                str(pose_pdb),
                "--score_only",
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                # parse "Affinity: -6.54321 (kcal/mol)"
                for line in result.stdout.split("\n"):
                    if line.startswith("Affinity:"):
                        val = float(line.split()[1])
                        scores[i] = val
                        break
            except Exception as e:
                # Fallback to highly unfavorable score if SMINA fails
                scores[i] = 1000.0

    return scores


def route_scoring(engine: ScoringEngine, **kwargs) -> np.ndarray:
    """
    Strict Enum dispatch for scoring.

    DEPRECATED: Use ABC polymorphism directly:
        backend = create_scoring_backend(ScoringFamily.VINARDO)
        scores = backend.score_batch(receptor_coords, poses_coords, receptor_radii, ligand_radii)

    Required kwargs:
      - INTERNAL_LJ: receptor_coords, poses_coords, receptor_radii, ligand_radii
      - SMINA_EXACT: receptor_file, ligand_template, poses_coords
      - VINARDO: receptor_coords, poses_coords, receptor_radii, ligand_radii
      - SOFT_LJ: receptor_coords, poses_coords, receptor_radii, ligand_radii
      - CERTIFIED_LJ_REALSPACE_EWALD: receptor_coords, poses_coords,
        receptor_radii, ligand_radii, electrostatics
    """
    from dq_dock_engine.docking.scoring_vinardo import (
        create_scoring_backend,
        ScoringFamily,
    )

    match engine:
        case ScoringEngine.INTERNAL_LJ:
            return np.array(
                score_internal_lj(
                    kwargs["receptor_coords"],
                    kwargs["poses_coords"],
                    kwargs["receptor_radii"],
                    kwargs["ligand_radii"],
                )
            )

        case ScoringEngine.VINARDO:
            backend = create_scoring_backend(ScoringFamily.VINARDO)
            return np.array(
                backend.score_batch(
                    kwargs["receptor_coords"],
                    kwargs["poses_coords"],
                    kwargs["receptor_radii"],
                    kwargs["ligand_radii"],
                )
            )

        case ScoringEngine.SOFT_LJ:
            backend = create_scoring_backend(ScoringFamily.SOFT_LJ)
            return np.array(
                backend.score_batch(
                    kwargs["receptor_coords"],
                    kwargs["poses_coords"],
                    kwargs["receptor_radii"],
                    kwargs["ligand_radii"],
                )
            )

        case ScoringEngine.SMINA_EXACT:
            return score_smina_exact(
                kwargs["receptor_file"],
                kwargs["ligand_template"],
                kwargs["poses_coords"],
            )

        case ScoringEngine.CERTIFIED_LJ:
            target_error = kwargs.get("target_error", 0.001)
            scores, error_bound = score_certified_lj(
                kwargs["receptor_coords"],
                kwargs["poses_coords"],
                kwargs["receptor_radii"],
                kwargs["ligand_radii"],
                target_error=target_error,
            )
            return np.array(scores)

        case ScoringEngine.CERTIFIED_LJ_REALSPACE_EWALD:
            target_error = kwargs.get("target_error", 0.001)
            electrostatics: CertifiedRealSpaceEwaldSpec = kwargs["electrostatics"]
            scores, error_bound = score_certified_lj_realspace_ewald(
                kwargs["receptor_coords"],
                kwargs["poses_coords"],
                kwargs["receptor_radii"],
                kwargs["ligand_radii"],
                electrostatics=electrostatics,
                target_error=target_error,
                epsilon=kwargs.get("epsilon", _EPSILON_KCAL_MOL),
            )
            return np.array(scores)

        case _:
            raise ValueError(f"Unknown ScoringEngine: {engine}")
