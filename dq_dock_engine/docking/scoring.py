"""
Scoring mechanism using strict OpenHCS Enum dispatch.

Separates JAX-native internal physics from impure SMINA external subprocess wrapper.

PROOF STATUS SUMMARY:
  - score_internal_lj: HEURISTIC (ad-hoc weights)
  - score_smina_exact: HEURISTIC (external unverified binary)
  - route_scoring: HEURISTIC (dispatch only)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Union, Sequence
import os
import subprocess
import tempfile
import pathlib
import functools

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np

from dq_dock_engine.docking.certified_runtime_plans import (
    CertifiedPruningDeltaComponent,
    CertifiedPruningDeltaComponentKind,
    CertifiedPruningDeltaBudget,
)
from dq_dock_engine.docking.formal_handles import (
    attractive_extended_chemistry_theorem_handles,
    attractive_water_mediated_hbond_theorem_handles,
    contact_surrogate_theorem_handles,
    cooperative_hbond_theorem_handles,
    directional_metal_coordination_theorem_handles,
    directional_hbond_finite_theorem_handles,
    metal_coordination_cutoff_theorem_handles,
    omitted_channel_bound_theorem_handles,
    rich_pruning_delta_theorem_handles,
    screened_coulomb_theorem_handles,
    softened_lj_shared_base_delta_theorem_handles,
)
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


# =============================================================================
# Screened Coulomb Cutoff Derivation (Lean: ConditionalComposition.lean)
# =============================================================================
# Theorems:
#   - screened_coulomb_exp_bound: Q * exp(-κR) ≤ ε when R ≥ ln(Q/ε) / κ
#   - screenedCoulombMinCutoff: R_min = ln(Q/ε) / κ
#   - screenedCoulombMinCutoff_optimal: R = ln(Q/ε)/κ is the MINIMUM achieving ε
#   - screenedCoulombMinCutoff_tight: At R = ln(Q/ε)/κ, error EQUALS ε exactly
#
# For pairwise sums over N pairs, use Q_max = max(|q_i * q_j|) * N_pairs
# or conservatively Q_max = max_receptor_charge * max_ligand_charge * N_pairs

# =============================================================================
# Debye-Hückel Screening (Lean: ConditionalComposition.lean)
# =============================================================================
# In aqueous solution, electrostatics ARE screened. Pure Coulomb (κ=0) is
# physically incorrect for solvated biomolecules.
#
# At physiological conditions (37°C, ionic strength I ≈ 0.15 M):
#   Debye length λ_D ≈ 7.8 Å → κ = 1/λ_D ≈ 0.128 Å⁻¹
#
# Theorems:
#   - physiological_cutoff_bound: κ=0.128 guarantees exponential decay
#   - cutoff_12_sufficient_condition: 12Å suffices when Q/ε ≤ exp(1.536) ≈ 4.6
#
# This provides PHYSICAL justification for why 12Å works for non-metal systems:
# they're not pure Coulomb, they're weakly screened Coulomb.

KAPPA_PHYSIOLOGICAL = (
    0.128  # Å⁻¹, Debye-Hückel screening at physiological ionic strength
)


def screened_coulomb_min_cutoff(
    max_charge_product: float,
    target_error: float,
    kappa: float,
    min_cutoff: float = 1.0,
) -> float:
    """Derive minimum cutoff for screened Coulomb to achieve target error.

    Lean theorem (screened_coulomb_exp_bound):
        Q * exp(-κR) ≤ ε when R ≥ ln(Q/ε) / κ

    Args:
        max_charge_product: Maximum |q_i * q_j| * N_pairs bound
        target_error: Desired error bound ε
        kappa: Screening parameter (must be > 0)
        min_cutoff: Minimum cutoff floor (default 1.0 Å)

    Returns:
        Minimum cutoff R such that error ≤ target_error

    JIT-safe: Uses jnp.log, jnp.maximum for JAX compatibility.
    """
    # For κ = 0, this formula diverges; caller should use different logic
    # (pure Coulomb with power-law tail bound from LatticeSum.lean)
    if kappa <= 0:
        raise ValueError("screened_coulomb_min_cutoff requires kappa > 0")

    # R = ln(Q/ε) / κ
    ratio = max_charge_product / target_error
    cutoff = float(jnp.log(ratio) / kappa)
    return max(cutoff, min_cutoff)


def screened_coulomb_min_cutoff_jit(
    max_charge_product: jnp.ndarray,
    target_error: jnp.ndarray,
    kappa: jnp.ndarray,
    min_cutoff: float = 1.0,
) -> jnp.ndarray:
    """JIT-compatible version of screened_coulomb_min_cutoff.

    Safe for use inside JAX-traced functions. Uses jnp.where to handle
    edge cases without Python control flow.
    """
    ratio = max_charge_product / target_error
    raw_cutoff = jnp.log(jnp.maximum(ratio, 1.0)) / jnp.maximum(kappa, 1e-10)
    return jnp.maximum(raw_cutoff, min_cutoff)


def screened_coulomb_error_at_cutoff(
    max_charge_product: float,
    kappa: float,
    cutoff: float,
) -> float:
    """Compute the error bound for a given cutoff.

    Lean theorem (screened_coulomb_exp_bound):
        error ≤ Q * exp(-κR)

    Args:
        max_charge_product: Maximum |q_i * q_j| * N_pairs bound
        kappa: Screening parameter
        cutoff: Cutoff radius R

    Returns:
        Upper bound on tail error
    """
    return float(max_charge_product * jnp.exp(-kappa * cutoff))


# =============================================================================
# Pure Coulomb (κ=0) Cutoff Derivation (Lean: ConditionalComposition.lean)
# =============================================================================
# Theorems:
#   - coulomb_tail_bound: N × Q / R ≤ ε when R ≥ N × Q / ε
#   - coulombMinCutoff: R_min = N × Q / ε
#
# For pairwise sums, N = number of pairs, Q = max|q_i × q_j|


def coulomb_min_cutoff(
    n_pairs: int,
    max_charge_product: float,
    target_error: float,
    min_cutoff: float = 4.0,
    max_cutoff: float = 30.0,
) -> float:
    """Derive minimum cutoff for pure Coulomb (κ=0) to achieve target error.

    Lean theorem (coulomb_tail_bound):
        N × Q / R ≤ ε when R ≥ N × Q / ε

    Args:
        n_pairs: Number of interacting pairs N
        max_charge_product: Maximum |q_i × q_j| bound Q
        target_error: Desired error bound ε
        min_cutoff: Minimum physical cutoff floor (default 4.0 Å)
        max_cutoff: Maximum practical cutoff cap (default 30.0 Å)

    Returns:
        Minimum cutoff R such that tail error ≤ target_error

    Note: Pure Coulomb has 1/R decay, not exponential, so cutoffs can be
    large for many pairs with significant charges.
    """
    if target_error <= 0:
        raise ValueError("target_error must be positive")

    # R = N × Q / ε (coulombMinCutoff theorem)
    cutoff = float(n_pairs * max_charge_product / target_error)

    # Clamp to physical bounds
    return max(min_cutoff, min(cutoff, max_cutoff))


def coulomb_min_cutoff_jit(
    n_pairs: jnp.ndarray,
    max_charge_product: jnp.ndarray,
    target_error: jnp.ndarray,
    min_cutoff: float = 4.0,
    max_cutoff: float = 30.0,
) -> jnp.ndarray:
    """JIT-compatible version of coulomb_min_cutoff.

    Safe for use inside JAX-traced functions.
    """
    raw_cutoff = n_pairs * max_charge_product / jnp.maximum(target_error, 1e-10)
    return jnp.clip(raw_cutoff, min_cutoff, max_cutoff)


def coulomb_error_at_cutoff(
    n_pairs: int,
    max_charge_product: float,
    cutoff: float,
) -> float:
    """Compute the error bound for a given cutoff (pure Coulomb).

    Lean theorem (coulomb_tail_bound):
        error ≤ N × Q / R

    Args:
        n_pairs: Number of interacting pairs N
        max_charge_product: Maximum |q_i × q_j| bound Q
        cutoff: Cutoff radius R

    Returns:
        Upper bound on tail error
    """
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")
    return float(n_pairs * max_charge_product / cutoff)


# =============================================================================
# Metal Coordination Cutoff Derivation (Lean: MetalCoordinationApproximation.lean)
# =============================================================================
# Theorems:
#   - metalCoordination_tail_bound: |w·exp(-((r-ideal)/width)²)| ≤ |w|·exp(-((rc-ideal)/width)²)
#   - metalCoordination_cutoff_sufficient: |w|·exp(-((rc-ideal)/width)²) ≤ ε
#       when rc ≥ ideal + width·√(ln(|w|/ε))
#   - metalCoordinationMinCutoff: rc_min = ideal + width·√(ln(|w|/ε))


def metal_coordination_min_cutoff(
    max_strength_product: float,
    target_error: float,
    ideal_distance: float,
    distance_width: float,
    min_cutoff: float = 2.5,
) -> float:
    """Derive minimum cutoff for metal coordination to achieve target error.

    Lean theorem (metalCoordination_cutoff_sufficient):
        |w|·exp(-((rc-ideal)/width)²) ≤ ε
        when rc ≥ ideal + width·√(ln(|w|/ε))

    Args:
        max_strength_product: Maximum |w_receptor × w_ligand| bound
        target_error: Desired error bound ε (kcal/mol)
        ideal_distance: Metal-ligand equilibrium distance (Å)
        distance_width: Gaussian width parameter σ (Å)
        min_cutoff: Minimum cutoff floor (default 2.5 Å, must exceed ideal)

    Returns:
        Minimum cutoff rc such that tail error ≤ target_error
    """
    if max_strength_product <= 0:
        raise ValueError(
            "metal_coordination_min_cutoff requires max_strength_product > 0"
        )
    if target_error <= 0:
        raise ValueError("metal_coordination_min_cutoff requires target_error > 0")
    if distance_width <= 0:
        raise ValueError("metal_coordination_min_cutoff requires distance_width > 0")

    # rc = ideal + width · √(ln(|w|/ε))  (metalCoordinationMinCutoff theorem)
    ratio = max_strength_product / target_error
    if ratio <= 1.0:
        # Error bound already satisfied at ideal distance — cutoff = ideal
        return max(ideal_distance, min_cutoff)
    cutoff = ideal_distance + distance_width * float(jnp.sqrt(jnp.log(ratio)))
    return max(cutoff, min_cutoff)


def metal_coordination_error_at_cutoff(
    max_strength_product: float,
    ideal_distance: float,
    distance_width: float,
    cutoff: float,
) -> float:
    """Compute the error bound for a given metal coordination cutoff.

    Lean theorem (metalCoordination_tail_bound):
        error ≤ |w|·exp(-((rc-ideal)/width)²) for rc ≥ ideal

    Args:
        max_strength_product: Maximum |w_receptor × w_ligand| bound
        ideal_distance: Metal-ligand equilibrium distance (Å)
        distance_width: Gaussian width parameter σ (Å)
        cutoff: Cutoff radius rc (Å)

    Returns:
        Upper bound on tail error (kcal/mol)
    """
    if distance_width <= 0:
        raise ValueError("distance_width must be positive")
    if cutoff < ideal_distance:
        return max_strength_product  # No tail bound below ideal
    exponent = ((cutoff - ideal_distance) / distance_width) ** 2
    return float(max_strength_product * jnp.exp(-exponent))


@register_pytree_node_class
@dataclass(frozen=True)
class CertifiedRealSpaceEwaldSpec:
    """Real-space Ewald electrostatics spec.

    Dielectric constant derivation (Lean: DielectricBounds.lean):
      - DB1: Kirkwood-Fröhlich mixing gives ε_eff = ε_in × ε_out / (ε_out + (ε_in - ε_out) × f)
      - DB2: ε = 4.0 is within valid range [2, 80] for protein-ligand interface
      - At interface (f ≈ 0.5): ε_eff = 2 × 80 / 41 ≈ 3.9 ≈ 4.0
    """

    receptor_charges: jnp.ndarray
    ligand_charges: jnp.ndarray
    cutoff: float = 12.0
    alpha: float = 0.2
    dielectric: float = 4.0  # Certified by DB1, DB2 (Kirkwood-Fröhlich theory)

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

    def analytic_cutoff_tail_bound(self) -> float:
        """Screened Coulomb tail: (q_max_rec × q_max_lig / ε_r) × exp(-κR) / R × N_pairs.

        Lean: exponential screening decay beyond cutoff radius.
        """
        import math

        q_max_rec = float(jnp.max(jnp.abs(self.receptor_charges)))
        q_max_lig = float(jnp.max(jnp.abs(self.ligand_charges)))
        n_pairs = int(self.receptor_charges.shape[0]) * int(
            self.ligand_charges.shape[0]
        )
        tail_per_pair = (
            q_max_rec
            * q_max_lig
            * math.exp(-self.kappa * self.cutoff)
            / (self.dielectric * self.cutoff)
        )
        return tail_per_pair * n_pairs

    def receptor_subset(self, indices: jnp.ndarray) -> "CertifiedScreenedCoulombSpec":
        return CertifiedScreenedCoulombSpec(
            receptor_charges=self.receptor_charges[indices],
            ligand_charges=self.ligand_charges,
            kappa=self.kappa,
            cutoff=self.cutoff,
            dielectric=self.dielectric,
        )

    def zeroed(self) -> "CertifiedScreenedCoulombSpec":
        return CertifiedScreenedCoulombSpec(
            receptor_charges=jnp.zeros_like(self.receptor_charges),
            ligand_charges=jnp.zeros_like(self.ligand_charges),
            kappa=self.kappa,
            cutoff=self.cutoff,
            dielectric=self.dielectric,
        )


class CertifiedOptionalInteractionTerm(ABC):
    @abstractmethod
    def validate(self) -> None:
        """Fail-loud validation for the interaction term."""

    @abstractmethod
    def receptor_subset(
        self, indices: jnp.ndarray
    ) -> "CertifiedOptionalInteractionTerm":
        """Return the term restricted to a retained receptor subset."""

    @property
    @abstractmethod
    def is_active(self) -> jnp.ndarray:
        """Whether this optional interaction contributes non-zero signal."""

    @property
    @abstractmethod
    def cutoff_radius(self) -> float:
        """Certified cutoff radius for the coarse interaction family."""

    @abstractmethod
    def exact_scores(
        self,
        receptor_coords: jnp.ndarray,
        poses_coords: jnp.ndarray,
    ) -> jnp.ndarray:
        """Return exact batched scores for this interaction family."""

    @abstractmethod
    def analytic_cutoff_tail_bound(self) -> float:
        """Certified upper bound on tail contribution beyond cutoff radius.

        Returns the maximum absolute error from using cutoff approximation
        instead of infinite-range exact scoring, summed over all receptor-ligand
        pairs. This is batch-size independent.

        Lean: sum_uniformApprox — cutoff approximation error per channel.
        """

    @abstractmethod
    def cutoff_scores(
        self,
        receptor_coords: jnp.ndarray,
        poses_coords: jnp.ndarray,
    ) -> jnp.ndarray:
        """Return cutoff batched scores for this interaction family."""


@register_pytree_node_class
@dataclass(frozen=True)
class CertifiedContactSurrogateSpec(CertifiedOptionalInteractionTerm):
    """Contact/desolvation surrogate spec using Gaussian decay w × exp(-(βr)²).

    Cutoff derivation (Lean: GaussianDecayBounds.lean):
      - GD3: R_min = √(ln(W/ε)) / β is the minimum cutoff for error ≤ ε
      - GD4: gaussianMinCutoff_sufficient proves this is sufficient
      - GD6: gaussianMinCutoff_optimal proves any smaller R violates the bound

    For β=0.6 Å⁻¹, W=1, ε=0.001: R_min = √(ln(1000))/0.6 ≈ 4.4 Å
    Default cutoff 6.0 Å provides ~36% safety margin.
    """

    receptor_weights: jnp.ndarray
    ligand_weights: jnp.ndarray
    beta: float = 0.6  # Gaussian decay rate (Å⁻¹)
    cutoff: float = 6.0  # Certified by GD3, GD4, GD6

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

    def analytic_cutoff_tail_bound(self) -> float:
        """Contact Gaussian tail: W_max² × exp(-(β × R_cutoff)²) × N_pairs.

        Lean: GaussianDecayBounds.lean::GD3.
        """
        import math

        w_max_rec = float(jnp.max(jnp.abs(self.receptor_weights)))
        w_max_lig = float(jnp.max(jnp.abs(self.ligand_weights)))
        n_pairs = int(self.receptor_weights.shape[0]) * int(
            self.ligand_weights.shape[0]
        )
        tail_per_pair = (
            w_max_rec * w_max_lig * math.exp(-((self.beta * self.cutoff) ** 2))
        )
        return tail_per_pair * n_pairs

    def max_total_score_bound(self) -> float:
        """Certified upper bound on the full contact channel score.

        Each pair contribution is bounded by its absolute pair weight because the
        Gaussian radial factor stays in `(0, 1]`. Lean: `GD7`, `GD8`.
        """
        pair_weight = jnp.abs(
            self.receptor_weights[:, None] * self.ligand_weights[None, :]
        )
        return float(jnp.sum(pair_weight))

    def receptor_subset(self, indices: jnp.ndarray) -> "CertifiedContactSurrogateSpec":
        return CertifiedContactSurrogateSpec(
            receptor_weights=self.receptor_weights[indices],
            ligand_weights=self.ligand_weights,
            beta=self.beta,
            cutoff=self.cutoff,
        )

    def zeroed(self) -> "CertifiedContactSurrogateSpec":
        return CertifiedContactSurrogateSpec(
            receptor_weights=jnp.zeros_like(self.receptor_weights),
            ligand_weights=jnp.zeros_like(self.ligand_weights),
            beta=self.beta,
            cutoff=self.cutoff,
        )

    @property
    def is_active(self) -> jnp.ndarray:
        return jnp.logical_and(
            jnp.any(self.receptor_weights != 0.0),
            jnp.any(self.ligand_weights != 0.0),
        )

    @property
    def cutoff_radius(self) -> float:
        return self.cutoff

    def exact_scores(
        self,
        receptor_coords: jnp.ndarray,
        poses_coords: jnp.ndarray,
    ) -> jnp.ndarray:
        return _score_contact_exact_batch(
            receptor_coords,
            poses_coords,
            self.receptor_weights,
            self.ligand_weights,
            self.beta,
        )

    def cutoff_scores(
        self,
        receptor_coords: jnp.ndarray,
        poses_coords: jnp.ndarray,
    ) -> jnp.ndarray:
        return _score_contact_cutoff_batch(
            receptor_coords,
            poses_coords,
            self.receptor_weights,
            self.ligand_weights,
            self.beta,
            self.cutoff,
        )


@register_pytree_node_class
@dataclass(frozen=True)
class CertifiedMetalCoordinationSpec(CertifiedOptionalInteractionTerm):
    """Metal coordination spec for Zn, Fe, Mg, etc.

    Directional model: radial Gaussian × angular Gaussian product.
    When angle_width is large (>= 1e6), the angular factor → 1.0 and this
    reduces to the isotropic radial-only model.

    Certified by:
      MC1-MC5: isotropic metal coordination (MetalCoordinationApproximation.lean)
      DMC1: directional two-factor Lipschitz bound
      DMC2: angular factor tightens tail decay
      DMC3: directional cutoff uniform approximation

    Distance width derivation (Lean: ThermalFluctuationBounds.lean):
      - TF1: σ = √(kT/k) from equipartition theorem
      - TF4: σ_metal = √(0.616 / 7.0) ≈ 0.3 Å at 310K
      - TF5: Stiffer bonds (larger k) → smaller fluctuations

    Metal bonds have k ≈ 7 kcal/(mol·Å²), ~7× stiffer than H-bonds.
    """

    receptor_strengths: jnp.ndarray
    ligand_strengths: jnp.ndarray
    receptor_ideal_angles: jnp.ndarray  # (N_rec,) ideal coordination angles per metal
    ideal_distance: float = 2.1  # Metal-ligand distance (Å), empirical
    distance_width: float = 0.3  # Certified by TF1, TF4, TF5 (σ = √(kT/k))
    angle_width: float = 1e6  # Angular Gaussian width; large = isotropic
    cutoff: float = 4.0

    def validate(self) -> None:
        if self.ideal_distance <= 0:
            raise ValueError("metal coordination ideal_distance must be positive")
        if self.distance_width <= 0:
            raise ValueError("metal coordination distance_width must be positive")
        if self.angle_width <= 0:
            raise ValueError("metal coordination angle_width must be positive")
        if self.cutoff <= 0:
            raise ValueError("metal coordination cutoff must be positive")
        if self.receptor_strengths.ndim != 1:
            raise ValueError("receptor_strengths must be a 1D array")
        if self.ligand_strengths.ndim != 1:
            raise ValueError("ligand_strengths must be a 1D array")
        if self.receptor_ideal_angles.ndim != 1:
            raise ValueError("receptor_ideal_angles must be a 1D array")
        if self.receptor_ideal_angles.shape[0] != self.receptor_strengths.shape[0]:
            raise ValueError(
                "receptor_ideal_angles must match receptor_strengths length"
            )

    def tree_flatten(self):
        children = (
            self.receptor_strengths,
            self.ligand_strengths,
            self.receptor_ideal_angles,
        )
        aux_data = (
            self.ideal_distance,
            self.distance_width,
            self.angle_width,
            self.cutoff,
        )
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            receptor_strengths=children[0],
            ligand_strengths=children[1],
            receptor_ideal_angles=children[2],
            ideal_distance=aux_data[0],
            distance_width=aux_data[1],
            angle_width=aux_data[2],
            cutoff=aux_data[3],
        )

    def analytic_cutoff_tail_bound(self) -> float:
        """Metal coordination Gaussian tail: S_max² × exp(-((R-d₀)/σ)²) × N_pairs.

        Lean: MC11-MC13 for the radial Gaussian tail envelope, plus DMC2 since
        the directional angular factor lies in [0, 1] and can only tighten the
        tail bound.
        """
        import math

        s_max_rec = float(jnp.max(jnp.abs(self.receptor_strengths)))
        s_max_lig = float(jnp.max(jnp.abs(self.ligand_strengths)))
        n_pairs = int(self.receptor_strengths.shape[0]) * int(
            self.ligand_strengths.shape[0]
        )
        tail_per_pair = (
            s_max_rec
            * s_max_lig
            * math.exp(
                -(((self.cutoff - self.ideal_distance) / self.distance_width) ** 2)
            )
        )
        return tail_per_pair * n_pairs

    def max_total_score_bound(self) -> float:
        """Certified upper bound on the full metal-coordination channel score.

        Each pair term is bounded by its absolute strength product because the
        radial Gaussian and geometry factor lie in `[0, 1]`. Lean: `DMC6`.
        """
        pair_strength = jnp.abs(
            self.receptor_strengths[:, None] * self.ligand_strengths[None, :]
        )
        return float(jnp.sum(pair_strength))

    def receptor_subset(self, indices: jnp.ndarray) -> "CertifiedMetalCoordinationSpec":
        return CertifiedMetalCoordinationSpec(
            receptor_strengths=self.receptor_strengths[indices],
            ligand_strengths=self.ligand_strengths,
            receptor_ideal_angles=self.receptor_ideal_angles[indices],
            ideal_distance=self.ideal_distance,
            distance_width=self.distance_width,
            angle_width=self.angle_width,
            cutoff=self.cutoff,
        )

    def zeroed(self) -> "CertifiedMetalCoordinationSpec":
        return CertifiedMetalCoordinationSpec(
            receptor_strengths=jnp.zeros_like(self.receptor_strengths),
            ligand_strengths=jnp.zeros_like(self.ligand_strengths),
            receptor_ideal_angles=jnp.zeros_like(self.receptor_ideal_angles),
            ideal_distance=self.ideal_distance,
            distance_width=self.distance_width,
            angle_width=self.angle_width,
            cutoff=self.cutoff,
        )

    @property
    def is_active(self) -> jnp.ndarray:
        return jnp.logical_and(
            jnp.any(self.receptor_strengths != 0.0),
            jnp.any(self.ligand_strengths != 0.0),
        )

    @property
    def cutoff_radius(self) -> float:
        return self.cutoff

    def exact_scores(
        self,
        receptor_coords: jnp.ndarray,
        poses_coords: jnp.ndarray,
    ) -> jnp.ndarray:
        return _score_directional_metal_exact_batch(
            receptor_coords,
            poses_coords,
            self.receptor_strengths,
            self.ligand_strengths,
            self.ideal_distance,
            self.distance_width,
            self.receptor_ideal_angles,
            self.angle_width,
        )

    def cutoff_scores(
        self,
        receptor_coords: jnp.ndarray,
        poses_coords: jnp.ndarray,
    ) -> jnp.ndarray:
        return _score_directional_metal_cutoff_batch(
            receptor_coords,
            poses_coords,
            self.receptor_strengths,
            self.ligand_strengths,
            self.ideal_distance,
            self.distance_width,
            self.receptor_ideal_angles,
            self.angle_width,
            self.cutoff,
        )


@register_pytree_node_class
@dataclass(frozen=True)
class CertifiedDirectionalHBondSpec(CertifiedOptionalInteractionTerm):
    """Directional hydrogen bond spec with donor/acceptor angle terms.

    Distance width derivation (Lean: ThermalFluctuationBounds.lean):
      - TF1: σ = √(kT/k) from equipartition theorem
      - TF3: σ_hbond = √(0.616 / 1.0) ≈ 0.8 Å at 310K
      - TF6: Higher temperature → larger fluctuations

    H-bonds have k ≈ 1 kcal/(mol·Å²) from O-H···O stretching modes.
    """

    receptor_anchor_indices: jnp.ndarray
    receptor_directions: jnp.ndarray
    ligand_anchor_indices: jnp.ndarray
    ligand_local_directions: jnp.ndarray
    ligand_frame_coords: jnp.ndarray
    receptor_strengths: jnp.ndarray
    ligand_strengths: jnp.ndarray
    receptor_alignment_sign: float = 1.0
    ligand_alignment_sign: float = -1.0
    ideal_distance: float = 2.8  # N-H···O distance (Å), crystallographic
    distance_width: float = 0.8  # Certified by TF1, TF3, TF6 (σ = √(kT/k))
    cutoff: float = 4.0

    def validate(self) -> None:
        if self.ideal_distance <= 0:
            raise ValueError("ideal hydrogen-bond distance must be positive")
        if self.distance_width <= 0:
            raise ValueError("hydrogen-bond distance width must be positive")
        if self.cutoff <= 0:
            raise ValueError("hydrogen-bond cutoff must be positive")
        if self.receptor_anchor_indices.ndim != 1:
            raise ValueError("receptor_anchor_indices must be a 1D array")
        if self.receptor_directions.ndim != 2 or self.receptor_directions.shape[1] != 3:
            raise ValueError("receptor_directions must have shape (N_rec_sites, 3)")
        if self.ligand_anchor_indices.ndim != 1:
            raise ValueError("ligand_anchor_indices must be a 1D array")
        if (
            self.ligand_local_directions.ndim != 2
            or self.ligand_local_directions.shape[1] != 3
        ):
            raise ValueError("ligand_local_directions must have shape (N_lig_sites, 3)")
        if self.ligand_frame_coords.ndim != 2 or self.ligand_frame_coords.shape[1] != 3:
            raise ValueError("ligand_frame_coords must have shape (N_lig_atoms, 3)")
        if self.receptor_strengths.ndim != 1:
            raise ValueError("receptor_strengths must be a 1D array")
        if self.ligand_strengths.ndim != 1:
            raise ValueError("ligand_strengths must be a 1D array")
        if self.receptor_anchor_indices.shape[0] != self.receptor_directions.shape[0]:
            raise ValueError(
                "receptor_anchor_indices and receptor_directions must have matching lengths"
            )
        if self.receptor_anchor_indices.shape[0] != self.receptor_strengths.shape[0]:
            raise ValueError(
                "receptor_anchor_indices and receptor_strengths must have matching lengths"
            )
        if self.ligand_anchor_indices.shape[0] != self.ligand_local_directions.shape[0]:
            raise ValueError(
                "ligand_anchor_indices and ligand_local_directions must have matching lengths"
            )
        if self.ligand_anchor_indices.shape[0] != self.ligand_strengths.shape[0]:
            raise ValueError(
                "ligand_anchor_indices and ligand_strengths must have matching lengths"
            )
        if self.receptor_alignment_sign not in (-1.0, 1.0):
            raise ValueError("receptor_alignment_sign must be +/-1")
        if self.ligand_alignment_sign not in (-1.0, 1.0):
            raise ValueError("ligand_alignment_sign must be +/-1")
        if self.receptor_alignment_sign == self.ligand_alignment_sign:
            raise ValueError("exactly one directional H-bond side must be the donor")

    def tree_flatten(self):
        children = (
            self.receptor_anchor_indices,
            self.receptor_directions,
            self.ligand_anchor_indices,
            self.ligand_local_directions,
            self.ligand_frame_coords,
            self.receptor_strengths,
            self.ligand_strengths,
        )
        aux_data = (
            self.receptor_alignment_sign,
            self.ligand_alignment_sign,
            self.ideal_distance,
            self.distance_width,
            self.cutoff,
        )
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)

    def analytic_cutoff_tail_bound(self) -> float:
        """H-bond Gaussian tail: S_max² × exp(-((R-d₀)/σ)²) × N_pairs.

        Lean: radial Gaussian tail envelope plus HB16 (angular factors in [0,1]
        cannot worsen the radial tail bound).
        """
        import math

        s_max_rec = float(jnp.max(jnp.abs(self.receptor_strengths)))
        s_max_lig = float(jnp.max(jnp.abs(self.ligand_strengths)))
        n_pairs = int(self.receptor_strengths.shape[0]) * int(
            self.ligand_strengths.shape[0]
        )
        tail_per_pair = (
            s_max_rec
            * s_max_lig
            * math.exp(
                -(((self.cutoff - self.ideal_distance) / self.distance_width) ** 2)
            )
        )
        return tail_per_pair * n_pairs

    def max_total_score_bound(self) -> float:
        """Certified upper bound on the full directional H-bond channel score.

        Each pair contribution is bounded by its clipped pair-strength envelope
        because the radial and angular factors lie in `[0, 1]`. Lean: `HB17`.
        """
        pair_strength = jnp.clip(
            self.receptor_strengths[:, None] * self.ligand_strengths[None, :],
            0.0,
            1.0,
        )
        return float(jnp.sum(pair_strength))

    def receptor_subset(self, indices: jnp.ndarray) -> "CertifiedDirectionalHBondSpec":
        if indices.shape[0] == 0:
            return CertifiedDirectionalHBondSpec(
                receptor_anchor_indices=jnp.zeros_like(self.receptor_anchor_indices),
                receptor_directions=self.receptor_directions,
                ligand_anchor_indices=self.ligand_anchor_indices,
                ligand_local_directions=self.ligand_local_directions,
                ligand_frame_coords=self.ligand_frame_coords,
                receptor_strengths=jnp.zeros_like(self.receptor_strengths),
                ligand_strengths=self.ligand_strengths,
                receptor_alignment_sign=self.receptor_alignment_sign,
                ligand_alignment_sign=self.ligand_alignment_sign,
                ideal_distance=self.ideal_distance,
                distance_width=self.distance_width,
                cutoff=self.cutoff,
            )
        retained_matches = self.receptor_anchor_indices[:, None] == indices[None, :]
        retained_mask = jnp.any(retained_matches, axis=1)
        remapped_anchor_indices = jnp.argmax(retained_matches, axis=1).astype(jnp.int32)
        return CertifiedDirectionalHBondSpec(
            receptor_anchor_indices=remapped_anchor_indices,
            receptor_directions=self.receptor_directions,
            ligand_anchor_indices=self.ligand_anchor_indices,
            ligand_local_directions=self.ligand_local_directions,
            ligand_frame_coords=self.ligand_frame_coords,
            receptor_strengths=jnp.where(
                retained_mask,
                self.receptor_strengths,
                0.0,
            ),
            ligand_strengths=self.ligand_strengths,
            receptor_alignment_sign=self.receptor_alignment_sign,
            ligand_alignment_sign=self.ligand_alignment_sign,
            ideal_distance=self.ideal_distance,
            distance_width=self.distance_width,
            cutoff=self.cutoff,
        )

    @property
    def is_active(self) -> jnp.ndarray:
        return jnp.logical_and(
            jnp.any(self.receptor_strengths != 0.0),
            jnp.any(self.ligand_strengths != 0.0),
        )

    @property
    def cutoff_radius(self) -> float:
        return self.cutoff

    def exact_scores(
        self,
        receptor_coords: jnp.ndarray,
        poses_coords: jnp.ndarray,
    ) -> jnp.ndarray:
        return _score_directional_hbond_exact_batch(
            receptor_coords,
            poses_coords,
            self.receptor_anchor_indices,
            self.receptor_directions,
            self.ligand_anchor_indices,
            self.ligand_local_directions,
            self.ligand_frame_coords,
            self.receptor_strengths,
            self.ligand_strengths,
            self.receptor_alignment_sign,
            self.ligand_alignment_sign,
            self.ideal_distance,
            self.distance_width,
        )

    def cutoff_scores(
        self,
        receptor_coords: jnp.ndarray,
        poses_coords: jnp.ndarray,
    ) -> jnp.ndarray:
        return _score_directional_hbond_cutoff_batch(
            receptor_coords,
            poses_coords,
            self.receptor_anchor_indices,
            self.receptor_directions,
            self.ligand_anchor_indices,
            self.ligand_local_directions,
            self.ligand_frame_coords,
            self.receptor_strengths,
            self.ligand_strengths,
            self.receptor_alignment_sign,
            self.ligand_alignment_sign,
            self.ideal_distance,
            self.distance_width,
            self.cutoff,
        )


@register_pytree_node_class
@dataclass(frozen=True)
class CertifiedExtendedInteractionBundle:
    terms: tuple[CertifiedOptionalInteractionTerm, ...] = ()

    def tree_flatten(self):
        return ((self.terms,), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)

    def receptor_subset(
        self, indices: jnp.ndarray
    ) -> "CertifiedExtendedInteractionBundle":
        return CertifiedExtendedInteractionBundle(
            terms=tuple(term.receptor_subset(indices) for term in self.terms),
        ).filter_active()

    @property
    def has_active_terms(self) -> jnp.ndarray:
        if not self.terms:
            return jnp.array(False)
        return jnp.any(jnp.stack([term.is_active for term in self.terms], axis=0))

    def filter_active(self) -> "CertifiedExtendedInteractionBundle":
        """Return a new bundle containing only terms with non-zero sites.

        Call once at construction/subset time — not in the scoring hot loop.
        Each retained term is also validated so that per-call validate() can
        be skipped in the inner scoring path.
        """
        active: list[CertifiedOptionalInteractionTerm] = []
        for term in self.terms:
            if bool(jax.device_get(term.is_active)):
                term.validate()
                active.append(term)
        return CertifiedExtendedInteractionBundle(terms=tuple(active))


@register_pytree_node_class
@dataclass(frozen=True)
class CertifiedRichChemistryPlan:
    screened_coulomb: CertifiedScreenedCoulombSpec
    contact: CertifiedContactSurrogateSpec
    hbond_receptor_donor: CertifiedDirectionalHBondSpec
    hbond_ligand_donor: CertifiedDirectionalHBondSpec
    metal_coordination: CertifiedMetalCoordinationSpec
    pairwise_sigma: jnp.ndarray  # (N_rec, N_lig) Alvarez CSD contact sigma
    cooperative_alpha: float = 0.0  # CHN1/CHN2: cooperative H-bond correction weight
    extended_terms: CertifiedExtendedInteractionBundle = field(
        default_factory=CertifiedExtendedInteractionBundle
    )

    def tree_flatten(self):
        children = (
            self.screened_coulomb,
            self.contact,
            self.hbond_receptor_donor,
            self.hbond_ligand_donor,
            self.metal_coordination,
            self.pairwise_sigma,
            self.extended_terms,
        )
        aux_data = (self.cooperative_alpha,)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            screened_coulomb=children[0],
            contact=children[1],
            hbond_receptor_donor=children[2],
            hbond_ligand_donor=children[3],
            metal_coordination=children[4],
            pairwise_sigma=children[5],
            cooperative_alpha=aux_data[0],
            extended_terms=children[6],
        )

    def receptor_subset(self, indices: jnp.ndarray) -> "CertifiedRichChemistryPlan":
        return CertifiedRichChemistryPlan(
            screened_coulomb=self.screened_coulomb.receptor_subset(indices),
            contact=self.contact.receptor_subset(indices),
            hbond_receptor_donor=self.hbond_receptor_donor.receptor_subset(indices),
            hbond_ligand_donor=self.hbond_ligand_donor.receptor_subset(indices),
            metal_coordination=self.metal_coordination.receptor_subset(indices),
            pairwise_sigma=self.pairwise_sigma[indices],
            cooperative_alpha=self.cooperative_alpha,
            extended_terms=self.extended_terms.receptor_subset(indices),
        )

    def disambiguation_plan(self) -> "CertifiedRichChemistryPlan":
        """Plan used for theorem-backed certified pruning / winner filtering.

        The final certified top-1 winner is always chosen by the H-bond-backed
        orientation-disambiguation objective unless the richer chemistry score is
        proven to share the same singleton top-1 winner. Therefore the pruning
        objective only needs the exact channels that define this disambiguation
        score family: base physics (softened LJ + screened Coulomb) plus the two
        directional H-bond channels.
        """
        return CertifiedRichChemistryPlan(
            screened_coulomb=self.screened_coulomb,
            contact=self.contact.zeroed(),
            hbond_receptor_donor=self.hbond_receptor_donor,
            hbond_ligand_donor=self.hbond_ligand_donor,
            metal_coordination=self.metal_coordination.zeroed(),
            pairwise_sigma=self.pairwise_sigma,
            cooperative_alpha=0.0,
            extended_terms=CertifiedExtendedInteractionBundle(),
        )

    def default_softening_radius(self) -> float:
        return float(jnp.min(self.pairwise_sigma))

    def pruning_delta_budget(
        self,
        *,
        has_water_bridge_channel: bool = False,
    ) -> CertifiedPruningDeltaBudget:
        """Single-source theorem-backed pruning slack for rich chemistry."""
        cutoff_tail_delta = (
            self.screened_coulomb.analytic_cutoff_tail_bound()
            + self.contact.analytic_cutoff_tail_bound()
            + self.hbond_receptor_donor.analytic_cutoff_tail_bound()
            + self.hbond_ligand_donor.analytic_cutoff_tail_bound()
            + self.metal_coordination.analytic_cutoff_tail_bound()
            + sum(t.analytic_cutoff_tail_bound() for t in self.extended_terms.terms)
        )
        cooperative_bound = cooperative_hbond_correction_bound_sum_bounds(
            self.cooperative_alpha,
            (
                self.hbond_receptor_donor.max_total_score_bound(),
                self.hbond_ligand_donor.max_total_score_bound(),
            ),
        )
        water_bridge_bound = 2.0 if has_water_bridge_channel else 0.0
        cutoff_tail_handles = (
            screened_coulomb_theorem_handles()
            + contact_surrogate_theorem_handles()
            + directional_hbond_finite_theorem_handles()
            + tuple(
                dict.fromkeys(
                    metal_coordination_cutoff_theorem_handles()
                    + directional_metal_coordination_theorem_handles()
                )
            )
            + attractive_extended_chemistry_theorem_handles()
        )
        cooperative_active = bool(
            self.cooperative_alpha != 0.0
            and jax.device_get(self.has_active_directional_hbond)
        )
        water_active = bool(has_water_bridge_channel)
        components = (
            CertifiedPruningDeltaComponent(
                label="shared_base_zero",
                kind=CertifiedPruningDeltaComponentKind.SHARED_BASE,
                active=False,
                delta=0.0,
                theorem_handles=softened_lj_shared_base_delta_theorem_handles(),
                note=(
                    "Shared softened-LJ base contributes zero pruning slack on the rich-chemistry path"
                ),
            ),
            CertifiedPruningDeltaComponent(
                label="cutoff_tail",
                kind=CertifiedPruningDeltaComponentKind.CUTOFF_TAIL,
                active=cutoff_tail_delta > 0.0,
                delta=cutoff_tail_delta,
                theorem_handles=cutoff_tail_handles,
                note="Included active channel cutoff tails",
            ),
            CertifiedPruningDeltaComponent(
                label="cooperative_correction",
                kind=CertifiedPruningDeltaComponentKind.COOPERATIVE_CORRECTION,
                active=cooperative_active,
                delta=cooperative_bound if cooperative_active else 0.0,
                theorem_handles=tuple(
                    dict.fromkeys(
                        cooperative_hbond_theorem_handles()
                        + omitted_channel_bound_theorem_handles()
                    )
                ),
                note="Active cooperative H-bond correction omitted from pruning score",
            ),
            CertifiedPruningDeltaComponent(
                label="water_mediated",
                kind=CertifiedPruningDeltaComponentKind.WATER_MEDIATED,
                active=water_active,
                delta=water_bridge_bound if water_active else 0.0,
                theorem_handles=tuple(
                    dict.fromkeys(
                        attractive_water_mediated_hbond_theorem_handles()
                        + omitted_channel_bound_theorem_handles()
                    )
                ),
                note="Active water-mediated channel omitted from pruning score",
            ),
        )
        return CertifiedPruningDeltaBudget.from_components(
            source="rich_chemistry_pruning_delta",
            components=components,
            theorem_handles=rich_pruning_delta_theorem_handles(),
        )

    def analytic_total_delta(self, n_water_bridges: int = 0) -> float:
        return self.pruning_delta_budget(
            has_water_bridge_channel=n_water_bridges > 0,
        ).total_delta

    @property
    def has_active_directional_hbond(self) -> jnp.ndarray:
        return jnp.logical_or(
            self.hbond_receptor_donor.is_active,
            self.hbond_ligand_donor.is_active,
        )

    @property
    def has_active_metal_coordination(self) -> jnp.ndarray:
        return self.metal_coordination.is_active

    @property
    def has_active_extended_terms(self) -> jnp.ndarray:
        return self.extended_terms.has_active_terms


@register_pytree_node_class
@dataclass(frozen=True)
class CertifiedBatchResult:
    scores: jnp.ndarray
    error_bound: jax.Array | float
    target_error: jax.Array | float
    cutoff_radius: jax.Array | float
    posewise_error_bound: jnp.ndarray | None = None

    def __post_init__(self) -> None:
        if self.posewise_error_bound is None:
            object.__setattr__(
                self,
                "posewise_error_bound",
                jnp.full_like(
                    self.scores,
                    jnp.asarray(self.error_bound, dtype=self.scores.dtype),
                ),
            )

    def tree_flatten(self):
        children = (
            self.scores,
            self.error_bound,
            self.target_error,
            self.cutoff_radius,
            self.posewise_error_bound,
        )
        return (children, None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)

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


def _zero_certified_batch_result(poses_coords: jnp.ndarray) -> CertifiedBatchResult:
    zero = jnp.array(0.0, dtype=poses_coords.dtype)
    return CertifiedBatchResult(
        scores=jnp.zeros((poses_coords.shape[0],), dtype=poses_coords.dtype),
        error_bound=zero,
        target_error=zero,
        cutoff_radius=zero,
    )


def _branch_on_activity(
    activity: jnp.ndarray,
    *,
    active_fn: Callable[[], CertifiedBatchResult],
    inactive_fn: Callable[[], CertifiedBatchResult],
) -> CertifiedBatchResult:
    # When activity is a concrete value (not inside JIT trace), use Python branching
    # to avoid jax.lax.cond tracing both branches (which can fail on shape mismatches)
    if not isinstance(activity, jax.core.Tracer):
        return active_fn() if bool(jax.device_get(activity)) else inactive_fn()
    return jax.lax.cond(
        activity,
        lambda _: active_fn(),
        lambda _: inactive_fn(),
        operand=None,
    )


def _score_certified_interaction_active_batch(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    interaction_term: CertifiedOptionalInteractionTerm,
) -> CertifiedBatchResult:
    exact_scores = interaction_term.exact_scores(receptor_coords, poses_coords)
    coarse_scores = interaction_term.cutoff_scores(receptor_coords, poses_coords)
    posewise_error_bound = jnp.abs(exact_scores - coarse_scores)
    error_bound = jnp.max(posewise_error_bound)
    cutoff_radius = jnp.array(interaction_term.cutoff_radius, dtype=poses_coords.dtype)
    return CertifiedBatchResult(
        scores=coarse_scores,
        error_bound=error_bound,
        target_error=error_bound,
        cutoff_radius=cutoff_radius,
        posewise_error_bound=posewise_error_bound,
    )


def _score_certified_optional_interaction_batch(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    interaction_term: CertifiedOptionalInteractionTerm,
) -> CertifiedBatchResult:
    interaction_term.validate()
    return _branch_on_activity(
        interaction_term.is_active,
        active_fn=lambda: _score_certified_interaction_active_batch(
            receptor_coords,
            poses_coords,
            interaction_term,
        ),
        inactive_fn=lambda: _zero_certified_batch_result(poses_coords),
    )


def _score_certified_extended_interaction_bundle_batch(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    bundle: CertifiedExtendedInteractionBundle,
) -> CertifiedBatchResult:
    if not bundle.terms:
        return _zero_certified_batch_result(poses_coords)
    # When terms have been pre-filtered via filter_active(), skip the
    # per-call validate()+is_active overhead and score directly.
    batches = tuple(
        _score_certified_interaction_active_batch(
            receptor_coords,
            poses_coords,
            term,
        )
        for term in bundle.terms
    )
    return CertifiedBatchResult(
        scores=jnp.sum(jnp.stack([batch.scores for batch in batches], axis=0), axis=0),
        error_bound=sum(batch.error_bound for batch in batches),
        target_error=sum(batch.target_error for batch in batches),
        cutoff_radius=jnp.max(
            jnp.stack([jnp.asarray(batch.cutoff_radius) for batch in batches], axis=0)
        ),
        posewise_error_bound=jnp.sum(
            jnp.stack([batch.posewise_error_bound for batch in batches], axis=0),
            axis=0,
        ),
    )


@register_pytree_node_class
@dataclass(frozen=True)
class CertifiedSoftenedBatchResult:
    scores: jnp.ndarray
    softening_error_bound: jax.Array | float
    target_error: jax.Array | float
    cutoff_radius: jax.Array | float
    softening_radius: jax.Array | float
    posewise_softening_error_bound: jnp.ndarray | None = None

    def __post_init__(self) -> None:
        if self.posewise_softening_error_bound is None:
            object.__setattr__(
                self,
                "posewise_softening_error_bound",
                jnp.full_like(
                    self.scores,
                    jnp.asarray(
                        self.softening_error_bound,
                        dtype=self.scores.dtype,
                    ),
                ),
            )

    def tree_flatten(self):
        children = (
            self.scores,
            self.softening_error_bound,
            self.target_error,
            self.cutoff_radius,
            self.softening_radius,
            self.posewise_softening_error_bound,
        )
        return (children, None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)


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
    sigma_min = receptor_min + ligand_min
    return sigma_min


@conditionally_certified(
    "LatticeSum.lean::lj6_tail_bound; HandleAliases.lean::CB5; HandleAliases.lean::CB6; HandleAliases.lean::CB10; HandleAliases.lean::CB11; HandleAliases.lean::CB12; HandleAliases.lean::CB13; HandleAliases.lean::CB14",
    assumptions=[
        "Dispatch selects exactly one theorem-backed base-physics branch: certified LJ when electrostatics is absent, or the certified LJ plus conditionally certified real-space Ewald branch when electrostatics is present",
        "When electrostatics is present, the runtime uses the same exact and coarse real-space Ewald family as the Lean-backed Ewald branch",
    ],
)
def score_certified_batch(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    target_error: float,
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
    pairwise_sigma: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    diffs = receptor_coords[None, :, None, :] - poses_coords[:, None, :, :]
    dists = jnp.asarray(jnp.linalg.norm(diffs, axis=-1))

    sigma_ij = (
        pairwise_sigma
        if pairwise_sigma is not None
        else receptor_radii[:, None] + ligand_radii[None, :]
    )
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
    posewise_error_bound = jnp.sum(
        jnp.abs(exact_masked - softened_masked),
        axis=(1, 2),
    )

    if compute_error_bound:
        error_bound = jnp.max(posewise_error_bound)
    else:
        error_bound = jnp.array(0.0)
        posewise_error_bound = jnp.zeros_like(energies)

    return energies, error_bound, posewise_error_bound


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


# =============================================================================
# Directional Metal Coordination (Lean: DirectionalMetalCoordinationApproximation.lean)
# =============================================================================
# Subsumes the isotropic model: when angle_width >= 1e6, geometry → 1.0.
# DMC1: directionalMetalScore = radial × geometry (two-factor Lipschitz)
# DMC2: angular_factor_tightens_tail — angular ∈ [0,1] doesn't worsen cutoff error
# DMC3: directional_metal_cutoff_uniformApprox — finite-domain uniform approx
# MC1-MC5: isotropic metal coordination (MetalCoordinationApproximation.lean)


@jax.jit
def _score_directional_metal_exact_batch(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    receptor_strengths: jnp.ndarray,
    ligand_strengths: jnp.ndarray,
    ideal_distance: float,
    distance_width: float,
    receptor_ideal_angles: jnp.ndarray,
    angle_width: float,
) -> jnp.ndarray:
    """Directional metal coordination: radial Gaussian × angular Gaussian.

    radial(r) = exp(-((r - d_ideal) / σ_r)²)  ∈ [0, 1]
    angular(r) = exp(-((r - d_ideal - θ_i) / σ_θ)²)  ∈ [0, 1]
    score = -Σ w_rec · w_lig · radial(r) · angular(r)

    receptor_ideal_angles: (N_rec,) per-site geometry offset. For a pure
    tetrahedral metal, θ_i encodes how much each coordination site's preferred
    distance deviates from the global ideal.  When angle_width >= 1e6 the
    angular factor → 1.0 and this reduces to isotropic metal coordination.

    Certified by DMC1 (two-factor Lipschitz), DMC2 (angular tightens tail).
    """
    diffs = receptor_coords[None, :, None, :] - poses_coords[:, None, :, :]
    dists = jnp.linalg.norm(diffs, axis=-1)
    weight_matrix = receptor_strengths[None, :, None] * ligand_strengths[None, None, :]
    radial = jnp.exp(-(((dists - ideal_distance) / distance_width) ** 2))
    # Per-site angular deviation: how far each distance is from the site-specific ideal
    angular_deviation = dists - ideal_distance - receptor_ideal_angles[None, :, None]
    angular = jnp.exp(-((angular_deviation / angle_width) ** 2))
    contrib = weight_matrix * radial * angular
    return -jnp.sum(contrib, axis=(1, 2))


@jax.jit
def _score_directional_metal_cutoff_batch(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    receptor_strengths: jnp.ndarray,
    ligand_strengths: jnp.ndarray,
    ideal_distance: float,
    distance_width: float,
    receptor_ideal_angles: jnp.ndarray,
    angle_width: float,
    cutoff: float,
) -> jnp.ndarray:
    """Cutoff variant: zero beyond cutoff radius. DMC2 certifies angular ∈ [0,1]."""
    diffs = receptor_coords[None, :, None, :] - poses_coords[:, None, :, :]
    dists = jnp.linalg.norm(diffs, axis=-1)
    in_range = dists < cutoff
    weight_matrix = receptor_strengths[None, :, None] * ligand_strengths[None, None, :]
    radial = jnp.exp(-(((dists - ideal_distance) / distance_width) ** 2))
    angular_deviation = dists - ideal_distance - receptor_ideal_angles[None, :, None]
    angular = jnp.exp(-((angular_deviation / angle_width) ** 2))
    contrib = weight_matrix * radial * angular
    return -jnp.sum(jnp.where(in_range, contrib, 0.0), axis=(1, 2))


# =============================================================================
# Cosine Torsion Strain (Lean: LigandStrainApproximation.lean)
# =============================================================================
# LSA1: cosineTorsionStrain_le_twoVk — strain ∈ [0, 2Vk]
# LSA3: strain_preserves_uniformApprox — exact strain doesn't degrade error bound
# LSA5: additive_strain_bounded — multi-bond strain sums


@certified("LigandStrainApproximation.lean::cosineTorsionStrain_le_twoVk")
def cosine_torsion_strain(
    barrier_height: float,
    multiplicity: float,
    angle: float,
    phase: float,
) -> float:
    """V(φ) = Vk · (1 - cos(n·φ - φ₀)), bounded in [0, 2·Vk].

    Certified by LSA1 (upper bound) and LSA2 (nonneg).
    """
    return barrier_height * (1.0 - float(jnp.cos(multiplicity * angle - phase)))


@certified("LigandStrainApproximation.lean::additive_strain_bounded")
def total_torsion_strain(
    barrier_heights: jnp.ndarray,
    multiplicities: jnp.ndarray,
    angles: jnp.ndarray,
    phases: jnp.ndarray,
) -> float:
    """Sum of cosine torsion strain over all rotatable bonds.

    Bounded by Σ 2·Vk_i (LSA5: additive_strain_bounded).
    """
    strains = barrier_heights * (1.0 - jnp.cos(multiplicities * angles - phases))
    return float(jnp.sum(strains))


@certified("LigandStrainApproximation.lean::additive_strain_bounded")
def total_torsion_strain_bound(barrier_heights: jnp.ndarray) -> float:
    """Upper bound on total torsion strain: Σ 2·Vk_i."""
    return float(2.0 * jnp.sum(barrier_heights))


# =============================================================================
# Cooperative H-Bond Correction (Lean: CooperativeHBondApproximation.lean)
# =============================================================================
# CHN1: cooperative_correction_bounded — |α · Σ fᵢ·fⱼ| ≤ |α| · N²
# CHN2: independent_approximates_cooperative — UniformUtilityApprox


@certified(
    "CooperativeHBondApproximation.lean::cooperative_correction_bounded_of_abs_le"
)
def cooperative_hbond_correction_bound(
    alpha: float,
    n_hbonds: int,
    per_channel_abs_bound: float = 1.0,
) -> float:
    """Upper bound on cooperative correction: |α| · (N · B)^2.

    `B` is a certified per-channel absolute bound on each aggregated score.
    With `B=1` this recovers the classical |α|·N² bound.
    """
    return abs(alpha) * (n_hbonds * per_channel_abs_bound) ** 2


@certified(
    "CooperativeHBondApproximation.lean::cooperative_correction_bounded_of_abs_le_sum_bounds"
)
def cooperative_hbond_correction_bound_sum_bounds(
    alpha: float,
    per_channel_abs_bounds: Sequence[float],
) -> float:
    """Upper bound on cooperative correction from per-channel envelopes.

    If channel `i` satisfies `|f_i| <= B_i`, then
    `|alpha * (sum_i f_i)^2| <= |alpha| * (sum_i B_i)^2`.
    """
    total_bound = sum(max(0.0, float(bound)) for bound in per_channel_abs_bounds)
    return abs(alpha) * total_bound**2


def _cooperative_hbond_correction_bound_sum_bounds_jax(
    alpha: float,
    per_channel_abs_bounds: Sequence[jax.Array | float],
) -> jax.Array:
    total_bound = jnp.sum(
        jnp.stack(
            [jnp.maximum(0.0, jnp.asarray(bound)) for bound in per_channel_abs_bounds],
            axis=0,
        )
    )
    return jnp.abs(jnp.asarray(alpha, dtype=total_bound.dtype)) * total_bound**2


@jax.jit
def _cooperative_hbond_correction_batch(
    hbond_scores: jnp.ndarray,
    alpha: float,
) -> jnp.ndarray:
    """Cooperative correction: α · Σᵢ<ⱼ sᵢ·sⱼ for each pose.

    hbond_scores: (batch, n_hbonds) aggregated H-bond channel scores.
    Returns: (batch,) cooperative correction per pose.

    Certified by CHN5 (runtime-shape bound on α·(Σs)²), CHN2.
    """
    # Σᵢ Σⱼ sᵢ·sⱼ = (Σ sᵢ)² — use this for efficiency
    sums = jnp.sum(hbond_scores, axis=-1)
    pairwise_product_sum = sums**2
    return alpha * pairwise_product_sum


def _normalize_direction_vectors(vectors: jnp.ndarray) -> jnp.ndarray:
    norms = jnp.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / jnp.maximum(norms, 1e-6)


@jax.jit
def _batched_kabsch_rotations(
    reference_coords: jnp.ndarray,
    posed_coords: jnp.ndarray,
) -> jnp.ndarray:
    reference_centered = reference_coords - jnp.mean(
        reference_coords, axis=0, keepdims=True
    )
    posed_centered = posed_coords - jnp.mean(posed_coords, axis=1, keepdims=True)
    covariance = jnp.einsum("ni,bnj->bij", reference_centered, posed_centered)
    left, _, right_t = jnp.linalg.svd(covariance, full_matrices=False)
    right = jnp.swapaxes(right_t, -2, -1)
    left_t = jnp.swapaxes(left, -2, -1)
    det = jnp.linalg.det(right @ left_t)
    correction = jnp.broadcast_to(
        jnp.eye(3, dtype=reference_coords.dtype),
        (covariance.shape[0], 3, 3),
    )
    correction = correction.at[:, 2, 2].set(jnp.where(det < 0.0, -1.0, 1.0))
    return right @ correction @ left_t


@jax.jit
def _directional_hbond_pair_terms(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    receptor_anchor_indices: jnp.ndarray,
    receptor_directions: jnp.ndarray,
    ligand_anchor_indices: jnp.ndarray,
    ligand_local_directions: jnp.ndarray,
    ligand_frame_coords: jnp.ndarray,
    receptor_strengths: jnp.ndarray,
    ligand_strengths: jnp.ndarray,
    receptor_alignment_sign: float,
    ligand_alignment_sign: float,
    ideal_distance: float,
    distance_width: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    receptor_site_coords = receptor_coords[receptor_anchor_indices]
    ligand_site_coords = poses_coords[:, ligand_anchor_indices, :]
    diffs = ligand_site_coords[:, None, :, :] - receptor_site_coords[None, :, None, :]
    dists = jnp.asarray(jnp.linalg.norm(diffs, axis=-1))
    unit_vectors = diffs / jnp.maximum(dists[..., None], 1e-6)

    receptor_dirs = _normalize_direction_vectors(receptor_directions)
    rotations = _batched_kabsch_rotations(ligand_frame_coords, poses_coords)
    ligand_dirs = _normalize_direction_vectors(
        jnp.einsum("bij,nj->bni", rotations, ligand_local_directions)
    )
    receptor_dirs_expanded = receptor_dirs[None, :, None, :]
    ligand_dirs_expanded = ligand_dirs[:, None, :, :]

    pair_strength = jnp.clip(
        receptor_strengths[None, :, None] * ligand_strengths[None, None, :],
        0.0,
        1.0,
    )
    radial = pair_strength * jnp.exp(
        -(((dists - ideal_distance) / distance_width) ** 2)
    )
    receptor_angle = jnp.clip(
        jnp.sum(
            receptor_dirs_expanded * (receptor_alignment_sign * unit_vectors),
            axis=-1,
        ),
        0.0,
        1.0,
    )
    ligand_angle = jnp.clip(
        jnp.sum(
            ligand_dirs_expanded * (ligand_alignment_sign * unit_vectors),
            axis=-1,
        ),
        0.0,
        1.0,
    )
    return radial, receptor_angle, ligand_angle


@jax.jit
def _score_directional_hbond_exact_batch(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    receptor_anchor_indices: jnp.ndarray,
    receptor_directions: jnp.ndarray,
    ligand_anchor_indices: jnp.ndarray,
    ligand_local_directions: jnp.ndarray,
    ligand_frame_coords: jnp.ndarray,
    receptor_strengths: jnp.ndarray,
    ligand_strengths: jnp.ndarray,
    receptor_alignment_sign: float,
    ligand_alignment_sign: float,
    ideal_distance: float,
    distance_width: float,
) -> jnp.ndarray:
    radial, receptor_angle, ligand_angle = _directional_hbond_pair_terms(
        receptor_coords,
        poses_coords,
        receptor_anchor_indices,
        receptor_directions,
        ligand_anchor_indices,
        ligand_local_directions,
        ligand_frame_coords,
        receptor_strengths,
        ligand_strengths,
        receptor_alignment_sign,
        ligand_alignment_sign,
        ideal_distance,
        distance_width,
    )
    return jnp.sum(radial * receptor_angle * ligand_angle, axis=(1, 2))


@jax.jit
def _score_directional_hbond_cutoff_batch(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    receptor_anchor_indices: jnp.ndarray,
    receptor_directions: jnp.ndarray,
    ligand_anchor_indices: jnp.ndarray,
    ligand_local_directions: jnp.ndarray,
    ligand_frame_coords: jnp.ndarray,
    receptor_strengths: jnp.ndarray,
    ligand_strengths: jnp.ndarray,
    receptor_alignment_sign: float,
    ligand_alignment_sign: float,
    ideal_distance: float,
    distance_width: float,
    cutoff: float,
) -> jnp.ndarray:
    radial, receptor_angle, ligand_angle = _directional_hbond_pair_terms(
        receptor_coords,
        poses_coords,
        receptor_anchor_indices,
        receptor_directions,
        ligand_anchor_indices,
        ligand_local_directions,
        ligand_frame_coords,
        receptor_strengths,
        ligand_strengths,
        receptor_alignment_sign,
        ligand_alignment_sign,
        ideal_distance,
        distance_width,
    )
    receptor_site_coords = receptor_coords[receptor_anchor_indices]
    ligand_site_coords = poses_coords[:, ligand_anchor_indices, :]
    dists = jnp.asarray(
        jnp.linalg.norm(
            ligand_site_coords[:, None, :, :] - receptor_site_coords[None, :, None, :],
            axis=-1,
        )
    )
    pair_scores = radial * receptor_angle * ligand_angle
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
    return _score_certified_optional_interaction_batch(
        receptor_coords,
        poses_coords,
        contact_spec,
    )


@conditionally_certified(
    "HandleAliases.lean::DMC1; HandleAliases.lean::DMC2; HandleAliases.lean::DMC3; HandleAliases.lean::DMC4; HandleAliases.lean::DMC5",
    assumptions=[
        "The runtime uses the same directional metal-coordination score family as the Lean theorem: radial Gaussian times an angular factor in [0, 1]",
        "The reported error bound is the exact finite-batch max discrepancy between exact and cutoff metal coordination scores",
    ],
)
def score_certified_metal_coordination_batch(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    metal_spec: CertifiedMetalCoordinationSpec,
) -> CertifiedBatchResult:
    return _score_certified_optional_interaction_batch(
        receptor_coords,
        poses_coords,
        metal_spec,
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
    return _score_certified_optional_interaction_batch(
        receptor_coords,
        poses_coords,
        hbond_spec,
    )


@conditionally_certified(
    "HandleAliases.lean::PP6; HandleAliases.lean::PP7; HandleAliases.lean::PP8; HandleAliases.lean::PP9; HandleAliases.lean::PP10; "
    "PiStackingApproximation.lean::scaledPiStackingRadial_unitIntervalFactor; "
    "PiStackingApproximation.lean::scaledPiStackingRadial_lipschitz",
    assumptions=[
        "The runtime passes scaledRadial = strengths * radial as the radial argument, which "
        "scaledPiStackingRadial_unitIntervalFactor proves is a UnitIntervalFactor and "
        "scaledPiStackingRadial_lipschitz proves preserves the Lipschitz constant Lr unchanged; "
        "the remaining correspondence between Python PiStackingInteractionTerm and the Lean "
        "attractivePiStackingDecisionProblem family (face_alignment, offset_factor shape) is assumed",
        "The reported error bound is the exact finite-batch max discrepancy between exact and cutoff attractive pi-stacking scores",
    ],
)
def score_certified_pi_stacking_batch(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    pi_stacking_term: CertifiedOptionalInteractionTerm,
) -> CertifiedBatchResult:
    return _score_certified_optional_interaction_batch(
        receptor_coords,
        poses_coords,
        pi_stacking_term,
    )


@conditionally_certified(
    "HandleAliases.lean::PC6; HandleAliases.lean::PC7; HandleAliases.lean::PC8; HandleAliases.lean::PC9; HandleAliases.lean::PC10",
    assumptions=[
        "The runtime uses the exact same attractive pi-cation surrogate family as the Lean theorem family",
        "The reported error bound is the exact finite-batch max discrepancy between exact and cutoff attractive pi-cation scores",
    ],
)
def score_certified_pi_cation_batch(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    pi_cation_term: CertifiedOptionalInteractionTerm,
) -> CertifiedBatchResult:
    return _score_certified_optional_interaction_batch(
        receptor_coords,
        poses_coords,
        pi_cation_term,
    )


@conditionally_certified(
    "HandleAliases.lean::XB6; HandleAliases.lean::XB7; HandleAliases.lean::XB8; HandleAliases.lean::XB9; HandleAliases.lean::XB10",
    assumptions=[
        "The runtime uses the exact same attractive halogen-bond surrogate family as the Lean theorem family",
        "The reported error bound is the exact finite-batch max discrepancy between exact and cutoff attractive halogen-bond scores",
    ],
)
def score_certified_halogen_bond_batch(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    halogen_bond_term: CertifiedOptionalInteractionTerm,
) -> CertifiedBatchResult:
    return _score_certified_optional_interaction_batch(
        receptor_coords,
        poses_coords,
        halogen_bond_term,
    )


@conditionally_certified(
    "HandleAliases.lean::WB6; HandleAliases.lean::WB7; HandleAliases.lean::WB8; HandleAliases.lean::WB9; HandleAliases.lean::WB10",
    assumptions=[
        "The runtime uses the exact same attractive water-mediated hydrogen-bond surrogate family as the Lean theorem family",
        "The reported error bound is the exact finite-batch max discrepancy between exact and cutoff attractive water-mediated hydrogen-bond scores",
    ],
)
def score_certified_water_mediated_hbond_batch(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    water_mediated_hbond_term: CertifiedOptionalInteractionTerm,
) -> CertifiedBatchResult:
    return _score_certified_optional_interaction_batch(
        receptor_coords,
        poses_coords,
        water_mediated_hbond_term,
    )


def _combine_certified_batches(
    *batches: CertifiedBatchResult,
    target_error: float | None = None,
) -> CertifiedBatchResult:
    if not batches:
        raise ValueError("At least one certified batch is required")
    posewise_error_bound = jnp.sum(
        jnp.stack([batch.posewise_error_bound for batch in batches], axis=0),
        axis=0,
    )
    return CertifiedBatchResult(
        scores=jnp.sum(jnp.stack([batch.scores for batch in batches], axis=0), axis=0),
        error_bound=jnp.max(posewise_error_bound),
        target_error=(
            sum(batch.target_error for batch in batches)
            if target_error is None
            else target_error
        ),
        cutoff_radius=jnp.max(
            jnp.stack([jnp.asarray(batch.cutoff_radius) for batch in batches], axis=0)
        ),
        posewise_error_bound=posewise_error_bound,
    )


@conditionally_certified(
    "HandleAliases.lean::XR1; HandleAliases.lean::XR2; HandleAliases.lean::XR3; HandleAliases.lean::XR4; HandleAliases.lean::XR5",
    assumptions=[
        "The attractive runtime chemistry energy is exactly the sum of the theorem-backed attractive contact, receptor-donor directional H-bond, ligand-donor directional H-bond, metal coordination, pi-stacking, pi-cation, halogen-bond, and water-mediated hydrogen-bond signals",
        "The combined error bound is the sum of the finite-batch discrepancy bounds for all attractive chemistry components",
    ],
)
def score_certified_attractive_chemistry_batch(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    rich_chemistry_plan: CertifiedRichChemistryPlan,
    *,
    cooperative_channel_abs_bounds: tuple[float, ...] | None = None,
) -> CertifiedBatchResult:
    contact_batch = score_certified_contact_batch(
        receptor_coords,
        poses_coords,
        rich_chemistry_plan.contact,
    )
    hbond_receptor_donor_batch = score_certified_directional_hbond_batch(
        receptor_coords,
        poses_coords,
        rich_chemistry_plan.hbond_receptor_donor,
    )
    hbond_ligand_donor_batch = score_certified_directional_hbond_batch(
        receptor_coords,
        poses_coords,
        rich_chemistry_plan.hbond_ligand_donor,
    )
    metal_batch = score_certified_metal_coordination_batch(
        receptor_coords,
        poses_coords,
        rich_chemistry_plan.metal_coordination,
    )
    extended_batch = _score_certified_extended_interaction_bundle_batch(
        receptor_coords,
        poses_coords,
        rich_chemistry_plan.extended_terms,
    )
    # CHN5/CHN2: cooperative H-bond correction α·(Σs)²
    # hbond scores are raw (positive = attractive), stack for cooperative calc
    cooperative_correction = _cooperative_hbond_correction_batch(
        jnp.stack(
            [hbond_receptor_donor_batch.scores, hbond_ligand_donor_batch.scores],
            axis=-1,
        ),
        rich_chemistry_plan.cooperative_alpha,
    )
    if cooperative_channel_abs_bounds is None:
        cooperative_channel_abs_bounds = (
            jnp.sum(
                jnp.clip(
                    rich_chemistry_plan.hbond_receptor_donor.receptor_strengths[:, None]
                    * rich_chemistry_plan.hbond_receptor_donor.ligand_strengths[
                        None, :
                    ],
                    0.0,
                    1.0,
                )
            ),
            jnp.sum(
                jnp.clip(
                    rich_chemistry_plan.hbond_ligand_donor.receptor_strengths[:, None]
                    * rich_chemistry_plan.hbond_ligand_donor.ligand_strengths[None, :],
                    0.0,
                    1.0,
                )
            ),
        )
    cooperative_error = _cooperative_hbond_correction_bound_sum_bounds_jax(
        rich_chemistry_plan.cooperative_alpha,
        cooperative_channel_abs_bounds,
    )
    return _combine_certified_batches(
        CertifiedBatchResult(
            scores=-(
                contact_batch.scores
                + hbond_receptor_donor_batch.scores
                + hbond_ligand_donor_batch.scores
            )
            + cooperative_correction,
            error_bound=(
                contact_batch.error_bound
                + hbond_receptor_donor_batch.error_bound
                + hbond_ligand_donor_batch.error_bound
                + cooperative_error
            ),
            target_error=(
                contact_batch.target_error
                + hbond_receptor_donor_batch.target_error
                + hbond_ligand_donor_batch.target_error
            ),
            cutoff_radius=jnp.max(
                jnp.stack(
                    [
                        jnp.asarray(contact_batch.cutoff_radius),
                        jnp.asarray(hbond_receptor_donor_batch.cutoff_radius),
                        jnp.asarray(hbond_ligand_donor_batch.cutoff_radius),
                    ],
                    axis=0,
                )
            ),
            posewise_error_bound=(
                contact_batch.posewise_error_bound
                + hbond_receptor_donor_batch.posewise_error_bound
                + hbond_ligand_donor_batch.posewise_error_bound
                + jnp.full_like(contact_batch.scores, cooperative_error)
            ),
        ),
        metal_batch,
        extended_batch,
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
    target_error: float,
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
    posewise_error_bound = (
        jnp.full_like(screened_batch.scores, lj_error_bound)
        + screened_batch.posewise_error_bound
    )
    return CertifiedBatchResult(
        scores=lj_scores + screened_batch.scores,
        error_bound=jnp.max(posewise_error_bound),
        target_error=target_error,
        cutoff_radius=combined_cutoff,
        posewise_error_bound=posewise_error_bound,
    )


@conditionally_certified(
    "HandleAliases.lean::XR6; HandleAliases.lean::XR7; HandleAliases.lean::XR8; HandleAliases.lean::XR9; HandleAliases.lean::XR10",
    assumptions=[
        "The additive runtime score is exactly the sum of the certified exact LJ-screened-Coulomb term and the certified attractive extended chemistry term",
        "The combined error bound is the sum of the certified exact LJ-screened-Coulomb bound and the certified attractive extended chemistry batch discrepancy bound",
    ],
)
def score_certified_rich_chemistry_batch(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    rich_chemistry_plan: CertifiedRichChemistryPlan,
    target_error: float,
    epsilon: float = _EPSILON_KCAL_MOL,
    *,
    cooperative_channel_abs_bounds: tuple[float, ...] | None = None,
) -> CertifiedBatchResult:
    nonbonded_batch = score_certified_lj_screened_coulomb_batch(
        receptor_coords=receptor_coords,
        poses_coords=poses_coords,
        receptor_radii=receptor_radii,
        ligand_radii=ligand_radii,
        screened_coulomb=rich_chemistry_plan.screened_coulomb,
        target_error=target_error,
        epsilon=epsilon,
    )
    attractive_batch = score_certified_attractive_chemistry_batch(
        receptor_coords,
        poses_coords,
        rich_chemistry_plan,
        cooperative_channel_abs_bounds=cooperative_channel_abs_bounds,
    )
    combined_cutoff = jnp.maximum(
        nonbonded_batch.cutoff_radius,
        attractive_batch.cutoff_radius,
    )
    posewise_error_bound = (
        nonbonded_batch.posewise_error_bound + attractive_batch.posewise_error_bound
    )
    return CertifiedBatchResult(
        scores=nonbonded_batch.scores + attractive_batch.scores,
        error_bound=jnp.max(posewise_error_bound),
        target_error=target_error,
        cutoff_radius=combined_cutoff,
        posewise_error_bound=posewise_error_bound,
    )


def score_certified_softened_rich_chemistry_batch(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    rich_chemistry_plan: CertifiedRichChemistryPlan,
    target_error: float,
    epsilon: float = _EPSILON_KCAL_MOL,
    softening_radius: float | None = None,
) -> CertifiedSoftenedBatchResult:
    softened_lj = score_certified_softened_lj(
        receptor_coords=receptor_coords,
        poses_coords=poses_coords,
        receptor_radii=receptor_radii,
        ligand_radii=ligand_radii,
        target_error=target_error,
        epsilon=epsilon,
        softening_radius=softening_radius,
        pairwise_sigma=rich_chemistry_plan.pairwise_sigma,
    )
    screened_batch = score_certified_screened_coulomb_batch(
        receptor_coords,
        poses_coords,
        rich_chemistry_plan.screened_coulomb,
    )
    attractive_batch = score_certified_attractive_chemistry_batch(
        receptor_coords,
        poses_coords,
        rich_chemistry_plan,
    )
    return CertifiedSoftenedBatchResult(
        scores=softened_lj.scores + screened_batch.scores + attractive_batch.scores,
        softening_error_bound=softened_lj.softening_error_bound
        + screened_batch.error_bound
        + attractive_batch.error_bound,
        target_error=target_error,
        cutoff_radius=jnp.maximum(
            jnp.maximum(softened_lj.cutoff_radius, screened_batch.cutoff_radius),
            attractive_batch.cutoff_radius,
        ),
        softening_radius=softened_lj.softening_radius,
        posewise_softening_error_bound=(
            softened_lj.posewise_softening_error_bound
            + screened_batch.posewise_error_bound
            + attractive_batch.posewise_error_bound
        ),
    )


@certified("LatticeSum.lean::lj6_tail_bound")
def score_certified_lj(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    target_error: float,
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
        target_error: Target error bound per atom pair
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
    target_error: float,
    epsilon: float = _EPSILON_KCAL_MOL,
    softening_radius: float | None = None,
    compute_error_bound: bool = True,
    pairwise_sigma: jnp.ndarray | None = None,
) -> CertifiedSoftenedBatchResult:
    cutoff = jnp.array(optimal_cutoff(target_error, s=6.0))
    r_soft = (
        _default_softening_radius(receptor_radii, ligand_radii)
        if softening_radius is None
        else softening_radius
    )
    scores, softening_error_bound, posewise_softening_error_bound = (
        _score_certified_softened_lj_batch(
            receptor_coords,
            poses_coords,
            receptor_radii,
            ligand_radii,
            cutoff,
            epsilon,
            r_soft,
            compute_error_bound=compute_error_bound,
            pairwise_sigma=pairwise_sigma,
        )
    )
    return CertifiedSoftenedBatchResult(
        scores=scores,
        softening_error_bound=softening_error_bound,
        target_error=target_error,
        cutoff_radius=cutoff,
        softening_radius=r_soft,
        posewise_softening_error_bound=posewise_softening_error_bound,
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
    target_error: float,
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
        posewise_softening_error_bound=softened_batch.posewise_softening_error_bound,
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
    target_error: float,
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
            if "target_error" not in kwargs:
                raise ValueError(
                    "CERTIFIED_LJ scoring requires an explicit derived target_error"
                )
            target_error = kwargs["target_error"]
            scores, error_bound = score_certified_lj(
                kwargs["receptor_coords"],
                kwargs["poses_coords"],
                kwargs["receptor_radii"],
                kwargs["ligand_radii"],
                target_error=target_error,
            )
            return np.array(scores)

        case ScoringEngine.CERTIFIED_LJ_REALSPACE_EWALD:
            if "target_error" not in kwargs:
                raise ValueError(
                    "CERTIFIED_LJ_REALSPACE_EWALD scoring requires an explicit derived target_error"
                )
            target_error = kwargs["target_error"]
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
