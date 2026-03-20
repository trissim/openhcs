"""
Scoring mechanism using strict OpenHCS Enum dispatch.

Separates JAX-native internal physics from impure SMINA external subprocess wrapper.

PROOF STATUS SUMMARY:
  - score_internal_lj: HEURISTIC (ad-hoc weights)
  - score_smina_exact: HEURISTIC (external unverified binary)
  - route_scoring: HEURISTIC (dispatch only)
"""

from dataclasses import dataclass
from typing import List, Dict, Callable
import os
import subprocess
import tempfile
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from dq_dock_engine.proof_status import certified, heuristic, ProofStatus
from dq_dock_engine.docking.core import ScoringEngine, ScoredPose, GapCertification
from dq_dock_engine.physics.lattice_sum import optimal_cutoff, lj6_cutoff_error
from dq_dock_engine.physics.kernels import typed_lennard_jones_matrix

_EPSILON_KCAL_MOL = 0.086


@dataclass(frozen=True)
class CertifiedBatchResult:
    scores: jnp.ndarray
    error_bound: float
    target_error: float
    cutoff_radius: float

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


def score_certified_batch(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    target_error: float = 0.001,
    epsilon: float = _EPSILON_KCAL_MOL,
) -> CertifiedBatchResult:
    scores, error_bound = score_certified_lj(
        receptor_coords,
        poses_coords,
        receptor_radii,
        ligand_radii,
        target_error=target_error,
        epsilon=epsilon,
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
    lj_contrib = typed_lennard_jones_matrix(dists_safe, epsilon_matrix, sigma_ij)

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
    error_bound = epsilon * lj6_cutoff_error(float(cutoff))

    return scores, error_bound


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

        case _:
            raise ValueError(f"Unknown ScoringEngine: {engine}")
