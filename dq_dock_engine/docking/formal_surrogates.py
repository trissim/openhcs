from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from dq_dock_engine.docking.formal_pruning import (
    CertifiedPruningCertificate,
    certified_pruning_certificate,
)
from dq_dock_engine.docking.scoring import optimal_cutoff, score_certified_batch


@dataclass(frozen=True)
class CertifiedCoarseScoreBundle:
    exact_scores: jax.Array
    coarse_scores: jax.Array
    delta: float
    exact_error_bound: float
    coarse_error_bound: float
    survivor_mask: jax.Array
    ambiguity_mask: jax.Array
    retained_receptor_indices: jax.Array
    pruning_certificate: CertifiedPruningCertificate


def select_trimmed_receptor_subset(
    receptor_coords: jax.Array,
    reference_coords: jax.Array,
    max_receptor_atoms: int,
) -> jax.Array:
    if max_receptor_atoms <= 0:
        raise ValueError("max_receptor_atoms must be positive")
    if max_receptor_atoms >= receptor_coords.shape[0]:
        return jnp.arange(receptor_coords.shape[0])

    reference_center = jnp.mean(reference_coords, axis=0)
    receptor_distances = jnp.linalg.norm(receptor_coords - reference_center, axis=1)
    return jnp.argsort(receptor_distances)[:max_receptor_atoms]


def select_trimmed_receptor_subset_for_batch(
    receptor_coords: jax.Array,
    reference_coords_batch: jax.Array,
    max_receptor_atoms: int,
) -> jax.Array:
    if max_receptor_atoms <= 0:
        raise ValueError("max_receptor_atoms must be positive")
    if max_receptor_atoms >= receptor_coords.shape[0]:
        return jnp.arange(receptor_coords.shape[0])

    batch_center = jnp.mean(reference_coords_batch, axis=(0, 1))
    receptor_distances = jnp.linalg.norm(receptor_coords - batch_center, axis=1)
    return jnp.argsort(receptor_distances)[:max_receptor_atoms]


def select_exact_receptor_subset_for_local_family(
    receptor_coords: jax.Array,
    receptor_radii: jax.Array,
    reference_coords_batch: jax.Array,
    ligand_radii: jax.Array,
    translation_step: float,
    target_error: float,
) -> jax.Array:
    """Return receptor atoms that can still interact with the sampled local family.

    This is the runtime geometric realization of the sampled inside-cutoff
    sufficiency bridge used by the formal docking path: atoms outside the family
    cutoff support are dropped before exact certified scoring.
    """
    if reference_coords_batch.ndim != 3:
        raise ValueError("reference_coords_batch must have shape (N, M, 3)")

    receptor_coords_np = np.asarray(receptor_coords)
    receptor_radii_np = np.asarray(receptor_radii)
    reference_coords_np = np.asarray(reference_coords_batch)
    ligand_radii_np = np.asarray(ligand_radii)

    pose_centers = np.mean(reference_coords_np, axis=1)
    ligand_extents = np.max(
        np.linalg.norm(reference_coords_np - pose_centers[:, None, :], axis=-1),
        axis=1,
    )
    cutoff = optimal_cutoff(target_error, s=6.0)
    max_ligand_radius = float(np.max(ligand_radii_np))

    center_distances = np.linalg.norm(
        receptor_coords_np[:, None, :] - pose_centers[None, :, :], axis=-1
    )
    safe_cutoff = np.maximum(cutoff, receptor_radii_np[:, None] + max_ligand_radius)
    support_radius = ligand_extents[None, :] + translation_step + safe_cutoff
    keep_mask = np.any(center_distances <= support_radius, axis=1)

    if not bool(np.any(keep_mask)):
        closest_index = int(np.argmin(np.min(center_distances, axis=1)))
        return jnp.array([closest_index], dtype=jnp.int32)

    return jnp.array(np.flatnonzero(keep_mask), dtype=jnp.int32)


def score_exact_and_coarse_local_family(
    receptor_coords: jax.Array,
    receptor_radii: jax.Array,
    ligand_radii: jax.Array,
    candidate_coords: jax.Array,
    target_error: float,
    max_receptor_atoms: int,
    translation_step: float,
) -> CertifiedCoarseScoreBundle:
    retained_indices = select_exact_receptor_subset_for_local_family(
        receptor_coords=receptor_coords,
        receptor_radii=receptor_radii,
        reference_coords_batch=candidate_coords,
        ligand_radii=ligand_radii,
        translation_step=translation_step,
        target_error=target_error,
    )
    exact_batch = score_certified_batch(
        receptor_coords=receptor_coords[retained_indices],
        poses_coords=candidate_coords,
        receptor_radii=receptor_radii[retained_indices],
        ligand_radii=ligand_radii,
        target_error=target_error,
    )
    coarse_scores = exact_batch.scores
    delta = 0.0
    pruning_certificate = certified_pruning_certificate(
        exact_scores=exact_batch.scores,
        coarse_scores=coarse_scores,
        k=1,
        delta=delta,
    )
    return CertifiedCoarseScoreBundle(
        exact_scores=exact_batch.scores,
        coarse_scores=coarse_scores,
        delta=delta,
        exact_error_bound=exact_batch.error_bound,
        coarse_error_bound=exact_batch.error_bound,
        survivor_mask=pruning_certificate.survivor_mask,
        ambiguity_mask=pruning_certificate.exact_ambiguity_mask,
        retained_receptor_indices=retained_indices,
        pruning_certificate=pruning_certificate,
    )


def score_exact_and_coarse_round(
    receptor_coords: jax.Array,
    receptor_radii: jax.Array,
    ligand_radii: jax.Array,
    candidate_batches: jax.Array,
    target_error: float,
    max_receptor_atoms: int,
    translation_step: float,
) -> tuple[jax.Array, jax.Array, float, float, jax.Array]:
    n_poses, n_actions, n_atoms, _ = candidate_batches.shape
    flat_candidates = candidate_batches.reshape((n_poses * n_actions, n_atoms, 3))
    retained_indices = select_exact_receptor_subset_for_local_family(
        receptor_coords=receptor_coords,
        receptor_radii=receptor_radii,
        reference_coords_batch=candidate_batches[:, 0, :, :],
        ligand_radii=ligand_radii,
        translation_step=translation_step,
        target_error=target_error,
    )
    exact_batch = score_certified_batch(
        receptor_coords=receptor_coords[retained_indices],
        poses_coords=flat_candidates,
        receptor_radii=receptor_radii[retained_indices],
        ligand_radii=ligand_radii,
        target_error=target_error,
    )
    exact_scores = exact_batch.scores.reshape((n_poses, n_actions))
    coarse_scores = exact_scores
    delta = 0.0
    return (
        exact_scores,
        coarse_scores,
        delta,
        exact_batch.error_bound,
        retained_indices,
    )
