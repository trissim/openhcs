from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from dq_dock_engine.docking.formal_pruning import (
    ambiguity_band_mask,
    certified_survivor_mask,
)
from dq_dock_engine.docking.scoring import score_certified_batch


@dataclass(frozen=True)
class CertifiedCoarseScoreBundle:
    exact_scores: jax.Array
    coarse_scores: jax.Array
    delta: float
    exact_error_bound: float
    survivor_mask: jax.Array
    ambiguity_mask: jax.Array
    retained_receptor_indices: jax.Array


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


def score_exact_and_coarse_local_family(
    receptor_coords: jax.Array,
    receptor_radii: jax.Array,
    ligand_radii: jax.Array,
    candidate_coords: jax.Array,
    target_error: float,
    max_receptor_atoms: int,
) -> CertifiedCoarseScoreBundle:
    exact_batch = score_certified_batch(
        receptor_coords=receptor_coords,
        poses_coords=candidate_coords,
        receptor_radii=receptor_radii,
        ligand_radii=ligand_radii,
        target_error=target_error,
    )
    retained_indices = select_trimmed_receptor_subset(
        receptor_coords=receptor_coords,
        reference_coords=candidate_coords[0],
        max_receptor_atoms=max_receptor_atoms,
    )
    coarse_batch = score_certified_batch(
        receptor_coords=receptor_coords[retained_indices],
        poses_coords=candidate_coords,
        receptor_radii=receptor_radii[retained_indices],
        ligand_radii=ligand_radii,
        target_error=target_error,
    )
    delta = float(jnp.max(jnp.abs(exact_batch.scores - coarse_batch.scores)))
    survivor_mask = certified_survivor_mask(
        exact_scores=exact_batch.scores,
        coarse_scores=coarse_batch.scores,
        k=1,
        delta=delta,
    )
    ambiguity_mask = ambiguity_band_mask(exact_batch.scores, k=1, epsilon=delta)
    return CertifiedCoarseScoreBundle(
        exact_scores=exact_batch.scores,
        coarse_scores=coarse_batch.scores,
        delta=delta,
        exact_error_bound=exact_batch.error_bound,
        survivor_mask=survivor_mask,
        ambiguity_mask=ambiguity_mask,
        retained_receptor_indices=retained_indices,
    )


def score_exact_and_coarse_round(
    receptor_coords: jax.Array,
    receptor_radii: jax.Array,
    ligand_radii: jax.Array,
    candidate_batches: jax.Array,
    target_error: float,
    max_receptor_atoms: int,
) -> tuple[jax.Array, jax.Array, float, float, jax.Array]:
    n_poses, n_actions, n_atoms, _ = candidate_batches.shape
    flat_candidates = candidate_batches.reshape((n_poses * n_actions, n_atoms, 3))
    exact_batch = score_certified_batch(
        receptor_coords=receptor_coords,
        poses_coords=flat_candidates,
        receptor_radii=receptor_radii,
        ligand_radii=ligand_radii,
        target_error=target_error,
    )
    retained_indices = select_trimmed_receptor_subset_for_batch(
        receptor_coords=receptor_coords,
        reference_coords_batch=candidate_batches[:, 0, :, :],
        max_receptor_atoms=max_receptor_atoms,
    )
    coarse_batch = score_certified_batch(
        receptor_coords=receptor_coords[retained_indices],
        poses_coords=flat_candidates,
        receptor_radii=receptor_radii[retained_indices],
        ligand_radii=ligand_radii,
        target_error=target_error,
    )
    exact_scores = exact_batch.scores.reshape((n_poses, n_actions))
    coarse_scores = coarse_batch.scores.reshape((n_poses, n_actions))
    delta = float(jnp.max(jnp.abs(exact_scores - coarse_scores)))
    return (
        exact_scores,
        coarse_scores,
        delta,
        exact_batch.error_bound,
        retained_indices,
    )
