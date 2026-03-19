from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from dq_dock_engine.docking.formal_actions import (
    CertifiedActionFamily,
    apply_action_family,
    create_certified_action_family,
)
from dq_dock_engine.docking.formal_belief import (
    CertifiedBeliefState,
    select_admissible_action,
    uniform_prior,
    update_posterior,
)
from dq_dock_engine.docking.formal_pruning import (
    ambiguity_band_mask,
    certified_survivor_mask,
)
from dq_dock_engine.docking.scoring import score_certified_batch


@dataclass(frozen=True)
class CertifiedOptimizerState:
    coords: jax.Array
    action_family: CertifiedActionFamily
    belief: CertifiedBeliefState


def _build_belief_state(
    exact_scores: jax.Array,
    exact_error_bound: float,
    step_index: int,
) -> CertifiedBeliefState:
    coarse_scores = exact_scores
    survivor_mask = certified_survivor_mask(
        exact_scores=exact_scores,
        coarse_scores=coarse_scores,
        k=1,
        delta=0.0,
    )
    ambiguity_mask = ambiguity_band_mask(
        exact_scores, k=1, epsilon=2.0 * exact_error_bound
    )
    prior = uniform_prior(len(exact_scores))
    posterior = update_posterior(prior, survivor_mask)
    selected_action = select_admissible_action(posterior, ambiguity_mask)
    return CertifiedBeliefState(
        prior=prior,
        posterior=posterior,
        coarse_scores=coarse_scores,
        exact_scores=exact_scores,
        exact_error_bound=exact_error_bound,
        survivor_mask=survivor_mask,
        ambiguity_mask=ambiguity_mask,
        selected_action=selected_action,
        step_index=step_index,
    )


def _translation_step_for_round(base_step: float, round_index: int) -> float:
    return base_step / float(2**round_index)


def _rotation_step_for_round(base_step: float, round_index: int) -> float:
    return base_step / float(2**round_index)


def _refine_round(
    coords_batch: jax.Array,
    receptor_coords: jax.Array,
    receptor_radii: jax.Array,
    ligand_radii: jax.Array,
    target_error: float,
    round_index: int,
    base_translation_step: float,
    base_rotation_step_rad: float,
) -> tuple[jax.Array, tuple[CertifiedOptimizerState, ...]]:
    action_family = create_certified_action_family(
        translation_step=_translation_step_for_round(
            base_translation_step, round_index
        ),
        rotation_step_rad=_rotation_step_for_round(base_rotation_step_rad, round_index),
        stencil_level=round_index,
    )

    candidate_batches = jax.vmap(apply_action_family, in_axes=(0, None))(
        coords_batch, action_family
    )
    n_poses, n_actions, n_atoms, _ = candidate_batches.shape
    flat_candidates = candidate_batches.reshape((n_poses * n_actions, n_atoms, 3))
    scored = score_certified_batch(
        receptor_coords=receptor_coords,
        poses_coords=flat_candidates,
        receptor_radii=receptor_radii,
        ligand_radii=ligand_radii,
        target_error=target_error,
    )
    exact_scores = scored.scores.reshape((n_poses, n_actions))

    next_coords = []
    states = []
    for pose_index in range(n_poses):
        belief = _build_belief_state(
            exact_scores=exact_scores[pose_index],
            exact_error_bound=scored.error_bound,
            step_index=round_index,
        )
        selected_coords = candidate_batches[pose_index, belief.selected_action]
        next_coords.append(selected_coords)
        states.append(
            CertifiedOptimizerState(
                coords=selected_coords,
                action_family=action_family,
                belief=belief,
            )
        )

    return jnp.stack(next_coords, axis=0), tuple(states)


def refine_poses_certified(
    coords_batch: jax.Array,
    receptor_coords: jax.Array,
    receptor_radii: jax.Array,
    ligand_radii: jax.Array,
    n_rounds: int,
    target_error: float,
    base_translation_step: float = 0.5,
    base_rotation_step_rad: float = jnp.pi / 12.0,
) -> tuple[jax.Array, tuple[tuple[CertifiedOptimizerState, ...], ...]]:
    current_coords = coords_batch
    history: list[tuple[CertifiedOptimizerState, ...]] = []

    for round_index in range(n_rounds):
        current_coords, states = _refine_round(
            coords_batch=current_coords,
            receptor_coords=receptor_coords,
            receptor_radii=receptor_radii,
            ligand_radii=ligand_radii,
            target_error=target_error,
            round_index=round_index,
            base_translation_step=base_translation_step,
            base_rotation_step_rad=float(base_rotation_step_rad),
        )
        history.append(states)

        if all(action_state.belief.selected_action == 0 for action_state in states):
            break

    return current_coords, tuple(history)
