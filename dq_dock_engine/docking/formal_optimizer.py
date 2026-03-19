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
    CertifiedPriorSpec,
    build_prior,
    select_admissible_action,
    update_posterior,
)
from dq_dock_engine.docking.formal_pruning import (
    ambiguity_band_mask,
    certified_survivor_mask,
)
from dq_dock_engine.docking.formal_surrogates import (
    CertifiedCoarseScoreBundle,
    score_exact_and_coarse_round,
)


@dataclass(frozen=True)
class CertifiedOptimizerState:
    coords: jax.Array
    action_family: CertifiedActionFamily
    belief: CertifiedBeliefState
    retained_receptor_indices: jax.Array


def _build_belief_state(
    bundle: CertifiedCoarseScoreBundle,
    prior_spec: CertifiedPriorSpec,
    step_index: int,
) -> CertifiedBeliefState:
    prior = build_prior(prior_spec, len(bundle.exact_scores))
    posterior = update_posterior(prior, bundle.survivor_mask)
    selected_action = select_admissible_action(posterior, bundle.ambiguity_mask)
    return CertifiedBeliefState(
        prior_spec=prior_spec,
        prior=prior,
        posterior=posterior,
        coarse_scores=bundle.coarse_scores,
        exact_scores=bundle.exact_scores,
        exact_error_bound=bundle.exact_error_bound,
        survivor_mask=bundle.survivor_mask,
        ambiguity_mask=bundle.ambiguity_mask,
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
    prior_spec: CertifiedPriorSpec,
    max_coarse_receptor_atoms: int,
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
    (
        exact_scores_matrix,
        coarse_scores_matrix,
        delta,
        exact_error_bound,
        retained_indices,
    ) = score_exact_and_coarse_round(
        receptor_coords=receptor_coords,
        receptor_radii=receptor_radii,
        ligand_radii=ligand_radii,
        candidate_batches=candidate_batches,
        target_error=target_error,
        max_receptor_atoms=max_coarse_receptor_atoms,
    )
    next_coords = []
    states = []
    for pose_index in range(candidate_batches.shape[0]):
        survivor_mask = certified_survivor_mask(
            exact_scores=exact_scores_matrix[pose_index],
            coarse_scores=coarse_scores_matrix[pose_index],
            k=1,
            delta=delta,
        )
        ambiguity_mask = ambiguity_band_mask(
            exact_scores_matrix[pose_index],
            k=1,
            epsilon=delta,
        )
        bundle = CertifiedCoarseScoreBundle(
            exact_scores=exact_scores_matrix[pose_index],
            coarse_scores=coarse_scores_matrix[pose_index],
            delta=delta,
            exact_error_bound=exact_error_bound,
            survivor_mask=survivor_mask,
            ambiguity_mask=ambiguity_mask,
            retained_receptor_indices=retained_indices,
        )
        belief = _build_belief_state(
            bundle=bundle,
            prior_spec=prior_spec,
            step_index=round_index,
        )
        selected_coords = candidate_batches[pose_index, belief.selected_action]
        next_coords.append(selected_coords)
        states.append(
            CertifiedOptimizerState(
                coords=selected_coords,
                action_family=action_family,
                belief=belief,
                retained_receptor_indices=bundle.retained_receptor_indices,
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
    prior_spec: CertifiedPriorSpec | None = None,
    max_coarse_receptor_atoms: int = 64,
) -> tuple[jax.Array, tuple[tuple[CertifiedOptimizerState, ...], ...]]:
    current_coords = coords_batch
    history: list[tuple[CertifiedOptimizerState, ...]] = []
    effective_prior_spec = (
        CertifiedPriorSpec(kind="uniform") if prior_spec is None else prior_spec
    )

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
            prior_spec=effective_prior_spec,
            max_coarse_receptor_atoms=max_coarse_receptor_atoms,
        )
        history.append(states)

        if all(action_state.belief.selected_action == 0 for action_state in states):
            break

    return current_coords, tuple(history)
