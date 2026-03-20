from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from dq_dock_engine.docking.formal_actions import (
    CertifiedActionFamily,
    apply_action_family,
    apply_action_family_batch,
    create_certified_action_family,
)
from dq_dock_engine.docking.formal_belief import (
    CertifiedBeliefState,
    CertifiedPriorSpec,
    build_prior,
    select_admissible_actions,
    select_admissible_action,
    update_posterior_batch,
    update_posterior,
)
from dq_dock_engine.docking.formal_pruning import (
    ambiguity_band_mask,
    certified_survivor_mask,
    CertifiedPruningCertificate,
    top_k_with_ties_mask,
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


def _noop_is_unique_admissible_action(state: CertifiedOptimizerState) -> bool:
    ambiguity_mask = state.belief.ambiguity_mask
    return (
        state.belief.selected_action == 0
        and bool(ambiguity_mask[0])
        and int(jnp.sum(ambiguity_mask.astype(jnp.int32))) == 1
    )


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

    candidate_batches = apply_action_family_batch(coords_batch, action_family)
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
        translation_step=action_family.translation_step,
    )
    prior = build_prior(prior_spec, exact_scores_matrix.shape[1])
    exact_top_k_masks = jax.vmap(
        lambda exact_scores: top_k_with_ties_mask(exact_scores, k=1)
    )(exact_scores_matrix)
    if delta <= 0.0:
        survivor_masks = exact_top_k_masks
        ambiguity_masks = exact_top_k_masks
    else:
        survivor_masks = jax.vmap(
            lambda exact_scores, coarse_scores: certified_survivor_mask(
                exact_scores=exact_scores,
                coarse_scores=coarse_scores,
                k=1,
                delta=delta,
            )
        )(exact_scores_matrix, coarse_scores_matrix)
        ambiguity_masks = jax.vmap(
            lambda exact_scores: ambiguity_band_mask(exact_scores, k=1, epsilon=delta)
        )(exact_scores_matrix)
    posterior_matrix = update_posterior_batch(prior, survivor_masks)
    selected_actions = select_admissible_actions(posterior_matrix, ambiguity_masks)
    next_coords = candidate_batches[
        jnp.arange(candidate_batches.shape[0]), selected_actions
    ]
    exact_scores_host = np.asarray(exact_scores_matrix)
    coarse_scores_host = np.asarray(coarse_scores_matrix)
    exact_top_k_host = np.asarray(exact_top_k_masks)
    survivor_masks_host = np.asarray(survivor_masks)
    ambiguity_masks_host = np.asarray(ambiguity_masks)
    posterior_host = np.asarray(posterior_matrix)
    selected_actions_host = np.asarray(selected_actions)
    next_coords_host = np.asarray(next_coords)
    states = []
    for pose_index in range(candidate_batches.shape[0]):
        pruning_certificate = CertifiedPruningCertificate(
            exact_top_k_mask=exact_top_k_host[pose_index],
            exact_ambiguity_mask=ambiguity_masks_host[pose_index],
            coarse_ambiguity_mask=ambiguity_masks_host[pose_index],
            survivor_mask=survivor_masks_host[pose_index],
            k=1,
            delta=delta,
        )
        bundle = CertifiedCoarseScoreBundle(
            exact_scores=exact_scores_host[pose_index],
            coarse_scores=coarse_scores_host[pose_index],
            delta=delta,
            exact_error_bound=exact_error_bound,
            coarse_error_bound=max(delta - exact_error_bound, 0.0),
            survivor_mask=survivor_masks_host[pose_index],
            ambiguity_mask=ambiguity_masks_host[pose_index],
            retained_receptor_indices=retained_indices,
            pruning_certificate=pruning_certificate,
        )
        belief = CertifiedBeliefState(
            prior_spec=prior_spec,
            prior=prior,
            posterior=posterior_host[pose_index],
            coarse_scores=bundle.coarse_scores,
            exact_scores=bundle.exact_scores,
            exact_error_bound=bundle.exact_error_bound,
            survivor_mask=bundle.survivor_mask,
            ambiguity_mask=bundle.ambiguity_mask,
            selected_action=int(selected_actions_host[pose_index]),
            step_index=round_index,
        )
        states.append(
            CertifiedOptimizerState(
                coords=next_coords_host[pose_index],
                action_family=action_family,
                belief=belief,
                retained_receptor_indices=bundle.retained_receptor_indices,
            )
        )

    return next_coords, tuple(states)


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
    if n_rounds <= 0:
        raise ValueError("n_rounds must be positive")
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

        if all(
            _noop_is_unique_admissible_action(action_state) for action_state in states
        ):
            break

    return current_coords, tuple(history)
