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
    CertifiedBeliefWitness,
    CertifiedBeliefState,
    CertifiedPriorSpec,
    PosteriorUpdateBranch,
    belief_witness,
    build_prior,
    selection_provenance,
    select_admissible_actions,
    select_admissible_action,
    update_posterior_batch,
    update_posterior,
)
from dq_dock_engine.docking.formal_handles import (
    CP1,
    FLO18,
    posterior_update_theorem_handle,
    optimizer_branch_witness_handle,
    selection_theorem_handle,
    optimizer_witness_handle,
    survivor_set_witness_handle,
)
from dq_dock_engine.docking.formal_pruning import (
    CertifiedSurvivorSetWitness,
    Top1PruningBranch,
    ambiguity_band_mask,
    certified_pruning_certificate,
    certified_survivor_mask,
    CertifiedPruningCertificate,
    top_k_with_ties_mask,
)
from dq_dock_engine.docking.formal_surrogates import (
    CertifiedCoarseScoreBundle,
    FastSingletonAcceptRoundResult,
    PerPoseFastSingletonAcceptRoundResult,
    StagedSingletonAcceptRoundResult,
    try_adaptive_singleton_accept_round,
    try_fast_singleton_accept_round,
    try_per_pose_fast_singleton_accept_round,
    score_exact_and_coarse_round,
)
from dq_dock_engine.docking.scoring import CertifiedRealSpaceEwaldSpec
from dq_dock_engine.docking.formal_pruning import (
    StagedTop1Guarantee,
    staged_top1_guarantee_from_coarse_scores,
)


@dataclass(frozen=True)
class CertifiedOptimizerState:
    coords: jax.Array
    action_family: CertifiedActionFamily
    belief: CertifiedBeliefState
    retained_receptor_indices: jax.Array
    pruning_certificate: CertifiedPruningCertificate


@dataclass(frozen=True)
class CertifiedOptimizerWitness:
    survivor_set: CertifiedSurvivorSetWitness
    belief: CertifiedBeliefWitness
    theorem_handle: str
    branch_witness_handle: str
    support_matches_survivors: bool
    coherence_theorem_handles: tuple[str, str]


@dataclass(frozen=True)
class StagedCertifiedDecisionState:
    coords: jax.Array
    selected_action: int
    coarse_scores: jax.Array
    delta: float
    retained_receptor_indices: jax.Array
    top1_guarantee: CertifiedSurvivorSetWitness | StagedTop1Guarantee
    theorem_handle: str


@dataclass(frozen=True)
class MinimalStagedRoundResult:
    next_coords: jax.Array
    selected_actions: jax.Array
    delta: float
    theorem_handle: str


@dataclass(frozen=True)
class StagedRoundAcceptanceDiagnostic:
    accepted_rounds: int
    first_failed_round: int | None
    total_rounds: int


@dataclass(frozen=True)
class HybridSingletonRefinementResult:
    coords: jax.Array
    singleton_round_history: tuple[MinimalStagedRoundResult, ...]
    exact_round_history: tuple[tuple[CertifiedOptimizerState, ...], ...]


def _noop_is_unique_admissible_action(state: CertifiedOptimizerState) -> bool:
    ambiguity_mask = state.belief.ambiguity_mask
    return (
        state.belief.selected_action == 0
        and bool(ambiguity_mask[0])
        and int(jnp.sum(ambiguity_mask.astype(jnp.int32))) == 1
    )


def active_pruning_branch(state: CertifiedOptimizerState) -> Top1PruningBranch:
    return Top1PruningBranch(state.pruning_certificate.rule)


def active_survivor_set_witness(
    state: CertifiedOptimizerState,
) -> CertifiedSurvivorSetWitness:
    theorem_handle = survivor_set_witness_handle(active_pruning_branch(state).value)
    return CertifiedSurvivorSetWitness(
        survivor_mask=state.pruning_certificate.survivor_mask,
        certificate=state.pruning_certificate,
        theorem_handle=theorem_handle,
    )


def active_belief_witness(state: CertifiedOptimizerState) -> CertifiedBeliefWitness:
    return belief_witness(
        posterior=jnp.asarray(state.belief.posterior),
        ambiguity_mask=jnp.asarray(state.belief.ambiguity_mask),
        selected_action=state.belief.selected_action,
    )


def active_optimizer_witness(
    state: CertifiedOptimizerState,
) -> CertifiedOptimizerWitness:
    theorem_handle = optimizer_witness_handle(active_pruning_branch(state).value)
    posterior_support = jnp.asarray(state.belief.posterior > 0)
    ambiguity_support = jnp.logical_and(
        jnp.asarray(state.belief.ambiguity_mask), posterior_support
    )
    selection_support = (
        ambiguity_support if bool(jnp.any(ambiguity_support)) else posterior_support
    )
    survivor_mask = jnp.asarray(state.pruning_certificate.survivor_mask)
    return CertifiedOptimizerWitness(
        survivor_set=active_survivor_set_witness(state),
        belief=active_belief_witness(state),
        theorem_handle=theorem_handle,
        branch_witness_handle=optimizer_branch_witness_handle(),
        support_matches_survivors=bool(
            jnp.array_equal(selection_support, survivor_mask)
        ),
        coherence_theorem_handles=("APX11", "APX12"),
    )


def staged_decision_states_from_singleton_accept_round(
    round_result: StagedSingletonAcceptRoundResult | FastSingletonAcceptRoundResult,
) -> tuple[StagedCertifiedDecisionState, ...]:
    if isinstance(round_result, FastSingletonAcceptRoundResult):
        return tuple(
            StagedCertifiedDecisionState(
                coords=round_result.next_coords[pose_index],
                selected_action=int(round_result.selected_actions[pose_index]),
                coarse_scores=round_result.coarse_scores[pose_index],
                delta=round_result.delta,
                retained_receptor_indices=round_result.retained_receptor_indices,
                top1_guarantee=staged_top1_guarantee_from_coarse_scores(
                    coarse_scores=jnp.asarray(round_result.coarse_scores[pose_index]),
                    delta=round_result.delta,
                ),
                theorem_handle=round_result.theorem_handle,
            )
            for pose_index in range(round_result.coarse_scores.shape[0])
        )
    return tuple(
        StagedCertifiedDecisionState(
            coords=round_result.next_coords[pose_index],
            selected_action=int(round_result.selected_actions[pose_index]),
            coarse_scores=round_result.round_result.coarse_scores[pose_index],
            delta=round_result.round_result.delta,
            retained_receptor_indices=round_result.round_result.retained_receptor_indices,
            top1_guarantee=round_result.round_result.decisions[pose_index].survivor_set,
            theorem_handle=round_result.round_result.decisions[
                pose_index
            ].survivor_set.theorem_handle,
        )
        for pose_index in range(len(round_result.round_result.decisions))
    )


def _minimal_round_from_per_pose_singleton_accept(
    round_result: PerPoseFastSingletonAcceptRoundResult,
) -> MinimalStagedRoundResult:
    return MinimalStagedRoundResult(
        next_coords=round_result.next_coords,
        selected_actions=round_result.selected_actions,
        delta=round_result.delta,
        theorem_handle=round_result.theorem_handle,
    )


def _try_singleton_accept(
    receptor_coords: jax.Array,
    receptor_radii: jax.Array,
    ligand_radii: jax.Array,
    candidate_batches: jax.Array,
    target_error: float,
    coarse_target_error: float,
    adaptive_coarse_target_errors: tuple[float, ...] | None,
    translation_step: float,
    use_softened_coarse: bool,
    electrostatics: CertifiedRealSpaceEwaldSpec | None,
) -> FastSingletonAcceptRoundResult | None:
    if adaptive_coarse_target_errors is not None:
        return try_adaptive_singleton_accept_round(
            receptor_coords=receptor_coords,
            receptor_radii=receptor_radii,
            ligand_radii=ligand_radii,
            candidate_batches=candidate_batches,
            target_error=target_error,
            coarse_target_errors=adaptive_coarse_target_errors,
            translation_step=translation_step,
            electrostatics=electrostatics,
        )

    return try_fast_singleton_accept_round(
        receptor_coords=receptor_coords,
        receptor_radii=receptor_radii,
        ligand_radii=ligand_radii,
        candidate_batches=candidate_batches,
        target_error=target_error,
        coarse_target_error=coarse_target_error,
        translation_step=translation_step,
        use_softened_coarse=use_softened_coarse,
        electrostatics=electrostatics,
    )


def try_refine_round_singleton_staged(
    coords_batch: jax.Array,
    receptor_coords: jax.Array,
    receptor_radii: jax.Array,
    ligand_radii: jax.Array,
    target_error: float,
    round_index: int,
    base_translation_step: float,
    base_rotation_step_rad: float,
    coarse_target_error: float = 0.004,
    adaptive_coarse_target_errors: tuple[float, ...] | None = None,
    use_softened_coarse: bool = False,
    electrostatics: CertifiedRealSpaceEwaldSpec | None = None,
) -> tuple[jax.Array, tuple[StagedCertifiedDecisionState, ...]] | None:
    action_family = create_certified_action_family(
        translation_step=base_translation_step / (2**round_index),
        rotation_step_rad=base_rotation_step_rad / (2**round_index),
        stencil_level=round_index,
    )
    candidate_batches = apply_action_family_batch(coords_batch, action_family)
    staged_result = _try_singleton_accept(
        receptor_coords=receptor_coords,
        receptor_radii=receptor_radii,
        ligand_radii=ligand_radii,
        candidate_batches=candidate_batches,
        target_error=target_error,
        coarse_target_error=coarse_target_error,
        adaptive_coarse_target_errors=adaptive_coarse_target_errors,
        translation_step=action_family.translation_step,
        use_softened_coarse=use_softened_coarse,
        electrostatics=electrostatics,
    )
    if staged_result is None:
        return None
    return (
        staged_result.next_coords,
        staged_decision_states_from_singleton_accept_round(staged_result),
    )


def try_refine_round_singleton_minimal(
    coords_batch: jax.Array,
    receptor_coords: jax.Array,
    receptor_radii: jax.Array,
    ligand_radii: jax.Array,
    target_error: float,
    round_index: int,
    base_translation_step: float,
    base_rotation_step_rad: float,
    coarse_target_error: float = 0.004,
    adaptive_coarse_target_errors: tuple[float, ...] | None = None,
    use_softened_coarse: bool = False,
    electrostatics: CertifiedRealSpaceEwaldSpec | None = None,
) -> MinimalStagedRoundResult | None:
    action_family = create_certified_action_family(
        translation_step=base_translation_step / (2**round_index),
        rotation_step_rad=base_rotation_step_rad / (2**round_index),
        stencil_level=round_index,
    )
    candidate_batches = apply_action_family_batch(coords_batch, action_family)
    staged_result = _try_singleton_accept(
        receptor_coords=receptor_coords,
        receptor_radii=receptor_radii,
        ligand_radii=ligand_radii,
        candidate_batches=candidate_batches,
        target_error=target_error,
        coarse_target_error=coarse_target_error,
        adaptive_coarse_target_errors=adaptive_coarse_target_errors,
        translation_step=action_family.translation_step,
        use_softened_coarse=use_softened_coarse,
        electrostatics=electrostatics,
    )
    if staged_result is None:
        return None
    return MinimalStagedRoundResult(
        next_coords=staged_result.next_coords,
        selected_actions=staged_result.selected_actions,
        delta=staged_result.delta,
        theorem_handle=staged_result.theorem_handle,
    )


def try_refine_poses_singleton_minimal(
    coords_batch: jax.Array,
    receptor_coords: jax.Array,
    receptor_radii: jax.Array,
    ligand_radii: jax.Array,
    n_rounds: int,
    target_error: float,
    base_translation_step: float = 0.5,
    base_rotation_step_rad: float = jnp.pi / 12.0,
    coarse_target_error: float = 0.004,
    adaptive_coarse_target_errors: tuple[float, ...] | None = None,
    use_softened_coarse: bool = False,
    electrostatics: CertifiedRealSpaceEwaldSpec | None = None,
) -> tuple[jax.Array, tuple[MinimalStagedRoundResult, ...]] | None:
    if n_rounds <= 0:
        raise ValueError("n_rounds must be positive")
    current_coords = coords_batch
    history: list[MinimalStagedRoundResult] = []
    for round_index in range(n_rounds):
        result = try_refine_round_singleton_minimal(
            coords_batch=current_coords,
            receptor_coords=receptor_coords,
            receptor_radii=receptor_radii,
            ligand_radii=ligand_radii,
            target_error=target_error,
            round_index=round_index,
            base_translation_step=base_translation_step,
            base_rotation_step_rad=base_rotation_step_rad,
            coarse_target_error=coarse_target_error,
            adaptive_coarse_target_errors=adaptive_coarse_target_errors,
            use_softened_coarse=use_softened_coarse,
            electrostatics=electrostatics,
        )
        if result is None:
            return None
        current_coords = result.next_coords
        history.append(result)
    return current_coords, tuple(history)


def diagnose_singleton_acceptance_schedule(
    coords_batch: jax.Array,
    receptor_coords: jax.Array,
    receptor_radii: jax.Array,
    ligand_radii: jax.Array,
    n_rounds: int,
    target_error: float,
    base_translation_step: float = 0.5,
    base_rotation_step_rad: float = jnp.pi / 12.0,
    coarse_target_error: float = 0.004,
    adaptive_coarse_target_errors: tuple[float, ...] | None = None,
    use_softened_coarse: bool = False,
    electrostatics: CertifiedRealSpaceEwaldSpec | None = None,
) -> StagedRoundAcceptanceDiagnostic:
    if n_rounds <= 0:
        raise ValueError("n_rounds must be positive")
    current_coords = coords_batch
    accepted_rounds = 0
    for round_index in range(n_rounds):
        result = try_refine_round_singleton_minimal(
            coords_batch=current_coords,
            receptor_coords=receptor_coords,
            receptor_radii=receptor_radii,
            ligand_radii=ligand_radii,
            target_error=target_error,
            round_index=round_index,
            base_translation_step=base_translation_step,
            base_rotation_step_rad=base_rotation_step_rad,
            coarse_target_error=coarse_target_error,
            adaptive_coarse_target_errors=adaptive_coarse_target_errors,
            use_softened_coarse=use_softened_coarse,
            electrostatics=electrostatics,
        )
        if result is None:
            return StagedRoundAcceptanceDiagnostic(
                accepted_rounds=accepted_rounds,
                first_failed_round=round_index,
                total_rounds=n_rounds,
            )
        accepted_rounds += 1
        current_coords = result.next_coords
    return StagedRoundAcceptanceDiagnostic(
        accepted_rounds=accepted_rounds,
        first_failed_round=None,
        total_rounds=n_rounds,
    )


def refine_poses_singleton_then_exact(
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
    coarse_target_error: float = 0.004,
    adaptive_coarse_target_errors: tuple[float, ...] | None = None,
    use_softened_coarse: bool = False,
    electrostatics: CertifiedRealSpaceEwaldSpec | None = None,
) -> HybridSingletonRefinementResult:
    if n_rounds <= 0:
        raise ValueError("n_rounds must be positive")
    effective_prior_spec = (
        CertifiedPriorSpec(kind="uniform") if prior_spec is None else prior_spec
    )
    current_coords = coords_batch
    singleton_history: list[MinimalStagedRoundResult] = []
    round_index = 0
    while round_index < n_rounds:
        staged = try_refine_round_singleton_minimal(
            coords_batch=current_coords,
            receptor_coords=receptor_coords,
            receptor_radii=receptor_radii,
            ligand_radii=ligand_radii,
            target_error=target_error,
            round_index=round_index,
            base_translation_step=base_translation_step,
            base_rotation_step_rad=base_rotation_step_rad,
            coarse_target_error=coarse_target_error,
            adaptive_coarse_target_errors=adaptive_coarse_target_errors,
            use_softened_coarse=use_softened_coarse,
            electrostatics=electrostatics,
        )
        if staged is None:
            action_family = create_certified_action_family(
                translation_step=base_translation_step / (2**round_index),
                rotation_step_rad=base_rotation_step_rad / (2**round_index),
                stencil_level=round_index,
            )
            candidate_batches = apply_action_family_batch(current_coords, action_family)
            per_pose = try_per_pose_fast_singleton_accept_round(
                receptor_coords=receptor_coords,
                receptor_radii=receptor_radii,
                ligand_radii=ligand_radii,
                candidate_batches=candidate_batches,
                target_error=target_error,
                coarse_target_error=coarse_target_error,
                translation_step=action_family.translation_step,
                electrostatics=electrostatics,
            )
            if per_pose is None:
                break
            staged = _minimal_round_from_per_pose_singleton_accept(per_pose)
        singleton_history.append(staged)
        current_coords = staged.next_coords
        round_index += 1

    exact_history: tuple[tuple[CertifiedOptimizerState, ...], ...] = ()
    if round_index < n_rounds:
        current_coords, exact_history = refine_poses_certified(
            coords_batch=current_coords,
            receptor_coords=receptor_coords,
            receptor_radii=receptor_radii,
            ligand_radii=ligand_radii,
            n_rounds=n_rounds - round_index,
            target_error=target_error,
            base_translation_step=base_translation_step / (2**round_index),
            base_rotation_step_rad=base_rotation_step_rad / (2**round_index),
            prior_spec=effective_prior_spec,
            max_coarse_receptor_atoms=max_coarse_receptor_atoms,
            coarse_target_error=coarse_target_error,
            use_softened_coarse=use_softened_coarse,
            electrostatics=electrostatics,
        )

    return HybridSingletonRefinementResult(
        coords=current_coords,
        singleton_round_history=tuple(singleton_history),
        exact_round_history=exact_history,
    )


def _run_exact_formal_refinement(**kwargs) -> jax.Array:
    opt_coords, _ = refine_poses_certified(**kwargs)
    return opt_coords


def _run_singleton_hybrid_formal_refinement(**kwargs) -> jax.Array:
    return refine_poses_singleton_then_exact(**kwargs).coords


def _build_belief_state(
    bundle: CertifiedCoarseScoreBundle,
    prior_spec: CertifiedPriorSpec,
    step_index: int,
) -> CertifiedBeliefState:
    prior = build_prior(prior_spec, len(bundle.exact_scores))
    posterior = update_posterior(prior, bundle.survivor_mask)
    selected_action = select_admissible_action(posterior, bundle.ambiguity_mask)
    selected_action_rule, selected_action_theorem = selection_provenance(
        posterior, bundle.ambiguity_mask
    )
    return CertifiedBeliefState(
        prior_spec=prior_spec,
        prior=prior,
        posterior=posterior,
        posterior_rule=PosteriorUpdateBranch.SURVIVOR_CONDITIONING.value,
        posterior_theorem=posterior_update_theorem_handle(),
        coarse_scores=bundle.coarse_scores,
        exact_scores=bundle.exact_scores,
        exact_error_bound=bundle.exact_error_bound,
        survivor_mask=bundle.survivor_mask,
        ambiguity_mask=bundle.ambiguity_mask,
        selected_action=selected_action,
        selected_action_rule=selected_action_rule,
        selected_action_theorem=selection_theorem_handle(selected_action_rule),
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
    coarse_target_error: float,
    round_index: int,
    base_translation_step: float,
    base_rotation_step_rad: float,
    prior_spec: CertifiedPriorSpec,
    max_coarse_receptor_atoms: int,
    electrostatics: CertifiedRealSpaceEwaldSpec | None = None,
    use_softened_coarse: bool = False,
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
        coarse_target_error=coarse_target_error,
        max_receptor_atoms=max_coarse_receptor_atoms,
        translation_step=action_family.translation_step,
        electrostatics=electrostatics,
        use_softened_coarse=use_softened_coarse,
    )
    prior = build_prior(prior_spec, exact_scores_matrix.shape[1])
    exact_scores_host = np.asarray(exact_scores_matrix)
    coarse_scores_host = np.asarray(coarse_scores_matrix)
    exact_top_k_host = np.asarray(
        [top_k_with_ties_mask(jnp.asarray(row), k=1) for row in exact_scores_host]
    )
    if delta <= 0.0:
        survivor_masks_host = exact_top_k_host
        ambiguity_masks_host = exact_top_k_host
    else:
        survivor_masks_host = np.asarray(
            [
                certified_survivor_mask(
                    exact_scores=jnp.asarray(exact_row),
                    coarse_scores=jnp.asarray(coarse_row),
                    k=1,
                    delta=delta,
                )
                for exact_row, coarse_row in zip(
                    exact_scores_host, coarse_scores_host, strict=True
                )
            ]
        )
        ambiguity_masks_host = np.asarray(
            [
                ambiguity_band_mask(jnp.asarray(exact_row), k=1, epsilon=delta)
                for exact_row in exact_scores_host
            ]
        )
    survivor_masks = jnp.asarray(survivor_masks_host)
    ambiguity_masks = jnp.asarray(ambiguity_masks_host)
    posterior_matrix = update_posterior_batch(prior, survivor_masks)
    selected_actions = select_admissible_actions(posterior_matrix, ambiguity_masks)
    next_coords = candidate_batches[
        jnp.arange(candidate_batches.shape[0]), selected_actions
    ]
    posterior_host = np.asarray(posterior_matrix)
    selected_actions_host = np.asarray(selected_actions)
    next_coords_host = np.asarray(next_coords)
    states = []
    for pose_index in range(candidate_batches.shape[0]):
        pruning_certificate = certified_pruning_certificate(
            exact_scores=jnp.asarray(exact_scores_host[pose_index]),
            coarse_scores=jnp.asarray(coarse_scores_host[pose_index]),
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
        selected_action_rule, selected_action_theorem = selection_provenance(
            jnp.asarray(posterior_host[pose_index]),
            jnp.asarray(ambiguity_masks_host[pose_index]),
        )
        belief = CertifiedBeliefState(
            prior_spec=prior_spec,
            prior=prior,
            posterior=posterior_host[pose_index],
            posterior_rule=PosteriorUpdateBranch.SURVIVOR_CONDITIONING.value,
            posterior_theorem=posterior_update_theorem_handle(),
            coarse_scores=bundle.coarse_scores,
            exact_scores=bundle.exact_scores,
            exact_error_bound=bundle.exact_error_bound,
            survivor_mask=bundle.survivor_mask,
            ambiguity_mask=bundle.ambiguity_mask,
            selected_action=int(selected_actions_host[pose_index]),
            selected_action_rule=selected_action_rule,
            selected_action_theorem=selection_theorem_handle(selected_action_rule),
            step_index=round_index,
        )
        states.append(
            CertifiedOptimizerState(
                coords=next_coords_host[pose_index],
                action_family=action_family,
                belief=belief,
                retained_receptor_indices=bundle.retained_receptor_indices,
                pruning_certificate=pruning_certificate,
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
    coarse_target_error: float = 0.004,
    base_translation_step: float = 0.5,
    base_rotation_step_rad: float = jnp.pi / 12.0,
    prior_spec: CertifiedPriorSpec | None = None,
    max_coarse_receptor_atoms: int = 64,
    electrostatics: CertifiedRealSpaceEwaldSpec | None = None,
    use_softened_coarse: bool = False,
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
            coarse_target_error=coarse_target_error,
            round_index=round_index,
            base_translation_step=base_translation_step,
            base_rotation_step_rad=float(base_rotation_step_rad),
            prior_spec=effective_prior_spec,
            max_coarse_receptor_atoms=max_coarse_receptor_atoms,
            electrostatics=electrostatics,
            use_softened_coarse=use_softened_coarse,
        )
        history.append(states)

        if all(
            _noop_is_unique_admissible_action(action_state) for action_state in states
        ):
            break

    return current_coords, tuple(history)
