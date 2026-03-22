from __future__ import annotations

import time
import functools
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Union, Any

import jax
import jax.numpy as jnp
import numpy as np

from dq_dock_engine.docking.core import (
    GapCertification,
    NativeCertification,
    ScoredPose,
    ScoringEngine,
)
from dq_dock_engine.docking.formal_actions import (
    create_roundwise_certified_action_family,
    apply_action_family_batch,
)
from dq_dock_engine.docking.formal_belief import (
    CertifiedBeliefState,
    CertifiedBeliefWitness,
    CertifiedPriorSpec,
    PosteriorUpdateBranch,
    SelectionBranch,
    build_prior,
    posterior_update_theorem_handle,
    selection_provenance,
    selection_theorem_handle,
    update_posterior,
    update_posterior_batch,
    select_admissible_action,
    select_admissible_actions,
)
from dq_dock_engine.docking.formal_handles import FLO10, FLO15, FLO3, FLO4
from dq_dock_engine.docking.formal_pruning import (
    Top1PruningBranch,
    ambiguity_band_mask,
    certified_pruning_certificate,
    certified_survivor_mask,
    top_k_with_ties_mask,
)
from dq_dock_engine.docking.formal_surrogates import (
    CertifiedCoarseScoreBundle,
    CertifiedRealSpaceEwaldSpec,
    StagedTop1BranchSummary,
    StagedTop1RoundResult,
    score_exact_and_coarse_round,
    staged_top1_decision_from_scores,
    staged_top1_round_from_coarse_scores,
    summarize_staged_top1_round,
    select_exact_receptor_subset_for_local_family,
)
from dq_dock_engine.physics.kernels import rigid_transform_3d


@dataclass(frozen=True)
class CertifiedOptimizerState:
    coords: jax.Array
    action_family: Any
    belief: CertifiedBeliefState
    retained_receptor_indices: jax.Array
    pruning_certificate: Any


@dataclass(frozen=True)
class MinimalStagedRoundResult:
    next_coords: jax.Array
    selected_actions: jax.Array
    delta: float
    theorem_handle: str


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
    return Top1PruningBranch(state.belief.selected_action_rule)


@functools.partial(
    jax.jit,
    static_argnames=(
        "max_coarse_receptor_atoms",
        "use_softened_coarse",
    ),
)
def _refine_round_jit_core(
    coords_batch: jax.Array,
    receptor_coords: jax.Array,
    receptor_radii: jax.Array,
    ligand_radii: jax.Array,
    target_error: float,
    coarse_target_error: float,
    translation_step: float,
    translation_deltas: jax.Array,
    quaternion_deltas: jax.Array,
    prior_spec: CertifiedPriorSpec,
    max_coarse_receptor_atoms: int,
    retained_indices: jax.Array,
    electrostatics: CertifiedRealSpaceEwaldSpec | None = None,
    use_softened_coarse: bool = False,
):
    n_poses, n_atoms, _ = coords_batch.shape
    n_actions = translation_deltas.shape[0]

    candidate_batches = jax.vmap(
        lambda pose_coords: jax.vmap(
            lambda t, q: rigid_transform_3d(pose_coords, q, t), in_axes=(0, 0)
        )(translation_deltas, quaternion_deltas),
        in_axes=0,
    )(coords_batch)

    (
        exact_scores_matrix,
        coarse_scores_matrix,
        delta,
        exact_error_bound,
        _,
    ) = score_exact_and_coarse_round(
        receptor_coords=receptor_coords,
        receptor_radii=receptor_radii,
        ligand_radii=ligand_radii,
        candidate_batches=candidate_batches,
        target_error=target_error,
        coarse_target_error=coarse_target_error,
        max_receptor_atoms=max_coarse_receptor_atoms,
        translation_step=translation_step,
        electrostatics=electrostatics,
        use_softened_coarse=use_softened_coarse,
    )

    return {
        "candidate_batches": candidate_batches,
        "exact_scores_matrix": exact_scores_matrix,
        "coarse_scores_matrix": coarse_scores_matrix,
        "delta": delta,
        "exact_error_bound": exact_error_bound,
    }


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
    support_expansion_level: int,
    prior_spec: CertifiedPriorSpec,
    max_coarse_receptor_atoms: int,
    retained_indices: jax.Array,
    electrostatics: CertifiedRealSpaceEwaldSpec | None = None,
    use_softened_coarse: bool = False,
) -> tuple[jax.Array, tuple[CertifiedOptimizerState, ...]]:
    action_family = create_roundwise_certified_action_family(
        base_translation_step=base_translation_step,
        base_rotation_step_rad=base_rotation_step_rad,
        round_index=round_index,
        support_expansion_level=support_expansion_level,
    )

    results = _refine_round_jit_core(
        coords_batch=coords_batch,
        receptor_coords=receptor_coords,
        receptor_radii=receptor_radii,
        ligand_radii=ligand_radii,
        target_error=target_error,
        coarse_target_error=coarse_target_error,
        translation_step=action_family.translation_step,
        translation_deltas=action_family.translation_deltas,
        quaternion_deltas=action_family.quaternion_deltas,
        prior_spec=prior_spec,
        max_coarse_receptor_atoms=max_coarse_receptor_atoms,
        retained_indices=retained_indices,
        electrostatics=electrostatics,
        use_softened_coarse=use_softened_coarse,
    )

    results_host = jax.device_get(results)

    candidate_batches = results_host["candidate_batches"]
    n_poses = candidate_batches.shape[0]

    prior = build_prior(prior_spec, results_host["exact_scores_matrix"].shape[1])
    exact_error_bound = float(results_host["exact_error_bound"])
    delta = float(results_host["delta"])

    exact_scores_host = np.asarray(results_host["exact_scores_matrix"])
    coarse_scores_host = np.asarray(results_host["coarse_scores_matrix"])
    survivor_masks_host = []
    ambiguity_masks_host = []
    posterior_rows = []
    selected_actions_host = []
    batched_decisions = []
    next_coords_rows = []

    for exact_row, coarse_row, candidate_row in zip(
        exact_scores_host,
        coarse_scores_host,
        candidate_batches,
        strict=True,
    ):
        exact_scores = jnp.asarray(exact_row)
        coarse_scores = jnp.asarray(coarse_row)
        if delta <= 0.0:
            survivor_mask = top_k_with_ties_mask(exact_scores, 1)
            ambiguity_mask = survivor_mask
        else:
            survivor_mask = certified_survivor_mask(
                exact_scores, coarse_scores, 1, delta
            )
            ambiguity_mask = ambiguity_band_mask(exact_scores, 1, delta)
        posterior = update_posterior(prior, survivor_mask)
        selected_action = select_admissible_action(posterior, ambiguity_mask)
        survivor_masks_host.append(np.asarray(survivor_mask))
        ambiguity_masks_host.append(np.asarray(ambiguity_mask))
        posterior_rows.append(np.asarray(posterior))
        selected_actions_host.append(int(selected_action))
        batched_decisions.append(
            staged_top1_decision_from_scores(exact_scores, coarse_scores, delta)
        )
        next_coords_rows.append(np.asarray(candidate_row[int(selected_action)]))

    next_coords = jnp.asarray(next_coords_rows)
    posterior_matrix = np.asarray(posterior_rows)

    states = []
    for i in range(n_poses):
        pose_posterior = posterior_matrix[i]
        pose_ambiguity = ambiguity_masks_host[i]
        decision_host = batched_decisions[i]

        rule, theorem = selection_provenance(pose_posterior, pose_ambiguity)

        belief = CertifiedBeliefState(
            prior_spec=prior_spec,
            prior=prior,
            posterior=pose_posterior,
            posterior_rule=PosteriorUpdateBranch.SURVIVOR_CONDITIONING.value,
            posterior_theorem=posterior_update_theorem_handle(),
            coarse_scores=coarse_scores_host[i],
            exact_scores=exact_scores_host[i],
            exact_error_bound=exact_error_bound,
            survivor_mask=survivor_masks_host[i],
            ambiguity_mask=pose_ambiguity,
            selected_action=selected_actions_host[i],
            selected_action_rule=rule,
            selected_action_theorem=selection_theorem_handle(rule),
            step_index=round_index,
        )

        states.append(
            CertifiedOptimizerState(
                coords=next_coords[i],
                action_family=action_family,
                belief=belief,
                retained_receptor_indices=retained_indices,
                pruning_certificate=decision_host,
            )
        )

    return next_coords, tuple(states)


def _has_non_singleton_ambiguity(state: CertifiedOptimizerState) -> bool:
    return int(np.count_nonzero(np.asarray(state.belief.ambiguity_mask))) > 1


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
    adaptive_coarse_target_errors: tuple[float, ...] | None = None,
) -> tuple[jax.Array, tuple[tuple[CertifiedOptimizerState, ...], ...]]:
    if n_rounds <= 0:
        raise ValueError("n_rounds must be positive")
    current_coords = coords_batch
    history: list[tuple[CertifiedOptimizerState, ...]] = []
    support_expansion_level = 0
    effective_prior_spec = (
        CertifiedPriorSpec(kind="uniform") if prior_spec is None else prior_spec
    )

    # Precompute a stable receptor subset once on the host
    # and use it for all rounds to ensure JIT-safe static shapes.
    retained_indices = select_exact_receptor_subset_for_local_family(
        receptor_coords=receptor_coords,
        receptor_radii=receptor_radii,
        reference_coords_batch=current_coords,
        ligand_radii=ligand_radii,
        translation_step=base_translation_step,
        target_error=target_error,
    )
    print(
        f"[DEBUG] Retained {len(retained_indices)} receptor atoms out of {len(receptor_coords)}"
    )
    retained_indices_jax = jnp.array(retained_indices)

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
            support_expansion_level=support_expansion_level,
            prior_spec=effective_prior_spec,
            max_coarse_receptor_atoms=max_coarse_receptor_atoms,
            retained_indices=retained_indices_jax,
            electrostatics=electrostatics,
            use_softened_coarse=use_softened_coarse,
        )
        history.append(states)

        if all(_noop_is_unique_admissible_action(s) for s in states):
            break

        if any(_has_non_singleton_ambiguity(state) for state in states):
            support_expansion_level += 1
        else:
            support_expansion_level = 0

    return current_coords, tuple(history)


def refine_poses_singleton_then_exact(
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
    adaptive_coarse_target_errors: tuple[float, ...] | None = None,
) -> HybridSingletonRefinementResult:
    opt_coords, history = refine_poses_certified(
        coords_batch=coords_batch,
        receptor_coords=receptor_coords,
        receptor_radii=receptor_radii,
        ligand_radii=ligand_radii,
        n_rounds=n_rounds,
        target_error=target_error,
        coarse_target_error=coarse_target_error,
        base_translation_step=base_translation_step,
        base_rotation_step_rad=base_rotation_step_rad,
        prior_spec=prior_spec,
        max_coarse_receptor_atoms=max_coarse_receptor_atoms,
        electrostatics=electrostatics,
        use_softened_coarse=use_softened_coarse,
        adaptive_coarse_target_errors=adaptive_coarse_target_errors,
    )
    return HybridSingletonRefinementResult(
        coords=opt_coords,
        singleton_round_history=(),
        exact_round_history=history,
    )


def _run_exact_formal_refinement(**kwargs) -> jax.Array:
    opt_coords, _ = refine_poses_certified(**kwargs)
    return opt_coords


def _run_singleton_hybrid_formal_refinement(**kwargs) -> jax.Array:
    return refine_poses_singleton_then_exact(**kwargs).coords
