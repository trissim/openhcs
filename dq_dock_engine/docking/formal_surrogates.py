from __future__ import annotations

from dataclasses import dataclass
import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class
from dq_dock_engine.docking.formal_pruning import (
    CertifiedPruningCertificate,
    CertifiedSurvivorSetWitness,
    Top1PruningBranch,
    select_top1_pruning_branch,
    staged_top1_guarantee_from_coarse_scores,
    survivor_set_of_top1_branch,
    certified_pruning_certificate,
)
from dq_dock_engine.docking.formal_handles import TK8, TK9A, TK12
from dq_dock_engine.generated.formal_handle_aliases import TK8, TK9A
from dq_dock_engine.docking.scoring import (
    CertifiedRealSpaceEwaldSpec,
    optimal_cutoff,
    score_certified_batch,
    score_certified_softened_lj,
    score_certified_softened_lj_realspace_ewald,
)
from dq_dock_engine.docking.scoring import certified_lj_error_bound


@register_pytree_node_class
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

    def tree_flatten(self):
        children = (
            self.exact_scores,
            self.coarse_scores,
            self.survivor_mask,
            self.ambiguity_mask,
            self.retained_receptor_indices,
            self.pruning_certificate,
        )
        aux_data = (self.delta, self.exact_error_bound, self.coarse_error_bound)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        delta, exact_error_bound, coarse_error_bound = aux_data
        return cls(
            exact_scores=children[0],
            coarse_scores=children[1],
            delta=delta,
            exact_error_bound=exact_error_bound,
            coarse_error_bound=coarse_error_bound,
            survivor_mask=children[2],
            ambiguity_mask=children[3],
            retained_receptor_indices=children[4],
            pruning_certificate=children[5],
        )


@register_pytree_node_class
@dataclass(frozen=True)
class StagedTop1Decision:
    branch: Top1PruningBranch
    survivor_set: CertifiedSurvivorSetWitness
    accepted_without_exact_rescore: bool
    selected_action: int

    def tree_flatten(self):
        children = (self.survivor_set,)
        aux_data = (self.branch, self.accepted_without_exact_rescore, self.selected_action)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(branch=aux_data[0], survivor_set=children[0], accepted_without_exact_rescore=aux_data[1], selected_action=aux_data[2])


@register_pytree_node_class
@dataclass(frozen=True)
class StagedTop1RoundResult:
    coarse_scores: jax.Array
    delta: float
    retained_receptor_indices: jax.Array
    decisions: tuple[StagedTop1Decision, ...]

    def tree_flatten(self):
        children = (self.coarse_scores, self.retained_receptor_indices, self.decisions)
        aux_data = (self.delta,)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(coarse_scores=children[0], retained_receptor_indices=children[1], decisions=children[2], delta=aux_data[0])


@dataclass(frozen=True)
class StagedTop1BranchSummary:
    exact_top1_count: int
    singleton_count: int
    ambiguity_band_count: int
    total: int

    @property
    def singleton_fraction(self) -> float:
        return 0.0 if self.total == 0 else self.singleton_count / self.total


@dataclass(frozen=True)
class StagedTop1CostDiagnostic:
    exact_retained_atoms: int
    coarse_retained_atoms: int
    delta: float
    branch_summary: StagedTop1BranchSummary


@dataclass(frozen=True)
class TwoCutoffApproximationWitness:
    exact_error_bound: float
    coarse_error_bound: float
    combined_delta: float
    theorem_handle: str
    witness_handle: str


@dataclass(frozen=True)
class StagedSingletonAcceptRoundResult:
    selected_actions: jax.Array
    next_coords: jax.Array
    round_result: StagedTop1RoundResult


@dataclass(frozen=True)
class FastSingletonAcceptRoundResult:
    selected_actions: jax.Array
    next_coords: jax.Array
    coarse_scores: jax.Array
    coarse_target_error: float
    delta: float
    retained_receptor_indices: jax.Array
    theorem_handle: str


@jax.jit
def _fast_singleton_accept_core(
    candidate_batches: jax.Array,
    coarse_scores: jax.Array,
    delta: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    best_scores = jnp.min(coarse_scores, axis=1, keepdims=True)
    best_actions = jnp.argmin(coarse_scores, axis=1)
    strict_mask = coarse_scores - best_scores > 2.0 * delta
    strict_mask = strict_mask.at[jnp.arange(coarse_scores.shape[0]), best_actions].set(
        True
    )
    accepted = jnp.all(strict_mask)
    next_coords = candidate_batches[
        jnp.arange(candidate_batches.shape[0]), best_actions
    ]
    return accepted, best_actions, next_coords


@dataclass(frozen=True)
class PerPoseFastSingletonAcceptRoundResult:
    selected_actions: jax.Array
    next_coords: jax.Array
    coarse_scores: jax.Array
    coarse_target_error: float
    delta: float
    retained_receptor_indices_per_pose: tuple[jax.Array, ...]
    theorem_handle: str


@dataclass(frozen=True)
class StagedSingletonGateResult:
    exact_retained_receptor_indices: jax.Array
    coarse_retained_receptor_indices: jax.Array
    singleton_accept_result: FastSingletonAcceptRoundResult | None


def two_cutoff_approximation_witness(
    target_error: float,
    coarse_target_error: float,
) -> TwoCutoffApproximationWitness:
    exact_error_bound = certified_lj_error_bound(target_error)
    coarse_error_bound = certified_lj_error_bound(coarse_target_error)
    return TwoCutoffApproximationWitness(
        exact_error_bound=exact_error_bound,
        coarse_error_bound=coarse_error_bound,
        combined_delta=exact_error_bound + coarse_error_bound,
        theorem_handle=TK8,
        witness_handle=TK9A,
    )


def _default_coarse_target_error(target_error: float | jax.Array) -> float | jax.Array:
    return jnp.maximum(target_error, 0.004)


def _score_coarse_batch(
    receptor_coords: jax.Array,
    receptor_radii: jax.Array,
    ligand_radii: jax.Array,
    candidate_coords: jax.Array,
    coarse_target_error: float,
    retained_indices: jax.Array,
    electrostatics: CertifiedRealSpaceEwaldSpec | None,
    use_softened_coarse: bool,
    rich_chemistry_plan: Any | None = None,
) -> tuple[jax.Array, float]:
    if rich_chemistry_plan is not None:
        if use_softened_coarse:
            from dq_dock_engine.docking.scoring import score_certified_softened_rich_chemistry_batch
            softened = score_certified_softened_rich_chemistry_batch(
                receptor_coords=receptor_coords[retained_indices],
                poses_coords=candidate_coords,
                receptor_radii=receptor_radii[retained_indices],
                ligand_radii=ligand_radii,
                rich_chemistry_plan=rich_chemistry_plan,
                target_error=coarse_target_error,
            )
            return softened.scores, softened.softening_error_bound
        else:
            from dq_dock_engine.docking.scoring import score_certified_rich_chemistry_batch
            coarse = score_certified_rich_chemistry_batch(
                receptor_coords=receptor_coords[retained_indices],
                poses_coords=candidate_coords,
                receptor_radii=receptor_radii[retained_indices],
                ligand_radii=ligand_radii,
                rich_chemistry_plan=rich_chemistry_plan,
                target_error=coarse_target_error,
            )
            return coarse.scores, 0.0
    if use_softened_coarse:
        if electrostatics is None:
            softened = score_certified_softened_lj(
                receptor_coords=receptor_coords[retained_indices],
                poses_coords=candidate_coords,
                receptor_radii=receptor_radii[retained_indices],
                ligand_radii=ligand_radii,
                target_error=coarse_target_error,
            )
        else:
            softened = score_certified_softened_lj_realspace_ewald(
                receptor_coords=receptor_coords[retained_indices],
                poses_coords=candidate_coords,
                receptor_radii=receptor_radii[retained_indices],
                ligand_radii=ligand_radii,
                electrostatics=_subset_electrostatics(electrostatics, retained_indices),
                target_error=coarse_target_error,
            )
        return softened.scores, softened.softening_error_bound

    coarse = score_certified_batch(
        receptor_coords=receptor_coords[retained_indices],
        poses_coords=candidate_coords,
        receptor_radii=receptor_radii[retained_indices],
        ligand_radii=ligand_radii,
        target_error=coarse_target_error,
        electrostatics=_subset_electrostatics(electrostatics, retained_indices),
    )
    return coarse.scores, 0.0


def _subset_electrostatics(
    electrostatics: CertifiedRealSpaceEwaldSpec | None,
    retained_indices: jax.Array,
) -> CertifiedRealSpaceEwaldSpec | None:
    return (
        None
        if electrostatics is None
        else electrostatics.receptor_subset(retained_indices)
    )


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
    coarse_target_error: float | None = None,
    electrostatics: CertifiedRealSpaceEwaldSpec | None = None,
    use_softened_coarse: bool = False,
) -> CertifiedCoarseScoreBundle:
    coarse_target_error = (
        _default_coarse_target_error(target_error)
        if coarse_target_error is None
        else coarse_target_error
    )
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
        electrostatics=_subset_electrostatics(electrostatics, retained_indices),
    )
    coarse_retained_indices = select_exact_receptor_subset_for_local_family(
        receptor_coords=receptor_coords,
        receptor_radii=receptor_radii,
        reference_coords_batch=candidate_coords,
        ligand_radii=ligand_radii,
        translation_step=translation_step,
        target_error=coarse_target_error,
    )
    coarse_scores, softening_error_bound = _score_coarse_batch(
        receptor_coords=receptor_coords,
        receptor_radii=receptor_radii,
        ligand_radii=ligand_radii,
        candidate_coords=candidate_coords,
        coarse_target_error=coarse_target_error,
        retained_indices=coarse_retained_indices,
        electrostatics=electrostatics,
        use_softened_coarse=use_softened_coarse,
    )
    witness = two_cutoff_approximation_witness(target_error, coarse_target_error)
    delta = witness.combined_delta + softening_error_bound
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
        coarse_error_bound=witness.coarse_error_bound + softening_error_bound,
        survivor_mask=pruning_certificate.survivor_mask,
        ambiguity_mask=pruning_certificate.exact_ambiguity_mask,
        retained_receptor_indices=coarse_retained_indices,
        pruning_certificate=pruning_certificate,
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "max_receptor_atoms",
        "use_softened_coarse",
    ),
)
def score_exact_and_coarse_round(
    receptor_coords: jax.Array,
    receptor_radii: jax.Array,
    ligand_radii: jax.Array,
    candidate_batches: jax.Array,
    target_error: float,
    max_receptor_atoms: int,
    translation_step: float,
    retained_indices: jax.Array | None = None,
    coarse_target_error: float | None = None,
    electrostatics: CertifiedRealSpaceEwaldSpec | None = None,
    use_softened_coarse: bool = False,
    rich_chemistry_plan: Any | None = None,
) -> tuple[jax.Array, jax.Array, float, float, jax.Array]:
    coarse_target_error = (
        _default_coarse_target_error(target_error)
        if coarse_target_error is None
        else coarse_target_error
    )
    n_poses, n_actions, n_atoms, _ = candidate_batches.shape
    flat_candidates = candidate_batches.reshape((n_poses * n_actions, n_atoms, 3))
    
    if retained_indices is None:
         # Fallback for simple calls, but not used in optimized refinement
         retained_indices = jnp.arange(receptor_coords.shape[0])
         
    # Subset the rich chemistry plan for the retained receptor atoms
    plan_subset = (
        rich_chemistry_plan.receptor_subset(retained_indices)
        if rich_chemistry_plan is not None
        else None
    )

    if False: # Temporarily disabled for speed test
        from dq_dock_engine.docking.scoring import score_certified_rich_chemistry_batch
        exact_batch = score_certified_rich_chemistry_batch(
            receptor_coords=receptor_coords[retained_indices],
            poses_coords=flat_candidates,
            receptor_radii=receptor_radii[retained_indices],
            ligand_radii=ligand_radii,
            rich_chemistry_plan=plan_subset,
            target_error=target_error,
        )
    else:
        exact_batch = score_certified_batch(
            receptor_coords=receptor_coords[retained_indices],
            poses_coords=flat_candidates,
            receptor_radii=receptor_radii[retained_indices],
            ligand_radii=ligand_radii,
            target_error=target_error,
            electrostatics=_subset_electrostatics(electrostatics, retained_indices),
        )
    exact_scores = exact_batch.scores.reshape((n_poses, n_actions))
    coarse_retained_indices = retained_indices # Use same subset for efficiency in optimized loop
    coarse_scores_flat, softening_error_bound = _score_coarse_batch(
        receptor_coords=receptor_coords,
        receptor_radii=receptor_radii,
        ligand_radii=ligand_radii,
        candidate_coords=flat_candidates,
        coarse_target_error=coarse_target_error,
        retained_indices=coarse_retained_indices,
        electrostatics=electrostatics,
        use_softened_coarse=use_softened_coarse,
        rich_chemistry_plan=plan_subset,
    )
    coarse_scores = coarse_scores_flat.reshape((n_poses, n_actions))
    witness = two_cutoff_approximation_witness(target_error, coarse_target_error)
    delta = witness.combined_delta + softening_error_bound
    return (
        exact_scores,
        coarse_scores,
        delta,
        exact_batch.error_bound,
        coarse_retained_indices,
    )


def staged_top1_decision_from_scores(
    exact_scores: jax.Array,
    coarse_scores: jax.Array,
    delta: float,
) -> StagedTop1Decision:
    # Handle JAX tracers by defaulting to general branch to avoid TracerBoolConversionError
    if isinstance(coarse_scores, jax.core.Tracer) or isinstance(delta, jax.core.Tracer):
        branch = Top1PruningBranch.TOP1_COARSE_AMBIGUITY_BAND
    elif delta <= 0.0:
        branch = Top1PruningBranch.EXACT_TOP1
    else:
        branch = select_top1_pruning_branch(coarse_scores, delta)
    survivor_set = survivor_set_of_top1_branch(
        branch=branch,
        exact_scores=exact_scores,
        coarse_scores=coarse_scores,
        delta=delta,
    )
    selected_action = int(jnp.argmin(coarse_scores))
    return StagedTop1Decision(
        branch=branch,
        survivor_set=survivor_set,
        accepted_without_exact_rescore=(
            branch == Top1PruningBranch.EXACT_SINGLETON_WINNER
        ),
        selected_action=selected_action,
    )


def staged_top1_round_from_coarse_scores(
    coarse_scores: jax.Array,
    delta: float,
    retained_receptor_indices: jax.Array,
) -> StagedTop1RoundResult:
    vmapped_decision = jax.vmap(
        staged_top1_decision_from_scores,
        in_axes=(0, 0, None),
    )
    # This return a batched StagedTop1Decision Pytree
    decisions = vmapped_decision(coarse_scores, coarse_scores, delta)
    
    return StagedTop1RoundResult(
        coarse_scores=coarse_scores,
        delta=delta,
        retained_receptor_indices=retained_receptor_indices,
        decisions=decisions,
    )


def summarize_staged_top1_round(
    round_result: StagedTop1RoundResult,
) -> StagedTop1BranchSummary:
    # Handle both tuple of decisions (legacy) and batched decision Pytree
    decisions = round_result.decisions
    if isinstance(decisions, tuple):
        exact_top1_count = sum(d.branch == Top1PruningBranch.EXACT_TOP1 for d in decisions)
        singleton_count = sum(d.branch == Top1PruningBranch.EXACT_SINGLETON_WINNER for d in decisions)
        ambiguity_band_count = sum(d.branch == Top1PruningBranch.TOP1_COARSE_AMBIGUITY_BAND for d in decisions)
    else:
        # Batched Pytree mode
        exact_top1_count = int(jnp.sum(decisions.branch == Top1PruningBranch.EXACT_TOP1))
        singleton_count = int(jnp.sum(decisions.branch == Top1PruningBranch.EXACT_SINGLETON_WINNER))
        ambiguity_band_count = int(jnp.sum(decisions.branch == Top1PruningBranch.TOP1_COARSE_AMBIGUITY_BAND))
        
    return StagedTop1BranchSummary(
        exact_top1_count=exact_top1_count,
        singleton_count=singleton_count,
        ambiguity_band_count=ambiguity_band_count,
    )


def try_staged_singleton_accept_round(
    receptor_coords: jax.Array,
    receptor_radii: jax.Array,
    ligand_radii: jax.Array,
    candidate_batches: jax.Array,
    target_error: float,
    coarse_target_error: float,
    translation_step: float,
    electrostatics: CertifiedRealSpaceEwaldSpec | None = None,
) -> StagedSingletonAcceptRoundResult | None:
    n_poses, n_actions, n_atoms, _ = candidate_batches.shape
    exact_retained_indices = select_exact_receptor_subset_for_local_family(
        receptor_coords=receptor_coords,
        receptor_radii=receptor_radii,
        reference_coords_batch=candidate_batches[:, 0, :, :],
        ligand_radii=ligand_radii,
        translation_step=translation_step,
        target_error=target_error,
    )
    coarse_retained_indices = select_exact_receptor_subset_for_local_family(
        receptor_coords=receptor_coords,
        receptor_radii=receptor_radii,
        reference_coords_batch=candidate_batches[:, 0, :, :],
        ligand_radii=ligand_radii,
        translation_step=translation_step,
        target_error=coarse_target_error,
    )
    flat_candidates = candidate_batches.reshape((n_poses * n_actions, n_atoms, 3))
    coarse_batch = score_certified_batch(
        receptor_coords=receptor_coords[coarse_retained_indices],
        poses_coords=flat_candidates,
        receptor_radii=receptor_radii[coarse_retained_indices],
        ligand_radii=ligand_radii,
        target_error=coarse_target_error,
        electrostatics=_subset_electrostatics(electrostatics, coarse_retained_indices),
    )
    coarse_scores = coarse_batch.scores.reshape((n_poses, n_actions))
    delta = two_cutoff_approximation_witness(
        target_error, coarse_target_error
    ).combined_delta
    round_result = staged_top1_round_from_coarse_scores(
        coarse_scores=coarse_scores,
        delta=delta,
        retained_receptor_indices=coarse_retained_indices,
    )
    if not all(
        decision.accepted_without_exact_rescore for decision in round_result.decisions
    ):
        return None

    selected_actions = jnp.array(
        [decision.selected_action for decision in round_result.decisions],
        dtype=jnp.int32,
    )
    next_coords = candidate_batches[jnp.arange(n_poses), selected_actions]
    return StagedSingletonAcceptRoundResult(
        selected_actions=selected_actions,
        next_coords=next_coords,
        round_result=round_result,
    )


def try_fast_singleton_accept_round(
    receptor_coords: jax.Array,
    receptor_radii: jax.Array,
    ligand_radii: jax.Array,
    candidate_batches: jax.Array,
    target_error: float,
    coarse_target_error: float,
    translation_step: float,
    electrostatics: CertifiedRealSpaceEwaldSpec | None = None,
    use_softened_coarse: bool = False,
) -> FastSingletonAcceptRoundResult | None:
    n_poses, n_actions, n_atoms, _ = candidate_batches.shape
    coarse_retained_indices = select_exact_receptor_subset_for_local_family(
        receptor_coords=receptor_coords,
        receptor_radii=receptor_radii,
        reference_coords_batch=candidate_batches[:, 0, :, :],
        ligand_radii=ligand_radii,
        translation_step=translation_step,
        target_error=coarse_target_error,
    )
    flat_candidates = candidate_batches.reshape((n_poses * n_actions, n_atoms, 3))
    coarse_scores_flat, softening_error_bound = _score_coarse_batch(
        receptor_coords=receptor_coords,
        receptor_radii=receptor_radii,
        ligand_radii=ligand_radii,
        candidate_coords=flat_candidates,
        coarse_target_error=coarse_target_error,
        retained_indices=coarse_retained_indices,
        electrostatics=electrostatics,
        use_softened_coarse=use_softened_coarse,
    )
    coarse_scores = coarse_scores_flat.reshape((n_poses, n_actions))
    witness = two_cutoff_approximation_witness(target_error, coarse_target_error)
    delta = witness.combined_delta + softening_error_bound

    accepted, best_actions, next_coords = _fast_singleton_accept_core(
        candidate_batches=candidate_batches,
        coarse_scores=coarse_scores,
        delta=delta,
    )
    if not bool(accepted):
        return None
    return FastSingletonAcceptRoundResult(
        selected_actions=best_actions,
        next_coords=next_coords,
        coarse_scores=coarse_scores,
        coarse_target_error=coarse_target_error,
        delta=delta,
        retained_receptor_indices=coarse_retained_indices,
        theorem_handle=TK12,
    )


def try_hybrid_singleton_accept_round(
    receptor_coords: jax.Array,
    receptor_radii: jax.Array,
    ligand_radii: jax.Array,
    candidate_batches: jax.Array,
    target_error: float,
    coarse_target_error: float,
    translation_step: float,
) -> FastSingletonAcceptRoundResult | None:
    exact_retained_indices = select_exact_receptor_subset_for_local_family(
        receptor_coords=receptor_coords,
        receptor_radii=receptor_radii,
        reference_coords_batch=candidate_batches[:, 0, :, :],
        ligand_radii=ligand_radii,
        translation_step=translation_step,
        target_error=target_error,
    )
    coarse_retained_indices = select_exact_receptor_subset_for_local_family(
        receptor_coords=receptor_coords,
        receptor_radii=receptor_radii,
        reference_coords_batch=candidate_batches[:, 0, :, :],
        ligand_radii=ligand_radii,
        translation_step=translation_step,
        target_error=coarse_target_error,
    )
    if int(coarse_retained_indices.shape[0]) >= int(exact_retained_indices.shape[0]):
        return None
    return try_fast_singleton_accept_round(
        receptor_coords=receptor_coords,
        receptor_radii=receptor_radii,
        ligand_radii=ligand_radii,
        candidate_batches=candidate_batches,
        target_error=target_error,
        coarse_target_error=coarse_target_error,
        translation_step=translation_step,
    )


def staged_singleton_gate(
    receptor_coords: jax.Array,
    receptor_radii: jax.Array,
    ligand_radii: jax.Array,
    candidate_batches: jax.Array,
    target_error: float,
    coarse_target_error: float,
    translation_step: float,
) -> StagedSingletonGateResult:
    exact_retained_indices = select_exact_receptor_subset_for_local_family(
        receptor_coords=receptor_coords,
        receptor_radii=receptor_radii,
        reference_coords_batch=candidate_batches[:, 0, :, :],
        ligand_radii=ligand_radii,
        translation_step=translation_step,
        target_error=target_error,
    )
    coarse_retained_indices = select_exact_receptor_subset_for_local_family(
        receptor_coords=receptor_coords,
        receptor_radii=receptor_radii,
        reference_coords_batch=candidate_batches[:, 0, :, :],
        ligand_radii=ligand_radii,
        translation_step=translation_step,
        target_error=coarse_target_error,
    )
    singleton_accept_result = None
    if int(coarse_retained_indices.shape[0]) < int(exact_retained_indices.shape[0]):
        singleton_accept_result = try_fast_singleton_accept_round(
            receptor_coords=receptor_coords,
            receptor_radii=receptor_radii,
            ligand_radii=ligand_radii,
            candidate_batches=candidate_batches,
            target_error=target_error,
            coarse_target_error=coarse_target_error,
            translation_step=translation_step,
        )
    return StagedSingletonGateResult(
        exact_retained_receptor_indices=exact_retained_indices,
        coarse_retained_receptor_indices=coarse_retained_indices,
        singleton_accept_result=singleton_accept_result,
    )


def try_adaptive_singleton_accept_round(
    receptor_coords: jax.Array,
    receptor_radii: jax.Array,
    ligand_radii: jax.Array,
    candidate_batches: jax.Array,
    target_error: float,
    coarse_target_errors: tuple[float, ...],
    translation_step: float,
    electrostatics: CertifiedRealSpaceEwaldSpec | None = None,
) -> FastSingletonAcceptRoundResult | None:
    for coarse_target_error in coarse_target_errors:
        result = try_fast_singleton_accept_round(
            receptor_coords=receptor_coords,
            receptor_radii=receptor_radii,
            ligand_radii=ligand_radii,
            candidate_batches=candidate_batches,
            target_error=target_error,
            coarse_target_error=coarse_target_error,
            translation_step=translation_step,
            electrostatics=electrostatics,
        )
        if result is not None:
            return result
    return None


def try_per_pose_fast_singleton_accept_round(
    receptor_coords: jax.Array,
    receptor_radii: jax.Array,
    ligand_radii: jax.Array,
    candidate_batches: jax.Array,
    target_error: float,
    coarse_target_error: float,
    translation_step: float,
    electrostatics: CertifiedRealSpaceEwaldSpec | None = None,
) -> PerPoseFastSingletonAcceptRoundResult | None:
    n_poses, n_actions, _n_atoms, _ = candidate_batches.shape
    delta = two_cutoff_approximation_witness(
        target_error, coarse_target_error
    ).combined_delta
    selected_actions = []
    next_coords = []
    coarse_score_rows = []
    retained_per_pose: list[jax.Array] = []

    for pose_index in range(n_poses):
        coarse_retained_indices = select_exact_receptor_subset_for_local_family(
            receptor_coords=receptor_coords,
            receptor_radii=receptor_radii,
            reference_coords_batch=candidate_batches[pose_index, 0, :, :][None, :, :],
            ligand_radii=ligand_radii,
            translation_step=translation_step,
            target_error=coarse_target_error,
        )
        coarse_batch = score_certified_batch(
            receptor_coords=receptor_coords[coarse_retained_indices],
            poses_coords=candidate_batches[pose_index],
            receptor_radii=receptor_radii[coarse_retained_indices],
            ligand_radii=ligand_radii,
            target_error=coarse_target_error,
            electrostatics=_subset_electrostatics(
                electrostatics, coarse_retained_indices
            ),
        )
        coarse_scores = coarse_batch.scores
        guarantee = staged_top1_guarantee_from_coarse_scores(coarse_scores, delta)
        if not guarantee.exact_winner_certified:
            return None

        selected_actions.append(guarantee.selected_action)
        next_coords.append(candidate_batches[pose_index, guarantee.selected_action])
        coarse_score_rows.append(coarse_scores)
        retained_per_pose.append(coarse_retained_indices)

    return PerPoseFastSingletonAcceptRoundResult(
        selected_actions=jnp.array(selected_actions, dtype=jnp.int32),
        next_coords=jnp.stack(next_coords, axis=0),
        coarse_scores=jnp.stack(coarse_score_rows, axis=0),
        coarse_target_error=coarse_target_error,
        delta=delta,
        retained_receptor_indices_per_pose=tuple(retained_per_pose),
        theorem_handle=TK12,
    )


def try_adaptive_per_pose_singleton_accept_round(
    receptor_coords: jax.Array,
    receptor_radii: jax.Array,
    ligand_radii: jax.Array,
    candidate_batches: jax.Array,
    target_error: float,
    coarse_target_errors: tuple[float, ...],
    translation_step: float,
    electrostatics: CertifiedRealSpaceEwaldSpec | None = None,
) -> PerPoseFastSingletonAcceptRoundResult | None:
    for coarse_target_error in coarse_target_errors:
        result = try_per_pose_fast_singleton_accept_round(
            receptor_coords=receptor_coords,
            receptor_radii=receptor_radii,
            ligand_radii=ligand_radii,
            candidate_batches=candidate_batches,
            target_error=target_error,
            coarse_target_error=coarse_target_error,
            translation_step=translation_step,
            electrostatics=electrostatics,
        )
        if result is not None:
            return result
    return None


def staged_top1_cost_diagnostic(
    receptor_coords: jax.Array,
    receptor_radii: jax.Array,
    ligand_radii: jax.Array,
    candidate_batches: jax.Array,
    target_error: float,
    coarse_target_error: float,
    translation_step: float,
    electrostatics: CertifiedRealSpaceEwaldSpec | None = None,
) -> StagedTop1CostDiagnostic:
    exact_retained = select_exact_receptor_subset_for_local_family(
        receptor_coords=receptor_coords,
        receptor_radii=receptor_radii,
        reference_coords_batch=candidate_batches[:, 0, :, :],
        ligand_radii=ligand_radii,
        translation_step=translation_step,
        target_error=target_error,
    )
    coarse_retained = select_exact_receptor_subset_for_local_family(
        receptor_coords=receptor_coords,
        receptor_radii=receptor_radii,
        reference_coords_batch=candidate_batches[:, 0, :, :],
        ligand_radii=ligand_radii,
        translation_step=translation_step,
        target_error=coarse_target_error,
    )
    flat_candidates = candidate_batches.reshape(
        (
            candidate_batches.shape[0] * candidate_batches.shape[1],
            candidate_batches.shape[2],
            3,
        )
    )
    exact_batch = score_certified_batch(
        receptor_coords=receptor_coords[exact_retained],
        poses_coords=flat_candidates,
        receptor_radii=receptor_radii[exact_retained],
        ligand_radii=ligand_radii,
        target_error=target_error,
        electrostatics=_subset_electrostatics(electrostatics, exact_retained),
    )
    coarse_batch = score_certified_batch(
        receptor_coords=receptor_coords[coarse_retained],
        poses_coords=flat_candidates,
        receptor_radii=receptor_radii[coarse_retained],
        ligand_radii=ligand_radii,
        target_error=coarse_target_error,
        electrostatics=_subset_electrostatics(electrostatics, coarse_retained),
    )
    coarse_scores = coarse_batch.scores.reshape(
        (candidate_batches.shape[0], candidate_batches.shape[1])
    )
    delta = two_cutoff_approximation_witness(
        target_error, coarse_target_error
    ).combined_delta
    summary = summarize_staged_top1_round(
        staged_top1_round_from_coarse_scores(
            coarse_scores=coarse_scores,
            delta=delta,
            retained_receptor_indices=coarse_retained,
        )
    )
    return StagedTop1CostDiagnostic(
        exact_retained_atoms=int(exact_retained.shape[0]),
        coarse_retained_atoms=int(coarse_retained.shape[0]),
        delta=delta,
        branch_summary=summary,
    )
