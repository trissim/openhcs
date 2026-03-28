"""
End-to-End OpenHCS Pose Prediction Pipeline.

Ties together pure JAX batched generation and Enum-dispatched scoring
with multi-stage filtering and pocket-guided sampling.
"""

import inspect
import math
import os
import time
from abc import ABC, abstractmethod
from dataclasses import (
    MISSING,
    dataclass,
    field,
    fields as dataclass_fields,
    is_dataclass,
    replace,
)
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generic,
    List,
    Optional,
    Self,
    TypeVar,
    Union,
    cast,
)

import jax
import jax.numpy as jnp
import numpy as np
from scipy import special as scipy_special


def _configure_jax_compilation_cache() -> None:
    if os.environ.get("OPENHCS_DISABLE_JAX_CACHE") == "1":
        return
    cache_dir = os.environ.get("OPENHCS_JAX_CACHE_DIR")
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/openhcs/jax")
    try:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    except Exception:
        return


_configure_jax_compilation_cache()

from dq_dock_engine.docking.core import (
    BindingSite,
    BlindDockingPlan,
    CertifiedBindingSite,
    CertifiedBlindDockingResult,
    CertifiedBlindDockingPlan,
    CertifiedPocketFailureReason,
    DockingBox,
    GeometricBindingSite,
    GeometricBlindDockingPlan,
    GeometricBlindDockingResult,
    LigandContext,
    SamplingStrategy,
    ScoringEngine,
    ScoredPose,
    PoseVector,
    GapCertification,
    NativeCertification,
    CertificationDecision,
    ReturnedPoseCertification,
    ReturnedPoseContractDecision,
)
from dq_dock_engine.docking.charges import ChargeMethod, create_charge_assigner
from dq_dock_engine.docking.certified_runtime_plans import (
    ActiveConformerEnergyGapWitness,
    ActiveConformerReturnedPoseWitness,
    ActiveRigidEnergyGapWitness,
    CertifiedActiveSubsetBudget,
    CertifiedActiveSubsetBudgetFamily,
    CertifiedConformerCoveragePlan,
    CertifiedConformerAwareReturnedPoseWitness,
    CertifiedPipelineCostModel,
    CertifiedPipelineExecutionPlan,
    CertifiedPruningDeltaComponent,
    CertifiedPruningDeltaComponentKind,
    CertifiedPruningDeltaBudget,
    CertifiedRefinementBudget,
    CertifiedRefinementBudgetKind,
    CertifiedRigidSeedFamilyPlan,
    CertifiedRigidSeedRegionKind,
    CertifiedReturnedPoseProofPlan,
    CertifiedSeedBudgetCandidate,
    CertifiedSeedBudgetPlan,
    InactiveConformerReturnedPoseWitness,
    PoseSpecificImprovementBudget,
    PoseSpecificImprovementBudgetFamily,
    PoseSpecificImprovementBudgetKind,
    ReturnedPoseProofCase,
)
from dq_dock_engine.docking.scoring import (
    CertifiedBatchResult,
    CertifiedContactSurrogateSpec,
    CertifiedDirectionalHBondSpec,
    CertifiedMetalCoordinationSpec,
    CertifiedRealSpaceEwaldSpec,
    CertifiedRichChemistryPlan,
    cooperative_hbond_correction_bound,
    route_scoring,
    score_certified_lj,
)
from dq_dock_engine.docking.pocket_analysis import (
    CertifiedDetectedPocket,
    GeometricDetectedPocket,
    detect_certified_pocket,
    detect_geometric_pocket,
)
from dq_dock_engine.docking.formal_pruning import (
    ambiguity_band_mask,
    certified_pruning_certificate,
    coarse_top1_ambiguity_mask,
)
from dq_dock_engine.docking.scoring import (
    score_certified_softened_lj,
    score_certified_softened_lj_realspace_ewald,
)
from dq_dock_engine.docking.pocket_sampling import (
    extract_local_pocket_region,
    extract_local_pocket_region_view,
)
from dq_dock_engine.docking.placement import apply_poses
from dq_dock_engine.docking.chemistry_runtime import (
    HalogenBondInteractionTerm,
    PiCationInteractionTerm,
    PiStackingInteractionTerm,
    WaterMediatedHBondInteractionTerm,
    indexed_site_positions,
)
from dq_dock_engine.docking.se3_refinement import (
    RefinementCertificate,
    compute_se3_spectral_certificate,
    make_se3_energy_fn,
    make_se3_kinematics_fn,
    observe_gd_trajectory,
    observe_gd_trajectory_with_reason,
    optimize_certified_gd,
    optimize_certified_gd_with_reason,
    _pose_to_se3_params,
    _se3_params_to_pose,
)
from dq_dock_engine.docking_config import RefinementCertificationMode
from dq_dock_engine.docking.scoring_context import (
    CertifiedScoringContext,
    build_certified_scoring_context,
)
from dq_dock_engine.docking.conformer_search import (
    BranchAndBoundConfig,
    CertifiedCellLowerBoundState,
    RotatableBond,
    TorsionCell,
    TorsionStrainParams,
    build_torsion_kinematics,
    compute_raw_lj_lipschitz,
    compute_softened_lipschitz_constant,
    derive_mmff_torsion_current_headroom,
    detect_rotatable_bonds,
    derive_uff_torsion_barrier_heights,
    search_conformers,
    search_conformers_support_grid,
    search_conformers_sequential_scan,
)
from dq_dock_engine.docking.formal_handles import (
    ambiguity_output_energy_contract_theorem_handles,
    ambiguity_output_contract_theorem_handles,
    branch_and_bound_cross_docking_handles,
    conformer_coverage_theorem_handles,
    enriched_support_selection_transfer_theorem_handles,
    joint_pruning_budget_optimality_handles,
    omitted_channel_bound_theorem_handles,
    optimizer_output_contract_theorem_handles,
    pocket_cross_docking_handles,
    pose_specific_improvement_budget_theorem_handles,
    rigid_posewise_improvement_budget_theorem_handles,
    returned_pose_energy_guarantee_theorem_handles,
    returned_pose_guarantee_theorem_handles,
    rigid_seed_family_theorem_handles,
    rigid_seed_runtime_bridge_theorem_handles,
    receptor_flex_cross_docking_handles,
    receptor_flexibility_theorem_handles,
    seed_budget_minimality_theorem_handles,
)
from dq_dock_engine.physics.kernels import rigid_transform_3d
from dq_dock_engine.docking_config import (
    CertifiedScoringFamily,
    ConformerSearchMode,
    compute_certified_cutoff,
    DockingConfig,
    DockingMode,
    ExactChemistryMode,
    FormalRoundStrategy,
    OptimizerBackend,
    SofteningPolicy,
)

if TYPE_CHECKING:
    from dq_dock_engine.docking.formal_sampling import CertifiedGlobalActionFamily


def _runtime_profile_enabled() -> bool:
    value = os.environ.get("OPENHCS_PROFILE_RUNTIME", "")
    return value not in ("", "0", "false", "False")


def _certify_all_refined_enabled() -> bool:
    value = os.environ.get("OPENHCS_CERTIFY_ALL_REFINED", "")
    return value in ("1", "true", "True")


def _force_seed_probe_enabled() -> bool:
    value = os.environ.get("OPENHCS_FORCE_SEED_PROBE", "")
    return value in ("1", "true", "True")


def _runtime_profile_log(label: str, start_time: float) -> None:
    if _runtime_profile_enabled():
        print(
            f"[RUNTIME PROFILE] {label}: {time.perf_counter() - start_time:.3f}s",
            flush=True,
        )


# Certified Pruning Constants
# We use a fixed power-of-two size for the survivor set to stabilize XLA caching.
# Two-phase seed-budget derivation uses a small probe set to estimate a certified
# local contraction rate and derive the full seed budget mechanically from the
# target RMSD. These constants control only the calibration phase size.
SEED_BUDGET_PROBE_POSES = 16
SEED_BUDGET_PROBE_TOP_K = 4
RIGID_EXACT_INCUMBENT_PREFILTER_TOP_K = 16

# Exact certified rescoring can be invoked over the full canonical retain survivor set.
# Keep it chunked so theorem-honest Top-K pruning remains usable on large pose sets
# without allocating one giant batched exact-score tensor.
EXACT_RESCORING_CHUNK_SIZE = 2048
CERTIFIED_PRUNING_CHUNK_SIZE = 512
CERTIFIED_PRUNING_MONOLITHIC_THRESHOLD = 2048
FINAL_EXACT_RESCORING_PAD_SIZE = 64

# Formal local refinement expands each survivor into a finite local action family.
# Chunk that stage as well so large certified survivor sets do not need one giant
# action tensor resident on device at once.
FORMAL_REFINEMENT_CHUNK_SIZE = 512

RUNTIME_ORIENTATION_SIGNAL_UNCERTIFIED = "RUNTIME_ORIENTATION_SIGNAL_UNCERTIFIED"
RUNTIME_ORIENTATION_SIGNAL_INACTIVE = RUNTIME_ORIENTATION_SIGNAL_UNCERTIFIED
RUNTIME_RIGID_EQUIVALENCE_AMBIGUITY = "RUNTIME_RIGID_EQUIVALENCE_AMBIGUITY"
RUNTIME_RIGID_CLEARANCE_WITNESS_UNCERTIFIED = (
    "RUNTIME_RIGID_CLEARANCE_WITNESS_UNCERTIFIED"
)


def _merge_theorem_handles(*groups: tuple[str, ...]) -> tuple[str, ...]:
    merged: list[str] = []
    for group in groups:
        merged.extend(group)
    return tuple(dict.fromkeys(merged))


def _detect_rigid_equivalence_ambiguity(
    coords_batch: jnp.ndarray,
    scores: jnp.ndarray | np.ndarray,
    *,
    error_bound: float | None,
    target_rmsd: float,
    max_candidates: int = 8,
) -> bool:
    if (
        coords_batch.shape[0] < 2
        or error_bound is None
        or not math.isfinite(error_bound)
    ):
        return False

    from dq_dock_engine.docking.metrics import (
        compute_docking_rmsd_batched,
        compute_rmsd_batched,
    )

    limit = min(int(coords_batch.shape[0]), max_candidates)
    ranked = jnp.argsort(scores)[:limit]
    ranked_coords = coords_batch[ranked]
    ranked_scores = scores[ranked]
    if ranked_coords.shape[0] < 2:
        return False

    best_coords = ranked_coords[0]
    others = ranked_coords[1:]
    kabsch = np.asarray(jax.device_get(compute_rmsd_batched(others, best_coords)))
    raw = np.asarray(jax.device_get(compute_docking_rmsd_batched(others, best_coords)))
    score_gaps = np.abs(
        np.asarray(
            jax.device_get(ranked_scores[1:] - ranked_scores[0]), dtype=np.float64
        )
    )
    aligned_tol = target_rmsd / 2.0
    separated_tol = target_rmsd
    ambiguity_gap = 2.0 * float(error_bound)
    return bool(
        np.any(
            (kabsch <= aligned_tol)
            & (raw >= separated_tol)
            & (score_gaps <= ambiguity_gap)
        )
    )


def _certified_top1_gap(
    scores: jnp.ndarray | np.ndarray, error_bound: float | None
) -> bool:
    """Finite certified singleton top-1 check from a top-2 score gap.

    Lean: TK16 (`exact_top1_eq_singleton_of_coarse_energy_gap_margin`).
    Lower energies are better, so the theorem is applied to negated utilities.
    """
    if scores.shape[0] < 2 or error_bound is None or not math.isfinite(error_bound):
        return False
    scores_np = np.asarray(jax.device_get(scores), dtype=np.float64)
    top2 = np.argsort(scores_np)[:2]
    cert = GapCertification.from_energies(
        float(scores_np[top2[0]]),
        float(scores_np[top2[1]]),
        float(error_bound),
    )
    return cert.is_certified


def _shared_certified_singleton_top1(
    scores_a: jnp.ndarray | np.ndarray,
    error_bound_a: float | None,
    scores_b: jnp.ndarray | np.ndarray,
    error_bound_b: float | None,
) -> bool:
    """Both score families certify the same singleton top-1 winner.

    Lean: FLO22 (`exact_energy_gap_certified_choice_agreement`), combining TK16
    singleton certification for each score family with FLO21 witness agreement.
    """
    if scores_a.shape[0] == 0 or scores_b.shape[0] == 0:
        return False
    top1_a = int(jnp.argmin(scores_a))
    top1_b = int(jnp.argmin(scores_b))
    return (
        top1_a == top1_b
        and _certified_top1_gap(scores_a, error_bound_a)
        and _certified_top1_gap(scores_b, error_bound_b)
    )


def _orientation_margin_certified_singleton_top1(
    scoring_context: CertifiedScoringContext | None,
    scores: jnp.ndarray | np.ndarray,
    error_bound: float | None,
) -> bool:
    return bool(
        scoring_context is not None
        and scoring_context.rich_orientation_disambiguation_active()
        and _certified_top1_gap(scores, error_bound)
    )


def _orientation_margin_theorem_handles(
    scoring_context: CertifiedScoringContext | None,
) -> tuple[str, ...]:
    if scoring_context is None or scoring_context.rich_chemistry_plan is None:
        return ()
    return ("HB15", "AH11")


def _winner_ambiguity_size(
    scores: jnp.ndarray | np.ndarray,
    error_bound: float | None,
) -> int | None:
    if error_bound is None or not math.isfinite(error_bound):
        return None
    ambiguity_mask = np.asarray(
        jax.device_get(
            ambiguity_band_mask(jnp.asarray(scores), k=1, epsilon=2.0 * error_bound)
        ),
        dtype=bool,
    )
    return int(np.count_nonzero(ambiguity_mask))


def _winner_ambiguity_indices(
    scores: jnp.ndarray | np.ndarray,
    error_bound: float | None,
) -> tuple[int, ...]:
    if error_bound is None or not math.isfinite(error_bound):
        return ()
    ambiguity_mask = np.asarray(
        jax.device_get(
            ambiguity_band_mask(jnp.asarray(scores), k=1, epsilon=2.0 * error_bound)
        ),
        dtype=bool,
    )
    return tuple(int(index) for index in np.flatnonzero(ambiguity_mask).tolist())


def _conformer_runtime_contract_handles(
    *,
    coverage_plan: CertifiedConformerCoveragePlan | None,
    proof_case: ReturnedPoseProofCase,
) -> tuple[str, ...]:
    if coverage_plan is None:
        return ()
    base_handles = _merge_theorem_handles(
        coverage_plan.theorem_handles,
        conformer_coverage_theorem_handles(),
    )
    if proof_case == ReturnedPoseProofCase.CERTIFIED_SINGLETON:
        return _merge_theorem_handles(
            base_handles,
            optimizer_output_contract_theorem_handles(),
            returned_pose_guarantee_theorem_handles(),
        )
    if proof_case == ReturnedPoseProofCase.CERTIFIED_AMBIGUITY_SET:
        return _merge_theorem_handles(
            base_handles,
            optimizer_output_contract_theorem_handles(),
            ambiguity_output_contract_theorem_handles(),
        )
    if proof_case == ReturnedPoseProofCase.CERTIFIED_ENERGY_SINGLETON:
        return _merge_theorem_handles(
            base_handles,
            returned_pose_energy_guarantee_theorem_handles(),
        )
    if proof_case == ReturnedPoseProofCase.CERTIFIED_ENERGY_AMBIGUITY_SET:
        return _merge_theorem_handles(
            base_handles,
            ambiguity_output_energy_contract_theorem_handles(),
        )
    return base_handles


def _rigid_runtime_energy_contract_handles(
    *,
    rigid_seed_family_plan: CertifiedRigidSeedFamilyPlan,
) -> tuple[str, ...]:
    return _merge_theorem_handles(
        rigid_seed_family_plan.theorem_handles,
        rigid_seed_runtime_bridge_theorem_handles(),
        returned_pose_energy_guarantee_theorem_handles(),
    )


def _derive_rigid_energy_gap_witness(
    request: "PipelineDockingRequest",
) -> ActiveRigidEnergyGapWitness | None:
    if request.rigid_seed_family_plan is None or request.config is None:
        return None
    rigid_summary = request.rigid_seed_family_plan.debug_summary()
    if not bool(rigid_summary.get("csc63_csc97_ready", False)):
        return None
    translation_cover_radius = rigid_summary.get("translation_cover_radius_over_box")
    signed_witness_radius = rigid_summary.get(
        "quaternion_signed_distance_witness_radius"
    )
    if not isinstance(translation_cover_radius, (int, float)):
        return None
    if not isinstance(signed_witness_radius, (int, float)):
        return None
    base_coords = np.asarray(request.ligand_ctx.base_coords, dtype=np.float64)
    arm_bound = float(np.max(np.sum(np.abs(base_coords), axis=1)))
    cover_rmsd_radius = float(
        translation_cover_radius + 48.0 * arm_bound * float(signed_witness_radius)
    )
    cover_gap_budget = float(
        _derive_exact_score_lipschitz_constant(
            request.receptor_radii,
            request.ligand_ctx.base_radii,
            request.exact_chemistry_mode,
            request.config.softening_policy,
        )
        * cover_rmsd_radius
    )
    return ActiveRigidEnergyGapWitness(
        cover_rmsd_radius=cover_rmsd_radius,
        cover_gap_budget=cover_gap_budget,
        certified_energy_gap=cover_gap_budget,
        theorem_handles=_rigid_runtime_energy_contract_handles(
            rigid_seed_family_plan=request.rigid_seed_family_plan
        ),
    )


def _with_rigid_selection_gap(
    witness: ActiveRigidEnergyGapWitness,
    *,
    selection_gap_budget: float,
) -> ActiveRigidEnergyGapWitness:
    if selection_gap_budget <= 0.0:
        return witness
    return ActiveRigidEnergyGapWitness(
        cover_rmsd_radius=witness.cover_rmsd_radius,
        cover_gap_budget=witness.cover_gap_budget,
        certified_energy_gap=witness.cover_gap_budget + selection_gap_budget,
        theorem_handles=_merge_theorem_handles(
            witness.theorem_handles,
            enriched_support_selection_transfer_theorem_handles(),
        ),
        selection_gap_budget=selection_gap_budget,
        witness_handles=witness.witness_handles,
    )


def _combined20_base_subset_mask(survivor_global_indices: np.ndarray) -> np.ndarray:
    return np.asarray(survivor_global_indices % 20 < 8, dtype=bool)


def _select_base_anchored_rigid_candidate(
    *,
    request: "PipelineDockingRequest",
    survivor_scores: jnp.ndarray | np.ndarray,
    survivor_global_indices: np.ndarray,
) -> tuple[int, float, dict[str, object] | None]:
    if request.rigid_seed_family_plan is None:
        return (
            int(
                np.argmin(np.asarray(jax.device_get(survivor_scores), dtype=np.float64))
            ),
            0.0,
            None,
        )
    rigid_summary = request.rigid_seed_family_plan.debug_summary()
    if rigid_summary.get("quaternion_dictionary_mode") != "combined20":
        return (
            int(
                np.argmin(np.asarray(jax.device_get(survivor_scores), dtype=np.float64))
            ),
            0.0,
            None,
        )

    scores_np = np.asarray(jax.device_get(survivor_scores), dtype=np.float64)
    exact_winner_index = int(np.argmin(scores_np))
    base_mask = _combined20_base_subset_mask(survivor_global_indices)
    base_indices = np.flatnonzero(base_mask)
    if base_indices.size == 0:
        return (
            exact_winner_index,
            0.0,
            {
                "policy": "combined20_base_anchor",
                "applied": False,
                "reason": "no surviving base-subset candidates remained after certified pruning",
                "exact_winner_index": exact_winner_index,
            },
        )

    base_local_index = int(base_indices[np.argmin(scores_np[base_indices])])
    candidate_gap = max(
        0.0, float(scores_np[base_local_index] - scores_np[exact_winner_index])
    )
    if base_local_index == exact_winner_index:
        return (
            exact_winner_index,
            0.0,
            {
                "policy": "combined20_base_anchor",
                "applied": False,
                "reason": "exact enriched winner already lies in the certified base subset",
                "exact_winner_index": exact_winner_index,
                "base_winner_index": base_local_index,
                "base_subset_survivor_count": int(base_indices.size),
            },
        )

    return (
        base_local_index,
        candidate_gap,
        {
            "policy": "combined20_base_anchor",
            "applied": True,
            "reason": "returned pose anchored to exact base-subset winner with theorem-backed additive candidate-gap transfer",
            "exact_winner_index": exact_winner_index,
            "base_winner_index": base_local_index,
            "base_subset_survivor_count": int(base_indices.size),
            "selection_gap_budget": candidate_gap,
            "exact_winner_energy": float(scores_np[exact_winner_index]),
            "base_winner_energy": float(scores_np[base_local_index]),
        },
    )


def _derive_rigid_energy_gap_proof_plan(
    *,
    request: "PipelineDockingRequest",
    witness: ActiveRigidEnergyGapWitness,
    winner_theorem_handles: tuple[str, ...],
    winner_refinement_failure_reason: str,
    note: str,
) -> CertifiedReturnedPoseProofPlan:
    theorem_handles = _merge_theorem_handles(
        winner_theorem_handles,
        witness.theorem_handles,
        returned_pose_energy_guarantee_theorem_handles(),
    )
    return CertifiedReturnedPoseProofPlan(
        proof_case=ReturnedPoseProofCase.CERTIFIED_ENERGY_SINGLETON,
        target_rmsd=request.target_rmsd,
        winner_index=0,
        support_indices=(0,),
        ambiguity_band_width=0.0,
        atom_count=int(request.ligand_ctx.base_coords.shape[0]),
        dimension=int(3 * request.ligand_ctx.base_coords.shape[0]),
        total_gap_budget=witness.certified_energy_gap,
        theorem_handles=theorem_handles,
        conformer_witness=witness,
        winner_refinement_certificate_present=False,
        winner_basin_witness_source="missing",
        winner_refinement_failure_reason=winner_refinement_failure_reason,
        note=note,
    )


def _contains_theorem_handle_group(
    theorem_handles: tuple[str, ...], required_handles: tuple[str, ...]
) -> bool:
    present_handles = set(theorem_handles)
    return all(handle in present_handles for handle in required_handles)


def _has_conformer_returned_pose_certificate_chain(
    proof_plan: CertifiedReturnedPoseProofPlan,
) -> bool:
    witness = proof_plan.conformer_witness
    if not isinstance(witness, ActiveConformerReturnedPoseWitness):
        return False
    if proof_plan.proof_case != ReturnedPoseProofCase.CERTIFIED_SINGLETON:
        return False
    if not proof_plan.winner_singleton or proof_plan.total_gap_budget is None:
        return False
    if proof_plan.total_gap_budget > witness.target_energy_gap:
        return False
    return _contains_theorem_handle_group(
        proof_plan.theorem_handles,
        returned_pose_guarantee_theorem_handles(),
    )


def _has_conformer_energy_singleton_certificate_chain(
    proof_plan: CertifiedReturnedPoseProofPlan,
) -> bool:
    witness = proof_plan.conformer_witness
    if not isinstance(witness, ActiveConformerEnergyGapWitness):
        return False
    if proof_plan.proof_case != ReturnedPoseProofCase.CERTIFIED_ENERGY_SINGLETON:
        return False
    if not proof_plan.winner_singleton or proof_plan.total_gap_budget is None:
        return False
    if proof_plan.total_gap_budget > witness.certified_energy_gap:
        return False
    return _contains_theorem_handle_group(
        proof_plan.theorem_handles,
        returned_pose_energy_guarantee_theorem_handles(),
    )


def _has_rigid_energy_singleton_certificate_chain(
    proof_plan: CertifiedReturnedPoseProofPlan,
) -> bool:
    witness = proof_plan.conformer_witness
    if not isinstance(witness, ActiveRigidEnergyGapWitness):
        return False
    if proof_plan.proof_case != ReturnedPoseProofCase.CERTIFIED_ENERGY_SINGLETON:
        return False
    if not proof_plan.winner_singleton or proof_plan.total_gap_budget is None:
        return False
    if proof_plan.total_gap_budget > witness.certified_energy_gap:
        return False
    if not _contains_theorem_handle_group(
        proof_plan.theorem_handles,
        returned_pose_energy_guarantee_theorem_handles(),
    ):
        return False
    if witness.selection_gap_budget > 0.0:
        return _contains_theorem_handle_group(
            proof_plan.theorem_handles,
            enriched_support_selection_transfer_theorem_handles(),
        )
    return True


def _has_conformer_ambiguity_set_certificate_chain(
    proof_plan: CertifiedReturnedPoseProofPlan,
) -> bool:
    witness = proof_plan.conformer_witness
    if not isinstance(witness, ActiveConformerReturnedPoseWitness):
        return False
    if not proof_plan.ambiguity_set_certified or proof_plan.total_gap_budget is None:
        return False
    if proof_plan.total_gap_budget > witness.target_energy_gap:
        return False
    if len(proof_plan.support_indices) <= 1:
        return False
    return _contains_theorem_handle_group(
        proof_plan.theorem_handles,
        ambiguity_output_contract_theorem_handles(),
    ) and _contains_theorem_handle_group(
        proof_plan.theorem_handles,
        optimizer_output_contract_theorem_handles(),
    )


def _has_conformer_energy_ambiguity_set_certificate_chain(
    proof_plan: CertifiedReturnedPoseProofPlan,
) -> bool:
    witness = proof_plan.conformer_witness
    if not isinstance(witness, ActiveConformerEnergyGapWitness):
        return False
    if proof_plan.proof_case != ReturnedPoseProofCase.CERTIFIED_ENERGY_AMBIGUITY_SET:
        return False
    if proof_plan.total_gap_budget is None:
        return False
    if proof_plan.total_gap_budget > witness.certified_energy_gap:
        return False
    if len(proof_plan.support_indices) <= 1:
        return False
    return _contains_theorem_handle_group(
        proof_plan.theorem_handles,
        ambiguity_output_energy_contract_theorem_handles(),
    )


def _has_rigid_energy_ambiguity_set_certificate_chain(
    proof_plan: CertifiedReturnedPoseProofPlan,
) -> bool:
    witness = proof_plan.conformer_witness
    if not isinstance(witness, ActiveRigidEnergyGapWitness):
        return False
    if proof_plan.proof_case != ReturnedPoseProofCase.CERTIFIED_ENERGY_AMBIGUITY_SET:
        return False
    if proof_plan.total_gap_budget is None:
        return False
    if proof_plan.total_gap_budget > witness.certified_energy_gap + proof_plan.ambiguity_band_width:
        return False
    if len(proof_plan.support_indices) <= 1:
        return False
    if not _contains_theorem_handle_group(
        proof_plan.theorem_handles,
        ambiguity_output_energy_contract_theorem_handles(),
    ):
        return False
    if witness.selection_gap_budget > 0.0:
        return _contains_theorem_handle_group(
            proof_plan.theorem_handles,
            enriched_support_selection_transfer_theorem_handles(),
        )
    return True


def _missing_conformer_proof_requirements(
    *,
    conformer_coverage_plan: CertifiedConformerCoveragePlan | None,
    cover_rmsd_radius: float | None,
    cover_gap_budget: float | None,
    basin_mu_coord: float | None,
    target_energy_gap: float | None,
    total_gap_budget: float | None,
) -> tuple[str, ...]:
    missing: list[str] = []
    if conformer_coverage_plan is None:
        missing.append("coverage_plan")
    if cover_rmsd_radius is None:
        missing.append("cover_rmsd_radius")
    if cover_gap_budget is None:
        missing.append("cover_gap_budget")
    if basin_mu_coord is None:
        missing.append("basin_mu_coord")
    if target_energy_gap is None:
        missing.append("target_energy_gap")
    if total_gap_budget is None:
        missing.append("total_gap_budget")
    return tuple(missing)


def _derive_returned_pose_proof_plan(
    *,
    request: "PipelineDockingRequest",
    final_scores: jnp.ndarray | np.ndarray,
    final_error_bound: float | None,
    final_pose_coords: jnp.ndarray | np.ndarray | None,
    refinement_certificates: list[RefinementCertificate | None],
    winner_refinement_failure_reason: str | None = None,
    pose_basin_mu_coords: tuple[float | None, ...] | None = None,
    conformer_improved_mask: tuple[bool, ...] | None = None,
    conformer_coverage_plan: CertifiedConformerCoveragePlan | None,
    winner_theorem_handles: tuple[str, ...],
    do_conf: bool,
) -> CertifiedReturnedPoseProofPlan:
    scores_np = np.asarray(jax.device_get(final_scores), dtype=np.float64)
    if scores_np.shape[0] == 0:
        return CertifiedReturnedPoseProofPlan(
            proof_case=ReturnedPoseProofCase.NO_POSE,
            target_rmsd=request.target_rmsd,
            winner_index=None,
            support_indices=(),
            ambiguity_band_width=final_error_bound,
            atom_count=int(request.ligand_ctx.base_coords.shape[0]),
            dimension=int(3 * request.ligand_ctx.base_coords.shape[0]),
            total_gap_budget=None,
            theorem_handles=_merge_theorem_handles(winner_theorem_handles, ("TK11",)),
            note="no returned pose available",
        )

    winner_index = int(np.argmin(scores_np))
    winner_singleton = _certified_top1_gap(final_scores, final_error_bound)
    support_indices = (
        (winner_index,)
        if winner_singleton
        else _winner_ambiguity_indices(final_scores, final_error_bound)
    )
    if not support_indices:
        support_indices = (winner_index,)
    winner_refinement_cert = refinement_certificates[winner_index]
    print(
        f"[PROOF_PLAN] winner_index={winner_index}, winner_singleton={winner_singleton}, "
        f"refinement_certificates length={len(refinement_certificates)}, "
        f"winner_cert is None: {winner_refinement_cert is None}",
        flush=True,
    )
    winner_conformer_improved = False
    if conformer_improved_mask is not None and winner_index < len(
        conformer_improved_mask
    ):
        winner_conformer_improved = bool(conformer_improved_mask[winner_index])
    fallback_basin_mu_coord = None
    if pose_basin_mu_coords is not None and winner_index < len(pose_basin_mu_coords):
        fallback_basin_mu_coord = pose_basin_mu_coords[winner_index]
    basin_mu_coord = None
    basin_witness_source = "missing"
    if (
        winner_refinement_cert is not None
        and winner_refinement_cert.spectral is not None
    ):
        basin_mu_coord = float(winner_refinement_cert.spectral.mu_coord)
        basin_witness_source = "refinement_certificate"
    elif fallback_basin_mu_coord is not None:
        basin_mu_coord = fallback_basin_mu_coord
        basin_witness_source = "spectral_only"
    print(
        f"[PROOF_PLAN] basin_witness_source={basin_witness_source}, "
        f"fallback_basin_mu_coord={fallback_basin_mu_coord}, "
        f"winner_refinement_cert.spectral={winner_refinement_cert.spectral if winner_refinement_cert else 'N/A'}",
        flush=True,
    )
    cover_rmsd_radius = None
    cover_gap_budget = None
    if conformer_coverage_plan is not None:
        cover_rmsd_radius = float(
            conformer_coverage_plan.max_arm * conformer_coverage_plan.min_cell_radius
        )
        cover_gap_budget = float(
            conformer_coverage_plan.score_lipschitz_constant * cover_rmsd_radius
        )
    rigid_energy_gap_witness = (
        None if do_conf else _derive_rigid_energy_gap_witness(request)
    )
    print(
        f"[PROOF_PLAN] conformer_coverage_plan={conformer_coverage_plan is not None}, "
        f"cover_rmsd_radius={cover_rmsd_radius}, cover_gap_budget={cover_gap_budget}, do_conf={do_conf}",
        flush=True,
    )
    target_rmsd_energy_gap = (
        None
        if basin_mu_coord is None
        else float(
            basin_mu_coord
            * float(request.ligand_ctx.base_coords.shape[0])
            * request.target_rmsd**2
            / 2.0
        )
    )
    total_gap_budget = None
    band_width = 0.0
    if do_conf:
        band_width = (
            0.0
            if winner_singleton or final_error_bound is None
            else float(final_error_bound)
        )
        total_gap_budget = (
            None if cover_gap_budget is None else float(cover_gap_budget + band_width)
        )
    else:
        total_gap_budget = (
            0.0 if (winner_singleton and winner_refinement_cert is not None) else None
        )
    rigid_ambiguity_detected = bool(
        final_pose_coords is not None
        and _detect_rigid_equivalence_ambiguity(
            jnp.asarray(final_pose_coords),
            final_scores,
            error_bound=final_error_bound,
            target_rmsd=request.target_rmsd,
        )
    )
    has_complete_conformer_chain = bool(
        do_conf
        and conformer_coverage_plan is not None
        and cover_rmsd_radius is not None
        and cover_gap_budget is not None
        and basin_mu_coord is not None
        and target_rmsd_energy_gap is not None
        and total_gap_budget is not None
    )
    missing_conformer_requirements = _missing_conformer_proof_requirements(
        conformer_coverage_plan=conformer_coverage_plan,
        cover_rmsd_radius=cover_rmsd_radius,
        cover_gap_budget=cover_gap_budget,
        basin_mu_coord=basin_mu_coord,
        target_energy_gap=target_rmsd_energy_gap,
        total_gap_budget=total_gap_budget,
    )
    ambiguity_set_certified = bool(
        has_complete_conformer_chain
        and not winner_singleton
        and cast(float, total_gap_budget) <= cast(float, target_rmsd_energy_gap)
    )
    conformer_path_certified = bool(
        has_complete_conformer_chain
        and winner_singleton
        and cast(float, total_gap_budget) <= cast(float, target_rmsd_energy_gap)
    )
    energy_gap_certified = bool(
        do_conf
        and conformer_coverage_plan is not None
        and cover_rmsd_radius is not None
        and cover_gap_budget is not None
        and total_gap_budget is not None
    )
    proof_case = ReturnedPoseProofCase.DOWNGRADED
    downgrade_decision = (
        ReturnedPoseContractDecision.DOWNGRADED_NO_REFINEMENT_CERTIFICATE
    )
    note = None
    if ambiguity_set_certified:
        proof_case = ReturnedPoseProofCase.CERTIFIED_AMBIGUITY_SET
        downgrade_decision = None
        note = (
            "returned pose belongs to a certified ambiguity set whose members all satisfy "
            "the requested target_rmsd contract"
        )
    elif not do_conf and total_gap_budget is not None:
        proof_case = ReturnedPoseProofCase.CERTIFIED_SINGLETON
        downgrade_decision = None
    elif not do_conf and rigid_energy_gap_witness is not None and winner_singleton:
        proof_case = ReturnedPoseProofCase.CERTIFIED_ENERGY_SINGLETON
        total_gap_budget = rigid_energy_gap_witness.certified_energy_gap
        downgrade_decision = None
        note = (
            "returned rigid pose is certified to lie within the rigid-cover energy gap budget "
            f"({cast(float, total_gap_budget):.3f} kcal/mol) of the optimal covered rigid pose"
        )
    elif not do_conf and rigid_energy_gap_witness is not None and not winner_singleton:
        if final_error_bound is not None:
            proof_case = ReturnedPoseProofCase.CERTIFIED_ENERGY_AMBIGUITY_SET
            total_gap_budget = rigid_energy_gap_witness.certified_energy_gap + final_error_bound
            downgrade_decision = None
            note = (
                "every pose in the returned rigid ambiguity band is certified to lie within the combined "
                f"cover-plus-band energy gap budget ({cast(float, total_gap_budget):.3f} kcal/mol) "
                "of the optimal covered rigid pose"
            )
        else:
            downgrade_decision = (
                ReturnedPoseContractDecision.DOWNGRADED_NO_FINAL_SCORE_CERTIFICATE
            )
            note = "final exact rigid ambiguity band lacks an error bound to certify its width"
    elif rigid_ambiguity_detected:
        proof_case = ReturnedPoseProofCase.RIGID_AMBIGUITY
        downgrade_decision = None
        note = "rigid-equivalence ambiguity prevents a unique returned-pose guarantee"
    elif not winner_singleton:
        downgrade_decision = (
            ReturnedPoseContractDecision.DOWNGRADED_NO_FINAL_SCORE_CERTIFICATE
        )
        note = "final exact winner does not have a certified singleton top-1 gap"
    elif total_gap_budget is None:
        downgrade_decision = (
            ReturnedPoseContractDecision.DOWNGRADED_NO_REFINEMENT_CERTIFICATE
        )
        note = "returned winner lacks the quantitative witness budget needed for certification"
        if not do_conf and winner_refinement_failure_reason is not None:
            note += f" (rigid refinement failure: {winner_refinement_failure_reason})"
        else:
            note += f" (missing: {', '.join(missing_conformer_requirements)})"
    elif conformer_path_certified:
        proof_case = ReturnedPoseProofCase.CERTIFIED_SINGLETON
        downgrade_decision = None
    elif has_complete_conformer_chain:
        downgrade_decision = ReturnedPoseContractDecision.DOWNGRADED_CONFORMER_PATH
        note = (
            "conformer-active returned pose is missing the full support-aware theorem chain "
            "needed for a certified target_rmsd guarantee"
        )
    elif energy_gap_certified and winner_singleton:
        proof_case = ReturnedPoseProofCase.CERTIFIED_ENERGY_SINGLETON
        downgrade_decision = None
        note = (
            "returned pose is certified to lie within the conformer-cover energy gap budget "
            f"({cast(float, total_gap_budget):.3f} kcal/mol) of the optimal covered conformer"
        )
    elif energy_gap_certified:
        proof_case = ReturnedPoseProofCase.CERTIFIED_ENERGY_AMBIGUITY_SET
        downgrade_decision = None
        note = (
            "every pose in the returned ambiguity band is certified to lie within the combined "
            f"cover-plus-band energy gap budget ({cast(float, total_gap_budget):.3f} kcal/mol) "
            "of the optimal covered conformer"
        )
    else:
        downgrade_decision = (
            ReturnedPoseContractDecision.DOWNGRADED_NO_REFINEMENT_CERTIFICATE
        )
        if winner_conformer_improved:
            note = (
                "conformer-active returned pose improved beyond the certified rigid refinement witness; "
                "a conformer-updated winner currently invalidates the rigid refinement certificate"
                f" (missing: {', '.join(missing_conformer_requirements)})"
            )
        else:
            note = (
                "conformer-active returned pose lacks the quantitative cover/refinement witnesses "
                "needed for certification"
                f" (missing: {', '.join(missing_conformer_requirements)})"
            )
    conformer_witness: CertifiedConformerAwareReturnedPoseWitness | None = None
    conformer_handles = ()
    if do_conf:
        if proof_case in (
            ReturnedPoseProofCase.CERTIFIED_SINGLETON,
            ReturnedPoseProofCase.CERTIFIED_AMBIGUITY_SET,
            ReturnedPoseProofCase.CERTIFIED_ENERGY_SINGLETON,
            ReturnedPoseProofCase.CERTIFIED_ENERGY_AMBIGUITY_SET,
        ):
            conformer_handles = _conformer_runtime_contract_handles(
                coverage_plan=conformer_coverage_plan,
                proof_case=proof_case,
            )
        if has_complete_conformer_chain:
            conformer_witness = ActiveConformerReturnedPoseWitness(
                coverage_plan=cast(
                    CertifiedConformerCoveragePlan, conformer_coverage_plan
                ),
                cover_rmsd_radius=cast(float, cover_rmsd_radius),
                cover_gap_budget=cast(float, cover_gap_budget),
                basin_mu_coord=cast(float, basin_mu_coord),
                target_energy_gap=cast(float, target_rmsd_energy_gap),
                theorem_handles=_merge_theorem_handles(
                    winner_theorem_handles,
                    conformer_handles,
                    () if winner_refinement_cert is None else ("ERC39",),
                ),
            )
        elif proof_case in (
            ReturnedPoseProofCase.CERTIFIED_ENERGY_SINGLETON,
            ReturnedPoseProofCase.CERTIFIED_ENERGY_AMBIGUITY_SET,
        ):
            conformer_witness = ActiveConformerEnergyGapWitness(
                coverage_plan=cast(
                    CertifiedConformerCoveragePlan, conformer_coverage_plan
                ),
                cover_rmsd_radius=cast(float, cover_rmsd_radius),
                cover_gap_budget=cast(float, cover_gap_budget),
                certified_energy_gap=cast(float, total_gap_budget),
                theorem_handles=_merge_theorem_handles(
                    winner_theorem_handles,
                    conformer_handles,
                ),
            )
        else:
            conformer_witness = InactiveConformerReturnedPoseWitness(
                theorem_handles=winner_theorem_handles,
                note=note,
            )
    elif winner_index is not None:
        conformer_witness = InactiveConformerReturnedPoseWitness(
            theorem_handles=winner_theorem_handles,
            note="conformer search inactive for returned-pose certification",
        )
    if (
        not do_conf
        and rigid_energy_gap_witness is not None
        and proof_case in (
            ReturnedPoseProofCase.CERTIFIED_ENERGY_SINGLETON,
            ReturnedPoseProofCase.CERTIFIED_ENERGY_AMBIGUITY_SET,
        )
    ):
        conformer_witness = rigid_energy_gap_witness
    proof_case_handles = ()
    if proof_case == ReturnedPoseProofCase.CERTIFIED_SINGLETON and not do_conf:
        proof_case_handles = returned_pose_guarantee_theorem_handles()
    elif proof_case == ReturnedPoseProofCase.CERTIFIED_ENERGY_SINGLETON:
        proof_case_handles = returned_pose_energy_guarantee_theorem_handles()
    elif proof_case == ReturnedPoseProofCase.CERTIFIED_ENERGY_AMBIGUITY_SET:
        proof_case_handles = ambiguity_output_energy_contract_theorem_handles()
    return CertifiedReturnedPoseProofPlan(
        proof_case=proof_case,
        target_rmsd=request.target_rmsd,
        winner_index=winner_index,
        support_indices=support_indices,
        ambiguity_band_width=final_error_bound,
        atom_count=int(request.ligand_ctx.base_coords.shape[0]),
        dimension=int(3 * request.ligand_ctx.base_coords.shape[0]),
        total_gap_budget=total_gap_budget,
        theorem_handles=_merge_theorem_handles(
            winner_theorem_handles,
            conformer_handles,
            proof_case_handles,
            () if winner_refinement_cert is None else ("ERC39",),
        ),
        conformer_witness=conformer_witness,
        winner_refinement_certificate_present=winner_refinement_cert is not None,
        winner_basin_witness_source=basin_witness_source,
        winner_refinement_failure_reason=winner_refinement_failure_reason,
        winner_conformer_improved=winner_conformer_improved,
        missing_conformer_requirements=(
            missing_conformer_requirements
            if do_conf
            and proof_case
            not in (
                ReturnedPoseProofCase.CERTIFIED_ENERGY_SINGLETON,
                ReturnedPoseProofCase.CERTIFIED_ENERGY_AMBIGUITY_SET,
            )
            else ()
        ),
        downgrade_decision=downgrade_decision,
        note=note,
    )


def _build_returned_pose_certification(
    *,
    proof_plan: CertifiedReturnedPoseProofPlan,
) -> ReturnedPoseCertification | None:
    if (
        proof_plan.decision
        == ReturnedPoseContractDecision.CERTIFIED_AMBIGUITY_SET_TARGET_RMSD
    ):
        if not _has_conformer_ambiguity_set_certificate_chain(proof_plan):
            raise ValueError(
                "Certified ambiguity-set proof plan lacks a complete conformer certificate chain"
            )
    elif (
        proof_plan.decision
        == ReturnedPoseContractDecision.CERTIFIED_AMBIGUITY_SET_ENERGY_GAP
    ):
        if isinstance(proof_plan.conformer_witness, ActiveConformerEnergyGapWitness):
            if not _has_conformer_energy_ambiguity_set_certificate_chain(proof_plan):
                raise ValueError(
                    "Certified energy ambiguity-set proof plan lacks a complete conformer certificate chain"
                )
        elif isinstance(proof_plan.conformer_witness, ActiveRigidEnergyGapWitness):
            if not _has_rigid_energy_ambiguity_set_certificate_chain(proof_plan):
                raise ValueError(
                    "Certified energy ambiguity-set proof plan lacks a complete rigid certificate chain"
                )
        else:
            raise ValueError(
                "Certified energy ambiguity-set proof plan has invalid witness type"
            )
    elif (
        proof_plan.decision == ReturnedPoseContractDecision.CERTIFIED_TARGET_RMSD
        and isinstance(proof_plan.conformer_witness, ActiveConformerReturnedPoseWitness)
        and not _has_conformer_returned_pose_certificate_chain(proof_plan)
    ):
        raise ValueError(
            "Certified singleton proof plan lacks a complete conformer certificate chain"
        )
    elif (
        proof_plan.decision == ReturnedPoseContractDecision.CERTIFIED_ENERGY_GAP
        and isinstance(proof_plan.conformer_witness, ActiveConformerEnergyGapWitness)
        and not _has_conformer_energy_singleton_certificate_chain(proof_plan)
    ):
        raise ValueError(
            "Certified energy singleton proof plan lacks a complete conformer certificate chain"
        )
    elif (
        proof_plan.decision == ReturnedPoseContractDecision.CERTIFIED_ENERGY_GAP
        and isinstance(proof_plan.conformer_witness, ActiveRigidEnergyGapWitness)
        and not _has_rigid_energy_singleton_certificate_chain(proof_plan)
    ):
        raise ValueError(
            "Certified energy singleton proof plan lacks a complete rigid certificate chain"
        )
    if not proof_plan.has_winner:
        return ReturnedPoseCertification(
            decision=proof_plan.decision,
            target_rmsd=proof_plan.target_rmsd,
            theorem_handles=proof_plan.theorem_handles,
            certified_energy_gap=proof_plan.total_gap_budget,
            note=proof_plan.note or "no pose was available to certify",
        )
    return ReturnedPoseCertification(
        decision=proof_plan.decision,
        target_rmsd=proof_plan.target_rmsd,
        theorem_handles=proof_plan.theorem_handles,
        certified_energy_gap=proof_plan.total_gap_budget,
        winner_index=int(cast(int, proof_plan.winner_index)),
        ambiguity_size=proof_plan.ambiguity_size,
        support_indices=proof_plan.support_indices,
        note=proof_plan.note,
    )


@dataclass(frozen=True)
class CertifiedPoseGeneration:
    pose_vecs: PoseVector
    family: "CertifiedGlobalActionFamily | None"


BindingSiteT = TypeVar("BindingSiteT", bound=BindingSite)
DetectedPocketT = TypeVar("DetectedPocketT")
PlanT = TypeVar("PlanT", bound=BlindDockingPlan)


@dataclass(frozen=True, kw_only=True)
class BlindDockingPreparation:
    protein_coords: jnp.ndarray
    receptor_radii: jnp.ndarray
    receptor_elements: tuple[str, ...] | None
    precomputed_receptor_charges: jnp.ndarray | None
    box: DockingBox


@dataclass(frozen=True, kw_only=True)
class CertifiedPocketPreparation(BlindDockingPreparation):
    detected_pocket: CertifiedDetectedPocket | None
    plan: CertifiedBlindDockingPlan


@dataclass(frozen=True, kw_only=True)
class GeometricPocketPreparation(BlindDockingPreparation):
    detected_pocket: GeometricDetectedPocket | None
    plan: GeometricBlindDockingPlan


def derive_seed_budget(
    confidence: float,
    box_size: jnp.ndarray,
    target_rmsd: float,
    ligand_radius: float,
    n_torsions: int = 0,
    probe_certificate: RefinementCertificate | None = None,
    translation_search_volume: float | None = None,
) -> int:
    """Derive the rigid initial-seed count from capture probability.

    Uses the SE(3) covering number argument: the search volume (translation,
    rotation, and optionally torsion for a *joint* sampler) divided by a
    conservative capture-basin volume gives the expected number of trials needed.
    The confidence level converts this to a concrete seed count via the
    geometric distribution CDF inversion:

        N ≥ ln(1 - P) / ln(1 - V_basin / V_total)

    Lean: SeedBudgetDerivation.lean — sufficient_seed_budget (SB2),
    minSeedBudget_antitone (SB5), composed_two_phase_seed_budget (SB7).

    In the current docking pipeline, random seeds are rigid SE(3) seeds *per
    conformer*; conformer search is budgeted separately by the branch-and-bound
    support theorems. That means runtime callers should pass `n_torsions=0`
    unless they truly sample the joint rigid+tortion space at this stage.

    Without a probe certificate, this uses the theorem-honest zero-step capture
    basin: a seed is a hit only if it already lies within `target_rmsd` of the
    optimum. With a probe certificate, we only expand this radius when the probe
    certifies both a coordinate-space lower curvature `μ`, an upper curvature
    `M`, and a linear contraction factor `q` over `T` steps. The certified
    larger capture radius is then

        capture_rmsd = target_rmsd * sqrt(μ / (M * q^T)).

    Lean: EnergyRMSDConvergence.lean —
      rmsd_target_of_initial_rmsd_and_linear_energy_convergence.

    This removes the old ad hoc 2Å / π/4 / π/6 constants. The probe-based
    protocol remains conservative by underestimating the true basin volume.
    """
    import math

    if confidence <= 0.0 or confidence >= 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if target_rmsd <= 0.0:
        raise ValueError(f"target_rmsd must be positive, got {target_rmsd}")
    if n_torsions < 0:
        raise ValueError(f"n_torsions must be nonnegative, got {n_torsions}")

    capture_rmsd = _certified_capture_rmsd(
        target_rmsd=target_rmsd,
        probe_certificate=probe_certificate,
    )

    # --- Search volumes ---

    # Translation: box volume in Å³
    box_vol = (
        float(jnp.prod(box_size))
        if translation_search_volume is None
        else float(translation_search_volume)
    )

    # Rotation: SO(3) has volume 2π² ≈ 19.739
    so3_volume = 2.0 * math.pi**2

    # Torsion: [-π, π]^n_torsions
    torsion_search = (2.0 * math.pi) ** n_torsions if n_torsions > 0 else 1.0

    v_total = box_vol * so3_volume * torsion_search

    # --- Basin capture volumes ---
    # Translation: a pure translation by r shifts every atom by r, so RMSD=r.
    trans_capture = min(
        box_vol,
        (4.0 / 3.0) * math.pi * capture_rmsd**3,
    )

    # Rotation: a rigid rotation by angle θ moves atoms by at most ligand_radius * θ.
    if ligand_radius > 1e-6 and math.isfinite(capture_rmsd):
        rot_capture_radius = min(math.pi, capture_rmsd / ligand_radius)
        rot_capture = min(
            so3_volume,
            (4.0 / 3.0) * math.pi * rot_capture_radius**3,
        )
    else:
        rot_capture = so3_volume

    # Torsion: require every torsion component to fit inside an RMSD ball using
    # the ligand radius as a conservative arm-length bound.
    if n_torsions > 0:
        if ligand_radius > 1e-6 and math.isfinite(capture_rmsd):
            torsion_capture_radius = min(
                math.pi,
                capture_rmsd / (ligand_radius * math.sqrt(float(n_torsions))),
            )
            torsion_capture = min(
                torsion_search,
                (2.0 * torsion_capture_radius) ** n_torsions,
            )
        else:
            torsion_capture = torsion_search
    else:
        torsion_capture = 1.0

    v_basin = trans_capture * rot_capture * torsion_capture

    # --- Geometric CDF inversion (SB2/SB4) ---
    ratio = min(1.0, v_basin / v_total)
    if ratio >= 1.0:
        return 1  # basin covers the entire search space
    if ratio <= 0.0:
        raise ValueError("derived basin volume fraction must be positive")

    n_poses = math.ceil(math.log(1.0 - confidence) / math.log(1.0 - ratio))
    return max(1, n_poses)


def _certified_capture_rmsd(
    *,
    target_rmsd: float,
    probe_certificate: RefinementCertificate | None,
) -> float:
    import math

    capture_rmsd = target_rmsd
    if probe_certificate is not None and probe_certificate.n_steps > 0:
        q = float(probe_certificate.q)
        mu_coord = float(probe_certificate.spectral.mu_coord)
        M_coord = float(probe_certificate.spectral.M_coord)
        if (
            0.0 < q < 1.0
            and math.isfinite(mu_coord)
            and math.isfinite(M_coord)
            and mu_coord > 0.0
            and M_coord > 0.0
        ):
            contraction = math.pow(q, probe_certificate.n_steps)
            if contraction > 0.0 and math.isfinite(contraction):
                amplification_sq = mu_coord / (M_coord * contraction)
                if amplification_sq > 1.0 and math.isfinite(amplification_sq):
                    capture_rmsd = target_rmsd * math.sqrt(amplification_sq)
    return capture_rmsd


def _translation_search_volume_for_seed_family(
    *,
    box_size: jnp.ndarray,
    rigid_seed_box: DockingBox | None,
    certified_binding_site: CertifiedBindingSite | None,
) -> float:
    if certified_binding_site is not None:
        radius = float(certified_binding_site.radius)
        return (4.0 / 3.0) * math.pi * radius**3
    if rigid_seed_box is not None:
        return float(np.prod(np.asarray(rigid_seed_box.size, dtype=np.float64)))
    return float(jnp.prod(box_size))


def _derive_rigid_seed_family_plan_for_budget(
    *,
    rigid_seed_box: DockingBox | None,
    certified_binding_site: CertifiedBindingSite | None,
    budget: int,
    target_rmsd: float | None = None,
) -> CertifiedRigidSeedFamilyPlan | None:
    if rigid_seed_box is None:
        return None
    from dq_dock_engine.docking.formal_sampling import (
        derive_certified_rigid_seed_family_plan,
    )

    return derive_certified_rigid_seed_family_plan(
        rigid_seed_box,
        budget,
        certified_binding_site=certified_binding_site,
        target_translation_cover_radius=target_rmsd,
    )


def derive_seed_budget_plan(
    *,
    confidence: float,
    box_size: jnp.ndarray,
    target_rmsd: float,
    ligand_radius: float,
    n_torsions: int = 0,
    probe_candidates: tuple[tuple[int, RefinementCertificate | None], ...] = (),
    rigid_seed_box: DockingBox | None = None,
    certified_binding_site: CertifiedBindingSite | None = None,
) -> CertifiedSeedBudgetPlan:
    translation_search_volume = _translation_search_volume_for_seed_family(
        box_size=box_size,
        rigid_seed_box=rigid_seed_box,
        certified_binding_site=certified_binding_site,
    )
    candidate_specs: list[dict[str, object]] = []
    baseline_budget = derive_seed_budget(
        confidence=confidence,
        box_size=box_size,
        target_rmsd=target_rmsd,
        ligand_radius=ligand_radius,
        n_torsions=n_torsions,
        probe_certificate=None,
        translation_search_volume=translation_search_volume,
    )
    candidate_specs.append(
        {
            "budget": baseline_budget,
            "source": "baseline_zero_step_capture",
            "adequate": True,
            "capture_rmsd": _certified_capture_rmsd(
                target_rmsd=target_rmsd,
                probe_certificate=None,
            ),
            "theorem_handles": ("SB2", "SB5"),
            "probe_rank": None,
            "provenance_note": None,
        }
    )
    for probe_rank, probe_certificate in probe_candidates:
        if probe_certificate is None:
            continue
        probe_budget = derive_seed_budget(
            confidence=confidence,
            box_size=box_size,
            target_rmsd=target_rmsd,
            ligand_radius=ligand_radius,
            n_torsions=n_torsions,
            probe_certificate=probe_certificate,
            translation_search_volume=translation_search_volume,
        )
        candidate_specs.append(
            {
                "budget": probe_budget,
                "source": "observed_probe_capture",
                "adequate": True,
                "capture_rmsd": _certified_capture_rmsd(
                    target_rmsd=target_rmsd,
                    probe_certificate=probe_certificate,
                ),
                "theorem_handles": ("SB2", "SB5", "SB7", "ERC43"),
                "probe_rank": probe_rank,
                "provenance_note": (
                    "Probe ranking limits are engineering calibration bounds and do not "
                    "change the selected theorem-backed seed budget fact once a probe "
                    "certificate is admitted"
                ),
            }
        )
    selected_index = min(
        range(len(candidate_specs)),
        key=lambda idx: (cast(int, candidate_specs[idx]["budget"]), idx),
    )
    selected_family_plan = _derive_rigid_seed_family_plan_for_budget(
        rigid_seed_box=rigid_seed_box,
        certified_binding_site=certified_binding_site,
        budget=cast(int, candidate_specs[selected_index]["budget"]),
        target_rmsd=target_rmsd,
    )
    candidates = tuple(
        CertifiedSeedBudgetCandidate(
            budget=cast(int, spec["budget"]),
            source=cast(str, spec["source"]),
            adequate=cast(bool, spec["adequate"]),
            capture_rmsd=cast(float, spec["capture_rmsd"]),
            theorem_handles=cast(tuple[str, ...], spec["theorem_handles"]),
            rigid_seed_family_plan=(
                selected_family_plan if idx == selected_index else None
            ),
            probe_rank=cast(int | None, spec["probe_rank"]),
            provenance_note=cast(str | None, spec["provenance_note"]),
        )
        for idx, spec in enumerate(candidate_specs)
    )
    return CertifiedSeedBudgetPlan(
        candidates=candidates,
        selected_index=selected_index,
        theorem_handles=seed_budget_minimality_theorem_handles(),
        engineering_probe_pose_cap=SEED_BUDGET_PROBE_POSES,
        engineering_probe_top_k=SEED_BUDGET_PROBE_TOP_K,
    )


def _ligand_radius(ligand_ctx: LigandContext) -> float:
    # `base_coords` are already centered in build_ligand_context, so the ligand
    # radius is just the farthest atom from the origin in that centered frame.
    return float(jnp.max(jnp.linalg.norm(ligand_ctx.base_coords, axis=-1)))


def _fit_pose_vector_from_coords(
    base_coords: jnp.ndarray,
    world_coords: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Recover the rigid translation/quaternion mapping `base_coords` to `world_coords`.

    Formal round refinement operates directly on coordinates. The theorem-backed
    SE(3) refinement layer expects the equivalent rigid pose vector, so we recover
    it deterministically with the Kabsch solution.
    """
    from scipy.spatial.transform import Rotation

    base_np = np.asarray(base_coords, dtype=np.float64)
    world_np = np.asarray(world_coords, dtype=np.float64)
    base_mean = base_np.mean(axis=0)
    world_mean = world_np.mean(axis=0)
    base_centered = base_np - base_mean
    world_centered = world_np - world_mean
    covariance = base_centered.T @ world_centered
    u, _, vt = np.linalg.svd(covariance)
    # `rigid_transform_3d` applies coordinates as `coords @ rotation.T + translation`.
    # For this row-vector convention, the Kabsch solution is R = V U^T.
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = world_mean - base_mean @ rotation.T
    quat_xyzw = Rotation.from_matrix(rotation).as_quat()
    quat_wxyz = np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float32,
    )
    dtype = world_coords.dtype
    return (
        jnp.asarray(translation, dtype=dtype),
        jnp.asarray(quat_wxyz, dtype=dtype),
    )


def _fit_pose_vectors_from_coords_batch(
    base_coords: jnp.ndarray,
    world_coords_batch: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    translations: list[jnp.ndarray] = []
    quaternions: list[jnp.ndarray] = []
    for coords in np.asarray(world_coords_batch):
        translation, quaternion = _fit_pose_vector_from_coords(
            base_coords,
            jnp.asarray(coords, dtype=world_coords_batch.dtype),
        )
        translations.append(translation)
        quaternions.append(quaternion)
    return jnp.stack(translations), jnp.stack(quaternions)


def _rotatable_bond_count(ligand_ctx: LigandContext) -> int:
    adjacency = ligand_ctx.adjacency
    elements = ligand_ctx.elements
    if adjacency is None or not elements:
        return 0
    coords_np = np.asarray(ligand_ctx.base_coords, dtype=np.float32)
    return len(detect_rotatable_bonds(adjacency, coords_np, elements))


def _min_pairwise_sigma_from_radii(
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
) -> float:
    if receptor_radii.size == 0 or ligand_radii.size == 0:
        raise ValueError("Cannot derive sigma without receptor and ligand radii")
    return float(jnp.min(receptor_radii) + jnp.min(ligand_radii))


def _derive_local_rotation_step_rad(base_step: float, ligand_radius: float) -> float:
    """Derive a local rigid-rotation step from the translational cell scale.

    The formal local optimizer uses rigid moves whose translation and
    rotation-induced pointwise displacements should live on the same geometric
    scale. If the translation shell starts at `base_step`, the matching rigid
    rotation shell is the angle whose maximal atomic displacement is also
    `base_step`, i.e. `angle = base_step / ligand_radius`.

    We cap at `pi/2`, the coarsest global-orientation shell already assumed by
    the certified SE(3) refinement round budget.
    """
    if base_step <= 0.0:
        raise ValueError(f"base_step must be positive, got {base_step}")
    if ligand_radius <= 1e-8:
        return float(jnp.pi / 2.0)
    return min(float(jnp.pi / 2.0), base_step / ligand_radius)


def _derive_realspace_ewald_lipschitz_constant(
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    electrostatics: CertifiedRealSpaceEwaldSpec | None,
) -> float:
    """Conservative pairwise Lipschitz bound for real-space Ewald electrostatics.

    For a single pair with charge product `Q` and kernel

        f(r) = Q * erfc(alpha * r) / (dielectric * r),

    the radial derivative satisfies

        |f'(r)| <= |Q| / dielectric * (2 * alpha / (sqrt(pi) * r) + 1 / r^2)

    because `exp(-(alpha r)^2) <= 1` and `erfc(alpha r) <= 1`. Summing the
    per-pair bounds gives a conservative coordinate-space Lipschitz constant for
    the rigid local-optimization objective.

    Closest Lean support today lives in
    `ScreenedCoulombApproximation.lean` / `ConditionalComposition.lean` for the
    exact screened-Coulomb kernel and cutoff/tail control; this explicit radial
    derivative enclosure is the runtime-side bridge still to mechanize.
    """
    if electrostatics is None:
        return 0.0
    r_min = _min_pairwise_sigma_from_radii(receptor_radii, ligand_radii)
    if r_min <= 0.0:
        raise ValueError(f"r_min must be positive, got {r_min}")
    charge_sum = float(
        jnp.sum(
            jnp.abs(
                electrostatics.receptor_charges[:, None]
                * electrostatics.ligand_charges[None, :]
            )
        )
    )
    if charge_sum <= 0.0:
        return 0.0
    alpha = float(electrostatics.alpha)
    dielectric = float(electrostatics.dielectric)
    if alpha <= 0.0 or dielectric <= 0.0:
        raise ValueError(
            f"real-space Ewald requires positive alpha/dielectric, got {alpha}, {dielectric}"
        )
    radial_bound = (2.0 * alpha) / (math.sqrt(math.pi) * r_min) + 1.0 / (r_min**2)
    return (charge_sum / dielectric) * radial_bound


def _derive_optimization_score_lipschitz_constant(
    request: "PipelineDockingRequest",
    scoring_context: CertifiedScoringContext,
) -> float:
    """Physics-derived Lipschitz constant for the formal local optimizer score."""
    from dq_dock_engine.docking.scoring import _EPSILON_KCAL_MOL

    min_sigma = _min_pairwise_sigma_from_radii(
        request.receptor_radii,
        request.ligand_ctx.base_radii,
    )
    lj_lipschitz = compute_raw_lj_lipschitz(_EPSILON_KCAL_MOL, min_sigma)
    electro_lipschitz = _derive_realspace_ewald_lipschitz_constant(
        request.receptor_radii,
        request.ligand_ctx.base_radii,
        scoring_context.electrostatics,
    )
    return lj_lipschitz + electro_lipschitz


def _derive_subset_optimization_score_lipschitz_constant(
    *,
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    electrostatics: CertifiedRealSpaceEwaldSpec | None,
) -> float:
    from dq_dock_engine.docking.scoring import _EPSILON_KCAL_MOL

    min_sigma = _min_pairwise_sigma_from_radii(receptor_radii, ligand_radii)
    lj_lipschitz = compute_raw_lj_lipschitz(_EPSILON_KCAL_MOL, min_sigma)
    electro_lipschitz = _derive_realspace_ewald_lipschitz_constant(
        receptor_radii,
        ligand_radii,
        electrostatics,
    )
    return lj_lipschitz + electro_lipschitz


def _dyadic_action_path_displacement_bound(
    base_translation_step: float,
    base_rotation_step_rad: float,
    ligand_radius: float,
    n_rounds: int,
) -> float:
    """Worst-case pointwise displacement over a dyadic rigid local-search path.

    One certified local action is taken per round. The action family offers
    either a pure translation step or a pure rotation step, so the maximal
    pointwise displacement at round `r` is

        max(trans_r, ligand_radius * rot_r)

    rather than their sum. Summing that dyadic schedule over the remaining
    rounds yields a theorem-aligned bound on total coordinate motion available to
    the local optimizer.
    """
    if base_translation_step <= 0.0:
        raise ValueError(
            f"base_translation_step must be positive, got {base_translation_step}"
        )
    if n_rounds <= 0:
        raise ValueError(f"n_rounds must be positive, got {n_rounds}")
    base_rotation_displacement = ligand_radius * base_rotation_step_rad
    base_round_displacement = max(base_translation_step, base_rotation_displacement)
    return float(
        sum(
            base_round_displacement / (2.0**round_index)
            for round_index in range(n_rounds)
        )
    )


def _rigid_local_improvement_bound(
    request: "PipelineDockingRequest",
    scoring_context: CertifiedScoringContext,
    *,
    base_translation_step: float,
    base_rotation_step_rad: float,
    ligand_radius: float,
    n_rounds: int,
) -> float:
    """Bound the total possible score improvement from rigid local refinement.

    Runtime composition of existing proof families:

      - `LipschitzStepBounds.lean::descent_bounded_change`
      - `LipschitzStepBounds.lean::n_step_error_bound`
      - `SupportExpansion.lean::dyadicTranslationStep`

    applied to the exact optimization score and the dyadic rigid-action family.
    """
    lipschitz = _derive_optimization_score_lipschitz_constant(request, scoring_context)
    path_displacement = _dyadic_action_path_displacement_bound(
        base_translation_step,
        base_rotation_step_rad,
        ligand_radius,
        n_rounds,
    )
    return lipschitz * path_displacement


_LJ_SINGULARITY_FLOOR = 1.0e-10


def _exact_lj_pair_score_with_floor(
    *,
    epsilon_pair: float,
    sigma: np.ndarray,
    distance: np.ndarray,
) -> np.ndarray:
    positive_mask = distance > _LJ_SINGULARITY_FLOOR
    score = np.full(distance.shape, 1.0e12, dtype=np.float64)
    if not np.any(positive_mask):
        return score
    sigma_pos = np.asarray(sigma[positive_mask], dtype=np.float64)
    dist_pos = np.asarray(distance[positive_mask], dtype=np.float64)
    sr = sigma_pos / dist_pos
    sr6 = sr**6
    score[positive_mask] = 4.0 * float(epsilon_pair) * (sr6 * sr6 - sr6)
    return score


def _exact_lj_interval_lower_bound(
    *,
    epsilon_pair: float,
    sigma: np.ndarray,
    current_distance: np.ndarray,
    max_displacement: np.ndarray,
    cutoff_safe: np.ndarray,
) -> np.ndarray:
    """Piecewise theorem-backed lower bound for reachable exact LJ on an interval.

    Uses:
    - `LJ27`: repulsive-side right endpoint lower bound above floor
    - `LJ28`: repulsive-side right endpoint lower bound with floor clamp
    - `LJ26`: attractive/tail-side left endpoint lower bound
    - `LJ25`: global fallback lower bound `-epsilon`
    """

    reachable_lo = current_distance - max_displacement
    reachable_hi = current_distance + max_displacement
    in_range_possible = reachable_lo <= cutoff_safe

    lj_min_distance = np.asarray(
        sigma * float(2.0 ** (1.0 / 6.0)),
        dtype=np.float64,
    )
    floor_safe = reachable_lo > _LJ_SINGULARITY_FLOOR
    repulsive_mask = reachable_hi <= lj_min_distance
    tail_mask = np.logical_and(floor_safe, reachable_lo >= lj_min_distance)

    repulsive_lower = np.minimum(
        _exact_lj_pair_score_with_floor(
            epsilon_pair=epsilon_pair,
            sigma=sigma,
            distance=reachable_hi,
        ),
        (10.0**12.0),
    )
    tail_lower = _exact_lj_pair_score_with_floor(
        epsilon_pair=epsilon_pair,
        sigma=sigma,
        distance=reachable_lo,
    )

    lower_in_range = np.where(
        repulsive_mask,
        repulsive_lower,
        np.where(tail_mask, tail_lower, -epsilon_pair),
    )
    return np.where(in_range_possible, lower_in_range, 0.0)


def _dyadic_action_path_displacement_bound_per_atom(
    base_translation_step: float,
    base_rotation_step_rad: float,
    atom_arm_lengths: np.ndarray,
    n_rounds: int,
) -> np.ndarray:
    scales = np.array([1.0 / (2.0**k) for k in range(n_rounds)], dtype=np.float64)
    trans = base_translation_step * scales[None, :]
    rot = (
        np.asarray(atom_arm_lengths, dtype=np.float64)[:, None]
        * base_rotation_step_rad
        * scales[None, :]
    )
    return np.sum(np.maximum(trans, rot), axis=1)


def _per_receptor_atom_rigid_omission_bounds(
    *,
    epsilon_pair: float,
    dists: np.ndarray,
    sigma: np.ndarray,
    atom_displacements: np.ndarray,
    cutoff: float,
    receptor_charges_local: np.ndarray | None,
    ligand_charges: np.ndarray | None,
    electro_cutoff: float | None,
    electro_alpha: float | None,
    electro_dielectric: float | None,
) -> np.ndarray:
    disp = np.broadcast_to(
        np.asarray(atom_displacements, dtype=np.float64)[None, :], dists.shape
    )
    cutoff_safe = np.maximum(cutoff, sigma)
    current_in_range = dists < cutoff_safe
    current_lj = np.where(
        current_in_range,
        _exact_lj_pair_score_with_floor(
            epsilon_pair=epsilon_pair,
            sigma=sigma,
            distance=dists,
        ),
        0.0,
    )
    lower_lj = _exact_lj_interval_lower_bound(
        epsilon_pair=epsilon_pair,
        sigma=sigma,
        current_distance=dists,
        max_displacement=disp,
        cutoff_safe=cutoff_safe,
    )
    bounds = np.sum(np.maximum(0.0, current_lj - lower_lj), axis=1, dtype=np.float64)

    if (
        receptor_charges_local is not None
        and ligand_charges is not None
        and electro_cutoff is not None
        and electro_alpha is not None
        and electro_dielectric is not None
    ):
        charge_products = (
            receptor_charges_local[:, None] * ligand_charges[None, :]
        ) / electro_dielectric
        electro_current_in_range = dists < electro_cutoff
        safe_electro_dists = np.where(
            electro_current_in_range,
            np.maximum(dists, 1.0e-6),
            electro_cutoff,
        )
        electro_current = np.where(
            electro_current_in_range,
            charge_products
            * scipy_special.erfc(electro_alpha * safe_electro_dists)
            / safe_electro_dists,
            0.0,
        )
        reach_lo = dists - disp
        reach_hi = dists + disp
        electro_lower = np.where(
            reach_lo > electro_cutoff,
            0.0,
            np.where(
                charge_products < 0.0,
                charge_products
                * scipy_special.erfc(electro_alpha * np.maximum(reach_lo, 1.0e-6))
                / np.maximum(reach_lo, 1.0e-6),
                charge_products
                * scipy_special.erfc(electro_alpha * np.maximum(reach_hi, 1.0e-6))
                / np.maximum(reach_hi, 1.0e-6),
            ),
        )
        bounds += np.sum(
            np.maximum(0.0, electro_current - electro_lower),
            axis=1,
            dtype=np.float64,
        )

    return np.asarray(bounds, dtype=np.float64)


def _posewise_directional_hbond_current_upper_bound(
    request: "PipelineDockingRequest",
    spec: CertifiedDirectionalHBondSpec,
    *,
    poses_coords: jnp.ndarray,
) -> np.ndarray:
    poses_np = np.asarray(jax.device_get(poses_coords), dtype=np.float32)
    if not bool(np.asarray(jax.device_get(spec.is_active))):
        return np.zeros((poses_np.shape[0],), dtype=np.float64)
    receptor_anchor_idx = np.asarray(
        jax.device_get(spec.receptor_anchor_indices), dtype=np.int32
    )
    ligand_anchor_idx = np.asarray(
        jax.device_get(spec.ligand_anchor_indices), dtype=np.int32
    )
    receptor_strengths = np.asarray(
        jax.device_get(spec.receptor_strengths), dtype=np.float32
    )
    ligand_strengths = np.asarray(
        jax.device_get(spec.ligand_strengths), dtype=np.float32
    )
    receptor_anchor_coords = np.asarray(
        jax.device_get(request.protein_coords[receptor_anchor_idx]),
        dtype=np.float32,
    )
    ligand_anchor_coords = poses_np[:, ligand_anchor_idx, :]
    dists = np.linalg.norm(
        receptor_anchor_coords[None, :, None, :] - ligand_anchor_coords[:, None, :, :],
        axis=-1,
    )
    pair_strength = np.clip(
        receptor_strengths[:, None] * ligand_strengths[None, :],
        0.0,
        1.0,
    ).astype(np.float64)
    radial = np.exp(
        -(((dists - float(spec.ideal_distance)) / float(spec.distance_width)) ** 2)
    )
    return np.asarray(
        np.sum(pair_strength[None, :, :] * radial, axis=(1, 2), dtype=np.float64),
        dtype=np.float64,
    )


def _posewise_anchored_indexed_gaussian_upper_bound_current(
    receptor_positions: np.ndarray,
    receptor_strengths: np.ndarray,
    ligand_positions: np.ndarray,
    ligand_strengths: np.ndarray,
    *,
    ideal_distance: float,
    distance_width: float,
) -> np.ndarray:
    if receptor_positions.shape[0] == 0 or ligand_positions.shape[1] == 0:
        return np.zeros((ligand_positions.shape[0],), dtype=np.float64)
    dists = np.linalg.norm(
        receptor_positions[None, :, None, :] - ligand_positions[:, None, :, :],
        axis=-1,
    )
    pair_strength = np.abs(
        receptor_strengths[None, :, None] * ligand_strengths[None, None, :]
    ).astype(np.float64)
    radial = np.exp(-(((dists - ideal_distance) / distance_width) ** 2))
    return np.asarray(
        np.sum(pair_strength * radial, axis=(1, 2), dtype=np.float64),
        dtype=np.float64,
    )


def _posewise_contact_omission_bounds_fast(
    request: "PipelineDockingRequest",
    contact_spec: "CertifiedContactSurrogateSpec",
    *,
    poses_coords: jnp.ndarray,
) -> np.ndarray:
    if not bool(np.asarray(jax.device_get(contact_spec.is_active))):
        return np.zeros((int(poses_coords.shape[0]),), dtype=np.float64)
    receptor_coords = np.asarray(
        jax.device_get(request.protein_coords), dtype=np.float32
    )
    poses_np = np.asarray(jax.device_get(poses_coords), dtype=np.float32)
    receptor_weights = np.asarray(
        jax.device_get(contact_spec.receptor_weights), dtype=np.float32
    )
    ligand_weights = np.asarray(
        jax.device_get(contact_spec.ligand_weights), dtype=np.float32
    )
    dists = np.linalg.norm(
        receptor_coords[None, :, None, :] - poses_np[:, None, :, :],
        axis=-1,
    )
    pair_strength = np.abs(
        receptor_weights[None, :, None] * ligand_weights[None, None, :]
    )
    return np.asarray(
        np.sum(
            pair_strength * np.exp(-((float(contact_spec.beta) * dists) ** 2)),
            axis=(1, 2),
            dtype=np.float64,
        ),
        dtype=np.float64,
    )


def _posewise_metal_omission_bounds_fast(
    request: "PipelineDockingRequest",
    metal_spec: "CertifiedMetalCoordinationSpec",
    *,
    poses_coords: jnp.ndarray,
) -> np.ndarray:
    if not bool(np.asarray(jax.device_get(metal_spec.is_active))):
        return np.zeros((int(poses_coords.shape[0]),), dtype=np.float64)
    receptor_coords = np.asarray(
        jax.device_get(request.protein_coords), dtype=np.float32
    )
    poses_np = np.asarray(jax.device_get(poses_coords), dtype=np.float32)
    receptor_strengths = np.asarray(
        jax.device_get(metal_spec.receptor_strengths), dtype=np.float32
    )
    ligand_strengths = np.asarray(
        jax.device_get(metal_spec.ligand_strengths), dtype=np.float32
    )
    dists = np.linalg.norm(
        receptor_coords[None, :, None, :] - poses_np[:, None, :, :],
        axis=-1,
    )
    pair_strength = np.abs(
        receptor_strengths[None, :, None] * ligand_strengths[None, None, :]
    )
    radial = np.exp(
        -(
            (
                (dists - float(metal_spec.ideal_distance))
                / float(metal_spec.distance_width)
            )
            ** 2
        )
    )
    return np.asarray(
        np.sum(pair_strength * radial, axis=(1, 2), dtype=np.float64),
        dtype=np.float64,
    )


def _posewise_rich_channel_omission_bounds_fast(
    request: "PipelineDockingRequest",
    rich_plan: CertifiedRichChemistryPlan,
    *,
    poses_coords: jnp.ndarray,
) -> np.ndarray:
    poses_np = np.asarray(jax.device_get(poses_coords), dtype=np.float32)
    contact_bound = _posewise_contact_omission_bounds_fast(
        request,
        rich_plan.contact,
        poses_coords=poses_coords,
    )
    hbond_rec_bound = _posewise_directional_hbond_current_upper_bound(
        request,
        rich_plan.hbond_receptor_donor,
        poses_coords=poses_coords,
    )
    hbond_lig_bound = _posewise_directional_hbond_current_upper_bound(
        request,
        rich_plan.hbond_ligand_donor,
        poses_coords=poses_coords,
    )
    hbond_bound = hbond_rec_bound + hbond_lig_bound
    metal_bound = _posewise_metal_omission_bounds_fast(
        request,
        rich_plan.metal_coordination,
        poses_coords=poses_coords,
    )
    cooperative_bound = (
        np.abs(float(rich_plan.cooperative_alpha))
        * (hbond_rec_bound + hbond_lig_bound) ** 2
    )

    extended_bound = np.zeros((poses_np.shape[0],), dtype=np.float64)
    for term in rich_plan.extended_terms.terms:
        if not bool(np.asarray(jax.device_get(term.is_active))):
            continue
        if isinstance(term, PiStackingInteractionTerm):
            ligand_positions = np.asarray(
                jax.device_get(indexed_site_positions(poses_coords, term.ligand_rings)),
                dtype=np.float32,
            )
            extended_bound += _posewise_anchored_indexed_gaussian_upper_bound_current(
                np.asarray(
                    jax.device_get(term.receptor_rings.positions), dtype=np.float32
                ),
                np.asarray(
                    jax.device_get(term.receptor_rings.strengths), dtype=np.float32
                ),
                ligand_positions,
                np.asarray(
                    jax.device_get(term.ligand_rings.strengths), dtype=np.float32
                ),
                ideal_distance=float(term.ideal_distance),
                distance_width=float(term.distance_width),
            )
        elif isinstance(term, PiCationInteractionTerm):
            ligand_cation_positions = np.asarray(
                jax.device_get(
                    indexed_site_positions(poses_coords, term.ligand_cations)
                ),
                dtype=np.float32,
            )
            ligand_ring_positions = np.asarray(
                jax.device_get(indexed_site_positions(poses_coords, term.ligand_rings)),
                dtype=np.float32,
            )
            extended_bound += _posewise_anchored_indexed_gaussian_upper_bound_current(
                np.asarray(
                    jax.device_get(term.receptor_rings.positions), dtype=np.float32
                ),
                np.asarray(
                    jax.device_get(term.receptor_rings.strengths), dtype=np.float32
                ),
                ligand_cation_positions,
                np.asarray(
                    jax.device_get(term.ligand_cations.strengths), dtype=np.float32
                ),
                ideal_distance=float(term.ideal_distance),
                distance_width=float(term.distance_width),
            )
            extended_bound += _posewise_anchored_indexed_gaussian_upper_bound_current(
                np.asarray(
                    jax.device_get(term.receptor_cations.positions), dtype=np.float32
                ),
                np.asarray(
                    jax.device_get(term.receptor_cations.strengths), dtype=np.float32
                ),
                ligand_ring_positions,
                np.asarray(
                    jax.device_get(term.ligand_rings.strengths), dtype=np.float32
                ),
                ideal_distance=float(term.ideal_distance),
                distance_width=float(term.distance_width),
            )
        elif isinstance(term, HalogenBondInteractionTerm):
            ligand_donor_positions = np.asarray(
                jax.device_get(
                    indexed_site_positions(poses_coords, term.ligand_donors)
                ),
                dtype=np.float32,
            )
            ligand_acceptor_positions = np.asarray(
                jax.device_get(
                    indexed_site_positions(poses_coords, term.ligand_acceptors)
                ),
                dtype=np.float32,
            )
            extended_bound += _posewise_anchored_indexed_gaussian_upper_bound_current(
                np.asarray(
                    jax.device_get(term.receptor_acceptors.positions), dtype=np.float32
                ),
                np.asarray(
                    jax.device_get(term.receptor_acceptors.strengths), dtype=np.float32
                ),
                ligand_donor_positions,
                np.asarray(
                    jax.device_get(term.ligand_donors.strengths), dtype=np.float32
                ),
                ideal_distance=float(term.ideal_distance),
                distance_width=float(term.distance_width),
            )
            extended_bound += _posewise_anchored_indexed_gaussian_upper_bound_current(
                np.asarray(
                    jax.device_get(term.receptor_donors.positions), dtype=np.float32
                ),
                np.asarray(
                    jax.device_get(term.receptor_donors.strengths), dtype=np.float32
                ),
                ligand_acceptor_positions,
                np.asarray(
                    jax.device_get(term.ligand_acceptors.strengths), dtype=np.float32
                ),
                ideal_distance=float(term.ideal_distance),
                distance_width=float(term.distance_width),
            )
        elif isinstance(term, WaterMediatedHBondInteractionTerm):
            ligand_polar_positions = np.asarray(
                jax.device_get(
                    indexed_site_positions(poses_coords, term.ligand_polar_sites)
                ),
                dtype=np.float32,
            )
            extended_bound += _posewise_anchored_indexed_gaussian_upper_bound_current(
                np.asarray(
                    jax.device_get(term.receptor_waters.positions), dtype=np.float32
                ),
                np.asarray(
                    jax.device_get(term.receptor_waters.strengths), dtype=np.float32
                ),
                ligand_polar_positions,
                np.asarray(
                    jax.device_get(term.ligand_polar_sites.strengths), dtype=np.float32
                ),
                ideal_distance=float(term.ideal_distance),
                distance_width=float(term.distance_width),
            )
        else:
            bound_fn = getattr(term, "max_total_score_bound", None)
            if bound_fn is None:
                return np.full((poses_np.shape[0],), np.inf, dtype=np.float64)
            extended_bound += float(bound_fn())
    return (
        contact_bound + hbond_bound + metal_bound + cooperative_bound + extended_bound
    )


def _posewise_rigid_local_improvement_bounds(
    request: "PipelineDockingRequest",
    scoring_context: CertifiedScoringContext,
    *,
    poses_coords: jnp.ndarray,
    base_translation_step: float,
    base_rotation_step_rad: float,
    ligand_radius: float,
    n_rounds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from dq_dock_engine.docking.scoring import _EPSILON_KCAL_MOL

    global_improvement_cap = _rigid_local_improvement_bound(
        request,
        scoring_context,
        base_translation_step=base_translation_step,
        base_rotation_step_rad=base_rotation_step_rad,
        ligand_radius=ligand_radius,
        n_rounds=n_rounds,
    )
    posewise_bounds = np.empty((int(poses_coords.shape[0]),), dtype=np.float32)
    clearance_safe_mask = np.ones((int(poses_coords.shape[0]),), dtype=bool)
    posewise_clearance = np.empty((int(poses_coords.shape[0]),), dtype=np.float32)
    receptor_coords = np.asarray(
        jax.device_get(request.protein_coords), dtype=np.float32
    )
    receptor_radii = np.asarray(
        jax.device_get(request.receptor_radii), dtype=np.float32
    )
    ligand_radii = np.asarray(
        jax.device_get(request.ligand_ctx.base_radii), dtype=np.float32
    )
    max_ligand_radius = float(np.max(ligand_radii))
    poses_np = np.asarray(jax.device_get(poses_coords), dtype=np.float32)
    cutoff = float(compute_certified_cutoff(request.target_error))
    epsilon_pair = float(_EPSILON_KCAL_MOL / 4.0)

    receptor_charges = ligand_charges = None
    electro_cutoff = electro_alpha = electro_dielectric = None
    ligand_charge_sum_abs = 0.0
    if scoring_context.electrostatics is not None:
        receptor_charges = np.asarray(
            jax.device_get(scoring_context.electrostatics.receptor_charges),
            dtype=np.float32,
        )
        ligand_charges = np.asarray(
            jax.device_get(scoring_context.electrostatics.ligand_charges),
            dtype=np.float32,
        )
        electro_cutoff = float(scoring_context.electrostatics.cutoff)
        electro_alpha = float(scoring_context.electrostatics.alpha)
        electro_dielectric = float(scoring_context.electrostatics.dielectric)
        ligand_charge_sum_abs = float(np.sum(np.abs(ligand_charges)))

    for idx, pose_coords_np in enumerate(poses_np):
        pose_center = np.mean(pose_coords_np, axis=0)
        atom_arm_lengths = np.linalg.norm(
            pose_coords_np - pose_center[None, :], axis=-1
        )
        atom_path_displacement = _dyadic_action_path_displacement_bound_per_atom(
            base_translation_step,
            base_rotation_step_rad,
            atom_arm_lengths,
            n_rounds,
        )
        path_displacement = float(np.max(atom_path_displacement))
        ligand_extent = float(np.max(atom_arm_lengths))
        center_distances = np.linalg.norm(
            receptor_coords - pose_center[None, :], axis=-1
        )
        safe_cutoff_center = np.maximum(cutoff, receptor_radii + max_ligand_radius)
        support_radius = ligand_extent + path_displacement + safe_cutoff_center
        retained_mask = center_distances <= support_radius
        if not bool(np.any(retained_mask)):
            retained_mask[int(np.argmin(center_distances))] = True

        receptor_coords_local = receptor_coords[retained_mask]
        receptor_radii_local = receptor_radii[retained_mask]
        receptor_charges_local = (
            None if receptor_charges is None else receptor_charges[retained_mask]
        )

        local_lj_lipschitz = 0.0
        local_electro_lipschitz = 0.0
        local_improvement_cap = global_improvement_cap

        dists = np.linalg.norm(
            receptor_coords_local[:, None, :] - pose_coords_np[None, :, :],
            axis=-1,
        )
        sigma = receptor_radii_local[:, None] + ligand_radii[None, :]
        disp = np.broadcast_to(atom_path_displacement[None, :], dists.shape)
        cutoff_safe = np.maximum(cutoff, sigma)
        in_range_possible = (dists - disp) <= cutoff_safe
        if bool(np.any(in_range_possible)):
            min_sigma_interacting = float(np.min(sigma[in_range_possible]))
            local_lj_lipschitz = compute_raw_lj_lipschitz(
                _EPSILON_KCAL_MOL,
                min_sigma_interacting,
            )
            if (
                receptor_charges_local is not None
                and electro_alpha is not None
                and electro_dielectric is not None
                and ligand_charge_sum_abs > 0.0
            ):
                receptor_active_mask = np.any(in_range_possible, axis=1)
                receptor_charge_sum_abs = float(
                    np.sum(np.abs(receptor_charges_local[receptor_active_mask]))
                )
                if receptor_charge_sum_abs > 0.0:
                    radial_bound = (2.0 * electro_alpha) / (
                        math.sqrt(math.pi) * min_sigma_interacting
                    ) + 1.0 / (min_sigma_interacting**2)
                    local_electro_lipschitz = (
                        (receptor_charge_sum_abs * ligand_charge_sum_abs)
                        / electro_dielectric
                    ) * radial_bound
        local_improvement_cap = min(
            global_improvement_cap,
            (local_lj_lipschitz + local_electro_lipschitz) * path_displacement,
        )
        posewise_clearance[idx] = np.float32(np.min(dists - sigma))

        omission_bounds = _per_receptor_atom_rigid_omission_bounds(
            epsilon_pair=epsilon_pair,
            dists=dists,
            sigma=sigma,
            atom_displacements=atom_path_displacement,
            cutoff=cutoff,
            receptor_charges_local=receptor_charges_local,
            ligand_charges=ligand_charges,
            electro_cutoff=electro_cutoff,
            electro_alpha=electro_alpha,
            electro_dielectric=electro_dielectric,
        )
        local_bound = float(np.sum(omission_bounds))

        tail_interval_mask = np.logical_and(
            dists - disp >= sigma,
            dists - disp > _LJ_SINGULARITY_FLOOR,
        )
        if bool(np.all(np.logical_or(~in_range_possible, tail_interval_mask))):
            tail_lj_cap = float(
                np.sum(
                    np.where(
                        in_range_possible,
                        (
                            24.0
                            * epsilon_pair
                            / np.maximum(dists - disp, _LJ_SINGULARITY_FLOOR)
                        )
                        * disp,
                        0.0,
                    ),
                    dtype=np.float64,
                )
            )
            local_improvement_cap = min(
                local_improvement_cap,
                tail_lj_cap + local_electro_lipschitz * path_displacement,
            )

        if omission_bounds.shape[0] > 1:
            local_subset, omitted_bound = _select_receptor_subset_by_omission_budget(
                omission_bounds,
                omission_budget=request.target_error,
            )
            if int(local_subset.shape[0]) < omission_bounds.shape[0]:
                subset_idx = np.asarray(local_subset, dtype=np.int32)
                subset_radii = receptor_radii_local[subset_idx]
                subset_sigma = subset_radii[:, None] + ligand_radii[None, :]
                subset_in_range = in_range_possible[subset_idx]
                subset_lj_lipschitz = 0.0
                if bool(np.any(subset_in_range)):
                    subset_lj_lipschitz = compute_raw_lj_lipschitz(
                        _EPSILON_KCAL_MOL,
                        float(np.min(subset_sigma[subset_in_range])),
                    )
                subset_electro_lipschitz = 0.0
                if (
                    receptor_charges_local is not None
                    and electro_alpha is not None
                    and electro_dielectric is not None
                    and ligand_charge_sum_abs > 0.0
                    and bool(np.any(subset_in_range))
                ):
                    subset_charge_sum_abs = float(
                        np.sum(np.abs(receptor_charges_local[subset_idx]))
                    )
                    if subset_charge_sum_abs > 0.0:
                        r_min_subset = float(np.min(subset_sigma[subset_in_range]))
                        radial_bound_subset = (2.0 * electro_alpha) / (
                            math.sqrt(math.pi) * r_min_subset
                        ) + 1.0 / (r_min_subset**2)
                        subset_electro_lipschitz = (
                            (subset_charge_sum_abs * ligand_charge_sum_abs)
                            / electro_dielectric
                        ) * radial_bound_subset
                local_improvement_cap = min(
                    local_improvement_cap,
                    (subset_lj_lipschitz + subset_electro_lipschitz) * path_displacement
                    + float(omitted_bound),
                )
        posewise_bounds[idx] = np.float32(min(local_bound, local_improvement_cap))
    return posewise_bounds, clearance_safe_mask, posewise_clearance


@dataclass(frozen=True)
class CertifiedRigidLocalRefinementPlan:
    translation_cell_width: float
    base_translation_step: float
    ligand_radius: float
    base_rotation_step_rad: float
    n_search_rounds: int
    local_improvement_bound: float


def _rigid_seed_translation_support_size(
    request: "PipelineDockingRequest",
) -> jnp.ndarray:
    rigid_seed_family_plan = request.rigid_seed_family_plan
    if rigid_seed_family_plan is None:
        return request.box.size
    if rigid_seed_family_plan.region_kind == CertifiedRigidSeedRegionKind.BOX:
        if rigid_seed_family_plan.box_size is None:
            raise ValueError("box rigid seed family plan requires box_size")
        return jnp.asarray(rigid_seed_family_plan.box_size, dtype=jnp.float32)
    if (
        rigid_seed_family_plan.region_kind
        == CertifiedRigidSeedRegionKind.CERTIFIED_BINDING_SITE
    ):
        if rigid_seed_family_plan.binding_site_radius is None:
            raise ValueError(
                "binding-site rigid seed family plan requires binding_site_radius"
            )
        return jnp.full(
            (3,),
            2.0 * float(rigid_seed_family_plan.binding_site_radius),
            dtype=jnp.float32,
        )
    raise ValueError(
        f"unsupported rigid seed region kind {rigid_seed_family_plan.region_kind}"
    )


def _derive_certified_rigid_local_refinement_plan(
    request: "PipelineDockingRequest",
    scoring_context: CertifiedScoringContext,
) -> CertifiedRigidLocalRefinementPlan:
    from dq_dock_engine.docking.formal_actions import (
        least_positive_joint_adequate_dyadic_round,
    )

    if request.rigid_seed_family_plan is None:
        raise ValueError(
            "Certified rigid local refinement requires rigid_seed_family_plan"
        )
    rigid_seed_family_plan = request.rigid_seed_family_plan
    translation_support_size = _rigid_seed_translation_support_size(request)
    translation_cell_width = float(jnp.min(translation_support_size)) / float(
        rigid_seed_family_plan.lattice_resolution
    )
    base_translation_step = translation_cell_width / 2.0
    ligand_radius = float(
        jnp.max(jnp.linalg.norm(request.ligand_ctx.base_coords, axis=-1))
    )
    rotation_displacement_step = ligand_radius * float(jnp.pi / 2.0)
    base_rotation_step_rad = _derive_local_rotation_step_rad(
        base_translation_step, ligand_radius
    )
    if request.config is None:
        raise ValueError("Certified rigid local refinement requires docking config")
    n_search_rounds = least_positive_joint_adequate_dyadic_round(
        base_translation_step,
        rotation_displacement_step,
        request.config.target_rmsd,
    )
    local_improvement_bound = _rigid_local_improvement_bound(
        request,
        scoring_context,
        base_translation_step=base_translation_step,
        base_rotation_step_rad=base_rotation_step_rad,
        ligand_radius=ligand_radius,
        n_rounds=n_search_rounds,
    )
    return CertifiedRigidLocalRefinementPlan(
        translation_cell_width=translation_cell_width,
        base_translation_step=base_translation_step,
        ligand_radius=ligand_radius,
        base_rotation_step_rad=base_rotation_step_rad,
        n_search_rounds=n_search_rounds,
        local_improvement_bound=local_improvement_bound,
    )


def _derive_exact_score_lipschitz_constant(
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    exact_chemistry_mode: ExactChemistryMode,
    softening_policy: SofteningPolicy,
) -> float:
    from dq_dock_engine.docking.scoring import _EPSILON_KCAL_MOL

    min_sigma = _min_pairwise_sigma_from_radii(receptor_radii, ligand_radii)
    if (
        exact_chemistry_mode == ExactChemistryMode.EXTENDED_RICH
        and softening_policy != SofteningPolicy.NONE
    ):
        return 24.0 * _EPSILON_KCAL_MOL / min_sigma
    return compute_raw_lj_lipschitz(_EPSILON_KCAL_MOL, min_sigma)


def _derive_target_error_from_rmsd(
    target_rmsd: float,
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    exact_chemistry_mode: ExactChemistryMode,
    softening_policy: SofteningPolicy,
) -> float:
    """Derive a certified score-resolution budget from target RMSD.

    Lean: CSC48 (`energyBudget_of_rmsdTarget`).
    Using a certified physical Lipschitz constant `L`, an RMSD tolerance `eps`
    induces the energy resolution budget `L * eps`.
    """
    if target_rmsd <= 0:
        raise ValueError(f"target_rmsd must be positive, got {target_rmsd}")
    return (
        _derive_exact_score_lipschitz_constant(
            receptor_radii,
            ligand_radii,
            exact_chemistry_mode,
            softening_policy,
        )
        * target_rmsd
    )


def _resolve_softening_radius(
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    *,
    policy: SofteningPolicy,
    mode: DockingMode,
    heuristic_ratio: float,
) -> float | None:
    sigma_min = _min_pairwise_sigma_from_radii(receptor_radii, ligand_radii)
    if policy == SofteningPolicy.NONE:
        return None
    if policy in (
        SofteningPolicy.CANONICAL_MAX_SIGMA,
        SofteningPolicy.DERIVED_FROM_ERROR_BUDGET,
    ):
        return sigma_min
    if mode == DockingMode.CERTIFIED:
        raise ValueError("Certified mode does not allow EMPIRICAL_RATIO softening")
    if heuristic_ratio <= 0:
        raise ValueError(
            f"heuristic_softening_ratio must be positive, got {heuristic_ratio}"
        )
    return sigma_min * heuristic_ratio


def _seed_budget_torsion_count(request: "DockingRequestBase") -> int:
    """Count torsions included in the rigid seed-budget volume model.

    The current pipeline samples rigid SE(3) seeds first and performs conformer
    search only after a rigid pose has already been selected/refined. Theorem
    `derive_seed_budget()` therefore applies *per conformer*, not over the full
    joint rigid+torsion volume, so torsional degrees of freedom must not inflate
    the initial rigid seed count.

    Keeping this at zero is what prevents flexible ligands like `1hk4` from
    exploding into billions of rigid seeds before the separate conformer-search
    budget even runs.
    """
    del request
    return 0


def _strictly_smaller_positive(value: float, *, name: str) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value}")
    strict_value = math.nextafter(value, 0.0)
    if not math.isfinite(strict_value) or strict_value <= 0.0:
        raise ValueError(
            f"{name}={value} is too small to derive a strict positive bound"
        )
    return strict_value


def _argmax_subdivision_convergence_max_cells(
    per_bond_lipschitz: tuple[float, ...],
    target_slack: float,
) -> int:
    """Bound the full argmax-subdivision tree size from the CS14 contraction rate.

    CS14 gives a worst-case one-step contraction factor

        slack' <= (1 - 1 / (2m)) * slack

    for `m` active torsions when we always bisect the coordinate with maximal
    weighted-L1 slack contribution. A complete binary tree of depth `d` therefore
    suffices once the root slack `pi * sum(L_i)` contracts below `target_slack`.
    """
    if target_slack <= 0.0:
        raise ValueError(f"target_slack must be positive, got {target_slack}")
    if not per_bond_lipschitz:
        return 1
    n_active = len(per_bond_lipschitz)
    initial_slack = math.pi * sum(max(0.0, float(li)) for li in per_bond_lipschitz)
    if initial_slack <= target_slack:
        return 1
    contraction = 1.0 - 1.0 / (2.0 * n_active)
    if not (0.0 < contraction < 1.0):
        raise ValueError(f"invalid argmax-subdivision contraction factor {contraction}")
    depth = max(
        0,
        int(
            math.ceil(
                math.log(initial_slack / target_slack) / math.log(1.0 / contraction)
            )
        ),
    )
    while initial_slack * (contraction**depth) > target_slack:
        depth += 1
    return max(1, (1 << (depth + 1)) - 1)


def _derive_adaptive_torsion_support_spec(
    per_bond_lipschitz: tuple[float, ...],
    target_delta: float,
    target_rmsd: float,
    max_arm: float,
) -> tuple[int, float, tuple[int, ...]]:
    plan = _derive_conformer_coverage_plan_from_lipschitz(
        score_lipschitz_constant=max(per_bond_lipschitz) if per_bond_lipschitz else 0.0,
        per_bond_lipschitz=per_bond_lipschitz,
        target_delta=target_delta,
        target_rmsd=target_rmsd,
        max_arm=max_arm,
    )
    return plan.max_cells, plan.min_cell_radius, plan.canonical_segments


def _derive_conformer_coverage_plan_from_lipschitz(
    *,
    score_lipschitz_constant: float,
    per_bond_lipschitz: tuple[float, ...],
    target_delta: float,
    target_rmsd: float,
    max_arm: float,
) -> CertifiedConformerCoveragePlan:
    """Derive theorem-backed torsion search budgets from slack and radius certificates.

    The runtime stopping radius must align with the actual B&B cell metric:
    `TorsionCell.radius()` is an L2 radius, so the old `min(half_widths)` derivation
    did not match the Lean `bb_stopping_radius_yields_coverage` / CSC50 bridge.

    We therefore derive the stopping radius from the composed parameter-space
    Lipschitz constant,

        min_cell_radius < target_error / L_param,

    where `L_param = max_i L_i = score_lipschitz * kinematics_lipschitz`.
    When `target_error` itself came from CSC48 (`target_error = score_lipschitz *
    target_rmsd`), this is exactly the CSC50 rule `target_rmsd / K`.

    To turn that stopping radius into a finite worst-case tree budget, we tighten to
    a weighted-L1 slack target that is strict enough to force the L2 cell radius below
    `min_cell_radius`, then take the tighter of:

      - CSC44's support-cardinality bound from canonical adaptive segments, and
      - CS14's geometric argmax-subdivision contraction bound.

    The strict `math.nextafter(..., 0.0)` steps are only there to match the runtime's
    strict `< config.min_cell_radius` stopping predicate without overshooting the
    certified radius.
    """
    if target_delta <= 0.0:
        raise ValueError(f"target_delta must be positive, got {target_delta}")
    if not per_bond_lipschitz:
        return CertifiedConformerCoveragePlan(
            source="zero_torsion_conformer_coverage",
            n_torsions=0,
            score_lipschitz_constant=max(0.0, float(score_lipschitz_constant)),
            per_bond_lipschitz=(),
            canonical_segments=(),
            support_size=1,
            max_cells=1,
            min_cell_radius=float(2.0 * np.pi),
            support_target_delta=target_delta,
            target_delta=target_delta,
            target_rmsd=target_rmsd,
            max_arm=max_arm,
            theorem_handles=conformer_coverage_theorem_handles(),
        )
    if target_rmsd <= 0.0:
        raise ValueError(f"target_rmsd must be positive, got {target_rmsd}")
    if max_arm <= 0.0:
        raise ValueError(f"max_arm must be positive, got {max_arm}")

    lipschitz = tuple(max(0.0, float(li)) for li in per_bond_lipschitz)
    max_lipschitz = max(lipschitz)
    min_lipschitz = min(lipschitz)
    if max_lipschitz <= 0.0:
        raise ValueError(
            "per_bond_lipschitz must contain a positive composed Lipschitz constant"
        )
    if min_lipschitz <= 0.0:
        raise ValueError(
            "all active torsions must have positive per-bond Lipschitz constants"
        )

    raw_min_cell_radius = min(target_delta / max_lipschitz, target_rmsd / max_arm)
    min_cell_radius = _strictly_smaller_positive(
        raw_min_cell_radius,
        name="min_cell_radius",
    )
    support_target_delta = _strictly_smaller_positive(
        min(target_delta, min_lipschitz * min_cell_radius),
        name="support_target_delta",
    )

    n_active = len(lipschitz)
    segments = tuple(
        max(
            1,
            int(math.ceil((2.0 * math.pi * n_active * li) / support_target_delta)),
        )
        for li in lipschitz
    )
    support_size = 1
    for seg in segments:
        support_size *= seg + 1
    # Theorem CSC44: a full binary refinement tree with `support_size` certified
    # leaf cells evaluates at most `2 * support_size - 1` cell centers.
    support_max_cells = max(1, 2 * support_size - 1)
    convergence_max_cells = _argmax_subdivision_convergence_max_cells(
        lipschitz,
        support_target_delta,
    )
    max_cells = min(support_max_cells, convergence_max_cells)

    return CertifiedConformerCoveragePlan(
        source="canonical_adaptive_torsion_support",
        n_torsions=n_active,
        score_lipschitz_constant=max(0.0, float(score_lipschitz_constant)),
        per_bond_lipschitz=lipschitz,
        canonical_segments=segments,
        support_size=support_size,
        max_cells=max_cells,
        min_cell_radius=min_cell_radius,
        support_target_delta=support_target_delta,
        target_delta=target_delta,
        target_rmsd=target_rmsd,
        max_arm=max_arm,
        theorem_handles=conformer_coverage_theorem_handles(),
    )


def _derive_conformer_coverage_plan(
    request: "PipelineDockingRequest",
    *,
    rotatable_bonds: tuple[RotatableBond, ...],
) -> CertifiedConformerCoveragePlan:
    from dq_dock_engine.docking.scoring import _EPSILON_KCAL_MOL

    min_sigma = _min_pairwise_sigma_from_radii(
        request.receptor_radii,
        request.ligand_ctx.base_radii,
    )
    score_lipschitz_constant = _derive_exact_score_lipschitz_constant(
        request.receptor_radii,
        request.ligand_ctx.base_radii,
        request.exact_chemistry_mode,
        request.config.softening_policy
        if request.config is not None
        else SofteningPolicy.NONE,
    )
    r_soft = request.softening_radius
    if r_soft is not None:
        if r_soft <= 0:
            raise ValueError(
                f"resolved softening radius must be positive, got {r_soft}"
            )
        softened = compute_softened_lipschitz_constant(
            _EPSILON_KCAL_MOL,
            min_sigma,
            r_soft,
        )
        if 0 < softened < score_lipschitz_constant:
            score_lipschitz_constant = softened

    if not rotatable_bonds:
        return _derive_conformer_coverage_plan_from_lipschitz(
            score_lipschitz_constant=score_lipschitz_constant,
            per_bond_lipschitz=(),
            target_delta=request.target_error,
            target_rmsd=request.target_rmsd,
            max_arm=float(2.0 * np.pi),
        )

    per_bond_lipschitz = tuple(
        score_lipschitz_constant * bond.max_arm_length for bond in rotatable_bonds
    )
    max_arm = max(bond.max_arm_length for bond in rotatable_bonds)
    return _derive_conformer_coverage_plan_from_lipschitz(
        score_lipschitz_constant=score_lipschitz_constant,
        per_bond_lipschitz=per_bond_lipschitz,
        target_delta=request.target_error,
        target_rmsd=request.target_rmsd,
        max_arm=max_arm,
    )


def _active_rotatable_bonds_for_pose(
    world_coords: jnp.ndarray,
    receptor_coords: jnp.ndarray,
    rotatable_bonds: tuple[RotatableBond, ...],
    scoring_cutoff: float = 6.0,
) -> tuple[tuple[RotatableBond, ...], np.ndarray]:
    if not rotatable_bonds:
        return (), np.zeros((0,), dtype=bool)
    pose_np = np.asarray(jax.device_get(world_coords), dtype=np.float32)
    receptor_np = np.asarray(jax.device_get(receptor_coords), dtype=np.float32)
    active_mask = np.zeros((len(rotatable_bonds),), dtype=bool)
    for bond_idx, bond in enumerate(rotatable_bonds):
        if not bond.rotating_atom_indices:
            continue
        rotating_coords = pose_np[
            np.asarray(bond.rotating_atom_indices, dtype=np.int32)
        ]
        dists = np.linalg.norm(
            rotating_coords[:, None, :] - receptor_np[None, :, :],
            axis=-1,
        )
        arm_lengths = (
            np.asarray(bond.rotating_atom_arm_lengths, dtype=np.float32)
            if bond.rotating_atom_arm_lengths
            else np.full(
                (rotating_coords.shape[0],),
                float(bond.max_arm_length),
                dtype=np.float32,
            )
        )
        max_rotation_displacement = 2.0 * arm_lengths
        active_mask[bond_idx] = bool(
            np.any(np.min(dists, axis=1) <= scoring_cutoff + max_rotation_displacement)
        )
    active_bonds = tuple(
        bond for bond, is_active in zip(rotatable_bonds, active_mask) if is_active
    )
    return active_bonds, active_mask


def _posewise_active_torsion_counts(
    poses_coords: jnp.ndarray,
    receptor_coords: jnp.ndarray,
    rotatable_bonds: tuple[RotatableBond, ...],
    *,
    scoring_cutoff: float,
) -> np.ndarray:
    if not rotatable_bonds:
        return np.zeros((int(poses_coords.shape[0]),), dtype=np.int32)
    poses_np = np.asarray(jax.device_get(poses_coords), dtype=np.float32)
    receptor_np = np.asarray(jax.device_get(receptor_coords), dtype=np.float32)
    counts = np.empty((int(poses_coords.shape[0]),), dtype=np.int32)
    for pose_idx, pose_np in enumerate(poses_np):
        active_count = 0
        for bond in rotatable_bonds:
            if not bond.rotating_atom_indices:
                continue
            rotating_coords = pose_np[
                np.asarray(bond.rotating_atom_indices, dtype=np.int32)
            ]
            dists = np.linalg.norm(
                rotating_coords[:, None, :] - receptor_np[None, :, :],
                axis=-1,
            )
            arm_lengths = (
                np.asarray(bond.rotating_atom_arm_lengths, dtype=np.float32)
                if bond.rotating_atom_arm_lengths
                else np.full(
                    (rotating_coords.shape[0],),
                    float(bond.max_arm_length),
                    dtype=np.float32,
                )
            )
            max_rotation_displacement = 2.0 * arm_lengths
            if np.any(
                np.min(dists, axis=1) <= scoring_cutoff + max_rotation_displacement
            ):
                active_count += 1
        counts[pose_idx] = active_count
    return counts


def _quaternion_rotation_matrix_np(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float32)
    norm = np.linalg.norm(q)
    if norm <= 1e-8:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _build_cell_local_activity_mask_fn(
    *,
    receptor_coords: jnp.ndarray,
    quaternion: jnp.ndarray,
    translation: jnp.ndarray,
    rotatable_bonds: tuple[RotatableBond, ...],
    scoring_cutoff: float,
) -> Callable[[TorsionCell, np.ndarray], np.ndarray] | None:
    if not rotatable_bonds:
        return None
    receptor_np = np.asarray(jax.device_get(receptor_coords), dtype=np.float32)
    quat_np = np.asarray(jax.device_get(quaternion), dtype=np.float32)
    translation_np = np.asarray(jax.device_get(translation), dtype=np.float32)
    rotation = _quaternion_rotation_matrix_np(quat_np)
    receptor_local = (receptor_np - translation_np[None, :]) @ rotation
    n_atoms = 0
    for bond in rotatable_bonds:
        if bond.rotating_atom_indices:
            n_atoms = max(n_atoms, max(bond.rotating_atom_indices) + 1)

    def _mask(cell: TorsionCell, center_coords_local: np.ndarray) -> np.ndarray:
        half_widths = np.asarray(jax.device_get(cell.half_widths()), dtype=np.float32)
        atom_displacement = np.zeros((n_atoms,), dtype=np.float32)
        for half_width, bond in zip(half_widths.tolist(), rotatable_bonds):
            half_angle = min(math.pi, abs(float(half_width)))
            arm_lengths = (
                np.asarray(bond.rotating_atom_arm_lengths, dtype=np.float32)
                if bond.rotating_atom_arm_lengths
                else np.full(
                    (len(bond.rotating_atom_indices),),
                    float(bond.max_arm_length),
                    dtype=np.float32,
                )
            )
            displacement_bound = 2.0 * arm_lengths * math.sin(half_angle / 2.0)
            if displacement_bound.size == 0 or float(np.max(displacement_bound)) <= 0.0:
                continue
            atom_displacement[list(bond.rotating_atom_indices)] += displacement_bound

        active_mask = np.zeros((len(rotatable_bonds),), dtype=bool)
        for bond_idx, bond in enumerate(rotatable_bonds):
            if not bond.rotating_atom_indices:
                continue
            rotating_idx = np.asarray(bond.rotating_atom_indices, dtype=np.int32)
            dists = np.linalg.norm(
                center_coords_local[rotating_idx, None, :] - receptor_local[None, :, :],
                axis=-1,
            )
            min_dist_per_atom = np.min(dists, axis=1)
            safe_margin = min_dist_per_atom - atom_displacement[rotating_idx]
            active_mask[bond_idx] = not np.all(safe_margin > scoring_cutoff)
        return active_mask

    return _mask


def _select_receptor_subset_for_conformer_family(
    *,
    receptor_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    ligand_world_coords: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    rotatable_bonds: tuple[RotatableBond, ...],
    target_error: float,
) -> jnp.ndarray:
    receptor_np = np.asarray(jax.device_get(receptor_coords), dtype=np.float32)
    receptor_radii_np = np.asarray(jax.device_get(receptor_radii), dtype=np.float32)
    ligand_np = np.asarray(jax.device_get(ligand_world_coords), dtype=np.float32)
    ligand_radii_np = np.asarray(jax.device_get(ligand_radii), dtype=np.float32)
    cutoff = compute_certified_cutoff(target_error)

    atom_displacement = np.zeros((ligand_np.shape[0],), dtype=np.float32)
    for bond in rotatable_bonds:
        arm_lengths = (
            np.asarray(bond.rotating_atom_arm_lengths, dtype=np.float32)
            if bond.rotating_atom_arm_lengths
            else np.full(
                (len(bond.rotating_atom_indices),),
                float(bond.max_arm_length),
                dtype=np.float32,
            )
        )
        atom_displacement[list(bond.rotating_atom_indices)] += 2.0 * arm_lengths

    pairwise_dist = np.linalg.norm(
        ligand_np[:, None, :] - receptor_np[None, :, :],
        axis=-1,
    )
    safe_cutoff = np.maximum(
        cutoff,
        ligand_radii_np[:, None] + receptor_radii_np[None, :],
    )
    possible_contact = pairwise_dist - atom_displacement[:, None] <= safe_cutoff
    keep_mask = np.any(possible_contact, axis=0)
    if not bool(np.any(keep_mask)):
        closest_index = int(np.argmin(np.min(pairwise_dist, axis=0)))
        return jnp.array([closest_index], dtype=jnp.int32)
    return jnp.array(np.flatnonzero(keep_mask), dtype=jnp.int32)


def _per_receptor_atom_conformer_omission_bounds(
    *,
    ligand_world_coords: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    receptor_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    rotatable_bonds: tuple[RotatableBond, ...],
    ligand_charges: jnp.ndarray | None,
    receptor_charges: jnp.ndarray | None,
    electro_alpha: float | None,
    electro_dielectric: float | None,
) -> np.ndarray:
    from dq_dock_engine.docking.scoring import _EPSILON_KCAL_MOL

    ligand_np = np.asarray(jax.device_get(ligand_world_coords), dtype=np.float32)
    ligand_radii_np = np.asarray(jax.device_get(ligand_radii), dtype=np.float32)
    receptor_np = np.asarray(jax.device_get(receptor_coords), dtype=np.float32)
    receptor_radii_np = np.asarray(jax.device_get(receptor_radii), dtype=np.float32)

    atom_displacement = np.zeros((ligand_np.shape[0],), dtype=np.float32)
    moving_mask = np.zeros((ligand_np.shape[0],), dtype=bool)
    for bond in rotatable_bonds:
        atom_idx = np.asarray(bond.rotating_atom_indices, dtype=np.int32)
        if atom_idx.size == 0:
            continue
        arm_lengths = (
            np.asarray(bond.rotating_atom_arm_lengths, dtype=np.float32)
            if bond.rotating_atom_arm_lengths
            else np.full((atom_idx.size,), float(bond.max_arm_length), dtype=np.float32)
        )
        atom_displacement[atom_idx] += 2.0 * arm_lengths
        moving_mask[atom_idx] = True

    moving_indices = np.flatnonzero(moving_mask)
    bounds = np.zeros((receptor_np.shape[0],), dtype=np.float32)
    if moving_indices.size == 0:
        return bounds

    epsilon_pair = float(_EPSILON_KCAL_MOL / 4.0)
    if ligand_charges is not None:
        ligand_charges_np = np.asarray(jax.device_get(ligand_charges), dtype=np.float32)
    else:
        ligand_charges_np = None
    if receptor_charges is not None:
        receptor_charges_np = np.asarray(
            jax.device_get(receptor_charges), dtype=np.float32
        )
    else:
        receptor_charges_np = None

    for atom_idx in moving_indices.tolist():
        atom_coord = ligand_np[atom_idx]
        dists = np.linalg.norm(receptor_np - atom_coord[None, :], axis=-1)
        sigma = ligand_radii_np[atom_idx] + receptor_radii_np
        lower = dists - atom_displacement[atom_idx]
        rm = sigma * (2.0 ** (1.0 / 6.0))
        sr = sigma / np.maximum(lower, sigma)
        tail_abs = np.abs(4.0 * epsilon_pair * (sr**12 - sr**6))
        lj_abs = np.where(
            lower <= sigma,
            np.float32(np.inf),
            np.where(lower < rm, epsilon_pair, tail_abs),
        )
        bounds += lj_abs.astype(np.float32)

        if (
            ligand_charges_np is not None
            and receptor_charges_np is not None
            and electro_alpha is not None
            and electro_dielectric is not None
        ):
            charge_abs = (
                np.abs(ligand_charges_np[atom_idx] * receptor_charges_np)
                / electro_dielectric
            )
            electro_abs = (
                charge_abs
                * np.exp(-((electro_alpha * np.maximum(lower, 1.0e-6)) ** 2))
                / np.maximum(lower, 1.0e-6)
            )
            electro_abs = np.where(lower <= 0.0, np.float32(np.inf), electro_abs)
            bounds += electro_abs.astype(np.float32)

    return bounds


def _select_receptor_subset_by_omission_budget(
    omission_bounds: np.ndarray,
    *,
    omission_budget: float,
) -> tuple[jnp.ndarray, float]:
    if omission_bounds.ndim != 1:
        raise ValueError("omission_bounds must be 1D")
    inf_mask = np.isinf(omission_bounds)
    keep_mask = inf_mask.copy()
    finite_indices = np.flatnonzero(~inf_mask)
    finite_bounds = omission_bounds[finite_indices]
    total_finite = float(np.sum(finite_bounds))
    if total_finite > omission_budget:
        order = finite_indices[np.argsort(finite_bounds)[::-1]]
        running_omitted = total_finite
        for idx in order.tolist():
            if running_omitted <= omission_budget:
                break
            keep_mask[idx] = True
            running_omitted -= float(omission_bounds[idx])
    omitted_total = float(np.sum(omission_bounds[~keep_mask & ~inf_mask]))
    if not bool(np.any(keep_mask)):
        keep_mask[int(np.argmax(omission_bounds))] = True
        omitted_total = float(np.sum(omission_bounds[~keep_mask & ~inf_mask]))
    return jnp.array(np.flatnonzero(keep_mask), dtype=jnp.int32), omitted_total


def _restrict_strain_params(
    strain_params: TorsionStrainParams | None,
    active_mask: np.ndarray,
) -> TorsionStrainParams | None:
    def _empty_params() -> TorsionStrainParams:
        return TorsionStrainParams(
            barrier_heights=jnp.zeros((0,), dtype=jnp.float32),
            multiplicities=jnp.zeros((0,), dtype=jnp.float32),
            phases=jnp.zeros((0,), dtype=jnp.float32),
        )

    if strain_params is None or active_mask.size == 0:
        return None if strain_params is None else _empty_params()
    if not np.any(active_mask):
        return _empty_params()
    idx = np.flatnonzero(active_mask)
    return TorsionStrainParams(
        barrier_heights=jnp.asarray(strain_params.barrier_heights)[idx],
        multiplicities=jnp.asarray(strain_params.multiplicities)[idx],
        phases=jnp.asarray(strain_params.phases)[idx],
    )


def _probe_seed_budget_certificate(
    request: "PipelineDockingRequest",
    route: "PipelineRoute",
) -> tuple["PipelineDockingRequest", RefinementCertificate | None]:
    from dq_dock_engine.docking.placement import apply_poses

    probe_request = cast(
        "PipelineDockingRequest",
        request.with_updates(
            n_poses_override=SEED_BUDGET_PROBE_POSES,
            rigid_seed_family_plan=_derive_request_rigid_seed_family_plan(
                request,
                n_poses=SEED_BUDGET_PROBE_POSES,
                enforce_target_translation_cover_radius=False,
            ),
        ),
    )
    probe_batch = route.generate_pose_batch(probe_request)
    probe_request = probe_batch.request
    probe_coords = apply_poses(probe_request.ligand_ctx, probe_batch.pose_vecs)
    probe_scores = route.score_pose_batch(
        probe_request, probe_coords, probe_batch.pose_vecs
    ).final_scores
    probe_scores_np = np.asarray(jax.device_get(probe_scores), dtype=np.float64)
    ranked = np.argsort(probe_scores_np)

    successful_probe_candidates: list[tuple[int, RefinementCertificate | None]] = []
    n_torsions = _seed_budget_torsion_count(probe_request)
    ligand_radius = _ligand_radius(probe_request.ligand_ctx)

    probe_candidate_indices = ranked[: min(SEED_BUDGET_PROBE_TOP_K, ranked.shape[0])]
    attempted_probe_indices = probe_candidate_indices
    if attempted_probe_indices.size > 0:
        probe_failure_reasons: list[str | None] = []
        _opt_t, _opt_q, probe_certs = _certified_refinement(
            request=probe_request,
            initial_translations=probe_batch.pose_vecs.translation[
                jnp.asarray(attempted_probe_indices)
            ],
            initial_quaternions=probe_batch.pose_vecs.quaternion[
                jnp.asarray(attempted_probe_indices)
            ],
            mode_override=RefinementCertificationMode.OBSERVED,
            failure_reasons_out=probe_failure_reasons,
        )
        del _opt_t, _opt_q
        for probe_rank, cert in enumerate(probe_certs):
            if cert is None:
                continue
            successful_probe_candidates.append((probe_rank, cert))
        if not successful_probe_candidates:
            print(
                "[SEED BUDGET PROBE] Top-ranked observed probe failures: "
                + ", ".join(
                    f"rank{rank}={reason}"
                    for rank, reason in enumerate(probe_failure_reasons)
                ),
                flush=True,
            )
    if (
        not successful_probe_candidates
        and ranked.shape[0] > probe_candidate_indices.size
    ):
        print(
            "[SEED BUDGET PROBE] No observed certificate in the top-ranked probe subset; expanding to the full engineering probe cap",
            flush=True,
        )
        attempted_probe_indices = ranked[
            : min(SEED_BUDGET_PROBE_POSES, ranked.shape[0])
        ]
        probe_failure_reasons = []
        _opt_t, _opt_q, probe_certs = _certified_refinement(
            request=probe_request,
            initial_translations=probe_batch.pose_vecs.translation[
                jnp.asarray(attempted_probe_indices)
            ],
            initial_quaternions=probe_batch.pose_vecs.quaternion[
                jnp.asarray(attempted_probe_indices)
            ],
            mode_override=RefinementCertificationMode.OBSERVED,
            failure_reasons_out=probe_failure_reasons,
        )
        del _opt_t, _opt_q
        for probe_rank, cert in enumerate(probe_certs):
            if cert is None:
                continue
            successful_probe_candidates.append((probe_rank, cert))
        if not successful_probe_candidates:
            print(
                "[SEED BUDGET PROBE] Full-cap observed probe failures: "
                + ", ".join(
                    f"rank{rank}={reason}"
                    for rank, reason in enumerate(probe_failure_reasons)
                ),
                flush=True,
            )

    config = cast(DockingConfig, probe_request.config)
    seed_budget_plan = derive_seed_budget_plan(
        confidence=config.confidence,
        box_size=probe_request.box.size,
        target_rmsd=config.target_rmsd,
        ligand_radius=ligand_radius,
        n_torsions=n_torsions,
        probe_candidates=tuple(successful_probe_candidates),
        rigid_seed_box=probe_request.box,
        certified_binding_site=_resolved_certified_seed_binding_site(probe_request),
    )
    selected_candidate = seed_budget_plan.selected_candidate
    best_cert = None
    if selected_candidate.probe_rank is not None:
        selected_probe_rank = selected_candidate.probe_rank
        for probe_rank, cert in successful_probe_candidates:
            if probe_rank == selected_probe_rank:
                best_cert = cert
                break
    if best_cert is None and successful_probe_candidates:
        best_cert = min(
            successful_probe_candidates,
            key=lambda item: (
                derive_seed_budget(
                    confidence=config.confidence,
                    box_size=probe_request.box.size,
                    target_rmsd=config.target_rmsd,
                    ligand_radius=ligand_radius,
                    n_torsions=n_torsions,
                    probe_certificate=item[1],
                    translation_search_volume=_translation_search_volume_for_seed_family(
                        box_size=probe_request.box.size,
                        rigid_seed_box=probe_request.box,
                        certified_binding_site=_resolved_certified_seed_binding_site(
                            probe_request
                        ),
                    ),
                ),
                item[0],
            ),
        )[1]
    return cast(
        "PipelineDockingRequest",
        probe_request.with_updates(
            n_poses_override=None,
            seed_budget_plan=seed_budget_plan,
            rigid_seed_family_plan=seed_budget_plan.selected_family_plan,
        ),
    ), best_cert


@dataclass(frozen=True, kw_only=True)
class DockingRequestBase:
    protein_coords: jnp.ndarray
    receptor_radii: jnp.ndarray
    ligand_ctx: LigandContext
    box: DockingBox
    n_poses_override: int | None = None
    seed_budget_plan: CertifiedSeedBudgetPlan | None = None
    rigid_seed_family_plan: CertifiedRigidSeedFamilyPlan | None = None
    conformer_coverage_plan: CertifiedConformerCoveragePlan | None = None
    key: jax.Array
    receptor_elements: tuple[str, ...] | None = None
    charge_method: ChargeMethod | None = None
    receptor_file: str | Path | None = None
    ligand_source_path: str | Path | None = None
    precomputed_receptor_charges: jnp.ndarray | None = None
    config: DockingConfig | None = None
    optimize: bool = True
    include_native: bool = False
    debug_native_coords: jnp.ndarray | None = None
    scoring_kwargs: dict[str, object] = field(default_factory=dict)

    def with_updates(self, **changes: object) -> Self:
        return replace(self, **changes)

    @property
    def normalized_key(self) -> jax.Array:
        return _normalize_sampling_key(self.key)

    @property
    def n_poses(self) -> int:
        """Seed budget: explicit if set, otherwise derived from confidence."""
        if self.n_poses_override is not None:
            return self.n_poses_override
        if self.rigid_seed_family_plan is not None:
            if (
                self.seed_budget_plan is not None
                and self.seed_budget_plan.selected_budget
                != self.rigid_seed_family_plan.adequate_pose_count
            ):
                raise ValueError(
                    "rigid_seed_family_plan.adequate_pose_count must agree with the selected seed budget"
                )
            return self.rigid_seed_family_plan.pose_count
        if self.seed_budget_plan is not None:
            return self.seed_budget_plan.selected_budget
        config = self.config
        if config is None:
            raise ValueError(
                "n_poses is None and no DockingConfig provided — "
                "cannot derive seed budget without confidence + target_rmsd"
            )
        ligand_radius = _ligand_radius(self.ligand_ctx)
        return derive_seed_budget_plan(
            confidence=config.confidence,
            box_size=self.box.size,
            target_rmsd=config.target_rmsd,
            ligand_radius=ligand_radius,
            n_torsions=_seed_budget_torsion_count(self),
            rigid_seed_box=self.box,
            certified_binding_site=_resolved_certified_seed_binding_site(self),
        ).selected_budget

    @property
    def resolved_target_error(self) -> float:
        if self.config is None:
            raise ValueError("Certified docking requires an explicit docking config")
        if self.config.target_error <= 0:
            return _derive_target_error_from_rmsd(
                self.config.target_rmsd,
                self.receptor_radii,
                self.ligand_radii,
                self.config.exact_chemistry_mode,
                self.config.softening_policy,
            )
        return self.config.target_error

    @property
    def target_error(self) -> float:
        return self.resolved_target_error

    @property
    def certified_binding_site(self) -> CertifiedBindingSite | None:
        return None if self.config is None else self.config.certified_binding_site

    @property
    def coarse_target_error(self) -> float:
        if self.config is None or self.config.coarse_target_error <= 0:
            return self.target_error
        return self.config.coarse_target_error

    @property
    def adaptive_coarse_target_errors(self) -> tuple[float, ...] | None:
        if self.config is None:
            return (self.coarse_target_error,)
        if self.config.adaptive_coarse_target_errors is None:
            return (self.coarse_target_error,)
        return self.config.adaptive_coarse_target_errors

    @property
    def use_softened_coarse_prefilter(self) -> bool:
        if self.config is None:
            return False
        if self.config.mode == DockingMode.CERTIFIED:
            return self.config.softening_policy != SofteningPolicy.NONE
        return self.config.use_softened_coarse_prefilter or (
            self.config.softening_policy != SofteningPolicy.NONE
        )

    @property
    def softening_radius(self) -> float | None:
        if self.config is None:
            return None
        return _resolve_softening_radius(
            self.receptor_radii,
            self.ligand_radii,
            policy=self.config.softening_policy,
            mode=self.config.mode,
            heuristic_ratio=self.config.heuristic_softening_ratio,
        )

    @property
    def reuse_initial_conformer(self) -> bool:
        return False if self.config is None else self.config.reuse_initial_conformer

    @property
    def target_rmsd(self) -> float:
        if self.config is None:
            raise ValueError(
                "target_rmsd is only defined when a docking config is present"
            )
        return self.config.target_rmsd

    @property
    def receptor_coords(self) -> jnp.ndarray:
        return self.protein_coords

    @property
    def ligand_radii(self) -> jnp.ndarray:
        return self.ligand_ctx.base_radii


@dataclass(frozen=True, kw_only=True)
class BlindDockingRequest(DockingRequestBase):
    pass


@dataclass(frozen=True, kw_only=True)
class CertifiedBlindDockingRequest(BlindDockingRequest):
    pass


@dataclass(frozen=True, kw_only=True)
class RoutedDockingRequest(DockingRequestBase):
    engine: ScoringEngine = ScoringEngine.INTERNAL_LJ


@dataclass(frozen=True, kw_only=True)
class GeometricBlindDockingRequest(RoutedDockingRequest):
    pass


@dataclass(frozen=True, kw_only=True)
class BlindDockingPreparationRequest:
    protein_coords: jnp.ndarray
    receptor_radii: jnp.ndarray
    receptor_elements: tuple[str, ...] | None
    precomputed_receptor_charges: jnp.ndarray | None
    ligand_ctx: LigandContext
    box: DockingBox
    target_error: float


@dataclass(frozen=True, kw_only=True)
class CertifiedPreparationRequest(BlindDockingPreparationRequest):
    explicit_binding_site: CertifiedBindingSite | None = None
    coarse_target_error: float = 0.0
    adaptive_coarse_target_errors: tuple[float, ...] | None = None
    use_softened_coarse_prefilter: bool = False

    @classmethod
    def from_request(
        cls, request: "PipelineDockingRequest | CertifiedBlindDockingRequest"
    ) -> "CertifiedPreparationRequest":
        return derive_request(
            cls,
            request,
            target_error=request.target_error,
            explicit_binding_site=request.certified_binding_site,
            coarse_target_error=request.coarse_target_error,
            adaptive_coarse_target_errors=request.adaptive_coarse_target_errors,
            use_softened_coarse_prefilter=request.use_softened_coarse_prefilter,
        )


@dataclass(frozen=True, kw_only=True)
class GeometricPreparationRequest(BlindDockingPreparationRequest):
    @classmethod
    def from_request(
        cls, request: "PipelineDockingRequest | GeometricBlindDockingRequest"
    ) -> "GeometricPreparationRequest":
        return derive_request(
            cls,
            request,
            target_error=request.target_error,
        )


@dataclass(frozen=True, kw_only=True)
class PipelineDockingRequest(RoutedDockingRequest):
    use_pocket_guided: bool = True
    use_multi_stage: bool = False
    certified_pocket_prep: CertifiedPocketPreparation | None = None

    @property
    def is_certified_mode(self) -> bool:
        return self.config is not None and self.config.mode == DockingMode.CERTIFIED

    @property
    def requires_fixed_size_padding(self) -> bool:
        return self.config is not None and self.config.mode != DockingMode.CERTIFIED

    @property
    def certified_scoring_family(self) -> CertifiedScoringFamily | None:
        return None if self.config is None else self.config.certified_scoring_family

    @property
    def effective_engine(self) -> ScoringEngine:
        if not self.is_certified_mode:
            return self.engine
        if self.certified_scoring_family == CertifiedScoringFamily.LJ:
            return ScoringEngine.CERTIFIED_LJ
        return ScoringEngine.CERTIFIED_LJ_REALSPACE_EWALD

    @property
    def formal_backend(self) -> OptimizerBackend:
        return (
            self.config.optimizer_backend
            if self.config is not None
            else OptimizerBackend.GRADIENT
        )

    @property
    def formal_round_strategy(self) -> FormalRoundStrategy:
        return (
            self.config.formal_round_strategy
            if self.config is not None
            else FormalRoundStrategy.SINGLETON_HYBRID
        )

    @property
    def exact_chemistry_mode(self) -> ExactChemistryMode:
        return (
            self.config.exact_chemistry_mode
            if self.config is not None
            else ExactChemistryMode.NONE
        )

    def with_scoring_override(
        self, **scoring_overrides: object
    ) -> "PipelineDockingRequest":
        return self.with_updates(
            scoring_kwargs=dict(self.scoring_kwargs) | dict(scoring_overrides)
        )

    def with_preparation(
        self,
        prep: BlindDockingPreparation,
        *,
        certified_pocket_prep: CertifiedPocketPreparation | None = None,
    ) -> "PipelineDockingRequest":
        return self.with_updates(
            protein_coords=prep.protein_coords,
            receptor_radii=prep.receptor_radii,
            receptor_elements=prep.receptor_elements,
            precomputed_receptor_charges=prep.precomputed_receptor_charges,
            box=prep.box,
            certified_pocket_prep=certified_pocket_prep,
        )

    def with_fixed_size_padding(self) -> "PipelineDockingRequest":
        if not self.requires_fixed_size_padding:
            return self
        assert self.config is not None
        padded_receptor_charges = self.precomputed_receptor_charges
        if padded_receptor_charges is not None:
            padded_receptor_charges = _pad_to_size(
                padded_receptor_charges,
                self.config.max_receptor_atoms,
                axis=0,
                value=0.0,
            )
        padded_ligand_elements = _pad_tuple_to_size(
            self.ligand_ctx.elements,
            self.config.max_ligand_atoms,
            value="C",
        )
        padded_ligand_charges = None
        if self.ligand_ctx.charges is not None:
            padded_ligand_charges = _pad_to_size(
                self.ligand_ctx.charges,
                self.config.max_ligand_atoms,
                axis=0,
                value=0.0,
            )
        return self.with_updates(
            protein_coords=_pad_to_size(
                self.protein_coords,
                self.config.max_receptor_atoms,
                axis=0,
                value=1e4,
            ),
            receptor_radii=_pad_to_size(
                self.receptor_radii,
                self.config.max_receptor_atoms,
                axis=0,
                value=0.0,
            ),
            receptor_elements=_pad_tuple_to_size(
                self.receptor_elements,
                self.config.max_receptor_atoms,
                value="C",
            ),
            precomputed_receptor_charges=padded_receptor_charges,
            ligand_ctx=LigandContext(
                base_coords=_pad_to_size(
                    self.ligand_ctx.base_coords,
                    self.config.max_ligand_atoms,
                    axis=0,
                    value=1e4,
                ),
                base_radii=_pad_to_size(
                    self.ligand_ctx.base_radii,
                    self.config.max_ligand_atoms,
                    axis=0,
                    value=0.0,
                ),
                elements=()
                if padded_ligand_elements is None
                else padded_ligand_elements,
                charges=padded_ligand_charges,
                center_of_mass=self.ligand_ctx.center_of_mass,
            ),
        )


class RequestMatchBase:
    @classmethod
    def matches_request(cls, request: PipelineDockingRequest) -> bool:
        del request
        return True


class CertifiedModeMatch(RequestMatchBase):
    @classmethod
    def matches_request(cls, request: PipelineDockingRequest) -> bool:
        return super().matches_request(request) and request.is_certified_mode


class NonCertifiedModeMatch(RequestMatchBase):
    @classmethod
    def matches_request(cls, request: PipelineDockingRequest) -> bool:
        return super().matches_request(request) and not request.is_certified_mode


class GuidedSamplingMatch(RequestMatchBase):
    @classmethod
    def matches_request(cls, request: PipelineDockingRequest) -> bool:
        return super().matches_request(request) and request.use_pocket_guided


class BoxSamplingMatch(RequestMatchBase):
    @classmethod
    def matches_request(cls, request: PipelineDockingRequest) -> bool:
        return super().matches_request(request) and not request.use_pocket_guided


class DirectScoringMatch(RequestMatchBase):
    @classmethod
    def matches_request(cls, request: PipelineDockingRequest) -> bool:
        return super().matches_request(request) and not request.use_multi_stage


class MultiStageScoringMatch(RequestMatchBase):
    @classmethod
    def matches_request(cls, request: PipelineDockingRequest) -> bool:
        return super().matches_request(request) and request.use_multi_stage


class CertifiedPreparationMixin:
    def prepare(self) -> "PreparedCertifiedDirectPipelineRequest":
        request = cast(Any, self)
        prep = request.certified_pocket_prep
        if prep is None:
            prep = _prepare_certified_blind_docking(
                CertifiedPreparationRequest.from_request(request)
            )
        return derive_request(
            PreparedCertifiedDirectPipelineRequest,
            request.with_preparation(prep, certified_pocket_prep=prep),
            certified_pocket_prep=prep,
        )


class GeometricPreparationMixin:
    def prepare(self) -> "PreparedGeometricPipelineRequest":
        request = cast(Any, self)
        prep = _prepare_geometric_blind_docking(
            GeometricPreparationRequest.from_request(request)
        )
        return derive_request(
            PreparedGeometricPipelineRequest,
            request.with_preparation(prep),
            geometric_pocket_prep=prep,
            sampling_plan=derive_geometric_sampling_plan(prep),
        )


@dataclass(frozen=True, kw_only=True)
class NominalPipelineDockingRequest(RequestMatchBase, PipelineDockingRequest):
    route_type_name: ClassVar[str | None] = None
    _registered_types: ClassVar[list[type["NominalPipelineDockingRequest"]]] = []

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if cls.__dict__.get("route_type_name") is not None:
            cls._registered_types.append(cls)

    @classmethod
    def from_request(
        cls, request: PipelineDockingRequest
    ) -> "NominalPipelineDockingRequest":
        matches = [
            candidate
            for candidate in cls._registered_types
            if candidate.matches_request(request)
        ]
        if not matches:
            raise ValueError(
                "CERTIFIED mode does not support the heuristic multi-stage scoring pipeline."
            )
        if len(matches) != 1:
            raise TypeError(
                f"Ambiguous nominal pipeline request refinement for {type(request).__name__}: {[candidate.__name__ for candidate in matches]}"
            )
        return derive_request(matches[0], request)

    def create_route(self) -> "PipelineRoute":
        if self.route_type_name is None:
            raise TypeError(
                f"Nominal request type {type(self).__name__} does not declare a route type."
            )
        return cast(type[PipelineRoute], globals()[self.route_type_name])()


@dataclass(frozen=True, kw_only=True)
class DirectPipelineDockingRequest(DirectScoringMatch, NominalPipelineDockingRequest):
    use_multi_stage: bool = False


@dataclass(frozen=True, kw_only=True)
class MultiStagePipelineDockingRequest(
    MultiStageScoringMatch, NominalPipelineDockingRequest
):
    use_multi_stage: bool = True
    charge_method: ChargeMethod | None = None

    def __post_init__(self) -> None:
        _require_present_fields(self, "charge_method")


@dataclass(frozen=True, kw_only=True)
class CertifiedDirectPipelineRequest(
    CertifiedPreparationMixin,
    CertifiedModeMatch,
    DirectPipelineDockingRequest,
):
    route_type_name: ClassVar[str] = "CertifiedPipelineRoute"
    use_pocket_guided: bool = True


@dataclass(frozen=True, kw_only=True)
class GeometricDirectPipelineRequest(
    GeometricPreparationMixin,
    GuidedSamplingMatch,
    NonCertifiedModeMatch,
    DirectPipelineDockingRequest,
):
    route_type_name: ClassVar[str] = "GeometricPocketRoute"
    use_pocket_guided: bool = True


@dataclass(frozen=True, kw_only=True)
class BoxDirectPipelineRequest(
    BoxSamplingMatch,
    NonCertifiedModeMatch,
    DirectPipelineDockingRequest,
):
    route_type_name: ClassVar[str] = "BoxSamplingRoute"
    use_pocket_guided: bool = False


@dataclass(frozen=True, kw_only=True)
class GeometricMultiStagePipelineRequest(
    GeometricPreparationMixin,
    GuidedSamplingMatch,
    NonCertifiedModeMatch,
    MultiStagePipelineDockingRequest,
):
    route_type_name: ClassVar[str] = "GeometricMultiStageRoute"
    use_pocket_guided: bool = True


@dataclass(frozen=True, kw_only=True)
class BoxMultiStagePipelineRequest(
    BoxSamplingMatch,
    NonCertifiedModeMatch,
    MultiStagePipelineDockingRequest,
):
    route_type_name: ClassVar[str] = "BoxMultiStageRoute"
    use_pocket_guided: bool = False


def _require_present_fields(instance: object, *field_names: str) -> None:
    missing = [name for name in field_names if getattr(instance, name) is None]
    if missing:
        raise ValueError(
            f"{type(instance).__name__} requires non-null fields: {', '.join(missing)}"
        )


@dataclass(frozen=True, kw_only=True)
class PreparedCertifiedDirectPipelineRequest(CertifiedDirectPipelineRequest):
    certified_pocket_prep: CertifiedPocketPreparation | None = None

    def __post_init__(self) -> None:
        _require_present_fields(self, "certified_pocket_prep")

    def prepare(self) -> "PreparedCertifiedDirectPipelineRequest":
        return self


class GeometricSamplingPlan(ABC):
    @abstractmethod
    def sample(
        self, request: "PreparedGeometricPipelineRequest"
    ) -> tuple[jax.Array, PoseVector]:
        """Sample a pose batch for the prepared geometric request."""


class DerivedSamplingPlan(GeometricSamplingPlan, ABC):
    @property
    @abstractmethod
    def sampler(self) -> Callable[..., tuple[jax.Array, PoseVector]]:
        """Concrete sampler function."""

    def sample(
        self, request: "PreparedGeometricPipelineRequest"
    ) -> tuple[jax.Array, PoseVector]:
        return call_with_derived_kwargs(
            self.sampler,
            request,
            aliases=None,
            **self.sampling_kwargs(),
        )

    def sampling_kwargs(self) -> dict[str, object]:
        return {}


@dataclass(frozen=True, kw_only=True)
class PreparedGeometricPipelineRequest(PipelineDockingRequest):
    geometric_pocket_prep: GeometricPocketPreparation
    sampling_plan: GeometricSamplingPlan

    def prepare(self) -> "PreparedGeometricPipelineRequest":
        return self

    def sample_pose_batch(self) -> tuple[jax.Array, PoseVector]:
        return self.sampling_plan.sample(self)


@dataclass(frozen=True)
class PocketGuidedSamplingPlan(DerivedSamplingPlan):
    geometric_detected_pocket: GeometricDetectedPocket

    @property
    def sampler(self) -> Callable[..., tuple[jax.Array, PoseVector]]:
        return _sample_geometric_pocket_guided_pose_vectors

    def sampling_kwargs(self) -> dict[str, object]:
        return {"geometric_detected_pocket": self.geometric_detected_pocket}


@dataclass(frozen=True)
class BoxFallbackSamplingPlan(DerivedSamplingPlan):
    @property
    def sampler(self) -> Callable[..., tuple[jax.Array, PoseVector]]:
        return _sample_box_guided_pose_vectors


def derive_geometric_sampling_plan(
    prep: GeometricPocketPreparation,
) -> GeometricSamplingPlan:
    if prep.detected_pocket is None:
        return BoxFallbackSamplingPlan()
    return PocketGuidedSamplingPlan(geometric_detected_pocket=prep.detected_pocket)


RequestTypeT = TypeVar("RequestTypeT")


def derive_request_kwargs(
    request_type: type[RequestTypeT],
    source: object | dict[str, Any],
    /,
    **overrides: Any,
) -> dict[str, Any]:
    if isinstance(source, dict):
        source_values = source
    elif is_dataclass(source):
        source_values = {
            field.name: getattr(source, field.name)
            for field in dataclass_fields(source)
        }
    else:
        raise TypeError(
            f"Cannot derive {request_type.__name__} from non-dataclass source {type(source).__name__}."
        )
    derived = {
        field.name: source_values[field.name]
        for field in dataclass_fields(cast(Any, request_type))
        if field.init and field.name in source_values
    }
    derived.update(overrides)
    return derived


def derive_request(
    request_type: type[RequestTypeT],
    source: object | dict[str, Any],
    /,
    **overrides: Any,
) -> RequestTypeT:
    return request_type(**derive_request_kwargs(request_type, source, **overrides))


def derive_callable_kwargs(
    func: Callable[..., Any],
    source: object | dict[str, Any],
    /,
    *,
    aliases: dict[str, str] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    alias_map = {} if aliases is None else aliases
    signature = inspect.signature(func)
    if isinstance(source, dict):
        source_values = source
    elif is_dataclass(source):
        source_values = {
            field.name: getattr(source, field.name)
            for field in dataclass_fields(source)
        }
    else:
        source_values = None
    kwargs: dict[str, Any] = {}
    accepts_var_keyword = False
    for name, parameter in signature.parameters.items():
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL,):
            continue
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            accepts_var_keyword = True
            continue
        if name in overrides:
            kwargs[name] = overrides[name]
            continue
        source_name = alias_map.get(name, name)
        if source_values is not None and source_name in source_values:
            kwargs[name] = source_values[source_name]
            continue
        if (
            not hasattr(source, source_name)
            and parameter.default is not inspect.Parameter.empty
        ):
            continue
        kwargs[name] = getattr(source, source_name)
    if accepts_var_keyword:
        for name, value in overrides.items():
            if name not in kwargs:
                kwargs[name] = value
    return kwargs


def call_with_derived_kwargs(
    func: Callable[..., Any],
    source: object | dict[str, Any],
    /,
    *,
    aliases: dict[str, str] | None = None,
    **overrides: Any,
) -> Any:
    return func(**derive_callable_kwargs(func, source, aliases=aliases, **overrides))


def resolve_request_electrostatics(
    request: RoutedDockingRequest,
    *,
    engine: ScoringEngine | None = None,
) -> CertifiedRealSpaceEwaldSpec | None:
    return call_with_derived_kwargs(
        _resolve_route_scoring_electrostatics,
        request,
        engine=request.engine if engine is None else engine,
    )


def resolve_request_scoring_context(
    request: RoutedDockingRequest,
    *,
    engine: ScoringEngine | None = None,
) -> CertifiedScoringContext:
    effective_engine = request.engine if engine is None else engine
    electrostatics = resolve_request_electrostatics(request, engine=effective_engine)
    if not isinstance(request, PipelineDockingRequest):
        return CertifiedScoringContext(
            exact_chemistry_mode=ExactChemistryMode.NONE,
            electrostatics=electrostatics,
            rich_chemistry_plan=None,
        )
    return build_certified_scoring_context(
        exact_chemistry_mode=request.exact_chemistry_mode,
        electrostatics=electrostatics,
        receptor_coords=request.protein_coords,
        receptor_radii=request.receptor_radii,
        receptor_elements=request.receptor_elements,
        receptor_file=request.receptor_file,
        ligand_source_path=request.ligand_source_path,
        ligand_ctx=request.ligand_ctx,
        target_electrostatic_error=request.target_error,
    )


def derive_route_scoring_kwargs(
    request: RoutedDockingRequest,
    *,
    poses_coords: jnp.ndarray,
    electrostatics: CertifiedRealSpaceEwaldSpec | None,
    engine: ScoringEngine | None = None,
    **extra_overrides: object,
) -> dict[str, Any]:
    return {
        "engine": request.engine if engine is None else engine,
        "receptor_coords": request.receptor_coords,
        "receptor_radii": request.receptor_radii,
        "ligand_radii": request.ligand_radii,
        "poses_coords": poses_coords,
        "electrostatics": electrostatics,
        **dict(request.scoring_kwargs),
        **dict(extra_overrides),
    }


def _ligand_extent_radius(ligand_ctx: LigandContext) -> float:
    centered = ligand_ctx.base_coords - ligand_ctx.center_of_mass
    if centered.shape[0] == 0:
        return 0.0
    return float(jnp.max(jnp.linalg.norm(centered, axis=1)))


def _apply_binding_site_restriction(
    protein_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    receptor_elements: tuple[str, ...] | None,
    precomputed_receptor_charges: jnp.ndarray | None,
    ligand_ctx: LigandContext,
    box: DockingBox,
    binding_site: BindingSite,
    target_error: float,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    tuple[str, ...] | None,
    jnp.ndarray | None,
    DockingBox,
]:
    interaction_cutoff = compute_certified_cutoff(target_error)
    restriction_radius = (
        binding_site.radius + interaction_cutoff + _ligand_extent_radius(ligand_ctx)
    )
    distances = jnp.linalg.norm(protein_coords - binding_site.center, axis=1)
    keep_mask = distances <= restriction_radius
    if not bool(jnp.any(keep_mask)):
        return (
            protein_coords,
            receptor_radii,
            receptor_elements,
            precomputed_receptor_charges,
            box,
        )

    kept_indices = jnp.nonzero(keep_mask, size=int(jnp.sum(keep_mask)))[0]
    restricted_coords = protein_coords[kept_indices]
    restricted_radii = receptor_radii[kept_indices]
    restricted_elements = (
        None
        if receptor_elements is None
        else tuple(receptor_elements[int(i)] for i in np.asarray(kept_indices))
    )
    restricted_charges = (
        None
        if precomputed_receptor_charges is None
        else precomputed_receptor_charges[kept_indices]
    )
    restricted_box = DockingBox(
        center=binding_site.center,
        size=jnp.full((3,), 2.0 * binding_site.radius),
    )
    return (
        restricted_coords,
        restricted_radii,
        restricted_elements,
        restricted_charges,
        restricted_box,
    )


def _derive_certified_binding_site_from_box(
    protein_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    receptor_elements: tuple[str, ...] | None,
    box: DockingBox,
) -> tuple[CertifiedDetectedPocket | None, CertifiedPocketFailureReason | None]:
    region = extract_local_pocket_region_view(
        protein_coords=protein_coords,
        receptor_elements=receptor_elements,
        box_center=box.center,
        box_size=float(jnp.max(box.size)),
    )
    if region.coords.shape[0] == 0:
        return None, CertifiedPocketFailureReason.NO_LOCAL_REGION
    pocket_radii = receptor_radii[region.indices]
    pocket = detect_certified_pocket(
        region.coords, region.elements, pocket_radii=pocket_radii
    )
    if pocket is None:
        return None, CertifiedPocketFailureReason.NO_CERTIFIED_POCKET
    return pocket, None


def _derive_geometric_pocket_from_box(
    protein_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    receptor_elements: tuple[str, ...] | None,
    box: DockingBox,
) -> GeometricDetectedPocket | None:
    region = extract_local_pocket_region_view(
        protein_coords=protein_coords,
        receptor_elements=receptor_elements,
        box_center=box.center,
        box_size=float(jnp.max(box.size)),
    )
    pocket_radii = receptor_radii[region.indices]
    return detect_geometric_pocket(
        region.coords,
        region.elements,
        pocket_radii=pocket_radii,
    )


def _create_certified_pose_vectors(
    box: DockingBox,
    n_poses: int,
    certified_binding_site: CertifiedBindingSite | None,
    rigid_seed_family_plan: CertifiedRigidSeedFamilyPlan | None = None,
) -> CertifiedPoseGeneration:
    from dq_dock_engine.docking.formal_sampling import (
        create_certified_binding_site_action_family,
        create_certified_global_action_family,
        materialize_certified_rigid_seed_family,
    )

    certified_family = (
        materialize_certified_rigid_seed_family(rigid_seed_family_plan)
        if rigid_seed_family_plan is not None
        else (
            create_certified_global_action_family(box, n_poses)
            if certified_binding_site is None
            else create_certified_binding_site_action_family(
                certified_binding_site, n_poses
            )
        )
    )
    pose_vecs = PoseVector(
        translation=certified_family.translations,
        quaternion=certified_family.quaternions,
    )
    return CertifiedPoseGeneration(pose_vecs=pose_vecs, family=certified_family)


def _resolved_certified_seed_binding_site(
    request: DockingRequestBase,
) -> CertifiedBindingSite | None:
    if (
        isinstance(request, PipelineDockingRequest)
        and request.certified_pocket_prep is not None
    ):
        return cast(
            CertifiedPocketPreparation, request.certified_pocket_prep
        ).plan.binding_site
    return request.certified_binding_site


def _derive_request_rigid_seed_family_plan(
    request: DockingRequestBase,
    *,
    n_poses: int,
    enforce_target_translation_cover_radius: bool = True,
) -> CertifiedRigidSeedFamilyPlan:
    from dq_dock_engine.docking.formal_sampling import (
        derive_certified_rigid_seed_family_plan,
    )

    return derive_certified_rigid_seed_family_plan(
        request.box,
        n_poses,
        certified_binding_site=_resolved_certified_seed_binding_site(request),
        target_translation_cover_radius=(
            None
            if request.config is None or not enforce_target_translation_cover_radius
            else request.config.target_rmsd
        ),
    )


def _sample_box_guided_pose_vectors(
    key: jax.Array,
    box: DockingBox,
    n_poses: int,
    protein_coords: jnp.ndarray,
    receptor_elements: tuple[str, ...] | None,
    ligand_ctx: LigandContext,
) -> tuple[jax.Array, PoseVector]:
    from dq_dock_engine.docking.pocket_sampling import (
        sample_intelligent_poses,
        SamplingStrategy,
    )

    key_samp, next_key = jax.random.split(key)
    translations, quaternions = sample_intelligent_poses(
        key=key_samp,
        box_center=box.center,
        box_size=float(box.size[0]),
        n_poses=n_poses,
        protein_coords=protein_coords,
        receptor_elements=receptor_elements,
        ligand_com=ligand_ctx.center_of_mass,
        strategy=SamplingStrategy.HYBRID,
    )
    return next_key, PoseVector(translation=translations, quaternion=quaternions)


def _sample_certified_pocket_guided_pose_vectors(
    key: jax.Array,
    n_poses: int,
    certified_detected_pocket: CertifiedDetectedPocket,
    ligand_ctx: LigandContext,
) -> tuple[jax.Array, PoseVector]:
    from dq_dock_engine.docking.pocket_sampling import (
        sample_intelligent_poses_from_certified_pocket,
        SamplingStrategy,
    )

    key_samp, next_key = jax.random.split(key)
    translations, quaternions = sample_intelligent_poses_from_certified_pocket(
        key=key_samp,
        n_poses=n_poses,
        certified_pocket=certified_detected_pocket,
        ligand_com=ligand_ctx.center_of_mass,
        strategy=SamplingStrategy.HYBRID,
    )
    return next_key, PoseVector(translation=translations, quaternion=quaternions)


def _sample_geometric_pocket_guided_pose_vectors(
    key: jax.Array,
    n_poses: int,
    geometric_detected_pocket: GeometricDetectedPocket,
    ligand_ctx: LigandContext,
) -> tuple[jax.Array, PoseVector]:
    from dq_dock_engine.docking.pocket_sampling import (
        sample_intelligent_poses_from_geometric_pocket,
        SamplingStrategy,
    )

    key_samp, next_key = jax.random.split(key)
    translations, quaternions = sample_intelligent_poses_from_geometric_pocket(
        key=key_samp,
        n_poses=n_poses,
        geometric_pocket=geometric_detected_pocket,
        ligand_com=ligand_ctx.center_of_mass,
        strategy=SamplingStrategy.HYBRID,
    )
    return next_key, PoseVector(translation=translations, quaternion=quaternions)


class BlindDockingPreparer(ABC, Generic[BindingSiteT, DetectedPocketT, PlanT]):
    preparation_type: ClassVar[type[BlindDockingPreparation]]
    plan_type: ClassVar[type[BlindDockingPlan]]

    def prepare_request(
        self,
        request: BlindDockingPreparationRequest,
    ) -> BlindDockingPreparation:
        return self.prepare(**derive_callable_kwargs(self.prepare, request))

    def prepare(
        self,
        *,
        protein_coords: jnp.ndarray,
        receptor_radii: jnp.ndarray,
        receptor_elements: tuple[str, ...] | None,
        precomputed_receptor_charges: jnp.ndarray | None,
        ligand_ctx: LigandContext,
        box: DockingBox,
        target_error: float,
        explicit_binding_site: BindingSiteT | None = None,
        coarse_target_error: float = 0.0,
        adaptive_coarse_target_errors: tuple[float, ...] | None = None,
        use_softened_coarse_prefilter: bool = False,
    ) -> BlindDockingPreparation:
        detected_pocket: DetectedPocketT | None = None
        failure_reason: CertifiedPocketFailureReason | None = None
        binding_site = explicit_binding_site
        if binding_site is None:
            detected_pocket, failure_reason = self.detect_pocket_from_box(
                protein_coords=protein_coords,
                receptor_radii=receptor_radii,
                receptor_elements=receptor_elements,
                box=box,
            )
            if detected_pocket is not None:
                binding_site = self.binding_site_from_detected_pocket(detected_pocket)

        restricted_coords = protein_coords
        restricted_radii = receptor_radii
        restricted_elements = receptor_elements
        restricted_charges = precomputed_receptor_charges
        restricted_box = box
        theorem_handles = self.binding_site_theorem_handles(binding_site)
        if binding_site is not None:
            (
                restricted_coords,
                restricted_radii,
                restricted_elements,
                restricted_charges,
                restricted_box,
            ) = _apply_binding_site_restriction(
                protein_coords=protein_coords,
                receptor_radii=receptor_radii,
                receptor_elements=receptor_elements,
                precomputed_receptor_charges=precomputed_receptor_charges,
                ligand_ctx=ligand_ctx,
                box=box,
                binding_site=binding_site,
                target_error=target_error,
            )
        theorem_handles = self.merge_theorem_handles(detected_pocket, theorem_handles)
        plan = self.build_plan(
            binding_site=binding_site,
            restricted_box=restricted_box,
            restricted_atom_count=int(restricted_coords.shape[0]),
            detected_pocket=detected_pocket,
            failure_reason=failure_reason,
            theorem_handles=theorem_handles,
            coarse_target_error=coarse_target_error,
            adaptive_coarse_target_errors=adaptive_coarse_target_errors,
            use_softened_coarse_prefilter=use_softened_coarse_prefilter,
        )
        return self.build_preparation(
            protein_coords=restricted_coords,
            receptor_radii=restricted_radii,
            receptor_elements=restricted_elements,
            precomputed_receptor_charges=restricted_charges,
            box=restricted_box,
            detected_pocket=detected_pocket,
            plan=plan,
        )

    @abstractmethod
    def detect_pocket_from_box(
        self,
        *,
        protein_coords: jnp.ndarray,
        receptor_radii: jnp.ndarray,
        receptor_elements: tuple[str, ...] | None,
        box: DockingBox,
    ) -> tuple[DetectedPocketT | None, CertifiedPocketFailureReason | None]:
        """Derive a local pocket object from the docking box."""

    @abstractmethod
    def binding_site_from_detected_pocket(
        self, detected_pocket: DetectedPocketT
    ) -> BindingSiteT:
        """Project a detected pocket down to its binding-site abstraction."""

    def build_plan(
        self,
        *,
        binding_site: BindingSiteT | None,
        restricted_box: DockingBox,
        restricted_atom_count: int,
        detected_pocket: DetectedPocketT | None,
        failure_reason: CertifiedPocketFailureReason | None,
        theorem_handles: tuple[str, ...],
        coarse_target_error: float,
        adaptive_coarse_target_errors: tuple[float, ...] | None,
        use_softened_coarse_prefilter: bool,
    ) -> PlanT:
        return cast(
            PlanT,
            self.plan_type(
                binding_site=binding_site,
                restricted_box=restricted_box,
                restricted_atom_count=restricted_atom_count,
                **self.plan_extras(
                    detected_pocket=detected_pocket,
                    failure_reason=failure_reason,
                    theorem_handles=theorem_handles,
                    coarse_target_error=coarse_target_error,
                    adaptive_coarse_target_errors=adaptive_coarse_target_errors,
                    use_softened_coarse_prefilter=use_softened_coarse_prefilter,
                ),
            ),
        )

    def build_preparation(
        self,
        *,
        protein_coords: jnp.ndarray,
        receptor_radii: jnp.ndarray,
        receptor_elements: tuple[str, ...] | None,
        precomputed_receptor_charges: jnp.ndarray | None,
        box: DockingBox,
        detected_pocket: DetectedPocketT | None,
        plan: PlanT,
    ) -> BlindDockingPreparation:
        return cast(Any, self.preparation_type)(
            protein_coords=protein_coords,
            receptor_radii=receptor_radii,
            receptor_elements=receptor_elements,
            precomputed_receptor_charges=precomputed_receptor_charges,
            box=box,
            detected_pocket=detected_pocket,
            plan=plan,
        )

    @abstractmethod
    def plan_extras(
        self,
        *,
        detected_pocket: DetectedPocketT | None,
        failure_reason: CertifiedPocketFailureReason | None,
        theorem_handles: tuple[str, ...],
        coarse_target_error: float,
        adaptive_coarse_target_errors: tuple[float, ...] | None,
        use_softened_coarse_prefilter: bool,
    ) -> dict[str, object]:
        """Route-specific plan fields beyond the shared blind-docking skeleton."""

    def binding_site_theorem_handles(
        self, binding_site: BindingSiteT | None
    ) -> tuple[str, ...]:
        return ()

    def merge_theorem_handles(
        self,
        detected_pocket: DetectedPocketT | None,
        theorem_handles: tuple[str, ...],
    ) -> tuple[str, ...]:
        return theorem_handles


class CertifiedBlindDockingPreparer(
    BlindDockingPreparer[
        CertifiedBindingSite,
        CertifiedDetectedPocket,
        CertifiedBlindDockingPlan,
    ]
):
    preparation_type = CertifiedPocketPreparation
    plan_type = CertifiedBlindDockingPlan

    def detect_pocket_from_box(
        self,
        *,
        protein_coords: jnp.ndarray,
        receptor_radii: jnp.ndarray,
        receptor_elements: tuple[str, ...] | None,
        box: DockingBox,
    ) -> tuple[CertifiedDetectedPocket | None, CertifiedPocketFailureReason | None]:
        return _derive_certified_binding_site_from_box(
            protein_coords=protein_coords,
            receptor_radii=receptor_radii,
            receptor_elements=receptor_elements,
            box=box,
        )

    def binding_site_from_detected_pocket(
        self, detected_pocket: CertifiedDetectedPocket
    ) -> CertifiedBindingSite:
        return detected_pocket.binding_site

    def plan_extras(
        self,
        *,
        detected_pocket: CertifiedDetectedPocket | None,
        failure_reason: CertifiedPocketFailureReason | None,
        theorem_handles: tuple[str, ...],
        coarse_target_error: float,
        adaptive_coarse_target_errors: tuple[float, ...] | None,
        use_softened_coarse_prefilter: bool,
    ) -> dict[str, object]:
        return dict(
            certified_pocket_found=detected_pocket is not None,
            certified_failure_reason=failure_reason,
            coarse_target_error=coarse_target_error,
            adaptive_coarse_target_errors=adaptive_coarse_target_errors,
            use_softened_coarse_prefilter=use_softened_coarse_prefilter,
            theorem_handles=theorem_handles,
        )

    def binding_site_theorem_handles(
        self, binding_site: CertifiedBindingSite | None
    ) -> tuple[str, ...]:
        return (
            ()
            if binding_site is None
            else _merge_theorem_handles(
                binding_site.theorem_handles,
                pocket_cross_docking_handles(),
            )
        )

    def merge_theorem_handles(
        self,
        detected_pocket: CertifiedDetectedPocket | None,
        theorem_handles: tuple[str, ...],
    ) -> tuple[str, ...]:
        if detected_pocket is None:
            return theorem_handles
        return _merge_theorem_handles(
            detected_pocket.theorem_handles,
            theorem_handles,
            pocket_cross_docking_handles(),
        )


class GeometricBlindDockingPreparer(
    BlindDockingPreparer[
        GeometricBindingSite,
        GeometricDetectedPocket,
        GeometricBlindDockingPlan,
    ]
):
    preparation_type = GeometricPocketPreparation
    plan_type = GeometricBlindDockingPlan

    def detect_pocket_from_box(
        self,
        *,
        protein_coords: jnp.ndarray,
        receptor_radii: jnp.ndarray,
        receptor_elements: tuple[str, ...] | None,
        box: DockingBox,
    ) -> tuple[GeometricDetectedPocket | None, CertifiedPocketFailureReason | None]:
        return (
            _derive_geometric_pocket_from_box(
                protein_coords=protein_coords,
                receptor_radii=receptor_radii,
                receptor_elements=receptor_elements,
                box=box,
            ),
            None,
        )

    def binding_site_from_detected_pocket(
        self, detected_pocket: GeometricDetectedPocket
    ) -> GeometricBindingSite:
        return detected_pocket.binding_site

    def plan_extras(
        self,
        *,
        detected_pocket: GeometricDetectedPocket | None,
        failure_reason: CertifiedPocketFailureReason | None,
        theorem_handles: tuple[str, ...],
        coarse_target_error: float,
        adaptive_coarse_target_errors: tuple[float, ...] | None,
        use_softened_coarse_prefilter: bool,
    ) -> dict[str, object]:
        del detected_pocket
        del failure_reason
        del theorem_handles
        del coarse_target_error
        del adaptive_coarse_target_errors
        del use_softened_coarse_prefilter
        return dict(
            sampling_strategy=SamplingStrategy.HYBRID,
        )


_CERTIFIED_PREPARER = CertifiedBlindDockingPreparer()
_GEOMETRIC_PREPARER = GeometricBlindDockingPreparer()


def _prepare_certified_blind_docking(
    request: CertifiedPreparationRequest,
) -> CertifiedPocketPreparation:
    prep = _CERTIFIED_PREPARER.prepare_request(request)
    return cast(CertifiedPocketPreparation, prep)


def _prepare_geometric_blind_docking(
    request: GeometricPreparationRequest,
) -> GeometricPocketPreparation:
    prep = _GEOMETRIC_PREPARER.prepare_request(request)
    return cast(GeometricPocketPreparation, prep)


def _resolve_route_scoring_electrostatics(
    effective_engine: ScoringEngine,
    ligand_ctx: LigandContext,
    receptor_elements: tuple[str, ...] | None,
    charge_method: ChargeMethod | None,
    receptor_file: str | Path | None,
    precomputed_receptor_charges: jnp.ndarray | None = None,
) -> CertifiedRealSpaceEwaldSpec | None:
    if effective_engine != ScoringEngine.CERTIFIED_LJ_REALSPACE_EWALD:
        return None

    if precomputed_receptor_charges is not None:
        if ligand_ctx.charges is None:
            raise ValueError(
                "Precomputed receptor charges require ligand_ctx.charges for electrostatic scoring."
            )
        return CertifiedRealSpaceEwaldSpec(
            receptor_charges=precomputed_receptor_charges,
            ligand_charges=ligand_ctx.charges,
        )

    if charge_method is None:
        raise ValueError(
            "CERTIFIED_LJ_REALSPACE_EWALD requires an explicit ChargeMethod."
        )

    assigner = create_charge_assigner(charge_method)

    if assigner.method == ChargeMethod.SIMPLE:
        if receptor_elements is None:
            raise ValueError("SIMPLE electrostatic scoring requires receptor_elements.")
        if not ligand_ctx.elements and ligand_ctx.charges is None:
            raise ValueError(
                "SIMPLE electrostatic scoring requires ligand elements or precomputed ligand charges."
            )
        receptor_charges = assigner.assign(receptor_elements).charges
        ligand_charges = (
            ligand_ctx.charges
            if ligand_ctx.charges is not None
            else assigner.assign(ligand_ctx.elements).charges
        )
        return CertifiedRealSpaceEwaldSpec(
            receptor_charges=receptor_charges,
            ligand_charges=ligand_charges,
        )

    if receptor_file is None:
        raise ValueError(
            f"ChargeMethod {assigner.method.name} requires receptor_file for electrostatic scoring."
        )
    if ligand_ctx.charges is None:
        raise ValueError(
            f"ChargeMethod {assigner.method.name} requires precomputed ligand_ctx.charges for electrostatic scoring."
        )

    receptor_charges = assigner.assign(receptor_file).charges
    return CertifiedRealSpaceEwaldSpec(
        receptor_charges=receptor_charges,
        ligand_charges=ligand_ctx.charges,
    )


def _posewise_active_torsion_mask(
    poses_coords: jnp.ndarray,
    receptor_coords: jnp.ndarray,
    bonds: tuple[RotatableBond, ...],
    scoring_cutoff: float = 6.0,
) -> jnp.ndarray:
    """For each pose, which torsion bonds have rotating atoms in receptor range?

    Returns shape (B, n_bonds) boolean mask.

    Theorem CSC45 (torsion_locality_radius): If a scoring function evaluates to
    exactly 0 beyond a cutoff radius R_c, and a kinematic rotation keeps an atom
    strictly beyond R_c from the receptor for ALL possible rotation angles,
    then that torsion has a Lipschitz constant of exactly 0 with respect to that
    receptor atom.

    Therefore, torsion bonds whose rotating atoms are ALL > scoring_cutoff from
    ALL receptor atoms are "inactive" and can be excluded from conformer search.

    The cutoff should be the maximum across all scoring terms:
      - Gaussian contact surrogate: certified cutoff from GaussianDecayBounds
      - Softened LJ: R_cutoff from optimal_cutoff
      - Electrostatics: Coulomb/screened Coulomb cutoff
      Default 6.0 Å is conservative overestimate for typical parameters.
    """
    masks = []
    for bond in bonds:
        rotating_coords = poses_coords[:, list(bond.rotating_atom_indices), :]
        dists = jnp.linalg.norm(
            rotating_coords[:, :, None, :] - receptor_coords[None, None, :, :],
            axis=-1,
        )
        min_dist_per_pose = jnp.min(dists, axis=(1, 2))
        max_rotation_displacement = 2.0 * float(bond.max_arm_length)
        masks.append(min_dist_per_pose <= scoring_cutoff + max_rotation_displacement)
    return jnp.stack(masks, axis=-1)


def _posewise_active_torsion_mask_np(
    poses_coords: jnp.ndarray,
    receptor_coords: jnp.ndarray,
    bonds: tuple[RotatableBond, ...],
    scoring_cutoff: float = 6.0,
) -> np.ndarray:
    if not bonds:
        return np.zeros((int(poses_coords.shape[0]), 0), dtype=bool)
    poses_np = np.asarray(jax.device_get(poses_coords), dtype=np.float32)
    receptor_np = np.asarray(jax.device_get(receptor_coords), dtype=np.float32)
    masks = np.empty((poses_np.shape[0], len(bonds)), dtype=bool)
    for bond_idx, bond in enumerate(bonds):
        rotating_coords = poses_np[
            :, np.asarray(bond.rotating_atom_indices, dtype=np.int32), :
        ]
        dists = np.linalg.norm(
            rotating_coords[:, :, None, :] - receptor_np[None, None, :, :],
            axis=-1,
        )
        min_dist_per_pose = np.min(dists, axis=(1, 2))
        max_rotation_displacement = 2.0 * float(bond.max_arm_length)
        masks[:, bond_idx] = min_dist_per_pose <= (
            scoring_cutoff + max_rotation_displacement
        )
    return masks


def _posewise_conformer_correction(
    poses_coords: jnp.ndarray,
    receptor_coords: jnp.ndarray,
    rotatable_bonds: tuple[RotatableBond, ...],
    per_bond_local_bounds: jnp.ndarray | None,
    scoring_cutoff: float = 6.0,
) -> jnp.ndarray:
    family = _active_subset_budget_family(
        poses_coords,
        receptor_coords,
        rotatable_bonds,
        per_bond_local_bounds,
        scoring_cutoff=scoring_cutoff,
    )
    return jnp.asarray(family.totals_array())


def _merge_pruning_delta_components(
    lhs: tuple[CertifiedPruningDeltaComponent, ...],
    rhs: tuple[CertifiedPruningDeltaComponent, ...],
) -> tuple[CertifiedPruningDeltaComponent, ...]:
    merged: dict[
        tuple[CertifiedPruningDeltaComponentKind, str],
        CertifiedPruningDeltaComponent,
    ] = {}
    for component in (*lhs, *rhs):
        key = (component.kind, component.label)
        if key not in merged:
            merged[key] = component
            continue
        prev = merged[key]
        chosen = component if component.delta >= prev.delta else prev
        merged[key] = CertifiedPruningDeltaComponent(
            label=chosen.label,
            kind=chosen.kind,
            active=prev.active or component.active,
            delta=max(prev.delta, component.delta),
            theorem_handles=_merge_theorem_handles(
                prev.theorem_handles,
                component.theorem_handles,
            ),
            witness_handles=_merge_theorem_handles(
                prev.witness_handles,
                component.witness_handles,
            ),
            note=chosen.note,
        )
    return tuple(merged.values())


def _merge_pruning_delta_budgets(
    lhs: CertifiedPruningDeltaBudget,
    rhs: CertifiedPruningDeltaBudget,
) -> CertifiedPruningDeltaBudget:
    assert lhs.components is not None
    assert rhs.components is not None
    merged_components = _merge_pruning_delta_components(
        lhs.components,
        rhs.components,
    )
    return CertifiedPruningDeltaBudget.from_components(
        source=(
            lhs.source if lhs.source == rhs.source else f"{lhs.source}+{rhs.source}"
        ),
        components=merged_components,
        theorem_handles=_merge_theorem_handles(
            lhs.theorem_handles,
            rhs.theorem_handles,
        ),
        witness_handles=_merge_theorem_handles(
            lhs.witness_handles,
            rhs.witness_handles,
        ),
    )


def _active_subset_budget_family(
    poses_coords: jnp.ndarray,
    receptor_coords: jnp.ndarray,
    rotatable_bonds: tuple[RotatableBond, ...],
    per_bond_local_bounds: jnp.ndarray | None,
    *,
    scoring_cutoff: float = 6.0,
    pose_indices: np.ndarray | None = None,
    active_subset_source: str = "posewise_interaction_cutoff",
) -> CertifiedActiveSubsetBudgetFamily:
    theorem_handles = pose_specific_improvement_budget_theorem_handles()
    n_poses = int(poses_coords.shape[0])
    use_batched_mask = n_poses > 8
    if pose_indices is None:
        pose_indices = np.arange(n_poses, dtype=np.int32)
    if per_bond_local_bounds is None or not rotatable_bonds:
        return CertifiedActiveSubsetBudgetFamily(
            budgets=tuple(
                CertifiedActiveSubsetBudget(
                    pose_index=int(pose_index),
                    active_torsion_indices=(),
                    active_torsion_mask=(),
                    per_bond_local_bounds=(),
                    total_budget=0.0,
                    active_subset_source=active_subset_source,
                    theorem_handles=theorem_handles,
                )
                for pose_index in pose_indices.tolist()
            ),
            theorem_handles=theorem_handles,
        )
    per_bond_bounds_np = np.asarray(
        jax.device_get(per_bond_local_bounds),
        dtype=np.float64,
    )
    if per_bond_bounds_np.ndim == 1:
        per_pose_bounds_np = np.tile(per_bond_bounds_np[None, :], (n_poses, 1))
    else:
        per_pose_bounds_np = per_bond_bounds_np
    active_mask_batched = None
    if use_batched_mask:
        active_mask_batched = _posewise_active_torsion_mask_np(
            poses_coords,
            receptor_coords,
            rotatable_bonds,
            scoring_cutoff=scoring_cutoff,
        )
    budgets: list[CertifiedActiveSubsetBudget] = []
    for local_pose_index, pose_index in enumerate(pose_indices.tolist()):
        if active_mask_batched is not None:
            pose_active_mask_np = active_mask_batched[local_pose_index]
        else:
            _, pose_active_mask_np = _active_rotatable_bonds_for_pose(
                poses_coords[local_pose_index],
                receptor_coords,
                rotatable_bonds,
                scoring_cutoff=scoring_cutoff,
            )
        pose_mask = tuple(bool(flag) for flag in pose_active_mask_np.tolist())
        budgets.append(
            CertifiedActiveSubsetBudget(
                pose_index=int(pose_index),
                active_torsion_indices=tuple(
                    int(idx) for idx in np.flatnonzero(pose_active_mask_np).tolist()
                ),
                active_torsion_mask=pose_mask,
                per_bond_local_bounds=tuple(
                    float(v) for v in per_pose_bounds_np[local_pose_index].tolist()
                ),
                total_budget=float(
                    np.sum(pose_active_mask_np * per_pose_bounds_np[local_pose_index])
                ),
                active_subset_source=active_subset_source,
                theorem_handles=theorem_handles,
            )
        )
    return CertifiedActiveSubsetBudgetFamily(
        budgets=tuple(budgets),
        theorem_handles=theorem_handles,
    )


def _posewise_improvement_budget_family(
    poses_coords: jnp.ndarray,
    receptor_coords: jnp.ndarray,
    rotatable_bonds: tuple[RotatableBond, ...],
    per_bond_local_bounds: jnp.ndarray | None,
    *,
    scoring_cutoff: float = 6.0,
    pose_indices: np.ndarray | None = None,
    active_subset_source: str = "posewise_interaction_cutoff",
) -> PoseSpecificImprovementBudgetFamily:
    return _active_subset_budget_family(
        poses_coords,
        receptor_coords,
        rotatable_bonds,
        per_bond_local_bounds,
        scoring_cutoff=scoring_cutoff,
        pose_indices=pose_indices,
        active_subset_source=active_subset_source,
    ).as_pose_specific_improvement_family()


def _posewise_rigid_improvement_budget_family(
    request: "PipelineDockingRequest",
    scoring_context: CertifiedScoringContext,
    *,
    poses_coords: jnp.ndarray,
    base_translation_step: float,
    base_rotation_step_rad: float,
    ligand_radius: float,
    n_rounds: int,
    pose_indices: np.ndarray | None = None,
    budget_source: str = "rigid_local_refinement",
) -> tuple[PoseSpecificImprovementBudgetFamily, np.ndarray, np.ndarray, np.ndarray]:
    posewise_bounds, clearance_safe_mask, posewise_clearance = (
        _posewise_rigid_local_improvement_bounds(
            request,
            scoring_context,
            poses_coords=poses_coords,
            base_translation_step=base_translation_step,
            base_rotation_step_rad=base_rotation_step_rad,
            ligand_radius=ligand_radius,
            n_rounds=n_rounds,
        )
    )
    if pose_indices is None:
        pose_indices = np.arange(int(poses_coords.shape[0]), dtype=np.int32)
    theorem_handles = rigid_posewise_improvement_budget_theorem_handles()
    correction_np = np.zeros((int(poses_coords.shape[0]),), dtype=np.float32)
    budgets: list[PoseSpecificImprovementBudget] = []
    for local_idx, pose_index in enumerate(pose_indices.tolist()):
        if not bool(clearance_safe_mask[local_idx]):
            continue
        total_budget = float(posewise_bounds[local_idx])
        correction_np[local_idx] = np.float32(total_budget)
        budgets.append(
            PoseSpecificImprovementBudget(
                pose_index=int(pose_index),
                active_torsion_indices=(),
                active_torsion_mask=(),
                per_bond_local_bounds=(),
                total_budget=total_budget,
                active_subset_source=budget_source,
                theorem_handles=theorem_handles,
                kind=PoseSpecificImprovementBudgetKind.RIGID_LOCAL,
            )
        )
    return (
        PoseSpecificImprovementBudgetFamily(
            budgets=tuple(budgets),
            theorem_handles=theorem_handles,
        ),
        correction_np,
        np.asarray(clearance_safe_mask, dtype=bool),
        np.asarray(posewise_clearance, dtype=np.float32),
    )


def _certified_pruning_pass(
    request: PipelineDockingRequest,
    poses_coords: jnp.ndarray,
    electrostatics: Optional[CertifiedRealSpaceEwaldSpec],
    *,
    scoring_context: CertifiedScoringContext | None = None,
    extra_pruning_delta_budget: CertifiedPruningDeltaBudget | None = None,
    exact_chunk_scoring_context: CertifiedScoringContext | None = None,
    extra_posewise_correction_fn: Callable[[jnp.ndarray], np.ndarray] | None = None,
    rotatable_bonds: tuple[RotatableBond, ...] = (),
    rigid_improvement_scoring_context: CertifiedScoringContext | None = None,
    rigid_refinement_plan: "CertifiedRigidLocalRefinementPlan | None" = None,
) -> CertifiedPipelineExecutionPlan:
    """
    Perform a formally justified pruning pass on the global pose set.

    Uses the Lean-proven top-1 coarse ambiguity band (TK11, BD5) to eliminate
    poses that cannot possibly be the global minimum under the exact engine.
    """
    n_total = int(poses_coords.shape[0])
    if n_total == 0:
        empty = np.zeros((0,), dtype=np.float32)
        zero_budget = CertifiedPruningDeltaBudget.from_components(
            source="empty_pruning_delta",
            components=(),
            theorem_handles=("TK11",),
        )
        return CertifiedPipelineExecutionPlan(
            coarse_scores=jnp.asarray(empty),
            lower_bounds=jnp.asarray(empty),
            tau=0.0,
            retain_mask=jnp.asarray(empty.astype(bool)),
            pruning_delta_budget=zero_budget,
            theorem_handles=("TK11",),
        )

    # Runtime-execution detail only: evaluate the certified pruning quantities in
    # host-side chunks so large theorem-backed seed budgets do not require one
    # monolithic receptor x pose tensor on device.
    chunk_size = (
        n_total
        if n_total <= CERTIFIED_PRUNING_MONOLITHIC_THRESHOLD
        else CERTIFIED_PRUNING_CHUNK_SIZE
    )
    coarse_scores_np = np.empty((n_total,), dtype=np.float32)
    use_conf = _request_uses_conformer_search(request)
    use_rigid_improvement = (
        (not use_conf)
        and rigid_improvement_scoring_context is not None
        and rigid_refinement_plan is not None
    )
    needs_receptor_flex = (
        scoring_context is not None
        and scoring_context.receptor_conformations is not None
    )
    additive_correction_np = np.zeros((n_total,), dtype=np.float32)
    active_subset_budgets: list[CertifiedActiveSubsetBudget] = []
    rigid_improvement_budgets: list[PoseSpecificImprovementBudget] = []
    rigid_clearance_safe_mask_full: np.ndarray | None = (
        np.ones((n_total,), dtype=bool) if use_rigid_improvement else None
    )
    rigid_forced_prune_mask: np.ndarray | None = (
        np.zeros((n_total,), dtype=bool) if use_rigid_improvement else None
    )

    conformer_config = None
    per_bond_bounds = None
    if use_conf:
        conformer_config = _build_conformer_search_config(
            request,
            rotatable_bonds=rotatable_bonds,
        )
        per_bond_bounds = _conformer_local_improvement_bounds(
            request,
            rotatable_bonds,
            conformer_config.per_bond_lipschitz,
        )

    pruning_delta_budget: CertifiedPruningDeltaBudget | None = None

    for start in range(0, n_total, chunk_size):
        stop = min(start + chunk_size, n_total)
        chunk_coords = poses_coords[start:stop]
        n_chunk_real = int(stop - start)
        scored_chunk_coords = chunk_coords
        if n_chunk_real < chunk_size:
            pad_count = int(chunk_size - n_chunk_real)
            pad_coords = jnp.repeat(chunk_coords[:1], pad_count, axis=0)
            scored_chunk_coords = jnp.concatenate((chunk_coords, pad_coords), axis=0)
        phase_start = time.perf_counter()
        if exact_chunk_scoring_context is not None:
            exact_zero_delta_handles = (
                ("CB10", "CB11", "CB12")
                if electrostatics is not None
                else ("LJ10", "LJ11", "LJ12")
            )
            chunk_scores = _score_rigid_exact_pose_batch(
                request,
                poses_coords=scored_chunk_coords,
                electrostatics=electrostatics,
                scoring_context=exact_chunk_scoring_context,
            )
            chunk_delta_budget = CertifiedPruningDeltaBudget.from_components(
                source="exact_base_physics_pruning_delta_zero",
                components=(),
                theorem_handles=exact_zero_delta_handles,
            )
            chunk_posewise_softening_error = jnp.zeros_like(chunk_scores)
        else:
            (
                chunk_scores,
                chunk_delta_budget,
                chunk_posewise_softening_error,
            ) = _score_softened_pose_batch(
                request,
                poses_coords=scored_chunk_coords,
                electrostatics=electrostatics,
                scoring_context=scoring_context,
            )
        _runtime_profile_log("certified_pruning_chunk_softened_scores", phase_start)
        chunk_scores_np = np.asarray(
            jax.device_get(chunk_scores[:n_chunk_real]), dtype=np.float32
        )
        coarse_scores_np[start:stop] = chunk_scores_np

        if pruning_delta_budget is None:
            pruning_delta_budget = chunk_delta_budget
        else:
            pruning_delta_budget = _merge_pruning_delta_budgets(
                pruning_delta_budget,
                chunk_delta_budget,
            )

        chunk_correction_np = np.array(
            jax.device_get(chunk_posewise_softening_error[:n_chunk_real]),
            dtype=np.float32,
            copy=True,
        )
        if extra_posewise_correction_fn is not None:
            chunk_correction_np += np.asarray(
                extra_posewise_correction_fn(chunk_coords),
                dtype=np.float32,
            )
        if use_conf:
            chunk_family = _active_subset_budget_family(
                chunk_coords,
                request.protein_coords,
                rotatable_bonds,
                per_bond_bounds,
                scoring_cutoff=compute_certified_cutoff(request.target_error),
                pose_indices=np.arange(start, stop, dtype=np.int32),
            )
            active_subset_budgets.extend(chunk_family.budgets)
            chunk_correction_np += chunk_family.totals_array()
        elif use_rigid_improvement:
            # Defer rigid local-improvement tightening to a second pass so we can
            # cheaply pre-prune by the global certified rigid cap first.
            pass
        if needs_receptor_flex:
            assert scoring_context is not None
            posewise_flex_error, coarse_phys_delta = (
                scoring_context.posewise_receptor_flex_error_softened_batch(
                    receptor_coords=request.protein_coords,
                    poses_coords=chunk_coords,
                    receptor_radii=request.receptor_radii,
                    ligand_radii=request.ligand_ctx.base_radii,
                    target_error=request.target_error,
                    epsilon=0.2,
                    softening_radius=request.softening_radius,
                )
            )
            chunk_correction_np += np.asarray(
                jax.device_get(posewise_flex_error),
                dtype=np.float32,
            )
            chunk_flex_delta_np = np.asarray(
                jax.device_get(coarse_phys_delta),
                dtype=np.float32,
            )
            if chunk_flex_delta_np.ndim == 0:
                chunk_correction_np += np.float32(2.0 * float(chunk_flex_delta_np))
            else:
                chunk_correction_np += np.float32(2.0) * chunk_flex_delta_np
        additive_correction_np[start:stop] = chunk_correction_np

    assert pruning_delta_budget is not None
    if extra_pruning_delta_budget is not None:
        pruning_delta_budget = _merge_pruning_delta_budgets(
            pruning_delta_budget,
            extra_pruning_delta_budget,
        )
    if use_rigid_improvement:
        assert rigid_refinement_plan is not None
        assert rigid_improvement_scoring_context is not None
        rigid_global_cap = float(rigid_refinement_plan.local_improvement_bound)
        additive_correction_np += np.float32(rigid_global_cap)

        best_coarse = (
            float(np.min(coarse_scores_np)) if coarse_scores_np.size > 0 else 0.0
        )
        admissible_cutoff = (
            best_coarse
            + float(pruning_delta_budget.total_delta)
            + additive_correction_np
        )
        rigid_tighten_mask = coarse_scores_np <= admissible_cutoff
        rigid_tighten_indices = np.flatnonzero(rigid_tighten_mask).astype(np.int32)

        rigid_pre_exact_count = int(rigid_tighten_indices.size)
        if rigid_tighten_indices.size > 0:
            phase_start = time.perf_counter()
            incumbent_prefilter_count = min(
                RIGID_EXACT_INCUMBENT_PREFILTER_TOP_K,
                int(rigid_tighten_indices.size),
            )
            incumbent_probe_indices = rigid_tighten_indices[
                np.argsort(coarse_scores_np[rigid_tighten_indices])[
                    :incumbent_prefilter_count
                ]
            ]
            incumbent_probe_scores = _score_rigid_exact_pose_batch(
                request,
                poses_coords=poses_coords[incumbent_probe_indices],
                electrostatics=electrostatics,
                scoring_context=rigid_improvement_scoring_context,
            )
            incumbent_exact_best = float(
                np.min(
                    np.asarray(jax.device_get(incumbent_probe_scores), dtype=np.float32)
                )
            )
            coarse_lower_bound = (
                coarse_scores_np[rigid_tighten_indices]
                - float(pruning_delta_budget.total_delta)
                - rigid_global_cap
            )
            prefilter_keep = coarse_lower_bound <= incumbent_exact_best
            if rigid_forced_prune_mask is not None:
                rigid_forced_prune_mask[rigid_tighten_indices[~prefilter_keep]] = True
            rigid_tighten_indices = rigid_tighten_indices[prefilter_keep]
            _runtime_profile_log(
                "certified_pruning_exact_incumbent_prefilter", phase_start
            )

        if rigid_tighten_indices.size > 0:
            phase_start = time.perf_counter()
            exact_candidates_np = np.empty(
                (int(rigid_tighten_indices.size),),
                dtype=np.float32,
            )
            exact_chunk_size = (
                int(rigid_tighten_indices.size)
                if int(rigid_tighten_indices.size)
                <= CERTIFIED_PRUNING_MONOLITHIC_THRESHOLD
                else CERTIFIED_PRUNING_CHUNK_SIZE
            )
            for s in range(0, int(rigid_tighten_indices.size), exact_chunk_size):
                e = min(s + exact_chunk_size, int(rigid_tighten_indices.size))
                idx_chunk = rigid_tighten_indices[s:e]
                exact_chunk = _score_rigid_exact_pose_batch(
                    request,
                    poses_coords=poses_coords[idx_chunk],
                    electrostatics=electrostatics,
                    scoring_context=rigid_improvement_scoring_context,
                )
                exact_candidates_np[s:e] = np.asarray(
                    jax.device_get(exact_chunk), dtype=np.float32
                )
            if exact_candidates_np.size > 0:
                best_exact = float(np.min(exact_candidates_np))
                keep_exact_mask = exact_candidates_np <= (best_exact + rigid_global_cap)
                if rigid_forced_prune_mask is not None:
                    rigid_forced_prune_mask[rigid_tighten_indices[~keep_exact_mask]] = (
                        True
                    )
                rigid_tighten_indices = rigid_tighten_indices[keep_exact_mask]
            else:
                best_exact = float("inf")
            _runtime_profile_log("certified_pruning_exact_rigid_scores", phase_start)
        else:
            best_exact = float("inf")

        if rigid_tighten_indices.size > 0:
            phase_start = time.perf_counter()
            rigid_chunk_size = (
                rigid_tighten_indices.size
                if rigid_tighten_indices.size <= CERTIFIED_PRUNING_MONOLITHIC_THRESHOLD
                else CERTIFIED_PRUNING_CHUNK_SIZE
            )
            for s in range(0, rigid_tighten_indices.size, rigid_chunk_size):
                e = min(s + rigid_chunk_size, rigid_tighten_indices.size)
                idx_chunk = rigid_tighten_indices[s:e]
                (
                    chunk_family,
                    chunk_rigid_correction_np,
                    chunk_clearance_safe_mask,
                    _chunk_posewise_clearance,
                ) = _posewise_rigid_improvement_budget_family(
                    request,
                    rigid_improvement_scoring_context,
                    poses_coords=poses_coords[idx_chunk],
                    base_translation_step=rigid_refinement_plan.base_translation_step,
                    base_rotation_step_rad=rigid_refinement_plan.base_rotation_step_rad,
                    ligand_radius=rigid_refinement_plan.ligand_radius,
                    n_rounds=rigid_refinement_plan.n_search_rounds,
                    pose_indices=idx_chunk,
                )
                rigid_improvement_budgets.extend(chunk_family.budgets)
                if rigid_clearance_safe_mask_full is not None:
                    rigid_clearance_safe_mask_full[idx_chunk] = (
                        chunk_clearance_safe_mask
                    )
                additive_correction_np[idx_chunk] += (
                    chunk_rigid_correction_np - np.float32(rigid_global_cap)
                )
            _runtime_profile_log("certified_pruning_rigid_posewise_bounds", phase_start)
        print(
            "[CERTIFIED PRUNING DEBUG] "
            f"rigid_tighten_candidates={int(rigid_tighten_indices.size)}/{n_total} "
            f"(pre_exact={rigid_pre_exact_count}, best_exact={best_exact:.3f}) "
            f"global_cap={rigid_global_cap:.3f}",
            flush=True,
        )

    survivor_mask, tau, lower_bounds = _canonical_retain_mask(
        jnp.asarray(coarse_scores_np),
        delta=pruning_delta_budget.total_delta,
        additive_correction=jnp.asarray(additive_correction_np),
    )
    survivor_mask_np = np.asarray(jax.device_get(survivor_mask), dtype=bool)
    lower_bounds_np = np.asarray(jax.device_get(lower_bounds), dtype=np.float32)
    rigid_clearance_safe_mask_np = rigid_clearance_safe_mask_full
    if rigid_clearance_safe_mask_np is not None:
        survivor_mask_np = np.logical_or(
            survivor_mask_np, ~rigid_clearance_safe_mask_np
        )
    if rigid_forced_prune_mask is not None:
        survivor_mask_np = np.logical_and(survivor_mask_np, ~rigid_forced_prune_mask)

    n_surv_val = int(np.count_nonzero(survivor_mask_np))
    efficiency = 100.0 * (1.0 - n_surv_val / n_total)
    print(
        f"[CERTIFIED PRUNING] Pruned {n_total} -> {n_surv_val} poses "
        f"({efficiency:.1f}% reduction, delta={pruning_delta_budget.total_delta:.3f} kcal/mol)"
    )
    if pruning_delta_budget.total_delta > 0.0:
        print(
            "[CERTIFIED PRUNING DEBUG] "
            f"delta_source={pruning_delta_budget.source} "
            f"delta_breakdown={[(item.label, round(float(item.value), 3)) for item in pruning_delta_budget.breakdown]}",
            flush=True,
        )
    if rigid_improvement_budgets:
        rigid_budget_totals_np = np.asarray(
            [budget.total_budget for budget in rigid_improvement_budgets],
            dtype=np.float64,
        )
        global_cap = (
            None
            if rigid_refinement_plan is None
            else float(rigid_refinement_plan.local_improvement_bound)
        )
        topk = min(5, rigid_budget_totals_np.size)
        top_idx = np.argsort(rigid_budget_totals_np)[-topk:][::-1]
        top_pose_indices = [
            int(cast(int, rigid_improvement_budgets[idx].pose_index))
            for idx in top_idx.tolist()
        ]
        top_bounds = [float(rigid_budget_totals_np[idx]) for idx in top_idx.tolist()]
        top_scores = [float(coarse_scores_np[idx]) for idx in top_pose_indices]
        print(
            "[CERTIFIED PRUNING DEBUG] "
            f"rigid_budget_p50={float(np.median(rigid_budget_totals_np)):.3f} "
            f"p90={float(np.quantile(rigid_budget_totals_np, 0.9)):.3f} "
            f"p99={float(np.quantile(rigid_budget_totals_np, 0.99)):.3f} "
            f"max={float(np.max(rigid_budget_totals_np)):.3f} "
            f"gt1e12={int(np.count_nonzero(rigid_budget_totals_np > 1.0e12))} "
            f"gt1e15={int(np.count_nonzero(rigid_budget_totals_np > 1.0e15))} "
            f"gt1e18={int(np.count_nonzero(rigid_budget_totals_np > 1.0e18))}"
            + (
                ""
                if global_cap is None
                else (
                    f" cap={global_cap:.3f} "
                    f"at_cap={int(np.count_nonzero(np.isclose(rigid_budget_totals_np, global_cap, rtol=1.0e-6, atol=1.0e-6)))}"
                )
            ),
            flush=True,
        )
        print(
            "[CERTIFIED PRUNING DEBUG] "
            f"top_pose_indices={top_pose_indices} "
            f"top_bounds={[round(val, 3) for val in top_bounds]} "
            f"top_coarse_scores={[round(val, 3) for val in top_scores]}",
            flush=True,
        )
        if coarse_scores_np.size > 0:
            best_coarse = float(np.min(coarse_scores_np))
            coarse_margin = coarse_scores_np - best_coarse
            correction_np = np.asarray(additive_correction_np, dtype=np.float64)
            print(
                "[CERTIFIED PRUNING DEBUG] "
                f"coarse_margin_p50={float(np.median(coarse_margin)):.3f} "
                f"p90={float(np.quantile(coarse_margin, 0.9)):.3f} "
                f"max={float(np.max(coarse_margin)):.3f} "
                f"corr_p50={float(np.median(correction_np)):.3f} "
                f"p90={float(np.quantile(correction_np, 0.9)):.3f} "
                f"max={float(np.max(correction_np)):.3f} "
                f"margin_minus_corr_p50={float(np.median(coarse_margin - correction_np)):.3f} "
                f"margin_minus_corr_max={float(np.max(coarse_margin - correction_np)):.3f}",
                flush=True,
            )
    active_subset_budget_family = (
        CertifiedActiveSubsetBudgetFamily(
            budgets=tuple(active_subset_budgets),
            theorem_handles=pose_specific_improvement_budget_theorem_handles(),
        )
        if use_conf
        else None
    )
    improvement_budget_family = (
        None
        if active_subset_budget_family is None
        else active_subset_budget_family.as_pose_specific_improvement_family()
    )
    if improvement_budget_family is None and rigid_improvement_budgets:
        improvement_budget_family = PoseSpecificImprovementBudgetFamily(
            budgets=tuple(rigid_improvement_budgets),
            theorem_handles=rigid_posewise_improvement_budget_theorem_handles(),
        )
    pruning_optimality_handles = (
        joint_pruning_budget_optimality_handles()
        if rigid_clearance_safe_mask_np is None
        or bool(np.all(rigid_clearance_safe_mask_np))
        else ()
    )
    return CertifiedPipelineExecutionPlan(
        coarse_scores=jnp.asarray(coarse_scores_np),
        lower_bounds=jnp.asarray(lower_bounds_np),
        tau=float(np.asarray(jax.device_get(tau))),
        retain_mask=jnp.asarray(survivor_mask_np),
        pruning_delta_budget=pruning_delta_budget,
        active_subset_budget_family=active_subset_budget_family,
        improvement_budget_family=improvement_budget_family,
        theorem_handles=_merge_theorem_handles(
            ("TK11", "BD5"),
            pruning_optimality_handles,
            pruning_delta_budget.theorem_handles,
            ()
            if improvement_budget_family is None
            else improvement_budget_family.theorem_handles,
        ),
    )


def _request_uses_conformer_search(request: "PipelineDockingRequest") -> bool:
    """Check if conformer search is enabled for this request.

    Requires: config with ConformerSearchMode.ENABLED AND ligand adjacency.
    """
    return (
        request.config is not None
        and request.config.conformer_search == ConformerSearchMode.ENABLED
        and request.ligand_ctx.adjacency is not None
    )


def _request_rotatable_bonds(
    request: "PipelineDockingRequest",
) -> tuple[RotatableBond, ...]:
    adjacency = request.ligand_ctx.adjacency
    elements = request.ligand_ctx.elements
    if adjacency is None or elements is None:
        return ()
    coords_np = np.asarray(request.ligand_ctx.base_coords, dtype=np.float32)
    return detect_rotatable_bonds(adjacency, coords_np, elements)


def _build_conformer_search_config(
    request: "PipelineDockingRequest",
    *,
    rotatable_bonds: tuple[RotatableBond, ...],
) -> BranchAndBoundConfig:
    coverage_plan = request.conformer_coverage_plan
    if coverage_plan is not None and coverage_plan.n_torsions != len(rotatable_bonds):
        print(
            "[CONFORMER COVERAGE] Re-deriving stale coverage plan: "
            f"stored n_torsions={coverage_plan.n_torsions}, "
            f"runtime rotatable_bonds={len(rotatable_bonds)}",
            flush=True,
        )
        coverage_plan = None
    if coverage_plan is None:
        coverage_plan = _derive_conformer_coverage_plan(
            request,
            rotatable_bonds=rotatable_bonds,
        )
    return cast(
        BranchAndBoundConfig,
        coverage_plan.as_branch_and_bound_config(
            reuse_initial_conformer=request.reuse_initial_conformer,
            max_conformers=1,
        ),
    )


def _restrict_conformer_search_config(
    request: "PipelineDockingRequest",
    config: BranchAndBoundConfig,
    active_mask: np.ndarray,
    active_bonds: tuple[RotatableBond, ...],
) -> BranchAndBoundConfig:
    if config.per_bond_lipschitz is None:
        return config
    if len(config.per_bond_lipschitz) != int(active_mask.shape[0]):
        raise ValueError(
            "conformer config per_bond_lipschitz must match the full rotatable-bond set "
            f"(got {len(config.per_bond_lipschitz)} vs {int(active_mask.shape[0])})"
        )
    restricted_lipschitz = tuple(
        float(value)
        for value, active in zip(config.per_bond_lipschitz, active_mask.tolist())
        if active
    )
    if not active_bonds:
        return BranchAndBoundConfig(
            max_cells=1,
            min_cell_radius=float(2.0 * np.pi),
            score_lipschitz_constant=config.score_lipschitz_constant,
            max_conformers=config.max_conformers,
            per_bond_lipschitz=(),
            reuse_initial_conformer=config.reuse_initial_conformer,
        )
    restricted_plan = _derive_conformer_coverage_plan_from_lipschitz(
        score_lipschitz_constant=config.score_lipschitz_constant,
        per_bond_lipschitz=restricted_lipschitz,
        target_delta=request.target_error,
        target_rmsd=request.target_rmsd,
        max_arm=max(bond.max_arm_length for bond in active_bonds),
    )
    restricted_config = cast(
        BranchAndBoundConfig,
        restricted_plan.as_branch_and_bound_config(
            reuse_initial_conformer=config.reuse_initial_conformer,
            max_conformers=config.max_conformers,
        ),
    )
    print(
        "[CONFORMER COVERAGE] Restricted active-bond config: "
        f"torsions {len(config.per_bond_lipschitz)} -> {len(restricted_lipschitz)}, "
        f"max_cells {config.max_cells} -> {restricted_config.max_cells}",
        flush=True,
    )
    return restricted_config


def _conformer_runtime_theorem_handles(
    *,
    has_rotatable_bonds: bool,
    include_pocket_handles: bool,
    include_receptor_flex_handles: bool,
) -> tuple[str, ...]:
    return _merge_theorem_handles(
        (
            "CS2",
            "CS5",
            "CS6",
            "CS8",
            "CS9",
            "CS10",
            "CS12",
            "CS13",
            "CS14",
            "GAP2",
        )
        if has_rotatable_bonds
        else (),
        (
            "CSC14",
            "CSC15",
            "CSC16",
            "CSC29",
            "CSC30",
            "CSC31",
            "CSC32",
            "CSC33",
            "CSC34",
            "CSC35",
            "CSC36",
            "CSC44",
            "CSC50",
            "BCRC1",
            "BCRC2",
            "BCRC3",
        )
        if has_rotatable_bonds
        else (),
        branch_and_bound_cross_docking_handles() if has_rotatable_bonds else (),
        pocket_cross_docking_handles() if include_pocket_handles else (),
        receptor_flexibility_theorem_handles() if include_receptor_flex_handles else (),
        receptor_flex_cross_docking_handles() if include_receptor_flex_handles else (),
    )


def _conformer_coverage_plan_matches_rotatable_bonds(
    plan: CertifiedConformerCoveragePlan | None,
    rotatable_bonds: tuple[RotatableBond, ...],
) -> bool:
    if plan is None:
        return False
    return plan.n_torsions == len(rotatable_bonds)


def _per_bond_lipschitz_improvement_bounds(
    per_bond_lipschitz: tuple[float, ...] | None,
) -> jnp.ndarray | None:
    if per_bond_lipschitz is None:
        return None
    if len(per_bond_lipschitz) == 0:
        return jnp.zeros((0,), dtype=jnp.float32)
    return (2.0 * math.pi) * jnp.asarray(per_bond_lipschitz, dtype=jnp.float32)


def _conformer_local_improvement_bounds(
    request: "PipelineDockingRequest",
    rotatable_bonds: tuple[RotatableBond, ...],
    per_bond_lipschitz: tuple[float, ...] | None,
) -> jnp.ndarray | None:
    if not rotatable_bonds:
        return jnp.zeros((0,), dtype=jnp.float32)

    interaction_bounds = _per_bond_lipschitz_improvement_bounds(per_bond_lipschitz)
    mmff_headroom = derive_mmff_torsion_current_headroom(
        request.ligand_source_path,
        request.ligand_ctx.elements,
        rotatable_bonds,
    )
    if mmff_headroom is not None:
        if interaction_bounds is None:
            return jnp.asarray(mmff_headroom, dtype=jnp.float32)
        return jnp.asarray(mmff_headroom, dtype=jnp.float32) + jnp.asarray(
            interaction_bounds,
            dtype=jnp.float32,
        )

    physical_barriers = derive_uff_torsion_barrier_heights(
        request.ligand_source_path,
        request.ligand_ctx.elements,
        rotatable_bonds,
    )
    if physical_barriers is not None:
        base_bounds = 2.0 * jnp.asarray(physical_barriers, dtype=jnp.float32)
        if interaction_bounds is None:
            return base_bounds
        return base_bounds + jnp.asarray(interaction_bounds, dtype=jnp.float32)

    return interaction_bounds


def _posewise_conformer_interaction_bounds(
    request: "PipelineDockingRequest",
    poses_coords: jnp.ndarray,
    rotatable_bonds: tuple[RotatableBond, ...],
    global_per_bond_bounds: jnp.ndarray | None,
    electrostatics: CertifiedRealSpaceEwaldSpec | None,
) -> np.ndarray | None:
    from dq_dock_engine.docking.scoring import _EPSILON_KCAL_MOL

    if global_per_bond_bounds is None or not rotatable_bonds:
        return None
    receptor_coords = np.asarray(
        jax.device_get(request.protein_coords), dtype=np.float32
    )
    receptor_radii = np.asarray(
        jax.device_get(request.receptor_radii), dtype=np.float32
    )
    ligand_radii = np.asarray(
        jax.device_get(request.ligand_ctx.base_radii), dtype=np.float32
    )
    poses_np = np.asarray(jax.device_get(poses_coords), dtype=np.float32)
    global_bounds_np = np.asarray(
        jax.device_get(global_per_bond_bounds), dtype=np.float32
    )
    posewise_bounds = np.tile(global_bounds_np[None, :], (poses_np.shape[0], 1))
    cutoff = float(compute_certified_cutoff(request.target_error))
    epsilon_pair = float(_EPSILON_KCAL_MOL / 4.0)
    receptor_charges = None
    ligand_charges = None
    electro_cutoff = None
    electro_alpha = None
    electro_dielectric = None
    if electrostatics is not None:
        receptor_charges = np.asarray(
            jax.device_get(electrostatics.receptor_charges), dtype=np.float32
        )
        ligand_charges = np.asarray(
            jax.device_get(electrostatics.ligand_charges), dtype=np.float32
        )
        electro_cutoff = float(electrostatics.cutoff)
        electro_alpha = float(electrostatics.alpha)
        electro_dielectric = float(electrostatics.dielectric)

    for pose_idx in range(poses_np.shape[0]):
        pose_coords = poses_np[pose_idx]
        for bond_idx, bond in enumerate(rotatable_bonds):
            if not bond.rotating_atom_indices:
                continue
            rotating_idx = np.asarray(bond.rotating_atom_indices, dtype=np.int32)
            rotating_coords = pose_coords[rotating_idx]
            dists = np.linalg.norm(
                rotating_coords[:, None, :] - receptor_coords[None, :, :],
                axis=-1,
            )
            sigma = ligand_radii[rotating_idx, None] + receptor_radii[None, :]
            arm_lengths = (
                np.asarray(bond.rotating_atom_arm_lengths, dtype=np.float32)
                if bond.rotating_atom_arm_lengths
                else np.full(
                    (rotating_coords.shape[0],),
                    float(bond.max_arm_length),
                    dtype=np.float32,
                )
            )
            max_displacement = 2.0 * arm_lengths[:, None]
            cutoff_safe = np.maximum(cutoff, sigma)
            current_in_range = dists < cutoff_safe
            current_pair_energy = np.where(
                current_in_range,
                _exact_lj_pair_score_with_floor(
                    epsilon_pair=epsilon_pair,
                    sigma=sigma,
                    distance=dists,
                ),
                0.0,
            )
            reachable_dists = dists - max_displacement
            lower_bound_pair = _exact_lj_interval_lower_bound(
                epsilon_pair=epsilon_pair,
                sigma=sigma,
                current_distance=dists,
                max_displacement=max_displacement,
                cutoff_safe=cutoff_safe,
            )
            local_lj_bound = np.float32(
                np.sum(np.maximum(0.0, current_pair_energy - lower_bound_pair))
            )
            local_bound = float(local_lj_bound)

            if (
                receptor_charges is not None
                and ligand_charges is not None
                and electro_alpha is not None
                and electro_cutoff is not None
                and electro_dielectric is not None
            ):
                charge_products = (
                    receptor_charges[None, :] * ligand_charges[rotating_idx][:, None]
                ) / electro_dielectric
                electro_current_in_range = dists < electro_cutoff
                safe_electro_dists = np.where(
                    electro_current_in_range,
                    dists,
                    electro_cutoff,
                )
                electro_current = np.where(
                    electro_current_in_range,
                    charge_products
                    * scipy_special.erfc(electro_alpha * safe_electro_dists)
                    / safe_electro_dists,
                    0.0,
                )
                electro_lower = np.where(
                    reachable_dists > electro_cutoff,
                    0.0,
                    np.where(
                        charge_products < 0.0,
                        charge_products
                        * scipy_special.erfc(
                            electro_alpha * np.maximum(reachable_dists, 1.0e-6)
                        )
                        / np.maximum(reachable_dists, 1.0e-6),
                        charge_products
                        * scipy_special.erfc(
                            electro_alpha * np.maximum(dists + max_displacement, 1.0e-6)
                        )
                        / np.maximum(dists + max_displacement, 1.0e-6),
                    ),
                )
                local_bound += float(
                    np.sum(np.maximum(0.0, electro_current - electro_lower))
                )

            posewise_bounds[pose_idx, bond_idx] = min(
                posewise_bounds[pose_idx, bond_idx],
                np.float32(local_bound),
            )
    return posewise_bounds


def _posewise_atomwise_conformer_budget_totals(
    request: "PipelineDockingRequest",
    poses_coords: jnp.ndarray,
    rotatable_bonds: tuple[RotatableBond, ...],
    per_bond_local_bounds: jnp.ndarray | None,
    electrostatics: CertifiedRealSpaceEwaldSpec | None,
) -> np.ndarray | None:
    from dq_dock_engine.docking.scoring import _EPSILON_KCAL_MOL

    if per_bond_local_bounds is None or not rotatable_bonds:
        return None
    poses_np = np.asarray(jax.device_get(poses_coords), dtype=np.float32)
    receptor_coords = np.asarray(
        jax.device_get(request.protein_coords), dtype=np.float32
    )
    receptor_radii = np.asarray(
        jax.device_get(request.receptor_radii), dtype=np.float32
    )
    ligand_radii = np.asarray(
        jax.device_get(request.ligand_ctx.base_radii), dtype=np.float32
    )
    cutoff = float(compute_certified_cutoff(request.target_error))
    epsilon_pair = float(_EPSILON_KCAL_MOL / 4.0)
    active_masks = _posewise_active_torsion_mask_np(
        poses_coords,
        request.protein_coords,
        rotatable_bonds,
        scoring_cutoff=cutoff,
    )
    bond_bounds_np = np.asarray(jax.device_get(per_bond_local_bounds), dtype=np.float32)
    if bond_bounds_np.ndim == 1:
        per_pose_bond_bounds = np.tile(
            bond_bounds_np[None, :],
            (poses_np.shape[0], 1),
        )
    else:
        per_pose_bond_bounds = bond_bounds_np

    receptor_charges = ligand_charges = None
    electro_cutoff = electro_alpha = electro_dielectric = None
    if electrostatics is not None:
        receptor_charges = np.asarray(
            jax.device_get(electrostatics.receptor_charges), dtype=np.float32
        )
        ligand_charges = np.asarray(
            jax.device_get(electrostatics.ligand_charges), dtype=np.float32
        )
        electro_cutoff = float(electrostatics.cutoff)
        electro_alpha = float(electrostatics.alpha)
        electro_dielectric = float(electrostatics.dielectric)

    totals = np.zeros((poses_np.shape[0],), dtype=np.float32)
    for pose_idx, pose_coords in enumerate(poses_np):
        active_mask = active_masks[pose_idx]
        totals[pose_idx] += float(
            np.sum(per_pose_bond_bounds[pose_idx][np.asarray(active_mask, dtype=bool)])
        )

        atom_displacement = np.zeros((pose_coords.shape[0],), dtype=np.float32)
        moving_atom_mask = np.zeros((pose_coords.shape[0],), dtype=bool)
        for is_active, bond in zip(active_mask.tolist(), rotatable_bonds):
            if not is_active:
                continue
            atom_idx = np.asarray(bond.rotating_atom_indices, dtype=np.int32)
            arm_lengths = (
                np.asarray(bond.rotating_atom_arm_lengths, dtype=np.float32)
                if bond.rotating_atom_arm_lengths
                else np.full(
                    (len(atom_idx),), float(bond.max_arm_length), dtype=np.float32
                )
            )
            atom_displacement[atom_idx] += 2.0 * arm_lengths
            moving_atom_mask[atom_idx] = True

        moving_indices = np.flatnonzero(moving_atom_mask)
        for atom_idx in moving_indices.tolist():
            atom_coord = pose_coords[atom_idx]
            dists = np.linalg.norm(receptor_coords - atom_coord[None, :], axis=-1)
            sigma = ligand_radii[atom_idx] + receptor_radii
            displacement = atom_displacement[atom_idx]

            cutoff_safe = np.maximum(cutoff, sigma)
            current_in_range = dists < cutoff_safe
            current_lj = np.where(
                current_in_range,
                _exact_lj_pair_score_with_floor(
                    epsilon_pair=epsilon_pair,
                    sigma=sigma,
                    distance=dists,
                ),
                0.0,
            )
            reachable_dists = dists - displacement
            lower_lj = _exact_lj_interval_lower_bound(
                epsilon_pair=epsilon_pair,
                sigma=sigma,
                current_distance=dists,
                max_displacement=np.full_like(dists, displacement),
                cutoff_safe=cutoff_safe,
            )
            atom_bound = np.sum(np.maximum(0.0, current_lj - lower_lj))

            if (
                receptor_charges is not None
                and ligand_charges is not None
                and electro_cutoff is not None
                and electro_alpha is not None
                and electro_dielectric is not None
            ):
                charge_products = (
                    receptor_charges * ligand_charges[atom_idx]
                ) / electro_dielectric
                electro_current_in_range = dists < electro_cutoff
                safe_electro_dists = np.where(
                    electro_current_in_range,
                    dists,
                    electro_cutoff,
                )
                electro_current = np.where(
                    electro_current_in_range,
                    charge_products
                    * scipy_special.erfc(electro_alpha * safe_electro_dists)
                    / safe_electro_dists,
                    0.0,
                )
                electro_lower = np.where(
                    reachable_dists > electro_cutoff,
                    0.0,
                    np.where(
                        charge_products < 0.0,
                        charge_products
                        * scipy_special.erfc(
                            electro_alpha * np.maximum(reachable_dists, 1.0e-6)
                        )
                        / np.maximum(reachable_dists, 1.0e-6),
                        charge_products
                        * scipy_special.erfc(
                            electro_alpha * np.maximum(dists + displacement, 1.0e-6)
                        )
                        / np.maximum(dists + displacement, 1.0e-6),
                    ),
                )
                atom_bound += np.sum(np.maximum(0.0, electro_current - electro_lower))

            totals[pose_idx] += float(atom_bound)
    return totals


def _conformer_improvement_bound(per_bond_bounds: jnp.ndarray | None) -> float:
    if per_bond_bounds is None:
        return 0.0
    return float(jnp.sum(jnp.asarray(per_bond_bounds, dtype=jnp.float32)))


def _canonical_retain_mask(
    reference_scores: jnp.ndarray,
    *,
    delta: float,
    additive_correction: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Retain poses whose certified lower bound is below the coarse incumbent.

    The authoritative lower bound is:

      coarse_score - delta - additive_correction

    where `delta` is the certified coarse approximation slack and
    `additive_correction` aggregates pose-specific certified omission /
    improvement budgets. If those pruning ingredients are unavailable or
    non-finite, the theorem-honest fallback is to retain every pose.

    Returns (mask, tau, lower_bounds).
    """
    if reference_scores.ndim != 1:
        raise ValueError("reference_scores must be 1D")
    if additive_correction.shape != reference_scores.shape:
        raise ValueError("additive_correction must match reference_scores shape")
    if reference_scores.shape[0] == 0:
        return (
            jnp.zeros_like(reference_scores, dtype=bool),
            jnp.array(jnp.inf, dtype=reference_scores.dtype),
            jnp.full_like(reference_scores, jnp.inf),
        )
    if not np.isfinite(float(delta)):
        tau = jnp.max(reference_scores)
        return jnp.ones_like(reference_scores, dtype=bool), tau, reference_scores
    additive_correction_np = np.asarray(
        jax.device_get(additive_correction), dtype=np.float32
    )
    if not np.all(np.isfinite(additive_correction_np)):
        tau = jnp.max(reference_scores)
        return jnp.ones_like(reference_scores, dtype=bool), tau, reference_scores
    best_score = jnp.min(reference_scores)
    tau = best_score
    lower_bounds = (
        reference_scores
        - jnp.asarray(delta, dtype=reference_scores.dtype)
        - additive_correction
    )
    return lower_bounds <= tau, tau, lower_bounds


def _retain_mask_for_explicit_threshold(
    reference_scores: jnp.ndarray,
    *,
    threshold: float,
    additive_correction: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    if reference_scores.ndim != 1:
        raise ValueError("reference_scores must be 1D")
    if additive_correction.shape != reference_scores.shape:
        raise ValueError("additive_correction must match reference_scores shape")
    if not math.isfinite(float(threshold)):
        return jnp.ones_like(reference_scores, dtype=bool), reference_scores
    additive_correction_np = np.asarray(
        jax.device_get(additive_correction), dtype=np.float32
    )
    if not np.all(np.isfinite(additive_correction_np)):
        return jnp.ones_like(reference_scores, dtype=bool), reference_scores
    lower_bounds = reference_scores - additive_correction
    return lower_bounds <= jnp.asarray(
        threshold, dtype=reference_scores.dtype
    ), lower_bounds


def _score_softened_pose_batch(
    request: "PipelineDockingRequest",
    *,
    poses_coords: jnp.ndarray,
    electrostatics: CertifiedRealSpaceEwaldSpec | None,
    scoring_context: CertifiedScoringContext | None,
) -> tuple[jnp.ndarray, CertifiedPruningDeltaBudget, jnp.ndarray]:
    softening_radius = request.softening_radius
    if scoring_context is not None:
        coarse_batch = scoring_context.score_softened_batch(
            receptor_coords=request.protein_coords,
            poses_coords=poses_coords,
            receptor_radii=request.receptor_radii,
            ligand_radii=request.ligand_ctx.base_radii,
            target_error=request.target_error,
            epsilon=0.2,
            softening_radius=softening_radius,
        )
        delta_budget = scoring_context.pruning_delta_budget(
            softening_error_bound=coarse_batch.softening_error_bound,
            softening_radius=softening_radius,
            use_posewise_softening=True,
        )
        return (
            coarse_batch.scores,
            delta_budget,
            (
                jnp.zeros_like(coarse_batch.scores)
                if scoring_context.uses_extended_rich
                else cast(jnp.ndarray, coarse_batch.posewise_softening_error_bound)
            ),
        )

    if electrostatics is not None:
        coarse_batch = score_certified_softened_lj_realspace_ewald(
            receptor_coords=request.receptor_coords,
            poses_coords=poses_coords,
            receptor_radii=request.receptor_radii,
            ligand_radii=request.ligand_radii,
            electrostatics=electrostatics,
            target_error=request.target_error,
            compute_error_bound=True,
            softening_radius=softening_radius,
        )
    else:
        coarse_batch = score_certified_softened_lj(
            receptor_coords=request.receptor_coords,
            poses_coords=poses_coords,
            receptor_radii=request.receptor_radii,
            ligand_radii=request.ligand_radii,
            target_error=request.target_error,
            compute_error_bound=True,
            softening_radius=softening_radius,
        )
    delta_budget = CertifiedScoringContext(
        exact_chemistry_mode=ExactChemistryMode.NONE,
        electrostatics=electrostatics,
    ).pruning_delta_budget(
        softening_error_bound=coarse_batch.softening_error_bound,
        use_posewise_softening=True,
    )
    return (
        coarse_batch.scores,
        delta_budget,
        cast(jnp.ndarray, coarse_batch.posewise_softening_error_bound),
    )


def _build_conformer_score_fns(
    request: "PipelineDockingRequest",
    quaternion: jnp.ndarray,
    translation: jnp.ndarray,
    scoring_context: CertifiedScoringContext | None,
    electrostatics: "CertifiedRealSpaceEwaldSpec | None",
    receptor_coords: jnp.ndarray | None = None,
    receptor_radii: jnp.ndarray | None = None,
) -> tuple[
    Callable[[jnp.ndarray], float],
    Callable[[jnp.ndarray], jnp.ndarray] | None,
]:
    """Build scoring closures for conformer search at a given rigid pose.

    Returns (score_fn, score_fn_batch):
      - score_fn: (N_atoms, 3) → float  (single-pose fallback)
      - score_fn_batch: (PAD, N_atoms, 3) → (PAD,) scores  (fixed-size batched path)
        Caller MUST always pass the same padded shape to avoid JIT recompilation.
    """

    def score_fn(candidate_coords: jnp.ndarray) -> float:
        transformed = rigid_transform_3d(candidate_coords, quaternion, translation)
        posed = jnp.expand_dims(jnp.asarray(transformed), 0)
        if scoring_context is not None:
            scores = scoring_context.score_rigid_exact_batch(
                receptor_coords=(
                    request.protein_coords
                    if receptor_coords is None
                    else receptor_coords
                ),
                poses_coords=posed,
                receptor_radii=(
                    request.receptor_radii if receptor_radii is None else receptor_radii
                ),
                ligand_radii=request.ligand_ctx.base_radii,
                target_error=request.target_error,
                epsilon=0.2,
            ).scores
        else:
            scores = route_scoring(
                **derive_route_scoring_kwargs(
                    request,
                    engine=request.effective_engine,
                    poses_coords=posed,
                    electrostatics=electrostatics,
                )
            )
        return float(np.asarray(scores[0]))

    score_fn_batch: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    if scoring_context is not None:

        def _score_batch(coords_batch: jnp.ndarray) -> jnp.ndarray:
            """Score a fixed-size padded batch: (PAD, N_atoms, 3) → (PAD,)."""
            posed = jax.vmap(lambda c: rigid_transform_3d(c, quaternion, translation))(
                coords_batch
            )
            return scoring_context.score_rigid_exact_batch(
                receptor_coords=(
                    request.protein_coords
                    if receptor_coords is None
                    else receptor_coords
                ),
                poses_coords=posed,
                receptor_radii=(
                    request.receptor_radii if receptor_radii is None else receptor_radii
                ),
                ligand_radii=request.ligand_ctx.base_radii,
                target_error=request.target_error,
                epsilon=0.2,
            ).scores

        score_fn_batch = _score_batch

    return score_fn, score_fn_batch


def _build_conformer_coarse_score_fns(
    request: "PipelineDockingRequest",
    quaternion: jnp.ndarray,
    translation: jnp.ndarray,
    scoring_context: CertifiedScoringContext | None,
    electrostatics: "CertifiedRealSpaceEwaldSpec | None",
    receptor_coords: jnp.ndarray | None = None,
    receptor_radii: jnp.ndarray | None = None,
) -> tuple[
    Callable[[jnp.ndarray], float],
    Callable[[jnp.ndarray], jnp.ndarray] | None,
]:
    def score_fn(candidate_coords: jnp.ndarray) -> float:
        transformed = rigid_transform_3d(candidate_coords, quaternion, translation)
        posed = jnp.expand_dims(jnp.asarray(transformed), 0)
        if scoring_context is not None:
            scores = scoring_context.score_softened_batch(
                receptor_coords=(
                    request.protein_coords
                    if receptor_coords is None
                    else receptor_coords
                ),
                poses_coords=posed,
                receptor_radii=(
                    request.receptor_radii if receptor_radii is None else receptor_radii
                ),
                ligand_radii=request.ligand_ctx.base_radii,
                target_error=request.target_error,
                epsilon=0.2,
                softening_radius=request.softening_radius,
            ).scores
        else:
            scores = route_scoring(
                **derive_route_scoring_kwargs(
                    request,
                    engine=request.effective_engine,
                    poses_coords=posed,
                    electrostatics=electrostatics,
                )
            )
        return float(np.asarray(scores[0]))

    score_fn_batch: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    if scoring_context is not None:

        def _score_batch(coords_batch: jnp.ndarray) -> jnp.ndarray:
            posed = jax.vmap(lambda c: rigid_transform_3d(c, quaternion, translation))(
                coords_batch
            )
            return scoring_context.score_softened_batch(
                receptor_coords=(
                    request.protein_coords
                    if receptor_coords is None
                    else receptor_coords
                ),
                poses_coords=posed,
                receptor_radii=(
                    request.receptor_radii if receptor_radii is None else receptor_radii
                ),
                ligand_radii=request.ligand_ctx.base_radii,
                target_error=request.target_error,
                epsilon=0.2,
                softening_radius=request.softening_radius,
            ).scores

        score_fn_batch = _score_batch

    return score_fn, score_fn_batch


@dataclass(frozen=True)
class _ConformerCellOmissionPayload:
    retained_indices: jnp.ndarray


def _build_conformer_cell_bound_state_fns(
    *,
    request: "PipelineDockingRequest",
    quaternion: jnp.ndarray,
    translation: jnp.ndarray,
    scoring_context: CertifiedScoringContext | None,
    electrostatics: "CertifiedRealSpaceEwaldSpec | None",
    rotatable_bonds: tuple[RotatableBond, ...],
    exact_subset_indices: jnp.ndarray,
    omission_budget: float,
    local_activity_mask_fn: Callable[[TorsionCell, np.ndarray], np.ndarray] | None,
) -> tuple[
    Callable[[TorsionCell, np.ndarray], CertifiedCellLowerBoundState | None] | None,
    Callable[
        [CertifiedCellLowerBoundState, TorsionCell],
        CertifiedCellLowerBoundState | None,
    ]
    | None,
]:
    if (
        scoring_context is None
        or not rotatable_bonds
        or omission_budget <= 0.0
        or int(exact_subset_indices.shape[0]) == 0
    ):
        return None, None

    coords_np = np.asarray(
        jax.device_get(request.ligand_ctx.base_coords), dtype=np.float32
    )
    kinematics = build_torsion_kinematics(
        rotatable_bonds,
        coords_np,
        request.ligand_ctx.base_coords.shape[0],
    )
    full_subset_tuple = tuple(
        int(i) for i in np.asarray(exact_subset_indices, dtype=np.int32).tolist()
    )
    scorer_cache: dict[
        tuple[int, ...],
        tuple[
            Callable[[jnp.ndarray], float], Callable[[jnp.ndarray], jnp.ndarray] | None
        ],
    ] = {}

    def _get_subset_scorers(
        retained_indices: jnp.ndarray,
    ) -> tuple[
        Callable[[jnp.ndarray], float], Callable[[jnp.ndarray], jnp.ndarray] | None
    ]:
        subset_tuple = tuple(
            int(i) for i in np.asarray(retained_indices, dtype=np.int32).tolist()
        )
        cached = scorer_cache.get(subset_tuple)
        if cached is not None:
            return cached
        local_context = scoring_context.receptor_subset(retained_indices)
        local_electrostatics = (
            None
            if electrostatics is None
            else electrostatics.receptor_subset(retained_indices)
        )
        built = _build_conformer_score_fns(
            request,
            quaternion,
            translation,
            local_context,
            local_electrostatics,
            receptor_coords=request.protein_coords[retained_indices],
            receptor_radii=request.receptor_radii[retained_indices],
        )
        scorer_cache[subset_tuple] = built
        return built

    def _active_mask_for_cell(
        cell: TorsionCell,
        center_coords_local: np.ndarray,
    ) -> np.ndarray:
        if local_activity_mask_fn is None:
            return np.ones((len(rotatable_bonds),), dtype=bool)
        return np.asarray(local_activity_mask_fn(cell, center_coords_local), dtype=bool)

    def _build_state(
        *,
        cell: TorsionCell,
        center_coords_local: np.ndarray,
        parent_retained_indices: jnp.ndarray,
        parent_omitted_bound: float,
    ) -> CertifiedCellLowerBoundState:
        active_mask = _active_mask_for_cell(cell, center_coords_local)
        active_bonds = tuple(
            bond
            for is_active, bond in zip(
                active_mask.tolist(), rotatable_bonds, strict=False
            )
            if is_active
        )
        actual_subset = parent_retained_indices
        additional_omitted_bound = 0.0
        if active_bonds and int(parent_retained_indices.shape[0]) > 1:
            center_world_coords = rigid_transform_3d(
                jnp.asarray(
                    center_coords_local, dtype=request.ligand_ctx.base_coords.dtype
                ),
                quaternion,
                translation,
            )
            omission_bounds = _per_receptor_atom_conformer_omission_bounds(
                ligand_world_coords=center_world_coords,
                ligand_radii=request.ligand_ctx.base_radii,
                receptor_coords=request.protein_coords[parent_retained_indices],
                receptor_radii=request.receptor_radii[parent_retained_indices],
                rotatable_bonds=active_bonds,
                ligand_charges=(
                    None
                    if request.ligand_ctx.charges is None
                    else request.ligand_ctx.charges
                ),
                receptor_charges=(
                    None
                    if electrostatics is None
                    else electrostatics.receptor_charges[parent_retained_indices]
                ),
                electro_alpha=(
                    None if electrostatics is None else electrostatics.alpha
                ),
                electro_dielectric=(
                    None if electrostatics is None else electrostatics.dielectric
                ),
            )
            local_subset, additional_omitted_bound = (
                _select_receptor_subset_by_omission_budget(
                    omission_bounds,
                    omission_budget=omission_budget,
                )
            )
            actual_subset = parent_retained_indices[local_subset]

        total_omitted_bound = float(parent_omitted_bound + additional_omitted_bound)
        local_score_fn, local_score_fn_batch = _get_subset_scorers(actual_subset)
        actual_subset_tuple = tuple(
            int(i) for i in np.asarray(actual_subset, dtype=np.int32).tolist()
        )
        uses_omission = (
            total_omitted_bound > 0.0 or actual_subset_tuple != full_subset_tuple
        )
        return CertifiedCellLowerBoundState(
            omitted_energy_bound=total_omitted_bound,
            theorem_handles=(
                omitted_channel_bound_theorem_handles() if uses_omission else ()
            ),
            score_fn=local_score_fn,
            score_fn_batch=local_score_fn_batch,
            active_mask=active_mask,
            score_is_exact=not uses_omission,
            payload=_ConformerCellOmissionPayload(retained_indices=actual_subset),
        )

    def _root_state_fn(
        cell: TorsionCell,
        center_coords_local: np.ndarray,
    ) -> CertifiedCellLowerBoundState:
        return _build_state(
            cell=cell,
            center_coords_local=np.asarray(center_coords_local, dtype=np.float32),
            parent_retained_indices=exact_subset_indices,
            parent_omitted_bound=0.0,
        )

    def _child_state_fn(
        parent_state: CertifiedCellLowerBoundState,
        child_cell: TorsionCell,
    ) -> CertifiedCellLowerBoundState:
        payload = cast(_ConformerCellOmissionPayload | None, parent_state.payload)
        parent_retained_indices = (
            exact_subset_indices if payload is None else payload.retained_indices
        )
        child_center_coords = np.asarray(
            jax.device_get(
                kinematics.forward(
                    request.ligand_ctx.base_coords,
                    child_cell.center(),
                )
            ),
            dtype=np.float32,
        )
        return _build_state(
            cell=child_cell,
            center_coords_local=child_center_coords,
            parent_retained_indices=parent_retained_indices,
            parent_omitted_bound=float(parent_state.omitted_energy_bound),
        )

    return _root_state_fn, _child_state_fn


def _run_conformer_search_for_pose(
    request: "PipelineDockingRequest",
    quaternion: jnp.ndarray,
    translation: jnp.ndarray,
    scoring_context: CertifiedScoringContext | None,
    electrostatics: "CertifiedRealSpaceEwaldSpec | None",
    *,
    rotatable_bonds: tuple[RotatableBond, ...] | None = None,
    conformer_config: BranchAndBoundConfig | None = None,
    strain_params: TorsionStrainParams | None = None,
    pruning_incumbent_energy: float | None = None,
    omission_budget: float | None = None,
    scan_only: bool = False,
) -> tuple[jnp.ndarray, float, tuple[str, ...]] | None:
    """Run conformer search for a single posed ligand.

    Returns (best_conformer_coords_in_world, energy, theorem_handles) if a
    conformer improves over the rigid baseline, else None.
    Fails loud — no silent fallback.
    """
    adjacency = request.ligand_ctx.adjacency
    if _runtime_profile_enabled():
        print(
            f"[RUNTIME PROFILE] conformer_pose_entry: scan_only={scan_only}, incumbent={pruning_incumbent_energy}",
            flush=True,
        )
    if adjacency is None:
        return None
    adjacency = cast(tuple[tuple[int, ...], ...], adjacency)

    bonds = (
        _request_rotatable_bonds(request)
        if rotatable_bonds is None
        else rotatable_bonds
    )
    active_mask = np.zeros((len(bonds),), dtype=bool)
    if bonds:
        base_world_coords = rigid_transform_3d(
            request.ligand_ctx.base_coords,
            quaternion,
            translation,
        )
        bonds, active_mask = _active_rotatable_bonds_for_pose(
            base_world_coords,
            request.protein_coords,
            bonds,
            scoring_cutoff=compute_certified_cutoff(request.target_error),
        )
    else:
        base_world_coords = rigid_transform_3d(
            request.ligand_ctx.base_coords,
            quaternion,
            translation,
        )

    subset_indices = _select_receptor_subset_for_conformer_family(
        receptor_coords=request.protein_coords,
        receptor_radii=request.receptor_radii,
        ligand_world_coords=base_world_coords,
        ligand_radii=request.ligand_ctx.base_radii,
        rotatable_bonds=bonds,
        target_error=request.target_error,
    )
    if _runtime_profile_enabled():
        print(
            f"[RUNTIME PROFILE] conformer_pose_setup: active_bonds={len(bonds)}, subset={int(subset_indices.shape[0])}",
            flush=True,
        )
    effective_omission_budget = (
        request.target_error if omission_budget is None else omission_budget
    )
    search_subset_indices = subset_indices
    search_omitted_bound = 0.0
    scan_subset_indices = subset_indices
    scan_omitted_bound = 0.0
    if bonds:
        omission_bounds = _per_receptor_atom_conformer_omission_bounds(
            ligand_world_coords=base_world_coords,
            ligand_radii=request.ligand_ctx.base_radii,
            receptor_coords=request.protein_coords[subset_indices],
            receptor_radii=request.receptor_radii[subset_indices],
            rotatable_bonds=bonds,
            ligand_charges=(
                None
                if request.ligand_ctx.charges is None
                else request.ligand_ctx.charges
            ),
            receptor_charges=(
                None
                if electrostatics is None
                else electrostatics.receptor_charges[subset_indices]
            ),
            electro_alpha=(None if electrostatics is None else electrostatics.alpha),
            electro_dielectric=(
                None if electrostatics is None else electrostatics.dielectric
            ),
        )
        local_search_subset, search_omitted_bound = (
            _select_receptor_subset_by_omission_budget(
                omission_bounds,
                omission_budget=effective_omission_budget,
            )
        )
        search_subset_indices = subset_indices[local_search_subset]
        if scan_only:
            scan_subset_indices = search_subset_indices
            scan_omitted_bound = search_omitted_bound
    local_activity_mask_fn = _build_cell_local_activity_mask_fn(
        receptor_coords=request.protein_coords,
        quaternion=quaternion,
        translation=translation,
        rotatable_bonds=bonds,
        scoring_cutoff=compute_certified_cutoff(request.target_error),
    )
    exact_conformer_scoring_context = (
        None
        if scoring_context is None
        else scoring_context.receptor_subset(subset_indices)
    )
    exact_conformer_electrostatics = (
        None
        if electrostatics is None
        else electrostatics.receptor_subset(subset_indices)
    )
    score_fn, score_fn_batch = _build_conformer_score_fns(
        request,
        quaternion,
        translation,
        exact_conformer_scoring_context,
        exact_conformer_electrostatics,
        receptor_coords=request.protein_coords[subset_indices],
        receptor_radii=request.receptor_radii[subset_indices],
    )
    scan_scoring_context = (
        None
        if scoring_context is None
        else scoring_context.receptor_subset(scan_subset_indices)
    )
    scan_electrostatics = (
        None
        if electrostatics is None
        else electrostatics.receptor_subset(scan_subset_indices)
    )
    coarse_score_fn, coarse_score_fn_batch = _build_conformer_coarse_score_fns(
        request,
        quaternion,
        translation,
        scan_scoring_context if scan_only else exact_conformer_scoring_context,
        scan_electrostatics if scan_only else exact_conformer_electrostatics,
        receptor_coords=(
            request.protein_coords[scan_subset_indices]
            if scan_only
            else request.protein_coords[subset_indices]
        ),
        receptor_radii=(
            request.receptor_radii[scan_subset_indices]
            if scan_only
            else request.receptor_radii[subset_indices]
        ),
    )
    if not bonds:
        return None
    active_strain_params = _restrict_strain_params(strain_params, active_mask)
    config = (
        _build_conformer_search_config(request, rotatable_bonds=bonds)
        if conformer_config is None
        else conformer_config
    )
    if conformer_config is not None:
        config = _restrict_conformer_search_config(
            request,
            config,
            active_mask,
            bonds,
        )

    cell_bound_state_fn = None
    child_cell_bound_state_fn = None
    if not scan_only:
        cell_bound_state_fn, child_cell_bound_state_fn = (
            _build_conformer_cell_bound_state_fns(
                request=request,
                quaternion=quaternion,
                translation=translation,
                scoring_context=scoring_context,
                electrostatics=electrostatics,
                rotatable_bonds=bonds,
                exact_subset_indices=subset_indices,
                omission_budget=float(effective_omission_budget),
                local_activity_mask_fn=local_activity_mask_fn,
            )
        )

    single_bond_scan_provider = None
    if scan_only and bonds:
        scan_score_cache: dict[
            tuple[int, ...],
            tuple[
                Callable[[jnp.ndarray], float],
                Callable[[jnp.ndarray], jnp.ndarray] | None,
            ],
        ] = {}
        single_bond_budgets = np.asarray(
            jax.device_get(
                _conformer_local_improvement_bounds(
                    request,
                    bonds,
                    tuple(float(v) for v in config.per_bond_lipschitz)
                    if config.per_bond_lipschitz is not None
                    else None,
                )
            )
            if config.per_bond_lipschitz is not None
            else np.zeros((len(bonds),), dtype=np.float32),
            dtype=np.float32,
        )

        any_single_bond_omission = False
        for bond_idx, bond in enumerate(bonds):
            omission_bounds = _per_receptor_atom_conformer_omission_bounds(
                ligand_world_coords=base_world_coords,
                ligand_radii=request.ligand_ctx.base_radii,
                receptor_coords=request.protein_coords[subset_indices],
                receptor_radii=request.receptor_radii[subset_indices],
                rotatable_bonds=(bond,),
                ligand_charges=(
                    None
                    if request.ligand_ctx.charges is None
                    else request.ligand_ctx.charges
                ),
                receptor_charges=(
                    None
                    if electrostatics is None
                    else electrostatics.receptor_charges[subset_indices]
                ),
                electro_alpha=(
                    None if electrostatics is None else electrostatics.alpha
                ),
                electro_dielectric=(
                    None if electrostatics is None else electrostatics.dielectric
                ),
            )
            local_subset, _ = _select_receptor_subset_by_omission_budget(
                omission_bounds,
                omission_budget=float(single_bond_budgets[bond_idx]),
            )
            if int(local_subset.shape[0]) < int(subset_indices.shape[0]):
                any_single_bond_omission = True
                break

        if any_single_bond_omission:

            def _single_bond_scan_provider(
                bond_idx: int,
                current_local_coords: jnp.ndarray,
            ) -> tuple[
                Callable[[jnp.ndarray], float],
                Callable[[jnp.ndarray], jnp.ndarray] | None,
            ]:
                current_world_coords = rigid_transform_3d(
                    current_local_coords,
                    quaternion,
                    translation,
                )
                single_bond = (bonds[bond_idx],)
                omission_bounds = _per_receptor_atom_conformer_omission_bounds(
                    ligand_world_coords=current_world_coords,
                    ligand_radii=request.ligand_ctx.base_radii,
                    receptor_coords=request.protein_coords[subset_indices],
                    receptor_radii=request.receptor_radii[subset_indices],
                    rotatable_bonds=single_bond,
                    ligand_charges=(
                        None
                        if request.ligand_ctx.charges is None
                        else request.ligand_ctx.charges
                    ),
                    receptor_charges=(
                        None
                        if electrostatics is None
                        else electrostatics.receptor_charges[subset_indices]
                    ),
                    electro_alpha=(
                        None if electrostatics is None else electrostatics.alpha
                    ),
                    electro_dielectric=(
                        None if electrostatics is None else electrostatics.dielectric
                    ),
                )
                local_subset, _ = _select_receptor_subset_by_omission_budget(
                    omission_bounds,
                    omission_budget=float(single_bond_budgets[bond_idx]),
                )
                actual_subset = subset_indices[local_subset]
                actual_subset_tuple = tuple(
                    int(i) for i in np.asarray(actual_subset, dtype=np.int32).tolist()
                )
                full_subset_tuple = tuple(
                    int(i) for i in np.asarray(subset_indices, dtype=np.int32).tolist()
                )
                if actual_subset_tuple == full_subset_tuple:
                    return coarse_score_fn, coarse_score_fn_batch
                cached = scan_score_cache.get(actual_subset_tuple)
                if cached is not None:
                    return cached
                local_ctx = (
                    None
                    if scoring_context is None
                    else scoring_context.receptor_subset(actual_subset)
                )
                local_electro = (
                    None
                    if electrostatics is None
                    else electrostatics.receptor_subset(actual_subset)
                )
                built = _build_conformer_coarse_score_fns(
                    request,
                    quaternion,
                    translation,
                    local_ctx,
                    local_electro,
                    receptor_coords=request.protein_coords[actual_subset],
                    receptor_radii=request.receptor_radii[actual_subset],
                )
                scan_score_cache[actual_subset_tuple] = built
                return built

            single_bond_scan_provider = _single_bond_scan_provider
    active_coverage_plan = None
    if bonds and config.per_bond_lipschitz is not None:
        active_coverage_plan = _derive_conformer_coverage_plan_from_lipschitz(
            score_lipschitz_constant=config.score_lipschitz_constant,
            per_bond_lipschitz=tuple(float(v) for v in config.per_bond_lipschitz),
            target_delta=request.target_error,
            target_rmsd=request.target_rmsd,
            max_arm=max(bond.max_arm_length for bond in bonds),
        )
    if _runtime_profile_enabled():
        print(
            "[RUNTIME PROFILE] conformer_search_start: "
            f"active_bonds={len(bonds)}, max_cells={config.max_cells}, "
            f"incumbent={pruning_incumbent_energy}, receptor_subset={int(search_subset_indices.shape[0])}, "
            f"scan_subset={int(scan_subset_indices.shape[0])}, omitted_bound={scan_omitted_bound:.3f}, "
            f"search_omitted_bound={search_omitted_bound:.3f}",
            flush=True,
        )
    phase_start = time.perf_counter()
    result = (
        search_conformers_sequential_scan(
            base_coords=request.ligand_ctx.base_coords,
            adjacency=adjacency,
            elements=request.ligand_ctx.elements,
            score_fn=coarse_score_fn,
            strain_params=active_strain_params,
            score_fn_batch=coarse_score_fn_batch,
            rotatable_bonds=bonds,
            single_bond_score_provider=single_bond_scan_provider,
        )
        if scan_only
        else search_conformers_support_grid(
            base_coords=request.ligand_ctx.base_coords,
            adjacency=adjacency,
            elements=request.ligand_ctx.elements,
            score_fn=score_fn,
            config=config,
            strain_params=active_strain_params,
            score_fn_batch=score_fn_batch,
            rotatable_bonds=bonds,
            canonical_segments=(
                None
                if active_coverage_plan is None
                else active_coverage_plan.canonical_segments
            ),
        )
        if (
            not scan_only
            and active_coverage_plan is not None
            and active_coverage_plan.support_size <= config.max_cells
        )
        else search_conformers(
            base_coords=request.ligand_ctx.base_coords,
            adjacency=adjacency,
            elements=request.ligand_ctx.elements,
            score_fn=score_fn,
            config=config,
            strain_params=active_strain_params,
            score_fn_batch=score_fn_batch,
            rotatable_bonds=bonds,
            pruning_incumbent_energy=(
                None
                if pruning_incumbent_energy is None
                else (
                    float(pruning_incumbent_energy)
                    if cell_bound_state_fn is not None
                    else float(pruning_incumbent_energy + search_omitted_bound)
                )
            ),
            local_activity_mask_fn=local_activity_mask_fn,
            cell_bound_state_fn=cell_bound_state_fn,
            child_cell_bound_state_fn=child_cell_bound_state_fn,
            score_is_exact=True,
        )
    )
    _runtime_profile_log("single_pose_conformer_search", phase_start)
    if not result.conformer_coords:
        return None
    best_conf = result.conformer_coords[0]
    world_coords = rigid_transform_3d(best_conf, quaternion, translation)
    exact_energy = _score_exact_pose_batch(
        request,
        poses_coords=world_coords[None, ...],
        electrostatics=electrostatics,
        scoring_context=scoring_context,
    )
    best_energy = float(np.asarray(jax.device_get(exact_energy[0])))
    theorem_handles = _merge_theorem_handles(
        result.theorem_handles,
        _conformer_runtime_theorem_handles(
            has_rotatable_bonds=bool(bonds),
            include_pocket_handles=request.certified_binding_site is not None,
            include_receptor_flex_handles=(
                scoring_context is not None
                and scoring_context.receptor_conformations is not None
            ),
        ),
        (
            omitted_channel_bound_theorem_handles()
            if (scan_only and scan_omitted_bound > 0.0)
            or ((not scan_only) and search_omitted_bound > 0.0)
            else ()
        ),
    )
    return world_coords, best_energy, theorem_handles


def _recertify_conformer_updated_pose(
    request: "PipelineDockingRequest",
    pose_coords: jnp.ndarray,
    *,
    scoring_context: CertifiedScoringContext | None,
    electrostatics: "CertifiedRealSpaceEwaldSpec | None",
) -> tuple[
    jnp.ndarray,
    float,
    RefinementCertificate | None,
    tuple[str, ...],
    float | None,
]:
    if not request.is_certified_mode:
        score = _score_exact_pose_batch(
            request,
            poses_coords=pose_coords[None, ...],
            electrostatics=electrostatics,
            scoring_context=scoring_context,
        )
        return pose_coords, float(score[0]), None, (), None

    pose_center = jnp.mean(pose_coords, axis=0)
    local_base_coords = pose_coords - pose_center
    local_ligand_ctx = LigandContext(
        base_coords=local_base_coords,
        base_radii=request.ligand_ctx.base_radii,
        center_of_mass=jnp.mean(local_base_coords, axis=0),
        elements=request.ligand_ctx.elements,
        charges=request.ligand_ctx.charges,
        adjacency=request.ligand_ctx.adjacency,
    )
    local_request = cast(
        "PipelineDockingRequest",
        request.with_updates(ligand_ctx=local_ligand_ctx),
    )

    initial_translation = pose_center[None, ...]
    initial_quaternion = jnp.array([[1.0, 0.0, 0.0, 0.0]], dtype=pose_coords.dtype)

    refined_translations, refined_quaternions, certificates = _certified_refinement(
        request=local_request,
        initial_translations=initial_translation,
        initial_quaternions=initial_quaternion,
        mode_override=RefinementCertificationMode.OBSERVED,
    )

    spectral_mu_coord = None
    if certificates[0] is not None and certificates[0].spectral is not None:
        cert_mu = certificates[0].spectral.mu_coord
        if math.isfinite(cert_mu) and cert_mu > 0:
            spectral_mu_coord = float(cert_mu)
    else:
        print(
            f"[RECERTIFY DEBUG] No certificate found. Certificate is None: {certificates[0] is None}, "
            f"spectral is None: {certificates[0].spectral if certificates[0] is not None else 'N/A'}",
            flush=True,
        )

    if certificates[0] is not None:
        print(
            f"[RECERTIFY DEBUG] Got certificate: spectral={certificates[0].spectral}",
            flush=True,
        )

    refined_coords = apply_poses(
        local_ligand_ctx,
        PoseVector(
            translation=refined_translations,
            quaternion=refined_quaternions,
        ),
    )[0]

    score = _score_exact_pose_batch(
        request,
        poses_coords=refined_coords[None, ...],
        electrostatics=electrostatics,
        scoring_context=scoring_context,
    )

    print(
        f"[RECERTIFY] About to return: spectral_mu_coord={spectral_mu_coord}, cert={certificates[0] is not None}",
        flush=True,
    )
    return (
        refined_coords,
        float(score[0]),
        certificates[0],
        (
            ()
            if certificates[0] is None or certificates[0].iteration_budget_plan is None
            else certificates[0].iteration_budget_plan.theorem_handles
        ),
        spectral_mu_coord,
    )


def _certify_single_rigid_pose(
    request: "PipelineDockingRequest",
    *,
    translation: jnp.ndarray,
    quaternion: jnp.ndarray,
    scoring_context: CertifiedScoringContext | None,
    electrostatics: "CertifiedRealSpaceEwaldSpec | None",
    site_center: jnp.ndarray | None,
    site_radius: float,
    failure_reasons_out: list[str | None] | None = None,
) -> tuple[jnp.ndarray, float, RefinementCertificate | None, jnp.ndarray, jnp.ndarray]:
    attempt_modes: list[RefinementCertificationMode | None] = [None]
    if (
        request.config is not None
        and request.config.refinement_certification
        != RefinementCertificationMode.OBSERVED
    ):
        attempt_modes.append(RefinementCertificationMode.OBSERVED)

    best_result: (
        tuple[
            jnp.ndarray,
            float,
            RefinementCertificate | None,
            jnp.ndarray,
            jnp.ndarray,
        ]
        | None
    ) = None
    best_failure_reason: str | None = None

    for mode_override in attempt_modes:
        attempt_failure_reasons: list[str | None] = []
        opt_t_subset, opt_q_subset, subset_certificates = _certified_refinement(
            request=request,
            initial_translations=translation[None, ...],
            initial_quaternions=quaternion[None, ...],
            mode_override=mode_override,
            failure_reasons_out=attempt_failure_reasons,
        )
        subset_coords = apply_poses(
            request.ligand_ctx,
            PoseVector(translation=opt_t_subset, quaternion=opt_q_subset),
        )[0]
        cert = subset_certificates[0]
        if site_center is not None and site_radius > 0.0:
            subset_center = np.asarray(jax.device_get(jnp.mean(subset_coords, axis=0)))
            if np.linalg.norm(subset_center - np.asarray(site_center)) > float(
                site_radius
            ):
                subset_coords = apply_poses(
                    request.ligand_ctx,
                    PoseVector(
                        translation=translation[None, ...],
                        quaternion=quaternion[None, ...],
                    ),
                )[0]
                cert = None
                opt_t_subset = translation[None, ...]
                opt_q_subset = quaternion[None, ...]
        exact_score = _score_exact_pose_batch(
            request,
            poses_coords=subset_coords[None, ...],
            electrostatics=electrostatics,
            scoring_context=scoring_context,
        )
        result = (
            subset_coords,
            float(exact_score[0]),
            cert,
            opt_t_subset[0],
            opt_q_subset[0],
        )
        if best_result is None or result[1] < best_result[1]:
            best_result = result
            best_failure_reason = attempt_failure_reasons[0]
        if cert is not None:
            if mode_override == RefinementCertificationMode.OBSERVED:
                print(
                    "[REFINE_CERTS] Winner-only certification fell back to observed mode",
                    flush=True,
                )
            if failure_reasons_out is not None:
                failure_reasons_out.append(None)
            return result

    assert best_result is not None
    if failure_reasons_out is not None:
        failure_reasons_out.append(best_failure_reason)
    return best_result


def _score_exact_pose_batch(
    request: PipelineDockingRequest,
    *,
    poses_coords: jnp.ndarray,
    electrostatics: CertifiedRealSpaceEwaldSpec | None,
    scoring_context: CertifiedScoringContext | None = None,
    chunk_size: int = EXACT_RESCORING_CHUNK_SIZE,
) -> jnp.ndarray:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if poses_coords.shape[0] == 0:
        return jnp.zeros((0,), dtype=request.protein_coords.dtype)

    if request.is_certified_mode:
        resolved_scoring_context = (
            resolve_request_scoring_context(
                request,
                engine=request.effective_engine,
            )
            if scoring_context is None
            else scoring_context
        )

        def score_chunk(chunk_coords: jnp.ndarray) -> jnp.ndarray:
            return jnp.asarray(
                resolved_scoring_context.score_exact_batch(
                    receptor_coords=request.protein_coords,
                    poses_coords=chunk_coords,
                    receptor_radii=request.receptor_radii,
                    ligand_radii=request.ligand_ctx.base_radii,
                    target_error=request.target_error,
                    epsilon=0.2,
                ).scores
            )
    else:

        def score_chunk(chunk_coords: jnp.ndarray) -> jnp.ndarray:
            return jnp.asarray(
                route_scoring(
                    **derive_route_scoring_kwargs(
                        request,
                        engine=request.effective_engine,
                        poses_coords=chunk_coords,
                        electrostatics=electrostatics,
                    )
                )
            )

    if poses_coords.shape[0] <= chunk_size:
        return score_chunk(poses_coords)

    scored_chunks: list[jnp.ndarray] = []
    for start in range(0, poses_coords.shape[0], chunk_size):
        stop = min(start + chunk_size, poses_coords.shape[0])
        scored_chunks.append(score_chunk(poses_coords[start:stop]))

    return jnp.concatenate(scored_chunks, axis=0)


def _score_exact_pose_batch_padded(
    request: PipelineDockingRequest,
    *,
    poses_coords: jnp.ndarray,
    electrostatics: CertifiedRealSpaceEwaldSpec | None,
    scoring_context: CertifiedScoringContext | None = None,
    pad_size: int = FINAL_EXACT_RESCORING_PAD_SIZE,
) -> jnp.ndarray:
    if poses_coords.shape[0] == 0:
        return jnp.zeros((0,), dtype=request.protein_coords.dtype)
    if pad_size <= 0:
        raise ValueError("pad_size must be positive")
    if poses_coords.shape[0] > pad_size:
        return _score_exact_pose_batch(
            request,
            poses_coords=poses_coords,
            electrostatics=electrostatics,
            scoring_context=scoring_context,
        )
    padded = poses_coords
    if poses_coords.shape[0] < pad_size:
        pad_count = int(pad_size - poses_coords.shape[0])
        pad_coords = jnp.repeat(poses_coords[:1], pad_count, axis=0)
        padded = jnp.concatenate((poses_coords, pad_coords), axis=0)
    scored = _score_exact_pose_batch(
        request,
        poses_coords=padded,
        electrostatics=electrostatics,
        scoring_context=scoring_context,
    )
    return scored[: poses_coords.shape[0]]


def _score_certified_batch_padded(
    score_fn: Callable[[jnp.ndarray], CertifiedBatchResult],
    poses_coords: jnp.ndarray,
    *,
    pad_size: int = FINAL_EXACT_RESCORING_PAD_SIZE,
) -> CertifiedBatchResult:
    if poses_coords.shape[0] == 0:
        raise ValueError("poses_coords must be non-empty")
    padded = poses_coords
    if 0 < poses_coords.shape[0] < pad_size:
        pad_count = int(pad_size - poses_coords.shape[0])
        pad_coords = jnp.repeat(poses_coords[:1], pad_count, axis=0)
        padded = jnp.concatenate((poses_coords, pad_coords), axis=0)
    batch = score_fn(padded)
    if padded.shape[0] == poses_coords.shape[0]:
        return batch
    return replace(
        batch,
        scores=batch.scores[: poses_coords.shape[0]],
        posewise_error_bound=(
            None
            if batch.posewise_error_bound is None
            else batch.posewise_error_bound[: poses_coords.shape[0]]
        ),
    )


def _score_rigid_exact_pose_batch(
    request: PipelineDockingRequest,
    *,
    poses_coords: jnp.ndarray,
    electrostatics: CertifiedRealSpaceEwaldSpec | None,
    scoring_context: CertifiedScoringContext | None = None,
    chunk_size: int = EXACT_RESCORING_CHUNK_SIZE,
) -> jnp.ndarray:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if poses_coords.shape[0] == 0:
        return jnp.zeros((0,), dtype=request.protein_coords.dtype)

    if request.is_certified_mode:
        resolved_scoring_context = (
            resolve_request_scoring_context(
                request,
                engine=request.effective_engine,
            )
            if scoring_context is None
            else scoring_context
        )

        def score_chunk(chunk_coords: jnp.ndarray) -> jnp.ndarray:
            return jnp.asarray(
                resolved_scoring_context.score_rigid_exact_batch(
                    receptor_coords=request.protein_coords,
                    poses_coords=chunk_coords,
                    receptor_radii=request.receptor_radii,
                    ligand_radii=request.ligand_ctx.base_radii,
                    target_error=request.target_error,
                    epsilon=0.2,
                ).scores
            )

    else:

        def score_chunk(chunk_coords: jnp.ndarray) -> jnp.ndarray:
            return jnp.asarray(
                route_scoring(
                    **derive_route_scoring_kwargs(
                        request,
                        engine=request.effective_engine,
                        poses_coords=chunk_coords,
                        electrostatics=electrostatics,
                    )
                )
            )

    if poses_coords.shape[0] <= chunk_size:
        return score_chunk(poses_coords)

    scored_chunks: list[jnp.ndarray] = []
    for start in range(0, poses_coords.shape[0], chunk_size):
        stop = min(start + chunk_size, poses_coords.shape[0])
        scored_chunks.append(score_chunk(poses_coords[start:stop]))

    return jnp.concatenate(scored_chunks, axis=0)


def _pad_to_size(
    arr: jax.Array, size: int, axis: int = 0, value: float = 0.0
) -> jax.Array:
    """Pad or clip an array to a fixed size along a specific axis."""
    current_size = arr.shape[axis]
    if current_size == size:
        return arr
    if current_size > size:
        return jax.lax.dynamic_slice_in_dim(arr, 0, size, axis=axis)

    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, size - current_size)
    return jnp.pad(arr, pad_width, constant_values=value)


def _pad_tuple_to_size(
    tup: tuple[str, ...] | None, size: int, value: str = "G"
) -> tuple[str, ...] | None:
    """Pad or clip a tuple to a fixed size."""
    if tup is None:
        return None
    current_size = len(tup)
    if current_size == size:
        return tup
    if current_size > size:
        return tup[:size]
    return tup + (value,) * (size - current_size)


def _normalize_sampling_key(key: jax.Array | None) -> jax.Array:
    return jax.random.PRNGKey(0) if key is None else key


@dataclass(frozen=True)
class PipelinePoseBatch:
    request: PipelineDockingRequest
    pose_vecs: PoseVector
    certified_family: "CertifiedGlobalActionFamily | None" = None


@dataclass(frozen=True)
class PipelineInitialScores:
    final_scores: jnp.ndarray | np.ndarray
    survivor_pose_vecs: PoseVector | None = None
    survivor_exact_scores: jnp.ndarray | np.ndarray | None = None
    valid_survivor_mask: jnp.ndarray | np.ndarray | None = None
    survivor_coords: jnp.ndarray | np.ndarray | None = None
    execution_plan: CertifiedPipelineExecutionPlan | None = None


class PipelineRoute(ABC):
    def prepare_request(
        self, request: PipelineDockingRequest
    ) -> PipelineDockingRequest:
        return request

    @abstractmethod
    def generate_pose_batch(self, request: PipelineDockingRequest) -> PipelinePoseBatch:
        """Generate the initial pose batch for the route."""

    @abstractmethod
    def score_pose_batch(
        self,
        request: PipelineDockingRequest,
        batched_coords: jnp.ndarray,
        pose_vecs: PoseVector,
    ) -> PipelineInitialScores:
        """Score the initial pose batch."""

    def best_index_limit(
        self, request: PipelineDockingRequest, initial_scores: PipelineInitialScores
    ) -> int:
        return request.n_poses

    def optimization_inputs(
        self,
        request: PipelineDockingRequest,
        pose_vecs: PoseVector,
        initial_scores: PipelineInitialScores,
    ) -> tuple[jnp.ndarray, jnp.ndarray, int]:
        # Optimize all canonical retain survivors — the retain set IS the
        # optimization set. Its size is emergent from delta, not a heuristic.
        n_to_opt = request.n_poses
        return (
            pose_vecs.translation,
            pose_vecs.quaternion,
            n_to_opt,
        )

    def validate_backend(
        self, request: PipelineDockingRequest, backend: OptimizerBackend
    ) -> None:
        del request, backend


class DirectScoringRoute(PipelineRoute, ABC):
    def prepare_request(
        self, request: PipelineDockingRequest
    ) -> PipelineDockingRequest:
        return request.with_fixed_size_padding()

    def score_pose_batch(
        self,
        request: PipelineDockingRequest,
        batched_coords: jnp.ndarray,
        pose_vecs: PoseVector,
    ) -> PipelineInitialScores:
        del pose_vecs
        kwargs = derive_route_scoring_kwargs(
            request,
            engine=request.effective_engine,
            poses_coords=batched_coords,
            electrostatics=resolve_request_electrostatics(
                request,
                engine=request.effective_engine,
            ),
        )
        return PipelineInitialScores(final_scores=route_scoring(**kwargs))


class MultiStageScoringRoute(PipelineRoute, ABC):
    def score_pose_batch(
        self,
        request: PipelineDockingRequest,
        batched_coords: jnp.ndarray,
        pose_vecs: PoseVector,
    ) -> PipelineInitialScores:
        del pose_vecs
        from dq_dock_engine.docking.scoring_stages import (
            StageLevel,
            create_pipeline,
            create_receptor_data,
        )
        from dq_dock_engine.docking.charges import create_charge_assigner, ChargeMethod

        typed_request = cast(MultiStagePipelineDockingRequest, request)
        receptor_elements = typed_request.receptor_elements
        if receptor_elements is None:
            receptor_elements = tuple(["C"] * len(typed_request.protein_coords))

        assigner = create_charge_assigner(
            cast(ChargeMethod, typed_request.charge_method)
        )
        if assigner.method == ChargeMethod.SIMPLE:
            receptor_charges = assigner.assign(receptor_elements).charges
        else:
            if typed_request.receptor_file is None:
                raise ValueError(
                    f"ChargeMethod {assigner.method.name} requires a receptor file path or RDKit Mol to assign charges."
                )
            receptor_charges = assigner.assign(typed_request.receptor_file).charges

        receptor_data = create_receptor_data(
            coords=typed_request.protein_coords,
            radii=typed_request.receptor_radii,
            charges=receptor_charges,
            elements=receptor_elements,
        )
        pipeline = create_pipeline(
            (
                StageLevel.STAGE1_GEOMETRIC,
                StageLevel.STAGE2_MEDIUM,
                StageLevel.STAGE3_FULL,
            )
        )
        stage_results, validation = pipeline.run(
            receptor_data,
            batched_coords,
            validate=False,
            ligand_radii=typed_request.ligand_ctx.base_radii,
            ligand_charges=typed_request.ligand_ctx.charges,
            ligand_elements=typed_request.ligand_ctx.elements,
        )
        if validation is not None:
            print(
                f"Stage validation: Spearman 1-3={validation.spearman_1_3:.2f}, Top-10 overlap={validation.top10_overlap_1_3:.2f}"
            )
        return PipelineInitialScores(final_scores=stage_results[-1].scores)


class CertifiedPipelineRoute(PipelineRoute):
    def prepare_request(
        self, request: PipelineDockingRequest
    ) -> PipelineDockingRequest:
        prepared_request = cast(CertifiedPreparationMixin, request).prepare()
        return prepared_request.with_scoring_override(target_error=request.target_error)

    def generate_pose_batch(self, request: PipelineDockingRequest) -> PipelinePoseBatch:
        prepared_request = cast(PreparedCertifiedDirectPipelineRequest, request)
        generation = _create_certified_pose_vectors(
            box=prepared_request.box,
            n_poses=prepared_request.n_poses,
            certified_binding_site=cast(
                CertifiedPocketPreparation,
                prepared_request.certified_pocket_prep,
            ).plan.binding_site,
            rigid_seed_family_plan=(
                prepared_request.rigid_seed_family_plan
                if prepared_request.rigid_seed_family_plan is not None
                else (
                    None
                    if prepared_request.seed_budget_plan is None
                    else prepared_request.seed_budget_plan.selected_family_plan
                )
            ),
        )
        return PipelinePoseBatch(
            request=prepared_request,
            pose_vecs=generation.pose_vecs,
            certified_family=generation.family,
        )

    def score_pose_batch(
        self,
        request: PipelineDockingRequest,
        batched_coords: jnp.ndarray,
        pose_vecs: PoseVector,
    ) -> PipelineInitialScores:
        electrostatics = resolve_request_electrostatics(
            request,
            engine=request.effective_engine,
        )
        full_scoring_context = (
            resolve_request_scoring_context(
                request,
                engine=request.effective_engine,
            )
            if request.is_certified_mode
            else None
        )
        do_conf = _request_uses_conformer_search(request)
        pruning_scoring_context = (
            None
            if full_scoring_context is None
            else full_scoring_context.pruning_context()
        )
        scoring_context = (
            None
            if full_scoring_context is None
            else full_scoring_context.optimization_context()
            if full_scoring_context.uses_extended_rich
            else full_scoring_context
        )
        rotatable_bonds = _request_rotatable_bonds(request) if do_conf else ()
        rigid_refinement_plan = _derive_certified_rigid_local_refinement_plan(
            request, cast(CertifiedScoringContext, scoring_context)
        )
        extra_pruning_delta_budget = None
        exact_pruning_scoring_context = None
        if (
            full_scoring_context is not None
            and full_scoring_context.uses_extended_rich
            and not do_conf
        ):
            exact_pruning_scoring_context = full_scoring_context.optimization_context()
            pruning_scoring_context = None
        phase_start = time.perf_counter()
        execution_plan = _certified_pruning_pass(
            request,
            poses_coords=batched_coords,
            electrostatics=electrostatics,
            scoring_context=pruning_scoring_context,
            extra_pruning_delta_budget=extra_pruning_delta_budget,
            exact_chunk_scoring_context=exact_pruning_scoring_context,
            rotatable_bonds=rotatable_bonds,
            rigid_improvement_scoring_context=cast(
                CertifiedScoringContext, scoring_context
            ),
            rigid_refinement_plan=rigid_refinement_plan,
        )
        _runtime_profile_log("certified_pruning_pass", phase_start)
        survivor_mask = execution_plan.retain_mask
        native_witness_debug: dict[str, object] | None = None
        native_witness_pose_index: int | None = None
        if request.debug_native_coords is not None and batched_coords.shape[0] > 0:
            from dq_dock_engine.docking.metrics import compute_docking_rmsd_batched

            native_rmsds = np.asarray(
                jax.device_get(
                    compute_docking_rmsd_batched(
                        batched_coords, request.debug_native_coords
                    )
                ),
                dtype=np.float64,
            )
            native_witness_pose_index = int(np.argmin(native_rmsds))
            native_witness_debug = {
                "global_pose_index": native_witness_pose_index,
                "initial_rmsd": float(native_rmsds[native_witness_pose_index]),
                "coarse_score": float(
                    np.asarray(jax.device_get(execution_plan.coarse_scores))[
                        native_witness_pose_index
                    ]
                ),
                "retained_after_coarse_pruning": bool(
                    np.asarray(jax.device_get(survivor_mask), dtype=bool)[
                        native_witness_pose_index
                    ]
                ),
            }
        valid_survivor_indices = jnp.asarray(
            np.flatnonzero(np.asarray(jax.device_get(survivor_mask), dtype=bool))
        )
        survivor_coords = batched_coords[valid_survivor_indices]
        survivor_pose_vecs = PoseVector(
            translation=pose_vecs.translation[valid_survivor_indices],
            quaternion=pose_vecs.quaternion[valid_survivor_indices],
        )

        exact_improvement_family: PoseSpecificImprovementBudgetFamily | None = None
        if do_conf:
            survivor_rigid_scores = _score_rigid_exact_pose_batch(
                request,
                poses_coords=survivor_coords,
                electrostatics=electrostatics,
                scoring_context=scoring_context,
            )
            conformer_config = _build_conformer_search_config(
                request,
                rotatable_bonds=rotatable_bonds,
            )
            per_bond_bounds = _conformer_local_improvement_bounds(
                request,
                rotatable_bonds,
                conformer_config.per_bond_lipschitz,
            )
            strain_params = None
            exact_improvement_family = _posewise_improvement_budget_family(
                survivor_coords,
                request.protein_coords,
                rotatable_bonds,
                per_bond_bounds,
                scoring_cutoff=compute_certified_cutoff(request.target_error),
                pose_indices=np.asarray(
                    jax.device_get(valid_survivor_indices),
                    dtype=np.int32,
                ),
                active_subset_source="survivor_exact_interaction_cutoff",
            )
            exact_additive_correction = jnp.asarray(
                exact_improvement_family.totals_array(),
                dtype=survivor_rigid_scores.dtype,
            )
            (
                posewise_rigid_improvement_bounds,
                exact_clearance_safe_mask,
                posewise_clearance,
            ) = _posewise_rigid_local_improvement_bounds(
                request,
                cast(CertifiedScoringContext, scoring_context),
                poses_coords=survivor_coords,
                base_translation_step=rigid_refinement_plan.base_translation_step,
                base_rotation_step_rad=rigid_refinement_plan.base_rotation_step_rad,
                ligand_radius=rigid_refinement_plan.ligand_radius,
                n_rounds=rigid_refinement_plan.n_search_rounds,
            )
            exact_additive_correction = exact_additive_correction + jnp.asarray(
                posewise_rigid_improvement_bounds,
                dtype=survivor_rigid_scores.dtype,
            )
            if (
                scoring_context is not None
                and scoring_context.receptor_conformations is not None
            ):
                exact_additive_correction = (
                    exact_additive_correction
                    + scoring_context.posewise_receptor_flex_error_exact_batch(
                        receptor_coords=request.protein_coords,
                        poses_coords=survivor_coords,
                        receptor_radii=request.receptor_radii,
                        ligand_radii=request.ligand_ctx.base_radii,
                        target_error=request.target_error,
                        epsilon=0.2,
                    )
                )
            bound_exact_survivor_mask, _, _ = _canonical_retain_mask(
                survivor_rigid_scores,
                delta=0.0,
                additive_correction=exact_additive_correction,
            )
            exact_survivor_mask_np = np.asarray(
                jax.device_get(bound_exact_survivor_mask), dtype=bool
            )
            exact_survivor_mask_np = np.logical_or(
                exact_survivor_mask_np,
                ~np.asarray(exact_clearance_safe_mask, dtype=bool),
            )
            exact_survivor_mask = jnp.asarray(exact_survivor_mask_np, dtype=bool)
            if (
                native_witness_debug is not None
                and native_witness_pose_index is not None
            ):
                survivor_global_indices = np.asarray(
                    jax.device_get(valid_survivor_indices), dtype=np.int32
                )
                matches = np.where(
                    survivor_global_indices == native_witness_pose_index
                )[0]
                native_witness_debug["retained_after_exact_pruning"] = False
                native_witness_debug.update(
                    {
                        "exact_pruning_clearance_safe_pose_count": int(
                            np.count_nonzero(exact_clearance_safe_mask)
                        ),
                        "exact_pruning_clearance_unsafe_pose_count": int(
                            np.count_nonzero(~np.asarray(exact_clearance_safe_mask))
                        ),
                        "exact_pruning_bound_retained_pose_count": int(
                            np.count_nonzero(
                                np.asarray(
                                    jax.device_get(bound_exact_survivor_mask),
                                    dtype=bool,
                                )
                            )
                        ),
                        "exact_pruning_combined_retained_pose_count": int(
                            np.count_nonzero(exact_survivor_mask_np)
                        ),
                    }
                )
                if matches.size > 0:
                    witness_local_idx = int(matches[0])
                    survivor_scores_np = np.asarray(
                        jax.device_get(survivor_rigid_scores), dtype=np.float64
                    )
                    correction_np = np.asarray(
                        jax.device_get(exact_additive_correction), dtype=np.float64
                    )
                    bound_exact_survivor_mask_np = np.asarray(
                        jax.device_get(bound_exact_survivor_mask), dtype=bool
                    )
                    native_witness_debug.update(
                        {
                            "survivor_local_index": witness_local_idx,
                            "rigid_exact_score": float(
                                survivor_scores_np[witness_local_idx]
                            ),
                            "rigid_exact_rank_among_coarse_survivors": int(
                                np.where(
                                    np.argsort(survivor_scores_np) == witness_local_idx
                                )[0][0]
                            )
                            + 1,
                            "rigid_exact_best_score": float(np.min(survivor_scores_np)),
                            "rigid_local_improvement_bound": float(
                                posewise_rigid_improvement_bounds[witness_local_idx]
                            ),
                            "posewise_improvement_budget": float(
                                correction_np[witness_local_idx]
                            ),
                            "exact_pruning_clearance_safe": bool(
                                np.asarray(exact_clearance_safe_mask, dtype=bool)[
                                    witness_local_idx
                                ]
                            ),
                            "exact_pruning_clearance": float(
                                np.asarray(posewise_clearance, dtype=np.float32)[
                                    witness_local_idx
                                ]
                            ),
                            "retained_after_exact_pruning_by_bound": bool(
                                bound_exact_survivor_mask_np[witness_local_idx]
                            ),
                            "retained_after_exact_pruning_by_clearance_fallback": bool(
                                (
                                    not np.asarray(
                                        exact_clearance_safe_mask, dtype=bool
                                    )[witness_local_idx]
                                )
                                and exact_survivor_mask_np[witness_local_idx]
                            ),
                            "retained_after_exact_pruning": bool(
                                exact_survivor_mask_np[witness_local_idx]
                            ),
                        }
                    )
            valid_survivor_indices = valid_survivor_indices[exact_survivor_mask]
            survivor_coords = survivor_coords[exact_survivor_mask]
            survivor_pose_vecs = PoseVector(
                translation=survivor_pose_vecs.translation[exact_survivor_mask],
                quaternion=survivor_pose_vecs.quaternion[exact_survivor_mask],
            )
        else:
            strain_params = None

        execution_plan = execution_plan.with_final_survivor_indices(
            tuple(
                int(index)
                for index in np.asarray(
                    jax.device_get(valid_survivor_indices),
                    dtype=np.int32,
                ).tolist()
            )
        )
        if native_witness_debug is not None:
            execution_plan = execution_plan.with_native_witness_debug(
                native_witness_debug
            )

        phase_start = time.perf_counter()
        survivor_exact_scores = _score_exact_pose_batch(
            request,
            poses_coords=survivor_coords,
            electrostatics=electrostatics,
            scoring_context=scoring_context,
        )
        _runtime_profile_log("survivor_exact_scoring", phase_start)
        if (
            full_scoring_context is not None
            and full_scoring_context.uses_extended_rich
            and survivor_coords.shape[0] > 0
            and do_conf
        ):
            rich_survivor_batch = full_scoring_context.score_exact_batch(
                receptor_coords=request.protein_coords,
                poses_coords=survivor_coords,
                receptor_radii=request.receptor_radii,
                ligand_radii=request.ligand_ctx.base_radii,
                target_error=request.target_error,
                epsilon=0.2,
            )
            disambiguation_batch = full_scoring_context.score_flip_disambiguation_batch(
                receptor_coords=request.protein_coords,
                poses_coords=survivor_coords,
                receptor_radii=request.receptor_radii,
                ligand_radii=request.ligand_ctx.base_radii,
                target_error=request.target_error,
                epsilon=0.2,
            )
            if _shared_certified_singleton_top1(
                rich_survivor_batch.scores,
                float(np.asarray(jax.device_get(rich_survivor_batch.error_bound))),
                disambiguation_batch.scores,
                float(np.asarray(jax.device_get(disambiguation_batch.error_bound))),
            ):
                survivor_exact_scores = rich_survivor_batch.scores
            else:
                survivor_exact_scores = disambiguation_batch.scores
            if native_witness_debug is not None:
                native_witness_debug["orientation_margin_certified_singleton"] = (
                    _orientation_margin_certified_singleton_top1(
                        full_scoring_context,
                        disambiguation_batch.scores,
                        float(
                            np.asarray(jax.device_get(disambiguation_batch.error_bound))
                        ),
                    )
                )
        if do_conf and not request.optimize:
            phase_start = time.perf_counter()
            updated_scores = np.asarray(jax.device_get(survivor_exact_scores)).copy()
            updated_coords = np.asarray(jax.device_get(survivor_coords)).copy()
            conformer_config = _build_conformer_search_config(
                request, rotatable_bonds=rotatable_bonds
            )
            if exact_improvement_family is None:
                ordered_indices = np.argsort(updated_scores)
            else:
                lower_bound_order = updated_scores - np.asarray(
                    [
                        budget.total_budget
                        for budget in exact_improvement_family.budgets
                    ],
                    dtype=np.float64,
                )
                ordered_indices = np.argsort(lower_bound_order)
            best_energy_with_conf = float(np.min(updated_scores))
            scan_start = time.perf_counter()
            for i in ordered_indices.tolist():
                pose_conf_improvement_budget = (
                    0.0
                    if exact_improvement_family is None
                    else exact_improvement_family.budgets[i].total_budget
                )
                if (
                    updated_scores[i]
                    > best_energy_with_conf + pose_conf_improvement_budget
                ):
                    continue
                scan_conf = _run_conformer_search_for_pose(
                    request,
                    quaternion=survivor_pose_vecs.quaternion[i],
                    translation=survivor_pose_vecs.translation[i],
                    scoring_context=scoring_context,
                    electrostatics=electrostatics,
                    rotatable_bonds=rotatable_bonds,
                    conformer_config=conformer_config,
                    strain_params=strain_params,
                    scan_only=True,
                )
                if scan_conf is not None and scan_conf[1] < best_energy_with_conf:
                    best_energy_with_conf = scan_conf[1]
            _runtime_profile_log("survivor_conformer_scan", scan_start)
            for i in ordered_indices.tolist():
                pose_conf_improvement_budget = (
                    0.0
                    if exact_improvement_family is None
                    else exact_improvement_family.budgets[i].total_budget
                )
                if (
                    updated_scores[i]
                    > best_energy_with_conf + pose_conf_improvement_budget
                ):
                    print(
                        "[CONFORMER SKIP] Skipping survivor conformer search: "
                        f"pose={i}, rigid_score={updated_scores[i]:.3f}, "
                        f"best={best_energy_with_conf:.3f}, "
                        f"improvement_budget={pose_conf_improvement_budget:.3f}",
                        flush=True,
                    )
                    continue
                conf = _run_conformer_search_for_pose(
                    request,
                    quaternion=survivor_pose_vecs.quaternion[i],
                    translation=survivor_pose_vecs.translation[i],
                    scoring_context=scoring_context,
                    electrostatics=electrostatics,
                    rotatable_bonds=rotatable_bonds,
                    conformer_config=conformer_config,
                    strain_params=strain_params,
                    pruning_incumbent_energy=float(best_energy_with_conf),
                )
                if conf is not None and conf[1] < float(updated_scores[i]):
                    updated_coords[i] = np.asarray(conf[0])
                    updated_scores[i] = conf[1]
                    if conf[1] < best_energy_with_conf:
                        best_energy_with_conf = conf[1]
            survivor_exact_scores = jnp.asarray(updated_scores)
            survivor_coords = jnp.asarray(updated_coords)
            _runtime_profile_log("survivor_conformer_search", phase_start)
        elif do_conf and _runtime_profile_enabled():
            print(
                "[RUNTIME PROFILE] survivor_conformer_search: deferred to post-refinement stage",
                flush=True,
            )

        final_scores = (
            jnp.full((batched_coords.shape[0],), 1e6)
            .at[valid_survivor_indices]
            .set(survivor_exact_scores, indices_are_sorted=False)
        )
        return PipelineInitialScores(
            final_scores=final_scores,
            survivor_pose_vecs=survivor_pose_vecs,
            survivor_exact_scores=survivor_exact_scores,
            valid_survivor_mask=jnp.ones(survivor_exact_scores.shape, dtype=bool),
            survivor_coords=survivor_coords,
            execution_plan=execution_plan,
        )

    def best_index_limit(
        self, request: PipelineDockingRequest, initial_scores: PipelineInitialScores
    ) -> int:
        del request
        assert initial_scores.survivor_exact_scores is not None
        return int(initial_scores.survivor_exact_scores.shape[0])

    def optimization_inputs(
        self,
        request: PipelineDockingRequest,
        pose_vecs: PoseVector,
        initial_scores: PipelineInitialScores,
    ) -> tuple[jnp.ndarray, jnp.ndarray, int]:
        del request, pose_vecs
        assert initial_scores.survivor_pose_vecs is not None
        assert initial_scores.survivor_exact_scores is not None
        n_valid_survivors = initial_scores.survivor_pose_vecs.translation.shape[0]
        survivor_ranked = jnp.argsort(initial_scores.survivor_exact_scores)
        return (
            initial_scores.survivor_pose_vecs.translation[survivor_ranked],
            initial_scores.survivor_pose_vecs.quaternion[survivor_ranked],
            n_valid_survivors,
        )

    def validate_backend(
        self, request: PipelineDockingRequest, backend: OptimizerBackend
    ) -> None:
        if backend != OptimizerBackend.FORMAL:
            raise ValueError(
                "CERTIFIED mode requires the formal optimizer backend; gradient refinement is heuristic."
            )


class GeometricPocketRoute(DirectScoringRoute):
    def generate_pose_batch(self, request: PipelineDockingRequest) -> PipelinePoseBatch:
        prepared_request = cast(GeometricPreparationMixin, request).prepare()
        key, pose_vecs = prepared_request.sample_pose_batch()
        return PipelinePoseBatch(
            request=prepared_request.with_updates(key=key),
            pose_vecs=pose_vecs,
        )


class BoxSamplingRoute(DirectScoringRoute):
    def generate_pose_batch(self, request: PipelineDockingRequest) -> PipelinePoseBatch:
        from dq_dock_engine.docking.placement import sample_random_poses

        return PipelinePoseBatch(
            request=request,
            pose_vecs=sample_random_poses(
                request.normalized_key,
                request.box,
                request.n_poses,
            ),
        )


class GeometricMultiStageRoute(MultiStageScoringRoute, GeometricPocketRoute):
    pass


class BoxMultiStageRoute(MultiStageScoringRoute, BoxSamplingRoute):
    pass


def nominalize_pipeline_request(
    request: PipelineDockingRequest,
) -> PipelineDockingRequest:
    if isinstance(request, NominalPipelineDockingRequest):
        return request
    return NominalPipelineDockingRequest.from_request(request)


def derive_pipeline_route(request: PipelineDockingRequest) -> PipelineRoute:
    if not isinstance(request, NominalPipelineDockingRequest):
        raise TypeError(
            f"Cannot derive a pipeline route from non-nominal request type {type(request).__name__}."
        )
    return request.create_route()


def _run_docking_pipeline_request(
    request: PipelineDockingRequest,
) -> tuple[List[ScoredPose], Union[NativeCertification, GapCertification, None]]:
    """
    Run a two-stage pose prediction pipeline.
    """
    request = nominalize_pipeline_request(
        request.with_updates(key=request.normalized_key)
    )
    if request.is_certified_mode and not request.optimize:
        raise ValueError(
            "certified mode requires optimize=True so the returned pose can carry a "
            "returned-pose RMSD contract"
        )
    route = derive_pipeline_route(request)
    phase_start = time.perf_counter()
    request = route.prepare_request(request)
    _runtime_profile_log("route.prepare_request", phase_start)
    if request.is_certified_mode and request.rigid_seed_family_plan is None:
        if request.n_poses_override is not None:
            request = request.with_updates(
                rigid_seed_family_plan=_derive_request_rigid_seed_family_plan(
                    request,
                    n_poses=request.n_poses_override,
                )
            )
    if (
        request.n_poses_override is None
        and request.seed_budget_plan is None
        and request.config is not None
    ):
        should_probe_seed_budget = (
            _force_seed_probe_enabled() or _request_uses_conformer_search(request)
        )
        if not should_probe_seed_budget:
            baseline_plan = derive_seed_budget_plan(
                confidence=request.config.confidence,
                box_size=request.box.size,
                target_rmsd=request.config.target_rmsd,
                ligand_radius=_ligand_radius(request.ligand_ctx),
                n_torsions=_seed_budget_torsion_count(request),
                rigid_seed_box=request.box,
                certified_binding_site=_resolved_certified_seed_binding_site(request),
            )
            request = request.with_updates(
                seed_budget_plan=baseline_plan,
                rigid_seed_family_plan=baseline_plan.selected_family_plan,
            )
        else:
            request, _ = _probe_seed_budget_certificate(request, route)
    if _request_uses_conformer_search(request):
        rotatable_bonds = _request_rotatable_bonds(request)
        if not _conformer_coverage_plan_matches_rotatable_bonds(
            request.conformer_coverage_plan,
            rotatable_bonds,
        ):
            if request.conformer_coverage_plan is not None:
                print(
                    "[CONFORMER COVERAGE] Refreshing request coverage plan: "
                    f"stored n_torsions={request.conformer_coverage_plan.n_torsions}, "
                    f"runtime rotatable_bonds={len(rotatable_bonds)}",
                    flush=True,
                )
        request = request.with_updates(
            conformer_coverage_plan=_derive_conformer_coverage_plan(
                request,
                rotatable_bonds=rotatable_bonds,
            )
        )
    phase_start = time.perf_counter()
    pose_batch = route.generate_pose_batch(request)
    _runtime_profile_log("route.generate_pose_batch", phase_start)
    request = pose_batch.request

    from dq_dock_engine.docking.placement import apply_poses

    phase_start = time.perf_counter()
    batched_coords = apply_poses(request.ligand_ctx, pose_batch.pose_vecs)
    _runtime_profile_log("apply_poses", phase_start)

    phase_start = time.perf_counter()
    initial_scores = route.score_pose_batch(
        request, batched_coords, pose_batch.pose_vecs
    )
    _runtime_profile_log("route.score_pose_batch", phase_start)
    execution_plan = initial_scores.execution_plan

    final_scores = initial_scores.final_scores

    best_indices = jnp.argsort(final_scores)[
        : route.best_index_limit(request, initial_scores)
    ]

    if not request.optimize:
        do_conf = _request_uses_conformer_search(request)
        scoring_context = (
            resolve_request_scoring_context(
                request,
                engine=request.effective_engine,
            ).ranking_context()
            if do_conf and request.is_certified_mode
            else None
        )
        electrostatics = (
            resolve_request_electrostatics(request, engine=request.effective_engine)
            if do_conf and not request.is_certified_mode
            else None
        )
        outputs = []
        for idx in best_indices:
            idx_i = int(idx)
            th_handles: tuple[str, ...] = ()
            coords_out = batched_coords[idx_i]
            energy_out = float(final_scores[idx_i])

            if do_conf:
                conf = _run_conformer_search_for_pose(
                    request,
                    quaternion=pose_batch.pose_vecs.quaternion[idx_i],
                    translation=pose_batch.pose_vecs.translation[idx_i],
                    scoring_context=scoring_context,
                    electrostatics=electrostatics,
                    pruning_incumbent_energy=float(
                        np.asarray(jax.device_get(jnp.min(final_scores)))
                    ),
                )
                if conf is not None and conf[1] < energy_out:
                    coords_out, energy_out, th_handles = conf

            outputs.append(
                ScoredPose(
                    coords=coords_out,
                    energy=energy_out,
                    engine=request.effective_engine,
                    theorem_handles=th_handles,
                )
            )
        cert = _compute_native_certification(
            config=request.config,
            protein_coords=request.protein_coords,
            coords=batched_coords,
            pre_opt_scores=final_scores,
            receptor_radii=request.receptor_radii,
            ligand_ctx=request.ligand_ctx,
            include_native=request.include_native,
        )
        if request.is_certified_mode:
            raise ValueError(
                "certified mode must not reach the non-optimized output path; "
                "returned-pose certification requires optimization/refinement witnesses"
            )
        return outputs, cert

    # A rigid energy-gap certificate is a valid fallback returned-pose contract,
    # but it is not by itself a theorem-backed reason to skip formal refinement.
    # Continue through local search/refinement and only use the rigid witness later
    # if stronger winner-side certification is unavailable.
    opt_translations, opt_quaternions, n_to_opt = route.optimization_inputs(
        request,
        pose_batch.pose_vecs,
        initial_scores,
    )

    pre_opt_scores = final_scores

    backend = request.formal_backend
    route.validate_backend(request, backend)

    scoring_context = resolve_request_scoring_context(
        request,
        engine=request.effective_engine,
    )
    ranking_scoring_context = scoring_context.ranking_context()
    # The formal local optimizer searches on the certified base physics
    # objective, while final pose scoring/ranking can still use the richer exact
    # chemistry context.
    optimization_scoring_context = scoring_context.optimization_context()
    electrostatics = scoring_context.electrostatics
    do_conf = _request_uses_conformer_search(request)
    opt_pose_translations = opt_translations
    opt_pose_quaternions = opt_quaternions
    opt_coords = apply_poses(
        request.ligand_ctx,
        PoseVector(translation=opt_pose_translations, quaternion=opt_pose_quaternions),
    )
    local_refine_indices = np.arange(int(opt_coords.shape[0]), dtype=np.int32)
    initial_ranked_exact_scores: jnp.ndarray | None = None
    binding_site_center: jnp.ndarray | None = None
    binding_site_radius = -1.0

    refinement_certificates: list[RefinementCertificate | None] = []
    rich_ranking_certified = False
    rigid_clearance_certified = True
    use_softened_local_objective = False
    winner_incumbent_filter_applied = False
    winner_global_index: int | None = None
    winner_pre_refined_score: float | None = None

    if backend == OptimizerBackend.FORMAL:
        from dq_dock_engine.docking.formal_optimizer import (
            _run_exact_formal_refinement,
            _run_singleton_hybrid_formal_refinement,
        )

        initial_opt_vecs = PoseVector(
            translation=opt_translations,
            quaternion=opt_quaternions,
        )
        if initial_scores.survivor_coords is not None:
            assert initial_scores.survivor_exact_scores is not None
            survivor_ranked = jnp.argsort(initial_scores.survivor_exact_scores)
            initial_coords = jnp.asarray(initial_scores.survivor_coords)[
                survivor_ranked
            ]
            initial_ranked_exact_scores = jnp.asarray(
                initial_scores.survivor_exact_scores
            )[survivor_ranked]
        else:
            initial_coords = apply_poses(request.ligand_ctx, initial_opt_vecs)
            if initial_scores.survivor_exact_scores is not None:
                initial_ranked_exact_scores = jnp.asarray(
                    initial_scores.survivor_exact_scores
                )
        rigid_refinement_plan = _derive_certified_rigid_local_refinement_plan(
            request, optimization_scoring_context
        )
        translation_cell_width = rigid_refinement_plan.translation_cell_width
        # Keep the formal local-search translation step tied to the certified pose
        # lattice itself. The current formal optimizer searches exact candidate
        # energies while softened scoring only appears in the coarse pruning lane;
        # inflating the local action family by the softened-Lipschitz speedup can
        # jump over narrow native basins (observed on 1hk4 known-conformer runs).
        # Use the half-cell exact lattice step here and leave PERF5-style scaling
        # to explicitly softened local search paths instead.
        base_step = translation_cell_width / 2.0
        # Use the canonical least positive joint dyadic refinement round from the
        # Lean support-expansion semantics: repeatedly halve both the translation
        # step and the radius-scaled rotation displacement until they are each at
        # most the target RMSD scale, and stop at the first such round.
        ligand_radius = rigid_refinement_plan.ligand_radius
        base_step = rigid_refinement_plan.base_translation_step
        base_rotation_step_rad = rigid_refinement_plan.base_rotation_step_rad
        n_search_rounds = rigid_refinement_plan.n_search_rounds
        binding_site = (
            None
            if request.certified_pocket_prep is None
            else cast(
                CertifiedPocketPreparation,
                request.certified_pocket_prep,
            ).plan.binding_site
        )
        binding_site_center = None if binding_site is None else binding_site.center
        binding_site_radius = -1.0 if binding_site is None else binding_site.radius
        local_improvement_bound = rigid_refinement_plan.local_improvement_bound
        initial_local_scores = np.asarray(
            jax.device_get(
                _score_exact_pose_batch(
                    request,
                    poses_coords=initial_coords,
                    electrostatics=electrostatics,
                    scoring_context=optimization_scoring_context,
                )
            ),
            dtype=np.float32,
        )
        (
            posewise_local_improvement_bounds,
            clearance_safe_mask,
            posewise_clearance,
        ) = _posewise_rigid_local_improvement_bounds(
            request,
            optimization_scoring_context,
            poses_coords=initial_coords,
            base_translation_step=base_step,
            base_rotation_step_rad=base_rotation_step_rad,
            ligand_radius=ligand_radius,
            n_rounds=n_search_rounds,
        )
        best_local_score = float(np.min(initial_local_scores))
        local_refine_mask = initial_local_scores <= (
            best_local_score + posewise_local_improvement_bounds
        )
        if request.is_certified_mode and not do_conf:
            local_refine_mask = np.logical_or(local_refine_mask, ~clearance_safe_mask)
            rigid_clearance_certified = bool(np.any(clearance_safe_mask))
            use_softened_local_objective = not rigid_clearance_certified
            print(
                "[LOCAL REFINEMENT DEBUG] "
                f"safe={int(np.count_nonzero(clearance_safe_mask))} "
                f"unsafe={int(np.count_nonzero(~clearance_safe_mask))} "
                f"bound_retained={int(np.count_nonzero(initial_local_scores <= (best_local_score + posewise_local_improvement_bounds)))} "
                f"combined_retained={int(np.count_nonzero(local_refine_mask))} "
                f"clearance_min={float(np.min(posewise_clearance)):.3f} "
                f"clearance_med={float(np.median(posewise_clearance)):.3f} "
                f"bound_max={float(np.max(posewise_local_improvement_bounds)):.3f}",
                flush=True,
            )
        if not np.any(local_refine_mask):
            local_refine_mask[int(np.argmin(initial_local_scores))] = True
        local_refine_indices = np.flatnonzero(local_refine_mask).astype(np.int32)
        if local_refine_indices.size < int(initial_coords.shape[0]):
            print(
                "[LOCAL REFINEMENT] "
                f"Retaining {int(local_refine_indices.size)}/{int(initial_coords.shape[0])} "
                f"poses (Delta_max={float(np.max(posewise_local_improvement_bounds)):.3f} kcal/mol)",
                flush=True,
            )
        elif request.is_certified_mode and not do_conf:
            print(
                "[LOCAL REFINEMENT] retaining all poses because every pose either passes the exact local bound "
                "or fails the LJ-clearance witness",
                flush=True,
            )
        if request.is_certified_mode:
            if execution_plan is None:
                raise ValueError(
                    "certified formal refinement requires an authoritative execution plan"
                )
            refinement_budget = CertifiedRefinementBudget(
                kind=CertifiedRefinementBudgetKind.DYADIC_ROUNDS,
                n_steps=n_search_rounds,
                pose_indices=tuple(int(i) for i in local_refine_indices.tolist()),
                theorem_handles=_merge_theorem_handles(
                    ("SH12", "SH13"),
                    joint_pruning_budget_optimality_handles(),
                    ("LJ20",) if request.is_certified_mode and not do_conf else (),
                    ("LJ19", "LJ21", "LJ22", "PERF6")
                    if use_softened_local_objective
                    else (),
                ),
                max_total_improvement=local_improvement_bound,
                note="Certified local dyadic refinement budget",
            )
            execution_plan = execution_plan.with_refinement_budget(
                refinement_budget,
                postfilter_cost_model=CertifiedPipelineCostModel(
                    input_pose_count=int(initial_coords.shape[0]),
                    retained_pose_count=int(initial_coords.shape[0]),
                    refinement_pose_count=int(local_refine_indices.size),
                    theorem_handles=joint_pruning_budget_optimality_handles(),
                ),
            )
            initial_scores = replace(initial_scores, execution_plan=execution_plan)
            local_refine_indices = np.asarray(
                execution_plan.refinement_pose_indices,
                dtype=np.int32,
            )
            local_refinement_budget = cast(
                CertifiedRefinementBudget,
                execution_plan.refinement_budget,
            )
            n_search_rounds = local_refinement_budget.n_steps
        refinement_input_coords = initial_coords[jnp.asarray(local_refine_indices)]

        refinement_kwargs: dict[str, object] = dict(
            coords_batch=refinement_input_coords,
            receptor_coords=request.protein_coords,
            receptor_radii=request.receptor_radii,
            ligand_radii=request.ligand_ctx.base_radii,
            n_rounds=n_search_rounds,
            target_error=request.target_error,
            coarse_target_error=request.coarse_target_error,
            adaptive_coarse_target_errors=request.adaptive_coarse_target_errors,
            # Keep the formal local optimizer on exact local-family scoring.
            # On 1hk4, softened local coarse scoring makes the ambiguity band so
            # wide that the singleton selector repeatedly chooses the noop action,
            # even when exact local candidates are much better than the incumbent.
            use_softened_exact=use_softened_local_objective,
            use_softened_coarse=use_softened_local_objective,
            base_translation_step=base_step,
            base_rotation_step_rad=base_rotation_step_rad,
            scoring_context=optimization_scoring_context,
            binding_site_center=binding_site_center,
            binding_site_radius=binding_site_radius,
        )
        formal_refiners = {
            FormalRoundStrategy.EXACT: _run_exact_formal_refinement,
            FormalRoundStrategy.SINGLETON_HYBRID: _run_singleton_hybrid_formal_refinement,
        }
        formal_refiner = formal_refiners[request.formal_round_strategy]
        winner_pre_refined_coords: jnp.ndarray | None = None

        if (
            request.is_certified_mode
            and not do_conf
            and not use_softened_local_objective
            and local_refine_indices.size > 1
        ):
            winner_global_index = int(
                local_refine_indices[
                    int(np.argmin(initial_local_scores[local_refine_indices]))
                ]
            )
            winner_input = initial_coords[winner_global_index : winner_global_index + 1]
            winner_refined_batch = formal_refiner(
                **(dict(refinement_kwargs) | {"coords_batch": winner_input})
            )
            winner_pre_refined_coords = winner_refined_batch[0]
            winner_refined_score = float(
                np.asarray(
                    jax.device_get(
                        _score_exact_pose_batch(
                            request,
                            poses_coords=winner_refined_batch,
                            electrostatics=electrostatics,
                            scoring_context=optimization_scoring_context,
                        )
                    ),
                    dtype=np.float32,
                )[0]
            )
            winner_pre_refined_score = winner_refined_score
            candidate_lower_bounds = (
                initial_local_scores[local_refine_indices]
                - posewise_local_improvement_bounds[local_refine_indices]
            )
            retain_mask = candidate_lower_bounds <= winner_refined_score
            winner_local_pos = int(
                np.flatnonzero(local_refine_indices == winner_global_index)[0]
            )
            retain_mask[winner_local_pos] = True
            if int(np.count_nonzero(retain_mask)) < local_refine_indices.size:
                old_count = int(local_refine_indices.size)
                local_refine_indices = local_refine_indices[retain_mask]
                refinement_input_coords = initial_coords[
                    jnp.asarray(local_refine_indices)
                ]
                winner_incumbent_filter_applied = True
                print(
                    "[LOCAL REFINEMENT DEBUG] "
                    f"winner_incumbent_filter retained={int(local_refine_indices.size)}/{old_count} "
                    f"winner_refined_score={winner_refined_score:.3f}",
                    flush=True,
                )

        phase_start = time.perf_counter()
        if winner_pre_refined_coords is not None and local_refine_indices.size == 1:
            refined_subset = winner_pre_refined_coords[None, ...]
        elif (
            winner_pre_refined_coords is not None
            and winner_global_index is not None
            and local_refine_indices.size > 1
        ):
            winner_local_pos = int(
                np.flatnonzero(local_refine_indices == winner_global_index)[0]
            )
            other_positions = np.array(
                [i for i in range(local_refine_indices.size) if i != winner_local_pos],
                dtype=np.int32,
            )
            other_coords = refinement_input_coords[jnp.asarray(other_positions)]
            if other_coords.shape[0] > 0:
                refined_others = formal_refiner(
                    **(dict(refinement_kwargs) | {"coords_batch": other_coords})
                )
            else:
                refined_others = jnp.empty(
                    (0,) + tuple(refinement_input_coords.shape[1:]),
                    dtype=refinement_input_coords.dtype,
                )
            assembled: list[jnp.ndarray | None] = [None] * int(
                local_refine_indices.size
            )
            winner_coords = cast(jnp.ndarray, winner_pre_refined_coords)
            assembled[winner_local_pos] = winner_coords
            for pos, coords in zip(
                other_positions.tolist(), np.asarray(refined_others), strict=False
            ):
                assembled[pos] = jnp.asarray(
                    coords, dtype=refinement_input_coords.dtype
                )
            refined_subset = jnp.stack(
                cast(list[jnp.ndarray], assembled),
                axis=0,
            )
        elif refinement_input_coords.shape[0] <= FORMAL_REFINEMENT_CHUNK_SIZE:
            refined_subset = formal_refiner(**refinement_kwargs)
        else:
            refined_chunks: list[jnp.ndarray] = []
            for start in range(
                0,
                int(refinement_input_coords.shape[0]),
                FORMAL_REFINEMENT_CHUNK_SIZE,
            ):
                stop = min(
                    start + FORMAL_REFINEMENT_CHUNK_SIZE,
                    int(refinement_input_coords.shape[0]),
                )
                chunk_kwargs = dict(refinement_kwargs)
                chunk_kwargs["coords_batch"] = refinement_input_coords[start:stop]
                refined_chunks.append(formal_refiner(**chunk_kwargs))
            refined_subset = jnp.concatenate(refined_chunks, axis=0)
        opt_coords = jnp.array(initial_coords)
        opt_coords = opt_coords.at[jnp.asarray(local_refine_indices)].set(
            refined_subset
        )
        opt_pose_translations = jnp.array(opt_translations)
        opt_pose_quaternions = jnp.array(opt_quaternions)
        refined_translations, refined_quaternions = _fit_pose_vectors_from_coords_batch(
            request.ligand_ctx.base_coords,
            refined_subset,
        )
        opt_pose_translations = opt_pose_translations.at[
            jnp.asarray(local_refine_indices)
        ].set(refined_translations)
        opt_pose_quaternions = opt_pose_quaternions.at[
            jnp.asarray(local_refine_indices)
        ].set(refined_quaternions)
        _runtime_profile_log("formal_local_refinement", phase_start)
        # FORMAL search proof complete — now add the refinement proof.
        # The Bayesian rounds certify basin selection; SE(3) refinement
        # certifies convergence to within ε RMSD of the basin minimum.
    # --- Certified SE(3) refinement witness ---
    # Even when conformer search is active, we still compute the rigid SE(3)
    # certificate for the pre-conformer rigid pose. If conformer search later
    # improves a pose, that rigid witness is explicitly invalidated for that pose.
    site_center = binding_site_center
    site_radius = binding_site_radius
    refinement_certificates = [None] * int(opt_coords.shape[0])
    refinement_failure_reasons: list[str | None] = [None] * int(opt_coords.shape[0])
    print(
        f"[REFINE_CERTS] local_refine_indices size: {local_refine_indices.size}, "
        f"opt_coords shape[0]: {opt_coords.shape[0]}, "
        f"winner index in refine: {0 in local_refine_indices}",
        flush=True,
    )
    subset_certificates_debug = None
    if local_refine_indices.size > 0 and not do_conf and _certify_all_refined_enabled():
        phase_start = time.perf_counter()
        pre_se3_translations = opt_pose_translations[jnp.asarray(local_refine_indices)]
        pre_se3_quaternions = opt_pose_quaternions[jnp.asarray(local_refine_indices)]
        pre_se3_coords = opt_coords[jnp.asarray(local_refine_indices)]
        subset_failure_reasons: list[str | None] = []
        opt_t_subset, opt_q_subset, subset_certificates = _certified_refinement(
            request=request,
            initial_translations=pre_se3_translations,
            initial_quaternions=pre_se3_quaternions,
            failure_reasons_out=subset_failure_reasons,
        )
        print(
            f"[REFINE_CERTS] After refinement: subset_certificates = {subset_certificates}, "
            f"len={len(subset_certificates) if subset_certificates else 'None'}",
            flush=True,
        )
        _runtime_profile_log("rigid_refinement_certificates", phase_start)
        subset_coords = apply_poses(
            request.ligand_ctx,
            PoseVector(translation=opt_t_subset, quaternion=opt_q_subset),
        )
        subset_in_site = np.ones((int(local_refine_indices.size),), dtype=bool)
        if site_center is not None and site_radius > 0.0:
            subset_centers = np.asarray(jax.device_get(jnp.mean(subset_coords, axis=1)))
            subset_in_site = np.linalg.norm(
                subset_centers - np.asarray(site_center),
                axis=1,
            ) <= float(site_radius)
            subset_coords = jnp.where(
                jnp.asarray(subset_in_site)[:, None, None],
                subset_coords,
                pre_se3_coords,
            )
            opt_t_subset = jnp.where(
                jnp.asarray(subset_in_site)[:, None],
                opt_t_subset,
                pre_se3_translations,
            )
            opt_q_subset = jnp.where(
                jnp.asarray(subset_in_site)[:, None],
                opt_q_subset,
                pre_se3_quaternions,
            )
        opt_coords = opt_coords.at[jnp.asarray(local_refine_indices)].set(subset_coords)
        opt_pose_translations = opt_pose_translations.at[
            jnp.asarray(local_refine_indices)
        ].set(opt_t_subset)
        opt_pose_quaternions = opt_pose_quaternions.at[
            jnp.asarray(local_refine_indices)
        ].set(opt_q_subset)
        for local_idx, pose_idx in enumerate(local_refine_indices.tolist()):
            if subset_in_site[local_idx]:
                cert = subset_certificates[local_idx]
                refinement_certificates[pose_idx] = cert
                refinement_failure_reasons[pose_idx] = subset_failure_reasons[local_idx]
                print(
                    f"[REFINE_CERTS] Assigned cert to pose_idx={pose_idx}: {cert is not None}, "
                    f"spectral={cert.spectral if cert else 'None'}, reason={subset_failure_reasons[local_idx]}",
                    flush=True,
                )
            else:
                refinement_failure_reasons[pose_idx] = (
                    "refined_pose_left_certified_site"
                )
    elif local_refine_indices.size > 0:
        print(
            "[REFINE_CERTS] Deferring rigid certificate computation to winner pose",
            flush=True,
        )

    refinement_cert_count = int(
        sum(cert is not None for cert in refinement_certificates)
    )
    print(
        "[PROOF_PLAN] Before proof plan: "
        f"refinement_cert_count={refinement_cert_count}/{len(refinement_certificates)}, "
        "winner would be index 0",
        flush=True,
    )

    orientation_margin_certified = False
    patched_rich_support_fast_path = False
    patched_rich_support_indices: tuple[int, ...] = ()
    if request.is_certified_mode:
        phase_start = time.perf_counter()
        if scoring_context.uses_extended_rich:
            if not do_conf and scoring_context.rich_chemistry_plan is not None:
                base_scores = _score_exact_pose_batch_padded(
                    request,
                    poses_coords=opt_coords,
                    electrostatics=electrostatics,
                    scoring_context=optimization_scoring_context,
                )
                witness_index = int(
                    np.argmin(np.asarray(jax.device_get(base_scores), dtype=np.float64))
                )
                witness_coords = opt_coords[witness_index : witness_index + 1]
                witness_rich_batch = scoring_context.score_exact_batch(
                    receptor_coords=request.protein_coords,
                    poses_coords=witness_coords,
                    receptor_radii=request.receptor_radii,
                    ligand_radii=request.ligand_ctx.base_radii,
                    target_error=request.target_error,
                    epsilon=0.2,
                )
                witness_disambiguation_batch = (
                    scoring_context.score_flip_disambiguation_batch(
                        receptor_coords=request.protein_coords,
                        poses_coords=witness_coords,
                        receptor_radii=request.receptor_radii,
                        ligand_radii=request.ligand_ctx.base_radii,
                        target_error=request.target_error,
                        epsilon=0.2,
                    )
                )
                witness_threshold = max(
                    float(
                        np.asarray(
                            jax.device_get(witness_rich_batch.scores), dtype=np.float32
                        )[0]
                    ),
                    float(
                        np.asarray(
                            jax.device_get(witness_disambiguation_batch.scores),
                            dtype=np.float32,
                        )[0]
                    ),
                )
                rich_omitted_bounds = _posewise_rich_channel_omission_bounds_fast(
                    request,
                    scoring_context.rich_chemistry_plan,
                    poses_coords=opt_coords,
                )
                rich_support_mask, _ = _retain_mask_for_explicit_threshold(
                    base_scores,
                    threshold=witness_threshold,
                    additive_correction=jnp.asarray(
                        rich_omitted_bounds, dtype=jnp.float32
                    ),
                )
                rich_support_indices_np = np.flatnonzero(
                    np.asarray(jax.device_get(rich_support_mask), dtype=bool)
                ).astype(np.int32)
                if 0 < rich_support_indices_np.size < int(opt_coords.shape[0]):
                    support_coords = opt_coords[jnp.asarray(rich_support_indices_np)]
                    rescoring_coords = support_coords
                    if 0 < support_coords.shape[0] < FINAL_EXACT_RESCORING_PAD_SIZE:
                        pad_count = int(
                            FINAL_EXACT_RESCORING_PAD_SIZE - support_coords.shape[0]
                        )
                        pad_coords = jnp.repeat(support_coords[:1], pad_count, axis=0)
                        rescoring_coords = jnp.concatenate(
                            (support_coords, pad_coords), axis=0
                        )
                    rich_final_batch = scoring_context.score_exact_batch(
                        receptor_coords=request.protein_coords,
                        poses_coords=rescoring_coords,
                        receptor_radii=request.receptor_radii,
                        ligand_radii=request.ligand_ctx.base_radii,
                        target_error=request.target_error,
                        epsilon=0.2,
                    )
                    disambiguation_batch = (
                        scoring_context.score_flip_disambiguation_batch(
                            receptor_coords=request.protein_coords,
                            poses_coords=rescoring_coords,
                            receptor_radii=request.receptor_radii,
                            ligand_radii=request.ligand_ctx.base_radii,
                            target_error=request.target_error,
                            epsilon=0.2,
                        )
                    )
                    rich_scores = rich_final_batch.scores[: support_coords.shape[0]]
                    disambiguation_scores = disambiguation_batch.scores[
                        : support_coords.shape[0]
                    ]
                    support_singleton = support_coords.shape[0] == 1
                    support_rich_singleton = support_singleton or _certified_top1_gap(
                        rich_scores,
                        float(np.asarray(jax.device_get(rich_final_batch.error_bound))),
                    )
                    support_dis_singleton = support_singleton or _certified_top1_gap(
                        disambiguation_scores,
                        float(
                            np.asarray(jax.device_get(disambiguation_batch.error_bound))
                        ),
                    )
                    rich_ranking_certified = (
                        int(jnp.argmin(rich_scores))
                        == int(jnp.argmin(disambiguation_scores))
                        and support_rich_singleton
                        and support_dis_singleton
                    )
                    orientation_margin_certified = (
                        support_dis_singleton
                        or _orientation_margin_certified_singleton_top1(
                            scoring_context,
                            disambiguation_scores,
                            float(
                                np.asarray(
                                    jax.device_get(disambiguation_batch.error_bound)
                                )
                            ),
                        )
                    )
                    chosen_scores = (
                        rich_scores if rich_ranking_certified else disambiguation_scores
                    )
                    chosen_error_bound = float(
                        np.asarray(
                            jax.device_get(
                                rich_final_batch.error_bound
                                if rich_ranking_certified
                                else disambiguation_batch.error_bound
                            )
                        )
                    )
                    fallback_score = witness_threshold + max(1.0e-6, chosen_error_bound)
                    final_scores_np = np.full(
                        (int(opt_coords.shape[0]),),
                        fallback_score,
                        dtype=np.float32,
                    )
                    final_scores_np[rich_support_indices_np] = np.asarray(
                        jax.device_get(chosen_scores), dtype=np.float32
                    )
                    final_scores = jnp.asarray(final_scores_np)
                    final_error_bound = chosen_error_bound
                    patched_rich_support_fast_path = True
                    patched_rich_support_indices = tuple(
                        int(i) for i in rich_support_indices_np.tolist()
                    )
                    print(
                        "[RICH SUPPORT FASTPATH] "
                        f"support={len(patched_rich_support_indices)}/{int(opt_coords.shape[0])} "
                        f"witness_idx={witness_index} threshold={witness_threshold:.3f} "
                        f"fallback={fallback_score:.3f}",
                        flush=True,
                    )
                else:
                    rescoring_coords = opt_coords
                    if 0 < opt_coords.shape[0] < FINAL_EXACT_RESCORING_PAD_SIZE:
                        pad_count = int(
                            FINAL_EXACT_RESCORING_PAD_SIZE - opt_coords.shape[0]
                        )
                        pad_coords = jnp.repeat(opt_coords[:1], pad_count, axis=0)
                        rescoring_coords = jnp.concatenate(
                            (opt_coords, pad_coords), axis=0
                        )
                    rich_final_batch = scoring_context.score_exact_batch(
                        receptor_coords=request.protein_coords,
                        poses_coords=rescoring_coords,
                        receptor_radii=request.receptor_radii,
                        ligand_radii=request.ligand_ctx.base_radii,
                        target_error=request.target_error,
                        epsilon=0.2,
                    )
                    disambiguation_batch = (
                        scoring_context.score_flip_disambiguation_batch(
                            receptor_coords=request.protein_coords,
                            poses_coords=rescoring_coords,
                            receptor_radii=request.receptor_radii,
                            ligand_radii=request.ligand_ctx.base_radii,
                            target_error=request.target_error,
                            epsilon=0.2,
                        )
                    )
                    rich_scores = rich_final_batch.scores[: opt_coords.shape[0]]
                    disambiguation_scores = disambiguation_batch.scores[
                        : opt_coords.shape[0]
                    ]
                    rich_ranking_certified = _shared_certified_singleton_top1(
                        rich_scores,
                        float(np.asarray(jax.device_get(rich_final_batch.error_bound))),
                        disambiguation_scores,
                        float(
                            np.asarray(jax.device_get(disambiguation_batch.error_bound))
                        ),
                    )
                    orientation_margin_certified = (
                        _orientation_margin_certified_singleton_top1(
                            scoring_context,
                            disambiguation_scores,
                            float(
                                np.asarray(
                                    jax.device_get(disambiguation_batch.error_bound)
                                )
                            ),
                        )
                    )
                    final_scores = (
                        rich_scores if rich_ranking_certified else disambiguation_scores
                    )
                    final_error_bound = float(
                        np.asarray(
                            jax.device_get(
                                rich_final_batch.error_bound
                                if rich_ranking_certified
                                else disambiguation_batch.error_bound
                            )
                        )
                    )
            else:
                rescoring_coords = opt_coords
                if 0 < opt_coords.shape[0] < FINAL_EXACT_RESCORING_PAD_SIZE:
                    pad_count = int(
                        FINAL_EXACT_RESCORING_PAD_SIZE - opt_coords.shape[0]
                    )
                    pad_coords = jnp.repeat(opt_coords[:1], pad_count, axis=0)
                    rescoring_coords = jnp.concatenate((opt_coords, pad_coords), axis=0)
                rich_final_batch = scoring_context.score_exact_batch(
                    receptor_coords=request.protein_coords,
                    poses_coords=rescoring_coords,
                    receptor_radii=request.receptor_radii,
                    ligand_radii=request.ligand_ctx.base_radii,
                    target_error=request.target_error,
                    epsilon=0.2,
                )
                disambiguation_batch = scoring_context.score_flip_disambiguation_batch(
                    receptor_coords=request.protein_coords,
                    poses_coords=rescoring_coords,
                    receptor_radii=request.receptor_radii,
                    ligand_radii=request.ligand_ctx.base_radii,
                    target_error=request.target_error,
                    epsilon=0.2,
                )
                rich_scores = rich_final_batch.scores[: opt_coords.shape[0]]
                disambiguation_scores = disambiguation_batch.scores[
                    : opt_coords.shape[0]
                ]
                rich_ranking_certified = _shared_certified_singleton_top1(
                    rich_scores,
                    float(np.asarray(jax.device_get(rich_final_batch.error_bound))),
                    disambiguation_scores,
                    float(np.asarray(jax.device_get(disambiguation_batch.error_bound))),
                )
                orientation_margin_certified = (
                    _orientation_margin_certified_singleton_top1(
                        scoring_context,
                        disambiguation_scores,
                        float(
                            np.asarray(jax.device_get(disambiguation_batch.error_bound))
                        ),
                    )
                )
                final_scores = (
                    rich_scores if rich_ranking_certified else disambiguation_scores
                )
                final_error_bound = float(
                    np.asarray(
                        jax.device_get(
                            rich_final_batch.error_bound
                            if rich_ranking_certified
                            else disambiguation_batch.error_bound
                        )
                    )
                )
        else:
            rich_ranking_certified = False
            orientation_margin_certified = False
            if (
                initial_ranked_exact_scores is not None
                and ranking_scoring_context.receptor_conformations is None
                and local_refine_indices.size > 0
            ):
                can_reuse_winner_refined_score = bool(
                    winner_pre_refined_score is not None
                    and winner_global_index is not None
                    and local_refine_indices.size == 1
                    and int(local_refine_indices[0]) == int(winner_global_index)
                )
                if can_reuse_winner_refined_score:
                    winner_idx_for_reuse = int(cast(int, winner_global_index))
                    winner_score_for_reuse = float(
                        cast(float, winner_pre_refined_score)
                    )
                    final_scores = (
                        jnp.asarray(initial_ranked_exact_scores)
                        .at[jnp.asarray([winner_idx_for_reuse])]
                        .set(jnp.asarray([winner_score_for_reuse], dtype=jnp.float32))
                    )
                    final_error_bound = 0.0
                else:
                    rescored_subset = ranking_scoring_context.score_exact_batch(
                        receptor_coords=request.protein_coords,
                        poses_coords=opt_coords[jnp.asarray(local_refine_indices)],
                        receptor_radii=request.receptor_radii,
                        ligand_radii=request.ligand_ctx.base_radii,
                        target_error=request.target_error,
                        epsilon=0.2,
                    )
                    final_scores = (
                        jnp.asarray(initial_ranked_exact_scores)
                        .at[jnp.asarray(local_refine_indices)]
                        .set(rescored_subset.scores)
                    )
                    final_error_bound = float(
                        np.asarray(jax.device_get(rescored_subset.error_bound))
                    )
            else:
                final_batch = _score_certified_batch_padded(
                    lambda coords: ranking_scoring_context.score_exact_batch(
                        receptor_coords=request.protein_coords,
                        poses_coords=coords,
                        receptor_radii=request.receptor_radii,
                        ligand_radii=request.ligand_ctx.base_radii,
                        target_error=request.target_error,
                        epsilon=0.2,
                    ),
                    opt_coords,
                )
                final_scores = final_batch.scores
                final_error_bound = float(
                    np.asarray(jax.device_get(final_batch.error_bound))
                )
        _runtime_profile_log("post_refinement_exact_scoring", phase_start)
    else:
        phase_start = time.perf_counter()
        final_scores = route_scoring(
            **derive_route_scoring_kwargs(
                request,
                engine=request.effective_engine,
                poses_coords=opt_coords,
                electrostatics=electrostatics,
            )
        )
        final_error_bound = None
        _runtime_profile_log("post_refinement_route_scoring", phase_start)

    final_scores = jnp.asarray(final_scores)
    pre_conformer_final_scores = final_scores

    phase_start = time.perf_counter()
    cert = _compute_native_certification(
        config=request.config,
        protein_coords=request.protein_coords,
        coords=opt_coords,
        pre_opt_scores=pre_opt_scores,
        receptor_radii=request.receptor_radii,
        ligand_ctx=request.ligand_ctx,
        include_native=request.include_native,
    )
    _runtime_profile_log("native_certification", phase_start)

    output_limit = 1 if patched_rich_support_fast_path else n_to_opt
    best_final_indices = jnp.argsort(final_scores)[:output_limit]

    runtime_diagnostic_handles: tuple[str, ...] = ()
    if winner_incumbent_filter_applied:
        runtime_diagnostic_handles = _merge_theorem_handles(
            runtime_diagnostic_handles,
            ("PERF8", "PERF9"),
        )
    if patched_rich_support_fast_path:
        runtime_diagnostic_handles = _merge_theorem_handles(
            runtime_diagnostic_handles,
            (
                "BCRP5",
                "BCRP9",
                "BCRP11",
                "BCRP12",
                "BCRP15",
                "BCRP16",
                "RPG11",
            ),
        )
    rigid_ambiguity_detected = False
    if (
        request.is_certified_mode
        and scoring_context.uses_extended_rich
        and not rich_ranking_certified
        and not orientation_margin_certified
    ):
        runtime_diagnostic_handles = _merge_theorem_handles(
            runtime_diagnostic_handles,
            (RUNTIME_ORIENTATION_SIGNAL_INACTIVE,),
        )
    if orientation_margin_certified:
        runtime_diagnostic_handles = _merge_theorem_handles(
            runtime_diagnostic_handles,
            _orientation_margin_theorem_handles(scoring_context),
        )
    if rich_ranking_certified:
        runtime_diagnostic_handles = _merge_theorem_handles(
            runtime_diagnostic_handles,
            ("TK16", "FLO21", "FLO22"),
        )
    if not rigid_clearance_certified:
        runtime_diagnostic_handles = _merge_theorem_handles(
            runtime_diagnostic_handles,
            (RUNTIME_RIGID_CLEARANCE_WITNESS_UNCERTIFIED, "LJ20", "BCPO2"),
        )
    rigid_ambiguity_detected = (
        request.is_certified_mode
        and not orientation_margin_certified
        and _detect_rigid_equivalence_ambiguity(
            opt_coords,
            final_scores,
            error_bound=final_error_bound,
            target_rmsd=request.target_rmsd,
        )
    )
    if rigid_ambiguity_detected:
        runtime_diagnostic_handles = _merge_theorem_handles(
            runtime_diagnostic_handles,
            (RUNTIME_RIGID_EQUIVALENCE_AMBIGUITY,),
        )

    rotatable_bonds: tuple[RotatableBond, ...] = ()
    conformer_config: BranchAndBoundConfig | None = None
    final_pose_improvement_family: PoseSpecificImprovementBudgetFamily | None = None
    atomwise_final_budget_totals: np.ndarray | None = None
    strain_params: TorsionStrainParams | None = None
    conformer_handles = ()
    if do_conf:
        phase_start = time.perf_counter()
        rotatable_bonds = _request_rotatable_bonds(request)
        _runtime_profile_log("final_conf_rotatable_bonds", phase_start)
        phase_start = time.perf_counter()
        conformer_config = _build_conformer_search_config(
            request, rotatable_bonds=rotatable_bonds
        )
        _runtime_profile_log("final_conf_build_config", phase_start)
        phase_start = time.perf_counter()
        per_bond_bounds = _conformer_local_improvement_bounds(
            request,
            rotatable_bonds,
            conformer_config.per_bond_lipschitz,
        )
        _runtime_profile_log("final_conf_per_bond_bounds", phase_start)
        phase_start = time.perf_counter()
        posewise_final_interaction_bounds = _posewise_conformer_interaction_bounds(
            request,
            opt_coords,
            rotatable_bonds,
            _per_bond_lipschitz_improvement_bounds(conformer_config.per_bond_lipschitz),
            electrostatics,
        )
        if (
            posewise_final_interaction_bounds is not None
            and per_bond_bounds is not None
        ):
            mmff_or_barrier_component = np.asarray(
                jax.device_get(per_bond_bounds),
                dtype=np.float32,
            ) - np.asarray(
                jax.device_get(
                    _per_bond_lipschitz_improvement_bounds(
                        conformer_config.per_bond_lipschitz
                    )
                ),
                dtype=np.float32,
            )
            per_bond_bounds = jnp.asarray(
                posewise_final_interaction_bounds + mmff_or_barrier_component[None, :],
                dtype=jnp.float32,
            )
        _runtime_profile_log("final_conf_local_interaction_bounds", phase_start)
        phase_start = time.perf_counter()
        final_pose_improvement_family = _posewise_improvement_budget_family(
            opt_coords,
            request.protein_coords,
            rotatable_bonds,
            per_bond_bounds,
            scoring_cutoff=compute_certified_cutoff(request.target_error),
            pose_indices=np.arange(int(opt_coords.shape[0]), dtype=np.int32),
            active_subset_source="final_pose_interaction_cutoff",
        )
        _runtime_profile_log("final_conf_improvement_family", phase_start)
        phase_start = time.perf_counter()
        atomwise_final_budget_totals = _posewise_atomwise_conformer_budget_totals(
            request,
            opt_coords,
            rotatable_bonds,
            per_bond_bounds,
            electrostatics,
        )
        _runtime_profile_log("final_conf_atomwise_budget_totals", phase_start)
        strain_params = None
        conformer_handles = _conformer_runtime_theorem_handles(
            has_rotatable_bonds=bool(rotatable_bonds),
            include_pocket_handles=request.certified_binding_site is not None,
            include_receptor_flex_handles=(
                scoring_context is not None
                and scoring_context.receptor_conformations is not None
            ),
        )
    best_energy_with_conf = (
        float(pre_conformer_final_scores[jnp.argmin(pre_conformer_final_scores)])
        if pre_conformer_final_scores.shape[0] > 0
        else float("inf")
    )
    per_pose_theorem_handles: list[tuple[str, ...]] | None = None
    conformer_improved_mask: list[bool] | None = None
    pose_basin_mu_coords: list[float | None] | None = None
    if do_conf:
        budget_totals = None
        if final_pose_improvement_family is not None:
            budget_totals = np.asarray(
                [
                    budget.total_budget
                    for budget in final_pose_improvement_family.budgets
                ],
                dtype=np.float64,
            )
            if atomwise_final_budget_totals is not None:
                budget_totals = np.minimum(
                    budget_totals,
                    np.asarray(atomwise_final_budget_totals, dtype=np.float64),
                )
        phase_start = time.perf_counter()
        active_torsion_counts = _posewise_active_torsion_counts(
            opt_coords,
            request.protein_coords,
            rotatable_bonds,
            scoring_cutoff=compute_certified_cutoff(request.target_error),
        )
        _runtime_profile_log("final_conf_active_torsion_counts", phase_start)
        if final_pose_improvement_family is None:
            exact_order = np.asarray(
                jax.device_get(pre_conformer_final_scores),
                dtype=np.float64,
            )
            ordered_pose_indices = np.lexsort(
                (exact_order, active_torsion_counts)
            ).astype(np.int32)
        else:
            lower_bound_order = np.asarray(
                jax.device_get(pre_conformer_final_scores),
                dtype=np.float64,
            ) - cast(np.ndarray, budget_totals)
            ordered_pose_indices = np.lexsort(
                (lower_bound_order, active_torsion_counts)
            ).astype(np.int32)
        resolved_coords = np.array(
            jax.device_get(opt_coords),
            dtype=np.float32,
            copy=True,
        )
        resolved_energies = np.array(
            jax.device_get(pre_conformer_final_scores),
            dtype=np.float64,
            copy=True,
        )
        coverage_handles = (
            ()
            if request.conformer_coverage_plan is None
            else request.conformer_coverage_plan.theorem_handles
        )
        per_pose_theorem_handles = [
            _merge_theorem_handles(
                conformer_handles,
                runtime_diagnostic_handles,
                coverage_handles,
            )
            for _ in range(int(opt_coords.shape[0]))
        ]
        conformer_improved_mask = [False] * int(opt_coords.shape[0])
        pose_basin_mu_coords = [
            cast(float | None, None) for _ in range(int(opt_coords.shape[0]))
        ]
        scan_start = time.perf_counter()
        scan_candidate_indices: list[int] = []
        scan_candidate_coords: list[jnp.ndarray] = []
        for idx_i in ordered_pose_indices.tolist():
            pose_conf_improvement_budget = (
                0.0 if budget_totals is None else float(budget_totals[idx_i])
            )
            if _runtime_profile_enabled():
                print(
                    "[RUNTIME PROFILE] final_conf_scan_candidate: "
                    f"pose={idx_i}, rigid_score={resolved_energies[idx_i]:.3f}, "
                    f"incumbent={best_energy_with_conf:.3f}, "
                    f"budget={pose_conf_improvement_budget:.3f}, "
                    f"active_torsions={active_torsion_counts[idx_i]}",
                    flush=True,
                )
            if (
                resolved_energies[idx_i]
                > best_energy_with_conf + pose_conf_improvement_budget
            ):
                continue
            if active_torsion_counts[idx_i] == 0 or pose_conf_improvement_budget <= 0.0:
                continue
            scan_conf = _run_conformer_search_for_pose(
                request,
                quaternion=opt_pose_quaternions[idx_i],
                translation=opt_pose_translations[idx_i],
                scoring_context=scoring_context,
                electrostatics=electrostatics,
                rotatable_bonds=rotatable_bonds,
                conformer_config=conformer_config,
                strain_params=strain_params,
                omission_budget=pose_conf_improvement_budget,
                scan_only=True,
            )
            if scan_conf is not None:
                scan_candidate_indices.append(idx_i)
                scan_candidate_coords.append(jnp.asarray(scan_conf[0]))
        if scan_candidate_coords:
            scan_coords_batch = jnp.stack(scan_candidate_coords)
            scan_exact_scores = _score_exact_pose_batch(
                request,
                poses_coords=scan_coords_batch,
                electrostatics=electrostatics,
                scoring_context=scoring_context,
            )
            scan_exact_np = np.asarray(
                jax.device_get(scan_exact_scores), dtype=np.float64
            )
            if scan_exact_np.size > 0:
                best_energy_with_conf = min(
                    best_energy_with_conf, float(np.min(scan_exact_np))
                )
        _runtime_profile_log("final_pose_conformer_scan", scan_start)
        for idx_i in ordered_pose_indices.tolist():
            pose_conf_improvement_budget = (
                0.0 if budget_totals is None else float(budget_totals[idx_i])
            )
            if _runtime_profile_enabled():
                print(
                    "[RUNTIME PROFILE] final_conf_full_candidate: "
                    f"pose={idx_i}, rigid_score={resolved_energies[idx_i]:.3f}, "
                    f"incumbent={best_energy_with_conf:.3f}, "
                    f"budget={pose_conf_improvement_budget:.3f}, "
                    f"active_torsions={active_torsion_counts[idx_i]}",
                    flush=True,
                )
            if (
                resolved_energies[idx_i]
                <= best_energy_with_conf + pose_conf_improvement_budget
            ):
                if (
                    active_torsion_counts[idx_i] == 0
                    or pose_conf_improvement_budget <= 0.0
                ):
                    continue
                conf = _run_conformer_search_for_pose(
                    request,
                    quaternion=opt_pose_quaternions[idx_i],
                    translation=opt_pose_translations[idx_i],
                    scoring_context=scoring_context,
                    electrostatics=electrostatics,
                    rotatable_bonds=rotatable_bonds,
                    conformer_config=conformer_config,
                    strain_params=strain_params,
                    pruning_incumbent_energy=float(best_energy_with_conf),
                    omission_budget=pose_conf_improvement_budget,
                )
                if conf is not None and conf[1] < resolved_energies[idx_i]:
                    conformer_improved_mask[idx_i] = True
                    print(
                        f"[CONFORMER] Before recertify call for pose {idx_i}",
                        flush=True,
                    )
                    (
                        recert_coords,
                        recert_energy,
                        recert_certificate,
                        recert_handles,
                        recert_basin_mu_coord,
                    ) = _recertify_conformer_updated_pose(
                        request,
                        jnp.asarray(conf[0]),
                        scoring_context=scoring_context,
                        electrostatics=electrostatics,
                    )
                    resolved_coords[idx_i] = np.asarray(
                        jax.device_get(recert_coords),
                        dtype=np.float32,
                    )
                    resolved_energies[idx_i] = float(recert_energy)
                    original_cert = refinement_certificates[idx_i]
                    if original_cert is not None:
                        refinement_certificates[idx_i] = original_cert
                        print(
                            f"[CONFORMER IMPROVE] Keeping original cert for pose {idx_i}: "
                            f"spectral={original_cert.spectral}, "
                            f"recert_basin_mu_coord={recert_basin_mu_coord}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[CONFORMER IMPROVE] No original cert for pose {idx_i}, "
                            f"recert_basin_mu_coord={recert_basin_mu_coord}",
                            flush=True,
                        )
                    assert pose_basin_mu_coords is not None
                    pose_basin_mu_coords[idx_i] = recert_basin_mu_coord
                    assert per_pose_theorem_handles is not None
                    per_pose_theorem_handles[idx_i] = _merge_theorem_handles(
                        per_pose_theorem_handles[idx_i],
                        conf[2],
                        recert_handles,
                    )
            best_energy_with_conf = min(best_energy_with_conf, resolved_energies[idx_i])
        opt_coords = jnp.asarray(resolved_coords, dtype=opt_coords.dtype)
        if request.is_certified_mode:
            if scoring_context.uses_extended_rich:
                rich_final_batch = scoring_context.score_exact_batch(
                    receptor_coords=request.protein_coords,
                    poses_coords=opt_coords,
                    receptor_radii=request.receptor_radii,
                    ligand_radii=request.ligand_ctx.base_radii,
                    target_error=request.target_error,
                    epsilon=0.2,
                )
                disambiguation_batch = scoring_context.score_flip_disambiguation_batch(
                    receptor_coords=request.protein_coords,
                    poses_coords=opt_coords,
                    receptor_radii=request.receptor_radii,
                    ligand_radii=request.ligand_ctx.base_radii,
                    target_error=request.target_error,
                    epsilon=0.2,
                )
                rich_ranking_certified = _shared_certified_singleton_top1(
                    rich_final_batch.scores,
                    float(np.asarray(jax.device_get(rich_final_batch.error_bound))),
                    disambiguation_batch.scores,
                    float(np.asarray(jax.device_get(disambiguation_batch.error_bound))),
                )
                final_batch = (
                    rich_final_batch if rich_ranking_certified else disambiguation_batch
                )
            else:
                rich_ranking_certified = False
                final_batch = ranking_scoring_context.score_exact_batch(
                    receptor_coords=request.protein_coords,
                    poses_coords=opt_coords,
                    receptor_radii=request.receptor_radii,
                    ligand_radii=request.ligand_ctx.base_radii,
                    target_error=request.target_error,
                    epsilon=0.2,
                )
            final_scores = final_batch.scores
            final_error_bound = float(
                np.asarray(jax.device_get(final_batch.error_bound))
            )
        else:
            final_scores = route_scoring(
                **derive_route_scoring_kwargs(
                    request,
                    engine=request.effective_engine,
                    poses_coords=opt_coords,
                    electrostatics=electrostatics,
                )
            )
            final_error_bound = None

    rigid_energy_gap_witness_for_skip = (
        None if do_conf else _derive_rigid_energy_gap_witness(request)
    )
    winner_refinement_failure_reason: str | None = None
    if request.is_certified_mode:
        winner_index = int(
            np.argmin(np.asarray(jax.device_get(final_scores), dtype=np.float64))
        )
        if (
            refinement_certificates[winner_index] is None
            and rigid_energy_gap_witness_for_skip is not None
        ):
            winner_refinement_failure_reason = (
                "skipped_due_to_rigid_energy_gap_certificate"
            )
            print(
                f"[REFINE_CERTS] Skipping winner-only rigid certification for pose_idx={winner_index} because a theorem-backed rigid energy-gap certificate is already available",
                flush=True,
            )
        elif refinement_certificates[winner_index] is None:
            phase_start = time.perf_counter()
            winner_failure_reasons: list[str | None] = []
            cert_coords, cert_energy, cert_obj, cert_t, cert_q = (
                _certify_single_rigid_pose(
                    request,
                    translation=opt_pose_translations[winner_index],
                    quaternion=opt_pose_quaternions[winner_index],
                    scoring_context=scoring_context,
                    electrostatics=electrostatics,
                    site_center=site_center,
                    site_radius=site_radius,
                    failure_reasons_out=winner_failure_reasons,
                )
            )
            refinement_certificates[winner_index] = cert_obj
            refinement_failure_reasons[winner_index] = (
                None if cert_obj is not None else winner_failure_reasons[0]
            )
            winner_refinement_failure_reason = refinement_failure_reasons[winner_index]
            opt_coords = jnp.asarray(opt_coords).at[winner_index].set(cert_coords)
            opt_pose_translations = (
                jnp.asarray(opt_pose_translations).at[winner_index].set(cert_t)
            )
            opt_pose_quaternions = (
                jnp.asarray(opt_pose_quaternions).at[winner_index].set(cert_q)
            )
            final_scores = jnp.asarray(final_scores).at[winner_index].set(cert_energy)
            print(
                f"[REFINE_CERTS] Winner-only certification pose_idx={winner_index}: {cert_obj is not None}, score={cert_energy:.3f}, reason={refinement_failure_reasons[winner_index]}",
                flush=True,
            )
            _runtime_profile_log("winner_rigid_certification", phase_start)
    best_final_indices = jnp.argsort(final_scores)[:n_to_opt]

    best_poses = []
    for idx in best_final_indices:
        idx_i = int(idx)
        coords_out = opt_coords[idx_i]
        energy_out = float(final_scores[idx_i])
        th_handles: tuple[str, ...] = (
            per_pose_theorem_handles[idx_i]
            if do_conf and per_pose_theorem_handles is not None
            else _merge_theorem_handles(
                conformer_handles if do_conf else (),
                runtime_diagnostic_handles,
            )
        )

        best_poses.append(
            ScoredPose(
                coords=coords_out,
                energy=energy_out,
                engine=request.effective_engine,
                theorem_handles=th_handles,
            )
        )

    if best_poses:
        winner_handles = _merge_theorem_handles(
            best_poses[0].theorem_handles,
            runtime_diagnostic_handles,
        )
        pipeline_debug_summary = (
            None if execution_plan is None else execution_plan.debug_summary()
        )
        if request.rigid_seed_family_plan is not None:
            if pipeline_debug_summary is None:
                pipeline_debug_summary = {}
            pipeline_debug_summary["rigid_seed_family_plan"] = (
                request.rigid_seed_family_plan.debug_summary()
            )
        if request.seed_budget_plan is not None:
            if pipeline_debug_summary is None:
                pipeline_debug_summary = {}
            pipeline_debug_summary["seed_budget_plan"] = (
                request.seed_budget_plan.debug_summary()
            )
        phase_start = time.perf_counter()
        returned_pose_proof_plan = _derive_returned_pose_proof_plan(
            request=request,
            final_scores=final_scores,
            final_error_bound=final_error_bound,
            final_pose_coords=opt_coords,
            refinement_certificates=refinement_certificates,
            winner_refinement_failure_reason=winner_refinement_failure_reason,
            pose_basin_mu_coords=(
                None if pose_basin_mu_coords is None else tuple(pose_basin_mu_coords)
            ),
            conformer_improved_mask=(
                None
                if conformer_improved_mask is None
                else tuple(conformer_improved_mask)
            ),
            conformer_coverage_plan=request.conformer_coverage_plan,
            winner_theorem_handles=winner_handles,
            do_conf=do_conf,
        )
        returned_pose_cert = _build_returned_pose_certification(
            proof_plan=returned_pose_proof_plan,
        )
        _runtime_profile_log("returned_pose_certification", phase_start)
        best_poses[0] = replace(
            best_poses[0],
            theorem_handles=(
                best_poses[0].theorem_handles
                if returned_pose_cert is None
                else _merge_theorem_handles(
                    best_poses[0].theorem_handles,
                    returned_pose_cert.theorem_handles,
                )
            ),
            returned_pose_certification=returned_pose_cert,
            returned_pose_proof_debug=returned_pose_proof_plan.debug_summary(),
            pipeline_debug_summary=pipeline_debug_summary,
        )
        if returned_pose_cert is not None:
            if (
                request.is_certified_mode
                and not returned_pose_cert.is_target_rmsd_certified
            ):
                tag = (
                    "[RETURNED POSE CERTIFIED] "
                    if returned_pose_cert.is_energy_gap_certified
                    else "[RETURNED POSE DOWNGRADE] "
                )
                print(
                    tag
                    + f"{returned_pose_cert.summary()} | proof_plan={returned_pose_proof_plan.debug_summary()}",
                    flush=True,
                )

    return best_poses, cert


def run_docking_pipeline_request(
    request: PipelineDockingRequest,
) -> tuple[List[ScoredPose], Union[NativeCertification, GapCertification, None]]:
    return _run_docking_pipeline_request(request)


def run_certified_blind_docking_request(
    request: CertifiedBlindDockingRequest,
) -> CertifiedBlindDockingResult:
    effective_request = request.with_updates(
        config=(
            DockingConfig(
                mode=DockingMode.CERTIFIED, optimizer_backend=OptimizerBackend.FORMAL
            )
            if request.config is None
            else request.config
        )
    )
    prep = _prepare_certified_blind_docking(
        CertifiedPreparationRequest.from_request(effective_request)
    )
    if not prep.plan.certified_pocket_found and prep.plan.binding_site is None:
        raise ValueError(
            "Certified blind docking could not derive a theorem-backed pocket/binding-site plan"
            f" ({prep.plan.certified_failure_reason.name if prep.plan.certified_failure_reason is not None else 'UNKNOWN'})."
        )
    poses, certification = run_docking_pipeline_request(
        derive_request(
            PipelineDockingRequest,
            effective_request,
            engine=ScoringEngine.CERTIFIED_LJ_REALSPACE_EWALD,
            use_pocket_guided=True,
            certified_pocket_prep=prep,
            scoring_kwargs=dict(effective_request.scoring_kwargs),
        )
    )
    return CertifiedBlindDockingResult(
        plan=prep.plan,
        poses=tuple(poses),
        certification=certification,
    )


def run_geometric_blind_docking_request(
    request: GeometricBlindDockingRequest,
) -> GeometricBlindDockingResult:
    prep = _prepare_geometric_blind_docking(
        GeometricPreparationRequest.from_request(request)
    )
    poses, _ = run_docking_pipeline_request(
        derive_request(
            PipelineDockingRequest,
            request,
            use_pocket_guided=True,
            scoring_kwargs=dict(request.scoring_kwargs),
        )
    )
    return GeometricBlindDockingResult(plan=prep.plan, poses=tuple(poses))


@dataclass(frozen=True)
class GeneratedRequestWrapperSpec:
    name: str
    request_type: type[DockingRequestBase]
    runner: Callable[[Any], Any]
    middle_positional_fields: tuple[str, ...] = ()
    signature_defaults: dict[str, object] = field(default_factory=dict)

    @property
    def positional_fields(self) -> tuple[str, ...]:
        return (
            "protein_coords",
            "receptor_radii",
            "ligand_ctx",
            "box",
            *self.middle_positional_fields,
            "key",
            "receptor_elements",
        )


def _build_request_wrapper_signature(
    spec: GeneratedRequestWrapperSpec,
) -> inspect.Signature:
    parameters: list[inspect.Parameter] = []
    ordered_fields = {
        field_info.name: field_info
        for field_info in dataclass_fields(spec.request_type)
        if field_info.init and field_info.name != "scoring_kwargs"
    }
    parameter_names = list(spec.positional_fields) + [
        name for name in ordered_fields if name not in spec.positional_fields
    ]
    for name in parameter_names:
        field_info = ordered_fields[name]
        default = inspect._empty
        if name in spec.signature_defaults:
            default = spec.signature_defaults[name]
        elif field_info.default is not MISSING:
            default = field_info.default
        elif field_info.default_factory is not MISSING:  # type: ignore[attr-defined]
            default = field_info.default_factory()  # type: ignore[misc]
        kind = (
            inspect.Parameter.POSITIONAL_OR_KEYWORD
            if name in spec.positional_fields
            else inspect.Parameter.KEYWORD_ONLY
        )
        parameters.append(
            inspect.Parameter(
                name,
                kind,
                default=default,
                annotation=field_info.type,
            )
        )
    parameters.append(
        inspect.Parameter("scoring_kwargs", inspect.Parameter.VAR_KEYWORD)
    )
    return inspect.Signature(
        parameters,
        return_annotation=inspect.signature(spec.runner).return_annotation,
    )


def _make_request_wrapper(spec: GeneratedRequestWrapperSpec) -> Callable[..., Any]:
    signature = _build_request_wrapper_signature(spec)

    def wrapper(*args: object, **kwargs: object) -> Any:
        bound = signature.bind(*args, **kwargs)
        request_kwargs = dict(bound.arguments)
        scoring_kwargs = request_kwargs.pop("scoring_kwargs", {})
        if "key" in request_kwargs:
            request_kwargs["key"] = _normalize_sampling_key(
                cast(jax.Array | None, request_kwargs["key"])
            )
        request_kwargs["scoring_kwargs"] = dict(cast(dict[str, object], scoring_kwargs))
        return spec.runner(derive_request(spec.request_type, request_kwargs))

    wrapper.__name__ = spec.name
    wrapper.__qualname__ = spec.name
    wrapper.__doc__ = (
        f"Auto-generated convenience wrapper for `{spec.request_type.__name__}`."
    )
    setattr(wrapper, "__signature__", signature)
    return wrapper


REQUEST_WRAPPER_SPECS = (
    GeneratedRequestWrapperSpec(
        name="run_docking_pipeline",
        request_type=PipelineDockingRequest,
        runner=run_docking_pipeline_request,
        signature_defaults={"key": None},
    ),
    GeneratedRequestWrapperSpec(
        name="run_certified_blind_docking",
        request_type=CertifiedBlindDockingRequest,
        runner=run_certified_blind_docking_request,
    ),
    GeneratedRequestWrapperSpec(
        name="run_geometric_blind_docking",
        request_type=GeometricBlindDockingRequest,
        runner=run_geometric_blind_docking_request,
        middle_positional_fields=("engine",),
        signature_defaults={"engine": inspect._empty},
    ),
)

globals().update(
    {spec.name: _make_request_wrapper(spec) for spec in REQUEST_WRAPPER_SPECS}
)


def _certified_refinement(
    request: "PipelineDockingRequest",
    initial_translations: jnp.ndarray,
    initial_quaternions: jnp.ndarray,
    *,
    mode_override: RefinementCertificationMode | None = None,
    failure_reasons_out: list[str | None] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, list[RefinementCertificate | None]]:
    """Certified SE(3) refinement: replaces fixed n_opt_steps with per-pose
    theorem-derived budgets.

    Both approaches start from the initial pose and run SE(3) GD in
    axis-angle parameterization:

      OBSERVED:     Fixed-α GD with scan-recorded energy trajectory →
                    empirical q from worst-case contraction ratio.
      CERTIFIED_GD: Two-phase (probe + Hessian-derived α and budget) →
                    analytic q = (lmax - lmin) / (lmax + lmin).

    Returns (optimized_translations, optimized_quaternions, certificates).
    The optimized poses come from the certified optimizer itself — this is
    "fixed-precision docking" not "fixed-cost docking".
    """
    from dq_dock_engine.docking.scoring import _EPSILON_KCAL_MOL

    config = request.config
    assert config is not None
    mode = config.refinement_certification if mode_override is None else mode_override
    target_rmsd = config.target_rmsd

    cutoff = jnp.array(compute_certified_cutoff(request.target_error))
    base_coords = request.ligand_ctx.base_coords
    receptor_coords = request.protein_coords
    receptor_radii = request.receptor_radii
    ligand_radii = request.ligand_ctx.base_radii
    ligand_radius = float(jnp.max(jnp.linalg.norm(base_coords, axis=-1)))
    n_atoms = base_coords.shape[0]

    refinement_scoring_context = None
    if request.is_certified_mode:
        refinement_scoring_context = resolve_request_scoring_context(
            request,
            engine=request.effective_engine,
        ).optimization_context()

    energy_fn = make_se3_energy_fn(
        base_coords,
        receptor_coords,
        receptor_radii,
        ligand_radii,
        cutoff,
        _EPSILON_KCAL_MOL,
        scoring_context=refinement_scoring_context,
        target_error=request.target_error,
    )
    kinematics_fn = make_se3_kinematics_fn(base_coords)

    from dq_dock_engine.docking.formal_actions import (
        least_positive_joint_adequate_dyadic_round,
    )

    translation_extent = (
        float(jnp.max(_rigid_seed_translation_support_size(request))) / 2.0
    )
    rotation_displacement_extent = ligand_radius * float(jnp.pi / 2.0)
    # Use the same canonical joint dyadic adequacy budget that drives the formal
    # local optimizer shells, rather than a fixed heuristic probe length.
    observed_probe_steps = least_positive_joint_adequate_dyadic_round(
        translation_extent,
        rotation_displacement_extent,
        target_rmsd,
    )

    n_poses = initial_translations.shape[0]
    opt_translations_list: list[jnp.ndarray] = []
    opt_quaternions_list: list[jnp.ndarray] = []
    certificates: list[RefinementCertificate | None] = []

    for i in range(n_poses):
        initial_params = _pose_to_se3_params(
            initial_translations[i],
            initial_quaternions[i],
        )

        if mode == RefinementCertificationMode.OBSERVED:
            optimized_params, cert, failure_reason = observe_gd_trajectory_with_reason(
                initial_params=initial_params,
                energy_fn=energy_fn,
                kinematics_fn=kinematics_fn,
                n_steps=observed_probe_steps,
                target_rmsd=target_rmsd,
                n_atoms=n_atoms,
                ligand_radius=ligand_radius,
            )
        elif mode == RefinementCertificationMode.CERTIFIED_GD:
            optimized_params, cert, failure_reason = optimize_certified_gd_with_reason(
                initial_params=initial_params,
                energy_fn=energy_fn,
                kinematics_fn=kinematics_fn,
                target_rmsd=target_rmsd,
                n_atoms=n_atoms,
                ligand_radius=ligand_radius,
                max_probe_steps=observed_probe_steps,
            )
        else:
            raise ValueError(f"Unexpected refinement certification mode: {mode}")

        t_opt, q_opt = _se3_params_to_pose(optimized_params)
        opt_translations_list.append(t_opt)
        opt_quaternions_list.append(q_opt)
        certificates.append(cert)
        if failure_reasons_out is not None:
            failure_reasons_out.append(None if cert is not None else failure_reason)

    return (
        jnp.stack(opt_translations_list),
        jnp.stack(opt_quaternions_list),
        certificates,
    )


def _compute_native_certification(
    config: DockingConfig | None,
    protein_coords: jnp.ndarray,
    coords: jnp.ndarray,
    pre_opt_scores: jnp.ndarray | np.ndarray,
    receptor_radii: jnp.ndarray,
    ligand_ctx: LigandContext,
    include_native: bool,
) -> Union[NativeCertification, GapCertification, None]:
    if config is None or config.mode != DockingMode.CERTIFIED:
        return None

    target_error = (
        config.target_error
        if config.target_error > 0
        else _derive_target_error_from_rmsd(
            config.target_rmsd,
            receptor_radii,
            ligand_ctx.base_radii,
            config.exact_chemistry_mode,
            config.softening_policy,
        )
    )
    _, error_bound = score_certified_lj(
        protein_coords,
        coords[:1],
        receptor_radii,
        ligand_ctx.base_radii,
        target_error=target_error,
    )
    error_bound = float(error_bound)

    # Convert to numpy once to avoid tracer issues in list comprehensions
    # We use jax.device_get() to pull values from the device concretely.
    try:
        pre_opt_scores_np = jax.device_get(pre_opt_scores)
    except Exception:
        # If we can't device_get, we are likely in a transformation where we shouldn't be anyway.
        # But for robustness in the benchmark, we'll try to convert.
        pre_opt_scores_np = np.asarray(pre_opt_scores)

    pre_scores_list = pre_opt_scores_np.tolist()
    best_energy = float(np.min(pre_opt_scores_np))

    if include_native:
        native_coords = ligand_ctx.base_coords + ligand_ctx.center_of_mass
        native_score_arr, _ = score_certified_lj(
            protein_coords,
            native_coords[None],
            receptor_radii,
            ligand_ctx.base_radii,
            target_error=target_error,
        )
        native_energy = float(native_score_arr[0])
        native_rank = int(np.sum(pre_opt_scores_np < native_energy)) + 1
        gap = abs(native_energy - best_energy)

        two_bound = 2 * error_bound
        decision = (
            CertificationDecision.CERTIFIED_BETTER
            if gap > two_bound
            else CertificationDecision.UNCERTIFIED
        )
        return NativeCertification(
            decision=decision,
            energy_gap=gap,
            error_bound=error_bound,
            native_rank=native_rank,
        )
    else:
        sorted_indices = sorted(
            range(len(pre_scores_list)), key=lambda i: pre_scores_list[i]
        )
        if len(sorted_indices) < 2:
            return None
        return GapCertification.from_energies(
            pre_scores_list[sorted_indices[0]],
            pre_scores_list[sorted_indices[1]],
            error_bound,
        )
