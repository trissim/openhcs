"""
End-to-End OpenHCS Pose Prediction Pipeline.

Ties together pure JAX batched generation and Enum-dispatched scoring
with multi-stage filtering and pocket-guided sampling.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np

from dq_dock_engine.docking.core import (
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
)
from dq_dock_engine.docking.charges import ChargeMethod, create_charge_assigner
from dq_dock_engine.docking.scoring import (
    CertifiedRealSpaceEwaldSpec,
    route_scoring,
    score_certified_lj,
)
from dq_dock_engine.docking.pocket_analysis import (
    CertifiedDetectedPocket,
    GeometricDetectedPocket,
    detect_certified_pocket,
    detect_geometric_pocket,
)
from dq_dock_engine.docking.pocket_sampling import (
    extract_local_pocket_region,
    extract_local_pocket_region_view,
)
from dq_dock_engine.docking.optimization import optimize_poses_batched
from dq_dock_engine.docking_config import (
    CertifiedScoringFamily,
    compute_certified_cutoff,
    DockingConfig,
    DockingMode,
    FormalRoundStrategy,
    OptimizerBackend,
)

if TYPE_CHECKING:
    from dq_dock_engine.docking.formal_sampling import CertifiedGlobalActionFamily


@dataclass(frozen=True)
class CertifiedPoseGeneration:
    pose_vecs: PoseVector
    family: "CertifiedGlobalActionFamily | None"


@dataclass(frozen=True)
class CertifiedPocketPreparation:
    protein_coords: jnp.ndarray
    receptor_radii: jnp.ndarray
    receptor_elements: tuple[str, ...] | None
    precomputed_receptor_charges: jnp.ndarray | None
    box: DockingBox
    detected_pocket: CertifiedDetectedPocket | None
    plan: CertifiedBlindDockingPlan


@dataclass(frozen=True)
class GeometricPocketPreparation:
    protein_coords: jnp.ndarray
    receptor_radii: jnp.ndarray
    receptor_elements: tuple[str, ...] | None
    precomputed_receptor_charges: jnp.ndarray | None
    box: DockingBox
    detected_pocket: GeometricDetectedPocket | None
    plan: GeometricBlindDockingPlan


def _ligand_extent_radius(ligand_ctx: LigandContext) -> float:
    centered = ligand_ctx.base_coords - ligand_ctx.center_of_mass
    if centered.shape[0] == 0:
        return 0.0
    return float(jnp.max(jnp.linalg.norm(centered, axis=1)))


def _apply_binding_site_restriction_impl(
    protein_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    receptor_elements: tuple[str, ...] | None,
    precomputed_receptor_charges: jnp.ndarray | None,
    ligand_ctx: LigandContext,
    box: DockingBox,
    binding_site: CertifiedBindingSite | GeometricBindingSite,
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


def _apply_certified_binding_site_restriction(
    protein_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    receptor_elements: tuple[str, ...] | None,
    precomputed_receptor_charges: jnp.ndarray | None,
    ligand_ctx: LigandContext,
    box: DockingBox,
    binding_site: CertifiedBindingSite,
    target_error: float,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    tuple[str, ...] | None,
    jnp.ndarray | None,
    DockingBox,
]:
    return _apply_binding_site_restriction_impl(
        protein_coords=protein_coords,
        receptor_radii=receptor_radii,
        receptor_elements=receptor_elements,
        precomputed_receptor_charges=precomputed_receptor_charges,
        ligand_ctx=ligand_ctx,
        box=box,
        binding_site=binding_site,
        target_error=target_error,
    )


def _apply_geometric_binding_site_restriction(
    protein_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    receptor_elements: tuple[str, ...] | None,
    precomputed_receptor_charges: jnp.ndarray | None,
    ligand_ctx: LigandContext,
    box: DockingBox,
    binding_site: GeometricBindingSite,
    target_error: float,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    tuple[str, ...] | None,
    jnp.ndarray | None,
    DockingBox,
]:
    return _apply_binding_site_restriction_impl(
        protein_coords=protein_coords,
        receptor_radii=receptor_radii,
        receptor_elements=receptor_elements,
        precomputed_receptor_charges=precomputed_receptor_charges,
        ligand_ctx=ligand_ctx,
        box=box,
        binding_site=binding_site,
        target_error=target_error,
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
) -> CertifiedPoseGeneration:
    from dq_dock_engine.docking.formal_sampling import (
        create_certified_binding_site_action_family,
        create_certified_global_action_family,
    )

    certified_family = (
        create_certified_global_action_family(box, n_poses)
        if certified_binding_site is None
        else create_certified_binding_site_action_family(
            certified_binding_site, n_poses
        )
    )
    pose_vecs = PoseVector(
        translation=certified_family.translations,
        quaternion=certified_family.quaternions,
    )
    return CertifiedPoseGeneration(pose_vecs=pose_vecs, family=certified_family)


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


def _prepare_certified_blind_docking(
    protein_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    receptor_elements: tuple[str, ...] | None,
    precomputed_receptor_charges: jnp.ndarray | None,
    ligand_ctx: LigandContext,
    box: DockingBox,
    target_error: float,
    certified_binding_site: CertifiedBindingSite | None,
    coarse_target_error: float = 0.004,
    adaptive_coarse_target_errors: tuple[float, ...] | None = None,
    use_softened_coarse_prefilter: bool = False,
) -> CertifiedPocketPreparation:
    detected_pocket: CertifiedDetectedPocket | None = None
    binding_site = certified_binding_site
    failure_reason: CertifiedPocketFailureReason | None = None
    if binding_site is None:
        detected_pocket, failure_reason = _derive_certified_binding_site_from_box(
            protein_coords=protein_coords,
            receptor_radii=receptor_radii,
            receptor_elements=receptor_elements,
            box=box,
        )
        binding_site = None if detected_pocket is None else detected_pocket.binding_site

    restricted_box = box
    restricted_coords = protein_coords
    restricted_radii = receptor_radii
    restricted_elements = receptor_elements
    restricted_charges = precomputed_receptor_charges
    theorem_handles: tuple[str, ...] = ()
    if binding_site is not None:
        (
            restricted_coords,
            restricted_radii,
            restricted_elements,
            restricted_charges,
            restricted_box,
        ) = _apply_certified_binding_site_restriction(
            protein_coords=protein_coords,
            receptor_radii=receptor_radii,
            receptor_elements=receptor_elements,
            precomputed_receptor_charges=precomputed_receptor_charges,
            ligand_ctx=ligand_ctx,
            box=box,
            binding_site=binding_site,
            target_error=target_error,
        )
        theorem_handles = binding_site.theorem_handles
    if detected_pocket is not None:
        theorem_handles = tuple(
            dict.fromkeys(detected_pocket.theorem_handles + theorem_handles)
        )

    plan = CertifiedBlindDockingPlan(
        binding_site=binding_site,
        restricted_box=restricted_box,
        restricted_atom_count=int(restricted_coords.shape[0]),
        certified_pocket_found=detected_pocket is not None,
        certified_failure_reason=failure_reason,
        coarse_target_error=coarse_target_error,
        adaptive_coarse_target_errors=adaptive_coarse_target_errors,
        use_softened_coarse_prefilter=use_softened_coarse_prefilter,
        theorem_handles=theorem_handles,
    )
    return CertifiedPocketPreparation(
        protein_coords=restricted_coords,
        receptor_radii=restricted_radii,
        receptor_elements=restricted_elements,
        precomputed_receptor_charges=restricted_charges,
        box=restricted_box,
        detected_pocket=detected_pocket,
        plan=plan,
    )


def _prepare_geometric_blind_docking(
    protein_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    receptor_elements: tuple[str, ...] | None,
    precomputed_receptor_charges: jnp.ndarray | None,
    ligand_ctx: LigandContext,
    box: DockingBox,
    target_error: float,
) -> GeometricPocketPreparation:
    detected_pocket = _derive_geometric_pocket_from_box(
        protein_coords=protein_coords,
        receptor_radii=receptor_radii,
        receptor_elements=receptor_elements,
        box=box,
    )
    binding_site = None if detected_pocket is None else detected_pocket.binding_site
    restricted_box = box
    restricted_coords = protein_coords
    restricted_radii = receptor_radii
    restricted_elements = receptor_elements
    restricted_charges = precomputed_receptor_charges
    if binding_site is not None:
        (
            restricted_coords,
            restricted_radii,
            restricted_elements,
            restricted_charges,
            restricted_box,
        ) = _apply_geometric_binding_site_restriction(
            protein_coords=protein_coords,
            receptor_radii=receptor_radii,
            receptor_elements=receptor_elements,
            precomputed_receptor_charges=precomputed_receptor_charges,
            ligand_ctx=ligand_ctx,
            box=box,
            binding_site=binding_site,
            target_error=target_error,
        )
    plan = GeometricBlindDockingPlan(
        binding_site=binding_site,
        restricted_box=restricted_box,
        restricted_atom_count=int(restricted_coords.shape[0]),
        sampling_strategy=SamplingStrategy.HYBRID,
    )
    return GeometricPocketPreparation(
        protein_coords=restricted_coords,
        receptor_radii=restricted_radii,
        receptor_elements=restricted_elements,
        precomputed_receptor_charges=restricted_charges,
        box=restricted_box,
        detected_pocket=detected_pocket,
        plan=plan,
    )


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


def run_docking_pipeline(
    protein_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    ligand_ctx: LigandContext,
    box: DockingBox,
    n_poses: int,
    engine: ScoringEngine,
    key: jax.Array,
    receptor_elements: tuple[str, ...] | None = None,
    *,
    charge_method: ChargeMethod | None = None,
    receptor_file: str | Path | None = None,
    precomputed_receptor_charges: jnp.ndarray | None = None,
    config: DockingConfig | None = None,
    top_k: int = 10,
    optimize: bool = True,
    n_opt_steps: int = 50,
    top_k_to_optimize: int = 200,
    use_pocket_guided: bool = True,
    use_multi_stage: bool = False,
    include_native: bool = False,
    **scoring_kwargs,
) -> tuple[List[ScoredPose], Union[NativeCertification, GapCertification, None]]:
    """
    Run a two-stage pose prediction pipeline:
    Stage 1: Fast global search (screening n_poses)
    Stage 2: Local refinement (optimizing top_k_to_optimize poses)

    Args:
        protein_coords: Receptor atom positions
        receptor_radii: Receptor VdW radii
        ligand_ctx: Ligand context
        box: Docking box constraints
        n_poses: Number of poses to generate
        engine: Base scoring engine (may be overridden by config)
        key: JAX random key
        top_k: Number of top poses to return
        optimize: Whether to run local optimization
        n_opt_steps: Optimization steps
        top_k_to_optimize: Top poses to optimize
        use_pocket_guided: Use pocket-guided sampling
        use_multi_stage: Use multi-stage filtering
        config: DockingConfig for CERTIFIED or HEURISTIC mode
        include_native: If True, compute certification against the native pose.
            This does not inject the native pose into the returned ranked poses.
    """
    # Determine effective engine based on config
    target_error = 0.001
    certified_family = None
    certified_binding_site = None
    certified_detected_pocket = None
    if config is not None and config.mode == DockingMode.CERTIFIED:
        if config.certified_scoring_family == CertifiedScoringFamily.LJ:
            effective_engine = ScoringEngine.CERTIFIED_LJ
        else:
            effective_engine = ScoringEngine.CERTIFIED_LJ_REALSPACE_EWALD
        target_error = config.target_error if config.target_error > 0 else 0.001
        scoring_kwargs["target_error"] = target_error
        certified_pocket_prep = _prepare_certified_blind_docking(
            protein_coords=protein_coords,
            receptor_radii=receptor_radii,
            receptor_elements=receptor_elements,
            precomputed_receptor_charges=precomputed_receptor_charges,
            ligand_ctx=ligand_ctx,
            box=box,
            target_error=target_error,
            certified_binding_site=config.certified_binding_site,
            coarse_target_error=config.coarse_target_error,
            adaptive_coarse_target_errors=config.adaptive_coarse_target_errors,
            use_softened_coarse_prefilter=config.use_softened_coarse_prefilter,
        )
        protein_coords = certified_pocket_prep.protein_coords
        receptor_radii = certified_pocket_prep.receptor_radii
        receptor_elements = certified_pocket_prep.receptor_elements
        precomputed_receptor_charges = (
            certified_pocket_prep.precomputed_receptor_charges
        )
        box = certified_pocket_prep.box
        certified_detected_pocket = certified_pocket_prep.detected_pocket
        certified_binding_site = certified_pocket_prep.plan.binding_site
    else:
        effective_engine = engine

    # --- POSE GENERATION ---
    certified_family = None
    if config is not None and config.mode == DockingMode.CERTIFIED:
        certified_generation = _create_certified_pose_vectors(
            box=box,
            n_poses=n_poses,
            certified_binding_site=certified_binding_site,
        )
        pose_vecs = certified_generation.pose_vecs
        certified_family = certified_generation.family
    elif use_pocket_guided:
        geometric_pocket_prep = _prepare_geometric_blind_docking(
            protein_coords=protein_coords,
            receptor_radii=receptor_radii,
            receptor_elements=receptor_elements,
            precomputed_receptor_charges=precomputed_receptor_charges,
            ligand_ctx=ligand_ctx,
            box=box,
            target_error=target_error,
        )
        protein_coords = geometric_pocket_prep.protein_coords
        receptor_radii = geometric_pocket_prep.receptor_radii
        receptor_elements = geometric_pocket_prep.receptor_elements
        precomputed_receptor_charges = (
            geometric_pocket_prep.precomputed_receptor_charges
        )
        box = geometric_pocket_prep.box
        geometric_detected_pocket = geometric_pocket_prep.detected_pocket
        key, pose_vecs = (
            _sample_geometric_pocket_guided_pose_vectors(
                key=key,
                n_poses=n_poses,
                geometric_detected_pocket=geometric_detected_pocket,
                ligand_ctx=ligand_ctx,
            )
            if geometric_detected_pocket is not None
            else _sample_box_guided_pose_vectors(
                key=key,
                box=box,
                n_poses=n_poses,
                protein_coords=protein_coords,
                receptor_elements=receptor_elements,
                ligand_ctx=ligand_ctx,
            )
        )
    else:
        from dq_dock_engine.docking.placement import sample_random_poses

        pose_vecs = sample_random_poses(key, box, n_poses)

    from dq_dock_engine.docking.placement import apply_poses

    batched_coords = apply_poses(ligand_ctx, pose_vecs)

    # --- MULTI-STAGE SCORING ---
    if use_multi_stage:
        from dq_dock_engine.docking.scoring_stages import (
            StageLevel,
            create_stage_calculator,
            create_pipeline,
            create_receptor_data,
        )
        from dq_dock_engine.docking.charges import create_charge_assigner, ChargeMethod

        # Require explicit charge method selection for multi-stage pipeline
        if charge_method is None:
            raise ValueError(
                "A ChargeMethod must be provided to run the multi-stage pipeline (no implicit fallback)."
            )

        # Use provided receptor_elements if available, otherwise fallback to carbon
        if receptor_elements is None:
            receptor_elements = tuple(["C"] * len(protein_coords))

        # Create assigner and assign receptor charges. For non-SIMPLE methods a receptor_file
        # (path or RDKit Mol) MUST be provided because RDKit/antechamber require a molecule input.
        assigner = create_charge_assigner(charge_method)
        if assigner.method == ChargeMethod.SIMPLE:
            receptor_charges = assigner.assign(receptor_elements).charges
        else:
            if receptor_file is None:
                raise ValueError(
                    f"ChargeMethod {assigner.method.name} requires a receptor file path or RDKit Mol to assign charges."
                )
            receptor_charges = assigner.assign(receptor_file).charges

        receptor_data = create_receptor_data(
            coords=protein_coords,
            radii=receptor_radii,
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
            ligand_radii=ligand_ctx.base_radii,
            ligand_charges=ligand_ctx.charges,
            ligand_elements=ligand_ctx.elements,
        )

        if validation is not None:
            print(
                f"Stage validation: Spearman 1-3={validation.spearman_1_3:.2f}, Top-10 overlap={validation.top10_overlap_1_3:.2f}"
            )

        final_scores = stage_results[-1].scores
    else:
        electrostatics = _resolve_route_scoring_electrostatics(
            effective_engine,
            ligand_ctx,
            receptor_elements,
            charge_method,
            receptor_file,
            precomputed_receptor_charges,
        )
        kwargs = {
            "receptor_coords": protein_coords,
            "receptor_radii": receptor_radii,
            "ligand_radii": ligand_ctx.base_radii,
            "poses_coords": batched_coords,
            "electrostatics": electrostatics,
            **scoring_kwargs,
        }
        final_scores = route_scoring(effective_engine, **kwargs)

    # --- SELECT TOP POSES ---
    best_indices = jnp.argsort(final_scores)[: min(top_k, n_poses)]

    if not optimize:
        outputs = []
        for idx in best_indices:
            idx_i = int(idx)
            outputs.append(
                ScoredPose(
                    coords=batched_coords[idx_i],
                    energy=float(final_scores[idx_i]),
                    engine=effective_engine,
                )
            )
        cert = _compute_native_certification(
            config=config,
            protein_coords=protein_coords,
            coords=batched_coords,
            pre_opt_scores=final_scores,
            receptor_radii=receptor_radii,
            ligand_ctx=ligand_ctx,
            include_native=include_native,
        )
        return outputs, cert

    # --- LOCAL OPTIMIZATION ---
    pre_opt_scores = final_scores

    from dq_dock_engine.docking.placement import apply_poses

    n_to_opt = min(top_k_to_optimize, n_poses)
    opt_indices = jnp.argsort(pre_opt_scores)[:n_to_opt]

    top_translations = pose_vecs.translation[opt_indices]
    top_quaternions = pose_vecs.quaternion[opt_indices]

    backend = (
        config.optimizer_backend if config is not None else OptimizerBackend.GRADIENT
    )

    if config is not None and config.mode == DockingMode.CERTIFIED:
        if backend != OptimizerBackend.FORMAL:
            raise ValueError(
                "CERTIFIED mode requires the formal optimizer backend; gradient refinement is heuristic."
            )

    if backend == OptimizerBackend.FORMAL:
        from dq_dock_engine.docking.formal_optimizer import (
            _run_exact_formal_refinement,
            _run_singleton_hybrid_formal_refinement,
        )

        initial_opt_vecs = PoseVector(
            translation=top_translations,
            quaternion=top_quaternions,
        )
        initial_coords = apply_poses(ligand_ctx, initial_opt_vecs)
        translation_cell_width = 1.0
        if certified_family is not None:
            translation_cell_width = float(jnp.min(box.size)) / float(
                certified_family.lattice_resolution
            )
        refinement_kwargs: dict[str, object] = dict(
            coords_batch=initial_coords,
            receptor_coords=protein_coords,
            receptor_radii=receptor_radii,
            ligand_radii=ligand_ctx.base_radii,
            n_rounds=n_opt_steps,
            target_error=target_error,
            coarse_target_error=(
                config.coarse_target_error if config is not None else 0.004
            ),
            adaptive_coarse_target_errors=(
                config.adaptive_coarse_target_errors if config is not None else None
            ),
            use_softened_coarse=(
                config.use_softened_coarse_prefilter if config is not None else False
            ),
            base_translation_step=translation_cell_width / 2.0,
            base_rotation_step_rad=float(jnp.pi / 2.0),
        )
        formal_electrostatics = _resolve_route_scoring_electrostatics(
            effective_engine,
            ligand_ctx,
            receptor_elements,
            charge_method,
            receptor_file,
            precomputed_receptor_charges,
        )
        refinement_kwargs["electrostatics"] = formal_electrostatics
        strategy = (
            config.formal_round_strategy
            if config is not None
            else FormalRoundStrategy.SINGLETON_HYBRID
        )
        formal_refiners = {
            FormalRoundStrategy.EXACT: _run_exact_formal_refinement,
            FormalRoundStrategy.SINGLETON_HYBRID: _run_singleton_hybrid_formal_refinement,
        }
        opt_coords = formal_refiners[strategy](**refinement_kwargs)
        opt_vecs = None
    else:
        opt_t, opt_q = optimize_poses_batched(
            translations=top_translations,
            quaternions=top_quaternions,
            ligand_base_coords=ligand_ctx.base_coords,
            receptor_coords=protein_coords,
            receptor_radii=receptor_radii,
            ligand_radii=ligand_ctx.base_radii,
            n_steps=n_opt_steps,
            lr_t=0.05,
            lr_q=0.05,
            config=config,
        )
        opt_vecs = PoseVector(translation=opt_t, quaternion=opt_q)
        opt_coords = apply_poses(ligand_ctx, opt_vecs)

    # --- FINAL SCORING ---
    electrostatics = _resolve_route_scoring_electrostatics(
        effective_engine,
        ligand_ctx,
        receptor_elements,
        charge_method,
        receptor_file,
        precomputed_receptor_charges,
    )
    kwargs = {
        "receptor_coords": protein_coords,
        "receptor_radii": receptor_radii,
        "ligand_radii": ligand_ctx.base_radii,
        "poses_coords": opt_coords,
        "electrostatics": electrostatics,
    }
    final_scores = route_scoring(effective_engine, **kwargs)

    # --- CERTIFICATION ---
    cert = _compute_native_certification(
        config=config,
        protein_coords=protein_coords,
        coords=opt_coords,
        pre_opt_scores=pre_opt_scores,
        receptor_radii=receptor_radii,
        ligand_ctx=ligand_ctx,
        include_native=include_native,
    )

    # --- RANKING ---
    best_final_indices = jnp.argsort(final_scores)[: min(top_k, n_to_opt)]

    best_poses = []
    for idx in best_final_indices:
        idx_i = int(idx)
        best_poses.append(
            ScoredPose(
                coords=opt_coords[idx_i],
                energy=float(final_scores[idx_i]),
                engine=effective_engine,
            )
        )

    return best_poses, cert


def run_certified_blind_docking(
    protein_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    ligand_ctx: LigandContext,
    box: DockingBox,
    n_poses: int,
    key: jax.Array,
    receptor_elements: tuple[str, ...] | None = None,
    *,
    charge_method: ChargeMethod | None = None,
    receptor_file: str | Path | None = None,
    precomputed_receptor_charges: jnp.ndarray | None = None,
    config: DockingConfig | None = None,
    top_k: int = 10,
    optimize: bool = True,
    n_opt_steps: int = 50,
    top_k_to_optimize: int = 200,
    include_native: bool = False,
    **scoring_kwargs,
) -> CertifiedBlindDockingResult:
    effective_config = (
        DockingConfig(
            mode=DockingMode.CERTIFIED, optimizer_backend=OptimizerBackend.FORMAL
        )
        if config is None
        else config
    )
    target_error = (
        effective_config.target_error if effective_config.target_error > 0 else 0.001
    )
    prep = _prepare_certified_blind_docking(
        protein_coords=protein_coords,
        receptor_radii=receptor_radii,
        receptor_elements=receptor_elements,
        precomputed_receptor_charges=precomputed_receptor_charges,
        ligand_ctx=ligand_ctx,
        box=box,
        target_error=target_error,
        certified_binding_site=effective_config.certified_binding_site,
        coarse_target_error=effective_config.coarse_target_error,
        adaptive_coarse_target_errors=effective_config.adaptive_coarse_target_errors,
        use_softened_coarse_prefilter=effective_config.use_softened_coarse_prefilter,
    )
    if not prep.plan.certified_pocket_found and prep.plan.binding_site is None:
        raise ValueError(
            "Certified blind docking could not derive a theorem-backed pocket/binding-site plan"
            f" ({prep.plan.certified_failure_reason.name if prep.plan.certified_failure_reason is not None else 'UNKNOWN'})."
        )
    poses, certification = run_docking_pipeline(
        protein_coords=protein_coords,
        receptor_radii=receptor_radii,
        ligand_ctx=ligand_ctx,
        box=box,
        n_poses=n_poses,
        engine=ScoringEngine.CERTIFIED_LJ_REALSPACE_EWALD,
        key=key,
        receptor_elements=receptor_elements,
        charge_method=charge_method,
        receptor_file=receptor_file,
        precomputed_receptor_charges=precomputed_receptor_charges,
        config=effective_config,
        top_k=top_k,
        optimize=optimize,
        n_opt_steps=n_opt_steps,
        top_k_to_optimize=top_k_to_optimize,
        use_pocket_guided=True,
        include_native=include_native,
        **scoring_kwargs,
    )
    return CertifiedBlindDockingResult(
        plan=prep.plan,
        poses=tuple(poses),
        certification=certification,
    )


def run_geometric_blind_docking(
    protein_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    ligand_ctx: LigandContext,
    box: DockingBox,
    n_poses: int,
    engine: ScoringEngine,
    key: jax.Array,
    receptor_elements: tuple[str, ...] | None = None,
    *,
    charge_method: ChargeMethod | None = None,
    receptor_file: str | Path | None = None,
    precomputed_receptor_charges: jnp.ndarray | None = None,
    config: DockingConfig | None = None,
    top_k: int = 10,
    optimize: bool = True,
    n_opt_steps: int = 50,
    top_k_to_optimize: int = 200,
    include_native: bool = False,
    **scoring_kwargs,
) -> GeometricBlindDockingResult:
    prep = _prepare_geometric_blind_docking(
        protein_coords=protein_coords,
        receptor_radii=receptor_radii,
        receptor_elements=receptor_elements,
        precomputed_receptor_charges=precomputed_receptor_charges,
        ligand_ctx=ligand_ctx,
        box=box,
        target_error=(
            config.target_error
            if config is not None and config.target_error > 0
            else 0.001
        ),
    )
    poses, _ = run_docking_pipeline(
        protein_coords=protein_coords,
        receptor_radii=receptor_radii,
        ligand_ctx=ligand_ctx,
        box=box,
        n_poses=n_poses,
        engine=engine,
        key=key,
        receptor_elements=receptor_elements,
        charge_method=charge_method,
        receptor_file=receptor_file,
        precomputed_receptor_charges=precomputed_receptor_charges,
        config=config,
        top_k=top_k,
        optimize=optimize,
        n_opt_steps=n_opt_steps,
        top_k_to_optimize=top_k_to_optimize,
        use_pocket_guided=True,
        include_native=include_native,
        **scoring_kwargs,
    )
    return GeometricBlindDockingResult(plan=prep.plan, poses=tuple(poses))


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

    target_error = config.target_error if config.target_error > 0 else 0.001
    _, error_bound = score_certified_lj(
        protein_coords,
        coords[:1],
        receptor_radii,
        ligand_ctx.base_radii,
        target_error=target_error,
    )
    error_bound = float(error_bound)

    pre_scores_list = [float(s) for s in pre_opt_scores]
    best_energy = min(pre_scores_list)

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
        native_rank = int((pre_opt_scores < native_energy).sum()) + 1
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
