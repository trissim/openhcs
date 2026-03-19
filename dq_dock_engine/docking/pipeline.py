"""
End-to-End OpenHCS Pose Prediction Pipeline.

Ties together pure JAX batched generation and Enum-dispatched scoring
with multi-stage filtering and pocket-guided sampling.
"""

from pathlib import Path
from typing import List, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np

from dq_dock_engine.docking.core import (
    DockingBox,
    LigandContext,
    ScoringEngine,
    ScoredPose,
    PoseVector,
    GapCertification,
    NativeCertification,
    CertificationDecision,
)
from dq_dock_engine.docking.charges import ChargeMethod
from dq_dock_engine.docking.scoring import route_scoring, score_certified_lj
from dq_dock_engine.docking.optimization import optimize_poses_batched
from dq_dock_engine.docking_config import DockingConfig, DockingMode


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
    """
    # Determine effective engine based on config
    if config is not None and config.mode == DockingMode.CERTIFIED:
        effective_engine = ScoringEngine.CERTIFIED_LJ
        target_error = config.target_error if config.target_error > 0 else 0.001
        scoring_kwargs["target_error"] = target_error
    else:
        effective_engine = engine

    # --- POSE GENERATION ---
    if use_pocket_guided:
        from dq_dock_engine.docking.pocket_sampling import (
            sample_intelligent_poses,
            SamplingStrategy,
        )

        key_samp, key = jax.random.split(key)
        translations, quaternions = sample_intelligent_poses(
            key=key_samp,
            box_center=box.center,
            box_size=float(box.size[0]),
            n_poses=n_poses,
            protein_coords=protein_coords,
            ligand_com=ligand_ctx.center_of_mass,
            strategy=SamplingStrategy.HYBRID,
        )
        pose_vecs = PoseVector(translation=translations, quaternion=quaternions)
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
        kwargs = {
            "receptor_coords": protein_coords,
            "receptor_radii": receptor_radii,
            "ligand_radii": ligand_ctx.base_radii,
            "poses_coords": batched_coords,
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
                    engine=engine,
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
    kwargs = {
        "receptor_coords": protein_coords,
        "receptor_radii": receptor_radii,
        "ligand_radii": ligand_ctx.base_radii,
        "poses_coords": opt_coords,
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
                engine=engine,
            )
        )

    return best_poses, cert


def _compute_native_certification(
    config: DockingConfig | None,
    protein_coords: jnp.ndarray,
    coords: jnp.ndarray,
    pre_opt_scores: jnp.ndarray,
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
