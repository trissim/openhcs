"""
End-to-End OpenHCS Pose Prediction Pipeline.

Ties together pure JAX batched generation and Enum-dispatched scoring.
"""

from typing import List, Optional
import jax
import jax.numpy as jnp
import numpy as np

from dq_dock_engine.docking.core import DockingBox, LigandContext, ScoringEngine, ScoredPose, PoseVector
from dq_dock_engine.docking.placement import sample_random_poses, apply_poses
from dq_dock_engine.docking.scoring import route_scoring


from dq_dock_engine.docking.optimization import optimize_poses_batched

def run_docking_pipeline(
    protein_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    ligand_ctx: LigandContext,
    box: DockingBox,
    n_poses: int,
    engine: ScoringEngine,
    key: jax.Array,
    top_k: int = 10,
    optimize: bool = True,
    n_opt_steps: int = 50,
    top_k_to_optimize: int = 200,
    **scoring_kwargs
) -> List[ScoredPose]:
    """
    Run a two-stage pose prediction pipeline:
    Stage 1: Fast global search (screening n_poses)
    Stage 2: Local refinement (optimizing top_k_to_optimize poses)
    """
    # --- STAGE 1: GLOBAL SCREENING ---
    pose_vecs = sample_random_poses(key, box, n_poses)
    batched_coords = apply_poses(ligand_ctx, pose_vecs)
    
    # Initial Scoring
    kwargs = {
        'receptor_coords': protein_coords,
        'receptor_radii': receptor_radii,
        'ligand_radii': ligand_ctx.base_radii,
        'poses_coords': batched_coords,
        **scoring_kwargs
    }
    initial_energies = route_scoring(engine, **kwargs)
    
    if not optimize or engine != ScoringEngine.INTERNAL_LJ:
        # Return top_k from initial screening
        best_indices = jnp.argsort(initial_energies)[:min(top_k, n_poses)]
        outputs = []
        for idx in best_indices:
            idx_i = int(idx)
            outputs.append(ScoredPose(coords=batched_coords[idx_i], energy=float(initial_energies[idx_i]), engine=engine))
        return outputs

    # --- STAGE 2: LOCAL REFINEMENT ---
    n_to_opt = min(top_k_to_optimize, n_poses)
    screening_best_indices = jnp.argsort(initial_energies)[:n_to_opt]
    
    # Extract top poses for optimization
    top_translations = pose_vecs.translation[screening_best_indices]
    top_quaternions = pose_vecs.quaternion[screening_best_indices]
    
    # Gradient Descent Optimization (Top Poses Only)
    opt_t, opt_q = optimize_poses_batched(
        translations=top_translations,
        quaternions=top_quaternions,
        ligand_base_coords=ligand_ctx.base_coords,
        receptor_coords=protein_coords,
        receptor_radii=receptor_radii,
        ligand_radii=ligand_ctx.base_radii,
        n_steps=n_opt_steps,
        lr_t=0.05,
        lr_q=0.05
    )
    
    # Apply optimized transformations
    opt_vecs = PoseVector(translation=opt_t, quaternion=opt_q)
    opt_coords = apply_poses(ligand_ctx, opt_vecs)
    
    # Final Scoring
    final_energies = route_scoring(
        engine, 
        receptor_coords=protein_coords,
        receptor_radii=receptor_radii,
        ligand_radii=ligand_ctx.base_radii,
        poses_coords=opt_coords
    )
    
    # Ranking & Selection
    best_final_indices = jnp.argsort(final_energies)[:min(top_k, n_to_opt)]
    
    best_poses = []
    for idx in best_final_indices:
        idx_i = int(idx)
        pose = ScoredPose(
            coords=opt_coords[idx_i],
            energy=float(final_energies[idx_i]),
            engine=engine
        )
        best_poses.append(pose)
        
    return best_poses

