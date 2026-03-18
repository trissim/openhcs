"""
End-to-End OpenHCS Pose Prediction Pipeline.

Ties together pure JAX batched generation and Enum-dispatched scoring.
"""

from typing import List, Optional
import jax
import jax.numpy as jnp
import numpy as np

from dq_dock_engine.docking.core import DockingBox, LigandContext, ScoringEngine, ScoredPose
from dq_dock_engine.docking.placement import sample_random_poses, apply_poses
from dq_dock_engine.docking.scoring import route_scoring


def run_docking_pipeline(
    protein_coords: jnp.ndarray,
    ligand_ctx: LigandContext,
    box: DockingBox,
    n_poses: int,
    engine: ScoringEngine,
    key: jax.Array,
    top_k: int = 10,
    **scoring_kwargs
) -> List[ScoredPose]:
    """
    Run the complete pose prediction pipeline.
    
    Args:
        protein_coords: (N_rec, 3) array of receptor coordinates
        ligand_ctx: Immunable LigandContext
        box: DockingBox defining search space
        n_poses: Number of batched poses to sample
        engine: ScoringEngine to use
        key: JAX PRNG key
        top_k: Number of best poses to return
        scoring_kwargs: Additional arguments required by the chosen scoring engine.
            (e.g., receptor_file and ligand_template for SMINA_EXACT)
            
    Returns:
        List of the top_k ScoredPose objects, sorted by best (lowest) energy.
    """
    # 1. State Generation (Pure JAX, Batched)
    pose_vecs = sample_random_poses(key, box, n_poses)
    batched_coords = apply_poses(ligand_ctx, pose_vecs)  # shape: (n_poses, N_lig, 3)
    
    # 2. Scoring (Strict Enum Dispatch)
    kwargs = {
        'receptor_coords': protein_coords,
        'poses_coords': batched_coords,
        **scoring_kwargs
    }
    energies = route_scoring(engine, **kwargs)  # shape: (n_poses,)
    
    # 3. Ranking & Selection
    energies_jnp = jnp.asarray(energies)
    
    # Handle cases where we asked for more poses than we generated
    actual_k = min(top_k, n_poses)
    
    # Get indices of the lowest energies
    best_indices = jnp.argsort(energies_jnp)[:actual_k]
    
    # 4. Construct output
    best_poses = []
    # Convert back to numpy for easy downstream usage in lists, 
    # but keep JAX arrays internally if we want
    for idx in best_indices:
        idx_val = int(idx)
        pose = ScoredPose(
            coords=batched_coords[idx_val],
            energy=float(energies[idx_val]),
            engine=engine
        )
        best_poses.append(pose)
        
    return best_poses
