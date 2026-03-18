"""
Pure JAX RMSD metrics evaluation.

Implmements batched Kabsch algorithm for comparing generated poses
to native crystal structures.
"""

import jax
import jax.numpy as jnp


@jax.jit
def _kabsch_rmsd_single(coords_pose: jnp.ndarray, coords_native: jnp.ndarray) -> float:
    """
    Compute RMSD between two sets of coordinates using the Kabsch algorithm.
    Both should have shape (N, 3).
    """
    # 1. Center the coordinates
    centroid_p = jnp.mean(coords_pose, axis=0)
    centroid_n = jnp.mean(coords_native, axis=0)
    
    p_centered = coords_pose - centroid_p
    n_centered = coords_native - centroid_n
    
    # 2. Compute covariance matrix H
    H = p_centered.T @ n_centered
    
    # 3. SVD
    U, S, Vt = jnp.linalg.svd(H)
    
    # 4. Determine rotation matrix R
    R = U @ Vt
    
    # 5. Handle reflection case (det(R) < 0)
    # We use jnp.where to keep it JAX-differentiable/jittable
    det = jnp.linalg.det(R)
    
    # If det < 0, we flip the sign of the last column of U
    # Construct a reflection corrector matrix
    corrector = jnp.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, jnp.sign(det)]
    ])
    
    # Recompute R with corrector if needed
    R = U @ corrector @ Vt
    
    # 6. Apply rotation and compute RMSD
    p_rotated = p_centered @ R
    
    # Mean squared error
    diffs = p_rotated - n_centered
    mse = jnp.mean(jnp.sum(diffs**2, axis=1))
    
    return jnp.sqrt(mse)


@jax.jit
def compute_rmsd_batched(poses: jnp.ndarray, native: jnp.ndarray) -> jnp.ndarray:
    """
    Batched Kabsch RMSD calculation (structural similarity, optimally aligned).
    
    Args:
        poses: (N_poses, N_atoms, 3)
        native: (N_atoms, 3)
        
    Returns:
        (N_poses,) array of RMSD values in Angstroms
    """
    batched_rmsd = jax.vmap(_kabsch_rmsd_single, in_axes=(0, None))
    return batched_rmsd(poses, native)


@jax.jit
def _absolute_rmsd_single(coords_pose: jnp.ndarray, coords_native: jnp.ndarray) -> float:
    """Compute true docking RMSD (absolute Cartesian error without alignment)."""
    diffs = coords_pose - coords_native
    mse = jnp.mean(jnp.sum(diffs**2, axis=1))
    return jnp.sqrt(mse)

@jax.jit
def compute_docking_rmsd_batched(poses: jnp.ndarray, native: jnp.ndarray) -> jnp.ndarray:
    """
    Batched Docking RMSD calculation (absolute positional error).
    """
    batched_docking_rmsd = jax.vmap(_absolute_rmsd_single, in_axes=(0, None))
    return batched_docking_rmsd(poses, native)
