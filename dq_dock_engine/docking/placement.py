"""
Pure JAX batched pose generation and geometric transformation.

Avoids Python loops by executing full sampling over `jax.vmap`.
"""

import jax
import jax.numpy as jnp
from dq_dock_engine.docking.core import DockingBox, LigandContext, PoseVector


def _uniform_quaternions(key: jax.Array, n: int) -> jnp.ndarray:
    """
    Shoemake's algorithm for uniform SO(3) sampling of quaternions.
    """
    k1, k2, k3 = jax.random.split(key, 3)
    u1 = jax.random.uniform(k1, shape=(n,))
    u2 = jax.random.uniform(k2, shape=(n,))
    u3 = jax.random.uniform(k3, shape=(n,))
    
    sqrt1_u1 = jnp.sqrt(1.0 - u1)
    sqrt_u1 = jnp.sqrt(u1)
    
    theta1 = 2 * jnp.pi * u2
    theta2 = 2 * jnp.pi * u3
    
    w = sqrt1_u1 * jnp.sin(theta1)
    x = sqrt1_u1 * jnp.cos(theta1)
    y = sqrt_u1 * jnp.sin(theta2)
    z = sqrt_u1 * jnp.cos(theta2)
    
    return jnp.stack([w, x, y, z], axis=-1)


def sample_random_poses(key: jax.Array, box: DockingBox, n_poses: int) -> PoseVector:
    """Pure geometric sampling within the constraints of the DockingBox."""
    key_t, key_r = jax.random.split(key)
    
    half_size = box.size / 2.0
    translations = jax.random.uniform(
        key_t, shape=(n_poses, 3),
        minval=box.center - half_size,
        maxval=box.center + half_size
    )
    
    quaternions = _uniform_quaternions(key_r, n_poses)
    return PoseVector(translation=translations, quaternion=quaternions)


@jax.jit
def _apply_single_pose(base_coords: jnp.ndarray, translation: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
    """
    Apply R*x + T to a single set of coordinates.
    base_coords: (N, 3)
    translation: (3,)
    q: (4,) quaternion [w, x, y, z]
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    
    # Rotation matrix from quaternion
    R = jnp.array([
        [1 - 2*y**2 - 2*z**2,   2*x*y - 2*z*w,       2*x*z + 2*y*w],
        [2*x*y + 2*z*w,         1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w,       1 - 2*x**2 - 2*y**2]
    ])
    
    # Apply R*x + T
    return (base_coords @ R.T) + translation


def apply_poses(context: LigandContext, poses: PoseVector) -> jnp.ndarray:
    """
    Applies translation and rotation over a batch of poses using vmap.
    
    Returns:
        jnp.ndarray: shape (N_poses, N_atoms, 3)
    """
    # Vectorize over poses (axis 0 of translation and quaternion).
    # LigandContext base_coords is identical for all poses (axis None).
    batched_apply = jax.vmap(_apply_single_pose, in_axes=(None, 0, 0))
    return batched_apply(context.base_coords, poses.translation, poses.quaternion)
