"""
Local optimization of poses using JAX automatic differentiation.
"""

import jax
import jax.numpy as jnp

from dq_dock_engine.docking.core import LigandContext, PoseVector
from dq_dock_engine.docking.placement import _apply_single_pose
from dq_dock_engine.docking.scoring import _score_single_lj


def _pose_loss_fn(
    translation: jnp.ndarray,
    quaternion: jnp.ndarray,
    base_coords: jnp.ndarray,
    receptor_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
) -> jnp.ndarray:
    """
    Computes the energy of a single pose given its transformation parameters.
    """
    q_norm = quaternion / jnp.linalg.norm(quaternion)
    pose_coords = _apply_single_pose(base_coords, translation, q_norm)
    energy = _score_single_lj(
        receptor_coords, pose_coords, receptor_radii, ligand_radii
    )
    return energy


def _optimize_single(
    t: jnp.ndarray,
    q: jnp.ndarray,
    ligand_base_coords: jnp.ndarray,
    receptor_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    n_steps: int,
    lr_t: float,
    lr_q: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Optimize a single pose using gradient descent."""
    value_and_grad_fn = jax.value_and_grad(_pose_loss_fn, argnums=(0, 1))

    def step_fn(i, val):
        curr_t, curr_q = val
        energy, (grad_t, grad_q) = value_and_grad_fn(
            curr_t,
            curr_q,
            ligand_base_coords,
            receptor_coords,
            receptor_radii,
            ligand_radii,
        )

        grad_t_norm = jnp.linalg.norm(grad_t)
        grad_q_norm = jnp.linalg.norm(grad_q)

        grad_t_dir = jnp.where(
            grad_t_norm > 1e-6, grad_t / grad_t_norm, jnp.zeros_like(grad_t)
        )
        grad_q_dir = jnp.where(
            grad_q_norm > 1e-6, grad_q / grad_q_norm, jnp.zeros_like(grad_q)
        )

        step_t = jnp.minimum(grad_t_norm * lr_t, 0.1) * grad_t_dir
        step_q = jnp.minimum(grad_q_norm * lr_q, 0.05) * grad_q_dir

        next_t = curr_t - step_t
        next_q = curr_q - step_q
        next_q = next_q / jnp.linalg.norm(next_q)

        return next_t, next_q

    final_t, final_q = jax.lax.fori_loop(0, n_steps, step_fn, (t, q))
    return final_t, final_q


_optimize_single_jit = jax.jit(_optimize_single, static_argnames=["n_steps"])


def optimize_poses_batched(
    translations: jnp.ndarray,
    quaternions: jnp.ndarray,
    ligand_base_coords: jnp.ndarray,
    receptor_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    n_steps: int = 50,
    lr_t: float = 0.05,
    lr_q: float = 0.05,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Optimize a batch of poses using JAX gradient descent.

    Args:
        translations: (N, 3) translations
        quaternions: (N, 4) quaternions
        ligand_base_coords: Ligand coordinates centered at origin
        receptor_coords: Protein coordinates
        receptor_radii: (N_rec,) VdW radii
        ligand_radii: (N_lig,) VdW radii
        n_steps: Number of gradient descent steps
        lr_t: Learning rate for translation
        lr_q: Learning rate for quaternion (rotation)

    Returns:
        tuple containing optimized (translations, quaternions)
    """

    def batch_fn(args):
        t, q = args
        return _optimize_single_jit(
            t,
            q,
            ligand_base_coords,
            receptor_coords,
            receptor_radii,
            ligand_radii,
            n_steps,
            lr_t,
            lr_q,
        )

    batched_optimize = jax.vmap(batch_fn, in_axes=((0, 0),))
    opt_t, opt_q = batched_optimize((translations, quaternions))

    return opt_t, opt_q
