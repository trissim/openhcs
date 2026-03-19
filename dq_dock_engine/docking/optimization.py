"""
Local optimization of poses using JAX automatic differentiation.
"""

import jax
import jax.numpy as jnp

from dq_dock_engine.docking.core import LigandContext, PoseVector
from dq_dock_engine.docking.placement import _apply_single_pose
from dq_dock_engine.docking.scoring import _score_single_lj, _score_certified_lj
from dq_dock_engine.docking_config import DockingConfig, DockingMode


def _pose_loss_heuristic(
    translation: jnp.ndarray,
    quaternion: jnp.ndarray,
    base_coords: jnp.ndarray,
    receptor_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
) -> jnp.ndarray:
    q_norm = quaternion / jnp.linalg.norm(quaternion)
    pose_coords = _apply_single_pose(base_coords, translation, q_norm)
    return _score_single_lj(receptor_coords, pose_coords, receptor_radii, ligand_radii)


def _pose_loss_certified(
    translation: jnp.ndarray,
    quaternion: jnp.ndarray,
    base_coords: jnp.ndarray,
    receptor_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    cutoff: jnp.ndarray,
    epsilon: float,
) -> jnp.ndarray:
    q_norm = quaternion / jnp.linalg.norm(quaternion)
    pose_coords = _apply_single_pose(base_coords, translation, q_norm)
    energy, _ = _score_certified_lj(
        receptor_coords, pose_coords, receptor_radii, ligand_radii, cutoff, epsilon
    )
    return energy


def _step_body(curr_t, curr_q, lr_t, lr_q, *loss_extra):
    energy, (grad_t, grad_q) = jax.value_and_grad(
        _pose_loss_heuristic, argnums=(0, 1), has_aux=False
    )(curr_t, curr_q, *loss_extra)

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


def _step_body_certified(curr_t, curr_q, lr_t, lr_q, lc, rc, rr, lr, cutoff, epsilon):
    energy, (grad_t, grad_q) = jax.value_and_grad(
        _pose_loss_certified, argnums=(0, 1), has_aux=False
    )(curr_t, curr_q, lc, rc, rr, lr, cutoff, epsilon)

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
    use_certified: bool,
    cutoff: jnp.ndarray | None,
    epsilon: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    if use_certified:

        def body_fn(i, val):
            return _step_body_certified(
                val[0],
                val[1],
                lr_t,
                lr_q,
                ligand_base_coords,
                receptor_coords,
                receptor_radii,
                ligand_radii,
                cutoff,
                epsilon,
            )

        return jax.lax.fori_loop(0, n_steps, body_fn, (t, q))
    else:

        def body_fn(i, val):
            return _step_body(
                val[0],
                val[1],
                lr_t,
                lr_q,
                ligand_base_coords,
                receptor_coords,
                receptor_radii,
                ligand_radii,
            )

        return jax.lax.fori_loop(0, n_steps, body_fn, (t, q))


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
    config: DockingConfig | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    if config is not None and config.mode == DockingMode.CERTIFIED:
        target_error = config.target_error if config.target_error > 0 else 0.001
        from dq_dock_engine.physics.lattice_sum import optimal_cutoff

        cutoff = jnp.array(optimal_cutoff(target_error, s=6.0))
        use_certified = True
        epsilon = 0.086
    else:
        cutoff = None
        use_certified = False
        epsilon = 0.0

    def batch_fn(args):
        t, q = args
        return _optimize_single(
            t,
            q,
            ligand_base_coords,
            receptor_coords,
            receptor_radii,
            ligand_radii,
            n_steps,
            lr_t,
            lr_q,
            use_certified,
            cutoff,
            epsilon,
        )

    batched_optimize = jax.vmap(batch_fn, in_axes=((0, 0),))
    opt_t, opt_q = batched_optimize((translations, quaternions))

    return opt_t, opt_q
