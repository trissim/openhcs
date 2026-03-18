import jax
import jax.numpy as jnp
from typing import Callable, Optional
from enum import Enum, auto

from .engine import MDState, Integrator, VelocityVerlet, Langevin, hamiltonian
from .constraints import rattle_step

"""
Algorithm backends — dispatch targets for pipeline/router.py.
Each Tractability case maps to a run function via enum-driven dispatch.
"""

def run_md(
    state: MDState,
    integrator: Integrator,
    force_fn: Callable,
    dt: float,
    n_steps: int,
    constraints: Optional[jnp.ndarray] = None,
    target_distances: Optional[jnp.ndarray] = None
) -> MDState:
    """
    Core MD loop. All backends reduce to this.
    Optionally applies RATTLE constraints per step.
    """
    for _ in range(n_steps):
        if constraints is not None:
            q_next, v_next = rattle_step(
                state.positions, state.velocities, state.masses,
                force_fn, constraints, target_distances, dt
            )
            state = MDState(q_next, v_next, state.masses, state.time + dt, state.key)
        else:
            state = integrator.step(state, force_fn, dt)
    return state

def run_standard_md(
    state: MDState,
    force_fn: Callable,
    dt: float,
    n_steps: int,
    gamma: float = 1.0,
    temperature: float = 300.0,
    constraints: Optional[jnp.ndarray] = None,
    target_distances: Optional[jnp.ndarray] = None
) -> MDState:
    """HARD tractability: full NVT Langevin MD."""
    integrator = Langevin(gamma, temperature)
    return run_md(state, integrator, force_fn, dt, n_steps, constraints, target_distances)

def run_factorized(
    state: MDState,
    force_fn: Callable,
    dt: float,
    n_steps: int
) -> MDState:
    """TENSOR_RANK tractability: Verlet on reduced coordinates (low srank)."""
    integrator = VelocityVerlet()
    return run_md(state, integrator, force_fn, dt, n_steps)

def run_separable(state: MDState, force_fn: Callable) -> float:
    """SEPARABLE tractability: srank=0, utility is constant. Just evaluate."""
    return force_fn(state.positions)
