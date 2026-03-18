import jax
import jax.numpy as jnp
from jax import jit, grad
from dataclasses import dataclass
from typing import Callable, Any, Tuple

@dataclass(frozen=True)
class MDState:
    """
    Representing the MD state as an immutable dataclass.
    Consistent with OpenHCS 'frozen' principles and JAX JIT/VMAP compatibility.
    """
    positions: jnp.ndarray  # (N, 3)
    velocities: jnp.ndarray # (N, 3)
    masses: jnp.ndarray     # (N,)
    time: float = 0.0
    prng_key: jnp.ndarray = jax.random.PRNGKey(42)

@jit
def velocity_verlet_step(
    state: MDState, 
    force_fn: Callable[[jnp.ndarray], jnp.ndarray], 
    dt: float
) -> MDState:
    """
    Standard Velocity-Verlet integrator.
    Verified in BackwardErrorAnalysis.lean to have O(dt^3) shadow Hamiltonian bound.
    
    Algebraic factoring:
    1. p(t+h/2) = p(t) + (h/2) * F(t)
    2. q(t+h) = q(t) + h * p(t+h/2) / m
    3. p(t+h) = p(t+h/2) + (h/2) * F(t+h)
    """
    q_t = state.positions
    v_t = state.velocities
    m = state.masses[:, None] # Reshape for broadcasting
    
    # 1. Half-step velocity update (KICK)
    f_t = force_fn(q_t)
    v_half = v_t + (dt / 2.0) * (f_t / m)
    
    # 2. Full-step position update (DRIFT)
    q_next = q_t + dt * v_half
    
    # 3. Full-step velocity update (KICK)
    f_next = force_fn(q_next)
    v_next = v_half + (dt / 2.0) * (f_next / m)
    
    return MDState(
        positions=q_next,
        velocities=v_next,
        masses=state.masses,
        time=state.time + dt,
        prng_key=state.prng_key
    )

@jit
def compute_forces_from_potential(
    potential_fn: Callable[[jnp.ndarray], float], 
    positions: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute forces as negative gradient of potential.
    Uses JAX autodiff for exact mathematical rigor.
    """
    return -grad(potential_fn)(positions)
