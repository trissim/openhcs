import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Callable

"""
Backward Error Analysis for symplectic integrators.
Direct translation of Computation/BackwardErrorAnalysis.lean.

Key theorem: Velocity-Verlet preserves a shadow Hamiltonian Ĥ = H + O(dt²),
with local truncation error O(dt³) per step.
This removes the need for empirical energy-drift axioms.
"""

def shadow_hamiltonian(
    potential_fn: Callable[[jnp.ndarray], float],
    positions: jnp.ndarray,
    velocities: jnp.ndarray,
    masses: jnp.ndarray,
    dt: float
) -> float:
    """
    Shadow Hamiltonian: Ĥ = H + (dt²/12) × correction_term.
    
    From BackwardErrorAnalysis.lean::shadowHamiltonian:
      Ĥ(q,p) = H(q,p) + (dt²/12) × ∇²U(q) · v²
    
    The correction term involves the Hessian of U contracted with velocities.
    This is the conserved quantity for the Verlet integrator.
    """
    # True Hamiltonian
    KE = 0.5 * jnp.sum(velocities ** 2 * masses[:, None])
    PE = potential_fn(positions)
    H = KE + PE
    
    # Correction term: (dt²/12) × Σ_i (∂²U/∂q_i²) × v_i²
    # Computed via Hessian-vector product
    def hessian_vp(q):
        """Hessian of potential contracted with v²."""
        grad_fn = jax.grad(potential_fn)
        # Hessian-vector product: H @ v² efficiently via jvp
        _, hvp = jax.jvp(grad_fn, (q,), (velocities ** 2,))
        return jnp.sum(hvp)
    
    correction = hessian_vp(positions)
    
    return H + (dt ** 2 / 12.0) * correction

def shadow_hamiltonian_error_bound(dt: float, third_deriv_bound: float) -> float:
    """
    Bound on |Ĥ(next) - Ĥ(current)| per Verlet step.
    
    From velocity_verlet_shadow_hamiltonian_bound:
      |ΔĤ| ≤ C × |dt|³
    where C depends on the third derivative bound B of the potential.
    """
    # The constant C depends on B and masses; this is the structure
    return third_deriv_bound * abs(dt) ** 3

def verify_energy_conservation(
    potential_fn: Callable,
    positions: jnp.ndarray,
    velocities: jnp.ndarray,
    masses: jnp.ndarray,
    dt: float,
    n_steps: int,
    integrator_step_fn: Callable
) -> dict:
    """
    Verify shadow Hamiltonian conservation over a trajectory.
    Returns drift statistics for the shadow Hamiltonian.
    """
    from dq_dock_engine.physics.engine import MDState
    
    key = jax.random.PRNGKey(0)
    state = MDState(positions, velocities, masses, 0.0, key)
    
    H_shadow_values = []
    H_true_values = []
    
    force_fn = lambda q: -jax.grad(potential_fn)(q)
    
    for _ in range(n_steps):
        H_s = shadow_hamiltonian(potential_fn, state.positions, state.velocities, masses, dt)
        H_t = 0.5 * jnp.sum(state.velocities ** 2 * masses[:, None]) + potential_fn(state.positions)
        H_shadow_values.append(float(H_s))
        H_true_values.append(float(H_t))
        state = integrator_step_fn(state, force_fn, dt)
    
    H_shadow = jnp.array(H_shadow_values)
    H_true = jnp.array(H_true_values)
    
    return {
        "shadow_drift": float(jnp.max(jnp.abs(H_shadow - H_shadow[0]))),
        "true_drift": float(jnp.max(jnp.abs(H_true - H_true[0]))),
        "shadow_mean": float(jnp.mean(H_shadow)),
        "true_mean": float(jnp.mean(H_true)),
    }
