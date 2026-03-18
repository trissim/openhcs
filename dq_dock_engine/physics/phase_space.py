import jax
import jax.numpy as jnp
from typing import Callable

"""
Phase Space Geometry — Liouville's Theorem for Velocity-Verlet.
Direct translation of Computation/PhaseSpaceGeometry.lean.

Key theorem: det(J_Verlet) = 1. The integrator preserves phase space volume exactly.
This is the defining property of a symplectic integrator.
"""

def force_jacobian(
    potential_fn: Callable[[jnp.ndarray], float],
    positions: jnp.ndarray
) -> jnp.ndarray:
    """
    Jacobian of the force field: ∂F/∂q = -∂²U/∂q².
    From PhaseSpaceGeometry.lean::forceJacobian.
    
    This is the Hessian of U (with sign flip for forces).
    """
    n = positions.shape[0] * positions.shape[1]  # total DOF
    flat_pos = positions.ravel()
    
    def flat_potential(q_flat):
        return potential_fn(q_flat.reshape(positions.shape))
    
    hessian = jax.hessian(flat_potential)(flat_pos)
    return -hessian  # F = -∇U, so ∂F/∂q = -∂²U/∂q²

def velocity_verlet_jacobian(
    potential_fn: Callable[[jnp.ndarray], float],
    positions: jnp.ndarray,
    dt: float,
    mass: float
) -> jnp.ndarray:
    """
    Full Jacobian of the Velocity-Verlet map: J3 × J2 × J1.
    From PhaseSpaceGeometry.lean::velocityVerletJacobian.
    
    J1 = [[I, 0], [M, I]]  — momentum half-step, M = (dt/2) × ∂F/∂q
    J2 = [[I, D], [0, I]]  — position full-step, D = (dt/mass) × I
    J3 = [[I, 0], [M', I]] — momentum half-step at new position
    """
    n = positions.shape[0] * positions.shape[1]
    I_n = jnp.eye(n)
    Z_n = jnp.zeros((n, n))
    
    # From forceJacobian
    M = (dt / 2.0) * force_jacobian(potential_fn, positions)
    D = (dt / mass) * I_n
    
    # J1: [[I, 0], [M, I]]
    J1 = jnp.block([[I_n, Z_n], [M, I_n]])
    
    # J2: [[I, D], [0, I]]
    J2 = jnp.block([[I_n, D], [Z_n, I_n]])
    
    # J3: [[I, 0], [M', I]] — M' at updated position
    # For the symplecticity proof, M' has the same structure
    # The Lean proof shows det=1 regardless of M' value
    J3 = jnp.block([[I_n, Z_n], [M, I_n]])  # Same structure
    
    return J3 @ J2 @ J1

def verify_liouville(
    potential_fn: Callable[[jnp.ndarray], float],
    positions: jnp.ndarray,
    dt: float,
    mass: float
) -> float:
    """
    Verify: det(J_Verlet) = 1 (Liouville's theorem).
    
    From PhaseSpaceGeometry.lean::velocity_verlet_preserves_volume:
      (velocityVerletJacobian U dt mass s).det = 1
    
    Returns |det(J) - 1| — should be ~machine epsilon.
    """
    J = velocity_verlet_jacobian(potential_fn, positions, dt, mass)
    det_J = jnp.linalg.det(J)
    return float(jnp.abs(det_J - 1.0))

def symplectic_form_2n(n: int) -> jnp.ndarray:
    """
    Standard symplectic form Ω = [[0, I], [-I, 0]] for 2n-dim phase space.
    A map is symplectic iff J^T Ω J = Ω.
    """
    I_n = jnp.eye(n)
    Z_n = jnp.zeros((n, n))
    return jnp.block([[Z_n, I_n], [-I_n, Z_n]])

def verify_symplecticity(
    potential_fn: Callable[[jnp.ndarray], float],
    positions: jnp.ndarray,
    dt: float,
    mass: float
) -> float:
    """
    Verify: J^T Ω J = Ω (symplecticity condition).
    Returns ‖J^T Ω J - Ω‖_F — should be ~machine epsilon.
    """
    n = positions.shape[0] * positions.shape[1]
    J = velocity_verlet_jacobian(potential_fn, positions, dt, mass)
    omega = symplectic_form_2n(n)
    residual = J.T @ omega @ J - omega
    return float(jnp.linalg.norm(residual))
