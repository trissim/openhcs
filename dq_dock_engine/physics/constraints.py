import jax
import jax.numpy as jnp
from jax import jit
from .kernels import elementwise_binary_add, elementwise_binary_sub, distance

"""
SHAKE/RATTLE constraint dynamics.
Direct translation of GeometricConstraints.lean.

Structure:
  - PhaseSpace2D → (positions, velocities) arrays
  - verletHalf → unconstrained half-step
  - analyticLambdaV → Lagrange multiplier for velocity projection
  - rattleStep2D → full RATTLE constraint step
"""

@jit
def shake_position_project(
    q_unconstrained: jnp.ndarray,
    constraints: jnp.ndarray,
    target_distances: jnp.ndarray,
    masses: jnp.ndarray,
    max_iter: int = 100,
    tol: float = 1e-10
) -> jnp.ndarray:
    """
    SHAKE: project positions onto constraint manifold g(q) = 0.
    constraints: (n_constraints, 2) — pairs of atom indices.
    target_distances: (n_constraints,) — target bond lengths.
    
    Newton iteration to satisfy |q_i - q_j| = d_ij for all constraints.
    """
    q = q_unconstrained.copy()

    def body_fn(carry):
        q, _ = carry
        for c_idx in range(constraints.shape[0]):
            i, j = constraints[c_idx]
            r_ij = q[i] - q[j]
            current_dist_sq = jnp.sum(r_ij ** 2)
            target_sq = target_distances[c_idx] ** 2
            
            # Lagrange multiplier
            m_eff = 1.0 / masses[i] + 1.0 / masses[j]
            lam = (current_dist_sq - target_sq) / (2.0 * m_eff * current_dist_sq)
            
            # Apply correction
            correction = lam * r_ij
            q = q.at[i].add(-correction / masses[i])
            q = q.at[j].add(correction / masses[j])
        return q, None

    q, _ = jax.lax.fori_loop(0, max_iter, lambda i, carry: body_fn(carry), (q, None))
    return q

@jit
def rattle_velocity_project(
    q: jnp.ndarray,
    v: jnp.ndarray,
    constraints: jnp.ndarray,
    masses: jnp.ndarray
) -> jnp.ndarray:
    """
    RATTLE velocity projection: project velocities onto tangent bundle.
    Ensures bond_next · rel_v_next = 0 (from rattleStep2D in Lean).
    
    Direct translation of analyticLambdaV:
      lambda_V = (p1 - p2) / (2 * bond)
    Generalized to N-body constraints.
    """
    v_out = v.copy()
    for c_idx in range(constraints.shape[0]):
        i, j = constraints[c_idx]
        r_ij = q[i] - q[j]                         # bond vector
        v_ij = v_out[i] - v_out[j]                  # relative velocity
        
        # lambda_V = (v_ij · r_ij) / (r_ij · r_ij * (1/m_i + 1/m_j))
        m_eff = 1.0 / masses[i] + 1.0 / masses[j]
        lam_v = jnp.dot(v_ij, r_ij) / (jnp.dot(r_ij, r_ij) * m_eff)
        
        # Apply: v_i -= lam_v * r_ij / m_i, v_j += lam_v * r_ij / m_j
        v_out = v_out.at[i].add(-lam_v * r_ij / masses[i])
        v_out = v_out.at[j].add(lam_v * r_ij / masses[j])
    
    return v_out

def rattle_step(
    positions: jnp.ndarray,
    velocities: jnp.ndarray,
    masses: jnp.ndarray,
    force_fn,
    constraints: jnp.ndarray,
    target_distances: jnp.ndarray,
    dt: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Full RATTLE step: Verlet + SHAKE position projection + velocity projection.
    Preserves both position constraints and tangent bundle orthogonality.
    """
    m = masses[:, None]
    
    # Half-kick
    F = force_fn(positions)
    v_half = velocities + (dt / 2.0) * (F / m)
    
    # Drift (unconstrained)
    q_unconstrained = positions + dt * v_half
    
    # SHAKE: project positions
    q_next = shake_position_project(q_unconstrained, constraints, target_distances, masses)
    
    # Correct velocities for position projection
    v_corrected = (q_next - positions) / dt
    
    # Second half-kick
    F_next = force_fn(q_next)
    v_full = v_corrected + (dt / 2.0) * (F_next / m)
    
    # RATTLE: project velocities onto tangent bundle
    v_next = rattle_velocity_project(q_next, v_full, constraints, masses)
    
    return q_next, v_next
