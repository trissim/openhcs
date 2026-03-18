import jax.numpy as jnp
from dataclasses import dataclass

"""
Thermodynamic Uncertainty Relations (TUR).
Direct translation of Physics/TUR.lean.

Key theorem (TUR1): Var(J) / ⟨J⟩² ≥ 2 / σ_Σ
Precision costs entropy production. Independent of Landauer.
"""

@dataclass(frozen=True)
class MarkovChain:
    """
    Discrete-time Markov chain on finite state space.
    From TUR.lean::DiscreteMarkovChain.
    """
    transition_matrix: jnp.ndarray  # P[s, s'] = P(s'|s)
    stationary_dist: jnp.ndarray    # π(s)
    
    def validate(self) -> bool:
        """Check normalization: rows sum to 1, π sums to 1."""
        row_sums = jnp.sum(self.transition_matrix, axis=1)
        pi_sum = jnp.sum(self.stationary_dist)
        return bool(
            jnp.allclose(row_sums, 1.0) and
            jnp.allclose(pi_sum, 1.0)
        )

def expected_value(pi: jnp.ndarray, J: jnp.ndarray) -> float:
    """⟨J⟩ = Σ_s π(s) × J(s). From TUR.lean::expectedValue."""
    return float(jnp.sum(pi * J))

def variance(pi: jnp.ndarray, J: jnp.ndarray) -> float:
    """Var(J) = Σ_s π(s) × (J(s) - ⟨J⟩)². From TUR.lean::variance."""
    mu = expected_value(pi, J)
    return float(jnp.sum(pi * (J - mu) ** 2))

def entropy_production(P: jnp.ndarray, pi: jnp.ndarray) -> float:
    """
    σ_Σ = Σ_{s,s'} π(s) P(s'|s) ln(P(s'|s) / P(s|s')).
    From TUR.lean::entropyProduction.
    Measures irreversibility. Zero iff detailed balance.
    """
    n = P.shape[0]
    sigma = 0.0
    for s in range(n):
        for sp in range(n):
            p_forward = P[s, sp]
            p_reverse = P[sp, s]
            if p_forward > 1e-30 and p_reverse > 1e-30:
                sigma += pi[s] * p_forward * jnp.log(p_forward / p_reverse)
    return float(sigma)

def tur_bound(
    pi: jnp.ndarray,
    J: jnp.ndarray,
    P: jnp.ndarray
) -> dict:
    """
    TUR1: Var(J) / ⟨J⟩² ≥ 2 / σ_Σ.
    From TUR.lean::tur_bound.
    
    Returns TUR statistics.
    """
    mu = expected_value(pi, J)
    var_J = variance(pi, J)
    sigma = entropy_production(P, pi)
    
    lhs = var_J / mu ** 2 if abs(mu) > 1e-30 else float('inf')
    rhs = 2.0 / sigma if sigma > 1e-30 else float('inf')
    
    return {
        "lhs": lhs,                    # Var(J)/⟨J⟩²
        "rhs": rhs,                    # 2/σ_Σ
        "satisfied": lhs >= rhs - 1e-10,  # TUR holds?
        "entropy_production": sigma,
        "mean": mu,
        "variance": var_J,
    }

def multiple_futures_check(P: jnp.ndarray) -> bool:
    """
    From multiple_futures_entropy_production:
    If forward ≠ reverse (P asymmetric), entropy is produced.
    """
    return not bool(jnp.allclose(P, P.T))
