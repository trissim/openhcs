import jax.numpy as jnp
from dataclasses import dataclass
from typing import Callable, Optional, Set

"""
Action Dominance detection.
Direct translation of Tractability/Dominance.lean.

Key results:
- Strict global dominance → empty set sufficient (srank = 0)
- Constant optimal set → empty set sufficient
- Weak dominance → dominant action always optimal
"""

def detect_strict_dominance(utility_matrix: jnp.ndarray) -> Optional[int]:
    """
    From Dominance.lean::hasStrictDominant.
    Check if any action strictly dominates all others at every state.
    
    utility_matrix[a, s] = U(a, s)
    Returns index of dominant action, or None.
    """
    n_actions = utility_matrix.shape[0]
    for a in range(n_actions):
        is_dominant = True
        for a2 in range(n_actions):
            if a2 == a:
                continue
            # Must strictly dominate: U(a, s) > U(a2, s) for all s
            if not bool(jnp.all(utility_matrix[a] > utility_matrix[a2])):
                is_dominant = False
                break
        if is_dominant:
            return a
    return None

def detect_constant_optimal_set(utility_matrix: jnp.ndarray) -> bool:
    """
    From ConstantOptimalSet.
    Check if optimal action set is the same for all states.
    """
    optimal_per_state = jnp.argmax(utility_matrix, axis=0)
    return bool(jnp.all(optimal_per_state == optimal_per_state[0]))

def detect_weak_dominance(utility_matrix: jnp.ndarray) -> Optional[int]:
    """
    From WeakGlobalDominance.
    One action is at least as good everywhere, and strictly better somewhere.
    """
    n_actions = utility_matrix.shape[0]
    for a in range(n_actions):
        weakly_dominates_all = True
        strictly_better_somewhere = False
        for a2 in range(n_actions):
            if a2 == a:
                continue
            if not bool(jnp.all(utility_matrix[a] >= utility_matrix[a2])):
                weakly_dominates_all = False
                break
            if bool(jnp.any(utility_matrix[a] > utility_matrix[a2])):
                strictly_better_somewhere = True
        if weakly_dominates_all and strictly_better_somewhere:
            return a
    return None

@dataclass(frozen=True)
class DominanceResult:
    """Result of dominance analysis."""
    strict_dominant: Optional[int]    # Strict global dominance
    weak_dominant: Optional[int]      # Weak global dominance
    constant_optimal: bool            # Constant optimal set
    srank_zero: bool                  # Any dominance → srank effectively 0
    
    @property
    def has_dominance(self) -> bool:
        return self.strict_dominant is not None or self.constant_optimal

def analyze_dominance(utility_matrix: jnp.ndarray) -> DominanceResult:
    """Full dominance analysis of a utility matrix."""
    strict = detect_strict_dominance(utility_matrix)
    weak = detect_weak_dominance(utility_matrix)
    constant = detect_constant_optimal_set(utility_matrix)
    
    return DominanceResult(
        strict_dominant=strict,
        weak_dominant=weak,
        constant_optimal=constant,
        srank_zero=(strict is not None or constant)
    )
