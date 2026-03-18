import jax.numpy as jnp
from dataclasses import dataclass

"""
Bounded Actions tractability case.
Direct translation of Tractability/BoundedActions.lean.

Key result: When |A| ≤ k (constant), SUFFICIENCY-CHECK runs in O(|S|² · k²).
"""

def bounded_actions_complexity(n_states: int, n_actions: int) -> int:
    """
    From BoundedActions.lean::totalCheckCost_le_pow:
      totalCheckCost ≤ |S|² × (1 + k²)
    """
    return n_states ** 2 * (1 + n_actions ** 2)

def is_bounded_actions(n_actions: int, bound: int = 100) -> bool:
    """
    Check if |A| is small enough for brute-force sufficiency check.
    From sufficiency_poly_bounded_actions: polynomial when k is constant.
    """
    return n_actions <= bound

@dataclass(frozen=True)
class BoundedActionsResult:
    """Result of bounded actions analysis."""
    n_actions: int
    n_states: int
    complexity: int
    is_tractable: bool  # |A| ≤ bound
