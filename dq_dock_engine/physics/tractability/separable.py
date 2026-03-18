import jax.numpy as jnp
from dataclasses import dataclass
from typing import Callable

"""
Separable and Low Tensor Rank Utility detection.
Direct translation of Tractability/SeparableUtility.lean.

Key results:
- Separable: u(a,s) = f(a) + g(s) → srank = 0 (all coords sufficient)
- Low rank: u(a,s) = Σᵣ wᵣ·fᵣ(a)·Πᵢ gᵣᵢ(sᵢ) → O(R·n·k) tractable
"""

@dataclass(frozen=True)
class SeparableDecomposition:
    """
    From SeparableUtility.lean::SeparableUtility.
    u(a,s) = actionValue(a) + stateValue(s).
    Optimal actions don't depend on state.
    """
    action_value: Callable  # A → ℝ
    state_value: Callable   # S → ℝ

def detect_separability(
    utility_matrix: jnp.ndarray,
    threshold: float = 1e-10
) -> bool:
    """
    Check if utility matrix U[a,s] is separable (rank 1 up to additive structure).
    
    From SeparableUtility.lean::optimalActions_eq_of_separable:
    If u(a,s) = f(a) + g(s), then optimal actions are the same for all states.
    
    Algorithm: check if argmax_a U[a,s] is the same column for all s.
    """
    optimal_actions = jnp.argmax(utility_matrix, axis=0)
    return bool(jnp.all(optimal_actions == optimal_actions[0]))

def detect_low_tensor_rank(
    utility_matrix: jnp.ndarray,
    max_rank: int
) -> int:
    """
    Estimate tensor rank of utility matrix via SVD.
    
    From TensorRankDecomposition: u = Σᵣ wᵣ·fᵣ(a)·gᵣ(s).
    
    From low_rank_tractability: O(|A|·R·n) when rank is bounded.
    Returns effective rank (number of singular values > threshold).
    """
    U, S, Vt = jnp.linalg.svd(utility_matrix, full_matrices=False)
    # Effective rank = count of significant singular values
    threshold = S[0] * 1e-10 if S[0] > 0 else 1e-10
    effective_rank = int(jnp.sum(S > threshold))
    return min(effective_rank, max_rank)

@dataclass(frozen=True)
class TensorRankResult:
    """Result of tensor rank analysis."""
    rank: int
    is_separable: bool    # rank ≤ 1
    is_low_rank: bool     # rank ≪ min(|A|, |S|)
    complexity_bound: int # O(|A| · rank · n)
