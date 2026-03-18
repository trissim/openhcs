import jax
import jax.numpy as jnp
from jax import jit, vmap
from typing import Callable

"""
Structural Rank (srank) implementation.
Aligns with StructuralRank.lean: srank(dp) = |{i | dp.isRelevant i}|.

Uses jax.grad for relevance detection instead of finite differences.
"""

def compute_srank(
    positions: jnp.ndarray,
    utility_fn: Callable[[jnp.ndarray], float],
    threshold: float = 1e-6
) -> int:
    """
    Compute structural rank: number of atoms whose gradient is non-zero.
    Uses jax.grad (DSL: array_gradient) for exact relevance, not finite differences.
    """
    grad_u = jax.grad(utility_fn)(positions)  # (N, 3) gradient
    atom_grad_norms = jnp.linalg.norm(grad_u, axis=-1)  # (N,) per-atom gradient magnitude
    return int(jnp.sum(atom_grad_norms > threshold))

def find_relevant_features(
    positions: jnp.ndarray,
    utility_fn: Callable[[jnp.ndarray], float],
    threshold: float = 1e-6
) -> jnp.ndarray:
    """Return indices of relevant atoms (gradient magnitude > threshold)."""
    grad_u = jax.grad(utility_fn)(positions)
    atom_grad_norms = jnp.linalg.norm(grad_u, axis=-1)
    return jnp.where(atom_grad_norms > threshold)[0]
