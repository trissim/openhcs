import jax.numpy as jnp
from dataclasses import dataclass
from typing import Callable

"""
Tree Structure and Bounded Treewidth detection.
Direct translation of Tractability/TreeStructure.lean.

Key results:
- TreeStructured: deps point only to smaller indices → DP tractable
- InteractionGraph: pairs of interacting coordinates
- BoundedTreewidth: treewidth ≤ w → CSP in O(n · k^(w+1))
"""

def is_tree_structured(adjacency: jnp.ndarray) -> bool:
    """
    Check if adjacency matrix has tree structure.
    From TreeStructure.lean::TreeStructured:
      deps point only to strictly smaller indices.
    
    In JAX: check if adjacency is lower-triangular (after removing diagonal).
    """
    n = adjacency.shape[0]
    # Upper triangle (excluding diagonal) should be all zeros
    upper = jnp.triu(adjacency, k=1)
    return bool(jnp.all(upper == 0))

def build_interaction_graph(
    potential_fn: Callable,
    positions: jnp.ndarray,
    threshold: float = 1e-8
) -> jnp.ndarray:
    """
    Build interaction graph from the Hessian of the potential.
    From TreeStructure.lean::InteractionGraph.
    
    Two coordinates interact if ∂²U/∂q_i∂q_j ≠ 0.
    Returns symmetric adjacency matrix.
    """
    import jax
    
    n = positions.shape[0] * positions.shape[1]
    flat = positions.ravel()
    
    def flat_potential(q):
        return potential_fn(q.reshape(positions.shape))
    
    H = jax.hessian(flat_potential)(flat)
    # Adjacency: off-diagonal Hessian entries above threshold
    adjacency = (jnp.abs(H) > threshold).astype(jnp.float32)
    adjacency = adjacency - jnp.diag(jnp.diag(adjacency))  # Remove diagonal
    return adjacency

def estimate_treewidth(adjacency: jnp.ndarray) -> int:
    """
    Estimate treewidth of the interaction graph.
    From TreeStructure.lean::treewidth_le.
    
    Uses minimum degree heuristic (upper bound on treewidth).
    Exact treewidth is NP-hard, so we use the greedy approximation.
    """
    n = adjacency.shape[0]
    adj = jnp.array(adjacency, copy=True)
    max_bag = 0
    remaining = list(range(n))
    
    for _ in range(n):
        if not remaining:
            break
        # Find vertex with minimum degree
        degrees = [int(jnp.sum(adj[v, remaining])) for v in remaining]
        min_idx = int(jnp.argmin(jnp.array(degrees)))
        v = remaining[min_idx]
        
        # Bag size = degree + 1
        deg = degrees[min_idx]
        max_bag = max(max_bag, deg + 1)
        
        # Remove vertex
        remaining.remove(v)
        
        # Add edges between neighbors (fill-in)
        neighbors = [u for u in remaining if adj[v, u] > 0]
        for i, u in enumerate(neighbors):
            for w in neighbors[i+1:]:
                adj = adj.at[u, w].set(1.0)
                adj = adj.at[w, u].set(1.0)
    
    return max(max_bag - 1, 0)  # treewidth = max bag size - 1

@dataclass(frozen=True)
class TreewidthResult:
    """Result of treewidth analysis."""
    treewidth: int
    is_tree: bool              # treewidth ≤ 1
    complexity_bound: int      # O(n · k^(w+1))
    is_tractable: bool         # treewidth ≤ threshold
