import jax
import jax.numpy as jnp
from jax import jit, vmap
from typing import Callable, List, Set

"""
Structural Rank (srank) implementation.
Aligns with StructuralRank.lean: srank(dp) = |{i | dp.isRelevant i}|.
"""

@jit
def is_relevant(
    coord_idx: int,
    positions: jnp.ndarray,
    utility_fn: Callable[[jnp.ndarray], float],
    epsilon: float = 1e-4
) -> bool:
    """
    Check if a coordinate is relevant to the decision.
    A coordinate is relevant if perturbing it changes the optimal action.
    Simplified for docking: check if the utility gradient with respect to this coord 
    is non-zero (i.e., it affects the binding energy).
    """
    # Perturb the coordinate
    perturbation = jnp.zeros_like(positions)
    # This is a bit tricky in JAX because coord_idx needs to be handled
    # We can use jax.lax.dynamic_update_slice or similar, but for simplicity:
    perturbed_pos = positions.at[coord_idx].add(epsilon)
    
    u_orig = utility_fn(positions)
    u_pert = utility_fn(perturbed_pos)
    
    # relevance = |du/dqi| > 0
    return jnp.abs(u_pert - u_orig) > epsilon

@jit
def compute_srank(
    positions: jnp.ndarray,
    utility_fn: Callable[[jnp.ndarray], float]
) -> int:
    """
    Compute structural rank: number of coordinates that affect the binding utility.
    """
    # Vectorized relevance check over all atoms/coordinates
    # atomic positions are (N, 3), so we check N atoms or 3N coordinates
    n_atoms = positions.shape[0]
    
    def check_atom_relevance(i):
        # Check if any of the 3 coordinates of atom i are relevant
        return is_relevant(i, positions, utility_fn)

    relevance_mask = vmap(check_atom_relevance)(jnp.arange(n_atoms))
    return jnp.sum(relevance_mask.astype(jnp.int32))

def find_relevant_features(
    positions: jnp.ndarray,
    utility_fn: Callable[[jnp.ndarray], float]
) -> jnp.ndarray:
    """Return indices of relevant atoms."""
    n_atoms = positions.shape[0]
    relevance_mask = vmap(lambda i: is_relevant(i, positions, utility_fn))(jnp.arange(n_atoms))
    return jnp.where(relevance_mask)[0]
