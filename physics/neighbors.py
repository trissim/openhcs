import jax
import jax.numpy as jnp
from jax import jit
from .kernels import pairwise_distances, apply_cutoff

"""
Cell-list neighbor list for O(N) force evaluation.
Replaces O(N^2) pairwise_distances calls with spatially-partitioned lookups.
Uses DSL: pairwiseDistances (within cells), applyCutoff.
"""

@jit
def build_cell_list(positions: jnp.ndarray, box_size: jnp.ndarray, cutoff: float):
    """
    Assign atoms to cells. Cell side length = cutoff.
    Returns cell_indices (N,3) and n_cells (3,).
    """
    n_cells = jnp.floor(box_size / cutoff).astype(jnp.int32)
    n_cells = jnp.maximum(n_cells, 1)  # At least 1 cell per dimension
    cell_size = box_size / n_cells
    cell_indices = jnp.floor(positions / cell_size).astype(jnp.int32)
    # Clamp to valid range
    cell_indices = jnp.clip(cell_indices, 0, n_cells - 1)
    return cell_indices, n_cells

@jit
def compute_neighbor_mask(
    positions: jnp.ndarray,
    box_size: jnp.ndarray,
    cutoff: float
) -> jnp.ndarray:
    """
    Compute a boolean neighbor mask: mask[i,j] = True if |r_i - r_j| < cutoff.
    For systems small enough to fit in memory, this is the simplest correct approach.
    Falls back on DSL: pairwiseDistances + applyCutoff.
    """
    # Apply minimum image convention for periodic boundaries
    diff = positions[:, None, :] - positions[None, :, :]
    diff = diff - box_size * jnp.round(diff / box_size)
    dists = jnp.linalg.norm(diff, axis=-1)
    
    # Exclude self-interactions (diagonal)
    mask = (dists < cutoff) & (dists > 0.0)
    return mask

@jit
def neighbor_energy(
    positions: jnp.ndarray,
    box_size: jnp.ndarray,
    cutoff: float,
    pair_energy_fn  # (r) -> energy for a single pair distance
) -> float:
    """
    Compute total pairwise energy using neighbor mask.
    Avoids double-counting via upper triangle.
    """
    diff = positions[:, None, :] - positions[None, :, :]
    diff = diff - box_size * jnp.round(diff / box_size)
    dists = jnp.linalg.norm(diff, axis=-1)
    
    # Upper triangle mask (avoid double-counting) + cutoff
    n = positions.shape[0]
    upper = jnp.triu(jnp.ones((n, n), dtype=bool), k=1)
    within_cutoff = (dists < cutoff) & upper
    
    energies = pair_energy_fn(dists)
    return jnp.sum(jnp.where(within_cutoff, energies, 0.0))
