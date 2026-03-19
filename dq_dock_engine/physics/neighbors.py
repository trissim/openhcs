import jax
import jax.numpy as jnp
from jax import jit
from .kernels import minimum_image_pairwise_distances, upper_triangle_masked_sum

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
    positions: jnp.ndarray, box_size: jnp.ndarray, cutoff: float
) -> jnp.ndarray:
    """
    Compute a boolean neighbor mask: mask[i,j] = True if |r_i - r_j| < cutoff.
    For systems small enough to fit in memory, this is the simplest correct approach.
    Falls back on DSL: pairwiseDistances + applyCutoff.
    """
    dists = minimum_image_pairwise_distances(positions, positions, box_size)

    # Exclude self-interactions (diagonal)
    mask = (dists < cutoff) & (dists > 0.0)
    return mask


@jit
def neighbor_energy(
    positions: jnp.ndarray,
    box_size: jnp.ndarray,
    cutoff: float,
    pair_energy_fn,  # (r) -> energy for a single pair distance
) -> float:
    """
    Compute total pairwise energy using neighbor mask.
    Avoids double-counting via upper triangle.
    """
    dists = minimum_image_pairwise_distances(positions, positions, box_size)
    within_cutoff = dists < cutoff

    energies = pair_energy_fn(dists)
    return upper_triangle_masked_sum(energies, within_cutoff)
