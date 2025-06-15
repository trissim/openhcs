"""
Position Reconstruction from MST for MIST Algorithm

Functions for rebuilding tile positions from minimum spanning tree.
"""
from __future__ import annotations 

from typing import TYPE_CHECKING

from openhcs.core.utils import optional_import

# For type checking only
if TYPE_CHECKING:
    import cupy as cp

# Import CuPy as an optional dependency
cp = optional_import("cupy")


def _validate_cupy_array(array, name: str = "input") -> None:  # type: ignore
    """Validate that the input is a CuPy array."""
    if not isinstance(array, cp.ndarray):
        raise TypeError(f"{name} must be a CuPy array, got {type(array)}")


def rebuild_positions_from_mst_gpu(
    initial_positions: "cp.ndarray",  # type: ignore
    mst_edges: dict,
    num_tiles: int,
    anchor_tile_index: int = 0
) -> "cp.ndarray":  # type: ignore
    """
    Rebuild tile positions from MST edges using GPU operations.
    
    Args:
        initial_positions: Initial position estimates (Z, 2) array
        mst_edges: Dictionary with 'edges' list containing MST edges
        num_tiles: Number of tiles
        anchor_tile_index: Index of anchor tile (fixed at origin)
    
    Returns:
        Reconstructed positions as (Z, 2) CuPy array
    """
    _validate_cupy_array(initial_positions, "initial_positions")
    
    if initial_positions.shape != (num_tiles, 2):
        raise ValueError(f"Initial positions must be ({num_tiles}, 2), got {initial_positions.shape}")
    
    edges = mst_edges.get('edges', [])
    if not edges:
        print("🔥 WARNING: No MST edges provided, returning initial positions")
        return initial_positions.copy()
    
    print(f"Position reconstruction: {len(edges)} MST edges, {num_tiles} tiles")
    
    # Initialize new positions (GPU)
    new_positions = cp.zeros((num_tiles, 2), dtype=cp.float32)
    visited = cp.zeros(num_tiles, dtype=cp.bool_)
    
    # Set anchor tile position
    new_positions[anchor_tile_index] = cp.array([0.0, 0.0])
    visited[anchor_tile_index] = True
    
    print(f"Anchor tile {anchor_tile_index}: (0.0, 0.0)")
    
    # Build adjacency list for efficient traversal
    adjacency = [[] for _ in range(num_tiles)]
    for edge in edges:
        from_idx = edge['from']
        to_idx = edge['to']
        dx = edge['dx']
        dy = edge['dy']
        
        # Add bidirectional edges
        adjacency[from_idx].append({'to': to_idx, 'dx': dx, 'dy': dy})
        adjacency[to_idx].append({'to': from_idx, 'dx': -dx, 'dy': -dy})
    
    # Breadth-first traversal to set positions
    queue = [anchor_tile_index]
    
    while queue:
        current_tile = queue.pop(0)
        current_pos = new_positions[current_tile]
        
        # Process all neighbors
        for neighbor_info in adjacency[current_tile]:
            neighbor_tile = neighbor_info['to']
            
            if not visited[neighbor_tile]:
                # Calculate neighbor position
                dx = neighbor_info['dx']
                dy = neighbor_info['dy']
                neighbor_pos = current_pos + cp.array([dx, dy])
                
                new_positions[neighbor_tile] = neighbor_pos
                visited[neighbor_tile] = True
                queue.append(neighbor_tile)
    
    # Check if all tiles were visited
    unvisited_count = int(cp.sum(~visited))
    if unvisited_count > 0:
        print(f"🔥 WARNING: {unvisited_count} tiles not reachable from anchor tile")
        
        # For unvisited tiles, use initial positions
        unvisited_mask = ~visited
        new_positions[unvisited_mask] = initial_positions[unvisited_mask]
    
    return new_positions


def build_mst_gpu(
    connection_from: "cp.ndarray",  # type: ignore
    connection_to: "cp.ndarray",  # type: ignore
    connection_dx: "cp.ndarray",  # type: ignore
    connection_dy: "cp.ndarray",  # type: ignore
    connection_quality: "cp.ndarray",  # type: ignore
    num_tiles: int
) -> dict:
    """
    Build MST using GPU Borůvka's algorithm.
    
    This is a wrapper that imports and calls the Borůvka implementation.
    """
    from .boruvka_mst import build_mst_gpu_boruvka
    
    return build_mst_gpu_boruvka(
        connection_from, connection_to, connection_dx,
        connection_dy, connection_quality, num_tiles
    )
