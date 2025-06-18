"""
Borůvka's Minimum Spanning Tree Algorithm for MIST

GPU-accelerated MST construction using parallel Borůvka's algorithm.
"""

from __future__ import annotations 
from typing import TYPE_CHECKING

from openhcs.core.utils import optional_import
from .gpu_kernels import (
    launch_reset_flatten_kernel,
    launch_find_minimum_edges_kernel, 
    launch_union_components_kernel,
    gpu_component_count
)

# For type checking only
if TYPE_CHECKING:
    import cupy as cp

# Import CuPy as an optional dependency
cp = optional_import("cupy")


def _validate_cupy_array(array, name: str = "input") -> None:  # type: ignore
    """Validate that the input is a CuPy array."""
    if not isinstance(array, cp.ndarray):
        raise TypeError(f"{name} must be a CuPy array, got {type(array)}")


def _validate_mst_inputs(
    connection_from: "cp.ndarray",  # type: ignore
    connection_to: "cp.ndarray",  # type: ignore
    connection_dx: "cp.ndarray",  # type: ignore
    connection_dy: "cp.ndarray",  # type: ignore
    connection_quality: "cp.ndarray",  # type: ignore
    num_nodes: int
) -> None:
    """Validate MST input arrays."""
    arrays = [connection_from, connection_to, connection_dx, connection_dy, connection_quality]
    names = ["connection_from", "connection_to", "connection_dx", "connection_dy", "connection_quality"]
    
    for array, name in zip(arrays, names):
        _validate_cupy_array(array, name)
    
    # Check all arrays have same length
    lengths = [len(arr) for arr in arrays]
    if not all(length == lengths[0] for length in lengths):
        raise ValueError(f"All connection arrays must have same length, got {lengths}")
    
    # Check node indices are valid
    if len(connection_from) > 0:
        max_from = int(cp.max(connection_from))
        max_to = int(cp.max(connection_to))
        max_node = max(max_from, max_to)
        if max_node >= num_nodes:
            raise ValueError(f"Node index {max_node} exceeds num_nodes {num_nodes}")


def build_mst_gpu_boruvka(
    connection_from: "cp.ndarray",  # type: ignore
    connection_to: "cp.ndarray",  # type: ignore
    connection_dx: "cp.ndarray",  # type: ignore
    connection_dy: "cp.ndarray",  # type: ignore
    connection_quality: "cp.ndarray",  # type: ignore
    num_nodes: int
) -> dict:
    """
    Full GPU Borůvka's algorithm for minimum spanning tree.

    Uses JIT kernels with atomic operations for true parallel execution.
    All operations remain on GPU with no CPU-GPU synchronization in inner loops.
    """


    # Validate inputs
    _validate_mst_inputs(connection_from, connection_to, connection_dx,
                        connection_dy, connection_quality, num_nodes)

    if len(connection_from) == 0:
        return {'edges': []}

    # Initialize GPU data structures
    num_edges = len(connection_from)

    # Union-find structure (flattened for O(1) lookups)
    parent = cp.arange(num_nodes, dtype=cp.int32)
    rank = cp.zeros(num_nodes, dtype=cp.int32)

    # Component minimum edge tracking (use int32 for atomic operations)
    cheapest_edge_idx = cp.full(num_nodes, -1, dtype=cp.int32)
    cheapest_edge_weight_int = cp.full(num_nodes, 2147483647, dtype=cp.int32)  # Max int32 value

    # MST result storage
    mst_edges_from = cp.zeros(num_nodes - 1, dtype=cp.int32)
    mst_edges_to = cp.zeros(num_nodes - 1, dtype=cp.int32)
    mst_edges_dx = cp.zeros(num_nodes - 1, dtype=cp.float32)
    mst_edges_dy = cp.zeros(num_nodes - 1, dtype=cp.float32)
    mst_count = cp.array([0], dtype=cp.int32)

    # Sort edges by source vertex for cache locality
    sort_indices = cp.argsort(connection_from)
    edges_from = connection_from[sort_indices]
    edges_to = connection_to[sort_indices]
    edges_dx = connection_dx[sort_indices]
    edges_dy = connection_dy[sort_indices]
    edges_quality = connection_quality[sort_indices]

    # Main Borůvka's loop - O(log V) iterations
    max_iterations = int(cp.ceil(cp.log2(num_nodes))) + 1


    for iteration in range(max_iterations):
        print(f"🔥 Iteration {iteration}: Starting...")

        # Kernel 1: Reset and flatten union-find trees
        print(f"🔥 Iteration {iteration}: Launching reset/flatten kernel...")
        launch_reset_flatten_kernel(
            parent, rank, cheapest_edge_idx, cheapest_edge_weight_int, num_nodes
        )

        # Kernel 2: Find minimum edge per component (parallel)
        print(f"🔥 Iteration {iteration}: Launching find minimum edges kernel...")
        launch_find_minimum_edges_kernel(
            edges_from, edges_to, edges_quality, parent,
            cheapest_edge_idx, cheapest_edge_weight_int, num_edges
        )

        # Kernel 3: Union components and update MST
        print(f"🔥 Iteration {iteration}: Launching union components kernel...")
        launch_union_components_kernel(
            cheapest_edge_idx, edges_from, edges_to, edges_dx, edges_dy,
            parent, rank, mst_edges_from, mst_edges_to, mst_edges_dx, mst_edges_dy,
            mst_count, num_nodes
        )

        print(f"🔥 Iteration {iteration}: Kernel launched (pure GPU)")

        # Pure GPU termination check - no CPU sync
        # Use fixed iteration count instead of dynamic checking
        # This eliminates CPU-GPU synchronization bottleneck

    # Convert result to expected format
    final_mst_count = int(mst_count[0])
    selected_edges = []

    # Debug: Print MST construction results
    print(f"Borůvka MST: {final_mst_count} edges constructed (expected: {num_nodes-1})")

    # Bounds check to prevent crash
    max_edges = num_nodes - 1
    if final_mst_count > max_edges:
        print(f"🔥 WARNING: MST count {final_mst_count} exceeds maximum {max_edges}, clamping")
        final_mst_count = max_edges

    for i in range(final_mst_count):
        edge = {
            'from': int(mst_edges_from[i]),
            'to': int(mst_edges_to[i]),
            'dx': float(mst_edges_dx[i]),
            'dy': float(mst_edges_dy[i]),
            'quality': 0.0  # Could be stored if needed
        }
        selected_edges.append(edge)

        # Debug: Print first few edges
        if i < 3:
            print(f"  Edge {i}: {edge['from']} -> {edge['to']}, dx={edge['dx']:.3f}, dy={edge['dy']:.3f}")

    return {'edges': selected_edges}
