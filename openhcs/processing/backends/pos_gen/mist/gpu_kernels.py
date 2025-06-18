"""
GPU JIT Kernels for Borůvka's MST Algorithm

All CUDA kernels for parallel MST construction using CuPy JIT.
"""
from __future__ import annotations 

from typing import TYPE_CHECKING

from openhcs.core.utils import optional_import

jit = optional_import("cupyx.jit")
# For type checking only
if TYPE_CHECKING:
    import cupy as cp

# Import CuPy as an optional dependency
cp = optional_import("cupy")


@jit.rawkernel() if jit else lambda f: f
def _reset_and_flatten_kernel(
    parent, rank, cheapest_edge_idx, cheapest_edge_weight_int, num_nodes
):
    """
    Kernel 1: Reset cheapest edge arrays and flatten union-find trees.
    """
    tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x

    if tid < num_nodes:
        # Reset cheapest edge tracking for this node
        cheapest_edge_idx[tid] = -1
        cheapest_edge_weight_int[tid] = 2147483647  # Max int32 value

        # Flatten union-find tree: make this node point directly to root
        # Correct iterative path compression
        current = tid
        while parent[current] != current:
            parent[current] = parent[parent[current]]  # Compress one level at a time
            current = parent[current]


@jit.rawkernel() if jit else lambda f: f
def _find_minimum_edges_kernel(
    edges_from, edges_to, edges_quality, parent,
    cheapest_edge_idx, cheapest_edge_weight_int, num_edges
):
    """
    Kernel 2: Find minimum weight edge for each component (using int32 for atomics).
    """
    tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x

    if tid < num_edges:
        # Get edge endpoints and their components
        from_node = edges_from[tid]
        to_node = edges_to[tid]
        from_comp = parent[from_node]
        to_comp = parent[to_node]

        # Only process if NOT a self-edge (no return statement)
        if from_comp != to_comp:
            # Get edge quality (higher is better, so negate for min comparison)
            edge_quality = edges_quality[tid]
            # Convert to integer by scaling and negating (higher quality = lower int value)
            # Scale by 1000000 to preserve precision, then negate
            edge_weight_int = int(-edge_quality * 1000000)

            # Atomic update cheapest edge for 'from' component
            jit.atomic_min(cheapest_edge_weight_int, from_comp, edge_weight_int)
            if cheapest_edge_weight_int[from_comp] == edge_weight_int:
                cheapest_edge_idx[from_comp] = tid

            # Atomic update cheapest edge for 'to' component
            jit.atomic_min(cheapest_edge_weight_int, to_comp, edge_weight_int)
            if cheapest_edge_weight_int[to_comp] == edge_weight_int:
                cheapest_edge_idx[to_comp] = tid


@jit.rawkernel() if jit else lambda f: f
def _union_components_kernel(
    cheapest_edge_idx, edges_from, edges_to, edges_dx, edges_dy,
    parent, rank, mst_from, mst_to, mst_dx, mst_dy, mst_count, num_nodes
):
    """
    Kernel 3: Union components (no return statements).
    """
    tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x

    if tid < num_nodes:
        edge_idx = cheapest_edge_idx[tid]

        # Only process if valid edge exists (no return statement)
        if edge_idx >= 0:
            # Get edge details
            from_node = edges_from[edge_idx]
            to_node = edges_to[edge_idx]

            # Find current component roots
            from_root = parent[from_node]
            to_root = parent[to_node]

            # Only process if NOT already in same component (no return statement)
            if from_root != to_root:
                # Atomic union operation with rank-based optimization
                rank1 = rank[from_root]
                rank2 = rank[to_root]

                union_success = False
                if rank1 < rank2:
                    # Make to_root the parent of from_root
                    old_parent = jit.atomic_cas(parent, from_root, from_root, to_root)
                    union_success = (old_parent == from_root)
                elif rank1 > rank2:
                    # Make from_root the parent of to_root
                    old_parent = jit.atomic_cas(parent, to_root, to_root, from_root)
                    union_success = (old_parent == to_root)
                else:
                    # Equal ranks: make from_root parent and increment its rank
                    old_parent = jit.atomic_cas(parent, to_root, to_root, from_root)
                    if old_parent == to_root:
                        jit.atomic_add(rank, from_root, 1)
                        union_success = True

                # If union was successful, atomically add edge to MST
                if union_success:
                    mst_slot = jit.atomic_add(mst_count, 0, 1)
                    if mst_slot < num_nodes - 1:
                        mst_from[mst_slot] = from_node
                        mst_to[mst_slot] = to_node
                        mst_dx[mst_slot] = edges_dx[edge_idx]
                        mst_dy[mst_slot] = edges_dy[edge_idx]


def launch_reset_flatten_kernel(
    parent: "cp.ndarray",  # type: ignore
    rank: "cp.ndarray",  # type: ignore
    cheapest_edge_idx: "cp.ndarray",  # type: ignore
    cheapest_edge_weight_int: "cp.ndarray",  # type: ignore
    num_nodes: int
) -> None:
    """
    Launch the reset and flatten kernel with appropriate grid/block dimensions.
    """
    threads_per_block = 256
    blocks_per_grid = (num_nodes + threads_per_block - 1) // threads_per_block

    _reset_and_flatten_kernel(
        (blocks_per_grid,), (threads_per_block,),
        (parent, rank, cheapest_edge_idx, cheapest_edge_weight_int, num_nodes)
    )


def launch_find_minimum_edges_kernel(
    edges_from: "cp.ndarray",  # type: ignore
    edges_to: "cp.ndarray",  # type: ignore
    edges_quality: "cp.ndarray",  # type: ignore
    parent: "cp.ndarray",  # type: ignore
    cheapest_edge_idx: "cp.ndarray",  # type: ignore
    cheapest_edge_weight_int: "cp.ndarray",  # type: ignore
    num_edges: int
) -> None:
    """
    Launch the minimum edge finding kernel with appropriate dimensions.
    """
    threads_per_block = 256
    blocks_per_grid = (num_edges + threads_per_block - 1) // threads_per_block

    _find_minimum_edges_kernel(
        (blocks_per_grid,), (threads_per_block,),
        (edges_from, edges_to, edges_quality, parent,
         cheapest_edge_idx, cheapest_edge_weight_int, num_edges)
    )


def launch_union_components_kernel(
    cheapest_edge_idx: "cp.ndarray",  # type: ignore
    edges_from: "cp.ndarray",  # type: ignore
    edges_to: "cp.ndarray",  # type: ignore
    edges_dx: "cp.ndarray",  # type: ignore
    edges_dy: "cp.ndarray",  # type: ignore
    parent: "cp.ndarray",  # type: ignore
    rank: "cp.ndarray",  # type: ignore
    mst_from: "cp.ndarray",  # type: ignore
    mst_to: "cp.ndarray",  # type: ignore
    mst_dx: "cp.ndarray",  # type: ignore
    mst_dy: "cp.ndarray",  # type: ignore
    mst_count: "cp.ndarray",  # type: ignore
    num_nodes: int
) -> None:
    """
    Launch the union components kernel - pure GPU, no CPU sync.
    """
    # Launch kernel without CPU synchronization
    threads_per_block = 256
    blocks_per_grid = (num_nodes + threads_per_block - 1) // threads_per_block

    _union_components_kernel(
        (blocks_per_grid,), (threads_per_block,),
        (cheapest_edge_idx, edges_from, edges_to, edges_dx, edges_dy,
         parent, rank, mst_from, mst_to, mst_dx, mst_dy, mst_count, num_nodes)
    )


def gpu_component_count(parent: "cp.ndarray") -> "cp.ndarray":  # type: ignore
    """
    Count number of distinct components in flattened union-find - pure GPU.
    Returns GPU array, no CPU sync.
    """
    # After flattening, roots are nodes where parent[i] == i
    roots = (parent == cp.arange(len(parent)))
    return cp.sum(roots)  # Return GPU array, not CPU int
