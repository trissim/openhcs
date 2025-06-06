# plan_04_kernel_architecture.md
## Component: Three-Kernel GPU Borůvka's Implementation

### Objective
Implement the three-kernel architecture for full GPU Borůvka's algorithm based on CMU research. Each kernel handles a specific phase of the algorithm with optimized memory access patterns and minimal CPU-GPU synchronization.

### Plan
1. **Kernel 1: Reset and Flatten (`reset_and_flatten_kernel`)**
   - Reset cheapest edge arrays for new iteration
   - Flatten union-find trees for O(1) component lookup
   - Prepare data structures for parallel edge processing

2. **Kernel 2: Find Minimum Edges (`find_minimum_edges_kernel`)**
   - Parallel processing of edge list
   - Atomic updates of cheapest edge per component
   - Warp-level optimizations for memory access

3. **Kernel 3: Union Components (`union_components_kernel`)**
   - Process selected minimum edges
   - Perform atomic union operations
   - Update MST edge list
   - Count remaining components for termination

4. **Host coordination and memory management**
   - Minimal CPU-GPU synchronization
   - Component count checking for termination
   - Memory allocation and cleanup

5. **Integration with existing MIST pipeline**
   - Replace current `_build_mst_gpu()` function
   - Maintain same input/output interface
   - Preserve error handling and validation

### Findings
**Three-Kernel Architecture Benefits:**
- Separates concerns for better optimization
- Minimizes CPU-GPU synchronization
- Allows kernel-specific memory access patterns
- Enables overlapped execution where possible

**Kernel 1: Reset and Flatten**
- Parallel tree flattening eliminates SIMD divergence
- Reset operations prepare for new iteration
- Minimal computation, memory bandwidth bound
- Can be fused with other operations if beneficial

**Kernel 2: Find Minimum Edges**
- Most computationally intensive kernel
- Memory access pattern critical for performance
- Atomic operations require careful optimization
- Warp-level reductions for efficiency

**Kernel 3: Union Components**
- Processes results from kernel 2
- Atomic union operations on selected edges
- Updates MST and component counts
- Relatively lightweight compared to kernel 2

**Memory Management Strategy:**
- Persistent GPU memory for data structures
- Minimal host-device transfers
- Coalesced memory access patterns
- Atomic operation clustering

**Synchronization Points:**
- Between kernels (implicit via kernel launches)
- Component count check on host (for termination)
- MST result transfer at completion
- Error checking after each kernel

**Performance Considerations:**
- Kernel 2 dominates execution time (edge processing)
- Memory bandwidth utilization critical
- Atomic contention minimization
- Warp occupancy optimization

**Integration Requirements:**
- Same function signature as current `_build_mst_gpu()`
- Compatible with existing error handling
- Maintains input validation
- Preserves output format

### Implementation Draft

```python
import cupyx.jit

@cupyx.jit.rawkernel()
def _union_components_kernel(
    cheapest_edge_idx, edges_from, edges_to, edges_dx, edges_dy,
    parent, rank, mst_from, mst_to, mst_dx, mst_dy, mst_count, num_nodes
):
    """
    Kernel 3: Union components based on selected minimum edges.

    Uses JIT compilation with atomic operations for parallel component union.
    """
    tid = cupyx.jit.blockIdx.x * cupyx.jit.blockDim.x + cupyx.jit.threadIdx.x

    if tid < num_nodes:
        edge_idx = cheapest_edge_idx[tid]

        # Skip if no valid edge for this component
        if edge_idx < 0:
            return

        # Get edge details
        from_node = edges_from[edge_idx]
        to_node = edges_to[edge_idx]

        # Find current component roots (should be O(1) after flattening)
        from_root = parent[from_node]
        to_root = parent[to_node]

        # Skip if already in same component (shouldn't happen but safety check)
        if from_root == to_root:
            return

        # Perform atomic union operation
        union_success = False
        rank1 = rank[from_root]
        rank2 = rank[to_root]

        if rank1 < rank2:
            # Make to_root the parent of from_root
            old_parent = cupyx.jit.atomic_cas(parent, from_root, from_root, to_root)
            union_success = (old_parent == from_root)
        elif rank1 > rank2:
            # Make from_root the parent of to_root
            old_parent = cupyx.jit.atomic_cas(parent, to_root, to_root, from_root)
            union_success = (old_parent == to_root)
        else:
            # Equal ranks: make from_root parent and increment its rank
            old_parent = cupyx.jit.atomic_cas(parent, to_root, to_root, from_root)
            if old_parent == to_root:
                cupyx.jit.atomic_add(rank, from_root, 1)
                union_success = True

        # If union was successful, add edge to MST
        if union_success:
            # Atomically get next MST slot
            mst_slot = cupyx.jit.atomic_add(mst_count, 0, 1)

            # Add edge to MST arrays
            if mst_slot < num_nodes - 1:  # Safety check
                mst_from[mst_slot] = from_node
                mst_to[mst_slot] = to_node
                mst_dx[mst_slot] = edges_dx[edge_idx]
                mst_dy[mst_slot] = edges_dy[edge_idx]


@cupyx.jit.rawkernel()
def _count_valid_edges_kernel(cheapest_edge_idx, valid_count, num_nodes):
    """
    Count how many components have valid cheapest edges.

    Used for termination detection.
    """
    tid = cupyx.jit.blockIdx.x * cupyx.jit.blockDim.x + cupyx.jit.threadIdx.x

    if tid < num_nodes:
        if cheapest_edge_idx[tid] >= 0:
            cupyx.jit.atomic_add(valid_count, 0, 1)


def _launch_union_components_kernel(
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
) -> int:
    """
    Launch the union components kernel and return number of edges added.
    """
    # Store initial MST count
    initial_count = int(mst_count[0])

    # Launch kernel
    threads_per_block = 256
    blocks_per_grid = (num_nodes + threads_per_block - 1) // threads_per_block

    _union_components_kernel(
        (blocks_per_grid,), (threads_per_block,),
        (cheapest_edge_idx, edges_from, edges_to, edges_dx, edges_dy,
         parent, rank, mst_from, mst_to, mst_dx, mst_dy, mst_count, num_nodes)
    )

    # Return number of edges added
    final_count = int(mst_count[0])
    return final_count - initial_count


def _launch_count_valid_edges_kernel(
    cheapest_edge_idx: "cp.ndarray",  # type: ignore
    num_nodes: int
) -> int:
    """
    Count number of components with valid cheapest edges.
    """
    valid_count = cp.array([0], dtype=cp.int32)

    threads_per_block = 256
    blocks_per_grid = (num_nodes + threads_per_block - 1) // threads_per_block

    _count_valid_edges_kernel(
        (blocks_per_grid,), (threads_per_block,),
        (cheapest_edge_idx, valid_count, num_nodes)
    )

    return int(valid_count[0])


def _performance_profile_kernels():
    """
    Profile individual kernel performance for optimization.

    Measures execution time of each kernel phase.
    """
    import time

    # Kernel timing results
    kernel_times = {
        'reset_flatten': [],
        'find_minimum': [],
        'union_components': []
    }

    # Add profiling hooks to kernel launches
    # Implementation would add timing around each kernel call

    return kernel_times


def _optimize_kernel_parameters():
    """
    Optimize kernel launch parameters based on GPU characteristics.

    Adjusts block size, grid size, and memory usage for target GPU.
    """
    # Query GPU properties
    device = cp.cuda.Device()
    props = device.attributes

    # Calculate optimal thread block size
    max_threads_per_block = props['MaxThreadsPerBlock']
    warp_size = 32  # Standard for NVIDIA GPUs

    # Optimize for memory coalescing and occupancy
    optimal_block_size = min(256, max_threads_per_block)

    return {
        'threads_per_block': optimal_block_size,
        'shared_memory_size': props['SharedMemoryPerBlock'],
        'max_blocks_per_sm': props['MaxBlocksPerMultiprocessor']
    }
```
