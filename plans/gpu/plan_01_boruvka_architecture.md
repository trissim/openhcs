# plan_01_boruvka_architecture.md
## Component: Full GPU Borůvka's MST Algorithm Architecture

### Objective
Design and implement a fully GPU-accelerated Borůvka's algorithm for minimum spanning tree construction in the MIST cupy processor. This will replace the current CPU-hybrid Kruskal's approach with true parallel execution where all operations remain on GPU.

### Plan
1. **Replace current `_build_mst_gpu()` with parallel Borůvka's implementation**
   - Remove sequential edge processing loops
   - Implement parallel component-based minimum edge finding
   - Use atomic operations for concurrent union-find operations

2. **Design GPU-native data structures**
   - Flattened union-find with path compression between iterations
   - Component-based edge storage for parallel minimum finding
   - Atomic cheapest edge tracking per component

3. **Implement three-kernel architecture** (based on CMU research)
   - Kernel 1: Reset and flatten union-find trees
   - Kernel 2: Find minimum edge per component (parallel)
   - Kernel 3: Union components and update MST

4. **Optimize memory access patterns**
   - Sort edges by source vertex for cache locality
   - Use warp-level primitives for reduction operations
   - Minimize CPU-GPU memory transfers

5. **Handle edge cases and termination**
   - Detect when single component remains
   - Handle tie-breaking consistently across parallel execution
   - Manage self-edges and multi-edges

### Findings
**Current Kruskal's Limitations:**
- Sequential edge processing: `for start_idx in range(0, len(sorted_from), batch_size)`
- CPU list operations: `selected_from.extend(batch_from[valid_indices].tolist())`
- Sequential union operations: `for idx in valid_indices: parent[to_node] = from_node`

**Borůvka's Advantages for GPU:**
- Each component finds minimum edge in parallel
- All union operations can happen simultaneously
- No sequential dependency between edge selections
- Natural fit for SIMD execution model

**Research Insights from CMU Implementation:**
- 29x speedup achieved on GeForce RTX 2080
- Three-kernel approach with flattened union-find
- Memory access pattern optimization critical
- Tie-breaking must be consistent across parallel threads

**Key Algorithmic Requirements:**
1. **Component identification** - each vertex knows its component root
2. **Parallel minimum finding** - each component finds cheapest outgoing edge
3. **Atomic edge selection** - prevent race conditions in edge selection
4. **Parallel union** - merge components simultaneously
5. **Tree flattening** - maintain O(1) component lookup between iterations

**Memory Layout Considerations:**
- Edge list sorted by source vertex for cache efficiency
- Component roots stored in flattened array
- Cheapest edge per component in separate array
- MST edges accumulated in GPU memory

**Termination Condition:**
- Algorithm terminates when no valid edges found (single component)
- Track number of components to detect completion
- Early termination when MST has n-1 edges

### Implementation Draft

```python
import cupyx.jit

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

    # Component minimum edge tracking with tie-breaking
    cheapest_edge_idx = cp.full(num_nodes, -1, dtype=cp.int32)
    cheapest_edge_weight = cp.full(num_nodes, cp.inf, dtype=cp.float32)
    cheapest_edge_tiebreaker = cp.full(num_nodes, cp.iinfo(cp.int32).max, dtype=cp.int32)

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
        # Kernel 1: Reset and flatten union-find trees
        _launch_reset_flatten_kernel(
            parent, rank, cheapest_edge_idx, cheapest_edge_weight, num_nodes
        )

        # Kernel 2: Find minimum edge per component (parallel)
        _launch_find_minimum_edges_kernel(
            edges_from, edges_to, edges_quality, parent,
            cheapest_edge_idx, cheapest_edge_weight, num_edges
        )

        # Kernel 3: Union components and update MST
        edges_added = _launch_union_components_kernel(
            cheapest_edge_idx, edges_from, edges_to, edges_dx, edges_dy,
            parent, rank, mst_edges_from, mst_edges_to, mst_edges_dx, mst_edges_dy,
            mst_count, num_nodes
        )

        # Check termination condition (minimal CPU-GPU sync)
        if edges_added == 0:
            break

        # Early termination check
        component_count = _gpu_component_count(parent)
        if component_count <= 1:
            break

    # Convert result to expected format
    final_mst_count = int(mst_count[0])
    selected_edges = []

    for i in range(final_mst_count):
        selected_edges.append({
            'from': int(mst_edges_from[i]),
            'to': int(mst_edges_to[i]),
            'dx': float(mst_edges_dx[i]),
            'dy': float(mst_edges_dy[i]),
            'quality': 0.0  # Could be stored if needed
        })

    return {'edges': selected_edges}


def _replace_build_mst_gpu_with_boruvka():
    """
    Replace the existing _build_mst_gpu function with Borůvka's implementation.

    Maintains same interface for backward compatibility.
    """
    import openhcs.processing.backends.pos_gen.mist_processor_cupy as mist_module

    # Store reference to old implementation for fallback
    mist_module._build_mst_gpu_kruskal = mist_module._build_mst_gpu

    # Replace with Borůvka's implementation
    mist_module._build_mst_gpu = build_mst_gpu_boruvka
```
