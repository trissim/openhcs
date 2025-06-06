# plan_02_union_find_gpu.md
## Component: GPU-Native Union-Find Data Structure

### Objective
Implement a fully GPU-optimized union-find data structure specifically designed for Borůvka's algorithm. This will support parallel find operations, atomic union operations, and tree flattening to avoid SIMD divergence.

### Plan
1. **Design flattened union-find structure**
   - Parent array with atomic updates
   - Rank/size tracking for union by rank
   - Component root caching for O(1) lookups

2. **Implement atomic union operations**
   - Compare-and-swap for parent updates
   - Atomic rank updates for balanced trees
   - Race condition handling for simultaneous unions

3. **Add tree flattening kernel**
   - Parallel path compression between Borůvka iterations
   - Ensure all nodes point directly to root
   - Eliminate divergent tree traversal

4. **Optimize for warp-level execution**
   - Minimize thread divergence in find operations
   - Use warp primitives for parallel reductions
   - Coalesced memory access patterns

5. **Integration with Borůvka's algorithm**
   - Component identification for edge processing
   - Parallel component merging
   - Efficient root finding for large graphs

### Findings
**Current Union-Find Issues:**
- Path compression causes thread divergence: `while parent[from_root] != from_root`
- Sequential union operations: `parent[to_node] = from_node`
- Mixed CPU/GPU operations in validation

**GPU-Specific Requirements:**
- Avoid loops in SIMD execution (causes warp stall)
- Atomic operations for concurrent access
- Flattened trees for predictable execution time
- Warp-level synchronization primitives

**Research-Based Design:**
- CMU paper shows flattening between iterations eliminates divergence
- Union by rank prevents deep trees
- Atomic CAS operations for thread-safe updates
- Component counting for termination detection

**Memory Access Patterns:**
- Parent array accessed randomly (component lookups)
- Rank array accessed during union operations
- Root cache for frequently accessed components
- Coalesced access where possible

**Atomic Operations Needed:**
- `atomicCAS` for parent updates
- `atomicAdd` for rank updates
- `atomicMin` for cheapest edge selection
- Memory fences for consistency

### Implementation Draft

```python
import cupyx.jit

@cupyx.jit.rawkernel()
def _reset_and_flatten_kernel(
    parent, rank, cheapest_edge_idx, cheapest_edge_weight, num_nodes
):
    """
    Kernel 1: Reset cheapest edge arrays and flatten union-find trees.

    Uses JIT compilation for true GPU parallel execution with atomic operations.
    """
    tid = cupyx.jit.blockIdx.x * cupyx.jit.blockDim.x + cupyx.jit.threadIdx.x

    if tid < num_nodes:
        # Reset cheapest edge tracking for this node
        cheapest_edge_idx[tid] = -1
        cheapest_edge_weight[tid] = float('inf')

        # Flatten union-find tree: make this node point directly to root
        # Use iterative path compression to avoid recursion
        current = tid
        while parent[current] != current:
            next_parent = parent[current]
            parent[current] = parent[next_parent]  # Path compression
            current = next_parent


@cupyx.jit.rawkernel()
def _find_roots_kernel(parent, roots, num_nodes):
    """
    Find root for each node after flattening (should be O(1)).
    """
    tid = cupyx.jit.blockIdx.x * cupyx.jit.blockDim.x + cupyx.jit.threadIdx.x

    if tid < num_nodes:
        roots[tid] = parent[tid]


@cupyx.jit.rawkernel()
def _union_components_atomic_kernel(
    parent, rank, from_roots, to_roots, valid_unions, num_edges
):
    """
    Perform atomic union operations for selected edges.

    Uses atomic compare-and-swap for thread-safe union operations.
    """
    tid = cupyx.jit.blockIdx.x * cupyx.jit.blockDim.x + cupyx.jit.threadIdx.x

    if tid < num_edges and valid_unions[tid]:
        root1 = from_roots[tid]
        root2 = to_roots[tid]

        if root1 != root2:
            # Union by rank with atomic operations
            rank1 = rank[root1]
            rank2 = rank[root2]

            if rank1 < rank2:
                # Make root2 the parent of root1
                cupyx.jit.atomic_cas(parent, root1, root1, root2)
            elif rank1 > rank2:
                # Make root1 the parent of root2
                cupyx.jit.atomic_cas(parent, root2, root2, root1)
            else:
                # Equal ranks: make root1 parent and increment its rank
                if cupyx.jit.atomic_cas(parent, root2, root2, root1) == root2:
                    cupyx.jit.atomic_add(rank, root1, 1)


def _launch_reset_flatten_kernel(
    parent: "cp.ndarray",  # type: ignore
    rank: "cp.ndarray",  # type: ignore
    cheapest_edge_idx: "cp.ndarray",  # type: ignore
    cheapest_edge_weight: "cp.ndarray",  # type: ignore
    num_nodes: int
) -> None:
    """
    Launch the reset and flatten kernel with appropriate grid/block dimensions.
    """
    threads_per_block = 256
    blocks_per_grid = (num_nodes + threads_per_block - 1) // threads_per_block

    _reset_and_flatten_kernel(
        (blocks_per_grid,), (threads_per_block,),
        (parent, rank, cheapest_edge_idx, cheapest_edge_weight, num_nodes)
    )


def _gpu_component_count(parent: "cp.ndarray") -> int:  # type: ignore
    """
    Count number of distinct components in flattened union-find.
    """
    # After flattening, roots are nodes where parent[i] == i
    roots = (parent == cp.arange(len(parent)))
    return int(cp.sum(roots))
```
