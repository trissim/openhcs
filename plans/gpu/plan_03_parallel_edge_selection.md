# plan_03_parallel_edge_selection.md
## Component: Parallel Minimum Edge Selection per Component

### Objective
Implement the core Borůvka's operation where each component simultaneously finds its minimum weight outgoing edge. This replaces sequential edge processing with true parallel execution across all components.

### Plan
1. **Design parallel edge scanning kernel**
   - Each thread processes subset of edges
   - Atomic updates for component minimum edges
   - Warp-level reductions for efficiency

2. **Implement component-based edge filtering**
   - Identify edges connecting different components
   - Skip self-edges within same component
   - Handle multi-edges between components

3. **Add atomic minimum edge tracking**
   - Per-component cheapest edge storage
   - Atomic compare-and-swap for edge updates
   - Tie-breaking using edge indices for consistency

4. **Optimize memory access patterns**
   - Sort edges by source component for locality
   - Coalesced memory reads within warps
   - Minimize atomic contention

5. **Integration with union-find**
   - Use flattened component roots for fast lookups
   - Update component information after unions
   - Maintain edge validity across iterations

### Findings
**Current Sequential Limitations:**
- Edge processing in batches: `for start_idx in range(0, len(sorted_from), batch_size)`
- Sequential validation: `valid_mask, parent = _gpu_union_find_operations(...)`
- CPU list accumulation: `selected_from.extend(...)`

**Parallel Edge Selection Requirements:**
- Each component finds minimum edge simultaneously
- No dependencies between component operations
- Atomic updates to prevent race conditions
- Consistent tie-breaking across threads

**Memory Access Optimization:**
- Edges sorted by source component for cache locality
- Warp-level processing of consecutive edges
- Atomic operations clustered by component
- Reduced memory bandwidth requirements

**Atomic Operations Strategy:**
- `atomicMin` for edge weight comparison
- `atomicCAS` for complete edge replacement
- Memory coalescing for better performance
- Warp-level primitives where possible

**Edge Processing Algorithm:**
1. Each thread processes chunk of edge list
2. For each edge (u,v,w):
   - Find components of u and v using flattened union-find
   - If different components, attempt atomic update of minimum edge
   - Use edge index for tie-breaking
3. Warp-level reduction to minimize atomic contention

**Tie-Breaking Strategy:**
- Use edge index in original list for consistent ordering
- Ensures deterministic results across parallel execution
- Prevents cycles from same-weight edges
- Compatible with atomic compare-and-swap operations

### Implementation Draft

```python
import cupyx.jit

@cupyx.jit.rawkernel()
def _find_minimum_edges_kernel(
    edges_from, edges_to, edges_quality, parent,
    cheapest_edge_idx, cheapest_edge_weight, num_edges
):
    """
    Kernel 2: Find minimum weight edge for each component in parallel.

    Uses atomic operations for true parallel edge selection per component.
    """
    tid = cupyx.jit.blockIdx.x * cupyx.jit.blockDim.x + cupyx.jit.threadIdx.x

    if tid < num_edges:
        # Get edge endpoints and their components
        from_node = edges_from[tid]
        to_node = edges_to[tid]
        from_comp = parent[from_node]
        to_comp = parent[to_node]

        # Skip self-edges within same component
        if from_comp == to_comp:
            return

        # Get edge quality (higher is better, so negate for min comparison)
        edge_quality = edges_quality[tid]
        edge_weight = -edge_quality  # Negative for min-heap behavior

        # Try to update cheapest edge for 'from' component
        cupyx.jit.atomic_min(cheapest_edge_weight, from_comp, edge_weight)

        # Check if we actually set the minimum and update edge index
        if cheapest_edge_weight[from_comp] == edge_weight:
            # Use atomic exchange to handle race conditions
            cupyx.jit.atomic_exch(cheapest_edge_idx, from_comp, tid)

        # Try to update cheapest edge for 'to' component
        cupyx.jit.atomic_min(cheapest_edge_weight, to_comp, edge_weight)

        # Check if we actually set the minimum and update edge index
        if cheapest_edge_weight[to_comp] == edge_weight:
            # Use atomic exchange to handle race conditions
            cupyx.jit.atomic_exch(cheapest_edge_idx, to_comp, tid)


@cupyx.jit.rawkernel()
def _find_minimum_edges_with_tiebreaker_kernel(
    edges_from, edges_to, edges_quality, parent,
    cheapest_edge_idx, cheapest_edge_weight, cheapest_edge_tiebreaker,
    num_edges
):
    """
    Enhanced version with tie-breaking for deterministic results.

    Uses edge index as tie-breaker for consistent parallel execution.
    """
    tid = cupyx.jit.blockIdx.x * cupyx.jit.blockDim.x + cupyx.jit.threadIdx.x

    if tid < num_edges:
        from_node = edges_from[tid]
        to_node = edges_to[tid]
        from_comp = parent[from_node]
        to_comp = parent[to_node]

        if from_comp == to_comp:
            return

        edge_quality = edges_quality[tid]
        edge_weight = -edge_quality

        # Update 'from' component with tie-breaking
        _atomic_update_cheapest_with_tiebreaker(
            cheapest_edge_weight, cheapest_edge_idx, cheapest_edge_tiebreaker,
            from_comp, edge_weight, tid, tid  # Use edge index as tie-breaker
        )

        # Update 'to' component with tie-breaking
        _atomic_update_cheapest_with_tiebreaker(
            cheapest_edge_weight, cheapest_edge_idx, cheapest_edge_tiebreaker,
            to_comp, edge_weight, tid, tid
        )


@cupyx.jit.rawkernel()
def _atomic_update_cheapest_with_tiebreaker(
    cheapest_weight, cheapest_idx, cheapest_tiebreaker,
    component, new_weight, new_idx, new_tiebreaker
):
    """
    Atomically update cheapest edge with tie-breaking logic.

    Updates only if new edge is better (lower weight) or same weight but better tie-breaker.
    """
    # Read current values
    current_weight = cheapest_weight[component]
    current_tiebreaker = cheapest_tiebreaker[component]

    # Check if we should update
    should_update = (new_weight < current_weight or
                    (new_weight == current_weight and new_tiebreaker < current_tiebreaker))

    if should_update:
        # Try atomic compare-and-swap for weight
        old_weight = cupyx.jit.atomic_cas(cheapest_weight, component, current_weight, new_weight)

        if old_weight == current_weight:
            # Successfully updated weight, now update index and tie-breaker
            cupyx.jit.atomic_exch(cheapest_idx, component, new_idx)
            cupyx.jit.atomic_exch(cheapest_tiebreaker, component, new_tiebreaker)


def _launch_find_minimum_edges_kernel(
    edges_from: "cp.ndarray",  # type: ignore
    edges_to: "cp.ndarray",  # type: ignore
    edges_quality: "cp.ndarray",  # type: ignore
    parent: "cp.ndarray",  # type: ignore
    cheapest_edge_idx: "cp.ndarray",  # type: ignore
    cheapest_edge_weight: "cp.ndarray",  # type: ignore
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
         cheapest_edge_idx, cheapest_edge_weight, num_edges)
    )
```
