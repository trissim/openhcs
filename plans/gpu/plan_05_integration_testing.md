# plan_05_integration_testing.md
## Component: Integration and Performance Testing

### Objective
Integrate the full GPU Borůvka's implementation into the MIST cupy processor, ensure correctness, and validate performance improvements over the current CPU-hybrid approach.

### Plan
1. **Replace existing MST implementation**
   - Swap out current `_build_mst_gpu()` with Borůvka's version
   - Maintain backward compatibility with calling code
   - Preserve input validation and error handling

2. **Comprehensive correctness testing**
   - Unit tests for union-find operations
   - Edge case testing (disconnected graphs, single nodes)
   - Comparison with known MST results
   - Stress testing with large graphs

3. **Performance benchmarking**
   - Compare against current Kruskal's implementation
   - Measure GPU memory usage and bandwidth
   - Profile kernel execution times
   - Test scaling with different graph sizes

4. **Integration with MIST workflow**
   - Test within full MIST tile positioning pipeline
   - Validate with real microscopy data
   - Ensure no regressions in positioning accuracy
   - Performance testing with typical MIST workloads

5. **Documentation and optimization**
   - Document algorithm parameters and tuning
   - Identify performance bottlenecks
   - Optimize based on profiling results
   - Create usage guidelines

### Findings
**Integration Points:**
- Current call site: `_global_optimization_gpu_only()` function
- Input: edge arrays (from, to, dx, dy, quality) and node count
- Output: MST edges dictionary with same format
- Error handling: maintain existing validation patterns

**Testing Requirements:**
- Correctness: MST weight must match known algorithms
- Performance: should show speedup over current implementation
- Memory: must fit within GPU memory constraints
- Accuracy: positioning results should be identical or better

**Benchmarking Strategy:**
- Small graphs (10x10 MIST grids): correctness focus
- Medium graphs (20x20 MIST grids): performance comparison
- Large graphs (synthetic): scalability testing
- Real data: integration validation

**Performance Metrics:**
- Total execution time (GPU kernels only)
- Memory bandwidth utilization
- Atomic operation efficiency
- Kernel occupancy and warp efficiency

**Expected Improvements:**
- Elimination of CPU-GPU transfers in MST construction
- Parallel edge processing vs sequential batching
- Better memory access patterns
- Reduced algorithm complexity for dense graphs

**Risk Mitigation:**
- Fallback to current implementation if GPU memory insufficient
- Extensive testing with edge cases
- Performance regression detection
- Gradual rollout with feature flags

**Success Criteria:**
- Correctness: identical or better MST results
- Performance: measurable speedup on typical MIST workloads
- Integration: no regressions in positioning pipeline
- Maintainability: clean, documented implementation

### Implementation Draft
*Code will be written here after smell loop approval*
