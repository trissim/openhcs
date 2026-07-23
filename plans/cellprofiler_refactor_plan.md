# CellProfiler Absorbed Functions Refactoring Plan

**Date:** 2025-12-27
**Status:** Architecture Review Phase
**Scope:** 88 absorbed CellProfiler functions in `benchmark/cellprofiler_library/functions/`

---

## Executive Summary

This plan addresses critical architectural issues discovered in the absorbed CellProfiler functions:

1. **Contract Mismatch**: 41 functions use `PURE_2D` contract (meant for external libraries) instead of `PURE_3D` (for native OpenHCS functions)
2. **Special Outputs Format**: Functions return lists instead of aggregated structures (inconsistent with existing OpenHCS functions like `skan_axon_analysis`)
3. **Missing 3D Support**: Functions with CellProfiler 3D variants need `FLEXIBLE` contract
4. **Tuple Handling Bug**: `_execute_pure_2d` doesn't handle tuple returns (special outputs)

---

## Background Context

### CellProfiler 3D Support

CellProfiler 3.0+ supports both:
- **Plane-wise processing**: 2D slice-by-slice analysis
- **Volumetric processing**: True 3D algorithms with z-connectivity

**Sources:**
- [CellProfiler 3.0: Next-generation image processing](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.2005970)
- [How to replicate Identify modules on volumetric images](https://cellprofiler-manual.s3.amazonaws.com/CellProfiler-4.0.7/help/other_3d_identify.html)
- [CellProfiler goes 3D - Allen Institute](https://alleninstitute.org/news/cellprofiler-goes-3d/)

### Processing Contract Semantics

**PURE_2D** (for external library functions):
- Function expects 2D input
- Framework unstacks 3D → 2D slices
- Calls function on each slice
- Framework restacks results

**PURE_3D** (default for OpenHCS native functions):
- Function expects 3D input
- Function handles internal slicing if needed
- No framework unstack/restack

**FLEXIBLE** (for functions with both modes):
- Framework auto-injects `slice_by_slice: bool` parameter
- When `slice_by_slice=True`: Framework unstacks/restacks (like PURE_2D)
- When `slice_by_slice=False`: Pass-through (like PURE_3D)
- Function writes 3D logic; framework handles the rest

### Special Outputs Semantics

**Current OpenHCS pattern** (from `skan_axon_analysis`):
```python
@special_outputs(("axon_analysis", materialize_fn))
def analyze(...) -> Tuple[np.ndarray, Dict[str, Any]]:
    # Single aggregated dict for ALL slices
    results = {
        'slice_indices': [0, 1, 2, ...],
        'measurements': [...]
    }
    return image_stack, results
```

**Current absorbed functions pattern** (incorrect):
```python
@special_outputs(("stats", materialize_fn))
def analyze(...) -> Tuple[np.ndarray, List[StatsObject]]:
    # List of per-slice objects
    results = [stats0, stats1, stats2, ...]
    return image_stack, results
```

**Target pattern** (consistent aggregation):
Special outputs should be single structures (dict/DataFrame/3D array), not lists.

---

## Problem Analysis

### Issue 1: Contract Mismatch

**Current state:**
```python
@numpy(contract=ProcessingContract.PURE_2D)  # ← WRONG
def identify_primary_objects(image: np.ndarray, ...):
    # 2D logic, no context about what slice this is
```

**Why this is wrong:**
- PURE_2D is for external libraries (scikit-image, pyclesperanto) discovered via runtime testing
- Absorbed functions are OpenHCS native code
- Framework calls function with each slice, but function has no slice index context
- Special outputs can't properly track slice_index

### Issue 2: Tuple Handling in _execute_pure_2d

**Current implementation** (`unified_registry.py:367-373`):
```python
def _execute_pure_2d(self, func, image, *args, **kwargs):
    memory_type = func.output_memory_type
    slices = unstack_slices(image, memory_type, 0)
    results = [func(sl, *args, **kwargs) for sl in slices]
    return stack_slices(results, memory_type, 0)  # ← CRASH on tuples
```

**Problem:**
- If function returns `(image, stats, labels)`, results = `[(img0, s0, l0), (img1, s1, l1), ...]`
- `stack_slices()` expects list of 2D arrays, not tuples
- Crashes at validation: `if not _is_2d(slice_data)` fails on tuples

### Issue 3: Special Outputs Format

**Inconsistency:**
- `count_cells_single_channel`: Returns `List[CellCountResult]`
- `skan_axon_skeletonize_and_analyze`: Returns `Dict[str, Any]`
- Absorbed functions: Return `List[DataclassObject]`

**Target:** All special outputs should be single aggregated structures.

---

## Function Categorization

### FLEXIBLE Contract (14 functions)
**Support both 2D slice-by-slice and true 3D volumetric processing**

| Function | Has 3D Variant | Special Outputs | Notes |
|----------|---------------|-----------------|-------|
| dilateobjects.py | ✓ (dilate_objects_3d) | ✓ | Merge variants |
| erodeobjects.py | Helpers only | ✓ | |
| expandorshrinkobjects.py | 8 helpers | ✓ | |
| fillobjects.py | | ✓ | |
| removeholes.py | ✓ (_3d variant) | ✗ | |
| resizeobjects.py | ✓ (_3d variant) | ✓ | |
| shrinktoobjectcenters.py | ✓ (_3d variant) | ✓ | |
| watershed.py | | ✓ | CP's primary 3D seg |
| measureimageskeleton.py | ✓ (4 _3d funcs) | ✓ | |
| measureobjectskeleton.py | | ✓ | |
| morphologicalskeleton.py | ✓ (_3d variant) | ✗ | |
| measureobjectsizeshape.py | volumetric param | ✓ | |
| saveimages.py | ✓ (_3d variant) | ✓ | |
| makeprojection.py | | ✓ | Native z-stack |

**12 out of 14 have special outputs** → Must handle tuples in `_execute_flexible` → `_execute_pure_2d` path

### PURE_3D Contract (74 functions)
**Always process slices internally, no true 3D algorithm in CellProfiler**

#### Identification Modules (6)
- identifyprimaryobjects.py (CP docs: "2D only, use Watershed for 3D")
- identifysecondaryobjects.py
- identifytertiaryobjects.py
- identifyobjectsingrid.py
- identifyobjectsmanually.py
- identifydeadworms.py

#### Measurement Modules (14)
- measurecolocalization.py
- measuregranularity.py
- measureimageareaoccupied.py
- measureimageintensity.py
- measureimageoverlap.py
- measureimagequality.py
- measureobjectintensity.py
- measureobjectintensitydistribution.py
- measureobjectneighbors.py
- measureobjectoverlap.py
- measuretexture.py
- calculatemath.py
- calculatestatistics.py
- relateobjects.py

#### Image Processing (20)
- closing.py, opening.py, morph.py
- dilateimage.py, erodeimage.py
- gaussianfilter.py, medianfilter.py, smooth.py, reducenoise.py
- enhanceedges.py, enhanceorsuppressfeatures.py
- correctilluminationapply.py, correctilluminationcalculate.py
- rescaleintensity.py, invertforprinting.py
- imagemath.py, unmixcolors.py
- threshold.py, findmaxima.py
- medialaxis.py

#### Image Manipulation (12)
- crop.py, resize.py, tile.py
- flipandrotate.py
- maskimage.py, maskobjects.py
- colortogray.py, graytocolor.py
- convertimagetoobjects.py, convertobjectstoimage.py
- overlayobjects.py, overlayoutlines.py

#### Classification & Filtering (5)
- classifyobjects.py
- filterobjects.py
- combineobjects.py
- splitormergeobjects.py
- matchtemplate.py

#### Object Operations (4)
- editobjectsmanually.py
- definegrid.py
- labelimages.py
- trackobjects.py

#### Worm-specific (2)
- straightenworms.py
- untangleworms.py

#### Display/Export (7)
- displaydataonimage.py, displaydensityplot.py, displayhistogram.py
- displayplatemap.py, displayscatterplot.py
- exporttodatabase.py, exporttospreadsheet.py

#### Utility (4)
- createbatchfiles.py, savecroppedobjects.py
- flagimage.py, runimagejmacro.py

---

## Refactoring Plan

### Phase 0: Fix Core Infrastructure (CRITICAL)

**File:** `openhcs/processing/backends/lib_registry/unified_registry.py`

**Fix `_execute_pure_2d` to handle tuple returns:**

```python
def _execute_pure_2d(self, func, image, *args, **kwargs):
    """Execute 2D→2D function with unstack/restack wrapper."""
    memory_type = func.output_memory_type
    slices = unstack_slices(image, memory_type, 0)
    results = [func(sl, *args, **kwargs) for sl in slices]

    # Handle tuple returns (functions with @special_outputs)
    if results and isinstance(results[0], tuple):
        # Transpose: [(m1,s1,l1), (m2,s2,l2)] → ([m1,m2], [s1,s2], [l1,l2])
        separated = list(zip(*results))

        # Stack main output (first element)
        stacked_main = stack_slices(list(separated[0]), memory_type, 0)

        # Special outputs stay as lists (same format as current functions expect)
        # NOTE: This is temporary - Phase 2/3 will refactor to aggregated format
        special_outputs_lists = [list(col) for col in separated[1:]]

        return (stacked_main, *special_outputs_lists)

    # Single output - normal stacking
    return stack_slices(results, memory_type, 0)
```

**Why this is first:**
- Without this fix, FLEXIBLE functions with `slice_by_slice=True` will crash
- Blocks all testing of FLEXIBLE contract functions
- Low risk, high impact fix

**Validation:**
- Create test with mock function returning tuple
- Verify unstacking, processing, and restacking works
- Verify tuple structure is preserved

---

### Phase 1: Refactor FLEXIBLE Functions (14 functions)

**Goal:** Merge 2D/_3d variants, implement true 3D logic, use aggregated special outputs

#### Step 1.1: Pilot Implementation (2 functions)

**Function 1: dilateobjects.py** (has separate _3d variant)
**Function 2: measureobjectsizeshape.py** (has volumetric parameter)

**Changes per function:**

1. **Merge variants into single function:**
```python
# Before: Two separate functions
@numpy(contract=ProcessingContract.PURE_2D)
def dilate_objects(...): ...

@numpy(contract=ProcessingContract.PURE_3D)
def dilate_objects_3d(...): ...

# After: Single FLEXIBLE function
@numpy(contract=ProcessingContract.FLEXIBLE)
@special_outputs(("dilation_stats", materialize_fn), ("dilated_labels", materialize_fn))
def dilate_objects(
    image: np.ndarray,  # 3D input (Z, Y, X)
    labels: np.ndarray,  # 3D labels
    structuring_element_shape: StructuringElementShape = StructuringElementShape.BALL,
    structuring_element_size: int = 1,
    # slice_by_slice auto-injected by FLEXIBLE contract
) -> Tuple[np.ndarray, Dict, np.ndarray]:
    """
    Dilate labeled objects using morphological dilation.

    Supports both 2D slice-by-slice (slice_by_slice=True) and
    true 3D volumetric dilation (slice_by_slice=False).
    """
    # Write TRUE 3D logic
    # Framework handles unstacking if slice_by_slice=True

    if labels.ndim == 3:
        # True 3D processing
        props_before = regionprops(labels.astype(np.int32))
        volumes_before = [p.area for p in props_before]  # 'area' is volume in 3D

        # Create 3D structuring element
        if structuring_element_shape == StructuringElementShape.BALL:
            selem = ball(structuring_element_size)
        elif structuring_element_shape == StructuringElementShape.CUBE:
            size = 2 * structuring_element_size + 1
            selem = np.ones((size, size, size), dtype=bool)
        else:
            selem = ball(structuring_element_size)

        # Perform grey dilation on 3D labels
        dilated_labels = grey_dilation(labels.astype(np.int32), footprint=selem)

        props_after = regionprops(dilated_labels)
        volumes_after = [p.area for p in props_after]

        # Aggregated stats dict (not list!)
        stats = {
            'object_count': len(props_after),
            'mean_volume_before': float(np.mean(volumes_before)) if volumes_before else 0.0,
            'mean_volume_after': float(np.mean(volumes_after)) if volumes_after else 0.0,
        }

        return image, stats, dilated_labels.astype(np.float32)

    else:
        # 2D fallback (shouldn't happen with FLEXIBLE, but defensive)
        raise ValueError(f"Expected 3D input, got {labels.ndim}D")
```

2. **Convert special outputs to aggregated format:**
   - Change from `List[DilationStats]` → `Dict[str, Any]`
   - Single dict contains all measurements across slices
   - When `slice_by_slice=True`, framework unstacks, function processes each as 3D with Z=1

3. **Update materialization functions:**
   - Accept dict instead of list
   - Convert dict to DataFrame/CSV

**Validation:**
- Test with `slice_by_slice=True` (framework unstacks)
- Test with `slice_by_slice=False` (true 3D)
- Verify special outputs are properly aggregated
- Verify materialization works

#### Step 1.2: Batch Update Remaining FLEXIBLE Functions (12 functions)

Apply same pattern to:
- erodeobjects.py
- expandorshrinkobjects.py
- fillobjects.py
- removeholes.py
- resizeobjects.py
- shrinktoobjectcenters.py
- watershed.py
- measureimageskeleton.py
- measureobjectskeleton.py
- morphologicalskeleton.py
- saveimages.py
- makeprojection.py

**Automation opportunity:**
- Script to update contract from PURE_2D → FLEXIBLE
- Manual merge of _3d variants
- Manual conversion of special outputs format

---

### Phase 2: Refactor PURE_3D Functions with Special Outputs (41 functions)

**Goal:** Change contract, internalize slicing, aggregate special outputs

**Example: identifyprimaryobjects.py**

```python
# Before
@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(("object_stats", csv_materializer(...)), ("labels", materialize_fn))
def identify_primary_objects(image: np.ndarray, ...) -> Tuple[np.ndarray, ObjectStats, np.ndarray]:
    # 2D logic, no slice context
    stats = ObjectStats(slice_index=0, ...)  # ← Wrong index!
    return image, stats, labels

# After
@numpy  # Default PURE_3D
@special_outputs(("object_stats", csv_materializer(...)), ("labels", materialize_fn))
def identify_primary_objects(
    image_stack: np.ndarray,  # 3D input (Z, Y, X)
    min_diameter: int = 10,
    max_diameter: int = 40,
    ...
) -> Tuple[np.ndarray, Dict, np.ndarray]:
    """
    Identify primary objects in 3D stack (processed slice-by-slice).

    Note: This is 2D-only in CellProfiler. For true 3D segmentation, use watershed.
    """
    if image_stack.ndim != 3:
        raise ValueError(f"Expected 3D input, got {image_stack.ndim}D")

    # Aggregate stats across all slices
    stats = {
        'slice_indices': [],
        'object_counts': [],
        'mean_areas': [],
        'median_areas': [],
        'thresholds_used': []
    }

    # Pre-allocate 3D labels array
    labels_3d = np.zeros_like(image_stack, dtype=np.int32)

    # Process each slice internally
    for z in range(image_stack.shape[0]):
        slice_img = image_stack[z]

        # ... 2D processing logic ...
        labels_2d, count, mean_area, median_area, threshold = process_slice(...)

        # Aggregate into dict
        stats['slice_indices'].append(z)
        stats['object_counts'].append(count)
        stats['mean_areas'].append(mean_area)
        stats['median_areas'].append(median_area)
        stats['thresholds_used'].append(threshold)

        # Store in 3D array
        labels_3d[z] = labels_2d

    return image_stack, stats, labels_3d
```

**Changes:**
1. Remove `contract=ProcessingContract.PURE_2D`
2. Accept 3D input
3. Internal loop over slices
4. Aggregate special outputs into dict (not list)
5. Return 3D arrays for image outputs

**Affected functions:** 41 functions currently using PURE_2D with special_outputs (see earlier categorization)

---

### Phase 3: Refactor PURE_3D Functions without Special Outputs (33 functions)

**Goal:** Change contract, accept 3D input, internalize slicing

**Example: gaussianfilter.py**

```python
# Before
@numpy(contract=ProcessingContract.PURE_2D)
def gaussian_filter(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    return gaussian(image, sigma=sigma)

# After
@numpy  # Default PURE_3D
def gaussian_filter(image_stack: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian filter to each slice in 3D stack."""
    if image_stack.ndim != 3:
        raise ValueError(f"Expected 3D input, got {image_stack.ndim}D")

    result = np.zeros_like(image_stack)
    for z in range(image_stack.shape[0]):
        result[z] = gaussian(image_stack[z], sigma=sigma)

    return result
```

**Simpler than Phase 2:**
- No special outputs to aggregate
- Just loop and stack

**Automation opportunity:**
- Script to wrap existing 2D logic in 3D loop
- High success rate for simple filters

---

## Implementation Timeline

### Week 1: Infrastructure
- [ ] Phase 0: Fix `_execute_pure_2d` tuple handling
- [ ] Write comprehensive tests for contract execution
- [ ] Document tuple handling semantics

### Week 2: Pilot FLEXIBLE
- [ ] Phase 1.1: Refactor dilateobjects.py (pilot)
- [ ] Phase 1.1: Refactor measureobjectsizeshape.py (pilot)
- [ ] Review and approve pattern
- [ ] Update materialization functions

### Week 3-4: Batch FLEXIBLE
- [ ] Phase 1.2: Refactor remaining 12 FLEXIBLE functions
- [ ] Test all FLEXIBLE functions with both slice_by_slice modes
- [ ] Update documentation

### Week 5-7: PURE_3D with Special Outputs
- [ ] Phase 2: Refactor 41 PURE_3D functions with special outputs
- [ ] Can be parallelized (independent functions)
- [ ] Test special outputs aggregation

### Week 8-9: PURE_3D without Special Outputs
- [ ] Phase 3: Refactor 33 PURE_3D functions without special outputs
- [ ] Automation script for simple filters
- [ ] Manual review for complex functions

### Week 10: Testing & Documentation
- [ ] End-to-end pipeline tests
- [ ] Performance benchmarks (2D vs 3D modes)
- [ ] Update user documentation
- [ ] Migration guide for existing pipelines

---

## Risk Mitigation

### Breaking Changes
- **Risk:** Existing pipelines using absorbed functions will break
- **Mitigation:**
  - Version bump
  - Migration script to update pipeline files
  - Backward compatibility shim (optional)

### Performance Regression
- **Risk:** Internal looping slower than framework unstacking
- **Mitigation:**
  - Benchmark both approaches
  - Profile hot paths
  - Consider vectorization where possible

### Testing Coverage
- **Risk:** 88 functions is large surface area
- **Mitigation:**
  - Automated test generation for contract compliance
  - Property-based testing for special outputs format
  - Visual inspection of 10% sample

---

## Success Criteria

1. **All functions use correct contracts:**
   - 14 FLEXIBLE: Support both modes
   - 74 PURE_3D: Always 3D input, internal slicing

2. **Consistent special outputs:**
   - All return aggregated structures (dict/DataFrame/3D array)
   - No lists of per-slice objects

3. **Zero crashes:**
   - `_execute_pure_2d` handles tuples
   - `_execute_flexible` works with special outputs
   - All special outputs materialize correctly

4. **Tests pass:**
   - Unit tests for each function
   - Integration tests for contract execution
   - End-to-end pipeline tests

5. **Documentation complete:**
   - Function signatures updated
   - Contract semantics documented
   - Migration guide for users

---

## Open Questions

1. Should we keep backward compatibility with old special outputs format?
2. Do we need migration script for existing .py pipeline files?
3. Should `slice_by_slice` default to True or False for FLEXIBLE functions?
4. Do we benchmark performance difference between approaches?
5. Should we expose 3D capabilities in UI dropdown/toggle?

---

## References

- Architecture docs: `docs/source/architecture/function_registry_system.rst`
- Contract implementation: `openhcs/processing/backends/lib_registry/unified_registry.py`
- Special outputs system: `openhcs/core/pipeline/function_contracts.py`
- Stack utilities: `openhcs/core/memory/stack_utils.py`
- CellProfiler docs: https://cellprofiler-manual.s3.amazonaws.com/

---

**Next Steps:**
1. Review and approve this plan
2. Start Phase 0: Fix `_execute_pure_2d`
3. Implement Phase 1.1 pilots
4. Iterate based on feedback
