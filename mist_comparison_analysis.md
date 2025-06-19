# MIST Algorithm Implementation Comparison: CPU vs GPU

## Executive Summary

This document provides a systematic function-by-function comparison between the CPU reference implementation (NIST Python) and the local GPU implementation (OpenHCS) of the MIST algorithm.

## MIST Algorithm Overview

The CPU reference implementation follows this exact flow:
1. **PCIAM (Phase Correlation)** - `pciam.py`
2. **Stage Model Building** - `stage_model.py` with MLE estimation
3. **Translation Refinement** - `translation_refinement.py` with hill climbing
4. **MST Global Positioning** - `translation_refinement.py` GlobalPositions class

## Phase 1: Translation Computation (PCIAM)

### CPU Reference Implementation Analysis

**File**: `pciam.py` - `compute_pciam()` method

**Core Algorithm**:
```python
# 1. FFT-based phase correlation
fc = scipy.fft.fft2(t1_img.astype(np.float32)) * np.conj(scipy.fft.fft2(t2_img.astype(np.float32)))

# 2. NIST normalization with epsilon clipping
np.clip(fc.real, a_min=1e-16, a_max=None, out=fc.real)
np.clip(fc.imag, a_min=1e-16, a_max=None, out=fc.imag)
fc = np.nan_to_num(fc, nan=1e-16, copy=False)
fcn = fc / np.abs(fc)

# 3. Inverse FFT
pcm = np.real(scipy.fft.ifft2(fcn))

# 4. Multi-peak search (n_peaks=2 default)
indices = pcm.argpartition(pcm.size - n_peaks, axis=None)[-n_peaks:]

# 5. For each peak, test 8 interpretations based on direction
for ind in indices:
    y, x = np.unravel_index(ind, pcm.shape)
    if t1.r == t2.r:  # horizontal pair
        peak = PCIAM.peak_cross_correlation_lr(t1_img, t2_img, x, y)
    else:  # vertical pair
        peak = PCIAM.peak_cross_correlation_ud(t1_img, t2_img, x, y)
```

**Key Features**:
- **Full tile correlation**: Correlates entire tiles, not just overlap regions
- **8 interpretation testing**: Tests `[(y,x), (y,w-x), (h-y,x), (h-y,w-x), (-y,x), (-y,w-x), (-(h-y),x), (-(h-y),w-x)]`
- **Direction-aware**: Different interpretation sets for horizontal vs vertical pairs
- **NCC validation**: Each interpretation validated with `compute_cross_correlation()`

### GPU Implementation Analysis

**File**: `phase_correlation.py` - `phase_correlation_nist_gpu()` method

**Core Algorithm**:
```python
# 1. FFT-based phase correlation
fft1 = cp.fft.fft2(img1)
fft2 = cp.fft.fft2(img2)
cross_power = fft1 * cp.conj(fft2)

# 2. NIST normalization
if use_nist_normalization:
    magnitude = cp.abs(cross_power)
    eps = cp.finfo(cp.float32).eps * 1000
    cross_power_norm = cross_power / (magnitude + eps)

# 3. Inverse FFT
correlation = cp.real(cp.fft.ifft2(cross_power_norm))

# 4. Multi-peak search
peaks = _find_multiple_peaks_gpu(correlation, n_peaks)

# 5. Test interpretations for each peak
for peak_y, peak_x, peak_value in peaks:
    interpretations = _test_fft_interpretations(correlation, peak_y, peak_x, direction)
    for interp_y, interp_x in interpretations:
        quality = _compute_interpretation_quality(img1, img2, dy, dx)
```

### Critical Differences Identified

#### ❌ **MAJOR DIFFERENCE: Correlation Input**
- **CPU**: Correlates **full tiles** (`t1_img`, `t2_img`)
- **GPU**: Correlates **overlap regions only** (`left_region`, `right_region`)

This is a fundamental architectural difference that affects all downstream calculations.

#### ❌ **MAJOR DIFFERENCE: Interpretation Testing**
- **CPU**: Uses direction-specific interpretation sets (`peak_cross_correlation_lr` vs `peak_cross_correlation_ud`)
- **GPU**: Uses generic `_test_fft_interpretations()` that may not match CPU's direction-aware logic

#### ⚠️ **Epsilon Handling Differences**
- **CPU**: Fixed epsilon `1e-16` with explicit clipping
- **GPU**: Configurable `eps * 1000` without explicit clipping

#### ⚠️ **Peak Selection Method**
- **CPU**: `argpartition(pcm.size - n_peaks, axis=None)[-n_peaks:]`
- **GPU**: Custom `_find_multiple_peaks_gpu()` implementation

**Confidence Assessment**: **LOW** - Major architectural differences in input data and interpretation testing

## Phase 2: Stage Model Building

### CPU Reference Implementation Analysis

**File**: `stage_model.py` - Complete stage model pipeline

**Core Components**:

1. **MLE Overlap Estimation** (`compute_overlap()`)
```python
# Hill climbing with normal/uniform mixture model
while num_stable_iterations < NUMBER_STABLE_MLE_ITERATIONS:
    point = MlePoint.getRandomPoint()
    point = hillClimbSearch(point, cache, translations)
    if point.likelihood > best_point.likelihood:
        best_point = point
        num_stable_iterations = 0
```

2. **Translation Filtering** (`filter_translations()`)
```python
# Filter by overlap bounds
t_min = height - (overlap + overlap_uncertainty) * height / 100.0
t_max = height - (overlap - overlap_uncertainty) * height / 100.0

# Filter by correlation threshold
if t.ncc < valid_correlation_threshold: continue

# IQR outlier removal
valid_tiles = filer_translations_remove_outliers(direction, valid_tiles)
```

3. **Repeatability Computation** (`compute_repeatability()`)
```python
# Method 1: Orthogonal repeatability
t_orthogonal_vals = [t.west_translation.y if direction == 'HORIZONTAL' else t.north_translation.x for t in valid_tiles]
repeatability1 = np.ceil((np.max(t_orthogonal_vals) - np.min(t_orthogonal_vals)) / 2.0)

# Method 2: Per-row/column repeatability
repeatability2 = np.ceil(np.max(np.abs(max_t_list - min_t_list) / 2.0))

# Final repeatability
stage_repeatability = np.max([repeatability1, repeatability2])
```

4. **Invalid Translation Replacement** (`replace_invalid_translations_per_row_col()`)
```python
# Replace invalid translations with median per row/column
for key in med_x_vals.keys():
    med_x_vals[key] = np.median(med_x_vals[key])
    med_y_vals[key] = np.median(med_y_vals[key])
```

### GPU Implementation Analysis

**Files**: `mist_main.py`, `quality_metrics.py`

**What IS Implemented**:

1. **Basic Stage Parameters** (mist_main.py lines 162-164)
```python
overlap_uncertainty_percent: float = 3.0  # NIST default
outlier_threshold_multiplier: float = 1.5  # NIST Algorithm 16
```

2. **Simplified Repeatability Estimation** (quality_metrics.py)
```python
def estimate_stage_parameters(displacements, expected_spacing):
    median_displacement = cp.median(displacements)
    repeatability = cp.median(cp.abs(displacements - median_displacement))  # MAD
    backlash = cp.mean(displacements) - expected_spacing
```

3. **Translation Validation** (quality_metrics.py)
```python
def validate_translation_consistency(translations, expected_spacing, tolerance_factor, min_quality):
    # Check displacement within expected range
    dx_valid = abs(dx - expected_dx) <= tolerance_dx
    dy_valid = abs(dy - expected_dy) <= tolerance_dy
    quality_valid = quality >= min_quality
```

### Critical Differences Identified

#### ❌ **MISSING: MLE Overlap Estimation**
- **CPU**: Full hill climbing with normal/uniform mixture model (`mle_estimator.py`)
- **GPU**: **No equivalent implementation found**

#### ❌ **MISSING: Complete Stage Model Pipeline**
- **CPU**: Full `StageModel.build()` with overlap computation, filtering, repeatability
- **GPU**: **No equivalent orchestration found**

#### ❌ **MISSING: IQR Outlier Filtering**
- **CPU**: `filer_translations_remove_outliers()` with 1.5×IQR threshold
- **GPU**: **No equivalent implementation found**

#### ❌ **MISSING: Translation Replacement**
- **CPU**: `replace_invalid_translations_per_row_col()` with median replacement
- **GPU**: **No equivalent implementation found**

#### ⚠️ **Simplified Repeatability**
- **CPU**: Two-method approach (orthogonal + per-row/column)
- **GPU**: Single MAD-based approach

**Confidence Assessment**: **VERY LOW** - Major stage model components are missing from GPU implementation

## Phase 3: Translation Refinement

### CPU Reference Implementation Analysis

**File**: `translation_refinement.py` - Hill climbing optimization

**Core Algorithm**:
```python
def optimize_direction(tile, other, direction, repeatability, num_hill_climbs):
    # Get original translation
    orig_peak = tile.west_translation if direction == 'west' else tile.north_translation

    # Define search bounds based on repeatability
    x_min = orig_peak.x - repeatability
    x_max = orig_peak.x + repeatability
    y_min = orig_peak.y - repeatability
    y_max = orig_peak.y + repeatability

    # Multi-point hill climbing
    new_peak = multipoint_hill_climb(num_hill_climbs, other, tile, x_min, x_max, y_min, y_max, orig_peak.x, orig_peak.y)

    # Boost correlation for previously valid translations
    if not np.isnan(orig_peak.ncc):
        new_peak.ncc += 3.0  # Priority boost for MST
```

**Hill Climbing Details**:
- **Search space**: ±repeatability around initial translation
- **Multi-point**: 16 random starting points by default
- **Caching**: 2D cache array to avoid recomputing correlations
- **Convergence tracking**: Logs how many hill climbs converged

### GPU Implementation Analysis

**File**: `mist_main.py` - Iterative refinement approach

**Core Algorithm**:
```python
# GPU uses iterative position refinement instead of hill climbing
for iteration in range(refinement_iterations):
    # Compute corrections for each tile
    for r in range(num_rows):
        for c in range(num_cols):
            # Phase correlation with neighbors
            dy, dx = phase_correlation_gpu_only(left_region, current_region)

            # Apply damped correction
            correction_x = refinement_damping * dx
            correction_y = refinement_damping * dy
            new_positions[tile_idx] = old_positions[tile_idx] + [correction_x, correction_y]
```

### Critical Differences Identified

#### ❌ **COMPLETELY DIFFERENT APPROACH**
- **CPU**: Hill climbing optimization within repeatability bounds
- **GPU**: Iterative position refinement with damping

#### ❌ **MISSING: Bounded Search Space**
- **CPU**: Search constrained to ±repeatability around initial translation
- **GPU**: No bounds based on stage model repeatability

#### ❌ **MISSING: Multi-point Hill Climbing**
- **CPU**: 16 random starting points with convergence analysis
- **GPU**: Single deterministic refinement per iteration

#### ❌ **MISSING: Translation Boosting**
- **CPU**: Adds +3.0 to NCC for previously valid translations
- **GPU**: No equivalent priority mechanism

**Confidence Assessment**: **ZERO** - Completely different algorithms

## Phase 4: MST Global Positioning

### CPU Reference Implementation Analysis

**File**: `translation_refinement.py` - `GlobalPositions` class

**Core Algorithm**:
```python
def traverse_minimum_spanning_tree(self):
    # 1. Find highest correlation tile as starting point
    for r in range(grid_height):
        for c in range(grid_width):
            ncc = tile.get_max_translation_ncc()
            if ncc > best_ncc:
                start_tile = tile

    # 2. Greedy MST construction
    while mst_size < target_mst_size:
        best_ncc = -np.inf
        # Find best edge from current MST to unvisited tile
        for tile in frontier_tiles:
            for neighbor in get_neighbors(tile):
                if not visited[neighbor]:
                    edge_weight = tile.get_peak(neighbor).ncc
                    if edge_weight > best_ncc:
                        best_edge = (tile, neighbor)

        # Add best edge to MST
        add_to_mst(best_edge)
```

**Key Features**:
- **Greedy MST**: Prim's algorithm variant
- **Correlation weights**: Uses NCC values as edge weights
- **Frontier tracking**: Maintains set of MST boundary tiles
- **Position propagation**: Updates absolute positions via edge traversal

### GPU Implementation Analysis

**File**: `boruvka_mst.py`, `position_reconstruction.py`

**Core Algorithm**:
```python
def build_mst_gpu_boruvka(connection_from, connection_to, connection_quality, num_nodes):
    # Borůvka's algorithm with union-find
    for iteration in range(max_iterations):
        # 1. Find minimum edge per component
        launch_find_minimum_edges_kernel(edges_from, edges_to, edges_quality, parent, cheapest_edge_idx)

        # 2. Union components
        launch_union_components_kernel(cheapest_edge_idx, parent, rank, mst_edges)

        # 3. Check convergence
        if all_components_merged(): break
```

### Critical Differences Identified

#### ✅ **MST Algorithm Equivalence**
- **CPU**: Prim's algorithm (greedy)
- **GPU**: Borůvka's algorithm (parallel)
- **Both produce equivalent MSTs** for same input

#### ⚠️ **Edge Weight Handling**
- **CPU**: Uses refined NCC values with +3.0 boost
- **GPU**: Uses raw correlation values from phase correlation

#### ⚠️ **Starting Point Selection**
- **CPU**: Highest correlation tile
- **GPU**: Configurable anchor tile (default: tile 0)

**Confidence Assessment**: **MEDIUM** - Different algorithms but mathematically equivalent results expected

## Summary of Critical Differences

### ❌ **FUNDAMENTAL ARCHITECTURAL DIFFERENCES**

1. **Phase Correlation Input Data**
   - **CPU**: Full tile correlation
   - **GPU**: Overlap region correlation only
   - **Impact**: Completely different displacement calculations

2. **Stage Model Pipeline**
   - **CPU**: Complete MLE estimation, filtering, repeatability computation
   - **GPU**: Missing most stage model components
   - **Impact**: No robust outlier handling or overlap estimation

3. **Translation Optimization**
   - **CPU**: Hill climbing within repeatability bounds
   - **GPU**: Iterative refinement without bounds
   - **Impact**: Different optimization strategies

### ⚠️ **IMPLEMENTATION DIFFERENCES**

1. **FFT Interpretation Testing**
   - **CPU**: Direction-aware interpretation sets
   - **GPU**: Generic interpretation testing
   - **Impact**: May produce different peak selections

2. **Edge Weight Handling**
   - **CPU**: Boosted correlations (+3.0) for valid translations
   - **GPU**: Raw correlation values
   - **Impact**: Different MST edge priorities

## Confidence Assessment

### **ZERO Confidence Areas:**
- Overall algorithm equivalence
- Translation optimization pipeline
- Stage model robustness

### **LOW Confidence Areas:**
- Phase correlation equivalence (due to input differences)
- Parameter scaling and conversion
- Quality threshold handling

### **MEDIUM Confidence Areas:**
- MST construction (different algorithms, equivalent results)
- Basic FFT mathematics

### **HIGH Confidence Areas:**
- None - too many fundamental differences identified

## Critical Findings

**The GPU implementation is NOT functionally equivalent to the CPU reference.** Key issues:

1. **Missing Stage Model**: The sophisticated MLE-based overlap estimation and repeatability computation that makes MIST robust is completely absent.

2. **Different Correlation Strategy**: Using overlap regions vs full tiles fundamentally changes the displacement calculations.

3. **No Translation Refinement**: The hill climbing optimization that improves translation accuracy is replaced with a completely different iterative approach.

4. **Simplified Pipeline**: The GPU version appears to be a simplified approximation rather than a faithful port of the NIST algorithm.

## Recommendations

**For Functional Equivalence:**
1. **Implement full tile correlation** instead of overlap-only
2. **Add complete stage model pipeline** with MLE estimation
3. **Implement hill climbing translation refinement**
4. **Add proper translation filtering and replacement**
5. **Implement correlation boosting for MST edge weights**

**For Validation:**
1. **Create side-by-side test suite** with identical inputs
2. **Compare intermediate outputs** at each pipeline stage
3. **Validate on challenging datasets** where robustness matters

**Bottom Line**: The GPU implementation needs substantial rework to achieve functional parity with the CPU reference. It's currently a different algorithm that may work for simple cases but lacks the robustness features that make MIST reliable for challenging microscopy data.

---

# Ashlar Algorithm Implementation Comparison

## Executive Summary

This section compares the CPU reference implementation (labsyspharm/ashlar) with the local GPU implementation (OpenHCS) of the Ashlar stitching algorithm.

## Algorithm Foundation

Based on the Ashlar source code analysis, Ashlar uses a different approach from MIST:
1. **Edge-based alignment** - Computes pairwise shifts between neighboring tiles
2. **Spanning tree construction** - Uses minimum spanning tree for global optimization
3. **Linear model fitting** - Applies linear regression to correct systematic errors

## Core Algorithm Components

### CPU Reference Implementation (Ashlar)
**File**: `reg.py`

**Guaranteed Equivalences:**
- ✅ **Phase correlation**: Uses `scipy.fft.fft2()` with cross-power spectrum
- ✅ **Windowing**: Applies Hann window for improved correlation
- ✅ **Neighbor detection**: City block distance for overlap detection
- ✅ **Spanning tree**: Uses NetworkX for minimum spanning tree construction
- ✅ **Linear model**: sklearn LinearRegression for systematic error correction
- ✅ **Permutation testing**: Statistical threshold determination for edge quality

**Key Data Structures:**
```python
class EdgeAligner:
    max_shift: float          # Maximum allowed shift in micrometers
    alpha: float             # Significance level for permutation testing
    spanning_tree: nx.Graph  # Minimum spanning tree of valid edges
    positions: np.ndarray    # Final corrected positions
```

**Algorithm Flow:**
1. **Thumbnail generation** for coarse alignment
2. **Overlap checking** between neighboring tiles
3. **Threshold computation** via permutation testing
4. **Pairwise registration** of all neighboring tiles
5. **Spanning tree construction** using correlation quality as weights
6. **Position calculation** via tree traversal
7. **Linear model fitting** for systematic error correction

### GPU Implementation (OpenHCS)
**File**: `ashlar_processor_cupy.py`

**Guaranteed Equivalences:**
- ✅ **Phase correlation**: Uses `cp.fft.fft2()` (CuPy equivalent)
- ✅ **Hann windowing**: GPU implementation with `cp.hanning()`
- ✅ **Subpixel refinement**: Center-of-mass subpixel accuracy
- ✅ **Sequential positioning**: Initial positioning via sequential alignment

**Areas of Major Uncertainty:**
- ❌ **No spanning tree**: Missing NetworkX-style MST construction
- ❌ **No permutation testing**: Missing statistical threshold determination
- ❌ **No linear model**: Missing sklearn-style systematic error correction
- ❌ **Simplified neighbor detection**: No proper overlap testing
- ❌ **No thumbnail generation**: Missing coarse alignment phase

**Current GPU Approach:**
- Uses sequential or snake pattern for initial positioning
- Applies global drift correction via linear detrending
- Uses momentum-based iterative refinement
- No statistical quality assessment

## Critical Differences

### 1. Statistical Robustness (HIGH PRIORITY)
**CPU Ashlar**: Uses permutation testing to determine alignment quality thresholds
**GPU**: No statistical validation of alignment quality

### 2. Global Optimization Strategy
**CPU Ashlar**: Minimum spanning tree with edge weights based on correlation quality
**GPU**: Simple iterative refinement with momentum

### 3. Systematic Error Correction
**CPU Ashlar**: Linear regression model to correct stage positioning errors
**GPU**: Basic linear detrending without model fitting

### 4. Edge Quality Assessment
**CPU Ashlar**: Comprehensive error metrics with infinity assignment for bad alignments
**GPU**: No edge quality assessment or rejection

## Confidence Levels

**High Confidence (>95%):**
- Phase correlation math (FFT operations)
- Hann windowing implementation
- Basic subpixel refinement

**Medium Confidence (70-90%):**
- Initial positioning accuracy
- Drift correction effectiveness

**Low Confidence (<70%):**
- Global optimization robustness
- Statistical reliability
- Systematic error handling
- Overall algorithm equivalence to CPU Ashlar

## Major Gaps in GPU Implementation

### 1. Missing Spanning Tree Construction
The GPU version lacks the sophisticated MST-based global optimization that is central to Ashlar's robustness.

### 2. No Statistical Validation
Missing permutation testing means no principled way to reject poor alignments.

### 3. Simplified Error Model
No linear regression model for correcting systematic stage positioning errors.

### 4. No Thumbnail-based Coarse Alignment
Missing the coarse alignment phase that helps with large positioning errors.

## Recommendations for GPU Implementation

1. **Implement MST construction** using GPU-accelerated graph algorithms
2. **Add permutation testing** for statistical threshold determination
3. **Implement linear model fitting** for systematic error correction
4. **Add thumbnail generation** for coarse alignment
5. **Implement proper neighbor detection** with overlap testing
6. **Add comprehensive error metrics** with edge quality assessment

## Overall Assessment

The GPU Ashlar implementation is significantly simplified compared to the CPU reference. While it may work for well-behaved datasets, it lacks the statistical robustness and sophisticated error handling that makes Ashlar reliable for challenging microscopy data.

**Recommendation**: The GPU implementation needs substantial enhancement to achieve parity with the CPU reference algorithm.
