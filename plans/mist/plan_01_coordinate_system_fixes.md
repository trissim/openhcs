# plan_01_coordinate_system_fixes.md
## Component: Phase Correlation Coordinate System Fixes

### Objective
Fix the fundamental coordinate system mismatch in the MIST phase correlation implementation that's causing severe stitching failures. The current implementation returns overlap-region-relative displacements (±51 pixels) but the MST algorithm expects tile-center-to-tile-center displacements (~921 pixels).

### Plan
1. **Analyze Current Coordinate System Bug**
   - Phase correlation returns displacements relative to overlap regions
   - MST expects absolute tile-to-tile center displacements
   - Quality filtering eliminates expected connections, causing MST to find "accidental" small displacements

2. **Implement Coordinate System Conversion**
   - Add proper conversion from overlap-region coordinates to tile-center coordinates
   - Ensure `expected_dx + dx` logic executes correctly
   - Validate coordinate transformations match NIST specification

3. **Fix Quality Threshold Issues**
   - Lower quality thresholds to prevent elimination of valid connections
   - Implement adaptive thresholding based on image content
   - Add fallback mechanisms for low-quality image pairs

4. **Add Coordinate System Validation**
   - Implement coordinate system unit tests
   - Add runtime validation of displacement magnitudes
   - Create debugging output for coordinate transformations

### Findings

**Current Implementation Analysis:**
From `analysis_notes.md`, the key issues identified:

1. **Coordinate System Mismatch:**
   - Phase correlation returns overlap-region-relative displacements
   - Expected tile spacing is ~921 pixels, but getting ±51 pixel corrections
   - The `expected_dx + dx` conversion isn't working properly

2. **Quality Filtering Problems:**
   - Quality threshold too high, eliminating valid connections
   - MST finding "accidental" connections with small displacements
   - Need adaptive quality assessment

3. **GPU vs CPU Differences:**
   - Algorithm works perfectly on CPU but fails on GPU
   - Suggests GPU-specific coordinate system bugs
   - Numerical precision or memory layout issues

**NIST Algorithm Specification:**
From the NIST documentation, the correct approach:

1. **Translation Interpretation (Algorithm 5):**
   - Tests 16 possible interpretations due to FFT periodicity
   - Uses directional constraints (left-right vs up-down pairs)
   - Selects interpretation with maximum normalized cross correlation

2. **Coordinate System:**
   - Translations are tile-to-tile displacements in pixels
   - x increases left to right, y increases top to bottom
   - Translation tuple: `<ncc, x, y>` where x,y are pixel displacements

3. **Multiple Peak Testing:**
   - Tests n=2 peaks by default (configurable)
   - Each peak represents potential translation
   - Handles FFT periodicity ambiguity

**Root Cause Analysis:**
The current implementation likely has:
- Incorrect coordinate space conversion from overlap regions to tile centers
- Missing directional constraints for translation interpretation
- Inadequate handling of FFT periodicity (only testing single peak)
- Quality thresholds not adapted for GPU numerical differences

### Implementation Draft

**Critical Finding from Cross-Validation:**
After analyzing the NIST reference implementation, the fundamental issue is clear:

1. **NIST uses 8-16 interpretation testing per peak** - our implementation only tests 1
2. **NIST uses directional constraints** - we have none
3. **NIST uses multiple peak testing (n=2 default)** - we only find 1 peak
4. **NIST uses normalized cross-power spectrum** - we use Hann windowing

**Phase 1: Fix Coordinate System Conversion**

```python
def _convert_overlap_to_tile_coordinates(
    dy: float, dx: float,
    overlap_h: int, overlap_w: int,
    tile_h: int, tile_w: int,
    direction: str
) -> Tuple[float, float]:
    """
    Convert overlap-region-relative displacements to tile-center coordinates.

    Args:
        dy, dx: Phase correlation displacements in overlap region coordinates
        overlap_h, overlap_w: Overlap region dimensions
        tile_h, tile_w: Full tile dimensions
        direction: 'horizontal' or 'vertical'

    Returns:
        (tile_dy, tile_dx): Displacements in tile-center coordinates
    """
    if direction == 'horizontal':
        # For horizontal connections (left-right)
        # Expected displacement is approximately tile_w - overlap_w
        expected_dx = tile_w - overlap_w
        tile_dx = expected_dx + dx  # Add phase correlation correction
        tile_dy = dy  # Vertical should be minimal

    elif direction == 'vertical':
        # For vertical connections (top-bottom)
        # Expected displacement is approximately tile_h - overlap_h
        expected_dy = tile_h - overlap_h
        tile_dy = expected_dy + dy  # Add phase correlation correction
        tile_dx = dx  # Horizontal should be minimal

    return tile_dy, tile_dx
```

**Phase 2: Implement Multiple Interpretation Testing**

```python
def _test_fft_interpretations(
    correlation_matrix: "cp.ndarray",
    peak_y: int, peak_x: int,
    direction: str
) -> List[Tuple[int, int]]:
    """
    Generate all possible interpretations for FFT periodicity.

    Based on NIST Algorithm 5 - tests 8 interpretations with directional constraints.
    """
    h, w = correlation_matrix.shape

    # Four FFT periodicity possibilities
    fft_interpretations = [
        (peak_y, peak_x),
        (peak_y, w - peak_x),
        (h - peak_y, peak_x),
        (h - peak_y, w - peak_x)
    ]

    # Apply directional constraints (NIST optimization)
    if direction == 'horizontal':
        # Left-right pairs: test (x, ±y)
        interpretations = []
        for fy, fx in fft_interpretations:
            interpretations.extend([
                (fy, fx),
                (-fy, fx)  # Only vary y direction
            ])
    else:  # vertical
        # Up-down pairs: test (±x, y)
        interpretations = []
        for fy, fx in fft_interpretations:
            interpretations.extend([
                (fy, fx),
                (fy, -fx)  # Only vary x direction
            ])

    return interpretations
```

**Phase 3: Implement Multi-Peak Testing**

```python
def _find_multiple_peaks_gpu(
    correlation_matrix: "cp.ndarray",
    n_peaks: int = 2
) -> List[Tuple[int, int, float]]:
    """
    Find top n peaks in correlation matrix (GPU implementation).

    Based on NIST Algorithm 4 - finds n highest values.
    """
    # Flatten and find top n indices
    flat_corr = correlation_matrix.flatten()
    top_indices = cp.argpartition(flat_corr, -n_peaks)[-n_peaks:]

    peaks = []
    for idx in top_indices:
        y, x = cp.unravel_index(idx, correlation_matrix.shape)
        value = correlation_matrix[y, x]
        peaks.append((int(y), int(x), float(value)))

    # Sort by correlation value (highest first)
    peaks.sort(key=lambda p: p[2], reverse=True)
    return peaks
```

**Phase 4: Integrate with Quality Assessment**

```python
def _compute_interpretation_quality(
    region1: "cp.ndarray",
    region2: "cp.ndarray",
    dy: float, dx: float
) -> float:
    """
    Compute normalized cross correlation for a specific interpretation.

    Based on NIST cross_correlation function.
    """
    # Apply shift and extract overlapping regions
    shift_y, shift_x = int(round(dy)), int(round(dx))

    # Extract overlapping regions after applying shift
    if shift_y >= 0 and shift_x >= 0:
        r1_overlap = region1[shift_y:, shift_x:]
        r2_overlap = region2[:-shift_y if shift_y > 0 else None,
                           :-shift_x if shift_x > 0 else None]
    # Handle negative shifts...

    if r1_overlap.size == 0 or r2_overlap.size == 0:
        return -1.0

    # Compute normalized cross correlation
    r1_flat = r1_overlap.flatten().astype(cp.float32)
    r2_flat = r2_overlap.flatten().astype(cp.float32)

    r1_flat -= cp.mean(r1_flat)
    r2_flat -= cp.mean(r2_flat)

    numerator = cp.dot(r1_flat, r2_flat)
    denominator = cp.sqrt(cp.dot(r1_flat, r1_flat) * cp.dot(r2_flat, r2_flat))

    if denominator == 0:
        return -1.0

    return float(numerator / denominator)
```

**Implementation Targets with Line Numbers:**

1. **Replace phase_correlation_gpu_only function** (phase_correlation.py:94-197):
   - Add NIST multi-peak testing and interpretation logic
   - Replace single peak detection with n_peaks parameter
   - Add directional constraints for FFT interpretation

2. **Update _global_optimization_gpu_only** (mist_main.py:131-132, 171-172):
   - Fix coordinate system conversion at lines where expected_dx/dy are added
   - Current: `connection_dx[conn_idx] = expected_dx + dx`
   - Should use proper coordinate space conversion function

3. **Add new phase_correlation_nist_gpu function** (phase_correlation.py:198+):
   - Insert after existing phase_correlation_gpu_only function
   - Implement NIST Algorithm 4 (multi-peak) and Algorithm 5 (interpretation testing)

4. **Update quality threshold logic** (mist_main.py:127, 167):
   - Replace fixed quality_threshold with adaptive thresholding
   - Add validation of displacement magnitudes before MST inclusion

**Specific Code Changes:**

**File: phase_correlation.py, Line 198+ (new function)**
```python
def phase_correlation_nist_gpu(
    image1: "cp.ndarray", image2: "cp.ndarray",
    direction: str, n_peaks: int = 2
) -> Tuple[float, float, float]:
    # Implementation from plan_02 GPU NIST robustness
```

**File: mist_main.py, Lines 131-132 (coordinate fix)**
```python
# OLD:
connection_dx[conn_idx] = expected_dx + dx
connection_dy[conn_idx] = dy

# NEW:
tile_dy, tile_dx = _convert_overlap_to_tile_coordinates(
    dy, dx, int(overlap_h), int(overlap_w), H, W, 'horizontal'
)
connection_dx[conn_idx] = tile_dx
connection_dy[conn_idx] = tile_dy
```

**File: mist_main.py, Lines 171-172 (coordinate fix)**
```python
# OLD:
connection_dx[conn_idx] = dx
connection_dy[conn_idx] = expected_dy + dy

# NEW:
tile_dy, tile_dx = _convert_overlap_to_tile_coordinates(
    dy, dx, int(overlap_h), int(overlap_w), H, W, 'vertical'
)
connection_dx[conn_idx] = tile_dx
connection_dy[conn_idx] = tile_dy
```
