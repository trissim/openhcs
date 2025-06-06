# plan_02_gpu_nist_robustness.md
## Component: GPU-Accelerated NIST Robustness Techniques

### Objective
Implement NIST MIST's proven robustness techniques in a GPU-native manner while preserving the superior architecture of the current OpenHCS implementation. Add multiple peak testing, directional constraints, and adaptive quality metrics without sacrificing GPU performance.

### Plan
1. **Multiple Peak Testing (GPU-Native)**
   - Implement GPU-accelerated multi-peak detection in phase correlation matrix
   - Test n=2 peaks by default (configurable parameter)
   - GPU kernel for finding top-n peaks efficiently
   - Parallel testing of multiple translation interpretations

2. **FFT Periodicity Handling**
   - Implement 16-interpretation testing for each peak
   - Add directional constraints (left-right vs up-down pairs)
   - GPU-optimized interpretation testing
   - Select interpretation with maximum normalized cross correlation

3. **Adaptive Quality Metrics**
   - Implement NIST-style quality assessment
   - Adaptive thresholding based on image characteristics
   - GPU-accelerated normalized cross correlation computation
   - Quality-based translation filtering

4. **Normalized Cross-Power Spectrum Option**
   - Add NIST's `fc / abs(fc)` normalization as configurable option
   - Compare against current Hann windowing approach
   - GPU-optimized complex number operations
   - Benchmarking framework for different approaches

5. **Integration with Existing Architecture**
   - Preserve current GPU-native MST implementation
   - Maintain overlap-region optimization
   - Keep configurable parameters and iterative refinement
   - Ensure compatibility with existing pipeline

### Findings

**NIST Algorithm Strengths to Adopt:**

1. **Multiple Peak Testing (Algorithm 4):**
   ```
   n = 2  // number of peaks to find (configurable)
   Peaks = multiPeakMax(PCM, n)
   foreach peak in Peaks:
       peak.ncc, peak.x, peak.y = interpretTranslation(I1, I2, peak.x, peak.y)
   return max(Peaks)  // peak with maximum ncc
   ```

2. **Translation Interpretation (Algorithm 5):**
   - Tests 16 possible interpretations: 4 FFT periodicities × 4 directions
   - Directional constraints reduce to 8 interpretations
   - Left-right pairs: test (x, ±y) with 4 FFT possibilities
   - Up-down pairs: test (±x, y) with 4 FFT possibilities

3. **Normalized Cross-Power Spectrum:**
   ```
   FC = F1 .* conj(F2)
   PCM = ifft2D(FC ./ abs(FC))  // NIST normalization
   ```
   vs current approach with Hann windowing

4. **Quality Assessment:**
   - Multiple correlation tests per peak
   - Adaptive thresholds based on image content
   - Robust handling of low-SNR cases

**Current Architecture Advantages to Preserve:**

1. **GPU-Native Throughout:**
   - No CPU fallbacks, pure GPU pipeline
   - CuPy-based implementation
   - Parallel processing of multiple image pairs

2. **Overlap-Region Optimization:**
   - More efficient than full-tile correlation
   - Focuses computation where it matters
   - Reduces memory bandwidth requirements

3. **Modern MST Algorithm:**
   - Borůvka with union-find (parallel-friendly)
   - Better than NIST's sequential approach
   - Configurable global optimization

4. **Iterative Refinement:**
   - Multiple passes to improve accuracy
   - Better than NIST's hill climbing
   - Configurable damping and convergence

**Integration Strategy:**
- Add NIST robustness as optional enhancements
- Maintain current architecture as foundation
- Use feature flags for different approaches
- Comprehensive benchmarking framework

**GPU Implementation Challenges:**
1. **Memory Management:**
   - Multiple peak storage and processing
   - Efficient interpretation testing
   - Complex number operations optimization

2. **Parallel Algorithm Design:**
   - Peak detection parallelization
   - Interpretation testing distribution
   - Load balancing across GPU cores

3. **Numerical Precision:**
   - GPU float32 vs CPU float64 differences
   - Consistent results across platforms
   - Robust correlation computations

### Implementation Draft

**Core Enhancement: GPU-Native NIST Phase Correlation**

Based on cross-validation with NIST reference implementation, here's the GPU-accelerated version:

```python
def phase_correlation_nist_gpu(
    image1: "cp.ndarray",
    image2: "cp.ndarray",
    direction: str,
    n_peaks: int = 2,
    use_nist_normalization: bool = True
) -> Tuple[float, float, float]:
    """
    GPU-native implementation of NIST MIST phase correlation with robustness features.

    Args:
        image1, image2: Input images (CuPy arrays)
        direction: 'horizontal' or 'vertical' for directional constraints
        n_peaks: Number of peaks to test (NIST default: 2)
        use_nist_normalization: Use fc/abs(fc) instead of Hann windowing

    Returns:
        (dy, dx, quality): Best displacement and correlation quality
    """
    # Ensure float32 and remove DC component
    img1 = image1.astype(cp.float32)
    img2 = image2.astype(cp.float32)

    img1 = img1 - cp.mean(img1)
    img2 = img2 - cp.mean(img2)

    # FFT operations
    fft1 = cp.fft.fft2(img1)
    fft2 = cp.fft.fft2(img2)

    # Cross-power spectrum
    cross_power = fft1 * cp.conj(fft2)

    if use_nist_normalization:
        # NIST normalization: fc / abs(fc)
        magnitude = cp.abs(cross_power)
        # Prevent division by zero with small epsilon
        eps = cp.finfo(cp.float32).eps * 1000
        cross_power_norm = cross_power / (magnitude + eps)
    else:
        # Current OpenHCS approach with regularization
        magnitude = cp.abs(cross_power)
        eps = cp.finfo(cp.float32).eps * 1000.0
        magnitude_threshold = cp.maximum(eps, cp.mean(magnitude) * 1e-6)
        cross_power_norm = cross_power / (magnitude + magnitude_threshold)

    # Inverse FFT to get correlation matrix
    correlation = cp.real(cp.fft.ifft2(cross_power_norm))

    # Find multiple peaks
    peaks = _find_multiple_peaks_gpu(correlation, n_peaks)

    best_quality = -1.0
    best_dy, best_dx = 0.0, 0.0

    # Test each peak with multiple interpretations
    for peak_y, peak_x, peak_value in peaks:
        interpretations = _test_fft_interpretations(
            correlation, peak_y, peak_x, direction
        )

        # Test each interpretation
        for interp_y, interp_x in interpretations:
            # Convert to signed displacements
            h, w = correlation.shape
            dy = interp_y if interp_y < h // 2 else interp_y - h
            dx = interp_x if interp_x < w // 2 else interp_x - w

            # Compute quality for this interpretation
            quality = _compute_interpretation_quality(img1, img2, dy, dx)

            if quality > best_quality:
                best_quality = quality
                best_dy, best_dx = dy, dx

    return float(best_dy), float(best_dx), float(best_quality)
```

**Enhanced Multi-Peak Detection with GPU Optimization**

```python
def _find_multiple_peaks_gpu_optimized(
    correlation_matrix: "cp.ndarray",
    n_peaks: int = 2,
    min_distance: int = 5
) -> List[Tuple[int, int, float]]:
    """
    GPU-optimized multi-peak detection with minimum distance constraint.

    Prevents finding multiple peaks that are too close together.
    """
    h, w = correlation_matrix.shape

    # Use GPU-accelerated peak finding
    flat_corr = correlation_matrix.flatten()

    # Find top candidates (more than needed)
    n_candidates = min(n_peaks * 4, flat_corr.size)
    top_indices = cp.argpartition(flat_corr, -n_candidates)[-n_candidates:]

    # Convert to 2D coordinates and sort by value
    candidates = []
    for idx in top_indices:
        y, x = cp.unravel_index(idx, correlation_matrix.shape)
        value = correlation_matrix[y, x]
        candidates.append((int(y), int(x), float(value)))

    candidates.sort(key=lambda p: p[2], reverse=True)

    # Apply minimum distance constraint
    selected_peaks = []
    for y, x, value in candidates:
        # Check distance from already selected peaks
        too_close = False
        for sel_y, sel_x, _ in selected_peaks:
            distance = cp.sqrt((y - sel_y)**2 + (x - sel_x)**2)
            if distance < min_distance:
                too_close = True
                break

        if not too_close:
            selected_peaks.append((y, x, value))

        if len(selected_peaks) >= n_peaks:
            break

    return selected_peaks
```

**GPU-Accelerated Interpretation Testing**

```python
def _test_interpretations_batch_gpu(
    region1: "cp.ndarray",
    region2: "cp.ndarray",
    interpretations: List[Tuple[int, int]]
) -> List[float]:
    """
    Batch test multiple interpretations on GPU for efficiency.

    Processes all interpretations in parallel where possible.
    """
    qualities = []

    # Pre-compute common values
    r1_mean = cp.mean(region1)
    r2_mean = cp.mean(region2)
    r1_centered = region1 - r1_mean
    r2_centered = region2 - r2_mean

    for dy, dx in interpretations:
        quality = _compute_interpretation_quality_fast(
            r1_centered, r2_centered, dy, dx
        )
        qualities.append(quality)

    return qualities

def _compute_interpretation_quality_fast(
    region1_centered: "cp.ndarray",
    region2_centered: "cp.ndarray",
    dy: float, dx: float
) -> float:
    """
    Fast GPU implementation of interpretation quality assessment.

    Uses pre-centered regions for efficiency.
    """
    shift_y, shift_x = int(round(dy)), int(round(dx))
    h, w = region1_centered.shape

    # Calculate overlap bounds
    y1_start = max(0, shift_y)
    y1_end = min(h, h + shift_y)
    x1_start = max(0, shift_x)
    x1_end = min(w, w + shift_x)

    y2_start = max(0, -shift_y)
    y2_end = min(h, h - shift_y)
    x2_start = max(0, -shift_x)
    x2_end = min(w, w - shift_x)

    # Extract overlapping regions
    r1_overlap = region1_centered[y1_start:y1_end, x1_start:x1_end]
    r2_overlap = region2_centered[y2_start:y2_end, x2_start:x2_end]

    if r1_overlap.size == 0 or r2_overlap.size == 0:
        return -1.0

    # Ensure same size (should be guaranteed by bounds calculation)
    min_h = min(r1_overlap.shape[0], r2_overlap.shape[0])
    min_w = min(r1_overlap.shape[1], r2_overlap.shape[1])

    r1_overlap = r1_overlap[:min_h, :min_w]
    r2_overlap = r2_overlap[:min_h, :min_w]

    # GPU-accelerated correlation computation
    r1_flat = r1_overlap.flatten()
    r2_flat = r2_overlap.flatten()

    numerator = cp.dot(r1_flat, r2_flat)
    norm1 = cp.linalg.norm(r1_flat)
    norm2 = cp.linalg.norm(r2_flat)

    denominator = norm1 * norm2

    if denominator == 0:
        return -1.0

    return float(numerator / denominator)
```

**Implementation Targets with Line Numbers:**

1. **Add NIST functions to phase_correlation.py** (after line 198):
   - Insert `phase_correlation_nist_gpu()` function
   - Insert helper functions: `_find_multiple_peaks_gpu()`, `_test_fft_interpretations()`
   - Insert `_compute_interpretation_quality()` function

2. **Update mist_main.py phase correlation calls**:
   - Line 113-118: Replace `phase_correlation_gpu_only()` with `phase_correlation_nist_gpu()`
   - Line 153-158: Replace `phase_correlation_gpu_only()` with `phase_correlation_nist_gpu()`
   - Add direction parameter: 'horizontal' for horizontal connections, 'vertical' for vertical

3. **Add configuration parameters to mist_compute_tile_positions** (mist_main.py:221-245):
   - Add `use_nist_robustness: bool = True` parameter
   - Add `n_peaks: int = 2` parameter
   - Add `use_nist_normalization: bool = True` parameter

4. **Update quality computation calls** (mist_main.py:121, 161):
   - Replace `compute_correlation_quality_gpu_aligned()` calls
   - Use quality value returned from NIST phase correlation

**Specific Code Changes:**

**File: phase_correlation.py, Line 198+ (new functions)**
```python
def phase_correlation_nist_gpu(
    image1: "cp.ndarray", image2: "cp.ndarray",
    direction: str, n_peaks: int = 2,
    use_nist_normalization: bool = True
) -> Tuple[float, float, float]:
    """GPU-native NIST MIST phase correlation with robustness features."""
    # Full implementation from plan draft above

def _find_multiple_peaks_gpu(correlation_matrix, n_peaks=2):
    """Find top n peaks in correlation matrix."""
    # Implementation from plan draft above

def _test_fft_interpretations(correlation_matrix, peak_y, peak_x, direction):
    """Generate FFT periodicity interpretations with directional constraints."""
    # Implementation from plan draft above
```

**File: mist_main.py, Lines 113-118 (horizontal connection)**
```python
# OLD:
dy, dx = phase_correlation_gpu_only(
    left_region, right_region,
    subpixel=subpixel,
    subpixel_radius=subpixel_radius,
    regularization_eps_multiplier=regularization_eps_multiplier
)

# NEW:
if use_nist_robustness:
    dy, dx, quality = phase_correlation_nist_gpu(
        left_region, right_region,
        direction='horizontal',
        n_peaks=n_peaks,
        use_nist_normalization=use_nist_normalization
    )
else:
    dy, dx = phase_correlation_gpu_only(
        left_region, right_region,
        subpixel=subpixel,
        subpixel_radius=subpixel_radius,
        regularization_eps_multiplier=regularization_eps_multiplier
    )
    quality = compute_correlation_quality_gpu_aligned(left_region, right_region, dx, dy)
```

**File: mist_main.py, Lines 153-158 (vertical connection)**
```python
# OLD:
dy, dx = phase_correlation_gpu_only(
    top_region, bottom_region,
    subpixel=subpixel,
    subpixel_radius=subpixel_radius,
    regularization_eps_multiplier=regularization_eps_multiplier
)

# NEW:
if use_nist_robustness:
    dy, dx, quality = phase_correlation_nist_gpu(
        top_region, bottom_region,
        direction='vertical',
        n_peaks=n_peaks,
        use_nist_normalization=use_nist_normalization
    )
else:
    dy, dx = phase_correlation_gpu_only(
        top_region, bottom_region,
        subpixel=subpixel,
        subpixel_radius=subpixel_radius,
        regularization_eps_multiplier=regularization_eps_multiplier
    )
    quality = compute_correlation_quality_gpu_aligned(top_region, bottom_region, dx, dy)
```

**File: mist_main.py, Lines 241-245 (add parameters)**
```python
# Add after line 244:
use_nist_robustness: bool = True,
n_peaks: int = 2,
use_nist_normalization: bool = True,
```
