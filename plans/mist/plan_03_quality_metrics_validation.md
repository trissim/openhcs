# plan_03_quality_metrics_validation.md
## Component: Quality Metrics and Validation Framework

### Objective
Implement comprehensive quality metrics and validation framework to ensure the coordinate system fixes and NIST robustness improvements work correctly. Create testing infrastructure to validate against known-good results and provide debugging capabilities.

### Plan
1. **Quality Metrics Implementation**
   - Implement NIST-style correlation quality assessment
   - Add image content analysis for adaptive thresholding
   - GPU-accelerated quality computation
   - Multi-scale quality evaluation

2. **Validation Framework**
   - Create test cases with known-good stitching results
   - Implement pixel-level accuracy validation
   - Add regression testing for coordinate system fixes
   - Performance benchmarking suite

3. **Debugging and Diagnostics**
   - Add detailed logging for coordinate transformations
   - Implement visualization tools for phase correlation matrices
   - Create debugging output for translation interpretations
   - Add runtime validation checks

4. **Comparative Analysis**
   - Compare GPU vs CPU results
   - Benchmark different phase correlation approaches
   - Validate against NIST reference implementation
   - Performance vs accuracy trade-off analysis

5. **Integration Testing**
   - End-to-end pipeline validation
   - Memory usage and performance profiling
   - Error handling and edge case testing
   - Compatibility with existing OpenHCS components

### Findings

**NIST Quality Assessment Approach:**

1. **Stage Model Validation (Algorithm 9-13):**
   - Estimates overlap and stage repeatability
   - Filters translations by overlap and correlation (ncc >= 0.5)
   - Outlier detection using inter-quartile distance
   - Repeatability-based filtering and reclamation

2. **Translation Quality Metrics:**
   - Normalized cross correlation as primary metric
   - Multiple interpretation testing for robustness
   - Directional consistency validation
   - Stage model compliance checking

3. **Filtering Pipeline:**
   ```
   validTranslations = filterByOverlapAndCorrelation(T, overlap, pou)
   validTranslations = filterOutliers(validTranslations)
   validTranslations = filterByRepeatability(T, validTranslations, r)
   ```

**Current Implementation Gaps:**

1. **Limited Quality Assessment:**
   - Single correlation threshold
   - No adaptive thresholding
   - Missing stage model validation
   - Insufficient outlier detection

2. **Debugging Capabilities:**
   - Limited visualization tools
   - Insufficient logging for coordinate transformations
   - No runtime validation of displacement magnitudes
   - Missing comparative analysis tools

3. **Testing Infrastructure:**
   - No regression testing for coordinate fixes
   - Limited validation against known-good results
   - Missing performance benchmarking
   - No systematic GPU vs CPU comparison

**Validation Requirements:**

1. **Coordinate System Validation:**
   - Test overlap-region to tile-center conversion
   - Validate displacement magnitude ranges
   - Check FFT periodicity handling
   - Verify directional constraints

2. **Quality Metrics Validation:**
   - Compare correlation computations across platforms
   - Validate adaptive thresholding effectiveness
   - Test outlier detection accuracy
   - Benchmark quality assessment performance

3. **End-to-End Testing:**
   - Known-good test image sets
   - Pixel-level stitching accuracy validation
   - Performance regression testing
   - Memory usage profiling

**Implementation Strategy:**

1. **Incremental Validation:**
   - Unit tests for each coordinate transformation
   - Component-level quality metric testing
   - Integration testing with existing pipeline
   - Performance benchmarking at each stage

2. **Comparative Framework:**
   - Side-by-side GPU vs CPU comparison
   - NIST reference implementation validation
   - Different phase correlation approach testing
   - Quality vs performance trade-off analysis

3. **Debugging Infrastructure:**
   - Detailed coordinate transformation logging
   - Phase correlation matrix visualization
   - Translation interpretation debugging
   - Runtime validation and error reporting

### Implementation Draft

**Implementation Targets with Line Numbers:**

1. **Add adaptive quality functions to quality_metrics.py** (after existing functions):
   - Insert `compute_adaptive_quality_threshold()` function
   - Insert `validate_translation_consistency()` function
   - Insert debugging and visualization functions

2. **Update quality threshold logic in mist_main.py**:
   - Line 127: Replace `if quality >= quality_threshold:` with adaptive threshold
   - Line 167: Replace `if quality >= quality_threshold:` with adaptive threshold
   - Lines 185-190: Enhance quality filtering summary with validation

3. **Add validation calls in _global_optimization_gpu_only** (mist_main.py:69-191):
   - After line 191: Add translation consistency validation
   - Add displacement magnitude validation before MST inclusion

4. **Add debugging output in mist_main.py**:
   - Lines 74-89: Enhance existing debug output with coordinate validation
   - Add visualization calls for correlation matrices (optional)

**Adaptive Quality Thresholding**

```python
def compute_adaptive_quality_threshold(
    all_qualities: List[float],
    base_threshold: float = 0.3,
    percentile_threshold: float = 0.25
) -> float:
    """
    Compute adaptive quality threshold based on distribution of correlation values.

    Based on NIST stage model validation approach.
    """
    if not all_qualities:
        return base_threshold

    qualities_array = cp.array(all_qualities)

    # Remove invalid correlations
    valid_qualities = qualities_array[qualities_array >= 0]

    if len(valid_qualities) == 0:
        return base_threshold

    # Use percentile-based threshold
    percentile_value = float(cp.percentile(valid_qualities, percentile_threshold * 100))

    # Ensure minimum threshold
    adaptive_threshold = max(base_threshold, percentile_value)

    return adaptive_threshold

def validate_translation_consistency(
    translations: List[Tuple[float, float, float]],
    expected_spacing: Tuple[float, float],
    tolerance_factor: float = 0.2
) -> List[bool]:
    """
    Validate translation consistency against expected grid spacing.

    Based on NIST stage model validation.
    """
    expected_dx, expected_dy = expected_spacing
    tolerance_dx = expected_dx * tolerance_factor
    tolerance_dy = expected_dy * tolerance_factor

    valid_flags = []

    for dy, dx, quality in translations:
        # Check if displacement is within expected range
        dx_valid = abs(dx - expected_dx) <= tolerance_dx
        dy_valid = abs(dy - expected_dy) <= tolerance_dy
        quality_valid = quality >= 0.3  # Minimum quality threshold

        is_valid = dx_valid and dy_valid and quality_valid
        valid_flags.append(is_valid)

    return valid_flags
```

**Debugging and Visualization Tools**

```python
def debug_phase_correlation_matrix(
    correlation_matrix: "cp.ndarray",
    peaks: List[Tuple[int, int, float]],
    save_path: str = None
) -> None:
    """
    Create visualization of phase correlation matrix with detected peaks.
    """
    import matplotlib.pyplot as plt

    # Convert to CPU for visualization
    corr_cpu = cp.asnumpy(correlation_matrix)

    plt.figure(figsize=(10, 8))
    plt.imshow(corr_cpu, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Correlation Value')

    # Mark detected peaks
    for i, (y, x, value) in enumerate(peaks):
        plt.plot(x, y, 'bo', markersize=8, label=f'Peak {i+1}: {value:.3f}')

    plt.legend()
    plt.title('Phase Correlation Matrix with Detected Peaks')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()

def log_coordinate_transformation(
    original_dy: float, original_dx: float,
    tile_dy: float, tile_dx: float,
    direction: str,
    tile_index: Tuple[int, int]
) -> None:
    """
    Log coordinate transformation details for debugging.
    """
    logging.info(f"Coordinate Transform - Tile {tile_index}, Direction: {direction}")
    logging.info(f"  Original (overlap coords): dy={original_dy:.2f}, dx={original_dx:.2f}")
    logging.info(f"  Transformed (tile coords): dy={tile_dy:.2f}, dx={tile_dx:.2f}")
    logging.info(f"  Delta: dy_delta={tile_dy-original_dy:.2f}, dx_delta={tile_dx-original_dx:.2f}")
```

**Performance Benchmarking Framework**

```python
def benchmark_phase_correlation_methods(
    test_images: List[Tuple["cp.ndarray", "cp.ndarray"]],
    methods: Dict[str, callable],
    num_iterations: int = 10
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark different phase correlation methods for performance and accuracy.
    """
    import time

    results = {}

    for method_name, method_func in methods.items():
        print(f"Benchmarking {method_name}...")

        times = []
        accuracies = []

        for iteration in range(num_iterations):
            start_time = time.time()

            total_error = 0.0
            num_pairs = 0

            for img1, img2 in test_images:
                try:
                    dy, dx = method_func(img1, img2)
                    # Compute error against known ground truth if available
                    # For now, just measure consistency
                    total_error += abs(dy) + abs(dx)  # Placeholder
                    num_pairs += 1
                except Exception as e:
                    print(f"Error in {method_name}: {e}")
                    continue

            elapsed_time = time.time() - start_time
            times.append(elapsed_time)

            if num_pairs > 0:
                avg_error = total_error / num_pairs
                accuracies.append(avg_error)

        results[method_name] = {
            'avg_time': sum(times) / len(times),
            'std_time': cp.std(cp.array(times)),
            'avg_accuracy': sum(accuracies) / len(accuracies) if accuracies else float('inf'),
            'std_accuracy': cp.std(cp.array(accuracies)) if len(accuracies) > 1 else 0.0
        }

    return results
```

**Specific Code Changes:**

**File: quality_metrics.py, end of file (new functions)**
```python
# Add all validation and debugging functions above
```

**File: mist_main.py, Lines 127 and 167 (adaptive thresholding)**
```python
# OLD:
if quality >= quality_threshold:

# NEW:
# Compute adaptive threshold after collecting all qualities
if conn_idx == 0:  # First connection, initialize adaptive threshold
    adaptive_threshold = quality_threshold
else:
    adaptive_threshold = compute_adaptive_quality_threshold(
        all_qualities, base_threshold=quality_threshold
    )

if quality >= adaptive_threshold:
```

**File: mist_main.py, After line 191 (validation)**
```python
# Add translation consistency validation
if conn_idx > 0:
    # Extract all translations for validation
    translations = []
    for i in range(conn_idx):
        translations.append((
            float(connection_dy[i]),
            float(connection_dx[i]),
            float(connection_quality[i])
        ))

    # Validate consistency
    valid_flags = validate_translation_consistency(
        translations, (float(expected_dx), float(expected_dy))
    )

    num_valid = sum(valid_flags)
    print(f"🔥 TRANSLATION VALIDATION: {num_valid}/{len(valid_flags)} translations are consistent")
```

**File: mist_main.py, Lines 74-89 (enhanced debugging)**
```python
# Add coordinate validation to existing debug output
print(f"🔥 COORDINATE VALIDATION:")
print(f"   Expected tile spacing: dx={float(expected_dx):.1f}, dy={float(expected_dy):.1f}")
print(f"   Overlap regions: H*ratio={H*overlap_ratio:.1f}, W*ratio={W*overlap_ratio:.1f}")
print(f"   Min overlap: {min_overlap_pixels} pixels")
```
