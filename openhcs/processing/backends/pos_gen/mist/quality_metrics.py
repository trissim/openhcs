"""
Quality Metrics for MIST Algorithm

Functions for computing correlation quality and adaptive thresholds.
"""
from __future__ import annotations 

from typing import TYPE_CHECKING, List, Tuple, Dict
import logging

from openhcs.core.utils import optional_import

# For type checking only
if TYPE_CHECKING:
    import cupy as cp

# Import CuPy as an optional dependency
cp = optional_import("cupy")


def _validate_cupy_array(array, name: str = "input") -> None:  # type: ignore
    """Validate that the input is a CuPy array."""
    if not isinstance(array, cp.ndarray):
        raise TypeError(f"{name} must be a CuPy array, got {type(array)}")


def compute_correlation_quality_gpu(region1: "cp.ndarray", region2: "cp.ndarray") -> float:  # type: ignore
    """GPU-only normalized cross-correlation quality metric."""
    # Validate input regions have same shape
    if region1.shape != region2.shape:
        return 0.0

    # Check for empty or single-pixel regions
    if region1.size <= 1:
        return 0.0

    r1_flat = region1.flatten()
    r2_flat = region2.flatten()

    # Normalize (GPU)
    r1_mean = cp.mean(r1_flat)
    r2_mean = cp.mean(r2_flat)
    r1_norm = r1_flat - r1_mean
    r2_norm = r2_flat - r2_mean

    # Correlation (GPU)
    numerator = cp.sum(r1_norm * r2_norm)
    denom1 = cp.sqrt(cp.sum(r1_norm ** 2))
    denom2 = cp.sqrt(cp.sum(r2_norm ** 2))

    # Avoid division by zero with more robust threshold (GPU)
    eps = cp.finfo(cp.float32).eps * 1000.0
    correlation = cp.where((denom1 > eps) & (denom2 > eps),
                          cp.abs(numerator / (denom1 * denom2)),
                          cp.float32(0.0))

    return float(correlation)


def compute_correlation_quality_gpu_aligned(region1: "cp.ndarray", region2: "cp.ndarray", dx: float, dy: float) -> float:  # type: ignore
    """
    GPU-only normalized cross-correlation quality metric after applying computed shift.

    This measures how well the regions align after applying the phase correlation shift.
    """
    # Convert shifts to integer pixels for alignment
    shift_x = int(round(dx))
    shift_y = int(round(dy))

    # Get region dimensions
    h1, w1 = region1.shape
    h2, w2 = region2.shape

    # Calculate overlap region after applying shift
    # For horizontal alignment: region1 is left, region2 is right
    # For vertical alignment: region1 is top, region2 is bottom

    # Determine overlap bounds considering the shift
    if abs(shift_x) >= min(w1, w2) or abs(shift_y) >= min(h1, h2):
        # No overlap after shift
        return 0.0

    # Calculate actual overlap region
    if shift_x >= 0:
        # region2 shifted right
        x1_start = max(0, shift_x)
        x1_end = min(w1, w2 + shift_x)
        x2_start = max(0, -shift_x)
        x2_end = min(w2, w1 - shift_x)
    else:
        # region2 shifted left
        x1_start = max(0, -shift_x)
        x1_end = min(w1, w2 - shift_x)
        x2_start = max(0, shift_x)
        x2_end = min(w2, w1 + shift_x)

    if shift_y >= 0:
        # region2 shifted down
        y1_start = max(0, shift_y)
        y1_end = min(h1, h2 + shift_y)
        y2_start = max(0, -shift_y)
        y2_end = min(h2, h1 - shift_y)
    else:
        # region2 shifted up
        y1_start = max(0, -shift_y)
        y1_end = min(h1, h2 - shift_y)
        y2_start = max(0, shift_y)
        y2_end = min(h2, h1 + shift_y)

    # Extract aligned overlap regions
    if x1_end <= x1_start or y1_end <= y1_start or x2_end <= x2_start or y2_end <= y2_start:
        return 0.0

    aligned_region1 = region1[y1_start:y1_end, x1_start:x1_end]
    aligned_region2 = region2[y2_start:y2_end, x2_start:x2_end]

    # Ensure regions have the same size
    min_h = min(aligned_region1.shape[0], aligned_region2.shape[0])
    min_w = min(aligned_region1.shape[1], aligned_region2.shape[1])

    if min_h <= 0 or min_w <= 0:
        return 0.0

    aligned_region1 = aligned_region1[:min_h, :min_w]
    aligned_region2 = aligned_region2[:min_h, :min_w]

    # Compute normalized cross-correlation on aligned regions
    return compute_correlation_quality_gpu(aligned_region1, aligned_region2)


def compute_adaptive_threshold(correlations: "cp.ndarray") -> float:  # type: ignore
    """
    Compute threshold using permutation test like ASHLAR.

    Args:
        correlations: Array of correlation values (CuPy array)

    Returns:
        Adaptive threshold value as float
    """
    _validate_cupy_array(correlations, "correlations")

    # Sample random non-adjacent pairs for null distribution
    # Use 99th percentile as threshold (following ASHLAR approach)
    if len(correlations) == 0:
        return 0.0

    # For small arrays, use all values
    if len(correlations) <= 100:
        sample_correlations = correlations
    else:
        # Sample random subset for efficiency
        n_samples = min(1000, len(correlations))
        indices = cp.random.choice(len(correlations), size=n_samples, replace=False)
        sample_correlations = correlations[indices]

    # Use 99th percentile as adaptive threshold
    threshold = cp.percentile(sample_correlations, 99.0)

    return float(threshold)


def estimate_stage_parameters(
    displacements: "cp.ndarray",  # type: ignore
    expected_spacing: float
) -> tuple[float, float]:
    """
    Estimate repeatability and backlash from measured displacements.

    This implements MIST's key innovation for stage model estimation.

    Args:
        displacements: Array of measured displacements (CuPy array)
        expected_spacing: Expected spacing between tiles

    Returns:
        Tuple of (repeatability, backlash) as floats
    """
    _validate_cupy_array(displacements, "displacements")

    # Estimate repeatability as MAD (Median Absolute Deviation) of displacements
    median_displacement = cp.median(displacements)
    repeatability = cp.median(cp.abs(displacements - median_displacement))

    # Estimate systematic bias (backlash)
    backlash = cp.mean(displacements) - expected_spacing

    return float(repeatability), float(backlash)


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
    tolerance_factor: float = 0.2,
    min_quality: float = 0.3
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
        quality_valid = quality >= min_quality  # Minimum quality threshold

        is_valid = dx_valid and dy_valid and quality_valid
        valid_flags.append(is_valid)

    return valid_flags


def debug_phase_correlation_matrix(
    correlation_matrix: "cp.ndarray",
    peaks: List[Tuple[int, int, float]],
    save_path: str = None
) -> None:
    """
    Create visualization of phase correlation matrix with detected peaks.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib not available, skipping correlation matrix visualization")
        return

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
