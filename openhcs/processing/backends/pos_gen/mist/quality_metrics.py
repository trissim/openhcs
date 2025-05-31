"""
Quality Metrics for MIST Algorithm

Functions for computing correlation quality and adaptive thresholds.
"""

from typing import TYPE_CHECKING

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
