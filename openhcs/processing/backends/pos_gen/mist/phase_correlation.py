"""
Phase Correlation Functions for MIST Algorithm

GPU-accelerated phase correlation with subpixel accuracy.
"""
from __future__ import annotations 

from typing import TYPE_CHECKING, Tuple, List

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


def constrained_hill_climbing(
    correlation_surface: "cp.ndarray",  # type: ignore
    initial_peak: Tuple[int, int],
    max_shift: int
) -> Tuple[float, float]:
    """
    Find optimal shift within constrained region using gradient ascent.

    Args:
        correlation_surface: 2D correlation surface (CuPy array)
        initial_peak: (y, x) coordinates of initial peak
        max_shift: Maximum allowed shift from initial peak

    Returns:
        Tuple of (dy, dx) refined shift values
    """
    _validate_cupy_array(correlation_surface, "correlation_surface")

    if correlation_surface.ndim != 2:
        raise ValueError(f"Correlation surface must be 2D, got {correlation_surface.ndim}D")

    h, w = correlation_surface.shape
    y_init, x_init = initial_peak

    # Define search bounds
    y_min = max(0, y_init - max_shift)
    y_max = min(h, y_init + max_shift + 1)
    x_min = max(0, x_init - max_shift)
    x_max = min(w, x_init + max_shift + 1)

    # Extract constrained region
    region = correlation_surface[y_min:y_max, x_min:x_max]

    if region.size == 0:
        return float(y_init), float(x_init)

    # Find peak in constrained region
    peak_idx = cp.unravel_index(cp.argmax(region), region.shape)
    y_peak_local = peak_idx[0]
    x_peak_local = peak_idx[1]

    # Convert back to global coordinates
    y_peak_global = y_min + y_peak_local
    x_peak_global = x_min + x_peak_local

    # Subpixel refinement using center of mass in 3x3 neighborhood
    if (1 <= y_peak_local < region.shape[0] - 1 and
        1 <= x_peak_local < region.shape[1] - 1):

        # Extract 3x3 neighborhood around peak
        neighborhood = region[y_peak_local-1:y_peak_local+2,
                             x_peak_local-1:x_peak_local+2]

        # Compute center of mass
        total_mass = cp.sum(neighborhood)
        if total_mass > 0:
            y_coords, x_coords = cp.mgrid[0:3, 0:3]
            y_com = cp.sum(y_coords * neighborhood) / total_mass
            x_com = cp.sum(x_coords * neighborhood) / total_mass

            # Adjust to global coordinates with subpixel precision
            y_refined = y_min + y_peak_local - 1 + y_com
            x_refined = x_min + x_peak_local - 1 + x_com

            return float(y_refined), float(x_refined)

    return float(y_peak_global), float(x_peak_global)


def phase_correlation_gpu_only(
    image1: "cp.ndarray",  # type: ignore
    image2: "cp.ndarray",  # type: ignore
    *,
    window: bool = True,
    subpixel: bool = True,
    subpixel_radius: int = 3,
    regularization_eps_multiplier: float = 1000.0
) -> Tuple[float, float]:
    """
    Full GPU phase correlation with all operations on device.
    
    Args:
        image1: First image (CuPy array)
        image2: Second image (CuPy array)
        window: Apply Hann window
        subpixel: Enable subpixel accuracy
        subpixel_radius: Radius for subpixel interpolation
        regularization_eps_multiplier: Multiplier for numerical stability
    
    Returns:
        (dy, dx) shift values
    """
    _validate_cupy_array(image1, "image1")
    _validate_cupy_array(image2, "image2")

    if image1.shape != image2.shape:
        raise ValueError(f"Images must have the same shape, got {image1.shape} and {image2.shape}")

    # Ensure float32 and remove DC component (all GPU operations)
    img1 = image1.astype(cp.float32)
    img2 = image2.astype(cp.float32)
    
    img1 = img1 - cp.mean(img1)
    img2 = img2 - cp.mean(img2)

    # Apply Hann window (all GPU)
    if window:
        h, w = img1.shape
        win_y = cp.hanning(h).reshape(-1, 1)
        win_x = cp.hanning(w).reshape(1, -1)
        window_2d = win_y * win_x
        img1 = img1 * window_2d
        img2 = img2 * window_2d

    # FFT operations (GPU)
    fft1 = cp.fft.fft2(img1)
    fft2 = cp.fft.fft2(img2)

    # Cross-power spectrum with configurable regularization (GPU)
    cross_power = fft1 * cp.conj(fft2)
    magnitude = cp.abs(cross_power)

    # More robust regularization - use relative threshold
    eps = cp.finfo(cp.float32).eps * regularization_eps_multiplier
    magnitude_threshold = cp.maximum(eps, cp.mean(magnitude) * 1e-6)
    cross_power_norm = cross_power / (magnitude + magnitude_threshold)

    # Inverse FFT (GPU)
    correlation = cp.real(cp.fft.ifft2(cross_power_norm))

    # Find peak (GPU)
    peak_idx = cp.unravel_index(cp.argmax(correlation), correlation.shape)
    y_peak = peak_idx[0]  # Keep as CuPy scalar
    x_peak = peak_idx[1]  # Keep as CuPy scalar

    # Convert to signed shifts (GPU arithmetic)
    # For FFT shift conversion, peaks in second half represent negative shifts
    h, w = correlation.shape
    dy = cp.where(y_peak < h // 2, y_peak, y_peak - h)
    dx = cp.where(x_peak < w // 2, x_peak, x_peak - w)

    # Subpixel refinement (all GPU)
    if subpixel:
        # Convert to int for indexing
        y_peak_int = int(y_peak)
        x_peak_int = int(x_peak)
        
        y_min = cp.maximum(0, y_peak_int - subpixel_radius)
        y_max = cp.minimum(h, y_peak_int + subpixel_radius + 1)
        x_min = cp.maximum(0, x_peak_int - subpixel_radius)
        x_max = cp.minimum(w, x_peak_int + subpixel_radius + 1)

        region = correlation[y_min:y_max, x_min:x_max]
        
        total_mass = cp.sum(region)
        if total_mass > 0:
            # Create local coordinates for the region, then convert to global
            region_h, region_w = region.shape
            y_local, x_local = cp.mgrid[0:region_h, 0:region_w]

            # Calculate center of mass in local coordinates
            y_com_local = cp.sum(y_local * region) / total_mass
            x_com_local = cp.sum(x_local * region) / total_mass

            # Convert local COM to global coordinates
            y_com = y_min + y_com_local
            x_com = x_min + x_com_local

            # Apply same FFT coordinate conversion for subpixel values
            dy = cp.where(y_com < h // 2, y_com, y_com - h)
            dx = cp.where(x_com < w // 2, x_com, x_com - w)

    return float(dy), float(dx)


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


def _find_multiple_peaks_gpu(
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


def _test_fft_interpretations(
    correlation_matrix: "cp.ndarray",
    peak_y: int,
    peak_x: int,
    direction: str
) -> List[Tuple[int, int]]:
    """
    Generate FFT periodicity interpretations with directional constraints.

    Args:
        correlation_matrix: Phase correlation matrix
        peak_y, peak_x: Peak coordinates
        direction: 'horizontal' or 'vertical' for directional constraints

    Returns:
        List of (y, x) interpretation coordinates
    """
    h, w = correlation_matrix.shape
    interpretations = []

    # NIST Algorithm 5: Test 16 interpretations with directional constraints
    if direction == 'horizontal':
        # Left-right pairs: test (x, ±y) with 4 FFT possibilities
        for y_sign in [1, -1]:
            for x_offset in [0, w]:  # FFT periodicity in x
                for y_offset in [0, h]:  # FFT periodicity in y
                    interp_x = (peak_x + x_offset) % w
                    interp_y = (peak_y * y_sign + y_offset) % h
                    interpretations.append((interp_y, interp_x))

    elif direction == 'vertical':
        # Up-down pairs: test (±x, y) with 4 FFT possibilities
        for x_sign in [1, -1]:
            for x_offset in [0, w]:  # FFT periodicity in x
                for y_offset in [0, h]:  # FFT periodicity in y
                    interp_x = (peak_x * x_sign + x_offset) % w
                    interp_y = (peak_y + y_offset) % h
                    interpretations.append((interp_y, interp_x))

    # Remove duplicates while preserving order
    seen = set()
    unique_interpretations = []
    for interp in interpretations:
        if interp not in seen:
            seen.add(interp)
            unique_interpretations.append(interp)

    return unique_interpretations


def _compute_interpretation_quality(
    region1: "cp.ndarray",
    region2: "cp.ndarray",
    dy: float,
    dx: float
) -> float:
    """
    Compute quality for a specific displacement interpretation.

    Args:
        region1, region2: Input image regions
        dy, dx: Displacement to test

    Returns:
        Normalized cross-correlation quality
    """
    # Pre-center regions
    r1_mean = cp.mean(region1)
    r2_mean = cp.mean(region2)
    r1_centered = region1 - r1_mean
    r2_centered = region2 - r2_mean

    shift_y, shift_x = int(round(dy)), int(round(dx))
    h, w = r1_centered.shape

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
    r1_overlap = r1_centered[y1_start:y1_end, x1_start:x1_end]
    r2_overlap = r2_centered[y2_start:y2_end, x2_start:x2_end]

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
