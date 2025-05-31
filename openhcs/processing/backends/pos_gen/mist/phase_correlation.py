"""
Phase Correlation Functions for MIST Algorithm

GPU-accelerated phase correlation with subpixel accuracy.
"""

from typing import TYPE_CHECKING, Tuple

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
