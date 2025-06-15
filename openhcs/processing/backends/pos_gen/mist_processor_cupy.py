"""
MIST (Microscopy Image Stitching Tool) GPU Implementation

This module provides GPU-accelerated MIST implementation using CuPy.
All legacy functions have been moved to the modular implementation in the mist/ subfolder.
"""
from __future__ import annotations 

import logging
from typing import TYPE_CHECKING, Tuple

from openhcs.constants.constants import DEFAULT_PATCH_SIZE, DEFAULT_SEARCH_RADIUS
from openhcs.core.utils import optional_import

# For type checking only
if TYPE_CHECKING:
    import cupy as cp

# Import CuPy as an optional dependency
cp = optional_import("cupy")

logger = logging.getLogger(__name__)


def _validate_cupy_array(array, name: str = "input") -> None:  # type: ignore
    """Validate that the input is a CuPy array."""
    if not isinstance(array, cp.ndarray):
        raise TypeError(f"{name} must be a CuPy array, got {type(array)}")


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
    eps = cp.finfo(cp.float32).eps * regularization_eps_multiplier
    cross_power_norm = cross_power / (magnitude + eps)

    # Inverse FFT (GPU)
    correlation = cp.real(cp.fft.ifft2(cross_power_norm))

    # Find peak (GPU)
    peak_idx = cp.unravel_index(cp.argmax(correlation), correlation.shape)
    y_peak = peak_idx[0]  # Keep as CuPy scalar
    x_peak = peak_idx[1]  # Keep as CuPy scalar

    # Convert to signed shifts (GPU arithmetic)
    h, w = correlation.shape
    dy = cp.where(y_peak <= h // 2, y_peak, y_peak - h)
    dx = cp.where(x_peak <= w // 2, x_peak, x_peak - w)

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
            y_coords, x_coords = cp.mgrid[y_min:y_max, x_min:x_max]
            y_com = cp.sum(y_coords * region) / total_mass
            x_com = cp.sum(x_coords * region) / total_mass
            
            dy = cp.where(y_com <= h // 2, y_com, y_com - h)
            dx = cp.where(x_com <= w // 2, x_com, x_com - w)

    return float(dy), float(dx)


# Import the modular MIST implementation
from .mist.mist_main import mist_compute_tile_positions

# Re-export for backward compatibility
__all__ = ['mist_compute_tile_positions', 'phase_correlation_gpu_only']
