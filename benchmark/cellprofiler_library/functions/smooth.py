"""
Converted from CellProfiler: Smooth
Original: Smooth.run

Smooths (blurs) images using various filtering methods.
"""

import numpy as np
from enum import Enum
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


class SmoothingMethod(Enum):
    FIT_POLYNOMIAL = "fit_polynomial"
    GAUSSIAN_FILTER = "gaussian_filter"
    MEDIAN_FILTER = "median_filter"
    SMOOTH_KEEPING_EDGES = "smooth_keeping_edges"
    CIRCULAR_AVERAGE_FILTER = "circular_average_filter"
    SMOOTH_TO_AVERAGE = "smooth_to_average"


def _fit_polynomial(image: np.ndarray, clip: bool = True) -> np.ndarray:
    """
    Fit a polynomial to the image intensity.
    Fits: A*x^2 + B*y^2 + C*x*y + D*x + E*y + F
    """
    h, w = image.shape
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    
    # Normalize coordinates to [-1, 1] for numerical stability
    x_norm = (x_coords - w/2) / (w/2)
    y_norm = (y_coords - h/2) / (h/2)
    
    # Build design matrix for polynomial fit
    # Columns: x^2, y^2, xy, x, y, 1
    design = np.column_stack([
        (x_norm**2).ravel(),
        (y_norm**2).ravel(),
        (x_norm * y_norm).ravel(),
        x_norm.ravel(),
        y_norm.ravel(),
        np.ones(h * w)
    ])
    
    # Solve least squares
    coeffs, _, _, _ = np.linalg.lstsq(design, image.ravel(), rcond=None)
    
    # Reconstruct fitted image
    output = design @ coeffs
    output = output.reshape(h, w)
    
    if clip:
        output = np.clip(output, 0, 1)
    
    return output.astype(np.float32)


def _circular_average_filter(image: np.ndarray, radius: float) -> np.ndarray:
    """
    Apply circular averaging filter (pillbox filter).
    """
    from scipy.ndimage import convolve
    
    # Create circular kernel
    size = int(2 * radius + 1)
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    kernel = mask.astype(np.float32)
    kernel = kernel / kernel.sum()
    
    return convolve(image, kernel, mode='constant', cval=0)


def _median_filter(image: np.ndarray, radius: float) -> np.ndarray:
    """
    Apply median filter with given radius.
    """
    from scipy.ndimage import median_filter
    
    size = int(2 * radius + 1)
    return median_filter(image, size=size, mode='constant', cval=0)


@numpy(contract=ProcessingContract.PURE_2D)
def smooth(
    image: np.ndarray,
    smoothing_method: SmoothingMethod = SmoothingMethod.GAUSSIAN_FILTER,
    auto_object_size: bool = True,
    object_size: float = 16.0,
    edge_intensity_difference: float = 0.1,
    clip_polynomial: bool = True,
) -> np.ndarray:
    """
    Smooth (blur) an image using various filtering methods.
    
    Args:
        image: Input grayscale image (H, W)
        smoothing_method: Method to use for smoothing
        auto_object_size: If True, calculate artifact diameter automatically
        object_size: Typical artifact diameter in pixels (used if auto_object_size=False)
        edge_intensity_difference: Edge intensity threshold for smooth_keeping_edges method
        clip_polynomial: Whether to clip polynomial fit results to [0, 1]
    
    Returns:
        Smoothed image (H, W)
    """
    from scipy.ndimage import gaussian_filter
    from skimage.restoration import denoise_bilateral
    
    # Determine object size
    if auto_object_size:
        calculated_size = max(1, np.mean(image.shape) / 40)
        calculated_size = min(30, calculated_size)
    else:
        calculated_size = object_size
    
    # Convert object size to sigma (FWHM to sigma conversion)
    sigma = calculated_size / 2.35
    
    if smoothing_method == SmoothingMethod.GAUSSIAN_FILTER:
        output = gaussian_filter(image.astype(np.float64), sigma, mode='constant', cval=0)
        
    elif smoothing_method == SmoothingMethod.MEDIAN_FILTER:
        radius = calculated_size / 2 + 1
        output = _median_filter(image, radius)
        
    elif smoothing_method == SmoothingMethod.SMOOTH_KEEPING_EDGES:
        output = denoise_bilateral(
            image=image.astype(np.float64),
            channel_axis=None,
            sigma_color=edge_intensity_difference,
            sigma_spatial=sigma,
        )
        
    elif smoothing_method == SmoothingMethod.FIT_POLYNOMIAL:
        output = _fit_polynomial(image, clip=clip_polynomial)
        
    elif smoothing_method == SmoothingMethod.CIRCULAR_AVERAGE_FILTER:
        radius = calculated_size / 2 + 1
        output = _circular_average_filter(image, radius)
        
    elif smoothing_method == SmoothingMethod.SMOOTH_TO_AVERAGE:
        mean_val = np.mean(image)
        output = np.full(image.shape, mean_val, dtype=np.float32)
        
    else:
        raise ValueError(f"Unsupported smoothing method: {smoothing_method}")
    
    return output.astype(np.float32)