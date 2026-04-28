"""
Converted from CellProfiler: GaussianFilter
Original: gaussianfilter
"""

import numpy as np
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


@numpy(contract=ProcessingContract.PURE_2D)
def gaussian_filter(
    image: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Apply Gaussian smoothing filter to an image.
    
    Args:
        image: Input image array with shape (H, W)
        sigma: Standard deviation for Gaussian kernel. Higher values produce
               more smoothing. Default is 1.0.
    
    Returns:
        Smoothed image with same shape as input.
    """
    from scipy.ndimage import gaussian_filter as scipy_gaussian_filter
    
    return scipy_gaussian_filter(image, sigma=sigma)