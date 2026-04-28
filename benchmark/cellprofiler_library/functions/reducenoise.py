"""
Converted from CellProfiler: ReduceNoise
Original: reducenoise
"""

import numpy as np
from typing import Optional
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


@numpy(contract=ProcessingContract.PURE_2D)
def reducenoise(
    image: np.ndarray,
    patch_size: int = 5,
    patch_distance: int = 6,
    cutoff_distance: float = 0.1,
) -> np.ndarray:
    """
    Reduce noise in an image using non-local means denoising.
    
    This function applies non-local means denoising which works by comparing
    patches of the image and averaging similar patches to reduce noise while
    preserving edges and details.
    
    Args:
        image: Input image array with shape (H, W)
        patch_size: Size of patches used for denoising. Larger values give
            more smoothing but may blur fine details. Default: 5
        patch_distance: Maximum distance in pixels to search for patches.
            Larger values search more of the image but are slower. Default: 6
        cutoff_distance: Cut-off distance (h parameter) that controls the
            decay of weights as a function of patch distances. Higher values
            give more smoothing. Default: 0.1
    
    Returns:
        Denoised image with same shape as input (H, W)
    """
    from skimage.restoration import denoise_nl_means, estimate_sigma
    
    # Ensure image is float for processing
    if image.dtype != np.float32 and image.dtype != np.float64:
        image = image.astype(np.float32)
    
    # Estimate noise standard deviation if cutoff_distance is very small
    # This helps with automatic parameter selection
    sigma_est = estimate_sigma(image)
    
    # The h parameter in skimage is related to cutoff_distance
    # Scale it by the estimated noise level for better results
    h = cutoff_distance if cutoff_distance > 0.01 else sigma_est * 1.15
    
    # Apply non-local means denoising
    # fast_mode=True uses a faster but slightly less accurate algorithm
    denoised = denoise_nl_means(
        image,
        h=h,
        patch_size=patch_size,
        patch_distance=patch_distance,
        fast_mode=True,
        channel_axis=None,  # 2D grayscale image
    )
    
    return denoised.astype(np.float32)