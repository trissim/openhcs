"""
Converted from CellProfiler: MedianFilter
Original: medianfilter
"""

import numpy as np
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


@numpy(contract=ProcessingContract.PURE_2D)
def medianfilter(
    image: np.ndarray,
    window_size: int = 3,
    mode: str = "reflect",
) -> np.ndarray:
    """
    Apply median filter to image for noise reduction.
    
    Median filtering is a nonlinear operation that replaces each pixel with
    the median value of neighboring pixels. It is particularly effective at
    removing salt-and-pepper noise while preserving edges.
    
    Args:
        image: Input image array with shape (H, W)
        window_size: Size of the median filter window. Must be odd integer.
                    Larger values provide more smoothing but may blur edges.
                    Default: 3
        mode: How to handle boundaries. Options:
              - 'reflect': Reflect values at boundary (d c b a | a b c d | d c b a)
              - 'constant': Pad with constant value (0)
              - 'nearest': Extend with nearest value (a a a a | a b c d | d d d d)
              - 'mirror': Mirror values at boundary (d c b | a b c d | c b a)
              - 'wrap': Wrap around (a b c d | a b c d | a b c d)
              Default: 'reflect'
    
    Returns:
        Median filtered image with same shape (H, W)
    """
    from scipy.ndimage import median_filter as scipy_median_filter
    
    # Ensure window_size is odd
    if window_size % 2 == 0:
        window_size += 1
    
    # Apply median filter
    filtered = scipy_median_filter(image, size=window_size, mode=mode)
    
    return filtered.astype(image.dtype)