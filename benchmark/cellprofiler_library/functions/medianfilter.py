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
    if window_size <= 1:
        return image
    if mode == "reflect" and image.ndim == 2 and window_size > 1:
        try:
            import cv2

            if image.dtype in (np.uint8, np.uint16, np.float32):
                return cv2.medianBlur(
                    np.ascontiguousarray(image),
                    int(window_size),
                ).astype(image.dtype, copy=False)
        except ImportError:
            pass
    
    # Apply median filter
    filtered = scipy_median_filter(image, size=window_size, mode=mode)
    return filtered.astype(image.dtype)


def _medianfilter_batch(
    func,
    slices_2d: tuple,
    kwargs: dict,
    slice_count: int,
    execute_slice,
) -> list:
    """Batch executor for aligned pure-2D median filtering."""
    del func
    window_size = int(kwargs.get("window_size", 3))
    if window_size % 2 == 0:
        window_size += 1
    if window_size <= 1:
        return list(slices_2d)
    if kwargs.get("mode", "reflect") == "reflect":
        try:
            import cv2

            outputs = []
            for slice_2d in slices_2d:
                data = np.asarray(slice_2d)
                if data.ndim != 2 or data.dtype not in (
                    np.uint8,
                    np.uint16,
                    np.float32,
                ):
                    break
                outputs.append(
                    cv2.medianBlur(
                        np.ascontiguousarray(data),
                        int(window_size),
                    ).astype(data.dtype, copy=False)
                )
            else:
                return outputs
        except ImportError:
            pass
    return [
        execute_slice(
            medianfilter,
            slice_2d,
            kwargs,
            slice_index,
            slice_count,
        )
        for slice_index, slice_2d in enumerate(slices_2d)
    ]


medianfilter.__openhcs_pure_2d_batch_executor__ = _medianfilter_batch
