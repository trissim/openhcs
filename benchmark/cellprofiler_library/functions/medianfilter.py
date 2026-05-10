"""
Converted from CellProfiler: MedianFilter
Original: medianfilter
"""

import numpy as np
from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.callable_contract import runtime_image_execution_mode
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import (
    pure_2d_batch_executor,
)
from openhcs.processing.backends.cellprofiler.median_filter import median_filter_backend
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


@runtime_image_execution_mode(ImagePayloadExecutionMode.FULL_STACK)
@numpy(contract=ProcessingContract.PURE_2D)
def medianfilter(
    image: np.ndarray,
    window_size: int = 3,
    mode: str = "constant",
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
              Default: 'constant', matching CellProfiler's MedianFilter module
    
    Returns:
        Median filtered image with same shape (H, W)
    """
    return median_filter_backend().filter(
        np.asarray(image),
        window_size=int(window_size),
        mode=str(mode),
    )


pure_2d_batch_executor(median_filter_backend().filter_batch)(medianfilter)
