"""
Converted from CellProfiler: MedianFilter
Original: medianfilter
"""

import numpy as np
from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.callable_contract import runtime_image_execution_mode
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import (
    RuntimePure2DSliceBatchRequest,
    pure_2d_batch_executor,
)
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
    from scipy.ndimage import median_filter as scipy_median_filter
    
    # Ensure window_size is odd
    if window_size % 2 == 0:
        window_size += 1
    if window_size <= 1:
        return image
    if mode == "constant" and image.ndim == 2 and window_size > 1:
        try:
            import cv2

            if image.dtype in (np.uint8, np.uint16, np.float32, np.float64):
                pad_width = window_size // 2
                cv2_input_dtype = np.float32 if image.dtype == np.float64 else image.dtype
                padded = np.pad(
                    np.ascontiguousarray(image, dtype=cv2_input_dtype),
                    pad_width,
                    mode="constant",
                    constant_values=0,
                )
                return cv2.medianBlur(padded, int(window_size))[
                    pad_width:-pad_width,
                    pad_width:-pad_width,
                ].astype(image.dtype, copy=False)
        except ImportError:
            pass
    if mode == "reflect" and image.ndim == 2 and window_size > 1:
        try:
            import cv2

            if image.dtype in (np.uint8, np.uint16, np.float32, np.float64):
                cv2_input_dtype = np.float32 if image.dtype == np.float64 else image.dtype
                return cv2.medianBlur(
                    np.ascontiguousarray(image, dtype=cv2_input_dtype),
                    int(window_size),
                ).astype(image.dtype, copy=False)
        except ImportError:
            pass
    
    # Apply median filter
    filtered = scipy_median_filter(image, size=window_size, mode=mode)
    return filtered.astype(image.dtype)


def _medianfilter_batch(request: RuntimePure2DSliceBatchRequest) -> list:
    """Batch executor for aligned pure-2D median filtering."""
    slices_2d = request.slices_2d
    kwargs = request.kwargs
    window_size = int(kwargs.get("window_size", 3))
    if window_size % 2 == 0:
        window_size += 1
    if window_size <= 1:
        return list(slices_2d)
    mode = kwargs.get("mode", "constant")
    if mode == "constant":
        try:
            import cv2

            outputs = []
            pad_width = window_size // 2
            for slice_2d in slices_2d:
                data = np.asarray(slice_2d)
                if data.ndim != 2 or data.dtype not in (
                    np.uint8,
                    np.uint16,
                    np.float32,
                    np.float64,
                ):
                    break
                cv2_input_dtype = np.float32 if data.dtype == np.float64 else data.dtype
                padded = np.pad(
                    np.ascontiguousarray(data, dtype=cv2_input_dtype),
                    pad_width,
                    mode="constant",
                    constant_values=0,
                )
                outputs.append(
                    cv2.medianBlur(padded, int(window_size))[
                        pad_width:-pad_width,
                        pad_width:-pad_width,
                    ].astype(data.dtype, copy=False)
                )
            else:
                return outputs
        except ImportError:
            pass
    if mode == "reflect":
        try:
            import cv2

            outputs = []
            for slice_2d in slices_2d:
                data = np.asarray(slice_2d)
                if data.ndim != 2 or data.dtype not in (
                    np.uint8,
                    np.uint16,
                    np.float32,
                    np.float64,
                ):
                    break
                cv2_input_dtype = np.float32 if data.dtype == np.float64 else data.dtype
                outputs.append(
                    cv2.medianBlur(
                        np.ascontiguousarray(data, dtype=cv2_input_dtype),
                        int(window_size),
                    ).astype(data.dtype, copy=False)
                )
            else:
                return outputs
        except ImportError:
            pass
    slice_arrays = tuple(np.asarray(slice_2d) for slice_2d in slices_2d)
    if len({array.shape for array in slice_arrays}) == 1:
        from scipy.ndimage import median_filter as scipy_median_filter

        stack = np.stack(slice_arrays, axis=0)
        filtered = scipy_median_filter(
            stack,
            size=(1, window_size, window_size),
            mode=mode,
        )
        return [
            filtered[index].astype(array.dtype, copy=False)
            for index, array in enumerate(slice_arrays)
        ]
    try:
        from scipy.ndimage import median_filter as scipy_median_filter

        return [
            scipy_median_filter(array, size=window_size, mode=mode).astype(
                array.dtype,
                copy=False,
            )
            for array in slice_arrays
        ]
    except ImportError:
        return [
            request.execute_one(slice_index)
            for slice_index in range(request.slice_count)
        ]


pure_2d_batch_executor(_medianfilter_batch)(medianfilter)
