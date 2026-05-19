"""
CuPy Image Processor Implementation

This module implements the ImageProcessorInterface using CuPy as the backend.
It leverages GPU acceleration for image processing operations.

Doctrinal Clauses:
- Clause 3 — Declarative Primacy: All functions are pure and stateless
- Clause 88 — No Inferred Capabilities: Explicit CuPy dependency
- Clause 106-A — Declared Memory Types: All methods specify CuPy arrays
"""
from __future__ import annotations 

import logging
import os
from abc import abstractmethod
from typing import Any, List, Optional, Tuple

from metaclass_registry import AutoRegisterMeta
from openhcs.core.memory import cupy as cupy_func
from openhcs.core.utils import optional_import
from openhcs.processing.backends.processors.method_axes import (
    RegisteredProcessorMethodStrategy,
)

logger = logging.getLogger(__name__)

# Check if we're in subprocess runner mode and should skip GPU imports
if os.getenv('OPENHCS_SUBPROCESS_NO_GPU') == '1':
    # Subprocess runner mode - skip GPU imports
    cp = None
    ndimage = None
    cucim_filters = None
    logger.info("Subprocess runner mode - skipping cupy import")
else:
    # Normal mode - import CuPy as an optional dependency
    cp = optional_import("cupy")
    ndimage = None
    if cp is not None:
        cupyx_scipy = optional_import("cupyx.scipy")
        if cupyx_scipy is not None:
            ndimage = cupyx_scipy.ndimage

    # Import CuCIM for edge detection
    cucim_filters = optional_import("cucim.skimage.filters")

logger = logging.getLogger(__name__)


@cupy_func
def create_linear_weight_mask(height: int, width: int, margin_ratio: float = 0.1) -> "cp.ndarray":
    """
    Create a 2D weight mask that linearly ramps from 0 at the edges to 1 in the center.

    Args:
        height: Height of the mask
        width: Width of the mask
        margin_ratio: Ratio of the margin to the image size

    Returns:
        2D CuPy weight mask of shape (height, width)
    """
    # The compiler will ensure this function is only called when CuPy is available
    # No need to check for CuPy availability here

    margin_y = int(cp.floor(height * margin_ratio))
    margin_x = int(cp.floor(width * margin_ratio))

    weight_y = cp.ones(height, dtype=cp.float32)
    if margin_y > 0:
        ramp_top = cp.linspace(0, 1, margin_y, endpoint=False)
        ramp_bottom = cp.linspace(1, 0, margin_y, endpoint=False)
        weight_y[:margin_y] = ramp_top
        weight_y[-margin_y:] = ramp_bottom

    weight_x = cp.ones(width, dtype=cp.float32)
    if margin_x > 0:
        ramp_left = cp.linspace(0, 1, margin_x, endpoint=False)
        ramp_right = cp.linspace(1, 0, margin_x, endpoint=False)
        weight_x[:margin_x] = ramp_left
        weight_x[-margin_x:] = ramp_right

    # Create 2D weight mask
    weight_mask = cp.outer(weight_y, weight_x)

    return weight_mask


def _validate_3d_array(array: Any, name: str = "input") -> None:
    """
    Validate that the input is a 3D CuPy array.

    Args:
        array: Array to validate
        name: Name of the array for error messages

    Raises:
        TypeError: If the array is not a CuPy array
        ValueError: If the array is not 3D
        ImportError: If CuPy is not available
    """
    # The compiler will ensure this function is only called when CuPy is available
    # No need to check for CuPy availability here

    if not isinstance(array, cp.ndarray):
        raise TypeError(f"{name} must be a CuPy array, got {type(array)}. "
                       f"No automatic conversion is performed to maintain explicit contracts.")

    if array.ndim != 3:
        raise ValueError(f"{name} must be a 3D array, got {array.ndim}D")

@cupy_func
def sharpen(image: "cp.ndarray", radius: float = 1.0, amount: float = 1.0) -> "cp.ndarray":
    """
    Sharpen a 3D image using unsharp masking - GPU PARALLELIZED.

    This applies sharpening to each Z-slice independently using vectorized operations
    for maximum GPU utilization. Normalization and rescaling are fully parallelized.

    Args:
        image: 3D CuPy array of shape (Z, Y, X)
        radius: Radius of Gaussian blur
        amount: Sharpening strength

    Returns:
        Sharpened 3D CuPy array of shape (Z, Y, X)
    """
    _validate_3d_array(image)

    # Store original dtype
    dtype = image.dtype

    # Convert to float32 for processing
    image_float = image.astype(cp.float32)

    # Vectorized per-slice normalization - GPU PARALLELIZED
    # Get max value per slice: shape (Z, 1, 1) for broadcasting
    max_per_slice = cp.max(image_float, axis=(1, 2), keepdims=True)
    image_norm = image_float / max_per_slice

    # Apply 3D Gaussian blur with sigma_z=0 for slice-wise processing - GPU PARALLELIZED
    # This processes all slices simultaneously while keeping Z-slices independent
    blurred = ndimage.gaussian_filter(image_norm, sigma=(0, radius, radius))

    # Apply unsharp mask: original + amount * (original - blurred) - GPU PARALLELIZED
    sharpened = image_norm + amount * (image_norm - blurred)

    # Clip to valid range - GPU PARALLELIZED
    sharpened = cp.clip(sharpened, 0, 1.0)

    # Vectorized rescaling back to original range - GPU PARALLELIZED
    min_per_slice = cp.min(sharpened, axis=(1, 2), keepdims=True)
    max_per_slice = cp.max(sharpened, axis=(1, 2), keepdims=True)

    # Avoid division by zero using broadcasting
    range_per_slice = max_per_slice - min_per_slice
    # Only rescale slices where max > min
    valid_range = range_per_slice > 0
    sharpened_rescaled = cp.where(
        valid_range,
        (sharpened - min_per_slice) * 65535 / range_per_slice,
        sharpened * 65535
    )

    # Convert back to original dtype
    return sharpened_rescaled.astype(dtype)

@cupy_func
def percentile_normalize(
    image: "cp.ndarray",
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    target_min: float = None,
    target_max: float = None,
    preserve_dtype: bool = True
) -> "cp.ndarray":
    """
    Normalize a 3D image using percentile-based contrast stretching.

    This applies normalization to each Z-slice independently using slice-by-slice
    processing for algorithmic reasons (each slice needs different percentile values).
    Each slice operation is GPU parallelized across Y,X dimensions.

    Args:
        image: 3D CuPy array of shape (Z, Y, X)
        low_percentile: Lower percentile (0-100)
        high_percentile: Upper percentile (0-100)
        target_min: Target minimum value (auto-detected from dtype if None)
        target_max: Target maximum value (auto-detected from dtype if None)
        preserve_dtype: If True, output same dtype as input. If False, use target range.

    Returns:
        Normalized 3D CuPy array of shape (Z, Y, X) with same or specified dtype
    """
    _validate_3d_array(image)

    # Import shared utilities
    from .percentile_utils import resolve_target_range, slice_percentile_normalize_core

    # Auto-detect target range based on input dtype if not specified
    target_min, target_max = resolve_target_range(image.dtype, target_min, target_max)

    # Use shared core logic with CuPy-specific functions
    return slice_percentile_normalize_core(
        image=image,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
        target_min=target_min,
        target_max=target_max,
        percentile_func=cp.percentile,
        clip_func=cp.clip,
        ones_like_func=cp.ones_like,
        zeros_like_func=lambda arr, dtype=None: cp.zeros_like(arr, dtype=dtype or cp.float32),
        preserve_dtype=preserve_dtype
    )

@cupy_func
def stack_percentile_normalize(
    stack: "cp.ndarray",
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    target_min: float = None,
    target_max: float = None,
    preserve_dtype: bool = True
) -> "cp.ndarray":
    """
    Normalize a stack using global percentile-based contrast stretching.

    This ensures consistent normalization across all Z-slices by computing
    global percentiles across the entire stack.

    Args:
        stack: 3D CuPy array of shape (Z, Y, X)
        low_percentile: Lower percentile (0-100)
        high_percentile: Upper percentile (0-100)
        target_min: Target minimum value (auto-detected from dtype if None)
        target_max: Target maximum value (auto-detected from dtype if None)
        preserve_dtype: If True, output same dtype as input. If False, use target range.

    Returns:
        Normalized 3D CuPy array of shape (Z, Y, X) with same or specified dtype
    """
    _validate_3d_array(stack)

    # Import shared utilities
    from .percentile_utils import resolve_target_range, percentile_normalize_core

    # Auto-detect target range based on input dtype if not specified
    target_min, target_max = resolve_target_range(stack.dtype, target_min, target_max)

    # Use shared core logic with CuPy-specific functions
    return percentile_normalize_core(
        stack=stack,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
        target_min=target_min,
        target_max=target_max,
        percentile_func=lambda arr, pct: cp.percentile(arr, pct),
        clip_func=cp.clip,
        ones_like_func=cp.ones_like,
        preserve_dtype=preserve_dtype
    )

@cupy_func
def create_composite(
    stack: "cp.ndarray", weights: Optional[List[float]] = None
) -> "cp.ndarray":
    """
    Create a composite image from a 3D stack where each slice is a channel.

    Args:
        stack: 3D CuPy array of shape (N, Y, X) where N is number of channel slices
        weights: List of weights for each slice. If None, equal weights are used.

    Returns:
        Composite 3D CuPy array of shape (1, Y, X)
    """
    # 🔄 MEMORY CONVERSION LOGGING: Log what we actually received
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"🔄 CREATE_COMPOSITE: Called with stack type: {type(stack)}, shape: {getattr(stack, 'shape', 'no shape')}")
    logger.info(f"🔄 CREATE_COMPOSITE: weights parameter - type: {type(weights)}, value: {weights}")

    # Validate input is 3D array
    if not hasattr(stack, 'shape') or len(stack.shape) != 3:
        raise TypeError(f"stack must be a 3D CuPy array, got shape: {getattr(stack, 'shape', 'no shape')}")

    n_slices, height, width = stack.shape

    # Default weights if none provided
    if weights is None:
        # Equal weights for all slices
        weights = [1.0 / n_slices] * n_slices
        logger.info(f"🔄 CREATE_COMPOSITE: Using default equal weights: {weights}")
    elif isinstance(weights, (list, tuple)):
        # Convert tuple to list if needed
        weights = list(weights)
        logger.info(f"🔄 CREATE_COMPOSITE: Using provided weights: {weights}")
        if len(weights) != n_slices:
            raise ValueError(f"Number of weights ({len(weights)}) must match number of slices ({n_slices})")
    else:
        # Log the problematic type and value for debugging
        logger.error(f"🔄 CREATE_COMPOSITE: Invalid weights type - expected list/tuple/None, got {type(weights)}: {weights}")
        raise TypeError(f"weights must be a list of values or None, got {type(weights)}: {weights}")

    # Normalize weights to sum to 1
    weight_sum = sum(weights)
    if weight_sum == 0:
        raise ValueError("Sum of weights cannot be zero")
    normalized_weights = [w / weight_sum for w in weights]

    # Convert weights to CuPy array for efficient computation
    # CRITICAL: Use float32 for weights to preserve fractional values, not stack.dtype
    weights_array = cp.array(normalized_weights, dtype=cp.float32)

    # Reshape weights for broadcasting: (N, 1, 1) to multiply with (N, Y, X)
    weights_array = weights_array.reshape(n_slices, 1, 1)

    # Create composite by weighted sum along the first axis
    # Convert stack to float32 for computation to avoid precision loss
    stack_float = stack.astype(cp.float32)
    weighted_stack = stack_float * weights_array
    composite_slice = cp.sum(weighted_stack, axis=0, keepdims=True)  # Keep as (1, Y, X)

    # Convert back to original dtype
    composite_slice = composite_slice.astype(stack.dtype)

    return composite_slice

@cupy_func
def apply_mask(image: "cp.ndarray", mask: "cp.ndarray") -> "cp.ndarray":
    """
    Apply a mask to a 3D image.

    This applies the mask to each Z-slice independently if mask is 2D,
    or applies the 3D mask directly if mask is 3D.

    Args:
        image: 3D CuPy array of shape (Z, Y, X)
        mask: 3D CuPy array of shape (Z, Y, X) or 2D CuPy array of shape (Y, X)

    Returns:
        Masked 3D CuPy array of shape (Z, Y, X)
    """
    _validate_3d_array(image)

    # Handle 2D mask (apply to each Z-slice)
    if isinstance(mask, cp.ndarray) and mask.ndim == 2:
        if mask.shape != image.shape[1:]:
            raise ValueError(
                f"2D mask shape {mask.shape} doesn't match image slice shape {image.shape[1:]}"
            )

        # Apply 2D mask to all Z-slices simultaneously using broadcasting - GPU PARALLELIZED
        # Broadcasting mask from (Y, X) to (1, Y, X) allows vectorized operation across all slices
        mask_3d = mask[None, :, :]  # Shape: (1, Y, X)
        result = image.astype(cp.float32) * mask_3d.astype(cp.float32)

        return result.astype(image.dtype)

    # Handle 3D mask
    if isinstance(mask, cp.ndarray) and mask.ndim == 3:
        if mask.shape != image.shape:
            raise ValueError(
                f"3D mask shape {mask.shape} doesn't match image shape {image.shape}"
            )

        # Apply 3D mask directly - MATCH NUMPY EXACTLY
        masked = image.astype(cp.float32) * mask.astype(cp.float32)
        return masked.astype(image.dtype)

    # If we get here, the mask is neither 2D nor 3D CuPy array
    raise TypeError(f"mask must be a 2D or 3D CuPy array, got {type(mask)}")

@cupy_func
def create_weight_mask(shape: Tuple[int, int], margin_ratio: float = 0.1) -> "cp.ndarray":
    """
    Create a weight mask for blending images.

    Args:
        shape: Shape of the mask (height, width)
        margin_ratio: Ratio of image size to use as margin

    Returns:
        2D CuPy weight mask of shape (Y, X)
    """
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise TypeError("shape must be a tuple of (height, width)")

    height, width = shape
    return create_linear_weight_mask(height, width, margin_ratio)

@cupy_func
def max_projection(stack: "cp.ndarray") -> "cp.ndarray":
    """
    Create a maximum intensity projection from a Z-stack.

    Args:
        stack: 3D CuPy array of shape (Z, Y, X)

    Returns:
        3D CuPy array of shape (1, Y, X)
    """
    _validate_3d_array(stack)

    # Create max projection
    projection_2d = cp.max(stack, axis=0)
    return projection_2d.reshape(1, projection_2d.shape[0], projection_2d.shape[1])

@cupy_func
def mean_projection(stack: "cp.ndarray") -> "cp.ndarray":
    """
    Create a mean intensity projection from a Z-stack.

    Args:
        stack: 3D CuPy array of shape (Z, Y, X)

    Returns:
        3D CuPy array of shape (1, Y, X)
    """
    _validate_3d_array(stack)

    # Create mean projection
    projection_2d = cp.mean(stack, axis=0).astype(stack.dtype)
    return projection_2d.reshape(1, projection_2d.shape[0], projection_2d.shape[1])


class CupySpatialBinStrategy(
    RegisteredProcessorMethodStrategy, metaclass=AutoRegisterMeta
):
    @abstractmethod
    def apply(self, array: "cp.ndarray", axis: tuple[int, ...]) -> "cp.ndarray":
        raise NotImplementedError


class CupyMeanSpatialBinStrategy(CupySpatialBinStrategy):
    method = "mean"

    def apply(self, array: "cp.ndarray", axis: tuple[int, ...]) -> "cp.ndarray":
        return cp.mean(array, axis=axis)


class CupySumSpatialBinStrategy(CupySpatialBinStrategy):
    method = "sum"

    def apply(self, array: "cp.ndarray", axis: tuple[int, ...]) -> "cp.ndarray":
        return cp.sum(array, axis=axis)


class CupyMaxSpatialBinStrategy(CupySpatialBinStrategy):
    method = "max"

    def apply(self, array: "cp.ndarray", axis: tuple[int, ...]) -> "cp.ndarray":
        return cp.max(array, axis=axis)


class CupyMinSpatialBinStrategy(CupySpatialBinStrategy):
    method = "min"

    def apply(self, array: "cp.ndarray", axis: tuple[int, ...]) -> "cp.ndarray":
        return cp.min(array, axis=axis)


class CupyStackProjectionStrategy(
    RegisteredProcessorMethodStrategy, metaclass=AutoRegisterMeta
):
    @abstractmethod
    def apply(self, stack: "cp.ndarray") -> "cp.ndarray":
        raise NotImplementedError


class CupyMaxStackProjectionStrategy(CupyStackProjectionStrategy):
    method = "max_projection"

    def apply(self, stack: "cp.ndarray") -> "cp.ndarray":
        return max_projection(stack)


class CupyMeanStackProjectionStrategy(CupyStackProjectionStrategy):
    method = "mean_projection"

    def apply(self, stack: "cp.ndarray") -> "cp.ndarray":
        return mean_projection(stack)


@cupy_func
def spatial_bin_2d(
    stack: "cp.ndarray",
    bin_size: int = 2,
    method: str = "mean"
) -> "cp.ndarray":
    """
    Apply 2D spatial binning to each slice in the stack - GPU accelerated.

    Reduces spatial resolution by combining neighboring pixels in 2D blocks.
    Each slice is processed independently using efficient GPU operations.

    Args:
        stack: 3D CuPy array of shape (Z, Y, X)
        bin_size: Size of the square binning kernel (e.g., 2 = 2x2 binning)
        method: Binning method - "mean", "sum", "max", or "min"

    Returns:
        Binned 3D CuPy array of shape (Z, Y//bin_size, X//bin_size)
    """
    _validate_3d_array(stack)

    if bin_size <= 0:
        raise ValueError("bin_size must be positive")

    z_slices, height, width = stack.shape

    # Calculate output dimensions
    new_height = height // bin_size
    new_width = width // bin_size

    if new_height == 0 or new_width == 0:
        raise ValueError(f"bin_size {bin_size} is too large for image dimensions {height}x{width}")

    # Crop to make dimensions divisible by bin_size
    crop_height = new_height * bin_size
    crop_width = new_width * bin_size
    cropped_stack = stack[:, :crop_height, :crop_width]

    # Reshape for binning: (Z, new_height, bin_size, new_width, bin_size)
    reshaped = cropped_stack.reshape(z_slices, new_height, bin_size, new_width, bin_size)

    result = CupySpatialBinStrategy.for_method(method).apply(reshaped, axis=(2, 4))

    return result.astype(stack.dtype)

@cupy_func
def spatial_bin_3d(
    stack: "cp.ndarray",
    bin_size: int = 2,
    method: str = "mean"
) -> "cp.ndarray":
    """
    Apply 3D spatial binning to the entire stack - GPU accelerated.

    Reduces spatial resolution by combining neighboring voxels in 3D blocks
    using efficient GPU operations.

    Args:
        stack: 3D CuPy array of shape (Z, Y, X)
        bin_size: Size of the cubic binning kernel (e.g., 2 = 2x2x2 binning)
        method: Binning method - "mean", "sum", "max", or "min"

    Returns:
        Binned 3D CuPy array of shape (Z//bin_size, Y//bin_size, X//bin_size)
    """
    _validate_3d_array(stack)

    if bin_size <= 0:
        raise ValueError("bin_size must be positive")

    depth, height, width = stack.shape

    # Calculate output dimensions
    new_depth = depth // bin_size
    new_height = height // bin_size
    new_width = width // bin_size

    if new_depth == 0 or new_height == 0 or new_width == 0:
        raise ValueError(f"bin_size {bin_size} is too large for stack dimensions {depth}x{height}x{width}")

    # Crop to make dimensions divisible by bin_size
    crop_depth = new_depth * bin_size
    crop_height = new_height * bin_size
    crop_width = new_width * bin_size
    cropped_stack = stack[:crop_depth, :crop_height, :crop_width]

    # Reshape for 3D binning: (new_depth, bin_size, new_height, bin_size, new_width, bin_size)
    reshaped = cropped_stack.reshape(new_depth, bin_size, new_height, bin_size, new_width, bin_size)

    result = CupySpatialBinStrategy.for_method(method).apply(reshaped, axis=(1, 3, 5))

    return result.astype(stack.dtype)

@cupy_func
def stack_equalize_histogram(
    stack: "cp.ndarray",
    bins: int = 65536,
    range_min: float = 0.0,
    range_max: float = 65535.0
) -> "cp.ndarray":
    """
    Apply histogram equalization to an entire stack.

    This ensures consistent contrast enhancement across all Z-slices by
    computing a global histogram across the entire stack.

    Args:
        stack: 3D CuPy array of shape (Z, Y, X)
        bins: Number of bins for histogram computation
        range_min: Minimum value for histogram range
        range_max: Maximum value for histogram range

    Returns:
        Equalized 3D CuPy array of shape (Z, Y, X)
    """
    _validate_3d_array(stack)

    # Remember input dtype to preserve it
    input_dtype = stack.dtype

    # MATCH NUMPY EXACTLY - Flatten the entire stack to compute the global histogram
    flat_stack = stack.flatten()

    # Calculate the histogram and cumulative distribution function (CDF) - MATCH NUMPY EXACTLY
    hist, bin_edges = cp.histogram(flat_stack, bins=bins, range=(range_min, range_max))
    cdf = hist.cumsum()

    # Normalize the CDF to the input dtype range
    # Avoid division by zero
    if cdf[-1] > 0:
        if cp.issubdtype(input_dtype, cp.integer):
            dtype_info = cp.iinfo(input_dtype)
            cdf = dtype_info.max * cdf / cdf[-1]
        else:
            # For float dtypes, normalize to [0, 1]
            cdf = cdf / cdf[-1]

    # Use linear interpolation to map input values to equalized values - MATCH NUMPY EXACTLY
    equalized_stack = cp.interp(stack.flatten(), bin_edges[:-1], cdf).reshape(stack.shape)

    # Convert back to input dtype
    if cp.issubdtype(input_dtype, cp.integer):
        dtype_info = cp.iinfo(input_dtype)
        return cp.clip(equalized_stack, dtype_info.min, dtype_info.max).astype(input_dtype)
    else:
        return equalized_stack.astype(input_dtype)

@cupy_func
def create_projection(
    stack: "cp.ndarray", method: str = "max_projection"
) -> "cp.ndarray":
    """
    Create a projection from a stack using the specified method.

    Args:
        stack: 3D CuPy array of shape (Z, Y, X)
        method: Projection method (max_projection, mean_projection)

    Returns:
        3D CuPy array of shape (1, Y, X)
    """
    _validate_3d_array(stack)

    return CupyStackProjectionStrategy.for_method(method).apply(stack)


@cupy_func
def crop(
    input_image: "cp.ndarray",
    start_x: int = 0,
    start_y: int = 0,
    start_z: int = 0,
    width: int = 1,
    height: int = 1,
    depth: int = 1
) -> "cp.ndarray":
    """
    Crop a given substack out of a given image stack.

    Equivalent to pyclesperanto.crop() but using CuPy operations.

    Parameters
    ----------
    input_image: cp.ndarray
        Input 3D image to process of shape (Z, Y, X)
    start_x: int (= 0)
        Starting index coordinate x
    start_y: int (= 0)
        Starting index coordinate y
    start_z: int (= 0)
        Starting index coordinate z
    width: int (= 1)
        Width size of the region to crop
    height: int (= 1)
        Height size of the region to crop
    depth: int (= 1)
        Depth size of the region to crop

    Returns
    -------
    cp.ndarray
        Cropped 3D array of shape (depth, height, width)
    """
    _validate_3d_array(input_image)

    # Validate crop parameters
    if width <= 0 or height <= 0 or depth <= 0:
        raise ValueError(f"Crop dimensions must be positive: width={width}, height={height}, depth={depth}")

    if start_x < 0 or start_y < 0 or start_z < 0:
        raise ValueError(f"Start coordinates must be non-negative: start_x={start_x}, start_y={start_y}, start_z={start_z}")

    # Get input dimensions
    input_depth, input_height, input_width = input_image.shape

    # Calculate end coordinates
    end_x = start_x + width
    end_y = start_y + height
    end_z = start_z + depth

    # Validate bounds
    if end_x > input_width or end_y > input_height or end_z > input_depth:
        raise ValueError(
            f"Crop region extends beyond image bounds. "
            f"Image shape: {input_image.shape}, "
            f"Crop region: ({start_z}:{end_z}, {start_y}:{end_y}, {start_x}:{end_x})"
        )

    # Perform the crop using CuPy slicing
    cropped = input_image[start_z:end_z, start_y:end_y, start_x:end_x]

    return cropped

def _create_disk_cupy(radius: int) -> "cp.ndarray":
    """Create a disk structuring element using CuPy - MATCH NUMPY EXACTLY"""
    y, x = cp.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x*x + y*y <= radius*radius
    return mask.astype(cp.uint8)

def _resize_cupy_better_match(image: "cp.ndarray", output_shape: tuple, anti_aliasing: bool = True, preserve_range: bool = True) -> "cp.ndarray":
    """
    Resize image using CuPy to better match scikit-image's transform.resize behavior.

    Key differences from original:
    1. Better anti-aliasing sigma calculation
    2. Proper preserve_range handling without rescaling
    3. More accurate zoom parameters
    """
    from cupyx.scipy import ndimage as cupy_ndimage

    # Calculate zoom factors
    zoom_factors = [output_shape[i] / image.shape[i] for i in range(len(output_shape))]

    # Convert to float32 for processing
    image_float = image.astype(cp.float32)

    # Apply anti-aliasing if downsampling
    if anti_aliasing and any(z < 1.0 for z in zoom_factors):
        # Use scikit-image's sigma calculation: sigma = (1/zoom - 1) / 2
        # But ensure minimum sigma and handle edge cases
        sigma = []
        for z in zoom_factors:
            if z < 1.0:
                s = (1.0/z - 1.0) / 2.0
                sigma.append(max(s, 0.5))  # Minimum sigma for stability
            else:
                sigma.append(0.0)

        # Apply Gaussian smoothing before downsampling
        if any(s > 0 for s in sigma):
            image_float = cupy_ndimage.gaussian_filter(image_float, sigma)

    # Perform zoom with bilinear interpolation (order=1)
    resized = cupy_ndimage.zoom(image_float, zoom_factors, order=1)

    # Handle preserve_range properly - don't rescale, just maintain dtype range
    if preserve_range:
        # Clip to valid range for the original dtype
        if image.dtype == cp.uint16:
            resized = cp.clip(resized, 0, 65535)
        elif image.dtype == cp.uint8:
            resized = cp.clip(resized, 0, 255)
        # For other dtypes, keep as-is

    return resized

@cupy_func
def tophat(
    image: "cp.ndarray",
    selem_radius: int = 50,
    downsample_factor: int = 4,
    downsample_anti_aliasing: bool = True,
    upsample_anti_aliasing: bool = False
) -> "cp.ndarray":
    """
    Apply white top-hat filter to a 3D image for background removal.

    This applies the filter to each Z-slice independently using slice-by-slice
    processing for algorithmic reasons (complex multi-step processing with
    slice-specific intermediate results). Each slice operation is GPU parallelized.

    Args:
        image: 3D CuPy array of shape (Z, Y, X)
        selem_radius: Radius of the structuring element disk
        downsample_factor: Factor by which to downsample the image for processing
        downsample_anti_aliasing: Whether to use anti-aliasing when downsampling
        upsample_anti_aliasing: Whether to use anti-aliasing when upsampling

    Returns:
        Filtered 3D CuPy array of shape (Z, Y, X)
    """
    _validate_3d_array(image)

    # Process each Z-slice independently - IMPROVED MATCH TO NUMPY
    # NOTE: For loop is ALGORITHMICALLY NECESSARY here due to complex multi-step
    # processing (downsample, morphology, upsample, subtract) with slice-specific
    # intermediate results. Vectorization would require significant algorithm restructuring.
    result = cp.zeros_like(image)

    for z in range(image.shape[0]):
        # Store original data type - MATCH NUMPY EXACTLY
        input_dtype = image[z].dtype

        # 1) Downsample - IMPROVED MATCH TO NUMPY
        target_shape = (image[z].shape[0]//downsample_factor, image[z].shape[1]//downsample_factor)
        image_small = _resize_cupy_better_match(
            image[z],
            target_shape,
            anti_aliasing=downsample_anti_aliasing,
            preserve_range=True
        )

        # 2) Build structuring element for the smaller image - MATCH NUMPY EXACTLY
        selem_small = _create_disk_cupy(selem_radius // downsample_factor)

        # 3) White top-hat on the smaller image - MATCH NUMPY EXACTLY
        tophat_small = ndimage.white_tophat(image_small, structure=selem_small)

        # 4) Upscale background to original size - IMPROVED MATCH TO NUMPY
        background_small = image_small - tophat_small
        background_large = _resize_cupy_better_match(
            background_small,
            image[z].shape,
            anti_aliasing=upsample_anti_aliasing,
            preserve_range=True
        )

        # 5) Subtract background and clip negative values - MATCH NUMPY EXACTLY
        slice_result = cp.maximum(image[z] - background_large, 0)

        # 6) Convert back to original data type - MATCH NUMPY EXACTLY
        result[z] = slice_result.astype(input_dtype)

    return result

# Lazy initialization of ElementwiseKernel to avoid import errors
_sobel_2d_parallel = None

def _get_sobel_2d_kernel():
    """Get or create the Sobel 2D parallel kernel."""
    global _sobel_2d_parallel
    if _sobel_2d_parallel is None:
        if cp is None:
            raise ImportError("CuPy is required for GPU Sobel operations")

        _sobel_2d_parallel = cp.ElementwiseKernel(
            'raw T input, int32 Y, int32 X, int32 mode, T cval',
            'T output',
            '''
            // Calculate which slice, row, col this thread handles
            int z = i / (Y * X);
            int y = (i % (Y * X)) / X;
            int x = i % X;

            // Calculate base index for this slice
            int slice_base = z * Y * X;

            // Helper function to get pixel with configurable boundary handling
            auto get_pixel = [&](int py, int px) -> T {
                if (mode == 0) {  // constant
                    if (py < 0 || py >= Y || px < 0 || px >= X) return cval;
                } else if (mode == 1) {  // reflect
                    py = py < 0 ? -py : (py >= Y ? 2*Y - py - 2 : py);
                    px = px < 0 ? -px : (px >= X ? 2*X - px - 2 : px);
                } else if (mode == 2) {  // nearest
                    py = py < 0 ? 0 : (py >= Y ? Y-1 : py);
                    px = px < 0 ? 0 : (px >= X ? X-1 : px);
                } else if (mode == 3) {  // wrap
                    py = py < 0 ? py + Y : (py >= Y ? py - Y : py);
                    px = px < 0 ? px + X : (px >= X ? px - X : px);
                }
                return input[slice_base + py * X + px];
            };

            // Sobel X kernel: [[-1,0,1],[-2,0,2],[-1,0,1]] (within slice only)
            T gx = -get_pixel(y-1, x-1) + get_pixel(y-1, x+1) +
                   -2*get_pixel(y, x-1) + 2*get_pixel(y, x+1) +
                   -get_pixel(y+1, x-1) + get_pixel(y+1, x+1);

            // Sobel Y kernel: [[-1,-2,-1],[0,0,0],[1,2,1]] (within slice only)
            T gy = -get_pixel(y-1, x-1) - 2*get_pixel(y-1, x) - get_pixel(y-1, x+1) +
                    get_pixel(y+1, x-1) + 2*get_pixel(y+1, x) + get_pixel(y+1, x+1);

            // Calculate magnitude
            output = sqrt(gx*gx + gy*gy);
            ''',
            'sobel_2d_parallel'
        )
    return _sobel_2d_parallel

@cupy_func
def sobel_2d_vectorized(image: "cp.ndarray", mode: str = "reflect", cval: float = 0.0) -> "cp.ndarray":
    """
    Apply 2D Sobel edge detection to all slices simultaneously - TRUE GPU PARALLELIZED.

    Each slice is treated as an independent 2D grayscale image. All pixels across
    all slices are processed in parallel on the GPU with slice independence guaranteed.
    Uses ElementwiseKernel for maximum performance and true parallelization.

    Args:
        image: 3D CuPy array of shape (Z, Y, X)
        mode: Boundary handling mode ('constant', 'reflect', 'nearest', 'wrap')
        cval: Constant value for 'constant' mode

    Returns:
        Edge magnitude as 3D CuPy array of shape (Z, Y, X)
    """
    _validate_3d_array(image)

    # Map mode string to integer
    mode_map = {'constant': 0, 'reflect': 1, 'nearest': 2, 'wrap': 3}
    if mode not in mode_map:
        raise ValueError(f"Unknown mode: {mode}. Valid modes: {list(mode_map.keys())}")

    mode_int = mode_map[mode]
    Z, Y, X = image.shape
    input_float = image.astype(cp.float32)
    output = cp.zeros_like(input_float)

    # Launch parallel kernel - each thread processes one pixel
    sobel_kernel = _get_sobel_2d_kernel()
    sobel_kernel(input_float, Y, X, mode_int, cp.float32(cval), output)

    return output.astype(image.dtype)

@cupy_func
def sobel_3d_voxel(image: "cp.ndarray", mode: str = "reflect", cval: float = 0.0) -> "cp.ndarray":
    """
    Apply true 3D voxel Sobel edge detection including Z-axis gradients - GPU PARALLELIZED.

    This computes gradients in all three spatial dimensions (X, Y, Z) for true
    volumetric edge detection. Useful for 3D structure analysis where Z-dimension
    has spatial meaning (not just independent slices).

    Args:
        image: 3D CuPy array of shape (Z, Y, X)
        mode: Boundary handling mode ('constant', 'reflect', 'nearest', 'wrap', 'mirror')
        cval: Constant value for 'constant' mode

    Returns:
        3D edge magnitude as 3D CuPy array of shape (Z, Y, X)
    """
    _validate_3d_array(image)

    # Convert to float32 for processing
    image_float = image.astype(cp.float32)

    # Apply Sobel filters in all three directions - GPU PARALLELIZED
    sobel_x = ndimage.sobel(image_float, axis=2, mode=mode, cval=cval)  # X-direction gradients
    sobel_y = ndimage.sobel(image_float, axis=1, mode=mode, cval=cval)  # Y-direction gradients
    sobel_z = ndimage.sobel(image_float, axis=0, mode=mode, cval=cval)  # Z-direction gradients

    # Calculate 3D magnitude - GPU PARALLELIZED
    magnitude = cp.sqrt(sobel_x**2 + sobel_y**2 + sobel_z**2)

    return magnitude.astype(image.dtype)

@cupy_func
def sobel_components(image: "cp.ndarray", include_z: bool = False, mode: str = "reflect", cval: float = 0.0) -> tuple:
    """
    Return individual Sobel gradient components - GPU PARALLELIZED.

    This provides access to directional gradients for advanced analysis.
    Useful when you need to analyze edge orientation or directional information.

    Args:
        image: 3D CuPy array of shape (Z, Y, X)
        include_z: Whether to include Z-direction gradients (3D analysis)
        mode: Boundary handling mode ('constant', 'reflect', 'nearest', 'wrap', 'mirror')
        cval: Constant value for 'constant' mode

    Returns:
        Tuple of gradient components:
        - If include_z=False: (sobel_x, sobel_y) - 2D gradients per slice
        - If include_z=True: (sobel_x, sobel_y, sobel_z) - 3D gradients
    """
    _validate_3d_array(image)

    # Convert to float32 for processing
    image_float = image.astype(cp.float32)

    # Apply Sobel filters - GPU PARALLELIZED
    sobel_x = ndimage.sobel(image_float, axis=2, mode=mode, cval=cval).astype(image.dtype)  # X-direction
    sobel_y = ndimage.sobel(image_float, axis=1, mode=mode, cval=cval).astype(image.dtype)  # Y-direction

    if include_z:
        sobel_z = ndimage.sobel(image_float, axis=0, mode=mode, cval=cval).astype(image.dtype)  # Z-direction
        return sobel_x, sobel_y, sobel_z
    else:
        return sobel_x, sobel_y


class CupyEdgeMagnitudeStrategy(
    RegisteredProcessorMethodStrategy, metaclass=AutoRegisterMeta
):
    @abstractmethod
    def apply(
        self, image: "cp.ndarray", *, mode: str, cval: float
    ) -> "cp.ndarray":
        raise NotImplementedError


class CupySlice2DEdgeMagnitudeStrategy(CupyEdgeMagnitudeStrategy):
    method = "2d"

    def apply(
        self, image: "cp.ndarray", *, mode: str, cval: float
    ) -> "cp.ndarray":
        return sobel_2d_vectorized(image, mode=mode, cval=cval)


class CupyVolume3DEdgeMagnitudeStrategy(CupyEdgeMagnitudeStrategy):
    method = "3d"

    def apply(
        self, image: "cp.ndarray", *, mode: str, cval: float
    ) -> "cp.ndarray":
        return sobel_3d_voxel(image, mode=mode, cval=cval)


@cupy_func
def edge_magnitude(image: "cp.ndarray", method: str = "2d", mode: str = "reflect", cval: float = 0.0) -> "cp.ndarray":
    """
    Compute edge magnitude using specified method - GPU PARALLELIZED.

    This dispatcher function provides a unified interface for different Sobel
    approaches, following the pattern of create_projection.

    Args:
        image: 3D CuPy array of shape (Z, Y, X)
        method: Edge detection method
            - "2d": 2D Sobel applied to each slice independently (slice-wise)
            - "3d": True 3D voxel Sobel including Z-axis gradients (volumetric)
        mode: Boundary handling mode ('constant', 'reflect', 'nearest', 'wrap', 'mirror')
        cval: Constant value for 'constant' mode

    Returns:
        Edge magnitude as 3D CuPy array of shape (Z, Y, X)
    """
    _validate_3d_array(image)

    return CupyEdgeMagnitudeStrategy.for_method(method).apply(
        image, mode=mode, cval=cval
    )


@cupy_func
def sobel(image: "cp.ndarray", mask: Optional["cp.ndarray"] = None, *,
          axis: Optional[int] = None, mode: str = "reflect", cval: float = 0.0) -> "cp.ndarray":
    """
    Find edges in an image using the Sobel filter (CuCIM backend).

    This function wraps CuCIM's sobel filter to provide a manual, pickleable
    sobel function with full OpenHCS features (slice_by_slice, dtype_conversion, etc.).

    The @cupy_func decorator automatically provides slice_by_slice processing,
    so this function can handle both 2D and 3D inputs depending on the setting.

    Args:
        image: CuPy array (2D when slice_by_slice=True, 3D when slice_by_slice=False)
        mask: Optional mask array to clip the output (values where mask=0 will be set to 0)
        axis: Compute the edge filter along this axis. If not provided, edge magnitude is computed
        mode: Boundary handling mode ('reflect', 'constant', 'nearest', 'wrap')
        cval: Constant value for 'constant' mode

    Returns:
        Edge-filtered CuPy array (same shape and dtype as input)

    Note:
        This is a manual wrapper around CuCIM's sobel function that provides
        the same functionality as auto-discovered functions but is pickleable
        for subprocess execution.

        CuCIM's sobel normalizes integer inputs to [0, 1] range and returns float64.
        This wrapper always preserves the input dtype by forcing dtype conversion.
    """
    if cucim_filters is None:
        raise ImportError("CuCIM is required for sobel edge detection but is not available")

    # Import here to avoid circular dependency
    from openhcs.core.memory import DtypeConversion
    from openhcs.core.memory import SCALING_FUNCTIONS
    from openhcs.constants.constants import MemoryType

    # Store original dtype
    original_dtype = image.dtype

    # Call CuCIM sobel
    result = cucim_filters.sobel(
        image,
        mask=mask,
        axis=axis,
        mode=mode,
        cval=cval
    )

    # Always preserve input dtype (CuCIM normalizes integer inputs to [0, 1])
    if result.dtype != original_dtype:
        scale_func = SCALING_FUNCTIONS[MemoryType.CUPY.value]
        result = scale_func(result, original_dtype)

    return result
