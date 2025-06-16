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
from typing import Any, List, Optional, Tuple

from openhcs.core.memory.decorators import cupy as cupy_func
from openhcs.core.utils import optional_import

# Import CuPy as an optional dependency
cp = optional_import("cupy")
ndimage = None
if cp is not None:
    cupyx_scipy = optional_import("cupyx.scipy")
    if cupyx_scipy is not None:
        ndimage = cupyx_scipy.ndimage

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
    Sharpen a 3D image using unsharp masking.

    This applies sharpening to each Z-slice independently.

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

    # Process each Z-slice independently
    result = cp.zeros_like(image, dtype=cp.float32)

    for z in range(image.shape[0]):
        # Convert to float for processing
        slice_float = image[z].astype(cp.float32) / cp.max(image[z])

        # Create blurred version for unsharp mask
        # Use CuPy's ndimage.gaussian_filter instead of scikit-image's filters.gaussian
        blurred = ndimage.gaussian_filter(slice_float, sigma=radius)

        # Apply unsharp mask: original + amount * (original - blurred)
        sharpened = slice_float + amount * (slice_float - blurred)

        # Clip to valid range
        sharpened = cp.clip(sharpened, 0, 1.0)

        # Scale back to original range
        # CuPy doesn't have exposure.rescale_intensity, so implement manually
        min_val = cp.min(sharpened)
        max_val = cp.max(sharpened)
        if max_val > min_val:
            sharpened = (sharpened - min_val) * 65535 / (max_val - min_val)

        result[z] = sharpened

    # Convert back to original dtype
    return result.astype(dtype)

@cupy_func
def percentile_normalize(
    image: "cp.ndarray",
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    target_min: float = 0.0,
    target_max: float = 65535.0
) -> "cp.ndarray":
    """
    Normalize a 3D image using percentile-based contrast stretching.

    This applies normalization to each Z-slice independently.

    Args:
        image: 3D CuPy array of shape (Z, Y, X)
        low_percentile: Lower percentile (0-100)
        high_percentile: Upper percentile (0-100)
        target_min: Target minimum value
        target_max: Target maximum value

    Returns:
        Normalized 3D CuPy array of shape (Z, Y, X)
    """
    _validate_3d_array(image)

    # Process each Z-slice independently (memory-efficient)
    result = cp.zeros_like(image, dtype=cp.uint16)  # Use final dtype directly

    for z in range(image.shape[0]):
        # Get percentile values for this slice
        p_low = cp.percentile(image[z], low_percentile)
        p_high = cp.percentile(image[z], high_percentile)

        # Avoid division by zero
        if p_high == p_low:
            result[z] = target_min  # Direct assignment, no temporary arrays
            continue

        # Clip and normalize to target range (in-place operations)
        # Use the original image slice directly to avoid copies
        slice_data = image[z].astype(cp.float32)  # Only convert one slice at a time
        cp.clip(slice_data, p_low, p_high, out=slice_data)  # In-place clipping
        slice_data -= p_low  # In-place subtraction
        slice_data *= (target_max - target_min) / (p_high - p_low)  # In-place scaling
        slice_data += target_min  # In-place addition

        # Convert and store result
        result[z] = slice_data.astype(cp.uint16)

    return result

@cupy_func
def stack_percentile_normalize(
    stack: "cp.ndarray",
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    target_min: float = 0.0,
    target_max: float = 65535.0
) -> "cp.ndarray":
    """
    Normalize a stack using global percentile-based contrast stretching.

    This ensures consistent normalization across all Z-slices by computing
    global percentiles across the entire stack.

    Args:
        stack: 3D CuPy array of shape (Z, Y, X)
        low_percentile: Lower percentile (0-100)
        high_percentile: Upper percentile (0-100)
        target_min: Target minimum value
        target_max: Target maximum value

    Returns:
        Normalized 3D CuPy array of shape (Z, Y, X)
    """
    _validate_3d_array(stack)

    # Calculate global percentiles across the entire stack
    p_low = cp.percentile(stack, low_percentile)
    p_high = cp.percentile(stack, high_percentile)

    # Avoid division by zero
    if p_high == p_low:
        return cp.ones_like(stack) * target_min

    # Clip and normalize to target range (match NumPy implementation exactly)
    clipped = cp.clip(stack, p_low, p_high)
    normalized = (clipped - p_low) * (target_max - target_min) / (p_high - p_low) + target_min
    normalized = normalized.astype(cp.uint16)

    return normalized

@cupy_func
def create_composite(
    images: List["cp.ndarray"], weights: Optional[List[float]] = None
) -> "cp.ndarray":
    """
    Create a composite image from multiple 3D arrays.

    Args:
        images: List of 3D CuPy arrays, each of shape (Z, Y, X)
        weights: List of weights for each image. If None, equal weights are used.

    Returns:
        Composite 3D CuPy array of shape (Z, Y, X)
    """
    # Ensure images is a list
    if not isinstance(images, list):
        raise TypeError("images must be a list of CuPy arrays")

    # Check for empty list early
    if not images:
        raise ValueError("images list cannot be empty")

    # Validate all images are 3D CuPy arrays with the same shape
    for i, img in enumerate(images):
        _validate_3d_array(img, f"images[{i}]")
        if img.shape != images[0].shape:
            raise ValueError(
                f"All images must have the same shape. "
                f"images[0] has shape {images[0].shape}, "
                f"images[{i}] has shape {img.shape}"
            )

    # Default weights if none provided
    if weights is None:
        # Equal weights for all images
        weights = [1.0 / len(images)] * len(images)
    elif not isinstance(weights, list):
        raise TypeError("weights must be a list of values")

    # Make sure weights list is at least as long as images list
    if len(weights) < len(images):
        weights = weights + [0.0] * (len(images) - len(weights))
    # Truncate weights if longer than images
    weights = weights[:len(images)]

    first_image = images[0]
    shape = first_image.shape
    dtype = first_image.dtype

    # Create empty composite
    composite = cp.zeros(shape, dtype=cp.float32)
    total_weight = 0.0

    # Add each image with its weight (memory-efficient)
    for i, image in enumerate(images):
        weight = weights[i]
        if weight <= 0.0:
            continue

        # Add to composite without creating large temporary arrays
        # Process in-place to reduce memory usage
        composite += image * weight  # CuPy handles mixed-type operations efficiently
        total_weight += weight

    # Normalize by total weight
    if total_weight > 0:
        composite /= total_weight

    # Convert back to original dtype (usually uint16)
    if cp.issubdtype(dtype, cp.integer):
        max_val = cp.iinfo(dtype).max
        composite = cp.clip(composite, 0, max_val).astype(dtype)
    else:
        composite = composite.astype(dtype)

    return composite

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

        # Apply 2D mask to each Z-slice (memory-efficient)
        result = cp.zeros_like(image)
        for z in range(image.shape[0]):
            # Use in-place operations to reduce memory usage
            result[z] = image[z] * mask  # CuPy handles mixed-type operations

        return result

    # Handle 3D mask
    if isinstance(mask, cp.ndarray) and mask.ndim == 3:
        if mask.shape != image.shape:
            raise ValueError(
                f"3D mask shape {mask.shape} doesn't match image shape {image.shape}"
            )

        # Apply 3D mask directly (memory-efficient)
        return image * mask  # CuPy handles mixed-type operations efficiently

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

    # Memory-efficient histogram equalization without creating large temporary arrays
    # Calculate histogram directly from the stack without flattening
    hist = cp.histogram(stack, bins=bins, range=(range_min, range_max))[0]
    cdf = hist.cumsum()

    # Normalize the CDF to the range [0, 65535]
    # Avoid division by zero
    if cdf[-1] > 0:
        cdf = 65535 * cdf / cdf[-1]

    # Process the stack in-place to avoid creating large temporary arrays
    bin_width = (range_max - range_min) / bins

    # Create output array
    equalized_stack = cp.zeros_like(stack, dtype=cp.uint16)

    # Process each Z-slice to reduce memory usage
    for z in range(stack.shape[0]):
        # Calculate bin indices for this slice only
        slice_data = stack[z]
        indices = cp.clip(
            cp.floor((slice_data - range_min) / bin_width).astype(cp.int32),
            0, bins - 1
        )

        # Look up CDF values and store directly in output
        equalized_stack[z] = cdf[indices].astype(cp.uint16)

    return equalized_stack

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

    if method == "max_projection":
        return max_projection(stack)

    if method == "mean_projection":
        return mean_projection(stack)

    # FAIL FAST: No fallback projection methods
    raise ValueError(f"Unknown projection method: {method}. Valid methods: max_projection, mean_projection")

@cupy_func
def tophat(
    image: "cp.ndarray",
    selem_radius: int = 50,
    downsample_factor: int = 4
) -> "cp.ndarray":
    """
    Apply white top-hat filter to a 3D image for background removal.

    This applies the filter to each Z-slice independently using CuPy's
    morphological operations.

    Args:
        image: 3D CuPy array of shape (Z, Y, X)
        selem_radius: Radius of the structuring element disk
        downsample_factor: Factor by which to downsample the image for processing

    Returns:
        Filtered 3D CuPy array of shape (Z, Y, X)
    """
    _validate_3d_array(image)

    # Process each Z-slice independently
    result = cp.zeros_like(image)

    # Create a circular structuring element
    # CuPy doesn't have a direct disk function, so we'll create one manually
    # (This is used as a reference for the per-slice structuring elements)

    for z in range(image.shape[0]):
        # Store original data type
        input_dtype = image[z].dtype

        # 1) Downsample using CuPy's resize function
        # Calculate new dimensions
        new_h = image[z].shape[0] // downsample_factor
        new_w = image[z].shape[1] // downsample_factor

        # Resize using CuPy's resize function
        # Note: CuPy's resize is different from scikit-image's, but works for our purpose
        image_small = cp.resize(image[z], (new_h, new_w))

        # 2) Resize the structuring element to match the downsampled image
        small_selem_radius = max(1, selem_radius // downsample_factor)

        # Create grid for structuring element
        y_range = cp.arange(-small_selem_radius, small_selem_radius+1)
        x_range = cp.arange(-small_selem_radius, small_selem_radius+1)
        grid_y, grid_x = cp.meshgrid(y_range, x_range)

        # Create circular mask
        small_mask = grid_x**2 + grid_y**2 <= small_selem_radius**2
        small_selem = cp.asarray(small_mask, dtype=cp.uint8)

        # 3) Apply white top-hat using CuPy's morphology functions
        # White top-hat is opening subtracted from the original image
        # Opening is erosion followed by dilation

        # Perform opening (erosion followed by dilation)
        eroded = ndimage.binary_erosion(image_small, structure=small_selem)
        opened = ndimage.binary_dilation(eroded, structure=small_selem)

        # White top-hat is original minus opening
        tophat_small = image_small - opened

        # 4) Calculate background
        background_small = image_small - tophat_small

        # 5) Upscale background to original size
        background_large = cp.resize(background_small, image[z].shape)

        # 6) Subtract background and clip negative values
        slice_result = cp.maximum(image[z] - background_large, 0)

        # 7) Convert back to original data type
        result[z] = slice_result.astype(input_dtype)

    return result
