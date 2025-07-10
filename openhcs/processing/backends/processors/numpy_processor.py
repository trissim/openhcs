"""
NumPy Image Processor Implementation

This module implements the ImageProcessorInterface using NumPy as the backend.
It serves as the reference implementation and works on CPU.

Doctrinal Clauses:
- Clause 3 — Declarative Primacy: All functions are pure and stateless
- Clause 88 — No Inferred Capabilities: Explicit NumPy dependency
- Clause 106-A — Declared Memory Types: All methods specify NumPy arrays
"""
from __future__ import annotations 

import logging
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np
from skimage import exposure, filters
from skimage import morphology as morph
from skimage import transform as trans

# Use direct import from core memory decorators to avoid circular imports
from openhcs.core.memory.decorators import numpy as numpy_func

logger = logging.getLogger(__name__)


@numpy_func
def create_linear_weight_mask(height: int, width: int, margin_ratio: float = 0.1) -> np.ndarray:
    """
    Create a 2D weight mask that linearly ramps from 0 at the edges to 1 in the center.

    Args:
        height: Height of the mask
        width: Width of the mask
        margin_ratio: Ratio of the margin to the image size

    Returns:
        2D weight mask of shape (height, width)
    """
    margin_y = int(np.floor(height * margin_ratio))
    margin_x = int(np.floor(width * margin_ratio))

    weight_y = np.ones(height, dtype=np.float32)
    if margin_y > 0:
        ramp_top = np.linspace(0, 1, margin_y, endpoint=False)
        ramp_bottom = np.linspace(1, 0, margin_y, endpoint=False)
        weight_y[:margin_y] = ramp_top
        weight_y[-margin_y:] = ramp_bottom

    weight_x = np.ones(width, dtype=np.float32)
    if margin_x > 0:
        ramp_left = np.linspace(0, 1, margin_x, endpoint=False)
        ramp_right = np.linspace(1, 0, margin_x, endpoint=False)
        weight_x[:margin_x] = ramp_left
        weight_x[-margin_x:] = ramp_right

    # Create 2D weight mask
    weight_mask = np.outer(weight_y, weight_x)

    return weight_mask


def _validate_3d_array(array: Any, name: str = "input") -> None:
    """
    Validate that the input is a 3D NumPy array.

    Args:
        array: Array to validate
        name: Name of the array for error messages

    Raises:
        TypeError: If the array is not a NumPy array
        ValueError: If the array is not 3D
    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array, got {type(array)}")

    if array.ndim != 3:
        raise ValueError(f"{name} must be a 3D array, got {array.ndim}D")

@numpy_func
def sharpen(image: np.ndarray, radius: float = 1.0, amount: float = 1.0) -> np.ndarray:
    """
    Sharpen a 3D image using unsharp masking.

    This applies sharpening to each Z-slice independently.

    Args:
        image: 3D NumPy array of shape (Z, Y, X)
        radius: Radius of Gaussian blur
        amount: Sharpening strength

    Returns:
        Sharpened 3D NumPy array of shape (Z, Y, X)
    """
    _validate_3d_array(image)

    # Store original dtype
    dtype = image.dtype

    # Process each Z-slice independently
    result = np.zeros_like(image, dtype=np.float32)

    for z in range(image.shape[0]):
        # Convert to float for processing
        slice_float = image[z].astype(np.float32) / np.max(image[z])

        # Create blurred version for unsharp mask
        blurred = filters.gaussian(slice_float, sigma=radius)

        # Apply unsharp mask: original + amount * (original - blurred)
        sharpened = slice_float + amount * (slice_float - blurred)

        # Clip to valid range
        sharpened = np.clip(sharpened, 0, 1.0)

        # Scale back to original range
        sharpened = exposure.rescale_intensity(sharpened, in_range='image', out_range=(0, 65535))
        result[z] = sharpened

    # Convert back to original dtype
    return result.astype(dtype)

@numpy_func
def percentile_normalize(image: np.ndarray,
                        low_percentile: float = 1.0,
                        high_percentile: float = 99.0,
                        target_min: float = 0.0,
                        target_max: float = 65535.0) -> np.ndarray:
    """
    Normalize a 3D image using percentile-based contrast stretching.

    This applies normalization to each Z-slice independently.

    Args:
        image: 3D NumPy array of shape (Z, Y, X)
        low_percentile: Lower percentile (0-100)
        high_percentile: Upper percentile (0-100)
        target_min: Target minimum value
        target_max: Target maximum value

    Returns:
        Normalized 3D NumPy array of shape (Z, Y, X)
    """
    _validate_3d_array(image)

    # Process each Z-slice independently
    result = np.zeros_like(image, dtype=np.float32)

    for z in range(image.shape[0]):
        # Get percentile values for this slice
        p_low, p_high = np.percentile(image[z], (low_percentile, high_percentile))

        # Avoid division by zero
        if p_high == p_low:
            result[z] = np.ones_like(image[z]) * target_min
            continue

        # Clip and normalize to target range
        clipped = np.clip(image[z], p_low, p_high)
        normalized = (clipped - p_low) * (target_max - target_min) / (p_high - p_low) + target_min
        result[z] = normalized

    # Convert to uint16
    return result.astype(np.uint16)

@numpy_func
def stack_percentile_normalize(stack: np.ndarray,
                              low_percentile: float = 1.0,
                              high_percentile: float = 99.0,
                              target_min: float = 0.0,
                              target_max: float = 65535.0) -> np.ndarray:
    """
    Normalize a stack using global percentile-based contrast stretching.

    This ensures consistent normalization across all Z-slices by computing
    global percentiles across the entire stack.

    Args:
        stack: 3D NumPy array of shape (Z, Y, X)
        low_percentile: Lower percentile (0-100)
        high_percentile: Upper percentile (0-100)
        target_min: Target minimum value
        target_max: Target maximum value

    Returns:
        Normalized 3D NumPy array of shape (Z, Y, X)
    """
    _validate_3d_array(stack)

    # Calculate global percentiles across the entire stack
    p_low = np.percentile(stack, low_percentile, axis=None)
    p_high = np.percentile(stack, high_percentile, axis=None)

    # Avoid division by zero
    if p_high == p_low:
        return np.ones_like(stack) * target_min

    # Clip and normalize to target range
    clipped = np.clip(stack, p_low, p_high)
    normalized = (clipped - p_low) * (target_max - target_min) / (p_high - p_low) + target_min
    normalized = normalized.astype(np.uint16)

    return normalized

@numpy_func
def create_composite(
    stack: np.ndarray, weights: Optional[List[float]] = None
) -> np.ndarray:
    """
    Create a composite image from a 3D stack where each slice is a channel.

    Args:
        stack: 3D NumPy array of shape (N, Y, X) where N is number of channel slices
        weights: List of weights for each slice. If None, equal weights are used.

    Returns:
        Composite 3D NumPy array of shape (1, Y, X)
    """
    # Validate input is 3D array
    _validate_3d_array(stack)

    n_slices, height, width = stack.shape

    # Default weights if none provided
    if weights is None:
        # Equal weights for all slices
        weights = [1.0 / n_slices] * n_slices
    elif isinstance(weights, (list, tuple)):
        # Convert tuple to list if needed
        weights = list(weights)
        if len(weights) != n_slices:
            raise ValueError(f"Number of weights ({len(weights)}) must match number of slices ({n_slices})")
    else:
        raise TypeError(f"weights must be a list of values or None, got {type(weights)}: {weights}")

    # Normalize weights to sum to 1
    weight_sum = sum(weights)
    if weight_sum == 0:
        raise ValueError("Sum of weights cannot be zero")
    normalized_weights = [w / weight_sum for w in weights]

    # Convert weights to NumPy array for efficient computation
    # CRITICAL: Use float32 for weights to preserve fractional values, not stack.dtype
    weights_array = np.array(normalized_weights, dtype=np.float32)

    # Reshape weights for broadcasting: (N, 1, 1) to multiply with (N, Y, X)
    weights_array = weights_array.reshape(n_slices, 1, 1)

    # Create composite by weighted sum along the first axis
    # Convert stack to float32 for computation to avoid precision loss
    stack_float = stack.astype(np.float32)
    weighted_stack = stack_float * weights_array
    composite_slice = np.sum(weighted_stack, axis=0, keepdims=True)  # Keep as (1, Y, X)

    # Convert back to original dtype
    composite_slice = composite_slice.astype(stack.dtype)

    return composite_slice

@numpy_func
def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a mask to a 3D image.

    This applies the mask to each Z-slice independently if mask is 2D,
    or applies the 3D mask directly if mask is 3D.

    Args:
        image: 3D NumPy array of shape (Z, Y, X)
        mask: 3D NumPy array of shape (Z, Y, X) or 2D NumPy array of shape (Y, X)

    Returns:
        Masked 3D NumPy array of shape (Z, Y, X)
    """
    _validate_3d_array(image)

    # Handle 2D mask (apply to each Z-slice)
    if isinstance(mask, np.ndarray) and mask.ndim == 2:
        if mask.shape != image.shape[1:]:
            raise ValueError(
                f"2D mask shape {mask.shape} doesn't match image slice shape {image.shape[1:]}"
            )

        # Apply 2D mask to each Z-slice
        result = np.zeros_like(image)
        for z in range(image.shape[0]):
            result[z] = image[z].astype(np.float32) * mask.astype(np.float32)

        return result.astype(image.dtype)

    # Handle 3D mask
    if isinstance(mask, np.ndarray) and mask.ndim == 3:
        if mask.shape != image.shape:
            raise ValueError(
                f"3D mask shape {mask.shape} doesn't match image shape {image.shape}"
            )

        # Apply 3D mask directly
        masked = image.astype(np.float32) * mask.astype(np.float32)
        return masked.astype(image.dtype)

    # If we get here, the mask is neither 2D nor 3D NumPy array
    raise TypeError(f"mask must be a 2D or 3D NumPy array, got {type(mask)}")

@numpy_func
def create_weight_mask(shape: Tuple[int, int], margin_ratio: float = 0.1) -> np.ndarray:
    """
    Create a weight mask for blending images.

    Args:
        shape: Shape of the mask (height, width)
        margin_ratio: Ratio of image size to use as margin

    Returns:
        2D NumPy weight mask of shape (Y, X)
    """
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise TypeError("shape must be a tuple of (height, width)")

    height, width = shape
    return create_linear_weight_mask(height, width, margin_ratio)

@numpy_func
def max_projection(stack: np.ndarray) -> np.ndarray:
    """
    Create a maximum intensity projection from a Z-stack.

    Args:
        stack: 3D NumPy array of shape (Z, Y, X)

    Returns:
        3D NumPy array of shape (1, Y, X)
    """
    _validate_3d_array(stack)

    # Create max projection
    projection_2d = np.max(stack, axis=0)
    return projection_2d.reshape(1, projection_2d.shape[0], projection_2d.shape[1])

@numpy_func
def mean_projection(stack: np.ndarray) -> np.ndarray:
    """
    Create a mean intensity projection from a Z-stack.

    Args:
        stack: 3D NumPy array of shape (Z, Y, X)

    Returns:
        3D NumPy array of shape (1, Y, X)
    """
    _validate_3d_array(stack)

    # Create mean projection
    projection_2d = np.mean(stack, axis=0).astype(stack.dtype)
    return projection_2d.reshape(1, projection_2d.shape[0], projection_2d.shape[1])

@numpy_func
def spatial_bin_2d(
    stack: np.ndarray,
    bin_size: int = 2,
    method: str = "mean"
) -> np.ndarray:
    """
    Apply 2D spatial binning to each slice in the stack.

    Reduces spatial resolution by combining neighboring pixels in 2D blocks.
    Each slice is processed independently.

    Args:
        stack: 3D NumPy array of shape (Z, Y, X)
        bin_size: Size of the square binning kernel (e.g., 2 = 2x2 binning)
        method: Binning method - "mean", "sum", "max", or "min"

    Returns:
        Binned 3D NumPy array of shape (Z, Y//bin_size, X//bin_size)
    """
    _validate_3d_array(stack)

    if bin_size <= 0:
        raise ValueError("bin_size must be positive")
    if method not in ["mean", "sum", "max", "min"]:
        raise ValueError("method must be one of: mean, sum, max, min")

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

    # Apply binning operation
    if method == "mean":
        result = np.mean(reshaped, axis=(2, 4))
    elif method == "sum":
        result = np.sum(reshaped, axis=(2, 4))
    elif method == "max":
        result = np.max(reshaped, axis=(2, 4))
    elif method == "min":
        result = np.min(reshaped, axis=(2, 4))

    return result.astype(stack.dtype)

@numpy_func
def spatial_bin_3d(
    stack: np.ndarray,
    bin_size: int = 2,
    method: str = "mean"
) -> np.ndarray:
    """
    Apply 3D spatial binning to the entire stack.

    Reduces spatial resolution by combining neighboring voxels in 3D blocks.

    Args:
        stack: 3D NumPy array of shape (Z, Y, X)
        bin_size: Size of the cubic binning kernel (e.g., 2 = 2x2x2 binning)
        method: Binning method - "mean", "sum", "max", or "min"

    Returns:
        Binned 3D NumPy array of shape (Z//bin_size, Y//bin_size, X//bin_size)
    """
    _validate_3d_array(stack)

    if bin_size <= 0:
        raise ValueError("bin_size must be positive")
    if method not in ["mean", "sum", "max", "min"]:
        raise ValueError("method must be one of: mean, sum, max, min")

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

    # Apply binning operation across the three bin_size dimensions
    if method == "mean":
        result = np.mean(reshaped, axis=(1, 3, 5))
    elif method == "sum":
        result = np.sum(reshaped, axis=(1, 3, 5))
    elif method == "max":
        result = np.max(reshaped, axis=(1, 3, 5))
    elif method == "min":
        result = np.min(reshaped, axis=(1, 3, 5))

    return result.astype(stack.dtype)

@numpy_func
def stack_equalize_histogram(
    stack: np.ndarray,
    bins: int = 65536,
    range_min: float = 0.0,
    range_max: float = 65535.0
) -> np.ndarray:
    """
    Apply histogram equalization to an entire stack.

    This ensures consistent contrast enhancement across all Z-slices by
    computing a global histogram across the entire stack.

    Args:
        stack: 3D NumPy array of shape (Z, Y, X)
        bins: Number of bins for histogram computation
        range_min: Minimum value for histogram range
        range_max: Maximum value for histogram range

    Returns:
        Equalized 3D NumPy array of shape (Z, Y, X)
    """
    _validate_3d_array(stack)

    # Flatten the entire stack to compute the global histogram
    flat_stack = stack.flatten()

    # Calculate the histogram and cumulative distribution function (CDF)
    hist, bin_edges = np.histogram(flat_stack, bins=bins, range=(range_min, range_max))
    cdf = hist.cumsum()

    # Normalize the CDF to the range [0, 65535]
    # Avoid division by zero
    if cdf[-1] > 0:
        cdf = 65535 * cdf / cdf[-1]

    # Use linear interpolation to map input values to equalized values
    equalized_stack = np.interp(stack.flatten(), bin_edges[:-1], cdf).reshape(stack.shape)

    # Convert to uint16
    return equalized_stack.astype(np.uint16)

@numpy_func
def create_projection(stack: np.ndarray, method: str = "max_projection") -> np.ndarray:
    """
    Create a projection from a stack using the specified method.

    Args:
        stack: 3D NumPy array of shape (Z, Y, X)
        method: Projection method (max_projection, mean_projection)

    Returns:
        3D NumPy array of shape (1, Y, X)
    """
    _validate_3d_array(stack)

    if method == "max_projection":
        return max_projection(stack)

    if method == "mean_projection":
        return mean_projection(stack)

    # FAIL FAST: No fallback projection methods
    raise ValueError(f"Unknown projection method: {method}. Valid methods: max_projection, mean_projection")

@numpy_func
def tophat(
    image: np.ndarray,
    selem_radius: int = 50,
    downsample_factor: int = 4,
    downsample_anti_aliasing: bool = True,
    upsample_order: int = 0
) -> np.ndarray:
    """
    Apply white top-hat filter to a 3D image for background removal.

    This applies the filter to each Z-slice independently.

    Args:
        image: 3D NumPy array of shape (Z, Y, X)
        selem_radius: Radius of the structuring element disk
        downsample_factor: Factor by which to downsample the image for processing
        downsample_anti_aliasing: Whether to use anti-aliasing when downsampling
        upsample_order: Interpolation order for upsampling (0=nearest, 1=linear, etc.)

    Returns:
        Filtered 3D NumPy array of shape (Z, Y, X)
    """
    _validate_3d_array(image)

    # Process each Z-slice independently
    result = np.zeros_like(image)

    for z in range(image.shape[0]):
        # Store original data type
        input_dtype = image[z].dtype

        # 1) Downsample
        image_small = trans.resize(
            image[z],
            (image[z].shape[0]//downsample_factor, image[z].shape[1]//downsample_factor),
            anti_aliasing=downsample_anti_aliasing,
            preserve_range=True
        )

        # 2) Build structuring element for the smaller image
        selem_small = morph.disk(selem_radius // downsample_factor)

        # 3) White top-hat on the smaller image
        tophat_small = morph.white_tophat(image_small, selem_small)

        # 4) Upscale background to original size
        background_small = image_small - tophat_small
        background_large = trans.resize(
            background_small,
            image[z].shape,
            order=upsample_order,
            preserve_range=True
        )

        # 5) Subtract background and clip negative values
        slice_result = np.maximum(image[z] - background_large, 0)

        # 6) Convert back to original data type
        result[z] = slice_result.astype(input_dtype)

    return result
