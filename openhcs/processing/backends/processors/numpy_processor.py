"""
NumPy Image Processor Implementation

This module implements the ImageProcessorInterface using NumPy as the backend.
It serves as the reference implementation and works on CPU.

Doctrinal Clauses:
- Clause 3 — Declarative Primacy: All functions are pure and stateless
- Clause 88 — No Inferred Capabilities: Explicit NumPy dependency
- Clause 106-A — Declared Memory Types: All methods specify NumPy arrays
"""

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
    p_low = np.percentile(stack, low_percentile)
    p_high = np.percentile(stack, high_percentile)

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
    cls, images: List[np.ndarray], weights: Optional[List[float]] = None
) -> np.ndarray:
    """
    Create a composite image from multiple 3D arrays.

    Args:
        images: List of 3D NumPy arrays, each of shape (Z, Y, X)
        weights: List of weights for each image. If None, equal weights are used.

    Returns:
        Composite 3D NumPy array of shape (Z, Y, X)
    """
    # Ensure images is a list
    if not isinstance(images, list):
        raise TypeError("images must be a list of NumPy arrays")

    # Check for empty list early
    if not images:
        raise ValueError("images list cannot be empty")

    # Validate all images are 3D NumPy arrays with the same shape
    for i, img in enumerate(images):
        _validate_3d_array(img, f"images[{i}]")
        if img.shape != images[0].shape:
            raise ValueError(f"All images must have the same shape. "
                            f"images[0] has shape {images[0].shape}, "
                            f"images[{i}] has shape {img.shape}")

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
    composite = np.zeros(shape, dtype=np.float32)
    total_weight = 0.0

    # Add each image with its weight
    for i, image in enumerate(images):
        weight = weights[i]
        if weight <= 0.0:
            continue

        # Add to composite
        composite += image.astype(np.float32) * weight
        total_weight += weight

    # Normalize by total weight
    if total_weight > 0:
        composite /= total_weight

    # Convert back to original dtype (usually uint16)
    if np.issubdtype(dtype, np.integer):
        max_val = np.iinfo(dtype).max
        composite = np.clip(composite, 0, max_val).astype(dtype)
    else:
        composite = composite.astype(dtype)

    return composite

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
def stack_equalize_histogram(
    cls, stack: np.ndarray,
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
    cls, image: np.ndarray,
    selem_radius: int = 50,
    downsample_factor: int = 4
) -> np.ndarray:
    """
    Apply white top-hat filter to a 3D image for background removal.

    This applies the filter to each Z-slice independently.

    Args:
        image: 3D NumPy array of shape (Z, Y, X)
        selem_radius: Radius of the structuring element disk
        downsample_factor: Factor by which to downsample the image for processing

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
            anti_aliasing=True,
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
            anti_aliasing=False,
            preserve_range=True
        )

        # 5) Subtract background and clip negative values
        slice_result = np.maximum(image[z] - background_large, 0)

        # 6) Convert back to original data type
        result[z] = slice_result.astype(input_dtype)

    return result