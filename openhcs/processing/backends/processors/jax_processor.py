"""
JAX Image Processor Implementation

This module implements the ImageProcessorInterface using JAX as the backend.
It leverages GPU acceleration for image processing operations.

Doctrinal Clauses:
- Clause 3 — Declarative Primacy: All functions are pure and stateless
- Clause 88 — No Inferred Capabilities: Explicit JAX dependency
- Clause 106-A — Declared Memory Types: All methods specify JAX arrays
"""
from __future__ import annotations 

import logging
from typing import Any, List, Optional, Tuple

from openhcs.core.memory.decorators import jax as jax_func
from openhcs.core.utils import optional_import

# Import JAX as an optional dependency
jax = optional_import("jax")
jnp = optional_import("jax.numpy") if jax is not None else None
lax = jax.lax if jax is not None else None

logger = logging.getLogger(__name__)


@jax_func
def create_linear_weight_mask(height: int, width: int, margin_ratio: float = 0.1) -> "jnp.ndarray":
    """
    Create a 2D weight mask that linearly ramps from 0 at the edges to 1 in the center.

    Args:
        height: Height of the mask
        width: Width of the mask
        margin_ratio: Ratio of the margin to the image size

    Returns:
        2D JAX weight mask of shape (height, width)
    """
    # The compiler will ensure this function is only called when JAX is available
    # No need to check for JAX availability here

    margin_y = int(jnp.floor(height * margin_ratio))
    margin_x = int(jnp.floor(width * margin_ratio))

    weight_h = jnp.ones(height, dtype=jnp.float32)
    if margin_y > 0:
        ramp_top = jnp.linspace(0, 1, margin_y, endpoint=False)
        ramp_bottom = jnp.linspace(1, 0, margin_y, endpoint=False)
        weight_h = weight_h.at[:margin_y].set(ramp_top)
        weight_h = weight_h.at[-margin_y:].set(ramp_bottom)

    weight_x = jnp.ones(width, dtype=jnp.float32)
    if margin_x > 0:
        ramp_left = jnp.linspace(0, 1, margin_x, endpoint=False)
        ramp_right = jnp.linspace(1, 0, margin_x, endpoint=False)
        weight_x = weight_x.at[:margin_x].set(ramp_left)
        weight_x = weight_x.at[-margin_x:].set(ramp_right)

    # Create 2D weight mask using outer product
    weight_mask = jnp.outer(weight_h, weight_x)

    return weight_mask


def _validate_3d_array(array: Any, name: str = "input") -> None:
    """
    Validate that the input is a 3D JAX array.

    Args:
        array: Array to validate
        name: Name of the array for error messages

    Raises:
        TypeError: If the array is not a JAX array
        ValueError: If the array is not 3D
        ImportError: If JAX is not available
    """
    # The compiler will ensure this function is only called when JAX is available
    # No need to check for JAX availability here

    if not isinstance(array, jnp.ndarray):
        raise TypeError(f"{name} must be a JAX array, got {type(array)}. "
                       f"No automatic conversion is performed to maintain explicit contracts.")

    if array.ndim != 3:
        raise ValueError(f"{name} must be a 3D array, got {array.ndim}D")

@jax_func
def _gaussian_kernel(sigma: float, kernel_size: int) -> "jnp.ndarray":
    """
    Create a 2D Gaussian kernel.

    Args:
        sigma: Standard deviation of the Gaussian kernel
        kernel_size: Size of the kernel (must be odd)

    Returns:
        2D JAX array of shape (kernel_size, kernel_size)
    """
    # Ensure kernel_size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Create 1D Gaussian kernel
    x = jnp.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=jnp.float32)
    kernel_1d = jnp.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / jnp.sum(kernel_1d)

    # Create 2D Gaussian kernel
    kernel_2d = jnp.outer(kernel_1d, kernel_1d)

    return kernel_2d

@jax_func
def _gaussian_blur(image: "jnp.ndarray", sigma: float) -> "jnp.ndarray":
    """
    Apply Gaussian blur to a 2D image.

    Args:
        image: 2D JAX array of shape (H, W)
        sigma: Standard deviation of the Gaussian kernel

    Returns:
        Blurred 2D JAX array of shape (H, W)
    """
    # Calculate kernel size based on sigma
    kernel_size = max(3, int(2 * 4 * sigma + 1))

    # Create Gaussian kernel
    kernel = _gaussian_kernel(sigma, kernel_size)

    # Pad the image for convolution
    pad_size = kernel_size // 2
    padded = jnp.pad(image, ((pad_size, pad_size), (pad_size, pad_size)), mode='reflect')

    # Apply convolution
    # JAX doesn't have a direct 2D convolution function for arbitrary kernels
    # We'll use lax.conv_general_dilated with appropriate parameters

    # Reshape inputs for lax.conv_general_dilated
    kernel_reshaped = kernel.reshape(kernel_size, kernel_size, 1, 1)
    padded_reshaped = padded.reshape(1, padded.shape[0], padded.shape[1], 1)

    # Apply convolution
    result = lax.conv_general_dilated(
        padded_reshaped,
        kernel_reshaped,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )

    # Reshape back to 2D
    return result[0, :, :, 0]

@jax_func
def sharpen(image: "jnp.ndarray", radius: float = 1.0, amount: float = 1.0
) -> "jnp.ndarray":
    """
    Sharpen a 3D image using unsharp masking.

    This applies sharpening to each Z-slice independently.

    Args:
        image: 3D JAX array of shape (Z, Y, X)
        radius: Radius of Gaussian blur
        amount: Sharpening strength

    Returns:
        Sharpened 3D JAX array of shape (Z, Y, X)
    """
    _validate_3d_array(image)

    # Store original dtype
    dtype = image.dtype

    # Process each Z-slice independently
    result_list = []

    for z in range(image.shape[0]):
        # Convert to float for processing
        slice_float = image[z].astype(jnp.float32) / jnp.max(image[z])

        # Create blurred version for unsharp mask
        blurred = _gaussian_blur(slice_float, sigma=radius)

        # Apply unsharp mask: original + amount * (original - blurred)
        sharpened = slice_float + amount * (slice_float - blurred)

        # Clip to valid range
        sharpened = jnp.clip(sharpened, 0.0, 1.0)

        # Scale back to original range
        min_val = jnp.min(sharpened)
        max_val = jnp.max(sharpened)
        if max_val > min_val:
            sharpened = (sharpened - min_val) * 65535.0 / (max_val - min_val)

        result_list.append(sharpened)

    # Stack results back into a 3D array
    result = jnp.stack(result_list, axis=0)

    # Convert back to original dtype
    if jnp.issubdtype(dtype, jnp.integer):
        result = jnp.clip(result, 0, 65535).astype(jnp.uint16)
    else:
        result = result.astype(dtype)

    return result

@jax_func
def percentile_normalize(
    image: "jnp.ndarray",
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    target_min: float = 0.0,
    target_max: float = 65535.0
) -> "jnp.ndarray":
    """
    Normalize a 3D image using percentile-based contrast stretching.

    This applies normalization to each Z-slice independently.

    Args:
        image: 3D JAX array of shape (Z, Y, X)
        low_percentile: Lower percentile (0-100)
        high_percentile: Upper percentile (0-100)
        target_min: Target minimum value
        target_max: Target maximum value

    Returns:
        Normalized 3D JAX array of shape (Z, Y, X)
    """
    _validate_3d_array(image)

    # Process each Z-slice independently
    result_list = []

    # Define a function to normalize a single slice
    def normalize_single_slice(slice_idx):
        slice_data = image[slice_idx]

        # Get percentile values for this slice
        p_low = jnp.percentile(slice_data, low_percentile)
        p_high = jnp.percentile(slice_data, high_percentile)

        # Avoid division by zero
        equal_percentiles = jnp.isclose(p_high, p_low)

        # Function to normalize when percentiles are different
        def normalize_slice(args):
            p_low, p_high, slice_data = args
            # Clip and normalize to target range
            clipped = jnp.clip(slice_data.astype(jnp.float32), p_low, p_high)
            scale = (target_max - target_min) / (p_high - p_low)
            normalized = (clipped - p_low) * scale + target_min
            return normalized

        # Function for the case where percentiles are equal
        def return_constant(args):
            _, _, slice_data = args
            return jnp.ones_like(slice_data, dtype=jnp.float32) * target_min

        # Handle the case where percentiles are equal
        normalized = jax.lax.cond(
            equal_percentiles,
            return_constant,
            normalize_slice,
            (p_low, p_high, slice_data)
        )

        return normalized

    # Process each slice
    for z in range(image.shape[0]):
        result_list.append(normalize_single_slice(z))

    # Stack results back into a 3D array
    result = jnp.stack(result_list, axis=0)

    # Convert to uint16
    result = jnp.clip(result, 0, 65535).astype(jnp.uint16)

    return result

@jax_func
def stack_percentile_normalize(
    stack: "jnp.ndarray",
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    target_min: float = 0.0,
    target_max: float = 65535.0
) -> "jnp.ndarray":
    """
    Normalize a stack using global percentile-based contrast stretching.

    This ensures consistent normalization across all Z-slices by computing
    global percentiles across the entire stack.

    Args:
        stack: 3D JAX array of shape (Z, Y, X)
        low_percentile: Lower percentile (0-100)
        high_percentile: Upper percentile (0-100)
        target_min: Target minimum value
        target_max: Target maximum value

    Returns:
        Normalized 3D JAX array of shape (Z, Y, X)
    """
    _validate_3d_array(stack)

    # Calculate global percentiles across the entire stack
    p_low = jnp.percentile(stack, low_percentile)
    p_high = jnp.percentile(stack, high_percentile)

    # Avoid division by zero
    if p_high == p_low:
        return jnp.ones_like(stack) * target_min

    # Clip and normalize to target range (match NumPy implementation exactly)
    clipped = jnp.clip(stack, p_low, p_high)
    normalized = (clipped - p_low) * (target_max - target_min) / (p_high - p_low) + target_min
    normalized = normalized.astype(jnp.uint16)

    return normalized

@jax_func
def create_composite(
    stack: "jnp.ndarray", weights: Optional[List[float]] = None
) -> "jnp.ndarray":
    """
    Create a composite image from a 3D stack where each slice is a channel.

    Args:
        stack: 3D JAX array of shape (N, Y, X) where N is number of channel slices
        weights: List of weights for each slice. If None, equal weights are used.

    Returns:
        Composite 3D JAX array of shape (1, Y, X)
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

    # Convert weights to JAX array for efficient computation
    # CRITICAL: Use float32 for weights to preserve fractional values, not stack.dtype
    weights_array = jnp.array(normalized_weights, dtype=jnp.float32)

    # Reshape weights for broadcasting: (N, 1, 1) to multiply with (N, Y, X)
    weights_array = weights_array.reshape(n_slices, 1, 1)

    # Create composite by weighted sum along the first axis
    # Convert stack to float32 for computation to avoid precision loss
    stack_float = stack.astype(jnp.float32)
    weighted_stack = stack_float * weights_array
    composite_slice = jnp.sum(weighted_stack, axis=0, keepdims=True)  # Keep as (1, Y, X)

    # Convert back to original dtype
    composite_slice = composite_slice.astype(stack.dtype)

    return composite_slice

@jax_func
def apply_mask(image: "jnp.ndarray", mask: "jnp.ndarray") -> "jnp.ndarray":
    """
    Apply a mask to a 3D image.

    This applies the mask to each Z-slice independently if mask is 2D,
    or applies the 3D mask directly if mask is 3D.

    Args:
        image: 3D JAX array of shape (Z, Y, X)
        mask: 3D JAX array of shape (Z, Y, X) or 2D JAX array of shape (Y, X)

    Returns:
        Masked 3D JAX array of shape (Z, Y, X)
    """
    _validate_3d_array(image)

    # Handle 2D mask (apply to each Z-slice)
    if isinstance(mask, jnp.ndarray) and mask.ndim == 2:
        if mask.shape != image.shape[1:]:
            raise ValueError(
                f"2D mask shape {mask.shape} doesn't match image slice shape {image.shape[1:]}"
            )

        # Apply 2D mask to each Z-slice
        result_list = []
        for z in range(image.shape[0]):
            result_list.append(image[z].astype(jnp.float32) * mask.astype(jnp.float32))

        result = jnp.stack(result_list, axis=0)
        return result.astype(image.dtype)

    # Handle 3D mask
    if isinstance(mask, jnp.ndarray) and mask.ndim == 3:
        if mask.shape != image.shape:
            raise ValueError(
                f"3D mask shape {mask.shape} doesn't match image shape {image.shape}"
            )

        # Apply 3D mask directly
        masked = image.astype(jnp.float32) * mask.astype(jnp.float32)
        return masked.astype(image.dtype)

    # If we get here, the mask is neither 2D nor 3D JAX array
    raise TypeError(f"mask must be a 2D or 3D JAX array, got {type(mask)}")

@jax_func
def create_weight_mask(
    shape: Tuple[int, int], margin_ratio: float = 0.1
) -> "jnp.ndarray":
    """
    Create a weight mask for blending images.

    Args:
        shape: Shape of the mask (height, width)
        margin_ratio: Ratio of image size to use as margin

    Returns:
        2D JAX weight mask of shape (Y, X)
    """
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise TypeError("shape must be a tuple of (height, width)")

    height, width = shape
    return create_linear_weight_mask(height, width, margin_ratio)

@jax_func
def max_projection(stack: "jnp.ndarray") -> "jnp.ndarray":
    """
    Create a maximum intensity projection from a Z-stack.

    Args:
        stack: 3D JAX array of shape (Z, Y, X)

    Returns:
        3D JAX array of shape (1, Y, X)
    """
    _validate_3d_array(stack)

    # Create max projection
    projection_2d = jnp.max(stack, axis=0)
    return jnp.expand_dims(projection_2d, axis=0)

@jax_func
def mean_projection(stack: "jnp.ndarray") -> "jnp.ndarray":
    """
    Create a mean intensity projection from a Z-stack.

    Args:
        stack: 3D JAX array of shape (Z, Y, X)

    Returns:
        3D JAX array of shape (1, Y, X)
    """
    _validate_3d_array(stack)

    # Create mean projection
    projection_2d = jnp.mean(stack.astype(jnp.float32), axis=0).astype(stack.dtype)
    return jnp.expand_dims(projection_2d, axis=0)

@jax_func
def stack_equalize_histogram(
    stack: "jnp.ndarray",
    bins: int = 65536,
    range_min: float = 0.0,
    range_max: float = 65535.0
) -> "jnp.ndarray":
    """
    Apply histogram equalization to an entire stack.

    This ensures consistent contrast enhancement across all Z-slices by
    computing a global histogram across the entire stack.

    Args:
        stack: 3D JAX array of shape (Z, Y, X)
        bins: Number of bins for histogram computation
        range_min: Minimum value for histogram range
        range_max: Maximum value for histogram range

    Returns:
        Equalized 3D JAX array of shape (Z, Y, X)
    """
    _validate_3d_array(stack)

    # Flatten the entire stack to compute the global histogram
    flat_stack = stack.flatten()

    # Calculate the histogram
    hist, _ = jnp.histogram(flat_stack, bins=bins, range=(range_min, range_max))

    # Calculate cumulative distribution function (CDF)
    cdf = jnp.cumsum(hist)

    # Normalize the CDF to the range [0, 65535]
    # Avoid division by zero
    cdf_max = jnp.max(cdf)
    cdf_normalized = jax.lax.cond(
        cdf_max > 0,
        lambda x: 65535.0 * x / cdf_max,
        lambda x: x,
        cdf
    )

    # Scale input values to bin indices
    bin_width = (range_max - range_min) / bins
    indices = jnp.clip(
        jnp.floor((flat_stack - range_min) / bin_width).astype(jnp.int32),
        0, bins - 1
    )

    # Look up CDF values
    equalized_flat = jnp.take(cdf_normalized, indices)

    # Reshape back to original shape
    equalized_stack = equalized_flat.reshape(stack.shape)

    # Convert to uint16
    return equalized_stack.astype(jnp.uint16)

@jax_func
def create_projection(
    stack: "jnp.ndarray", method: str = "max_projection"
) -> "jnp.ndarray":
    """
    Create a projection from a stack using the specified method.

    Args:
        stack: 3D JAX array of shape (Z, Y, X)
        method: Projection method (max_projection, mean_projection)

    Returns:
        3D JAX array of shape (1, Y, X)
    """
    _validate_3d_array(stack)

    if method == "max_projection":
        return max_projection(stack)

    if method == "mean_projection":
        return mean_projection(stack)

    # FAIL FAST: No fallback projection methods
    raise ValueError(f"Unknown projection method: {method}. Valid methods: max_projection, mean_projection")

@jax_func
def tophat(
    image: "jnp.ndarray",
    selem_radius: int = 50,
    downsample_factor: int = 4
) -> "jnp.ndarray":
    """
    Apply white top-hat filter to a 3D image for background removal.

    This applies the filter to each Z-slice independently using JAX's
    native operations.

    Args:
        image: 3D JAX array of shape (Z, Y, X)
        selem_radius: Radius of the structuring element disk
        downsample_factor: Factor by which to downsample the image for processing

    Returns:
        Filtered 3D JAX array of shape (Z, Y, X)
    """
    _validate_3d_array(image)

    # Process each Z-slice independently
    result_list = []

    # Define a function to process a single slice
    def process_slice(slice_idx):
        slice_data = image[slice_idx]
        input_dtype = slice_data.dtype

        # 1) Downsample
        # JAX doesn't have a direct resize function, so we'll use a simple approach
        # This is a simplified version and might not match scikit-image's resize exactly
        new_h = slice_data.shape[0] // downsample_factor
        new_w = slice_data.shape[1] // downsample_factor

        # Simple block averaging for downsampling
        slice_data_float = slice_data.astype(jnp.float32)
        blocks = slice_data_float.reshape(
            new_h, downsample_factor, new_w, downsample_factor
        )
        image_small = jnp.mean(blocks, axis=(1, 3))

        # 2) Create a circular structuring element
        small_selem_radius = max(1, selem_radius // downsample_factor)

        # Create grid for structuring element
        y_range = jnp.arange(-small_selem_radius, small_selem_radius + 1)
        x_range = jnp.arange(-small_selem_radius, small_selem_radius + 1)
        grid_y, grid_x = jnp.meshgrid(y_range, x_range, indexing='ij')

        # Create circular mask
        small_mask = (grid_x**2 + grid_y**2) <= small_selem_radius**2
        small_selem = small_mask.astype(jnp.float32)

        # 3) Apply white top-hat
        # JAX doesn't have built-in morphological operations
        # This is a simplified implementation that approximates the behavior

        # Pad the image for convolution
        pad_size = small_selem_radius
        padded = jnp.pad(image_small, pad_size, mode='reflect')

        # Implement erosion (minimum filter)
        # For each pixel, find the minimum value in the neighborhood defined by the structuring element
        eroded = jnp.zeros_like(image_small)

        # This is a simplified approach - in a real implementation, we would use a more efficient method
        for y in range(new_h):
            for x in range(new_w):
                # Extract neighborhood
                neighborhood = padded[y:y+2*pad_size+1, x:x+2*pad_size+1]
                # Apply structuring element and find minimum
                masked_values = jnp.where(small_selem, neighborhood, jnp.inf)
                eroded = eroded.at[y, x].set(jnp.min(masked_values))

        # Implement dilation (maximum filter)
        # For each pixel, find the maximum value in the neighborhood defined by the structuring element
        opened = jnp.zeros_like(image_small)

        # Pad the eroded image
        padded_eroded = jnp.pad(eroded, pad_size, mode='reflect')

        # This is a simplified approach - in a real implementation, we would use a more efficient method
        for y in range(new_h):
            for x in range(new_w):
                # Extract neighborhood
                neighborhood = padded_eroded[y:y+2*pad_size+1, x:x+2*pad_size+1]
                # Apply structuring element and find maximum
                masked_values = jnp.where(small_selem, neighborhood, -jnp.inf)
                opened = opened.at[y, x].set(jnp.max(masked_values))

        # White top-hat is original minus opening
        tophat_small = image_small - opened

        # 4) Calculate background
        background_small = image_small - tophat_small

        # 5) Upscale background to original size
        # Simple nearest neighbor upscaling
        background_large = jnp.repeat(
            jnp.repeat(background_small, downsample_factor, axis=0),
            downsample_factor, axis=1
        )

        # Crop to original size if needed
        if background_large.shape != slice_data.shape:
            background_large = background_large[:slice_data.shape[0], :slice_data.shape[1]]

        # 6) Subtract background and clip negative values
        slice_result = jnp.maximum(slice_data.astype(jnp.float32) - background_large, 0)

        # 7) Convert back to original data type
        return slice_result.astype(input_dtype)

    # Process each slice
    for z in range(image.shape[0]):
        result_list.append(process_slice(z))

    # Stack results back into a 3D array
    result = jnp.stack(result_list, axis=0)

    return result
