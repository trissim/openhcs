"""
TensorFlow Image Processor Implementation

This module implements the ImageProcessorInterface using TensorFlow as the backend.
It leverages GPU acceleration for image processing operations.

Doctrinal Clauses:
- Clause 3 — Declarative Primacy: All functions are pure and stateless
- Clause 88 — No Inferred Capabilities: Explicit TensorFlow dependency
- Clause 106-A — Declared Memory Types: All methods specify TensorFlow tensors
"""
from __future__ import annotations 

import logging
from typing import Any, List, Optional, Tuple

import pkg_resources

from openhcs.core.memory.decorators import tensorflow as tensorflow_func
from openhcs.core.utils import optional_import

# Define error variable
TENSORFLOW_ERROR = ""

# Import TensorFlow as an optional dependency
tf = optional_import("tensorflow")

# Check TensorFlow version for DLPack compatibility if available
if tf is not None:
    try:
        tf_version = pkg_resources.parse_version(tf.__version__)
        min_version = pkg_resources.parse_version("2.12.0")

        if tf_version < min_version:
            TENSORFLOW_ERROR = (
                f"TensorFlow version {tf.__version__} is not supported for DLPack operations. "
                f"Version 2.12.0 or higher is required for stable DLPack support. "
                f"Clause 88 violation: Cannot infer DLPack capability."
            )
            tf = None
    except Exception as e:
        TENSORFLOW_ERROR = str(e)
        tf = None

logger = logging.getLogger(__name__)


@tensorflow_func
def create_linear_weight_mask(height: int, width: int, margin_ratio: float = 0.1) -> "tf.Tensor":
    """
    Create a 2D weight mask that linearly ramps from 0 at the edges to 1 in the center.

    Args:
        height: Height of the mask
        width: Width of the mask
        margin_ratio: Ratio of the margin to the image size

    Returns:
        2D TensorFlow weight mask of shape (height, width)
    """
    # The compiler will ensure this function is only called when TensorFlow is available
    # No need to check for TensorFlow availability here

    margin_y = int(tf.math.floor(height * margin_ratio))
    margin_x = int(tf.math.floor(width * margin_ratio))

    weight_y = tf.ones(height, dtype=tf.float32)
    if margin_y > 0:
        ramp_top = tf.linspace(0.0, 1.0, margin_y)
        ramp_bottom = tf.linspace(1.0, 0.0, margin_y)

        # Update slices of the weight_y tensor
        weight_y = tf.tensor_scatter_nd_update(
            weight_y,
            tf.stack([tf.range(margin_y)], axis=1),
            ramp_top
        )
        weight_y = tf.tensor_scatter_nd_update(
            weight_y,
            tf.stack([tf.range(height - margin_y, height)], axis=1),
            ramp_bottom
        )

    weight_x = tf.ones(width, dtype=tf.float32)
    if margin_x > 0:
        ramp_left = tf.linspace(0.0, 1.0, margin_x)
        ramp_right = tf.linspace(1.0, 0.0, margin_x)

        # Update slices of the weight_x tensor
        weight_x = tf.tensor_scatter_nd_update(
            weight_x,
            tf.stack([tf.range(margin_x)], axis=1),
            ramp_left
        )
        weight_x = tf.tensor_scatter_nd_update(
            weight_x,
            tf.stack([tf.range(width - margin_x, width)], axis=1),
            ramp_right
        )

    # Create 2D weight mask using outer product
    weight_mask = tf.tensordot(weight_y, weight_x, axes=0)

    return weight_mask


def _validate_3d_array(array: Any, name: str = "input") -> None:
    """
    Validate that the input is a 3D TensorFlow tensor.

    Args:
        array: Array to validate
        name: Name of the array for error messages

    Raises:
        TypeError: If the array is not a TensorFlow tensor
        ValueError: If the array is not 3D
        ImportError: If TensorFlow is not available
    """
    # The compiler will ensure this function is only called when TensorFlow is available
    # No need to check for TensorFlow availability here

    if not isinstance(array, tf.Tensor):
        raise TypeError(f"{name} must be a TensorFlow tensor, got {type(array)}. "
                       f"No automatic conversion is performed to maintain explicit contracts.")

    if len(array.shape) != 3:
        raise ValueError(f"{name} must be a 3D tensor, got {len(array.shape)}D")

def _gaussian_blur(image: "tf.Tensor", sigma: float) -> "tf.Tensor":
    """
    Apply Gaussian blur to a 2D image.

    Args:
        image: 2D TensorFlow tensor of shape (H, W)
        sigma: Standard deviation of the Gaussian kernel

    Returns:
        Blurred 2D TensorFlow tensor of shape (H, W)
    """
    # Calculate kernel size based on sigma
    kernel_size = max(3, int(2 * 4 * sigma + 1))
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd kernel size

    # Add batch and channel dimensions for tf.image.gaussian_blur
    img = tf.expand_dims(tf.expand_dims(image, 0), -1)

    # Apply Gaussian blur
    blurred = tf.image.gaussian_blur(
        img,
        [kernel_size, kernel_size],
        sigma,
        "REFLECT"
    )

    # Remove batch and channel dimensions
    return tf.squeeze(blurred)

@tensorflow_func
def sharpen(
    image: "tf.Tensor", radius: float = 1.0, amount: float = 1.0
) -> "tf.Tensor":
    """
    Sharpen a 3D image using unsharp masking.

    This applies sharpening to each Z-slice independently.

    Args:
        image: 3D TensorFlow tensor of shape (Z, Y, X)
        radius: Radius of Gaussian blur
        amount: Sharpening strength

    Returns:
        Sharpened 3D TensorFlow tensor of shape (Z, Y, X)
    """
    _validate_3d_array(image)

    # Store original dtype
    dtype = image.dtype

    # Process each Z-slice independently
    result_list = []

    for z in range(image.shape[0]):
        # Convert to float for processing
        slice_float = tf.cast(image[z], tf.float32) / tf.reduce_max(image[z])

        # Create blurred version for unsharp mask
        blurred = _gaussian_blur(slice_float, sigma=radius)

        # Apply unsharp mask: original + amount * (original - blurred)
        sharpened = slice_float + amount * (slice_float - blurred)

        # Clip to valid range
        sharpened = tf.clip_by_value(sharpened, 0.0, 1.0)

        # Scale back to original range
        min_val = tf.reduce_min(sharpened)
        max_val = tf.reduce_max(sharpened)
        if max_val > min_val:
            sharpened = (sharpened - min_val) * 65535.0 / (max_val - min_val)

        result_list.append(sharpened)

    # Stack results back into a 3D tensor
    result = tf.stack(result_list, axis=0)

    # Convert back to original dtype
    if dtype == tf.uint16:
        result = tf.cast(tf.clip_by_value(result, 0, 65535), tf.uint16)
    else:
        result = tf.cast(result, dtype)

    return result

@tensorflow_func
def percentile_normalize(
    image: "tf.Tensor",
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    target_min: float = 0.0,
    target_max: float = 65535.0
) -> "tf.Tensor":
    """
    Normalize a 3D image using percentile-based contrast stretching.

    This applies normalization to each Z-slice independently.

    Args:
        image: 3D TensorFlow tensor of shape (Z, Y, X)
        low_percentile: Lower percentile (0-100)
        high_percentile: Upper percentile (0-100)
        target_min: Target minimum value
        target_max: Target maximum value

    Returns:
        Normalized 3D TensorFlow tensor of shape (Z, Y, X)
    """
    _validate_3d_array(image)

    # Process each Z-slice independently
    result_list = []

    for z in range(image.shape[0]):
        # Get percentile values for this slice
        # TensorFlow doesn't have a direct percentile function, so we use a workaround
        flat_slice = tf.reshape(image[z], [-1])
        sorted_slice = tf.sort(flat_slice)

        # Calculate indices for percentiles
        slice_size = tf.cast(tf.size(flat_slice), tf.float32)
        low_idx = tf.cast(tf.math.floor(slice_size * low_percentile / 100.0), tf.int32)
        high_idx = tf.cast(tf.math.floor(slice_size * high_percentile / 100.0), tf.int32)

        # Get percentile values
        p_low = sorted_slice[low_idx]
        p_high = sorted_slice[high_idx]

        # Avoid division by zero
        if p_high == p_low:
            result_list.append(tf.ones_like(image[z], dtype=tf.float32) * target_min)
            continue

        # Clip and normalize to target range
        clipped = tf.clip_by_value(tf.cast(image[z], tf.float32), p_low, p_high)
        scale = (target_max - target_min) / (p_high - p_low)
        normalized = (clipped - p_low) * scale + target_min
        result_list.append(normalized)

    # Stack results back into a 3D tensor
    result = tf.stack(result_list, axis=0)

    # Convert to uint16
    result = tf.cast(tf.clip_by_value(result, 0, 65535), tf.uint16)

    return result

@tensorflow_func
def stack_percentile_normalize(
    stack: "tf.Tensor",
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    target_min: float = 0.0,
    target_max: float = 65535.0
) -> "tf.Tensor":
    """
    Normalize a stack using global percentile-based contrast stretching.

    This ensures consistent normalization across all Z-slices by computing
    global percentiles across the entire stack.

    Args:
        stack: 3D TensorFlow tensor of shape (Z, Y, X)
        low_percentile: Lower percentile (0-100)
        high_percentile: Upper percentile (0-100)
        target_min: Target minimum value
        target_max: Target maximum value

    Returns:
        Normalized 3D TensorFlow tensor of shape (Z, Y, X)
    """
    _validate_3d_array(stack)

    # Calculate global percentiles across the entire stack using TensorFlow Probability
    # This is memory-efficient and doesn't require sorting the entire array
    try:
        import tensorflow_probability as tfp
        p_low = tf.cast(tfp.stats.percentile(stack, low_percentile), tf.float32)
        p_high = tf.cast(tfp.stats.percentile(stack, high_percentile), tf.float32)
    except ImportError:
        # Fallback to manual calculation if TensorFlow Probability is not available
        # This is less memory-efficient but works
        flat_stack = tf.reshape(stack, [-1])
        sorted_stack = tf.sort(flat_stack)

        # Calculate indices for percentiles
        stack_size = tf.cast(tf.size(flat_stack), tf.float32)
        low_idx = tf.cast(tf.math.floor(stack_size * low_percentile / 100.0), tf.int32)
        high_idx = tf.cast(tf.math.floor(stack_size * high_percentile / 100.0), tf.int32)

        # Get percentile values and cast to float32 for consistency
        p_low = tf.cast(sorted_stack[low_idx], tf.float32)
        p_high = tf.cast(sorted_stack[high_idx], tf.float32)

    # Avoid division by zero
    if p_high == p_low:
        return tf.ones_like(stack) * target_min

    # Clip and normalize to target range (match NumPy implementation exactly)
    clipped = tf.clip_by_value(stack, p_low, p_high)
    normalized = (clipped - p_low) * (target_max - target_min) / (p_high - p_low) + target_min
    normalized = tf.cast(normalized, tf.uint16)

    return normalized

@tensorflow_func
def create_composite(
    images: List["tf.Tensor"], weights: Optional[List[float]] = None
) -> "tf.Tensor":
    """
    Create a composite image from multiple 3D arrays.

    Args:
        images: List of 3D TensorFlow tensors, each of shape (Z, Y, X)
        weights: List of weights for each image. If None, equal weights are used.

    Returns:
        Composite 3D TensorFlow tensor of shape (Z, Y, X)
    """
    # Ensure images is a list
    if not isinstance(images, list):
        raise TypeError("images must be a list of TensorFlow tensors")

    # Check for empty list early
    if not images:
        raise ValueError("images list cannot be empty")

    # Validate all images are 3D TensorFlow tensors with the same shape
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
    composite = tf.zeros(shape, dtype=tf.float32)
    total_weight = 0.0

    # Add each image with its weight
    for i, image in enumerate(images):
        weight = weights[i]
        if weight <= 0.0:
            continue

        # Add to composite
        composite += tf.cast(image, tf.float32) * weight
        total_weight += weight

    # Normalize by total weight
    if total_weight > 0:
        composite /= total_weight

    # Convert back to original dtype (usually uint16)
    if dtype in [tf.uint8, tf.uint16, tf.uint32, tf.int8, tf.int16, tf.int32, tf.int64]:
        # Get the maximum value for the specific integer dtype
        if dtype == tf.uint8:
            max_val = 255
        elif dtype == tf.uint16:
            max_val = 65535
        elif dtype == tf.uint32:
            max_val = 4294967295
        elif dtype == tf.int8:
            max_val = 127
        elif dtype == tf.int16:
            max_val = 32767
        elif dtype == tf.int32:
            max_val = 2147483647
        elif dtype == tf.int64:
            max_val = 9223372036854775807

        composite = tf.cast(tf.clip_by_value(composite, 0, max_val), dtype)
    else:
        composite = tf.cast(composite, dtype)

    return composite

@tensorflow_func
def apply_mask(image: "tf.Tensor", mask: "tf.Tensor") -> "tf.Tensor":
    """
    Apply a mask to a 3D image.

    This applies the mask to each Z-slice independently if mask is 2D,
    or applies the 3D mask directly if mask is 3D.

    Args:
        image: 3D TensorFlow tensor of shape (Z, Y, X)
        mask: 3D TensorFlow tensor of shape (Z, Y, X) or 2D TensorFlow tensor of shape (Y, X)

    Returns:
        Masked 3D TensorFlow tensor of shape (Z, Y, X)
    """
    _validate_3d_array(image)

    # Handle 2D mask (apply to each Z-slice)
    if isinstance(mask, tf.Tensor) and len(mask.shape) == 2:
        if mask.shape != image.shape[1:]:
            raise ValueError(
                f"2D mask shape {mask.shape} doesn't match image slice shape {image.shape[1:]}"
            )

        # Apply 2D mask to each Z-slice
        result_list = []
        for z in range(image.shape[0]):
            result_list.append(tf.cast(image[z], tf.float32) * tf.cast(mask, tf.float32))

        result = tf.stack(result_list, axis=0)
        return tf.cast(result, image.dtype)

    # Handle 3D mask
    if isinstance(mask, tf.Tensor) and len(mask.shape) == 3:
        if mask.shape != image.shape:
            raise ValueError(
                f"3D mask shape {mask.shape} doesn't match image shape {image.shape}"
            )

        # Apply 3D mask directly
        masked = tf.cast(image, tf.float32) * tf.cast(mask, tf.float32)
        return tf.cast(masked, image.dtype)

    # If we get here, the mask is neither 2D nor 3D TensorFlow tensor
    raise TypeError(f"mask must be a 2D or 3D TensorFlow tensor, got {type(mask)}")

@tensorflow_func
def create_weight_mask(
    shape: Tuple[int, int], margin_ratio: float = 0.1
) -> "tf.Tensor":
    """
    Create a weight mask for blending images.

    Args:
        shape: Shape of the mask (height, width)
        margin_ratio: Ratio of image size to use as margin

    Returns:
        2D TensorFlow weight mask of shape (Y, X)
    """
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise TypeError("shape must be a tuple of (height, width)")

    height, width = shape
    return create_linear_weight_mask(height, width, margin_ratio)

@tensorflow_func
def max_projection(stack: "tf.Tensor") -> "tf.Tensor":
    """
    Create a maximum intensity projection from a Z-stack.

    Args:
        stack: 3D TensorFlow tensor of shape (Z, Y, X)

    Returns:
        3D TensorFlow tensor of shape (1, Y, X)
    """
    _validate_3d_array(stack)

    # Create max projection
    projection_2d = tf.reduce_max(stack, axis=0)
    return tf.expand_dims(projection_2d, axis=0)

@tensorflow_func
def mean_projection(stack: "tf.Tensor") -> "tf.Tensor":
    """
    Create a mean intensity projection from a Z-stack.

    Args:
        stack: 3D TensorFlow tensor of shape (Z, Y, X)

    Returns:
        3D TensorFlow tensor of shape (1, Y, X)
    """
    _validate_3d_array(stack)

    # Create mean projection
    projection_2d = tf.cast(tf.reduce_mean(tf.cast(stack, tf.float32), axis=0), stack.dtype)
    return tf.expand_dims(projection_2d, axis=0)

@tensorflow_func
def stack_equalize_histogram(
    stack: "tf.Tensor",
    bins: int = 65536,
    range_min: float = 0.0,
    range_max: float = 65535.0
) -> "tf.Tensor":
    """
    Apply histogram equalization to an entire stack.

    This ensures consistent contrast enhancement across all Z-slices by
    computing a global histogram across the entire stack.

    Args:
        stack: 3D TensorFlow tensor of shape (Z, Y, X)
        bins: Number of bins for histogram computation
        range_min: Minimum value for histogram range
        range_max: Maximum value for histogram range

    Returns:
        Equalized 3D TensorFlow tensor of shape (Z, Y, X)
    """
    _validate_3d_array(stack)

    # TensorFlow doesn't have a direct histogram equalization function
    # We'll implement it manually

    # Flatten the entire stack to compute the global histogram
    flat_stack = tf.reshape(tf.cast(stack, tf.float32), [-1])

    # Calculate the histogram
    # TensorFlow doesn't have a direct equivalent to numpy's histogram
    # We'll use tf.histogram_fixed_width
    hist = tf.histogram_fixed_width(
        flat_stack,
        [range_min, range_max],
        nbins=bins
    )

    # Calculate cumulative distribution function (CDF)
    cdf = tf.cumsum(hist)

    # Normalize the CDF to the range [0, 65535]
    # Avoid division by zero
    if tf.reduce_max(cdf) > 0:
        cdf = 65535.0 * cdf / tf.cast(cdf[-1], tf.float32)

    # We don't need bin width for the lookup table approach

    # Scale input values to bin indices
    indices = tf.cast(tf.clip_by_value(
        tf.math.floor((flat_stack - range_min) / (range_max - range_min) * bins),
        0, bins - 1
    ), tf.int32)

    # Look up CDF values
    equalized_flat = tf.gather(cdf, indices)

    # Reshape back to original shape
    equalized_stack = tf.reshape(equalized_flat, stack.shape)

    # Convert to uint16
    return tf.cast(equalized_stack, tf.uint16)

@tensorflow_func
def create_projection(
    stack: "tf.Tensor", method: str = "max_projection"
) -> "tf.Tensor":
    """
    Create a projection from a stack using the specified method.

    Args:
        stack: 3D TensorFlow tensor of shape (Z, Y, X)
        method: Projection method (max_projection, mean_projection)

    Returns:
        3D TensorFlow tensor of shape (1, Y, X)
    """
    _validate_3d_array(stack)

    if method == "max_projection":
        return max_projection(stack)

    if method == "mean_projection":
        return mean_projection(stack)

    # FAIL FAST: No fallback projection methods
    raise ValueError(f"Unknown projection method: {method}. Valid methods: max_projection, mean_projection")

@tensorflow_func
def tophat(
    image: "tf.Tensor",
    selem_radius: int = 50,
    downsample_factor: int = 4
) -> "tf.Tensor":
    """
    Apply white top-hat filter to a 3D image for background removal.

    This applies the filter to each Z-slice independently using TensorFlow's
    native operations.

    Args:
        image: 3D TensorFlow tensor of shape (Z, Y, X)
        selem_radius: Radius of the structuring element disk
        downsample_factor: Factor by which to downsample the image for processing

    Returns:
        Filtered 3D TensorFlow tensor of shape (Z, Y, X)
    """
    _validate_3d_array(image)

    # Process each Z-slice independently
    result_list = []

    for z in range(image.shape[0]):
        # Store original data type
        input_dtype = image[z].dtype

        # 1) Downsample using TensorFlow's resize function
        # First, add batch and channel dimensions for resize
        img_4d = tf.expand_dims(tf.expand_dims(tf.cast(image[z], tf.float32), 0), -1)

        # Calculate new dimensions
        new_h = tf.cast(tf.shape(image[z])[0] // downsample_factor, tf.int32)
        new_w = tf.cast(tf.shape(image[z])[1] // downsample_factor, tf.int32)

        # Resize using TensorFlow's resize function
        image_small = tf.squeeze(tf.image.resize(
            img_4d,
            [new_h, new_w],
            method=tf.image.ResizeMethod.BILINEAR
        ), axis=[0, -1])

        # 2) Create a circular structuring element
        small_selem_radius = tf.maximum(1, selem_radius // downsample_factor)
        small_grid_size = 2 * small_selem_radius + 1

        # Create grid for structuring element
        y_range = tf.range(-small_selem_radius, small_selem_radius + 1, dtype=tf.float32)
        x_range = tf.range(-small_selem_radius, small_selem_radius + 1, dtype=tf.float32)
        small_y_grid, small_x_grid = tf.meshgrid(y_range, x_range)

        # Create circular mask
        small_mask = tf.cast(
            tf.sqrt(tf.square(small_y_grid) + tf.square(small_x_grid)) <= small_selem_radius,
            tf.float32
        )

        # 3) Apply white top-hat using TensorFlow's morphological operations
        # White top-hat is opening subtracted from the original image
        # Opening is erosion followed by dilation

        # Implement erosion using TensorFlow's conv2d with custom kernel

        # Pad the image to handle boundary conditions
        pad_size = small_selem_radius
        padded = tf.pad(
            tf.expand_dims(tf.expand_dims(image_small, 0), -1),
            [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]],
            mode='SYMMETRIC'
        )

        # For erosion, we need to find the minimum value in the neighborhood
        # We can use a trick: negate the image, apply max pooling, then negate again
        neg_padded = -padded

        # Apply convolution with the kernel
        # We use a large negative value for pixels outside the mask
        mask_expanded = tf.reshape(small_mask, [small_grid_size, small_grid_size, 1, 1])
        mask_complement = 1.0 - mask_expanded
        large_neg = tf.constant(-1e9, dtype=tf.float32)

        # Custom erosion using depthwise_conv2d
        eroded_neg = tf.nn.depthwise_conv2d(
            neg_padded + mask_complement * large_neg,
            tf.tile(mask_expanded, [1, 1, 1, 1]),
            strides=[1, 1, 1, 1],
            padding='VALID'
        )

        # Convert back to positive
        eroded = -eroded_neg

        # Implement dilation using similar approach
        # Pad the eroded image
        padded_eroded = tf.pad(
            eroded,
            [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]],
            mode='SYMMETRIC'
        )

        # For dilation, we need to find the maximum value in the neighborhood
        # Apply convolution with the kernel
        opened = tf.nn.depthwise_conv2d(
            padded_eroded,
            tf.tile(mask_expanded, [1, 1, 1, 1]),
            strides=[1, 1, 1, 1],
            padding='VALID'
        )

        # Remove batch and channel dimensions
        opened = tf.squeeze(opened, axis=[0, -1])

        # White top-hat is original minus opening
        tophat_small = image_small - opened

        # 4) Calculate background
        background_small = image_small - tophat_small

        # 5) Upscale background to original size
        background_4d = tf.expand_dims(tf.expand_dims(background_small, 0), -1)
        background_large = tf.squeeze(tf.image.resize(
            background_4d,
            tf.shape(image[z])[:2],
            method=tf.image.ResizeMethod.BILINEAR
        ), axis=[0, -1])

        # 6) Subtract background and clip negative values
        slice_result = tf.maximum(tf.cast(image[z], tf.float32) - background_large, 0.0)

        # 7) Convert back to original data type
        result_list.append(tf.cast(slice_result, input_dtype))

    # Stack results back into a 3D tensor
    result = tf.stack(result_list, axis=0)

    return result
