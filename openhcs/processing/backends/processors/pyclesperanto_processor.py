"""
pyclesperanto GPU processor for OpenHCS.

This processor uses pyclesperanto for GPU-accelerated image processing,
providing excellent compatibility with OpenCL devices and seamless integration
with OpenHCS patterns.
"""

import logging
from typing import List, Optional, Union
import numpy as np

# Import OpenHCS decorator
from openhcs.core.memory.decorators import pyclesperanto as pyclesperanto_func

# Set up logging
logger = logging.getLogger(__name__)

# Try to import pyclesperanto
try:
    import pyclesperanto as cle
    PYCLESPERANTO_AVAILABLE = True
    logger.info("pyclesperanto available - GPU acceleration enabled")
except ImportError:
    PYCLESPERANTO_AVAILABLE = False
    logger.warning("pyclesperanto not available - install with: pip install pyclesperanto")

def _check_pyclesperanto_available():
    """Check if pyclesperanto is available and raise error if not."""
    if not PYCLESPERANTO_AVAILABLE:
        raise ImportError("pyclesperanto is required but not available. Install with: pip install pyclesperanto")

def _validate_3d_array(array) -> None:
    """Validate that the input is a 3D array."""
    if array.ndim != 3:
        raise ValueError(f"Expected 3D array, got {array.ndim}D array")

@pyclesperanto_func
def percentile_normalize(
    image: "cle.Array",
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    target_min: int = 0,
    target_max: int = 65535
) -> "cle.Array":
    """
    Normalize image intensities using percentile clipping - GPU accelerated.

    Args:
        image: 3D pyclesperanto Array of shape (Z, Y, X)
        low_percentile: Lower percentile for clipping
        high_percentile: Higher percentile for clipping
        target_min: Minimum value in output range
        target_max: Maximum value in output range

    Returns:
        Normalized 3D pyclesperanto Array of shape (Z, Y, X) with dtype uint16
    """
    _check_pyclesperanto_available()

    # Import pyclesperanto
    import pyclesperanto as cle

    # Validate 3D array (check shape directly on GPU array)
    if len(image.shape) != 3:
        raise ValueError(f"Expected 3D array, got {len(image.shape)}D array")

    # Create result array on GPU
    result = cle.create_like(image)

    for z in range(image.shape[0]):
        # Extract Z-slice (stays on GPU)
        gpu_slice = cle.copy_slice(image, z)

        # Calculate percentiles - LIMITATION: pyclesperanto doesn't have percentile functions
        # We need to pull to CPU for this calculation only
        slice_np = cle.pull(gpu_slice)
        import numpy as np
        p_low, p_high = np.percentile(slice_np, (low_percentile, high_percentile))

        # Avoid division by zero
        if p_high == p_low:
            # Fill slice with target_min (stays on GPU)
            cle.set(gpu_slice, target_min)
            cle.copy_slice(gpu_slice, result, z)
            continue

        # All operations stay on GPU from here
        # Clip to [p_low, p_high]
        gpu_clipped = cle.clip(gpu_slice, min_intensity=p_low, max_intensity=p_high)

        # Normalize: (clipped - p_low) * (target_max - target_min) / (p_high - p_low) + target_min
        # Step 1: subtract p_low
        gpu_shifted = cle.subtract_image_from_scalar(gpu_clipped, scalar=p_low)

        # Step 2: scale by (target_max - target_min) / (p_high - p_low)
        scale_factor = (target_max - target_min) / (p_high - p_low)
        gpu_scaled = cle.multiply_image_and_scalar(gpu_shifted, scalar=scale_factor)

        # Step 3: add target_min
        gpu_normalized = cle.add_image_and_scalar(gpu_scaled, scalar=target_min)

        # Copy normalized slice back to result (stays on GPU)
        cle.copy_slice(gpu_normalized, result, z)

    return result

@pyclesperanto_func
def stack_percentile_normalize(
    image: "cle.Array",
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    target_min: int = 0,
    target_max: int = 65535
) -> "cle.Array":
    """
    Normalize image intensities using global stack percentiles - GPU accelerated.

    Args:
        image: 3D pyclesperanto Array of shape (Z, Y, X)
        low_percentile: Lower percentile for clipping
        high_percentile: Higher percentile for clipping
        target_min: Minimum value in output range
        target_max: Maximum value in output range

    Returns:
        Normalized 3D pyclesperanto Array of shape (Z, Y, X) with dtype uint16
    """
    _check_pyclesperanto_available()

    # Import pyclesperanto
    import pyclesperanto as cle

    # Validate 3D array
    if len(image.shape) != 3:
        raise ValueError(f"Expected 3D array, got {len(image.shape)}D array")

    # Calculate global percentiles from entire stack
    # LIMITATION: pyclesperanto doesn't have percentile functions
    # We need to pull to CPU for this calculation only
    image_np = cle.pull(image)
    import numpy as np
    p_low, p_high = np.percentile(image_np, (low_percentile, high_percentile))

    # Avoid division by zero
    if p_high == p_low:
        result = cle.create_like(image)
        cle.set(result, target_min)
        return result

    # All operations stay on GPU from here
    # Clip to [p_low, p_high]
    gpu_clipped = cle.clip(image, min_intensity=p_low, max_intensity=p_high)

    # Normalize: (clipped - p_low) * (target_max - target_min) / (p_high - p_low) + target_min
    # Step 1: subtract p_low
    gpu_shifted = cle.subtract_image_from_scalar(gpu_clipped, scalar=p_low)

    # Step 2: scale by (target_max - target_min) / (p_high - p_low)
    scale_factor = (target_max - target_min) / (p_high - p_low)
    gpu_scaled = cle.multiply_image_and_scalar(gpu_shifted, scalar=scale_factor)

    # Step 3: add target_min
    gpu_normalized = cle.add_image_and_scalar(gpu_scaled, scalar=target_min)

    return gpu_normalized

@pyclesperanto_func
def sharpen(
    image: "cle.Array",
    radius: float = 1.0,
    amount: float = 1.0
) -> "cle.Array":
    """
    Apply unsharp mask sharpening to a 3D image - GPU accelerated.

    Args:
        image: 3D pyclesperanto Array of shape (Z, Y, X)
        radius: Gaussian blur radius for unsharp mask
        amount: Sharpening strength

    Returns:
        Sharpened 3D pyclesperanto Array of shape (Z, Y, X)
    """
    _check_pyclesperanto_available()

    # Import pyclesperanto
    import pyclesperanto as cle

    # Validate 3D array
    if len(image.shape) != 3:
        raise ValueError(f"Expected 3D array, got {len(image.shape)}D array")

    # Create result array on GPU
    result = cle.create_like(image)

    for z in range(image.shape[0]):
        # Extract Z-slice (stays on GPU)
        gpu_slice = cle.copy_slice(image, z)

        # Apply Gaussian blur
        gpu_blurred = cle.gaussian_blur(gpu_slice, sigma_x=radius, sigma_y=radius)

        # Unsharp mask: original + amount * (original - blurred)
        gpu_diff = cle.subtract_images(gpu_slice, gpu_blurred)
        gpu_scaled_diff = cle.multiply_image_and_scalar(gpu_diff, scalar=amount)
        gpu_sharpened = cle.add_images(gpu_slice, gpu_scaled_diff)

        # Clip to valid range
        gpu_clipped = cle.clip(gpu_sharpened, min_intensity=0, max_intensity=65535)

        # Copy result slice back to result (stays on GPU)
        cle.copy_slice(gpu_clipped, result, z)

    return result

@pyclesperanto_func
def max_projection(stack: "cle.Array") -> "cle.Array":
    """
    Create a maximum intensity projection from a Z-stack - GPU accelerated.

    Args:
        stack: 3D pyclesperanto Array of shape (Z, Y, X)

    Returns:
        3D pyclesperanto Array of shape (1, Y, X)
    """
    _check_pyclesperanto_available()

    # Import pyclesperanto
    import pyclesperanto as cle

    # Validate 3D array
    if len(stack.shape) != 3:
        raise ValueError(f"Expected 3D array, got {len(stack.shape)}D array")

    # Create max projection (stays on GPU)
    gpu_projection_2d = cle.maximum_z_projection(stack)

    # Reshape to (1, Y, X) by creating a new 3D array
    result_shape = (1, gpu_projection_2d.shape[0], gpu_projection_2d.shape[1])
    result = cle.create(result_shape, dtype=gpu_projection_2d.dtype)
    cle.copy_slice(gpu_projection_2d, result, 0)

    return result

@pyclesperanto_func
def mean_projection(stack: "cle.Array") -> "cle.Array":
    """
    Create a mean intensity projection from a Z-stack - GPU accelerated.

    Args:
        stack: 3D pyclesperanto Array of shape (Z, Y, X)

    Returns:
        3D pyclesperanto Array of shape (1, Y, X)
    """
    _check_pyclesperanto_available()

    # Import pyclesperanto
    import pyclesperanto as cle

    # Validate 3D array
    if len(stack.shape) != 3:
        raise ValueError(f"Expected 3D array, got {len(stack.shape)}D array")

    # Create mean projection (stays on GPU)
    gpu_projection_2d = cle.mean_z_projection(stack)

    # Reshape to (1, Y, X) by creating a new 3D array
    result_shape = (1, gpu_projection_2d.shape[0], gpu_projection_2d.shape[1])
    result = cle.create(result_shape, dtype=gpu_projection_2d.dtype)
    cle.copy_slice(gpu_projection_2d, result, 0)

    return result

@pyclesperanto_func
def create_projection(
    stack: "cle.Array", method: str = "max_projection"
) -> "cle.Array":
    """
    Create a projection from a stack using the specified method - GPU accelerated.

    Args:
        stack: 3D pyclesperanto Array of shape (Z, Y, X)
        method: Projection method (max_projection, mean_projection)

    Returns:
        3D pyclesperanto Array of shape (1, Y, X)
    """
    _check_pyclesperanto_available()

    # Validate 3D array
    if len(stack.shape) != 3:
        raise ValueError(f"Expected 3D array, got {len(stack.shape)}D array")

    if method == "max_projection":
        return max_projection(stack)

    if method == "mean_projection":
        return mean_projection(stack)

    # FAIL FAST: No fallback projection methods
    raise ValueError(f"Unknown projection method: {method}. Valid methods: max_projection, mean_projection")

@pyclesperanto_func
def tophat(
    image: "cle.Array",
    selem_radius: int = 50,
    downsample_factor: int = 4,
    downsample_interpolate: bool = True,
    upsample_interpolate: bool = False
) -> "cle.Array":
    """
    Apply white top-hat filter to a 3D image for background removal - GPU accelerated.

    Args:
        image: 3D pyclesperanto Array of shape (Z, Y, X)
        selem_radius: Radius of the structuring element
        downsample_factor: Factor by which to downsample for processing
        downsample_interpolate: Whether to use interpolation when downsampling
        upsample_interpolate: Whether to use interpolation when upsampling

    Returns:
        Filtered 3D pyclesperanto Array of shape (Z, Y, X)
    """
    _check_pyclesperanto_available()

    # Import pyclesperanto
    import pyclesperanto as cle

    # Validate 3D array (check shape directly on GPU array)
    if len(image.shape) != 3:
        raise ValueError(f"Expected 3D array, got {len(image.shape)}D array")

    # Create result array on GPU
    result = cle.create_like(image)

    for z in range(image.shape[0]):
        # Extract Z-slice (stays on GPU)
        gpu_slice = cle.copy_slice(image, z)

        # 1) Downsample
        scale_factor = 1.0 / downsample_factor
        gpu_small = cle.scale(
            gpu_slice,
            factor_x=scale_factor,
            factor_y=scale_factor,
            resize=True,
            interpolate=downsample_interpolate
        )

        # 2) Apply top-hat filter using sphere structuring element
        gpu_tophat_small = cle.top_hat_sphere(
            gpu_small,
            radius_x=selem_radius // downsample_factor,
            radius_y=selem_radius // downsample_factor
        )

        # 3) Calculate background on small image
        gpu_background_small = cle.subtract_images(gpu_small, gpu_tophat_small)

        # 4) Upscale background to original size
        gpu_background_large = cle.scale(
            gpu_background_small,
            factor_x=downsample_factor,
            factor_y=downsample_factor,
            resize=True,
            interpolate=upsample_interpolate
        )

        # 5) Subtract background and clip negative values
        gpu_subtracted = cle.subtract_images(gpu_slice, gpu_background_large)
        gpu_result_slice = cle.maximum_image_and_scalar(gpu_subtracted, scalar=0)

        # 6) Copy result slice back to result (stays on GPU)
        cle.copy_slice(gpu_result_slice, result, z)

    return result

@pyclesperanto_func
def apply_mask(image: "cle.Array", mask: "cle.Array") -> "cle.Array":
    """
    Apply a mask to a 3D image - GPU accelerated.

    Args:
        image: 3D pyclesperanto Array of shape (Z, Y, X)
        mask: 3D pyclesperanto Array of shape (Z, Y, X) or 2D pyclesperanto Array of shape (Y, X)

    Returns:
        Masked 3D pyclesperanto Array of shape (Z, Y, X)
    """
    _check_pyclesperanto_available()

    # Import pyclesperanto
    import pyclesperanto as cle

    # Validate 3D image
    if len(image.shape) != 3:
        raise ValueError(f"Expected 3D image array, got {len(image.shape)}D array")

    # Handle 2D mask (apply to each Z-slice)
    if len(mask.shape) == 2:
        if mask.shape != image.shape[1:]:
            raise ValueError(
                f"2D mask shape {mask.shape} doesn't match image slice shape {image.shape[1:]}"
            )

        # Create result array on GPU
        result = cle.create_like(image)

        for z in range(image.shape[0]):
            # Extract Z-slice (stays on GPU)
            gpu_slice = cle.copy_slice(image, z)
            # Apply mask (both stay on GPU)
            gpu_masked = cle.multiply_images(gpu_slice, mask)
            # Copy result back (stays on GPU)
            cle.copy_slice(gpu_masked, result, z)

        return result

    # Handle 3D mask
    elif len(mask.shape) == 3:
        if mask.shape != image.shape:
            raise ValueError(
                f"3D mask shape {mask.shape} doesn't match image shape {image.shape}"
            )

        # Apply mask directly (both stay on GPU)
        return cle.multiply_images(image, mask)

    # If we get here, the mask is neither 2D nor 3D
    else:
        raise TypeError(f"mask must be a 2D or 3D pyclesperanto Array, got shape {mask.shape}")

@pyclesperanto_func
def create_composite(
    images: List["cle.Array"], weights: Optional[List[float]] = None
) -> "cle.Array":
    """
    Create a composite image from multiple 3D arrays - GPU accelerated.

    Args:
        images: List of 3D pyclesperanto Arrays, each of shape (Z, Y, X)
        weights: List of weights for each image. If None, equal weights are used.

    Returns:
        Composite 3D pyclesperanto Array of shape (Z, Y, X)
    """
    _check_pyclesperanto_available()

    # Import pyclesperanto
    import pyclesperanto as cle

    # Validate inputs
    if not isinstance(images, list):
        raise TypeError("images must be a list of pyclesperanto Arrays")

    if not images:
        raise ValueError("images list cannot be empty")

    # Validate all images are 3D with the same shape
    for i, img in enumerate(images):
        if len(img.shape) != 3:
            raise ValueError(f"Expected 3D array, got {len(img.shape)}D array for images[{i}]")
        if img.shape != images[0].shape:
            raise ValueError(f"All images must have the same shape. "
                            f"images[0] has shape {images[0].shape}, "
                            f"images[{i}] has shape {img.shape}")

    # Default weights if none provided
    if weights is None:
        weights = [1.0 / len(images)] * len(images)
    elif not isinstance(weights, list):
        raise TypeError("weights must be a list of values")

    # Ensure weights list matches images list
    if len(weights) < len(images):
        weights = weights + [0.0] * (len(images) - len(weights))
    weights = weights[:len(images)]

    # Initialize composite with first image (stays on GPU)
    first_image = images[0]
    gpu_composite = cle.multiply_image_and_scalar(first_image, scalar=weights[0])

    total_weight = weights[0] if weights[0] > 0.0 else 0.0

    # Add remaining images with their weights (all stay on GPU)
    for i in range(1, len(images)):
        weight = weights[i]
        if weight <= 0.0:
            continue

        gpu_weighted = cle.multiply_image_and_scalar(images[i], scalar=weight)
        gpu_composite = cle.add_images(gpu_composite, gpu_weighted)
        total_weight += weight

    # Normalize by total weight (stays on GPU)
    if total_weight > 0:
        gpu_composite = cle.multiply_image_and_scalar(gpu_composite, scalar=1.0/total_weight)

    # Clip to valid range (stays on GPU)
    # Assume uint16 range for now - pyclesperanto doesn't have dtype introspection
    gpu_clipped = cle.clip(gpu_composite, min_intensity=0, max_intensity=65535)

    return gpu_clipped

@pyclesperanto_func
def stack_equalize_histogram(
    stack: "cle.Array",
    bins: int = 65536,
    range_min: float = 0.0,
    range_max: float = 65535.0
) -> "cle.Array":
    """
    Apply histogram equalization to an entire stack - GPU accelerated.

    Args:
        stack: 3D pyclesperanto Array of shape (Z, Y, X)
        bins: Number of bins for histogram computation
        range_min: Minimum value for histogram range
        range_max: Maximum value for histogram range

    Returns:
        Equalized 3D pyclesperanto Array of shape (Z, Y, X)
    """
    _check_pyclesperanto_available()

    # Import pyclesperanto
    import pyclesperanto as cle

    # Validate 3D array
    if len(stack.shape) != 3:
        raise ValueError(f"Expected 3D array, got {len(stack.shape)}D array")

    # pyclesperanto has CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # which is more advanced than basic histogram equalization
    gpu_equalized = cle.clahe(stack, block_size_x=64, block_size_y=64, clip_limit=3.0)

    # Clip to valid range (stays on GPU)
    gpu_clipped = cle.clip(gpu_equalized, min_intensity=range_min, max_intensity=range_max)

    return gpu_clipped

def create_linear_weight_mask(height: int, width: int, margin_ratio: float = 0.1) -> np.ndarray:
    """
    Create a linear weight mask for blending images.

    This is a CPU-only helper function since it's typically called once.
    """
    # Calculate margin size
    margin_h = int(height * margin_ratio)
    margin_w = int(width * margin_ratio)

    # Create weight mask
    mask = np.ones((height, width), dtype=np.float32)

    # Apply linear fade at edges
    for i in range(margin_h):
        weight = i / margin_h
        mask[i, :] *= weight
        mask[-(i+1), :] *= weight

    for j in range(margin_w):
        weight = j / margin_w
        mask[:, j] *= weight
        mask[:, -(j+1)] *= weight

    return mask

def create_weight_mask(shape: tuple, margin_ratio: float = 0.1) -> np.ndarray:
    """
    Create a weight mask for blending images.

    Args:
        shape: Shape of the mask (height, width)
        margin_ratio: Ratio of image size to use as margin

    Returns:
        2D numpy weight mask of shape (Y, X)
    """
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise TypeError("shape must be a tuple of (height, width)")

    height, width = shape
    return create_linear_weight_mask(height, width, margin_ratio)
