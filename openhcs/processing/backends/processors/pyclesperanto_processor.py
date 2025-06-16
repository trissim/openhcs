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

        # Calculate percentiles (need to pull slice to CPU for percentile calculation)
        # This is a limitation - pyclesperanto doesn't have percentile functions yet
        slice_np = cle.pull(gpu_slice)
        p_low, p_high = np.percentile(slice_np, (low_percentile, high_percentile))

        # Avoid division by zero
        if p_high == p_low:
            # Fill slice with target_min
            cle.set(gpu_slice, target_min)
            cle.copy_slice(gpu_slice, result, z)
            continue

        # Clip and normalize using pyclesperanto operations (all on GPU)
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
    # Need to pull to CPU for percentile calculation (limitation of pyclesperanto)
    image_np = cle.pull(image)
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
    image: np.ndarray,
    radius: float = 1.0,
    amount: float = 1.0
) -> np.ndarray:
    """
    Apply unsharp mask sharpening to a 3D image - GPU accelerated.

    Args:
        image: 3D numpy array of shape (Z, Y, X)
        radius: Gaussian blur radius for unsharp mask
        amount: Sharpening strength

    Returns:
        Sharpened 3D numpy array of shape (Z, Y, X)
    """
    _check_pyclesperanto_available()
    _validate_3d_array(image)

    # Process each Z-slice independently
    result = np.zeros_like(image)

    for z in range(image.shape[0]):
        # Push slice to GPU
        gpu_slice = cle.push(image[z].astype(np.float32))
        
        # Apply Gaussian blur
        gpu_blurred = cle.gaussian_blur(gpu_slice, sigma_x=radius, sigma_y=radius)
        
        # Unsharp mask: original + amount * (original - blurred)
        gpu_diff = cle.subtract_images(gpu_slice, gpu_blurred)
        gpu_scaled_diff = cle.multiply_image_and_scalar(gpu_diff, scalar=amount)
        gpu_sharpened = cle.add_images(gpu_slice, gpu_scaled_diff)
        
        # Clip to valid range and pull back
        gpu_clipped = cle.clip(gpu_sharpened, min_intensity=0, max_intensity=65535)
        result[z] = cle.pull(gpu_clipped).astype(image.dtype)

    return result

@pyclesperanto_func
def max_projection(stack: np.ndarray) -> np.ndarray:
    """
    Create a maximum intensity projection from a Z-stack - GPU accelerated.

    Args:
        stack: 3D numpy array of shape (Z, Y, X)

    Returns:
        3D numpy array of shape (1, Y, X)
    """
    _check_pyclesperanto_available()
    _validate_3d_array(stack)

    # Push to GPU and create max projection
    gpu_stack = cle.push(stack)
    gpu_projection = cle.maximum_z_projection(gpu_stack)
    
    # Pull back and reshape to (1, Y, X)
    projection_2d = cle.pull(gpu_projection)
    return projection_2d.reshape(1, projection_2d.shape[0], projection_2d.shape[1])

@pyclesperanto_func
def mean_projection(stack: np.ndarray) -> np.ndarray:
    """
    Create a mean intensity projection from a Z-stack - GPU accelerated.

    Args:
        stack: 3D numpy array of shape (Z, Y, X)

    Returns:
        3D numpy array of shape (1, Y, X)
    """
    _check_pyclesperanto_available()
    _validate_3d_array(stack)

    # Push to GPU and create mean projection
    gpu_stack = cle.push(stack)
    gpu_projection = cle.mean_z_projection(gpu_stack)
    
    # Pull back and reshape to (1, Y, X)
    projection_2d = cle.pull(gpu_projection)
    return projection_2d.reshape(1, projection_2d.shape[0], projection_2d.shape[1])

@pyclesperanto_func
def create_projection(
    stack: np.ndarray, method: str = "max_projection"
) -> np.ndarray:
    """
    Create a projection from a stack using the specified method - GPU accelerated.

    Args:
        stack: 3D numpy array of shape (Z, Y, X)
        method: Projection method (max_projection, mean_projection)

    Returns:
        3D numpy array of shape (1, Y, X)
    """
    _check_pyclesperanto_available()
    _validate_3d_array(stack)

    if method == "max_projection":
        return max_projection(stack)

    if method == "mean_projection":
        return mean_projection(stack)

    # FAIL FAST: No fallback projection methods
    raise ValueError(f"Unknown projection method: {method}. Valid methods: max_projection, mean_projection")

@pyclesperanto_func
def tophat(
    image: np.ndarray,
    selem_radius: int = 50,
    downsample_factor: int = 4,
    downsample_interpolate: bool = True,
    upsample_interpolate: bool = False
) -> np.ndarray:
    """
    Apply white top-hat filter to a 3D image for background removal - GPU accelerated.

    Args:
        image: 3D numpy array of shape (Z, Y, X)
        selem_radius: Radius of the structuring element
        downsample_factor: Factor by which to downsample for processing
        downsample_interpolate: Whether to use interpolation when downsampling
        upsample_interpolate: Whether to use interpolation when upsampling

    Returns:
        Filtered 3D numpy array of shape (Z, Y, X)
    """
    _check_pyclesperanto_available()
    _validate_3d_array(image)

    # Import pyclesperanto
    import pyclesperanto as cle

    # Process each Z-slice independently
    result = np.zeros_like(image)

    for z in range(image.shape[0]):
        # Store original data type
        input_dtype = image[z].dtype

        # Push slice to GPU
        gpu_slice = cle.push(image[z].astype(np.float32))

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
        gpu_result = cle.maximum_image_and_scalar(gpu_subtracted, scalar=0)

        # 6) Pull back and convert to original data type
        slice_result = cle.pull(gpu_result)
        result[z] = slice_result.astype(input_dtype)

    return result

@pyclesperanto_func
def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a mask to a 3D image - GPU accelerated.

    Args:
        image: 3D numpy array of shape (Z, Y, X)
        mask: 3D numpy array of shape (Z, Y, X) or 2D numpy array of shape (Y, X)

    Returns:
        Masked 3D numpy array of shape (Z, Y, X)
    """
    _check_pyclesperanto_available()
    _validate_3d_array(image)

    # Handle 2D mask (apply to each Z-slice)
    if mask.ndim == 2:
        if mask.shape != image.shape[1:]:
            raise ValueError(
                f"2D mask shape {mask.shape} doesn't match image slice shape {image.shape[1:]}"
            )

        # Push mask to GPU once
        gpu_mask = cle.push(mask.astype(np.float32))
        result = np.zeros_like(image)

        for z in range(image.shape[0]):
            gpu_slice = cle.push(image[z].astype(np.float32))
            gpu_masked = cle.multiply_images(gpu_slice, gpu_mask)
            result[z] = cle.pull(gpu_masked).astype(image.dtype)

        return result

    # Handle 3D mask
    if mask.ndim == 3:
        if mask.shape != image.shape:
            raise ValueError(
                f"3D mask shape {mask.shape} doesn't match image shape {image.shape}"
            )

        # Push both to GPU and apply mask
        gpu_image = cle.push(image.astype(np.float32))
        gpu_mask = cle.push(mask.astype(np.float32))
        gpu_masked = cle.multiply_images(gpu_image, gpu_mask)

        return cle.pull(gpu_masked).astype(image.dtype)

    # If we get here, the mask is neither 2D nor 3D
    raise TypeError(f"mask must be a 2D or 3D numpy array, got {type(mask)}")

@pyclesperanto_func
def create_composite(
    images: List[np.ndarray], weights: Optional[List[float]] = None
) -> np.ndarray:
    """
    Create a composite image from multiple 3D arrays - GPU accelerated.

    Args:
        images: List of 3D numpy arrays, each of shape (Z, Y, X)
        weights: List of weights for each image. If None, equal weights are used.

    Returns:
        Composite 3D numpy array of shape (Z, Y, X)
    """
    _check_pyclesperanto_available()

    # Validate inputs
    if not isinstance(images, list):
        raise TypeError("images must be a list of numpy arrays")

    if not images:
        raise ValueError("images list cannot be empty")

    # Validate all images are 3D with the same shape
    for i, img in enumerate(images):
        _validate_3d_array(img)
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

    # Push first image to GPU and initialize composite
    first_image = images[0]
    gpu_composite = cle.push(first_image.astype(np.float32))
    gpu_composite = cle.multiply_image_and_scalar(gpu_composite, scalar=weights[0])

    total_weight = weights[0] if weights[0] > 0.0 else 0.0

    # Add remaining images with their weights
    for i in range(1, len(images)):
        weight = weights[i]
        if weight <= 0.0:
            continue

        gpu_image = cle.push(images[i].astype(np.float32))
        gpu_weighted = cle.multiply_image_and_scalar(gpu_image, scalar=weight)
        gpu_composite = cle.add_images(gpu_composite, gpu_weighted)
        total_weight += weight

    # Normalize by total weight
    if total_weight > 0:
        gpu_composite = cle.multiply_image_and_scalar(gpu_composite, scalar=1.0/total_weight)

    # Pull back and convert to original dtype
    result = cle.pull(gpu_composite)

    # Clip to valid range for integer types
    if np.issubdtype(first_image.dtype, np.integer):
        max_val = np.iinfo(first_image.dtype).max
        result = np.clip(result, 0, max_val)

    return result.astype(first_image.dtype)

@pyclesperanto_func
def stack_equalize_histogram(
    stack: np.ndarray,
    bins: int = 65536,
    range_min: float = 0.0,
    range_max: float = 65535.0
) -> np.ndarray:
    """
    Apply histogram equalization to an entire stack - GPU accelerated.

    Args:
        stack: 3D numpy array of shape (Z, Y, X)
        bins: Number of bins for histogram computation
        range_min: Minimum value for histogram range
        range_max: Maximum value for histogram range

    Returns:
        Equalized 3D numpy array of shape (Z, Y, X)
    """
    _check_pyclesperanto_available()
    _validate_3d_array(stack)

    # Push stack to GPU
    gpu_stack = cle.push(stack.astype(np.float32))

    # pyclesperanto has CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # which is more advanced than basic histogram equalization
    gpu_equalized = cle.clahe(gpu_stack, block_size_x=64, block_size_y=64, clip_limit=3.0)

    # Pull back and convert to uint16
    result = cle.pull(gpu_equalized)
    return np.clip(result, range_min, range_max).astype(np.uint16)

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
