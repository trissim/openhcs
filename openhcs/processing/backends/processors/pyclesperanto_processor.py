"""
pyclesperanto GPU processor for OpenHCS.

This processor uses pyclesperanto for GPU-accelerated image processing,
providing excellent compatibility with OpenCL devices and seamless integration
with OpenHCS patterns.
"""

import logging
from typing import List, Optional, Union

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

def _gpu_minmax_normalize_range(image: "cle.Array" ) -> tuple:
    """Calculate normalization range using min/max instead of percentiles - pure GPU."""
    import pyclesperanto as cle

    # Use min/max instead of percentiles to stay purely on GPU
    # This is similar to how CLAHE works internally
    min_val = cle.minimum_of_all_pixels(image)
    max_val = cle.maximum_of_all_pixels(image)

    # For compatibility, we could apply a small margin based on percentile values
    # but for pure GPU operation, we use full min/max range
    return float(min_val), float(max_val)

@pyclesperanto_func
def per_slice_minmax_normalize(
    image: "cle.Array",
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    target_min: int = 0,
    target_max: int = 65535
) -> "cle.Array":
    """
    Normalize image intensities using per-slice min/max values - GPU accelerated.

    PER-SLICE OPERATION: Uses min/max values independently for each Z-slice,
    then normalizes each slice to its own min/max range. Each slice gets
    its own normalization parameters. Pure GPU implementation using min/max
    instead of percentiles (pyclesperanto limitation).

    Args:
        image: 3D pyclesperanto Array of shape (Z, Y, X)
        low_percentile: Ignored - kept for API compatibility
        high_percentile: Ignored - kept for API compatibility
        target_min: Minimum value in output range
        target_max: Maximum value in output range

    Returns:
        Normalized 3D pyclesperanto Array of shape (Z, Y, X) with dtype uint16
    """
    # Validate 3D array
    if len(image.shape) != 3:
        raise ValueError(f"Expected 3D array, got {len(image.shape)}D array")

    # Build result by concatenating pairs of slices
    result_slices = []

    for z in range(image.shape[0]):
        # Work directly with slice views - no copying needed
        gpu_slice = image[z]  # Direct slice access

        # Calculate min/max range for normalization (pure GPU)
        p_low, p_high = _gpu_minmax_normalize_range(gpu_slice, low_percentile, high_percentile)

        # Avoid division by zero
        if p_high == p_low:
            # Fill slice with target_min
            gpu_result_slice = cle.create_like(gpu_slice)
            cle.set(gpu_result_slice, target_min)
            result_slices.append(gpu_result_slice)
            continue

        # All normalization operations stay on GPU using pure pyclesperanto
        gpu_clipped = cle.clip(gpu_slice, min_intensity=p_low, max_intensity=p_high)
        gpu_shifted = cle.subtract_image_from_scalar(gpu_clipped, scalar=p_low)

        scale_factor = (target_max - target_min) / (p_high - p_low)
        gpu_normalized = cle.add_image_and_scalar(
            cle.multiply_image_and_scalar(gpu_shifted, scalar=scale_factor),
            scalar=target_min
        )

        result_slices.append(gpu_normalized)

    # Concatenate slices back into 3D array using pairwise concatenation
    result = result_slices[0]
    for i in range(1, len(result_slices)):
        result = cle.concatenate_along_z(result, result_slices[i])

    return result

@pyclesperanto_func
def stack_minmax_normalize(
    image: "cle.Array",
    target_min: int = 0,
    target_max: int = 65535
) -> "cle.Array":
    """
    Normalize image intensities using global stack min/max values - GPU accelerated.

    STACK-WIDE OPERATION: Uses global min/max values from the entire 3D stack
    for normalization. All slices use the same global normalization parameters.
    Pure GPU implementation using min/max instead of percentiles (pyclesperanto limitation).

    Args:
        image: 3D pyclesperanto Array of shape (Z, Y, X)
        low_percentile: Ignored - kept for API compatibility
        high_percentile: Ignored - kept for API compatibility
        target_min: Minimum value in output range
        target_max: Maximum value in output range

    Returns:
        Normalized 3D pyclesperanto Array of shape (Z, Y, X) with dtype uint16
    """
    _check_pyclesperanto_available()

    # Validate 3D array
    if len(image.shape) != 3:
        raise ValueError(f"Expected 3D array, got {len(image.shape)}D array")

    # Calculate global min/max range from entire stack (pure GPU)
    p_low, p_high = _gpu_minmax_normalize_range(image, low_percentile, high_percentile)

    # Avoid division by zero
    if p_high == p_low:
        result = cle.create_like(image)
        cle.set(result, target_min)
        return result

    # All normalization operations stay on GPU using pure pyclesperanto
    gpu_clipped = cle.clip(image, min_intensity=p_low, max_intensity=p_high)
    gpu_shifted = cle.subtract_image_from_scalar(gpu_clipped, scalar=p_low)

    scale_factor = (target_max - target_min) / (p_high - p_low)
    gpu_normalized = cle.add_image_and_scalar(
        cle.multiply_image_and_scalar(gpu_shifted, scalar=scale_factor),
        scalar=target_min
    )

    return gpu_normalized

@pyclesperanto_func
def sharpen(
    image: "cle.Array",
    radius: float = 1.0,
    amount: float = 1.0
) -> "cle.Array":
    """
    Apply unsharp mask sharpening to a 3D image - GPU accelerated.

    PER-SLICE OPERATION: Applies 2D Gaussian blur (sigma_z=0) and unsharp masking
    to each Z-slice independently. Each slice is sharpened using only its own
    2D neighborhood, not considering adjacent Z-slices.

    Args:
        image: 3D pyclesperanto Array of shape (Z, Y, X)
        radius: Gaussian blur radius for unsharp mask (applied in X,Y only)
        amount: Sharpening strength

    Returns:
        Sharpened 3D pyclesperanto Array of shape (Z, Y, X)
    """
    _check_pyclesperanto_available()

    # Validate 3D array
    if len(image.shape) != 3:
        raise ValueError(f"Expected 3D array, got {len(image.shape)}D array")

    # Apply 3D Gaussian blur (pyclesperanto handles Z-dimension efficiently)
    gpu_blurred = cle.gaussian_blur(image, sigma_x=radius, sigma_y=radius, sigma_z=0)

    # Unsharp mask: original + amount * (original - blurred)
    # Use add_images_weighted for efficiency: result = 1*original + amount*(original - blurred)
    gpu_diff = cle.subtract_images(image, gpu_blurred)
    gpu_sharpened = cle.add_images_weighted(image, gpu_diff, factor1=1.0, factor2=amount)

    # Clip to valid range
    gpu_clipped = cle.clip(gpu_sharpened, min_intensity=0, max_intensity=65535)

    return gpu_clipped

@pyclesperanto_func
def max_projection(stack: "cle.Array") -> "cle.Array":
    """
    Create a maximum intensity projection from a Z-stack - GPU accelerated.

    TRUE 3D OPERATION: Collapses the Z-dimension by taking the maximum value
    across all Z-slices for each (Y,X) position. Uses pyclesperanto's
    maximum_z_projection function.

    Args:
        stack: 3D pyclesperanto Array of shape (Z, Y, X)

    Returns:
        3D pyclesperanto Array of shape (1, Y, X)
    """
    _check_pyclesperanto_available()

    # Validate 3D array
    if len(stack.shape) != 3:
        raise ValueError(f"Expected 3D array, got {len(stack.shape)}D array")

    # Create max projection (stays on GPU)
    gpu_projection_2d = cle.maximum_z_projection(stack)

    # Reshape to (1, Y, X) by creating a new 3D array
    result_shape = (1, gpu_projection_2d.shape[0], gpu_projection_2d.shape[1])
    result = cle.create(result_shape, dtype=gpu_projection_2d.dtype)
    result[0] = gpu_projection_2d  # Direct assignment

    return result

@pyclesperanto_func
def mean_projection(stack: "cle.Array") -> "cle.Array":
    """
    Create a mean intensity projection from a Z-stack - GPU accelerated.

    TRUE 3D OPERATION: Collapses the Z-dimension by taking the mean value
    across all Z-slices for each (Y,X) position. Uses pyclesperanto's
    mean_z_projection function.

    Args:
        stack: 3D pyclesperanto Array of shape (Z, Y, X)

    Returns:
        3D pyclesperanto Array of shape (1, Y, X)
    """
    _check_pyclesperanto_available()

    # Validate 3D array
    if len(stack.shape) != 3:
        raise ValueError(f"Expected 3D array, got {len(stack.shape)}D array")

    # Create mean projection (stays on GPU)
    gpu_projection_2d = cle.mean_z_projection(stack)

    # Reshape to (1, Y, X) by creating a new 3D array
    result_shape = (1, gpu_projection_2d.shape[0], gpu_projection_2d.shape[1])
    result = cle.create(result_shape, dtype=gpu_projection_2d.dtype)
    result[0] = gpu_projection_2d  # Direct assignment

    return result

@pyclesperanto_func
def create_projection(
    stack: "cle.Array", method: str = "max_projection"
) -> "cle.Array":
    """
    Create a projection from a stack using the specified method - GPU accelerated.

    TRUE 3D OPERATION: Dispatcher function that calls the appropriate projection
    method. All projection methods collapse the Z-dimension.

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

    PER-SLICE OPERATION: Applies 2D top-hat morphological filtering to each Z-slice
    independently using a sequential loop. Each slice is processed with its own
    2D structuring element, not considering adjacent Z-slices.

    Implementation: Downsamples entire stack, applies 2D top-hat per slice,
    calculates background, upsamples background, then subtracts from original.

    Args:
        image: 3D pyclesperanto Array of shape (Z, Y, X)
        selem_radius: Radius of the 2D structuring element (applied per slice)
        downsample_factor: Factor by which to downsample for processing
        downsample_interpolate: Whether to use interpolation when downsampling
        upsample_interpolate: Whether to use interpolation when upsampling

    Returns:
        Filtered 3D pyclesperanto Array of shape (Z, Y, X)
    """
    _check_pyclesperanto_available()

    # Validate 3D array
    if len(image.shape) != 3:
        raise ValueError(f"Expected 3D array, got {len(image.shape)}D array")

    # 1) Downsample entire stack at once (more efficient)
    scale_factor = 1.0 / downsample_factor
    gpu_small = cle.scale(
        image,
        factor_x=scale_factor,
        factor_y=scale_factor,
        factor_z=1.0,  # Don't scale Z dimension
        resize=True,
        interpolate=downsample_interpolate
    )

    # 2) Apply top-hat filter using sphere structuring element to entire stack
    # Process slice by slice using direct array access
    result_small = cle.create_like(gpu_small)
    for z in range(gpu_small.shape[0]):
        gpu_slice = gpu_small[z]  # Direct slice access
        gpu_tophat_slice = cle.top_hat_sphere(
            gpu_slice,
            radius_x=selem_radius // downsample_factor,
            radius_y=selem_radius // downsample_factor
        )
        result_small[z] = gpu_tophat_slice  # Direct assignment

    # 3) Calculate background on small image
    gpu_background_small = cle.subtract_images(gpu_small, result_small)

    # 4) Upscale background to original size
    gpu_background_large = cle.scale(
        gpu_background_small,
        factor_x=downsample_factor,
        factor_y=downsample_factor,
        factor_z=1.0,  # Don't scale Z dimension
        resize=True,
        interpolate=upsample_interpolate
    )

    # 5) Subtract background and clip negative values (entire stack at once)
    gpu_subtracted = cle.subtract_images(image, gpu_background_large)
    gpu_result = cle.maximum_image_and_scalar(gpu_subtracted, scalar=0)

    return gpu_result

@pyclesperanto_func
def apply_mask(image: "cle.Array", mask: "cle.Array") -> "cle.Array":
    """
    Apply a mask to a 3D image - GPU accelerated.

    HYBRID OPERATION:
    - If 3D mask: TRUE 3D OPERATION (direct element-wise multiplication)
    - If 2D mask: PER-SLICE OPERATION (applies same 2D mask to each Z-slice via sequential loop)

    Args:
        image: 3D pyclesperanto Array of shape (Z, Y, X)
        mask: 3D pyclesperanto Array of shape (Z, Y, X) or 2D pyclesperanto Array of shape (Y, X)

    Returns:
        Masked 3D pyclesperanto Array of shape (Z, Y, X)
    """
    _check_pyclesperanto_available()

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
            # Work directly with slice views
            gpu_slice = image[z]  # Direct slice access
            # Apply mask (both stay on GPU)
            gpu_masked = cle.multiply_images(gpu_slice, mask)
            # Assign result directly
            result[z] = gpu_masked

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
    stack: "cle.Array", weights: Optional[List[float]] = None
) -> "cle.Array":
    """
    Create a composite image from a 3D stack where each slice is a channel - GPU accelerated.

    TRUE 3D OPERATION: Performs element-wise weighted addition across slices
    to create a composite. All mathematical operations are applied using
    efficient pyclesperanto functions.

    Args:
        stack: 3D pyclesperanto Array of shape (N, Y, X) where N is number of channel slices
        weights: List of weights for each slice. If None, equal weights are used.

    Returns:
        Composite 3D pyclesperanto Array of shape (1, Y, X)
    """
    _check_pyclesperanto_available()

    # Validate input is 3D array
    if len(stack.shape) != 3:
        raise ValueError(f"Expected 3D array, got {len(stack.shape)}D array")

    n_slices, height, width = stack.shape

    # Default weights if none provided
    if weights is None:
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

    # Create result array with shape (1, Y, X)
    result = cle.create((1, height, width), dtype=stack.dtype)

    # Initialize with zeros
    cle.set(result, 0.0)

    # Add each weighted slice
    for i, weight in enumerate(normalized_weights):
        if weight > 0.0:
            # Get slice i from the stack
            slice_i = stack[i]  # This gives us a 2D slice

            # Multiply slice by its weight
            weighted_slice = cle.multiply_image_and_scalar(slice_i, scalar=weight)

            # Add to result (need to handle 2D slice + 3D result)
            # Extract the single slice from result for addition
            result_slice = result[0]
            result_slice = cle.add_images(result_slice, weighted_slice)

            # Put it back (this might need adjustment based on pyclesperanto API)
            result[0] = result_slice

    return result

@pyclesperanto_func
def equalize_histogram_3d(
    stack: "cle.Array",
    tile_size: int = 8,
    clip_limit: float = 0.01,
    range_min: float = 0.0,
    range_max: float = 65535.0
) -> "cle.Array":
    """
    Apply 3D CLAHE histogram equalization to a volume - GPU accelerated.

    TRUE 3D OPERATION: Uses 3D CLAHE (Contrast Limited Adaptive Histogram Equalization)
    with 3D tiles (cubes) that consider voxel neighborhoods in X, Y, and Z dimensions.
    Appropriate for Z-stacks where adjacent slices are spatially continuous.

    Args:
        stack: 3D pyclesperanto Array of shape (Z, Y, X)
        tile_size: Size of 3D tiles (cubes) for adaptive equalization
        clip_limit: Clipping limit for histogram equalization (0.0-1.0)
        range_min: Minimum value for output range
        range_max: Maximum value for output range

    Returns:
        Equalized 3D pyclesperanto Array of shape (Z, Y, X)
    """
    _check_pyclesperanto_available()

    # Validate 3D array
    if len(stack.shape) != 3:
        raise ValueError(f"Expected 3D array, got {len(stack.shape)}D array")

    # Use 3D CLAHE with 3D tiles (cubes)
    gpu_equalized = cle.clahe(stack, tile_size=tile_size, clip_limit=clip_limit)

    # Clip to valid range using pure pyclesperanto
    gpu_clipped = cle.clip(gpu_equalized, min_intensity=range_min, max_intensity=range_max)

    return gpu_clipped

@pyclesperanto_func
def equalize_histogram_per_slice(
    stack: "cle.Array",
    tile_size: int = 8,
    clip_limit: float = 0.01,
    range_min: float = 0.0,
    range_max: float = 65535.0
) -> "cle.Array":
    """
    Apply 2D CLAHE histogram equalization to each slice independently - GPU accelerated.

    PER-SLICE OPERATION: Applies 2D CLAHE to each Z-slice independently using 2D tiles.
    Each slice gets its own adaptive histogram equalization. Appropriate for stacks
    of different images (different X,Y content) or when slices should be treated independently.

    Args:
        stack: 3D pyclesperanto Array of shape (Z, Y, X)
        tile_size: Size of 2D tiles (squares) for adaptive equalization per slice
        clip_limit: Clipping limit for histogram equalization (0.0-1.0)
        range_min: Minimum value for output range
        range_max: Maximum value for output range

    Returns:
        Equalized 3D pyclesperanto Array of shape (Z, Y, X)
    """
    _check_pyclesperanto_available()

    # Validate 3D array
    if len(stack.shape) != 3:
        raise ValueError(f"Expected 3D array, got {len(stack.shape)}D array")

    # Create result array
    result = cle.create_like(stack)

    # Apply 2D CLAHE to each slice independently
    for z in range(stack.shape[0]):
        # Work directly with slice views
        gpu_slice = stack[z]  # Direct slice access

        # Apply 2D CLAHE to this slice only
        gpu_equalized_slice = cle.clahe(gpu_slice, tile_size=tile_size, clip_limit=clip_limit)

        # Clip to valid range
        gpu_clipped_slice = cle.clip(gpu_equalized_slice, min_intensity=range_min, max_intensity=range_max)

        # Assign result directly
        result[z] = gpu_clipped_slice

    return result

@pyclesperanto_func
def stack_equalize_histogram(
    stack: "cle.Array",
    bins: int = 65536,
    range_min: float = 0.0,
    range_max: float = 65535.0
) -> "cle.Array":
    """
    Apply histogram equalization to a stack - GPU accelerated.

    COMPATIBILITY FUNCTION: Alias for equalize_histogram_3d to maintain API compatibility
    with numpy processor. Uses 3D CLAHE for true 3D histogram equalization.

    Args:
        stack: 3D pyclesperanto Array of shape (Z, Y, X)
        bins: Number of bins for histogram computation (unused - CLAHE parameter)
        range_min: Minimum value for histogram range
        range_max: Maximum value for histogram range

    Returns:
        Equalized 3D pyclesperanto Array of shape (Z, Y, X)
    """
    # Delegate to the 3D version with default parameters
    return equalize_histogram_3d(stack, range_min=range_min, range_max=range_max)

# API compatibility aliases - these maintain the original function names
# but delegate to the more accurately named implementations

@pyclesperanto_func
def percentile_normalize(
    image: "cle.Array",
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    target_min: int = 0,
    target_max: int = 65535
) -> "cle.Array":
    """
    COMPATIBILITY ALIAS: Delegates to per_slice_minmax_normalize.

    Note: Uses min/max normalization instead of true percentiles due to
    pyclesperanto limitations. Kept for API compatibility with other processors.
    """
    return per_slice_minmax_normalize(image, low_percentile, high_percentile, target_min, target_max)

@pyclesperanto_func
def stack_percentile_normalize(
    image: "cle.Array",
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    target_min: int = 0,
    target_max: int = 65535
) -> "cle.Array":
    """
    COMPATIBILITY ALIAS: Delegates to stack_minmax_normalize.

    Note: Uses min/max normalization instead of true percentiles due to
    pyclesperanto limitations. Kept for API compatibility with other processors.
    """
    return stack_minmax_normalize(image, target_min, target_max)

def create_linear_weight_mask(height: int, width: int, margin_ratio: float = 0.1) -> "cle.Array":
    """
    Create a linear weight mask for blending images - GPU accelerated.

    Pure pyclesperanto implementation using GPU operations only.
    """
    # Create coordinate arrays for X and Y positions
    y_coords = cle.create((height, width), dtype=float)
    x_coords = cle.create((height, width), dtype=float)

    # Fill coordinate arrays
    cle.set_ramp_y(y_coords)
    cle.set_ramp_x(x_coords)

    # Calculate margin sizes
    margin_h = int(height * margin_ratio)
    margin_w = int(width * margin_ratio)

    # Create weight mask starting with ones
    mask = cle.create((height, width), dtype=float)
    cle.set(mask, 1.0)

    # Apply fade from top edge: weight = min(1.0, y / margin_h)
    if margin_h > 0:
        top_weight = cle.multiply_image_and_scalar(y_coords, scalar=1.0/margin_h)
        top_weight = cle.minimum_image_and_scalar(top_weight, scalar=1.0)
        mask = cle.multiply_images(mask, top_weight)

    # Apply fade from bottom edge: weight = min(1.0, (height - 1 - y) / margin_h)
    if margin_h > 0:
        bottom_coords = cle.subtract_image_from_scalar(y_coords, scalar=height - 1)
        bottom_coords = cle.absolute(bottom_coords)
        bottom_weight = cle.multiply_image_and_scalar(bottom_coords, scalar=1.0/margin_h)
        bottom_weight = cle.minimum_image_and_scalar(bottom_weight, scalar=1.0)
        mask = cle.multiply_images(mask, bottom_weight)

    # Apply fade from left edge: weight = min(1.0, x / margin_w)
    if margin_w > 0:
        left_weight = cle.multiply_image_and_scalar(x_coords, scalar=1.0/margin_w)
        left_weight = cle.minimum_image_and_scalar(left_weight, scalar=1.0)
        mask = cle.multiply_images(mask, left_weight)

    # Apply fade from right edge: weight = min(1.0, (width - 1 - x) / margin_w)
    if margin_w > 0:
        right_coords = cle.subtract_image_from_scalar(x_coords, scalar=width - 1)
        right_coords = cle.absolute(right_coords)
        right_weight = cle.multiply_image_and_scalar(right_coords, scalar=1.0/margin_w)
        right_weight = cle.minimum_image_and_scalar(right_weight, scalar=1.0)
        mask = cle.multiply_images(mask, right_weight)

    return mask

def create_weight_mask(shape: tuple, margin_ratio: float = 0.1) -> "cle.Array":
    """
    Create a weight mask for blending images - GPU accelerated.

    Args:
        shape: Shape of the mask (height, width)
        margin_ratio: Ratio of image size to use as margin

    Returns:
        2D pyclesperanto Array of shape (Y, X)
    """
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise TypeError("shape must be a tuple of (height, width)")

    height, width = shape
    return create_linear_weight_mask(height, width, margin_ratio)

