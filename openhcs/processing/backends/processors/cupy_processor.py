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

    # Process each Z-slice independently - MATCH NUMPY EXACTLY
    result = cp.zeros_like(image, dtype=cp.float32)  # Use float32 like NumPy

    for z in range(image.shape[0]):
        # Get percentile values for this slice - MATCH NUMPY EXACTLY
        p_low, p_high = cp.percentile(image[z], (low_percentile, high_percentile))

        # Avoid division by zero - MATCH NUMPY EXACTLY
        if p_high == p_low:
            result[z] = cp.ones_like(image[z]) * target_min
            continue

        # Clip and normalize to target range - MATCH NUMPY EXACTLY
        clipped = cp.clip(image[z], p_low, p_high)
        normalized = (clipped - p_low) * (target_max - target_min) / (p_high - p_low) + target_min
        result[z] = normalized

    # Convert to uint16 - MATCH NUMPY EXACTLY
    return result.astype(cp.uint16)

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

    # Add each image with its weight - MATCH NUMPY EXACTLY
    for i, image in enumerate(images):
        weight = weights[i]
        if weight <= 0.0:
            continue

        # Add to composite - MATCH NUMPY EXACTLY
        composite += image.astype(cp.float32) * weight
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

        # Apply 2D mask to each Z-slice - MATCH NUMPY EXACTLY
        result = cp.zeros_like(image)
        for z in range(image.shape[0]):
            result[z] = image[z].astype(cp.float32) * mask.astype(cp.float32)

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

    # MATCH NUMPY EXACTLY - Flatten the entire stack to compute the global histogram
    flat_stack = stack.flatten()

    # Calculate the histogram and cumulative distribution function (CDF) - MATCH NUMPY EXACTLY
    hist, bin_edges = cp.histogram(flat_stack, bins=bins, range=(range_min, range_max))
    cdf = hist.cumsum()

    # Normalize the CDF to the range [0, 65535] - MATCH NUMPY EXACTLY
    # Avoid division by zero
    if cdf[-1] > 0:
        cdf = 65535 * cdf / cdf[-1]

    # Use linear interpolation to map input values to equalized values - MATCH NUMPY EXACTLY
    equalized_stack = cp.interp(stack.flatten(), bin_edges[:-1], cdf).reshape(stack.shape)

    # Convert to uint16 - MATCH NUMPY EXACTLY
    return equalized_stack.astype(cp.uint16)

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

    This applies the filter to each Z-slice independently - IMPROVED MATCH TO NUMPY.

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

#@cupy_func
#def clahe_2d(
#    image: "cp.ndarray",
#    clip_limit: float = 2.0,
#    tile_grid_size: tuple = None,
#    nbins: int = None,
#    adaptive_bins: bool = True,
#    adaptive_tiles: bool = True
#) -> "cp.ndarray":
#    """
#    Fixed version of 2D CLAHE with proper bilinear interpolation.
#    """
#    _validate_3d_array(image)
#    
#    result = cp.zeros_like(image)
#    
#    for z in range(image.shape[0]):
#        slice_2d = image[z]
#        height, width = slice_2d.shape
#        
#        # Adaptive parameters (same as before)
#        if nbins is None:
#            if adaptive_bins:
#                data_range = cp.max(slice_2d) - cp.min(slice_2d)
#                adaptive_nbins = min(512, max(64, int(cp.sqrt(data_range))))
#            else:
#                adaptive_nbins = 256
#        else:
#            adaptive_nbins = nbins
#            
#        if tile_grid_size is None:
#            if adaptive_tiles:
#                target_tile_size = 80
#                adaptive_tile_rows = max(2, min(16, height // target_tile_size))
#                adaptive_tile_cols = max(2, min(16, width // target_tile_size))
#                adaptive_tile_grid = (adaptive_tile_rows, adaptive_tile_cols)
#            else:
#                adaptive_tile_grid = (8, 8)
#        else:
#            adaptive_tile_grid = tile_grid_size
#            
#        result[z] = _clahe_2d(
#            slice_2d, clip_limit, adaptive_tile_grid, adaptive_nbins
#        )
#    
#    return result
#
#def _clahe_2d(
#    image: "cp.ndarray",
#    clip_limit: float,
#    tile_grid_size: tuple,
#    nbins: int
#) -> "cp.ndarray":
#    """
#    Fixed CLAHE implementation with proper pixel-level bilinear interpolation.
#    """
#    if image.ndim != 2:
#        raise ValueError("Input must be 2D array")
#    
#    height, width = image.shape
#    tile_rows, tile_cols = tile_grid_size
#    
#    # Calculate tile dimensions
#    tile_height = height // tile_rows
#    tile_width = width // tile_cols
#    
#    # Ensure image dimensions are divisible by tile dimensions
#    crop_height = tile_height * tile_rows
#    crop_width = tile_width * tile_cols
#    image_crop = image[:crop_height, :crop_width]
#    
#    # Calculate actual clip limit based on tile size
#    actual_clip_limit = max(1, int(clip_limit * tile_height * tile_width / nbins))
#    
#    # Determine value range for histogram binning
#    min_val = cp.min(image)
#    max_val = cp.max(image)
#    value_range = (float(min_val), float(max_val))
#    
#    # Calculate transformation functions (CDFs) for each tile center
#    tile_cdfs = cp.zeros((tile_rows, tile_cols, nbins), dtype=cp.float32)
#    bin_edges = cp.linspace(min_val, max_val, nbins + 1)
#    
#    for row in range(tile_rows):
#        for col in range(tile_cols):
#            # Extract tile
#            y_start = row * tile_height
#            y_end = (row + 1) * tile_height
#            x_start = col * tile_width  
#            x_end = (col + 1) * tile_width
#            
#            tile = image_crop[y_start:y_end, x_start:x_end]
#            
#            # Compute histogram
#            if max_val > min_val:
#                hist, _ = cp.histogram(tile, bins=nbins, range=value_range)
#            else:
#                hist = cp.zeros(nbins, dtype=cp.int32)
#                hist[nbins // 2] = tile.size
#            
#            # Clip and redistribute histogram
#            hist = _clip_histogram(hist, actual_clip_limit)
#            
#            # Calculate CDF (this becomes the transformation function)
#            cdf = cp.cumsum(hist).astype(cp.float32)
#            if cdf[-1] > 0:
#                # Normalize CDF to output range
#                cdf = (cdf / cdf[-1]) * (max_val - min_val) + min_val
#            else:
#                cdf = cp.full_like(cdf, min_val)
#            
#            tile_cdfs[row, col, :] = cdf
#    
#    # Apply pixel-level bilinear interpolation
#    result = _apply_pixel_level_interpolation(
#        image_crop, tile_cdfs, tile_rows, tile_cols, 
#        tile_height, tile_width, bin_edges, min_val, max_val
#    )
#    
#    # Handle original image size if needed
#    if result.shape != image.shape:
#        full_result = cp.zeros_like(image, dtype=result.dtype)
#        full_result[:crop_height, :crop_width] = result
#        # Fill remaining areas
#        if crop_height < height:
#            full_result[crop_height:, :crop_width] = result[-1:, :]
#        if crop_width < width:
#            full_result[:crop_height, crop_width:] = result[:, -1:]
#        if crop_height < height and crop_width < width:
#            full_result[crop_height:, crop_width:] = result[-1, -1]
#        result = full_result
#    
#    return result.astype(image.dtype)
#
#
#@cupy_func
#def clahe_3d(
#    stack: "cp.ndarray",
#    clip_limit: float = 2.0,
#    tile_grid_size_3d: tuple = None,
#    nbins: int = None,
#    adaptive_bins: bool = True,
#    adaptive_tiles: bool = True
#) -> "cp.ndarray":
#    """
#    Fixed version of true 3D CLAHE with proper trilinear interpolation.
#    """
#    _validate_3d_array(stack)
#    
#    depth, height, width = stack.shape
#    
#    # Adaptive parameters (same as before)
#    if nbins is None:
#        if adaptive_bins:
#            data_range = cp.max(stack) - cp.min(stack)
#            adaptive_nbins = min(512, max(128, int(cp.cbrt(data_range * 64))))
#        else:
#            adaptive_nbins = 256
#    else:
#        adaptive_nbins = nbins
#        
#    if tile_grid_size_3d is None:
#        if adaptive_tiles:
#            target_tile_size = 48
#            adaptive_z_tiles = max(1, min(depth // 4, depth // target_tile_size))
#            adaptive_y_tiles = max(2, min(8, height // target_tile_size))
#            adaptive_x_tiles = max(2, min(8, width // target_tile_size))
#            adaptive_tile_grid_3d = (adaptive_z_tiles, adaptive_y_tiles, adaptive_x_tiles)
#        else:
#            adaptive_tile_grid_3d = (max(1, depth // 8), 4, 4)
#    else:
#        adaptive_tile_grid_3d = tile_grid_size_3d
#    
#    return _clahe_3d(stack, clip_limit, adaptive_tile_grid_3d, adaptive_nbins)
#
#@cupy_func
#def _clahe_3d(
#    stack: "cp.ndarray",
#    clip_limit: float,
#    tile_grid_size_3d: tuple,
#    nbins: int
#) -> "cp.ndarray":
#    """
#    Fixed 3D CLAHE implementation with proper voxel-level trilinear interpolation.
#    """
#    depth, height, width = stack.shape
#    tile_z, tile_y, tile_x = tile_grid_size_3d
#    
#    # Calculate 3D tile dimensions
#    tile_depth = max(1, depth // tile_z)
#    tile_height = max(4, height // tile_y)
#    tile_width = max(4, width // tile_x)
#    
#    # Recalculate actual number of tiles
#    actual_tile_z = depth // tile_depth
#    actual_tile_y = height // tile_y
#    actual_tile_x = width // tile_x
#    
#    # Calculate crop dimensions
#    crop_depth = tile_depth * actual_tile_z
#    crop_height = tile_height * actual_tile_y
#    crop_width = tile_width * actual_tile_x
#    stack_crop = stack[:crop_depth, :crop_height, :crop_width]
#    
#    # Adaptive clip limit based on 3D tile volume
#    voxels_per_tile = tile_depth * tile_height * tile_width
#    actual_clip_limit = max(1, int(clip_limit * voxels_per_tile / nbins))
#    
#    # Determine value range for histogram binning
#    min_val = cp.min(stack)
#    max_val = cp.max(stack)
#    value_range = (float(min_val), float(max_val))
#    
#    # Calculate transformation functions (CDFs) for each 3D tile center
#    tile_cdfs = cp.zeros((actual_tile_z, actual_tile_y, actual_tile_x, nbins), dtype=cp.float32)
#    bin_edges = cp.linspace(min_val, max_val, nbins + 1)
#    
#    for z_idx in range(actual_tile_z):
#        for y_idx in range(actual_tile_y):
#            for x_idx in range(actual_tile_x):
#                # Extract 3D tile
#                z_start = z_idx * tile_depth
#                z_end = (z_idx + 1) * tile_depth
#                y_start = y_idx * tile_height
#                y_end = (y_idx + 1) * tile_height
#                x_start = x_idx * tile_width
#                x_end = (x_idx + 1) * tile_width
#                
#                tile_3d = stack_crop[z_start:z_end, y_start:y_end, x_start:x_end]
#                
#                # Compute 3D histogram
#                if max_val > min_val:
#                    hist, _ = cp.histogram(tile_3d.flatten(), bins=nbins, range=value_range)
#                else:
#                    hist = cp.zeros(nbins, dtype=cp.int32)
#                    hist[nbins // 2] = tile_3d.size
#                
#                # Clip and redistribute histogram
#                hist = _clip_histogram(hist, actual_clip_limit)
#                
#                # Calculate CDF (transformation function)
#                cdf = cp.cumsum(hist).astype(cp.float32)
#                if cdf[-1] > 0:
#                    # Normalize CDF to output range
#                    cdf = (cdf / cdf[-1]) * (max_val - min_val) + min_val
#                else:
#                    cdf = cp.full_like(cdf, min_val)
#                
#                tile_cdfs[z_idx, y_idx, x_idx, :] = cdf
#    
#    # Apply voxel-level trilinear interpolation
#    result = _apply_voxel_level_trilinear_interpolation(
#        stack_crop, tile_cdfs,
#        actual_tile_z, actual_tile_y, actual_tile_x,
#        tile_depth, tile_height, tile_width,
#        bin_edges, min_val, max_val
#    )
#    
#    # Handle original stack size if needed
#    if result.shape != stack.shape:
#        full_result = cp.zeros_like(stack, dtype=result.dtype)
#        full_result[:crop_depth, :crop_height, :crop_width] = result
#        
#        # Fill remaining regions by replicating edge values
#        if crop_depth < depth:
#            full_result[crop_depth:, :crop_height, :crop_width] = result[-1:, :, :]
#        if crop_height < height:
#            full_result[:crop_depth, crop_height:, :crop_width] = result[:, -1:, :]
#        if crop_width < width:
#            full_result[:crop_depth, :crop_height, crop_width:] = result[:, :, -1:]
#            
#        # Handle corner cases
#        if crop_depth < depth and crop_height < height:
#            full_result[crop_depth:, crop_height:, :crop_width] = result[-1, -1:, :]
#        if crop_depth < depth and crop_width < width:
#            full_result[crop_depth:, :crop_height, crop_width:] = result[-1:, :, -1:]
#        if crop_height < height and crop_width < width:
#            full_result[:crop_depth, crop_height:, crop_width:] = result[:, -1:, -1:]
#        if crop_depth < depth and crop_height < height and crop_width < width:
#            full_result[crop_depth:, crop_height:, crop_width:] = result[-1, -1, -1]
#            
#        result = full_result
#    
#    return result.astype(stack.dtype)
#
#
#def _apply_pixel_level_interpolation(
#    image: "cp.ndarray",
#    tile_cdfs: "cp.ndarray",
#    tile_rows: int,
#    tile_cols: int,
#    tile_height: int,
#    tile_width: int,
#    bin_edges: "cp.ndarray",
#    min_val: float,
#    max_val: float
#) -> "cp.ndarray":
#    """
#    Apply proper pixel-level bilinear interpolation for CLAHE.
#    
#    Each pixel gets its value by interpolating between the transformation
#    functions (CDFs) of surrounding tile centers.
#    """
#    height, width = image.shape
#    result = cp.zeros_like(image, dtype=cp.float32)
#    
#    # Calculate tile center positions
#    tile_center_y = cp.arange(tile_rows) * tile_height + tile_height // 2
#    tile_center_x = cp.arange(tile_cols) * tile_width + tile_width // 2
#    
#    # Process each pixel
#    for y in range(height):
#        for x in range(width):
#            pixel_val = image[y, x]
#            
#            # Find which bin this pixel value belongs to
#            if max_val > min_val:
#                bin_idx = cp.searchsorted(bin_edges[1:], pixel_val)
#                bin_idx = min(bin_idx, len(bin_edges) - 2)
#            else:
#                bin_idx = len(bin_edges) // 2
#            
#            # Find surrounding tile centers
#            # Find the tile centers that surround this pixel
#            tile_y_low = cp.searchsorted(tile_center_y, y) - 1
#            tile_x_low = cp.searchsorted(tile_center_x, x) - 1
#            
#            tile_y_low = max(0, min(tile_y_low, tile_rows - 2))
#            tile_x_low = max(0, min(tile_x_low, tile_cols - 2))
#            
#            tile_y_high = tile_y_low + 1
#            tile_x_high = tile_x_low + 1
#            
#            # Get the 4 surrounding transformation values
#            val_tl = tile_cdfs[tile_y_low, tile_x_low, bin_idx]    # top-left
#            val_tr = tile_cdfs[tile_y_low, tile_x_high, bin_idx]   # top-right
#            val_bl = tile_cdfs[tile_y_high, tile_x_low, bin_idx]   # bottom-left
#            val_br = tile_cdfs[tile_y_high, tile_x_high, bin_idx]  # bottom-right
#            
#            # Calculate interpolation weights based on position
#            center_y_low = tile_center_y[tile_y_low]
#            center_y_high = tile_center_y[tile_y_high]
#            center_x_low = tile_center_x[tile_x_low]
#            center_x_high = tile_center_x[tile_x_high]
#            
#            # Bilinear interpolation weights
#            if center_y_high != center_y_low:
#                wy = (y - center_y_low) / (center_y_high - center_y_low)
#            else:
#                wy = 0.0
#                
#            if center_x_high != center_x_low:
#                wx = (x - center_x_low) / (center_x_high - center_x_low)
#            else:
#                wx = 0.0
#            
#            # Clamp weights to [0, 1]
#            wy = max(0.0, min(1.0, wy))
#            wx = max(0.0, min(1.0, wx))
#            
#            # Bilinear interpolation
#            val_top = (1 - wx) * val_tl + wx * val_tr
#            val_bottom = (1 - wx) * val_bl + wx * val_br
#            interpolated_val = (1 - wy) * val_top + wy * val_bottom
#            
#            result[y, x] = interpolated_val
#    
#    return result
#
#def _apply_voxel_level_trilinear_interpolation(
#    stack: "cp.ndarray",
#    tile_cdfs: "cp.ndarray",
#    tile_z: int,
#    tile_y: int,
#    tile_x: int,
#    tile_depth: int,
#    tile_height: int,
#    tile_width: int,
#    bin_edges: "cp.ndarray",
#    min_val: float,
#    max_val: float
#) -> "cp.ndarray":
#    """
#    Apply proper voxel-level trilinear interpolation for 3D CLAHE.
#    
#    Each voxel gets its value by trilinearly interpolating between the 
#    transformation functions (CDFs) of the 8 surrounding 3D tile centers.
#    """
#    depth, height, width = stack.shape
#    result = cp.zeros_like(stack, dtype=cp.float32)
#    
#    # Calculate 3D tile center positions
#    tile_center_z = cp.arange(tile_z) * tile_depth + tile_depth // 2
#    tile_center_y = cp.arange(tile_y) * tile_height + tile_height // 2
#    tile_center_x = cp.arange(tile_x) * tile_width + tile_width // 2
#    
#    # Process each voxel
#    for z in range(depth):
#        for y in range(height):
#            for x in range(width):
#                voxel_val = stack[z, y, x]
#                
#                # Find which bin this voxel value belongs to
#                if max_val > min_val:
#                    bin_idx = cp.searchsorted(bin_edges[1:], voxel_val)
#                    bin_idx = min(bin_idx, len(bin_edges) - 2)
#                else:
#                    bin_idx = len(bin_edges) // 2
#                
#                # Find surrounding tile centers in 3D
#                tile_z_low = cp.searchsorted(tile_center_z, z) - 1
#                tile_y_low = cp.searchsorted(tile_center_y, y) - 1
#                tile_x_low = cp.searchsorted(tile_center_x, x) - 1
#                
#                # Clamp to valid range
#                tile_z_low = max(0, min(tile_z_low, tile_z - 2)) if tile_z > 1 else 0
#                tile_y_low = max(0, min(tile_y_low, tile_y - 2))
#                tile_x_low = max(0, min(tile_x_low, tile_x - 2))
#                
#                tile_z_high = min(tile_z_low + 1, tile_z - 1)
#                tile_y_high = tile_y_low + 1
#                tile_x_high = tile_x_low + 1
#                
#                # Get the 8 surrounding transformation values for trilinear interpolation
#                # Front face (z_low)
#                val_000 = tile_cdfs[tile_z_low, tile_y_low, tile_x_low, bin_idx]     # front-bottom-left
#                val_001 = tile_cdfs[tile_z_low, tile_y_low, tile_x_high, bin_idx]    # front-bottom-right
#                val_010 = tile_cdfs[tile_z_low, tile_y_high, tile_x_low, bin_idx]    # front-top-left
#                val_011 = tile_cdfs[tile_z_low, tile_y_high, tile_x_high, bin_idx]   # front-top-right
#                
#                # Back face (z_high)
#                val_100 = tile_cdfs[tile_z_high, tile_y_low, tile_x_low, bin_idx]    # back-bottom-left
#                val_101 = tile_cdfs[tile_z_high, tile_y_low, tile_x_high, bin_idx]   # back-bottom-right
#                val_110 = tile_cdfs[tile_z_high, tile_y_high, tile_x_low, bin_idx]   # back-top-left
#                val_111 = tile_cdfs[tile_z_high, tile_y_high, tile_x_high, bin_idx]  # back-top-right
#                
#                # Calculate interpolation weights based on position
#                center_z_low = tile_center_z[tile_z_low]
#                center_z_high = tile_center_z[tile_z_high]
#                center_y_low = tile_center_y[tile_y_low]
#                center_y_high = tile_center_y[tile_y_high]
#                center_x_low = tile_center_x[tile_x_low]
#                center_x_high = tile_center_x[tile_x_high]
#                
#                # Trilinear interpolation weights
#                if center_z_high != center_z_low:
#                    wz = (z - center_z_low) / (center_z_high - center_z_low)
#                else:
#                    wz = 0.0
#                    
#                if center_y_high != center_y_low:
#                    wy = (y - center_y_low) / (center_y_high - center_y_low)
#                else:
#                    wy = 0.0
#                    
#                if center_x_high != center_x_low:
#                    wx = (x - center_x_low) / (center_x_high - center_x_low)
#                else:
#                    wx = 0.0
#                
#                # Clamp weights to [0, 1]
#                wz = max(0.0, min(1.0, wz))
#                wy = max(0.0, min(1.0, wy))
#                wx = max(0.0, min(1.0, wx))
#                
#                # Trilinear interpolation
#                # Interpolate along x-axis for each face
#                val_00 = (1 - wx) * val_000 + wx * val_001  # front-bottom
#                val_01 = (1 - wx) * val_010 + wx * val_011  # front-top
#                val_10 = (1 - wx) * val_100 + wx * val_101  # back-bottom
#                val_11 = (1 - wx) * val_110 + wx * val_111  # back-top
#                
#                # Interpolate along y-axis for each face
#                val_0 = (1 - wy) * val_00 + wy * val_01  # front face
#                val_1 = (1 - wy) * val_10 + wy * val_11  # back face
#                
#                # Final interpolation along z-axis
#                interpolated_val = (1 - wz) * val_0 + wz * val_1
#                
#                result[z, y, x] = interpolated_val
#    
#    return result
#
#def _clip_histogram(hist: "cp.ndarray", clip_limit: int) -> "cp.ndarray":
#    """
#    Simplified histogram clipping with uniform redistribution.
#    
#    Args:
#        hist: Input histogram array
#        clip_limit: Maximum allowed value for any histogram bin
#        
#    Returns:
#        Clipped and redistributed histogram
#    """
#    if clip_limit <= 0:
#        return hist
#    
#    # Find excess above clip limit
#    excess = cp.maximum(hist - clip_limit, 0)
#    total_excess = cp.sum(excess)
#    
#    # Clip the histogram
#    clipped_hist = cp.minimum(hist, clip_limit)
#    
#    # Redistribute excess uniformly
#    if total_excess > 0:
#        nbins = len(hist)
#        redistribution = total_excess // nbins
#        remainder = total_excess % nbins
#        
#        # Add uniform redistribution to all bins
#        clipped_hist = clipped_hist + redistribution
#        
#        # Distribute remainder to first bins
#        if remainder > 0:
#            clipped_hist[:remainder] += 1
#            
#        # Final clipping in case redistribution caused overflow
#        clipped_hist = cp.minimum(clipped_hist, clip_limit)
#    
#    return clipped_hist
#
#def _apply_clahe_mapping(
#    tile: "cp.ndarray",
#    tile_mappings: "cp.ndarray", 
#    row: int,
#    col: int,
#    tile_rows: int,
#    tile_cols: int,
#    nbins: int,
#    max_val: float
#) -> "cp.ndarray":
#    """
#    Apply CLAHE mapping with bilinear interpolation between neighboring tiles.
#    """
#    # Convert pixel values to bin indices
#    if max_val > 0:
#        bin_indices = ((tile.astype(cp.float32) / max_val) * (nbins - 1)).astype(cp.int32)
#        bin_indices = cp.clip(bin_indices, 0, nbins - 1)
#    else:
#        bin_indices = cp.zeros_like(tile, dtype=cp.int32)
#    
#    # For interior tiles, use bilinear interpolation
#    if 0 < row < tile_rows - 1 and 0 < col < tile_cols - 1:
#        # Get the four surrounding tile mappings
#        tl = tile_mappings[row, col, bin_indices]       # top-left
#        tr = tile_mappings[row, col + 1, bin_indices]   # top-right  
#        bl = tile_mappings[row + 1, col, bin_indices]   # bottom-left
#        br = tile_mappings[row + 1, col + 1, bin_indices] # bottom-right
#        
#        # Simple bilinear interpolation
#        result = 0.25 * (tl + tr + bl + br)
#    else:
#        # For edge/corner tiles, just use the tile's own mapping
#        result = tile_mappings[row, col, bin_indices]
#    
#    return result