"""
PyTorch Image Processor Implementation

This module implements the ImageProcessorInterface using PyTorch as the backend.
It leverages GPU acceleration for image processing operations.

Doctrinal Clauses:
- Clause 3 — Declarative Primacy: All functions are pure and stateless
- Clause 88 — No Inferred Capabilities: Explicit PyTorch dependency
- Clause 106-A — Declared Memory Types: All methods specify PyTorch tensors
"""
from __future__ import annotations 

import logging
from typing import Any, List, Optional, Tuple

from openhcs.core.utils import optional_import
from openhcs.core.memory.decorators import torch as torch_func

# Import PyTorch as an optional dependency
torch = optional_import("torch")
F = optional_import("torch.nn.functional") if torch is not None else None
HAS_TORCH = torch is not None

logger = logging.getLogger(__name__)


def create_linear_weight_mask(height: int, width: int, margin_ratio: float = 0.1) -> "torch.Tensor":
    """
    Create a 2D weight mask that linearly ramps from 0 at the edges to 1 in the center.

    Args:
        height: Height of the mask
        width: Width of the mask
        margin_ratio: Ratio of the margin to the image size

    Returns:
        2D PyTorch weight mask of shape (height, width)
    """
    if torch is None:
        raise ImportError("PyTorch is required for TorchImageProcessor")

    margin_y = int(torch.floor(torch.tensor(height * margin_ratio)))
    margin_x = int(torch.floor(torch.tensor(width * margin_ratio)))

    weight_y = torch.ones(height, dtype=torch.float32)
    if margin_y > 0:
        ramp_top = torch.linspace(0, 1, margin_y, dtype=torch.float32)
        ramp_bottom = torch.linspace(1, 0, margin_y, dtype=torch.float32)
        weight_y[:margin_y] = ramp_top
        weight_y[-margin_y:] = ramp_bottom

    weight_x = torch.ones(width, dtype=torch.float32)
    if margin_x > 0:
        ramp_left = torch.linspace(0, 1, margin_x, dtype=torch.float32)
        ramp_right = torch.linspace(1, 0, margin_x, dtype=torch.float32)
        weight_x[:margin_x] = ramp_left
        weight_x[-margin_x:] = ramp_right

    # Create 2D weight mask using outer product
    weight_mask = torch.outer(weight_y, weight_x)

    return weight_mask


def _validate_3d_array(array: Any, name: str = "input") -> None:
    """
    Validate that the input is a 3D PyTorch tensor.

    Args:
        array: Array to validate
        name: Name of the array for error messages

    Raises:
        TypeError: If the array is not a PyTorch tensor
        ValueError: If the array is not 3D
        ImportError: If PyTorch is not available
    """
    if torch is None:
        raise ImportError("PyTorch is required for TorchImageProcessor")

    if not isinstance(array, torch.Tensor):
        raise TypeError(f"{name} must be a PyTorch tensor, got {type(array)}. "
                       f"No automatic conversion is performed to maintain explicit contracts.")

    if array.ndim != 3:
        raise ValueError(f"{name} must be a 3D tensor, got {array.ndim}D")

def _gaussian_blur(image: "torch.Tensor", sigma: float) -> "torch.Tensor":
    """
    Apply Gaussian blur to a 2D image.

    Args:
        image: 2D PyTorch tensor of shape (H, W)
        sigma: Standard deviation of the Gaussian kernel

    Returns:
        Blurred 2D PyTorch tensor of shape (H, W)
    """
    # Calculate kernel size based on sigma
    kernel_size = max(3, int(2 * 4 * sigma + 1))
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd kernel size

    # Create 1D Gaussian kernel
    coords = torch.arange(kernel_size, dtype=torch.float32, device=image.device)
    coords -= (kernel_size - 1) / 2

    # Calculate Gaussian values
    gauss = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = gauss / gauss.sum()

    # Reshape for 2D convolution
    kernel_x = kernel.view(1, 1, kernel_size, 1)
    kernel_y = kernel.view(1, 1, 1, kernel_size)

    # Add batch and channel dimensions to image
    img = image.unsqueeze(0).unsqueeze(0)

    # Apply separable convolution
    blurred = F.conv2d(img, kernel_x, padding=(kernel_size//2, 0))
    blurred = F.conv2d(blurred, kernel_y, padding=(0, kernel_size//2))

    # Remove batch and channel dimensions
    return blurred.squeeze(0).squeeze(0)

@torch_func
def sharpen(image: "torch.Tensor", radius: float = 1.0, amount: float = 1.0) -> "torch.Tensor":
    """
    Sharpen a 3D image using unsharp masking.

    This applies sharpening to each Z-slice independently.

    Args:
        image: 3D PyTorch tensor of shape (Z, Y, X)
        radius: Radius of Gaussian blur
        amount: Sharpening strength

    Returns:
        Sharpened 3D PyTorch tensor of shape (Z, Y, X)
    """
    _validate_3d_array(image)

    # Store original dtype
    dtype = image.dtype

    # Process each Z-slice independently
    result = torch.zeros_like(image, dtype=torch.float32)

    for z in range(image.shape[0]):
        # Convert to float for processing
        slice_float_raw = image[z].float()
        slice_float = slice_float_raw / torch.max(slice_float_raw)

        # Create blurred version for unsharp mask
        blurred = _gaussian_blur(slice_float, sigma=radius)

        # Apply unsharp mask: original + amount * (original - blurred)
        sharpened = slice_float + amount * (slice_float - blurred)

        # Clip to valid range
        sharpened = torch.clamp(sharpened, 0, 1.0)

        # Scale back to original range
        min_val = torch.min(sharpened)
        max_val = torch.max(sharpened)
        if max_val > min_val:
            sharpened = (sharpened - min_val) * 65535 / (max_val - min_val)

        result[z] = sharpened

    # Convert back to original dtype
    if dtype == torch.uint16:
        result = torch.clamp(result, 0, 65535).to(torch.uint16)
    else:
        result = result.to(dtype)

    return result

@torch_func
def percentile_normalize(
    image: "torch.Tensor",
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    target_min: float = 0.0,
    target_max: float = 65535.0
) -> "torch.Tensor":
    """
    Normalize a 3D image using percentile-based contrast stretching.

    This applies normalization to each Z-slice independently.

    Args:
        image: 3D PyTorch tensor of shape (Z, Y, X)
        low_percentile: Lower percentile (0-100)
        high_percentile: Upper percentile (0-100)
        target_min: Target minimum value
        target_max: Target maximum value

    Returns:
        Normalized 3D PyTorch tensor of shape (Z, Y, X)
    """
    _validate_3d_array(image)

    # Process each Z-slice independently
    result = torch.zeros_like(image, dtype=torch.float32)

    for z in range(image.shape[0]):
        # Get percentile values for this slice
        # Handle large slices that exceed PyTorch's quantile() size limits
        slice_float = image[z].float()
        slice_elements = slice_float.numel()

        # PyTorch quantile() fails on very large tensors, so we use sampling for large slices
        max_elements_for_quantile = 10_000_000  # ~10M elements, conservative limit for quantile()

        logger.debug(f"🔥 QUANTILE DEBUG: percentile_normalize slice {z} shape {image[z].shape}, {slice_elements:,} elements")

        if slice_elements > max_elements_for_quantile:
            # Use random sampling for large slices to estimate percentiles
            sample_size = min(max_elements_for_quantile, slice_elements // 10)  # Sample 10% or max size
            flat_slice = slice_float.flatten()

            # Generate random indices for sampling (memory efficient)
            # Use torch.randint instead of torch.randperm to avoid creating huge tensors
            indices = torch.randint(0, slice_elements, (sample_size,), device=image.device)
            sampled_values = flat_slice[indices]

            p_low = torch.quantile(sampled_values, low_percentile / 100.0)
            p_high = torch.quantile(sampled_values, high_percentile / 100.0)
        else:
            # Use full slice for smaller slices
            p_low = torch.quantile(slice_float, low_percentile / 100.0)
            p_high = torch.quantile(slice_float, high_percentile / 100.0)

        # Avoid division by zero
        if p_high == p_low:
            result[z] = torch.ones_like(image[z], dtype=torch.float32) * target_min
            continue

        # Clip and normalize to target range
        clipped = torch.clamp(image[z].float(), p_low, p_high)
        scale = (target_max - target_min) / (p_high - p_low)
        normalized = (clipped - p_low) * scale + target_min
        result[z] = normalized

    # Convert to uint16
    result = torch.clamp(result, 0, 65535).to(torch.uint16)

    return result

@torch_func
def stack_percentile_normalize(
    stack: "torch.Tensor",
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    target_min: float = 0.0,
    target_max: float = 65535.0
) -> "torch.Tensor":
    """
    Normalize a stack using global percentile-based contrast stretching.

    This ensures consistent normalization across all Z-slices by computing
    global percentiles across the entire stack.

    Args:
        stack: 3D PyTorch tensor of shape (Z, Y, X)
        low_percentile: Lower percentile (0-100)
        high_percentile: Upper percentile (0-100)
        target_min: Target minimum value
        target_max: Target maximum value

    Returns:
        Normalized 3D PyTorch tensor of shape (Z, Y, X)
    """
    _validate_3d_array(stack)

    # Calculate global percentiles across the entire stack
    # Handle large tensors that exceed PyTorch's quantile() size limits
    stack_float = stack.float()
    total_elements = stack_float.numel()

    # PyTorch quantile() fails on very large tensors, so we use sampling for large stacks
    max_elements_for_quantile = 10_000_000  # ~10M elements, conservative limit for quantile()

    logger.debug(f"🔥 QUANTILE DEBUG: stack_percentile_normalize called with tensor shape {stack.shape}, {total_elements:,} elements")

    if total_elements > max_elements_for_quantile:
        # Use random sampling for large tensors to estimate percentiles
        sample_size = min(max_elements_for_quantile, total_elements // 10)  # Sample 10% or max size
        flat_stack = stack_float.flatten()

        # Generate random indices for sampling (memory efficient)
        # Use torch.randint instead of torch.randperm to avoid creating huge tensors
        indices = torch.randint(0, total_elements, (sample_size,), device=stack.device)
        sampled_values = flat_stack[indices]

        p_low = torch.quantile(sampled_values, low_percentile / 100.0)
        p_high = torch.quantile(sampled_values, high_percentile / 100.0)

        logger.debug(f"Used sampling ({sample_size:,} of {total_elements:,} elements) for percentile calculation due to large tensor size")
    else:
        # Use full tensor for smaller stacks
        p_low = torch.quantile(stack_float, low_percentile / 100.0)
        p_high = torch.quantile(stack_float, high_percentile / 100.0)

    # Avoid division by zero
    if p_high == p_low:
        return torch.ones_like(stack, dtype=torch.float32) * target_min

    # Clip and normalize to target range
    clipped = torch.clamp(stack.float(), p_low, p_high)
    normalized = (clipped - p_low) * (target_max - target_min) / (p_high - p_low) + target_min
    normalized = torch.clamp(normalized, 0, 65535).to(torch.uint16)

    return normalized

@torch_func
def create_composite(
    stack: "torch.Tensor", weights: Optional[List[float]] = None
) -> "torch.Tensor":
    """
    Create a composite image from a 3D stack of 2D images.

    Args:
        stack: 3D PyTorch tensor of shape (N, Y, X) where N is number of images
        weights: List of weights for each image. If None, equal weights are used.

    Returns:
        Composite 3D PyTorch tensor of shape (1, Y, X)
    """
    # Validate input is 3D tensor
    _validate_3d_array(stack)

    n_images, height, width = stack.shape

    # Default weights if none provided
    if weights is None:
        # Equal weights for all images
        weights = [1.0 / n_images] * n_images
    elif not isinstance(weights, list):
        raise TypeError("weights must be a list of values")

    # FAIL FAST: No fallback weights - weights must match exactly
    if len(weights) != n_images:
        raise ValueError(
            f"Weights list length ({len(weights)}) must exactly match number of images ({n_images}). "
            f"No automatic padding or truncation allowed."
        )

    dtype = stack.dtype
    device = stack.device

    # Create empty composite
    composite = torch.zeros((height, width), dtype=torch.float32, device=device)
    total_weight = 0.0

    # Add each image with its weight
    for i in range(n_images):
        weight = weights[i]
        if weight <= 0.0:
            continue

        # Add to composite
        composite += stack[i].float() * weight
        total_weight += weight

    # Normalize by total weight
    if total_weight > 0:
        composite /= total_weight

    # Convert back to original dtype (usually uint16)
    if dtype == torch.uint16:
        composite = torch.clamp(composite, 0, 65535).to(torch.uint16)
    else:
        composite = composite.to(dtype)

    # Return as 3D tensor with shape (1, Y, X)
    return composite.reshape(1, height, width)

@torch_func
def apply_mask(image: "torch.Tensor", mask: "torch.Tensor") -> "torch.Tensor":
    """
    Apply a mask to a 3D image while maintaining 3D structure.

    This applies the mask to each Z-slice independently if mask is 2D,
    or applies the 3D mask directly if mask is 3D.

    Args:
        image: 3D PyTorch tensor of shape (Z, Y, X)
        mask: 3D PyTorch tensor of shape (Z, Y, X) or 2D PyTorch tensor of shape (Y, X)

    Returns:
        Masked 3D PyTorch tensor of shape (Z, Y, X) - dimensionality preserved
    """
    _validate_3d_array(image)

    # Handle 2D mask (apply to each Z-slice)
    if isinstance(mask, torch.Tensor) and mask.ndim == 2:
        if mask.shape != image.shape[1:]:
            raise ValueError(
                f"2D mask shape {mask.shape} doesn't match image slice shape {image.shape[1:]}"
            )

        # Apply 2D mask to each Z-slice
        result = torch.zeros_like(image)
        for z in range(image.shape[0]):
            result[z] = image[z].float() * mask.float()

        return result.to(image.dtype)

    # Handle 3D mask
    if isinstance(mask, torch.Tensor) and mask.ndim == 3:
        if mask.shape != image.shape:
            raise ValueError(
                f"3D mask shape {mask.shape} doesn't match image shape {image.shape}"
            )

        # Apply 3D mask directly
        masked = image.float() * mask.float()
        return masked.to(image.dtype)

    # If we get here, the mask is neither 2D nor 3D PyTorch tensor
    raise TypeError(f"mask must be a 2D or 3D PyTorch tensor, got {type(mask)}")

@torch_func
def create_weight_mask(
    shape: Tuple[int, int], margin_ratio: float = 0.1
) -> "torch.Tensor":
    """
    Create a weight mask for blending images.

    Args:
        shape: Shape of the mask (height, width)
        margin_ratio: Ratio of image size to use as margin

    Returns:
        2D PyTorch weight mask of shape (Y, X)
    """
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise TypeError("shape must be a tuple of (height, width)")

    height, width = shape
    return create_linear_weight_mask(height, width, margin_ratio)

@torch_func
def max_projection(stack: "torch.Tensor") -> "torch.Tensor":
    """
    Create a maximum intensity projection from a Z-stack.

    Args:
        stack: 3D PyTorch tensor of shape (Z, Y, X)

    Returns:
        3D PyTorch tensor of shape (1, Y, X)
    """
    _validate_3d_array(stack)

    # Store original dtype for conversion back
    original_dtype = stack.dtype

    # Convert to float32 if needed for GPU operations
    if stack.dtype == torch.uint16:
        stack_float = stack.float()
    else:
        stack_float = stack

    # Create max projection
    projection_2d = torch.max(stack_float, dim=0)[0]

    # Convert back to original dtype
    projection_2d = projection_2d.to(original_dtype)

    return projection_2d.reshape(1, projection_2d.shape[0], projection_2d.shape[1])

@torch_func
def mean_projection(stack: "torch.Tensor") -> "torch.Tensor":
    """
    Create a mean intensity projection from a Z-stack.

    Args:
        stack: 3D PyTorch tensor of shape (Z, Y, X)

    Returns:
        3D PyTorch tensor of shape (1, Y, X)
    """
    _validate_3d_array(stack)

    # Store original dtype for conversion back
    original_dtype = stack.dtype

    # Convert to float32 for mean calculation (always needed for mean)
    stack_float = stack.float()

    # Create mean projection
    projection_2d = torch.mean(stack_float, dim=0)

    # Convert back to original dtype
    projection_2d = projection_2d.to(original_dtype)

    return projection_2d.reshape(1, projection_2d.shape[0], projection_2d.shape[1])

@torch_func
def stack_equalize_histogram(
    stack: "torch.Tensor",
    bins: int = 65536,
    range_min: float = 0.0,
    range_max: float = 65535.0
) -> "torch.Tensor":
    """
    Apply histogram equalization to an entire stack.

    This ensures consistent contrast enhancement across all Z-slices by
    computing a global histogram across the entire stack.

    Args:
        stack: 3D PyTorch tensor of shape (Z, Y, X)
        bins: Number of bins for histogram computation
        range_min: Minimum value for histogram range
        range_max: Maximum value for histogram range

    Returns:
        Equalized 3D PyTorch tensor of shape (Z, Y, X)
    """
    _validate_3d_array(stack)

    # PyTorch doesn't have a direct histogram equalization function
    # We'll implement it manually using torch.histc for the histogram

    # Flatten the entire stack to compute the global histogram
    flat_stack = stack.float().flatten()

    # For very large stacks, use sampling to avoid memory issues
    max_elements_for_histogram = 50_000_000  # 50M elements limit
    if flat_stack.numel() > max_elements_for_histogram:
        # Use random sampling for histogram computation
        sample_size = max_elements_for_histogram
        indices = torch.randint(0, flat_stack.numel(), (sample_size,), device=stack.device)
        sampled_stack = flat_stack[indices]
        hist = torch.histc(sampled_stack, bins=bins, min=range_min, max=range_max)
        logger.debug(f"Used sampling ({sample_size:,} of {flat_stack.numel():,} elements) for histogram computation")
    else:
        # Use full stack for smaller stacks
        hist = torch.histc(flat_stack, bins=bins, min=range_min, max=range_max)

    # We don't need bin edges for the lookup table approach

    # Calculate cumulative distribution function (CDF)
    cdf = torch.cumsum(hist, dim=0)

    # Normalize the CDF to the range [0, 65535]
    # Avoid division by zero
    if cdf[-1] > 0:
        cdf = 65535 * cdf / cdf[-1]

    # PyTorch doesn't have a direct equivalent to numpy's interp
    # We'll use a lookup table approach

    # Scale input values to bin indices
    indices = torch.clamp(
        ((flat_stack - range_min) / (range_max - range_min) * (bins - 1)).long(),
        0, bins - 1
    )

    # Look up CDF values
    equalized_flat = torch.gather(cdf, 0, indices)

    # Reshape back to original shape
    equalized_stack = equalized_flat.reshape(stack.shape)

    # Convert to uint16
    return equalized_stack.to(torch.uint16)

@torch_func
def create_projection(
    stack: "torch.Tensor", method: str = "max_projection"
) -> "torch.Tensor":
    """
    Create a projection from a stack using the specified method.

    Args:
        stack: 3D PyTorch tensor of shape (Z, Y, X)
        method: Projection method (max_projection, mean_projection)

    Returns:
        3D PyTorch tensor of shape (1, Y, X)
    """
    _validate_3d_array(stack)

    if method == "max_projection":
        return max_projection(stack)

    if method == "mean_projection":
        return mean_projection(stack)

    # FAIL FAST: No fallback projection methods
    raise ValueError(f"Unknown projection method: {method}. Valid methods: max_projection, mean_projection")

@torch_func
def tophat(
    image: "torch.Tensor",
    selem_radius: int = 50,
    downsample_factor: int = 4
) -> "torch.Tensor":
    """
    Apply white top-hat filter to a 3D image for background removal.

    This applies the filter to each Z-slice independently using PyTorch's
    native operations.

    Args:
        image: 3D PyTorch tensor of shape (Z, Y, X)
        selem_radius: Radius of the structuring element disk
        downsample_factor: Factor by which to downsample the image for processing

    Returns:
        Filtered 3D PyTorch tensor of shape (Z, Y, X)
    """
    _validate_3d_array(image)

    # Store device for later use
    device = image.device

    # Process each Z-slice independently
    result = torch.zeros_like(image)

    # We'll create structuring elements for each slice as needed

    for z in range(image.shape[0]):
        # Store original data type
        input_dtype = image[z].dtype

        # 1) Downsample using PyTorch's interpolate function
        # First, add batch and channel dimensions for interpolate
        img_4d = image[z].float().unsqueeze(0).unsqueeze(0)

        # Calculate new dimensions
        new_h = image[z].shape[0] // downsample_factor
        new_w = image[z].shape[1] // downsample_factor

        # Resize using PyTorch's interpolate function
        image_small = F.interpolate(
            img_4d,
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0)

        # 2) Resize the structuring element to match the downsampled image
        small_selem_radius = max(1, selem_radius // downsample_factor)
        small_grid_size = 2 * small_selem_radius + 1
        small_grid_y, small_grid_x = torch.meshgrid(
            torch.arange(small_grid_size, device=device) - small_selem_radius,
            torch.arange(small_grid_size, device=device) - small_selem_radius,
            indexing='ij'
        )
        small_mask = (small_grid_x.pow(2) + small_grid_y.pow(2)) <= small_selem_radius**2
        small_selem = small_mask.float()

        # 3) Apply white top-hat using PyTorch's convolution operations
        # White top-hat is opening subtracted from the original image
        # Opening is erosion followed by dilation

        # Implement erosion using min pooling with custom kernel
        # First, pad the image to handle boundary conditions
        pad_size = small_selem_radius
        padded = F.pad(
            image_small.unsqueeze(0).unsqueeze(0),
            (pad_size, pad_size, pad_size, pad_size),
            mode='reflect'
        )

        # Unfold the padded image into patches
        patches = F.unfold(padded, kernel_size=small_grid_size, stride=1)

        # Reshape patches for processing
        patch_size = small_grid_size * small_grid_size
        patches = patches.reshape(1, patch_size, new_h, new_w)

        # Apply the structuring element as a mask
        masked_patches = patches * small_selem.reshape(-1, 1, 1)

        # Perform erosion (min pooling)
        eroded = torch.min(
            masked_patches + (1 - small_selem.reshape(-1, 1, 1)) * 1e9,
            dim=1
        )[0]

        # Implement dilation using max pooling with custom kernel
        # Pad the eroded image
        padded_eroded = F.pad(
            eroded.unsqueeze(0).unsqueeze(0),
            (pad_size, pad_size, pad_size, pad_size),
            mode='reflect'
        )

        # Unfold the padded eroded image into patches
        patches_eroded = F.unfold(padded_eroded, kernel_size=small_grid_size, stride=1)

        # Reshape patches for processing
        patch_size = small_grid_size * small_grid_size
        patches_eroded = patches_eroded.reshape(1, patch_size, new_h, new_w)

        # Apply the structuring element as a mask
        masked_patches_eroded = patches_eroded * small_selem.reshape(-1, 1, 1)

        # Perform dilation (max pooling)
        opened = torch.max(masked_patches_eroded, dim=1)[0]

        # White top-hat is original minus opening
        tophat_small = image_small - opened

        # 4) Calculate background
        background_small = image_small - tophat_small

        # 5) Upscale background to original size
        background_4d = background_small.unsqueeze(0).unsqueeze(0)
        background_large = F.interpolate(
            background_4d,
            size=image[z].shape,
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0)

        # 6) Subtract background and clip negative values
        slice_result = torch.clamp(image[z].float() - background_large, min=0.0)

        # 7) Convert back to original data type
        result[z] = slice_result.to(input_dtype)

    return result
