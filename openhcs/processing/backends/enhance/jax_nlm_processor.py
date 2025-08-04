"""
JAX-based Non-Local Means Denoising Implementation

This module provides OpenHCS-decorated wrapper functions for non-local means denoising
using JAX backend with automatic output rescaling to prevent clipping issues when
converting to uint16.

Non-local means is an advanced denoising algorithm that preserves fine details
and textures by comparing patches across the entire image rather than just
local neighborhoods. This JAX implementation provides GPU acceleration with
automatic output normalization.

Doctrinal Clauses:
- Clause 3 — Declarative Primacy: All functions are pure and stateless
- Clause 65 — Fail Loudly: No silent fallbacks or inferred capabilities
- Clause 88 — No Inferred Capabilities: Explicit JAX dependency
- Clause 273 — Memory Backend Restrictions: JAX-only implementation
"""
from __future__ import annotations

import logging
from typing import Optional

from openhcs.utils.import_utils import optional_import
from openhcs.core.memory.decorators import jax as jax_func

# Import JAX modules as optional dependencies
jax = optional_import("jax")
jnp = optional_import("jax.numpy") if jax is not None else None
lax = jax.lax if jax is not None else None
tree_util = jax.tree_util if jax is not None else None

logger = logging.getLogger(__name__)


def _validate_jax_array(image: "jnp.ndarray") -> None:
    """Validate that input is a JAX array (2D or 3D)."""
    if jax is None or jnp is None:
        raise ImportError("JAX is required for JAX NLM functions")

    if not isinstance(image, jnp.ndarray):
        raise TypeError(f"Input must be a jax.numpy.ndarray, got {type(image)}")

    if image.ndim not in [2, 3]:
        raise ValueError(f"Input must be a 2D or 3D array, got {image.ndim}D array")


def _rescale_to_unit_range(image: "jnp.ndarray") -> "jnp.ndarray":
    """
    Rescale image so that the minimum value across the entire stack is 0 
    and the maximum value is 1.
    
    This prevents clipping issues when converting to uint16.
    
    Args:
        image: 3D JAX array of shape (Z, Y, X)
        
    Returns:
        Rescaled 3D JAX array with values in [0, 1] range
    """
    # Calculate global min and max across the entire stack
    global_min = jnp.min(image)
    global_max = jnp.max(image)
    
    # Avoid division by zero
    range_val = global_max - global_min
    
    # If all values are the same, return zeros
    def rescale_normal(args):
        image, global_min, range_val = args
        return (image - global_min) / range_val
    
    def return_zeros(args):
        image, _, _ = args
        return jnp.zeros_like(image)
    
    # Use JAX conditional to handle zero range
    result = lax.cond(
        range_val > 0,
        rescale_normal,
        return_zeros,
        (image, global_min, range_val)
    )
    
    return result


def _ixs(y_ixs, x_ixs):
    """Create meshgrid for vectorized operations."""
    return jnp.meshgrid(x_ixs, y_ixs)


def _vmap_2d(f, y_ixs, x_ixs):
    """Apply function f over 2D grid using vectorized mapping."""
    _x, _y = _ixs(y_ixs, x_ixs)
    return jax.vmap(jax.vmap(f))(_y, _x)


# Use jax.tree_util.Partial instead of functools.partial for better JAX integration
# - jax.tree_util.Partial is a JAX pytree, compatible with JAX transformations
# - Enables proper serialization and JIT compilation
# - Better performance with JAX's internal machinery
@tree_util.Partial(jax.jit, static_argnums=(1, 2)) if jax is not None and tree_util is not None else lambda f: f
def _nlm_core(img: "jnp.ndarray", search_window_radius: int, filter_radius: int, h: float, sigma: float) -> "jnp.ndarray":
    """
    Core non-local means implementation based on Buades et al.

    This is a vectorized and JIT-compiled implementation adapted from:
    https://github.com/bhchiang/nlm

    Args:
        img: 2D image array
        search_window_radius: Radius of search window
        filter_radius: Radius of comparison patches
        h: Filter strength parameter
        sigma: Noise standard deviation

    Returns:
        Denoised 2D image
    """
    _h, _w = img.shape
    pad = search_window_radius
    img_pad = jnp.pad(img, pad, mode='reflect')

    filter_length = 2 * filter_radius + 1
    search_window_length = 2 * search_window_radius + 1

    win_y_ixs = win_x_ixs = jnp.arange(search_window_length - filter_length + 1)
    filter_size = (filter_length, filter_length)

    def compute(y, x):
        # (y + pad, x + pad) are the center of the current neighborhood
        win_center_y = y + pad
        win_center_x = x + pad

        center_patch = lax.dynamic_slice(
            img_pad,
            (win_center_y - filter_radius, win_center_x - filter_radius),
            filter_size
        )

        # Iterate over all patches in this neighborhood
        def _compare(center):
            center_y, center_x = center
            patch = lax.dynamic_slice(
                img_pad,
                (center_y - filter_radius, center_x - filter_radius),
                filter_size
            )
            d2 = jnp.sum((patch - center_patch) ** 2) / (filter_length ** 2)
            weight = jnp.exp(-(jnp.maximum(d2 - 2 * (sigma**2), 0) / (h**2)))
            intensity = img_pad[center_y, center_x]
            return (weight, intensity)

        def compare(patch_y, patch_x):
            patch_center_y = patch_y + filter_radius
            patch_center_x = patch_x + filter_radius

            # Skip if patch is out of image boundaries or this is the center patch
            skip = (lax.lt(patch_center_y, pad) |
                   lax.ge(patch_center_y, _h + pad) |
                   lax.lt(patch_center_x, pad) |
                   lax.ge(patch_center_x, _w + pad) |
                   (lax.eq(patch_center_y, win_center_y) & lax.eq(patch_center_x, win_center_x)))

            return lax.cond(
                skip,
                lambda _: (0., 0.),
                _compare,
                (patch_center_y, patch_center_x)
            )

        weights, intensities = _vmap_2d(compare, y + win_y_ixs, x + win_x_ixs)

        # Use max weight for the center patch
        max_weight = jnp.max(weights)
        total_weight = jnp.sum(weights) + max_weight
        pixel = ((jnp.sum(weights * intensities) +
                 max_weight * img_pad[win_center_y, win_center_x]) / total_weight)

        return pixel

    h_ixs = jnp.arange(_h)
    w_ixs = jnp.arange(_w)
    out = _vmap_2d(compute, h_ixs, w_ixs)

    return out


@jax_func
def non_local_means_denoise_jax(
    image: "jnp.ndarray",
    *,
    search_window_radius: int = 7,
    filter_radius: int = 1,
    h: Optional[float] = None,
    sigma: Optional[float] = None,
    slice_by_slice: bool = False,
    **kwargs
) -> "jnp.ndarray":
    """
    Apply Non-Local Means denoising to image(s) using JAX.

    This function applies vectorized and JIT-compiled non-local means denoising
    based on the implementation by Buades et al. The output is automatically
    rescaled to [0, 1] range to prevent clipping issues when converting to uint16.

    Can handle both 2D and 3D inputs:
    - 2D input: Direct processing (when called by decorator on individual slices)
    - 3D input: Slice-by-slice processing or raises error for 3D mode

    Args:
        image: 2D JAX array of shape (Y, X) or 3D JAX array of shape (Z, Y, X)
        search_window_radius: Radius of search window (default: 7)
        filter_radius: Radius of comparison patches (default: 1)
        h: Filter strength parameter (default: auto-estimated from image)
        sigma: Noise standard deviation (default: auto-estimated from image)
        slice_by_slice: Process each Z-slice independently (default: False, but effectively True).
                       If explicitly set to False, raises NotImplementedError for 3D processing.
        **kwargs: Additional arguments (ignored for compatibility)

    Returns:
        Denoised JAX array of same shape as input with values always rescaled to [0, 1] range

    Raises:
        ImportError: If JAX is not available
        TypeError: If input is not a jax.numpy.ndarray
        ValueError: If input is not 2D or 3D
        NotImplementedError: If slice_by_slice=False (3D processing not yet implemented)

    Additional OpenHCS Parameters
    -----------------------------
    slice_by_slice : bool, optional (default: False, but effectively True)
        If True or not explicitly set to False, process 3D arrays slice-by-slice using
        2D non-local means. If explicitly set to False, raises NotImplementedError.
        Note: 3D processing is not yet implemented for JAX backend.
    """
    _validate_jax_array(image)

    if jax is None or jnp is None:
        raise ImportError(
            "JAX is required for this function. "
            "Install with: pip install jax"
        )

    # Store original dtype for reference
    original_dtype = image.dtype

    # Convert to float32 for processing and normalize to [0, 1] range
    image_float = image.astype(jnp.float32)

    # Normalize input to [0, 1] for consistent parameter behavior
    img_min = jnp.min(image_float)
    img_max = jnp.max(image_float)
    if img_max > img_min:
        image_normalized = (image_float - img_min) / (img_max - img_min)
    else:
        image_normalized = jnp.zeros_like(image_float)

    # Auto-estimate parameters if not provided
    if sigma is None:
        # Simple noise estimation using Laplacian
        laplacian_kernel = jnp.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=jnp.float32)

        # Apply to appropriate slice for estimation
        if image.ndim == 3:
            estimation_slice = image_normalized[0]  # Use first slice for 3D
        else:
            estimation_slice = image_normalized     # Use the 2D image directly

        padded = jnp.pad(estimation_slice, 1, mode='reflect')
        laplacian = jnp.zeros_like(estimation_slice)
        for i in range(3):
            for j in range(3):
                shifted = padded[i:i + estimation_slice.shape[0], j:j + estimation_slice.shape[1]]
                laplacian += laplacian_kernel[i, j] * shifted
        sigma = jnp.sqrt(2) * jnp.std(laplacian) / 6.0
        sigma = jnp.maximum(sigma, 0.01)  # Minimum sigma

    if h is None:
        h = 0.75 * sigma  # Standard relationship

    # Handle different input dimensions
    if image.ndim == 2:
        # 2D input: Process directly (called by decorator on individual slices)
        result = _nlm_core(image_normalized, search_window_radius, filter_radius, h, sigma)
    elif image.ndim == 3:
        # 3D input: If we get here with 3D input, it means slice_by_slice=False
        # because when slice_by_slice=True, the decorator handles slicing
        raise NotImplementedError(
            "3D non-local means processing is not yet implemented for JAX backend. "
            "Use slice_by_slice=True for 2D slice-by-slice processing."
        )
    else:
        raise ValueError(f"Unexpected input dimensions: {image.ndim}D")

    # Always rescale output to [0, 1] range to prevent uint16 clipping
    result = _rescale_to_unit_range(result)
    logger.info("Rescaled NLM output to [0, 1] range to prevent uint16 clipping")

    return result


# Alias for convenience
jax_nlm_denoise = non_local_means_denoise_jax
