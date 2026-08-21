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
from enum import Enum
from typing import Optional

from openhcs.utils.import_utils import optional_import
from openhcs.core.memory import jax as jax_func

# Import JAX modules as optional dependencies
from openhcs.core.lazy_gpu_imports import jax

jnp = optional_import("jax.numpy") if jax else None
lax = jax.lax if jax else None
tree_util = jax.tree_util if jax else None

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
        range_val > 0, rescale_normal, return_zeros, (image, global_min, range_val)
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
@(
    tree_util.Partial(jax.jit, static_argnums=(1, 2))
    if jax is not None and tree_util is not None
    else lambda f: f
)
def _nlm_core(
    img: "jnp.ndarray",
    search_window_radius: int,
    filter_radius: int,
    h: float,
    sigma: float,
) -> "jnp.ndarray":
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
    img_pad = jnp.pad(img, pad, mode="reflect")

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
            filter_size,
        )

        # Iterate over all patches in this neighborhood
        def _compare(center):
            center_y, center_x = center
            patch = lax.dynamic_slice(
                img_pad,
                (center_y - filter_radius, center_x - filter_radius),
                filter_size,
            )
            d2 = jnp.sum((patch - center_patch) ** 2) / (filter_length**2)
            weight = jnp.exp(-(jnp.maximum(d2 - 2 * (sigma**2), 0) / (h**2)))
            intensity = img_pad[center_y, center_x]
            return (weight, intensity)

        def compare(patch_y, patch_x):
            patch_center_y = patch_y + filter_radius
            patch_center_x = patch_x + filter_radius

            # Skip if patch is out of image boundaries or this is the center patch
            skip = (
                lax.lt(patch_center_y, pad)
                | lax.ge(patch_center_y, _h + pad)
                | lax.lt(patch_center_x, pad)
                | lax.ge(patch_center_x, _w + pad)
                | (
                    lax.eq(patch_center_y, win_center_y)
                    & lax.eq(patch_center_x, win_center_x)
                )
            )

            return lax.cond(
                skip, lambda _: (0.0, 0.0), _compare, (patch_center_y, patch_center_x)
            )

        weights, intensities = _vmap_2d(compare, y + win_y_ixs, x + win_x_ixs)

        # Use max weight for the center patch
        max_weight = jnp.max(weights)
        total_weight = jnp.sum(weights) + max_weight
        pixel = (
            jnp.sum(weights * intensities)
            + max_weight * img_pad[win_center_y, win_center_x]
        ) / total_weight

        return pixel

    h_ixs = jnp.arange(_h)
    w_ixs = jnp.arange(_w)
    out = _vmap_2d(compute, h_ixs, w_ixs)

    return out


def _use_whole_image(image: "jnp.ndarray") -> "jnp.ndarray":
    return image


def _use_first_plane(image: "jnp.ndarray") -> "jnp.ndarray":
    return image[0]


def _denoise_2d(
    image: "jnp.ndarray",
    search_window_radius: int,
    filter_radius: int,
    h: float,
    sigma: float,
) -> "jnp.ndarray":
    return _nlm_core(image, search_window_radius, filter_radius, h, sigma)


def _reject_unsupported_3d(
    image: "jnp.ndarray",
    search_window_radius: int,
    filter_radius: int,
    h: float,
    sigma: float,
) -> "jnp.ndarray":
    del image, search_window_radius, filter_radius, h, sigma
    raise NotImplementedError(
        "3D non-local means processing is not yet implemented for JAX backend. "
        "Use slice_by_slice=True for 2D slice-by-slice processing."
    )


class JaxNlmInputDimensionality(Enum):
    """Input dimensionalities carrying their estimation and execution leaves."""

    IMAGE_2D = (2, _use_whole_image, _denoise_2d)
    VOLUME_3D = (3, _use_first_plane, _reject_unsupported_3d)

    def __new__(cls, ndim, estimation_selector, denoiser):
        member = object.__new__(cls)
        member._value_ = ndim
        member._estimation_selector = estimation_selector
        member._denoiser = denoiser
        return member

    @classmethod
    def from_ndim(cls, ndim: int) -> "JaxNlmInputDimensionality":
        try:
            return cls(ndim)
        except ValueError as exc:
            raise ValueError(f"Unexpected input dimensions: {ndim}D") from exc

    def estimation_slice(self, image_normalized: "jnp.ndarray") -> "jnp.ndarray":
        return self._estimation_selector(image_normalized)

    def denoise(
        self,
        image_normalized: "jnp.ndarray",
        search_window_radius: int,
        filter_radius: int,
        h: float,
        sigma: float,
    ) -> "jnp.ndarray":
        return self._denoiser(
            image_normalized,
            search_window_radius,
            filter_radius,
            h,
            sigma,
        )


@jax_func
def non_local_means_denoise_jax(
    image: "jnp.ndarray",
    *,
    search_window_radius: int = 7,
    filter_radius: int = 1,
    h: Optional[float] = None,
    sigma: Optional[float] = None,
) -> "jnp.ndarray":
    """
    Apply Non-Local Means denoising to image(s) using JAX.

    This function applies vectorized and JIT-compiled non-local means denoising
    based on the implementation by Buades et al. The output is automatically
    rescaled to [0, 1] range to prevent clipping issues when converting to uint16.

    Two-dimensional inputs are processed directly. For three-dimensional inputs,
    enable the decorator-owned ``slice_by_slice`` control; volumetric JAX NLM is
    not currently available.

    Args:
        image: 2D JAX array of shape (Y, X) or 3D JAX array of shape (Z, Y, X)
        search_window_radius: Radius of search window (default: 7)
        filter_radius: Radius of comparison patches (default: 1)
        h: Filter strength parameter (default: auto-estimated from image)
        sigma: Noise standard deviation (default: auto-estimated from image)

    Returns:
        Denoised JAX array of same shape as input with values always rescaled to [0, 1] range

    Raises:
        ImportError: If JAX is not available
        TypeError: If input is not a jax.numpy.ndarray
        ValueError: If input is not 2D or 3D
        NotImplementedError: If a 3D input is not processed slice by slice
    """
    _validate_jax_array(image)

    if jax is None or jnp is None:
        raise ImportError(
            "JAX is required for this function. " "Install with: pip install jax"
        )

    input_dimensionality = JaxNlmInputDimensionality.from_ndim(image.ndim)

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
        laplacian_kernel = jnp.array(
            [[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=jnp.float32
        )

        estimation_slice = input_dimensionality.estimation_slice(image_normalized)

        padded = jnp.pad(estimation_slice, 1, mode="reflect")
        laplacian = jnp.zeros_like(estimation_slice)
        for i in range(3):
            for j in range(3):
                shifted = padded[
                    i : i + estimation_slice.shape[0], j : j + estimation_slice.shape[1]
                ]
                laplacian += laplacian_kernel[i, j] * shifted
        sigma = jnp.sqrt(2) * jnp.std(laplacian) / 6.0
        sigma = jnp.maximum(sigma, 0.01)  # Minimum sigma

    if h is None:
        h = 0.75 * sigma  # Standard relationship

    result = input_dimensionality.denoise(
        image_normalized,
        search_window_radius,
        filter_radius,
        h,
        sigma,
    )

    # Always rescale output to [0, 1] range to prevent uint16 clipping
    result = _rescale_to_unit_range(result)
    logger.info("Rescaled NLM output to [0, 1] range to prevent uint16 clipping")

    return result


# Alias for convenience
jax_nlm_denoise = non_local_means_denoise_jax
