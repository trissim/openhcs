"""
GPU-accelerated implementation of Ashlar alignment algorithm.

This module provides a CuPy-based implementation of the Ashlar algorithm for
computing tile positions in a microscopy image stitching workflow.

The implementation follows the OpenHCS doctrinal principles:
- Clause 3 — Declarative Primacy: All functions are pure and stateless
- Clause 65 — Fail Loudly: No silent fallbacks or inferred capabilities
- Clause 88 — No Inferred Capabilities: Explicit CuPy dependency
- Clause 101 — Memory Declaration: Memory-resident output, no side effects
- Clause 273 — Memory Backend Restrictions: GPU-only implementation
"""

import logging
from typing import TYPE_CHECKING, Any, Tuple

from openhcs.constants.constants import SpecialKey
from openhcs.core.memory.decorators import cupy as cupy_func
from openhcs.core.pipeline.function_contracts import special_out, chain_breaker

# For type checking only
if TYPE_CHECKING:
    import cupy as cp
    from cupyx.scipy import ndimage

# Import CuPy with error handling
try:
    import cupy as cp  # type: ignore
    from cupyx.scipy import ndimage  # type: ignore
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    # Create dummy objects for type checking
    class DummyCupy:
        """Dummy class that raises ImportError when attributes are accessed."""
        def __getattr__(self, name):
            raise ImportError("CuPy is not installed. Please install it to use GPU-accelerated functions.")
    cp = DummyCupy()

logger = logging.getLogger(__name__)


def _validate_cupy_array(array: Any, name: str = "input") -> None:  # type: ignore
    """
    Validate that the input is a CuPy array.

    Args:
        array: Array to validate
        name: Name of the array for error messages

    Raises:
        ImportError: If CuPy is not available
        TypeError: If the array is not a CuPy array
        ValueError: If the array doesn't support DLPack
    """
    if not HAS_CUPY:
        raise ImportError("CuPy is required for GPU-accelerated Ashlar alignment")

    if not isinstance(array, cp.ndarray):
        raise TypeError(
            f"{name} must be a CuPy array, got {type(array)}. "
            f"No automatic conversion is performed to maintain explicit contracts. "
            f"Use DLPack for zero-copy GPU-to-GPU transfers."
        )


def phase_correlation(
    image1: "cp.ndarray",  # type: ignore
    image2: "cp.ndarray",  # type: ignore
    *,
    window: bool = False,
    subpixel: bool = False,
    subpixel_radius: int = 3
) -> Tuple[float, float]:
    """
    Compute the phase correlation between two images to find the translation.

    Args:
        image1: First image
        image2: Second image
        window: Whether to apply a Hann window to reduce edge effects
        subpixel: Whether to compute subpixel precision
        subpixel_radius: Radius for subpixel interpolation

    Returns:
        Tuple of (y_shift, x_shift) as floats
    """
    # Validate inputs
    _validate_cupy_array(image1, "image1")
    _validate_cupy_array(image2, "image2")

    if image1.shape != image2.shape:
        raise ValueError(f"Images must have the same shape, got {image1.shape} and {image2.shape}")

    # Apply window if requested
    if window:
        h, w = image1.shape
        window_y = cp.hanning(h).reshape(-1, 1)
        window_x = cp.hanning(w).reshape(1, -1)
        window_2d = window_y * window_x

        image1 = image1 * window_2d
        image2 = image2 * window_2d

    # Compute FFTs
    fft1 = cp.fft.fft2(image1)
    fft2 = cp.fft.fft2(image2)

    # Compute cross-power spectrum
    cross_power = fft1 * cp.conj(fft2)
    cross_power_norm = cross_power / (cp.abs(cross_power) + 1e-8)

    # Compute inverse FFT
    correlation = cp.abs(cp.fft.ifft2(cross_power_norm))

    # Find peak
    peak_idx = cp.unravel_index(cp.argmax(correlation), correlation.shape)
    y_peak, x_peak = peak_idx

    # Convert to shift
    h, w = correlation.shape
    y_shift = y_peak if y_peak < h // 2 else y_peak - h
    x_shift = x_peak if x_peak < w // 2 else x_peak - w

    # Compute subpixel precision if requested
    if subpixel:
        # Extract region around peak
        y_min = max(0, y_peak - subpixel_radius)
        y_max = min(h, y_peak + subpixel_radius + 1)
        x_min = max(0, x_peak - subpixel_radius)
        x_max = min(w, x_peak + subpixel_radius + 1)

        region = correlation[y_min:y_max, x_min:x_max]

        # Compute center of mass
        y_indices, x_indices = cp.mgrid[y_min:y_max, x_min:x_max]
        total_mass = cp.sum(region)

        if total_mass > 0:
            y_com = cp.sum(y_indices * region) / total_mass
            x_com = cp.sum(x_indices * region) / total_mass

            # Convert to shift
            y_shift = y_com if y_com < h // 2 else y_com - h
            x_shift = x_com if x_com < w // 2 else x_com - w

    return float(y_shift), float(x_shift)

@chain_breaker(SpecialKey.POSITION_ARRAY)
@special_out(SpecialKey.POSITION_ARRAY)
@cupy_func
def gpu_ashlar_align_cupy(
    tiles: "cp.ndarray",  # type: ignore
    num_rows: int,
    num_cols: int,
    *,
    patch_size: int = 128,
    search_radius: int = 20,
    normalize: bool = False,
    method: str = "phase_correlation",
    return_affine: bool = False,
    verbose: bool = False,
    **kwargs
) -> "cp.ndarray":  # type: ignore
    """
    Given a stack of 2D image tiles in ZYX order, estimate per-tile (x, y) shift using
    GPU-native phase correlation and graph-based global alignment.

    Args:
        tiles: 3D tensor of shape (Z, Y, X) where each Z slice is a 2D tile
        num_rows: Number of rows in the grid
        num_cols: Number of columns in the grid
        patch_size: Size of the patches used for correlation
        search_radius: Maximum search radius for correlation
        normalize: Whether to normalize the images before correlation
        method: Method for computing correlation (only "phase_correlation" is supported)
        return_affine: Whether to return affine transformation matrices
        verbose: Whether to print progress information
        **kwargs: Additional parameters

    Returns:
        If return_affine is False:
            3D array of shape (Z, 2) where each entry is an (y, x) float position
        If return_affine is True:
            3D array of shape (Z, 3, 3) containing affine transformation matrices

    Raises:
        ImportError: If CuPy is not available
        ValueError: If the input shape is invalid
        TypeError: If the input is not a CuPy array
    """
    # Validate inputs
    _validate_cupy_array(tiles, "tiles")

    if tiles.ndim != 3:
        raise ValueError(f"Input must be a 3D tensor, got {tiles.ndim}D")

    Z, H, W = tiles.shape
    if Z != num_rows * num_cols:
        raise ValueError(
            f"Number of tiles ({Z}) does not match grid size ({num_rows}x{num_cols}={num_rows*num_cols})"
        )

    if method != "phase_correlation":
        raise ValueError(f"Only 'phase_correlation' method is supported, got '{method}'")

    # Convert to float32 for better precision
    if normalize:
        # Normalize each tile to [0, 1]
        tiles = tiles.astype(cp.float32)
        for z in range(Z):
            tile = tiles[z]
            tile_min = cp.min(tile)
            tile_max = cp.max(tile)
            if tile_max > tile_min:
                tiles[z] = (tile - tile_min) / (tile_max - tile_min)
    else:
        tiles = tiles.astype(cp.float32)

    # Reshape to grid layout for easier access
    tiles_grid = tiles.reshape(num_rows, num_cols, H, W)

    # Initialize offsets array
    offsets = cp.zeros((Z, 2), dtype=cp.float32)

    # Build relative shift graph from neighbors
    for r in range(num_rows):
        for c in range(num_cols):
            idx = r * num_cols + c

            # Skip the top-left tile (anchor)
            if r == 0 and c == 0:
                continue

            # Process top neighbor
            if r > 0:
                top_idx = (r - 1) * num_cols + c
                top_tile = tiles_grid[r-1, c]
                current_tile = tiles_grid[r, c]

                # Compute shift using phase correlation
                dy, dx = phase_correlation(current_tile, top_tile)

                # Update offset based on top neighbor
                offsets[idx] = offsets[top_idx] + cp.array([dy, dx])

            # Process left neighbor (if no top neighbor)
            elif c > 0:
                left_idx = r * num_cols + (c - 1)
                left_tile = tiles_grid[r, c-1]
                current_tile = tiles_grid[r, c]

                # Compute shift using phase correlation
                dy, dx = phase_correlation(current_tile, left_tile)

                # Update offset based on left neighbor
                offsets[idx] = offsets[left_idx] + cp.array([dy, dx])

    # Apply drift correction (center around mean)
    mean_offset = cp.mean(offsets, axis=0)
    offsets -= mean_offset

    # Return affine transformation matrices if requested
    if return_affine:
        affine_mats = cp.zeros((Z, 3, 3), dtype=cp.float32)

        for z in range(Z):
            dy, dx = offsets[z]
            # Create affine transformation matrix
            # [ 1  0  dx ]
            # [ 0  1  dy ]
            # [ 0  0  1  ]
            affine_mats[z] = cp.array([
                [1.0, 0.0, dx],
                [0.0, 1.0, dy],
                [0.0, 0.0, 1.0]
            ])

        return affine_mats

    # Return offsets (y, x)
    return offsets
