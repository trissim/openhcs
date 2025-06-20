"""
CuPy implementation of image assembly functions.

This module provides GPU-accelerated functions for assembling microscopy images
using CuPy. It handles subpixel positioning and blending of image tiles.
"""
from __future__ import annotations 

import logging
from typing import TYPE_CHECKING, List, Tuple, Union, List, Tuple, Union

from openhcs.core.memory.decorators import cupy as cupy_func
from openhcs.core.pipeline.function_contracts import special_inputs
from openhcs.core.utils import optional_import

# For type checking only
if TYPE_CHECKING:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter
    from cupyx.scipy.ndimage import shift as subpixel_shift

# Import CuPy as an optional dependency
cp = optional_import("cupy")

# Import CuPy functions if available
if cp is not None:
    cupyx_scipy = optional_import("cupyx.scipy.ndimage")
    if cupyx_scipy is not None:
        gaussian_filter = cupyx_scipy.gaussian_filter
        subpixel_shift = cupyx_scipy.shift
    else:
        gaussian_filter = None
        subpixel_shift = None
else:
    gaussian_filter = None
    subpixel_shift = None

logger = logging.getLogger(__name__)

def _get_all_overlapping_pairs_gpu(positions: "cp.ndarray", tile_shape: tuple) -> list:  # type: ignore
    """
    GPU-accelerated detection of ALL overlapping tile pairs with edge directions.

    Args:
        positions: CuPy array of shape (N, 2) with (x, y) positions
        tile_shape: (height, width) of tiles

    Returns:
        List of (tile_i, tile_j, edge_direction, pixel_overlap) tuples
        edge_direction: 'left', 'right', 'top', 'bottom' relative to tile_i
    """
    height, width = tile_shape
    N = positions.shape[0]

    if N <= 1:
        return []

    # Vectorized computation of ALL pairwise overlaps (fully GPU-accelerated)
    # Broadcast positions for vectorized comparisons
    pos_i = positions[:, cp.newaxis, :]  # Shape: (N, 1, 2)
    pos_j = positions[cp.newaxis, :, :]  # Shape: (1, N, 2)

    # Extract coordinates
    xi, yi = pos_i[:, :, 0], pos_i[:, :, 1]  # Shape: (N, 1)
    xj, yj = pos_j[:, :, 0], pos_j[:, :, 1]  # Shape: (1, N)

    # Compute tile boundaries
    left_i, right_i = xi, xi + width
    top_i, bottom_i = yi, yi + height
    left_j, right_j = xj, xj + width
    top_j, bottom_j = yj, yj + height

    # Compute overlap amounts between ALL pairs (vectorized on GPU)
    x_overlap = cp.maximum(0, cp.minimum(right_i, right_j) - cp.maximum(left_i, left_j))
    y_overlap = cp.maximum(0, cp.minimum(bottom_i, bottom_j) - cp.maximum(top_i, top_j))

    # Valid overlaps (both x and y must overlap, and not self)
    valid_overlap = (x_overlap > 0) & (y_overlap > 0) & (cp.arange(N)[:, None] != cp.arange(N)[None, :])

    print(f"🔍 GPU DIRECT ADJACENCY: Checking all {N}×{N} pairs for overlaps")

    # VECTORIZED: Keep everything on GPU, eliminate CPU transfers
    overlapping_pairs = cp.where(valid_overlap)
    pair_indices_i = overlapping_pairs[0]
    pair_indices_j = overlapping_pairs[1]

    if len(pair_indices_i) == 0:
        return []

    # Extract overlap values and positions for valid pairs (all on GPU)
    pair_x_overlaps = x_overlap[pair_indices_i, pair_indices_j]
    pair_y_overlaps = y_overlap[pair_indices_i, pair_indices_j]

    # Get positions for all pairs
    pos_i = positions[pair_indices_i]  # Shape: (num_pairs, 2)
    pos_j = positions[pair_indices_j]  # Shape: (num_pairs, 2)

    # Vectorized direction determination
    xi_vals, yi_vals = pos_i[:, 0], pos_i[:, 1]
    xj_vals, yj_vals = pos_j[:, 0], pos_j[:, 1]

    # Create boolean masks for each direction (vectorized)
    has_x_overlap = pair_x_overlaps > 0
    has_y_overlap = pair_y_overlaps > 0

    j_left_of_i = xj_vals < xi_vals
    j_right_of_i = xj_vals > xi_vals
    j_above_i = yj_vals < yi_vals
    j_below_i = yj_vals > yi_vals

    # Build edge pairs list (minimal CPU transfer at the end)
    edge_pairs = []

    # Convert to CPU only for final list construction (much smaller data)
    indices_i_cpu = cp.asnumpy(pair_indices_i)
    indices_j_cpu = cp.asnumpy(pair_indices_j)
    x_overlaps_cpu = cp.asnumpy(pair_x_overlaps)
    y_overlaps_cpu = cp.asnumpy(pair_y_overlaps)

    has_x_cpu = cp.asnumpy(has_x_overlap)
    has_y_cpu = cp.asnumpy(has_y_overlap)
    left_cpu = cp.asnumpy(j_left_of_i)
    right_cpu = cp.asnumpy(j_right_of_i)
    above_cpu = cp.asnumpy(j_above_i)
    below_cpu = cp.asnumpy(j_below_i)

    # Vectorized edge pair construction
    for idx in range(len(indices_i_cpu)):
        i, j = indices_i_cpu[idx], indices_j_cpu[idx]
        x_overlap_val = float(x_overlaps_cpu[idx])
        y_overlap_val = float(y_overlaps_cpu[idx])

        # Horizontal overlaps
        if has_x_cpu[idx]:
            if left_cpu[idx]:
                edge_pairs.append((i, j, 'left', x_overlap_val))
            elif right_cpu[idx]:
                edge_pairs.append((i, j, 'right', x_overlap_val))

        # Vertical overlaps
        if has_y_cpu[idx]:
            if above_cpu[idx]:
                edge_pairs.append((i, j, 'top', y_overlap_val))
            elif below_cpu[idx]:
                edge_pairs.append((i, j, 'bottom', y_overlap_val))

    print(f"✅ GPU: Found {len(edge_pairs)} total edge overlaps from {len(indices_i_cpu)} overlapping pairs")
    return edge_pairs


def _create_batch_fixed_masks_gpu(
    tile_shape: tuple,
    all_edge_overlaps: list,
    margin_ratio: float = 0.1
) -> "cp.ndarray":
    """
    VECTORIZED: Create all fixed blend masks at once for 2-3x speedup.
    Uses batch operations instead of individual mask creation.
    """
    height, width = tile_shape
    num_tiles = len(all_edge_overlaps)

    # Pre-calculate margin pixels
    margin_pixels_y = int(height * margin_ratio)
    margin_pixels_x = int(width * margin_ratio)

    # Create batch of 1D weights - shape (N, height) and (N, width)
    y_weights = cp.ones((num_tiles, height), dtype=cp.float32)
    x_weights = cp.ones((num_tiles, width), dtype=cp.float32)

    # Pre-generate gradient arrays (reuse for all tiles)
    if margin_pixels_y > 0:
        top_gradient = cp.linspace(0, 1, margin_pixels_y, endpoint=False, dtype=cp.float32)
        bottom_gradient = cp.linspace(1, 0, margin_pixels_y, endpoint=False, dtype=cp.float32)

    if margin_pixels_x > 0:
        left_gradient = cp.linspace(0, 1, margin_pixels_x, endpoint=False, dtype=cp.float32)
        right_gradient = cp.linspace(1, 0, margin_pixels_x, endpoint=False, dtype=cp.float32)

    # Apply gradients to each tile (vectorized where possible)
    for i, edge_overlaps in enumerate(all_edge_overlaps):
        if 'top' in edge_overlaps and margin_pixels_y > 0:
            y_weights[i, :margin_pixels_y] = top_gradient

        if 'bottom' in edge_overlaps and margin_pixels_y > 0:
            y_weights[i, -margin_pixels_y:] = bottom_gradient

        if 'left' in edge_overlaps and margin_pixels_x > 0:
            x_weights[i, :margin_pixels_x] = left_gradient

        if 'right' in edge_overlaps and margin_pixels_x > 0:
            x_weights[i, -margin_pixels_x:] = right_gradient

    # Batch outer product using broadcasting: (N, H, 1) * (N, 1, W) = (N, H, W)
    masks = y_weights[:, :, cp.newaxis] * x_weights[:, cp.newaxis, :]

    return masks.astype(cp.float32)


def _create_batch_dynamic_masks_gpu(
    tile_shape: tuple,
    all_edge_overlaps: list,
    overlap_fraction: float = 1.0
) -> "cp.ndarray":
    """
    VECTORIZED: Create all dynamic blend masks at once for 2-3x speedup.
    """
    height, width = tile_shape
    num_tiles = len(all_edge_overlaps)

    # Create batch of 1D weights
    y_weights = cp.ones((num_tiles, height), dtype=cp.float32)
    x_weights = cp.ones((num_tiles, width), dtype=cp.float32)

    # Apply gradients to each tile
    for i, edge_overlaps in enumerate(all_edge_overlaps):
        if 'top' in edge_overlaps:
            overlap_pixels = int(edge_overlaps['top'] * overlap_fraction)
            if overlap_pixels > 0:
                y_weights[i, :overlap_pixels] = cp.linspace(0, 1, overlap_pixels, endpoint=False)

        if 'bottom' in edge_overlaps:
            overlap_pixels = int(edge_overlaps['bottom'] * overlap_fraction)
            if overlap_pixels > 0:
                y_weights[i, -overlap_pixels:] = cp.linspace(1, 0, overlap_pixels, endpoint=False)

        if 'left' in edge_overlaps:
            overlap_pixels = int(edge_overlaps['left'] * overlap_fraction)
            if overlap_pixels > 0:
                x_weights[i, :overlap_pixels] = cp.linspace(0, 1, overlap_pixels, endpoint=False)

        if 'right' in edge_overlaps:
            overlap_pixels = int(edge_overlaps['right'] * overlap_fraction)
            if overlap_pixels > 0:
                x_weights[i, -overlap_pixels:] = cp.linspace(1, 0, overlap_pixels, endpoint=False)

    # Batch outer product using broadcasting
    masks = y_weights[:, :, cp.newaxis] * x_weights[:, cp.newaxis, :]

    return masks.astype(cp.float32)


def _create_dynamic_blend_mask_gpu(
    tile_shape: tuple,
    edge_overlaps: dict,
    overlap_fraction: float = 1.0
) -> "cp.ndarray":
    """
    GPU version of dynamic blend mask using WORKING logic from CPU version.
    CRITICAL: Uses endpoint=False and same logic as working CPU version.
    """
    height, width = tile_shape

    # Create 1D weights
    y_weight = cp.ones(height, dtype=cp.float32)
    x_weight = cp.ones(width, dtype=cp.float32)

    # Process each edge based on actual overlap (same as working CPU version)
    # CRITICAL: endpoint=False (this is what made the CPU version work!)
    if 'top' in edge_overlaps:
        overlap_pixels = int(edge_overlaps['top'] * overlap_fraction)
        if overlap_pixels > 0:
            y_weight[:overlap_pixels] = cp.linspace(0, 1, overlap_pixels, endpoint=False)

    if 'bottom' in edge_overlaps:
        overlap_pixels = int(edge_overlaps['bottom'] * overlap_fraction)
        if overlap_pixels > 0:
            y_weight[-overlap_pixels:] = cp.linspace(1, 0, overlap_pixels, endpoint=False)

    if 'left' in edge_overlaps:
        overlap_pixels = int(edge_overlaps['left'] * overlap_fraction)
        if overlap_pixels > 0:
            x_weight[:overlap_pixels] = cp.linspace(0, 1, overlap_pixels, endpoint=False)

    if 'right' in edge_overlaps:
        overlap_pixels = int(edge_overlaps['right'] * overlap_fraction)
        if overlap_pixels > 0:
            x_weight[-overlap_pixels:] = cp.linspace(1, 0, overlap_pixels, endpoint=False)

    # Use outer product (same as working CPU version)
    mask = cp.outer(y_weight, x_weight)
    return mask.astype(cp.float32)


# Removed old complex function - using simpler _create_simple_dynamic_mask_gpu instead


def _create_gaussian_blend_mask(tile_shape: tuple, blend_radius: float) -> "cp.ndarray":  # type: ignore
    """
    Legacy function for backward compatibility.
    Use _create_blend_mask with blend_method="gaussian" instead.
    """
    return _create_blend_mask(tile_shape, "gaussian", blend_radius)


@special_inputs("positions") # The input name is "positions"
@cupy_func
def assemble_stack_cupy(
    image_tiles: "cp.ndarray",  # type: ignore
    positions: Union[List[Tuple[float, float]], "cp.ndarray"],  # type: ignore
    blend_method: str = "fixed",
    fixed_margin_ratio: float = 0.1,
    overlap_blend_fraction: float = 1.0
) -> "cp.ndarray":  # type: ignore
    """
    GPU-accelerated assembly using WORKING logic from CPU version.

    Args:
        image_tiles: 3D CuPy array of tiles (N, H, W)
        positions: List of (x, y) tuples or 2D array [N, 2]
        blend_method: "none", "fixed", or "dynamic"
        fixed_margin_ratio: Ratio for fixed blending (e.g., 0.1 = 10%)
        overlap_blend_fraction: For dynamic mode, fraction of overlap to blend

    Returns:
        3D CuPy array (1, H_canvas, W_canvas) with assembled image
    """
    # The compiler will ensure this function is only called when CuPy is available
    # No need to check for CuPy availability here
    # --- 1. Validate and standardize inputs ---
    if not isinstance(image_tiles, cp.ndarray) or image_tiles.ndim != 3:
        raise TypeError("image_tiles must be a 3D CuPy ndarray of shape (N, H, W).")
    if image_tiles.shape[0] == 0:
        logger.warning("image_tiles array is empty (0 tiles). Returning an empty array.")
        return cp.array([[[]]], dtype=cp.uint16) # Shape (1,0,0) to indicate empty 3D

    # Convert positions to CuPy array for GPU-native operations
    if isinstance(positions, list):
        # Convert list of tuples to CuPy array
        if not positions or not isinstance(positions[0], tuple) or len(positions[0]) != 2:
            raise TypeError("positions must be a list of (x, y) tuples.")
        positions = cp.array(positions, dtype=cp.float32)
    else:
        # Handle array input (backward compatibility)
        if not hasattr(positions, 'ndim') or positions.ndim != 2 or positions.shape[1] != 2:
            raise TypeError("positions must be an array of shape [N, 2] or list of (x, y) tuples.")
        positions = cp.asarray(positions)  # Convert to cupy for GPU operations

    # Debug: Print positions information
    print(f"Assembly: Received {positions.shape[0]} positions for {image_tiles.shape[0]} tiles")
    print(f"Position range: X=[{float(cp.min(positions[:, 0])):.1f}, {float(cp.max(positions[:, 0])):.1f}], Y=[{float(cp.min(positions[:, 1])):.1f}, {float(cp.max(positions[:, 1])):.1f}]")
    print(f"First 3 positions: {positions[:3].tolist()}")

    # Debug: Check image tile statistics
    print(f"🔥 ASSEMBLY DEBUG: Image tiles shape: {image_tiles.shape}")
    print(f"🔥 ASSEMBLY DEBUG: Image tiles dtype: {image_tiles.dtype}")
    for i in range(min(3, image_tiles.shape[0])):
        tile_min = float(cp.min(image_tiles[i]))
        tile_max = float(cp.max(image_tiles[i]))
        tile_mean = float(cp.mean(image_tiles[i]))
        tile_nonzero = int(cp.count_nonzero(image_tiles[i]))
        print(f"🔥 ASSEMBLY DEBUG: Tile {i}: min={tile_min:.3f}, max={tile_max:.3f}, mean={tile_mean:.3f}, nonzero={tile_nonzero}")

    # Debug: Check if tiles are all zeros
    total_nonzero = int(cp.count_nonzero(image_tiles))
    total_pixels = int(cp.prod(cp.array(image_tiles.shape)))
    print(f"🔥 ASSEMBLY DEBUG: Total nonzero pixels: {total_nonzero}/{total_pixels} ({100*total_nonzero/total_pixels:.1f}%)")

    if image_tiles.shape[0] != positions.shape[0]:
        raise ValueError(f"Mismatch between number of image_tiles ({image_tiles.shape[0]}) and positions ({positions.shape[0]}).")

    num_tiles, tile_h, tile_w = image_tiles.shape
    first_tile_shape = (tile_h, tile_w) # Used for blend mask, assumes all tiles same H, W

    # Note: Convert tiles to float32 one at a time to save memory
    # (removed bulk conversion to avoid doubling memory usage)

    # --- 2. Compute canvas bounds ---
    # positions_xy are for top-left corners.
    # Add tile dimensions to get bottom-right corners for each tile.
    # positions_xy[:, 0] is X (width dimension), positions_xy[:, 1] is Y (height dimension)

    # Min/max X coordinates of tile top-left corners
    min_x_pos = cp.min(positions[:, 0])
    max_x_pos = cp.max(positions[:, 0])

    # Min/max Y coordinates of tile top-left corners
    min_y_pos = cp.min(positions[:, 1])
    max_y_pos = cp.max(positions[:, 1])

    # Canvas dimensions need to encompass all tiles
    # Canvas origin will be (min_x_pos_rounded_down, min_y_pos_rounded_down)
    # Max extent is max_pos + tile_dim
    canvas_min_x = cp.floor(min_x_pos).astype(cp.int32) # cupy needs explicit int type for astype(int)
    canvas_min_y = cp.floor(min_y_pos).astype(cp.int32) # cupy needs explicit int type for astype(int)

    canvas_max_x = cp.ceil(max_x_pos + tile_w).astype(cp.int32) # cupy needs explicit int type for astype(int)
    canvas_max_y = cp.ceil(max_y_pos + tile_h).astype(cp.int32) # cupy needs explicit int type for astype(int)

    canvas_width = canvas_max_x - canvas_min_x
    canvas_height = canvas_max_y - canvas_min_y

    # Debug: Print canvas information
    print(f"Canvas: {int(canvas_width)}x{int(canvas_height)} pixels, origin=({float(canvas_min_x):.1f}, {float(canvas_min_y):.1f})")
    print(f"Tile size: {tile_w}x{tile_h} pixels")

    if canvas_width <= 0 or canvas_height <= 0:
        logger.warning(f"Calculated canvas dimensions are non-positive ({canvas_height}x{canvas_width}). Check positions and tile sizes.")
        return cp.array([], dtype=cp.uint16)

    composite_accum = cp.zeros((int(canvas_height), int(canvas_width)), dtype=cp.float32)
    weight_accum = cp.zeros((int(canvas_height), int(canvas_width)), dtype=cp.float32)

    # --- 3. Generate blend masks using WORKING logic from CPU version ---
    if blend_method == "none":
        blend_masks = [cp.ones(first_tile_shape, dtype=cp.float32) for _ in range(num_tiles)]

    else:
        # Find overlaps (same as working CPU version)
        edge_pairs = _get_all_overlapping_pairs_gpu(positions, first_tile_shape)
        tile_overlaps = [{} for _ in range(num_tiles)]

        # Build overlap info per tile
        for tile_i, tile_j, edge_direction, pixel_overlap in edge_pairs:
            if edge_direction not in tile_overlaps[tile_i]:
                tile_overlaps[tile_i][edge_direction] = pixel_overlap
            else:
                # Keep maximum overlap
                tile_overlaps[tile_i][edge_direction] = max(
                    tile_overlaps[tile_i][edge_direction], pixel_overlap
                )

        # VECTORIZED: Create all masks at once using batch operations
        if blend_method == "fixed":
            # Create all fixed masks in one batch operation
            masks_batch = _create_batch_fixed_masks_gpu(
                first_tile_shape,
                tile_overlaps,
                margin_ratio=fixed_margin_ratio
            )
        elif blend_method == "dynamic":
            # Create all dynamic masks in one batch operation
            masks_batch = _create_batch_dynamic_masks_gpu(
                first_tile_shape,
                tile_overlaps,
                overlap_fraction=overlap_blend_fraction
            )
        else:
            raise ValueError(f"Unknown blend_method: {blend_method}")

        # Convert batch tensor to list for compatibility with existing code
        blend_masks = [masks_batch[i] for i in range(num_tiles)]

    # --- 3.5. Batch convert to float32 for better memory efficiency ---
    image_tiles_float = image_tiles.astype(cp.float32)

    # --- 3.6. VECTORIZED: Pre-calculate all position data ---
    positions_array = cp.array(positions, dtype=cp.float32)  # Shape: (N, 2)
    target_canvas_positions = positions_array - cp.array([canvas_min_x, canvas_min_y], dtype=cp.float32)

    # Vectorized calculation of integer and fractional parts for all tiles
    canvas_starts_int = cp.floor(target_canvas_positions).astype(cp.int32)  # Shape: (N, 2)
    fractional_parts = target_canvas_positions - canvas_starts_int  # Shape: (N, 2)
    subpixel_shifts = -fractional_parts  # Shape: (N, 2) - negative for scipy.ndimage.shift

    # --- 4. Place tiles with subpixel shifts (using pre-calculated values) ---
    for i in range(num_tiles):
        tile_float = image_tiles_float[i]

        # Use pre-calculated values (vectorized above)
        canvas_x_start_int = int(canvas_starts_int[i, 0].item())
        canvas_y_start_int = int(canvas_starts_int[i, 1].item())
        shift_x_subpixel = subpixel_shifts[i, 0]
        shift_y_subpixel = subpixel_shifts[i, 1]

        shifted_tile = subpixel_shift(tile_float, shift=(shift_y_subpixel, shift_x_subpixel), order=1, mode='constant', cval=0.0)

        # Apply tile-specific blending mask
        blended_tile = shifted_tile * blend_masks[i]

        # Define where this tile (and its mask) go on the canvas
        y_start_on_canvas = canvas_y_start_int
        y_end_on_canvas = y_start_on_canvas + tile_h
        x_start_on_canvas = canvas_x_start_int
        x_end_on_canvas = x_start_on_canvas + tile_w

        # Define what part of the tile to take (in case it goes off-canvas)
        tile_y_start_src = 0
        tile_y_end_src = tile_h
        tile_x_start_src = 0
        tile_x_end_src = tile_w

        # Adjust for tile parts that are off the canvas (negative start)
        if y_start_on_canvas < 0:
            tile_y_start_src = -y_start_on_canvas
            y_start_on_canvas = 0
        if x_start_on_canvas < 0:
            tile_x_start_src = -x_start_on_canvas
            x_start_on_canvas = 0

        # Adjust for tile parts that are off the canvas (positive end)
        if y_end_on_canvas > canvas_height:
            tile_y_end_src -= (y_end_on_canvas - canvas_height)
            y_end_on_canvas = canvas_height
        if x_end_on_canvas > canvas_width:
            tile_x_end_src -= (x_end_on_canvas - canvas_width)
            x_end_on_canvas = canvas_width

        # If the tile is entirely off-canvas after adjustments, skip
        if tile_y_start_src >= tile_y_end_src or tile_x_start_src >= tile_x_end_src:
            continue
        if y_start_on_canvas >= y_end_on_canvas or x_start_on_canvas >= x_end_on_canvas:
            continue

        # Add to accumulators
        composite_accum[y_start_on_canvas:y_end_on_canvas, x_start_on_canvas:x_end_on_canvas] += \
            blended_tile[tile_y_start_src:tile_y_end_src, tile_x_start_src:tile_x_end_src]

        weight_accum[y_start_on_canvas:y_end_on_canvas, x_start_on_canvas:x_end_on_canvas] += \
            blend_masks[i][tile_y_start_src:tile_y_end_src, tile_x_start_src:tile_x_end_src]

    # --- 5. Normalize + cast ---
    epsilon = 1e-7 # To avoid division by zero
    stitched_image_float = composite_accum / (weight_accum + epsilon)

    # Clip to 0-65535 and cast to uint16
    stitched_image_uint16 = cp.clip(stitched_image_float, 0, 65535).astype(cp.uint16)

    # Return as a 3D array with a single Z-slice
    return stitched_image_uint16.reshape(1, canvas_height.item(), canvas_width.item()) # .item() to convert 0-dim cupy array to scalar
