"""
CPU implementation of image assembly functions with fixed blending.
"""
from __future__ import annotations 

import logging
from typing import TYPE_CHECKING, List, Tuple, Union

from openhcs.core.memory.decorators import numpy as numpy_func
from openhcs.core.pipeline.function_contracts import special_inputs

# For type checking only
if TYPE_CHECKING:
    import numpy as np
    from scipy.ndimage import shift as subpixel_shift

# Import NumPy
import numpy as np  # type: ignore
from scipy.ndimage import shift as subpixel_shift  # type: ignore

logger = logging.getLogger(__name__)


def _get_all_overlapping_pairs(positions: "np.ndarray", tile_shape: tuple) -> list:
    """
    Find ALL overlapping tile pairs with edge directions.
    [Keep this exactly as it was - it works fine]
    """
    height, width = tile_shape
    N = positions.shape[0]

    if N <= 1:
        return []

    # Vectorized computation of ALL pairwise overlaps
    pos_i = positions[:, np.newaxis, :]
    pos_j = positions[np.newaxis, :, :]

    xi, yi = pos_i[:, :, 0], pos_i[:, :, 1]
    xj, yj = pos_j[:, :, 0], pos_j[:, :, 1]

    left_i, right_i = xi, xi + width
    top_i, bottom_i = yi, yi + height
    left_j, right_j = xj, xj + width
    top_j, bottom_j = yj, yj + height

    x_overlap = np.maximum(0, np.minimum(right_i, right_j) - np.maximum(left_i, left_j))
    y_overlap = np.maximum(0, np.minimum(bottom_i, bottom_j) - np.maximum(top_i, top_j))

    valid_overlap = (x_overlap > 0) & (y_overlap > 0) & (np.arange(N)[:, None] != np.arange(N)[None, :])

    edge_pairs = []
    overlapping_pairs = np.where(valid_overlap)

    for idx in range(len(overlapping_pairs[0])):
        i, j = overlapping_pairs[0][idx], overlapping_pairs[1][idx]

        x_overlap_val = float(x_overlap[i, j])
        y_overlap_val = float(y_overlap[i, j])

        xi_val, yi_val = positions[i, 0], positions[i, 1]
        xj_val, yj_val = positions[j, 0], positions[j, 1]

        if x_overlap_val > 0:
            if xj_val < xi_val:
                edge_pairs.append((i, j, 'left', x_overlap_val))
            elif xj_val > xi_val:
                edge_pairs.append((i, j, 'right', x_overlap_val))

        if y_overlap_val > 0:
            if yj_val < yi_val:
                edge_pairs.append((i, j, 'top', y_overlap_val))
            elif yj_val > yi_val:
                edge_pairs.append((i, j, 'bottom', y_overlap_val))

    return edge_pairs


def _create_fixed_blend_mask(
    tile_shape: tuple,
    edge_overlaps: dict,
    margin_ratio: float = 0.1
) -> "np.ndarray":
    """
    Create blend mask with FIXED margin ratio using WORKING logic from old version.
    CRITICAL: Uses endpoint=False like the old working version.
    """
    height, width = tile_shape

    # Create 1D weights
    y_weight = np.ones(height, dtype=np.float32)
    x_weight = np.ones(width, dtype=np.float32)

    # Fixed margins (same as old working version)
    margin_pixels_y = int(height * margin_ratio)
    margin_pixels_x = int(width * margin_ratio)

    # Apply gradients ONLY where there are overlaps (same as old working version)
    # CRITICAL: endpoint=False (this is what made the old version work!)
    if 'top' in edge_overlaps and margin_pixels_y > 0:
        y_weight[:margin_pixels_y] = np.linspace(0, 1, margin_pixels_y, endpoint=False)

    if 'bottom' in edge_overlaps and margin_pixels_y > 0:
        y_weight[-margin_pixels_y:] = np.linspace(1, 0, margin_pixels_y, endpoint=False)

    if 'left' in edge_overlaps and margin_pixels_x > 0:
        x_weight[:margin_pixels_x] = np.linspace(0, 1, margin_pixels_x, endpoint=False)

    if 'right' in edge_overlaps and margin_pixels_x > 0:
        x_weight[-margin_pixels_x:] = np.linspace(1, 0, margin_pixels_x, endpoint=False)

    # Use outer product (same as old working version)
    mask = np.outer(y_weight, x_weight)
    return mask.astype(np.float32)


def _create_dynamic_blend_mask(
    tile_shape: tuple,
    edge_overlaps: dict,
    overlap_fraction: float = 1.0
) -> "np.ndarray":
    """
    Create blend mask based on actual overlap amounts using WORKING logic from old version.
    CRITICAL: Uses endpoint=False and same logic as old working version.
    """
    height, width = tile_shape

    # Create 1D weights
    y_weight = np.ones(height, dtype=np.float32)
    x_weight = np.ones(width, dtype=np.float32)

    # Process each edge based on actual overlap (same as old working version)
    # CRITICAL: endpoint=False (this is what made the old version work!)
    if 'top' in edge_overlaps:
        overlap_pixels = int(edge_overlaps['top'] * overlap_fraction)
        if overlap_pixels > 0:
            y_weight[:overlap_pixels] = np.linspace(0, 1, overlap_pixels, endpoint=False)

    if 'bottom' in edge_overlaps:
        overlap_pixels = int(edge_overlaps['bottom'] * overlap_fraction)
        if overlap_pixels > 0:
            y_weight[-overlap_pixels:] = np.linspace(1, 0, overlap_pixels, endpoint=False)

    if 'left' in edge_overlaps:
        overlap_pixels = int(edge_overlaps['left'] * overlap_fraction)
        if overlap_pixels > 0:
            x_weight[:overlap_pixels] = np.linspace(0, 1, overlap_pixels, endpoint=False)

    if 'right' in edge_overlaps:
        overlap_pixels = int(edge_overlaps['right'] * overlap_fraction)
        if overlap_pixels > 0:
            x_weight[-overlap_pixels:] = np.linspace(1, 0, overlap_pixels, endpoint=False)

    # Use outer product (same as old working version)
    mask = np.outer(y_weight, x_weight)
    return mask.astype(np.float32)


@special_inputs("positions")
@numpy_func
def assemble_stack_cpu(
    image_tiles: "np.ndarray",
    positions: Union[List[Tuple[float, float]], "np.ndarray"],
    blend_method: str = "fixed",
    fixed_margin_ratio: float = 0.1,
    overlap_blend_fraction: float = 1.0
) -> "np.ndarray":
    """
    Assembles tiles with simple, working blending approach.
    
    Args:
        image_tiles: 3D array of tiles (N, H, W)
        positions: List of (x, y) tuples or 2D array [N, 2]
        blend_method: "none", "fixed", or "dynamic"
        fixed_margin_ratio: Ratio for fixed blending (e.g., 0.1 = 10%)
        overlap_blend_fraction: For dynamic mode, fraction of overlap to blend
        use_endpoint: Whether to include endpoint in gradients
    """
    # --- 1. Validate inputs ---
    if not isinstance(image_tiles, np.ndarray) or image_tiles.ndim != 3:
        raise TypeError("image_tiles must be a 3D NumPy ndarray of shape (N, H, W).")
    
    if image_tiles.shape[0] == 0:
        logger.warning("image_tiles array is empty (0 tiles).")
        return np.array([[[]]], dtype=np.uint16)

    # Convert positions to numpy
    if isinstance(positions, list):
        if not positions or not isinstance(positions[0], tuple) or len(positions[0]) != 2:
            raise TypeError("positions must be a list of (x, y) tuples.")
        positions = np.array(positions, dtype=np.float32)
    else:
        if not isinstance(positions, np.ndarray):
            positions = to_numpy(positions)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise TypeError("positions must be an array of shape [N, 2].")

    if image_tiles.shape[0] != positions.shape[0]:
        raise ValueError(f"Mismatch: {image_tiles.shape[0]} tiles vs {positions.shape[0]} positions.")

    num_tiles, tile_h, tile_w = image_tiles.shape
    tile_shape = (tile_h, tile_w)

    # Convert to float32
    image_tiles_float = image_tiles.astype(np.float32)

    # --- 2. Compute canvas bounds ---
    min_x = np.floor(np.min(positions[:, 0])).astype(int)
    min_y = np.floor(np.min(positions[:, 1])).astype(int)
    max_x = np.ceil(np.max(positions[:, 0]) + tile_w).astype(int)
    max_y = np.ceil(np.max(positions[:, 1]) + tile_h).astype(int)

    canvas_width = max_x - min_x
    canvas_height = max_y - min_y

    if canvas_width <= 0 or canvas_height <= 0:
        logger.warning(f"Invalid canvas size: {canvas_height}x{canvas_width}")
        return np.array([], dtype=np.uint16)

    composite_accum = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    weight_accum = np.zeros((canvas_height, canvas_width), dtype=np.float32)

    # --- 3. Create blend masks ---
    if blend_method == "none":
        blend_masks = [np.ones(tile_shape, dtype=np.float32) for _ in range(num_tiles)]
        
    else:
        # Find overlaps
        edge_pairs = _get_all_overlapping_pairs(positions, tile_shape)
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
        
        # Create masks using WORKING logic from old version
        blend_masks = []
        for i in range(num_tiles):
            if blend_method == "fixed":
                mask = _create_fixed_blend_mask(
                    tile_shape,
                    tile_overlaps[i],
                    margin_ratio=fixed_margin_ratio
                )
            elif blend_method == "dynamic":
                mask = _create_dynamic_blend_mask(
                    tile_shape,
                    tile_overlaps[i],
                    overlap_fraction=overlap_blend_fraction
                )
            else:
                raise ValueError(f"Unknown blend_method: {blend_method}")

            blend_masks.append(mask)

    # --- 4. Place tiles ---
    for i in range(num_tiles):
        tile = image_tiles_float[i]
        pos_x, pos_y = positions[i]

        # Canvas position
        target_x = pos_x - min_x
        target_y = pos_y - min_y

        # Integer and fractional parts
        x_int = int(np.floor(target_x))
        y_int = int(np.floor(target_y))
        x_frac = target_x - x_int
        y_frac = target_y - y_int

        # Subpixel shift
        shift_x = -x_frac
        shift_y = -y_frac
        
        shifted_tile = subpixel_shift(
            tile, 
            shift=(shift_y, shift_x), 
            order=1, 
            mode='constant', 
            cval=0.0
        )

        # Apply blend mask
        blended_tile = shifted_tile * blend_masks[i]

        # Canvas bounds
        y_start = y_int
        y_end = y_start + tile_h
        x_start = x_int
        x_end = x_start + tile_w

        # Tile bounds (for edge cases)
        tile_y_start = 0
        tile_y_end = tile_h
        tile_x_start = 0
        tile_x_end = tile_w

        # Clip to canvas
        if y_start < 0:
            tile_y_start = -y_start
            y_start = 0
        if x_start < 0:
            tile_x_start = -x_start
            x_start = 0
        if y_end > canvas_height:
            tile_y_end -= (y_end - canvas_height)
            y_end = canvas_height
        if x_end > canvas_width:
            tile_x_end -= (x_end - canvas_width)
            x_end = canvas_width

        # Skip if out of bounds
        if (tile_y_start >= tile_y_end or tile_x_start >= tile_x_end or
            y_start >= y_end or x_start >= x_end):
            continue

        # Accumulate
        composite_accum[y_start:y_end, x_start:x_end] += \
            blended_tile[tile_y_start:tile_y_end, tile_x_start:tile_x_end]
        
        weight_accum[y_start:y_end, x_start:x_end] += \
            blend_masks[i][tile_y_start:tile_y_end, tile_x_start:tile_x_end]

    # --- 5. Normalize ---
    epsilon = 1e-7
    stitched = composite_accum / (weight_accum + epsilon)
    
    # Convert to uint16
    stitched_uint16 = np.clip(stitched, 0, 65535).astype(np.uint16)
    
    return stitched_uint16.reshape(1, canvas_height, canvas_width)


def to_numpy(tensor):
    """Convert various tensor types to numpy"""
    if hasattr(tensor, 'dtype') and tensor.__class__.__module__ == 'numpy':
        return tensor
    if hasattr(tensor, 'get'):  # CuPy
        return tensor.get()
    if hasattr(tensor, 'detach'):  # PyTorch
        return tensor.detach().cpu().numpy()
    if hasattr(tensor, 'numpy') and hasattr(tensor, 'device'):  # TF
        return tensor.numpy()
    raise ValueError(f"Unsupported tensor type: {type(tensor)}")