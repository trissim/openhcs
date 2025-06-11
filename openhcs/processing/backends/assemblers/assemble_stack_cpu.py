"""
CPU implementation of image assembly functions.

This module provides CPU-based functions for assembling microscopy images
using NumPy. It handles subpixel positioning and blending of image tiles.
"""

import logging
from typing import TYPE_CHECKING

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

def _create_blend_mask(tile_shape: tuple, blend_method: str = "rectangular", blend_radius: float = 10.0) -> "np.ndarray":  # type: ignore
    """
    Creates a 2D blending mask for tile assembly.

    Args:
        tile_shape: (height, width) of the tile
        blend_method: Blending method - "none", "gaussian", "rectangular", "linear_edge"
        blend_radius: Blending parameter (pixels for feathering)

    Returns:
        2D mask array with blending weights
    """
    height, width = tile_shape

    if blend_method == "none" or blend_radius <= 0:
        # No blending - uniform weights
        return np.ones(tile_shape, dtype=np.float32)

    elif blend_method == "gaussian":
        # Original circular Gaussian mask (for compatibility)
        coords_y = np.arange(height) - (height - 1) / 2.0
        coords_x = np.arange(width) - (width - 1) / 2.0
        yy, xx = np.meshgrid(coords_y, coords_x, indexing='ij')
        mask = np.exp(-(xx**2 + yy**2) / (2 * blend_radius**2))
        mask = mask / np.max(mask)  # Center will be 1.0
        return mask.astype(np.float32)

    elif blend_method == "rectangular":
        # Rectangular feathering - blend near edges only
        mask = np.ones(tile_shape, dtype=np.float32)
        blend_pixels = int(blend_radius)

        if blend_pixels > 0:
            # Create distance-to-edge maps
            y_coords = np.arange(height, dtype=np.float32)
            x_coords = np.arange(width, dtype=np.float32)

            # Distance from top/bottom edges
            dist_from_top = y_coords
            dist_from_bottom = height - 1 - y_coords
            y_edge_dist = np.minimum(dist_from_top, dist_from_bottom)

            # Distance from left/right edges
            dist_from_left = x_coords
            dist_from_right = width - 1 - x_coords
            x_edge_dist = np.minimum(dist_from_left, dist_from_right)

            # Create 2D distance maps
            y_dist_2d = y_edge_dist[:, np.newaxis]  # Shape: (height, 1)
            x_dist_2d = x_edge_dist[np.newaxis, :]  # Shape: (1, width)

            # Feathering weights based on distance to nearest edge
            y_weight = np.minimum(y_dist_2d / blend_pixels, 1.0)
            x_weight = np.minimum(x_dist_2d / blend_pixels, 1.0)

            # Combine weights (minimum ensures feathering at ALL edges)
            mask = np.minimum(y_weight, x_weight)

        return mask.astype(np.float32)

    elif blend_method == "linear_edge":
        # Linear falloff from edges (softer than rectangular)
        mask = np.ones(tile_shape, dtype=np.float32)
        blend_pixels = int(blend_radius)

        if blend_pixels > 0:
            # Create coordinate grids
            y_coords = np.arange(height, dtype=np.float32)
            x_coords = np.arange(width, dtype=np.float32)

            # Linear weights from edges
            y_weight_top = np.minimum(y_coords / blend_pixels, 1.0)
            y_weight_bottom = np.minimum((height - 1 - y_coords) / blend_pixels, 1.0)
            x_weight_left = np.minimum(x_coords / blend_pixels, 1.0)
            x_weight_right = np.minimum((width - 1 - x_coords) / blend_pixels, 1.0)

            # Create 2D weight maps
            y_weight = np.minimum(y_weight_top[:, np.newaxis], y_weight_bottom[:, np.newaxis])
            x_weight = np.minimum(x_weight_left[np.newaxis, :], x_weight_right[np.newaxis, :])

            # Combine weights multiplicatively for smoother transitions
            mask = y_weight * x_weight

        return mask.astype(np.float32)

    else:
        raise ValueError(f"Unknown blend_method: {blend_method}. Supported: 'none', 'gaussian', 'rectangular', 'linear_edge'")


def _create_gaussian_blend_mask(tile_shape: tuple, blend_radius: float) -> "np.ndarray":  # type: ignore
    """
    Legacy function for backward compatibility.
    Use _create_blend_mask with blend_method="gaussian" instead.
    """
    return _create_blend_mask(tile_shape, "gaussian", blend_radius)


@special_inputs("positions") # The input name is "positions"
@numpy_func
def assemble_stack_cpu(
    image_tiles: "np.ndarray",  # type: ignore
    positions: "np.ndarray",  # type: ignore # Renamed from positions_xy
    blend_radius: float = 10.0,
    blend_method: str = "rectangular"
) -> "np.ndarray":  # type: ignore
    """
    Assembles a 3D stack of 2D image tiles (N, H, W) into a composited 2D image,
    returned as a 3D array (1, H_canvas, W_canvas), using subpixel XY positions
    and configurable blending.

    Args:
        image_tiles: 3D array of tiles (N, H, W)
        positions: 2D array of tile positions (N, 2) as [x, y] coordinates
        blend_radius: Blending parameter in pixels (default: 10.0)
        blend_method: Blending method (default: "rectangular")
            - "none": No blending, uniform weights
            - "gaussian": Circular Gaussian mask (original behavior)
            - "rectangular": Rectangular feathering from edges
            - "linear_edge": Linear falloff from edges

    Returns:
        3D array (1, H_canvas, W_canvas) with assembled image
    """
    # --- 1. Validate and standardize inputs ---
    if not isinstance(positions, np.ndarray):
        positions = to_numpy(positions)

    if not isinstance(image_tiles, np.ndarray) or image_tiles.ndim != 3:
        raise TypeError("image_tiles must be a 3D NumPy ndarray of shape (N, H, W).")
    if image_tiles.shape[0] == 0:
        logger.warning("image_tiles array is empty (0 tiles). Returning an empty array.")
        return np.array([[[]]], dtype=np.uint16) # Shape (1,0,0) to indicate empty 3D

    if not isinstance(positions, np.ndarray) or positions.ndim != 2 or positions.shape[1] != 2:
        raise TypeError("positions must be a NumPy ndarray of shape [N, 2].") # Updated error message
    if image_tiles.shape[0] != positions.shape[0]:
        raise ValueError(f"Mismatch between number of image_tiles ({image_tiles.shape[0]}) and positions ({positions.shape[0]}).") # Updated error message

    num_tiles, tile_h, tile_w = image_tiles.shape
    first_tile_shape = (tile_h, tile_w) # Used for blend mask, assumes all tiles same H, W

    # Convert tiles to float32 for accumulation
    image_tiles_float = image_tiles.astype(np.float32)

    # --- 2. Compute canvas bounds ---
    # positions_xy are for top-left corners.
    # Add tile dimensions to get bottom-right corners for each tile.
    # positions_xy[:, 0] is X (width dimension), positions_xy[:, 1] is Y (height dimension)

    # Min/max X coordinates of tile top-left corners
    min_x_pos = np.min(positions[:, 0])
    max_x_pos = np.max(positions[:, 0])

    # Min/max Y coordinates of tile top-left corners
    min_y_pos = np.min(positions[:, 1])
    max_y_pos = np.max(positions[:, 1])

    # Canvas dimensions need to encompass all tiles
    # Canvas origin will be (min_x_pos_rounded_down, min_y_pos_rounded_down)
    # Max extent is max_pos + tile_dim
    canvas_min_x = np.floor(min_x_pos).astype(int)
    canvas_min_y = np.floor(min_y_pos).astype(int)

    canvas_max_x = np.ceil(max_x_pos + tile_w).astype(int)
    canvas_max_y = np.ceil(max_y_pos + tile_h).astype(int)

    canvas_width = canvas_max_x - canvas_min_x
    canvas_height = canvas_max_y - canvas_min_y

    if canvas_width <= 0 or canvas_height <= 0:
        logger.warning(f"Calculated canvas dimensions are non-positive ({canvas_height}x{canvas_width}). Check positions and tile sizes.")
        return np.array([], dtype=np.uint16)

    composite_accum = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    weight_accum = np.zeros((canvas_height, canvas_width), dtype=np.float32)

    # --- 3. Generate blending mask ---
    # Assuming all tiles have the same shape as the first tile for the blend mask
    blend_mask = _create_blend_mask(first_tile_shape, blend_method, blend_radius)

    # --- 4. Place tiles with subpixel shifts ---
    for i in range(num_tiles):
        tile_float = image_tiles_float[i]
        # Shape validation for individual tiles is implicitly handled by the 3D array input check,
        # assuming all slices (tiles) in the 3D array have consistent H, W.

        pos_x, pos_y = positions[i] # Original subpixel top-left of this tile

        # Calculate integer and fractional parts of the shift
        # The shift for subpixel_shift is (shift_y, shift_x)
        # We want to place the tile's origin (0,0) at (pos_x, pos_y) relative to global origin.
        # The canvas origin is (canvas_min_x, canvas_min_y).
        # So, target top-left on canvas is (pos_x - canvas_min_x, pos_y - canvas_min_y)

        target_canvas_x_float = pos_x - canvas_min_x
        target_canvas_y_float = pos_y - canvas_min_y

        # The subpixel_shift function shifts the *content* of the array.
        # If we want the tile's (0,0) to land on a subpixel grid,
        # the shift should be the negative of the fractional part of the target coordinate.
        # E.g., if target is 10.3, we place at 10, and shift content by -0.3.

        # Integer part for slicing
        canvas_x_start_int = np.floor(target_canvas_x_float).astype(int)
        canvas_y_start_int = np.floor(target_canvas_y_float).astype(int)

        # Fractional part for subpixel shift (scipy.ndimage.shift convention: positive shifts move data "down/right")
        # We want to shift the tile's origin. If target is 10.3, we want pixel 0 to effectively start at 10.3.
        # If we place the tile starting at integer coord 10, its content needs to be shifted by +0.3.
        # However, scipy.ndimage.shift shifts data: a positive shift moves data to higher indices.
        # If tile's (0,0) should be at (10.3, 20.7)
        # Place tile at canvas_int_coords = (floor(10.3), floor(20.7)) = (10, 20)
        # The content of the tile needs to be shifted by (10.3-10, 20.7-20) = (0.3, 0.7)
        # But subpixel_shift shifts data values, not coordinates.
        # A shift of (dy, dx) means output[iy,ix] = input[iy-dy, ix-dx]
        # So, if we want output at (canvas_y_start_int + y_tile, canvas_x_start_int + x_tile)
        # to correspond to input at (y_tile - (target_canvas_y_float - canvas_y_start_int), x_tile - (target_canvas_x_float - canvas_x_start_int))
        # The shift values are -(target_canvas_y_float - canvas_y_start_int) and -(target_canvas_x_float - canvas_x_start_int)

        shift_y_subpixel = -(target_canvas_y_float - canvas_y_start_int)
        shift_x_subpixel = -(target_canvas_x_float - canvas_x_start_int)

        shifted_tile = subpixel_shift(tile_float, shift=(shift_y_subpixel, shift_x_subpixel), order=1, mode='constant', cval=0.0)

        # Apply blending mask
        blended_tile = shifted_tile * blend_mask

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
            blend_mask[tile_y_start_src:tile_y_end_src, tile_x_start_src:tile_x_end_src]

    # --- 5. Normalize + cast ---
    epsilon = 1e-7 # To avoid division by zero
    stitched_image_float = composite_accum / (weight_accum + epsilon)

    # Clip to 0-65535 and cast to uint16
    stitched_image_uint16 = np.clip(stitched_image_float, 0, 65535).astype(np.uint16)

    # Return as a 3D array with a single Z-slice
    return stitched_image_uint16.reshape(1, canvas_height, canvas_width)

def to_numpy(tensor):
    """Convert CuPy, PyTorch, JAX, or TensorFlow tensor to numpy"""
    
    # Already numpy
    if hasattr(tensor, 'dtype') and tensor.__class__.__module__ == 'numpy':
        return tensor
    
    # CuPy
    if hasattr(tensor, 'get'):  # CuPy arrays have .get() method
        return tensor.get()
    
    # PyTorch
    if hasattr(tensor, 'detach'):  # PyTorch tensors have .detach()
        return tensor.detach().cpu().numpy()
    
#    # JAX
#    if hasattr(tensor, 'device'):  # JAX arrays have .device
#        import jax
#        return jax.device_get(tensor)
    
    # TensorFlow
    if hasattr(tensor, 'numpy') and hasattr(tensor, 'device'):  # TF tensors
        return tensor.numpy()
    
    raise ValueError(f"Unsupported tensor type: {type(tensor)}")
