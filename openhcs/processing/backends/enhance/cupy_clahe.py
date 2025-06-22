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
def clahe_2d(
    image: "cp.ndarray",
    clip_limit: float = 2.0,
    tile_grid_size: tuple = None,
    nbins: int = None,
    adaptive_bins: bool = True,
    adaptive_tiles: bool = True
) -> "cp.ndarray":
    """
    Optimized 2D CLAHE with vectorized bilinear interpolation.
    """
    
    result = cp.zeros_like(image)
    
    for z in range(image.shape[0]):
        slice_2d = image[z]
        height, width = slice_2d.shape
        
        # Adaptive parameters
        if nbins is None:
            if adaptive_bins:
                data_range = float(cp.max(slice_2d) - cp.min(slice_2d))
                adaptive_nbins = min(512, max(64, int(cp.sqrt(data_range))))
            else:
                adaptive_nbins = 256
        else:
            adaptive_nbins = nbins
            
        if tile_grid_size is None:
            if adaptive_tiles:
                target_tile_size = 80
                adaptive_tile_rows = max(2, min(16, height // target_tile_size))
                adaptive_tile_cols = max(2, min(16, width // target_tile_size))
                adaptive_tile_grid = (adaptive_tile_rows, adaptive_tile_cols)
            else:
                adaptive_tile_grid = (8, 8)
        else:
            adaptive_tile_grid = tile_grid_size
            
        result[z] = _clahe_2d_vectorized(
            slice_2d, clip_limit, adaptive_tile_grid, adaptive_nbins
        )
    
    return result

def _clahe_2d_vectorized(
    image: "cp.ndarray",
    clip_limit: float,
    tile_grid_size: tuple,
    nbins: int
) -> "cp.ndarray":
    """
    Vectorized CLAHE implementation for 2D images.
    """
    if image.ndim != 2:
        raise ValueError("Input must be 2D array")
    
    height, width = image.shape
    tile_rows, tile_cols = tile_grid_size
    
    # Calculate tile dimensions
    tile_height = height // tile_rows
    tile_width = width // tile_cols
    
    # Ensure we have valid tiles
    if tile_height < 1 or tile_width < 1:
        raise ValueError(f"Image too small for {tile_rows}x{tile_cols} tiles")
    
    # Calculate crop dimensions
    crop_height = tile_height * tile_rows
    crop_width = tile_width * tile_cols
    image_crop = image[:crop_height, :crop_width]
    
    # Calculate actual clip limit
    actual_clip_limit = max(1, int(clip_limit * tile_height * tile_width / nbins))
    
    # Get value range
    min_val = float(cp.min(image_crop))
    max_val = float(cp.max(image_crop))
    
    if max_val <= min_val:
        return image.astype(image.dtype)  # Constant image
    
    # Compute tile CDFs
    tile_cdfs = _compute_tile_cdfs_2d(
        image_crop, tile_rows, tile_cols, tile_height, tile_width,
        nbins, actual_clip_limit, min_val, max_val
    )
    
    # Apply vectorized interpolation
    result = _apply_vectorized_interpolation_2d(
        image_crop, tile_cdfs, tile_rows, tile_cols,
        tile_height, tile_width, nbins, min_val, max_val
    )
    
    # Handle original image size
    if result.shape != image.shape:
        full_result = cp.zeros_like(image, dtype=result.dtype)
        full_result[:crop_height, :crop_width] = result
        
        # Fill remaining areas by replication
        if crop_height < height:
            full_result[crop_height:, :crop_width] = result[-1:, :]
        if crop_width < width:
            full_result[:crop_height, crop_width:] = result[:, -1:]
        if crop_height < height and crop_width < width:
            full_result[crop_height:, crop_width:] = result[-1, -1]
        result = full_result
    
    return result.astype(image.dtype)

def _compute_tile_cdfs_2d(
    image: "cp.ndarray",
    tile_rows: int,
    tile_cols: int,
    tile_height: int,
    tile_width: int,
    nbins: int,
    clip_limit: int,
    min_val: float,
    max_val: float
) -> "cp.ndarray":
    """Compute CDFs for all tiles efficiently."""
    
    tile_cdfs = cp.zeros((tile_rows, tile_cols, nbins), dtype=cp.float32)
    
    # Precompute bin edges
    bin_edges = cp.linspace(min_val, max_val, nbins + 1, dtype=cp.float32)
    bin_width = (max_val - min_val) / nbins
    
    for row in range(tile_rows):
        for col in range(tile_cols):
            # Extract tile
            y_start = row * tile_height
            y_end = (row + 1) * tile_height
            x_start = col * tile_width
            x_end = (col + 1) * tile_width
            
            tile = image[y_start:y_end, x_start:x_end]
            
            # Compute histogram
            hist, _ = cp.histogram(tile, bins=bin_edges)
            
            # Clip and redistribute
            hist = _clip_histogram_optimized(hist, clip_limit)
            
            # Compute CDF and normalize properly
            cdf = cp.cumsum(hist, dtype=cp.float32)
            if cdf[-1] > 0:
                # Normalize to [0, 1] then scale to output range
                cdf = cdf / cdf[-1]
                # Map to intensity values (proper CLAHE transformation)
                tile_cdfs[row, col, :] = min_val + cdf * (max_val - min_val)
            else:
                tile_cdfs[row, col, :] = min_val
    
    return tile_cdfs

def _apply_vectorized_interpolation_2d(
    image: "cp.ndarray",
    tile_cdfs: "cp.ndarray",
    tile_rows: int,
    tile_cols: int,
    tile_height: int,
    tile_width: int,
    nbins: int,
    min_val: float,
    max_val: float
) -> "cp.ndarray":
    """Vectorized bilinear interpolation."""
    
    height, width = image.shape
    
    # Create coordinate grids
    y_coords, x_coords = cp.meshgrid(
        cp.arange(height, dtype=cp.float32),
        cp.arange(width, dtype=cp.float32),
        indexing='ij'
    )
    
    # Calculate tile centers
    tile_centers_y = cp.arange(tile_rows, dtype=cp.float32) * tile_height + tile_height // 2
    tile_centers_x = cp.arange(tile_cols, dtype=cp.float32) * tile_width + tile_width // 2
    
    # Find surrounding tiles for each pixel (vectorized)
    tile_y_low = cp.searchsorted(tile_centers_y, y_coords.flatten()) - 1
    tile_x_low = cp.searchsorted(tile_centers_x, x_coords.flatten()) - 1
    
    # Clamp to valid ranges
    tile_y_low = cp.clip(tile_y_low, 0, tile_rows - 2).reshape(height, width)
    tile_x_low = cp.clip(tile_x_low, 0, tile_cols - 2).reshape(height, width)
    
    tile_y_high = tile_y_low + 1
    tile_x_high = tile_x_low + 1
    
    # Convert pixel values to bin indices (vectorized)
    normalized_values = (image - min_val) / (max_val - min_val)
    bin_indices = cp.clip(
        (normalized_values * (nbins - 1)).astype(cp.int32),
        0, nbins - 1
    )
    
    # Calculate interpolation weights (vectorized)
    center_y_low = tile_centers_y[tile_y_low]
    center_y_high = tile_centers_y[tile_y_high]
    center_x_low = tile_centers_x[tile_x_low]
    center_x_high = tile_centers_x[tile_x_high]
    
    # Avoid division by zero
    dy = center_y_high - center_y_low
    dx = center_x_high - center_x_low
    
    wy = cp.where(dy > 0, (y_coords - center_y_low) / dy, 0.0)
    wx = cp.where(dx > 0, (x_coords - center_x_low) / dx, 0.0)
    
    # Clamp weights
    wy = cp.clip(wy, 0.0, 1.0)
    wx = cp.clip(wx, 0.0, 1.0)
    
    # Get transformation values (this is the tricky part - need advanced indexing)
    val_tl = tile_cdfs[tile_y_low, tile_x_low, bin_indices]
    val_tr = tile_cdfs[tile_y_low, tile_x_high, bin_indices]
    val_bl = tile_cdfs[tile_y_high, tile_x_low, bin_indices]
    val_br = tile_cdfs[tile_y_high, tile_x_high, bin_indices]
    
    # Bilinear interpolation (vectorized)
    val_top = (1 - wx) * val_tl + wx * val_tr
    val_bottom = (1 - wx) * val_bl + wx * val_br
    result = (1 - wy) * val_top + wy * val_bottom
    
    return result

def _clip_histogram_optimized(hist: "cp.ndarray", clip_limit: int) -> "cp.ndarray":
    """Optimized histogram clipping."""
    if clip_limit <= 0:
        return hist
    
    # Convert to float for precise calculations
    hist_float = hist.astype(cp.float32)
    
    # Find excess and clip
    excess = cp.maximum(hist_float - clip_limit, 0)
    total_excess = cp.sum(excess)
    
    clipped_hist = cp.minimum(hist_float, clip_limit)
    
    # Redistribute excess uniformly
    if total_excess > 0:
        nbins = len(hist)
        redistribution = total_excess / nbins
        clipped_hist += redistribution
        
        # Handle overflow after redistribution (iterative clipping)
        for _ in range(3):  # Max 3 iterations should be enough
            overflow = cp.maximum(clipped_hist - clip_limit, 0)
            total_overflow = cp.sum(overflow)
            
            if total_overflow < 1e-6:
                break
                
            clipped_hist = cp.minimum(clipped_hist, clip_limit)
            # Redistribute overflow to non-saturated bins
            non_saturated = clipped_hist < clip_limit
            if cp.any(non_saturated):
                available_space = cp.sum(cp.maximum(clip_limit - clipped_hist, 0))
                if available_space > 0:
                    redistrib_factor = min(1.0, total_overflow / available_space)
                    clipped_hist += cp.where(
                        non_saturated,
                        redistrib_factor * cp.maximum(clip_limit - clipped_hist, 0),
                        0
                    )
    
    return clipped_hist.astype(hist.dtype)

@cupy_func
def clahe_3d(
    stack: "cp.ndarray",
    clip_limit: float = 2.0,
    tile_grid_size_3d: tuple = None,
    nbins: int = None,
    adaptive_bins: bool = True,
    adaptive_tiles: bool = True,
    memory_efficient: bool = True
) -> "cp.ndarray":
    """
    Optimized 3D CLAHE with vectorized trilinear interpolation.
    
    Args:
        stack: 3D CuPy array of shape (Z, Y, X)
        clip_limit: Threshold for contrast limiting
        tile_grid_size_3d: Number of tiles (z_tiles, y_tiles, x_tiles)
        nbins: Number of histogram bins
        adaptive_bins: Whether to adapt bins based on data range
        adaptive_tiles: Whether to adapt tile size based on volume dimensions
        memory_efficient: Use chunked processing for large volumes
    """
    
    depth, height, width = stack.shape
    
    # Adaptive parameters
    if nbins is None:
        if adaptive_bins:
            data_range = float(cp.max(stack) - cp.min(stack))
            adaptive_nbins = min(512, max(128, int(cp.cbrt(data_range * 64))))
        else:
            adaptive_nbins = 256
    else:
        adaptive_nbins = nbins
        
    if tile_grid_size_3d is None:
        if adaptive_tiles:
            target_tile_size = 48
            adaptive_z_tiles = max(1, min(depth // 4, depth // target_tile_size))
            adaptive_y_tiles = max(2, min(8, height // target_tile_size))
            adaptive_x_tiles = max(2, min(8, width // target_tile_size))
            adaptive_tile_grid_3d = (adaptive_z_tiles, adaptive_y_tiles, adaptive_x_tiles)
        else:
            adaptive_tile_grid_3d = (max(1, depth // 8), 4, 4)
    else:
        adaptive_tile_grid_3d = tile_grid_size_3d
    
    # Check memory requirements and use chunked processing if needed
    total_voxels = depth * height * width
    if memory_efficient and total_voxels > 512**3:  # ~134M voxels threshold
        return _clahe_3d_chunked(stack, clip_limit, adaptive_tile_grid_3d, adaptive_nbins)
    else:
        return _clahe_3d_vectorized(stack, clip_limit, adaptive_tile_grid_3d, adaptive_nbins)

def _clahe_3d_vectorized(
    stack: "cp.ndarray",
    clip_limit: float,
    tile_grid_size_3d: tuple,
    nbins: int
) -> "cp.ndarray":
    """
    Full vectorized 3D CLAHE implementation.
    """
    depth, height, width = stack.shape
    tile_z, tile_y, tile_x = tile_grid_size_3d
    
    # Calculate 3D tile dimensions
    tile_depth = max(1, depth // tile_z)
    tile_height = max(4, height // tile_y)
    tile_width = max(4, width // tile_x)
    
    # Ensure valid tiles
    if tile_depth < 1 or tile_height < 1 or tile_width < 1:
        raise ValueError(f"Volume too small for {tile_z}x{tile_y}x{tile_x} tiles")
    
    # Recalculate actual number of tiles
    actual_tile_z = depth // tile_depth
    actual_tile_y = height // tile_height
    actual_tile_x = width // tile_width
    
    # Calculate crop dimensions
    crop_depth = tile_depth * actual_tile_z
    crop_height = tile_height * actual_tile_y
    crop_width = tile_width * actual_tile_x
    stack_crop = stack[:crop_depth, :crop_height, :crop_width]
    
    # Calculate actual clip limit
    voxels_per_tile = tile_depth * tile_height * tile_width
    actual_clip_limit = max(1, int(clip_limit * voxels_per_tile / nbins))
    
    # Get value range
    min_val = float(cp.min(stack_crop))
    max_val = float(cp.max(stack_crop))
    
    if max_val <= min_val:
        return stack.astype(stack.dtype)  # Constant volume
    
    # Compute 3D tile CDFs
    tile_cdfs = _compute_tile_cdfs_3d(
        stack_crop, actual_tile_z, actual_tile_y, actual_tile_x,
        tile_depth, tile_height, tile_width,
        nbins, actual_clip_limit, min_val, max_val
    )
    
    # Apply vectorized trilinear interpolation
    result = _apply_vectorized_trilinear_interpolation(
        stack_crop, tile_cdfs, actual_tile_z, actual_tile_y, actual_tile_x,
        tile_depth, tile_height, tile_width, nbins, min_val, max_val
    )
    
    # Handle original stack size
    if result.shape != stack.shape:
        full_result = cp.zeros_like(stack, dtype=result.dtype)
        full_result[:crop_depth, :crop_height, :crop_width] = result
        
        # Fill remaining regions efficiently
        _fill_3d_boundaries(full_result, result, crop_depth, crop_height, crop_width, 
                          depth, height, width)
        result = full_result
    
    return result.astype(stack.dtype)

def _compute_tile_cdfs_3d(
    stack: "cp.ndarray",
    tile_z: int,
    tile_y: int,
    tile_x: int,
    tile_depth: int,
    tile_height: int,
    tile_width: int,
    nbins: int,
    clip_limit: int,
    min_val: float,
    max_val: float
) -> "cp.ndarray":
    """Compute CDFs for all 3D tiles efficiently."""
    
    tile_cdfs = cp.zeros((tile_z, tile_y, tile_x, nbins), dtype=cp.float32)
    
    # Precompute bin edges
    bin_edges = cp.linspace(min_val, max_val, nbins + 1, dtype=cp.float32)
    
    for z_idx in range(tile_z):
        for y_idx in range(tile_y):
            for x_idx in range(tile_x):
                # Extract 3D tile
                z_start = z_idx * tile_depth
                z_end = (z_idx + 1) * tile_depth
                y_start = y_idx * tile_height
                y_end = (y_idx + 1) * tile_height
                x_start = x_idx * tile_width
                x_end = (x_idx + 1) * tile_width
                
                tile_3d = stack[z_start:z_end, y_start:y_end, x_start:x_end]
                
                # Compute 3D histogram efficiently
                hist, _ = cp.histogram(tile_3d.ravel(), bins=bin_edges)
                
                # Clip and redistribute
                hist = _clip_histogram_optimized(hist, clip_limit)
                
                # Compute CDF and normalize properly
                cdf = cp.cumsum(hist, dtype=cp.float32)
                if cdf[-1] > 0:
                    # Normalize to [0, 1] then scale to output range
                    cdf = cdf / cdf[-1]
                    tile_cdfs[z_idx, y_idx, x_idx, :] = min_val + cdf * (max_val - min_val)
                else:
                    tile_cdfs[z_idx, y_idx, x_idx, :] = min_val
    
    return tile_cdfs

def _apply_vectorized_trilinear_interpolation(
    stack: "cp.ndarray",
    tile_cdfs: "cp.ndarray",
    tile_z: int,
    tile_y: int,
    tile_x: int,
    tile_depth: int,
    tile_height: int,
    tile_width: int,
    nbins: int,
    min_val: float,
    max_val: float
) -> "cp.ndarray":
    """Vectorized trilinear interpolation for 3D CLAHE."""
    
    depth, height, width = stack.shape
    
    # Create 3D coordinate grids
    z_coords, y_coords, x_coords = cp.meshgrid(
        cp.arange(depth, dtype=cp.float32),
        cp.arange(height, dtype=cp.float32),
        cp.arange(width, dtype=cp.float32),
        indexing='ij'
    )
    
    # Calculate tile centers
    tile_centers_z = cp.arange(tile_z, dtype=cp.float32) * tile_depth + tile_depth // 2
    tile_centers_y = cp.arange(tile_y, dtype=cp.float32) * tile_height + tile_height // 2
    tile_centers_x = cp.arange(tile_x, dtype=cp.float32) * tile_width + tile_width // 2
    
    # Find surrounding tiles for each voxel (vectorized)
    total_voxels = depth * height * width
    coords_flat = cp.column_stack([
        z_coords.ravel(),
        y_coords.ravel(), 
        x_coords.ravel()
    ])
    
    # Use searchsorted to find tile indices
    tile_z_low = cp.searchsorted(tile_centers_z, coords_flat[:, 0]) - 1
    tile_y_low = cp.searchsorted(tile_centers_y, coords_flat[:, 1]) - 1
    tile_x_low = cp.searchsorted(tile_centers_x, coords_flat[:, 2]) - 1
    
    # Clamp to valid ranges
    tile_z_low = cp.clip(tile_z_low, 0, tile_z - 2).reshape(depth, height, width)
    tile_y_low = cp.clip(tile_y_low, 0, tile_y - 2).reshape(depth, height, width)
    tile_x_low = cp.clip(tile_x_low, 0, tile_x - 2).reshape(depth, height, width)
    
    # Handle edge case for single tile in z-dimension
    if tile_z == 1:
        tile_z_low = cp.zeros_like(tile_z_low)
    
    tile_z_high = cp.minimum(tile_z_low + 1, tile_z - 1)
    tile_y_high = tile_y_low + 1
    tile_x_high = tile_x_low + 1
    
    # Convert voxel values to bin indices (vectorized)
    normalized_values = (stack - min_val) / (max_val - min_val)
    bin_indices = cp.clip(
        (normalized_values * (nbins - 1)).astype(cp.int32),
        0, nbins - 1
    )
    
    # Calculate interpolation weights (vectorized)
    center_z_low = tile_centers_z[tile_z_low]
    center_z_high = tile_centers_z[tile_z_high]
    center_y_low = tile_centers_y[tile_y_low]
    center_y_high = tile_centers_y[tile_y_high]
    center_x_low = tile_centers_x[tile_x_low]
    center_x_high = tile_centers_x[tile_x_high]
    
    # Avoid division by zero
    dz = center_z_high - center_z_low
    dy = center_y_high - center_y_low
    dx = center_x_high - center_x_low
    
    wz = cp.where(dz > 0, (z_coords - center_z_low) / dz, 0.0)
    wy = cp.where(dy > 0, (y_coords - center_y_low) / dy, 0.0)
    wx = cp.where(dx > 0, (x_coords - center_x_low) / dx, 0.0)
    
    # Clamp weights
    wz = cp.clip(wz, 0.0, 1.0)
    wy = cp.clip(wy, 0.0, 1.0)
    wx = cp.clip(wx, 0.0, 1.0)
    
    # Get the 8 surrounding transformation values using advanced indexing
    val_000 = tile_cdfs[tile_z_low, tile_y_low, tile_x_low, bin_indices]
    val_001 = tile_cdfs[tile_z_low, tile_y_low, tile_x_high, bin_indices]
    val_010 = tile_cdfs[tile_z_low, tile_y_high, tile_x_low, bin_indices]
    val_011 = tile_cdfs[tile_z_low, tile_y_high, tile_x_high, bin_indices]
    val_100 = tile_cdfs[tile_z_high, tile_y_low, tile_x_low, bin_indices]
    val_101 = tile_cdfs[tile_z_high, tile_y_low, tile_x_high, bin_indices]
    val_110 = tile_cdfs[tile_z_high, tile_y_high, tile_x_low, bin_indices]
    val_111 = tile_cdfs[tile_z_high, tile_y_high, tile_x_high, bin_indices]
    
    # Trilinear interpolation (vectorized)
    # First interpolate along x-axis
    val_00 = (1 - wx) * val_000 + wx * val_001  # front-bottom
    val_01 = (1 - wx) * val_010 + wx * val_011  # front-top
    val_10 = (1 - wx) * val_100 + wx * val_101  # back-bottom
    val_11 = (1 - wx) * val_110 + wx * val_111  # back-top
    
    # Then interpolate along y-axis
    val_0 = (1 - wy) * val_00 + wy * val_01  # front face
    val_1 = (1 - wy) * val_10 + wy * val_11  # back face
    
    # Finally interpolate along z-axis
    result = (1 - wz) * val_0 + wz * val_1
    
    return result

def _clahe_3d_chunked(
    stack: "cp.ndarray",
    clip_limit: float,
    tile_grid_size_3d: tuple,
    nbins: int,
    chunk_size: int = 128
) -> "cp.ndarray":
    """
    Memory-efficient chunked processing for very large 3D volumes.
    
    Processes the volume in overlapping chunks to manage memory usage.
    """
    depth, height, width = stack.shape
    result = cp.zeros_like(stack)
    
    # Calculate overlap needed for smooth transitions
    tile_z, tile_y, tile_x = tile_grid_size_3d
    tile_depth = max(1, depth // tile_z)
    overlap = tile_depth // 2
    
    # Process volume in z-chunks
    for z_start in range(0, depth, chunk_size - overlap):
        z_end = min(z_start + chunk_size, depth)
        
        # Extract chunk with context
        chunk_start = max(0, z_start - overlap)
        chunk_end = min(depth, z_end + overlap)
        
        chunk = stack[chunk_start:chunk_end, :, :]
        
        # Adjust tile grid for chunk
        chunk_depth = chunk_end - chunk_start
        chunk_tile_z = max(1, min(tile_z, chunk_depth // tile_depth))
        chunk_tile_grid = (chunk_tile_z, tile_y, tile_x)
        
        # Process chunk
        chunk_result = _clahe_3d_vectorized(
            chunk, clip_limit, chunk_tile_grid, nbins
        )
        
        # Extract the relevant part (without overlap)
        extract_start = z_start - chunk_start
        extract_end = extract_start + (z_end - z_start)
        
        result[z_start:z_end, :, :] = chunk_result[extract_start:extract_end, :, :]
    
    return result

def _fill_3d_boundaries(
    full_result: "cp.ndarray",
    cropped_result: "cp.ndarray",
    crop_depth: int,
    crop_height: int,
    crop_width: int,
    depth: int,
    height: int,
    width: int
) -> None:
    """Efficiently fill boundary regions by replicating edge values."""
    
    # Fill z-direction boundaries
    if crop_depth < depth:
        full_result[crop_depth:, :crop_height, :crop_width] = cropped_result[-1:, :, :]
    
    # Fill y-direction boundaries
    if crop_height < height:
        full_result[:crop_depth, crop_height:, :crop_width] = cropped_result[:, -1:, :]
        if crop_depth < depth:
            full_result[crop_depth:, crop_height:, :crop_width] = cropped_result[-1:, -1:, :]
    
    # Fill x-direction boundaries
    if crop_width < width:
        full_result[:crop_depth, :crop_height, crop_width:] = cropped_result[:, :, -1:]
        if crop_height < height:
            full_result[:crop_depth, crop_height:, crop_width:] = cropped_result[:, -1:, -1:]
        if crop_depth < depth:
            full_result[crop_depth:, :crop_height, crop_width:] = cropped_result[-1:, :, -1:]
        if crop_depth < depth and crop_height < height:
            full_result[crop_depth:, crop_height:, crop_width:] = cropped_result[-1, -1, -1]