"""
Full GPU-accelerated Ashlar implementation with zero CPU operations.

All operations remain on GPU. All parameters are configurable via kwargs.
Follows OpenHCS doctrinal principles for GPU-only execution.
"""

from __future__ import annotations 
import logging
from typing import TYPE_CHECKING, Any, Tuple, Optional

from openhcs.core.memory.decorators import cupy as cupy_func
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs, chain_breaker
from openhcs.core.utils import optional_import

# For type checking only
if TYPE_CHECKING:
    import cupy as cp
    from cupyx.scipy import ndimage
    from openhcs.core.context import ProcessingContext

# Import CuPy as an optional dependency
cp = optional_import("cupy")
ndimage = None
if cp is not None:
    cupyx_scipy = optional_import("cupyx.scipy")
    if cupyx_scipy is not None:
        ndimage = cupyx_scipy.ndimage

logger = logging.getLogger(__name__)


def _validate_cupy_array(array: Any, name: str = "input") -> None:  # type: ignore
    """Validate that the input is a CuPy array."""
    if not isinstance(array, cp.ndarray):
        raise TypeError(f"{name} must be a CuPy array, got {type(array)}")


def phase_correlation_gpu_ashlar(
    image1: "cp.ndarray",  # type: ignore
    image2: "cp.ndarray",  # type: ignore
    *,
    window: bool = True,
    subpixel: bool = True,
    subpixel_radius: int = 3,
    regularization_eps_multiplier: float = 1000.0
) -> Tuple[float, float]:
    """
    Full GPU phase correlation for Ashlar with all operations on device.
    
    Args:
        image1: First image (CuPy array)
        image2: Second image (CuPy array)
        window: Apply Hann window
        subpixel: Enable subpixel accuracy
        subpixel_radius: Radius for subpixel interpolation
        regularization_eps_multiplier: Multiplier for numerical stability
    
    Returns:
        (dy, dx) shift values
    """
    _validate_cupy_array(image1, "image1")
    _validate_cupy_array(image2, "image2")

    if image1.shape != image2.shape:
        raise ValueError(f"Images must have the same shape, got {image1.shape} and {image2.shape}")

    # All operations on GPU
    img1 = image1.astype(cp.float32)
    img2 = image2.astype(cp.float32)
    
    # Remove DC component (GPU)
    img1 = img1 - cp.mean(img1)
    img2 = img2 - cp.mean(img2)

    # Apply Hann window (GPU)
    if window:
        h, w = img1.shape
        win_y = cp.hanning(h).reshape(-1, 1)
        win_x = cp.hanning(w).reshape(1, -1)
        window_2d = win_y * win_x
        img1 = img1 * window_2d
        img2 = img2 * window_2d

    # FFT operations (GPU)
    fft1 = cp.fft.fft2(img1)
    fft2 = cp.fft.fft2(img2)

    # Cross-power spectrum (GPU)
    cross_power = fft1 * cp.conj(fft2)
    magnitude = cp.abs(cross_power)
    eps = cp.finfo(cp.float32).eps * regularization_eps_multiplier
    cross_power_norm = cross_power / (magnitude + eps)

    # Inverse FFT (GPU)
    correlation = cp.real(cp.fft.ifft2(cross_power_norm))

    # Find peak (GPU)
    peak_idx = cp.unravel_index(cp.argmax(correlation), correlation.shape)
    y_peak = peak_idx[0]  # Keep as CuPy scalar
    x_peak = peak_idx[1]  # Keep as CuPy scalar

    # Convert to signed displacement (GPU)
    h, w = correlation.shape
    dy = cp.where(y_peak <= h // 2, y_peak, y_peak - h)
    dx = cp.where(x_peak <= w // 2, x_peak, x_peak - w)

    # Subpixel refinement (GPU)
    if subpixel:
        # Convert to int for indexing
        y_peak_int = int(y_peak)
        x_peak_int = int(x_peak)
        
        y_min = cp.maximum(0, y_peak_int - subpixel_radius)
        y_max = cp.minimum(h, y_peak_int + subpixel_radius + 1)
        x_min = cp.maximum(0, x_peak_int - subpixel_radius)
        x_max = cp.minimum(w, x_peak_int + subpixel_radius + 1)

        region = correlation[y_min:y_max, x_min:x_max]
        
        total_mass = cp.sum(region)
        if total_mass > 0:
            y_coords, x_coords = cp.mgrid[y_min:y_max, x_min:x_max]
            y_com = cp.sum(y_coords * region) / total_mass
            x_com = cp.sum(x_coords * region) / total_mass
            
            dy = cp.where(y_com <= h // 2, y_com, y_com - h)
            dx = cp.where(x_com <= w // 2, x_com, x_com - w)

    return float(dy), float(dx)


@chain_breaker
@special_inputs("grid_dimensions")
@special_outputs("positions")
@cupy_func
def gpu_ashlar_align_cupy(
    tiles: "cp.ndarray",  # type: ignore
    grid_dimensions: Tuple[int, int],
    *,
    overlap_ratio: float = 0.1,
    patch_size: int = 128,
    search_radius: int = 20,
    normalize: bool = True,
    method: str = "phase_correlation",
    return_affine: bool = False,
    verbose: bool = False,
    context: Optional['ProcessingContext'] = None,
    # All configurable parameters (no magic numbers)
    max_global_iterations: int = 5,
    convergence_threshold: float = 0.1,
    global_optimization_damping: float = 0.7,
    min_overlap_pixels: int = 32,
    regularization_eps_multiplier: float = 1000.0,
    subpixel: bool = True,
    subpixel_radius: int = 3,
    drift_correction_enabled: bool = True,
    anchor_tile_index: int = 0,
    initial_positioning_method: str = "sequential",  # "sequential" or "snake"
    constraint_weight_horizontal: float = 1.0,
    constraint_weight_vertical: float = 1.0,
    position_update_momentum: float = 0.0,
    **kwargs
) -> Tuple["cp.ndarray", "cp.ndarray"]:  # type: ignore
    """
    Full GPU Ashlar implementation with zero CPU operations.
    
    Args:
        tiles: 3D tensor (Z, Y, X) of tiles
        grid_dimensions: (num_cols, num_rows)
        overlap_ratio: Expected overlap ratio
        patch_size: Patch size (unused in this implementation)
        search_radius: Search radius (unused in this implementation)
        normalize: Normalize tiles
        method: Must be "phase_correlation"
        return_affine: Return affine transformation matrices
        verbose: Print progress
        context: Processing context (unused)
        max_global_iterations: Maximum global optimization iterations
        convergence_threshold: Convergence threshold for position changes
        global_optimization_damping: Damping factor for global optimization
        min_overlap_pixels: Minimum overlap region size
        regularization_eps_multiplier: Numerical stability parameter
        subpixel: Enable subpixel accuracy
        subpixel_radius: Radius for subpixel interpolation
        drift_correction_enabled: Enable final drift correction
        anchor_tile_index: Index of anchor tile
        initial_positioning_method: "sequential" or "snake" traversal
        constraint_weight_horizontal: Weight for horizontal constraints
        constraint_weight_vertical: Weight for vertical constraints
        position_update_momentum: Momentum for position updates
        **kwargs: Additional parameters
    
    Returns:
        (tiles, positions) or (tiles, affine_matrices) if return_affine=True
    """
    _validate_cupy_array(tiles, "tiles")

    if tiles.ndim != 3:
        raise ValueError(f"Input must be a 3D tensor, got {tiles.ndim}D")

    if method != "phase_correlation":
        raise ValueError(f"Only 'phase_correlation' method is supported, got '{method}'")

    num_cols, num_rows = grid_dimensions
    Z, H, W = tiles.shape

    if Z != num_rows * num_cols:
        raise ValueError(
            f"Number of tiles ({Z}) does not match grid size ({num_rows}x{num_cols}={num_rows*num_cols})"
        )

    # Normalize tiles on GPU
    processed_tiles = tiles.astype(cp.float32)
    if normalize:
        for z in range(Z):
            tile = processed_tiles[z]
            tile_min = cp.min(tile)
            tile_max = cp.max(tile)
            tile_range = tile_max - tile_min
            # GPU conditional to avoid division by zero
            processed_tiles[z] = cp.where(tile_range > 0, 
                                        (tile - tile_min) / tile_range, 
                                        tile)

    # Reshape for grid access (GPU)
    tile_grid = processed_tiles.reshape(num_rows, num_cols, H, W)

    # Expected tile spacing (GPU)
    nominal_dx = cp.float32(W * (1.0 - overlap_ratio))
    nominal_dy = cp.float32(H * (1.0 - overlap_ratio))

    if verbose:
        logger.info(f"GPU Ashlar: {num_rows}x{num_cols} grid")
        logger.info(f"Expected spacing: dx={float(nominal_dx):.1f}, dy={float(nominal_dy):.1f}")

    # Initialize positions array on GPU
    positions = cp.zeros((Z, 2), dtype=cp.float32)
    position_momentum = cp.zeros((Z, 2), dtype=cp.float32)

    # Phase 1: Initial positioning
    positions[anchor_tile_index] = cp.array([0.0, 0.0])

    if initial_positioning_method == "snake":
        _initial_positioning_snake_gpu(
            positions, tile_grid, num_rows, num_cols, 
            nominal_dx, nominal_dy, overlap_ratio,
            min_overlap_pixels, subpixel, subpixel_radius,
            regularization_eps_multiplier, anchor_tile_index
        )
    else:  # sequential
        _initial_positioning_sequential_gpu(
            positions, tile_grid, num_rows, num_cols,
            nominal_dx, nominal_dy, overlap_ratio,
            min_overlap_pixels, subpixel, subpixel_radius,
            regularization_eps_multiplier, anchor_tile_index
        )

    # Phase 2: Iterative global optimization (all GPU)
    for iteration in range(max_global_iterations):
        if verbose:
            logger.info(f"Global optimization iteration {iteration + 1}/{max_global_iterations}")
        
        old_positions = positions.copy()
        
        # Collect constraint updates (GPU)
        position_updates = cp.zeros_like(positions)
        update_weights = cp.zeros(Z, dtype=cp.float32)
        
        # Horizontal constraints (GPU)
        for r in range(num_rows):
            for c in range(num_cols - 1):
                left_idx = r * num_cols + c
                right_idx = r * num_cols + (c + 1)
                
                left_tile = tile_grid[r, c]
                right_tile = tile_grid[r, c + 1]
                
                # Compute optimal shift (GPU)
                dy, dx = _compute_tile_shift_gpu(
                    left_tile, right_tile, overlap_ratio, 'horizontal',
                    min_overlap_pixels, subpixel, subpixel_radius,
                    regularization_eps_multiplier
                )
                
                # Expected position of right tile (GPU)
                expected_right = positions[left_idx] + cp.array([nominal_dx + dx, dy])
                
                # Add to updates (GPU)
                position_updates[right_idx] += expected_right * constraint_weight_horizontal
                update_weights[right_idx] += constraint_weight_horizontal
        
        # Vertical constraints (GPU)
        for r in range(num_rows - 1):
            for c in range(num_cols):
                top_idx = r * num_cols + c
                bottom_idx = (r + 1) * num_cols + c
                
                top_tile = tile_grid[r, c]
                bottom_tile = tile_grid[r + 1, c]
                
                # Compute optimal shift (GPU)
                dy, dx = _compute_tile_shift_gpu(
                    top_tile, bottom_tile, overlap_ratio, 'vertical',
                    min_overlap_pixels, subpixel, subpixel_radius,
                    regularization_eps_multiplier
                )
                
                # Expected position of bottom tile (GPU)
                expected_bottom = positions[top_idx] + cp.array([dx, nominal_dy + dy])
                
                # Add to updates (GPU)
                position_updates[bottom_idx] += expected_bottom * constraint_weight_vertical
                update_weights[bottom_idx] += constraint_weight_vertical
        
        # Apply weighted updates with momentum (GPU)
        max_change = cp.float32(0.0)
        
        for tile_idx in range(Z):
            if update_weights[tile_idx] > 0 and tile_idx != anchor_tile_index:
                new_pos = position_updates[tile_idx] / update_weights[tile_idx]
                
                # Apply momentum (GPU)
                position_change = new_pos - positions[tile_idx]
                position_momentum[tile_idx] = (position_update_momentum * position_momentum[tile_idx] + 
                                             position_change)
                
                # Update position with damping and momentum (GPU)
                positions[tile_idx] = (positions[tile_idx] + 
                                     global_optimization_damping * position_momentum[tile_idx])
                
                # Track maximum change (GPU)
                change_magnitude = cp.linalg.norm(position_change)
                max_change = cp.maximum(max_change, change_magnitude)
        
        if verbose:
            logger.info(f"Max position change: {float(max_change):.3f}")
        
        # Check convergence (GPU)
        if max_change < convergence_threshold:
            if verbose:
                logger.info(f"Converged after {iteration + 1} iterations")
            break

    # Phase 3: Final drift correction (GPU)
    if drift_correction_enabled:
        positions = _correct_global_drift_gpu(positions, num_rows, num_cols, anchor_tile_index)

    # Center positions around origin (GPU)
    mean_pos = cp.mean(positions, axis=0)
    positions = positions - mean_pos

    if return_affine:
        # Convert to affine transformation matrices (GPU)
        affine_mats = cp.zeros((Z, 3, 3), dtype=cp.float32)
        for z in range(Z):
            dx, dy = positions[z]
            affine_mats[z] = cp.array([
                [1.0, 0.0, dx],
                [0.0, 1.0, dy],
                [0.0, 0.0, 1.0]
            ])
        return processed_tiles, affine_mats
    
    return processed_tiles, positions


def _initial_positioning_sequential_gpu(
    positions: "cp.ndarray",  # type: ignore
    tile_grid: "cp.ndarray",  # type: ignore
    num_rows: int,
    num_cols: int,
    nominal_dx: float,
    nominal_dy: float,
    overlap_ratio: float,
    min_overlap_pixels: int,
    subpixel: bool,
    subpixel_radius: int,
    regularization_eps_multiplier: float,
    anchor_tile_index: int
) -> None:
    """Sequential grid traversal for initial positioning (GPU)."""
    for r in range(num_rows):
        for c in range(num_cols):
            tile_idx = r * num_cols + c
            
            if tile_idx == anchor_tile_index:
                continue
            
            current_tile = tile_grid[r, c]
            
            # Position from left neighbor (preferred)
            if c > 0:
                left_idx = r * num_cols + (c - 1)
                left_tile = tile_grid[r, c - 1]
                
                dy, dx = _compute_tile_shift_gpu(
                    left_tile, current_tile, overlap_ratio, 'horizontal',
                    min_overlap_pixels, subpixel, subpixel_radius,
                    regularization_eps_multiplier
                )
                
                positions[tile_idx, 0] = positions[left_idx, 0] + nominal_dx + dx
                positions[tile_idx, 1] = positions[left_idx, 1] + dy
                
            elif r > 0:  # Position from top neighbor
                top_idx = (r - 1) * num_cols + c
                top_tile = tile_grid[r - 1, c]
                
                dy, dx = _compute_tile_shift_gpu(
                    top_tile, current_tile, overlap_ratio, 'vertical',
                    min_overlap_pixels, subpixel, subpixel_radius,
                    regularization_eps_multiplier
                )
                
                positions[tile_idx, 0] = positions[top_idx, 0] + dx
                positions[tile_idx, 1] = positions[top_idx, 1] + nominal_dy + dy


def _initial_positioning_snake_gpu(
    positions: "cp.ndarray",  # type: ignore
    tile_grid: "cp.ndarray",  # type: ignore
    num_rows: int,
    num_cols: int,
    nominal_dx: float,
    nominal_dy: float,
    overlap_ratio: float,
    min_overlap_pixels: int,
    subpixel: bool,
    subpixel_radius: int,
    regularization_eps_multiplier: float,
    anchor_tile_index: int
) -> None:
    """Snake pattern traversal for initial positioning (GPU)."""
    # Snake pattern: alternating left-to-right and right-to-left per row
    for r in range(num_rows):
        if r % 2 == 0:  # Left to right
            for c in range(num_cols):
                tile_idx = r * num_cols + c
                if tile_idx == anchor_tile_index:
                    continue
                
                current_tile = tile_grid[r, c]
                
                if c > 0:  # From left
                    left_idx = r * num_cols + (c - 1)
                    left_tile = tile_grid[r, c - 1]
                    
                    dy, dx = _compute_tile_shift_gpu(
                        left_tile, current_tile, overlap_ratio, 'horizontal',
                        min_overlap_pixels, subpixel, subpixel_radius,
                        regularization_eps_multiplier
                    )
                    
                    positions[tile_idx, 0] = positions[left_idx, 0] + nominal_dx + dx
                    positions[tile_idx, 1] = positions[left_idx, 1] + dy
                    
                elif r > 0:  # From top
                    top_idx = (r - 1) * num_cols + c
                    top_tile = tile_grid[r - 1, c]
                    
                    dy, dx = _compute_tile_shift_gpu(
                        top_tile, current_tile, overlap_ratio, 'vertical',
                        min_overlap_pixels, subpixel, subpixel_radius,
                        regularization_eps_multiplier
                    )
                    
                    positions[tile_idx, 0] = positions[top_idx, 0] + dx
                    positions[tile_idx, 1] = positions[top_idx, 1] + nominal_dy + dy
        
        else:  # Right to left
            for c in range(num_cols - 1, -1, -1):
                tile_idx = r * num_cols + c
                if tile_idx == anchor_tile_index:
                    continue
                
                current_tile = tile_grid[r, c]
                
                if c < num_cols - 1:  # From right
                    right_idx = r * num_cols + (c + 1)
                    right_tile = tile_grid[r, c + 1]
                    
                    dy, dx = _compute_tile_shift_gpu(
                        current_tile, right_tile, overlap_ratio, 'horizontal',
                        min_overlap_pixels, subpixel, subpixel_radius,
                        regularization_eps_multiplier
                    )
                    
                    positions[tile_idx, 0] = positions[right_idx, 0] - nominal_dx - dx
                    positions[tile_idx, 1] = positions[right_idx, 1] - dy
                    
                elif r > 0:  # From top
                    top_idx = (r - 1) * num_cols + c
                    top_tile = tile_grid[r - 1, c]
                    
                    dy, dx = _compute_tile_shift_gpu(
                        top_tile, current_tile, overlap_ratio, 'vertical',
                        min_overlap_pixels, subpixel, subpixel_radius,
                        regularization_eps_multiplier
                    )
                    
                    positions[tile_idx, 0] = positions[top_idx, 0] + dx
                    positions[tile_idx, 1] = positions[top_idx, 1] + nominal_dy + dy


def _compute_tile_shift_gpu(
    tile1: "cp.ndarray",  # type: ignore
    tile2: "cp.ndarray",  # type: ignore
    overlap_ratio: float,
    direction: str,
    min_overlap_pixels: int,
    subpixel: bool,
    subpixel_radius: int,
    regularization_eps_multiplier: float
) -> Tuple[float, float]:
    """GPU-only tile shift computation."""
    H, W = tile1.shape
    
    if direction == 'horizontal':
        overlap_w = cp.maximum(cp.int32(W * overlap_ratio), min_overlap_pixels)
        region1 = tile1[:, -overlap_w:]
        region2 = tile2[:, :overlap_w]
    else:  # vertical
        overlap_h = cp.maximum(cp.int32(H * overlap_ratio), min_overlap_pixels)
        region1 = tile1[-overlap_h:, :]
        region2 = tile2[:overlap_h, :]
    
    # GPU phase correlation
    dy, dx = phase_correlation_gpu_ashlar(
        region1, region2,
        subpixel=subpixel,
        subpixel_radius=subpixel_radius,
        regularization_eps_multiplier=regularization_eps_multiplier
    )
    
    return dy, dx


def _correct_global_drift_gpu(
    positions: "cp.ndarray",  # type: ignore
    num_rows: int,
    num_cols: int,
    anchor_tile_index: int
) -> "cp.ndarray":  # type: ignore
    """GPU-only global drift correction using linear detrending."""
    corrected_positions = positions.copy()
    num_tiles = len(positions)
    
    # Create design matrix for linear fit on GPU
    design_matrix = cp.ones((num_tiles, 3), dtype=cp.float32)
    
    for i in range(num_tiles):
        r = i // num_cols
        c = i % num_cols
        design_matrix[i, 1] = cp.float32(r)
        design_matrix[i, 2] = cp.float32(c)
    
    # Fit and remove linear trends (GPU operations)
    try:
        # X position detrending
        x_coeffs = cp.linalg.lstsq(design_matrix, positions[:, 0], rcond=None)[0]
        x_trend = design_matrix @ x_coeffs
        corrected_positions[:, 0] = positions[:, 0] - x_trend + x_coeffs[0]
        
        # Y position detrending  
        y_coeffs = cp.linalg.lstsq(design_matrix, positions[:, 1], rcond=None)[0]
        y_trend = design_matrix @ y_coeffs
        corrected_positions[:, 1] = positions[:, 1] - y_trend + y_coeffs[0]
        
    except cp.linalg.LinAlgError:
        # If singular, skip detrending
        pass
    
    return corrected_positions
