"""
Main MIST Implementation

Full GPU-accelerated MIST implementation with zero CPU operations.
Orchestrates all MIST components for tile position computation.
"""

import logging
from typing import TYPE_CHECKING, Tuple

from openhcs.constants.constants import DEFAULT_PATCH_SIZE, DEFAULT_SEARCH_RADIUS
from openhcs.core.memory.decorators import cupy as cupy_func
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs, chain_breaker
from openhcs.core.utils import optional_import

from .phase_correlation import phase_correlation_gpu_only
from .quality_metrics import compute_correlation_quality_gpu_aligned
from .position_reconstruction import build_mst_gpu, rebuild_positions_from_mst_gpu

# For type checking only
if TYPE_CHECKING:
    import cupy as cp

# Import CuPy as an optional dependency
cp = optional_import("cupy")

logger = logging.getLogger(__name__)


def _convert_overlap_to_tile_coordinates(
    dy: float, dx: float,
    overlap_h: int, overlap_w: int,
    tile_h: int, tile_w: int,
    direction: str
) -> Tuple[float, float]:
    """
    Convert overlap-region-relative displacements to tile-center coordinates.

    Args:
        dy, dx: Phase correlation displacements in overlap region coordinates
        overlap_h, overlap_w: Overlap region dimensions
        tile_h, tile_w: Full tile dimensions
        direction: 'horizontal' or 'vertical'

    Returns:
        (tile_dy, tile_dx): Displacements in tile-center coordinates
    """
    if direction == 'horizontal':
        # For horizontal connections (left-right)
        # Expected displacement is approximately tile_w - overlap_w
        expected_dx = tile_w - overlap_w
        tile_dx = expected_dx + dx  # Add phase correlation correction
        tile_dy = dy  # Vertical should be minimal

    elif direction == 'vertical':
        # For vertical connections (top-bottom)
        # Expected displacement is approximately tile_h - overlap_h
        expected_dy = tile_h - overlap_h
        tile_dy = expected_dy + dy  # Add phase correlation correction
        tile_dx = dx  # Horizontal should be minimal

    else:
        raise ValueError(f"Invalid direction: {direction}. Must be 'horizontal' or 'vertical'")

    return tile_dy, tile_dx


def _validate_cupy_array(array, name: str = "input") -> None:  # type: ignore
    """Validate that the input is a CuPy array."""
    if not isinstance(array, cp.ndarray):
        raise TypeError(f"{name} must be a CuPy array, got {type(array)}")


def _global_optimization_gpu_only(
    positions: "cp.ndarray",  # type: ignore
    tile_grid: "cp.ndarray",  # type: ignore
    num_rows: int,
    num_cols: int,
    expected_dx: float,
    expected_dy: float,
    overlap_ratio: float,
    subpixel: bool,
    *,
    min_overlap_pixels: int = 32,
    quality_threshold: float = 0.03,
    subpixel_radius: int = 3,
    regularization_eps_multiplier: float = 1000.0,
    anchor_tile_index: int = 0
) -> "cp.ndarray":  # type: ignore
    """
    GPU-only global optimization using simplified MST approach.
    """
    H, W = tile_grid.shape[2], tile_grid.shape[3]
    num_tiles = num_rows * num_cols
    
    # Pre-allocate GPU arrays for connections
    max_connections = 2 * num_tiles  # Each tile has at most 2 neighbors (right, bottom)
    connection_from = cp.full(max_connections, -1, dtype=cp.int32)
    connection_to = cp.full(max_connections, -1, dtype=cp.int32)
    connection_dx = cp.zeros(max_connections, dtype=cp.float32)
    connection_dy = cp.zeros(max_connections, dtype=cp.float32)
    connection_quality = cp.zeros(max_connections, dtype=cp.float32)
    
    conn_idx = 0

    # Debug: Track quality filtering
    total_correlations = 0
    passed_threshold = 0
    all_qualities = []

    # Debug: Print expected displacements
    print(f"🔥 EXPECTED DISPLACEMENTS: dx={float(expected_dx):.1f}, dy={float(expected_dy):.1f}")
    print(f"🔥 OVERLAP RATIO: {overlap_ratio}, H={H}, W={W}")

    # Debug: Check if images are black
    tile_stats = []
    for r in range(num_rows):
        for c in range(num_cols):
            tile = tile_grid[r, c]
            tile_min = float(cp.min(tile))
            tile_max = float(cp.max(tile))
            tile_mean = float(cp.mean(tile))
            tile_stats.append((tile_min, tile_max, tile_mean))

    print(f"🔥 TILE STATS: First 3 tiles - min/max/mean:")
    for i, (tmin, tmax, tmean) in enumerate(tile_stats[:3]):
        print(f"  Tile {i}: [{tmin:.1f}, {tmax:.1f}], mean={tmean:.1f}")

    # Build connections (GPU operations)
    for r in range(num_rows):
        for c in range(num_cols):
            tile_idx = r * num_cols + c
            current_tile = tile_grid[r, c]
            
            # Horizontal connection
            if c < num_cols - 1:
                right_idx = r * num_cols + (c + 1)
                right_tile = tile_grid[r, c + 1]

                overlap_w = cp.maximum(cp.int32(W * overlap_ratio), min_overlap_pixels)
                left_region = current_tile[:, -overlap_w:]  # Right edge of left tile
                right_region = right_tile[:, :overlap_w]   # Left edge of right tile

                # Debug: Check overlap region extraction
                if conn_idx < 3:
                    print(f"🔥 HORIZONTAL OVERLAP {conn_idx}: tiles {tile_idx}->{right_idx}")
                    print(f"   overlap_w={int(overlap_w)}, W={W}")
                    print(f"   left_region.shape={left_region.shape} (from tile[:, -{int(overlap_w)}:])")
                    print(f"   right_region.shape={right_region.shape} (from tile[:, :{int(overlap_w)}])")

                dy, dx = phase_correlation_gpu_only(
                    left_region, right_region,  # Standardized: left_region first
                    subpixel=subpixel,
                    subpixel_radius=subpixel_radius,
                    regularization_eps_multiplier=regularization_eps_multiplier
                )

                # Compute quality after applying the shift
                quality = compute_correlation_quality_gpu_aligned(left_region, right_region, dx, dy)

                # Debug: Track all quality values
                total_correlations += 1
                all_qualities.append(quality)

                if quality >= quality_threshold:
                    passed_threshold += 1
                    connection_from[conn_idx] = tile_idx
                    connection_to[conn_idx] = right_idx

                    # Convert overlap-region coordinates to tile-center coordinates
                    tile_dy, tile_dx = _convert_overlap_to_tile_coordinates(
                        dy, dx, int(overlap_w), int(overlap_w), H, W, 'horizontal'
                    )
                    connection_dx[conn_idx] = tile_dx
                    connection_dy[conn_idx] = tile_dy
                    connection_quality[conn_idx] = quality

                    # Debug: Print first few connections
                    if conn_idx < 3:
                        print(f"🔥 HORIZONTAL CONNECTION {conn_idx}: {tile_idx}->{right_idx}")
                        print(f"   overlap coords: dx={float(dx):.3f}, dy={float(dy):.3f}")
                        print(f"   tile coords: dx={float(tile_dx):.3f}, dy={float(tile_dy):.3f}")
                        print(f"   quality={float(quality):.6f}")

                    conn_idx += 1
            
            # Vertical connection
            if r < num_rows - 1:
                bottom_idx = (r + 1) * num_cols + c
                bottom_tile = tile_grid[r + 1, c]

                overlap_h = cp.maximum(cp.int32(H * overlap_ratio), min_overlap_pixels)
                top_region = current_tile[-overlap_h:, :]  # Bottom edge of top tile
                bottom_region = bottom_tile[:overlap_h, :] # Top edge of bottom tile

                dy, dx = phase_correlation_gpu_only(
                    top_region, bottom_region,  # Standardized: top_region first
                    subpixel=subpixel,
                    subpixel_radius=subpixel_radius,
                    regularization_eps_multiplier=regularization_eps_multiplier
                )

                # Compute quality after applying the shift
                quality = compute_correlation_quality_gpu_aligned(top_region, bottom_region, dx, dy)

                # Debug: Track all quality values
                total_correlations += 1
                all_qualities.append(quality)

                if quality >= quality_threshold:
                    passed_threshold += 1
                    connection_from[conn_idx] = tile_idx
                    connection_to[conn_idx] = bottom_idx
                    connection_dx[conn_idx] = dx               # Correction only (should be ~0 for vertical)
                    connection_dy[conn_idx] = expected_dy + dy  # Total displacement = expected + correction
                    connection_quality[conn_idx] = quality

                    # Debug: Print first few connections
                    if conn_idx < 6:  # Show a few more since we want to see vertical connections too
                        print(f"🔥 VERTICAL CONNECTION {conn_idx}: {tile_idx}->{bottom_idx}")
                        print(f"   expected_dy={float(expected_dy):.3f}, dy={float(dy):.3f}")
                        print(f"   stored connection_dy={float(connection_dy[conn_idx]):.3f}")
                        print(f"   quality={float(quality):.6f}")

                    conn_idx += 1

    # Debug: Print quality filtering summary
    print(f"🔥 QUALITY FILTERING: {passed_threshold}/{total_correlations} connections passed threshold {quality_threshold}")
    if len(all_qualities) > 0:
        min_q = float(cp.min(cp.array(all_qualities)))
        max_q = float(cp.max(cp.array(all_qualities)))
        mean_q = float(cp.mean(cp.array(all_qualities)))
        print(f"🔥 QUALITY RANGE: min={min_q:.6f}, max={max_q:.6f}, mean={mean_q:.6f}")

    # Trim arrays to actual size (GPU)
    if conn_idx > 0:
        valid_connections = cp.arange(conn_idx)
        connection_from = connection_from[:conn_idx]
        connection_to = connection_to[:conn_idx]
        connection_dx = connection_dx[:conn_idx]
        connection_dy = connection_dy[:conn_idx]
        connection_quality = connection_quality[:conn_idx]

        # Build MST using refactored GPU Borůvka's algorithm
        mst_edges = build_mst_gpu(
            connection_from, connection_to, connection_dx,
            connection_dy, connection_quality, num_tiles
        )

        # Rebuild positions using MST (GPU)
        new_positions = rebuild_positions_from_mst_gpu(
            positions, mst_edges, num_tiles, anchor_tile_index
        )

        return new_positions

    return positions


@special_inputs("grid_dimensions")
@special_outputs("positions")
@chain_breaker
@cupy_func
def mist_compute_tile_positions(
    image_stack: "cp.ndarray",  # type: ignore
    grid_dimensions: Tuple[int, int],
    *,
    patch_size: int = DEFAULT_PATCH_SIZE,
    search_radius: int = DEFAULT_SEARCH_RADIUS,
    stride: int = 64,
    method: str = "phase_correlation",
    normalize: bool = True,
    fft_backend: str = "cupy",
    verbose: bool = False,
    overlap_ratio: float = 0.1,
    subpixel: bool = True,
    refinement_iterations: int = 10,
    global_optimization: bool = True,
    # All configurable parameters (no magic numbers)
    min_overlap_pixels: int = 32,
    refinement_damping: float = 0.5,
    regularization_eps_multiplier: float = 1000.0,
    subpixel_radius: int = 3,
    mst_quality_threshold: float = 0.01,  # Very low threshold - phase correlation peaks can be low even for good alignments
    correlation_weight_horizontal: float = 1.0,
    correlation_weight_vertical: float = 1.0,
    anchor_tile_index: int = 0,
    **kwargs
) -> Tuple["cp.ndarray", "cp.ndarray"]:  # type: ignore
    """
    Full GPU MIST implementation with zero CPU operations.

    Args:
        image_stack: 3D tensor (Z, Y, X) of tiles
        grid_dimensions: (num_cols, num_rows)
        patch_size: Patch size for correlation
        search_radius: Search radius (unused in this implementation)
        stride: Stride for patches (unused in this implementation)
        method: Correlation method (must be "phase_correlation")
        normalize: Normalize tiles
        fft_backend: Must be "cupy"
        verbose: Print progress
        overlap_ratio: Expected overlap ratio
        subpixel: Enable subpixel accuracy
        refinement_iterations: Number of refinement passes
        global_optimization: Enable MST optimization
        min_overlap_pixels: Minimum overlap region size
        refinement_damping: Damping factor for refinement
        regularization_eps_multiplier: Numerical stability parameter
        subpixel_radius: Radius for subpixel interpolation
        mst_quality_threshold: Minimum correlation for MST edges
        correlation_weight_horizontal: Weight for horizontal correlations
        correlation_weight_vertical: Weight for vertical correlations
        anchor_tile_index: Index of anchor tile (usually 0)
        **kwargs: Additional parameters

    Returns:
        (image_stack, positions) where positions is (Z, 2) in (x, y) format
    """
    _validate_cupy_array(image_stack, "image_stack")

    if image_stack.ndim != 3:
        raise ValueError(f"Input must be a 3D tensor, got {image_stack.ndim}D")

    if fft_backend != "cupy":
        raise ValueError(f"FFT backend must be 'cupy', got '{fft_backend}'")

    if method != "phase_correlation":
        raise ValueError(f"Only 'phase_correlation' method is supported, got '{method}'")

    num_cols, num_rows = grid_dimensions
    Z, H, W = image_stack.shape

    # Debug: Log the actual overlap_ratio parameter being used
    print(f"🔥 MIST FUNCTION CALLED WITH overlap_ratio={overlap_ratio}")
    print(f"🔥 Expected: 0.1 (10% overlap), Actual: {overlap_ratio}")

    if Z != num_rows * num_cols:
        raise ValueError(
            f"Number of tiles ({Z}) does not match grid size ({num_rows}x{num_cols}={num_rows*num_cols})"
        )

    # Normalize on GPU
    tiles = image_stack.astype(cp.float32)
    if normalize:
        for z in range(Z):
            tile = tiles[z]
            tile_min = cp.min(tile)
            tile_max = cp.max(tile)
            tile_range = tile_max - tile_min
            # Use GPU conditional to avoid division by zero
            tiles[z] = cp.where(tile_range > 0, (tile - tile_min) / tile_range, tile)

    # Reshape to grid (GPU operation)
    tile_grid = tiles.reshape(num_rows, num_cols, H, W)

    # Calculate expected spacing (GPU)
    expected_dy = cp.float32(H * (1.0 - overlap_ratio))
    expected_dx = cp.float32(W * (1.0 - overlap_ratio))

    # Initialize positions on GPU
    positions = cp.zeros((Z, 2), dtype=cp.float32)

    if verbose:
        logger.info(f"GPU MIST: {num_rows}x{num_cols} grid, spacing: dx={float(expected_dx):.1f}, dy={float(expected_dy):.1f}")

    # Phase 1: Initial positioning (all GPU)
    for r in range(num_rows):
        for c in range(num_cols):
            tile_idx = r * num_cols + c

            if tile_idx == anchor_tile_index:
                positions[tile_idx] = cp.array([0.0, 0.0])
                continue

            current_tile = tile_grid[r, c]

            # Position from left neighbor (GPU operations)
            if c > 0:
                left_idx = r * num_cols + (c - 1)
                left_tile = tile_grid[r, c - 1]

                # Extract overlap regions (GPU)
                overlap_w = cp.maximum(cp.int32(W * overlap_ratio), min_overlap_pixels)
                left_region = left_tile[:, -overlap_w:]
                current_region = current_tile[:, :overlap_w]

                # GPU phase correlation
                dy, dx = phase_correlation_gpu_only(
                    left_region, current_region,
                    subpixel=subpixel,
                    subpixel_radius=subpixel_radius,
                    regularization_eps_multiplier=regularization_eps_multiplier
                )

                # Update position (GPU)
                new_x = positions[left_idx, 0] + expected_dx + dx
                new_y = positions[left_idx, 1] + dy
                positions[tile_idx] = cp.array([new_x, new_y])

            elif r > 0:  # Position from top neighbor
                top_idx = (r - 1) * num_cols + c
                top_tile = tile_grid[r - 1, c]

                # Extract overlap regions (GPU)
                overlap_h = cp.maximum(cp.int32(H * overlap_ratio), min_overlap_pixels)
                top_region = top_tile[-overlap_h:, :]
                current_region = current_tile[:overlap_h, :]

                # GPU phase correlation
                dy, dx = phase_correlation_gpu_only(
                    top_region, current_region,
                    subpixel=subpixel,
                    subpixel_radius=subpixel_radius,
                    regularization_eps_multiplier=regularization_eps_multiplier
                )

                # Update position (GPU)
                new_x = positions[top_idx, 0] + dx
                new_y = positions[top_idx, 1] + expected_dy + dy
                positions[tile_idx] = cp.array([new_x, new_y])

    # Phase 2: Refinement iterations (all GPU)
    for iteration in range(refinement_iterations):
        if verbose:
            logger.info(f"GPU refinement iteration {iteration + 1}/{refinement_iterations}")

        position_corrections = cp.zeros_like(positions)
        correction_weights = cp.zeros(Z, dtype=cp.float32)

        # Horizontal constraints (GPU)
        for r in range(num_rows):
            for c in range(num_cols - 1):
                left_idx = r * num_cols + c
                right_idx = r * num_cols + (c + 1)

                left_tile = tile_grid[r, c]
                right_tile = tile_grid[r, c + 1]

                overlap_w = cp.maximum(cp.int32(W * overlap_ratio), min_overlap_pixels)
                left_region = left_tile[:, -overlap_w:]   # Right edge of left tile
                right_region = right_tile[:, :overlap_w] # Left edge of right tile

                dy, dx = phase_correlation_gpu_only(
                    left_region, right_region,  # Standardized: left_region first
                    subpixel=subpixel,
                    subpixel_radius=subpixel_radius,
                    regularization_eps_multiplier=regularization_eps_multiplier
                )

                # Expected position (GPU)
                expected_right = positions[left_idx] + cp.array([expected_dx + dx, dy])

                # Accumulate updates (GPU)
                position_corrections[right_idx] += expected_right * correlation_weight_horizontal
                correction_weights[right_idx] += correlation_weight_horizontal

        # Vertical constraints (GPU)
        for r in range(num_rows - 1):
            for c in range(num_cols):
                top_idx = r * num_cols + c
                bottom_idx = (r + 1) * num_cols + c

                top_tile = tile_grid[r, c]
                bottom_tile = tile_grid[r + 1, c]

                overlap_h = cp.maximum(cp.int32(H * overlap_ratio), min_overlap_pixels)
                top_region = top_tile[-overlap_h:, :]    # Bottom edge of top tile
                bottom_region = bottom_tile[:overlap_h, :] # Top edge of bottom tile

                dy, dx = phase_correlation_gpu_only(
                    top_region, bottom_region,  # Standardized: top_region first
                    subpixel=subpixel,
                    subpixel_radius=subpixel_radius,
                    regularization_eps_multiplier=regularization_eps_multiplier
                )

                # Expected position (GPU)
                expected_bottom = positions[top_idx] + cp.array([dx, expected_dy + dy])

                # Accumulate updates (GPU)
                position_corrections[bottom_idx] += expected_bottom * correlation_weight_vertical
                correction_weights[bottom_idx] += correlation_weight_vertical

        # Apply corrections with damping (all GPU)
        for tile_idx in range(Z):
            if correction_weights[tile_idx] > 0 and tile_idx != anchor_tile_index:
                averaged_correction = position_corrections[tile_idx] / correction_weights[tile_idx]
                positions[tile_idx] = ((1 - refinement_damping) * positions[tile_idx] +
                                      refinement_damping * averaged_correction)

    # Phase 3: Global optimization MST (GPU operations)
    print(f"🔥 PHASE 3: global_optimization={global_optimization}")
    if global_optimization:
        print(f"🔥 STARTING MST GLOBAL OPTIMIZATION")
        positions = _global_optimization_gpu_only(
            positions, tile_grid, num_rows, num_cols,
            expected_dx, expected_dy, overlap_ratio, subpixel,
            min_overlap_pixels=min_overlap_pixels,
            quality_threshold=mst_quality_threshold,
            subpixel_radius=subpixel_radius,
            regularization_eps_multiplier=regularization_eps_multiplier,
            anchor_tile_index=anchor_tile_index
        )

    # Center positions (GPU)
    mean_pos = cp.mean(positions, axis=0)
    positions = positions - mean_pos

    print(f"🔥 MIST COMPLETE: Returning {positions.shape} positions")
    return tiles, positions
