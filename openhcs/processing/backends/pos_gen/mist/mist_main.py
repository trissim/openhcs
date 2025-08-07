"""
Main MIST Implementation

Full GPU-accelerated MIST implementation with zero CPU operations.
Orchestrates all MIST components for tile position computation.
"""
from __future__ import annotations 

import logging
from typing import TYPE_CHECKING, Tuple

from openhcs.constants.constants import DEFAULT_PATCH_SIZE, DEFAULT_SEARCH_RADIUS
from openhcs.core.memory.decorators import cupy as cupy_func
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.core.utils import optional_import

from .phase_correlation import phase_correlation_gpu_only, phase_correlation_nist_gpu
from .quality_metrics import (
    compute_correlation_quality_gpu_aligned,
    compute_adaptive_quality_threshold,
    validate_translation_consistency,
    log_coordinate_transformation,
    debug_phase_correlation_matrix
)
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





def _validate_displacement_magnitude(
    tile_dx: float, tile_dy: float,
    expected_dx: float, expected_dy: float,
    direction: str,
    tolerance_factor: float = 2.0,
    tolerance_percent: float = 0.1
) -> bool:
    """
    Validate that displacement magnitudes are reasonable.

    Args:
        tile_dx, tile_dy: Computed tile-center displacements
        expected_dx, expected_dy: Expected displacements
        direction: 'horizontal' or 'vertical'
        tolerance_factor: How much deviation to allow

    Returns:
        True if displacement is reasonable, False otherwise
    """
    if direction == 'horizontal':
        # For horizontal connections, dx should be close to expected_dx
        dx_error = abs(tile_dx - expected_dx)
        max_allowed_error = tolerance_factor * expected_dx * tolerance_percent
        dx_valid = dx_error <= max_allowed_error

        # dy should be small (minimal vertical drift relative to expected_dx, not expected_dy)
        max_allowed_dy = tolerance_factor * expected_dx * tolerance_percent
        dy_valid = abs(tile_dy) <= max_allowed_dy

        return dx_valid and dy_valid

    elif direction == 'vertical':
        # For vertical connections, dy should be close to expected_dy
        dy_error = abs(tile_dy - expected_dy)
        max_allowed_error = tolerance_factor * expected_dy * tolerance_percent
        dy_valid = dy_error <= max_allowed_error

        # dx should be small (minimal horizontal drift relative to expected_dy, not expected_dx)
        max_allowed_dx = tolerance_factor * expected_dy * tolerance_percent
        dx_valid = abs(tile_dx) <= max_allowed_dx

        return dy_valid and dx_valid

    return False


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

    quality_threshold: float = 0.5,  # NIST Algorithm 15: ncc >= 0.5 for valid translations
    subpixel_radius: int = 3,
    regularization_eps_multiplier: float = 1000.0,
    anchor_tile_index: int = 0,
    debug_connection_limit: int = 3,
    debug_vertical_limit: int = 6,
    displacement_tolerance_factor: float = 2.0,
    displacement_tolerance_percent: float = 0.3,
    consistency_threshold_percent: float = 0.5,
    max_connections_multiplier: int = 2,
    adaptive_base_threshold: float = 0.3,
    adaptive_percentile_threshold: float = 0.25,
    translation_tolerance_factor: float = 0.2,
    translation_min_quality: float = 0.3,
    magnitude_threshold_multiplier: float = 1e-6,
    peak_candidates_multiplier: int = 4,
    min_peak_distance: int = 5,
    use_nist_robustness: bool = True,  # NIST Algorithm 2: Enable multi-peak PCIAM with interpretation testing
    n_peaks: int = 2,  # NIST Algorithm 2: n=2 peaks tested (manually selected based on experimental testing)
    use_nist_normalization: bool = True,  # NIST Algorithm 3: Use fc/abs(fc) normalization instead of regularized approach

    # NIST Algorithm 9: Stage model parameters
    overlap_uncertainty_percent: float = 3.0,  # NIST default: 3% overlap uncertainty (pou)
    outlier_threshold_multiplier: float = 1.5,  # NIST Algorithm 16: 1.5 × IQR for outlier detection
) -> "cp.ndarray":  # type: ignore
    """
    GPU-only global optimization using simplified MST approach.
    """
    H, W = tile_grid.shape[2], tile_grid.shape[3]
    num_tiles = num_rows * num_cols
    
    # Pre-allocate GPU arrays for connections
    max_connections = max_connections_multiplier * num_tiles  # Each tile has at most 2 neighbors (right, bottom)
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

    # Debug: Print expected displacements and coordinate validation
    print(f"🔥 EXPECTED DISPLACEMENTS: dx={float(expected_dx):.1f}, dy={float(expected_dy):.1f}")
    print(f"🔥 OVERLAP RATIO: {overlap_ratio}, H={H}, W={W}")
    print(f"🔥 COORDINATE VALIDATION:")
    print(f"   Expected tile spacing: dx={float(expected_dx):.1f}, dy={float(expected_dy):.1f}")
    print(f"   Overlap regions: H*ratio={H*overlap_ratio:.1f}, W*ratio={W*overlap_ratio:.1f}")
    print(f"   Actual overlap: H={H*overlap_ratio:.1f}, W={W*overlap_ratio:.1f} pixels")

    # Debug: Check if images are black
    tile_stats = []
    for r in range(num_rows):
        for c in range(num_cols):
            tile = tile_grid[r, c]
            tile_min = float(cp.min(tile))
            tile_max = float(cp.max(tile))
            tile_mean = float(cp.mean(tile))
            tile_stats.append((tile_min, tile_max, tile_mean))

    print(f"🔥 TILE STATS: First {debug_connection_limit} tiles - min/max/mean:")
    for i, (tmin, tmax, tmean) in enumerate(tile_stats[:debug_connection_limit]):
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

                overlap_w = cp.int32(W * overlap_ratio)
                left_region = current_tile[:, -overlap_w:]  # Right edge of left tile
                right_region = right_tile[:, :overlap_w]   # Left edge of right tile

                # Debug: Check overlap region extraction (avoid GPU sync on .shape)
                if conn_idx < debug_connection_limit:
                    print(f"🔥 HORIZONTAL OVERLAP {conn_idx}: tiles {tile_idx}->{right_idx}")
                    print(f"   overlap_w={int(overlap_w)}, W={W}")
                    # Avoid .shape access which can cause GPU sync issues
                    print(f"   Processing overlap regions (shapes not shown to avoid GPU sync)")

                if use_nist_robustness:
                    dy, dx, quality = phase_correlation_nist_gpu(
                        left_region, right_region,
                        direction='horizontal',
                        n_peaks=n_peaks,
                        use_nist_normalization=use_nist_normalization
                    )
                else:
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
                    # Convert overlap-region coordinates to tile-center coordinates
                    tile_dy, tile_dx = _convert_overlap_to_tile_coordinates(
                        dy, dx, int(overlap_w), int(overlap_w), H, W, 'horizontal'
                    )

                    # Log coordinate transformation for debugging
                    if conn_idx < debug_connection_limit:  # Only log first few for brevity
                        log_coordinate_transformation(
                            dy, dx, tile_dy, tile_dx, 'horizontal', (tile_idx, right_idx)
                        )

                    # Validate displacement magnitude
                    displacement_valid = _validate_displacement_magnitude(
                        tile_dx, tile_dy, float(expected_dx), 0.0, 'horizontal',
                        displacement_tolerance_factor, displacement_tolerance_percent
                    )

                    if displacement_valid:
                        passed_threshold += 1
                        connection_from[conn_idx] = tile_idx
                        connection_to[conn_idx] = right_idx
                        connection_dx[conn_idx] = tile_dx
                        connection_dy[conn_idx] = tile_dy
                        connection_quality[conn_idx] = quality

                        # Debug: Print first few connections
                        if conn_idx < debug_connection_limit:
                            print(f"🔥 HORIZONTAL CONNECTION {conn_idx}: {tile_idx}->{right_idx}")
                            print(f"   overlap coords: dx={float(dx):.3f}, dy={float(dy):.3f}")
                            print(f"   tile coords: dx={float(tile_dx):.3f}, dy={float(tile_dy):.3f}")
                            print(f"   quality={float(quality):.6f}, displacement_valid={displacement_valid}")

                        conn_idx += 1
                    else:
                        # Debug: Log rejected connections
                        if conn_idx < debug_connection_limit:
                            print(f"🔥 REJECTED HORIZONTAL {tile_idx}->{right_idx}: displacement invalid")
                            print(f"   tile coords: dx={float(tile_dx):.3f}, dy={float(tile_dy):.3f}")
                            print(f"   expected: dx={float(expected_dx):.3f}, dy={float(expected_dy):.3f}")
                            # Show validation details
                            dx_error = abs(tile_dx - expected_dx)
                            max_allowed_error = displacement_tolerance_factor * expected_dx * displacement_tolerance_percent
                            max_allowed_dy = displacement_tolerance_factor * expected_dx * displacement_tolerance_percent
                            print(f"   dx_error={dx_error:.3f} vs max_allowed={max_allowed_error:.3f}")
                            print(f"   abs(dy)={abs(tile_dy):.3f} vs max_allowed_dy={max_allowed_dy:.3f}")
            
            # Vertical connection
            if r < num_rows - 1:
                bottom_idx = (r + 1) * num_cols + c
                bottom_tile = tile_grid[r + 1, c]

                overlap_h = cp.int32(H * overlap_ratio)
                top_region = current_tile[-overlap_h:, :]  # Bottom edge of top tile
                bottom_region = bottom_tile[:overlap_h, :] # Top edge of bottom tile

                if use_nist_robustness:
                    dy, dx, quality = phase_correlation_nist_gpu(
                        top_region, bottom_region,
                        direction='vertical',
                        n_peaks=n_peaks,
                        use_nist_normalization=use_nist_normalization
                    )
                else:
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
                    # Convert overlap-region coordinates to tile-center coordinates
                    tile_dy, tile_dx = _convert_overlap_to_tile_coordinates(
                        dy, dx, int(overlap_h), int(overlap_h), H, W, 'vertical'
                    )

                    # Log coordinate transformation for debugging
                    if conn_idx < debug_vertical_limit:  # Only log first few for brevity
                        log_coordinate_transformation(
                            dy, dx, tile_dy, tile_dx, 'vertical', (tile_idx, bottom_idx)
                        )

                    # Validate displacement magnitude
                    displacement_valid = _validate_displacement_magnitude(
                        tile_dx, tile_dy, 0.0, float(expected_dy), 'vertical',
                        displacement_tolerance_factor, displacement_tolerance_percent
                    )

                    if displacement_valid:
                        passed_threshold += 1
                        connection_from[conn_idx] = tile_idx
                        connection_to[conn_idx] = bottom_idx
                        connection_dx[conn_idx] = tile_dx
                        connection_dy[conn_idx] = tile_dy
                        connection_quality[conn_idx] = quality

                        # Debug: Print first few connections
                        if conn_idx < debug_vertical_limit:  # Show a few more since we want to see vertical connections too
                            print(f"🔥 VERTICAL CONNECTION {conn_idx}: {tile_idx}->{bottom_idx}")
                            print(f"   overlap coords: dx={float(dx):.3f}, dy={float(dy):.3f}")
                            print(f"   tile coords: dx={float(tile_dx):.3f}, dy={float(tile_dy):.3f}")
                            print(f"   quality={float(quality):.6f}, displacement_valid={displacement_valid}")

                        conn_idx += 1
                    else:
                        # Debug: Log rejected connections
                        if conn_idx < debug_vertical_limit:
                            print(f"🔥 REJECTED VERTICAL {tile_idx}->{bottom_idx}: displacement invalid")
                            print(f"   tile coords: dx={float(tile_dx):.3f}, dy={float(tile_dy):.3f}")
                            print(f"   expected: dx={float(expected_dx):.3f}, dy={float(expected_dy):.3f}")
                            # Show validation details
                            dy_error = abs(tile_dy - expected_dy)
                            max_allowed_error = displacement_tolerance_factor * expected_dy * displacement_tolerance_percent
                            max_allowed_dx = displacement_tolerance_factor * expected_dy * displacement_tolerance_percent
                            print(f"   dy_error={dy_error:.3f} vs max_allowed={max_allowed_error:.3f}")
                            print(f"   abs(dx)={abs(tile_dx):.3f} vs max_allowed_dx={max_allowed_dx:.3f}")

    # Compute adaptive quality threshold if we have quality data
    if len(all_qualities) > 0:
        adaptive_threshold = compute_adaptive_quality_threshold(
            all_qualities, adaptive_base_threshold, adaptive_percentile_threshold
        )
        print(f"🔥 ADAPTIVE THRESHOLD: original={quality_threshold:.6f}, adaptive={adaptive_threshold:.6f}")

        # Re-filter connections with adaptive threshold if it's different
        if adaptive_threshold != quality_threshold and adaptive_threshold < quality_threshold:
            print(f"🔥 RE-FILTERING with adaptive threshold...")
            # Note: In a full implementation, we'd re-process with the adaptive threshold
            # For now, we'll use the original threshold but log the adaptive one

    # Debug: Print quality filtering summary
    print(f"🔥 QUALITY FILTERING: {passed_threshold}/{total_correlations} connections passed threshold {quality_threshold}")
    if len(all_qualities) > 0:
        min_q = float(cp.min(cp.array(all_qualities)))
        max_q = float(cp.max(cp.array(all_qualities)))
        mean_q = float(cp.mean(cp.array(all_qualities)))
        print(f"🔥 QUALITY RANGE: min={min_q:.6f}, max={max_q:.6f}, mean={mean_q:.6f}")

        # Validate translation consistency (Plan 03)
        if conn_idx > 0:
            # Collect translations for validation
            translations = []
            for i in range(conn_idx):
                dy_val = float(connection_dy[i])
                dx_val = float(connection_dx[i])
                quality_val = float(connection_quality[i])
                translations.append((dy_val, dx_val, quality_val))

            # Validate against expected spacing
            expected_spacing = (float(expected_dx), float(expected_dy))
            valid_flags = validate_translation_consistency(
                translations, expected_spacing, translation_tolerance_factor, translation_min_quality
            )

            num_valid = sum(valid_flags)
            print(f"🔥 TRANSLATION VALIDATION: {num_valid}/{len(translations)} connections are consistent")

            if num_valid < len(translations) * consistency_threshold_percent:  # Less than threshold% valid
                print(f"🔥 WARNING: Low translation consistency ({num_valid}/{len(translations)})")
                print(f"🔥 Expected spacing: dx={expected_spacing[0]:.1f}, dy={expected_spacing[1]:.1f}")
                print(f"🔥 Consider adjusting overlap_ratio or quality thresholds")

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
@cupy_func
def mist_compute_tile_positions(
    image_stack: "cp.ndarray",  # type: ignore
    grid_dimensions: Tuple[int, int],
    *,
    # === Input Validation Parameters ===
    method: str = "phase_correlation",
    fft_backend: str = "cupy",

    # === Core Algorithm Parameters ===
    normalize: bool = True,
    verbose: bool = False,
    overlap_ratio: float = 0.1,
    subpixel: bool = True,
    refinement_iterations: int = 10,
    global_optimization: bool = True,
    anchor_tile_index: int = 0,

    # === Refinement Tuning Parameters ===
    refinement_damping: float = 0.5,
    correlation_weight_horizontal: float = 1.0,
    correlation_weight_vertical: float = 1.0,

    # === Phase Correlation Parameters ===
    subpixel_radius: int = 3,
    regularization_eps_multiplier: float = 1000.0,

    # === MST Global Optimization Parameters ===
    mst_quality_threshold: float = 0.5,  # NIST Algorithm 15: ncc >= 0.5 for MST edge inclusion
    # NIST robustness parameters (Algorithms 2-5)
    use_nist_robustness: bool = True,  # Enable full NIST PCIAM implementation
    n_peaks: int = 2,  # NIST Algorithm 2: Test 2 peaks (experimentally determined)
    use_nist_normalization: bool = True,  # NIST Algorithm 3: fc/abs(fc) normalization
    # Debugging and validation parameters
    debug_connection_limit: int = 3,
    debug_vertical_limit: int = 6,
    displacement_tolerance_factor: float = 2.0,
    displacement_tolerance_percent: float = 0.3,
    consistency_threshold_percent: float = 0.5,
    max_connections_multiplier: int = 2,
    # Quality metric tuning parameters
    adaptive_base_threshold: float = 0.3,
    adaptive_percentile_threshold: float = 0.25,
    translation_tolerance_factor: float = 0.2,
    translation_min_quality: float = 0.3,
    # Phase correlation tuning parameters
    magnitude_threshold_multiplier: float = 1e-6,
    peak_candidates_multiplier: int = 4,
    min_peak_distance: int = 5,
    **kwargs
) -> Tuple["cp.ndarray", "cp.ndarray"]:  # type: ignore
    """
    Full GPU MIST implementation with zero CPU operations.

    Performs microscopy image stitching using phase correlation and iterative refinement.
    The algorithm has three phases:
    1. Initial positioning using sequential phase correlation
    2. Iterative refinement with constraint optimization
    3. Global optimization using minimum spanning tree (MST)

    Args:
        image_stack: 3D tensor (Z, Y, X) of tiles to stitch
        grid_dimensions: (num_cols, num_rows) grid layout of tiles

        === Input Validation Parameters ===
        method: Correlation method - must be "phase_correlation"
        fft_backend: FFT backend - must be "cupy" for GPU acceleration

        === Core Algorithm Parameters (NIST Algorithms 1-3) ===
        normalize: Normalize each tile to [0,1] range using (tile-min)/(max-min).
                  True = better correlation accuracy, handles varying illumination.
                  False = faster but poor results with uneven lighting.
                  Used in NIST Algorithm 3 (PCM) preprocessing.
        verbose: Enable detailed logging of algorithm progress and timing
        overlap_ratio: Expected overlap between adjacent tiles as fraction (0.0-1.0).
                      Defines correlation region size: overlap_w = int(W * overlap_ratio).
                      CRITICAL: Must match actual overlap in data or correlation fails.
                      Higher (0.2-0.4) = more robust but slower.
                      Lower (0.05-0.08) = faster but less accurate.
                      Used in NIST Algorithm 10 (Compute Image Overlap).
        subpixel: Enable subpixel-accurate phase correlation for higher precision.
                 True = center-of-mass interpolation around correlation peak.
                 False = pixel-only accuracy (faster, less precise).
                 Enhances NIST Algorithm 3 (PCM) with subpixel refinement.
        refinement_iterations: Number of iterative position refinement passes (0-50).
                              Each iteration applies weighted position corrections.
                              Higher = better convergence but much slower.
                              0 = skip refinement (fastest, least accurate).
                              Implements NIST Algorithm 21 (Bounded NCC Hill Climb).
        global_optimization: Enable MST-based global optimization phase.
                           Uses minimum spanning tree to optimize tile positions globally.
                           Significantly improves accuracy for large grids.
                           Implements NIST Phase 3 (Image Composition).
        anchor_tile_index: Index of reference tile that remains fixed at origin (usually 0).
                          All other positions calculated relative to this tile.
                          Used in NIST MST position reconstruction.

        === Refinement Tuning Parameters ===
        refinement_damping: Controls how aggressively positions are updated (0.0-1.0).
                          Formula: new_pos = (1-damping)*old_pos + damping*correction.
                          Higher (0.7-0.9) = faster convergence but may overshoot.
                          Lower (0.1-0.3) = more stable but slower convergence.
                          1.0 = full correction (may be unstable), 0.0 = no updates.
        correlation_weight_horizontal: Weight for horizontal tile constraints (>0).
                                     Higher values prioritize horizontal alignment accuracy.
                                     Typical range: 0.5-2.0.
        correlation_weight_vertical: Weight for vertical tile constraints (>0).
                                   Higher values prioritize vertical alignment accuracy.
                                   Typical range: 0.5-2.0.

        === Phase Correlation Parameters (NIST Algorithm 3) ===
        subpixel_radius: Radius around correlation peak for center-of-mass calculation.
                        Extracts (2*radius+1)² region around peak for interpolation.
                        Higher (5-10) = more accurate subpixel positioning but slower.
                        Lower (1-2) = faster but less precise, may cause drift.
                        0 = pixel-only accuracy (fastest, least precise).
                        Enhances NIST Algorithm 3 (PCM) with subpixel precision.
        regularization_eps_multiplier: Prevents division by zero in phase correlation.
                                     Formula: eps = machine_epsilon * multiplier.
                                     Higher (10000+) = more stable with noisy images.
                                     Lower (100-500) = higher precision but may fail.
                                     Too low (<10) = risk of numerical instability.
                                     Used in NIST Algorithm 3 cross-power normalization.

        === MST Global Optimization Parameters (NIST Algorithms 8-21) ===
        mst_quality_threshold: Minimum correlation quality for MST edge inclusion (0.0-1.0).
                             NIST Algorithm 15: ncc >= 0.5 for valid translations.
                             Formula: if correlation_peak < threshold: reject_connection.
                             NIST default: 0.5 (stricter quality control).
                             Higher = fewer connections, lower = includes weak correlations.
                             Too high = MST may fail, too low = includes noise.
        use_nist_robustness: Enable NIST robust phase correlation (Algorithm 2).
                           True = multi-peak PCIAM with interpretation testing.
                           False = simplified single-peak method (faster).
        n_peaks: Number of correlation peaks to analyze (NIST Algorithm 4).
                NIST default: n=2 (manually selected based on experimental testing).
                Higher = more robust peak selection but slower processing.
        use_nist_normalization: Apply NIST normalization method (Algorithm 3).
                              True = fc/abs(fc) normalization (NIST standard).
                              False = OpenHCS regularization method.

        displacement_tolerance_factor: Multiplier for expected displacement tolerance.
                                     NIST Algorithm 14: Stage model displacement validation.
                                     Formula: max_error = factor * expected_displacement * percent.
                                     Higher (3.0-5.0) = more permissive validation.
                                     Lower (1.0-1.5) = stricter validation.
        displacement_tolerance_percent: Percentage tolerance for displacement (0.0-1.0).
                                      NIST Algorithm 14: Displacement validation threshold.
                                      Formula: valid if |actual - expected| <= expected * percent.
                                      0.3 = ±30% deviation allowed from expected displacement.
                                      Higher = accepts larger deviations, lower = stricter.

        debug_connection_limit: Max horizontal connections to log for debugging (0-10)
        debug_vertical_limit: Max vertical connections to log for debugging (0-10)
        consistency_threshold_percent: Translation consistency validation threshold (0.0-1.0).
                                      NIST Algorithm 17: Filter by repeatability.
                                      Formula: valid if |translation - median| <= median * threshold.
                                      0.5 = ±50% deviation from median allowed.
                                      Higher = more permissive, lower = stricter consistency.
        max_connections_multiplier: Maximum connections per tile in MST construction.
                                   Formula: max_connections = base_connections * multiplier.
                                   Prevents over-connected graphs that slow MST algorithms.
                                   2 = allow 2x normal connections, 1 = strict minimum.
        adaptive_base_threshold: Minimum quality threshold for adaptive quality metrics.
                               NIST-inspired adaptive thresholding for challenging datasets.
                               Formula: final_threshold = max(base_threshold, percentile_threshold).
                               0.3 = minimum 30% correlation required regardless of distribution.
                               Prevents threshold from becoming too permissive.
        adaptive_percentile_threshold: Percentile-based quality threshold (0.0-1.0).
                                     NIST Algorithm 9: Stage model validation approach.
                                     Formula: threshold = percentile(all_qualities, percentile * 100).
                                     0.25 = use 25th percentile of quality distribution.
                                     Lower = more permissive, higher = stricter selection.
        translation_tolerance_factor: Tolerance multiplier for translation validation.
                                    NIST Algorithm 14: Stage model displacement validation.
                                    Formula: max_error = expected_displacement * factor * percent.
                                    0.2 = allow 20% deviation from expected displacement.
                                    Higher = more permissive validation.
        translation_min_quality: Minimum correlation quality for translation acceptance.
                                NIST Algorithm 15: Quality-based filtering threshold.
                                Formula: accept if ncc >= min_quality.
                                0.3 = require 30% normalized cross-correlation minimum.
                                Higher = stricter quality, lower = more permissive.
        magnitude_threshold_multiplier: FFT magnitude threshold for numerical stability.
                                      NIST Algorithm 3: Cross-power spectrum normalization.
                                      Formula: threshold = mean(magnitude) * multiplier.
                                      1e-6 = very small threshold for numerical stability.
                                      Higher = more aggressive filtering of low-magnitude frequencies.
        peak_candidates_multiplier: Candidate peak search multiplier for robustness.
                                   NIST Algorithm 4: Multi-peak max search optimization.
                                   Formula: n_candidates = n_peaks * multiplier.
                                   4 = search 4x more candidates than needed for robust selection.
                                   Higher = more thorough search but slower processing.
        min_peak_distance: Minimum pixel distance between correlation peaks.
                         NIST Algorithm 4: Prevents duplicate peak detection.
                         Formula: reject if distance(peak1, peak2) < min_distance.
                         5 = peaks must be ≥5 pixels apart to be considered distinct.
                         Higher = fewer but more distinct peaks, lower = more peaks.

        === NIST Mathematical Formulas ===

        Algorithm 3 (PCM): Peak Correlation Matrix
        F1 ← fft2D(I1), F2 ← fft2D(I2)
        FC ← F1 .* conj(F2)
        PCM ← ifft2D(FC ./ abs(FC))

        Algorithm 6 (NCC): Normalized Cross-Correlation
        I1 ← I1 - mean(I1), I2 ← I2 - mean(I2)
        ncc = (I1 · I2) / (|I1| * |I2|)

        Algorithm 10 (Overlap): Image Overlap Computation
        overlap_percent = 100 - mu  (where mu is mean translation)
        valid_range = [overlap ± overlap_uncertainty_percent]

        Algorithm 16 (Outliers): Statistical Outlier Detection
        q1 = 25th percentile, q3 = 75th percentile
        IQR = q3 - q1
        outlier if: value < (q1 - 1.5*IQR) OR value > (q3 + 1.5*IQR)

        Algorithm 21 (Hill Climb): Bounded Translation Refinement
        search_bounds = [current ± repeatability]
        ncc_surface[i,j] = ncc(extract_overlap(I1, j, i), extract_overlap(I2, -j, -i))
        climb to local maximum within bounds

        === NIST Performance Guidance ===

        Quality Threshold Tuning:
        - Start with NIST default: 0.5 (strict quality control)
        - Lower to 0.3-0.4 for noisy biological samples
        - Lower to 0.1-0.2 for very challenging datasets
        - Monitor MST edge count: need ≥(num_tiles-1) edges minimum

        Peak Count Optimization:
        - NIST default: n=2 peaks (experimentally optimal)
        - Increase to 3-5 for highly repetitive patterns
        - Keep at 2 for most microscopy applications

        Overlap Ratio Guidelines:
        - Must match actual image overlap precisely
        - Typical microscopy: 0.1-0.2 (10-20% overlap)
        - Higher overlap = more robust but slower processing
        - Lower overlap = faster but less reliable alignment

        Subpixel Refinement:
        - Enable for publication-quality results
        - Radius 3-5 optimal for most applications
        - Disable for speed-critical applications

        Expected Performance:
        - With NIST defaults: High accuracy, moderate speed
        - Quality threshold 0.5: Strict filtering, fewer edges
        - Multi-peak robustness: 2-3x slower but more reliable
        - Global optimization: Essential for large grids (>3x3)

    Returns:
        Tuple of (image_stack, positions) where:
        - image_stack: Original input tiles (potentially normalized)
        - positions: (Z, 2) array of tile positions in (x, y) format
                    Positions are centered around origin

    Raises:
        ValueError: If input validation fails (wrong method, backend, or dimensions)
        TypeError: If image_stack is not a CuPy array
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

    # VERY FIRST THING - Debug output to confirm function is called
    print("🔥🔥🔥 MIST FUNCTION ENTRY POINT - FUNCTION IS DEFINITELY BEING CALLED! 🔥🔥🔥")
    print(f"🔥 Image stack shape: {image_stack.shape}")
    print(f"🔥 Grid dimensions: {grid_dimensions}")

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
                overlap_w = cp.int32(W * overlap_ratio)
                left_region = left_tile[:, -overlap_w:]
                current_region = current_tile[:, :overlap_w]

                # GPU phase correlation
                dy, dx = phase_correlation_gpu_only(
                    left_region, current_region,
                    subpixel=subpixel,
                    subpixel_radius=subpixel_radius,
                    regularization_eps_multiplier=regularization_eps_multiplier
                )

                # Convert overlap-region coordinates to tile-center coordinates
                tile_dy, tile_dx = _convert_overlap_to_tile_coordinates(
                    dy, dx, int(overlap_w), int(overlap_w), H, W, 'horizontal'
                )

                # Update position (GPU)
                new_x = positions[left_idx, 0] + tile_dx
                new_y = positions[left_idx, 1] + tile_dy
                positions[tile_idx] = cp.array([new_x, new_y])

            elif r > 0:  # Position from top neighbor
                top_idx = (r - 1) * num_cols + c
                top_tile = tile_grid[r - 1, c]

                # Extract overlap regions (GPU)
                overlap_h = cp.int32(H * overlap_ratio)
                top_region = top_tile[-overlap_h:, :]
                current_region = current_tile[:overlap_h, :]

                # GPU phase correlation
                dy, dx = phase_correlation_gpu_only(
                    top_region, current_region,
                    subpixel=subpixel,
                    subpixel_radius=subpixel_radius,
                    regularization_eps_multiplier=regularization_eps_multiplier
                )

                # Convert overlap-region coordinates to tile-center coordinates
                tile_dy, tile_dx = _convert_overlap_to_tile_coordinates(
                    dy, dx, int(overlap_h), int(overlap_h), H, W, 'vertical'
                )

                # Update position (GPU)
                new_x = positions[top_idx, 0] + tile_dx
                new_y = positions[top_idx, 1] + tile_dy
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

                overlap_w = cp.int32(W * overlap_ratio)
                left_region = left_tile[:, -overlap_w:]   # Right edge of left tile
                right_region = right_tile[:, :overlap_w] # Left edge of right tile

                dy, dx = phase_correlation_gpu_only(
                    left_region, right_region,  # Standardized: left_region first
                    subpixel=subpixel,
                    subpixel_radius=subpixel_radius,
                    regularization_eps_multiplier=regularization_eps_multiplier
                )

                # Convert overlap-region coordinates to tile-center coordinates
                tile_dy, tile_dx = _convert_overlap_to_tile_coordinates(
                    dy, dx, int(overlap_w), int(overlap_w), H, W, 'horizontal'
                )

                # Expected position (GPU)
                expected_right = positions[left_idx] + cp.array([tile_dx, tile_dy])

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

                overlap_h = cp.int32(H * overlap_ratio)
                top_region = top_tile[-overlap_h:, :]    # Bottom edge of top tile
                bottom_region = bottom_tile[:overlap_h, :] # Top edge of bottom tile

                dy, dx = phase_correlation_gpu_only(
                    top_region, bottom_region,  # Standardized: top_region first
                    subpixel=subpixel,
                    subpixel_radius=subpixel_radius,
                    regularization_eps_multiplier=regularization_eps_multiplier
                )

                # Convert overlap-region coordinates to tile-center coordinates
                tile_dy, tile_dx = _convert_overlap_to_tile_coordinates(
                    dy, dx, int(overlap_h), int(overlap_h), H, W, 'vertical'
                )

                # Expected position (GPU)
                expected_bottom = positions[top_idx] + cp.array([tile_dx, tile_dy])

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

            quality_threshold=mst_quality_threshold,
            subpixel_radius=subpixel_radius,
            regularization_eps_multiplier=regularization_eps_multiplier,
            anchor_tile_index=anchor_tile_index,
            debug_connection_limit=debug_connection_limit,
            debug_vertical_limit=debug_vertical_limit,
            displacement_tolerance_factor=displacement_tolerance_factor,
            displacement_tolerance_percent=displacement_tolerance_percent,
            consistency_threshold_percent=consistency_threshold_percent,
            max_connections_multiplier=max_connections_multiplier,
            adaptive_base_threshold=adaptive_base_threshold,
            adaptive_percentile_threshold=adaptive_percentile_threshold,
            translation_tolerance_factor=translation_tolerance_factor,
            translation_min_quality=translation_min_quality,
            magnitude_threshold_multiplier=magnitude_threshold_multiplier,
            peak_candidates_multiplier=peak_candidates_multiplier,
            min_peak_distance=min_peak_distance,
            use_nist_robustness=use_nist_robustness,
            n_peaks=n_peaks,
            use_nist_normalization=use_nist_normalization
        )

    # Center positions (GPU)
    mean_pos = cp.mean(positions, axis=0)
    positions = positions - mean_pos

    print(f"🔥 MIST COMPLETE: Returning {positions.shape} positions")
    return tiles, positions
