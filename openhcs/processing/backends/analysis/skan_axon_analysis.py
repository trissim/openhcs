"""
Skan-based axon skeletonization and analysis for OpenHCS.

This module provides comprehensive axon analysis using the skan library,
including segmentation, skeletonization, and quantitative skeleton analysis.
Supports both 2D and 3D analysis modes with multiple output formats.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import logging

# OpenHCS imports
from openhcs.core.memory import numpy as numpy_func
from openhcs.core.pipeline.function_contracts import artifact_outputs
from openhcs.processing.materialization import MaterializationSpec, CsvOptions, JsonOptions, ROIOptions, TiffStackOptions
from polystore.roi import ROI

logger = logging.getLogger(__name__)


class ThresholdMethod(Enum):
    """Segmentation methods for axon detection."""
    OTSU = "otsu"
    MANUAL = "manual"
    ADAPTIVE = "adaptive"


class OutputMode(Enum):
    """Output array format options."""
    SKELETON = "skeleton"
    SKELETON_OVERLAY = "skeleton_overlay"
    ORIGINAL = "original"
    COMPOSITE = "composite"


class AnalysisDimension(Enum):
    """Analysis dimension modes."""
    TWO_D = "2d"
    THREE_D = "3d"



@artifact_outputs(
    (
        "axon_analysis",
        MaterializationSpec(
            JsonOptions(filename_suffix=".json"),
            CsvOptions(source="branch_data", filename_suffix="_branches.csv"),
            primary=0,
        ),
    ),
    (
        "skeleton_visualizations",
        MaterializationSpec(
            TiffStackOptions(
                normalize_uint8=True,
                summary_suffix="_skeleton_summary.txt",
            )
        ),
    ),
    ("skeleton_masks", MaterializationSpec(ROIOptions())),
)
@numpy_func
def skan_axon_skeletonize_and_analyze(
    image_stack: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    threshold_method: ThresholdMethod = ThresholdMethod.OTSU,
    threshold_value: Optional[float] = None,
    min_object_size: int = 100,
    min_branch_length: float = 0.0,
    filter_edge: Optional[str] = None,  # Filter objects not touching this edge: 'left', 'right', 'top', 'bottom', None
    return_skeleton_visualizations: bool = False,
    skeleton_visualization_mode: OutputMode = OutputMode.SKELETON_OVERLAY,
    analysis_dimension: AnalysisDimension = AnalysisDimension.THREE_D,
    return_skeleton_mask: bool = True  # Return skeleton mask (gets converted to ROIs)
) -> Tuple[np.ndarray, Dict[str, Any], List[np.ndarray], np.ndarray]:
    """
    Skeletonize axon images and perform comprehensive skeleton analysis.

    Complete workflow: segmentation → skeletonization → analysis

    Args:
        image_stack: 3D grayscale image to skeletonize (Z, Y, X format)
        voxel_spacing: Physical voxel spacing (z, y, x) in micrometers
        threshold_method: Segmentation method (OTSU, MANUAL, ADAPTIVE)
        threshold_value: Manual threshold value (if threshold_method=MANUAL)
        min_object_size: Minimum object size for noise removal (voxels)
        min_branch_length: Minimum branch length threshold (micrometers)
        filter_edge: Keep only objects touching this edge ('left', 'right', 'top', 'bottom', or None for no filtering)
        return_skeleton_visualizations: Whether to generate skeleton visualizations as special output
        skeleton_visualization_mode: Type of visualization (SKELETON, SKELETON_OVERLAY, ORIGINAL, COMPOSITE)
        analysis_dimension: Analysis mode (TWO_D or THREE_D)
        return_skeleton_mask: Whether to return skeleton binary mask (converted to ROIs for Napari/Fiji)

    Returns:
        Tuple containing:
        - Original image stack: Input image unchanged (Z, Y, X)
        - Axon analysis results: Complete analysis data structure
        - Skeleton visualizations: (Special output) List of visualization arrays if return_skeleton_visualizations=True
        - Skeleton mask: (Special output) Binary skeleton mask (Z, Y, X) - gets converted to ROIs by materializer
    """
    # Validate input
    if len(image_stack.shape) != 3:
        raise ValueError(f"Expected 3D image, got {len(image_stack.shape)}D with shape {image_stack.shape}")
    
    if threshold_method == ThresholdMethod.MANUAL and threshold_value is None:
        raise ValueError("threshold_value required when threshold_method=MANUAL")
    
    logger.info(f"Starting skan axon analysis: {image_stack.shape} image (ndim={image_stack.ndim}, dtype={image_stack.dtype})")
    logger.info(f"Parameters: threshold={threshold_method.value}, "
                f"analysis={analysis_dimension.value}, visualizations={return_skeleton_visualizations}")
    
    # Step 1: Segmentation/Thresholding
    binary_stack = _segment_axons(image_stack, threshold_method, threshold_value)
    
    # Step 2: Noise removal
    if min_object_size > 0:
        binary_stack = _remove_small_objects(binary_stack, min_object_size)

    # Step 2.5: Edge filtering (keep only objects touching specified edge)
    if filter_edge is not None:
        binary_stack = _filter_by_edge(binary_stack, filter_edge)

    # Step 3: Skeletonization
    skeleton_stack = _skeletonize_3d(binary_stack)

    # Step 4: Skeleton analysis (initial pass to identify branches)
    if analysis_dimension == AnalysisDimension.THREE_D:
        branch_data = _analyze_3d_skeleton(skeleton_stack, voxel_spacing)
        analysis_type = "3D volumetric"
    elif analysis_dimension == AnalysisDimension.TWO_D:
        branch_data = _analyze_2d_slices(skeleton_stack, voxel_spacing)
        analysis_type = "2D slice-by-slice"
    else:
        raise ValueError(f"Invalid analysis_dimension: {analysis_dimension}")

    # Step 5: Prune small branches from skeleton (modifies skeleton geometry)
    if min_branch_length > 0 and len(branch_data) > 0:
        skeleton_stack = _prune_skeleton(skeleton_stack, branch_data, min_branch_length, voxel_spacing, analysis_dimension)

        # Re-analyze after pruning to get updated branch data
        if analysis_dimension == AnalysisDimension.THREE_D:
            branch_data = _analyze_3d_skeleton(skeleton_stack, voxel_spacing)
        else:
            branch_data = _analyze_2d_slices(skeleton_stack, voxel_spacing)

    # Step 6: Generate skeleton visualizations if requested
    skeleton_visualizations = []
    if return_skeleton_visualizations:
        # Generate visualization for each slice
        for z in range(image_stack.shape[0]):
            slice_image = image_stack[z]
            slice_binary = binary_stack[z]
            slice_skeleton = skeleton_stack[z]

            # Create visualization for this slice
            visualization = _create_output_array_2d(
                slice_image, slice_binary, slice_skeleton, skeleton_visualization_mode
            )
            skeleton_visualizations.append(visualization)

    # Step 7: Return skeleton mask if requested (converted to per-branch labels for ROI writer)
    skeleton_mask_output = (
        _skeleton_mask_to_labels(skeleton_stack)
        if return_skeleton_mask
        else np.zeros((0, 0, 0), dtype=np.uint16)
    )

    # Step 8: Compile comprehensive results
    results = _compile_analysis_results(
        branch_data, skeleton_stack, binary_stack, image_stack,
        voxel_spacing, analysis_type, threshold_method, min_object_size, min_branch_length
    )

    logger.info(f"Analysis complete: {len(branch_data)} branches found")
    if return_skeleton_mask:
        logger.info(f"Returning skeleton mask for ROI conversion: {skeleton_mask_output.shape}")

    # Return: original image, analysis results, skeleton visualizations, skeleton mask
    return image_stack, results, skeleton_visualizations, skeleton_mask_output


# Helper functions for segmentation and preprocessing
def _segment_axons(image_stack, threshold_method, threshold_value):
    """Segment axons from grayscale image."""
    from skimage import filters

    if threshold_method == ThresholdMethod.OTSU:
        # Global Otsu thresholding
        threshold = filters.threshold_otsu(image_stack)
        binary_stack = image_stack > threshold
        logger.debug(f"Otsu threshold: {threshold}")

    elif threshold_method == ThresholdMethod.MANUAL:
        # Manual threshold (threshold_value already validated)
        binary_stack = image_stack > threshold_value
        logger.debug(f"Manual threshold: {threshold_value}")

    elif threshold_method == ThresholdMethod.ADAPTIVE:
        # Slice-by-slice adaptive thresholding
        binary_stack = np.zeros_like(image_stack, dtype=bool)
        for z in range(image_stack.shape[0]):
            if image_stack[z].max() > 0:  # Skip empty slices
                threshold = filters.threshold_local(image_stack[z], block_size=51)
                binary_stack[z] = image_stack[z] > threshold
        logger.debug("Applied adaptive thresholding slice-by-slice")

    else:
        raise ValueError(f"Unknown threshold_method: {threshold_method}")

    return binary_stack


def _remove_small_objects(binary_stack, min_size):
    """Remove small objects from binary image."""
    from skimage import morphology

    # Apply to each slice to preserve 3D connectivity
    cleaned_stack = np.zeros_like(binary_stack)
    removed_count = 0

    for z in range(binary_stack.shape[0]):
        if binary_stack[z].any():
            original_objects = np.sum(binary_stack[z])
            cleaned_stack[z] = morphology.remove_small_objects(
                binary_stack[z], min_size=min_size
            )
            removed_objects = original_objects - np.sum(cleaned_stack[z])
            removed_count += removed_objects

    logger.debug(f"Removed {removed_count} small object pixels (min_size={min_size})")
    return cleaned_stack


def _skeletonize_3d(binary_stack):
    """Create 3D skeleton from binary image."""
    from skimage import morphology

    # Use 3D skeletonization to preserve connectivity
    skeleton_stack = morphology.skeletonize(binary_stack)

    # Count skeleton pixels for logging
    skeleton_pixels = np.sum(skeleton_stack)
    binary_pixels = np.sum(binary_stack)
    reduction_ratio = skeleton_pixels / binary_pixels if binary_pixels > 0 else 0

    logger.debug(f"Skeletonization: {binary_pixels} → {skeleton_pixels} pixels "
                f"(reduction: {reduction_ratio:.3f})")

    return skeleton_stack


def _prune_skeleton(skeleton_stack: np.ndarray, branch_data: pd.DataFrame,
                   min_branch_length: float, voxel_spacing: Tuple[float, float, float],
                   analysis_dimension: AnalysisDimension) -> np.ndarray:
    """
    Remove branches shorter than min_branch_length from the skeleton.

    This actually modifies the skeleton geometry, not just the analysis results.

    Args:
        skeleton_stack: Binary skeleton mask (Z, Y, X)
        branch_data: DataFrame from skan analysis
        min_branch_length: Minimum branch length threshold (micrometers)
        voxel_spacing: Physical voxel spacing (z, y, x)
        analysis_dimension: Whether this is 2D or 3D analysis

    Returns:
        Pruned skeleton mask
    """
    from skan import Skeleton

    if not skeleton_stack.any() or len(branch_data) == 0:
        return skeleton_stack

    # Identify branches to remove (below threshold)
    short_branches = branch_data[branch_data['branch_distance'] < min_branch_length]

    if len(short_branches) == 0:
        logger.info(f"No branches below {min_branch_length}µm threshold - no pruning needed")
        return skeleton_stack

    # Create skeleton object to get pixel indices for each branch
    if analysis_dimension == AnalysisDimension.THREE_D:
        skeleton_obj = Skeleton(skeleton_stack, spacing=voxel_spacing)

        # Get coordinates of pixels to remove
        pruned_skeleton = skeleton_stack.copy()

        for idx, row in short_branches.iterrows():
            # Get path coordinates for this branch
            path = skeleton_obj.path_coordinates(idx)

            # Remove these pixels from skeleton
            if len(path) > 0:
                coords = np.round(path).astype(int)
                # Ensure coordinates are within bounds
                valid_mask = (
                    (coords[:, 0] >= 0) & (coords[:, 0] < skeleton_stack.shape[0]) &
                    (coords[:, 1] >= 0) & (coords[:, 1] < skeleton_stack.shape[1]) &
                    (coords[:, 2] >= 0) & (coords[:, 2] < skeleton_stack.shape[2])
                )
                coords = coords[valid_mask]

                if len(coords) > 0:
                    pruned_skeleton[coords[:, 0], coords[:, 1], coords[:, 2]] = 0

    else:  # 2D analysis - prune each slice independently
        pruned_skeleton = skeleton_stack.copy()

        for z in range(skeleton_stack.shape[0]):
            slice_skeleton = skeleton_stack[z]

            if not slice_skeleton.any():
                continue

            # Get branches for this slice
            slice_branches = short_branches[short_branches['z_slice'] == z] if 'z_slice' in short_branches.columns else pd.DataFrame()

            if len(slice_branches) == 0:
                continue

            # Create 2D skeleton object for this slice
            skeleton_obj_2d = Skeleton(slice_skeleton, spacing=voxel_spacing[1:])

            for idx, row in slice_branches.iterrows():
                # Get path coordinates for this branch
                path = skeleton_obj_2d.path_coordinates(idx)

                if len(path) > 0:
                    coords = np.round(path).astype(int)
                    # Ensure coordinates are within bounds
                    valid_mask = (
                        (coords[:, 0] >= 0) & (coords[:, 0] < slice_skeleton.shape[0]) &
                        (coords[:, 1] >= 0) & (coords[:, 1] < slice_skeleton.shape[1])
                    )
                    coords = coords[valid_mask]

                    if len(coords) > 0:
                        pruned_skeleton[z, coords[:, 0], coords[:, 1]] = 0

    branches_before = len(branch_data)
    branches_after = branches_before - len(short_branches)
    logger.info(f"Pruned {len(short_branches)} branches < {min_branch_length}µm ({branches_before} → {branches_after} branches)")

    return pruned_skeleton


def _filter_by_edge(binary_stack: np.ndarray, edge: str) -> np.ndarray:
    """
    Keep only objects that touch the specified edge of the image.

    Removes artifacts and objects that don't originate from the specified edge.
    Useful for filtering out debris while keeping neurites/axons that enter from one side.

    Args:
        binary_stack: 3D binary mask (Z, Y, X)
        edge: Which edge to filter by ('left', 'right', 'top', 'bottom')

    Returns:
        Filtered binary mask with only objects touching the specified edge
    """
    from skimage.measure import label

    valid_edges = {'left', 'right', 'top', 'bottom'}
    if edge.lower() not in valid_edges:
        raise ValueError(f"Invalid edge '{edge}'. Must be one of: {valid_edges}")

    edge = edge.lower()

    # Process each Z slice independently
    filtered_stack = np.zeros_like(binary_stack)

    for z in range(binary_stack.shape[0]):
        slice_binary = binary_stack[z]

        if not slice_binary.any():
            continue

        # Label connected components
        labeled = label(slice_binary)

        # Get labels that touch the specified edge
        edge_labels = set()

        if edge == 'left':
            # First column (x=0)
            edge_labels.update(labeled[:, 0])
        elif edge == 'right':
            # Last column (x=-1)
            edge_labels.update(labeled[:, -1])
        elif edge == 'top':
            # First row (y=0)
            edge_labels.update(labeled[0, :])
        elif edge == 'bottom':
            # Last row (y=-1)
            edge_labels.update(labeled[-1, :])

        # Remove background label (0)
        edge_labels.discard(0)

        # Keep only objects that touch the edge
        if edge_labels:
            mask = np.isin(labeled, list(edge_labels))
            filtered_stack[z] = mask

    objects_before = np.unique(label(binary_stack))[1:]  # Exclude background
    objects_after = np.unique(label(filtered_stack))[1:]
    logger.info(f"Edge filtering ({edge}): {len(objects_before)} → {len(objects_after)} objects")

    return filtered_stack


def _create_empty_branch_dataframe(include_2d_columns: bool = False):
    """
    Create an empty DataFrame with the expected skan branch data schema.

    This ensures consistent DataFrame structure even when no branches are found,
    preventing KeyError when filtering or processing results.

    Args:
        include_2d_columns: If True, include additional columns for 2D slice analysis

    Returns:
        Empty DataFrame with proper column schema
    """
    # Core columns from skan.summarize()
    columns = [
        'skeleton_id',
        'node_id_src',
        'node_id_dst',
        'branch_distance',
        'branch_type',
        'mean_pixel_value',
        'stdev_pixel_value',
        'image_coord_src_0',
        'image_coord_src_1',
        'image_coord_src_2',
        'image_coord_dst_0',
        'image_coord_dst_1',
        'image_coord_dst_2',
        'coord_src_0',
        'coord_src_1',
        'coord_src_2',
        'coord_dst_0',
        'coord_dst_1',
        'coord_dst_2',
        'euclidean_distance',
    ]

    # Add 2D-specific columns if requested
    if include_2d_columns:
        columns.extend(['z_slice', 'z_coord', 'skeleton_id'])

    return pd.DataFrame(columns=columns)


def _analyze_3d_skeleton(skeleton_stack, voxel_spacing):
    """Analyze skeleton as single 3D network."""
    try:
        from skan import Skeleton, summarize
    except ImportError:
        raise ImportError("skan library is required for skeleton analysis. "
                         "Install with: pip install skan")

    if not skeleton_stack.any():
        logger.warning("Empty skeleton - returning empty analysis")
        return _create_empty_branch_dataframe()

    # Single 3D analysis - preserves Z-connections
    skeleton_obj = Skeleton(skeleton_stack, spacing=voxel_spacing)
    branch_data = summarize(skeleton_obj, separator='_')

    logger.debug(f"3D analysis: {len(branch_data)} branches found")
    return branch_data


def _analyze_2d_slices(skeleton_stack, voxel_spacing):
    """Analyze each Z-slice as separate 2D skeleton."""
    try:
        from skan import Skeleton, summarize
    except ImportError:
        raise ImportError("skan library is required for skeleton analysis. "
                         "Install with: pip install skan")

    all_branch_data = []
    z_spacing, y_spacing, x_spacing = voxel_spacing

    for z_idx, slice_skeleton in enumerate(skeleton_stack):
        if slice_skeleton.any():  # Skip empty slices
            # 2D analysis with XY spacing only
            skeleton_obj = Skeleton(slice_skeleton, spacing=(y_spacing, x_spacing))
            slice_data = summarize(skeleton_obj, separator='_')

            if len(slice_data) > 0:
                # Add Z-coordinate information
                slice_data['z_slice'] = z_idx
                slice_data['z_coord'] = z_idx * z_spacing
                slice_data['skeleton_id'] = f"slice_{z_idx:03d}"

                all_branch_data.append(slice_data)

    # Combine all slices
    if all_branch_data:
        combined_data = pd.concat(all_branch_data, ignore_index=True)
        logger.debug(f"2D analysis: {len(combined_data)} branches across "
                    f"{len(all_branch_data)} slices")
        return combined_data
    else:
        logger.warning("No skeleton data found in any slice")
        return _create_empty_branch_dataframe(include_2d_columns=True)


def _create_output_array_2d(slice_image, slice_binary, slice_skeleton, output_mode):
    """Generate 2D output array based on specified mode."""

    if output_mode == OutputMode.SKELETON:
        # Return binary skeleton
        return slice_skeleton.astype(np.uint8) * 255

    elif output_mode == OutputMode.SKELETON_OVERLAY:
        # Overlay skeleton on original image
        output = slice_image.copy()
        # Highlight skeleton pixels with maximum intensity
        if slice_skeleton.any():
            output[slice_skeleton] = slice_image.max()
        return output

    elif output_mode == OutputMode.ORIGINAL:
        # Return original unchanged
        return slice_image.copy()

    elif output_mode == OutputMode.COMPOSITE:
        # Side-by-side: original | binary | skeleton
        y, x = slice_image.shape
        composite = np.zeros((y, x * 3), dtype=slice_image.dtype)

        # Original image
        composite[:, :x] = slice_image

        # Binary segmentation (scaled to match original intensity range)
        binary_scaled = (slice_binary.astype(np.float32) * slice_image.max()).astype(slice_image.dtype)
        composite[:, x:2*x] = binary_scaled

        # Skeleton (scaled to match original intensity range)
        skeleton_scaled = (slice_skeleton.astype(np.float32) * slice_image.max()).astype(slice_image.dtype)
        composite[:, 2*x:3*x] = skeleton_scaled

        return composite

    else:
        raise ValueError(f"Unknown output_mode: {output_mode}")


def _create_output_array(image_stack, binary_stack, skeleton_stack, branch_data, output_mode):
    """Generate output array based on specified mode (legacy function, kept for compatibility)."""

    if output_mode == OutputMode.SKELETON:
        # Return binary skeleton
        return skeleton_stack.astype(np.uint8) * 255

    elif output_mode == OutputMode.SKELETON_OVERLAY:
        # Overlay skeleton on original image
        output = image_stack.copy()
        # Highlight skeleton pixels with maximum intensity
        if skeleton_stack.any():
            output[skeleton_stack] = image_stack.max()
        return output

    elif output_mode == OutputMode.ORIGINAL:
        # Return original unchanged
        return image_stack.copy()

    elif output_mode == OutputMode.COMPOSITE:
        # Side-by-side: original | binary | skeleton
        z, y, x = image_stack.shape
        composite = np.zeros((z, y, x * 3), dtype=image_stack.dtype)

        # Original image
        composite[:, :, :x] = image_stack

        # Binary segmentation (scaled to match original intensity range)
        binary_scaled = (binary_stack.astype(np.float32) * image_stack.max()).astype(image_stack.dtype)
        composite[:, :, x:2*x] = binary_scaled

        # Skeleton (scaled to match original intensity range)
        skeleton_scaled = (skeleton_stack.astype(np.float32) * image_stack.max()).astype(image_stack.dtype)
        composite[:, :, 2*x:3*x] = skeleton_scaled

        return composite

    else:
        raise ValueError(f"Unknown output_mode: {output_mode}")


def _compile_analysis_results(branch_data, skeleton_stack, binary_stack, image_stack,
                             voxel_spacing, analysis_type, threshold_method,
                             min_object_size, min_branch_length):
    """Compile complete analysis results."""

    # Compute summary metrics
    summary = _compute_summary_metrics(branch_data, skeleton_stack.shape, voxel_spacing)

    # Add segmentation metrics
    total_voxels = np.prod(image_stack.shape)
    binary_voxels = np.sum(binary_stack)
    skeleton_voxels = np.sum(skeleton_stack)

    segmentation_metrics = {
        'total_voxels': int(total_voxels),
        'segmented_voxels': int(binary_voxels),
        'skeleton_voxels': int(skeleton_voxels),
        'segmentation_fraction': float(binary_voxels / total_voxels),
        'skeleton_fraction': float(skeleton_voxels / binary_voxels) if binary_voxels > 0 else 0.0,
    }

    # Combine all results
    results = {
        'summary': {**summary, **segmentation_metrics},
        # Writer-based materialization expects list-of-dicts for tabular CSV.
        'branch_data': branch_data.to_dict('records') if len(branch_data) > 0 else [],
        'metadata': {
            'analysis_type': analysis_type,
            'voxel_spacing': voxel_spacing,
            'threshold_method': threshold_method.value,
            'min_object_size': min_object_size,
            'min_branch_length': min_branch_length,
            'image_shape': image_stack.shape,
            'image_dtype': str(image_stack.dtype),
            'intensity_range': (float(image_stack.min()), float(image_stack.max())),
            'processing_timestamp': datetime.now().isoformat(),
            'skan_version': _get_skan_version(),
        }
    }

    return results


def _compute_summary_metrics(branch_data, skeleton_shape, voxel_spacing):
    """Compute summary statistics from branch data."""
    if len(branch_data) == 0:
        return {
            'total_axon_length': 0.0,
            'num_branches': 0,
            'num_junction_points': 0,
            'num_endpoints': 0,
            'mean_branch_length': 0.0,
            'max_branch_length': 0.0,
            'mean_tortuosity': 0.0,
            'network_density': 0.0,
            'branching_ratio': 0.0,
            'total_volume': float(np.prod(skeleton_shape) * np.prod(voxel_spacing)),
        }

    # Basic metrics
    total_length = branch_data['branch_distance'].sum()
    num_branches = len(branch_data)
    mean_length = branch_data['branch_distance'].mean()
    max_length = branch_data['branch_distance'].max()

    # Tortuosity (branch_distance / euclidean_distance)
    tortuosity = branch_data['branch_distance'] / (branch_data['euclidean_distance'] + 1e-8)
    mean_tortuosity = tortuosity.mean()

    # Count junction points and endpoints based on branch types
    # Branch types: 0=endpoint-endpoint, 1=junction-endpoint, 2=junction-junction, 3=cycle
    junction_branches = branch_data[branch_data['branch_type'].isin([1, 2])]
    num_junction_points = len(junction_branches['node_id_src'].unique()) if len(junction_branches) > 0 else 0

    endpoint_branches = branch_data[branch_data['branch_type'].isin([0, 1])]
    num_endpoints = len(endpoint_branches) * 2 if len(endpoint_branches) > 0 else 0  # Each branch has 2 endpoints

    # Volume and density
    total_volume = float(np.prod(skeleton_shape) * np.prod(voxel_spacing))
    network_density = num_branches / total_volume if total_volume > 0 else 0.0

    # Branching ratio
    branching_ratio = num_junction_points / num_endpoints if num_endpoints > 0 else 0.0

    return {
        'total_axon_length': float(total_length),
        'num_branches': int(num_branches),
        'num_junction_points': int(num_junction_points),
        'num_endpoints': int(num_endpoints),
        'mean_branch_length': float(mean_length),
        'max_branch_length': float(max_length),
        'mean_tortuosity': float(mean_tortuosity),
        'network_density': float(network_density),
        'branching_ratio': float(branching_ratio),
        'total_volume': total_volume,
    }


def _get_skan_version():
    """Get skan library version."""
    try:
        import skan
        return skan.__version__
    except (ImportError, AttributeError):
        return "unknown"


def _skeleton_mask_to_labels(skeleton_stack: np.ndarray) -> np.ndarray:
    """
    Convert a binary skeleton mask into a label mask using skan's branch identification.

    Uses skan.Skeleton.path_label_image() to label each pixel according to which branch
    it belongs to. This ensures the visualization matches exactly what skan analyzes.

    Args:
        skeleton_stack: Binary skeleton mask (Z, Y, X)

    Returns:
        Label mask (Z, Y, X) where each branch has a unique integer label matching skan's analysis
    """
    from skan import Skeleton

    if skeleton_stack.ndim == 2:
        skeleton_stack = skeleton_stack[np.newaxis, :, :]
        was_2d = True
    elif skeleton_stack.ndim != 3:
        logger.error(f"🔍 SKELETON_TO_LABELS: Expected 2D/3D skeleton, got {skeleton_stack.ndim}D")
        return np.zeros_like(skeleton_stack, dtype=np.uint16)
    else:
        was_2d = False

    # Check if skeleton is empty
    if not skeleton_stack.any():
        logger.info("🔍 SKELETON_TO_LABELS: Empty skeleton mask, returning zeros")
        label_stack = np.zeros_like(skeleton_stack, dtype=np.uint16)
        return label_stack[0] if was_2d else label_stack

    # Create output label array
    label_stack = np.zeros_like(skeleton_stack, dtype=np.uint16)

    current_label = 1

    # Process each Z-slice independently
    for z_idx, skeleton_slice in enumerate(skeleton_stack):
        if not skeleton_slice.any():
            continue

        try:
            # Create skan Skeleton object for this slice
            skeleton_obj = Skeleton(skeleton_slice)

            # Get path label image - this labels ALL skeleton pixels according to branch
            # This matches exactly what skan analyzes
            slice_labels = skeleton_obj.path_label_image()

            # Renumber labels to be globally unique across all slices
            if slice_labels.max() > 0:
                # Shift labels to avoid conflicts between slices
                slice_labels_shifted = slice_labels.copy()
                slice_labels_shifted[slice_labels > 0] += (current_label - 1)
                label_stack[z_idx] = slice_labels_shifted.astype(np.uint16)

                num_branches = int(slice_labels.max())
                current_label += num_branches

                # Verify all skeleton pixels are labeled
                skeleton_pixels = skeleton_slice.sum()
                labeled_pixels = (slice_labels > 0).sum()
                if skeleton_pixels != labeled_pixels:
                    logger.warning(f"🔍 SKELETON_TO_LABELS: Z-slice {z_idx} has {skeleton_pixels} skeleton pixels but only {labeled_pixels} labeled")

                logger.debug(f"🔍 SKELETON_TO_LABELS: Labeled {num_branches} branches in Z-slice {z_idx} ({labeled_pixels} pixels)")

        except Exception as e:
            logger.warning(f"🔍 SKELETON_TO_LABELS: Failed to label Z-slice {z_idx}: {e}")
            continue

    total_labels = current_label - 1
    total_labeled_pixels = (label_stack > 0).sum()
    total_skeleton_pixels = skeleton_stack.sum()

    logger.info(f"🔍 SKELETON_TO_LABELS: Created label mask with {total_labels} branches using skan's branch identification")
    logger.info(f"🔍 SKELETON_TO_LABELS: Labeled {total_labeled_pixels}/{total_skeleton_pixels} skeleton pixels")

    return label_stack[0] if was_2d else label_stack


def _skeleton_mask_to_rois(skeleton_stack: np.ndarray) -> List[ROI]:
    """
    Convert a binary skeleton mask into ROI objects using skan branch paths.

    Uses skan's Skeleton object to extract actual branch paths, creating one ROI
    per continuous skeleton segment. This preserves skeleton connectivity and
    creates proper polyline ROIs instead of fragmenting the skeleton.

    Args:
        skeleton_stack: Binary skeleton mask (Z, Y, X)

    Returns:
        List of ROI objects, one per skeleton branch
    """
    from skan import Skeleton
    from polystore.roi import PolylineShape, ROI

    rois: List[ROI] = []

    if skeleton_stack.ndim == 2:
        skeleton_stack = skeleton_stack[np.newaxis, :, :]
    elif skeleton_stack.ndim != 3:
        logger.error(f"🔍 SKELETON_TO_ROIS: Expected 2D/3D skeleton, got {skeleton_stack.ndim}D")
        return rois

    # Check if skeleton is empty
    if not skeleton_stack.any():
        logger.info("🔍 SKELETON_TO_ROIS: Empty skeleton mask, no ROIs to extract")
        return rois

    # Process each Z-slice independently to create 2D ROIs
    for z_idx, skeleton_slice in enumerate(skeleton_stack):
        if not skeleton_slice.any():
            continue

        try:
            # Create skan Skeleton object for this slice
            skeleton_obj = Skeleton(skeleton_slice)

            # Get number of branches (paths) in this skeleton
            num_branches = skeleton_obj.n_paths

            if num_branches == 0:
                logger.debug(f"🔍 SKELETON_TO_ROIS: No branches found in Z-slice {z_idx}")
                continue

            # Extract each branch as a separate ROI
            for branch_idx in range(num_branches):
                # Get the pixel coordinates for this branch path
                # path_coordinates returns (N, 2) array of (row, col) = (y, x) coordinates
                path_coords = skeleton_obj.path_coordinates(branch_idx)

                if len(path_coords) < 2:
                    # Skip degenerate paths (single pixel or empty)
                    logger.debug(f"🔍 SKELETON_TO_ROIS: Skipping degenerate path {branch_idx} in Z-slice {z_idx} (length={len(path_coords)})")
                    continue

                # Create polyline shape from path coordinates
                # skan returns (row, col) which is (y, x) - exactly what PolylineShape expects
                shape = PolylineShape(coordinates=path_coords)

                # Create ROI with metadata
                metadata = {
                    'position': z_idx,
                    'label': f'Skeleton_Z{z_idx:03d}_Branch{branch_idx:03d}',
                    'branch_index': branch_idx,
                    'path_length': len(path_coords)
                }

                roi = ROI(shapes=[shape], metadata=metadata)
                rois.append(roi)

            logger.debug(f"🔍 SKELETON_TO_ROIS: Extracted {num_branches} branch ROIs from Z-slice {z_idx}")

        except Exception as e:
            logger.warning(f"🔍 SKELETON_TO_ROIS: Failed to extract ROIs from Z-slice {z_idx}: {e}")
            continue

    logger.info(f"🔍 SKELETON_TO_ROIS: Extracted {len(rois)} total branch ROIs from skeleton")
    return rois
