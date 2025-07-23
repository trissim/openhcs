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
from typing import Dict, Any, Tuple, Optional, Union, List
import logging

# OpenHCS imports
from openhcs.core.memory.decorators import numpy as numpy_func
from openhcs.core.pipeline.function_contracts import special_outputs

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


def materialize_axon_analysis(
    axon_analysis_data: Dict[str, Any],
    path: str,
    filemanager,
    **kwargs
) -> str:
    """
    Materialize axon analysis results to disk using filemanager.

    Creates multiple output files:
    - CSV file with detailed branch data
    - JSON file with summary metrics and metadata
    - Optional: Excel file with multiple sheets

    Args:
        axon_analysis_data: The axon analysis results dictionary
        path: Base path for output files (from special output path)
        filemanager: FileManager instance for consistent I/O
        **kwargs: Additional materialization options

    Returns:
        str: Path to the primary output file (JSON summary)
    """
    logger.info(f"🔬 SKAN_MATERIALIZE: Called with path={path}, data_keys={list(axon_analysis_data.keys()) if axon_analysis_data else 'None'}")
    import json
    from pathlib import Path
    from openhcs.constants.constants import Backend

    # Generate output file paths based on the input path
    # Use clean naming: preserve namespaced path structure, don't duplicate special output key
    base_path = path.replace('.pkl', '')
    json_path = f"{base_path}.json"
    csv_path = f"{base_path}_branches.csv"

    # Ensure output directory exists for disk backend
    output_dir = Path(json_path).parent
    filemanager.ensure_directory(str(output_dir), Backend.DISK.value)

    # 1. Save summary and metadata as JSON (primary output)
    summary_data = {
        'analysis_type': 'axon_skeleton_analysis',
        'summary': axon_analysis_data['summary'],
        'metadata': axon_analysis_data['metadata']
    }
    json_content = json.dumps(summary_data, indent=2, default=str)
    # Remove existing file if it exists using filemanager
    if filemanager.exists(json_path, Backend.DISK.value):
        filemanager.delete(json_path, Backend.DISK.value)
    filemanager.save(json_content, json_path, Backend.DISK.value)

    # 2. Save detailed branch data as CSV
    branch_df = pd.DataFrame(axon_analysis_data['branch_data'])
    csv_content = branch_df.to_csv(index=False)
    # Remove existing file if it exists using filemanager
    if filemanager.exists(csv_path, Backend.DISK.value):
        filemanager.delete(csv_path, Backend.DISK.value)
    filemanager.save(csv_content, csv_path, Backend.DISK.value)

    # 3. Optional: Create Excel file with multiple sheets (using direct file I/O for Excel)
    if kwargs.get('create_excel', True):
        excel_path = f"{base_path}_complete.xlsx"
        # Remove existing file if it exists
        if Path(excel_path).exists():
            Path(excel_path).unlink()
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Branch data sheet
            branch_df.to_excel(writer, sheet_name='Branch_Data', index=False)

            # Summary sheet
            summary_df = pd.DataFrame([axon_analysis_data['summary']])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

            # Metadata sheet
            metadata_df = pd.DataFrame([axon_analysis_data['metadata']])
            metadata_df.to_excel(writer, sheet_name='Metadata', index=False)

        logger.info(f"Created Excel file: {excel_path}")

    # 4. Log materialization
    logger.info(f"Materialized axon analysis:")
    logger.info(f"  - Summary: {json_path}")
    logger.info(f"  - Branch data: {csv_path}")

    return json_path


def materialize_skeleton_visualizations(data: List[np.ndarray], path: str, filemanager) -> str:
    """Materialize skeleton visualizations as individual TIFF files."""

    if not data:
        # Create empty summary file to indicate no visualizations were generated
        summary_path = path.replace('.pkl', '_skeleton_summary.txt')
        summary_content = "No skeleton visualizations generated (return_skeleton_visualizations=False)\n"
        from openhcs.constants.constants import Backend
        filemanager.save(summary_content, summary_path, Backend.DISK.value)
        return summary_path

    # Generate output file paths based on the input path
    base_path = path.replace('.pkl', '')

    # Save each visualization as a separate TIFF file
    for i, visualization in enumerate(data):
        viz_filename = f"{base_path}_slice_{i:03d}.tif"

        # Convert visualization to appropriate dtype for saving (uint16 to match input images)
        if visualization.dtype != np.uint16:
            # Normalize to uint16 range if needed
            if visualization.max() <= 1.0:
                viz_uint16 = (visualization * 65535).astype(np.uint16)
            else:
                viz_uint16 = visualization.astype(np.uint16)
        else:
            viz_uint16 = visualization

        # Save using filemanager
        from openhcs.constants.constants import Backend
        filemanager.save(viz_uint16, viz_filename, Backend.DISK.value)

    # Return summary path
    summary_path = f"{base_path}_skeleton_summary.txt"
    summary_content = f"Skeleton visualizations saved: {len(data)} files\n"
    summary_content += f"Base filename pattern: {base_path}_slice_XXX.tif\n"
    summary_content += f"Visualization dtype: {data[0].dtype}\n"
    summary_content += f"Visualization shape: {data[0].shape}\n"

    filemanager.save(summary_content, summary_path, Backend.DISK.value)

    return summary_path


@special_outputs(("axon_analysis", materialize_axon_analysis), ("skeleton_visualizations", materialize_skeleton_visualizations))
@numpy_func
def skan_axon_skeletonize_and_analyze(
    image_stack: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    threshold_method: ThresholdMethod = ThresholdMethod.OTSU,
    threshold_value: Optional[float] = None,
    min_object_size: int = 100,
    min_branch_length: float = 0.0,
    return_skeleton_visualizations: bool = False,
    skeleton_visualization_mode: OutputMode = OutputMode.SKELETON_OVERLAY,
    analysis_dimension: AnalysisDimension = AnalysisDimension.THREE_D
) -> Tuple[np.ndarray, Dict[str, Any], List[np.ndarray]]:
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
        return_skeleton_visualizations: Whether to generate skeleton visualizations as special output
        skeleton_visualization_mode: Type of visualization (SKELETON, SKELETON_OVERLAY, ORIGINAL, COMPOSITE)
        analysis_dimension: Analysis mode (TWO_D or THREE_D)

    Returns:
        Tuple containing:
        - Original image stack: Input image unchanged (Z, Y, X)
        - Axon analysis results: Complete analysis data structure
        - Skeleton visualizations: (Special output) List of visualization arrays if return_skeleton_visualizations=True
    """
    # Validate input
    if len(image_stack.shape) != 3:
        raise ValueError(f"Expected 3D image, got {len(image_stack.shape)}D")
    
    if threshold_method == ThresholdMethod.MANUAL and threshold_value is None:
        raise ValueError("threshold_value required when threshold_method=MANUAL")
    
    logger.info(f"Starting skan axon analysis: {image_stack.shape} image")
    logger.info(f"Parameters: threshold={threshold_method.value}, "
                f"analysis={analysis_dimension.value}, visualizations={return_skeleton_visualizations}")
    
    # Step 1: Segmentation/Thresholding
    binary_stack = _segment_axons(image_stack, threshold_method, threshold_value)
    
    # Step 2: Noise removal
    if min_object_size > 0:
        binary_stack = _remove_small_objects(binary_stack, min_object_size)
    
    # Step 3: Skeletonization
    skeleton_stack = _skeletonize_3d(binary_stack)
    
    # Step 4: Skeleton analysis
    if analysis_dimension == AnalysisDimension.THREE_D:
        branch_data = _analyze_3d_skeleton(skeleton_stack, voxel_spacing)
        analysis_type = "3D volumetric"
    elif analysis_dimension == AnalysisDimension.TWO_D:
        branch_data = _analyze_2d_slices(skeleton_stack, voxel_spacing)
        analysis_type = "2D slice-by-slice"
    else:
        raise ValueError(f"Invalid analysis_dimension: {analysis_dimension}")
    
    # Step 5: Filter results
    if min_branch_length > 0:
        branch_data = branch_data[branch_data['branch_distance'] >= min_branch_length]

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

    # Step 7: Compile comprehensive results
    results = _compile_analysis_results(
        branch_data, skeleton_stack, binary_stack, image_stack,
        voxel_spacing, analysis_type, threshold_method, min_object_size, min_branch_length
    )

    logger.info(f"Analysis complete: {len(branch_data)} branches found")

    # Always return original image, analysis results, and skeleton visualizations
    return image_stack, results, skeleton_visualizations


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


def _analyze_3d_skeleton(skeleton_stack, voxel_spacing):
    """Analyze skeleton as single 3D network."""
    try:
        from skan import Skeleton, summarize
    except ImportError:
        raise ImportError("skan library is required for skeleton analysis. "
                         "Install with: pip install skan")

    if not skeleton_stack.any():
        logger.warning("Empty skeleton - returning empty analysis")
        return pd.DataFrame()

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
        return pd.DataFrame()


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
        'branch_data': branch_data.to_dict('list') if len(branch_data) > 0 else {},
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
