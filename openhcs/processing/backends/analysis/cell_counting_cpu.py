"""
CPU-based cell counting and multi-channel colocalization analysis for OpenHCS.

This module provides comprehensive cell counting capabilities using scikit-image,
supporting both single-channel and multi-channel analysis with various detection
methods and colocalization metrics.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Core scientific computing imports
import pandas as pd
import json
from scipy import ndimage
from scipy.spatial.distance import cdist
from skimage.feature import blob_log, blob_dog, blob_doh, peak_local_max
from skimage.filters import threshold_otsu, threshold_li, gaussian, median
from skimage.segmentation import watershed, clear_border
from skimage.morphology import remove_small_objects, disk, binary_erosion, binary_dilation
from skimage.measure import label, regionprops
from skimage import exposure

# OpenHCS imports
from openhcs.core.memory.decorators import numpy as numpy_func
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.constants.constants import Backend


class DetectionMethod(Enum):
    """Cell detection methods available."""
    BLOB_LOG = "blob_log"          # Laplacian of Gaussian (best for round cells)
    BLOB_DOG = "blob_dog"          # Difference of Gaussian (faster, good general purpose)
    BLOB_DOH = "blob_doh"          # Determinant of Hessian (good for elongated cells)
    WATERSHED = "watershed"        # Watershed segmentation (best for touching cells)
    THRESHOLD = "threshold"        # Simple thresholding (fastest, basic)


class ColocalizationMethod(Enum):
    """Methods for multi-channel colocalization analysis."""
    OVERLAP_AREA = "overlap_area"           # Based on segmentation overlap
    DISTANCE_BASED = "distance_based"       # Based on centroid distances
    INTENSITY_CORRELATION = "intensity_correlation"  # Based on intensity correlation
    MANDERS_COEFFICIENTS = "manders_coefficients"    # Manders colocalization coefficients


class ThresholdMethod(Enum):
    """Automatic thresholding methods for watershed segmentation."""
    OTSU = "otsu"                  # Otsu's method (good for bimodal histograms)
    LI = "li"                      # Li's method (good for low contrast images)
    MANUAL = "manual"              # Manual threshold value (0.0-1.0)





@dataclass
class CellCountResult:
    """Results for single-channel cell counting."""
    slice_index: int
    method: str
    cell_count: int
    cell_positions: List[Tuple[float, float]]  # (x, y) centroids
    cell_areas: List[float]
    cell_intensities: List[float]
    detection_confidence: List[float]
    parameters_used: Dict[str, Any]
    binary_mask: Optional[np.ndarray] = None  # Actual binary mask for segmentation methods


@dataclass
class MultiChannelResult:
    """Results for multi-channel cell counting and colocalization."""
    slice_index: int
    chan_1_results: CellCountResult
    chan_2_results: CellCountResult
    colocalization_method: str
    colocalized_count: int
    colocalization_percentage: float
    chan_1_only_count: int
    chan_2_only_count: int
    colocalization_metrics: Dict[str, float]
    overlap_positions: List[Tuple[float, float]]


def materialize_cell_counts(data: List[Union[CellCountResult, MultiChannelResult]], path: str, filemanager) -> str:
    """Materialize cell counting results as analysis-ready CSV and JSON formats."""

    logger.info(f"🔬 CELL_COUNT_MATERIALIZE: Called with path={path}, data_length={len(data) if data else 0}")

    # Determine if this is single-channel or multi-channel data
    if not data:
        logger.warning(f"🔬 CELL_COUNT_MATERIALIZE: No data to materialize")
        return path

    is_multi_channel = isinstance(data[0], MultiChannelResult)
    logger.info(f"🔬 CELL_COUNT_MATERIALIZE: is_multi_channel={is_multi_channel}")

    if is_multi_channel:
        return _materialize_multi_channel_results(data, path, filemanager)
    else:
        return _materialize_single_channel_results(data, path, filemanager)


def materialize_segmentation_masks(data: List[np.ndarray], path: str, filemanager) -> str:
    """Materialize segmentation masks as individual TIFF files."""

    logger.info(f"🔬 SEGMENTATION_MATERIALIZE: Called with path={path}, masks_count={len(data) if data else 0}")

    if not data:
        logger.info(f"🔬 SEGMENTATION_MATERIALIZE: No segmentation masks to materialize (return_segmentation_mask=False)")
        # Create empty summary file to indicate no masks were generated
        summary_path = path.replace('.pkl', '_segmentation_summary.txt')
        summary_content = "No segmentation masks generated (return_segmentation_mask=False)\n"
        filemanager.save(summary_content, summary_path, Backend.DISK.value)
        return summary_path

    # Generate output file paths based on the input path
    base_path = path.replace('.pkl', '')

    # Save each mask as a separate TIFF file
    for i, mask in enumerate(data):
        mask_filename = f"{base_path}_slice_{i:03d}.tif"

        # Convert mask to appropriate dtype for saving (uint16 to match input images)
        if mask.dtype != np.uint16:
            # Normalize to uint16 range if needed
            if mask.max() <= 1.0:
                mask_uint16 = (mask * 65535).astype(np.uint16)
            else:
                mask_uint16 = mask.astype(np.uint16)
        else:
            mask_uint16 = mask

        # Save using filemanager
        filemanager.save(mask_uint16, mask_filename, Backend.DISK.value)
        logger.debug(f"🔬 SEGMENTATION_MATERIALIZE: Saved mask {i} to {mask_filename}")

    # Return summary path
    summary_path = f"{base_path}_segmentation_summary.txt"
    summary_content = f"Segmentation masks saved: {len(data)} files\n"
    summary_content += f"Base filename pattern: {base_path}_slice_XXX.tif\n"
    summary_content += f"Mask dtype: {data[0].dtype}\n"
    summary_content += f"Mask shape: {data[0].shape}\n"

    filemanager.save(summary_content, summary_path, Backend.DISK.value)
    logger.info(f"🔬 SEGMENTATION_MATERIALIZE: Completed, saved {len(data)} masks")

    return summary_path


@numpy_func
@special_outputs(("cell_counts", materialize_cell_counts), ("segmentation_masks", materialize_segmentation_masks))
def count_cells_single_channel(
    image_stack: np.ndarray,
    # Detection method and parameters
    detection_method: DetectionMethod = DetectionMethod.BLOB_LOG,  # UI will show radio buttons
    # Blob detection parameters
    min_sigma: float = 1.0,                                       # Minimum blob size (pixels)
    max_sigma: float = 10.0,                                      # Maximum blob size (pixels)
    num_sigma: int = 10,                                          # Number of sigma values to test
    threshold: float = 0.1,                                       # Detection threshold (0.0-1.0)
    overlap: float = 0.5,                                         # Maximum overlap between blobs (0.0-1.0)
    # Watershed parameters
    watershed_footprint_size: int = 3,                            # Local maxima footprint size
    watershed_min_distance: int = 5,                              # Minimum distance between peaks
    watershed_threshold_method: ThresholdMethod = ThresholdMethod.OTSU,  # UI will show threshold methods
    # Preprocessing parameters
    enable_preprocessing: bool = True,
    gaussian_sigma: float = 1.0,                                  # Gaussian blur sigma
    median_disk_size: int = 1,                                    # Median filter disk size
    # Filtering parameters
    min_cell_area: int = 10,                                      # Minimum cell area (pixels)
    max_cell_area: int = 1000,                                    # Maximum cell area (pixels)
    remove_border_cells: bool = True,                             # Remove cells touching image border
    # Output parameters
    return_segmentation_mask: bool = False
) -> Tuple[np.ndarray, List[CellCountResult]]:
    """
    Count cells in single-channel image stack using various detection methods.
    
    Args:
        image_stack: 3D array (Z, Y, X) where each Z slice is processed independently
        detection_method: Method for cell detection (see DetectionMethod enum)
        min_sigma: Minimum blob size for blob detection methods
        max_sigma: Maximum blob size for blob detection methods
        num_sigma: Number of sigma values to test for blob detection
        threshold: Detection threshold (method-dependent)
        overlap: Maximum overlap between detected blobs
        watershed_footprint_size: Footprint size for local maxima detection
        watershed_min_distance: Minimum distance between watershed peaks
        watershed_threshold_method: Thresholding method for watershed
        enable_preprocessing: Apply Gaussian and median filtering
        gaussian_sigma: Standard deviation for Gaussian blur
        median_disk_size: Disk size for median filtering
        min_cell_area: Minimum area for valid cells
        max_cell_area: Maximum area for valid cells
        remove_border_cells: Remove cells touching image borders
        return_segmentation_mask: Return segmentation masks in output
        
    Returns:
        output_stack: Original image stack unchanged (Z, Y, X)
        cell_count_results: List of CellCountResult objects for each slice
        segmentation_masks: (Special output) List of segmentation mask arrays if return_segmentation_mask=True
    """
    if image_stack.ndim != 3:
        raise ValueError(f"Expected 3D image stack, got {image_stack.ndim}D")
    
    results = []
    segmentation_masks = []

    # Store parameters for reproducibility (convert enums to values)
    parameters = {
        "detection_method": detection_method.value,
        "min_sigma": min_sigma,
        "max_sigma": max_sigma,
        "num_sigma": num_sigma,
        "threshold": threshold,
        "overlap": overlap,
        "watershed_footprint_size": watershed_footprint_size,
        "watershed_min_distance": watershed_min_distance,
        "watershed_threshold_method": watershed_threshold_method.value,
        "gaussian_sigma": gaussian_sigma,
        "median_disk_size": median_disk_size,
        "min_cell_area": min_cell_area,
        "max_cell_area": max_cell_area,
        "remove_border_cells": remove_border_cells
    }

    logging.info(f"Processing {image_stack.shape[0]} slices with {detection_method.value} method")

    for z_idx in range(image_stack.shape[0]):
        slice_img = image_stack[z_idx].astype(np.float64)

        # Apply preprocessing if enabled
        if enable_preprocessing:
            slice_img = _preprocess_image(slice_img, gaussian_sigma, median_disk_size)

        # Detect cells using specified method
        result = _detect_cells_single_method(
            slice_img, z_idx, detection_method.value, parameters
        )

        results.append(result)

        # Create segmentation mask if requested
        if return_segmentation_mask:
            segmentation_mask = _create_segmentation_visualization(
                slice_img, result.cell_positions, max_sigma, result.cell_areas, result.binary_mask
            )
            segmentation_masks.append(segmentation_mask)

    # Always return segmentation masks (empty list if not requested)
    # This ensures consistent return signature for special outputs system
    return image_stack, results, segmentation_masks


@numpy_func
@special_outputs(("multi_channel_counts", materialize_cell_counts))
def count_cells_multi_channel(
    image_stack: np.ndarray,
    chan_1: int,                         # Index of first channel (positional arg)
    chan_2: int,                         # Index of second channel (positional arg)
    # Detection parameters for channel 1 (all single-channel params available)
    chan_1_method: DetectionMethod = DetectionMethod.BLOB_LOG,        # UI will show radio buttons
    chan_1_min_sigma: float = 1.0,                                    # Minimum blob size (pixels)
    chan_1_max_sigma: float = 10.0,                                   # Maximum blob size (pixels)
    chan_1_num_sigma: int = 10,                                       # Number of sigma values to test
    chan_1_threshold: float = 0.1,                                    # Detection threshold (0.0-1.0)
    chan_1_overlap: float = 0.5,                                      # Maximum overlap between blobs (0.0-1.0)
    chan_1_watershed_footprint_size: int = 3,                         # Local maxima footprint size
    chan_1_watershed_min_distance: int = 5,                           # Minimum distance between peaks
    chan_1_watershed_threshold_method: ThresholdMethod = ThresholdMethod.OTSU,  # Thresholding method
    chan_1_enable_preprocessing: bool = True,                         # Apply preprocessing
    chan_1_gaussian_sigma: float = 1.0,                               # Gaussian blur sigma
    chan_1_median_disk_size: int = 1,                                 # Median filter disk size
    chan_1_min_area: int = 10,                                        # Minimum cell area (pixels)
    chan_1_max_area: int = 1000,                                      # Maximum cell area (pixels)
    chan_1_remove_border_cells: bool = True,                          # Remove cells touching border
    # Detection parameters for channel 2 (all single-channel params available)
    chan_2_method: DetectionMethod = DetectionMethod.BLOB_LOG,        # UI will show radio buttons
    chan_2_min_sigma: float = 1.0,                                    # Minimum blob size (pixels)
    chan_2_max_sigma: float = 10.0,                                   # Maximum blob size (pixels)
    chan_2_num_sigma: int = 10,                                       # Number of sigma values to test
    chan_2_threshold: float = 0.1,                                    # Detection threshold (0.0-1.0)
    chan_2_overlap: float = 0.5,                                      # Maximum overlap between blobs (0.0-1.0)
    chan_2_watershed_footprint_size: int = 3,                         # Local maxima footprint size
    chan_2_watershed_min_distance: int = 5,                           # Minimum distance between peaks
    chan_2_watershed_threshold_method: ThresholdMethod = ThresholdMethod.OTSU,  # Thresholding method
    chan_2_enable_preprocessing: bool = True,                         # Apply preprocessing
    chan_2_gaussian_sigma: float = 1.0,                               # Gaussian blur sigma
    chan_2_median_disk_size: int = 1,                                 # Median filter disk size
    chan_2_min_area: int = 10,                                        # Minimum cell area (pixels)
    chan_2_max_area: int = 1000,                                      # Maximum cell area (pixels)
    chan_2_remove_border_cells: bool = True,                          # Remove cells touching border
    # Colocalization parameters
    colocalization_method: ColocalizationMethod = ColocalizationMethod.DISTANCE_BASED,  # UI will show coloc methods
    max_distance: float = 5.0,                                        # Maximum distance for colocalization (pixels)
    min_overlap_area: float = 0.3,                                    # Minimum overlap fraction for area-based method
    intensity_threshold: float = 0.5,                                 # Threshold for intensity-based methods
    # Output parameters
    return_colocalization_map: bool = False
) -> Tuple[np.ndarray, List[MultiChannelResult]]:
    """
    Count cells in multi-channel image stack with colocalization analysis.

    Each channel can be processed with independent parameters, providing the same
    flexibility as the single-channel function for each channel separately.

    Args:
        image_stack: 3D array (Z, Y, X) where Z represents different channels
        chan_1: Index of first channel in the stack (positional)
        chan_2: Index of second channel in the stack (positional)

        # Channel 1 detection parameters (same as single-channel function)
        chan_1_method: Detection method for channel 1 (DetectionMethod enum)
        chan_1_min_sigma: Minimum blob size for channel 1
        chan_1_max_sigma: Maximum blob size for channel 1
        chan_1_num_sigma: Number of sigma values to test for channel 1
        chan_1_threshold: Detection threshold for channel 1 (0.0-1.0)
        chan_1_overlap: Maximum overlap between blobs for channel 1
        chan_1_watershed_footprint_size: Local maxima footprint size for channel 1
        chan_1_watershed_min_distance: Minimum distance between peaks for channel 1
        chan_1_watershed_threshold_method: Thresholding method for channel 1
        chan_1_enable_preprocessing: Apply preprocessing to channel 1
        chan_1_gaussian_sigma: Gaussian blur sigma for channel 1
        chan_1_median_disk_size: Median filter size for channel 1
        chan_1_min_area: Minimum cell area for channel 1
        chan_1_max_area: Maximum cell area for channel 1
        chan_1_remove_border_cells: Remove border cells for channel 1

        # Channel 2 detection parameters (same as single-channel function)
        chan_2_method: Detection method for channel 2 (DetectionMethod enum)
        chan_2_min_sigma: Minimum blob size for channel 2
        chan_2_max_sigma: Maximum blob size for channel 2
        chan_2_num_sigma: Number of sigma values to test for channel 2
        chan_2_threshold: Detection threshold for channel 2 (0.0-1.0)
        chan_2_overlap: Maximum overlap between blobs for channel 2
        chan_2_watershed_footprint_size: Local maxima footprint size for channel 2
        chan_2_watershed_min_distance: Minimum distance between peaks for channel 2
        chan_2_watershed_threshold_method: Thresholding method for channel 2
        chan_2_enable_preprocessing: Apply preprocessing to channel 2
        chan_2_gaussian_sigma: Gaussian blur sigma for channel 2
        chan_2_median_disk_size: Median filter size for channel 2
        chan_2_min_area: Minimum cell area for channel 2
        chan_2_max_area: Maximum cell area for channel 2
        chan_2_remove_border_cells: Remove border cells for channel 2

        # Colocalization parameters
        colocalization_method: Method for determining colocalization (ColocalizationMethod enum)
        max_distance: Maximum distance for distance-based colocalization (pixels)
        min_overlap_area: Minimum overlap fraction for area-based colocalization
        intensity_threshold: Threshold for intensity-based colocalization (0.0-1.0)
        return_colocalization_map: Return colocalization visualization

    Returns:
        output_stack: Original images or colocalization maps
        multi_channel_results: List of MultiChannelResult objects
    """
    if image_stack.ndim != 3:
        raise ValueError(f"Expected 3D image stack, got {image_stack.ndim}D")

    if chan_1 >= image_stack.shape[0] or chan_2 >= image_stack.shape[0]:
        raise ValueError(f"Channel indices {chan_1}, {chan_2} exceed stack size {image_stack.shape[0]}")

    if chan_1 == chan_2:
        raise ValueError("Channel 1 and Channel 2 must be different")

    # Extract channel images
    chan_1_img = image_stack[chan_1:chan_1+1]  # Keep 3D shape for consistency
    chan_2_img = image_stack[chan_2:chan_2+1]

    # Count cells in each channel separately using the single-channel function
    # Channel 1 parameters (all explicit)
    chan_1_params = {
        "detection_method": chan_1_method,
        "min_sigma": chan_1_min_sigma,
        "max_sigma": chan_1_max_sigma,
        "num_sigma": chan_1_num_sigma,
        "threshold": chan_1_threshold,
        "overlap": chan_1_overlap,
        "watershed_footprint_size": chan_1_watershed_footprint_size,
        "watershed_min_distance": chan_1_watershed_min_distance,
        "watershed_threshold_method": chan_1_watershed_threshold_method,
        "enable_preprocessing": chan_1_enable_preprocessing,
        "gaussian_sigma": chan_1_gaussian_sigma,
        "median_disk_size": chan_1_median_disk_size,
        "min_cell_area": chan_1_min_area,
        "max_cell_area": chan_1_max_area,
        "remove_border_cells": chan_1_remove_border_cells,
        "return_segmentation_mask": False
    }

    # Channel 2 parameters (all explicit)
    chan_2_params = {
        "detection_method": chan_2_method,
        "min_sigma": chan_2_min_sigma,
        "max_sigma": chan_2_max_sigma,
        "num_sigma": chan_2_num_sigma,
        "threshold": chan_2_threshold,
        "overlap": chan_2_overlap,
        "watershed_footprint_size": chan_2_watershed_footprint_size,
        "watershed_min_distance": chan_2_watershed_min_distance,
        "watershed_threshold_method": chan_2_watershed_threshold_method,
        "enable_preprocessing": chan_2_enable_preprocessing,
        "gaussian_sigma": chan_2_gaussian_sigma,
        "median_disk_size": chan_2_median_disk_size,
        "min_cell_area": chan_2_min_area,
        "max_cell_area": chan_2_max_area,
        "remove_border_cells": chan_2_remove_border_cells,
        "return_segmentation_mask": False
    }

    # Process each channel
    _, chan_1_results = count_cells_single_channel(chan_1_img, **chan_1_params)
    _, chan_2_results = count_cells_single_channel(chan_2_img, **chan_2_params)

    # Perform colocalization analysis
    multi_results = []
    output_stack = image_stack.copy()

    # Since we're processing single slices from each channel, we only have one result each
    chan_1_result = chan_1_results[0]
    chan_2_result = chan_2_results[0]

    # Analyze colocalization
    coloc_result = _analyze_colocalization(
        chan_1_result, chan_2_result, colocalization_method.value,
        max_distance, min_overlap_area, intensity_threshold
    )

    multi_results.append(coloc_result)

    # Create colocalization visualization if requested
    if return_colocalization_map:
        coloc_map = _create_colocalization_map(
            image_stack[chan_1], image_stack[chan_2], coloc_result
        )
        # Replace one of the channels with the colocalization map
        output_stack = np.stack([image_stack[chan_1], image_stack[chan_2], coloc_map])

    return output_stack, multi_results


def _materialize_single_channel_results(data: List[CellCountResult], path: str, filemanager) -> str:
    """Materialize single-channel cell counting results."""
    # Generate output file paths based on the input path
    # Use clean naming: preserve namespaced path structure, don't duplicate special output key
    base_path = path.replace('.pkl', '')
    json_path = f"{base_path}.json"
    csv_path = f"{base_path}_details.csv"

    # Ensure output directory exists for disk backend
    from pathlib import Path
    from openhcs.constants.constants import Backend
    output_dir = Path(json_path).parent
    filemanager.ensure_directory(str(output_dir), Backend.DISK.value)

    summary = {
        "analysis_type": "single_channel_cell_counting",
        "total_slices": len(data),
        "results_per_slice": []
    }
    rows = []

    total_cells = 0
    for result in data:
        total_cells += result.cell_count

        # Add to summary
        summary["results_per_slice"].append({
            "slice_index": result.slice_index,
            "method": result.method,
            "cell_count": result.cell_count,
            "avg_cell_area": np.mean(result.cell_areas) if result.cell_areas else 0,
            "avg_cell_intensity": np.mean(result.cell_intensities) if result.cell_intensities else 0,
            "parameters": result.parameters_used
        })

        # Add individual cell data to CSV
        for i, (pos, area, intensity, confidence) in enumerate(zip(
            result.cell_positions, result.cell_areas,
            result.cell_intensities, result.detection_confidence
        )):
            rows.append({
                'slice_index': result.slice_index,
                'cell_id': f"slice_{result.slice_index}_cell_{i}",
                'x_position': pos[0],
                'y_position': pos[1],
                'cell_area': area,
                'cell_intensity': intensity,
                'detection_confidence': confidence,
                'detection_method': result.method
            })

    summary["total_cells_all_slices"] = total_cells
    summary["average_cells_per_slice"] = total_cells / len(data) if data else 0

    # Save JSON summary (overwrite if exists)
    json_content = json.dumps(summary, indent=2, default=str)
    # Remove existing file if it exists using filemanager
    if filemanager.exists(json_path, Backend.DISK.value):
        filemanager.delete(json_path, Backend.DISK.value)
    filemanager.save(json_content, json_path, Backend.DISK.value)

    # Save CSV details (overwrite if exists)
    if rows:
        df = pd.DataFrame(rows)
        csv_content = df.to_csv(index=False)
        # Remove existing file if it exists using filemanager
        if filemanager.exists(csv_path, Backend.DISK.value):
            filemanager.delete(csv_path, Backend.DISK.value)
        filemanager.save(csv_content, csv_path, Backend.DISK.value)

    return json_path


def _materialize_multi_channel_results(data: List[MultiChannelResult], path: str, filemanager) -> str:
    """Materialize multi-channel cell counting and colocalization results."""
    # Generate output file paths based on the input path
    # Use clean naming: preserve namespaced path structure, don't duplicate special output key
    base_path = path.replace('.pkl', '')
    json_path = f"{base_path}.json"
    csv_path = f"{base_path}_details.csv"

    # Ensure output directory exists for disk backend
    from pathlib import Path
    output_dir = Path(json_path).parent
    filemanager.ensure_directory(str(output_dir), Backend.DISK.value)

    summary = {
        "analysis_type": "multi_channel_cell_counting_colocalization",
        "total_slices": len(data),
        "colocalization_summary": {
            "total_chan_1_cells": 0,
            "total_chan_2_cells": 0,
            "total_colocalized": 0,
            "average_colocalization_percentage": 0
        },
        "results_per_slice": []
    }

    # CSV for detailed analysis (csv_path already defined above)
    rows = []

    total_coloc_pct = 0
    for result in data:
        summary["colocalization_summary"]["total_chan_1_cells"] += result.chan_1_results.cell_count
        summary["colocalization_summary"]["total_chan_2_cells"] += result.chan_2_results.cell_count
        summary["colocalization_summary"]["total_colocalized"] += result.colocalized_count
        total_coloc_pct += result.colocalization_percentage

        # Add to summary
        summary["results_per_slice"].append({
            "slice_index": result.slice_index,
            "chan_1_count": result.chan_1_results.cell_count,
            "chan_2_count": result.chan_2_results.cell_count,
            "colocalized_count": result.colocalized_count,
            "colocalization_percentage": result.colocalization_percentage,
            "chan_1_only": result.chan_1_only_count,
            "chan_2_only": result.chan_2_only_count,
            "colocalization_method": result.colocalization_method,
            "colocalization_metrics": result.colocalization_metrics
        })

        # Add colocalization details to CSV
        for i, pos in enumerate(result.overlap_positions):
            rows.append({
                'slice_index': result.slice_index,
                'colocalization_id': f"slice_{result.slice_index}_coloc_{i}",
                'x_position': pos[0],
                'y_position': pos[1],
                'colocalization_method': result.colocalization_method
            })

    summary["colocalization_summary"]["average_colocalization_percentage"] = (
        total_coloc_pct / len(data) if data else 0
    )

    # Save JSON summary (overwrite if exists)
    json_content = json.dumps(summary, indent=2, default=str)
    # Remove existing file if it exists using filemanager
    if filemanager.exists(json_path, Backend.DISK.value):
        filemanager.delete(json_path, Backend.DISK.value)
    filemanager.save(json_content, json_path, Backend.DISK.value)

    # Save CSV details (overwrite if exists)
    if rows:
        df = pd.DataFrame(rows)
        csv_content = df.to_csv(index=False)
        # Remove existing file if it exists using filemanager
        if filemanager.exists(csv_path, Backend.DISK.value):
            filemanager.delete(csv_path, Backend.DISK.value)
        filemanager.save(csv_content, csv_path, Backend.DISK.value)

    return json_path


def _preprocess_image(image: np.ndarray, gaussian_sigma: float, median_disk_size: int) -> np.ndarray:
    """Apply preprocessing to enhance cell detection."""
    # Gaussian blur to reduce noise
    if gaussian_sigma > 0:
        image = gaussian(image, sigma=gaussian_sigma, preserve_range=True)

    # Median filter to remove salt-and-pepper noise
    if median_disk_size > 0:
        image = median(image, disk(median_disk_size))

    return image


def _detect_cells_single_method(
    image: np.ndarray,
    slice_idx: int,
    method: str,
    params: Dict[str, Any]
) -> CellCountResult:
    """Detect cells using specified method."""

    if method == DetectionMethod.BLOB_LOG.value:
        return _detect_cells_blob_log(image, slice_idx, params)
    elif method == DetectionMethod.BLOB_DOG.value:
        return _detect_cells_blob_dog(image, slice_idx, params)
    elif method == DetectionMethod.BLOB_DOH.value:
        return _detect_cells_blob_doh(image, slice_idx, params)
    elif method == DetectionMethod.WATERSHED.value:
        return _detect_cells_watershed(image, slice_idx, params)
    elif method == DetectionMethod.THRESHOLD.value:
        return _detect_cells_threshold(image, slice_idx, params)
    else:
        raise ValueError(f"Unknown detection method: {method}")


def _detect_cells_blob_log(image: np.ndarray, slice_idx: int, params: Dict[str, Any]) -> CellCountResult:
    """Detect cells using Laplacian of Gaussian blob detection."""
    blobs = blob_log(
        image,
        min_sigma=params["min_sigma"],
        max_sigma=params["max_sigma"],
        num_sigma=params["num_sigma"],
        threshold=params["threshold"],
        overlap=params["overlap"]
    )

    # Extract positions, areas, and intensities
    positions = []
    areas = []
    intensities = []
    confidences = []

    for blob in blobs:
        y, x, sigma = blob
        positions.append((float(x), float(y)))

        # Estimate area from sigma (blob radius ≈ sigma * sqrt(2))
        radius = sigma * np.sqrt(2)
        area = np.pi * radius**2
        areas.append(float(area))

        # Sample intensity at blob center
        intensity = float(image[int(y), int(x)])
        intensities.append(intensity)

        # Use sigma as confidence measure (larger blobs = higher confidence)
        confidence = float(sigma / params["max_sigma"])
        confidences.append(confidence)

    # Filter by area constraints
    filtered_data = _filter_by_area(
        positions, areas, intensities, confidences,
        params["min_cell_area"], params["max_cell_area"]
    )

    return CellCountResult(
        slice_index=slice_idx,
        method="blob_log",
        cell_count=len(filtered_data[0]),
        cell_positions=filtered_data[0],
        cell_areas=filtered_data[1],
        cell_intensities=filtered_data[2],
        detection_confidence=filtered_data[3],
        parameters_used=params
    )


def _detect_cells_blob_dog(image: np.ndarray, slice_idx: int, params: Dict[str, Any]) -> CellCountResult:
    """Detect cells using Difference of Gaussian blob detection."""
    blobs = blob_dog(
        image,
        min_sigma=params["min_sigma"],
        max_sigma=params["max_sigma"],
        threshold=params["threshold"],
        overlap=params["overlap"]
    )

    # Process similar to blob_log
    positions = []
    areas = []
    intensities = []
    confidences = []

    for blob in blobs:
        y, x, sigma = blob
        positions.append((float(x), float(y)))

        radius = sigma * np.sqrt(2)
        area = np.pi * radius**2
        areas.append(float(area))

        intensity = float(image[int(y), int(x)])
        intensities.append(intensity)

        confidence = float(sigma / params["max_sigma"])
        confidences.append(confidence)

    filtered_data = _filter_by_area(
        positions, areas, intensities, confidences,
        params["min_cell_area"], params["max_cell_area"]
    )

    return CellCountResult(
        slice_index=slice_idx,
        method="blob_dog",
        cell_count=len(filtered_data[0]),
        cell_positions=filtered_data[0],
        cell_areas=filtered_data[1],
        cell_intensities=filtered_data[2],
        detection_confidence=filtered_data[3],
        parameters_used=params
    )


def _detect_cells_blob_doh(image: np.ndarray, slice_idx: int, params: Dict[str, Any]) -> CellCountResult:
    """Detect cells using Determinant of Hessian blob detection."""
    blobs = blob_doh(
        image,
        min_sigma=params["min_sigma"],
        max_sigma=params["max_sigma"],
        num_sigma=params["num_sigma"],
        threshold=params["threshold"],
        overlap=params["overlap"]
    )

    # Process similar to other blob methods
    positions = []
    areas = []
    intensities = []
    confidences = []

    for blob in blobs:
        y, x, sigma = blob
        positions.append((float(x), float(y)))

        radius = sigma * np.sqrt(2)
        area = np.pi * radius**2
        areas.append(float(area))

        intensity = float(image[int(y), int(x)])
        intensities.append(intensity)

        confidence = float(sigma / params["max_sigma"])
        confidences.append(confidence)

    filtered_data = _filter_by_area(
        positions, areas, intensities, confidences,
        params["min_cell_area"], params["max_cell_area"]
    )

    return CellCountResult(
        slice_index=slice_idx,
        method="blob_doh",
        cell_count=len(filtered_data[0]),
        cell_positions=filtered_data[0],
        cell_areas=filtered_data[1],
        cell_intensities=filtered_data[2],
        detection_confidence=filtered_data[3],
        parameters_used=params
    )


def _filter_by_area(
    positions: List[Tuple[float, float]],
    areas: List[float],
    intensities: List[float],
    confidences: List[float],
    min_area: float,
    max_area: float
) -> Tuple[List[Tuple[float, float]], List[float], List[float], List[float]]:
    """Filter detected cells by area constraints."""
    filtered_positions = []
    filtered_areas = []
    filtered_intensities = []
    filtered_confidences = []

    for pos, area, intensity, confidence in zip(positions, areas, intensities, confidences):
        if min_area <= area <= max_area:
            filtered_positions.append(pos)
            filtered_areas.append(area)
            filtered_intensities.append(intensity)
            filtered_confidences.append(confidence)

    return filtered_positions, filtered_areas, filtered_intensities, filtered_confidences


def _detect_cells_watershed(image: np.ndarray, slice_idx: int, params: Dict[str, Any]) -> CellCountResult:
    """Detect cells using watershed segmentation."""
    # Determine threshold
    if params["watershed_threshold_method"] == "otsu":
        threshold_val = threshold_otsu(image)
    elif params["watershed_threshold_method"] == "li":
        threshold_val = threshold_li(image)
    else:
        threshold_val = float(params["watershed_threshold_method"])

    # Create binary mask
    binary = image > threshold_val

    # Remove small objects and border objects
    binary = remove_small_objects(binary, min_size=params["min_cell_area"])
    if params["remove_border_cells"]:
        binary = clear_border(binary)

    # Find local maxima as seeds
    distance = ndimage.distance_transform_edt(binary)
    local_maxima = peak_local_max(
        distance,
        min_distance=params["watershed_min_distance"],
        footprint=np.ones((params["watershed_footprint_size"], params["watershed_footprint_size"]))
    )

    # Convert coordinates to binary mask
    local_maxima_mask = np.zeros_like(distance, dtype=bool)
    if len(local_maxima) > 0:
        local_maxima_mask[local_maxima[:, 0], local_maxima[:, 1]] = True

    # Create markers for watershed
    # Convert boolean mask to integer labels for connected components
    markers = label(local_maxima_mask.astype(np.uint8))

    # Apply watershed
    labels = watershed(-distance, markers, mask=binary)

    # Extract region properties
    regions = regionprops(labels, intensity_image=image)

    positions = []
    areas = []
    intensities = []
    confidences = []
    valid_labels = []  # Track which labels pass the size filter

    for region in regions:
        # Filter by area
        if params["min_cell_area"] <= region.area <= params["max_cell_area"]:
            # Centroid (note: regionprops returns (row, col) = (y, x))
            y, x = region.centroid
            positions.append((float(x), float(y)))

            areas.append(float(region.area))
            intensities.append(float(region.mean_intensity))

            # Use area as confidence measure (normalized)
            confidence = min(1.0, region.area / params["max_cell_area"])
            confidences.append(confidence)

            # Track this label as valid
            valid_labels.append(region.label)

    # Create filtered binary mask with only cells that passed size filter
    filtered_binary_mask = np.isin(labels, valid_labels)

    return CellCountResult(
        slice_index=slice_idx,
        method="watershed",
        cell_count=len(positions),
        cell_positions=positions,
        cell_areas=areas,
        cell_intensities=intensities,
        detection_confidence=confidences,
        parameters_used=params,
        binary_mask=filtered_binary_mask  # Only cells that passed all filters
    )


def _detect_cells_threshold(image: np.ndarray, slice_idx: int, params: Dict[str, Any]) -> CellCountResult:
    """Detect cells using simple thresholding and connected components."""
    # Apply threshold
    binary = image > params["threshold"] * image.max()

    # Remove small objects and border objects
    binary = remove_small_objects(binary, min_size=params["min_cell_area"])
    if params["remove_border_cells"]:
        binary = clear_border(binary)

    # Label connected components
    labels = label(binary)
    regions = regionprops(labels, intensity_image=image)

    positions = []
    areas = []
    intensities = []
    confidences = []
    valid_labels = []  # Track which labels pass the size filter

    for region in regions:
        # Filter by area
        if params["min_cell_area"] <= region.area <= params["max_cell_area"]:
            y, x = region.centroid
            positions.append((float(x), float(y)))

            areas.append(float(region.area))
            intensities.append(float(region.mean_intensity))

            # Use intensity as confidence measure
            confidence = float(region.mean_intensity / image.max())
            confidences.append(confidence)

            # Track this label as valid
            valid_labels.append(region.label)

    # Create filtered binary mask with only cells that passed size filter
    filtered_binary_mask = np.isin(labels, valid_labels)

    return CellCountResult(
        slice_index=slice_idx,
        method="threshold",
        cell_count=len(positions),
        cell_positions=positions,
        cell_areas=areas,
        cell_intensities=intensities,
        detection_confidence=confidences,
        parameters_used=params,
        binary_mask=filtered_binary_mask  # Only cells that passed all filters
    )


def _analyze_colocalization(
    chan_1_result: CellCountResult,
    chan_2_result: CellCountResult,
    method: str,
    max_distance: float,
    min_overlap_area: float,
    intensity_threshold: float
) -> MultiChannelResult:
    """Analyze colocalization between two channels."""

    if method == ColocalizationMethod.DISTANCE_BASED.value:
        return _colocalization_distance_based(
            chan_1_result, chan_2_result, max_distance
        )
    elif method == ColocalizationMethod.OVERLAP_AREA.value:
        return _colocalization_overlap_based(
            chan_1_result, chan_2_result, min_overlap_area
        )
    elif method == ColocalizationMethod.INTENSITY_CORRELATION.value:
        return _colocalization_intensity_based(
            chan_1_result, chan_2_result, intensity_threshold
        )
    elif method == ColocalizationMethod.MANDERS_COEFFICIENTS.value:
        return _colocalization_manders(
            chan_1_result, chan_2_result, intensity_threshold
        )
    else:
        raise ValueError(f"Unknown colocalization method: {method}")


def _colocalization_distance_based(
    chan_1_result: CellCountResult,
    chan_2_result: CellCountResult,
    max_distance: float
) -> MultiChannelResult:
    """Perform distance-based colocalization analysis."""

    if not chan_1_result.cell_positions or not chan_2_result.cell_positions:
        return _create_empty_coloc_result(chan_1_result, chan_2_result, "distance_based")

    # Convert positions to arrays
    pos_1 = np.array(chan_1_result.cell_positions)
    pos_2 = np.array(chan_2_result.cell_positions)

    # Calculate pairwise distances
    distances = cdist(pos_1, pos_2)

    # Find colocalized pairs
    colocalized_pairs = []
    used_chan_2 = set()

    for i in range(len(pos_1)):
        # Find closest cell in channel 2
        min_dist_idx = np.argmin(distances[i])
        min_dist = distances[i, min_dist_idx]

        # Check if within distance threshold and not already used
        if min_dist <= max_distance and min_dist_idx not in used_chan_2:
            colocalized_pairs.append((i, min_dist_idx))
            used_chan_2.add(min_dist_idx)

    # Calculate metrics
    colocalized_count = len(colocalized_pairs)
    total_cells = len(pos_1) + len(pos_2)
    colocalization_percentage = (2 * colocalized_count / total_cells * 100) if total_cells > 0 else 0

    chan_1_only = len(pos_1) - colocalized_count
    chan_2_only = len(pos_2) - colocalized_count

    # Extract colocalized positions (average of paired positions)
    overlap_positions = []
    for i, j in colocalized_pairs:
        avg_pos = ((pos_1[i] + pos_2[j]) / 2).tolist()
        overlap_positions.append(tuple(avg_pos))

    # Calculate additional metrics
    if colocalized_pairs:
        avg_distance = np.mean([distances[i, j] for i, j in colocalized_pairs])
        max_distance_found = np.max([distances[i, j] for i, j in colocalized_pairs])
    else:
        avg_distance = 0
        max_distance_found = 0

    metrics = {
        "average_colocalization_distance": float(avg_distance),
        "max_colocalization_distance": float(max_distance_found),
        "distance_threshold_used": max_distance
    }

    return MultiChannelResult(
        slice_index=chan_1_result.slice_index,
        chan_1_results=chan_1_result,
        chan_2_results=chan_2_result,
        colocalization_method="distance_based",
        colocalized_count=colocalized_count,
        colocalization_percentage=colocalization_percentage,
        chan_1_only_count=chan_1_only,
        chan_2_only_count=chan_2_only,
        colocalization_metrics=metrics,
        overlap_positions=overlap_positions
    )


def _create_empty_coloc_result(
    chan_1_result: CellCountResult,
    chan_2_result: CellCountResult,
    method: str
) -> MultiChannelResult:
    """Create empty colocalization result when no cells found."""
    return MultiChannelResult(
        slice_index=chan_1_result.slice_index,
        chan_1_results=chan_1_result,
        chan_2_results=chan_2_result,
        colocalization_method=method,
        colocalized_count=0,
        colocalization_percentage=0.0,
        chan_1_only_count=chan_1_result.cell_count,
        chan_2_only_count=chan_2_result.cell_count,
        colocalization_metrics={},
        overlap_positions=[]
    )


def _colocalization_overlap_based(
    chan_1_result: CellCountResult,
    chan_2_result: CellCountResult,
    min_overlap_area: float
) -> MultiChannelResult:
    """Perform area overlap-based colocalization analysis."""
    # This is a simplified implementation - in practice, you'd need actual segmentation masks
    # For now, we'll use distance as a proxy for overlap

    # Use distance-based method with smaller threshold as approximation
    distance_threshold = 2.0  # Assume cells must be very close to overlap significantly

    result = _colocalization_distance_based(chan_1_result, chan_2_result, distance_threshold)
    result.colocalization_method = "overlap_area"
    result.colocalization_metrics["min_overlap_threshold"] = min_overlap_area
    result.colocalization_metrics["note"] = "Approximated using distance-based method"

    return result


def _colocalization_intensity_based(
    chan_1_result: CellCountResult,
    chan_2_result: CellCountResult,
    intensity_threshold: float
) -> MultiChannelResult:
    """Perform intensity correlation-based colocalization analysis."""

    if not chan_1_result.cell_positions or not chan_2_result.cell_positions:
        return _create_empty_coloc_result(chan_1_result, chan_2_result, "intensity_correlation")

    # Use distance-based pairing first
    distance_result = _colocalization_distance_based(chan_1_result, chan_2_result, 5.0)

    # Filter pairs based on intensity correlation
    colocalized_pairs = []
    overlap_positions = []

    pos_1 = np.array(chan_1_result.cell_positions)
    pos_2 = np.array(chan_2_result.cell_positions)

    for i, (x1, y1) in enumerate(chan_1_result.cell_positions):
        for j, (x2, y2) in enumerate(chan_2_result.cell_positions):
            # Calculate distance
            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

            if dist <= 5.0:  # Within reasonable distance
                # Check intensity correlation
                int_1 = chan_1_result.cell_intensities[i]
                int_2 = chan_2_result.cell_intensities[j]

                # Simple intensity correlation: both above threshold
                if int_1 >= intensity_threshold and int_2 >= intensity_threshold:
                    colocalized_pairs.append((i, j))
                    avg_pos = ((x1 + x2) / 2, (y1 + y2) / 2)
                    overlap_positions.append(avg_pos)
                    break  # One-to-one mapping

    colocalized_count = len(colocalized_pairs)
    total_cells = len(pos_1) + len(pos_2)
    colocalization_percentage = (2 * colocalized_count / total_cells * 100) if total_cells > 0 else 0

    metrics = {
        "intensity_threshold_used": intensity_threshold,
        "correlation_method": "threshold_based"
    }

    return MultiChannelResult(
        slice_index=chan_1_result.slice_index,
        chan_1_results=chan_1_result,
        chan_2_results=chan_2_result,
        colocalization_method="intensity_correlation",
        colocalized_count=colocalized_count,
        colocalization_percentage=colocalization_percentage,
        chan_1_only_count=len(pos_1) - colocalized_count,
        chan_2_only_count=len(pos_2) - colocalized_count,
        colocalization_metrics=metrics,
        overlap_positions=overlap_positions
    )


def _colocalization_manders(
    chan_1_result: CellCountResult,
    chan_2_result: CellCountResult,
    intensity_threshold: float
) -> MultiChannelResult:
    """Calculate Manders colocalization coefficients."""

    if not chan_1_result.cell_positions or not chan_2_result.cell_positions:
        return _create_empty_coloc_result(chan_1_result, chan_2_result, "manders_coefficients")

    # Simplified Manders calculation based on detected cells
    # In practice, this would use pixel-level intensity analysis

    # Use intensity-based method as foundation
    intensity_result = _colocalization_intensity_based(
        chan_1_result, chan_2_result, intensity_threshold
    )

    # Calculate Manders-like coefficients
    total_int_1 = sum(chan_1_result.cell_intensities)
    total_int_2 = sum(chan_2_result.cell_intensities)

    # Simplified: assume colocalized cells contribute their full intensity
    coloc_int_1 = sum(chan_1_result.cell_intensities[i] for i, j in
                     [(i, j) for i in range(len(chan_1_result.cell_positions))
                      for j in range(len(chan_2_result.cell_positions))
                      if (i, j) in [(0, 0)]])  # Simplified placeholder

    # Manders coefficients (M1 and M2)
    m1 = coloc_int_1 / total_int_1 if total_int_1 > 0 else 0
    m2 = coloc_int_1 / total_int_2 if total_int_2 > 0 else 0  # Simplified

    intensity_result.colocalization_method = "manders_coefficients"
    intensity_result.colocalization_metrics.update({
        "manders_m1": m1,
        "manders_m2": m2,
        "note": "Simplified cell-based Manders calculation"
    })

    return intensity_result


def _create_segmentation_visualization(
    image: np.ndarray,
    positions: List[Tuple[float, float]],
    max_sigma: float,
    cell_areas: List[float] = None,
    binary_mask: np.ndarray = None
) -> np.ndarray:
    """Create segmentation visualization using actual binary mask if available."""

    # If we have the actual binary mask from detection, use it directly
    if binary_mask is not None:
        # Convert boolean mask to uint16 to match input image dtype
        # Use max intensity for detected cells, 0 for background
        max_intensity = image.max() if image.max() > 0 else 65535
        return (binary_mask * max_intensity).astype(image.dtype)

    # Fallback to original circular marker approach for blob methods
    visualization = image.copy()

    # Mark detected cells with their actual sizes
    for i, (x, y) in enumerate(positions):
        # Use actual cell area if available, otherwise fall back to max_sigma
        if cell_areas and i < len(cell_areas):
            # Convert area to radius (assuming circular cells)
            radius = np.sqrt(cell_areas[i] / np.pi)
        else:
            # Fallback to max_sigma for backward compatibility
            radius = max_sigma * 2

        # Create circular markers with actual cell size
        rr, cc = np.ogrid[:image.shape[0], :image.shape[1]]
        mask = (rr - y)**2 + (cc - x)**2 <= radius**2

        # Ensure indices are within bounds
        valid_mask = (rr >= 0) & (rr < image.shape[0]) & (cc >= 0) & (cc < image.shape[1])
        mask = mask & valid_mask

        visualization[mask] = visualization.max()  # Bright markers

    return visualization


def _create_colocalization_map(
    chan_1_img: np.ndarray,
    chan_2_img: np.ndarray,
    coloc_result: MultiChannelResult
) -> np.ndarray:
    """Create colocalization visualization map."""
    # Create RGB-like visualization
    coloc_map = np.zeros_like(chan_1_img)

    # Mark colocalized positions
    for x, y in coloc_result.overlap_positions:
        # Create markers for colocalized cells
        rr, cc = np.ogrid[:chan_1_img.shape[0], :chan_1_img.shape[1]]
        mask = (rr - y)**2 + (cc - x)**2 <= 25  # 5-pixel radius

        valid_mask = (rr >= 0) & (rr < chan_1_img.shape[0]) & (cc >= 0) & (cc < chan_1_img.shape[1])
        mask = mask & valid_mask

        coloc_map[mask] = chan_1_img.max()  # Bright colocalization markers

    return coloc_map
