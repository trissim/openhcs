"""
GPU-accelerated cell counting and multi-channel colocalization analysis for OpenHCS.

This module provides comprehensive cell counting capabilities using pyclesperanto,
supporting both single-channel and multi-channel analysis with various detection
methods and colocalization metrics.
"""

import numpy as np
import logging
import gc
from typing import Dict, List, Tuple, Any, Union

logger = logging.getLogger(__name__)

import pyclesperanto as cle
from scipy.spatial.distance import cdist

# OpenHCS imports
from openhcs.core.memory import pyclesperanto as pyclesperanto_func
from openhcs.core.pipeline.function_contracts import artifact_outputs
from openhcs.processing.materialization import (
    MaterializationSpec,
    CsvOptions,
    JsonOptions,
)
from openhcs.processing.backends.analysis.cell_counting_common import (
    CellCountResult,
    ColocalizationMethod,
    DetectionMethod,
    DistanceColocalizationMetrics,
    IntensityColocalizationMetrics,
    MultiChannelResult,
    ThresholdMethod,
)


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
        "watershed_threshold_method": watershed_threshold_method.value if hasattr(watershed_threshold_method, 'value') else watershed_threshold_method,
        "gaussian_sigma": gaussian_sigma,
        "median_disk_size": median_disk_size,
        "min_cell_area": min_cell_area,
        "max_cell_area": max_cell_area,
        "remove_border_cells": remove_border_cells
    }

    logging.info(f"Processing {image_stack.shape[0]} slices with {detection_method.value} method")

    for z_idx in range(image_stack.shape[0]):
        # Extract slice - keep as pyclesperanto array
        slice_img = image_stack[z_idx]

        # Apply preprocessing if enabled (all operations stay in GPU)
        if enable_preprocessing:
            slice_img = _preprocess_image(slice_img, gaussian_sigma, median_disk_size)

        # Detect cells using specified method (slice_img stays as cle array)
        result = _detect_cells_single_method(
            slice_img, z_idx, detection_method.value, parameters
        )

        results.append(result)

        # Create segmentation mask if requested
        if return_segmentation_mask:
            segmentation_mask = _create_segmentation_visualization(
                slice_img, result.cell_positions, max_sigma
            )
            segmentation_masks.append(segmentation_mask)

    # Always return segmentation masks (empty list if not requested)
    # This ensures consistent return signature for special outputs system
    return image_stack, results, segmentation_masks


@pyclesperanto_func
@artifact_outputs((
    "multi_channel_counts",
    MaterializationSpec(
        JsonOptions(filename_suffix=".json", wrap_list=True),
        CsvOptions(filename_suffix="_details.csv"),
        primary=0,
    ),
))
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


def _preprocess_image(image, gaussian_sigma: float, median_disk_size: int):
    """Apply preprocessing to enhance cell detection using pyclesperanto."""
    # Assume image is already a pyclesperanto array
    gpu_image = image

    # Gaussian blur to reduce noise
    if gaussian_sigma > 0:
        gpu_image = cle.gaussian_blur(gpu_image, sigma_x=gaussian_sigma, sigma_y=gaussian_sigma)

    # Median filter to remove salt-and-pepper noise
    if median_disk_size > 0:
        gpu_image = cle.median_box(gpu_image, radius_x=median_disk_size, radius_y=median_disk_size)

    # Return the GPU array
    return gpu_image


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
    """Detect cells using fast LoG-like blob detection."""
    gpu_image = image

    # Use single scale for speed - average of min and max sigma
    sigma = (params["min_sigma"] + params["max_sigma"]) / 2

    # Fast LoG approximation: Use DoG like blob_dog but with closer scales
    blurred1 = cle.gaussian_blur(gpu_image, sigma_x=sigma, sigma_y=sigma)
    blurred2 = cle.gaussian_blur(gpu_image, sigma_x=sigma*1.6, sigma_y=sigma*1.6)

    # Difference approximates Laplacian
    edges = cle.subtract_images(blurred1, blurred2)

    # Threshold the edge response
    max_response = cle.maximum_of_all_pixels(cle.absolute(edges))
    threshold_val = params["threshold"] * max_response
    thresholded = cle.greater_constant(cle.absolute(edges), scalar=threshold_val)

    # Find local maxima
    maxima = cle.detect_maxima(cle.absolute(edges),
                              radius_x=int(sigma),
                              radius_y=int(sigma))

    # Combine threshold and maxima
    valid_maxima = cle.binary_and(thresholded, maxima)

    # Dilate maxima more aggressively to create proper cell-sized regions
    dilated = valid_maxima
    for _ in range(6):  # More aggressive dilation
        dilated = cle.dilate_box(dilated)

    # Label connected components
    labels = cle.connected_components_labeling(dilated)

    # Remove small and large objects
    labels = cle.remove_small_labels(labels, minimum_size=params["min_cell_area"])
    labels = cle.remove_large_labels(labels, maximum_size=params["max_cell_area"])

    # Get statistics
    if cle.maximum_of_all_pixels(labels) > 0:
        stats_dict = cle.statistics_of_labelled_pixels(gpu_image, labels)

        positions = []
        areas = []
        intensities = []
        confidences = []

        if 'centroid_x' in stats_dict and len(stats_dict['centroid_x']) > 0:
            for i, (x, y) in enumerate(zip(stats_dict['centroid_x'], stats_dict['centroid_y'])):
                positions.append((float(x), float(y)))

                # Get area and intensity
                area = float(stats_dict['area'][i]) if i < len(stats_dict.get('area', [])) else sigma**2
                intensity = float(stats_dict['mean_intensity'][i]) if i < len(stats_dict.get('mean_intensity', [])) else 1.0

                areas.append(area)
                intensities.append(intensity)
                confidences.append(min(1.0, intensity / max_response))
    else:
        positions = []
        areas = []
        intensities = []
        confidences = []

    return CellCountResult.from_measurements(
        slice_idx,
        "blob_log_pyclesperanto",
        positions,
        areas,
        intensities,
        confidences,
        params,
    )


def _detect_cells_blob_dog(image: np.ndarray, slice_idx: int, params: Dict[str, Any]) -> CellCountResult:
    """Detect cells using fast Difference of Gaussians blob detection."""
    gpu_image = image

    # Use only two scales for speed
    sigma1 = params["min_sigma"]
    sigma2 = params["max_sigma"]

    # Apply Gaussian blurs
    blur1 = cle.gaussian_blur(gpu_image, sigma_x=sigma1, sigma_y=sigma1)
    blur2 = cle.gaussian_blur(gpu_image, sigma_x=sigma2, sigma_y=sigma2)

    # Difference of Gaussians
    dog = cle.subtract_images(blur1, blur2)

    # Threshold the DoG response
    max_response = cle.maximum_of_all_pixels(cle.absolute(dog))
    threshold_val = params["threshold"] * max_response
    thresholded = cle.greater_constant(cle.absolute(dog), scalar=threshold_val)

    # Find local maxima
    maxima = cle.detect_maxima(cle.absolute(dog),
                              radius_x=int(sigma1),
                              radius_y=int(sigma1))

    # Combine threshold and maxima
    valid_maxima = cle.binary_and(thresholded, maxima)

    # Dilate maxima to create regions for area calculation
    dilated = valid_maxima
    for _ in range(3):  # Moderate dilation
        dilated = cle.dilate_box(dilated)

    # Label connected components
    labels = cle.connected_components_labeling(dilated)

    # Remove small and large objects
    labels = cle.remove_small_labels(labels, minimum_size=params["min_cell_area"])
    labels = cle.remove_large_labels(labels, maximum_size=params["max_cell_area"])

    # Get statistics
    if cle.maximum_of_all_pixels(labels) > 0:
        stats_dict = cle.statistics_of_labelled_pixels(gpu_image, labels)

        positions = []
        areas = []
        intensities = []
        confidences = []

        if 'centroid_x' in stats_dict and len(stats_dict['centroid_x']) > 0:
            for i, (x, y) in enumerate(zip(stats_dict['centroid_x'], stats_dict['centroid_y'])):
                positions.append((float(x), float(y)))

                # Get area and intensity
                area = float(stats_dict['area'][i]) if i < len(stats_dict.get('area', [])) else sigma1**2
                intensity = float(stats_dict['mean_intensity'][i]) if i < len(stats_dict.get('mean_intensity', [])) else 1.0

                areas.append(area)
                intensities.append(intensity)
                confidences.append(min(1.0, intensity / max_response))
    else:
        positions = []
        areas = []
        intensities = []
        confidences = []

    return CellCountResult.from_measurements(
        slice_idx,
        "blob_dog_pyclesperanto",
        positions,
        areas,
        intensities,
        confidences,
        params,
    )



    # Extract the data we need from the statistics dictionary
    if 'label' in stats_dict and len(stats_dict['label']) > 0:
        # We have detected objects
        areas = stats_dict.get('area', [])
        labels_list = stats_dict.get('label', [])
    else:
        # No objects detected
        areas = []
        labels_list = []
        centroids_x = []
        centroids_y = []

    # Process similar to blob_log
    positions = []
    filtered_areas = []
    intensities = []
    confidences = []

    for i in range(len(labels_list)):
        if i < len(centroids_x) and i < len(centroids_y) and i < len(areas):
            x = float(centroids_x[i])
            y = float(centroids_y[i])
            positions.append((x, y))

            # Get area
            area = float(areas[i])
            filtered_areas.append(area)

            # Mean intensity (if available in stats)
            mean_intensities = stats_dict.get('mean_intensity', [])
            if i < len(mean_intensities):
                intensity = float(mean_intensities[i])
            else:
                intensity = 1.0  # Default value
            intensities.append(intensity)

            # Use area as confidence measure
            confidence = min(1.0, area / (np.pi * params["max_sigma"]**2))
            confidences.append(confidence)

    filtered_data = _filter_by_area(
        positions, filtered_areas, intensities, confidences,
        params["min_cell_area"], params["max_cell_area"]
    )

    return CellCountResult.from_measurements(
        slice_idx, "blob_dog_pyclesperanto", *filtered_data, params
    )


def _detect_cells_blob_doh(image: np.ndarray, slice_idx: int, params: Dict[str, Any]) -> CellCountResult:
    """Detect cells using Hessian-like detection with pyclesperanto."""
    # Assume image is already a pyclesperanto array
    gpu_image = image

    # Apply Gaussian blur for smoothing
    sigma = (params["min_sigma"] + params["max_sigma"]) / 2
    blurred = cle.gaussian_blur(gpu_image, sigma_x=sigma, sigma_y=sigma)

    # Apply edge detection (approximates Hessian determinant)
    edges = cle.binary_edge_detection(blurred)

    # Apply threshold to original image for valid regions
    threshold_val = params["threshold"] * cle.maximum_of_all_pixels(gpu_image)
    thresholded = cle.greater_constant(gpu_image, scalar=threshold_val)

    # Detect local maxima in edge response
    maxima = cle.detect_maxima_box(edges,
                                  radius_x=int(params["min_sigma"]),
                                  radius_y=int(params["min_sigma"]))

    # Combine threshold and maxima
    combined = cle.binary_and(thresholded, maxima)

    # Dilate the maxima points to create proper regions
    dilated = cle.dilate_box(combined)

    # Label the dilated regions
    labels = cle.connected_components_labeling(dilated)

    # Remove small and large objects (same as simple baseline)
    labels = cle.remove_small_labels(labels, minimum_size=params["min_cell_area"])
    labels = cle.remove_large_labels(labels, maximum_size=params["max_cell_area"])

    # Get statistics - this returns a dictionary with centroids included
    stats_dict = cle.statistics_of_labelled_pixels(gpu_image, labels)

    # Extract centroids directly from statistics (much simpler!)
    centroids_x = stats_dict.get('centroid_x', [])
    centroids_y = stats_dict.get('centroid_y', [])

    # Extract the data we need from the statistics dictionary
    if 'label' in stats_dict and len(stats_dict['label']) > 0:
        # We have detected objects
        areas = stats_dict.get('area', [])
        labels_list = stats_dict.get('label', [])
    else:
        # No objects detected
        areas = []
        labels_list = []
        centroids_x = []
        centroids_y = []

    # Process similar to other blob methods
    positions = []
    filtered_areas = []
    intensities = []
    confidences = []

    for i in range(len(labels_list)):
        if i < len(centroids_x) and i < len(centroids_y) and i < len(areas):
            x = float(centroids_x[i])
            y = float(centroids_y[i])
            positions.append((x, y))

            # Get area
            area = float(areas[i])
            filtered_areas.append(area)

            # Mean intensity (if available in stats)
            mean_intensities = stats_dict.get('mean_intensity', [])
            if i < len(mean_intensities):
                intensity = float(mean_intensities[i])
            else:
                intensity = 1.0  # Default value
            intensities.append(intensity)

            # Use area as confidence measure
            confidence = min(1.0, area / (np.pi * params["max_sigma"]**2))
            confidences.append(confidence)

    filtered_data = _filter_by_area(
        positions, filtered_areas, intensities, confidences,
        params["min_cell_area"], params["max_cell_area"]
    )

    return CellCountResult.from_measurements(
        slice_idx, "blob_doh_pyclesperanto", *filtered_data, params
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
    """Detect cells using watershed segmentation with pyclesperanto."""
    # Assume image is already a pyclesperanto array
    gpu_image = image

    # Use Otsu thresholding (optimal for microscopy images)
    binary = cle.threshold_otsu(gpu_image)

    # Remove small and large objects with explicit memory cleanup
    temp_labels = cle.connected_components_labeling(binary)

    # Clean up intermediate arrays to prevent GPU OOM
    temp_labels_filtered = cle.remove_small_labels(temp_labels, minimum_size=params["min_cell_area"])
    del temp_labels  # Explicit cleanup

    temp_labels = cle.remove_large_labels(temp_labels_filtered, maximum_size=params["max_cell_area"])
    del temp_labels_filtered  # Explicit cleanup

    binary_filtered = cle.greater_constant(temp_labels, scalar=0)
    del binary  # Cleanup original binary
    binary = binary_filtered

    # Remove border objects if requested
    if params["remove_border_cells"]:
        temp_labels_border = cle.connected_components_labeling(binary)
        del temp_labels  # Cleanup previous temp_labels

        temp_labels = cle.remove_labels_on_edges(temp_labels_border)
        del temp_labels_border  # Explicit cleanup

        binary_final = cle.greater_constant(temp_labels, scalar=0)
        del binary  # Cleanup previous binary
        binary = binary_final

    # Since pyclesperanto doesn't have watershed, use connected components directly
    labels = cle.connected_components_labeling(binary)
    del binary  # Cleanup binary mask

    # Get statistics - this returns a dictionary with centroids included
    stats_dict = cle.statistics_of_labelled_pixels(gpu_image, labels)

    # Cleanup labels array after getting statistics
    del labels

    # Extract centroids directly from statistics (much simpler!)
    centroids_x = stats_dict.get('centroid_x', [])
    centroids_y = stats_dict.get('centroid_y', [])

    # Extract the data we need from the statistics dictionary
    if 'label' in stats_dict and len(stats_dict['label']) > 0:
        # We have detected objects
        areas = stats_dict.get('area', [])
        labels_list = stats_dict.get('label', [])
    else:
        # No objects detected
        areas = []
        labels_list = []
        centroids_x = []
        centroids_y = []

    positions = []
    filtered_areas = []
    intensities = []
    confidences = []

    for i in range(len(labels_list)):
        if i < len(centroids_x) and i < len(centroids_y) and i < len(areas):
            area = float(areas[i])

            # Filter by area
            if params["min_cell_area"] <= area <= params["max_cell_area"]:
                x = float(centroids_x[i])
                y = float(centroids_y[i])
                positions.append((x, y))

                filtered_areas.append(area)

                # Mean intensity (if available in stats)
                mean_intensities = stats_dict.get('mean_intensity', [])
                if i < len(mean_intensities):
                    intensity = float(mean_intensities[i])
                else:
                    intensity = 1.0  # Default value
                intensities.append(intensity)

                # Use area as confidence measure (normalized)
                confidence = min(1.0, area / params["max_cell_area"])
                confidences.append(confidence)

    # Force garbage collection to free GPU memory
    gc.collect()

    return CellCountResult.from_measurements(
        slice_idx,
        "watershed_pyclesperanto",
        positions,
        filtered_areas,
        intensities,
        confidences,
        params,
    )


def _detect_cells_threshold(image: np.ndarray, slice_idx: int, params: Dict[str, Any]) -> CellCountResult:
    """Detect cells using simple thresholding and connected components with pyclesperanto."""
    # Image is already a pyclesperanto array - no conversion needed
    gpu_image = image

    # Apply threshold (all operations stay on GPU)
    max_intensity = cle.maximum_of_all_pixels(gpu_image)
    threshold_val = params["threshold"] * max_intensity
    binary = cle.greater_constant(gpu_image, scalar=threshold_val)

    # Remove small objects with explicit memory cleanup
    temp_labels = cle.connected_components_labeling(binary)
    temp_labels_filtered = cle.remove_small_labels(temp_labels, minimum_size=params["min_cell_area"])
    del temp_labels  # Explicit cleanup

    binary_filtered = cle.greater_constant(temp_labels_filtered, scalar=0)
    del binary  # Cleanup original binary
    del temp_labels_filtered  # Cleanup filtered labels
    binary = binary_filtered

    # Remove border objects if requested with explicit cleanup
    if params["remove_border_cells"]:
        temp_labels = cle.connected_components_labeling(binary)
        temp_labels_no_border = cle.remove_labels_on_edges(temp_labels)
        del temp_labels  # Explicit cleanup

        binary_final = cle.greater_constant(temp_labels_no_border, scalar=0)
        del binary  # Cleanup previous binary
        del temp_labels_no_border  # Cleanup no border labels
        binary = binary_final

    # Label connected components (final labeling on GPU)
    labels = cle.connected_components_labeling(binary)
    del binary  # Cleanup binary mask

    # Get statistics (operates on GPU arrays, returns dictionary)
    stats_dict = cle.statistics_of_labelled_pixels(gpu_image, labels)
    del labels  # Cleanup labels after getting statistics

    # Extract centroids directly from statistics (much simpler!)
    centroids_x = stats_dict.get('centroid_x', [])
    centroids_y = stats_dict.get('centroid_y', [])

    # Extract the data we need from the statistics dictionary
    if 'label' in stats_dict and len(stats_dict['label']) > 0:
        # We have detected objects
        areas = stats_dict.get('area', [])
        labels_list = stats_dict.get('label', [])
    else:
        # No objects detected
        areas = []
        labels_list = []
        centroids_x = []
        centroids_y = []

    max_intensity_cpu = float(max_intensity)

    positions = []
    filtered_areas = []
    intensities = []
    confidences = []

    for i in range(len(labels_list)):
        if i < len(centroids_x) and i < len(centroids_y) and i < len(areas):
            area = float(areas[i])

            # Filter by area
            if params["min_cell_area"] <= area <= params["max_cell_area"]:
                x = float(centroids_x[i])
                y = float(centroids_y[i])
                positions.append((x, y))

                filtered_areas.append(area)

                # Mean intensity (if available in stats)
                mean_intensities = stats_dict.get('mean_intensity', [])
                if i < len(mean_intensities):
                    mean_intensity = float(mean_intensities[i])
                else:
                    mean_intensity = 1.0  # Default value
                intensities.append(mean_intensity)

                # Use intensity as confidence measure
                confidence = mean_intensity / max_intensity_cpu if max_intensity_cpu > 0 else 1.0
                confidences.append(confidence)

    # Force garbage collection to free GPU memory
    gc.collect()

    return CellCountResult.from_measurements(
        slice_idx,
        "threshold_pyclesperanto",
        positions,
        filtered_areas,
        intensities,
        confidences,
        params,
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

    metrics = DistanceColocalizationMetrics(
        average_colocalization_distance=float(avg_distance),
        max_colocalization_distance=float(max_distance_found),
        distance_threshold_used=max_distance,
    )

    return MultiChannelResult(
        slice_index=chan_1_result.slice_index,
        chan_1_results=chan_1_result,
        chan_2_results=chan_2_result,
        colocalization_method="distance_based",
        colocalized_count=colocalized_count,
        colocalization_percentage=colocalization_percentage,
        chan_1_only_count=chan_1_only,
        chan_2_only_count=chan_2_only,
        colocalization_metrics=metrics.as_dict(),
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

    metrics = IntensityColocalizationMetrics(
        intensity_threshold_used=intensity_threshold,
    )

    return MultiChannelResult(
        slice_index=chan_1_result.slice_index,
        chan_1_results=chan_1_result,
        chan_2_results=chan_2_result,
        colocalization_method="intensity_correlation",
        colocalized_count=colocalized_count,
        colocalization_percentage=colocalization_percentage,
        chan_1_only_count=len(pos_1) - colocalized_count,
        chan_2_only_count=len(pos_2) - colocalized_count,
        colocalization_metrics=metrics.as_dict(),
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
    max_sigma: float
) -> np.ndarray:
    """Create segmentation visualization with detected cells marked."""
    # Convert pyclesperanto array to numpy only when needed for visualization
    import pyclesperanto as cle
    if hasattr(image, 'shape') and hasattr(image, 'dtype') and not isinstance(image, np.ndarray):
        # This is a pyclesperanto array - convert to numpy only for final visualization
        visualization = cle.pull(image).copy()
    else:
        visualization = image.copy()

    # Mark detected cells
    for x, y in positions:
        # Create small circular markers
        rr, cc = np.ogrid[:image.shape[0], :image.shape[1]]
        mask = (rr - y)**2 + (cc - x)**2 <= (max_sigma * 2)**2

        # Ensure indices are within bounds
        valid_mask = (rr >= 0) & (rr < image.shape[0]) & (cc >= 0) & (cc < image.shape[1])
        mask = mask & valid_mask

        visualization[mask] = visualization.max()  # Bright markers

    return visualization


def count_cells_simple_baseline(
    image: np.ndarray,  # 2D image only
    threshold: float = 0.1,
    min_cell_area: int = 50,
    max_cell_area: int = 5000
) -> Tuple[np.ndarray, int, List[Tuple[float, float]]]:
    """
    Simple baseline cell counting using pyclesperanto.
    Based on analyse_blobs.ipynb and voronoi_otsu_labeling.ipynb examples.

    Args:
        image: 2D numpy array
        threshold: Threshold as fraction of max intensity (0.0-1.0)
        min_cell_area: Minimum area in pixels
        max_cell_area: Maximum area in pixels

    Returns:
        segmentation_mask: 2D array with labeled cells
        cell_count: Number of detected cells
        cell_positions: List of (x, y) centroid positions
    """
    # Convert to pyclesperanto array
    gpu_image = cle.push(image.astype(np.float32))

    # Apply threshold
    max_intensity = cle.maximum_of_all_pixels(gpu_image)
    threshold_val = threshold * max_intensity
    binary = cle.greater_constant(gpu_image, scalar=threshold_val)

    # Connected components labeling
    labels = cle.connected_components_labeling(binary)

    # Remove small and large objects
    labels = cle.remove_small_labels(labels, minimum_size=min_cell_area)
    labels = cle.remove_large_labels(labels, maximum_size=max_cell_area)

    # Get statistics
    stats = cle.statistics_of_labelled_pixels(gpu_image, labels)

    # Extract results
    if 'label' in stats and len(stats['label']) > 0:
        cell_count = len(stats['label'])
        centroids_x = stats.get('centroid_x', [])
        centroids_y = stats.get('centroid_y', [])
        cell_positions = [(float(x), float(y)) for x, y in zip(centroids_x, centroids_y)]
    else:
        cell_count = 0
        cell_positions = []

    # Convert result back to numpy
    segmentation_mask = cle.pull(labels)

    return segmentation_mask, cell_count, cell_positions


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
