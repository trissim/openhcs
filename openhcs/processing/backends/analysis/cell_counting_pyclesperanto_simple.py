"""
Simplified GPU-accelerated cell counting using pyclesperanto Voronoi-Otsu labeling.

This module provides a clean, simple implementation based on the pyclesperanto
Voronoi-Otsu labeling reference workflow, while maintaining compatibility with
the existing OpenHCS materialization system.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Core scientific computing imports
import pandas as pd
import json
import pyclesperanto as cle

# OpenHCS imports
from openhcs.core.memory import pyclesperanto as pyclesperanto_func
from openhcs.core.pipeline.function_contracts import artifact_outputs
from openhcs.processing.materialization import (
    MaterializationSpec,
    CsvOptions,
    JsonOptions,
)
from openhcs.constants.constants import Backend


class DetectionMethod(Enum):
    """Simplified cell detection methods."""
    VORONOI_OTSU = "voronoi_otsu"      # Voronoi-Otsu labeling (reference workflow)
    THRESHOLD = "threshold"            # Simple thresholding (fallback)


@dataclass
class CellCountResult:
    """Results for single-channel cell counting - compatible with existing system."""
    slice_index: int
    method: str
    cell_count: int
    cell_positions: List[Tuple[float, float]]  # (x, y) centroids
    cell_areas: List[float]
    cell_intensities: List[float]
    detection_confidence: List[float]
    parameters_used: Dict[str, Any]
    binary_mask: np.ndarray = None  # Labeled mask for ROI extraction


def count_cells_single_channel(
    image_stack: np.ndarray,
    # Simplified parameters
    detection_method: DetectionMethod = DetectionMethod.VORONOI_OTSU,
    # Core parameters
    gaussian_sigma: float = 1.0,
    min_cell_area: int = 50,
    max_cell_area: int = 5000,
    # Output parameters
    return_segmentation_mask: bool = False
) -> Tuple[np.ndarray, List[CellCountResult]]:
    """
    Count cells using simplified Voronoi-Otsu labeling workflow.
    
    This implementation follows the pyclesperanto reference workflow:
    1. Gaussian blur
    2. Detect spots
    3. Otsu threshold
    4. Masked Voronoi labeling
    
    Args:
        image_stack: 3D array (Z, Y, X) where each Z slice is processed independently
        detection_method: Method for cell detection (VORONOI_OTSU or THRESHOLD)
        gaussian_sigma: Standard deviation for Gaussian blur
        min_cell_area: Minimum area for valid cells (pixels)
        max_cell_area: Maximum area for valid cells (pixels)
        return_segmentation_mask: Return segmentation masks in output
        
    Returns:
        output_stack: Original image stack unchanged (Z, Y, X)
        cell_count_results: List of CellCountResult objects for each slice
        segmentation_masks: (Special output) List of segmentation mask arrays if requested
    """
    if image_stack.ndim != 3:
        raise ValueError(f"Expected 3D image stack, got {image_stack.ndim}D")
    
    results = []
    segmentation_masks = []

    # Store parameters for reproducibility
    parameters = {
        "detection_method": detection_method.value,
        "gaussian_sigma": gaussian_sigma,
        "min_cell_area": min_cell_area,
        "max_cell_area": max_cell_area,
        "return_segmentation_mask": return_segmentation_mask
    }

    logger.info(f"Processing {image_stack.shape[0]} slices with {detection_method.value} method")

    for z_idx in range(image_stack.shape[0]):
        # Extract slice and push to GPU
        slice_img = cle.push(image_stack[z_idx].astype(np.float32))

        # Detect cells using specified method
        if detection_method == DetectionMethod.VORONOI_OTSU:
            result = _detect_cells_voronoi_otsu(slice_img, z_idx, parameters)
        elif detection_method == DetectionMethod.THRESHOLD:
            result = _detect_cells_threshold(slice_img, z_idx, parameters)
        else:
            raise ValueError(f"Unknown detection method: {detection_method.value}")

        results.append(result)

        # Create segmentation mask if requested
        if return_segmentation_mask:
            segmentation_mask = _create_labeled_mask(slice_img, result)
            segmentation_masks.append(segmentation_mask)

    # Always return segmentation masks (empty list if not requested)
    return image_stack, results, segmentation_masks


def _detect_cells_voronoi_otsu(
    gpu_image: cle.Array,
    slice_idx: int,
    params: Dict[str, Any]
) -> CellCountResult:
    """
    Detect cells using Voronoi-Otsu labeling (reference workflow).
    
    This follows the exact workflow from the pyclesperanto notebook:
    1. Gaussian blur
    2. Detect spots
    3. Otsu threshold
    4. Masked Voronoi labeling
    """
    # Step 1: Apply Gaussian blur
    blurred = cle.gaussian_blur(gpu_image, sigma_x=params["gaussian_sigma"], sigma_y=params["gaussian_sigma"])
    
    # Step 2: Detect spots (simplified - use fixed radius)
    spots = cle.detect_spots(blurred, radius_x=1, radius_y=1)
    
    # Step 3: Create binary mask using Otsu threshold
    binary = cle.threshold_otsu(blurred)
    
    # Step 4: Apply Voronoi labeling
    voronoi_labels = cle.masked_voronoi_labeling(spots, binary)
    
    # Step 5: Filter by size and get statistics
    voronoi_labels = cle.remove_small_labels(voronoi_labels, minimum_size=params["min_cell_area"])
    voronoi_labels = cle.remove_large_labels(voronoi_labels, maximum_size=params["max_cell_area"])
    
    # Get statistics from pyclesperanto
    stats_dict = cle.statistics_of_labelled_pixels(gpu_image, voronoi_labels)
    
    # Extract results
    if 'label' in stats_dict and len(stats_dict['label']) > 0:
        positions = [(float(x), float(y)) for x, y in zip(stats_dict['centroid_x'], stats_dict['centroid_y'])]
        areas = stats_dict.get('area', [])
        intensities = stats_dict.get('mean_intensity', [])
        confidences = [1.0] * len(positions)  # Simple confidence for Voronoi method
        
        # Convert labeled mask to numpy for binary_mask field
        labeled_mask = cle.pull(voronoi_labels)
    else:
        positions = []
        areas = []
        intensities = []
        confidences = []
        labeled_mask = np.zeros_like(cle.pull(gpu_image), dtype=np.uint16)

    return CellCountResult(
        slice_index=slice_idx,
        method="voronoi_otsu_pyclesperanto",
        cell_count=len(positions),
        cell_positions=positions,
        cell_areas=areas,
        cell_intensities=intensities,
        detection_confidence=confidences,
        parameters_used=params,
        binary_mask=labeled_mask
    )


def _detect_cells_threshold(
    gpu_image: cle.Array,
    slice_idx: int,
    params: Dict[str, Any]
) -> CellCountResult:
    """
    Detect cells using simple thresholding (fallback method).
    """
    # Apply threshold
    max_intensity = cle.maximum_of_all_pixels(gpu_image)
    threshold_val = 0.1 * max_intensity  # Fixed threshold for simplicity
    binary = cle.greater_constant(gpu_image, scalar=threshold_val)
    
    # Connected components labeling
    labels = cle.connected_components_labeling(binary)
    
    # Remove small and large objects
    labels = cle.remove_small_labels(labels, minimum_size=params["min_cell_area"])
    labels = cle.remove_large_labels(labels, maximum_size=params["max_cell_area"])
    
    # Get statistics
    stats_dict = cle.statistics_of_labelled_pixels(gpu_image, labels)
    
    # Extract results
    if 'label' in stats_dict and len(stats_dict['label']) > 0:
        positions = [(float(x), float(y)) for x, y in zip(stats_dict['centroid_x'], stats_dict['centroid_y'])]
        areas = stats_dict.get('area', [])
        intensities = stats_dict.get('mean_intensity', [])
        
        # Use intensity as confidence measure
        max_intensity_cpu = float(max_intensity)
        confidences = [float(intensity / max_intensity_cpu) for intensity in intensities]
        
        # Convert labeled mask to numpy
        labeled_mask = cle.pull(labels)
    else:
        positions = []
        areas = []
        intensities = []
        confidences = []
        labeled_mask = np.zeros_like(cle.pull(gpu_image), dtype=np.uint16)

    return CellCountResult(
        slice_index=slice_idx,
        method="threshold_pyclesperanto",
        cell_count=len(positions),
        cell_positions=positions,
        cell_areas=areas,
        cell_intensities=intensities,
        detection_confidence=confidences,
        parameters_used=params,
        binary_mask=labeled_mask
    )


def _create_labeled_mask(gpu_image: cle.Array, result: CellCountResult) -> np.ndarray:
    """
    Create labeled segmentation mask for ROI extraction.
    
    Returns a labeled mask where each connected region has a unique integer ID.
    """
    # If we already have the binary mask from detection, use it
    if result.binary_mask is not None:
        return result.binary_mask
    
    # Fallback: create mask from positions
    image_shape = cle.pull(gpu_image).shape
    labeled_mask = np.zeros(image_shape, dtype=np.uint16)
    
    # Create simple circular regions around detected positions
    for i, (x, y) in enumerate(result.cell_positions):
        if i < len(result.cell_areas):
            # Use actual cell area if available
            radius = np.sqrt(result.cell_areas[i] / np.pi)
        else:
            # Default radius
            radius = 5.0
        
        # Create circular region
        rr, cc = np.ogrid[:image_shape[0], :image_shape[1]]
        mask = (rr - y)**2 + (cc - x)**2 <= radius**2
        
        # Ensure indices are within bounds
        valid_mask = (rr >= 0) & (rr < image_shape[0]) & (cc >= 0) & (cc < image_shape[1])
        mask = mask & valid_mask
        
        # Assign unique label ID (i+1 to avoid background label 0)
        labeled_mask[mask] = i + 1
    
    return labeled_mask


def count_cells_simple(
    image: np.ndarray,
    gaussian_sigma: float = 1.0,
    min_cell_area: int = 50,
    max_cell_area: int = 5000
) -> Tuple[int, List[Tuple[float, float]]]:
    """
    Quick cell counting with minimal parameters.
    
    Args:
        image: 2D numpy array
        gaussian_sigma: Gaussian blur sigma
        min_cell_area: Minimum cell area (pixels)
        max_cell_area: Maximum cell area (pixels)
        
    Returns:
        cell_count: Number of detected cells
        cell_positions: List of (x, y) centroid positions
    """
    # Convert 2D to 3D for the main function
    image_stack = image[np.newaxis, ...]
    
    # Call main function with Voronoi-Otsu method
    _, results, _ = count_cells_single_channel(
        image_stack,
        detection_method=DetectionMethod.VORONOI_OTSU,
        gaussian_sigma=gaussian_sigma,
        min_cell_area=min_cell_area,
        max_cell_area=max_cell_area,
        return_segmentation_mask=False
    )
    
    # Extract results from first (and only) slice
    result = results[0]
    return result.cell_count, result.cell_positions
