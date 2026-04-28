"""
Converted from CellProfiler: MeasureImageSkeleton
Original: MeasureImageSkeleton

Measures the number of branches and endpoints in a skeletonized structure
such as neurons, roots, or vasculature.

A branch is a pixel with more than two neighbors.
An endpoint is a pixel with only one neighbor.
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.processing.materialization import csv_materializer
import scipy.ndimage


@dataclass
class SkeletonMeasurement:
    """Measurements from skeleton analysis."""
    slice_index: int
    branches: int
    endpoints: int


def _neighbors_2d(image: np.ndarray) -> np.ndarray:
    """
    Counts the neighbor pixels for each pixel of a 2D image.
    
    Uses uniform filter to count neighbors in a 3x3 neighborhood.
    
    Args:
        image: A two-dimensional binary image (H, W)
        
    Returns:
        Array of neighbor counts for each pixel
    """
    padding = np.pad(image, 1, mode="constant", constant_values=0)
    mask = padding > 0
    padding = padding.astype(np.float64)
    
    # 3x3 neighborhood: 9 pixels, subtract 1 for center pixel
    response = 9 * scipy.ndimage.uniform_filter(padding, size=3) - 1
    labels = (response * mask)[1:-1, 1:-1]
    
    return labels.astype(np.uint16)


def _neighbors_3d(image: np.ndarray) -> np.ndarray:
    """
    Counts the neighbor pixels for each pixel of a 3D image.
    
    Uses uniform filter to count neighbors in a 3x3x3 neighborhood.
    
    Args:
        image: A three-dimensional binary image (D, H, W)
        
    Returns:
        Array of neighbor counts for each pixel
    """
    padding = np.pad(image, 1, mode="constant", constant_values=0)
    mask = padding > 0
    padding = padding.astype(np.float64)
    
    # 3x3x3 neighborhood: 27 pixels, subtract 1 for center pixel
    response = 27 * scipy.ndimage.uniform_filter(padding, size=3) - 1
    labels = (response * mask)[1:-1, 1:-1, 1:-1]
    
    return labels.astype(np.uint16)


def _count_branches_2d(image: np.ndarray) -> int:
    """Count branch points (pixels with more than 2 neighbors) in 2D."""
    neighbors = _neighbors_2d(image)
    return int(np.count_nonzero(neighbors > 2))


def _count_endpoints_2d(image: np.ndarray) -> int:
    """Count endpoints (pixels with exactly 1 neighbor) in 2D."""
    neighbors = _neighbors_2d(image)
    return int(np.count_nonzero(neighbors == 1))


def _count_branches_3d(image: np.ndarray) -> int:
    """Count branch points (pixels with more than 2 neighbors) in 3D."""
    neighbors = _neighbors_3d(image)
    return int(np.count_nonzero(neighbors > 2))


def _count_endpoints_3d(image: np.ndarray) -> int:
    """Count endpoints (pixels with exactly 1 neighbor) in 3D."""
    neighbors = _neighbors_3d(image)
    return int(np.count_nonzero(neighbors == 1))


@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(("skeleton_measurements", csv_materializer(
    fields=["slice_index", "branches", "endpoints"],
    analysis_type="skeleton_measurement"
)))
def measure_image_skeleton(
    image: np.ndarray,
) -> Tuple[np.ndarray, SkeletonMeasurement]:
    """
    Measure branches and endpoints in a skeletonized image.
    
    Analyzes a morphological skeleton to count:
    - Branches: pixels with more than two neighbors (junction points)
    - Endpoints: pixels with only one neighbor (terminal points)
    
    Args:
        image: Skeletonized binary image (H, W). Create with MorphologicalSkeleton.
        
    Returns:
        Tuple of:
        - Original image (passed through)
        - SkeletonMeasurement dataclass with branch and endpoint counts
    """
    # Ensure binary
    binary = (image > 0).astype(np.uint8)
    
    # Count branches and endpoints
    branch_count = _count_branches_2d(binary)
    endpoint_count = _count_endpoints_2d(binary)
    
    measurement = SkeletonMeasurement(
        slice_index=0,
        branches=branch_count,
        endpoints=endpoint_count
    )
    
    return image, measurement


@numpy(contract=ProcessingContract.PURE_3D)
@special_outputs(("skeleton_measurements_3d", csv_materializer(
    fields=["slice_index", "branches", "endpoints"],
    analysis_type="skeleton_measurement_3d"
)))
def measure_image_skeleton_3d(
    image: np.ndarray,
) -> Tuple[np.ndarray, SkeletonMeasurement]:
    """
    Measure branches and endpoints in a 3D skeletonized image.
    
    Analyzes a 3D morphological skeleton to count:
    - Branches: voxels with more than two neighbors (junction points)
    - Endpoints: voxels with only one neighbor (terminal points)
    
    Uses 26-connectivity (3x3x3 neighborhood) for neighbor counting.
    
    Args:
        image: 3D skeletonized binary image (D, H, W).
        
    Returns:
        Tuple of:
        - Original image (passed through)
        - SkeletonMeasurement dataclass with branch and endpoint counts
    """
    # Ensure binary
    binary = (image > 0).astype(np.uint8)
    
    # Count branches and endpoints in 3D
    branch_count = _count_branches_3d(binary)
    endpoint_count = _count_endpoints_3d(binary)
    
    measurement = SkeletonMeasurement(
        slice_index=0,
        branches=branch_count,
        endpoints=endpoint_count
    )
    
    return image, measurement