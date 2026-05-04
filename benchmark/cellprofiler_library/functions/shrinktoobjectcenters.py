"""
Converted from CellProfiler: ShrinkToObjectCenters
Original: ShrinkToObjectCenters.find_centroids

Transforms a set of labeled objects into a label image with single points
representing each object. The location of each point corresponds to the
centroid of the input objects.

Note: If the object is not sufficiently round, the resulting single pixel
may reside outside the original object (e.g., U-shaped objects).
"""

import numpy as np
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.processing.materialization import csv_materializer, segmentation_mask_rois
from openhcs.processing.backends.analysis.region_properties import (
    LabelRegionPropertiesBackendStrategy,
)
from dataclasses import dataclass
from typing import Tuple


@dataclass
class CentroidStats:
    slice_index: int
    object_count: int


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(
    ("centroid_stats", csv_materializer(fields=["slice_index", "object_count"], analysis_type="centroid")),
    ("centroid_labels", segmentation_mask_rois())
)
def shrink_to_object_centers(
    image: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, CentroidStats, np.ndarray]:
    """
    Transform labeled objects into single-point centroids.
    
    Takes a label image where each object has a unique integer label and
    produces a new label image where each object is represented by a single
    pixel at its centroid location.
    
    Args:
        image: Input image (H, W), passed through unchanged
        labels: Label image (H, W) where each object has unique integer label
    
    Returns:
        Tuple of:
            - Original image (unchanged)
            - CentroidStats dataclass with object count
            - Centroid label image (H, W) with single-pixel objects
    """
    region_props = LabelRegionPropertiesBackendStrategy.for_memory_type().measure_2d(
        labels.astype(np.int32, copy=False)
    )
    
    # Create output label image with same shape as input
    output_labels = np.zeros_like(labels, dtype=np.int32)
    
    # Place each object's label at its centroid location
    for index, label_id in enumerate(region_props.label):
        centroid_int = (
            int(round(float(region_props.centroid_y[index]))),
            int(round(float(region_props.centroid_x[index]))),
        )
        
        # Ensure centroid is within image bounds
        if all(0 <= centroid_int[i] < labels.shape[i] for i in range(len(centroid_int))):
            output_labels[centroid_int] = int(label_id)
    
    stats = CentroidStats(
        slice_index=0,
        object_count=int(region_props.label.size)
    )
    
    return image, stats, output_labels


@numpy(contract=ProcessingContract.PURE_3D)
@special_inputs("labels")
@special_outputs(
    ("centroid_stats", csv_materializer(fields=["slice_index", "object_count"], analysis_type="centroid")),
    ("centroid_labels", segmentation_mask_rois())
)
def shrink_to_object_centers_3d(
    image: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, CentroidStats, np.ndarray]:
    """
    Transform 3D labeled objects into single-point centroids.
    
    Takes a 3D label image where each object has a unique integer label and
    produces a new label image where each object is represented by a single
    voxel at its centroid location.
    
    Args:
        image: Input image (D, H, W), passed through unchanged
        labels: Label image (D, H, W) where each object has unique integer label
    
    Returns:
        Tuple of:
            - Original image (unchanged)
            - CentroidStats dataclass with object count
            - Centroid label image (D, H, W) with single-voxel objects
    """
    from skimage.measure import regionprops
    
    # Get region properties to find centroids
    props = regionprops(labels.astype(np.int32))
    
    # Create output label image with same shape as input
    output_labels = np.zeros_like(labels, dtype=np.int32)
    
    # Place each object's label at its centroid location
    for region in props:
        # Get centroid coordinates (z, row, col for 3D)
        centroid = region.centroid
        # Convert to integer indices
        centroid_int = tuple(int(round(c)) for c in centroid)
        
        # Ensure centroid is within image bounds
        if all(0 <= centroid_int[i] < labels.shape[i] for i in range(len(centroid_int))):
            output_labels[centroid_int] = region.label
    
    stats = CentroidStats(
        slice_index=0,
        object_count=len(props)
    )
    
    return image, stats, output_labels
