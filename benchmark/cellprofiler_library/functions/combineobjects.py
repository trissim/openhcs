"""
Converted from CellProfiler: CombineObjects
Original: combineobjects
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass
from enum import Enum
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.core.runtime_values import object_label_dense_array
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.processing.materialization import csv_materializer, segmentation_mask_rois


class CombineMethod(Enum):
    MERGE = "merge"
    PRESERVE = "preserve"
    DISCARD = "discard"
    SEGMENT = "segment"


@dataclass
class CombineObjectsStats:
    slice_index: int
    method: str
    input_objects_x: int
    input_objects_y: int
    output_objects: int


def _merge_objects(labels_x: np.ndarray, labels_y: np.ndarray) -> np.ndarray:
    """Merge overlapping objects from two label images into single objects."""
    from scipy.ndimage import label as scipy_label
    
    # Create combined binary mask
    combined_binary = ((labels_x > 0) | (labels_y > 0)).astype(np.uint8)
    
    # Relabel connected components
    merged_labels, _ = scipy_label(combined_binary)
    
    return merged_labels.astype(np.int32)


def _preserve_objects(labels_x: np.ndarray, labels_y: np.ndarray) -> np.ndarray:
    """Preserve objects from labels_x, add non-overlapping objects from labels_y."""
    # Start with labels_x
    result = labels_x.copy().astype(np.int32)
    
    # Find max label in labels_x
    max_label = labels_x.max()
    
    # Find regions in labels_y that don't overlap with labels_x
    non_overlapping_mask = (labels_y > 0) & (labels_x == 0)
    
    if non_overlapping_mask.any():
        # Get unique labels from labels_y in non-overlapping regions
        y_labels_in_mask = np.unique(labels_y[non_overlapping_mask])
        y_labels_in_mask = y_labels_in_mask[y_labels_in_mask > 0]
        
        # Add non-overlapping objects with new labels
        for i, y_label in enumerate(y_labels_in_mask):
            y_object_mask = (labels_y == y_label) & non_overlapping_mask
            result[y_object_mask] = max_label + i + 1
    
    return result


def _discard_objects(labels_x: np.ndarray, labels_y: np.ndarray) -> np.ndarray:
    """Discard objects from labels_x that overlap with labels_y."""
    from scipy.ndimage import label as scipy_label
    
    # Find labels in labels_x that overlap with labels_y
    overlap_mask = (labels_x > 0) & (labels_y > 0)
    overlapping_labels = np.unique(labels_x[overlap_mask])
    
    # Create result excluding overlapping objects
    result = labels_x.copy().astype(np.int32)
    for lbl in overlapping_labels:
        if lbl > 0:
            result[labels_x == lbl] = 0
    
    # Relabel to ensure consecutive labels
    if result.max() > 0:
        binary = result > 0
        result, _ = scipy_label(binary)
    
    return result.astype(np.int32)


def _segment_objects(labels_x: np.ndarray, labels_y: np.ndarray) -> np.ndarray:
    """Segment objects in labels_x using labels_y as seeds/markers."""
    from scipy.ndimage import label as scipy_label
    from skimage.segmentation import watershed
    from scipy.ndimage import distance_transform_edt
    
    # Use labels_y as markers within labels_x regions
    # Create distance transform of labels_x
    binary_x = labels_x > 0
    
    if not binary_x.any():
        return np.zeros_like(labels_x, dtype=np.int32)
    
    # Distance transform for watershed
    distance = distance_transform_edt(binary_x)
    
    # Use labels_y as markers, but only within labels_x regions
    markers = labels_y.copy()
    markers[~binary_x] = 0
    
    if markers.max() == 0:
        # No markers within labels_x, return labels_x as is
        return labels_x.astype(np.int32)
    
    # Apply watershed
    segmented = watershed(-distance, markers, mask=binary_x)
    
    return segmented.astype(np.int32)


@numpy
@special_outputs(
    ("combine_stats", csv_materializer(
        fields=["slice_index", "method", "input_objects_x", "input_objects_y", "output_objects"],
        analysis_type="combine_objects"
    )),
    ("labels", segmentation_mask_rois())
)
def combineobjects(
    image: np.ndarray,
    method: CombineMethod | str = CombineMethod.MERGE,
) -> Tuple[np.ndarray, CombineObjectsStats, np.ndarray]:
    """
    Combine objects from two label images using various methods.
    
    Args:
        image: Shape (2, H, W) - two label images stacked along dim 0.
               image[0] = labels_x (primary objects)
               image[1] = labels_y (secondary objects)
        method: How to combine objects:
               - MERGE: Merge overlapping objects into single objects
               - PRESERVE: Keep labels_x, add non-overlapping from labels_y
               - DISCARD: Remove objects from labels_x that overlap with labels_y
               - SEGMENT: Segment labels_x using labels_y as markers
    
    Returns:
        Tuple of (original image[0], stats, combined labels)
    """
    # Unstack the two label images from dim 0
    labels_x = object_label_dense_array(image[0], dtype=np.int32)
    labels_y = object_label_dense_array(image[1], dtype=np.int32)
    
    # Count input objects
    num_objects_x = len(np.unique(labels_x)) - (1 if 0 in labels_x else 0)
    num_objects_y = len(np.unique(labels_y)) - (1 if 0 in labels_y else 0)
    method = coerce_cellprofiler_enum(CombineMethod, method)
    
    # Apply the selected method
    if method == CombineMethod.MERGE:
        combined_labels = _merge_objects(labels_x, labels_y)
    elif method == CombineMethod.PRESERVE:
        combined_labels = _preserve_objects(labels_x, labels_y)
    elif method == CombineMethod.DISCARD:
        combined_labels = _discard_objects(labels_x, labels_y)
    elif method == CombineMethod.SEGMENT:
        combined_labels = _segment_objects(labels_x, labels_y)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Count output objects
    num_output = len(np.unique(combined_labels)) - (1 if 0 in combined_labels else 0)
    
    stats = CombineObjectsStats(
        slice_index=0,
        method=method.value,
        input_objects_x=num_objects_x,
        input_objects_y=num_objects_y,
        output_objects=num_output
    )
    
    # Return labels_x as the "image" output, plus stats and combined labels
    return labels_x.astype(np.float32), stats, combined_labels
