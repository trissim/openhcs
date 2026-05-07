"""
Converted from CellProfiler: ErodeObjects
Original: erode_objects
"""

import numpy as np
from typing import Tuple
from openhcs.core.memory.decorators import numpy
from openhcs.core.runtime_values import object_label_dense_array
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.processing.materialization import csv_materializer, segmentation_mask_rois
from dataclasses import dataclass

from benchmark.cellprofiler_library.functions.spatial_axes import (
    apply_over_trailing_spatial_axes,
)
from benchmark.cellprofiler_library.functions.structuring_elements import (
    StructuringElement,
    adapt_structuring_element_rank,
    build_structuring_element,
)


@dataclass
class ErosionStats:
    slice_index: int
    input_object_count: int
    output_object_count: int
    objects_removed: int


def _find_object_centers(labels: np.ndarray) -> dict:
    """Find the center pixel for each labeled object."""
    from scipy.ndimage import center_of_mass
    
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]
    
    centers = {}
    for label_id in unique_labels:
        mask = labels == label_id
        coords = np.argwhere(mask)
        if len(coords) > 0:
            # Use centroid, rounded to nearest pixel
            center = coords.mean(axis=0).astype(int)
            # Ensure center is within the object
            if not mask[tuple(center)]:
                # Find closest pixel in object to centroid
                distances = np.sum((coords - center) ** 2, axis=1)
                center = coords[np.argmin(distances)]
            centers[label_id] = tuple(center)
    
    return centers


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(
    ("erosion_stats", csv_materializer(
        fields=["slice_index", "input_object_count", "output_object_count", "objects_removed"],
        analysis_type="erosion"
    )),
    ("eroded_labels", segmentation_mask_rois())
)
def erode_objects(
    image: np.ndarray,
    labels: np.ndarray,
    structuring_element: StructuringElement | str = StructuringElement.DISK,
    size: int = 1,
    preserve_midpoints: bool = False,
    relabel_objects: bool = False,
) -> Tuple[np.ndarray, ErosionStats, np.ndarray]:
    """Erode objects based on the structuring element provided.
    
    This function erodes labeled objects using morphological erosion.
    Objects smaller than the structuring element will be removed entirely
    unless preserve_midpoints is enabled.
    
    Args:
        image: Input intensity image (passed through unchanged)
        labels: Input labeled objects array
        structuring_element: Shape of structuring element
        size: Size/radius of structuring element
        preserve_midpoints: If True, central pixels for each object will not be eroded
        relabel_objects: If True, resulting objects will be relabeled sequentially
        
    Returns:
        Tuple of (image, erosion_stats, eroded_labels)
    """
    from skimage.measure import label as relabel
    labels = object_label_dense_array(labels, dtype=np.int32)
    
    # Get structuring element
    selem = adapt_structuring_element_rank(
        build_structuring_element(structuring_element, size),
        labels.ndim,
    )
    
    # Count input objects
    input_labels = np.unique(labels)
    input_labels = input_labels[input_labels != 0]
    input_count = len(input_labels)
    
    # Store centers if preserving midpoints
    if preserve_midpoints:
        centers = _find_object_centers(labels)
    
    # Erode each object individually to maintain label identity
    eroded = np.zeros_like(labels)
    
    for label_id in input_labels:
        mask = labels == label_id
        eroded_mask = _binary_erosion_over_spatial_axes(mask, selem)
        
        # Preserve midpoint if requested and object was eroded away
        if preserve_midpoints and not eroded_mask.any() and label_id in centers:
            center = centers[label_id]
            eroded_mask = np.zeros_like(mask)
            eroded_mask[center] = True
        
        eroded[eroded_mask] = label_id
    
    # Relabel if requested
    if relabel_objects:
        eroded = relabel(eroded > 0).astype(labels.dtype)
    
    # Count output objects
    output_labels = np.unique(eroded)
    output_labels = output_labels[output_labels != 0]
    output_count = len(output_labels)
    
    stats = ErosionStats(
        slice_index=0,
        input_object_count=input_count,
        output_object_count=output_count,
        objects_removed=input_count - output_count
    )
    
    return image, stats, eroded


def _binary_erosion_over_spatial_axes(
    mask: np.ndarray,
    structure: np.ndarray,
) -> np.ndarray:
    from scipy.ndimage import binary_erosion

    return apply_over_trailing_spatial_axes(
        mask,
        structure.ndim,
        lambda spatial_mask: binary_erosion(spatial_mask, structure=structure),
        fill_value=False,
    )
