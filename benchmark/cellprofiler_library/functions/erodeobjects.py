"""
Converted from CellProfiler: ErodeObjects
Original: erode_objects
"""

import numpy as np
from typing import Tuple
from enum import Enum
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.processing.materialization import csv_materializer
from openhcs.processing.backends.analysis.cell_counting_cpu import materialize_segmentation_masks
from dataclasses import dataclass


class StructuringElementShape(Enum):
    DISK = "disk"
    SQUARE = "square"
    DIAMOND = "diamond"
    OCTAGON = "octagon"
    STAR = "star"


@dataclass
class ErosionStats:
    slice_index: int
    input_object_count: int
    output_object_count: int
    objects_removed: int


def _get_structuring_element_2d(shape: StructuringElementShape, size: int) -> np.ndarray:
    """Generate a 2D structuring element."""
    from skimage.morphology import disk, square, diamond, octagon, star
    
    if shape == StructuringElementShape.DISK:
        return disk(size)
    elif shape == StructuringElementShape.SQUARE:
        return square(size * 2 + 1)
    elif shape == StructuringElementShape.DIAMOND:
        return diamond(size)
    elif shape == StructuringElementShape.OCTAGON:
        return octagon(size, size)
    elif shape == StructuringElementShape.STAR:
        return star(size)
    else:
        return disk(size)


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
    ("eroded_labels", materialize_segmentation_masks)
)
def erode_objects(
    image: np.ndarray,
    labels: np.ndarray,
    structuring_element_shape: StructuringElementShape = StructuringElementShape.DISK,
    structuring_element_size: int = 1,
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
        structuring_element_shape: Shape of structuring element
        structuring_element_size: Size/radius of structuring element
        preserve_midpoints: If True, central pixels for each object will not be eroded
        relabel_objects: If True, resulting objects will be relabeled sequentially
        
    Returns:
        Tuple of (image, erosion_stats, eroded_labels)
    """
    from scipy.ndimage import binary_erosion
    from skimage.measure import label as relabel
    
    # Get structuring element
    selem = _get_structuring_element_2d(structuring_element_shape, structuring_element_size)
    
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
        eroded_mask = binary_erosion(mask, structure=selem)
        
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