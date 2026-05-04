"""Converted from CellProfiler: IdentifyTertiaryObjects

Identifies tertiary objects (e.g., cytoplasm) by removing smaller primary
objects (e.g., nuclei) from larger secondary objects (e.g., cells),
leaving a ring shape.
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass
from openhcs.core.memory import numpy
from openhcs.core.runtime_semantics import (
    ParentChildRelationshipPayload,
    object_label_parent_child_payload,
)
from openhcs.processing.backends.cellprofiler._backend import CellProfilerBackendProvider
from openhcs.processing.backends.cellprofiler.outlines import ObjectOutlineBackendStrategy


@dataclass
class TertiaryObjectStats:
    slice_index: int
    object_count: int
    mean_area: float
    primary_parent_count: int
    secondary_parent_count: int


def _outline(
    labels: np.ndarray,
    *,
    outline_backend_provider: CellProfilerBackendProvider | None,
) -> np.ndarray:
    """Find outline pixels of labeled objects.

    An outline pixel is a labeled pixel that has at least one neighbor
    with a different label (including background).
    """
    return ObjectOutlineBackendStrategy.for_memory_type(
        backend_provider=outline_backend_provider,
    ).outline(labels)


def _positive_label_count(labels: np.ndarray) -> int:
    return int(np.count_nonzero(np.bincount(np.asarray(labels).ravel())[1:]))


def _positive_label_mean_area(labels: np.ndarray) -> tuple[int, float]:
    areas = np.bincount(np.asarray(labels).ravel())[1:]
    positive_areas = areas[areas > 0]
    if positive_areas.size == 0:
        return 0, 0.0
    return int(positive_areas.size), float(np.mean(positive_areas))


def _parent_child_relationship(
    parent_labels: np.ndarray,
    child_labels: np.ndarray,
    *,
    parent_context_labels: np.ndarray | None = None,
) -> ParentChildRelationshipPayload:
    return object_label_parent_child_payload(
        parent_labels,
        child_labels,
        child_region_labels=parent_context_labels,
    )


@numpy
def identify_tertiary_objects(
    image: np.ndarray,
    primary_labels: np.ndarray,
    secondary_labels: np.ndarray,
    shrink_primary: bool = True,
    outline_backend_provider: CellProfilerBackendProvider | None = None,
) -> Tuple[
    np.ndarray,
    ParentChildRelationshipPayload,
    ParentChildRelationshipPayload,
    TertiaryObjectStats,
    np.ndarray,
]:
    """
    Identify tertiary objects by subtracting primary objects from secondary objects.
    
    Creates ring-shaped objects (e.g., cytoplasm) by removing smaller objects
    (e.g., nuclei) from larger objects (e.g., cells).
    
    Args:
        image: Input image, shape (D, H, W) - used as reference, passed through
        primary_labels: Label image of smaller objects (e.g., nuclei), shape (H, W)
        secondary_labels: Label image of larger objects (e.g., cells), shape (H, W)
        shrink_primary: If True, shrink primary objects by 1 pixel before subtraction
                       to ensure tertiary objects always have some area
    
    Returns:
        Tuple of:
        - Original image (passed through)
        - Secondary-parent to tertiary-child relationships
        - Primary-parent to tertiary-child relationships
        - TertiaryObjectStats dataclass with measurements
        - Tertiary label image (ring-shaped objects)

    CellProfiler Parameter Mapping:
    (CellProfiler setting -> Python parameter)
        'Select the larger identified objects' -> (pipeline-handled)
        'Select the smaller identified objects' -> (pipeline-handled)
        'Name the tertiary objects to be identified' -> (pipeline-handled)
        'Shrink smaller object prior to subtraction?' -> shrink_primary
    """
    # Handle 3D input - process slice by slice or take first slice
    if image.ndim == 3:
        # For FLEXIBLE contract, we process the first slice as reference
        ref_image = image[0]
    else:
        ref_image = image
    
    # Ensure labels are 2D
    if primary_labels.ndim == 3:
        primary_labels = primary_labels[0]
    if secondary_labels.ndim == 3:
        secondary_labels = secondary_labels[0]
    
    # Ensure shapes match
    if primary_labels.shape != secondary_labels.shape:
        raise ValueError(
            f"Primary and secondary label shapes must match. "
            f"Got {primary_labels.shape} vs {secondary_labels.shape}"
        )
    
    # Find outlines of primary objects
    primary_outline = _outline(
        primary_labels,
        outline_backend_provider=outline_backend_provider,
    )
    
    # Create tertiary labels by subtracting primary from secondary
    tertiary_labels = secondary_labels.copy()
    
    if shrink_primary:
        # Keep pixels that are either background OR on the outline of primary
        # This shrinks primary objects by 1 pixel
        primary_mask = np.logical_or(primary_labels == 0, primary_outline > 0)
    else:
        # Only keep pixels where primary is background
        primary_mask = primary_labels == 0
    
    # Remove primary object pixels from tertiary
    tertiary_labels[~primary_mask] = 0
    
    # Check for labels that were completely removed and restore a single pixel
    secondary_unique_labels, secondary_unique_indices = np.unique(
        secondary_labels, return_index=True
    )
    tertiary_unique_labels = np.unique(tertiary_labels)
    missing_labels = np.setdiff1d(secondary_unique_labels, tertiary_unique_labels)
    
    for missing_label in missing_labels:
        if missing_label == 0:
            continue
        # Add a single pixel to preserve the object
        idx = np.where(secondary_unique_labels == missing_label)[0][0]
        first_row, first_col = np.unravel_index(
            secondary_unique_indices[idx], secondary_labels.shape
        )
        tertiary_labels[first_row, first_col] = missing_label
    
    object_count, mean_area = _positive_label_mean_area(tertiary_labels)
    
    # Count unique parent objects
    primary_parent_count = _positive_label_count(primary_labels)
    secondary_parent_count = _positive_label_count(secondary_labels)
    
    stats = TertiaryObjectStats(
        slice_index=0,
        object_count=object_count,
        mean_area=float(mean_area),
        primary_parent_count=int(primary_parent_count),
        secondary_parent_count=int(secondary_parent_count)
    )
    
    # Ensure output has correct shape (D, H, W)
    if image.ndim == 3:
        tertiary_labels_out = np.expand_dims(tertiary_labels, axis=0)
    else:
        tertiary_labels_out = tertiary_labels
    
    return (
        image,
        _parent_child_relationship(secondary_labels, tertiary_labels),
        _parent_child_relationship(
            primary_labels,
            tertiary_labels,
            parent_context_labels=secondary_labels,
        ),
        stats,
        tertiary_labels_out,
    )
