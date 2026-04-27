"""Converted from CellProfiler: IdentifyTertiaryObjects

Identifies tertiary objects (e.g., cytoplasm) by removing smaller primary
objects (e.g., nuclei) from larger secondary objects (e.g., cells),
leaving a ring shape.
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass
from openhcs.core.memory import numpy


@dataclass
class TertiaryObjectStats:
    slice_index: int
    object_count: int
    mean_area: float
    primary_parent_count: int
    secondary_parent_count: int


def _outline(labels: np.ndarray) -> np.ndarray:
    """Find outline pixels of labeled objects.

    An outline pixel is a labeled pixel that has at least one neighbor
    with a different label (including background).
    """
    from scipy.ndimage import maximum_filter, minimum_filter
    
    # A pixel is on the outline if the max in its neighborhood differs from min
    max_labels = maximum_filter(labels, size=3, mode='constant', cval=0)
    min_labels = minimum_filter(labels, size=3, mode='constant', cval=0)
    
    outline_mask = (max_labels != min_labels) & (labels > 0)
    result = np.zeros_like(labels)
    result[outline_mask] = labels[outline_mask]
    return result


@numpy
def identify_tertiary_objects(
    image: np.ndarray,
    primary_labels: np.ndarray,
    secondary_labels: np.ndarray,
    shrink_primary: bool = True,
) -> Tuple[np.ndarray, TertiaryObjectStats, np.ndarray]:
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
        - TertiaryObjectStats dataclass with measurements
        - Tertiary label image (ring-shaped objects)

    CellProfiler Parameter Mapping:
    (CellProfiler setting -> Python parameter)
        'Select the larger identified objects' -> (pipeline-handled)
        'Select the smaller identified objects' -> (pipeline-handled)
        'Name the tertiary objects to be identified' -> (pipeline-handled)
        'Shrink smaller object prior to subtraction?' -> shrink_primary
    """
    from skimage.measure import regionprops
    
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
    primary_outline = _outline(primary_labels)
    
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
    
    # Compute measurements
    props = regionprops(tertiary_labels.astype(np.int32))
    object_count = len(props)
    mean_area = np.mean([p.area for p in props]) if props else 0.0
    
    # Count unique parent objects
    primary_parent_count = len(np.unique(primary_labels)) - (1 if 0 in primary_labels else 0)
    secondary_parent_count = len(np.unique(secondary_labels)) - (1 if 0 in secondary_labels else 0)
    
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
    
    return image, stats, tertiary_labels_out
