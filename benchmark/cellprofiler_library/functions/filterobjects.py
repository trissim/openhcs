"""
Converted from CellProfiler: FilterObjects
Original: FilterObjects module

FilterObjects eliminates objects based on their measurements (e.g., area, shape,
texture, intensity) or removes objects touching the image border.
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.processing.materialization import csv_materializer
from openhcs.processing.backends.analysis.cell_counting_cpu import materialize_segmentation_masks


class FilterMethod(Enum):
    MINIMAL = "minimal"
    MAXIMAL = "maximal"
    MINIMAL_PER_OBJECT = "minimal_per_object"
    MAXIMAL_PER_OBJECT = "maximal_per_object"
    LIMITS = "limits"


class FilterMode(Enum):
    MEASUREMENTS = "measurements"
    BORDER = "border"


@dataclass
class FilterObjectsStats:
    slice_index: int
    objects_pre_filter: int
    objects_post_filter: int
    objects_removed: int


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(
    ("filter_stats", csv_materializer(
        fields=["slice_index", "objects_pre_filter", "objects_post_filter", "objects_removed"],
        analysis_type="filter_objects"
    )),
    ("filtered_labels", materialize_segmentation_masks)
)
def filter_objects(
    image: np.ndarray,
    labels: np.ndarray,
    mode: FilterMode = FilterMode.MEASUREMENTS,
    filter_method: FilterMethod = FilterMethod.LIMITS,
    measurement_values: Optional[np.ndarray] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    use_minimum: bool = True,
    use_maximum: bool = True,
) -> Tuple[np.ndarray, FilterObjectsStats, np.ndarray]:
    """
    Filter objects based on measurements or border touching.
    
    Args:
        image: Input intensity image (H, W)
        labels: Label image with segmented objects (H, W)
        mode: Filtering mode - MEASUREMENTS or BORDER
        filter_method: Method for measurement-based filtering
        measurement_values: Array of measurement values per object (indexed by label-1)
        min_value: Minimum threshold for LIMITS method
        max_value: Maximum threshold for LIMITS method
        use_minimum: Whether to apply minimum threshold
        use_maximum: Whether to apply maximum threshold
    
    Returns:
        Tuple of (image, stats, filtered_labels)
    """
    from scipy import ndimage as ndi
    from skimage.measure import regionprops
    
    labels = labels.astype(np.int32)
    max_label = labels.max()
    
    if max_label == 0:
        # No objects to filter
        stats = FilterObjectsStats(
            slice_index=0,
            objects_pre_filter=0,
            objects_post_filter=0,
            objects_removed=0
        )
        return image, stats, labels
    
    # Get all unique labels (excluding background)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]
    num_objects_pre = len(unique_labels)
    
    if mode == FilterMode.BORDER:
        # Remove objects touching the border
        indexes_to_keep = _discard_border_objects(labels)
    elif mode == FilterMode.MEASUREMENTS:
        if measurement_values is None:
            # If no measurements provided, compute area as default
            props = regionprops(labels)
            measurement_values = np.array([p.area for p in props])
        
        if filter_method == FilterMethod.LIMITS:
            indexes_to_keep = _keep_within_limits(
                measurement_values, 
                min_value, 
                max_value,
                use_minimum,
                use_maximum
            )
        elif filter_method == FilterMethod.MINIMAL:
            indexes_to_keep = _keep_one(measurement_values, keep_max=False)
        elif filter_method == FilterMethod.MAXIMAL:
            indexes_to_keep = _keep_one(measurement_values, keep_max=True)
        else:
            # Default to keeping all
            indexes_to_keep = list(range(1, num_objects_pre + 1))
    else:
        indexes_to_keep = list(range(1, num_objects_pre + 1))
    
    # Create new label image with only kept objects
    new_object_count = len(indexes_to_keep)
    label_mapping = np.zeros(max_label + 1, dtype=np.int32)
    for new_idx, old_idx in enumerate(indexes_to_keep, start=1):
        if old_idx <= max_label:
            label_mapping[old_idx] = new_idx
    
    filtered_labels = label_mapping[labels]
    
    stats = FilterObjectsStats(
        slice_index=0,
        objects_pre_filter=num_objects_pre,
        objects_post_filter=new_object_count,
        objects_removed=num_objects_pre - new_object_count
    )
    
    return image, stats, filtered_labels


def _discard_border_objects(labels: np.ndarray) -> List[int]:
    """
    Return indices of objects not touching the image border.
    
    Args:
        labels: Label image
    
    Returns:
        List of label indices to keep
    """
    from scipy import ndimage as ndi
    
    # Create interior mask (erode by 1 pixel)
    interior_pixels = ndi.binary_erosion(np.ones_like(labels, dtype=bool))
    border_pixels = ~interior_pixels
    
    # Find labels touching the border
    border_labels = set(labels[border_pixels])
    
    # Get all labels and remove border-touching ones
    all_labels = set(labels.ravel())
    keep_labels = list(all_labels.difference(border_labels))
    
    # Remove background (0) if present
    if 0 in keep_labels:
        keep_labels.remove(0)
    
    keep_labels.sort()
    return keep_labels


def _keep_within_limits(
    values: np.ndarray,
    min_value: Optional[float],
    max_value: Optional[float],
    use_minimum: bool,
    use_maximum: bool
) -> List[int]:
    """
    Keep objects whose measurements fall within specified limits.
    
    Args:
        values: Measurement values per object (0-indexed)
        min_value: Minimum threshold
        max_value: Maximum threshold
        use_minimum: Whether to apply minimum threshold
        use_maximum: Whether to apply maximum threshold
    
    Returns:
        List of label indices (1-indexed) to keep
    """
    if len(values) == 0:
        return []
    
    hits = np.ones(len(values), dtype=bool)
    
    if use_minimum and min_value is not None:
        hits[values < min_value] = False
    
    if use_maximum and max_value is not None:
        hits[values > max_value] = False
    
    # Convert to 1-indexed labels
    indexes = np.argwhere(hits).flatten() + 1
    return indexes.tolist()


def _keep_one(values: np.ndarray, keep_max: bool = True) -> List[int]:
    """
    Keep only the object with the maximum or minimum measurement value.
    
    Args:
        values: Measurement values per object (0-indexed)
        keep_max: If True, keep maximum; if False, keep minimum
    
    Returns:
        List containing single label index (1-indexed) to keep
    """
    if len(values) == 0:
        return []
    
    if keep_max:
        best_idx = np.argmax(values) + 1
    else:
        best_idx = np.argmin(values) + 1
    
    return [int(best_idx)]


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(
    ("filter_stats", csv_materializer(
        fields=["slice_index", "objects_pre_filter", "objects_post_filter", "objects_removed"],
        analysis_type="filter_objects"
    )),
    ("filtered_labels", materialize_segmentation_masks)
)
def filter_objects_by_size(
    image: np.ndarray,
    labels: np.ndarray,
    min_area: float = 0.0,
    max_area: float = float('inf'),
    use_minimum: bool = True,
    use_maximum: bool = True,
) -> Tuple[np.ndarray, FilterObjectsStats, np.ndarray]:
    """
    Filter objects based on area measurements.
    
    This is a convenience function that computes area internally.
    
    Args:
        image: Input intensity image (H, W)
        labels: Label image with segmented objects (H, W)
        min_area: Minimum area threshold in pixels
        max_area: Maximum area threshold in pixels
        use_minimum: Whether to apply minimum threshold
        use_maximum: Whether to apply maximum threshold
    
    Returns:
        Tuple of (image, stats, filtered_labels)
    """
    from skimage.measure import regionprops
    
    labels = labels.astype(np.int32)
    max_label = labels.max()
    
    if max_label == 0:
        stats = FilterObjectsStats(
            slice_index=0,
            objects_pre_filter=0,
            objects_post_filter=0,
            objects_removed=0
        )
        return image, stats, labels
    
    # Compute area for each object
    props = regionprops(labels)
    areas = np.array([p.area for p in props])
    num_objects_pre = len(props)
    
    # Filter by area limits
    indexes_to_keep = _keep_within_limits(
        areas,
        min_area,
        max_area,
        use_minimum,
        use_maximum
    )
    
    # Create new label image
    new_object_count = len(indexes_to_keep)
    label_mapping = np.zeros(max_label + 1, dtype=np.int32)
    for new_idx, old_idx in enumerate(indexes_to_keep, start=1):
        if old_idx <= max_label:
            label_mapping[old_idx] = new_idx
    
    filtered_labels = label_mapping[labels]
    
    stats = FilterObjectsStats(
        slice_index=0,
        objects_pre_filter=num_objects_pre,
        objects_post_filter=new_object_count,
        objects_removed=num_objects_pre - new_object_count
    )
    
    return image, stats, filtered_labels


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(
    ("filter_stats", csv_materializer(
        fields=["slice_index", "objects_pre_filter", "objects_post_filter", "objects_removed"],
        analysis_type="filter_objects"
    )),
    ("filtered_labels", materialize_segmentation_masks)
)
def filter_border_objects(
    image: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, FilterObjectsStats, np.ndarray]:
    """
    Remove objects touching the image border.
    
    Args:
        image: Input intensity image (H, W)
        labels: Label image with segmented objects (H, W)
    
    Returns:
        Tuple of (image, stats, filtered_labels)
    """
    labels = labels.astype(np.int32)
    max_label = labels.max()
    
    if max_label == 0:
        stats = FilterObjectsStats(
            slice_index=0,
            objects_pre_filter=0,
            objects_post_filter=0,
            objects_removed=0
        )
        return image, stats, labels
    
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]
    num_objects_pre = len(unique_labels)
    
    indexes_to_keep = _discard_border_objects(labels)
    
    # Create new label image
    new_object_count = len(indexes_to_keep)
    label_mapping = np.zeros(max_label + 1, dtype=np.int32)
    for new_idx, old_idx in enumerate(indexes_to_keep, start=1):
        if old_idx <= max_label:
            label_mapping[old_idx] = new_idx
    
    filtered_labels = label_mapping[labels]
    
    stats = FilterObjectsStats(
        slice_index=0,
        objects_pre_filter=num_objects_pre,
        objects_post_filter=new_object_count,
        objects_removed=num_objects_pre - new_object_count
    )
    
    return image, stats, filtered_labels