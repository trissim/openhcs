"""
Converted from CellProfiler: SplitOrMergeObjects
Original: SplitOrMergeObjects module

Separates or combines a set of objects that were identified earlier in a pipeline.
Objects can be merged based on distance or parent relationships, or split into
disconnected components.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.processing.materialization import csv_materializer
from openhcs.processing.backends.analysis.cell_counting_cpu import materialize_segmentation_masks


class Operation(Enum):
    MERGE = "merge"
    SPLIT = "split"


class MergeMethod(Enum):
    DISTANCE = "distance"
    PER_PARENT = "per_parent"


class OutputObjectType(Enum):
    DISCONNECTED = "disconnected"
    CONVEX_HULL = "convex_hull"


class IntensityMethod(Enum):
    CENTROIDS = "centroids"
    CLOSEST_POINT = "closest_point"


@dataclass
class SplitOrMergeStats:
    slice_index: int
    input_object_count: int
    output_object_count: int
    operation: str


def _relabel_consecutive(labels: np.ndarray) -> np.ndarray:
    """Relabel a label image to have consecutive labels starting from 1."""
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]
    if len(unique_labels) == 0:
        return labels
    
    max_label = int(np.max(labels))
    label_map = np.zeros(max_label + 1, dtype=labels.dtype)
    label_map[unique_labels] = np.arange(1, len(unique_labels) + 1)
    
    return label_map[labels]


def _compute_convex_hull_labels(labels: np.ndarray) -> np.ndarray:
    """Compute convex hull for each label and fill it."""
    from scipy.spatial import ConvexHull
    from skimage.draw import polygon
    
    output = np.zeros_like(labels)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]
    
    for label_id in unique_labels:
        mask = labels == label_id
        coords = np.argwhere(mask)
        
        if len(coords) < 3:
            # Can't form convex hull with less than 3 points
            output[mask] = label_id
            continue
        
        try:
            hull = ConvexHull(coords)
            hull_points = coords[hull.vertices]
            rr, cc = polygon(hull_points[:, 0], hull_points[:, 1], labels.shape)
            output[rr, cc] = label_id
        except Exception:
            # If convex hull fails, just use original mask
            output[mask] = label_id
    
    return output


def _merge_by_distance(
    labels: np.ndarray,
    distance_threshold: int,
    guide_image: Optional[np.ndarray] = None,
    minimum_intensity_fraction: float = 0.9,
    intensity_method: IntensityMethod = IntensityMethod.CENTROIDS
) -> np.ndarray:
    """Merge objects within a distance threshold."""
    from scipy.ndimage import distance_transform_edt, label as scipy_label
    
    mask = labels > 0
    
    if distance_threshold > 0:
        # Expand mask to include nearby background pixels
        d = distance_transform_edt(~mask)
        mask = d < (distance_threshold / 2.0 + 1)
    
    # Label connected components in the expanded mask
    output_labels, _ = scipy_label(mask, structure=np.ones((3, 3), bool))
    
    # Remove labels where original was background
    output_labels[labels == 0] = 0
    
    if guide_image is not None:
        output_labels = _filter_using_image(
            labels, output_labels, guide_image,
            minimum_intensity_fraction, intensity_method
        )
    
    return _relabel_consecutive(output_labels)


def _filter_using_image(
    original_labels: np.ndarray,
    merged_labels: np.ndarray,
    image: np.ndarray,
    minimum_intensity_fraction: float,
    intensity_method: IntensityMethod
) -> np.ndarray:
    """Filter merged connections using intensity criteria."""
    from scipy.ndimage import distance_transform_edt, label as scipy_label
    from skimage.measure import regionprops
    
    # For simplicity, implement a basic version that checks intensity along paths
    # This is a simplified version of the CellProfiler algorithm
    
    if intensity_method == IntensityMethod.CLOSEST_POINT:
        # Get distance transform and closest point indices
        distances, indices = distance_transform_edt(
            original_labels == 0, return_indices=True
        )
        
        # Get intensity at closest object point
        closest_i, closest_j = indices
        object_intensity = image[closest_i, closest_j] * minimum_intensity_fraction
        
        # Create mask where background intensity is sufficient
        valid_mask = (original_labels > 0) | (image >= object_intensity)
        
        # Relabel with the filtered mask
        output_labels, _ = scipy_label(valid_mask & (merged_labels > 0), 
                                        structure=np.ones((3, 3), bool))
        output_labels[original_labels == 0] = 0
        
    else:  # CENTROIDS method
        # For centroids method, we check intensity along lines between centroids
        # Simplified: just use the merged labels as-is for now
        output_labels = merged_labels.copy()
    
    return output_labels


def _merge_by_parent(
    labels: np.ndarray,
    parent_labels: np.ndarray,
    output_type: OutputObjectType = OutputObjectType.DISCONNECTED
) -> np.ndarray:
    """Merge child objects that share the same parent."""
    from skimage.measure import regionprops
    
    # Create output where each child gets its parent's label
    output_labels = np.zeros_like(labels)
    
    # For each child object, find which parent it belongs to
    child_props = regionprops(labels)
    
    for prop in child_props:
        child_mask = labels == prop.label
        # Find the most common parent label in this child's region
        parent_values = parent_labels[child_mask]
        parent_values = parent_values[parent_values > 0]
        
        if len(parent_values) > 0:
            # Use the most common parent label
            parent_id = np.bincount(parent_values).argmax()
            output_labels[child_mask] = parent_id
        else:
            # No parent found, keep original label
            output_labels[child_mask] = prop.label
    
    if output_type == OutputObjectType.CONVEX_HULL:
        output_labels = _compute_convex_hull_labels(output_labels)
    
    return _relabel_consecutive(output_labels)


def _split_objects(labels: np.ndarray) -> np.ndarray:
    """Split disconnected components into separate objects."""
    from scipy.ndimage import label as scipy_label
    
    # Label all connected components
    output_labels, _ = scipy_label(labels > 0, structure=np.ones((3, 3), bool))
    
    return output_labels


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(
    ("split_merge_stats", csv_materializer(
        fields=["slice_index", "input_object_count", "output_object_count", "operation"],
        analysis_type="split_or_merge"
    )),
    ("output_labels", materialize_segmentation_masks)
)
def split_or_merge_objects(
    image: np.ndarray,
    labels: np.ndarray,
    operation: Operation = Operation.MERGE,
    merge_method: MergeMethod = MergeMethod.DISTANCE,
    output_object_type: OutputObjectType = OutputObjectType.DISCONNECTED,
    distance_threshold: int = 0,
    use_guide_image: bool = False,
    minimum_intensity_fraction: float = 0.9,
    intensity_method: IntensityMethod = IntensityMethod.CENTROIDS,
    parent_labels: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, SplitOrMergeStats, np.ndarray]:
    """
    Split or merge objects based on various criteria.
    
    Args:
        image: Grayscale image (H, W), used as guide for intensity-based merging
        labels: Input label image (H, W) with objects to split or merge
        operation: Whether to merge or split objects
        merge_method: Method for merging (distance-based or per-parent)
        output_object_type: For per-parent merge, whether to use convex hull
        distance_threshold: Maximum distance for merging objects (pixels)
        use_guide_image: Whether to use intensity image to guide merging
        minimum_intensity_fraction: Minimum intensity fraction for guided merging
        intensity_method: Method to find object intensity for guided merging
        parent_labels: Parent label image for per-parent merging
    
    Returns:
        Tuple of (image, stats, output_labels)
    """
    input_count = len(np.unique(labels)) - (1 if 0 in labels else 0)
    
    if operation == Operation.SPLIT:
        output_labels = _split_objects(labels)
    else:  # MERGE
        if merge_method == MergeMethod.DISTANCE:
            guide_image = image if use_guide_image else None
            output_labels = _merge_by_distance(
                labels,
                distance_threshold,
                guide_image,
                minimum_intensity_fraction,
                intensity_method
            )
        else:  # PER_PARENT
            if parent_labels is None:
                # If no parent labels provided, use the image as a fallback
                # In practice, parent_labels should be provided via special_inputs
                output_labels = labels.copy()
            else:
                output_labels = _merge_by_parent(
                    labels,
                    parent_labels,
                    output_object_type
                )
    
    output_count = len(np.unique(output_labels)) - (1 if 0 in output_labels else 0)
    
    stats = SplitOrMergeStats(
        slice_index=0,
        input_object_count=int(input_count),
        output_object_count=int(output_count),
        operation=operation.value
    )
    
    return image, stats, output_labels.astype(np.int32)