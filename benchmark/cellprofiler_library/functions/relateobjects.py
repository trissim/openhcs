"""
Converted from CellProfiler: RelateObjects
Original: RelateObjects module

Assigns relationships between parent and child objects.
All objects (e.g., speckles) within a parent object (e.g., nucleus) become its children.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import scipy.ndimage
import skimage.segmentation
from benchmark.cellprofiler_compat.relationship_payload import (
    CellProfilerRelationshipPayload,
)
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.processing.materialization import csv_materializer


class DistanceMethod(Enum):
    NONE = "none"
    CENTROID = "centroid"
    MINIMUM = "minimum"
    BOTH = "both"


@dataclass
class RelationshipMeasurements:
    """Measurements from relating parent and child objects."""
    slice_index: int
    parent_object_count: int
    child_object_count: int
    children_with_parents_count: int
    mean_children_per_parent: float
    mean_centroid_distance: float
    mean_minimum_distance: float


@numpy
@special_inputs("parent_labels", "child_labels")
@special_outputs(
    ("relationship_measurements", csv_materializer(
        fields=["slice_index", "parent_object_count", "child_object_count",
                "children_with_parents_count", "mean_children_per_parent",
                "mean_centroid_distance", "mean_minimum_distance"],
        analysis_type="relate_objects"
    ))
)
def relate_objects(
    image: np.ndarray,
    parent_labels: np.ndarray,
    child_labels: np.ndarray,
    calculate_distances: DistanceMethod = DistanceMethod.BOTH,
    calculate_per_parent_means: bool = False,
    save_children_with_parents: bool = False,
) -> Tuple[
    np.ndarray,
    CellProfilerRelationshipPayload,
    RelationshipMeasurements,
]:
    """
    Relate child objects to parent objects based on spatial overlap.
    
    Args:
        image: Main OpenHCS image payload (passed through unchanged for flow).
        parent_labels: Parent object labels (H, W)
        child_labels: Child object labels (H, W)
        calculate_distances: Method for calculating child-parent distances
        calculate_per_parent_means: Whether to calculate mean measurements per parent
        save_children_with_parents: Whether to output only children that have parents
    
    Returns:
        Tuple of:
        - child_labels with parent assignments encoded (H, W)
        - RelationshipMeasurements dataclass
    """
    parent_labels = parent_labels.astype(np.int32)
    child_labels = child_labels.astype(np.int32)
    
    # Get object counts
    parent_count = int(parent_labels.max()) if parent_labels.max() > 0 else 0
    child_count = int(child_labels.max()) if child_labels.max() > 0 else 0
    
    # Relate children to parents based on maximum overlap
    parents_of = _relate_children_to_parents(parent_labels, child_labels, child_count)
    
    # Count children per parent
    child_counts_per_parent = np.zeros(parent_count, dtype=np.int32)
    for parent_idx in parents_of:
        if parent_idx > 0 and parent_idx <= parent_count:
            child_counts_per_parent[parent_idx - 1] += 1
    
    children_with_parents = np.sum(parents_of > 0)
    mean_children = np.mean(child_counts_per_parent) if parent_count > 0 else 0.0
    
    # Calculate distances if requested
    mean_centroid_dist = np.nan
    mean_minimum_dist = np.nan
    
    if calculate_distances in (DistanceMethod.CENTROID, DistanceMethod.BOTH):
        centroid_distances = _calculate_centroid_distances(
            parent_labels, child_labels, parents_of
        )
        valid_dists = centroid_distances[~np.isnan(centroid_distances)]
        mean_centroid_dist = float(np.mean(valid_dists)) if len(valid_dists) > 0 else np.nan
    
    if calculate_distances in (DistanceMethod.MINIMUM, DistanceMethod.BOTH):
        minimum_distances = _calculate_minimum_distances(
            parent_labels, child_labels, parents_of
        )
        valid_dists = minimum_distances[~np.isnan(minimum_distances)]
        mean_minimum_dist = float(np.mean(valid_dists)) if len(valid_dists) > 0 else np.nan
    
    # Create output: child labels colored by parent assignment
    output_labels = np.zeros_like(child_labels)
    if save_children_with_parents:
        # Only keep children that have parents
        for child_idx in range(1, child_count + 1):
            if parents_of[child_idx - 1] > 0:
                output_labels[child_labels == child_idx] = child_idx
    else:
        # Keep all children, encode parent relationship
        output_labels = child_labels.copy()
    
    measurements = RelationshipMeasurements(
        slice_index=0,
        parent_object_count=parent_count,
        child_object_count=child_count,
        children_with_parents_count=int(children_with_parents),
        mean_children_per_parent=float(mean_children),
        mean_centroid_distance=mean_centroid_dist,
        mean_minimum_distance=mean_minimum_dist
    )

    related_child_ids = tuple(
        child_idx
        for child_idx, parent_idx in enumerate(parents_of, start=1)
        if parent_idx > 0
    )
    related_parent_ids = tuple(
        int(parent_idx)
        for parent_idx in parents_of
        if parent_idx > 0
    )

    return (
        output_labels.astype(np.float32),
        CellProfilerRelationshipPayload(
            parent_ids=related_parent_ids,
            child_ids=related_child_ids,
        ),
        measurements,
    )


def _relate_children_to_parents(
    parent_labels: np.ndarray,
    child_labels: np.ndarray,
    child_count: int
) -> np.ndarray:
    """
    Determine parent for each child based on maximum overlap.
    
    Returns:
        Array of length child_count with parent label for each child (0 if no parent)
    """
    parents_of = np.zeros(child_count, dtype=np.int32)
    
    if child_count == 0:
        return parents_of
    
    for child_idx in range(1, child_count + 1):
        child_mask = child_labels == child_idx
        overlapping_parents = parent_labels[child_mask]
        overlapping_parents = overlapping_parents[overlapping_parents > 0]
        
        if len(overlapping_parents) > 0:
            # Assign to parent with maximum overlap
            unique, counts = np.unique(overlapping_parents, return_counts=True)
            parents_of[child_idx - 1] = unique[np.argmax(counts)]
    
    return parents_of


def _calculate_centroid_distances(
    parent_labels: np.ndarray,
    child_labels: np.ndarray,
    parents_of: np.ndarray
) -> np.ndarray:
    """
    Calculate centroid-to-centroid distances between children and their parents.
    """
    child_count = len(parents_of)
    distances = np.full(child_count, np.nan)
    
    if child_count == 0:
        return distances
    
    # Get parent centroids
    parent_count = int(parent_labels.max())
    if parent_count == 0:
        return distances
    
    parent_centroids = scipy.ndimage.center_of_mass(
        np.ones_like(parent_labels),
        parent_labels,
        range(1, parent_count + 1)
    )
    parent_centroids = np.array(parent_centroids)
    
    # Get child centroids
    child_centroids = scipy.ndimage.center_of_mass(
        np.ones_like(child_labels),
        child_labels,
        range(1, child_count + 1)
    )
    child_centroids = np.array(child_centroids)
    
    # Calculate distances
    for child_idx in range(child_count):
        parent_idx = parents_of[child_idx]
        if parent_idx > 0 and parent_idx <= parent_count:
            child_center = child_centroids[child_idx]
            parent_center = parent_centroids[parent_idx - 1]
            distances[child_idx] = np.sqrt(np.sum((child_center - parent_center) ** 2))
    
    return distances


def _calculate_minimum_distances(
    parent_labels: np.ndarray,
    child_labels: np.ndarray,
    parents_of: np.ndarray
) -> np.ndarray:
    """
    Calculate minimum distances from child centroids to parent perimeters.
    """
    child_count = len(parents_of)
    distances = np.full(child_count, np.nan)
    
    if child_count == 0:
        return distances
    
    parent_count = int(parent_labels.max())
    if parent_count == 0:
        return distances
    
    # Get child centroids
    child_centroids = scipy.ndimage.center_of_mass(
        np.ones_like(child_labels),
        child_labels,
        range(1, child_count + 1)
    )
    child_centroids = np.array(child_centroids)
    
    # Find parent perimeters
    parent_perimeter = (
        skimage.segmentation.find_boundaries(parent_labels, mode='inner') *
        parent_labels
    )
    
    # Calculate minimum distance for each child
    for child_idx in range(child_count):
        parent_idx = parents_of[child_idx]
        if parent_idx > 0 and parent_idx <= parent_count:
            child_center = child_centroids[child_idx]
            
            # Get perimeter points of this parent
            perim_points = np.argwhere(parent_perimeter == parent_idx)
            
            if len(perim_points) > 0:
                # Calculate distance to all perimeter points
                dists = np.sqrt(np.sum((perim_points - child_center) ** 2, axis=1))
                distances[child_idx] = np.min(dists)
    
    return distances
