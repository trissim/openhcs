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
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.core.runtime_semantics import (
    ParentChildRelationshipPayload,
    aligned_dense_object_label_arrays,
)
from openhcs.processing.backends.cellprofiler._backend import CellProfilerBackendProvider
from openhcs.processing.backends.cellprofiler.relationships import (
    ObjectRelationshipBackendStrategy,
)
from openhcs.processing.materialization import csv_materializer
from ._enum import _coerce_function_enum


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
    calculate_distances: DistanceMethod | str = DistanceMethod.BOTH,
    calculate_per_parent_means: bool = False,
    save_children_with_parents: bool = False,
    relationship_backend_provider: CellProfilerBackendProvider | None = None,
) -> Tuple[
    np.ndarray,
    ParentChildRelationshipPayload,
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
    parent_labels, child_labels = aligned_dense_object_label_arrays(
        parent_labels,
        child_labels,
    )
    calculate_distances = _coerce_function_enum(
        DistanceMethod,
        calculate_distances,
    )
    
    # Get object counts
    parent_count = int(parent_labels.max()) if parent_labels.max() > 0 else 0
    child_count = int(child_labels.max()) if child_labels.max() > 0 else 0
    relationship_backend = ObjectRelationshipBackendStrategy.for_memory_type(
        backend_provider=relationship_backend_provider,
    )
    
    # Relate children to parents based on maximum overlap
    parents_of = relationship_backend.relate_children_to_parents(
        parent_labels,
        child_labels,
        child_count,
    )
    
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
        centroid_distances = relationship_backend.centroid_distances(
            parent_labels, child_labels, parents_of
        )
        valid_dists = centroid_distances[~np.isnan(centroid_distances)]
        mean_centroid_dist = float(np.mean(valid_dists)) if len(valid_dists) > 0 else np.nan
    
    if calculate_distances in (DistanceMethod.MINIMUM, DistanceMethod.BOTH):
        minimum_distances = relationship_backend.minimum_distances(
            parent_labels, child_labels, parents_of
        )
        valid_dists = minimum_distances[~np.isnan(minimum_distances)]
        mean_minimum_dist = float(np.mean(valid_dists)) if len(valid_dists) > 0 else np.nan
    
    # Create output: child labels colored by parent assignment
    output_labels = np.zeros_like(child_labels)
    if save_children_with_parents:
        keep_child = np.zeros(child_count + 1, dtype=bool)
        keep_child[1:] = parents_of > 0
        child_index = np.asarray(child_labels, dtype=np.intp)
        output_labels = np.where(keep_child[child_index], child_labels, 0)
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
        ParentChildRelationshipPayload(
            parent_ids=related_parent_ids,
            child_ids=related_child_ids,
        ),
        measurements,
    )
