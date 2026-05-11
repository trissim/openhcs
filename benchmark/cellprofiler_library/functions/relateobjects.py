"""
Converted from CellProfiler: RelateObjects
Original: RelateObjects module

Assigns relationships between parent and child objects.
All objects (e.g., speckles) within a parent object (e.g., nucleus) become its children.
"""

import numpy as np
from dataclasses import dataclass
from openhcs.interop.cellprofiler.relate_objects_settings import (
    RelateObjectsDistanceMethod as DistanceMethod,
)
from openhcs.interop.cellprofiler.relationship_measurements import (
    RelationshipMeasurements,
)
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import (
    special_inputs,
    special_outputs,
)
from openhcs.core.runtime_invocation import RuntimeOutputBundle
from openhcs.core.runtime_semantics import (
    ParentChildRelationshipPayload,
    aligned_dense_object_label_arrays,
)
from openhcs.core.runtime_values import object_label_dense_array
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
)
from openhcs.processing.backends.cellprofiler.relationships import (
    ObjectRelationshipBackendStrategy,
)
from openhcs.processing.materialization import csv_dataclass_materializer
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum


@dataclass(frozen=True, slots=True)
class RelateObjectsResult(RuntimeOutputBundle):
    """Nominal result bundle emitted by RelateObjects."""

    output_labels: np.ndarray
    parent_child_relationship: ParentChildRelationshipPayload
    relationship_measurements: RelationshipMeasurements
    saved_child_relationship: ParentChildRelationshipPayload | None = None

    def as_runtime_tuple(self) -> tuple[
        np.ndarray,
        ParentChildRelationshipPayload,
        RelationshipMeasurements,
    ] | tuple[
        np.ndarray,
        ParentChildRelationshipPayload,
        ParentChildRelationshipPayload,
        RelationshipMeasurements,
    ]:
        """Lower to the current positional function-contract ABI."""
        if self.saved_child_relationship is None:
            return (
                self.output_labels,
                self.parent_child_relationship,
                self.relationship_measurements,
            )
        return (
            self.output_labels,
            self.parent_child_relationship,
            self.saved_child_relationship,
            self.relationship_measurements,
        )

    def __iter__(self):
        """Preserve direct tuple-unpacking compatibility for function tests."""
        return iter(self.as_runtime_tuple())


@numpy
@special_inputs("parent_labels", "child_labels")
@special_outputs(
    (
        "relationship_measurements",
        csv_dataclass_materializer(
            RelationshipMeasurements,
            analysis_type="relate_objects",
        ),
    )
)
def relate_objects(
    image: np.ndarray,
    parent_labels: np.ndarray,
    child_labels: np.ndarray,
    calculate_distances: DistanceMethod | str = DistanceMethod.BOTH,
    calculate_per_parent_means: bool = False,
    save_children_with_parents: bool = False,
    relationship_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> RelateObjectsResult:
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
    raw_parent_labels = parent_labels
    raw_child_labels = child_labels
    calculate_distances = coerce_cellprofiler_enum(
        DistanceMethod,
        calculate_distances,
    )
    relationship_backend = ObjectRelationshipBackendStrategy.for_memory_type(
        backend_provider=relationship_backend_provider,
    )
    parent_child_relationship = relationship_backend.parent_child_payload_from_labels(
        raw_parent_labels,
        raw_child_labels,
    )

    parent_labels = object_label_dense_array(raw_parent_labels, dtype=np.int32)
    child_labels = object_label_dense_array(raw_child_labels, dtype=np.int32)
    parent_labels, child_labels = aligned_dense_object_label_arrays(
        parent_labels,
        child_labels,
    )
    
    # Get object counts
    parent_count = int(parent_labels.max()) if parent_labels.max() > 0 else 0
    child_count = int(child_labels.max()) if child_labels.max() > 0 else 0
    
    parents_of = relationship_backend.parents_of_from_payload(
        parent_child_relationship,
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
    
    if calculate_distances.calculates_centroid_distance:
        centroid_distances = relationship_backend.centroid_distances(
            parent_labels, child_labels, parents_of
        )
        valid_dists = centroid_distances[~np.isnan(centroid_distances)]
        mean_centroid_dist = float(np.mean(valid_dists)) if len(valid_dists) > 0 else np.nan
    
    if calculate_distances.calculates_minimum_distance:
        minimum_distances = relationship_backend.minimum_distances(
            parent_labels, child_labels, parents_of
        )
        valid_dists = minimum_distances[~np.isnan(minimum_distances)]
        mean_minimum_dist = float(np.mean(valid_dists)) if len(valid_dists) > 0 else np.nan
    
    saved_child_relationship: ParentChildRelationshipPayload | None = None
    if save_children_with_parents:
        retained_child_ids = np.flatnonzero(
            np.concatenate((np.zeros(1, dtype=bool), parents_of > 0))
        ).astype(np.int32, copy=False)
        label_indexes = np.zeros(child_count + 1, dtype=np.int32)
        label_indexes[retained_child_ids] = np.arange(
            1,
            len(retained_child_ids) + 1,
            dtype=np.int32,
        )
        child_index = np.asarray(child_labels, dtype=np.intp)
        output_labels = label_indexes[child_index]
        saved_child_relationship = relationship_backend.parent_child_payload_from_labels(
            child_labels,
            output_labels,
        )
    else:
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

    parent_child_relationship = ParentChildRelationshipPayload(
        parent_ids=related_parent_ids,
        child_ids=related_child_ids,
    )
    return RelateObjectsResult(
        output_labels.astype(np.float32),
        parent_child_relationship,
        measurements,
        saved_child_relationship=saved_child_relationship,
    )
