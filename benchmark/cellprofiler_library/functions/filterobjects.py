"""
Converted from CellProfiler: FilterObjects
Original: FilterObjects module

FilterObjects eliminates objects based on their measurements (e.g., area, shape,
texture, intensity) or removes objects touching the image border.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from collections.abc import Sequence
from typing import ClassVar, Optional, Tuple

from metaclass_registry import AutoRegisterMeta
from numba import njit
from openhcs.interop.cellprofiler.measurement_lookup import (
    child_count_feature_child_name,
)
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.core.memory.decorators import numpy
from openhcs.core.runtime_artifact_queries import (
    measurement_feature_candidates,
    ordered_measurement_feature_candidates,
    optional_measurement_value_index,
    measurement_values_for_feature,
    normalize_measurement_token,
)
from openhcs.core.runtime_semantics import (
    DenseObjectLabelExtentDomainDeclaration,
    ObjectShapeMeasurementFeature,
    ObjectLabelMeasurementValues,
    ParentChildRelationshipPayload,
    aligned_dense_object_label_arrays,
    dense_object_label_present_ids,
    project_dense_object_label_stack,
)
from openhcs.core.runtime_values import (
    MeasurementTable,
    ObjectLabelPayload,
    ObjectRelationship,
    object_label_dense_array,
    object_label_payload_with_dense_labels,
)
from openhcs.interop.cellprofiler.measurement_dialect import (
    CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
)
from openhcs.processing.backends.analysis.region_properties import (
    LabelRegionPropertiesBackendStrategy,
)
from openhcs.processing.backends.cellprofiler.relationships import (
    ObjectRelationshipBackendStrategy,
)
from openhcs.processing.backends.cellprofiler.object_filtering import (
    BorderFilterSelectionStrategy,
    FilterMethod,
    FilterMode,
    FilterObjectsLabelPlane,
    FilterObjectsMeasurementLimitWindow,
    FilterObjectsParentChildRelationship,
    FilterObjectsParentChildRelationships,
    FilterObjectsRelationshipEndpointIds,
    FilterObjectsSelectionRequest,
    FilterObjectsStats,
    FilterSelectionStrategy,
    PerObjectAssignment,
    PerObjectAssignmentStrategy,
)
from openhcs.processing.backends.cellprofiler.shape import form_factor_values
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.processing.materialization import csv_materializer, segmentation_mask_rois


@numpy(contract=ProcessingContract.FLEXIBLE)
@special_outputs(
    ("filter_stats", csv_materializer(
        fields=["slice_index", "objects_pre_filter", "objects_post_filter", "objects_removed"],
        analysis_type="filter_objects"
    )),
    ("filtered_labels", segmentation_mask_rois())
)
def filter_objects(
    image: np.ndarray,
    mode: FilterMode = FilterMode.MEASUREMENTS,
    filter_method: FilterMethod = FilterMethod.LIMITS,
    object_labels: tuple[np.ndarray, ...] = (),
    measurement_values: Optional[np.ndarray] = None,
    measurement_features: tuple[str, ...] = (),
    measurement_min_values: tuple[float | None, ...] = (),
    measurement_max_values: tuple[float | None, ...] = (),
    measurement_use_minimum: tuple[bool, ...] = (),
    measurement_use_maximum: tuple[bool, ...] = (),
    measurement_tables: tuple[MeasurementTable, ...] = (),
    enclosing_object_labels: Optional[np.ndarray] = None,
    parent_child_relationship: Optional[FilterObjectsParentChildRelationship] = None,
    parent_child_relationships: FilterObjectsParentChildRelationships = (),
    per_object_assignment: PerObjectAssignment = PerObjectAssignment.BOTH_PARENTS,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    use_minimum: bool = True,
    use_maximum: bool = True,
    additional_object_count: int = 0,
    outline_object_indices: tuple[int, ...] = (),
) -> tuple[
    np.ndarray,
    FilterObjectsStats,
    np.ndarray | ParentChildRelationshipPayload,
    ...,
]:
    """
    Filter objects based on measurements or border touching.
    
    Args:
        image: Input intensity image (H, W)
        object_labels: Primary labels followed by additional label sets to
            relabel using the retained primary-object mask.
        mode: Filtering mode - MEASUREMENTS or BORDER
        filter_method: Method for measurement-based filtering
        measurement_values: Array of measurement values per object (indexed by label-1)
        measurement_features: CellProfiler feature names used for limits filtering.
        measurement_tables: Prior object measurement tables from the runtime adapter.
        min_value: Minimum threshold for LIMITS method
        max_value: Maximum threshold for LIMITS method
        use_minimum: Whether to apply minimum threshold
        use_maximum: Whether to apply maximum threshold
    
    Returns:
        Tuple of (image, stats, filtered primary labels, additional relabeled
        objects, parent-child relationships, outline images).
    """
    if object_labels is None:
        object_labels = ()
    elif isinstance(object_labels, np.ndarray):
        object_labels = (object_labels,)
    if len(object_labels) == 0:
        raise ValueError("FilterObjects requires at least one object label input.")
    mode = coerce_cellprofiler_enum(FilterMode, mode)
    filter_method = coerce_cellprofiler_enum(FilterMethod, filter_method)
    per_object_assignment = coerce_cellprofiler_enum(
        PerObjectAssignment,
        per_object_assignment,
    )
    if additional_object_count != len(object_labels) - 1:
        raise ValueError(
            "FilterObjects additional_object_count must match additional object "
            "label inputs."
        )
    labels = FilterObjectsLabelPlane(object_labels[0]).projected
    labels = labels.astype(np.int32)
    additional_label_planes = tuple(
        FilterObjectsLabelPlane(value).aligned_to(labels)
        for value in object_labels[1:]
    )
    input_label_planes = (labels, *additional_label_planes)
    max_label = labels.max()
    
    if max_label == 0:
        # No objects to filter
        stats = FilterObjectsStats.from_counts(
            objects_pre_filter=0,
            objects_post_filter=0,
        )
        relabeled_objects = _filtered_object_payloads(
            object_labels,
            (labels, *additional_label_planes),
        )
        relationships = _object_transform_relationships(
            input_label_planes,
            relabeled_objects,
        )
        return (
            image,
            stats,
            *relabeled_objects,
            *relationships,
            *_outline_images(relabeled_objects, outline_object_indices),
        )
    
    object_ids = dense_object_label_present_ids(labels)
    num_objects_pre = len(object_ids)
    selection_measurement_values = (
        None
        if measurement_values is None
        else ObjectLabelMeasurementValues.from_label_indexed_values(
            object_ids,
            measurement_values,
        )
    )

    indexes_to_keep = FilterSelectionStrategy.for_mode_and_method(
        mode,
        filter_method,
    ).indexes_to_keep(
        FilterObjectsSelectionRequest(
            labels=labels,
            object_ids=object_ids,
            filter_method=filter_method,
            measurement_values=selection_measurement_values,
            measurement_features=measurement_features,
            measurement_min_values=measurement_min_values,
            measurement_max_values=measurement_max_values,
            measurement_use_minimum=measurement_use_minimum,
            measurement_use_maximum=measurement_use_maximum,
            measurement_tables=measurement_tables,
            enclosing_labels=FilterObjectsLabelPlane.optional_aligned_to(
                labels,
                enclosing_object_labels,
            ),
            parent_child_relationship=parent_child_relationship,
            parent_child_relationships=_relationship_tuple(
                parent_child_relationship,
                parent_child_relationships,
            ),
            per_object_assignment=per_object_assignment,
            min_value=min_value,
            max_value=max_value,
            use_minimum=use_minimum,
            use_maximum=use_maximum,
        )
    )
    
    # Create new label image with only kept objects
    new_object_count = len(indexes_to_keep)
    label_mapping = np.zeros(max_label + 1, dtype=np.int32)
    for new_idx, old_idx in enumerate(indexes_to_keep, start=1):
        if old_idx <= max_label:
            label_mapping[old_idx] = new_idx
    
    filtered_labels = label_mapping[labels]
    relabeled_objects = _filtered_object_payloads(
        object_labels,
        (
            filtered_labels,
            *(
                _relabel_overlapping_objects(additional, filtered_labels)
                for additional in additional_label_planes
            ),
        ),
    )
    relationships = _object_transform_relationships(
        input_label_planes,
        relabeled_objects,
    )
    
    stats = FilterObjectsStats.from_counts(
        objects_pre_filter=num_objects_pre,
        objects_post_filter=new_object_count,
    )
    
    return (
        image,
        stats,
        *relabeled_objects,
        *relationships,
        *_outline_images(relabeled_objects, outline_object_indices),
    )


def _filtered_object_payloads(
    inputs: Sequence[np.ndarray],
    outputs: Sequence[np.ndarray],
) -> tuple[np.ndarray | ObjectLabelPayload, ...]:
    return tuple(
        _filtered_object_payload(input_value, output_labels)
        for input_value, output_labels in zip(inputs, outputs, strict=True)
    )


def _filtered_object_payload(
    input_value: np.ndarray,
    output_labels: np.ndarray,
) -> np.ndarray | ObjectLabelPayload:
    return object_label_payload_with_dense_labels(
        input_value,
        output_labels,
        domain_declaration=DenseObjectLabelExtentDomainDeclaration(),
    )


def _keep_one(
    values: ObjectLabelMeasurementValues,
    keep_max: bool = True,
) -> list[int]:
    """Keep only the object with the maximum or minimum finite measurement."""
    selected_id = values.extremum_id(keep_max=keep_max)
    return [] if selected_id is None else [selected_id]


def _require_enclosing_labels(
    request: FilterObjectsSelectionRequest,
) -> np.ndarray:
    if request.enclosing_labels is not None:
        return request.enclosing_labels
    raise ValueError(
        "FilterObjects per-object filtering requires enclosing object labels."
    )


def _best_child_indexes_both_parents(
    child_labels: np.ndarray,
    enclosing_labels: np.ndarray,
    measurement_values: np.ndarray,
    keep_max: bool,
) -> list[int]:
    import scipy.ndimage

    values = np.asarray(measurement_values, dtype=np.float64)
    if values.size == 0:
        return []
    child_array = np.asarray(child_labels, dtype=np.int32)
    parent_array = np.asarray(enclosing_labels, dtype=np.int32)
    max_parent = int(parent_array.max()) if parent_array.size else 0
    if max_parent <= 0:
        return []

    pixel_values = np.empty(values.size + 1, dtype=np.float64)
    pixel_values[1:] = values
    pixel_values[0] = -np.inf if keep_max else np.inf
    source_values = pixel_values[child_array]
    parent_range = np.arange(1, max_parent + 1)
    position_fn = scipy.ndimage.maximum_position if keep_max else scipy.ndimage.minimum_position
    positions = position_fn(source_values, parent_array, parent_range)
    positions = np.asarray(
        (positions,) if isinstance(positions, tuple) else positions,
        dtype=np.uint32,
    )
    if positions.size == 0:
        return []
    indexes = tuple(map(tuple, positions.transpose()))
    selected = sorted(set(int(label) for label in child_array[indexes]))
    if selected and selected[0] == 0:
        selected = selected[1:]
    return selected


@njit(cache=True)
def _best_child_selected_mask_both_parents_numba(
    child_labels: np.ndarray,
    enclosing_labels: np.ndarray,
    measurement_values: np.ndarray,
    keep_max: bool,
) -> np.ndarray:
    max_child = int(np.max(child_labels))
    max_parent = int(np.max(enclosing_labels))
    selected = np.zeros(max_child + 1, dtype=np.bool_)
    if max_child <= 0 or max_parent <= 0:
        return selected

    best_child_by_parent = np.zeros(max_parent + 1, dtype=np.int32)
    best_value_by_parent = np.empty(max_parent + 1, dtype=np.float64)
    height, width = child_labels.shape
    for row in range(height):
        for col in range(width):
            child_id = int(child_labels[row, col])
            parent_id = int(enclosing_labels[row, col])
            if child_id <= 0 or parent_id <= 0:
                continue
            value_index = child_id - 1
            if value_index < 0 or value_index >= measurement_values.size:
                continue
            value = float(measurement_values[value_index])
            if not np.isfinite(value):
                continue

            best_child = int(best_child_by_parent[parent_id])
            if best_child == 0:
                best_child_by_parent[parent_id] = child_id
                best_value_by_parent[parent_id] = value
            else:
                best_value = float(best_value_by_parent[parent_id])
                if keep_max:
                    # CellProfiler delegates this path to
                    # scipy.ndimage.maximum_position, which uses the last
                    # maximal pixel within each enclosing label.
                    if value >= best_value:
                        best_child_by_parent[parent_id] = child_id
                        best_value_by_parent[parent_id] = value
                else:
                    # scipy.ndimage.minimum_position keeps the first minimal
                    # pixel, so equal values do not replace the current child.
                    if value < best_value:
                        best_child_by_parent[parent_id] = child_id
                        best_value_by_parent[parent_id] = value

    for parent_id in range(1, max_parent + 1):
        child_id = int(best_child_by_parent[parent_id])
        if child_id > 0:
            selected[child_id] = True
    return selected


def _best_child_indexes_parent_with_most_overlap(
    child_labels: np.ndarray,
    enclosing_labels: np.ndarray,
    measurement_values: np.ndarray,
    keep_max: bool,
) -> list[int]:
    max_child = int(np.max(child_labels))
    if max_child <= 0:
        return []
    child_ids, parent_ids, overlap_counts = _unique_overlap_counts(
        child_labels,
        enclosing_labels,
    )
    if child_ids.size == 0:
        return []
    return _selected_labels_to_list(
        _best_child_selected_mask_parent_with_most_overlap_numba(
            np.ascontiguousarray(child_ids, dtype=np.int32),
            np.ascontiguousarray(parent_ids, dtype=np.int32),
            np.ascontiguousarray(overlap_counts, dtype=np.int64),
            np.ascontiguousarray(measurement_values, dtype=np.float64),
            max_child,
            bool(keep_max),
        )
    )


def _unique_overlap_counts(
    child_labels: np.ndarray,
    enclosing_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    child_array = np.asarray(child_labels, dtype=np.int64)
    parent_array = np.asarray(enclosing_labels, dtype=np.int64)
    max_parent = int(parent_array.max())
    overlap_mask = (child_array > 0) & (parent_array > 0)
    if not np.any(overlap_mask):
        empty = np.array([], dtype=np.int64)
        return empty, empty, empty
    encoded_pairs = child_array[overlap_mask] * (max_parent + 1) + parent_array[
        overlap_mask
    ]
    unique_pairs, overlap_counts = np.unique(encoded_pairs, return_counts=True)
    return (
        unique_pairs // (max_parent + 1),
        unique_pairs % (max_parent + 1),
        overlap_counts,
    )


@njit(cache=True)
def _best_child_selected_mask_parent_with_most_overlap_numba(
    child_ids: np.ndarray,
    parent_ids: np.ndarray,
    overlap_counts: np.ndarray,
    measurement_values: np.ndarray,
    max_child: int,
    keep_max: bool,
) -> np.ndarray:
    selected = np.zeros(max_child + 1, dtype=np.bool_)
    if child_ids.size == 0:
        return selected

    max_parent = int(np.max(parent_ids))
    best_parent_by_child = np.zeros(max_child + 1, dtype=np.int32)
    best_overlap_by_child = np.zeros(max_child + 1, dtype=np.int64)
    for index in range(child_ids.size):
        child_id = int(child_ids[index])
        parent_id = int(parent_ids[index])
        overlap_count = int(overlap_counts[index])
        best_parent = int(best_parent_by_child[child_id])
        best_overlap = int(best_overlap_by_child[child_id])
        if (
            best_parent == 0
            or overlap_count > best_overlap
            or (overlap_count == best_overlap and parent_id < best_parent)
        ):
            best_parent_by_child[child_id] = parent_id
            best_overlap_by_child[child_id] = overlap_count

    best_child_by_parent = np.zeros(max_parent + 1, dtype=np.int32)
    best_value_by_parent = np.empty(max_parent + 1, dtype=np.float64)
    for child_id in range(1, max_child + 1):
        parent_id = int(best_parent_by_child[child_id])
        if parent_id <= 0:
            continue
        value_index = child_id - 1
        if value_index < 0 or value_index >= measurement_values.size:
            continue
        value = float(measurement_values[value_index])
        if not np.isfinite(value):
            continue

        best_child = int(best_child_by_parent[parent_id])
        if best_child == 0:
            best_child_by_parent[parent_id] = child_id
            best_value_by_parent[parent_id] = value
        else:
            best_value = float(best_value_by_parent[parent_id])
            if keep_max:
                if value > best_value or (
                    value == best_value and child_id < best_child
                ):
                    best_child_by_parent[parent_id] = child_id
                    best_value_by_parent[parent_id] = value
            else:
                if value < best_value or (
                    value == best_value and child_id < best_child
                ):
                    best_child_by_parent[parent_id] = child_id
                    best_value_by_parent[parent_id] = value

    for parent_id in range(1, max_parent + 1):
        child_id = int(best_child_by_parent[parent_id])
        if child_id > 0:
            selected[child_id] = True
    return selected


def _selected_labels_to_list(selected: np.ndarray) -> list[int]:
    return np.flatnonzero(np.asarray(selected, dtype=bool)).astype(int).tolist()


def _relationship_tuple(
    relationship: FilterObjectsParentChildRelationship | None,
    relationships: Sequence[FilterObjectsParentChildRelationship],
) -> FilterObjectsParentChildRelationships:
    ordered = (*(() if relationship is None else (relationship,)), *tuple(relationships))
    unique: list[FilterObjectsParentChildRelationship] = []
    seen: set[tuple[str, str] | int] = set()
    for value in ordered:
        key: tuple[str, str] | int
        if isinstance(value, ObjectRelationship):
            key = (value.source.name, value.target.name)
        else:
            key = id(value)
        if key in seen:
            continue
        seen.add(key)
        unique.append(value)
    return tuple(unique)
def _relabel_overlapping_objects(
    labels: np.ndarray,
    filtered_primary_labels: np.ndarray,
) -> np.ndarray:
    """Relabel additional objects by overlap with retained primary objects."""
    labels = labels.astype(np.int32)
    retained_mask = filtered_primary_labels > 0
    if labels.shape != retained_mask.shape:
        raise ValueError(
            "FilterObjects additional object labels must match primary labels."
        )
    retained_source_labels = np.unique(labels[retained_mask])
    retained_source_labels = retained_source_labels[retained_source_labels > 0]
    if retained_source_labels.size == 0:
        return np.zeros_like(labels, dtype=np.int32)
    mapping = np.zeros(labels.max() + 1, dtype=np.int32)
    for new_idx, old_idx in enumerate(retained_source_labels, start=1):
        mapping[int(old_idx)] = new_idx
    return mapping[labels]


def _object_transform_relationships(
    input_label_planes: tuple[np.ndarray, ...],
    relabeled_objects: tuple[np.ndarray, ...],
) -> tuple[ParentChildRelationshipPayload, ...]:
    if len(input_label_planes) != len(relabeled_objects):
        raise ValueError(
            "Object transform relationship derivation requires aligned input "
            "and output label planes."
        )
    relationship_backend = ObjectRelationshipBackendStrategy.for_memory_type()
    return tuple(
        relationship_backend.parent_child_payload_from_labels(
            np.asarray(input_labels),
            np.asarray(output_labels),
        )
        for input_labels, output_labels in zip(
            input_label_planes,
            relabeled_objects,
            strict=True,
        )
    )


def _outline_images(
    relabeled_objects: tuple[np.ndarray, ...],
    outline_object_indices: tuple[int, ...],
) -> tuple[np.ndarray, ...]:
    return tuple(
        _outline_image(relabeled_objects[index])
        for index in outline_object_indices
    )


def _outline_image(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).astype(np.int32)
    if labels.ndim != 2:
        raise ValueError("FilterObjects outline images require 2D labels.")
    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]
    boundary[1:, :] |= labels[:-1, :] != labels[1:, :]
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    boundary[:, 1:] |= labels[:, :-1] != labels[:, 1:]
    boundary &= labels > 0
    return boundary.astype(np.uint8)


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(
    ("filter_stats", csv_materializer(
        fields=["slice_index", "objects_pre_filter", "objects_post_filter", "objects_removed"],
        analysis_type="filter_objects"
    )),
    ("filtered_labels", segmentation_mask_rois())
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
    labels = object_label_dense_array(labels, dtype=np.int32)
    max_label = labels.max()
    
    if max_label == 0:
        stats = FilterObjectsStats.from_counts(
            objects_pre_filter=0,
            objects_post_filter=0,
        )
        return image, stats, labels
    
    # Compute area for each object through the shared dense-label backend.
    region_props = LabelRegionPropertiesBackendStrategy.for_memory_type().measure_2d(
        labels
    )
    areas = region_props.area
    num_objects_pre = len(region_props.label)
    
    # Filter by area limits
    indexes_to_keep = FilterObjectsMeasurementLimitWindow.from_label_indexed_values(
        areas,
        min_value=min_area,
        max_value=max_area,
        use_minimum=use_minimum,
        use_maximum=use_maximum,
    ).retained_ids
    
    # Create new label image
    new_object_count = len(indexes_to_keep)
    label_mapping = np.zeros(max_label + 1, dtype=np.int32)
    for new_idx, old_idx in enumerate(indexes_to_keep, start=1):
        if old_idx <= max_label:
            label_mapping[old_idx] = new_idx
    
    filtered_labels = label_mapping[labels]
    
    stats = FilterObjectsStats.from_counts(
        objects_pre_filter=num_objects_pre,
        objects_post_filter=new_object_count,
    )
    
    return image, stats, filtered_labels


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(
    ("filter_stats", csv_materializer(
        fields=["slice_index", "objects_pre_filter", "objects_post_filter", "objects_removed"],
        analysis_type="filter_objects"
    )),
    ("filtered_labels", segmentation_mask_rois())
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
    labels = object_label_dense_array(labels, dtype=np.int32)
    max_label = labels.max()
    
    if max_label == 0:
        stats = FilterObjectsStats.from_counts(
            objects_pre_filter=0,
            objects_post_filter=0,
        )
        return image, stats, labels
    
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]
    num_objects_pre = len(unique_labels)
    
    indexes_to_keep = BorderFilterSelectionStrategy.discard_border_objects(labels)
    
    # Create new label image
    new_object_count = len(indexes_to_keep)
    label_mapping = np.zeros(max_label + 1, dtype=np.int32)
    for new_idx, old_idx in enumerate(indexes_to_keep, start=1):
        if old_idx <= max_label:
            label_mapping[old_idx] = new_idx
    
    filtered_labels = label_mapping[labels]
    
    stats = FilterObjectsStats.from_counts(
        objects_pre_filter=num_objects_pre,
        objects_post_filter=new_object_count,
    )
    
    return image, stats, filtered_labels
