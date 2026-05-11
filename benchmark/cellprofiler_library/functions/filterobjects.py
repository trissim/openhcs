"""
Converted from CellProfiler: FilterObjects
Original: FilterObjects module

FilterObjects eliminates objects based on their measurements (e.g., area, shape,
texture, intensity) or removes objects touching the image border.
"""

import numpy as np
from typing import Optional, Tuple

from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.core.memory.decorators import numpy
from openhcs.core.runtime_semantics import (
    ObjectLabelMeasurementValues,
    ParentChildRelationshipPayload,
    dense_object_label_present_ids,
)
from openhcs.core.runtime_values import (
    MeasurementTable,
    object_label_dense_array,
)
from openhcs.processing.backends.analysis.region_properties import (
    LabelRegionPropertiesBackendStrategy,
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
    filter_objects_outline_images,
    filter_objects_relationship_tuple,
    filtered_object_payloads,
    object_transform_relationships,
    relabel_overlapping_objects,
)
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
        relabeled_objects = filtered_object_payloads(
            object_labels,
            (labels, *additional_label_planes),
        )
        relationships = object_transform_relationships(
            input_label_planes,
            relabeled_objects,
        )
        return (
            image,
            stats,
            *relabeled_objects,
            *relationships,
            *filter_objects_outline_images(relabeled_objects, outline_object_indices),
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
            parent_child_relationships=filter_objects_relationship_tuple(
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
    relabeled_objects = filtered_object_payloads(
        object_labels,
        (
            filtered_labels,
            *(
                relabel_overlapping_objects(additional, filtered_labels)
                for additional in additional_label_planes
            ),
        ),
    )
    relationships = object_transform_relationships(
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
        *filter_objects_outline_images(relabeled_objects, outline_object_indices),
    )


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
