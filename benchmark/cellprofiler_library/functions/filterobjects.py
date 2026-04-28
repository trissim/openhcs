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
from typing import ClassVar, Optional, Tuple

from benchmark.cellprofiler_compat.measurement_lookup import (
    measurement_values_for_feature,
)
from metaclass_registry import AutoRegisterMeta
from openhcs.core.memory.decorators import numpy
from openhcs.core.runtime_values import MeasurementTable
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.processing.materialization import csv_materializer, segmentation_mask_rois


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

    @classmethod
    def from_counts(
        cls,
        *,
        objects_pre_filter: int,
        objects_post_filter: int,
        slice_index: int = 0,
    ) -> "FilterObjectsStats":
        return cls(
            slice_index=slice_index,
            objects_pre_filter=objects_pre_filter,
            objects_post_filter=objects_post_filter,
            objects_removed=objects_pre_filter - objects_post_filter,
        )


@dataclass(frozen=True, slots=True)
class FilterObjectsSelectionRequest:
    """Inputs needed to choose retained primary object labels."""

    labels: np.ndarray
    num_objects_pre: int
    filter_method: FilterMethod
    measurement_values: np.ndarray | None
    measurement_features: tuple[str, ...]
    measurement_min_values: tuple[float | None, ...]
    measurement_max_values: tuple[float | None, ...]
    measurement_use_minimum: tuple[bool, ...]
    measurement_use_maximum: tuple[bool, ...]
    measurement_tables: tuple[MeasurementTable, ...]
    min_value: float | None
    max_value: float | None
    use_minimum: bool
    use_maximum: bool


@dataclass(frozen=True, slots=True)
class FilterSelectionKey:
    """Nominal retained-object selection identity."""

    mode: FilterMode
    method: FilterMethod | None = None

    @property
    def label(self) -> str:
        if self.method is None:
            return self.mode.value
        return f"{self.mode.value}:{self.method.value}"

    def lookup_candidates(self) -> tuple["FilterSelectionKey", ...]:
        return (self, FilterSelectionKey(self.mode))


class FilterSelectionStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal retained-object selection for each FilterObjects behavior."""

    __registry_key__ = "selection_key"
    __skip_if_no_key__ = True
    selection_key: ClassVar[FilterSelectionKey | None] = None

    @classmethod
    def for_mode_and_method(
        cls,
        mode: FilterMode,
        method: FilterMethod,
    ) -> "FilterSelectionStrategy":
        requested_key = FilterSelectionKey(mode, method)
        for key in requested_key.lookup_candidates():
            strategy_type = cls.__registry__.get(key)
            if strategy_type is not None:
                return strategy_type()
        raise ValueError(
            f"Unsupported FilterObjects selection {requested_key.label!r}."
        )

    @abstractmethod
    def indexes_to_keep(
        self,
        request: FilterObjectsSelectionRequest,
    ) -> list[int]:
        """Return one-indexed primary object labels to retain."""


class BorderFilterSelectionStrategy(FilterSelectionStrategy):
    """Remove primary objects touching the image border."""

    selection_key = FilterSelectionKey(FilterMode.BORDER)

    def indexes_to_keep(
        self,
        request: FilterObjectsSelectionRequest,
    ) -> list[int]:
        return _discard_border_objects(request.labels)


class LimitsFilterSelectionStrategy(FilterSelectionStrategy):
    """Keep objects whose measurement falls within configured limits."""

    selection_key = FilterSelectionKey(FilterMode.MEASUREMENTS, FilterMethod.LIMITS)

    def indexes_to_keep(
        self,
        request: FilterObjectsSelectionRequest,
    ) -> list[int]:
        if request.measurement_features:
            return _keep_matching_measurement_rules(request)
        values = request.measurement_values
        if values is None:
            values = _area_measurement_values(request.labels)
        return _keep_within_limits(
            values,
            request.min_value,
            request.max_value,
            request.use_minimum,
            request.use_maximum,
        )


class ExtremumFilterSelectionStrategy(FilterSelectionStrategy):
    """Keep one object selected by a measurement extremum."""

    keep_max: ClassVar[bool | None] = None

    def indexes_to_keep(
        self,
        request: FilterObjectsSelectionRequest,
    ) -> list[int]:
        keep_max = type(self).keep_max
        if keep_max is None:
            raise TypeError("ExtremumFilterSelectionStrategy must define keep_max.")
        values = request.measurement_values
        if values is None:
            values = _first_measurement_values(request)
        return _keep_one(values, keep_max=keep_max)


class MinimalFilterSelectionStrategy(ExtremumFilterSelectionStrategy):
    """Keep the object with the minimum measurement value."""

    selection_key = FilterSelectionKey(FilterMode.MEASUREMENTS, FilterMethod.MINIMAL)
    keep_max = False


class MaximalFilterSelectionStrategy(ExtremumFilterSelectionStrategy):
    """Keep the object with the maximum measurement value."""

    selection_key = FilterSelectionKey(FilterMode.MEASUREMENTS, FilterMethod.MAXIMAL)
    keep_max = True


class PerObjectFilterSelectionStrategy(FilterSelectionStrategy):
    """Reject per-object filtering until parent-object measurements are modeled."""

    selection_key: ClassVar[FilterSelectionKey | None] = None

    def indexes_to_keep(
        self,
        request: FilterObjectsSelectionRequest,
    ) -> list[int]:
        del request
        selection_key = type(self).selection_key
        if selection_key is None or selection_key.method is None:
            raise TypeError("PerObjectFilterSelectionStrategy must define method.")
        raise NotImplementedError(
            f"FilterObjects method {selection_key.method.value!r} requires "
            "parent-object assignment semantics."
        )


class MinimalPerObjectFilterSelectionStrategy(PerObjectFilterSelectionStrategy):
    """Fail loudly for minimal-per-parent filtering until relationships exist."""

    selection_key = FilterSelectionKey(
        FilterMode.MEASUREMENTS,
        FilterMethod.MINIMAL_PER_OBJECT,
    )


class MaximalPerObjectFilterSelectionStrategy(PerObjectFilterSelectionStrategy):
    """Fail loudly for maximal-per-parent filtering until relationships exist."""

    selection_key = FilterSelectionKey(
        FilterMode.MEASUREMENTS,
        FilterMethod.MAXIMAL_PER_OBJECT,
    )


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
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    use_minimum: bool = True,
    use_maximum: bool = True,
    additional_object_count: int = 0,
    outline_object_indices: tuple[int, ...] = (),
) -> tuple[np.ndarray, FilterObjectsStats, np.ndarray, ...]:
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
        objects, outline images).
    """
    if not object_labels:
        raise ValueError("FilterObjects requires at least one object label input.")
    if additional_object_count != len(object_labels) - 1:
        raise ValueError(
            "FilterObjects additional_object_count must match additional object "
            "label inputs."
        )
    labels = _label_plane(object_labels[0])
    labels = labels.astype(np.int32)
    max_label = labels.max()
    
    if max_label == 0:
        # No objects to filter
        stats = FilterObjectsStats.from_counts(
            objects_pre_filter=0,
            objects_post_filter=0,
        )
        relabeled_objects = (
            labels,
            *(_label_plane(value) for value in object_labels[1:]),
        )
        return (
            image,
            stats,
            *relabeled_objects,
            *_outline_images(relabeled_objects, outline_object_indices),
        )
    
    # Get all unique labels (excluding background)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]
    num_objects_pre = len(unique_labels)

    indexes_to_keep = FilterSelectionStrategy.for_mode_and_method(
        mode,
        filter_method,
    ).indexes_to_keep(
        FilterObjectsSelectionRequest(
            labels=labels,
            num_objects_pre=num_objects_pre,
            filter_method=filter_method,
            measurement_values=measurement_values,
            measurement_features=measurement_features,
            measurement_min_values=measurement_min_values,
            measurement_max_values=measurement_max_values,
            measurement_use_minimum=measurement_use_minimum,
            measurement_use_maximum=measurement_use_maximum,
            measurement_tables=measurement_tables,
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
    relabeled_objects = (
        filtered_labels,
        *(
            _relabel_overlapping_objects(_label_plane(additional), filtered_labels)
            for additional in object_labels[1:]
        ),
    )
    
    stats = FilterObjectsStats.from_counts(
        objects_pre_filter=num_objects_pre,
        objects_post_filter=new_object_count,
    )
    
    return (
        image,
        stats,
        *relabeled_objects,
        *_outline_images(relabeled_objects, outline_object_indices),
    )


def _discard_border_objects(labels: np.ndarray) -> list[int]:
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
) -> list[int]:
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
    hits[~np.isfinite(values)] = False
    
    if use_minimum and min_value is not None:
        hits[values < min_value] = False
    
    if use_maximum and max_value is not None:
        hits[values > max_value] = False
    
    # Convert to 1-indexed labels
    indexes = np.argwhere(hits).flatten() + 1
    return indexes.tolist()


def _keep_one(values: np.ndarray, keep_max: bool = True) -> list[int]:
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


def _keep_matching_measurement_rules(
    request: FilterObjectsSelectionRequest,
) -> list[int]:
    _validate_measurement_rule_lengths(request)
    hits = np.ones(request.num_objects_pre, dtype=bool)
    for index, feature_name in enumerate(request.measurement_features):
        values = measurement_values_for_feature(
            request.measurement_tables,
            feature_name,
            object_count=request.num_objects_pre,
        )
        keep_indexes = _keep_within_limits(
            values,
            request.measurement_min_values[index],
            request.measurement_max_values[index],
            request.measurement_use_minimum[index],
            request.measurement_use_maximum[index],
        )
        rule_hits = np.zeros(request.num_objects_pre, dtype=bool)
        for keep_index in keep_indexes:
            if 1 <= keep_index <= request.num_objects_pre:
                rule_hits[keep_index - 1] = True
        hits &= rule_hits
    return (np.argwhere(hits).flatten() + 1).tolist()


def _first_measurement_values(request: FilterObjectsSelectionRequest) -> np.ndarray:
    if request.measurement_values is not None:
        return request.measurement_values
    if request.measurement_features:
        return measurement_values_for_feature(
            request.measurement_tables,
            request.measurement_features[0],
            object_count=request.num_objects_pre,
        )
    return _area_measurement_values(request.labels)


def _area_measurement_values(labels: np.ndarray) -> np.ndarray:
    from skimage.measure import regionprops

    return np.array([prop.area for prop in regionprops(labels)])


def _validate_measurement_rule_lengths(
    request: FilterObjectsSelectionRequest,
) -> None:
    expected = len(request.measurement_features)
    lengths = {
        len(request.measurement_min_values),
        len(request.measurement_max_values),
        len(request.measurement_use_minimum),
        len(request.measurement_use_maximum),
    }
    if lengths == {expected}:
        return
    raise ValueError("FilterObjects measurement rule kwargs must align by row.")


def _label_plane(labels: np.ndarray) -> np.ndarray:
    """Return the label plane FilterObjects should operate on."""
    if labels.ndim == 3 and labels.shape[0] == 1:
        return labels[0]
    return labels


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


def _outline_images(
    relabeled_objects: tuple[np.ndarray, ...],
    outline_object_indices: tuple[int, ...],
) -> tuple[np.ndarray, ...]:
    return tuple(
        _outline_image(relabeled_objects[index])
        for index in outline_object_indices
    )


def _outline_image(labels: np.ndarray) -> np.ndarray:
    labels = labels.astype(np.int32)
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
    from skimage.measure import regionprops
    
    labels = labels.astype(np.int32)
    max_label = labels.max()
    
    if max_label == 0:
        stats = FilterObjectsStats.from_counts(
            objects_pre_filter=0,
            objects_post_filter=0,
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
    labels = labels.astype(np.int32)
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
    
    indexes_to_keep = _discard_border_objects(labels)
    
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
