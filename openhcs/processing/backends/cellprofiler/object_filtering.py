"""Object-filtering semantics for CellProfiler-compatible processing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

import numpy as np
from metaclass_registry import AutoRegisterMeta
from numba import njit

from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.core.runtime_artifact_queries import (
    measurement_values_for_feature,
    normalize_measurement_token,
    optional_measurement_value_index,
    ordered_measurement_feature_candidates,
)
from openhcs.core.runtime_semantics import (
    ObjectShapeMeasurementFeature,
    ObjectLabelMeasurementValues,
    ParentChildRelationshipPayload,
    aligned_dense_object_label_arrays,
    project_dense_object_label_stack,
)
from openhcs.interop.cellprofiler.measurement_dialect import (
    CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
)
from openhcs.interop.cellprofiler.measurement_lookup import (
    child_count_feature_child_name,
)
from openhcs.processing.backends.analysis.region_properties import (
    LabelRegionPropertiesBackendStrategy,
)
from openhcs.processing.backends.cellprofiler.shape import form_factor_values
from openhcs.core.runtime_values import (
    MeasurementTable,
    ObjectRelationship,
    object_label_dense_array,
)


class FilterMethod(Enum):
    """CellProfiler FilterObjects measurement selection modes."""

    MINIMAL = "minimal"
    MAXIMAL = "maximal"
    MINIMAL_PER_OBJECT = "minimal_per_object"
    MAXIMAL_PER_OBJECT = "maximal_per_object"
    LIMITS = "limits"


class FilterMode(Enum):
    """CellProfiler FilterObjects top-level filter modes."""

    MEASUREMENTS = "measurements"
    BORDER = "border"


class PerObjectAssignment(Enum):
    """How per-object FilterObjects assigns child objects to parents."""

    BOTH_PARENTS = "both_parents"
    PARENT_WITH_MOST_OVERLAP = "parent_with_most_overlap"


FilterObjectsParentChildRelationship = (
    ObjectRelationship | ParentChildRelationshipPayload
)
FilterObjectsParentChildRelationships = tuple[FilterObjectsParentChildRelationship, ...]


@dataclass(frozen=True, slots=True)
class FilterObjectsRelationshipEndpointIds:
    """Dense integer ids carried by a FilterObjects relationship endpoint."""

    values: object

    @property
    def ids(self) -> tuple[int, ...]:
        return tuple(int(value) for value in np.asarray(self.values).reshape(-1))


@dataclass(frozen=True, slots=True)
class FilterObjectsLabelPlane:
    """Dense label plane with FilterObjects projection/alignment semantics."""

    labels: np.ndarray

    @property
    def projected(self) -> np.ndarray:
        return project_dense_object_label_stack(
            object_label_dense_array(self.labels, dtype=np.int32)
        )

    def aligned_to(self, reference_labels: np.ndarray) -> np.ndarray:
        _aligned_reference, aligned_labels = aligned_dense_object_label_arrays(
            reference_labels,
            object_label_dense_array(self.labels, dtype=np.int32),
        )
        return aligned_labels.astype(np.int32, copy=False)

    @classmethod
    def optional_aligned_to(
        cls,
        reference_labels: np.ndarray,
        labels: np.ndarray | None,
    ) -> np.ndarray | None:
        if labels is None:
            return None
        return cls(labels).aligned_to(reference_labels)


@dataclass(frozen=True, slots=True)
class FilterObjectsMeasurementLimitWindow:
    """Object-id retention policy for FilterObjects measurement bounds."""

    values: ObjectLabelMeasurementValues
    min_value: float | None
    max_value: float | None
    use_minimum: bool
    use_maximum: bool

    @classmethod
    def from_label_indexed_values(
        cls,
        values: np.ndarray,
        *,
        min_value: float | None,
        max_value: float | None,
        use_minimum: bool,
        use_maximum: bool,
    ) -> "FilterObjectsMeasurementLimitWindow":
        object_ids = tuple(range(1, len(values) + 1))
        return cls(
            ObjectLabelMeasurementValues.from_label_indexed_values(
                object_ids,
                values,
            ),
            min_value=min_value,
            max_value=max_value,
            use_minimum=use_minimum,
            use_maximum=use_maximum,
        )

    @property
    def retained_ids(self) -> list[int]:
        return list(
            self.values.ids_within_limits(
                min_value=self.min_value,
                max_value=self.max_value,
                use_minimum=self.use_minimum,
                use_maximum=self.use_maximum,
            )
        )


@dataclass
class FilterObjectsStats:
    """FilterObjects object-count output row."""

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
    object_ids: tuple[int, ...]
    filter_method: FilterMethod
    measurement_values: ObjectLabelMeasurementValues | None
    measurement_features: tuple[str, ...]
    measurement_min_values: tuple[float | None, ...]
    measurement_max_values: tuple[float | None, ...]
    measurement_use_minimum: tuple[bool, ...]
    measurement_use_maximum: tuple[bool, ...]
    measurement_tables: tuple[MeasurementTable, ...]
    enclosing_labels: np.ndarray | None
    parent_child_relationship: FilterObjectsParentChildRelationship | None
    parent_child_relationships: FilterObjectsParentChildRelationships
    per_object_assignment: PerObjectAssignment
    min_value: float | None
    max_value: float | None
    use_minimum: bool
    use_maximum: bool

    @property
    def num_objects_pre(self) -> int:
        return len(self.object_ids)

    def measurement_values_for_feature(
        self,
        feature_name: str,
    ) -> ObjectLabelMeasurementValues:
        """Resolve one FilterObjects measurement rule through the nominal source chain."""
        return FilterObjectsMeasurementValuesSource.resolve_feature(self, feature_name)

    def first_measurement_values(self) -> ObjectLabelMeasurementValues:
        if self.measurement_values is not None:
            return self.measurement_values
        if self.measurement_features:
            return self.measurement_values_for_feature(self.measurement_features[0])
        return self.area_measurement_values()

    def area_measurement_values(self) -> ObjectLabelMeasurementValues:
        return ObjectLabelMeasurementValues.from_label_indexed_values(
            self.object_ids,
            DerivedMeasurementValuesStrategy.for_enum_member(
                ObjectShapeMeasurementFeature.AREA
            ).values(self.labels),
        )

    def matching_measurement_rule_ids(self) -> list[int]:
        self.validate_measurement_rule_lengths()
        retained_ids = set(self.object_ids)
        for index, feature_name in enumerate(self.measurement_features):
            keep_ids = FilterObjectsMeasurementLimitWindow(
                values=self.measurement_values_for_feature(feature_name),
                min_value=self.measurement_min_values[index],
                max_value=self.measurement_max_values[index],
                use_minimum=self.measurement_use_minimum[index],
                use_maximum=self.measurement_use_maximum[index],
            )
            retained_ids.intersection_update(keep_ids.retained_ids)
        return sorted(retained_ids)

    def validate_measurement_rule_lengths(self) -> None:
        expected = len(self.measurement_features)
        lengths = {
            len(self.measurement_min_values),
            len(self.measurement_max_values),
            len(self.measurement_use_minimum),
            len(self.measurement_use_maximum),
        }
        if lengths == {expected}:
            return
        raise ValueError("FilterObjects measurement rule kwargs must align by row.")


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


class DerivedMeasurementValuesStrategy(
    EnumKeyedStrategyMixin[ObjectShapeMeasurementFeature],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Derived object-measurement values available from dense labels."""

    __registry_key__ = "strategy_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "feature"
    feature: ClassVar[ObjectShapeMeasurementFeature]
    strategy_label: ClassVar[str | None] = None

    @classmethod
    def for_feature_name(cls, feature_name: str) -> "DerivedMeasurementValuesStrategy | None":
        candidates = ordered_measurement_feature_candidates(
            feature_name,
            dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
        )
        strategies_by_feature_name = {
            normalize_measurement_token(strategy_type.feature.value): strategy_type
            for strategy_type in cls.registered_strategy_types()
        }
        for candidate in candidates:
            strategy_type = strategies_by_feature_name.get(candidate)
            if strategy_type is not None:
                return strategy_type()
        return None

    @abstractmethod
    def values(self, labels: np.ndarray) -> np.ndarray:
        """Return one derived value per object label."""


class AreaDerivedMeasurementValuesStrategy(DerivedMeasurementValuesStrategy):
    """Area is directly measurable from the label geometry."""

    feature = ObjectShapeMeasurementFeature.AREA

    def values(self, labels: np.ndarray) -> np.ndarray:
        return LabelRegionPropertiesBackendStrategy.for_memory_type().measure_2d(
            labels.astype(np.int32, copy=False)
        ).area


class FormFactorDerivedMeasurementValuesStrategy(DerivedMeasurementValuesStrategy):
    """FormFactor can be derived from area/perimeter label geometry."""

    feature = ObjectShapeMeasurementFeature.FORM_FACTOR

    def values(self, labels: np.ndarray) -> np.ndarray:
        label_ids = np.arange(1, int(labels.max()) + 1, dtype=np.int32)
        if label_ids.size == 0:
            return np.array([], dtype=float)
        return form_factor_values(labels.astype(np.int32, copy=False), label_ids)


class FilterObjectsMeasurementValuesSource(ABC, metaclass=AutoRegisterMeta):
    """MRO-ordered FilterObjects measurement-value source chain."""

    __registry_key__ = "source_label"
    __skip_if_no_key__ = True
    source_label: ClassVar[str | None] = None

    @classmethod
    def active_source_type(cls) -> type["FilterObjectsMeasurementValuesSource"]:
        """Return the most-derived registered source; MRO defines precedence."""
        return max(
            cls.__registry__.values(),
            key=lambda source_type: len(source_type.__mro__),
        )

    @classmethod
    def resolve_feature(
        cls,
        request: FilterObjectsSelectionRequest,
        feature_name: str,
    ) -> ObjectLabelMeasurementValues:
        if cls is FilterObjectsMeasurementValuesSource:
            return cls.active_source_type().resolve_feature(request, feature_name)
        values = measurement_values_for_feature(
            request.measurement_tables,
            feature_name,
            object_count=request.num_objects_pre,
            object_ids=request.object_ids,
            dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
        )
        return ObjectLabelMeasurementValues(request.object_ids, values)


class DerivedFilterObjectsMeasurementValuesSource(FilterObjectsMeasurementValuesSource):
    """Resolve label-intrinsic measurements before falling back to tables."""

    source_label = "derived"

    @classmethod
    def resolve_feature(
        cls,
        request: FilterObjectsSelectionRequest,
        feature_name: str,
    ) -> ObjectLabelMeasurementValues:
        strategy = DerivedMeasurementValuesStrategy.for_feature_name(feature_name)
        if strategy is not None:
            return ObjectLabelMeasurementValues.from_label_indexed_values(
                request.object_ids,
                strategy.values(request.labels),
            )
        return super().resolve_feature(request, feature_name)


class TableFilterObjectsMeasurementValuesSource(
    DerivedFilterObjectsMeasurementValuesSource
):
    """Resolve explicit measurement tables before label-derived fallbacks."""

    source_label = "measurement_table"

    @classmethod
    def resolve_feature(
        cls,
        request: FilterObjectsSelectionRequest,
        feature_name: str,
    ) -> ObjectLabelMeasurementValues:
        if request.measurement_tables:
            value_index = optional_measurement_value_index(
                request.measurement_tables,
                feature_name,
                dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
            )
            if value_index is not None:
                values_by_label, positional_values = value_index
                if values_by_label:
                    return ObjectLabelMeasurementValues.from_value_mapping(
                        request.object_ids,
                        values_by_label,
                    )
                if positional_values:
                    return ObjectLabelMeasurementValues.from_positional_values(
                        request.object_ids,
                        positional_values,
                    )
        return super().resolve_feature(request, feature_name)


class RelationshipChildCountFilterObjectsMeasurementValuesSource(
    TableFilterObjectsMeasurementValuesSource
):
    """Resolve Children_* measurement rules from parent-child relationships."""

    source_label = "relationship_child_count"

    @classmethod
    def resolve_feature(
        cls,
        request: FilterObjectsSelectionRequest,
        feature_name: str,
    ) -> ObjectLabelMeasurementValues:
        child_name = child_count_feature_child_name(feature_name)
        if child_name is not None:
            for relationship in request.parent_child_relationships:
                parent_ids = cls.parent_ids_for_child(relationship, child_name)
                if parent_ids is None:
                    continue
                counts_by_parent_id: dict[int, float] = {
                    object_id: 0.0 for object_id in request.object_ids
                }
                for parent_id in parent_ids:
                    if parent_id in counts_by_parent_id:
                        counts_by_parent_id[parent_id] += 1.0
                return ObjectLabelMeasurementValues.from_value_mapping(
                    request.object_ids,
                    counts_by_parent_id,
                )
        return super().resolve_feature(request, feature_name)

    @classmethod
    def parent_ids_for_child(
        cls,
        relationship: FilterObjectsParentChildRelationship,
        child_name: str,
    ) -> tuple[int, ...] | None:
        if isinstance(relationship, ObjectRelationship):
            if relationship.target.name != child_name:
                return None
            return FilterObjectsRelationshipEndpointIds(relationship.source_ids).ids
        return FilterObjectsRelationshipEndpointIds(relationship.parent_ids).ids


class FilterSelectionStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal retained-object selection for each FilterObjects behavior."""

    __registry_key__ = "selection_label"
    __skip_if_no_key__ = True
    selection_label: ClassVar[str | None] = None
    selection_key: ClassVar[FilterSelectionKey | None] = None

    @classmethod
    def for_mode_and_method(
        cls,
        mode: FilterMode,
        method: FilterMethod,
    ) -> "FilterSelectionStrategy":
        requested_key = FilterSelectionKey(mode, method)
        for key in requested_key.lookup_candidates():
            strategy_type = cls.__registry__.get(key.label)
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
    selection_label = selection_key.label

    def indexes_to_keep(
        self,
        request: FilterObjectsSelectionRequest,
    ) -> list[int]:
        return self.discard_border_objects(request.labels)

    @staticmethod
    def discard_border_objects(labels: np.ndarray) -> list[int]:
        from scipy import ndimage as ndi

        interior_pixels = ndi.binary_erosion(np.ones_like(labels, dtype=bool))
        border_labels = set(labels[~interior_pixels])
        keep_labels = list(set(labels.ravel()).difference(border_labels))
        if 0 in keep_labels:
            keep_labels.remove(0)
        keep_labels.sort()
        return keep_labels


class LimitsFilterSelectionStrategy(FilterSelectionStrategy):
    """Keep objects whose measurement falls within configured limits."""

    selection_key = FilterSelectionKey(FilterMode.MEASUREMENTS, FilterMethod.LIMITS)
    selection_label = selection_key.label

    def indexes_to_keep(
        self,
        request: FilterObjectsSelectionRequest,
    ) -> list[int]:
        if request.measurement_features:
            return request.matching_measurement_rule_ids()
        values = request.measurement_values
        if values is None:
            values = request.area_measurement_values()
        return FilterObjectsMeasurementLimitWindow(
            values=values,
            min_value=request.min_value,
            max_value=request.max_value,
            use_minimum=request.use_minimum,
            use_maximum=request.use_maximum,
        ).retained_ids


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
            values = request.first_measurement_values()
        return keep_one_object(values, keep_max=keep_max)


class MinimalFilterSelectionStrategy(ExtremumFilterSelectionStrategy):
    """Keep the object with the minimum measurement value."""

    selection_key = FilterSelectionKey(FilterMode.MEASUREMENTS, FilterMethod.MINIMAL)
    selection_label = selection_key.label
    keep_max = False


class MaximalFilterSelectionStrategy(ExtremumFilterSelectionStrategy):
    """Keep the object with the maximum measurement value."""

    selection_key = FilterSelectionKey(FilterMode.MEASUREMENTS, FilterMethod.MAXIMAL)
    selection_label = selection_key.label
    keep_max = True


class PerObjectFilterSelectionStrategy(FilterSelectionStrategy):
    """Keep one child object per enclosing parent object."""

    selection_key: ClassVar[FilterSelectionKey | None] = None

    def indexes_to_keep(
        self,
        request: FilterObjectsSelectionRequest,
    ) -> list[int]:
        selection_key = type(self).selection_key
        if selection_key is None or selection_key.method is None:
            raise TypeError("PerObjectFilterSelectionStrategy must define method.")
        values = request.first_measurement_values().dense_label_indexed(
            max_label=int(request.labels.max()) if request.labels.size else 0
        )
        return PerObjectAssignmentStrategy.for_assignment(
            request.per_object_assignment,
        ).indexes_to_keep(
            PerObjectAssignmentRequest(
                child_labels=request.labels,
                enclosing_labels=require_enclosing_labels(request),
                measurement_values=values,
                child_count=request.num_objects_pre,
                keep_max=selection_key.method is FilterMethod.MAXIMAL_PER_OBJECT,
                parent_child_relationship=request.parent_child_relationship,
            )
        )


class MinimalPerObjectFilterSelectionStrategy(PerObjectFilterSelectionStrategy):
    """Fail loudly for minimal-per-parent filtering until relationships exist."""

    selection_key = FilterSelectionKey(
        FilterMode.MEASUREMENTS,
        FilterMethod.MINIMAL_PER_OBJECT,
    )
    selection_label = selection_key.label


class MaximalPerObjectFilterSelectionStrategy(PerObjectFilterSelectionStrategy):
    """Fail loudly for maximal-per-parent filtering until relationships exist."""

    selection_key = FilterSelectionKey(
        FilterMode.MEASUREMENTS,
        FilterMethod.MAXIMAL_PER_OBJECT,
    )
    selection_label = selection_key.label


@dataclass(frozen=True, slots=True)
class PerObjectAssignmentRequest:
    """Inputs for assigning candidate child objects to enclosing parents."""

    child_labels: np.ndarray
    enclosing_labels: np.ndarray
    measurement_values: np.ndarray
    child_count: int
    keep_max: bool
    parent_child_relationship: FilterObjectsParentChildRelationship | None = None

    def __post_init__(self) -> None:
        if self.child_labels.shape != self.enclosing_labels.shape:
            raise ValueError(
                "FilterObjects per-object child and enclosing labels must have "
                f"matching shape, got {self.child_labels.shape} and "
                f"{self.enclosing_labels.shape}."
            )


class PerObjectAssignmentStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal parent-assignment strategy for per-object filtering."""

    __registry_key__ = "assignment_label"
    __skip_if_no_key__ = True
    assignment_label: ClassVar[str | None] = None
    assignment: ClassVar[PerObjectAssignment | None] = None

    @classmethod
    def for_assignment(
        cls,
        assignment: PerObjectAssignment,
    ) -> "PerObjectAssignmentStrategy":
        strategy_type = cls.__registry__.get(assignment.value)
        if strategy_type is None:
            raise ValueError(
                f"Unsupported FilterObjects per-object assignment "
                f"{assignment.value!r}."
            )
        return strategy_type()

    def indexes_to_keep(self, request: PerObjectAssignmentRequest) -> list[int]:
        parent_children = self.parent_children_from_relationship(request) or (
            self.parent_children(request)
        )
        return self.best_child_indexes_by_parent(parent_children, request)

    def parent_children_from_relationship(
        self,
        request: PerObjectAssignmentRequest,
    ) -> dict[int, set[int]]:
        relationship = request.parent_child_relationship
        if relationship is None:
            return {}
        if isinstance(relationship, ObjectRelationship):
            parent_ids = relationship.source_ids
            child_ids = relationship.target_ids
        elif isinstance(relationship, ParentChildRelationshipPayload):
            parent_ids = relationship.parent_ids
            child_ids = relationship.child_ids
        else:
            raise TypeError(
                "FilterObjects parent_child_relationship must be "
                "ObjectRelationship or ParentChildRelationshipPayload, got "
                f"{type(relationship).__name__}."
            )

        parent_children: dict[int, set[int]] = {}
        for parent_id, child_id in zip(
            FilterObjectsRelationshipEndpointIds(parent_ids).ids,
            FilterObjectsRelationshipEndpointIds(child_ids).ids,
            strict=True,
        ):
            if parent_id > 0 and child_id > 0:
                parent_children.setdefault(parent_id, set()).add(child_id)
        return parent_children

    def best_child_indexes_by_parent(
        self,
        parent_children: dict[int, set[int]],
        request: PerObjectAssignmentRequest,
    ) -> list[int]:
        selected: set[int] = set()
        for child_ids in parent_children.values():
            child_values = tuple(
                (
                    child_id,
                    self.measurement_value_for_child(
                        request.measurement_values,
                        child_id,
                    ),
                )
                for child_id in child_ids
            )
            finite_child_values = tuple(
                (child_id, value)
                for child_id, value in child_values
                if np.isfinite(value)
            )
            if not finite_child_values:
                continue
            selected.add(
                min(
                    finite_child_values,
                    key=(
                        (lambda item: (-item[1], item[0]))
                        if request.keep_max
                        else (lambda item: (item[1], item[0]))
                    ),
                )[0]
            )
        return sorted(selected)

    @staticmethod
    def measurement_value_for_child(
        measurement_values: np.ndarray,
        child_id: int,
    ) -> float:
        value_index = child_id - 1
        if value_index < 0 or value_index >= len(measurement_values):
            return float("nan")
        return float(measurement_values[value_index])

    @staticmethod
    def overlap_label_pairs(
        request: PerObjectAssignmentRequest,
    ) -> tuple[tuple[int, int], ...]:
        overlap_mask = (request.child_labels > 0) & (request.enclosing_labels > 0)
        child_ids = request.child_labels[overlap_mask].astype(np.int64, copy=False)
        parent_ids = request.enclosing_labels[overlap_mask].astype(np.int64, copy=False)
        return tuple(
            (int(child_id), int(parent_id))
            for child_id, parent_id in zip(child_ids, parent_ids, strict=True)
        )

    @abstractmethod
    def parent_children(
        self,
        request: PerObjectAssignmentRequest,
    ) -> dict[int, set[int]]:
        """Return child labels eligible for each enclosing parent label."""


class BothParentsAssignmentStrategy(PerObjectAssignmentStrategy):
    """Assign an overlapping child as a candidate for every touched parent."""

    assignment = PerObjectAssignment.BOTH_PARENTS
    assignment_label = assignment.value

    def indexes_to_keep(self, request: PerObjectAssignmentRequest) -> list[int]:
        return best_child_indexes_both_parents(
            request.child_labels,
            request.enclosing_labels,
            request.measurement_values,
            request.keep_max,
        )

    def parent_children(
        self,
        request: PerObjectAssignmentRequest,
    ) -> dict[int, set[int]]:
        parent_children: dict[int, set[int]] = {}
        for child_id, parent_id in self.overlap_label_pairs(request):
            parent_children.setdefault(parent_id, set()).add(child_id)
        return parent_children


class ParentWithMostOverlapAssignmentStrategy(PerObjectAssignmentStrategy):
    """Assign each child only to its most-overlapped enclosing parent."""

    assignment = PerObjectAssignment.PARENT_WITH_MOST_OVERLAP
    assignment_label = assignment.value

    def indexes_to_keep(self, request: PerObjectAssignmentRequest) -> list[int]:
        parent_children = self.parent_children_from_relationship(request)
        if parent_children:
            return self.best_child_indexes_by_parent(
                parent_children,
                request,
            )
        return best_child_indexes_parent_with_most_overlap(
            request.child_labels,
            request.enclosing_labels,
            request.measurement_values,
            request.keep_max,
        )

    def parent_children(
        self,
        request: PerObjectAssignmentRequest,
    ) -> dict[int, set[int]]:
        counts_by_child: dict[int, dict[int, int]] = {}
        for child_id, parent_id in self.overlap_label_pairs(request):
            parent_counts = counts_by_child.setdefault(child_id, {})
            parent_counts[parent_id] = parent_counts.get(parent_id, 0) + 1

        parent_children: dict[int, set[int]] = {}
        for child_id, parent_counts in counts_by_child.items():
            parent_id = min(
                parent_counts,
                key=lambda candidate: (-parent_counts[candidate], candidate),
            )
            parent_children.setdefault(parent_id, set()).add(child_id)
        return parent_children


def keep_one_object(
    values: ObjectLabelMeasurementValues,
    keep_max: bool = True,
) -> list[int]:
    """Keep only the object with the maximum or minimum finite measurement."""
    selected_id = values.extremum_id(keep_max=keep_max)
    return [] if selected_id is None else [selected_id]


def require_enclosing_labels(
    request: FilterObjectsSelectionRequest,
) -> np.ndarray:
    if request.enclosing_labels is not None:
        return request.enclosing_labels
    raise ValueError(
        "FilterObjects per-object filtering requires enclosing object labels."
    )


def best_child_indexes_both_parents(
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


def best_child_indexes_parent_with_most_overlap(
    child_labels: np.ndarray,
    enclosing_labels: np.ndarray,
    measurement_values: np.ndarray,
    keep_max: bool,
) -> list[int]:
    max_child = int(np.max(child_labels))
    if max_child <= 0:
        return []
    child_ids, parent_ids, overlap_counts = unique_overlap_counts(
        child_labels,
        enclosing_labels,
    )
    if child_ids.size == 0:
        return []
    return selected_labels_to_list(
        _best_child_selected_mask_parent_with_most_overlap_numba(
            np.ascontiguousarray(child_ids, dtype=np.int32),
            np.ascontiguousarray(parent_ids, dtype=np.int32),
            np.ascontiguousarray(overlap_counts, dtype=np.int64),
            np.ascontiguousarray(measurement_values, dtype=np.float64),
            max_child,
            bool(keep_max),
        )
    )


def unique_overlap_counts(
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


def selected_labels_to_list(selected: np.ndarray) -> list[int]:
    return np.flatnonzero(np.asarray(selected, dtype=bool)).astype(int).tolist()


__all__ = [
    "FilterMethod",
    "FilterMode",
    "FilterObjectsMeasurementValuesSource",
    "FilterObjectsLabelPlane",
    "FilterObjectsMeasurementLimitWindow",
    "FilterObjectsParentChildRelationship",
    "FilterObjectsParentChildRelationships",
    "FilterObjectsRelationshipEndpointIds",
    "FilterObjectsSelectionRequest",
    "FilterObjectsStats",
    "FilterSelectionStrategy",
    "FilterSelectionKey",
    "PerObjectAssignment",
    "PerObjectAssignmentRequest",
    "PerObjectAssignmentStrategy",
    "best_child_indexes_both_parents",
    "best_child_indexes_parent_with_most_overlap",
    "keep_one_object",
    "require_enclosing_labels",
]
