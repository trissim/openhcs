"""Relationship measurement row authorities for CellProfiler runtime outputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import time
from typing import TYPE_CHECKING

import numpy as np

from openhcs.core.artifacts import ArtifactKind, ArtifactSpec
from openhcs.core.measurement_row_materialization import (
    MEASUREMENT_OBJECT_LABEL_FIELD,
    MEASUREMENT_OBJECT_NAME_FIELD,
)
from openhcs.core.registry_strategies import GeneratedEnumClassSpec
from openhcs.core.runtime_semantics import (
    MeasurementRowAxisField,
    ObjectLabelDomainMetadataStrategy,
    ObjectLabelDomainScope,
    ParentChildRelationshipPayload,
    dense_object_label_id_domain,
)
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValues
from openhcs.core.runtime_slice_projection import RuntimeSliceProjection
from openhcs.core.runtime_values import (
    ObjectLabelRuntimeSliceStackContract,
    object_label_dense_array,
)
from openhcs.interop.cellprofiler.measurement_lookup import (
    CellProfilerMeasurementFeature,
)
from openhcs.interop.cellprofiler.relationship_measurements import (
    RelationshipMeasurements,
)
from openhcs.interop.cellprofiler.runtime.mapping_lookup import MappingValueLookup
from openhcs.interop.cellprofiler.runtime.measurement_rows import (
    CellProfilerMeasurementRowProjection,
    FormattingMeasurementFeatureTemplate,
    LABEL_PAYLOAD_FINAL,
    ObjectLabelCountAuthority,
)
from openhcs.interop.cellprofiler.runtime.payload_types import (
    CellProfilerRuntimeValue,
    CellProfilerRuntimeValueSequence,
)
from openhcs.interop.cellprofiler.runtime.relationship_endpoints import (
    RelationshipEndpointResolver,
)
from openhcs.interop.cellprofiler.runtime.runtime_profile import (
    CellProfilerRuntimeProfileLogger,
)

if TYPE_CHECKING:
    from openhcs.interop.cellprofiler.runtime.module_execution import (
        CellProfilerOutputRecordRequest,
    )

RelationshipMeasurementValue = int | str | float


class RelationshipMeasurementRow(dict[str, RelationshipMeasurementValue]):
    """Nominal CellProfiler relationship measurement row."""

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, RelationshipMeasurementValue],
    ) -> "RelationshipMeasurementRow":
        return cls(values)

    def with_axis(self, *, slice_index: int | None) -> "RelationshipMeasurementRow":
        if slice_index is None:
            return self
        return type(self)(
            {
                **self,
                MeasurementRowAxisField.SLICE_INDEX.value: slice_index,
            }
        )


RelationshipMeasurementRowList = list[RelationshipMeasurementRow]
RelationshipDistanceRowTuple = tuple[RelationshipMeasurementRow, ...]
RelationshipObjectPair = tuple[int, int]
RelationshipObjectPairs = tuple[RelationshipObjectPair, ...]
RelationshipObjectPairsBySlice = tuple[tuple[int, RelationshipObjectPairs], ...]


for _measurement_feature_template_spec in (
    GeneratedEnumClassSpec(
        class_name="CellProfilerRelationshipMeasurementFeature",
        base_type=FormattingMeasurementFeatureTemplate,
        members={
            "PARENT": "Parent_{parent_object_name}",
        },
    ),
):
    _measurement_feature_template_spec.declare_in(globals())


@dataclass(frozen=True, slots=True)
class RelationshipMeasurementRows(
    CellProfilerMeasurementRowProjection,
):
    """Project parent-child relationship payloads into CP object measurement rows."""

    request: CellProfilerOutputRecordRequest
    _object_number_cache: dict[
        tuple[str, int | None, int | None],
        dict[int, int],
    ] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    @classmethod
    def for_request(
        cls,
        request: CellProfilerOutputRecordRequest,
    ) -> "RelationshipMeasurementRows":
        from openhcs.processing.backends.cellprofiler.module_classes import (
            CellProfilerModule,
        )

        module_type = CellProfilerModule.for_module(request.module_name)
        if module_type is None:
            return GenericRelationshipMeasurementRows(request)
        rows = module_type.relationship_measurement_rows(request)
        if not isinstance(rows, RelationshipMeasurementRows):
            raise TypeError(
                f"{module_type.__name__}.relationship_measurement_rows must return "
                "RelationshipMeasurementRows."
            )
        return rows

    def rows(self) -> RelationshipMeasurementRowList:
        rows_started_at = time.perf_counter()
        rows: RelationshipMeasurementRowList = []
        endpoint_resolver = RelationshipEndpointResolver.for_request(self.request)
        for relationship_spec, payload in self.output_entries():
            parent_spec, child_spec = endpoint_resolver.endpoint_specs(
                relationship_spec
            )
            rows.extend(
                self.child_count_rows(
                    parent_object_name=parent_spec.name,
                    child_object_name=child_spec.name,
                    payload=payload,
                )
            )
            rows.extend(
                self.parent_rows(
                    parent_object_name=parent_spec.name,
                    child_object_name=child_spec.name,
                    payload=payload,
                )
            )
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "relationship_measurement_rows",
            time.perf_counter() - rows_started_at,
            module=self.request.module_name,
            rows=len(rows),
        )
        return rows

    def output_entries(
        self,
    ) -> tuple[tuple[ArtifactSpec, ParentChildRelationshipPayload], ...]:
        entries = tuple(
            (spec, value)
            for spec in self.request.declared_outputs
            if spec.kind is ArtifactKind.RELATIONSHIPS
            for value in (self.request.output_values.get(spec.name),)
            if isinstance(value, ParentChildRelationshipPayload)
        )
        return entries

    def child_count_rows(
        self,
        *,
        parent_object_name: str,
        child_object_name: str,
        payload: ParentChildRelationshipPayload,
    ) -> tuple[RelationshipMeasurementRow, ...]:
        rows_started_at = time.perf_counter()
        sliced_pairs = self.payload_pairs_by_slice(
            payload,
            child_object_name=child_object_name,
        )
        if sliced_pairs is not None:
            slice_count = len(sliced_pairs)
            rows: RelationshipMeasurementRowList = []
            for slice_index, pairs in sliced_pairs:
                related_parent_ids = tuple(parent_id for parent_id, _child_id in pairs)
                rows.extend(
                    self.child_count_rows_for_ids(
                        parent_object_name=parent_object_name,
                        child_object_name=child_object_name,
                        related_parent_ids=related_parent_ids,
                        slice_index=slice_index,
                        slice_count=slice_count,
                    )
                )
            result = tuple(rows)
        else:
            result = self.child_count_rows_for_ids(
                parent_object_name=parent_object_name,
                child_object_name=child_object_name,
                related_parent_ids=tuple(
                    int(parent_id) for parent_id in payload.parent_ids
                ),
                slice_index=None,
            )
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "relationship_child_count_rows",
            time.perf_counter() - rows_started_at,
            module=self.request.module_name,
            parent=parent_object_name,
            child=child_object_name,
            rows=len(result),
        )
        return result

    def child_count_rows_for_ids(
        self,
        *,
        parent_object_name: str,
        child_object_name: str,
        related_parent_ids: tuple[int, ...],
        slice_index: int | None,
        slice_count: int | None = None,
    ) -> tuple[RelationshipMeasurementRow, ...]:
        related_parent_ids = tuple(int(parent_id) for parent_id in related_parent_ids)
        parent_numbers = self.object_numbers_by_label_id(
            parent_object_name,
            slice_index=slice_index,
            slice_count=slice_count,
        )
        counts = {parent_number: 0 for parent_number in parent_numbers.values()}
        for parent_id in related_parent_ids:
            parent_number = parent_numbers.get(parent_id)
            if parent_number is not None:
                counts[parent_number] = (
                    MappingValueLookup(counts, parent_number).value_or(0) + 1
                )
        feature_name = CellProfilerMeasurementFeature.child_count(
            child_object_name
        ).name
        return tuple(
            self.axis_qualified_row(
                {
                    MEASUREMENT_OBJECT_NAME_FIELD: parent_object_name,
                    MEASUREMENT_OBJECT_LABEL_FIELD: parent_id,
                    feature_name: count,
                },
                slice_index=slice_index,
            )
            for parent_id, count in counts.items()
        )

    def parent_rows(
        self,
        *,
        parent_object_name: str,
        child_object_name: str,
        payload: ParentChildRelationshipPayload,
    ) -> tuple[RelationshipMeasurementRow, ...]:
        rows_started_at = time.perf_counter()
        sliced_pairs = self.payload_pairs_by_slice(
            payload,
            child_object_name=child_object_name,
        )
        if sliced_pairs is not None:
            slice_count = len(sliced_pairs)
            rows: list[dict[str, int | str]] = []
            for slice_index, pairs in sliced_pairs:
                rows.extend(
                    self.parent_rows_for_pairs(
                        parent_object_name=parent_object_name,
                        child_object_name=child_object_name,
                        pairs=pairs,
                        slice_index=slice_index,
                        slice_count=slice_count,
                    )
                )
            result = tuple(rows)
        else:
            result = self.parent_rows_for_pairs(
                parent_object_name=parent_object_name,
                child_object_name=child_object_name,
                pairs=tuple(
                    (int(parent_id), int(child_id))
                    for parent_id, child_id in zip(
                        payload.parent_ids,
                        payload.child_ids,
                        strict=True,
                    )
                ),
                slice_index=None,
            )
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "relationship_parent_rows",
            time.perf_counter() - rows_started_at,
            module=self.request.module_name,
            parent=parent_object_name,
            child=child_object_name,
            rows=len(result),
        )
        return result

    def parent_rows_for_pairs(
        self,
        *,
        parent_object_name: str,
        child_object_name: str,
        pairs: tuple[tuple[int, int], ...],
        slice_index: int | None,
        slice_count: int | None = None,
    ) -> tuple[RelationshipMeasurementRow, ...]:
        parent_numbers = self.object_numbers_by_label_id(
            parent_object_name,
            slice_index=slice_index,
            slice_count=slice_count,
        )
        child_numbers = self.object_numbers_by_label_id(
            child_object_name,
            slice_index=slice_index,
            slice_count=slice_count,
        )
        parent_by_child = {}
        for parent_id, child_id in pairs:
            child_number = child_numbers.get(int(child_id))
            parent_number = parent_numbers.get(int(parent_id))
            if child_number is not None and parent_number is not None:
                parent_by_child[child_number] = parent_number
        feature_name = CellProfilerRelationshipMeasurementFeature.PARENT.feature_name(
            parent_object_name=parent_object_name
        )
        return tuple(
            self.axis_qualified_row(
                {
                    MEASUREMENT_OBJECT_NAME_FIELD: child_object_name,
                    MEASUREMENT_OBJECT_LABEL_FIELD: child_id,
                    feature_name: MappingValueLookup(
                        parent_by_child,
                        child_id,
                    ).value_or(0),
                },
                slice_index=slice_index,
            )
            for child_id in child_numbers.values()
        )

    def object_numbers_by_label_id(
        self,
        object_name: str,
        *,
        slice_index: int | None,
        slice_count: int | None = None,
    ) -> dict[int, int]:
        cache_key = (object_name, slice_index, slice_count)
        cached = self._object_number_cache.get(cache_key)
        if cached is not None:
            return cached
        started_at = time.perf_counter()
        labels = self.object_label_domain_value(
            object_name,
            slice_index=slice_index,
            slice_count=slice_count,
        )
        if labels is None:
            label_ids = ()
        else:
            label_ids = dense_object_label_id_domain(labels)
        numbers = {
            int(label_id): object_number
            for object_number, label_id in enumerate(label_ids, start=1)
        }
        self._object_number_cache[cache_key] = numbers
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "relationship_object_numbers_by_label_id",
            time.perf_counter() - started_at,
            module=self.request.module_name,
            object=object_name,
            slice_index=slice_index,
            ids=len(numbers),
        )
        return numbers

    def object_label_domain_value(
        self,
        object_name: str,
        *,
        slice_index: int | None = None,
        slice_count: int | None = None,
    ) -> CellProfilerRuntimeValue | None:
        labels = self.unprojected_object_labels(object_name)
        if labels is None:
            return None
        if slice_index is None:
            return RuntimeSliceProjection.object_label_endpoint(labels)
        context = RuntimeSliceProjection.context_for_value(
            labels,
            slice_index=slice_index,
            slice_count=slice_count,
            source_description=f"object labels {object_name!r}",
        )
        return RuntimeSliceProjection.object_label_endpoint(
            labels,
            context=context,
        )

    def object_label_count(
        self,
        object_name: str,
        *,
        slice_index: int | None = None,
    ) -> int:
        if object_name in self.request.output_values:
            return ObjectLabelCountAuthority.count_from_value(
                self.request.output_values[object_name],
                slice_index=slice_index,
            )
        return ObjectLabelCountAuthority.count_from_adapter(
            self.request.adapter,
            object_name,
            slice_index=slice_index,
        )

    def payload_pairs_by_slice(
        self,
        payload: ParentChildRelationshipPayload,
        *,
        child_object_name: str,
    ) -> RelationshipObjectPairsBySlice | None:
        if payload.slice_count is None and not payload.slice_indices:
            measurement_sliced_pairs = self._payload_pairs_by_measurement_slice(
                payload,
                child_object_name=child_object_name,
            )
            if measurement_sliced_pairs is not None:
                return measurement_sliced_pairs
            return self._payload_pairs_by_child_label_slices(
                payload,
                child_object_name=child_object_name,
            )
        if payload.slice_count is None:
            slice_count = 0
            if payload.slice_indices:
                slice_count = max(payload.slice_indices) + 1
        else:
            slice_count = payload.slice_count
        pairs_by_slice: list[list[RelationshipObjectPair]] = [
            [] for _ in range(slice_count)
        ]
        if payload.slice_indices:
            for slice_index, parent_id, child_id in zip(
                payload.slice_indices,
                payload.parent_ids,
                payload.child_ids,
                strict=True,
            ):
                pairs_by_slice[slice_index].append((parent_id, child_id))
        elif payload.parent_ids:
            if slice_count != 1:
                raise ValueError(
                    "ParentChildRelationshipPayload with multiple slices must carry "
                    "slice_indices for non-empty relationships."
                )
            pairs_by_slice[0].extend(
                zip(payload.parent_ids, payload.child_ids, strict=True)
            )
        return tuple(
            (slice_index, tuple(pairs))
            for slice_index, pairs in enumerate(pairs_by_slice)
        )

    def _payload_pairs_by_measurement_slice(
        self,
        payload: ParentChildRelationshipPayload,
        *,
        child_object_name: str,
    ) -> RelationshipObjectPairsBySlice | None:
        child_label_slice_count = self._object_label_stack_slice_count(
            child_object_name,
        )
        if child_label_slice_count is None:
            return None
        measurement_payloads = tuple(
            measurement
            for value in (
                self.request.output_value,
                *self.request.output_values.values(),
            )
            for measurement in (
                CellProfilerRelationshipMeasurementPayloads
                .from_value(value)
                .values
            )
        )
        slice_indices = {
            int(measurement.slice_index)
            for measurement in measurement_payloads
        }
        if len(slice_indices) != 1:
            return None
        slice_index = next(iter(slice_indices))
        slice_count = max(child_label_slice_count, slice_index + 1)
        pairs_by_slice: list[list[RelationshipObjectPair]] = [
            [] for _slice_index in range(slice_count)
        ]
        pairs_by_slice[slice_index].extend(
            (int(parent_id), int(child_id))
            for parent_id, child_id in zip(
                payload.parent_ids,
                payload.child_ids,
                strict=True,
            )
        )
        return tuple(
            (index, tuple(pairs))
            for index, pairs in enumerate(pairs_by_slice)
        )

    def _object_label_stack_slice_count(self, object_name: str) -> int | None:
        labels = self.unprojected_object_labels(object_name)
        if labels is None:
            return None
        return ObjectLabelRuntimeSliceStackContract.runtime_slice_count(labels)

    def _payload_pairs_by_child_label_slices(
        self,
        payload: ParentChildRelationshipPayload,
        *,
        child_object_name: str,
    ) -> RelationshipObjectPairsBySlice | None:
        child_value = self.unprojected_object_labels(child_object_name)
        if child_value is None:
            return None
        child_domain = ObjectLabelDomainMetadataStrategy.for_value(
            child_value,
        ).object_label_domain(child_value)
        if child_domain.scope is not ObjectLabelDomainScope.PLANE:
            return None
        child_labels = LABEL_PAYLOAD_FINAL.value(child_value)
        label_stack = object_label_dense_array(child_labels, dtype=np.int32)
        if label_stack.ndim != 3:
            return None
        pairs_by_slice: list[list[RelationshipObjectPair]] = [
            [] for _slice_index in range(int(label_stack.shape[0]))
        ]
        parent_by_child = {
            int(child_id): int(parent_id)
            for parent_id, child_id in zip(
                payload.parent_ids,
                payload.child_ids,
                strict=True,
            )
        }
        for slice_index, label_plane in enumerate(label_stack):
            child_ids = np.unique(label_plane[label_plane > 0])
            for child_id in child_ids:
                parent_id = parent_by_child.get(int(child_id))
                if parent_id is not None:
                    pairs_by_slice[slice_index].append((parent_id, int(child_id)))
        return tuple(
            (slice_index, tuple(pairs))
            for slice_index, pairs in enumerate(pairs_by_slice)
        )

    @staticmethod
    def axis_qualified_row(
        row: Mapping[str, RelationshipMeasurementValue],
        *,
        slice_index: int | None,
    ) -> RelationshipMeasurementRow:
        return RelationshipMeasurementRow.from_mapping(row).with_axis(
            slice_index=slice_index
        )

    def object_labels(
        self,
        object_name: str,
        *,
        slice_index: int | None = None,
        slice_count: int | None = None,
    ) -> CellProfilerRuntimeValue | None:
        labels = self.object_label_domain_value(
            object_name,
            slice_index=slice_index,
            slice_count=slice_count,
        )
        if labels is None:
            return None
        return LABEL_PAYLOAD_FINAL.value(labels)

    def raw_object_labels(self, object_name: str) -> CellProfilerRuntimeValue | None:
        value = self.unprojected_object_labels(object_name)
        if value is None:
            return None
        return LABEL_PAYLOAD_FINAL.value(value)

    def unprojected_object_labels(self, object_name: str) -> CellProfilerRuntimeValue | None:
        value = self.request.output_values.get(object_name)
        if value is None:
            value = self.request.adapter.get_objects(object_name)
        return value


class GenericRelationshipMeasurementRows(RelationshipMeasurementRows):
    """Default relationship rows: child counts plus parent ids."""


@dataclass(frozen=True, slots=True)
class CellProfilerRelationshipMeasurementPayloads:
    """Relationship measurement payloads carried by one output record."""

    values: tuple[RelationshipMeasurements, ...]

    @classmethod
    def from_value(cls, value: CellProfilerRuntimeValue) -> "CellProfilerRelationshipMeasurementPayloads":
        """Normalize scalar or slice-aligned relationship measurement payloads."""
        match value:
            case RelationshipMeasurements() as measurements:
                return cls((measurements,))
            case RuntimeSliceAlignedValues(slices=slices):
                return cls.from_values(slices)
            case tuple() | list() as values:
                return cls.from_values(values)
            case _:
                return cls(())

    @classmethod
    def from_values(
        cls,
        values: CellProfilerRuntimeValueSequence,
    ) -> "CellProfilerRelationshipMeasurementPayloads":
        """Normalize a sequence of candidate relationship measurement payloads."""
        return cls(
            tuple(
                value
                for value in values
                if isinstance(value, RelationshipMeasurements)
            )
        )

    @property
    def declares_distance_measurements(self) -> bool:
        """Return whether any payload declares distance measurements."""
        return any(value.declares_distance_measurements for value in self.values)
