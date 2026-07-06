"""Object-label measurement batch resolution for CellProfiler runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast
from weakref import WeakKeyDictionary

import numpy as np

from openhcs.core.artifacts import MeasurementsArtifactType, RelationshipsArtifactType
from openhcs.core.runtime_artifact_queries import (
    MeasurementLabelSliceFeatureQuery,
    MeasurementTableAxisProjection,
)
from openhcs.core.measurement_feature_queries import (
    MeasurementFeatureAxisScopeSelection,
    MeasurementFeatureQuery,
)
from openhcs.core.runtime_slice_projection import RuntimeSliceProjection
from openhcs.core.runtime_semantics import (
    MeasurementRowAxisField,
    ObjectLabelMeasurementValues,
    RuntimeObjectImageMeasurementQuery,
    RuntimeObjectLabelMeasurementQuery,
    dense_object_label_id_domain,
    parent_child_relationship_artifact_name,
)
from openhcs.core.runtime_stores import RuntimeValueStore, StoredRuntimeValue
from openhcs.core.runtime_values import (
    MeasurementTable,
    ObjectRelationship,
)
from openhcs.interop.cellprofiler.measurement_lookup import (
    child_count_feature_child_name,
)
from openhcs.interop.cellprofiler.measurement_dialect import (
    CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
)
from openhcs.interop.cellprofiler.runtime.adapter_profile import AdapterProfileLog
from openhcs.interop.cellprofiler.runtime.adapter_scope import (
    CellProfilerRuntimeScope,
    MultiplaneObjectMeasurementTableCacheKey,
    RuntimeGroupMatchScope,
)
from openhcs.interop.cellprofiler.runtime.payload_types import (
    CellProfilerMeasurementVector,
    DenseLabelPayload,
)
from openhcs.interop.cellprofiler.runtime.runtime_value_authorities import (
    DenseLabelStackRepeatPattern,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_tables import (
    CellProfilerMeasurementSliceValues,
    MeasurementQueryCacheInvalidationPolicy,
    MeasurementTablesByObject,
    MutableMeasurementTablesByObject,
    object_measurement_table_index,
    object_measurement_tables_for_object,
)
from openhcs.interop.cellprofiler.runtime.runtime_artifact_records import (
    RuntimeArtifactRecordResolver,
)
from openhcs.interop.cellprofiler.runtime.source_identity import (
    CellProfilerCurrentImage,
    RuntimeRecordSourceImageSetSelector,
)

ObjectLabelMeasurementValuesProcessCache = dict[
    RuntimeObjectLabelMeasurementQuery,
    CellProfilerMeasurementSliceValues,
]

_OBJECT_LABEL_MEASUREMENT_VALUES_PROCESS_CACHE: WeakKeyDictionary[
    RuntimeValueStore,
    ObjectLabelMeasurementValuesProcessCache,
] = WeakKeyDictionary()


def object_label_measurement_values_cache(
    store: RuntimeValueStore,
) -> ObjectLabelMeasurementValuesProcessCache:
    """Return the process cache for label-aligned object-feature vectors."""
    cache = _OBJECT_LABEL_MEASUREMENT_VALUES_PROCESS_CACHE.get(store)
    if cache is None:
        cache = {}
        _OBJECT_LABEL_MEASUREMENT_VALUES_PROCESS_CACHE[store] = cache
    return cache


class ObjectLabelMeasurementValuesCacheInvalidationPolicy(
    MeasurementQueryCacheInvalidationPolicy
):
    """Invalidate label-aligned measurement vector cache entries touched by a write."""

    policy_name = "object_label_measurement_values"
    entry_type = RuntimeObjectLabelMeasurementQuery
    feature_scoped = True
    store_cache_accessor = staticmethod(object_label_measurement_values_cache)


@dataclass(frozen=True, slots=True, kw_only=True)
class ObjectFeatureMeasurementContext(RuntimeObjectImageMeasurementQuery):
    """Nominal context for one object-feature measurement lookup."""

    current_image: CellProfilerCurrentImage | None = None

    @property
    def feature_query(self) -> MeasurementFeatureQuery:
        """Return the CellProfiler feature query for this object context."""
        return MeasurementFeatureQuery(
            self.feature_name,
            object_name=self.object_name,
            dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
        )

    @property
    def table_scope_current_image(self) -> CellProfilerCurrentImage | None:
        """Return ambient image scope only when the feature is not source-qualified."""
        if self.feature_query.source_candidates:
            return None
        return self.current_image

    @property
    def table_scope_image_number(self) -> int | None:
        """Return an image-number axis only when it belongs to the feature scope."""
        if self.feature_query.source_candidates:
            return None
        return self.image_number

    def measurement_tables(
        self,
        adapter: "CellProfilerRuntimeAdapter",
        *,
        match_group: bool = True,
    ) -> tuple[MeasurementTable, ...]:
        """Return adapter-visible tables for this object-feature context."""
        object_table_index = object_measurement_table_index(
            adapter,
            group_key=self.group_key,
            match_group=match_group,
            current_image=self.table_scope_current_image,
        )
        tables = object_table_index.for_object_feature(
            self.object_name,
            self.feature_name,
        )
        if tables is not None:
            return tables
        return object_measurement_tables_for_object(
            adapter,
            self.object_name,
            group_key=self.group_key,
            match_group=match_group,
            current_image=self.table_scope_current_image,
        )

    def axis_scope_measurement_tables(
        self,
        adapter: "CellProfilerRuntimeAdapter",
    ) -> tuple[MeasurementTable, ...]:
        """Return feature-bearing tables in their declared runtime axis scope."""
        scoped_table_sets = (
            self.measurement_tables(adapter),
            self.measurement_tables(adapter, match_group=False),
        )
        runtime_slice_plane_index = adapter.runtime_slice_plane_index()
        candidates: list[tuple[MeasurementTable, ...]] = []
        for tables in scoped_table_sets:
            image_number = self.table_scope_image_number
            if image_number is not None:
                candidates.append(
                    MeasurementTableAxisProjection(
                        MeasurementRowAxisField.IMAGE_NUMBER,
                        image_number,
                    ).tables(tables)
                )
            if runtime_slice_plane_index is not None:
                candidates.append(
                    MeasurementTableAxisProjection(
                        MeasurementRowAxisField.SLICE_INDEX,
                        runtime_slice_plane_index,
                    ).tables(tables)
                )
            candidates.append(tables)
        return MeasurementFeatureAxisScopeSelection(
            candidates=tuple(candidates),
            query=self.feature_query,
            fallback=scoped_table_sets[0],
        ).select()

class RelationshipChildCountUnavailable(RuntimeError):
    """Raised internally when relationship child-count projection is not applicable."""


@dataclass(frozen=True, slots=True)
class RuntimeRelationshipPlaneRecord:
    """Relationship record projected onto one runtime label plane."""

    relationship: ObjectRelationship
    plane_index: int | None = None


@dataclass(frozen=True, slots=True)
class RelationshipPlaneProjectionResolution:
    """Typed result for projecting grouped relationship records onto label planes."""

    records: tuple[RuntimeRelationshipPlaneRecord, ...] = ()
    unavailable_reason: str | None = None

    @property
    def is_available(self) -> bool:
        return self.unavailable_reason is None

    @classmethod
    def from_adapter_records(
        cls,
        adapter: "CellProfilerRuntimeAdapter",
        relationship_name: str,
        records: tuple[StoredRuntimeValue, ...],
        *,
        label_plane_count: int,
    ) -> "RelationshipPlaneProjectionResolution":
        """Project grouped relationship records onto runtime label planes."""
        if not records:
            return cls(unavailable_reason="no_records")
        if len(records) == 1:
            return cls(
                records=(
                    RuntimeRelationshipPlaneRecord(
                        ObjectRelationship.from_runtime_value(records[0].value),
                    ),
                )
            )

        group_order = RelationshipGroupPlaneOrder.for_adapter_relationship(
            adapter,
            relationship_name,
            label_plane_count=label_plane_count,
        )
        if not group_order.is_available:
            unavailable_reason = group_order.unavailable_reason
            if unavailable_reason is None:
                unavailable_reason = "group_order_unavailable"
            return cls(unavailable_reason=unavailable_reason)
        plane_index_by_group_key = group_order.plane_index_by_group_key
        indexed_records: list[tuple[int, StoredRuntimeValue]] = []
        for record in records:
            group_key = record.key.scope.group_key
            plane_index = plane_index_by_group_key.get(group_key)
            if plane_index is None:
                return cls(unavailable_reason="record_group_not_declared")
            indexed_records.append((plane_index, record))
        if len({plane_index for plane_index, _record in indexed_records}) != len(
            indexed_records
        ):
            return cls(unavailable_reason="duplicate_record_plane")
        return cls(
            records=tuple(
                RuntimeRelationshipPlaneRecord(
                    ObjectRelationship.from_runtime_value(record.value),
                    plane_index=plane_index,
                )
                for plane_index, record in sorted(indexed_records)
            )
        )

    def require_records(self) -> tuple[RuntimeRelationshipPlaneRecord, ...]:
        if not self.is_available:
            reason = self.unavailable_reason
            if reason is None:
                reason = "unavailable"
            raise RelationshipChildCountUnavailable(reason)
        return self.records


@dataclass(frozen=True, slots=True)
class RelationshipGroupPlaneOrder:
    """Declared group-key order for relationship runtime-slice projection."""

    plane_index_by_group_key: Mapping[str | None, int] = field(
        default_factory=lambda: MappingProxyType({})
    )
    unavailable_reason: str | None = None

    @property
    def is_available(self) -> bool:
        return self.unavailable_reason is None

    @classmethod
    def for_adapter_relationship(
        cls,
        adapter: "CellProfilerRuntimeAdapter",
        relationship_name: str,
        *,
        label_plane_count: int,
    ) -> "RelationshipGroupPlaneOrder":
        """Return declared runtime-slice group order for a relationship input."""
        input_plan = adapter.artifact_inputs.get(relationship_name)
        if input_plan is None:
            return cls(unavailable_reason="undeclared_relationship_input")
        input_group_keys = input_plan.group_keys
        if input_group_keys is None:
            group_keys = ()
        else:
            group_keys = tuple(input_group_keys)
        if len(group_keys) != label_plane_count:
            return cls(unavailable_reason="group_count_mismatch")
        if len(set(group_keys)) != len(group_keys):
            return cls(unavailable_reason="duplicate_group_keys")
        return cls(
            plane_index_by_group_key=MappingProxyType(
                {
                    group_key: plane_index
                    for plane_index, group_key in enumerate(group_keys)
                }
            )
        )

@dataclass(frozen=True, slots=True, kw_only=True)
class ObjectLabelMeasurementSliceRequest(ObjectFeatureMeasurementContext):
    """Complete context for one label-aligned object-measurement vector query."""

    labels: DenseLabelPayload

    @property
    def label_array(self) -> np.ndarray:
        return np.asarray(self.labels)

    @property
    def label_planes(self) -> tuple[np.ndarray, ...]:
        label_array = self.label_array
        if label_array.ndim <= 2:
            return (label_array,)
        return tuple(label_array[index] for index in range(label_array.shape[0]))

    def label_domain(self) -> tuple[int, ...]:
        """Return the dense label-id domain for this request's label payload."""
        return dense_object_label_id_domain(self.labels)

    def measurement_query(
        self,
        adapter: "CellProfilerRuntimeAdapter",
        *,
        label_domain: tuple[int, ...] | None = None,
    ) -> RuntimeObjectLabelMeasurementQuery:
        """Return the cache/query identity for this label-aligned feature."""
        runtime_scope = RuntimeGroupMatchScope(
            group_key=self.group_key,
        ).runtime_scope(
            adapter,
            current_image=self.current_image,
        )
        return runtime_scope.object_label_measurement_query(
            object_name=self.object_name,
            feature_name=self.feature_name,
            label_domain=(
                self.label_domain() if label_domain is None else label_domain
            ),
            image_number=self.table_scope_image_number,
        )

    def values(
        self,
        adapter: "CellProfilerRuntimeAdapter",
    ) -> CellProfilerMeasurementSliceValues:
        """Return object measurements aligned to this request's label planes."""
        label_array = self.label_array
        label_domain = self.label_domain()
        query = self.measurement_query(adapter, label_domain=label_domain)
        runtime_scope = RuntimeGroupMatchScope(
            group_key=self.group_key,
        ).runtime_scope(
            adapter,
            current_image=self.current_image,
        )
        object_label_values_cache = object_label_measurement_values_cache(
            adapter.runtime_value_store
        )
        cached = adapter._measurement_cache.get(query)
        if cached is None:
            cached = object_label_values_cache.get(query)
            if cached is not None:
                adapter._measurement_cache[query] = cached
        if cached is not None:
            return cached
        values = RelationshipChildCountLabelMeasurement(
            adapter=adapter,
            object_name=self.object_name,
            feature_name=self.feature_name,
            group_key=self.group_key,
            image_number=self.table_scope_image_number,
            current_image=self.current_image,
            labels=self.labels,
        ).values()
        if values is None:
            if label_array.ndim <= 2:
                measurement_context = (
                    ObjectFeatureMeasurementContext(
                        object_name=self.object_name,
                        feature_name=self.feature_name,
                        group_key=self.group_key,
                        image_number=self.table_scope_image_number,
                        current_image=None,
                    )
                    if self.image_number is not None
                    else self
                )
                tables = measurement_context.axis_scope_measurement_tables(adapter)
            else:
                tables = self.multiplane_measurement_tables(
                    adapter,
                    label_domain=label_domain,
                        runtime_scope=runtime_scope,
                    )
            values = MeasurementLabelSliceFeatureQuery(
                measurement_tables=tables,
                feature_name=self.feature_name,
                object_name=self.object_name,
                row_axis=MeasurementRowAxisField.SLICE_INDEX,
                dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
            ).values_for_labels(self.labels)
        adapter._measurement_cache[query] = values
        object_label_values_cache[query] = values
        return values

    def multiplane_measurement_tables(
        self,
        adapter: "CellProfilerRuntimeAdapter",
        *,
        label_domain: tuple[int, ...],
        runtime_scope: CellProfilerRuntimeScope,
    ) -> tuple[MeasurementTable, ...]:
        """Return object tables aligned from producer groups into label planes."""
        label_array = self.label_array
        cache_key = MultiplaneObjectMeasurementTableCacheKey(
            revision=adapter.runtime_value_store.revision,
            axis_identity=runtime_scope.adapter.axis_scope.axis_id,
            object_name=self.object_name,
            feature_name=self.feature_name,
            label_domain=label_domain,
            source_scope=runtime_scope.source_identity_cache_scope,
        )
        cached = adapter._measurement_cache.get(cache_key)
        if cached is not None:
            return cached

        records = RuntimeGroupMatchScope(
            group_key=self.group_key,
            match_group=False,
        ).runtime_scope(
            adapter,
            current_image=self.current_image,
        ).artifact_query_context().find(
            artifact_type=MeasurementsArtifactType,
        )
        scoped_tables: list[tuple[str | None, MeasurementTable]] = []
        feature_query = MeasurementFeatureQuery(
            self.feature_name,
            dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
        )
        for record in records:
            table = MeasurementTable.from_runtime_value(record.value)
            if (
                RuntimeSliceProjection.measurement_table_matches_object(
                    table,
                    self.object_name,
                )
                and feature_query.table_may_carry_feature(table)
            ):
                scoped_tables.append((record.key.scope.group_key, table))

        if not scoped_tables:
            tables = self.measurement_tables(adapter, match_group=False)
            adapter._measurement_cache[cache_key] = tables
            return tables

        group_order = tuple(
            dict.fromkeys(
                group_key for group_key, _table in scoped_tables if group_key is not None
            )
        )
        group_slice_counts = {
            group_key: max(
                RuntimeSliceProjection.measurement_table_effective_slice_count(table)
                for table_group_key, table in scoped_tables
                if table_group_key == group_key
            )
            for group_key in group_order
        }
        if (
            label_array.ndim > 2
            and not group_order
            and len(scoped_tables) == 1
            and RuntimeSliceProjection.measurement_table_effective_slice_count(
                scoped_tables[0][1]
            )
            == 1
            and DenseLabelStackRepeatPattern(label_array).repeats_first_plane
        ):
            tables = (
                RuntimeSliceProjection.measurement_table_broadcast_to_slice_count(
                    scoped_tables[0][1],
                    label_array.shape[0],
                ),
            )
            adapter._measurement_cache[cache_key] = tables
            return tables

        group_offsets: dict[str, int] = {}
        offset = 0
        for group_key in group_order:
            group_offsets[group_key] = offset
            offset += group_slice_counts[group_key]

        tables = tuple(
            (
                RuntimeSliceProjection.measurement_table_with_slice_offset(
                    table,
                    group_offsets[table_group_key],
                )
                if table_group_key is not None
                else table
            )
            for table_group_key, table in scoped_tables
        )
        if offset and label_array.ndim > 2 and offset != label_array.shape[0]:
            AdapterProfileLog.measurement_slice_mismatch(
                0.0,
                object_name=self.object_name,
                measurement_slices=offset,
                label_slices=label_array.shape[0],
            )
        adapter._measurement_cache[cache_key] = tables
        return tables


@dataclass(frozen=True, slots=True, kw_only=True)
class RelationshipChildCountLabelMeasurement(ObjectLabelMeasurementSliceRequest):
    """Resolve label-aligned child-count vectors from relationship artifacts."""

    adapter: "CellProfilerRuntimeAdapter"

    @property
    def child_count_child_name(self) -> str:
        child_name = child_count_feature_child_name(self.feature_name)
        if child_name is None:
            raise RelationshipChildCountUnavailable
        return child_name

    @property
    def child_count_relationship_name(self) -> str:
        return parent_child_relationship_artifact_name(
            self.object_name,
            self.child_count_child_name,
        )

    def values(self) -> tuple[np.ndarray, ...] | None:
        try:
            return self.required_values()
        except RelationshipChildCountUnavailable:
            return None

    def required_values(self) -> tuple[np.ndarray, ...]:
        plane_records = self.valid_plane_records()
        values = tuple(
            self.values_for_plane(slice_index, label_plane, plane_records)
            for slice_index, label_plane in enumerate(self.label_planes)
        )
        if any(value is None for value in values):
            raise RelationshipChildCountUnavailable
        return cast(tuple[np.ndarray, ...], values)

    def valid_plane_records(self) -> tuple[RuntimeRelationshipPlaneRecord, ...]:
        child_name = self.child_count_child_name
        plane_records = self.plane_records()
        if not plane_records:
            raise RelationshipChildCountUnavailable
        if any(
            plane_record.relationship.source.name != self.object_name
            or plane_record.relationship.target.name != child_name
            for plane_record in plane_records
        ):
            raise RelationshipChildCountUnavailable
        return plane_records

    def plane_records(self) -> tuple[RuntimeRelationshipPlaneRecord, ...]:
        return RelationshipPlaneProjectionResolution.from_adapter_records(
            self.adapter,
            self.child_count_relationship_name,
            self.selected_records(),
            label_plane_count=len(self.label_planes),
        ).require_records()

    def selected_records(self) -> tuple[StoredRuntimeValue, ...]:
        relationship_name = self.child_count_relationship_name
        if relationship_name in self.adapter.artifact_inputs:
            records = RuntimeArtifactRecordResolver(
                adapter=self.adapter,
                name=relationship_name,
                artifact_type=RelationshipsArtifactType,
                group_key=self.group_key,
                current_image=self.current_image,
            ).resolve()
        else:
            records = RuntimeGroupMatchScope(
                group_key=self.group_key
            ).runtime_scope(self.adapter).artifact_query_context().find(
                name=relationship_name,
                artifact_type=RelationshipsArtifactType,
            )
        return RuntimeRecordSourceImageSetSelector(
            self.adapter,
            self.current_image,
        ).select_runtime_scope(records)

    def values_for_plane(
        self,
        slice_index: int,
        label_plane: np.ndarray,
        plane_records: tuple[RuntimeRelationshipPlaneRecord, ...],
    ) -> np.ndarray | None:
        object_ids = dense_object_label_id_domain(label_plane)
        counts_by_parent = {object_id: 0.0 for object_id in object_ids}
        for plane_record in plane_records:
            relationship_slice_indices = self.relationship_slice_indices(
                plane_record,
                slice_index=slice_index,
            )
            if relationship_slice_indices is None:
                return None
            if not relationship_slice_indices:
                continue
            for parent_id, relationship_slice_index in zip(
                plane_record.relationship.source_ids,
                relationship_slice_indices,
                strict=True,
            ):
                if relationship_slice_index != slice_index:
                    continue
                parent_id = int(parent_id)
                if parent_id in counts_by_parent:
                    counts_by_parent[parent_id] += 1.0
        return ObjectLabelMeasurementValues.from_value_mapping(
            object_ids,
            counts_by_parent,
        ).values

    def relationship_slice_indices(
        self,
        plane_record: RuntimeRelationshipPlaneRecord,
        *,
        slice_index: int,
    ) -> tuple[int, ...] | None:
        relationship = plane_record.relationship
        if plane_record.plane_index is not None:
            if plane_record.plane_index != slice_index:
                return ()
            return tuple(plane_record.plane_index for _source_id in relationship.source_ids)
        relationship_slice_indices = tuple(relationship.slice_indices)
        if len(self.label_planes) > 1 and not relationship_slice_indices:
            return None
        if relationship_slice_indices:
            return relationship_slice_indices
        return tuple(0 for _source_id in relationship.source_ids)
