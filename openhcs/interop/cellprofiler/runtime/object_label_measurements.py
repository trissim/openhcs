"""Object-label measurement batch resolution for CellProfiler runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
from weakref import WeakKeyDictionary

import numpy as np

from openhcs.core.artifacts import (
    ArtifactSpec,
    ArtifactSpecCollection,
    ObjectLabelsArtifactType,
)
from openhcs.core.measurement_feature_queries import (
    MeasurementFeatureQuery,
    RuntimeObjectLabelMeasurementQuery,
    RuntimeObjectSliceMeasurementQuery,
)
from openhcs.core.runtime_artifact_queries import MeasurementLabelSliceFeatureQuery
from openhcs.core.runtime_measurements import (
    MeasurementTable,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelValue,
)
from openhcs.core.runtime_relationships import (
    ObjectRelationship,
    ObjectRelationshipDeclaration,
)
from openhcs.core.runtime_measurements import (
    MeasurementRowAxisField,
    ObjectLabelMeasurementValues,
)
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValues
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.interop.cellprofiler.measurement_dialect import (
    CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
)
from openhcs.interop.cellprofiler.measurement_lookup import (
    child_count_feature_child_name,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_tables import ObjectMeasurementTableIndex

if TYPE_CHECKING:
    from openhcs.interop.cellprofiler.runtime.adapter import CellProfilerRuntimeAdapter

ObjectLabelMeasurementValuesProcessCache = dict[
    RuntimeObjectLabelMeasurementQuery,
    tuple[np.ndarray, ...],
]

_OBJECT_LABEL_MEASUREMENT_VALUES_PROCESS_CACHE: WeakKeyDictionary[
    RuntimeValueStore,
    tuple[int, ObjectLabelMeasurementValuesProcessCache],
] = WeakKeyDictionary()


def object_label_measurement_values_cache(
    store: RuntimeValueStore,
) -> ObjectLabelMeasurementValuesProcessCache:
    """Return label-aligned vectors bound to the store's current revision."""
    cached_revision = _OBJECT_LABEL_MEASUREMENT_VALUES_PROCESS_CACHE.get(store)
    if cached_revision is not None and cached_revision[0] == store.revision:
        return cached_revision[1]
    cache: ObjectLabelMeasurementValuesProcessCache = {}
    _OBJECT_LABEL_MEASUREMENT_VALUES_PROCESS_CACHE[store] = (store.revision, cache)
    return cache


@dataclass(frozen=True, slots=True, kw_only=True)
class ObjectFeatureMeasurementContext(RuntimeObjectSliceMeasurementQuery):
    """Nominal context for one object-feature measurement lookup."""

    @property
    def feature_query(self) -> MeasurementFeatureQuery:
        """Return the CellProfiler feature query for this object context."""
        return MeasurementFeatureQuery(
            self.feature_name,
            object_name=self.object_name,
            dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
        )

    @property
    def table_scope_slice_index(self) -> int | None:
        """Return a runtime slice only when it belongs to the feature scope."""
        if self.feature_query.source_candidates:
            return None
        return self.slice_index

    def measurement_tables(
        self,
        adapter: "CellProfilerRuntimeAdapter",
        *,
        match_group: bool = True,
    ) -> tuple[MeasurementTable, ...]:
        """Return exact declared input tables for this object-feature context."""
        records = adapter.declared_measurement_input_records(
            group_key=self.group_key,
            match_group=match_group,
        )
        if not records:
            return ()
        object_table_index = ObjectMeasurementTableIndex.from_tables(
            tuple(cast(MeasurementTable, record.value.data) for record in records)
        )
        tables = object_table_index.for_object_feature(
            self.object_name,
            self.feature_name,
        )
        return () if tables is None else tables


@dataclass(frozen=True, slots=True)
class RuntimeRelationshipPlaneRecord:
    """Relationship record projected onto one runtime label plane."""

    relationship: ObjectRelationship
    plane_index: int | None = None


@dataclass(frozen=True, slots=True)
class RelationshipPlaneProjectionResolution:
    """Typed result for projecting relationship values onto label planes."""

    records: tuple[RuntimeRelationshipPlaneRecord, ...]

    @classmethod
    def from_value(
        cls,
        relationship_name: str,
        value: ObjectRelationship | RuntimeSliceAlignedValues[ObjectRelationship],
        *,
        label_plane_count: int,
    ) -> "RelationshipPlaneProjectionResolution":
        """Project an explicitly composed relationship value onto label planes."""
        if isinstance(value, ObjectRelationship):
            return cls(
                records=(
                    RuntimeRelationshipPlaneRecord(
                        value,
                    ),
                )
            )
        if value.slice_count != label_plane_count:
            raise ValueError(
                f"Relationship input {relationship_name!r} carries "
                f"{value.slice_count} runtime slices for {label_plane_count} "
                "label planes."
            )
        projected_records: list[RuntimeRelationshipPlaneRecord] = []
        for plane_index, relationship in enumerate(value.slices):
            if relationship.payload.slice_count not in (None, 1):
                raise ValueError(
                    "Slice-aligned relationship records must be payload-local, got "
                    f"slice_count={relationship.payload.slice_count!r}."
                )
            if relationship.payload.slice_indices and any(
                slice_index != 0
                for slice_index in relationship.payload.slice_indices
            ):
                raise ValueError(
                    "Slice-aligned relationship records must use payload-local "
                    "slice_index 0."
                )
            projected_records.append(
                RuntimeRelationshipPlaneRecord(
                    relationship,
                    plane_index=plane_index,
                )
            )
        return cls(records=tuple(projected_records))


@dataclass(frozen=True, slots=True, kw_only=True)
class ObjectLabelMeasurementSliceRequest(ObjectFeatureMeasurementContext):
    """Complete context for one label-aligned object-measurement vector query."""

    labels: ObjectLabelValue

    @property
    def label_data(self) -> object:
        return self.labels.variant_data.labels

    @property
    def label_plane_count(self) -> int:
        return len(self.labels.measurement_planes())

    @property
    def label_planes(self) -> tuple[object, ...]:
        return tuple(plane.labels for plane in self.labels.measurement_planes())

    def label_domain(self) -> tuple[int, ...]:
        """Return the declared label-ID domain for this request."""
        plane_domains = self.label_plane_domains()
        return tuple(
            sorted(
                {
                    object_id
                    for plane_domain in plane_domains
                    for object_id in plane_domain
                }
            )
        )

    def label_plane_domains(self) -> tuple[tuple[int, ...], ...]:
        """Return the declared object-ID domain of each label plane."""
        plane_domains = self.labels.measurement_plane_domains()
        if len(plane_domains) != self.label_plane_count:
            raise ValueError(
                "Object-label measurement plane cardinality must match declared "
                f"domains; got {self.label_plane_count} planes and "
                f"{len(plane_domains)} domains."
            )
        return plane_domains

    def measurement_query(
        self,
        adapter: "CellProfilerRuntimeAdapter",
        *,
        label_domain: tuple[int, ...] | None = None,
    ) -> RuntimeObjectLabelMeasurementQuery:
        """Return the cache/query identity for this label-aligned feature."""
        return RuntimeObjectLabelMeasurementQuery(
            axis_id=adapter.request.axis_scope.axis_id,
            group_key=self.group_key,
            object_name=self.object_name,
            feature_name=self.feature_name,
            label_domain=(
                self.label_domain() if label_domain is None else label_domain
            ),
            label_plane_domains=self.label_plane_domains(),
            slice_index=self.table_scope_slice_index,
        )

    def values(
        self,
        adapter: "CellProfilerRuntimeAdapter",
    ) -> tuple[np.ndarray, ...]:
        """Return object measurements aligned to this request's label planes."""
        label_domain = self.label_domain()
        query = self.measurement_query(adapter, label_domain=label_domain)
        object_label_values_cache = object_label_measurement_values_cache(
            adapter.request.context.runtime_value_store
        )
        cached = object_label_values_cache.get(query)
        if cached is not None:
            return cached
        child_name = child_count_feature_child_name(self.feature_name)
        relationship_spec = None
        if child_name is not None:
            source_ref = ArtifactSpec.input(
                self.object_name,
                ObjectLabelsArtifactType,
            ).ref()
            target_ref = ArtifactSpec.input(
                child_name,
                ObjectLabelsArtifactType,
            ).ref()
            matches = tuple(
                spec
                for spec, declaration in ArtifactSpecCollection(
                    edge_plan.spec
                    for edge_plan in adapter.request.artifact_inputs.values()
                ).relation_refs(ObjectRelationshipDeclaration)
                if declaration.source == source_ref and declaration.target == target_ref
            )
            if len(matches) > 1:
                raise ValueError(
                    "Child-count endpoints select multiple compiled relationship "
                    f"inputs: {source_ref!r} -> {target_ref!r}: "
                    f"{tuple(spec.name for spec in matches)!r}."
                )
            relationship_spec = matches[0] if matches else None
        if relationship_spec is not None:
            values = RelationshipChildCountLabelMeasurement(
                adapter=adapter,
                relationship_spec=relationship_spec,
                object_name=self.object_name,
                feature_name=self.feature_name,
                group_key=self.group_key,
                slice_index=self.table_scope_slice_index,
                labels=self.labels,
            ).required_values()
        else:
            tables = self.measurement_tables(adapter, match_group=False)
            if not tables:
                raise ValueError(
                    f"Object feature {self.feature_name!r} for {self.object_name!r} "
                    "requires a declared relationship or measurement artifact input."
                )
            values = MeasurementLabelSliceFeatureQuery(
                measurement_tables=tables,
                feature_name=self.feature_name,
                object_name=self.object_name,
                row_axis=MeasurementRowAxisField.SLICE_INDEX,
                plane_projector=adapter,
                dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
            ).values_for_labels(self.labels)
        object_label_values_cache[query] = values
        return values


@dataclass(frozen=True, slots=True, kw_only=True)
class RelationshipChildCountLabelMeasurement(ObjectLabelMeasurementSliceRequest):
    """Resolve label-aligned child-count vectors from relationship artifacts."""

    adapter: "CellProfilerRuntimeAdapter"
    relationship_spec: ArtifactSpec

    @property
    def child_count_child_name(self) -> str:
        child_name = child_count_feature_child_name(self.feature_name)
        if child_name is None:
            raise ValueError(
                f"Feature {self.feature_name!r} is not a declared child-count feature."
            )
        return child_name

    def required_values(self) -> tuple[np.ndarray, ...]:
        plane_records = self.valid_plane_records()
        return tuple(
            self.values_for_plane(slice_index, label_plane, plane_records)
            for slice_index, label_plane in enumerate(self.label_planes)
        )

    def valid_plane_records(self) -> tuple[RuntimeRelationshipPlaneRecord, ...]:
        child_name = self.child_count_child_name
        plane_records = self.plane_records()
        if not plane_records:
            raise ValueError(
                f"Relationship input {self.relationship_spec.name!r} has no records."
            )
        if any(
            plane_record.relationship.declaration.source.name != self.object_name
            or plane_record.relationship.declaration.target.name != child_name
            for plane_record in plane_records
        ):
            raise ValueError(
                f"Relationship input {self.relationship_spec.name!r} has "
                "endpoints that do not match the requested child-count feature."
            )
        return plane_records

    def plane_records(self) -> tuple[RuntimeRelationshipPlaneRecord, ...]:
        return RelationshipPlaneProjectionResolution.from_value(
            self.relationship_spec.name,
            self.adapter.get_relationship(
                self.relationship_spec.name,
                artifact_type=self.relationship_spec.artifact_type,
                group_key=self.group_key,
            ),
            label_plane_count=len(self.label_planes),
        ).records

    def values_for_plane(
        self,
        slice_index: int,
        label_plane: object,
        plane_records: tuple[RuntimeRelationshipPlaneRecord, ...],
    ) -> np.ndarray:
        del label_plane
        object_ids = self.label_plane_domains()[slice_index]
        counts_by_parent = {object_id: 0.0 for object_id in object_ids}
        for plane_record in plane_records:
            relationship_slice_indices = self.relationship_slice_indices(
                plane_record,
                slice_index=slice_index,
            )
            if not relationship_slice_indices:
                continue
            for parent_id, relationship_slice_index in zip(
                plane_record.relationship.payload.source_ids,
                relationship_slice_indices,
                strict=True,
            ):
                if relationship_slice_index != slice_index:
                    continue
                parent_id = int(parent_id)
                if parent_id not in counts_by_parent:
                    raise ValueError(
                        f"Relationship parent ID {parent_id} is outside the declared "
                        f"object domain {object_ids!r} on slice {slice_index}."
                    )
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
    ) -> tuple[int, ...]:
        relationship = plane_record.relationship
        if plane_record.plane_index is not None:
            if plane_record.plane_index != slice_index:
                return ()
            return tuple(
                plane_record.plane_index
                for _source_id in relationship.payload.source_ids
            )
        relationship_slice_indices = tuple(relationship.payload.slice_indices)
        if relationship.payload.slice_count is None:
            if self.label_plane_count != 1 or relationship_slice_indices:
                raise ValueError(
                    "Relationship records aligned to multiple label planes must "
                    "declare slice_count and slice_indices."
                )
            return tuple(0 for _source_id in relationship.payload.source_ids)
        if relationship.payload.slice_count != self.label_plane_count:
            raise ValueError(
                f"Relationship declares {relationship.payload.slice_count} slices for "
                f"{self.label_plane_count} label planes."
            )
        if relationship_slice_indices:
            return relationship_slice_indices
        if (
            relationship.payload.source_ids
            and relationship.payload.slice_count != 1
        ):
            raise ValueError(
                "Non-empty multi-slice relationships require one slice_index per pair."
            )
        return tuple(0 for _source_id in relationship.payload.source_ids)
