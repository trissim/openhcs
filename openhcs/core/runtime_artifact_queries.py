"""Semantic queries over typed OpenHCS runtime artifacts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace
from types import MappingProxyType
from typing import Any, ClassVar, cast
from weakref import WeakKeyDictionary

from metaclass_registry import AutoRegisterMeta
import numpy as np

from openhcs.core.artifacts import ArtifactKind
from openhcs.core.measurement_row_materialization import (
    MEASUREMENT_OBJECT_ID_FIELD,
    MEASUREMENT_OBJECT_ID_FIELDS,
    MEASUREMENT_OBJECT_LABEL_FIELD,
    MEASUREMENT_OBJECT_NAME_FIELD,
    MEASUREMENT_OBJECT_NUMBER_FIELD,
    MEASUREMENT_OBJECT_ROW_IDENTITY_FIELD,
    MEASUREMENT_LABEL_FIELD,
    MEASUREMENT_SOURCE_IMAGE_NAME_FIELD,
    ConcatenatedColumnarRows,
    DataclassMeasurementColumnarRows,
    MEASUREMENT_SPARSE_CELL,
    MeasurementColumnarRowsView,
    MeasurementObjectLabelResolution,
    MeasurementProjectedColumnarRows,
    MeasurementRowOwnership,
    MeasurementRowQualifier,
    MeasurementSliceIndexImageNumberProjection,
    MeasurementSparseCell,
    MeasurementSparseColumnarRows,
    ProjectedMeasurementRows,
    QualifiedMeasurementColumnarRows,
    columnar_row_count,
    columnar_row_values,
    is_structural_missing_measurement_cell as _is_structural_missing_measurement_cell,
    measurement_object_label,
    measurement_row_has_object_identity,
    measurement_row_object_name,
    measurement_row_source_image_name,
    measurement_rows,
    measurement_table_object_id_field,
    measurement_table_object_name,
)
from openhcs.core.measurement_feature_queries import (
    MEASUREMENT_FEATURE_NAME_FIELDS,
    MEASUREMENT_VALUE_FIELDS,
    ColumnarMeasurementTableSchema,
    IndexedObjectMeasurementLabelPlaneBinding,
    MeasurementAxisValueProjection,
    MeasurementFeatureQuery,
    MeasurementFeatureValueIndex,
    MeasurementObjectFeatureVectorBatchQuery,
    MeasurementTableFeatureQuery,
    MeasurementTableObjectFeatureSemantics,
    MeasurementValueIndexResult,
)
from openhcs.core.measurement_lookup_dialect import (
    CURRENT_RUNTIME_MEASUREMENT_LOOKUP_DIALECT,
    RuntimeMeasurementFeatureLookup,
    RuntimeMeasurementLookupDialect,
    RuntimeMeasurementLookupDialectLike,
    resolve_runtime_measurement_lookup_dialect,
)
from openhcs.core.process_local_cache import (
    ProcessLocalBoundedCache,
    identity_owner_tuples_match,
    named_identity_owner_tuples_match,
)
from openhcs.core.registry_strategies import NominalTypeStrategyFamilyMixin
from openhcs.core.runtime_semantics import (
    FieldSpec,
    MeasurementRowAxisField,
    MeasurementSubject,
    MeasurementTableRowLayout,
    MeasurementScope,
    ObjectLabelMeasurementValues,
    ObjectLabelIdDomainStrategy,
    dense_object_label_id_domain,
    measurement_axis_integer_domain,
    measurement_axis_integer_value,
    measurement_row_mapping,
    measurement_table_row_layout,
    measurement_table_row_layout_from_fields,
    normalize_measurement_table_rows,
)
from openhcs.core.runtime_identifier import normalize_runtime_identifier
from openhcs.core.runtime_stores import (
    RuntimeValueStore,
    StoredRuntimeValue,
)
from openhcs.core.runtime_values import (
    ColumnarRows,
    MeasurementTable,
    ObjectLabelValue,
    ObjectRelationship,
    SpatialGrid,
    object_label_dense_array,
)
from openhcs.core.source_image_provenance import SourceImageProvenance


_MEASUREMENT_TABLE_CACHE: WeakKeyDictionary[
    RuntimeValueStore,
    dict[tuple[int, str, str | None], tuple[MeasurementTable, ...]],
] = WeakKeyDictionary()


MeasurementLabelSliceFeatureBatchCacheValue = tuple[
    tuple[MeasurementTable, ...],
    tuple[tuple[str, object], ...],
    Mapping[str, tuple[Any, ...]],
]


@dataclass(frozen=True, slots=True)
class MeasurementLabelSliceFeatureBatchCacheKey:
    """Identity key for label-plane feature projections."""

    feature_name: str
    object_names: tuple[str, ...]
    dialect_identity: int
    row_axis: MeasurementRowAxisField
    table_identities: tuple[int, ...]
    label_identities: tuple[tuple[str, int], ...]
    row_axis_starts: tuple[tuple[str, int | None], ...]


class MeasurementLabelSliceFeatureBatchQueryCache(
    ProcessLocalBoundedCache[
        MeasurementLabelSliceFeatureBatchCacheKey,
        MeasurementLabelSliceFeatureBatchCacheValue,
    ]
):
    """Process-local cache for repeated label-plane feature projections."""

    max_entries = 1024


@dataclass(frozen=True, slots=True)
class MeasurementTableUnion:
    """Lossless row-owned view over same-artifact measurement subject tables."""

    name: str
    tables: tuple[MeasurementTable, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("MeasurementTableUnion.name cannot be empty.")
        if not self.tables:
            raise ValueError("MeasurementTableUnion.tables cannot be empty.")

    def as_table(self) -> MeasurementTable:
        if len(self.tables) == 1:
            return self.tables[0]
        schema = MeasurementTableUnionSchema.from_tables(self.tables)
        return MeasurementTable(
            name=self.name,
            rows=self.rows(),
            fields=schema.fields,
            object_name=schema.object_name,
            object_id_field=schema.object_id_field,
            source_image_name=schema.source_image_name,
            subject=schema.subject,
            validated_runtime_schema=schema.validated_runtime_schema,
            schema_loss_reasons=schema.schema_loss_reasons,
            source_provenance=schema.source_provenance,
        )

    def rows(self) -> Sequence[object] | ColumnarRows:
        if all(isinstance(table.rows, ColumnarRows) for table in self.tables):
            return ConcatenatedColumnarRows(
                tuple(table.rows for table in self.tables)
            )
        return tuple(
            row
            for table in self.tables
            for row in measurement_rows((table,))
        )


@dataclass(frozen=True, slots=True)
class MeasurementTableUnionSchema:
    """Schema facts preserved across compatible measurement-table unions."""

    source_provenance: SourceImageProvenance = field(
        default_factory=SourceImageProvenance,
    )
    fields: tuple[FieldSpec, ...] = ()
    object_name: str | None = None
    object_id_field: str | None = None
    source_image_name: str | None = None
    subject: MeasurementSubject | None = None
    validated_runtime_schema: bool = False
    schema_loss_reasons: frozenset[str] = frozenset()

    @classmethod
    def from_tables(
        cls,
        tables: tuple[MeasurementTable, ...],
    ) -> "MeasurementTableUnionSchema":
        fields, fields_reason = cls._common_value(
            tuple(table.fields for table in tables),
            "fields",
        )
        object_name, object_name_reason = cls._common_value(
            tuple(table.object_name for table in tables),
            "object_name",
        )
        object_id_field, object_id_field_reason = cls._common_value(
            tuple(table.object_id_field for table in tables),
            "object_id_field",
        )
        source_image_name, source_image_name_reason = cls._common_value(
            tuple(table.source_image_name for table in tables),
            "source_image_name",
        )
        subject, subject_reason = cls._common_value(
            tuple(table.subject for table in tables),
            "subject",
        )
        reasons = frozenset(
            reason
            for reason in (
                fields_reason,
                object_name_reason,
                object_id_field_reason,
                source_image_name_reason,
                subject_reason,
            )
            if reason is not None
        )
        return cls(
            source_provenance=cls._common_source_provenance(tables),
            fields=fields or (),
            object_name=object_name,
            object_id_field=object_id_field,
            source_image_name=source_image_name,
            subject=subject,
            validated_runtime_schema=bool(fields) and not reasons,
            schema_loss_reasons=reasons,
        )

    @staticmethod
    def _common_value(
        values: tuple[Any, ...],
        field_name: str,
    ) -> tuple[Any | None, str | None]:
        unique_values = tuple(dict.fromkeys(values))
        if len(unique_values) == 1:
            return unique_values[0], None
        return None, field_name

    @staticmethod
    def _common_source_provenance(
        tables: tuple[MeasurementTable, ...],
    ) -> SourceImageProvenance:
        first = tables[0].source_provenance
        first_identity = first.equality_identity
        if all(
            table.source_provenance.equality_identity == first_identity
            for table in tables
        ):
            return first
        return SourceImageProvenance()


@dataclass(frozen=True, slots=True)
class RuntimeArtifactQueryContext:
    """Execution-scope view over a RuntimeValueStore."""

    store: RuntimeValueStore
    axis_id: str
    group_key: str | None = None
    match_group: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.store, RuntimeValueStore):
            raise TypeError(
                "RuntimeArtifactQueryContext.store must be RuntimeValueStore, "
                f"got {type(self.store).__name__}."
            )
        if not self.axis_id:
            raise ValueError("RuntimeArtifactQueryContext.axis_id cannot be empty.")

    def find(
        self,
        *,
        kind: ArtifactKind | None = None,
        name: str | None = None,
    ) -> tuple[StoredRuntimeValue, ...]:
        """Find runtime records in this execution scope."""
        return self.store.find(
            name=name,
            kind=kind,
            axis_id=self.axis_id,
            group_key=self.group_key,
            match_group=self.match_group,
        )

    def resolve(
        self,
        *,
        name: str,
        kind: ArtifactKind,
        purpose: str = "runtime artifact",
    ) -> StoredRuntimeValue:
        """Resolve exactly one runtime record in this execution scope."""
        records = self.find(name=name, kind=kind)
        if not records:
            raise RuntimeError(
                f"Missing {purpose} '{name}' ({kind.value}) on axis "
                f"'{self.axis_id}'."
            )
        if len(records) > 1:
            raise RuntimeError(
                f"Ambiguous {purpose} '{name}' ({kind.value}) on axis "
                f"'{self.axis_id}': {runtime_record_locations(records)}."
            )
        return records[0]


def runtime_record_locations(records: Sequence[StoredRuntimeValue]) -> tuple[str, ...]:
    """Return compact runtime-record identities without formatting payload data."""
    return tuple(
        f"{record.key.scope.group_key or '<none>'}@{record.backend}:{record.path}"
        for record in records
    )


@dataclass(frozen=True, slots=True)
class MeasurementObjectQuery:
    """Query for measurement tables describing one object set."""

    object_name: str

    def __post_init__(self) -> None:
        if not self.object_name:
            raise ValueError("MeasurementObjectQuery.object_name cannot be empty.")

    def matches(self, table: MeasurementTable) -> bool:
        if table.subject.scope is MeasurementScope.OBJECT:
            return table.subject.name == self.object_name
        if isinstance(table.rows, ColumnarRows):
            return self.object_name in ColumnarMeasurementTableSchema.from_table(
                table
            ).object_names
        if not _measurement_table_may_declare_object_name(table):
            return False
        return any(
            measurement_row_object_name(measurement_row_mapping(row))
            == self.object_name
            for row in measurement_rows((table,))
        )


@dataclass(frozen=True, slots=True)
class MeasurementScopeQuery:
    """Query for measurement tables describing a semantic scope."""

    scope: MeasurementScope
    name: str | None = None

    def __post_init__(self) -> None:
        MeasurementScope(self.scope)
        if self.name == "":
            raise ValueError("MeasurementScopeQuery.name cannot be empty.")

    def matches(self, table: MeasurementTable) -> bool:
        if table.subject.scope is not MeasurementScope(self.scope):
            return False
        return self.name is None or table.subject.name == self.name


def runtime_measurement_tables(
    context: RuntimeArtifactQueryContext,
) -> tuple[MeasurementTable, ...]:
    """Return all measurement tables in a runtime query context."""
    cache_key = (context.store.revision, context.axis_id, context.group_key)
    store_cache = _MEASUREMENT_TABLE_CACHE.setdefault(context.store, {})
    cached = store_cache.get(cache_key)
    if cached is not None:
        return cached
    tables = tuple(
        MeasurementTable.from_runtime_value(record.value)
        for record in context.find(kind=ArtifactKind.MEASUREMENTS)
    )
    for key in tuple(store_cache):
        if key[0] != context.store.revision:
            del store_cache[key]
    store_cache[cache_key] = tables
    return tables


def runtime_measurement_tables_for_object(
    context: RuntimeArtifactQueryContext,
    object_name: str,
) -> tuple[MeasurementTable, ...]:
    """Return measurement tables whose subject is one object set."""
    query = MeasurementObjectQuery(object_name)
    return tuple(
        table
        for table in runtime_measurement_tables(context)
        if query.matches(table)
    )


def runtime_measurement_tables_for_scope(
    context: RuntimeArtifactQueryContext,
    scope: MeasurementScope,
    name: str | None = None,
) -> tuple[MeasurementTable, ...]:
    """Return measurement tables whose subject matches one semantic scope."""
    query = MeasurementScopeQuery(scope, name)
    return tuple(
        table
        for table in runtime_measurement_tables(context)
        if query.matches(table)
    )


def runtime_relationship(
    context: RuntimeArtifactQueryContext,
    name: str,
) -> ObjectRelationship:
    """Return one relationship artifact as native OpenHCS relationship value."""
    record = context.resolve(
        name=name,
        kind=ArtifactKind.RELATIONSHIPS,
        purpose="relationship artifact",
    )
    return ObjectRelationship.from_runtime_value(record.value)


def runtime_spatial_grid(
    context: RuntimeArtifactQueryContext,
    name: str,
) -> SpatialGrid:
    """Return one spatial-grid artifact as a native OpenHCS value."""
    record = context.resolve(
        name=name,
        kind=ArtifactKind.SPATIAL_GRID,
        purpose="spatial grid artifact",
    )
    return SpatialGrid.from_runtime_value(record.value)


def _measurement_table_may_declare_object_name(table: MeasurementTable) -> bool:
    """Return whether row-level fallback object-name scans can match."""
    if table.object_name is not None:
        return True
    if any(field.name == MEASUREMENT_OBJECT_NAME_FIELD for field in table.fields):
        return True

    column_names = table.column_names()
    if column_names is not None:
        return MEASUREMENT_OBJECT_NAME_FIELD in column_names
    rows = table.row_sequence_payloads()
    if rows is None:
        return False
    return any(
        MEASUREMENT_OBJECT_NAME_FIELD in measurement_row_mapping(row)
        for row in rows
    )


class MeasurementTableRowsAxisProjection(
    NominalTypeStrategyFamilyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered table-row projection selected by row payload representation."""

    @classmethod
    def for_table(cls, table: MeasurementTable) -> "MeasurementTableRowsAxisProjection":
        projection_types = cls.strategy_types_for_nominal_value(table.rows)
        if not projection_types:
            raise TypeError(
                "MeasurementTableRowsAxisProjection requires table-like rows, "
                f"got {type(table.rows).__name__}."
            )
        return cast(MeasurementTableRowsAxisProjection, projection_types[0]())

    @abstractmethod
    def project(
        self,
        projection: "MeasurementTableAxisProjection",
        table: MeasurementTable,
    ) -> MeasurementTable:
        """Return ``table`` narrowed by ``projection``."""


@dataclass(frozen=True, slots=True)
class MeasurementTableAxisProjection(MeasurementAxisValueProjection):
    """Projection of one measurement table onto a declared row-axis value."""

    table: MeasurementTable | None = None

    def apply(self, table: MeasurementTable | None = None) -> MeasurementTable:
        """Return the projected table, preserving schema only when still valid."""
        target_table = self._target_table(table)
        return MeasurementTableRowsAxisProjection.for_table(target_table).project(
            self,
            target_table,
        )

    def table_projection(self, table: MeasurementTable) -> MeasurementTable:
        """Return one table narrowed to this row-axis value."""
        return self.apply(table)

    def tables(
        self,
        measurement_tables: tuple[MeasurementTable, ...],
    ) -> tuple[MeasurementTable, ...]:
        """Return measurement tables narrowed to this row-axis value."""
        return tuple(self.table_projection(table) for table in measurement_tables)

    def _target_table(self, table: MeasurementTable | None) -> MeasurementTable:
        target_table = self.table if table is None else table
        if target_table is None:
            raise ValueError(
                "MeasurementTableAxisProjection.apply requires a table when the "
                "projection was not constructed with one."
            )
        return target_table

    def _row_sequence_table(
        self,
        table: MeasurementTable,
        rows: Sequence[object],
    ) -> MeasurementTable:
        declared_layout = measurement_table_row_layout_from_fields(table.fields)
        if declared_layout is not None:
            return self._with_rows(
                table,
                rows,
                table.fields,
                validated_runtime_schema=True,
            )
        normalized_rows = normalize_measurement_table_rows(rows, fields=())
        return self._with_rows(
            table,
            normalized_rows,
            self._compatible_fields(table, normalized_rows),
            validated_runtime_schema=False,
        )

    def _compatible_fields(
        self,
        table: MeasurementTable,
        rows: object,
    ) -> tuple[FieldSpec, ...]:
        declared_layout = measurement_table_row_layout_from_fields(table.fields)
        observed_layout = measurement_table_row_layout(rows)
        if declared_layout is None:
            return ()
        if observed_layout not in (declared_layout, MeasurementTableRowLayout.EMPTY):
            return ()
        return tuple(table.fields)

    def _with_rows(
        self,
        table: MeasurementTable,
        rows: object,
        fields: Iterable[FieldSpec],
        *,
        validated_runtime_schema: bool = False,
    ) -> MeasurementTable:
        return MeasurementTable(
            name=table.name,
            rows=rows,
            object_name=table.object_name,
            fields=tuple(fields),
            object_id_field=table.object_id_field,
            source_image_name=table.source_image_name,
            subject=table.subject,
            validated_runtime_schema=validated_runtime_schema,
            schema_loss_reasons=table.schema_loss_reasons,
            source_provenance=self.table_source_provenance(table),
        )


class ColumnarMeasurementTableRowsAxisProjection(MeasurementTableRowsAxisProjection):
    """Axis projection for columnar measurement-table row payloads."""

    value_type = ColumnarRows

    def project(
        self,
        projection: MeasurementTableAxisProjection,
        table: MeasurementTable,
    ) -> MeasurementTable:
        rows = cast(ColumnarRows, table.rows)
        column_names = tuple(str(column) for column in rows.columns)
        if projection.field_name not in column_names:
            return table
        axis_values = columnar_row_values(rows, projection.field_name)
        axis_mask = projection.columnar_mask(axis_values)
        if bool(np.all(axis_mask)):
            return table
        projected_rows = AxisFilteredMeasurementColumnarRows(
            rows,
            projection,
            axis_mask=axis_mask,
        )
        return projection._with_rows(table, projected_rows, table.fields)


class NativeMeasurementTableRowsAxisProjection(MeasurementTableRowsAxisProjection):
    """Axis projection for native mapping or sequence measurement-table rows."""

    value_type = (Mapping, Sequence)

    def project(
        self,
        projection: MeasurementTableAxisProjection,
        table: MeasurementTable,
    ) -> MeasurementTable:
        rows = measurement_rows((table,))
        if not rows:
            return table
        if not any(projection.field_name in measurement_row_mapping(row) for row in rows):
            return table

        row_mappings = tuple(measurement_row_mapping(row) for row in rows)
        projection_mask = projection.mask(
            tuple(row.get(projection.field_name) for row in row_mappings)
        )
        return projection._row_sequence_table(
            table,
            [row for row, keep in zip(rows, projection_mask, strict=True) if keep],
        )


def measurement_table_slice_indices(table: MeasurementTable) -> set[int]:
    """Return runtime slice indexes declared by one measurement table."""
    return measurement_table_axis_values(table, MeasurementRowAxisField.SLICE_INDEX)


def measurement_table_axis_values(
    table: MeasurementTable,
    axis: MeasurementRowAxisField,
) -> set[int]:
    """Return declared row-axis values for one measurement table."""
    axis_field = axis.value
    if isinstance(table.rows, ColumnarRows):
        column_names = tuple(str(column) for column in table.rows.columns)
        if axis_field not in column_names:
            return set()
        return set(
            measurement_axis_integer_domain(
                columnar_row_values(table.rows, axis_field),
                axis,
            )
        )
    return {
        axis_integer
        for row in measurement_rows((table,))
        for row_mapping in (measurement_row_mapping(row),)
        for axis_integer in (
            _measurement_axis_integer_value(row_mapping.get(axis_field), axis),
        )
        if axis_integer is not None
    }


def _measurement_axis_integer_value(
    value: object,
    axis: MeasurementRowAxisField,
) -> int | None:
    return measurement_axis_integer_value(value, axis)


@dataclass(frozen=True, slots=True)
class MeasurementLabelSliceAxisSelection:
    """Authoritative row-axis binding for label-stack measurement lookup."""

    row_axis: MeasurementRowAxisField
    row_axis_start: int | None = None

    def row_axis_value(self, slice_index: int) -> int:
        """Return the row-axis value corresponding to one label slice."""
        if self.row_axis_start is None:
            return slice_index
        return self.row_axis_start + slice_index

    def value_index_for_slice(
        self,
        values_by_slice: Mapping[int, MeasurementValueIndexResult],
        slice_index: int,
        *,
        label_slice_count: int | None = None,
    ) -> MeasurementValueIndexResult:
        """Return the measurement index for a label plane, broadcasting singletons."""
        row_axis_value = self.row_axis_value(slice_index)
        if row_axis_value in values_by_slice:
            return values_by_slice[row_axis_value]
        if slice_index in values_by_slice:
            return values_by_slice[slice_index]
        if -1 in values_by_slice:
            return values_by_slice[-1]
        concrete_slice_indexes = tuple(
            sorted(index for index in values_by_slice if index >= 0)
        )
        if (
            label_slice_count is not None
            and len(concrete_slice_indexes) == label_slice_count
            and slice_index < label_slice_count
        ):
            return values_by_slice[concrete_slice_indexes[slice_index]]
        if (
            label_slice_count is not None
            and concrete_slice_indexes
            and label_slice_count % len(concrete_slice_indexes) == 0
        ):
            return values_by_slice[
                concrete_slice_indexes[slice_index % len(concrete_slice_indexes)]
            ]
        if len(concrete_slice_indexes) == 1:
            return values_by_slice[concrete_slice_indexes[0]]
        return {}, []


@dataclass(frozen=True, slots=True, kw_only=True)
class MeasurementLabelSliceFeatureQuery(MeasurementTableFeatureQuery):
    """Query one measurement feature against a stack of object-label planes."""

    row_axis: MeasurementRowAxisField = MeasurementRowAxisField.SLICE_INDEX

    def select_axis(self) -> MeasurementLabelSliceAxisSelection:
        """Return the declared table axis and slice-to-row start offset."""
        for axis in self.candidate_axes():
            axis_values = self.axis_values(axis)
            if not axis_values:
                continue
            return MeasurementLabelSliceAxisSelection(
                row_axis=axis,
                row_axis_start=self.axis_start(axis_values),
            )
        return MeasurementLabelSliceAxisSelection(self.row_axis)

    def candidate_axes(self) -> tuple[MeasurementRowAxisField, ...]:
        """Return row axes in preferred order without duplicating candidates."""
        return tuple(
            dict.fromkeys(
                (
                    self.row_axis,
                    MeasurementRowAxisField.IMAGE_NUMBER,
                    MeasurementRowAxisField.SLICE_INDEX,
                )
            )
        )

    def axis_values(self, axis: MeasurementRowAxisField) -> tuple[int, ...]:
        """Return sorted row-axis values on tables that can carry the feature."""
        values = {
            value
            for table in self.measurement_tables
            if self.table_may_carry_feature(table)
            for value in measurement_table_axis_values(table, axis)
        }
        return tuple(sorted(values))

    @staticmethod
    def axis_start(axis_values: tuple[int, ...]) -> int | None:
        """Return the slice-index origin for a concrete table axis."""
        if not axis_values:
            return None
        first_value = axis_values[0]
        return None if first_value == 0 else first_value

    def values_for_labels(
        self,
        labels: object,
        *,
        row_axis_start: int | None = None,
    ) -> tuple[Any, ...]:
        """Return measurement values aligned to each label plane."""
        import numpy as np

        label_planes = self.label_planes(labels)
        axis_selection = self.select_axis()
        if row_axis_start is not None:
            axis_selection = dataclass_replace(
                axis_selection,
                row_axis_start=row_axis_start,
            )
        if not self.measurement_tables:
            label_domains = tuple(
                dense_object_label_id_domain(label_plane)
                for label_plane in label_planes
            )
            if not any(label_domains):
                return tuple(
                    ObjectLabelMeasurementValues(
                        label_domain,
                        np.empty(0, dtype=np.float64),
                    ).values
                    for label_domain in label_domains
                )
        indexed_values_by_plane = self.indexed_values_for_label_planes(
            label_planes,
            axis_selection=axis_selection,
        )
        return tuple(
            IndexedObjectMeasurementLabelPlaneBinding(
                measurement_tables=self.measurement_tables,
                object_name=self.object_name,
                feature_name=self.feature_name,
                labels=label_plane,
                dialect=self.dialect,
                indexed_values=indexed_values,
            ).values()
            for label_plane, indexed_values in zip(
                label_planes,
                indexed_values_by_plane,
                strict=True,
            )
        )

    @staticmethod
    def label_planes(labels: object) -> tuple[Any, ...]:
        """Return the label planes that define one object-measurement domain."""
        if isinstance(labels, ObjectLabelValue):
            label_array = object_label_dense_array(labels)
            if label_array.ndim <= 2:
                return (labels,)
            return tuple(
                labels.with_projected_plane(label_array[index], index)
                for index in range(label_array.shape[0])
            )
        label_array = np.asarray(labels)
        if label_array.ndim <= 2:
            return (label_array,)
        return tuple(label_array[index] for index in range(label_array.shape[0]))

    def indexed_values_for_label_planes(
        self,
        label_planes: tuple[Any, ...],
        *,
        axis_selection: MeasurementLabelSliceAxisSelection,
    ) -> tuple[MeasurementValueIndexResult, ...]:
        """Return feature indexes aligned to label planes through one axis authority."""
        batch_values_by_axis = self.object_batch_value_indexes_by_axis(
            axis_selection.row_axis,
        )
        if batch_values_by_axis is not None and self.object_name is not None:
            values_by_axis = {
                axis_value: values_by_object.get(self.object_name, ({}, []))
                for axis_value, values_by_object in batch_values_by_axis.items()
            }
            return tuple(
                axis_selection.value_index_for_slice(
                    values_by_axis,
                    slice_index,
                    label_slice_count=len(label_planes),
                )
                for slice_index in range(len(label_planes))
            )
        values_by_axis = (
            self.value_indexes_by_axis(axis_selection.row_axis)
            if len(label_planes) > 1
            else None
        )
        if values_by_axis is not None:
            return tuple(
                axis_selection.value_index_for_slice(
                    values_by_axis,
                    slice_index,
                    label_slice_count=len(label_planes),
                )
                for slice_index in range(len(label_planes))
            )
        return tuple(
            self.value_index(
                self.feature_tables_for_axis(
                    slice_index,
                    axis_selection=axis_selection,
                )
            )
            for slice_index in range(len(label_planes))
        )

    def object_batch_value_indexes_by_axis(
        self,
        row_axis: MeasurementRowAxisField,
    ) -> dict[int, Mapping[str, MeasurementValueIndexResult]] | None:
        """Return object-batch feature indexes for this query's object domain."""
        if self.object_name is None:
            return None
        return MeasurementObjectFeatureVectorBatchQuery(
            self.feature_name,
            (self.object_name,),
            dialect=self.dialect,
        ).value_indexes_by_axis(
            {self.object_name: self.measurement_tables},
            row_axis,
        )

    def feature_tables_for_axis(
        self,
        slice_index: int,
        *,
        axis_selection: MeasurementLabelSliceAxisSelection,
    ) -> tuple[MeasurementTable, ...]:
        """Return axis-projected tables, preserving axisless feature tables."""
        candidate_tables = tuple(
            table
            for table in self.measurement_tables
            if self.table_may_carry_feature(table)
        )
        projected_tables = MeasurementTableAxisProjection(
            axis_selection.row_axis,
            axis_selection.row_axis_value(slice_index),
        ).tables(candidate_tables)
        if projected_tables:
            return projected_tables
        axis_values = set()
        for table in candidate_tables:
            axis_values.update(
                measurement_table_axis_values(table, axis_selection.row_axis)
            )
        return candidate_tables if not axis_values else projected_tables

    def value_indexes_by_axis(
        self,
        row_axis: MeasurementRowAxisField,
    ) -> dict[int, MeasurementValueIndexResult] | None:
        """Return per-axis feature indexes without re-scanning tables per plane."""
        defaults: MeasurementValueIndexResult = ({}, [])
        by_slice: dict[int, MeasurementValueIndexResult] = {}
        query_object_name = self.query_object_name

        for table in self.measurement_tables:
            if not self.table_may_carry_feature(table):
                continue
            if query_object_name is not None:
                table_object = measurement_table_object_name(table)
                if table_object not in (None, query_object_name):
                    continue

            slice_indices = measurement_table_axis_values(table, row_axis)
            if not slice_indices:
                table_index = MeasurementFeatureValueIndex.from_table(
                    table,
                    self,
                )
                if table_index.present:
                    _merge_measurement_value_index(
                        defaults,
                        table_index.as_query_result(),
                    )
                continue

            for slice_index in sorted(slice_indices):
                table_index = MeasurementFeatureValueIndex.from_table(
                    MeasurementTableAxisProjection(
                        axis=row_axis,
                        value=slice_index,
                        table=table,
                    ).apply(),
                    self,
                )
                if not table_index.present:
                    continue
                target = by_slice.get(slice_index)
                if target is None:
                    target = ({}, [])
                    by_slice[slice_index] = target
                _merge_measurement_value_index(
                    target,
                    table_index.as_query_result(),
                )

        if defaults[0] or defaults[1]:
            for slice_index in tuple(by_slice):
                values_by_label = dict(defaults[0])
                values_by_label.update(by_slice[slice_index][0])
                positional_values = [*defaults[1], *by_slice[slice_index][1]]
                by_slice[slice_index] = (values_by_label, positional_values)
        if defaults[0] or defaults[1]:
            if -1 not in by_slice:
                by_slice[-1] = defaults
        if by_slice:
            return by_slice
        return None


@dataclass(frozen=True, slots=True, kw_only=True)
class MeasurementLabelSliceFeatureBatchQuery(MeasurementLabelSliceFeatureQuery):
    """Query one feature for multiple object-label payloads through one table pass."""

    labels_by_object: Mapping[str, object]
    row_axis_starts_by_object: Mapping[str, int | None] = field(default_factory=dict)

    @property
    def object_names(self) -> tuple[str, ...]:
        """Return requested object names in caller order."""
        return tuple(dict.fromkeys(str(name) for name in self.labels_by_object))

    def values_by_object(self) -> Mapping[str, tuple[Any, ...]]:
        """Return label-plane-aligned vectors keyed by object name."""
        cached = self.cached_values_by_object()
        if cached is not None:
            return cached

        label_planes_by_object = {
            object_name: self.label_planes(labels)
            for object_name, labels in self.labels_by_object.items()
        }
        object_names = self.object_names
        values_by_object: dict[str, tuple[Any, ...]] = {}
        axis_selections_by_object = {
            object_name: self.axis_selection_for_object(object_name)
            for object_name in object_names
        }
        batch_values_by_axis_by_row_axis = {
            row_axis: MeasurementObjectFeatureVectorBatchQuery(
                self.feature_name,
                tuple(
                    object_name
                    for object_name, axis_selection in axis_selections_by_object.items()
                    if axis_selection.row_axis is row_axis
                ),
                dialect=self.dialect,
            ).value_indexes_by_axis(
                {
                    object_name: self.measurement_tables
                    for object_name, axis_selection in axis_selections_by_object.items()
                    if axis_selection.row_axis is row_axis
                },
                row_axis,
            )
            for row_axis in tuple(
                dict.fromkeys(
                    axis_selection.row_axis
                    for axis_selection in axis_selections_by_object.values()
                )
            )
        }
        for object_name in object_names:
            label_planes = label_planes_by_object[object_name]
            object_query = self.object_feature_query(object_name)
            axis_selection = axis_selections_by_object[object_name]
            batch_values_by_axis = batch_values_by_axis_by_row_axis[
                axis_selection.row_axis
            ]
            if batch_values_by_axis is None:
                indexed_values_by_plane = object_query.indexed_values_for_label_planes(
                    label_planes,
                    axis_selection=axis_selection,
                )
            else:
                values_by_axis = {
                    axis_value: values_by_object.get(object_name, ({}, []))
                    for axis_value, values_by_object in batch_values_by_axis.items()
                }
                indexed_values_by_plane = tuple(
                    axis_selection.value_index_for_slice(
                        values_by_axis,
                        slice_index,
                        label_slice_count=len(label_planes),
                    )
                    for slice_index in range(len(label_planes))
                )
            values_by_object[object_name] = tuple(
                IndexedObjectMeasurementLabelPlaneBinding(
                    measurement_tables=object_query.measurement_tables,
                    object_name=object_name,
                    feature_name=object_query.feature_name,
                    labels=label_plane,
                    dialect=object_query.dialect,
                    indexed_values=indexed_values,
                ).values()
                for label_plane, indexed_values in zip(
                    label_planes,
                    indexed_values_by_plane,
                    strict=True,
                )
            )
        return self.cache_values_by_object(
            MappingProxyType(
                {
                    object_name: values_by_object[object_name]
                    for object_name in object_names
                }
            )
        )

    def object_feature_query(self, object_name: str) -> MeasurementLabelSliceFeatureQuery:
        """Return the single-object query that owns one batch member's semantics."""
        return MeasurementLabelSliceFeatureQuery(
            measurement_tables=self.measurement_tables,
            feature_name=self.feature_name,
            object_name=object_name,
            dialect=self.dialect,
            row_axis=self.row_axis,
        )

    def cache_key(self) -> MeasurementLabelSliceFeatureBatchCacheKey:
        """Return the identity key for this label-plane feature projection."""
        object_names = self.object_names
        return MeasurementLabelSliceFeatureBatchCacheKey(
            feature_name=self.feature_name,
            object_names=object_names,
            dialect_identity=id(resolve_runtime_measurement_lookup_dialect(self.dialect)),
            row_axis=self.row_axis,
            table_identities=tuple(id(table) for table in self.measurement_tables),
            label_identities=tuple(
                (object_name, id(self.labels_by_object[object_name]))
                for object_name in object_names
            ),
            row_axis_starts=tuple(
                (object_name, self.row_axis_starts_by_object.get(object_name))
                for object_name in object_names
            ),
        )

    def table_owners(self) -> tuple[MeasurementTable, ...]:
        """Return table owners used to protect identity-keyed cache entries."""
        return self.measurement_tables

    def label_owners(self) -> tuple[tuple[str, object], ...]:
        """Return label owners used to protect identity-keyed cache entries."""
        return tuple(
            (object_name, self.labels_by_object[object_name])
            for object_name in self.object_names
        )

    def cached_values_by_object(self) -> Mapping[str, tuple[Any, ...]] | None:
        """Return cached label-plane feature projections when owners still match."""
        cached = (
            MeasurementLabelSliceFeatureBatchQueryCache
            .process_cache()
            .cached_value(self.cache_key())
        )
        if cached is None:
            return None
        cached_tables, cached_labels, cached_values = cached
        if not identity_owner_tuples_match(cached_tables, self.table_owners()):
            return None
        if not named_identity_owner_tuples_match(cached_labels, self.label_owners()):
            return None
        return cached_values

    def cache_values_by_object(
        self,
        values_by_object: Mapping[str, tuple[Any, ...]],
    ) -> Mapping[str, tuple[Any, ...]]:
        """Store label-plane feature projections with identity-owner protection."""
        return (
            MeasurementLabelSliceFeatureBatchQueryCache
            .process_cache()
            .store_value(
                self.cache_key(),
                (
                    self.table_owners(),
                    self.label_owners(),
                    values_by_object,
                ),
            )[2]
        )

    def axis_selection_for_object(
        self,
        object_name: str,
    ) -> MeasurementLabelSliceAxisSelection:
        """Return row-axis selection with an optional object-specific start."""
        axis_selection = self.object_feature_query(object_name).select_axis()
        row_axis_start = self.row_axis_starts_by_object.get(object_name)
        if row_axis_start is None:
            return axis_selection
        return dataclass_replace(
            axis_selection,
            row_axis_start=row_axis_start,
        )

def _merge_measurement_value_index(
    target: MeasurementValueIndexResult,
    source: MeasurementValueIndexResult,
) -> None:
    target[0].update(source[0])
    target[1].extend(source[1])


@dataclass(slots=True)
class AxisFilteredMeasurementColumnarRows(MeasurementColumnarRowsView):
    """Columnar measurement rows filtered to one runtime/CellProfiler axis value."""

    rows: ColumnarRows
    projection: MeasurementAxisValueProjection
    axis_mask: Any | None = None

    def __post_init__(self) -> None:
        columns = {
            str(column): np.asarray(columnar_row_values(self.rows, str(column)))
            for column in self.rows.columns
        }
        if self.axis_mask is None:
            axis_values = columns.get(self.projection.field_name)
            if axis_values is None:
                self._columns = columns
                return
            axis_mask = self.projection.columnar_mask(axis_values)
        else:
            axis_mask = self.axis_mask
        if bool(np.all(axis_mask)):
            self._columns = columns
            return
        self._columns = {
            column_name: column_values[axis_mask]
            for column_name, column_values in columns.items()
        }


def _label_planes_are_empty(label_planes: tuple[Any, ...]) -> bool:
    import numpy as np

    return all(not np.any(label_plane > 0) for label_plane in label_planes)
