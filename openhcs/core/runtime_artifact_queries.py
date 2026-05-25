"""Semantic queries over typed OpenHCS runtime artifacts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field, fields as dataclass_fields, is_dataclass
from types import MappingProxyType
from typing import Any, ClassVar
from weakref import WeakKeyDictionary

from metaclass_registry import AutoRegisterMeta
from nominal_refactor_advisor.descriptor_algebra import AliasProperty
import numpy as np

from openhcs.core.artifacts import ArtifactKind
from openhcs.core.measurement_lookup_dialect import (
    CURRENT_RUNTIME_MEASUREMENT_LOOKUP_DIALECT,
    RuntimeMeasurementLookupDialect,
    RuntimeMeasurementLookupDialectLike,
    resolve_runtime_measurement_lookup_dialect,
)
from openhcs.core.process_local_cache import ProcessLocalBoundedCache
from openhcs.core.registry_strategies import NominalTypeKeyedStrategyMixin
from openhcs.core.runtime_semantics import (
    FieldSpec,
    MeasurementRowAxisField,
    MeasurementScalarLiteral,
    MeasurementSubject,
    MeasurementTableRowLayout,
    MeasurementScope,
    ObjectLabelMeasurementValues,
    ObjectLabelIdDomainStrategy,
    dense_object_label_id_domain,
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
    ObjectRelationship,
    SpatialGrid,
)


MEASUREMENT_FEATURE_NAME_FIELD = "feature_name"
MEASUREMENT_MEASUREMENT_NAME_FIELD = "measurement_name"
MEASUREMENT_OUTPUT_NAME_FIELD = "output_name"
MEASUREMENT_FEATURE_NAME_FIELDS = (
    MEASUREMENT_FEATURE_NAME_FIELD,
    MEASUREMENT_MEASUREMENT_NAME_FIELD,
    MEASUREMENT_OUTPUT_NAME_FIELD,
)
MEASUREMENT_RESULT_VALUE_FIELD = "result_value"
MEASUREMENT_MEASUREMENT_VALUE_FIELD = "measurement_value"
MEASUREMENT_VALUE_FIELD = "value"
MEASUREMENT_MEAN_VALUE_FIELD = "mean_value"
MEASUREMENT_VALUE_FIELDS = (
    MEASUREMENT_RESULT_VALUE_FIELD,
    MEASUREMENT_MEASUREMENT_VALUE_FIELD,
    MEASUREMENT_VALUE_FIELD,
    MEASUREMENT_MEAN_VALUE_FIELD,
)
MEASUREMENT_OBJECT_NAME_FIELD = "object_name"
MEASUREMENT_SOURCE_IMAGE_NAME_FIELD = "source_image_name"
MEASUREMENT_OBJECT_LABEL_FIELD = "object_label"
MEASUREMENT_OBJECT_NUMBER_FIELD = "object_number"
MEASUREMENT_OBJECT_ID_FIELD = "object_id"
MEASUREMENT_LABEL_FIELD = "label"
MEASUREMENT_OBJECT_ROW_IDENTITY_FIELD = "openhcs_object_row_identity"
MEASUREMENT_OBJECT_ID_FIELDS = (
    MEASUREMENT_OBJECT_LABEL_FIELD,
    MEASUREMENT_OBJECT_NUMBER_FIELD,
    MEASUREMENT_OBJECT_ID_FIELD,
    MEASUREMENT_LABEL_FIELD,
)
MEASUREMENT_UNQUALIFIED_SOURCE_NAMES = frozenset(("", MeasurementScope.IMAGE.value))
_MEASUREMENT_TABLE_CACHE: WeakKeyDictionary[
    RuntimeValueStore,
    dict[tuple[int, str, str | None], tuple[MeasurementTable, ...]],
] = WeakKeyDictionary()
MeasurementValueIndexResult = tuple[dict[int, float], list[float]]
OptionalMeasurementValueIndexResult = MeasurementValueIndexResult | None
MeasurementTablesByObject = Mapping[str, tuple[MeasurementTable, ...]]
MeasurementValueIndexesByObject = Mapping[str, MeasurementValueIndexResult]
MeasurementFeatureValueIndexesByObject = dict[str | None, "MeasurementFeatureValueIndex"]


def _is_structural_missing_measurement_cell(value: object) -> bool:
    """Return whether a columnar value marks structural absence, not a value."""
    return isinstance(value, MeasurementSparseCell)


@dataclass(frozen=True, slots=True)
class ColumnarMeasurementTableSchema:
    """Cached semantic projection for nominal columnar measurement rows."""

    columns: tuple[str, ...]
    normalized_columns: dict[str, str]
    object_names: tuple[str, ...]
    feature_names: frozenset[str]
    object_name_values: Sequence[object] | None
    source_image_name_values: Sequence[object] | None
    object_masks_by_name: dict[str, Any]
    source_masks_by_candidates: dict[tuple[str, ...], Any]

    @classmethod
    def from_table(cls, table: MeasurementTable) -> "ColumnarMeasurementTableSchema":
        rows = table.rows
        if not isinstance(rows, ColumnarRows):
            raise TypeError(
                "ColumnarMeasurementTableSchema requires ColumnarRows, "
                f"got {type(rows).__name__}."
            )
        cached = ColumnarMeasurementTableSchemaCache.process_cache().get_bound(rows)
        if cached is not None:
            return cached

        columns = tuple(str(column) for column in rows.columns)
        normalized_columns = {
            column: normalize_measurement_token(column) for column in columns
        }
        table_object_name = measurement_table_object_name(table)
        object_name_values = (
            columnar_row_values(rows, MEASUREMENT_OBJECT_NAME_FIELD)
            if table_object_name is None
            and MEASUREMENT_OBJECT_NAME_FIELD in columns
            else None
        )
        source_image_name_values = (
            columnar_row_values(rows, MEASUREMENT_SOURCE_IMAGE_NAME_FIELD)
            if MEASUREMENT_SOURCE_IMAGE_NAME_FIELD in columns
            else None
        )
        if table_object_name is not None:
            object_names = (table_object_name,)
        elif object_name_values is not None:
            object_names = tuple(
                dict.fromkeys(
                    object_name
                    for value in object_name_values
                    for object_name in (str(value).strip(),)
                    if object_name
                )
            )
        else:
            object_names = ()

        return ColumnarMeasurementTableSchemaCache.process_cache().put_bound(
            rows,
            cls(
                columns=columns,
                normalized_columns=normalized_columns,
                object_names=object_names,
                feature_names=MeasurementTableObjectFeatureSemantics.feature_names_from_names(
                    columns,
                    table,
                ),
                object_name_values=object_name_values,
                source_image_name_values=source_image_name_values,
                object_masks_by_name={},
                source_masks_by_candidates={},
            ),
        )

    def object_mask(self, object_name: str) -> Any | None:
        """Return a boolean row mask for a row-owned object table."""
        if self.object_name_values is None:
            return None
        cached = self.object_masks_by_name.get(object_name)
        if cached is not None:
            return cached
        import numpy as np

        normalized_object_name = normalize_runtime_identifier(object_name)
        normalized_objects = np.asarray(
            [
                normalize_runtime_identifier(value)
                for value in self.object_name_values
            ],
            dtype=object,
        )
        mask = normalized_objects == normalized_object_name
        self.object_masks_by_name[object_name] = mask
        return mask

    def source_mask(self, source_candidates: tuple[str, ...]) -> Any | None:
        """Return a boolean row mask for a source-qualified columnar table."""
        if self.source_image_name_values is None or not source_candidates:
            return None
        cached = self.source_masks_by_candidates.get(source_candidates)
        if cached is not None:
            return cached
        import numpy as np

        normalized_sources = np.asarray(
            [
                normalize_measurement_token(str(value))
                for value in self.source_image_name_values
            ],
            dtype=object,
        )
        mask = np.isin(normalized_sources, np.asarray(source_candidates, dtype=object))
        self.source_masks_by_candidates[source_candidates] = mask
        return mask

    def matching_feature_column(self, query: "MeasurementFeatureQuery") -> str | None:
        """Return the column matching a measurement feature query."""
        normalized_feature = normalize_measurement_token(query.feature_name)
        for column, normalized_column in self.normalized_columns.items():
            if normalized_column == normalized_feature:
                return column
        candidates = query.field_candidates
        for candidate in candidates:
            for column, normalized_column in self.normalized_columns.items():
                if normalized_column == candidate:
                    return column
        return None


class IdentityBoundProcessCache(
    ProcessLocalBoundedCache[
        int,
        tuple[object, Any],
    ],
    metaclass=AutoRegisterMeta,
):
    """Process-local cache whose keys are protected against id reuse."""

    __registry_key__ = "registry_key"
    __skip_if_no_key__ = True

    max_entries = 4096
    registry_key: ClassVar[str | None] = None

    def get_bound(
        self,
        owner: object,
    ) -> Any | None:
        cache_key = id(owner)
        cached = self.cached_value(cache_key)
        if cached is None:
            return None
        cached_owner, value = cached
        if cached_owner is not owner:
            del self.entries[cache_key]
            return None
        return value

    def put_bound(
        self,
        owner: object,
        value: Any,
    ) -> Any:
        return self.store_value(id(owner), (owner, value))[1]


class ColumnarMeasurementTableSchemaCache(IdentityBoundProcessCache):
    """Process-local semantic cache keyed by a columnar row object identity."""

    registry_key = "columnar_measurement_table_schema"


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
        scope = MeasurementScope(self.scope)
        object.__setattr__(self, "scope", scope)
        if self.name == "":
            raise ValueError("MeasurementScopeQuery.name cannot be empty.")

    def matches(self, table: MeasurementTable) -> bool:
        if table.subject.scope is not self.scope:
            return False
        return self.name is None or table.subject.name == self.name


@dataclass(frozen=True, slots=True)
class MeasurementFeatureQuery:
    """Query for measurement rows carrying one semantic feature value."""

    feature_name: str
    object_name: str | None = None
    dialect: RuntimeMeasurementLookupDialectLike = (
        CURRENT_RUNTIME_MEASUREMENT_LOOKUP_DIALECT
    )
    _field_candidates: tuple[str, ...] = field(init=False, repr=False)
    _source_candidates: tuple[str, ...] = field(init=False, repr=False)
    _query_object_name: str | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.feature_name:
            raise ValueError("MeasurementFeatureQuery.feature_name cannot be empty.")
        if self.object_name == "":
            raise ValueError("MeasurementFeatureQuery.object_name cannot be empty.")
        lookup = resolve_runtime_measurement_lookup_dialect(
            self.dialect
        ).feature_lookup(self.feature_name)
        object.__setattr__(self, "_field_candidates", lookup.field_aliases)
        object.__setattr__(self, "_source_candidates", lookup.source_aliases)
        object.__setattr__(
            self,
            "_query_object_name",
            lookup.query_object_name(self.object_name),
        )

    @property
    def field_candidates(self) -> tuple[str, ...]:
        return self._field_candidates

    @property
    def source_candidates(self) -> tuple[str, ...]:
        return self._source_candidates

    @property
    def query_object_name(self) -> str | None:
        """Return the dialect-effective object constraint for this feature."""
        return self._query_object_name

    def row_value(self, row: object) -> object | None:
        """Return the row value matching this feature query, if present."""
        row_mapping = measurement_row_mapping(row)
        if not self._matches_object(row_mapping):
            return None

        candidates = self.field_candidates
        if measurement_row_feature_matches(row_mapping, candidates):
            if not measurement_row_source_matches_feature(row_mapping, self):
                return None
            return measurement_row_first_value(row_mapping)

        if not measurement_row_source_matches_feature(row_mapping, self):
            return None
        for field_name in matching_measurement_fields(row_mapping, candidates):
            value = row_mapping[field_name]
            if value not in (None, "") and not _is_structural_missing_measurement_cell(value):
                return value
        return None

    def value_index(
        self,
        measurement_tables: tuple[MeasurementTable, ...],
    ) -> MeasurementValueIndexResult:
        """Return object-id and positional values for this feature."""
        value_index = self.optional_value_index(measurement_tables)
        if value_index is None:
            raise ValueError(
                f"Could not resolve measurement feature {self.feature_name!r}; "
                f"tables={self.table_summaries(measurement_tables)!r}; "
                f"matches={self.table_match_diagnostics(measurement_tables)!r}."
            )
        return value_index

    def table_match_diagnostics(
        self,
        measurement_tables: tuple[MeasurementTable, ...],
    ) -> tuple[str, ...]:
        """Return compact row/field diagnostics for unresolved feature queries."""
        diagnostics: list[str] = []
        for table in measurement_tables:
            rows = measurement_rows((table,))
            first_row = measurement_row_mapping(rows[0]) if rows else {}
            matching_fields = matching_measurement_fields(
                first_row,
                self.field_candidates,
            )
            diagnostics.append(
                f"{table.name}/object={measurement_table_object_name(table) or '<none>'}/"
                f"query_object={self.query_object_name or '<none>'}/"
                f"row_count={len(rows)}/first_object={measurement_row_object_name(first_row) or '<none>'}/"
                f"matching_fields={matching_fields}/"
                f"first_keys={tuple(str(key) for key in tuple(first_row)[:12])}"
            )
        return tuple(diagnostics)

    def table_summaries(
        self,
        measurement_tables: tuple[MeasurementTable, ...],
    ) -> tuple[str, ...]:
        """Return compact diagnostics for tables searched by this query."""
        summaries: list[str] = []
        for table in measurement_tables:
            semantics = MeasurementTableObjectFeatureSemantics.from_table(table)
            features = tuple(sorted(semantics.feature_names))
            feature_column = None
            row_count = "unknown"
            object_match_count = "unknown"
            axis_values = ()
            if isinstance(table.rows, ColumnarRows):
                schema = ColumnarMeasurementTableSchema.from_table(table)
                feature_column = schema.matching_feature_column(self)
                row_count = str(len(table.rows))
                query_object_name = self.query_object_name
                object_mask = (
                    schema.object_mask(query_object_name)
                    if query_object_name is not None
                    and measurement_table_object_name(table) is None
                    else None
                )
                if object_mask is not None:
                    object_match_count = str(int(object_mask.sum()))
                axis_values = tuple(
                    tuple(sorted(measurement_table_axis_values(table, axis)))
                    for axis in (
                        MeasurementRowAxisField.SLICE_INDEX,
                        MeasurementRowAxisField.IMAGE_NUMBER,
                    )
                )
            summaries.append(
                f"{table.name}/object={measurement_table_object_name(table) or '<none>'}/"
                f"source={table.source_image_name or '<none>'}/"
                f"rows={type(table.rows).__name__}/objects={semantics.object_names[:8]}/"
                f"feature_column={feature_column or '<none>'}/"
                f"row_count={row_count}/object_matches={object_match_count}/"
                f"axes={axis_values}/feature_count={len(features)}/features={features[:8]}"
            )
        return tuple(summaries)

    def optional_value_index(
        self,
        measurement_tables: tuple[MeasurementTable, ...],
    ) -> OptionalMeasurementValueIndexResult:
        """Return feature values when present, otherwise ``None``."""
        value_index = MeasurementFeatureValueIndex.from_tables(
            measurement_tables,
            self,
        )
        return value_index.as_query_result() if value_index.present else None

    def table_may_carry_feature(
        self,
        table: MeasurementTable,
        semantics: "MeasurementTableObjectFeatureSemantics | None" = None,
    ) -> bool:
        """Return whether table ownership and feature schema can satisfy this query."""
        if not self.table_source_matches_feature(table):
            return False
        table_semantics = (
            MeasurementTableObjectFeatureSemantics.from_table(table)
            if semantics is None
            else semantics
        )
        if not table_semantics.feature_names:
            return True
        candidates = frozenset(self.field_candidates)
        return any(
            normalize_measurement_token(feature_name) in candidates
            for feature_name in table_semantics.feature_names
        )

    def table_source_matches_feature(self, table: MeasurementTable) -> bool:
        """Return whether table-level source ownership matches this feature query."""
        source_image_name = table.source_image_name
        if source_image_name is None:
            return True
        normalized_source = normalize_measurement_token(source_image_name)
        if normalized_source in MEASUREMENT_UNQUALIFIED_SOURCE_NAMES:
            return True
        return normalized_source in self.source_candidates

    def scalar_value(self, measurement_tables: tuple[MeasurementTable, ...]) -> float:
        """Return exactly one scalar measurement value for this feature."""
        values_by_label, positional_values = self.value_index(measurement_tables)
        values = (
            tuple(values_by_label[label] for label in sorted(values_by_label))
            if values_by_label
            else tuple(positional_values)
        )
        if len(values) != 1:
            raise ValueError(
                f"Measurement feature {self.feature_name!r} resolved to "
                f"{len(values)} values; expected exactly one scalar value."
            )
        return float(values[0])

    def _matches_object(self, row: Mapping[str, object]) -> bool:
        row_object_name = measurement_row_object_name(row)
        query_object_name = self.query_object_name
        return (
            query_object_name is None
            or row_object_name is None
            or row_object_name == query_object_name
        )


@dataclass(frozen=True, slots=True)
class MeasurementObjectFeatureVectorBatchQuery:
    """Query multiple object-domain vectors for one measurement feature."""

    feature_name: str
    object_names: tuple[str, ...]
    dialect: RuntimeMeasurementLookupDialectLike = (
        CURRENT_RUNTIME_MEASUREMENT_LOOKUP_DIALECT
    )

    def __post_init__(self) -> None:
        if not self.feature_name:
            raise ValueError(
                "MeasurementObjectFeatureVectorBatchQuery.feature_name cannot be empty."
            )
        object_names = tuple(dict.fromkeys(str(name) for name in self.object_names))
        if any(not name for name in object_names):
            raise ValueError(
                "MeasurementObjectFeatureVectorBatchQuery.object_names cannot contain "
                "empty names."
            )
        object.__setattr__(self, "object_names", object_names)

    def value_indexes(
        self,
        measurement_tables_by_object: MeasurementTablesByObject,
    ) -> MeasurementValueIndexesByObject:
        """Return value indexes keyed by object name for this feature."""
        lookup = resolve_runtime_measurement_lookup_dialect(
            self.dialect
        ).feature_lookup(self.feature_name)
        query_objects_by_requested_object = {
            object_name: lookup.query_object_name(object_name)
            for object_name in self.object_names
        }
        indexes_by_object = {
            object_name: MeasurementFeatureValueIndex()
            for object_name in self.object_names
        }
        objects_by_table_id: dict[int, list[str]] = {}
        tables_by_id: dict[int, MeasurementTable] = {}
        for object_name in self.object_names:
            for table in measurement_tables_by_object[object_name]:
                table_id = id(table)
                tables_by_id[table_id] = table
                objects_by_table_id.setdefault(table_id, []).append(object_name)

        table_query = MeasurementFeatureQuery(
            self.feature_name,
            dialect=self.dialect,
        )
        for table_id, table in tables_by_id.items():
            table_object_names = tuple(dict.fromkeys(objects_by_table_id[table_id]))
            columnar_indexes = self._columnar_value_indexes(
                table,
                table_object_names,
                query_objects_by_requested_object,
                table_query,
            )
            if columnar_indexes is not None:
                for object_name, object_index in columnar_indexes.items():
                    indexes_by_object[object_name] = indexes_by_object[
                        object_name
                    ].merged(object_index)
                continue
            row_sequence_index = MeasurementRowSequenceFeatureValueIndexBuild(
                table,
                table_query,
            ).index()
            if row_sequence_index is not None:
                for object_name in table_object_names:
                    object_index = row_sequence_index.for_object(
                        query_objects_by_requested_object[object_name]
                    )
                    if object_index is not None:
                        indexes_by_object[object_name] = indexes_by_object[
                            object_name
                        ].merged(MeasurementFeatureValueIndex(*object_index))
                continue

            for object_name in table_object_names:
                object_index = MeasurementFeatureQuery(
                    self.feature_name,
                    object_name=query_objects_by_requested_object[object_name],
                    dialect=self.dialect,
                ).optional_value_index((table,))
                if object_index is not None:
                    indexes_by_object[object_name] = indexes_by_object[
                        object_name
                    ].merged(MeasurementFeatureValueIndex(*object_index))

        missing_object_names = tuple(
            object_name
            for object_name, index in indexes_by_object.items()
            if not index.present
        )
        if missing_object_names:
            table_summaries = tuple(
                f"{object_name}:"
                + ",".join(
                    f"{measurement_table_object_name(table) or '<none>'}/"
                    f"{type(table.rows).__name__}/"
                    f"{len(MeasurementTableObjectFeatureSemantics.from_table(table).feature_names)}"
                    for table in measurement_tables_by_object[object_name]
                )
                for object_name in missing_object_names
            )
            raise ValueError(
                f"Could not resolve measurement feature {self.feature_name!r} "
                f"for object(s) {missing_object_names!r}; tables={table_summaries!r}."
            )
        return {
            object_name: indexes_by_object[object_name].as_query_result()
            for object_name in self.object_names
        }

    def _columnar_value_indexes(
        self,
        table: MeasurementTable,
        requested_object_names: tuple[str, ...],
        query_objects_by_requested_object: Mapping[str, str | None],
        table_query: MeasurementFeatureQuery,
    ) -> dict[str, "MeasurementFeatureValueIndex"] | None:
        """Return per-object indexes from one columnar table scan, when possible."""
        rows = table.rows
        if not isinstance(rows, ColumnarRows):
            return None
        if not table_query.table_source_matches_feature(table):
            return {}

        schema = ColumnarMeasurementTableSchema.from_table(table)
        feature_column = schema.matching_feature_column(table_query)
        if feature_column is None:
            return {}

        raw_values = np.asarray(columnar_row_values(rows, feature_column), dtype=object)
        value_mask = np.asarray(
            [
                not _is_structural_missing_measurement_cell(value)
                and MeasurementScalarLiteral(value).is_present_measurement_value
                for value in raw_values
            ],
            dtype=bool,
        )
        source_mask = (
            None
            if table.source_image_name is not None
            else schema.source_mask(table_query.source_candidates)
        )
        base_mask = (
            value_mask
            if source_mask is None
            else np.logical_and(value_mask, source_mask)
        )
        values = raw_values.astype(float, copy=False)
        object_id_field = MeasurementRowSequenceFeatureValueIndex.matching_object_id_field(
            schema.columns,
            measurement_table_object_id_field(table),
        )
        object_ids = (
            None
            if object_id_field is None or object_id_field not in schema.columns
            else np.asarray(columnar_row_values(rows, object_id_field), dtype=object)
        )
        table_object_name = measurement_table_object_name(table)
        indexes: dict[str, MeasurementFeatureValueIndex] = {}
        for object_name in requested_object_names:
            query_object_name = query_objects_by_requested_object[object_name]
            object_mask: Any | None = None
            if query_object_name is not None:
                if table_object_name not in (None, query_object_name):
                    continue
                if table_object_name is None:
                    if schema.object_name_values is None:
                        continue
                    object_mask = schema.object_mask(query_object_name)
            effective_mask = (
                base_mask
                if object_mask is None
                else np.logical_and(base_mask, object_mask)
            )
            object_values = values[effective_mask]
            if object_id_field is not None and object_ids is not None:
                indexes[object_name] = MeasurementFeatureValueIndex(
                    {
                        object_label: float(value)
                        for raw_object_id, value in zip(
                            object_ids[effective_mask],
                            object_values,
                            strict=True,
                        )
                        for object_label in (
                            MeasurementObjectLabelResolution(raw_object_id).object_label,
                        )
                        if object_label is not None
                    },
                    [],
                )
                continue
            indexes[object_name] = MeasurementFeatureValueIndex(
                {},
                [float(value) for value in object_values],
            )
        return indexes


@dataclass(frozen=True, slots=True)
class MeasurementFeatureValueIndex:
    """Object-label and positional values for one measurement feature."""

    values_by_label: dict[int, float] = field(default_factory=dict)
    positional_values: list[float] = field(default_factory=list)

    @classmethod
    def from_tables(
        cls,
        measurement_tables: tuple[MeasurementTable, ...],
        query: MeasurementFeatureQuery,
    ) -> "MeasurementFeatureValueIndex":
        index = cls()
        for table in measurement_tables:
            index = index.merged(cls.from_table(table, query))
        return index

    @classmethod
    def from_table(
        cls,
        table: MeasurementTable,
        query: MeasurementFeatureQuery,
    ) -> "MeasurementFeatureValueIndex":
        columnar_index = cls.from_columnar_table(table, query)
        if columnar_index is not None:
            return columnar_index

        row_sequence_index = MeasurementRowSequenceFeatureValueIndexBuild(
            table,
            query,
        ).index()
        if row_sequence_index is not None:
            object_index = row_sequence_index.for_object(query.query_object_name)
            return (
                cls()
                if object_index is None
                else cls(*object_index)
            )

        if _is_wide_row_sequence_measurement_table(table):
            return cls()
        return cls.from_rows(
            measurement_rows((table,)),
            query,
            object_id_field=measurement_table_object_id_field(table),
        )

    @classmethod
    def from_columnar_table(
        cls,
        table: MeasurementTable,
        query: MeasurementFeatureQuery,
    ) -> "MeasurementFeatureValueIndex | None":
        """Return a feature vector directly from columnar rows when possible."""
        import numpy as np

        rows = table.rows
        if not isinstance(rows, ColumnarRows):
            return None

        schema = ColumnarMeasurementTableSchema.from_table(table)
        columns = schema.columns
        object_mask: Any | None = None
        if not query.table_source_matches_feature(table):
            return cls()
        source_mask = (
            None
            if table.source_image_name is not None
            else schema.source_mask(query.source_candidates)
        )
        query_object_name = query.query_object_name
        if query_object_name is not None:
            table_object = measurement_table_object_name(table)
            if table_object not in (None, query_object_name):
                return None
            if table_object is None and MEASUREMENT_OBJECT_NAME_FIELD in columns:
                object_mask = schema.object_mask(query_object_name)
            elif table_object is None:
                return None

        feature_column = schema.matching_feature_column(query)
        if feature_column is None:
            return None

        raw_values = np.asarray(columnar_row_values(rows, feature_column), dtype=object)
        value_mask = np.asarray(
            [
                not _is_structural_missing_measurement_cell(value)
                and MeasurementScalarLiteral(value).is_present_measurement_value
                for value in raw_values
            ],
            dtype=bool,
        )
        values = raw_values[value_mask].astype(float, copy=False)
        if object_mask is not None:
            values = values[object_mask[value_mask]]
        if source_mask is not None:
            effective_source_mask = source_mask[value_mask]
            values = values[
                effective_source_mask
                if object_mask is None
                else effective_source_mask[object_mask[value_mask]]
            ]
        object_id_field = MeasurementRowSequenceFeatureValueIndex.matching_object_id_field(
            columns,
            measurement_table_object_id_field(table),
        )
        if object_id_field is not None and object_id_field in columns:
            object_ids = np.asarray(
                columnar_row_values(rows, object_id_field),
                dtype=object,
            )
            object_ids = object_ids[value_mask]
            if object_mask is not None:
                object_ids = object_ids[object_mask[value_mask]]
            if source_mask is not None:
                effective_source_mask = source_mask[value_mask]
                object_ids = object_ids[
                    effective_source_mask
                    if object_mask is None
                    else effective_source_mask[object_mask[value_mask]]
                ]
            return cls(
                {
                    object_label: float(value)
                    for raw_object_id, value in zip(object_ids, values, strict=True)
                    for object_label in (
                        MeasurementObjectLabelResolution(raw_object_id).object_label,
                    )
                    if object_label is not None
                },
                [],
            )
        return cls({}, [float(value) for value in values])

    @classmethod
    def from_rows(
        cls,
        rows: tuple[object, ...],
        query: MeasurementFeatureQuery,
        *,
        object_id_field: str | None,
    ) -> "MeasurementFeatureValueIndex":
        index = cls()
        for row in rows:
            row_mapping = measurement_row_mapping(row)
            if (
                query.query_object_name is not None
                and MEASUREMENT_SOURCE_IMAGE_NAME_FIELD in row_mapping
                and not measurement_row_has_object_identity(row_mapping)
            ):
                continue
            value = query.row_value(row)
            if value is None:
                continue
            object_label = measurement_object_label(
                row_mapping,
                object_id_field=object_id_field,
            )
            index.add(object_label, value)
        return index

    @property
    def present(self) -> bool:
        return bool(self.values_by_label or self.positional_values)

    def add(self, object_label: int | None, value: object) -> None:
        if _is_structural_missing_measurement_cell(value):
            return
        numeric_value = float(value)
        if object_label is None:
            self.positional_values.append(numeric_value)
            return
        self.values_by_label[object_label] = numeric_value

    def merged(self, other: "MeasurementFeatureValueIndex") -> "MeasurementFeatureValueIndex":
        values_by_label = dict(self.values_by_label)
        values_by_label.update(other.values_by_label)
        return MeasurementFeatureValueIndex(
            values_by_label,
            [*self.positional_values, *other.positional_values],
        )

    def as_query_result(self) -> MeasurementValueIndexResult:
        return self.values_by_label, self.positional_values


@dataclass(frozen=True, slots=True)
class MeasurementRowSequenceFeatureValueIndex:
    """Feature values indexed once per row sequence and projected per object."""

    values_by_object: MeasurementFeatureValueIndexesByObject

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[object],
        *,
        feature_field: str,
        object_id_field: str | None,
        query: MeasurementFeatureQuery,
    ) -> "MeasurementRowSequenceFeatureValueIndex | None":
        """Build a row-sequence index once field ownership has been resolved."""
        values_by_object: MeasurementFeatureValueIndexesByObject = {}
        for row in rows:
            row_mapping = measurement_row_mapping(row)
            if not measurement_row_source_matches_feature(row_mapping, query):
                continue
            value = row_mapping.get(feature_field)
            if value in (None, "") or _is_structural_missing_measurement_cell(value):
                continue
            object_label = measurement_object_label(
                row_mapping,
                object_id_field=object_id_field,
            )
            object_name = measurement_row_object_name(row_mapping)
            values_by_object.setdefault(
                object_name,
                MeasurementFeatureValueIndex(),
            ).add(object_label, value)

        if not any(index.present for index in values_by_object.values()):
            return None
        return cls(values_by_object)

    @staticmethod
    def field_names(
        rows: Sequence[object],
        query: MeasurementFeatureQuery,
    ) -> tuple[str, ...]:
        """Return field names for homogeneous or heterogeneous row sequences."""
        first_row = measurement_row_mapping(rows[0])
        first_row_names = tuple(str(field_name) for field_name in first_row)
        if any(field in first_row for field in MEASUREMENT_FEATURE_NAME_FIELDS):
            return first_row_names

        field_names: list[str] = []
        seen: set[str] = set(first_row_names)
        field_names.extend(first_row_names)
        candidate_rank = {
            candidate: index for index, candidate in enumerate(query.field_candidates)
        }
        found_feature_rank = min(
            (
                candidate_rank[normalize_measurement_token(field_name)]
                for field_name in first_row_names
                if normalize_measurement_token(field_name) in candidate_rank
            ),
            default=None,
        )
        found_object_id = any(
            field_name in MEASUREMENT_OBJECT_ID_FIELDS
            for field_name in first_row_names
        )
        for row in rows[1:]:
            for field_name in measurement_row_mapping(row):
                normalized_name = str(field_name)
                if normalized_name not in seen:
                    seen.add(normalized_name)
                    field_names.append(normalized_name)
                normalized_token = normalize_measurement_token(normalized_name)
                if normalized_token in candidate_rank:
                    rank = candidate_rank[normalized_token]
                    found_feature_rank = (
                        rank
                        if found_feature_rank is None
                        else min(found_feature_rank, rank)
                    )
                if normalized_name in MEASUREMENT_OBJECT_ID_FIELDS:
                    found_object_id = True
            if found_feature_rank == 0 and found_object_id:
                break
        return tuple(field_names)

    @staticmethod
    def matching_row_value_field(
        field_names: tuple[str, ...],
        query: MeasurementFeatureQuery,
        *,
        table_source_image_name: str | None,
    ) -> str | None:
        fields = MeasurementRowSequenceFeatureValueIndex.matching_row_value_fields(
            field_names,
            query,
            table_source_image_name=table_source_image_name,
        )
        return fields[0] if fields else None

    @staticmethod
    def matching_row_value_fields(
        field_names: tuple[str, ...],
        query: MeasurementFeatureQuery,
        *,
        table_source_image_name: str | None,
    ) -> tuple[str, ...]:
        candidates = query.field_candidates
        if table_source_image_name is not None:
            normalized_source = normalize_measurement_token(table_source_image_name)
            if (
                normalized_source not in MEASUREMENT_UNQUALIFIED_SOURCE_NAMES
                and normalized_source not in query.source_candidates
            ):
                return ()
        return matching_measurement_fields(
            {field_name: None for field_name in field_names},
            candidates,
        )

    @staticmethod
    def matching_object_id_field(
        field_names: tuple[str, ...],
        declared_object_id_field: str | None,
    ) -> str | None:
        if (
            declared_object_id_field is not None
            and declared_object_id_field in field_names
        ):
            return declared_object_id_field
        for field_name in field_names:
            if field_name in MEASUREMENT_OBJECT_ID_FIELDS:
                return field_name
        return None

    def for_object(
        self,
        object_name: str | None,
    ) -> OptionalMeasurementValueIndexResult:
        if object_name is None:
            merged_index = MeasurementFeatureValueIndex()
            for index in self.values_by_object.values():
                merged_index = merged_index.merged(index)
            return merged_index.as_query_result() if merged_index.present else None

        default_index = self.values_by_object.get(None)
        object_index = self.values_by_object.get(object_name)
        if default_index is None:
            return object_index.as_query_result() if object_index is not None else None
        if object_index is None:
            return default_index.as_query_result()
        return default_index.merged(object_index).as_query_result()


@dataclass(frozen=True, slots=True)
class MeasurementRowSequenceLayout:
    """Declared row shape for sequence-backed measurement tables."""

    field_names: tuple[str, ...]
    declares_feature_names: bool

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[object],
        query: MeasurementFeatureQuery | None = None,
    ) -> "MeasurementRowSequenceLayout":
        field_names: list[str] = []
        seen: set[str] = set()
        declares_feature_names = False
        candidate_rank = (
            {candidate: index for index, candidate in enumerate(query.field_candidates)}
            if query is not None
            else {}
        )
        found_feature_rank = 0 if not candidate_rank else None
        found_object_id = False

        for row in rows:
            for field_name in measurement_row_mapping(row):
                normalized_name = str(field_name)
                if normalized_name not in seen:
                    seen.add(normalized_name)
                    field_names.append(normalized_name)
                declares_feature_names = (
                    declares_feature_names
                    or normalized_name in MEASUREMENT_FEATURE_NAME_FIELDS
                )
                normalized_token = normalize_measurement_token(normalized_name)
                if normalized_token in candidate_rank:
                    rank = candidate_rank[normalized_token]
                    found_feature_rank = (
                        rank
                        if found_feature_rank is None
                        else min(found_feature_rank, rank)
                    )
                if normalized_name in MEASUREMENT_OBJECT_ID_FIELDS:
                    found_object_id = True
            if declares_feature_names:
                found_feature_rank = 0
            if found_feature_rank == 0 and found_object_id:
                break
        return cls(tuple(field_names), declares_feature_names)

    @property
    def is_wide_only(self) -> bool:
        """Return whether rows expose only direct measurement columns."""
        return not self.declares_feature_names


@dataclass(frozen=True, slots=True)
class MeasurementRowSequenceFeatureValueIndexBuild:
    """Staged build request for row-sequence feature indexes."""

    table: MeasurementTable
    query: MeasurementFeatureQuery

    @property
    def rows(self) -> tuple[object, ...] | None:
        table_rows = self.table.rows
        if isinstance(table_rows, ColumnarRows):
            return None
        if not isinstance(table_rows, list | tuple) or not table_rows:
            return None
        return tuple(table_rows)

    @property
    def field_names(self) -> tuple[str, ...] | None:
        layout = self.layout
        return None if layout is None else layout.field_names

    @property
    def layout(self) -> MeasurementRowSequenceLayout | None:
        rows = self.rows
        if rows is None:
            return None
        return MeasurementRowSequenceLayout.from_rows(rows, self.query)

    @property
    def feature_field(self) -> str | None:
        feature_fields = self.feature_fields
        return feature_fields[0] if feature_fields else None

    @property
    def feature_fields(self) -> tuple[str, ...]:
        field_names = self.field_names
        if field_names is None:
            return ()
        return MeasurementRowSequenceFeatureValueIndex.matching_row_value_fields(
            field_names,
            self.query,
            table_source_image_name=self.table.source_image_name,
        )

    @property
    def object_id_field(self) -> str | None:
        field_names = self.field_names
        if field_names is None:
            return None
        return MeasurementRowSequenceFeatureValueIndex.matching_object_id_field(
            field_names,
            measurement_table_object_id_field(self.table),
        )

    def index(self) -> MeasurementRowSequenceFeatureValueIndex | None:
        rows = self.rows
        if rows is None:
            return None
        for feature_field in self.feature_fields:
            index = MeasurementRowSequenceFeatureValueIndex.from_rows(
                rows,
                feature_field=feature_field,
                object_id_field=self.object_id_field,
                query=self.query,
            )
            if index is not None:
                return index
        return None


class MeasurementTableObjectFeatureSemanticsCache(IdentityBoundProcessCache):
    """Bounded process-local cache for immutable measurement-table semantics."""

    registry_key = "measurement_table_object_feature_semantics"

@dataclass(frozen=True, slots=True)
class MeasurementTableObjectFeatureSemantics:
    """Object and feature declarations carried by one measurement table."""

    object_names: tuple[str, ...]
    feature_names: frozenset[str]

    @classmethod
    def from_table(cls, table: MeasurementTable) -> "MeasurementTableObjectFeatureSemantics":
        cache = MeasurementTableObjectFeatureSemanticsCache.process_cache()
        cached = cache.get_bound(table)
        if cached is not None:
            return cached
        declared = cls.from_table_declarations(table)
        if declared is not None:
            return cache.put_bound(table, declared)
        return cache.put_bound(
            table,
            cls(
                object_names=cls._object_names(table, measurement_rows((table,))),
                feature_names=cls._feature_names(table, measurement_rows((table,))),
            ),
        )

    @classmethod
    def from_table_declarations(
        cls,
        table: MeasurementTable,
    ) -> "MeasurementTableObjectFeatureSemantics | None":
        """Return semantics from table-level schema when rows need not be scanned."""
        object_name = measurement_table_object_name(table)
        if isinstance(table.rows, ColumnarRows):
            schema = ColumnarMeasurementTableSchema.from_table(table)
            if object_name is None and not schema.object_names:
                return None
            return cls(
                object_names=(
                    (object_name,)
                    if object_name is not None
                    else schema.object_names
                ),
                feature_names=schema.feature_names,
            )
        if isinstance(table.rows, list | tuple):
            return None
        if object_name is None:
            return None
        if not table.fields:
            return None
        field_names = tuple(field.name for field in table.fields)
        if any(field_name in MEASUREMENT_FEATURE_NAME_FIELDS for field_name in field_names):
            return None
        return cls(
            object_names=(object_name,),
            feature_names=cls.feature_names_from_names(field_names, table),
        )

    @staticmethod
    def _object_names(
        table: MeasurementTable,
        rows: tuple[object, ...],
    ) -> tuple[str, ...]:
        table_object_name = measurement_table_object_name(table)
        if table_object_name is not None:
            return (table_object_name,)
        return tuple(
            dict.fromkeys(
                object_name
                for row in rows
                for object_name in (
                    measurement_row_object_name(measurement_row_mapping(row)),
                )
                if object_name is not None
            )
        )

    @staticmethod
    def _feature_names(
        table: MeasurementTable,
        rows: tuple[object, ...],
    ) -> frozenset[str]:
        feature_names: set[str] = set()
        for row in rows:
            row_mapping = measurement_row_mapping(row)
            for field_name in MEASUREMENT_FEATURE_NAME_FIELDS:
                value = row_mapping.get(field_name)
                if value not in (None, ""):
                    feature_names.add(str(value))
        if feature_names:
            return frozenset(feature_names)
        return MeasurementTableObjectFeatureSemantics.feature_names_from_names(
            tuple(
                dict.fromkeys(
                    str(field_name)
                    for row in rows
                    for field_name in measurement_row_mapping(row)
                )
            ),
            table,
        )

    @staticmethod
    def feature_names_from_fields(
        fields: tuple[FieldSpec, ...],
        table: MeasurementTable,
    ) -> frozenset[str]:
        """Return wide-form feature names declared by measurement-table fields."""
        return MeasurementTableObjectFeatureSemantics.feature_names_from_names(
            tuple(field.name for field in fields),
            table,
        )

    @staticmethod
    def feature_names_from_names(
        field_names: tuple[str, ...],
        table: MeasurementTable,
    ) -> frozenset[str]:
        """Return wide-form feature names declared by measurement field names."""
        non_feature_fields = {
            MEASUREMENT_OBJECT_NAME_FIELD,
            MEASUREMENT_SOURCE_IMAGE_NAME_FIELD,
            *(str(field_name) for field_name in MEASUREMENT_OBJECT_ID_FIELDS),
            *(field_name for field_name in MEASUREMENT_FEATURE_NAME_FIELDS),
        }
        if table.object_id_field is not None:
            non_feature_fields.add(table.object_id_field)
        return frozenset(field_name for field_name in field_names if field_name not in non_feature_fields)


@dataclass(frozen=True, slots=True)
class ObjectMeasurementLabelPlaneBinding:
    """Bind one object measurement feature onto a label-plane object domain."""

    measurement_tables: tuple[MeasurementTable, ...]
    object_name: str
    feature_name: str
    labels: object
    dialect: RuntimeMeasurementLookupDialectLike = (
        CURRENT_RUNTIME_MEASUREMENT_LOOKUP_DIALECT
    )

    @property
    def object_domain(self) -> tuple[int, ...]:
        """Return the object IDs represented by this label plane."""
        return dense_object_label_id_domain(self.labels)

    def values(self) -> Any:
        """Return measurement values aligned to the label-plane domain."""
        policy = ObjectMeasurementLabelPlaneBindingPolicy.for_nominal_value(self)
        if policy is None:
            raise TypeError(
                "No ObjectMeasurementLabelPlaneBindingPolicy registered for "
                f"{type(self).__name__}."
            )
        return policy.values(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class IndexedObjectMeasurementLabelPlaneBinding(ObjectMeasurementLabelPlaneBinding):
    """Bind an already-indexed measurement feature onto a label plane."""

    indexed_values: MeasurementValueIndexResult


class ObjectMeasurementLabelPlaneBindingPolicy(
    NominalTypeKeyedStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered alignment policy for object measurements and label planes."""

    __registry_key__ = "value_type_label"
    __skip_if_no_key__ = True
    value_type: ClassVar[type[object] | tuple[type[object], ...] | None] = None
    value_type_label: ClassVar[str | None] = None

    @abstractmethod
    def values(self, binding: ObjectMeasurementLabelPlaneBinding) -> Any:
        """Return values aligned to ``binding.object_domain``."""


class DefaultObjectMeasurementLabelPlaneBindingPolicy(
    ObjectMeasurementLabelPlaneBindingPolicy
):
    """Align label-keyed measurements first, then compact positional rows."""

    value_type = ObjectMeasurementLabelPlaneBinding

    def values(self, binding: ObjectMeasurementLabelPlaneBinding) -> Any:
        import numpy as np

        domain = binding.object_domain
        values_by_label, positional_values = MeasurementFeatureQuery(
            binding.feature_name,
            object_name=binding.object_name,
            dialect=binding.dialect,
        ).value_index(binding.measurement_tables)
        return self.values_for_index(binding, values_by_label, positional_values)

    def values_for_index(
        self,
        binding: ObjectMeasurementLabelPlaneBinding,
        values_by_label: Mapping[int, float],
        positional_values: Sequence[float],
    ) -> Any:
        import numpy as np

        domain = binding.object_domain
        if not domain:
            return np.array([], dtype=float)
        if values_by_label:
            return np.array([values_by_label.get(label, np.nan) for label in domain])
        if positional_values:
            return np.array(positional_values[: len(domain)])
        if binding.measurement_tables:
            summaries = MeasurementFeatureQuery(
                binding.feature_name,
                object_name=binding.object_name or None,
                dialect=binding.dialect,
            ).table_summaries(binding.measurement_tables)
            raise ValueError(
                f"Could not resolve measurement feature {binding.feature_name!r}; "
                f"tables={summaries!r}."
            )
        raise ValueError(
            f"Could not resolve measurement feature {binding.feature_name!r}."
        )


class IndexedObjectMeasurementLabelPlaneBindingPolicy(
    DefaultObjectMeasurementLabelPlaneBindingPolicy
):
    """Align pre-indexed measurements to their label-plane domain."""

    value_type = IndexedObjectMeasurementLabelPlaneBinding

    def values(self, binding: ObjectMeasurementLabelPlaneBinding) -> Any:
        if not isinstance(binding, IndexedObjectMeasurementLabelPlaneBinding):
            raise TypeError(
                "IndexedObjectMeasurementLabelPlaneBindingPolicy requires "
                "IndexedObjectMeasurementLabelPlaneBinding."
            )
        values_by_label, positional_values = binding.indexed_values
        return self.values_for_index(binding, values_by_label, positional_values)


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


def iter_measurement_rows(
    measurement_tables: Iterable[MeasurementTable],
) -> Iterator[object]:
    """Yield row payloads from measurement tables without materializing them."""
    for table in measurement_tables:
        table_rows = table.rows
        if isinstance(table_rows, ColumnarRows):
            yield from _columnar_measurement_rows(table_rows)
            continue
        if isinstance(table_rows, list | tuple):
            yield from table_rows
            continue
        yield table_rows


def measurement_rows(
    measurement_tables: tuple[MeasurementTable, ...],
) -> tuple[object, ...]:
    """Flatten row payloads from measurement tables."""
    return tuple(iter_measurement_rows(measurement_tables))


def _measurement_table_may_declare_object_name(table: MeasurementTable) -> bool:
    """Return whether row-level fallback object-name scans can match."""
    if table.object_name is not None:
        return True
    if any(field.name == MEASUREMENT_OBJECT_NAME_FIELD for field in table.fields):
        return True

    rows = table.rows
    if isinstance(rows, ColumnarRows):
        return MEASUREMENT_OBJECT_NAME_FIELD in tuple(
            str(column) for column in rows.columns
        )
    if not isinstance(rows, list | tuple) or not rows:
        return False
    return any(
        MEASUREMENT_OBJECT_NAME_FIELD in measurement_row_mapping(row)
        for row in rows
    )


def _is_wide_row_sequence_measurement_table(table: MeasurementTable) -> bool:
    """Return whether row fields are direct measurement columns, not long rows."""
    rows = table.rows
    if not isinstance(rows, list | tuple) or not rows:
        return False
    return MeasurementRowSequenceLayout.from_rows(rows).is_wide_only


def _columnar_measurement_rows(rows: ColumnarRows) -> tuple[Mapping[str, object], ...]:
    """Return record mappings from a nominal columnar table payload."""
    return rows.row_mappings()


def columnar_row_values(rows: ColumnarRows, column: str) -> Sequence[object]:
    """Return one column from a nominal columnar payload."""
    return rows.column_values(column)


ProjectedMeasurementRows = Sequence[Mapping[str, Any]] | ColumnarRows


@dataclass(frozen=True, slots=True)
class MeasurementProjectedColumnarRows(ColumnarRows):
    """Columnar measurement rows with projected row-axis values."""

    columns: Mapping[str, Sequence[Any]]

    def __len__(self) -> int:
        return len(next(iter(self.columns.values()))) if self.columns else 0

    def __iter__(self):
        yield from self.row_mappings()

    def row_mappings(self) -> tuple[Mapping[str, object], ...]:
        return tuple(
            {
                field_name: value
                for field_name, value in row.items()
                if not _is_structural_missing_measurement_cell(value)
            }
            for row in ColumnarRows.row_mappings(self)
        )


@dataclass(frozen=True, slots=True)
class MeasurementSparseCell:
    """Structural missing-cell marker for sparse columnar row materialization."""


MEASUREMENT_SPARSE_CELL = MeasurementSparseCell()


@dataclass(frozen=True, slots=True)
class MeasurementSparseColumnarRows(ColumnarRows):
    """Columnar measurement rows whose missing cells are structural, not values."""

    columns: Mapping[str, Sequence[Any]]
    missing_cell: object = MEASUREMENT_SPARSE_CELL

    def __len__(self) -> int:
        return len(next(iter(self.columns.values()))) if self.columns else 0

    def __iter__(self):
        columns = self.columns
        for row_index in range(len(self)):
            yield {
                field_name: value
                for field_name, values in columns.items()
                for value in (values[row_index],)
                if not _is_structural_missing_measurement_cell(value)
            }

    def row_mappings(self) -> tuple[Mapping[str, object], ...]:
        return tuple(self)


@dataclass(frozen=True, slots=True)
class MeasurementSliceIndexImageNumberProjection:
    """Map runtime slice indices onto external image-number row values."""

    start: int
    image_numbers_by_slice: Mapping[int, int]

    def image_number_for_slice(self, slice_index: int) -> int:
        mapped = self.image_numbers_by_slice.get(slice_index)
        if mapped is not None:
            return mapped
        return slice_index + self.start


@dataclass(frozen=True, slots=True)
class MeasurementColumnarImageNumberProjection:
    """Projection for columnar rows that already declare image numbers."""

    columns: Mapping[str, Sequence[Any]]
    start: int

    def apply(self) -> ColumnarRows | None:
        image_number_field = MeasurementRowAxisField.IMAGE_NUMBER.value
        image_numbers = [
            int(value)
            for value in self.columns[image_number_field]
            if MeasurementScalarLiteral(value).is_present_axis_value
        ]
        if not image_numbers or min(image_numbers) >= self.start:
            return None
        offset = self.start - 1
        projected_columns = dict(self.columns)
        projected_columns[image_number_field] = tuple(
            int(value) + offset
            if MeasurementScalarLiteral(value).is_present_axis_value
            else value
            for value in self.columns[image_number_field]
        )
        return MeasurementProjectedColumnarRows(MappingProxyType(projected_columns))


@dataclass(frozen=True, slots=True)
class MeasurementColumnarSliceIndexProjection:
    """Projection for columnar rows that declare runtime slice_index only."""

    columns: Mapping[str, Sequence[Any]]
    image_numbers: MeasurementSliceIndexImageNumberProjection

    def apply(self) -> ColumnarRows:
        projected_columns = dict(self.columns)
        projected_columns[MeasurementRowAxisField.IMAGE_NUMBER.value] = (
            self.projected_image_numbers(
                self.columns[MeasurementRowAxisField.SLICE_INDEX.value]
            )
        )
        return MeasurementProjectedColumnarRows(MappingProxyType(projected_columns))

    def projected_image_numbers(self, slice_indices: Sequence[Any]) -> Sequence[Any]:
        """Return projected image numbers for one columnar slice-index vector."""
        values = np.asarray(slice_indices)
        if values.size == 0:
            return ()
        if np.issubdtype(values.dtype, np.integer):
            unique_values = np.unique(values)
            if unique_values.size == 1:
                return np.full(
                    values.shape,
                    self.image_numbers.image_number_for_slice(int(unique_values[0])),
                    dtype=np.int64,
                )
            mapping = {
                int(slice_index): self.image_numbers.image_number_for_slice(
                    int(slice_index)
                )
                for slice_index in unique_values
            }
            return np.asarray(
                [mapping[int(slice_index)] for slice_index in values],
                dtype=np.int64,
            )
        return tuple(
            self.image_numbers.image_number_for_slice(int(value))
            if MeasurementScalarLiteral(value).is_present_axis_value
            else value
            for value in slice_indices
        )


@dataclass(frozen=True, slots=True)
class MeasurementRowSequenceAxisProjection:
    """Projection for row-sequence measurements with runtime axis fields."""

    rows: Sequence[Any]
    row_mappings: tuple[Mapping[str, Any], ...]

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[Any],
        *,
        need_row_mappings: bool,
    ) -> "MeasurementRowSequenceAxisProjection":
        row_mappings: list[Mapping[str, Any]] = []
        has_axis_field = False
        for row in rows:
            row_mapping = measurement_row_mapping(row)
            if need_row_mappings:
                row_mappings.append(row_mapping)
            has_axis_field = has_axis_field or cls.row_has_axis(row_mapping)
            if has_axis_field and not need_row_mappings:
                row_mappings = [measurement_row_mapping(candidate) for candidate in rows]
                break
        return cls(rows=rows, row_mappings=tuple(row_mappings))

    @staticmethod
    def row_has_axis(row: Mapping[str, Any]) -> bool:
        return (
            MeasurementRowAxisField.IMAGE_NUMBER.value in row
            or MeasurementRowAxisField.SLICE_INDEX.value in row
        )

    @property
    def has_axis(self) -> bool:
        return any(self.row_has_axis(row) for row in self.row_mappings)

    @property
    def has_image_number(self) -> bool:
        return any(
            MeasurementRowAxisField.IMAGE_NUMBER.value in row
            for row in self.row_mappings
        )

    @property
    def has_slice_index(self) -> bool:
        return any(
            MeasurementRowAxisField.SLICE_INDEX.value in row
            for row in self.row_mappings
        )

    @property
    def has_source_qualified_image_rows(self) -> bool:
        return any(
            MEASUREMENT_SOURCE_IMAGE_NAME_FIELD in row
            and not measurement_row_has_object_identity(row)
            for row in self.row_mappings
        )

    def project_current_image_number(self, start: int) -> Sequence[Mapping[str, Any]]:
        image_number_field = MeasurementRowAxisField.IMAGE_NUMBER.value
        projected_rows = [dict(row) for row in self.row_mappings]
        for row in projected_rows:
            row.setdefault(image_number_field, start)
        return projected_rows

    def present_axis_values(self, field_name: str) -> tuple[int, ...]:
        """Return present integer axis values for one measurement row field."""
        return tuple(
            int(row[field_name])
            for row in self.row_mappings
            if MeasurementScalarLiteral(row.get(field_name)).is_present_axis_value
        )

    def project_axis_values(
        self,
        *,
        source_field_name: str,
        target_field_name: str,
        transform: Callable[[int], int],
    ) -> Sequence[Mapping[str, Any]]:
        """Return rows with present source-axis values projected into a target."""
        projected_rows = [dict(row) for row in self.row_mappings]
        for row in projected_rows:
            if MeasurementScalarLiteral(row.get(source_field_name)).is_present_axis_value:
                row[target_field_name] = transform(int(row[source_field_name]))
        return projected_rows

    def apply(
        self,
        image_numbers: MeasurementSliceIndexImageNumberProjection,
    ) -> Sequence[Any] | None:
        if not self.rows or not self.has_axis:
            return None
        if self.has_slice_index:
            return self.project_slice_index(image_numbers)
        if self.has_image_number:
            return self.project_image_number(image_numbers.start)
        return None

    def project_slice_index(
        self,
        image_numbers: MeasurementSliceIndexImageNumberProjection,
    ) -> Sequence[Mapping[str, Any]]:
        slice_index_field = MeasurementRowAxisField.SLICE_INDEX.value
        image_number_field = MeasurementRowAxisField.IMAGE_NUMBER.value
        return self.project_axis_values(
            source_field_name=slice_index_field,
            target_field_name=image_number_field,
            transform=image_numbers.image_number_for_slice,
        )

    def project_image_number(self, start: int) -> Sequence[Mapping[str, Any]] | None:
        image_number_field = MeasurementRowAxisField.IMAGE_NUMBER.value
        image_numbers = self.present_axis_values(image_number_field)
        if not image_numbers or min(image_numbers) >= start:
            return None

        offset = start - 1
        return self.project_axis_values(
            source_field_name=image_number_field,
            target_field_name=image_number_field,
            transform=lambda value: value + offset,
        )


def normalize_measurement_token(value: object) -> str:
    """Normalize feature/source names for runtime measurement lookup."""
    return normalize_runtime_identifier(value)


def measurement_feature_candidates(
    feature_name: str,
    *,
    dialect: RuntimeMeasurementLookupDialectLike = (
        CURRENT_RUNTIME_MEASUREMENT_LOOKUP_DIALECT
    ),
) -> frozenset[str]:
    """Return normalized feature aliases accepted for row/field lookup."""
    return frozenset(
        ordered_measurement_feature_candidates(feature_name, dialect=dialect)
    )


def ordered_measurement_feature_candidates(
    feature_name: str,
    *,
    dialect: RuntimeMeasurementLookupDialectLike = (
        CURRENT_RUNTIME_MEASUREMENT_LOOKUP_DIALECT
    ),
) -> tuple[str, ...]:
    """Return schema-safe feature-field aliases from most specific to least specific."""
    return (
        resolve_runtime_measurement_lookup_dialect(dialect)
        .feature_lookup(feature_name)
        .field_aliases
    )


def specific_measurement_feature_candidates(
    feature_name: str,
    *,
    dialect: RuntimeMeasurementLookupDialectLike = (
        CURRENT_RUNTIME_MEASUREMENT_LOOKUP_DIALECT
    ),
) -> frozenset[str]:
    """Return non-lossy aliases suitable for schema discovery."""
    return frozenset(
        ordered_measurement_feature_candidates(feature_name, dialect=dialect)
    )


def ordered_measurement_source_candidates(
    feature_name: str,
    *,
    dialect: RuntimeMeasurementLookupDialectLike = (
        CURRENT_RUNTIME_MEASUREMENT_LOOKUP_DIALECT
    ),
) -> tuple[str, ...]:
    """Return source-image aliases encoded by a source-qualified feature name."""
    return (
        resolve_runtime_measurement_lookup_dialect(dialect)
        .feature_lookup(feature_name)
        .source_aliases
    )


def matching_measurement_field(
    row: Mapping[str, object],
    candidates: Sequence[str],
) -> str | None:
    """Return the first row field matching the ordered feature alias set."""
    fields = matching_measurement_fields(row, candidates)
    return fields[0] if fields else None


def matching_measurement_fields(
    row: Mapping[str, object],
    candidates: Sequence[str],
) -> tuple[str, ...]:
    """Return row fields matching ordered feature aliases."""
    fields: list[str] = []
    for candidate in candidates:
        for field_name in row:
            normalized = normalize_measurement_token(field_name)
            if candidate == normalized and field_name not in fields:
                fields.append(field_name)
    return tuple(fields)


def measurement_row_feature_matches(
    row: Mapping[str, object],
    candidates: Sequence[str],
) -> bool:
    """Return whether the row explicitly names one matching feature."""
    for field_name in MEASUREMENT_FEATURE_NAME_FIELDS:
        value = row.get(field_name)
        if value is None:
            continue
        if normalize_measurement_token(value) in candidates:
            return True
    return False


def measurement_row_first_value(row: Mapping[str, object]) -> object | None:
    """Return the first recognized scalar value field on a measurement row."""
    for value_field in MEASUREMENT_VALUE_FIELDS:
        if value_field in row:
            return row[value_field]
    return None


def measurement_object_label(
    row: Mapping[str, object],
    *,
    object_id_field: str | None = None,
) -> int | None:
    """Return the object id encoded on a measurement row."""
    if object_id_field is not None and object_id_field in row:
        return MeasurementObjectLabelResolution(row[object_id_field]).object_label
    for key in MEASUREMENT_OBJECT_ID_FIELDS:
        if key in row:
            return MeasurementObjectLabelResolution(row[key]).object_label
    return None


def measurement_row_has_object_identity(
    row: Mapping[str, object],
    *,
    object_id_field: str | None = None,
) -> bool:
    """Return whether a measurement row carries object identity."""
    return (
        measurement_object_label(row, object_id_field=object_id_field)
        is not None
    )


@dataclass(frozen=True, slots=True)
class MeasurementObjectLabelResolution:
    """Integer object label resolved from runtime/CSV scalar encodings."""

    value: object

    @property
    def object_label(self) -> int | None:
        return MeasurementScalarLiteral(self.value).integer_value


def measurement_table_object_id_field(table: MeasurementTable) -> str | None:
    """Return the authoritative object-id field declared by a measurement table."""
    if table.object_id_field is not None:
        return table.object_id_field
    if table.subject and table.subject.scope is MeasurementScope.OBJECT:
        return table.subject.id_field
    return None


def measurement_table_object_name(table: MeasurementTable) -> str | None:
    """Return the authoritative object name for object-scoped measurement tables."""
    if table.object_name is not None:
        return table.object_name
    if table.subject and table.subject.scope is MeasurementScope.OBJECT:
        return table.subject.name
    return None


def measurement_row_source_matches_feature(
    row: Mapping[str, object],
    query: MeasurementFeatureQuery,
) -> bool:
    """Return whether the row source qualifier is compatible with a feature."""
    source_image_name = measurement_row_source_image_name(row)
    if source_image_name is None:
        return True
    normalized_source = normalize_measurement_token(source_image_name)
    if normalized_source in MEASUREMENT_UNQUALIFIED_SOURCE_NAMES:
        return True
    return normalized_source in query.source_candidates


def measurement_values_for_feature(
    measurement_tables: tuple[MeasurementTable, ...],
    feature_name: str,
    *,
    object_count: int,
    object_ids: Sequence[int] | None = None,
    object_name: str | None = None,
    dialect: RuntimeMeasurementLookupDialectLike = (
        CURRENT_RUNTIME_MEASUREMENT_LOOKUP_DIALECT
    ),
) -> Any:
    """Return object-indexed measurement values for one feature."""
    values_by_label, positional_values = MeasurementFeatureQuery(
        feature_name,
        object_name=object_name,
        dialect=dialect,
    ).value_index(measurement_tables)
    resolved_object_ids = (
        tuple(range(1, object_count + 1))
        if object_ids is None
        else tuple(int(object_id) for object_id in object_ids)
    )
    if values_by_label:
        return ObjectLabelMeasurementValues.from_value_mapping(
            resolved_object_ids,
            values_by_label,
        ).values
    if positional_values:
        return ObjectLabelMeasurementValues.from_positional_values(
            resolved_object_ids,
            positional_values,
        ).values
    raise ValueError(f"Could not resolve measurement feature {feature_name!r}.")


def measurement_value_indexes_for_object_feature_batch(
    measurement_tables_by_object: MeasurementTablesByObject,
    feature_name: str,
    *,
    object_names: Sequence[str],
    dialect: RuntimeMeasurementLookupDialectLike = (
        CURRENT_RUNTIME_MEASUREMENT_LOOKUP_DIALECT
    ),
) -> MeasurementValueIndexesByObject:
    """Return object-keyed feature indexes for one feature across object domains."""
    return MeasurementObjectFeatureVectorBatchQuery(
        feature_name,
        tuple(object_names),
        dialect=dialect,
    ).value_indexes(measurement_tables_by_object)


@dataclass(frozen=True, slots=True)
class MeasurementAxisValueProjection:
    """Projection rule for rows that may or may not declare one runtime axis."""

    axis: MeasurementRowAxisField
    value: int

    @property
    def field_name(self) -> str:
        return self.axis.value

    def matches_value(self, value: object) -> bool:
        """Return whether a row/column value survives this axis projection."""
        if not self.value_is_present(value):
            return True
        return int(value) == int(self.value)

    def mask(self, values: Sequence[object]) -> Any:
        """Return a boolean mask that keeps direct axis matches."""
        import numpy as np

        return np.asarray(
            [self.matches_value(value) for value in values],
            dtype=bool,
        )

    def columnar_mask(self, values: Sequence[object]) -> Any:
        """Return a columnar mask, allowing singleton-axis local projections."""
        import numpy as np

        concrete_values = tuple(value for value in values if self.value_is_present(value))
        direct_mask = self.mask(values)
        if bool(np.any(direct_mask)):
            return direct_mask
        concrete_domain = frozenset(int(value) for value in concrete_values)
        if len(concrete_domain) == 1:
            return np.ones(len(values), dtype=bool)
        return direct_mask

    @staticmethod
    def value_is_present(value: object) -> bool:
        """Return whether an axis value declares a concrete row domain."""
        return MeasurementScalarLiteral(value).is_present_axis_value


@dataclass(frozen=True, slots=True)
class MeasurementTableAxisProjection:
    """Projection of one measurement table onto a declared row-axis value."""

    table: MeasurementTable
    axis: MeasurementRowAxisField
    value: int

    @property
    def projection(self) -> MeasurementAxisValueProjection:
        return MeasurementAxisValueProjection(self.axis, self.value)

    @property
    def field_name(self) -> str:
        return self.projection.field_name

    def apply(self) -> MeasurementTable:
        """Return the projected table, preserving schema only when still valid."""
        if isinstance(self.table.rows, ColumnarRows):
            return self._columnar_table()

        rows = measurement_rows((self.table,))
        if not rows:
            return self.table
        if not any(self.field_name in measurement_row_mapping(row) for row in rows):
            return self.table

        row_mappings = tuple(measurement_row_mapping(row) for row in rows)
        projection_mask = self.projection.mask(
            tuple(row.get(self.field_name) for row in row_mappings)
        )
        return self._row_sequence_table(
            [row for row, keep in zip(rows, projection_mask, strict=True) if keep]
        )

    def _columnar_table(self) -> MeasurementTable:
        column_names = tuple(str(column) for column in self.table.rows.columns)
        if self.field_name not in column_names:
            return self.table
        axis_values = columnar_row_values(self.table.rows, self.field_name)
        axis_mask = self.projection.columnar_mask(axis_values)
        if bool(np.all(axis_mask)):
            return self.table
        rows = AxisFilteredMeasurementColumnarRows(
            self.table.rows,
            self.projection,
            axis_mask=axis_mask,
        )
        return self._with_rows(rows, self.table.fields)

    def _row_sequence_table(self, rows: Sequence[object]) -> MeasurementTable:
        declared_layout = measurement_table_row_layout_from_fields(self.table.fields)
        if declared_layout is not None:
            return self._with_rows(
                rows,
                self.table.fields,
                validated_runtime_schema=True,
            )
        normalized_rows = normalize_measurement_table_rows(rows, fields=())
        return self._with_rows(
            normalized_rows,
            self._compatible_fields(normalized_rows),
            validated_runtime_schema=False,
        )

    def _compatible_fields(self, rows: object) -> tuple[FieldSpec, ...]:
        declared_layout = measurement_table_row_layout_from_fields(self.table.fields)
        observed_layout = measurement_table_row_layout(rows)
        if declared_layout is None:
            return ()
        if observed_layout not in (declared_layout, MeasurementTableRowLayout.EMPTY):
            return ()
        return tuple(self.table.fields)

    def _with_rows(
        self,
        rows: object,
        fields: Iterable[FieldSpec],
        *,
        validated_runtime_schema: bool = False,
    ) -> MeasurementTable:
        return MeasurementTable(
            name=self.table.name,
            rows=rows,
            object_name=self.table.object_name,
            fields=tuple(fields),
            object_id_field=self.table.object_id_field,
            source_image_name=self.table.source_image_name,
            subject=self.table.subject,
            validated_runtime_schema=validated_runtime_schema,
        )


@dataclass(frozen=True, slots=True)
class MeasurementTableAxisQuery:
    """Reusable query for projecting measurement tables by one row axis."""

    axis: MeasurementRowAxisField
    value: int

    def tables(
        self,
        measurement_tables: tuple[MeasurementTable, ...],
    ) -> tuple[MeasurementTable, ...]:
        """Return measurement tables narrowed to this row-axis value."""
        return tuple(
            MeasurementTableAxisProjection(table, self.axis, self.value).apply()
            for table in measurement_tables
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
        return {
            int(axis_value)
            for axis_value in columnar_row_values(table.rows, axis_field)
            if axis_value is not None
        }
    return {
        int(row_mapping[axis_field])
        for row in measurement_rows((table,))
        for row_mapping in (measurement_row_mapping(row),)
        if row_mapping.get(axis_field) is not None
    }


def measurement_values_for_label_slices(
    measurement_tables: tuple[MeasurementTable, ...],
    feature_name: str,
    labels: object,
    *,
    object_name: str | None = None,
    row_axis: MeasurementRowAxisField = MeasurementRowAxisField.SLICE_INDEX,
    row_axis_start: int | None = None,
    dialect: RuntimeMeasurementLookupDialectLike = (
        CURRENT_RUNTIME_MEASUREMENT_LOOKUP_DIALECT
    ),
) -> tuple[Any, ...]:
    """Return measurement values aligned to positive label IDs in each label plane."""
    return MeasurementLabelSliceFeatureQuery(
        measurement_tables=measurement_tables,
        feature_query=MeasurementFeatureQuery(
            feature_name,
            object_name=object_name,
            dialect=dialect,
        ),
        row_axis=row_axis,
    ).values_for_labels(labels, row_axis_start=row_axis_start)


@dataclass(frozen=True, slots=True)
class MeasurementLabelSliceTableQuery(ABC):
    """Shared table/feature authority for label-slice measurement lookup."""

    measurement_tables: tuple[MeasurementTable, ...]
    feature_query: MeasurementFeatureQuery


@dataclass(frozen=True, slots=True)
class MeasurementLabelSliceAxisSelection:
    """Authoritative row-axis binding for label-stack measurement lookup."""

    row_axis: MeasurementRowAxisField
    row_axis_start: int | None = None


@dataclass(frozen=True, slots=True)
class MeasurementLabelSliceAxisResolver(MeasurementLabelSliceTableQuery):
    """Resolve the table row axis that indexes a label stack."""

    preferred_axis: MeasurementRowAxisField
    label_slice_count: int

    def select(self) -> MeasurementLabelSliceAxisSelection:
        """Return the declared table axis and slice-to-row start offset."""
        if self.label_slice_count <= 1:
            return MeasurementLabelSliceAxisSelection(self.preferred_axis)
        for axis in self.candidate_axes():
            axis_values = self.axis_values(axis)
            if not axis_values:
                continue
            return MeasurementLabelSliceAxisSelection(
                row_axis=axis,
                row_axis_start=self.axis_start(axis_values),
            )
        return MeasurementLabelSliceAxisSelection(self.preferred_axis)

    def candidate_axes(self) -> tuple[MeasurementRowAxisField, ...]:
        """Return row axes in preferred order without duplicating candidates."""
        return tuple(
            dict.fromkeys(
                (
                    self.preferred_axis,
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
            if self.feature_query.table_may_carry_feature(table)
            for value in measurement_table_axis_values(table, axis)
        }
        return tuple(sorted(values))

    def axis_start(self, axis_values: tuple[int, ...]) -> int | None:
        """Return the slice-index origin for a concrete table axis."""
        if not axis_values:
            return None
        first_value = axis_values[0]
        return None if first_value == 0 else first_value


@dataclass(frozen=True, slots=True)
class MeasurementLabelSliceFeatureQuery(MeasurementLabelSliceTableQuery):
    """Query one measurement feature against a stack of object-label planes."""

    row_axis: MeasurementRowAxisField = MeasurementRowAxisField.SLICE_INDEX

    @property
    def feature_name(self) -> str:
        return self.feature_query.feature_name

    @property
    def object_name(self) -> str | None:
        return self.feature_query.object_name

    @property
    def dialect(self) -> RuntimeMeasurementLookupDialectLike:
        return self.feature_query.dialect

    def values_for_labels(
        self,
        labels: object,
        *,
        row_axis_start: int | None = None,
    ) -> tuple[Any, ...]:
        """Return measurement values aligned to each label plane."""
        import numpy as np

        label_array = np.asarray(labels)
        label_planes = (
            (label_array,)
            if label_array.ndim <= 2
            else tuple(label_array[index] for index in range(label_array.shape[0]))
        )
        axis_selection = MeasurementLabelSliceAxisResolver(
            measurement_tables=self.measurement_tables,
            feature_query=self.feature_query,
            preferred_axis=self.row_axis,
            label_slice_count=len(label_planes),
        ).select()
        if axis_selection.row_axis is not self.row_axis:
            return type(self)(
                measurement_tables=self.measurement_tables,
                feature_query=self.feature_query,
                row_axis=axis_selection.row_axis,
            ).values_for_labels(
                labels,
                row_axis_start=(
                    row_axis_start
                    if row_axis_start is not None
                    else axis_selection.row_axis_start
                ),
            )
        if row_axis_start is None:
            row_axis_start = axis_selection.row_axis_start
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
        if label_array.ndim > 2:
            values_by_slice = self.value_indexes_by_axis()
            if values_by_slice is not None:
                return tuple(
                    IndexedObjectMeasurementLabelPlaneBinding(
                        measurement_tables=self.measurement_tables,
                        object_name=self.object_name or "",
                        feature_name=self.feature_name,
                        labels=label_plane,
                        dialect=self.dialect,
                        indexed_values=self.value_index_for_label_slice(
                            values_by_slice,
                            slice_index,
                            row_axis_start=row_axis_start,
                            label_slice_count=len(label_planes),
                        ),
                    ).values()
                    for slice_index, label_plane in enumerate(label_planes)
                )
        return tuple(
            IndexedObjectMeasurementLabelPlaneBinding(
                measurement_tables=slice_feature_tables,
                object_name=self.object_name or "",
                feature_name=self.feature_name,
                labels=label_plane,
                dialect=self.dialect,
                indexed_values=self.feature_query.value_index(slice_feature_tables),
            ).values()
            for slice_index, label_plane, slice_feature_tables in (
                (
                    index,
                    plane,
                    self.feature_tables_for_axis(
                        index,
                        row_axis_start=row_axis_start,
                    ),
                )
                for index, plane in enumerate(label_planes)
            )
        )

    def feature_tables_for_axis(
        self,
        slice_index: int,
        *,
        row_axis_start: int | None = None,
    ) -> tuple[MeasurementTable, ...]:
        """Return axis-projected tables, preserving axisless feature tables."""
        projected_tables = self.axis_projected_tables(
            slice_index,
            row_axis_start=row_axis_start,
        )
        if projected_tables:
            return projected_tables
        axis_values = set()
        for table in self.measurement_tables:
            axis_values.update(measurement_table_axis_values(table, self.row_axis))
        return self.measurement_tables if not axis_values else projected_tables

    def axis_projected_tables(
        self,
        slice_index: int,
        *,
        row_axis_start: int | None = None,
    ) -> tuple[MeasurementTable, ...]:
        """Return measurement tables projected to one label slice row-axis value."""
        row_axis_value = (
            slice_index
            if row_axis_start is None
            else row_axis_start + slice_index
        )
        return MeasurementTableAxisQuery(
            self.row_axis,
            row_axis_value,
        ).tables(self.measurement_tables)

    @staticmethod
    def value_index_for_label_slice(
        values_by_slice: Mapping[int, MeasurementValueIndexResult],
        slice_index: int,
        *,
        row_axis_start: int | None = None,
        label_slice_count: int | None = None,
    ) -> MeasurementValueIndexResult:
        """Return the measurement index for a label plane, broadcasting singletons."""
        if row_axis_start is not None:
            row_axis_value = row_axis_start + slice_index
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

    def value_indexes_by_axis(self) -> dict[int, MeasurementValueIndexResult] | None:
        """Return per-axis feature indexes without re-scanning tables per plane."""
        defaults: MeasurementValueIndexResult = ({}, [])
        by_slice: dict[int, MeasurementValueIndexResult] = {}
        query_object_name = self.feature_query.query_object_name

        for table in self.measurement_tables:
            if query_object_name is not None:
                table_object = measurement_table_object_name(table)
                if table_object not in (None, query_object_name):
                    continue

            slice_indices = measurement_table_axis_values(table, self.row_axis)
            if not slice_indices:
                table_index = MeasurementFeatureValueIndex.from_table(
                    table,
                    self.feature_query,
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
                        table,
                        self.row_axis,
                        slice_index,
                    ).apply(),
                    self.feature_query,
                )
                if not table_index.present:
                    continue
                _merge_measurement_value_index(
                    by_slice.setdefault(slice_index, ({}, [])),
                    table_index.as_query_result(),
                )

        if defaults[0] or defaults[1]:
            for slice_index in tuple(by_slice):
                values_by_label = dict(defaults[0])
                values_by_label.update(by_slice[slice_index][0])
                positional_values = [*defaults[1], *by_slice[slice_index][1]]
                by_slice[slice_index] = (values_by_label, positional_values)
        if defaults[0] or defaults[1]:
            by_slice.setdefault(-1, defaults)
        return by_slice if by_slice else None


def _merge_measurement_value_index(
    target: MeasurementValueIndexResult,
    source: MeasurementValueIndexResult,
) -> None:
    target[0].update(source[0])
    target[1].extend(source[1])


def measurement_row_object_name(row: Mapping[str, object]) -> str | None:
    """Return the object-set owner encoded on one measurement row."""
    value = row.get(MEASUREMENT_OBJECT_NAME_FIELD)
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def measurement_row_source_image_name(row: Mapping[str, object]) -> str | None:
    """Return the source-image owner encoded on one measurement row."""
    value = row.get(MEASUREMENT_SOURCE_IMAGE_NAME_FIELD)
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def annotate_measurement_row_object(
    row: object,
    object_name: str,
) -> Mapping[str, object]:
    """Return a measurement row with explicit object-set ownership."""
    normalized_object_name = object_name.strip()
    if not normalized_object_name:
        raise ValueError("object_name cannot be empty.")
    return {
        **dict(measurement_row_mapping(row)),
        MEASUREMENT_OBJECT_NAME_FIELD: normalized_object_name,
    }


def annotate_measurement_row_source_image(
    row: object,
    source_image_name: str,
) -> Mapping[str, object]:
    """Return a measurement row with explicit source-image ownership."""
    normalized_source_image_name = source_image_name.strip()
    if not normalized_source_image_name:
        raise ValueError("source_image_name cannot be empty.")
    return {
        **dict(measurement_row_mapping(row)),
        MEASUREMENT_SOURCE_IMAGE_NAME_FIELD: normalized_source_image_name,
    }


@dataclass(frozen=True, slots=True)
class MeasurementRowQualifier:
    """One typed ownership qualifier attached to a measurement row."""

    field_name: str
    value: str

    @classmethod
    def optional(
        cls,
        *,
        field_name: str,
        value: str | None,
    ) -> "MeasurementRowQualifier | None":
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{field_name} cannot be empty.")
        return cls(field_name=field_name, value=normalized)

    def apply(self, row: MutableMapping[str, object]) -> None:
        row[self.field_name] = self.value


@dataclass(frozen=True, slots=True)
class MeasurementRowOwnership:
    """Shared object/source ownership qualifiers for measurement rows."""

    object_name: str | None = None
    source_image_name: str | None = None

    @property
    def qualifiers(self) -> tuple[MeasurementRowQualifier, ...]:
        return tuple(
            qualifier
            for qualifier in (
                MeasurementRowQualifier.optional(
                    field_name=MEASUREMENT_OBJECT_NAME_FIELD,
                    value=self.object_name,
                ),
                MeasurementRowQualifier.optional(
                    field_name=MEASUREMENT_SOURCE_IMAGE_NAME_FIELD,
                    value=self.source_image_name,
                ),
            )
            if qualifier is not None
        )

    def annotate_rows(self, rows: Sequence[object] | ColumnarRows) -> Sequence[object] | ColumnarRows:
        """Attach ownership qualifiers, copying only non-mutable row values."""
        qualifiers = self.qualifiers
        if not qualifiers:
            return rows
        if isinstance(rows, ColumnarRows):
            return QualifiedMeasurementColumnarRows(rows, qualifiers)
        if (
            rows
            and is_dataclass(type(rows[0]))
            and all(type(row) is type(rows[0]) for row in rows)
        ):
            return QualifiedMeasurementColumnarRows(
                DataclassMeasurementColumnarRows(rows),
                qualifiers,
            )
        return [self.annotate_row(row, qualifiers=qualifiers) for row in rows]

    def annotate_row(
        self,
        row: object,
        *,
        qualifiers: Sequence[MeasurementRowQualifier],
    ) -> Mapping[str, object]:
        annotated_row: MutableMapping[str, object] = (
            row
            if isinstance(row, MutableMapping)
            else dict(measurement_row_mapping(row))
        )
        for qualifier in qualifiers:
            qualifier.apply(annotated_row)
        return annotated_row


@dataclass(frozen=True, slots=True)
class MeasurementColumnarRowsView(ColumnarRows, ABC):
    """Base for columnar measurement views that derive columns from another table."""

    rows: ColumnarRows
    _columns: Mapping[str, Sequence[object]] = field(
        init=False,
        repr=False,
        compare=False,
    )

    columns: ClassVar[AliasProperty[Mapping[str, Sequence[object]]]] = (
        AliasProperty("_columns")
    )

    def __len__(self) -> int:
        return len(next(iter(self._columns.values()))) if self._columns else 0


def columnar_row_count(rows: ColumnarRows) -> int:
    """Return row count for a nominal columnar payload."""
    return rows.row_count()


@dataclass(frozen=True, slots=True)
class ConcatenatedColumnarRowColumns(Mapping[str, Sequence[object]]):
    """Lazy mapping over columns concatenated from multiple columnar batches."""

    row_batches: tuple[ColumnarRows, ...]
    column_names: tuple[str, ...]

    @classmethod
    def from_row_batches(
        cls,
        row_batches: tuple[ColumnarRows, ...],
    ) -> "ConcatenatedColumnarRowColumns":
        return cls(
            row_batches=row_batches,
            column_names=tuple(
                dict.fromkeys(
                    str(column)
                    for row_batch in row_batches
                    for column in row_batch.columns
                )
            ),
        )

    def __getitem__(self, column_name: str) -> Sequence[object]:
        if column_name not in self.column_names:
            raise KeyError(column_name)
        return np.concatenate(
            tuple(
                self._batch_column_values(row_batch, column_name)
                for row_batch in self.row_batches
            )
        )

    def _batch_column_values(
        self,
        row_batch: ColumnarRows,
        column_name: str,
    ) -> Sequence[object]:
        batch_columns = {str(column): column for column in row_batch.columns}
        if column_name in batch_columns:
            return columnar_row_values(row_batch, batch_columns[column_name])
        return (None,) * columnar_row_count(row_batch)

    def __iter__(self):
        return iter(self.column_names)

    def __len__(self) -> int:
        return len(self.column_names)


@dataclass(frozen=True, slots=True)
class ConcatenatedColumnarRows(ColumnarRows):
    """Columnar table view over multiple columnar row batches."""

    row_batches: tuple[ColumnarRows, ...]
    _columns: Mapping[str, Sequence[object]] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_columns",
            ConcatenatedColumnarRowColumns.from_row_batches(self.row_batches),
        )

    columns: ClassVar[AliasProperty[Mapping[str, Sequence[object]]]] = (
        AliasProperty("_columns")
    )

    def __len__(self) -> int:
        return sum(columnar_row_count(row_batch) for row_batch in self.row_batches)

    def row_mappings(self) -> tuple[Mapping[str, object], ...]:
        return tuple(
            row
            for row_batch in self.row_batches
            for row in row_batch.row_mappings()
        )


@dataclass(frozen=True, slots=True)
class DataclassMeasurementColumnarRows(ColumnarRows):
    """Columnar view over homogeneous dataclass measurement rows."""

    rows: Sequence[object]
    _columns: Mapping[str, Sequence[object]] = field(
        init=False,
        repr=False,
        compare=False,
    )
    columns: ClassVar[AliasProperty[Mapping[str, Sequence[object]]]] = (
        AliasProperty("_columns")
    )

    def __post_init__(self) -> None:
        if not self.rows:
            object.__setattr__(self, "_columns", {})
            return
        row_type = type(self.rows[0])
        if not is_dataclass(row_type):
            raise TypeError(
                "DataclassMeasurementColumnarRows requires dataclass rows, "
                f"got {row_type.__name__}."
            )
        if not all(type(row) is row_type for row in self.rows):
            raise TypeError(
                "DataclassMeasurementColumnarRows requires homogeneous row types."
            )
        object.__setattr__(
            self,
            "_columns",
            {
                field_spec.name: tuple(
                    getattr(row, field_spec.name) for row in self.rows
                )
                for field_spec in dataclass_fields(row_type)
            },
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        columns = self._columns
        for row_index in range(len(self)):
            yield {
                field_name: values[row_index]
                for field_name, values in columns.items()
            }


@dataclass(frozen=True, slots=True)
class QualifiedMeasurementColumnarRows(MeasurementColumnarRowsView):
    """Columnar measurement rows with table-ownership qualifiers attached."""

    qualifiers: tuple[MeasurementRowQualifier, ...]

    def __post_init__(self) -> None:
        columns = dict(self.rows.columns)
        row_count = len(next(iter(columns.values()))) if columns else 0
        for qualifier in self.qualifiers:
            columns[qualifier.field_name] = (qualifier.value,) * row_count
        object.__setattr__(self, "_columns", columns)

    def __iter__(self):
        columns = self._columns
        for row_index in range(len(self)):
            yield {
                field_name: values[row_index]
                for field_name, values in columns.items()
            }


@dataclass(frozen=True, slots=True)
class AxisFilteredMeasurementColumnarRows(MeasurementColumnarRowsView):
    """Columnar measurement rows filtered to one runtime/CellProfiler axis value."""

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
                object.__setattr__(self, "_columns", columns)
                return
            axis_mask = self.projection.columnar_mask(axis_values)
        else:
            axis_mask = self.axis_mask
        if bool(np.all(axis_mask)):
            object.__setattr__(self, "_columns", columns)
            return
        object.__setattr__(
            self,
            "_columns",
            {
                column_name: column_values[axis_mask]
                for column_name, column_values in columns.items()
            },
        )


def _label_planes_are_empty(label_planes: tuple[Any, ...]) -> bool:
    import numpy as np

    return all(not np.any(label_plane > 0) for label_plane in label_planes)
