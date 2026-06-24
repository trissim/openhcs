"""Measurement feature query, schema, and value-index semantics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar

from metaclass_registry import AutoRegisterMeta
import numpy as np

from openhcs.core.measurement_lookup_dialect import (
    CURRENT_RUNTIME_MEASUREMENT_LOOKUP_DIALECT,
    RuntimeMeasurementFeatureLookup,
    RuntimeMeasurementLookupDialectLike,
    resolve_runtime_measurement_lookup_dialect,
)
from openhcs.core.measurement_row_materialization import (
    MEASUREMENT_FEATURE_NAME_FIELD,
    MEASUREMENT_FEATURE_NAME_FIELDS,
    MEASUREMENT_MEAN_VALUE_FIELD,
    MEASUREMENT_MEASUREMENT_NAME_FIELD,
    MEASUREMENT_MEASUREMENT_VALUE_FIELD,
    MEASUREMENT_OBJECT_ID_FIELDS,
    MEASUREMENT_OBJECT_NAME_FIELD,
    MEASUREMENT_OUTPUT_NAME_FIELD,
    MEASUREMENT_RESULT_VALUE_FIELD,
    MEASUREMENT_SOURCE_IMAGE_NAME_FIELD,
    MEASUREMENT_VALUE_FIELD,
    MEASUREMENT_VALUE_FIELDS,
    MeasurementObjectLabelResolution,
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
from openhcs.core.process_local_cache import (
    IdentityBoundProcessCache,
    ProcessLocalBoundedCache,
    identity_owner_tuples_match,
)
from openhcs.core.registry_strategies import NominalTypeKeyedStrategyMixin
from openhcs.core.runtime_identifier import normalize_runtime_identifier
from openhcs.core.runtime_semantics import (
    FieldSpec,
    MeasurementRowAxisField,
    MeasurementScalarLiteral,
    MeasurementScope,
    ObjectLabelMeasurementValues,
    dense_object_label_id_domain,
    measurement_axis_integer_domain,
    measurement_axis_integer_value,
    measurement_row_mapping,
)
from openhcs.core.runtime_values import ColumnarRows, MeasurementTable
from openhcs.core.source_image_provenance import SourceImageProvenance


MEASUREMENT_UNQUALIFIED_SOURCE_NAMES = frozenset(("", MeasurementScope.IMAGE.value))
MeasurementValueIndexResult = tuple[dict[int, float], list[float]]
OptionalMeasurementValueIndexResult = MeasurementValueIndexResult | None
MeasurementTablesByObject = Mapping[str, tuple[MeasurementTable, ...]]
MeasurementValueIndexesByObject = Mapping[str, MeasurementValueIndexResult]
MeasurementFeatureValueIndexesByObject = dict[str | None, "MeasurementFeatureValueIndex"]
MeasurementObjectFeatureVectorBatchCacheValue = tuple[
    tuple[MeasurementTable, ...],
    MeasurementValueIndexesByObject,
]
MeasurementObjectFeatureAxisBatchCacheValue = tuple[
    tuple[MeasurementTable, ...],
    Mapping[int, MeasurementValueIndexesByObject],
]
_DIAGNOSTIC_NONE = "<none>"


def _diagnostic_value(value: object | None) -> str:
    if value is None or value == "":
        return _DIAGNOSTIC_NONE
    return str(value)


def _first_measurement_row_mapping(
    rows: tuple[object, ...],
) -> Mapping[str, object]:
    if not rows:
        return MappingProxyType({})
    return measurement_row_mapping(rows[0])


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
        """Return a boolean mask that keeps direct axis matches and axisless rows."""
        return np.asarray(
            [self.matches_value(value) for value in values],
            dtype=bool,
        )

    def columnar_mask(self, values: Sequence[object]) -> Any:
        """Return a columnar mask, allowing singleton-axis local projections."""
        concrete_values = tuple(value for value in values if self.value_is_present(value))
        direct_mask = self.mask(values)
        if bool(np.any(direct_mask)):
            return direct_mask
        concrete_domain = frozenset(int(value) for value in concrete_values)
        if len(concrete_domain) == 1:
            return np.ones(len(values), dtype=bool)
        return direct_mask

    def table_source_provenance(
        self,
        table: MeasurementTable,
    ) -> SourceImageProvenance:
        """Return source provenance after applying this table-axis projection."""
        if self.axis is MeasurementRowAxisField.SLICE_INDEX:
            return table.source_provenance.for_source_plane(self.value)
        return table.source_provenance

    def value_is_present(self, value: object) -> bool:
        """Return whether an axis value declares a concrete row domain."""
        return measurement_axis_integer_value(value, self.axis) is not None


@dataclass(frozen=True, slots=True)
class MeasurementObjectFeatureVectorBatchCacheKey:
    """Identity key for a batch object-feature value-index lookup."""

    feature_name: str
    dialect_identity: int
    table_identities: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class MeasurementObjectFeatureAxisBatchCacheKey:
    """Identity key for a batch object-feature row-axis value-index lookup."""

    feature_name: str
    dialect_identity: int
    row_axis: MeasurementRowAxisField
    table_identities: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ColumnarMeasurementTableSchema:
    """Cached semantic projection for nominal columnar measurement rows."""

    columns: tuple[str, ...]
    normalized_columns: dict[str, str]
    object_names: tuple[str, ...]
    feature_names: frozenset[str]
    feature_name_values: Sequence[object] | None
    object_name_values: Sequence[object] | None
    source_image_name_values: Sequence[object] | None
    feature_masks_by_candidates: dict[tuple[str, ...], Any]
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
        feature_name_values = cls._feature_name_values(rows, columns)
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
                ) if feature_name_values is None else frozenset(
                    str(value)
                    for value in feature_name_values
                    if value not in (None, "")
                ),
                feature_name_values=feature_name_values,
                object_name_values=object_name_values,
                source_image_name_values=source_image_name_values,
                feature_masks_by_candidates={},
                object_masks_by_name={},
                source_masks_by_candidates={},
            ),
        )

    @staticmethod
    def _feature_name_values(
        rows: ColumnarRows,
        columns: tuple[str, ...],
    ) -> Sequence[object] | None:
        for field_name in MEASUREMENT_FEATURE_NAME_FIELDS:
            if field_name in columns:
                return columnar_row_values(rows, field_name)
        return None

    def feature_mask(self, candidates: tuple[str, ...]) -> Any | None:
        """Return a boolean row mask for row-declared feature ownership."""
        if self.feature_name_values is None:
            return None
        cached = self.feature_masks_by_candidates.get(candidates)
        if cached is not None:
            return cached
        normalized_features = np.asarray(
            [
                normalize_measurement_token(str(value))
                for value in self.feature_name_values
            ],
            dtype=object,
        )
        mask = np.isin(normalized_features, np.asarray(candidates, dtype=object))
        self.feature_masks_by_candidates[candidates] = mask
        return mask

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
        if self.feature_name_values is not None:
            feature_mask = self.feature_mask(query.field_candidates)
            if feature_mask is None or not bool(np.any(feature_mask)):
                return None
            return next(
                (
                    value_field
                    for value_field in MEASUREMENT_VALUE_FIELDS
                    if value_field in self.columns
                ),
                None,
            )
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


class ColumnarMeasurementTableSchemaCache(IdentityBoundProcessCache):
    """Process-local semantic cache keyed by a columnar row object identity."""

    registry_key = "columnar_measurement_table_schema"


@dataclass(frozen=True, slots=True)
class MeasurementFeatureQuery:
    """Query for measurement rows carrying one semantic feature value."""

    feature_name: str
    object_name: str | None = None
    dialect: RuntimeMeasurementLookupDialectLike = (
        CURRENT_RUNTIME_MEASUREMENT_LOOKUP_DIALECT
    )

    def __post_init__(self) -> None:
        if not self.feature_name:
            raise ValueError("MeasurementFeatureQuery.feature_name cannot be empty.")
        if self.object_name == "":
            raise ValueError("MeasurementFeatureQuery.object_name cannot be empty.")

    @property
    def feature_lookup(self) -> RuntimeMeasurementFeatureLookup:
        return resolve_runtime_measurement_lookup_dialect(
            self.dialect
        ).feature_lookup(self.feature_name)

    @property
    def field_candidates(self) -> tuple[str, ...]:
        return self.feature_lookup.field_aliases

    @property
    def source_candidates(self) -> tuple[str, ...]:
        return self.feature_lookup.source_aliases

    @property
    def query_object_name(self) -> str | None:
        """Return the dialect-effective object constraint for this feature."""
        return self.feature_lookup.query_object_name(self.object_name)

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
            first_row = _first_measurement_row_mapping(rows)
            matching_fields = matching_measurement_fields(
                first_row,
                self.field_candidates,
            )
            diagnostics.append(
                f"{table.name}/object={_diagnostic_value(measurement_table_object_name(table))}/"
                f"query_object={_diagnostic_value(self.query_object_name)}/"
                f"row_count={len(rows)}/first_object={_diagnostic_value(measurement_row_object_name(first_row))}/"
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
                object_mask = None
                if (
                    query_object_name is not None
                    and measurement_table_object_name(table) is None
                ):
                    object_mask = schema.object_mask(query_object_name)
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
                f"{table.name}/object={_diagnostic_value(measurement_table_object_name(table))}/"
                f"source={_diagnostic_value(table.source_image_name)}/"
                f"rows={type(table.rows).__name__}/objects={semantics.object_names[:8]}/"
                f"feature_column={_diagnostic_value(feature_column)}/"
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

    def values_for_domain(
        self,
        measurement_tables: tuple[MeasurementTable, ...],
        object_ids: Sequence[int],
    ) -> Any:
        """Return feature values aligned to an explicit object-label domain."""
        return MeasurementFeatureValueIndex(
            *self.value_index(measurement_tables)
        ).values_for_domain(object_ids)

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
class MeasurementFeatureAxisScopeSelection:
    """Select the narrowest measurement-table scope that carries a feature."""

    candidates: tuple[tuple[MeasurementTable, ...], ...]
    query: MeasurementFeatureQuery
    fallback: tuple[MeasurementTable, ...]

    def select(self) -> tuple[MeasurementTable, ...]:
        for candidate_tables in self.candidates:
            if self.query.optional_value_index(candidate_tables) is not None:
                return candidate_tables
        return self.fallback


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
        if any(not name for name in self.normalized_object_names):
            raise ValueError(
                "MeasurementObjectFeatureVectorBatchQuery.object_names cannot contain "
                "empty names."
            )

    @property
    def normalized_object_names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(str(name) for name in self.object_names))

    def value_indexes(
        self,
        measurement_tables_by_object: MeasurementTablesByObject,
    ) -> MeasurementValueIndexesByObject:
        """Return value indexes keyed by object name for this feature."""
        object_names = self.normalized_object_names
        cache_object_names = self.cache_object_names(measurement_tables_by_object)
        cached_indexes = self.cached_matching_value_indexes(
            measurement_tables_by_object,
        )
        if cached_indexes is not None:
            if all(object_name in cached_indexes for object_name in object_names):
                return {
                    object_name: cached_indexes[object_name]
                    for object_name in object_names
                }
            indexes_by_object = dict(cached_indexes)
        else:
            indexes_by_object = {}
        missing_cache_object_names = tuple(
            object_name
            for object_name in cache_object_names
            if object_name not in indexes_by_object
        )

        lookup = resolve_runtime_measurement_lookup_dialect(
            self.dialect
        ).feature_lookup(self.feature_name)
        query_objects_by_requested_object = {
            object_name: lookup.query_object_name(object_name)
            for object_name in cache_object_names
        }
        pending_indexes_by_object = {
            object_name: MeasurementFeatureValueIndex()
            for object_name in missing_cache_object_names
        }
        objects_by_table_id: dict[int, list[str]] = {}
        tables_by_id: dict[int, MeasurementTable] = {}
        for table in self.feature_measurement_tables(measurement_tables_by_object):
            table_id = id(table)
            tables_by_id[table_id] = table
            objects_by_table_id[table_id] = list(
                object_name
                for object_name in self.table_cache_object_names(
                    table,
                    object_names,
                )
                if object_name in missing_cache_object_names
            )

        table_query = MeasurementFeatureQuery(
            self.feature_name,
            dialect=self.dialect,
        )
        for table_id, table in tables_by_id.items():
            table_object_names = tuple(dict.fromkeys(objects_by_table_id[table_id]))
            if not table_object_names:
                continue
            columnar_indexes = MeasurementFeatureValueIndex.from_columnar_table_by_object(
                table,
                table_query,
                {
                    object_name: query_objects_by_requested_object[object_name]
                    for object_name in table_object_names
                },
            )
            if columnar_indexes is not None:
                for object_name, object_index in columnar_indexes.items():
                    pending_indexes_by_object[object_name] = pending_indexes_by_object[
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
                        pending_indexes_by_object[object_name] = pending_indexes_by_object[
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
                    pending_indexes_by_object[object_name] = pending_indexes_by_object[
                        object_name
                    ].merged(MeasurementFeatureValueIndex(*object_index))

        indexes_by_object.update(
            {
                object_name: object_index.as_query_result()
                for object_name, object_index in pending_indexes_by_object.items()
                if object_index.present
            }
        )
        missing_object_names = tuple(
            object_name
            for object_name in object_names
            if object_name not in indexes_by_object
        )
        if missing_object_names:
            diagnostic_query = MeasurementFeatureQuery(
                self.feature_name,
                dialect=self.dialect,
            )
            table_summaries = tuple(
                f"{object_name}:"
                + ";".join(
                    diagnostic_query.table_summaries(
                        measurement_tables_by_object[object_name]
                    )
                )
                for object_name in missing_object_names
            )
            raise ValueError(
                f"Could not resolve measurement feature {self.feature_name!r} "
                f"for object(s) {missing_object_names!r}; tables={table_summaries!r}."
            )
        resolved = {
            object_name: indexes_by_object[object_name]
            for object_name in cache_object_names
            if object_name in indexes_by_object
        }
        self.cache_value_indexes(measurement_tables_by_object, resolved)
        return {
            object_name: resolved[object_name]
            for object_name in object_names
        }

    def value_indexes_by_axis(
        self,
        measurement_tables_by_object: MeasurementTablesByObject,
        row_axis: MeasurementRowAxisField,
    ) -> dict[int, MeasurementValueIndexesByObject] | None:
        """Return object-keyed feature indexes grouped by one declared row axis."""
        cached_axis_indexes = self.cached_matching_axis_value_indexes(
            measurement_tables_by_object,
            row_axis,
        )
        object_names = self.normalized_object_names
        cache_object_names = self.cache_object_names(measurement_tables_by_object)
        if cached_axis_indexes is not None:
            requested = self.requested_axis_value_indexes(cached_axis_indexes)
            if self.requested_axis_indexes_present(requested):
                return requested
            cached_object_names = frozenset(
                object_name
                for object_indexes in cached_axis_indexes.values()
                for object_name in object_indexes
            )
        else:
            cached_axis_indexes = {}
            cached_object_names = frozenset()
        missing_cache_object_names = tuple(
            object_name
            for object_name in cache_object_names
            if object_name not in cached_object_names
        )
        if not missing_cache_object_names:
            return self.requested_axis_value_indexes(cached_axis_indexes)

        lookup = resolve_runtime_measurement_lookup_dialect(
            self.dialect
        ).feature_lookup(self.feature_name)
        query_objects_by_requested_object = {
            object_name: lookup.query_object_name(object_name)
            for object_name in cache_object_names
        }
        table_query = MeasurementFeatureQuery(
            self.feature_name,
            dialect=self.dialect,
        )
        defaults = self.empty_indexes(missing_cache_object_names)
        by_axis: dict[int, dict[str, MeasurementFeatureValueIndex]] = {}

        for table in self.feature_measurement_tables(measurement_tables_by_object):
            table_object_names = tuple(
                object_name
                for object_name in dict.fromkeys(
                    self.table_cache_object_names(table, object_names)
                )
                if object_name in missing_cache_object_names
            )
            if not table_object_names:
                continue
            axis_values = self.table_axis_values(table, row_axis)
            if not axis_values:
                self.merge_indexes(
                    defaults,
                    self.table_value_indexes(
                        table,
                        table_query,
                        table_object_names,
                        query_objects_by_requested_object,
                    ),
                )
                continue
            for axis_value in axis_values:
                target = by_axis.setdefault(
                    axis_value,
                    self.empty_indexes(missing_cache_object_names),
                )
                self.merge_indexes(
                    target,
                    self.table_value_indexes(
                        table,
                        table_query,
                        table_object_names,
                        query_objects_by_requested_object,
                        projection=MeasurementAxisValueProjection(
                            row_axis,
                            axis_value,
                        ),
                    ),
                )

        if self.indexes_present(defaults):
            for axis_value, indexes in tuple(by_axis.items()):
                merged = self.empty_indexes(missing_cache_object_names)
                self.merge_indexes(merged, defaults)
                self.merge_indexes(merged, indexes)
                by_axis[axis_value] = merged
            by_axis.setdefault(-1, defaults)
        if not by_axis:
            return self.requested_axis_value_indexes(cached_axis_indexes)
        resolved = {
            axis_value: {
                object_name: object_index.as_query_result()
                for object_name, object_index in indexes.items()
                if object_index.present
            }
            for axis_value, indexes in by_axis.items()
        }
        merged_resolved = {
            axis_value: dict(object_indexes)
            for axis_value, object_indexes in cached_axis_indexes.items()
        }
        for axis_value, object_indexes in resolved.items():
            merged_resolved.setdefault(axis_value, {}).update(object_indexes)
        self.cache_axis_value_indexes(
            measurement_tables_by_object,
            row_axis,
            merged_resolved,
        )
        return self.requested_axis_value_indexes(merged_resolved)

    @staticmethod
    def empty_indexes(
        object_names: tuple[str, ...],
    ) -> dict[str, "MeasurementFeatureValueIndex"]:
        """Return empty mutable feature indexes keyed by object name."""
        return {object_name: MeasurementFeatureValueIndex() for object_name in object_names}

    @staticmethod
    def indexes_present(
        indexes: Mapping[str, "MeasurementFeatureValueIndex"],
    ) -> bool:
        """Return whether any object index carries values."""
        return any(index.present for index in indexes.values())

    @staticmethod
    def merge_indexes(
        target: dict[str, "MeasurementFeatureValueIndex"],
        source: Mapping[str, "MeasurementFeatureValueIndex"],
    ) -> None:
        """Merge object feature indexes in place."""
        for object_name, object_index in source.items():
            if object_name not in target:
                target[object_name] = MeasurementFeatureValueIndex()
            target[object_name] = target[object_name].merged(object_index)

    def table_value_indexes(
        self,
        table: MeasurementTable,
        table_query: MeasurementFeatureQuery,
        table_object_names: tuple[str, ...],
        query_objects_by_requested_object: Mapping[str, str | None],
        *,
        projection: MeasurementAxisValueProjection | None = None,
    ) -> dict[str, "MeasurementFeatureValueIndex"]:
        """Return object indexes for one table, optionally narrowed by row axis."""
        columnar_indexes = self.columnar_table_value_indexes(
            table,
            table_query,
            table_object_names,
            query_objects_by_requested_object,
            projection=projection,
        )
        if columnar_indexes is not None:
            return columnar_indexes

        rows = self.projected_row_sequence(table, projection)
        row_sequence_index = MeasurementRowSequenceFeatureValueIndexBuild(
            table,
            table_query,
        ).index_for_rows(rows)
        if row_sequence_index is not None:
            return {
                object_name: MeasurementFeatureValueIndex(*object_index)
                for object_name in table_object_names
                for object_index in (
                    row_sequence_index.for_object(
                        query_objects_by_requested_object[object_name]
                    ),
                )
                if object_index is not None
            }
        if _is_wide_row_sequence_measurement_table(table):
            return {}
        return {
            object_name: MeasurementFeatureValueIndex.from_rows(
                rows,
                MeasurementFeatureQuery(
                    self.feature_name,
                    object_name=query_objects_by_requested_object[object_name],
                    dialect=self.dialect,
                ),
                object_id_field=measurement_table_object_id_field(table),
            )
            for object_name in table_object_names
        }

    def columnar_table_value_indexes(
        self,
        table: MeasurementTable,
        table_query: MeasurementFeatureQuery,
        table_object_names: tuple[str, ...],
        query_objects_by_requested_object: Mapping[str, str | None],
        *,
        projection: MeasurementAxisValueProjection | None = None,
    ) -> dict[str, "MeasurementFeatureValueIndex"] | None:
        """Return object indexes for one columnar table and optional axis projection."""
        rows = table.rows
        if not isinstance(rows, ColumnarRows):
            return None
        row_mask = None
        if projection is not None:
            column_names = tuple(str(column) for column in rows.columns)
            if projection.field_name in column_names:
                row_mask = projection.columnar_mask(
                    columnar_row_values(rows, projection.field_name)
                )
        columnar_indexes = MeasurementFeatureValueIndex.from_columnar_table_by_object(
            table,
            table_query,
            {
                object_name: query_objects_by_requested_object[object_name]
                for object_name in table_object_names
            },
            row_mask=row_mask,
        )
        return columnar_indexes

    @staticmethod
    def projected_row_sequence(
        table: MeasurementTable,
        projection: MeasurementAxisValueProjection | None,
    ) -> tuple[object, ...]:
        """Return table rows optionally narrowed by the shared row-axis projection."""
        rows = measurement_rows((table,))
        if projection is None or not rows:
            return rows
        field_name = projection.field_name
        if not any(field_name in measurement_row_mapping(row) for row in rows):
            return rows
        return tuple(
            row
            for row in rows
            if projection.matches_value(measurement_row_mapping(row).get(field_name))
        )

    @staticmethod
    def table_axis_values(
        table: MeasurementTable,
        row_axis: MeasurementRowAxisField,
    ) -> tuple[int, ...]:
        """Return sorted declared values for one row axis on one table."""
        axis_field = row_axis.value
        rows = table.rows
        if isinstance(rows, ColumnarRows):
            column_names = tuple(str(column) for column in rows.columns)
            if axis_field not in column_names:
                return ()
            return measurement_axis_integer_domain(
                columnar_row_values(rows, axis_field),
                row_axis,
            )
        return tuple(
            sorted(
                {
                    axis_value
                    for row in measurement_rows((table,))
                    for axis_value in (
                        measurement_axis_integer_value(
                            measurement_row_mapping(row).get(axis_field),
                            row_axis,
                        ),
                    )
                    if axis_value is not None
                }
            )
        )

    def cache_object_names(
        self,
        measurement_tables_by_object: MeasurementTablesByObject,
    ) -> tuple[str, ...]:
        """Return object domains worth indexing for this feature/table set."""
        return tuple(
            dict.fromkeys(
                (
                    *self.normalized_object_names,
                    *(
                        object_name
                        for table in self.feature_measurement_tables(
                            measurement_tables_by_object
                        )
                        for object_name in self.table_cache_object_names(
                            table,
                            self.normalized_object_names,
                        )
                    ),
                )
            )
        )

    def table_cache_object_names(
        self,
        table: MeasurementTable,
        fallback_object_names: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Return result object domains represented by one table scan."""
        semantics = MeasurementTableObjectFeatureSemantics.from_table(table)
        if not semantics.object_names:
            return fallback_object_names
        feature_lookup = resolve_runtime_measurement_lookup_dialect(
            self.dialect
        ).feature_lookup(self.feature_name)
        unconstrained_result_names = tuple(
            object_name
            for object_name in fallback_object_names
            if feature_lookup.query_object_name(object_name) is None
        )
        return tuple(dict.fromkeys((*semantics.object_names, *unconstrained_result_names)))

    def unique_measurement_tables(
        self,
        measurement_tables_by_object: MeasurementTablesByObject,
    ) -> tuple[MeasurementTable, ...]:
        """Return unique table owners across requested object domains."""
        tables: list[MeasurementTable] = []
        table_ids: set[int] = set()
        for object_name in self.normalized_object_names:
            for table in measurement_tables_by_object[object_name]:
                table_id = id(table)
                if table_id in table_ids:
                    continue
                table_ids.add(table_id)
                tables.append(table)
        return tuple(tables)

    def feature_measurement_tables(
        self,
        measurement_tables_by_object: MeasurementTablesByObject,
    ) -> tuple[MeasurementTable, ...]:
        """Return unique tables whose semantics can carry this feature."""
        query = MeasurementFeatureQuery(
            self.feature_name,
            dialect=self.dialect,
        )
        return tuple(
            table
            for table in self.unique_measurement_tables(measurement_tables_by_object)
            if query.table_may_carry_feature(table)
        )

    def cache_key(
        self,
        measurement_tables_by_object: MeasurementTablesByObject,
    ) -> MeasurementObjectFeatureVectorBatchCacheKey:
        """Return the process-local identity key for this feature/table batch."""
        return MeasurementObjectFeatureVectorBatchCacheKey(
            feature_name=self.feature_name,
            dialect_identity=id(resolve_runtime_measurement_lookup_dialect(self.dialect)),
            table_identities=tuple(
                id(table)
                for table in self.feature_measurement_tables(measurement_tables_by_object)
            ),
        )

    def axis_cache_key(
        self,
        measurement_tables_by_object: MeasurementTablesByObject,
        row_axis: MeasurementRowAxisField,
    ) -> MeasurementObjectFeatureAxisBatchCacheKey:
        """Return the process-local identity key for this feature/table/axis batch."""
        return MeasurementObjectFeatureAxisBatchCacheKey(
            feature_name=self.feature_name,
            dialect_identity=id(resolve_runtime_measurement_lookup_dialect(self.dialect)),
            row_axis=row_axis,
            table_identities=tuple(
                id(table)
                for table in self.feature_measurement_tables(measurement_tables_by_object)
            ),
        )

    def table_owners(
        self,
        measurement_tables_by_object: MeasurementTablesByObject,
    ) -> tuple[MeasurementTable, ...]:
        """Return table owners used to protect identity-keyed cache entries."""
        return self.feature_measurement_tables(measurement_tables_by_object)

    def cached_value_indexes(
        self,
        measurement_tables_by_object: MeasurementTablesByObject,
    ) -> MeasurementValueIndexesByObject | None:
        """Return cached value indexes when table identities still match."""
        cached_indexes = self.cached_matching_value_indexes(
            measurement_tables_by_object,
        )
        if cached_indexes is None:
            return None
        if not all(
            object_name in cached_indexes
            for object_name in self.normalized_object_names
        ):
            return None
        return {
            object_name: cached_indexes[object_name]
            for object_name in self.normalized_object_names
        }

    def cached_matching_value_indexes(
        self,
        measurement_tables_by_object: MeasurementTablesByObject,
    ) -> MeasurementValueIndexesByObject | None:
        """Return every cached object index for this feature/table identity."""
        cache = MeasurementObjectFeatureVectorBatchQueryCache.process_cache()
        cached = cache.cached_value(self.cache_key(measurement_tables_by_object))
        if cached is None:
            return None
        cached_owners, cached_indexes = cached
        if not identity_owner_tuples_match(
            cached_owners,
            self.table_owners(measurement_tables_by_object),
        ):
            return None
        return cached_indexes

    def cached_axis_value_indexes(
        self,
        measurement_tables_by_object: MeasurementTablesByObject,
        row_axis: MeasurementRowAxisField,
    ) -> dict[int, MeasurementValueIndexesByObject] | None:
        """Return cached row-axis indexes when table identities still match."""
        cached_indexes = self.cached_matching_axis_value_indexes(
            measurement_tables_by_object,
            row_axis,
        )
        if cached_indexes is None:
            return None
        requested = self.requested_axis_value_indexes(cached_indexes)
        if not self.requested_axis_indexes_present(requested):
            return None
        return requested

    def cached_matching_axis_value_indexes(
        self,
        measurement_tables_by_object: MeasurementTablesByObject,
        row_axis: MeasurementRowAxisField,
    ) -> Mapping[int, MeasurementValueIndexesByObject] | None:
        """Return every cached row-axis index for this feature/table identity."""
        cache = MeasurementObjectFeatureAxisBatchQueryCache.process_cache()
        cached = cache.cached_value(
            self.axis_cache_key(measurement_tables_by_object, row_axis)
        )
        if cached is None:
            return None
        cached_owners, cached_indexes = cached
        if not identity_owner_tuples_match(
            cached_owners,
            self.table_owners(measurement_tables_by_object),
        ):
            return None
        return cached_indexes

    def requested_axis_value_indexes(
        self,
        indexes_by_axis: Mapping[int, MeasurementValueIndexesByObject],
    ) -> dict[int, MeasurementValueIndexesByObject]:
        """Project cached row-axis indexes to requested object names."""
        object_names = self.normalized_object_names
        return {
            axis_value: {
                object_name: object_indexes[object_name]
                for object_name in object_names
                if object_name in object_indexes
            }
            for axis_value, object_indexes in indexes_by_axis.items()
        }

    def requested_axis_indexes_present(
        self,
        indexes_by_axis: Mapping[int, MeasurementValueIndexesByObject],
    ) -> bool:
        """Return whether cached axis indexes satisfy every requested object."""
        return all(
            any(object_name in object_indexes for object_indexes in indexes_by_axis.values())
            for object_name in self.normalized_object_names
        )

    def cache_value_indexes(
        self,
        measurement_tables_by_object: MeasurementTablesByObject,
        value_indexes: MeasurementValueIndexesByObject,
    ) -> None:
        """Store value indexes with table-owner references for id-reuse safety."""
        MeasurementObjectFeatureVectorBatchQueryCache.process_cache().store_value(
            self.cache_key(measurement_tables_by_object),
            (self.table_owners(measurement_tables_by_object), value_indexes),
        )

    def cache_axis_value_indexes(
        self,
        measurement_tables_by_object: MeasurementTablesByObject,
        row_axis: MeasurementRowAxisField,
        value_indexes: Mapping[int, MeasurementValueIndexesByObject],
    ) -> None:
        """Store row-axis value indexes with table-owner id-reuse protection."""
        MeasurementObjectFeatureAxisBatchQueryCache.process_cache().store_value(
            self.axis_cache_key(measurement_tables_by_object, row_axis),
            (self.table_owners(measurement_tables_by_object), value_indexes),
        )


class MeasurementObjectFeatureVectorBatchQueryCache(
    ProcessLocalBoundedCache[
        MeasurementObjectFeatureVectorBatchCacheKey,
        MeasurementObjectFeatureVectorBatchCacheValue,
    ]
):
    """Process-local cache for repeated object-feature table batch indexes."""

    max_entries = 1024


class MeasurementObjectFeatureAxisBatchQueryCache(
    ProcessLocalBoundedCache[
        MeasurementObjectFeatureAxisBatchCacheKey,
        MeasurementObjectFeatureAxisBatchCacheValue,
    ]
):
    """Process-local cache for repeated object-feature row-axis table indexes."""

    max_entries = 1024


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
        object_key = query.query_object_name
        indexes = cls.from_columnar_table_by_object(
            table,
            query,
            {object_key: object_key},
        )
        if indexes is None:
            return None
        return indexes.get(object_key)

    @classmethod
    def from_columnar_table_by_object(
        cls,
        table: MeasurementTable,
        query: MeasurementFeatureQuery,
        query_object_names_by_result: Mapping[str | None, str | None],
        *,
        row_mask: Any | None = None,
    ) -> dict[str | None, "MeasurementFeatureValueIndex"] | None:
        """Return feature indexes from one columnar table scan."""
        rows = table.rows
        if not isinstance(rows, ColumnarRows):
            return None

        if not query.table_source_matches_feature(table):
            return {
                result_object_name: cls()
                for result_object_name in query_object_names_by_result
            }

        schema = ColumnarMeasurementTableSchema.from_table(table)
        feature_column = schema.matching_feature_column(query)
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
            else schema.source_mask(query.source_candidates)
        )
        feature_mask = schema.feature_mask(query.field_candidates)
        source_feature_mask = (
            value_mask
            if source_mask is None
            else np.logical_and(value_mask, source_mask)
        )
        base_mask = (
            source_feature_mask
            if feature_mask is None
            else np.logical_and(source_feature_mask, feature_mask)
        )
        if row_mask is not None:
            base_mask = np.logical_and(base_mask, row_mask)
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
        indexes: dict[str | None, MeasurementFeatureValueIndex] = {}
        for result_object_name, query_object_name in query_object_names_by_result.items():
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
            object_values = raw_values[effective_mask].astype(float, copy=False)
            if object_id_field is None or object_ids is None:
                indexes[result_object_name] = cls(
                    {},
                    [float(value) for value in object_values],
                )
                continue
            indexes[result_object_name] = cls(
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
        return indexes

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

    def values_for_domain(self, object_ids: Sequence[int]) -> Any:
        """Return values aligned to an explicit object-label ID domain."""
        resolved_object_ids = tuple(int(object_id) for object_id in object_ids)
        if self.values_by_label:
            return ObjectLabelMeasurementValues.from_value_mapping(
                resolved_object_ids,
                self.values_by_label,
            ).values
        if self.positional_values:
            return ObjectLabelMeasurementValues.from_positional_values(
                resolved_object_ids,
                self.positional_values,
            ).values
        raise ValueError("Could not resolve measurement feature values.")

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
        cache_key = MeasurementRowSequenceLayoutCacheKey.from_rows(rows, query)
        cached = MeasurementRowSequenceLayoutCache.process_cache().cached_value(cache_key)
        if cached is not None:
            cached_rows, cached_layout = cached
            if cached_rows is rows:
                return cached_layout
        layout = cls._from_rows_uncached(rows, query)
        return MeasurementRowSequenceLayoutCache.process_cache().store_value(
            cache_key,
            (rows, layout),
        )[1]

    @classmethod
    def _from_rows_uncached(
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
class MeasurementRowSequenceLayoutCacheKey:
    """Identity key for row-sequence layout discovery."""

    rows_identity: int
    field_candidates: tuple[str, ...]

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[object],
        query: MeasurementFeatureQuery | None,
    ) -> "MeasurementRowSequenceLayoutCacheKey":
        return cls(
            id(rows),
            () if query is None else query.field_candidates,
        )


MeasurementRowSequenceLayoutCacheValue = tuple[
    Sequence[object],
    MeasurementRowSequenceLayout,
]


class MeasurementRowSequenceLayoutCache(
    ProcessLocalBoundedCache[
        MeasurementRowSequenceLayoutCacheKey,
        MeasurementRowSequenceLayoutCacheValue,
    ]
):
    """Process-local cache for row-sequence semantic layout discovery."""

    max_entries = 2048


def _is_wide_row_sequence_measurement_table(table: MeasurementTable) -> bool:
    """Return whether row fields are direct measurement columns, not long rows."""
    rows = table.row_sequence_payloads()
    if rows is None:
        return False
    return MeasurementRowSequenceLayout.from_rows(rows).is_wide_only


@dataclass(frozen=True, slots=True)
class MeasurementRowSequenceFeatureValueIndexBuild:
    """Staged build request for row-sequence feature indexes."""

    table: MeasurementTable
    query: MeasurementFeatureQuery

    @property
    def rows(self) -> tuple[object, ...] | None:
        return self.table.row_sequence_payloads()

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
        return self.index_for_rows(rows)

    def index_for_rows(
        self,
        rows: tuple[object, ...],
    ) -> MeasurementRowSequenceFeatureValueIndex | None:
        """Build the row-sequence index for an explicit row projection."""
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
        if table.column_names() is not None:
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
        if table.row_sequence_payloads() is not None:
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


@dataclass(frozen=True, slots=True, kw_only=True)
class MeasurementTableFeatureQuery(MeasurementFeatureQuery, ABC):
    """Feature query bound to a concrete measurement-table collection."""

    measurement_tables: tuple[MeasurementTable, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class ObjectMeasurementLabelPlaneBinding(MeasurementTableFeatureQuery):
    """Bind one object measurement feature onto a label-plane object domain."""

    labels: object

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
        values_by_label, positional_values = binding.value_index(
            binding.measurement_tables
        )
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
            summaries = binding.table_summaries(binding.measurement_tables)
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
    query = MeasurementFeatureQuery(
        feature_name,
        object_name=object_name,
        dialect=dialect,
    )
    resolved_object_ids = (
        tuple(range(1, object_count + 1))
        if object_ids is None
        else tuple(int(object_id) for object_id in object_ids)
    )
    return query.values_for_domain(measurement_tables, resolved_object_ids)


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
