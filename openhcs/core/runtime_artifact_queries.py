"""Semantic queries over typed OpenHCS runtime artifacts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field, fields as dataclass_fields, is_dataclass
import math
from typing import Any, ClassVar
from weakref import WeakKeyDictionary

from metaclass_registry import AutoRegisterMeta
from nominal_refactor_advisor.descriptor_algebra import AliasProperty

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

        mask = np.asarray(self.object_name_values, dtype=object) == object_name
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
class RuntimeArtifactQueryContext:
    """Execution-scope view over a RuntimeValueStore."""

    store: RuntimeValueStore
    axis_id: str
    group_key: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.store, RuntimeValueStore):
            raise TypeError(
                "RuntimeArtifactQueryContext.store must be RuntimeValueStore, "
                f"got {type(self.store).__name__}."
            )
        if not self.axis_id:
            raise ValueError("RuntimeArtifactQueryContext.axis_id cannot be empty.")

    @property
    def match_group(self) -> bool:
        return self.group_key is not None

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

    def __post_init__(self) -> None:
        if not self.feature_name:
            raise ValueError("MeasurementFeatureQuery.feature_name cannot be empty.")
        if self.object_name == "":
            raise ValueError("MeasurementFeatureQuery.object_name cannot be empty.")

    @property
    def field_candidates(self) -> tuple[str, ...]:
        return ordered_measurement_feature_candidates(
            self.feature_name,
            dialect=self.dialect,
        )

    @property
    def source_candidates(self) -> tuple[str, ...]:
        return ordered_measurement_source_candidates(
            self.feature_name,
            dialect=self.dialect,
        )

    @property
    def query_object_name(self) -> str | None:
        """Return the dialect-effective object constraint for this feature."""
        dialect = resolve_runtime_measurement_lookup_dialect(self.dialect)
        return dialect.feature_lookup(self.feature_name).query_object_name(
            self.object_name
        )

    def row_value(self, row: object) -> object | None:
        """Return the row value matching this feature query, if present."""
        row_mapping = measurement_row_mapping(row)
        if not self._matches_object(row_mapping):
            return None

        candidates = self.field_candidates
        if measurement_row_feature_matches(row_mapping, candidates):
            return measurement_row_first_value(row_mapping)

        field_name = matching_measurement_field(row_mapping, candidates)
        if field_name is None:
            return None
        if not measurement_row_source_matches_feature(row_mapping, self):
            return None
        return row_mapping[field_name]

    def value_index(
        self,
        measurement_tables: tuple[MeasurementTable, ...],
    ) -> MeasurementValueIndexResult:
        """Return object-id and positional values for this feature."""
        value_index = self.optional_value_index(measurement_tables)
        if value_index is None:
            raise ValueError(
                f"Could not resolve measurement feature {self.feature_name!r}; "
                f"tables={self.table_summaries(measurement_tables)!r}."
            )
        return value_index

    def table_summaries(
        self,
        measurement_tables: tuple[MeasurementTable, ...],
    ) -> tuple[str, ...]:
        """Return compact diagnostics for tables searched by this query."""
        summaries: list[str] = []
        for table in measurement_tables:
            semantics = MeasurementTableObjectFeatureSemantics.from_table(table)
            features = tuple(sorted(semantics.feature_names))
            summaries.append(
                f"{table.name}/object={measurement_table_object_name(table) or '<none>'}/"
                f"source={table.source_image_name or '<none>'}/"
                f"rows={type(table.rows).__name__}/features={features[:8]}"
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
            object_name: MeasurementFeatureValueIndex.empty()
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
            row_sequence_index = MeasurementRowSequenceFeatureValueIndex.from_table(
                table,
                table_query,
            )
            if row_sequence_index is not None:
                for object_name in table_object_names:
                    object_index = row_sequence_index.for_object(
                        query_objects_by_requested_object[object_name]
                    )
                    if object_index is not None:
                        indexes_by_object[object_name] = indexes_by_object[
                            object_name
                        ].merged(MeasurementFeatureValueIndex.from_query_result(object_index))
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
                    ].merged(MeasurementFeatureValueIndex.from_query_result(object_index))

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


@dataclass(frozen=True, slots=True)
class MeasurementFeatureValueIndex:
    """Object-label and positional values for one measurement feature."""

    values_by_label: dict[int, float]
    positional_values: list[float]

    @classmethod
    def empty(cls) -> "MeasurementFeatureValueIndex":
        return cls({}, [])

    @classmethod
    def from_query_result(
        cls,
        value_index: MeasurementValueIndexResult,
    ) -> "MeasurementFeatureValueIndex":
        return cls(dict(value_index[0]), list(value_index[1]))

    @classmethod
    def from_tables(
        cls,
        measurement_tables: tuple[MeasurementTable, ...],
        query: MeasurementFeatureQuery,
    ) -> "MeasurementFeatureValueIndex":
        index = cls.empty()
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

        row_sequence_index = MeasurementRowSequenceFeatureValueIndex.from_table(
            table,
            query,
        )
        if row_sequence_index is not None:
            object_index = row_sequence_index.for_object(query.query_object_name)
            return (
                cls.empty()
                if object_index is None
                else cls.from_query_result(object_index)
            )

        if _is_wide_row_sequence_measurement_table(table):
            return cls.empty()
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
        source_mask = schema.source_mask(query.source_candidates)
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

        values = np.asarray(columnar_row_values(rows, feature_column), dtype=float)
        if object_mask is not None:
            values = values[object_mask]
        if source_mask is not None:
            values = values[source_mask if object_mask is None else source_mask[object_mask]]
        object_id_field = MeasurementRowSequenceFeatureValueIndex.matching_object_id_field(
            columns,
            measurement_table_object_id_field(table),
        )
        if object_id_field is not None and object_id_field in columns:
            object_ids = np.asarray(
                columnar_row_values(rows, object_id_field),
                dtype=np.int64,
            )
            if object_mask is not None:
                object_ids = object_ids[object_mask]
            if source_mask is not None:
                object_ids = object_ids[
                    source_mask if object_mask is None else source_mask[object_mask]
                ]
            return cls(
                {
                    int(object_id): float(value)
                    for object_id, value in zip(object_ids, values, strict=True)
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
        index = cls.empty()
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
    def from_table(
        cls,
        table: MeasurementTable,
        query: MeasurementFeatureQuery,
    ) -> "MeasurementRowSequenceFeatureValueIndex | None":
        rows = table.rows
        if isinstance(rows, ColumnarRows):
            return None
        if not isinstance(rows, list | tuple) or not rows:
            return None

        field_names = cls.field_names(rows, query)
        table_source_image_name = (
            None
            if cls.table_declares_object_identity(table, field_names)
            else table.source_image_name
        )
        feature_field = cls.matching_row_value_field(
            field_names,
            query,
            table_source_image_name=table_source_image_name,
        )
        if feature_field is None:
            return None

        object_id_field = cls.matching_object_id_field(
            field_names,
            measurement_table_object_id_field(table),
        )
        values_by_object: MeasurementFeatureValueIndexesByObject = {}
        for row in rows:
            row_mapping = measurement_row_mapping(row)
            if not measurement_row_source_matches_feature(row_mapping, query):
                continue
            value = row_mapping.get(feature_field)
            if value in (None, ""):
                continue
            object_label = measurement_object_label(
                row_mapping,
                object_id_field=object_id_field,
            )
            object_name = measurement_row_object_name(row_mapping)
            values_by_object.setdefault(
                object_name,
                MeasurementFeatureValueIndex.empty(),
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
        specific_candidates = frozenset(query.field_candidates)
        found_feature = any(
            normalize_measurement_token(field_name) in specific_candidates
            for field_name in first_row_names
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
                if normalize_measurement_token(normalized_name) in specific_candidates:
                    found_feature = True
                if normalized_name in MEASUREMENT_OBJECT_ID_FIELDS:
                    found_object_id = True
            if found_feature and found_object_id:
                break
        return tuple(field_names)

    @staticmethod
    def matching_row_value_field(
        field_names: tuple[str, ...],
        query: MeasurementFeatureQuery,
        *,
        table_source_image_name: str | None,
    ) -> str | None:
        candidates = query.field_candidates
        if table_source_image_name is not None:
            normalized_source = normalize_measurement_token(table_source_image_name)
            if (
                normalized_source not in MEASUREMENT_UNQUALIFIED_SOURCE_NAMES
                and normalized_source not in query.source_candidates
            ):
                return None
        return matching_measurement_field(
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

    @classmethod
    def table_declares_object_identity(
        cls,
        table: MeasurementTable,
        field_names: tuple[str, ...],
    ) -> bool:
        """Return whether table-level image source should not qualify feature fields."""
        return (
            measurement_table_object_name(table) is not None
            or MEASUREMENT_OBJECT_NAME_FIELD in field_names
            or cls.matching_object_id_field(
                field_names,
                measurement_table_object_id_field(table),
            )
            is not None
        )

    def for_object(
        self,
        object_name: str | None,
    ) -> OptionalMeasurementValueIndexResult:
        if object_name is None:
            merged_index = MeasurementFeatureValueIndex.empty()
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
        if _is_wide_row_sequence_measurement_table(table):
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
        return MeasurementTableObjectFeatureSemantics.feature_names_from_fields(
            table.fields,
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

    @property
    def value_index(self) -> MeasurementValueIndexResult:
        """Return prior measurements keyed by label or compact position."""
        return measurement_value_index(
            self.measurement_tables,
            self.feature_name,
            object_name=self.object_name,
            dialect=self.dialect,
        )

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
    value_index: ClassVar[AliasProperty[MeasurementValueIndexResult]] = AliasProperty(
        "indexed_values"
    )

    @property
    def object_domain(self) -> tuple[int, ...]:
        """Return object IDs materially present in this indexed label plane."""
        return ObjectLabelIdDomainStrategy.for_value(self.labels).present_ids(self.labels)


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
        values_by_label, positional_values = binding.value_index
        if not domain:
            return np.array([], dtype=float)
        if values_by_label:
            return np.array([values_by_label.get(label, np.nan) for label in domain])
        if positional_values:
            return np.array(positional_values[: len(domain)])
        raise ValueError(
            f"Could not resolve measurement feature {binding.feature_name!r}."
        )


def measurement_values_for_label_plane(
    measurement_tables: tuple[MeasurementTable, ...],
    feature_name: str,
    labels: object,
    *,
    object_name: str,
    dialect: RuntimeMeasurementLookupDialectLike = (
        CURRENT_RUNTIME_MEASUREMENT_LOOKUP_DIALECT
    ),
) -> Any:
    """Return one feature vector aligned to a label-plane object domain."""
    return ObjectMeasurementLabelPlaneBinding(
        measurement_tables=measurement_tables,
        object_name=object_name,
        feature_name=feature_name,
        labels=labels,
        dialect=dialect,
    ).values()


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
    first_row = measurement_row_mapping(rows[0])
    return not any(field in first_row for field in MEASUREMENT_FEATURE_NAME_FIELDS)


def _columnar_measurement_rows(rows: ColumnarRows) -> tuple[Mapping[str, object], ...]:
    """Return record mappings from a nominal columnar table payload."""
    columns = tuple(str(column) for column in rows.columns)
    column_values = tuple(columnar_row_values(rows, column) for column in columns)
    return tuple(
        dict(zip(columns, values, strict=True))
        for values in zip(*column_values, strict=True)
    )


def columnar_row_values(rows: ColumnarRows, column: str) -> Sequence[object]:
    """Return one column from a nominal columnar payload."""
    columns = rows.columns
    if isinstance(columns, Mapping):
        return columns[column]
    return rows[column]


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
    for candidate in candidates:
        for field_name in row:
            normalized = normalize_measurement_token(field_name)
            if candidate == normalized:
                return field_name
    return None


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
        return _coerce_measurement_object_label(row[object_id_field])
    for key in MEASUREMENT_OBJECT_ID_FIELDS:
        if key in row:
            return _coerce_measurement_object_label(row[key])
    return None


def measurement_row_has_object_identity(
    row: Mapping[str, object],
    *,
    object_id_field: str | None = None,
) -> bool:
    """Return whether a measurement row carries object identity."""
    try:
        return measurement_object_label(row, object_id_field=object_id_field) is not None
    except (TypeError, ValueError):
        return False


def _coerce_measurement_object_label(value: object) -> int | None:
    """Return an integer object label from runtime/CSV scalar encodings."""
    if value in (None, ""):
        return None
    return int(float(value))


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


def measurement_value_index(
    measurement_tables: tuple[MeasurementTable, ...],
    feature_name: str,
    *,
    object_name: str | None = None,
    dialect: RuntimeMeasurementLookupDialectLike = (
        CURRENT_RUNTIME_MEASUREMENT_LOOKUP_DIALECT
    ),
) -> MeasurementValueIndexResult:
    """Return object-id and positional values for one feature."""
    return MeasurementFeatureQuery(
        feature_name,
        object_name=object_name,
        dialect=dialect,
    ).value_index(measurement_tables)


def optional_measurement_value_index(
    measurement_tables: tuple[MeasurementTable, ...],
    feature_name: str,
    *,
    object_name: str | None = None,
    dialect: RuntimeMeasurementLookupDialectLike = (
        CURRENT_RUNTIME_MEASUREMENT_LOOKUP_DIALECT
    ),
) -> OptionalMeasurementValueIndexResult:
    """Return object-id and positional values for one feature when present."""
    return MeasurementFeatureQuery(
        feature_name,
        object_name=object_name,
        dialect=dialect,
    ).optional_value_index(measurement_tables)


def measurement_scalar_value_for_feature(
    measurement_tables: tuple[MeasurementTable, ...],
    feature_name: str,
    *,
    object_name: str | None = None,
    dialect: RuntimeMeasurementLookupDialectLike = (
        CURRENT_RUNTIME_MEASUREMENT_LOOKUP_DIALECT
    ),
) -> float:
    """Return exactly one scalar measurement value for one feature."""
    return MeasurementFeatureQuery(
        feature_name,
        object_name=object_name,
        dialect=dialect,
    ).scalar_value(measurement_tables)


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
    values_by_label, positional_values = measurement_value_index(
        measurement_tables,
        feature_name,
        object_name=object_name,
        dialect=dialect,
    )
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
        if value in (None, ""):
            return False
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return True
        return math.isfinite(numeric_value)


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
        rows = AxisFilteredMeasurementColumnarRows(self.table.rows, self.projection)
        return self._with_rows(rows, self.table.fields)

    def _row_sequence_table(self, rows: Sequence[object]) -> MeasurementTable:
        normalized_rows = normalize_measurement_table_rows(rows, fields=())
        return self._with_rows(
            normalized_rows,
            self._compatible_fields(normalized_rows),
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
    ) -> MeasurementTable:
        return MeasurementTable(
            name=self.table.name,
            rows=rows,
            object_name=self.table.object_name,
            fields=tuple(fields),
            object_id_field=self.table.object_id_field,
            source_image_name=self.table.source_image_name,
            subject=self.table.subject,
        )


@dataclass(frozen=True, slots=True)
class MeasurementTableAxisQuery:
    """Reusable query for projecting measurement tables by one row axis."""

    axis: MeasurementRowAxisField
    value: int

    def slice(cls, slice_index: int) -> "MeasurementTableAxisQuery":
        """Return a query for one runtime slice index."""
        return cls(MeasurementRowAxisField.SLICE_INDEX, int(slice_index))

    @classmethod
    def image_number(cls, image_number: int) -> "MeasurementTableAxisQuery":
        """Return a query for one CellProfiler ImageNumber row domain."""
        return cls(MeasurementRowAxisField.IMAGE_NUMBER, int(image_number))

    def table(self, table: MeasurementTable) -> MeasurementTable:
        """Return one measurement table narrowed to this row-axis value."""
        return MeasurementTableAxisProjection(table, self.axis, self.value).apply()

    def tables(
        self,
        measurement_tables: tuple[MeasurementTable, ...],
    ) -> tuple[MeasurementTable, ...]:
        """Return measurement tables narrowed to this row-axis value."""
        return tuple(self.table(table) for table in measurement_tables)


def measurement_table_for_slice(
    table: MeasurementTable,
    slice_index: int,
) -> MeasurementTable:
    """Return a measurement table narrowed to one slice when rows declare slices."""
    return MeasurementTableAxisQuery.slice(slice_index).table(table)


def measurement_table_slice_indices(table: MeasurementTable) -> set[int]:
    """Return runtime slice indexes declared by one measurement table."""
    if not table.subject.scope.projects_runtime_slices:
        return set()
    slice_field = MeasurementRowAxisField.SLICE_INDEX.value
    if isinstance(table.rows, ColumnarRows):
        column_names = tuple(str(column) for column in table.rows.columns)
        if slice_field not in column_names:
            return set()
        return {
            int(slice_index)
            for slice_index in columnar_row_values(table.rows, slice_field)
            if slice_index is not None
        }
    return {
        int(row_mapping[slice_field])
        for row in measurement_rows((table,))
        for row_mapping in (measurement_row_mapping(row),)
        if row_mapping.get(slice_field) is not None
    }


def measurement_tables_for_slice(
    measurement_tables: tuple[MeasurementTable, ...],
    slice_index: int,
) -> tuple[MeasurementTable, ...]:
    """Return measurement tables narrowed to one slice where row axes permit it."""
    return MeasurementTableAxisQuery.slice(slice_index).tables(measurement_tables)


def measurement_table_for_image_number(
    table: MeasurementTable,
    image_number: int,
) -> MeasurementTable:
    """Return a measurement table narrowed to one CellProfiler ImageNumber."""
    return MeasurementTableAxisQuery.image_number(image_number).table(table)


def measurement_tables_for_image_number(
    measurement_tables: tuple[MeasurementTable, ...],
    image_number: int,
) -> tuple[MeasurementTable, ...]:
    """Return measurement tables narrowed to one CellProfiler ImageNumber."""
    return MeasurementTableAxisQuery.image_number(image_number).tables(
        measurement_tables,
    )


def measurement_values_for_label_slices(
    measurement_tables: tuple[MeasurementTable, ...],
    feature_name: str,
    labels: object,
    *,
    object_name: str | None = None,
    dialect: RuntimeMeasurementLookupDialectLike = (
        CURRENT_RUNTIME_MEASUREMENT_LOOKUP_DIALECT
    ),
) -> tuple[Any, ...]:
    """Return measurement values aligned to positive label IDs in each label plane."""
    import numpy as np

    label_array = np.asarray(labels)
    label_planes = (
        (label_array,)
        if label_array.ndim <= 2
        else tuple(label_array[index] for index in range(label_array.shape[0]))
    )
    if label_array.ndim > 2:
        values_by_slice = _measurement_value_indexes_by_slice(
            measurement_tables,
            feature_name,
            object_name=object_name,
            dialect=dialect,
        )
        if values_by_slice is not None:
            return tuple(
                IndexedObjectMeasurementLabelPlaneBinding(
                    measurement_tables=(),
                    object_name=object_name or "",
                    feature_name=feature_name,
                    labels=label_plane,
                    dialect=dialect,
                    indexed_values=_measurement_value_index_for_label_slice(
                        values_by_slice,
                        slice_index,
                    ),
                ).values()
                for slice_index, label_plane in enumerate(label_planes)
            )
    return tuple(
        IndexedObjectMeasurementLabelPlaneBinding(
            measurement_tables=sliced_tables,
            object_name=object_name or "",
            feature_name=feature_name,
            labels=label_plane,
            dialect=dialect,
            indexed_values=measurement_value_index(
                sliced_tables,
                feature_name,
                object_name=object_name,
                dialect=dialect,
            ),
        ).values()
        for slice_index, label_plane, sliced_tables in (
            (
                index,
                plane,
                measurement_tables_for_slice(measurement_tables, index),
            )
            for index, plane in enumerate(label_planes)
        )
    )


def _measurement_value_index_for_label_slice(
    values_by_slice: Mapping[int, MeasurementValueIndexResult],
    slice_index: int,
) -> MeasurementValueIndexResult:
    """Return the measurement index for a label plane, broadcasting singletons."""
    if slice_index in values_by_slice:
        return values_by_slice[slice_index]
    if -1 in values_by_slice:
        return values_by_slice[-1]
    concrete_slice_indexes = tuple(
        index for index in values_by_slice if index >= 0
    )
    if len(concrete_slice_indexes) == 1:
        return values_by_slice[concrete_slice_indexes[0]]
    return {}, []


def _measurement_value_indexes_by_slice(
    measurement_tables: tuple[MeasurementTable, ...],
    feature_name: str,
    *,
    object_name: str | None,
    dialect: RuntimeMeasurementLookupDialectLike,
) -> dict[int, MeasurementValueIndexResult] | None:
    """Return per-slice feature indexes without re-scanning tables per plane."""
    query = MeasurementFeatureQuery(
        feature_name,
        object_name=object_name,
        dialect=dialect,
    )
    defaults: MeasurementValueIndexResult = ({}, [])
    by_slice: dict[int, MeasurementValueIndexResult] = {}

    for table in measurement_tables:
        if object_name is not None:
            table_object = measurement_table_object_name(table)
            if table_object not in (None, object_name):
                continue

        slice_indices = measurement_table_slice_indices(table)
        if not slice_indices:
            table_index = MeasurementFeatureValueIndex.from_table(table, query)
            if table_index.present:
                _merge_measurement_value_index(
                    defaults,
                    table_index.as_query_result(),
                )
            continue

        for slice_index in sorted(slice_indices):
            table_index = MeasurementFeatureValueIndex.from_table(
                measurement_table_for_slice(table, slice_index),
                query,
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
    return by_slice


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

    def __post_init__(self) -> None:
        import numpy as np

        columns = {
            str(column): np.asarray(columnar_row_values(self.rows, str(column)))
            for column in self.rows.columns
        }
        axis_values = columns.get(self.projection.field_name)
        if axis_values is None:
            object.__setattr__(self, "_columns", columns)
            return
        axis_mask = self.projection.columnar_mask(axis_values)
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
