"""Semantic queries over typed OpenHCS runtime artifacts."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Any, ClassVar
from weakref import WeakKeyDictionary

from abc import ABC, abstractmethod
from metaclass_registry import AutoRegisterMeta

from openhcs.core.artifacts import ArtifactKind
from openhcs.core.measurement_lookup_dialect import (
    CURRENT_RUNTIME_MEASUREMENT_LOOKUP_DIALECT,
    RuntimeMeasurementLookupDialect,
    RuntimeMeasurementLookupDialectLike,
    resolve_runtime_measurement_lookup_dialect,
)
from openhcs.core.registry_strategies import NominalTypeKeyedStrategyMixin
from openhcs.core.runtime_semantics import (
    MeasurementScope,
    ObjectLabelMeasurementValues,
    dense_object_label_id_domain,
    dense_object_label_present_ids,
    measurement_row_mapping,
)
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
_COLUMNAR_RECORD_ORIENTATION = "records"
_MEASUREMENT_TABLE_CACHE: WeakKeyDictionary[
    RuntimeValueStore,
    dict[tuple[int, str, str | None], tuple[MeasurementTable, ...]],
] = WeakKeyDictionary()
_ROW_SEQUENCE_VALUE_INDEX_CACHE: dict[
    tuple[int, str, tuple[str, ...]],
    "MeasurementRowSequenceFeatureValueIndex | None",
] = {}


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
    def candidates(self) -> tuple[str, ...]:
        return ordered_measurement_feature_candidates(
            self.feature_name,
            dialect=self.dialect,
        )

    def row_value(self, row: object) -> object | None:
        """Return the row value matching this feature query, if present."""
        row_mapping = measurement_row_mapping(row)
        if not self._matches_object(row_mapping):
            return None

        candidates = self.candidates
        if measurement_row_feature_matches(row_mapping, candidates):
            return measurement_row_first_value(row_mapping)

        field_name = matching_measurement_field(row_mapping, candidates)
        if field_name is None:
            return None
        if not measurement_row_source_matches_feature(row_mapping, candidates):
            return None
        return row_mapping[field_name]

    def value_index(
        self,
        measurement_tables: tuple[MeasurementTable, ...],
    ) -> tuple[dict[int, float], list[float]]:
        """Return object-id and positional values for this feature."""
        value_index = self.optional_value_index(measurement_tables)
        if value_index is None:
            raise ValueError(
                f"Could not resolve measurement feature {self.feature_name!r}."
            )
        return value_index

    def optional_value_index(
        self,
        measurement_tables: tuple[MeasurementTable, ...],
    ) -> tuple[dict[int, float], list[float]] | None:
        """Return feature values when present, otherwise ``None``."""
        values_by_label: dict[int, float] = {}
        positional_values: list[float] = []
        for table in measurement_tables:
            columnar_index = _columnar_measurement_value_index(table, self)
            if columnar_index is not None:
                columnar_values_by_label, columnar_positional_values = columnar_index
                values_by_label.update(columnar_values_by_label)
                positional_values.extend(columnar_positional_values)
                continue
            row_sequence_index = _row_sequence_measurement_value_index(table, self)
            if row_sequence_index is not None:
                row_values_by_label, row_positional_values = row_sequence_index
                values_by_label.update(row_values_by_label)
                positional_values.extend(row_positional_values)
                continue
            if _is_wide_row_sequence_measurement_table(table):
                continue
            for row in measurement_rows((table,)):
                value = self.row_value(row)
                if value is None:
                    continue
                object_label = measurement_object_label(
                    measurement_row_mapping(row),
                    object_id_field=measurement_table_object_id_field(table),
                )
                if object_label is None:
                    positional_values.append(float(value))
                    continue
                values_by_label[object_label] = float(value)
        if not values_by_label and not positional_values:
            return None
        return values_by_label, positional_values

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
        return (
            self.object_name is None
            or row_object_name is None
            or row_object_name == self.object_name
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
        measurement_tables_by_object: Mapping[str, tuple[MeasurementTable, ...]],
    ) -> Mapping[str, tuple[dict[int, float], list[float]]]:
        """Return value indexes keyed by object name for this feature."""
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
                    object_index = row_sequence_index.for_object(object_name)
                    if object_index is not None:
                        indexes_by_object[object_name] = indexes_by_object[
                            object_name
                        ].merged(MeasurementFeatureValueIndex.from_query_result(object_index))
                continue

            for object_name in table_object_names:
                object_index = MeasurementFeatureQuery(
                    self.feature_name,
                    object_name=object_name,
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
            raise ValueError(
                f"Could not resolve measurement feature {self.feature_name!r} "
                f"for object(s) {missing_object_names!r}."
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
        value_index: tuple[dict[int, float], list[float]],
    ) -> "MeasurementFeatureValueIndex":
        return cls(dict(value_index[0]), list(value_index[1]))

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

    def as_query_result(self) -> tuple[dict[int, float], list[float]]:
        return self.values_by_label, self.positional_values


@dataclass(frozen=True, slots=True)
class MeasurementRowSequenceFeatureValueIndex:
    """Feature values indexed once per row sequence and projected per object."""

    values_by_object: dict[str | None, MeasurementFeatureValueIndex]

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

        field_names = _row_sequence_field_names(rows, query)
        table_source_image_name = (
            None
            if _measurement_table_declares_object_identity(table, field_names)
            else table.source_image_name
        )
        feature_field = _matching_row_value_field(
            field_names,
            query,
            table_source_image_name=table_source_image_name,
        )
        if feature_field is None:
            return None

        object_id_field = _matching_object_id_field(
            field_names,
            measurement_table_object_id_field(table),
        )
        candidates = query.candidates
        values_by_object: dict[str | None, MeasurementFeatureValueIndex] = {}
        for row in rows:
            row_mapping = measurement_row_mapping(row)
            if not measurement_row_source_matches_feature(row_mapping, candidates):
                continue
            value = row_mapping.get(feature_field)
            if value in (None, ""):
                continue
            if object_id_field is not None and object_id_field not in row_mapping:
                return None
            object_label = (
                None
                if object_id_field is None
                else measurement_object_label(
                    row_mapping,
                    object_id_field=object_id_field,
                )
            )
            object_name = measurement_row_object_name(row_mapping)
            values_by_object.setdefault(
                object_name,
                MeasurementFeatureValueIndex.empty(),
            ).add(object_label, value)

        if not any(index.present for index in values_by_object.values()):
            return None
        return cls(values_by_object)

    def for_object(
        self,
        object_name: str | None,
    ) -> tuple[dict[int, float], list[float]] | None:
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


@dataclass(frozen=True, slots=True)
class MeasurementTableObjectFeatureSemantics:
    """Object and feature declarations carried by one measurement table."""

    object_names: tuple[str, ...]
    feature_names: frozenset[str]

    @classmethod
    def from_table(cls, table: MeasurementTable) -> "MeasurementTableObjectFeatureSemantics":
        rows = measurement_rows((table,))
        return cls(
            object_names=cls._object_names(table, rows),
            feature_names=cls._feature_names(table, rows),
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

        non_feature_fields = {
            MEASUREMENT_OBJECT_NAME_FIELD,
            MEASUREMENT_SOURCE_IMAGE_NAME_FIELD,
            *(str(field_name) for field_name in MEASUREMENT_OBJECT_ID_FIELDS),
            *(field_name for field_name in MEASUREMENT_FEATURE_NAME_FIELDS),
        }
        if table.object_id_field is not None:
            non_feature_fields.add(table.object_id_field)
        return frozenset(
            field.name
            for field in table.fields
            if field.name not in non_feature_fields
        )


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
    def value_index(self) -> tuple[dict[int, float], list[float]]:
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

    indexed_values: tuple[dict[int, float], list[float]]

    @property
    def object_domain(self) -> tuple[int, ...]:
        """Return object IDs materially present in this indexed label plane."""
        return dense_object_label_present_ids(self.labels)

    @property
    def value_index(self) -> tuple[dict[int, float], list[float]]:
        """Return the supplied per-plane measurement index."""
        return self.indexed_values


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
    return tuple(rows)


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


def _columnar_measurement_value_index(
    table: MeasurementTable,
    query: MeasurementFeatureQuery,
) -> tuple[dict[int, float], list[float]] | None:
    """Return a feature vector directly from columnar rows when possible."""
    import numpy as np

    rows = table.rows
    if not isinstance(rows, ColumnarRows):
        return None
    if query.object_name is not None:
        table_object = measurement_table_object_name(table)
        if table_object not in (None, query.object_name):
            return None

    columns = tuple(str(column) for column in rows.columns)
    feature_column = _matching_columnar_feature_column(columns, query)
    if feature_column is None:
        return None

    values = np.asarray(rows[feature_column], dtype=float)
    object_id_field = measurement_table_object_id_field(table)
    if object_id_field is not None and object_id_field in columns:
        object_ids = np.asarray(rows[object_id_field], dtype=np.int64)
        return (
            {
                int(object_id): float(value)
                for object_id, value in zip(object_ids, values, strict=True)
            },
            [],
        )
    return {}, [float(value) for value in values]


def _row_sequence_measurement_value_index(
    table: MeasurementTable,
    query: MeasurementFeatureQuery,
) -> tuple[dict[int, float], list[float]] | None:
    """Return a feature vector directly from homogeneous row sequences."""
    table_rows = table.rows
    if not isinstance(table_rows, ColumnarRows) and isinstance(table_rows, list | tuple):
        cache_key = (
            id(table_rows),
            query.feature_name,
            query.candidates,
        )
        cached = _ROW_SEQUENCE_VALUE_INDEX_CACHE.get(cache_key)
        if cached is not None or cache_key in _ROW_SEQUENCE_VALUE_INDEX_CACHE:
            return None if cached is None else cached.for_object(query.object_name)
        value_index = MeasurementRowSequenceFeatureValueIndex.from_table(table, query)
        _ROW_SEQUENCE_VALUE_INDEX_CACHE[cache_key] = value_index
        return None if value_index is None else value_index.for_object(query.object_name)
    return _uncached_row_sequence_measurement_value_index(table, query)


def _uncached_row_sequence_measurement_value_index(
    table: MeasurementTable,
    query: MeasurementFeatureQuery,
) -> tuple[dict[int, float], list[float]] | None:
    """Return a feature vector directly from homogeneous row sequences."""
    rows = table.rows
    if isinstance(rows, ColumnarRows):
        return None
    if not isinstance(rows, list | tuple) or not rows:
        return None
    if query.object_name is not None:
        table_object = measurement_table_object_name(table)
        if table_object not in (None, query.object_name):
            return None

    field_names = _row_sequence_field_names(rows, query)
    table_source_image_name = (
        None
        if _measurement_table_declares_object_identity(table, field_names)
        else table.source_image_name
    )
    feature_field = _matching_row_value_field(
        field_names,
        query,
        table_source_image_name=table_source_image_name,
    )
    if feature_field is None:
        return None

    object_id_field = _matching_object_id_field(
        field_names,
        measurement_table_object_id_field(table),
    )
    candidates = query.candidates
    values_by_label: dict[int, float] = {}
    positional_values: list[float] = []
    for row in rows:
        row_mapping = measurement_row_mapping(row)
        if not query._matches_object(row_mapping):
            continue
        if not measurement_row_source_matches_feature(row_mapping, candidates):
            continue
        value = row_mapping.get(feature_field)
        if value in (None, ""):
            continue
        if object_id_field is None:
            positional_values.append(float(value))
            continue
        if object_id_field not in row_mapping:
            return None
        object_label = measurement_object_label(
            row_mapping,
            object_id_field=object_id_field,
        )
        if object_label is None:
            positional_values.append(float(value))
            continue
        values_by_label[object_label] = float(value)
    if not values_by_label and not positional_values:
        return None
    return values_by_label, positional_values


def _row_sequence_field_names(
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
    candidates = measurement_feature_candidates(
        query.feature_name,
        dialect=query.dialect,
    )
    found_feature = any(
        normalize_measurement_token(field_name) in candidates
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
            if normalize_measurement_token(normalized_name) in candidates:
                found_feature = True
            if normalized_name in MEASUREMENT_OBJECT_ID_FIELDS:
                found_object_id = True
        if found_feature and found_object_id:
            break
    return tuple(field_names)


def _is_wide_row_sequence_measurement_table(table: MeasurementTable) -> bool:
    """Return whether row fields are direct measurement columns, not long rows."""
    rows = table.rows
    if not isinstance(rows, list | tuple) or not rows:
        return False
    first_row = measurement_row_mapping(rows[0])
    return not any(field in first_row for field in MEASUREMENT_FEATURE_NAME_FIELDS)


def _matching_row_value_field(
    field_names: tuple[str, ...],
    query: MeasurementFeatureQuery,
    *,
    table_source_image_name: str | None,
) -> str | None:
    candidates = query.candidates
    if table_source_image_name is not None:
        normalized_source = normalize_measurement_token(table_source_image_name)
        if (
            normalized_source not in MEASUREMENT_UNQUALIFIED_SOURCE_NAMES
            and normalized_source not in candidates
        ):
            return None
    return matching_measurement_field(
        {field_name: None for field_name in field_names},
        candidates,
    )


def _matching_object_id_field(
    field_names: tuple[str, ...],
    declared_object_id_field: str | None,
) -> str | None:
    if declared_object_id_field is not None and declared_object_id_field in field_names:
        return declared_object_id_field
    for field_name in field_names:
        if field_name in MEASUREMENT_OBJECT_ID_FIELDS:
            return field_name
    return None


def _measurement_table_declares_object_identity(
    table: MeasurementTable,
    field_names: tuple[str, ...],
) -> bool:
    """Return whether table-level image source should not qualify feature fields."""
    return (
        measurement_table_object_name(table) is not None
        or MEASUREMENT_OBJECT_NAME_FIELD in field_names
        or _matching_object_id_field(
            field_names,
            measurement_table_object_id_field(table),
        )
        is not None
    )


def _matching_columnar_feature_column(
    columns: tuple[str, ...],
    query: MeasurementFeatureQuery,
) -> str | None:
    """Return the best column match for a feature query."""
    normalized_feature = normalize_measurement_token(query.feature_name)
    for column in columns:
        if normalize_measurement_token(column) == normalized_feature:
            return column
    candidates = measurement_feature_candidates(
        query.feature_name,
        dialect=query.dialect,
    )
    for column in columns:
        if normalize_measurement_token(column) in candidates:
            return column
    return None


def _columnar_measurement_rows(rows: ColumnarRows) -> tuple[Mapping[str, object], ...]:
    """Return record mappings from a nominal columnar table payload."""
    to_dict = getattr(rows, "to_dict", None)
    if callable(to_dict):
        try:
            records = to_dict(orient=_COLUMNAR_RECORD_ORIENTATION)
        except TypeError:
            records = to_dict(_COLUMNAR_RECORD_ORIENTATION)
        if isinstance(records, list | tuple) and all(
            isinstance(record, Mapping) for record in records
        ):
            return tuple(records)

    itertuples = getattr(rows, "itertuples", None)
    if callable(itertuples):
        columns = tuple(str(column) for column in rows.columns)
        return tuple(
            dict(zip(columns, values, strict=True))
            for values in itertuples(index=False, name=None)
        )

    raise TypeError(
        f"Columnar measurement rows {type(rows).__name__!r} must expose "
        "record mappings."
    )


def normalize_measurement_token(value: object) -> str:
    """Normalize feature/source names for runtime measurement lookup."""
    return _normalize_measurement_text(str(value))


@lru_cache(maxsize=8192)
def _normalize_measurement_text(text: str) -> str:
    """Normalize one measurement token string with bounded process-local reuse."""
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


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
    """Return feature aliases from most specific to least specific."""
    normalized = normalize_measurement_token(feature_name)
    parts = tuple(part for part in normalized.split("_") if part)
    candidates: list[str] = []

    def add(candidate: str) -> None:
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    add(normalized)
    add(normalized.replace("_", ""))
    resolved_dialect = resolve_runtime_measurement_lookup_dialect(dialect)
    dialect_parts = _measurement_dialect_feature_parts(parts, resolved_dialect)
    if dialect_parts != parts:
        dialect_name = "_".join(dialect_parts)
        add(dialect_name)
        add(dialect_name.replace("_", ""))
    if len(parts) >= 2:
        add("_".join(parts[1:]))
        add("".join(parts[1:]))
    if len(parts) >= 3:
        add("_".join(parts[1:-1]))
        add("".join(parts[1:-1]))

    for width in range(len(parts), 1, -1):
        for start in range(0, len(parts) - width + 1):
            stop = start + width
            candidate_parts = parts[start:stop]
            add("_".join(candidate_parts))
            add("".join(candidate_parts))

    for part in parts:
        add(part)
    return tuple(candidates)


def _measurement_dialect_feature_parts(
    parts: tuple[str, ...],
    dialect: RuntimeMeasurementLookupDialect,
) -> tuple[str, ...]:
    """Return dialect-normalized feature parts for runtime artifact lookup."""
    resolved_parts = parts
    for prefix in dialect.category_prefixes:
        if len(resolved_parts) > len(prefix) and resolved_parts[: len(prefix)] == prefix:
            resolved_parts = resolved_parts[len(prefix) :]
            break
    return dialect.feature_part_aliases.get(resolved_parts, resolved_parts)


def matching_measurement_field(
    row: Mapping[str, object],
    candidates: Sequence[str],
) -> str | None:
    """Return the first row field matching the ordered feature alias set."""
    for candidate in candidates:
        for field_name in row:
            normalized = normalize_measurement_token(field_name)
            if candidate in (normalized, normalized.replace("_", "")):
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
    candidates: Sequence[str],
) -> bool:
    """Return whether the row source qualifier is compatible with a feature."""
    source_image_name = measurement_row_source_image_name(row)
    if source_image_name is None:
        return True
    normalized_source = normalize_measurement_token(source_image_name)
    if normalized_source in MEASUREMENT_UNQUALIFIED_SOURCE_NAMES:
        return True
    return normalized_source in candidates


def measurement_value_index(
    measurement_tables: tuple[MeasurementTable, ...],
    feature_name: str,
    *,
    object_name: str | None = None,
    dialect: RuntimeMeasurementLookupDialectLike = (
        CURRENT_RUNTIME_MEASUREMENT_LOOKUP_DIALECT
    ),
) -> tuple[dict[int, float], list[float]]:
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
) -> tuple[dict[int, float], list[float]] | None:
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
    measurement_tables_by_object: Mapping[str, tuple[MeasurementTable, ...]],
    feature_name: str,
    *,
    object_names: Sequence[str],
    dialect: RuntimeMeasurementLookupDialectLike = (
        CURRENT_RUNTIME_MEASUREMENT_LOOKUP_DIALECT
    ),
) -> Mapping[str, tuple[dict[int, float], list[float]]]:
    """Return object-keyed feature indexes for one feature across object domains."""
    return MeasurementObjectFeatureVectorBatchQuery(
        feature_name,
        tuple(object_names),
        dialect=dialect,
    ).value_indexes(measurement_tables_by_object)


def measurement_table_for_slice(
    table: MeasurementTable,
    slice_index: int,
) -> MeasurementTable:
    """Return a measurement table narrowed to one slice when rows declare slices."""
    rows = measurement_rows((table,))
    if not rows:
        return table

    keyed_rows = tuple(
        row
        for row in rows
        if "slice_index" in measurement_row_mapping(row)
    )
    if not keyed_rows:
        return table

    return MeasurementTable(
        name=table.name,
        rows=[
            row
            for row in rows
            if int(measurement_row_mapping(row).get("slice_index", -1))
            == int(slice_index)
        ],
        object_name=table.object_name,
        fields=table.fields,
        object_id_field=table.object_id_field,
        source_image_name=table.source_image_name,
        subject=table.subject,
    )


def measurement_tables_for_slice(
    measurement_tables: tuple[MeasurementTable, ...],
    slice_index: int,
) -> tuple[MeasurementTable, ...]:
    """Return measurement tables narrowed to one slice where row axes permit it."""
    return tuple(
        measurement_table_for_slice(table, slice_index)
        for table in measurement_tables
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
    values_by_slice: Mapping[int, tuple[dict[int, float], list[float]]],
    slice_index: int,
) -> tuple[dict[int, float], list[float]]:
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
) -> dict[int, tuple[dict[int, float], list[float]]] | None:
    """Return per-slice feature indexes without re-scanning tables per plane."""
    query = MeasurementFeatureQuery(
        feature_name,
        object_name=object_name,
        dialect=dialect,
    )
    defaults: tuple[dict[int, float], list[float]] = ({}, [])
    by_slice: dict[int, tuple[dict[int, float], list[float]]] = {}

    for table in measurement_tables:
        if isinstance(table.rows, ColumnarRows):
            return None
        rows = table.rows
        if not isinstance(rows, list | tuple):
            return None
        if not rows:
            continue
        if object_name is not None:
            table_object = measurement_table_object_name(table)
            if table_object not in (None, object_name):
                continue

        row_mappings = tuple(measurement_row_mapping(row) for row in rows)
        has_slice_rows = any("slice_index" in row for row in row_mappings)
        if not has_slice_rows:
            row_index = _row_sequence_measurement_value_index(table, query)
            if row_index is not None:
                _merge_measurement_value_index(defaults, row_index)
                continue
            if _is_wide_row_sequence_measurement_table(table):
                continue
            return None

        field_names = _row_sequence_field_names(rows, query)
        table_source_image_name = (
            None
            if _measurement_table_declares_object_identity(table, field_names)
            else table.source_image_name
        )
        feature_field = _matching_row_value_field(
            field_names,
            query,
            table_source_image_name=table_source_image_name,
        )
        if feature_field is None:
            continue
        object_id_field = _matching_object_id_field(
            field_names,
            measurement_table_object_id_field(table),
        )
        candidates = query.candidates

        for row_mapping in row_mappings:
            if "slice_index" not in row_mapping:
                continue
            if not query._matches_object(row_mapping):
                continue
            if not measurement_row_source_matches_feature(row_mapping, candidates):
                continue
            value = row_mapping.get(feature_field)
            if value in (None, ""):
                continue
            slice_index = int(row_mapping["slice_index"])
            target = by_slice.setdefault(slice_index, ({}, []))
            if object_id_field is None:
                target[1].append(float(value))
                continue
            if object_id_field not in row_mapping:
                return None
            object_label = measurement_object_label(
                row_mapping,
                object_id_field=object_id_field,
            )
            if object_label is None:
                target[1].append(float(value))
                continue
            target[0][object_label] = float(value)

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
    target: tuple[dict[int, float], list[float]],
    source: tuple[dict[int, float], list[float]],
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

    def annotate_rows(self, rows: Sequence[object]) -> list[object]:
        """Attach ownership qualifiers, copying only non-mutable row values."""
        qualifiers = self.qualifiers
        if not qualifiers:
            return list(rows)
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


def _label_planes_are_empty(label_planes: tuple[Any, ...]) -> bool:
    import numpy as np

    return all(not np.any(label_plane > 0) for label_plane in label_planes)
