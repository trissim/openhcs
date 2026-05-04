"""Semantic queries over typed OpenHCS runtime artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from functools import lru_cache
import re
from typing import Any

from openhcs.core.artifacts import ArtifactKind
from openhcs.core.runtime_semantics import MeasurementScope
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
                f"'{self.axis_id}': {records!r}."
            )
        return records[0]


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

    def __post_init__(self) -> None:
        if not self.feature_name:
            raise ValueError("MeasurementFeatureQuery.feature_name cannot be empty.")
        if self.object_name == "":
            raise ValueError("MeasurementFeatureQuery.object_name cannot be empty.")

    @property
    def candidates(self) -> frozenset[str]:
        return measurement_feature_candidates(self.feature_name)

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
            raise ValueError(
                f"Could not resolve measurement feature {self.feature_name!r}."
            )
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


def runtime_measurement_tables(
    context: RuntimeArtifactQueryContext,
) -> tuple[MeasurementTable, ...]:
    """Return all measurement tables in a runtime query context."""
    return tuple(
        MeasurementTable.from_runtime_value(record.value)
        for record in context.find(kind=ArtifactKind.MEASUREMENTS)
    )


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


def measurement_rows(
    measurement_tables: tuple[MeasurementTable, ...],
) -> tuple[object, ...]:
    """Flatten row payloads from measurement tables."""
    rows: list[object] = []
    for table in measurement_tables:
        table_rows = table.rows
        if isinstance(table_rows, ColumnarRows):
            rows.extend(_columnar_measurement_rows(table_rows))
            continue
        if isinstance(table_rows, list | tuple):
            rows.extend(table_rows)
            continue
        rows.append(table_rows)
    return tuple(rows)


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
    feature_column = _matching_columnar_feature_column(columns, query.feature_name)
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
    rows = table.rows
    if isinstance(rows, ColumnarRows):
        return None
    if not isinstance(rows, list | tuple) or not rows:
        return None
    if query.object_name is not None:
        table_object = measurement_table_object_name(table)
        if table_object not in (None, query.object_name):
            return None

    field_names = _row_sequence_field_names(rows, query.feature_name)
    feature_field = _matching_row_value_field(
        field_names,
        query.feature_name,
        table_source_image_name=(
            None if query.object_name is not None else table.source_image_name
        ),
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
    feature_name: str,
) -> tuple[str, ...]:
    """Return field names for homogeneous or heterogeneous row sequences."""
    first_row = measurement_row_mapping(rows[0])
    first_row_names = tuple(str(field_name) for field_name in first_row)
    if any(field in first_row for field in MEASUREMENT_FEATURE_NAME_FIELDS):
        return first_row_names

    field_names: list[str] = []
    seen: set[str] = set(first_row_names)
    field_names.extend(first_row_names)
    candidates = measurement_feature_candidates(feature_name)
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
    feature_name: str,
    *,
    table_source_image_name: str | None,
) -> str | None:
    candidates = measurement_feature_candidates(feature_name)
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


def _matching_columnar_feature_column(
    columns: tuple[str, ...],
    feature_name: str,
) -> str | None:
    """Return the best column match for a feature query."""
    normalized_feature = normalize_measurement_token(feature_name)
    for column in columns:
        if normalize_measurement_token(column) == normalized_feature:
            return column
    candidates = measurement_feature_candidates(feature_name)
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


def measurement_row_mapping(row: object) -> Mapping[str, object]:
    """Return a mapping view for a supported measurement row payload."""
    if isinstance(row, Mapping):
        return row
    if is_dataclass(row):
        return {field.name: getattr(row, field.name) for field in fields(row)}
    try:
        return vars(row)
    except TypeError as exc:
        raise TypeError(
            f"Unsupported measurement row type {type(row).__name__}."
        ) from exc


def normalize_measurement_token(value: object) -> str:
    """Normalize feature/source names for runtime measurement lookup."""
    return _normalize_measurement_text(str(value))


@lru_cache(maxsize=8192)
def _normalize_measurement_text(text: str) -> str:
    """Normalize one measurement token string with bounded process-local reuse."""
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def measurement_feature_candidates(feature_name: str) -> frozenset[str]:
    """Return normalized feature aliases accepted for row/field lookup."""
    normalized = normalize_measurement_token(feature_name)
    parts = tuple(part for part in normalized.split("_") if part)
    candidates = {normalized}
    if len(parts) >= 2:
        candidates.add("_".join(parts[1:]))
        candidates.add(parts[-1])
    if len(parts) >= 3:
        candidates.add("_".join(parts[1:-1]))
    for start in range(len(parts)):
        for stop in range(start + 2, len(parts) + 1):
            candidates.add("_".join(parts[start:stop]))
    return frozenset(candidates)


def matching_measurement_field(
    row: Mapping[str, object],
    candidates: frozenset[str],
) -> str | None:
    """Return the first row field whose name matches one feature candidate."""
    for field_name in row:
        if normalize_measurement_token(field_name) in candidates:
            return field_name
    return None


def measurement_row_feature_matches(
    row: Mapping[str, object],
    candidates: frozenset[str],
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
    candidates: frozenset[str],
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
) -> tuple[dict[int, float], list[float]]:
    """Return object-id and positional values for one feature."""
    return MeasurementFeatureQuery(
        feature_name,
        object_name=object_name,
    ).value_index(measurement_tables)


def measurement_scalar_value_for_feature(
    measurement_tables: tuple[MeasurementTable, ...],
    feature_name: str,
    *,
    object_name: str | None = None,
) -> float:
    """Return exactly one scalar measurement value for one feature."""
    return MeasurementFeatureQuery(
        feature_name,
        object_name=object_name,
    ).scalar_value(measurement_tables)


def measurement_values_for_feature(
    measurement_tables: tuple[MeasurementTable, ...],
    feature_name: str,
    *,
    object_count: int,
    object_name: str | None = None,
) -> Any:
    """Return object-indexed measurement values for one feature."""
    import numpy as np

    values_by_label, positional_values = measurement_value_index(
        measurement_tables,
        feature_name,
        object_name=object_name,
    )
    if values_by_label:
        return np.array(
            [values_by_label.get(index, np.nan) for index in range(1, object_count + 1)]
        )
    if positional_values:
        return np.array(positional_values[:object_count])
    raise ValueError(f"Could not resolve measurement feature {feature_name!r}.")


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
        )
        if values_by_slice is not None:
            return tuple(
                (
                    np.array([], dtype=float)
                    if not np.any(np.asarray(label_plane) > 0)
                    else _measurement_values_for_label_plane(
                        label_plane,
                        *values_by_slice.get(
                            slice_index,
                            values_by_slice.get(-1, ({}, [])),
                        ),
                        feature_name,
                    )
                )
                for slice_index, label_plane in enumerate(label_planes)
            )
    return tuple(
        _measurement_values_for_label_slice(
            measurement_tables_for_slice(measurement_tables, slice_index),
            feature_name,
            label_plane,
            object_name=object_name,
        )
        for slice_index, label_plane in enumerate(label_planes)
    )


def _measurement_value_indexes_by_slice(
    measurement_tables: tuple[MeasurementTable, ...],
    feature_name: str,
    *,
    object_name: str | None,
) -> dict[int, tuple[dict[int, float], list[float]]] | None:
    """Return per-slice feature indexes without re-scanning tables per plane."""
    query = MeasurementFeatureQuery(feature_name, object_name=object_name)
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

        field_names = _row_sequence_field_names(rows, feature_name)
        feature_field = _matching_row_value_field(
            field_names,
            feature_name,
            table_source_image_name=(
                None if object_name is not None else table.source_image_name
            ),
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


def _label_planes_are_empty(label_planes: tuple[Any, ...]) -> bool:
    import numpy as np

    return all(not np.any(label_plane > 0) for label_plane in label_planes)


def _measurement_values_for_label_plane(
    label_plane: Any,
    values_by_label: Mapping[int, float],
    positional_values: list[float],
    feature_name: str,
) -> Any:
    import numpy as np

    positive_labels = _positive_label_ids(label_plane)
    if values_by_label:
        return np.array(
            [values_by_label.get(label, np.nan) for label in positive_labels]
        )
    if positional_values:
        return np.array(positional_values[: len(positive_labels)])
    raise ValueError(f"Could not resolve measurement feature {feature_name!r}.")


def _positive_label_ids(label_plane: Any) -> tuple[int, ...]:
    """Return present positive dense-label ids without sorting all pixels."""
    import numpy as np

    label_array = np.asarray(label_plane)
    if label_array.size == 0:
        return ()
    integer_labels = label_array.astype(np.int64, copy=False)
    max_label = int(integer_labels.max())
    if max_label <= 0:
        return ()
    if max_label <= integer_labels.size:
        present = np.bincount(integer_labels.ravel(), minlength=max_label + 1) > 0
        return tuple(int(label) for label in np.flatnonzero(present[1:]) + 1)
    return tuple(int(label) for label in np.unique(integer_labels) if int(label) > 0)


def _measurement_values_for_label_slice(
    measurement_tables: tuple[MeasurementTable, ...],
    feature_name: str,
    label_plane: Any,
    *,
    object_name: str | None,
) -> Any:
    import numpy as np

    if not np.any(np.asarray(label_plane) > 0):
        return np.array([], dtype=float)
    values_by_label, positional_values = measurement_value_index(
        measurement_tables,
        feature_name,
        object_name=object_name,
    )
    return _measurement_values_for_label_plane(
        label_plane,
        values_by_label,
        positional_values,
        feature_name,
    )
