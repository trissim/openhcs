"""Table snapshot records for runtime equivalence."""

from __future__ import annotations

import csv
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from nominal_refactor_advisor.collection_algebra import sorted_tuple

from openhcs.core.equivalence.arrays import canonical_scalar, semantic_array_payload
from openhcs.core.equivalence.cells import runtime_cell_signature
from openhcs.core.equivalence.policy import (
    RuntimeEquivalencePolicy,
    normalize_runtime_identifier,
)
from openhcs.core.runtime_artifact_queries import (
    MEASUREMENT_FEATURE_NAME_FIELDS,
    MEASUREMENT_OBJECT_ID_FIELDS,
    MEASUREMENT_OBJECT_NAME_FIELD,
    MEASUREMENT_SOURCE_IMAGE_NAME_FIELD,
    MEASUREMENT_VALUE_FIELDS,
    measurement_row_mapping,
    measurement_rows,
)
from openhcs.core.runtime_values import MeasurementTable

MEASUREMENT_IDENTITY_FIELDS = frozenset(
    {
        "image_number",
        "image_id",
        "slice_index",
        *MEASUREMENT_OBJECT_ID_FIELDS,
        MEASUREMENT_OBJECT_NAME_FIELD,
        MEASUREMENT_SOURCE_IMAGE_NAME_FIELD,
        "group_key",
        "number_object_number",
    }
)
CSV_HEADER_CONTEXT_STOPWORDS = frozenset(
    {
        "image",
        "object",
        "objects",
        "measurement",
        "measurements",
    }
)
RUNTIME_AGGREGATE_TABLE_IDENTITY_FIELDS = frozenset(
    {"image_id", "image_number", "slice_index"}
)


@dataclass(frozen=True, slots=True)
class RuntimeTableSnapshot:
    """Semantic snapshot of one exported runtime table."""

    path: Path
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    column_context: tuple[str | None, ...] = ()

    @classmethod
    def from_csv(cls, path: Path) -> "RuntimeTableSnapshot":
        """Read a CSV export into a semantic table snapshot."""
        with Path(path).open(newline="") as handle:
            header, rows, column_context = read_semantic_csv_table(csv.reader(handle))
        return cls(
            path=Path(path),
            header=header,
            rows=rows,
            column_context=column_context,
        )

    def __post_init__(self) -> None:
        path = Path(self.path)
        header = tuple(str(column).strip() for column in self.header)
        if not header:
            raise ValueError(f"Runtime table {path} has no header.")
        column_context = tuple(
            None if value is None or not str(value).strip() else str(value).strip()
            for value in self.column_context
        )
        if column_context and len(column_context) != len(header):
            raise ValueError(
                f"Runtime table {path} column context width "
                f"{len(column_context)} does not match header width {len(header)}."
            )
        duplicate_headers = duplicate_values(header)
        if duplicate_headers:
            raise ValueError(
                f"Runtime table {path} has duplicate headers "
                f"{duplicate_headers!r}."
            )
        rows = tuple(tuple(str(value).strip() for value in row) for row in self.rows)
        malformed_rows = tuple(
            index
            for index, row in enumerate(rows, start=1)
            if len(row) != len(header)
        )
        if malformed_rows:
            raise ValueError(
                f"Runtime table {path} rows do not match header width at "
                f"data rows {malformed_rows!r}."
            )
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "header", header)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "column_context", column_context)

    @property
    def schema_key(self) -> tuple[str, ...]:
        """File-order-independent schema identity for this table."""
        return sorted_tuple(self.header)

    def content_key(
        self,
        policy: RuntimeEquivalencePolicy,
    ) -> tuple[tuple[tuple[str, str], ...], ...]:
        """File-order-independent row identity for this table."""
        columns = self.schema_key
        indexes = {column: self.header.index(column) for column in self.header}
        return sorted_tuple(
            tuple(
                runtime_cell_signature(row[indexes[column]], policy).sort_key
                for column in columns
            )
            for row in self.rows
        )


def aggregate_measurement_table_key(
    table: MeasurementTable,
) -> tuple[object, ...] | None:
    """Return a semantic key for duplicate full-axis measurement tables.

    Grouped execution can materialize an already-aggregated measurement table
    once per group key. Row-local image identity fields carry the actual
    measurement scope, so exact duplicate full-axis tables should only
    contribute once. Group-local tables remain count-preserving.
    """
    rows = tuple(measurement_rows((table,)))
    if not rows:
        return None

    row_mappings = tuple(measurement_row_mapping(row) for row in rows)
    normalized_field_cache: dict[str, str] = {}

    def normalized_field(field_name: str) -> str:
        cached = normalized_field_cache.get(field_name)
        if cached is None:
            cached = normalize_runtime_identifier(field_name)
            normalized_field_cache[field_name] = cached
        return cached

    row_identity_values: set[tuple[str, object]] = set()
    for row_mapping in row_mappings:
        for field_name, value in row_mapping.items():
            normalized_field_name = normalized_field(str(field_name))
            if normalized_field_name in RUNTIME_AGGREGATE_TABLE_IDENTITY_FIELDS:
                row_identity_values.add(
                    (
                        normalized_field_name,
                        measurement_table_cell_payload(value),
                    )
                )
        if len(row_identity_values) > 1:
            break

    if len(row_identity_values) <= 1:
        return None

    row_payloads: list[tuple[tuple[str, object], ...]] = []
    for row_mapping in row_mappings:
        row_payloads.append(
            tuple(
                (
                    normalized_field(str(field_name)),
                    measurement_table_cell_payload(value),
                )
                for field_name, value in row_mapping.items()
            )
        )

    field_payloads = tuple(
        (field.name, field.dtype, field.required)
        for field in table.fields
    )
    return (
        table.name,
        repr(table.subject),
        table.object_name,
        table.object_id_field,
        table.source_image_name,
        field_payloads,
        tuple(row_payloads),
    )


def measurement_table_cell_payload(value: object) -> object:
    """Return a hashable exact payload for measurement-table dedupe."""
    value = canonical_scalar(value)
    array_payload = semantic_array_payload(value)
    if array_payload is not None:
        return array_payload
    if value is None:
        return None
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float) and math.isnan(value):
        return ("float", "nan")
    if isinstance(value, float):
        return ("float", repr(value))
    if isinstance(value, Mapping):
        return (
            "mapping",
            tuple(
                (
                    measurement_table_cell_payload(key),
                    measurement_table_cell_payload(nested_value),
                )
                for key, nested_value in value.items()
            ),
        )
    if isinstance(value, (tuple, list)):
        return (
            type(value).__name__,
            tuple(measurement_table_cell_payload(item) for item in value),
        )
    return (type(value).__name__, repr(value))


def is_static_wide_measurement_table(
    row: Mapping[str, object],
    qualifier_field_names: frozenset[str],
) -> bool:
    """Return whether a wide table has only static feature columns."""
    normalized_fields = {normalize_runtime_identifier(field_name) for field_name in row}
    if normalized_fields & frozenset(MEASUREMENT_FEATURE_NAME_FIELDS):
        return False
    if normalized_fields & frozenset(MEASUREMENT_VALUE_FIELDS):
        return False
    if normalized_fields & qualifier_field_names:
        return False
    return (
        MEASUREMENT_OBJECT_NAME_FIELD not in normalized_fields
        and MEASUREMENT_SOURCE_IMAGE_NAME_FIELD not in normalized_fields
    )


def is_wide_measurement_table(row: Mapping[str, object]) -> bool:
    """Return whether a table encodes measurements as feature columns."""
    normalized_fields = {normalize_runtime_identifier(field_name) for field_name in row}
    if normalized_fields & frozenset(MEASUREMENT_FEATURE_NAME_FIELDS):
        return False
    if normalized_fields & frozenset(MEASUREMENT_VALUE_FIELDS):
        return False
    return True


def read_semantic_csv_table(
    rows: Iterable[tuple[str, ...] | list[str]],
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...], tuple[str | None, ...]]:
    all_rows = tuple(tuple(row) for row in rows)
    for index, row in enumerate(all_rows):
        header = tuple(str(column).strip() for column in row)
        if index + 1 < len(all_rows):
            next_header = tuple(str(column).strip() for column in all_rows[index + 1])
            if (
                _is_contextual_semantic_csv_table_header(header, next_header)
                and len(header) == len(next_header)
            ):
                return (
                    _disambiguate_contextual_csv_header(next_header, header),
                    all_rows[index + 2 :],
                    _semantic_csv_column_context(header),
                )
        if _is_semantic_csv_header(header):
            return header, all_rows[index + 1 :], ()
        if _is_contextual_semantic_csv_header(header) and index > 0:
            context = tuple(str(column).strip() for column in all_rows[index - 1])
            if len(context) == len(header):
                return (
                    _disambiguate_contextual_csv_header(header, context),
                    all_rows[index + 1 :],
                    _semantic_csv_column_context(context),
                )
        if _is_contextual_semantic_csv_header(header):
            return _ensure_unique_header(header), all_rows[index + 1 :], ()
    return (), (), ()


def duplicate_values(values: tuple[str, ...]) -> tuple[str, ...]:
    """Return duplicate values in first-observed order."""
    counts = Counter(values)
    return tuple(value for value, count in counts.items() if count > 1)


def _semantic_csv_column_context(
    context: tuple[str, ...],
) -> tuple[str | None, ...]:
    return tuple(str(value).strip() or None for value in context)


def _is_semantic_csv_header(header: tuple[str, ...]) -> bool:
    if not header:
        return False
    if any(not column for column in header):
        return False
    return not duplicate_values(header)


def _is_contextual_semantic_csv_header(header: tuple[str, ...]) -> bool:
    if not header:
        return False
    if any(not column for column in header):
        return False
    if not duplicate_values(header):
        return False
    normalized_fields = {normalize_runtime_identifier(column) for column in header}
    return bool(normalized_fields & MEASUREMENT_IDENTITY_FIELDS)


def _is_contextual_semantic_csv_table_header(
    context: tuple[str, ...],
    header: tuple[str, ...],
) -> bool:
    """Return whether adjacent CSV rows encode context plus measurement header."""
    if _is_contextual_semantic_csv_header(header):
        return True
    if not _is_semantic_csv_header(header):
        return False
    if len(context) != len(header):
        return False
    normalized_context = tuple(normalize_runtime_identifier(column) for column in context)
    normalized_header = tuple(normalize_runtime_identifier(column) for column in header)
    if normalized_context == normalized_header:
        return False
    if not (frozenset(normalized_header) & MEASUREMENT_IDENTITY_FIELDS):
        return False
    if duplicate_values(normalized_context):
        return True
    return bool(frozenset(normalized_context) & CSV_HEADER_CONTEXT_STOPWORDS)


def _disambiguate_contextual_csv_header(
    header: tuple[str, ...],
    context: tuple[str, ...],
) -> tuple[str, ...]:
    duplicates = frozenset(duplicate_values(header))
    disambiguated = tuple(
        _contextual_csv_header_name(column, context_value, index, duplicates)
        for index, (column, context_value) in enumerate(zip(header, context, strict=True))
    )
    return _ensure_unique_header(disambiguated)


def _contextual_csv_header_name(
    column: str,
    context_value: str,
    index: int,
    duplicates: frozenset[str],
) -> str:
    if column not in duplicates:
        return column
    normalized_context = normalize_runtime_identifier(context_value)
    if normalized_context and normalized_context not in CSV_HEADER_CONTEXT_STOPWORDS:
        return f"{column}_{normalized_context}"
    return f"{column}_{index + 1}"


def _ensure_unique_header(header: tuple[str, ...]) -> tuple[str, ...]:
    counts: Counter[str] = Counter()
    unique: list[str] = []
    for index, column in enumerate(header, start=1):
        counts[column] += 1
        if counts[column] == 1:
            unique.append(column)
            continue
        unique.append(f"{column}_{index}")
    return tuple(unique)
