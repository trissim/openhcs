"""Table snapshot records for runtime equivalence."""

from __future__ import annotations

import csv
import hashlib
import math
import pickle
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    iter_measurement_rows,
    measurement_row_mapping,
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
class RuntimeMeasurementRowFingerprint:
    """Exact canonical fingerprint for measurement rows without retaining rows."""

    row_count: int
    digest: bytes


@dataclass(slots=True)
class RuntimeMeasurementRowFingerprintBuilder:
    """Incrementally fingerprint canonical row payloads."""

    digest_size: int = 32
    _hash: Any = field(init=False, repr=False)
    _row_count: int = field(init=False, repr=False)
    _normalized_field_cache: dict[str, str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._hash = hashlib.blake2b(digest_size=self.digest_size)
        self._row_count = 0
        self._normalized_field_cache = {}

    def add_row_mapping(self, row_mapping: Mapping[str, object]) -> None:
        self._hash.update(b"row:")
        for field_name, value in row_mapping.items():
            field_payload = ("field", self._normalized_field(str(field_name)))
            self._hash.update(
                pickle.dumps(field_payload, protocol=pickle.HIGHEST_PROTOCOL)
            )
            update_measurement_table_cell_hash(self._hash, value)
        self._hash.update(b":row")
        self._row_count += 1

    def _normalized_field(self, field_name: str) -> str:
        cached = self._normalized_field_cache.get(field_name)
        if cached is None:
            cached = normalize_runtime_identifier(field_name)
            self._normalized_field_cache[field_name] = cached
        return cached

    def finish(self) -> RuntimeMeasurementRowFingerprint:
        return RuntimeMeasurementRowFingerprint(
            row_count=self._row_count,
            digest=self._hash.digest(),
        )


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementTableIdentity:
    """Compact exact identity for runtime measurement table dedupe."""

    name: str | None
    subject: str
    object_name: str | None
    object_id_field: str | None
    source_image_name: str | None
    fields: tuple[tuple[str, object, bool], ...]
    rows: RuntimeMeasurementRowFingerprint

    @classmethod
    def from_table_rows(
        cls,
        table: MeasurementTable,
        rows: Iterable[object],
    ) -> "RuntimeMeasurementTableIdentity":
        builder = RuntimeMeasurementRowFingerprintBuilder()
        for row in rows:
            builder.add_row_mapping(measurement_row_mapping(row))
        return cls.from_table_row_fingerprint(table, builder.finish())

    @classmethod
    def from_table_row_fingerprint(
        cls,
        table: MeasurementTable,
        rows: RuntimeMeasurementRowFingerprint,
    ) -> "RuntimeMeasurementTableIdentity":
        return cls(
            table.name,
            repr(table.subject),
            table.object_name,
            table.object_id_field,
            table.source_image_name,
            tuple(
                (field.name, field.dtype, field.required)
                for field in table.fields
            ),
            rows,
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
) -> RuntimeMeasurementTableIdentity | None:
    """Return a nominal key for duplicate full-axis measurement tables.

    Grouped execution can materialize an already-aggregated measurement table
    once per group key. Row-local image identity fields carry the actual
    measurement scope, so repeated materializations of the same runtime table
    payload should only contribute once. Group-local tables remain
    count-preserving because distinct payload objects are not collapsed.
    """
    normalized_field_cache: dict[str, str] = {}

    def normalized_field(field_name: str) -> str:
        cached = normalized_field_cache.get(field_name)
        if cached is None:
            cached = normalize_runtime_identifier(field_name)
            normalized_field_cache[field_name] = cached
        return cached

    row_identity_values: set[tuple[str, object]] = set()
    row_count = 0
    for row in iter_measurement_rows((table,)):
        row_count += 1
        row_mapping = measurement_row_mapping(row)
        for field_name, value in row_mapping.items():
            normalized_field_name = normalized_field(str(field_name))
            if normalized_field_name in RUNTIME_AGGREGATE_TABLE_IDENTITY_FIELDS:
                row_identity_values.add(
                    (
                        normalized_field_name,
                        measurement_table_cell_payload(value),
                    )
                )
    if row_count == 0:
        return None

    if len(row_identity_values) <= 1:
        return None
    return RuntimeMeasurementTableIdentity.from_table_row_fingerprint(
        table,
        RuntimeMeasurementRowFingerprint(
            row_count=row_count,
            digest=f"runtime-payload:{id(table.rows)}".encode("ascii"),
        ),
    )


def exact_measurement_table_key(
    table: MeasurementTable,
) -> RuntimeMeasurementTableIdentity:
    """Return an exact semantic key for duplicate runtime measurement tables."""
    return RuntimeMeasurementTableIdentity.from_table_rows(
        table,
        iter_measurement_rows((table,)),
    )


def measurement_table_cell_payload(value: object) -> object:
    """Return a hashable exact payload for measurement-table dedupe."""
    value = canonical_scalar(value)
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
    array_payload = semantic_array_payload(value)
    if array_payload is not None:
        return array_payload
    return (type(value).__name__, repr(value))


def update_measurement_table_cell_hash(digest: Any, value: object) -> None:
    """Update an exact cell digest without materializing nested payload trees."""
    value = canonical_scalar(value)
    if value is None:
        digest.update(pickle.dumps(("none",), protocol=pickle.HIGHEST_PROTOCOL))
        return
    if isinstance(value, str):
        digest.update(pickle.dumps(("str", value), protocol=pickle.HIGHEST_PROTOCOL))
        return
    if isinstance(value, bool):
        digest.update(pickle.dumps(("bool", value), protocol=pickle.HIGHEST_PROTOCOL))
        return
    if isinstance(value, int):
        digest.update(pickle.dumps(("int", value), protocol=pickle.HIGHEST_PROTOCOL))
        return
    if isinstance(value, float) and math.isnan(value):
        digest.update(pickle.dumps(("float", "nan"), protocol=pickle.HIGHEST_PROTOCOL))
        return
    if isinstance(value, float):
        digest.update(
            pickle.dumps(("float", repr(value)), protocol=pickle.HIGHEST_PROTOCOL)
        )
        return
    array_payload = semantic_array_payload(value)
    if array_payload is not None:
        digest.update(pickle.dumps(array_payload, protocol=pickle.HIGHEST_PROTOCOL))
        return
    if isinstance(value, Mapping):
        digest.update(
            pickle.dumps(("mapping", len(value)), protocol=pickle.HIGHEST_PROTOCOL)
        )
        for key, nested_value in value.items():
            update_measurement_table_cell_hash(digest, key)
            update_measurement_table_cell_hash(digest, nested_value)
        digest.update(
            pickle.dumps(("mapping_end",), protocol=pickle.HIGHEST_PROTOCOL)
        )
        return
    if isinstance(value, tuple):
        digest.update(
            pickle.dumps(("tuple", len(value)), protocol=pickle.HIGHEST_PROTOCOL)
        )
        for item in value:
            update_measurement_table_cell_hash(digest, item)
        digest.update(pickle.dumps(("tuple_end",), protocol=pickle.HIGHEST_PROTOCOL))
        return
    if isinstance(value, list):
        digest.update(
            pickle.dumps(("list", len(value)), protocol=pickle.HIGHEST_PROTOCOL)
        )
        for item in value:
            update_measurement_table_cell_hash(digest, item)
        digest.update(pickle.dumps(("list_end",), protocol=pickle.HIGHEST_PROTOCOL))
        return
    digest.update(
        pickle.dumps((type(value).__name__, repr(value)), protocol=pickle.HIGHEST_PROTOCOL)
    )


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
