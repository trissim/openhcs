"""Table snapshot records for runtime equivalence."""

from __future__ import annotations

import csv
import hashlib
import pickle
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openhcs.core.equivalence.cells import (
    measurement_table_cell_payload,
    runtime_cell_signature,
    update_measurement_table_cell_hash,
)
from openhcs.core.equivalence.policy import (
    RuntimeEquivalencePolicy,
    normalize_runtime_identifier,
)
from openhcs.core.measurement_row_materialization import (
    MEASUREMENT_FEATURE_NAME_FIELDS,
    MEASUREMENT_OBJECT_ID_FIELDS,
    MEASUREMENT_OBJECT_NAME_FIELD,
    MEASUREMENT_OBJECT_ROW_IDENTITY_FIELD,
    MEASUREMENT_SOURCE_IMAGE_NAME_FIELD,
    MEASUREMENT_VALUE_FIELDS,
    iter_measurement_rows,
    measurement_row_has_long_form_measurement_fields,
    measurement_row_has_object_identity,
    measurement_row_identity_role,
)
from openhcs.core.measurement_row_materialization import measurement_table_axis_values
from openhcs.core.runtime_semantics import (
    FieldSpec,
    MeasurementObjectRowIdentity,
    MeasurementRowAxisField,
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
        MEASUREMENT_OBJECT_ROW_IDENTITY_FIELD,
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
DEFAULT_MEASUREMENT_TABLE_PADDING_GROUP = "measurements"


def measurement_table_padding_group(table_name: str) -> str:
    """Return the semantic padding group for one measurement table name."""
    normalized_name = normalize_runtime_identifier(table_name)
    if normalized_name:
        return normalized_name
    return DEFAULT_MEASUREMENT_TABLE_PADDING_GROUP


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
    fields: tuple[FieldSpec, ...]
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
            tuple(table.fields),
            rows,
        )


@dataclass(slots=True)
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
        self.path = path
        self.header = header
        self.rows = rows
        self.column_context = column_context

    @property
    def schema_key(self) -> tuple[str, ...]:
        """File-order-independent schema identity for this table."""
        return tuple(sorted(self.header))

    def content_key(
        self,
        policy: RuntimeEquivalencePolicy,
    ) -> tuple[tuple[tuple[str, str], ...], ...]:
        """File-order-independent row identity for this table."""
        columns = self.schema_key
        indexes = {column: self.header.index(column) for column in self.header}
        return tuple(sorted(
            tuple(
                runtime_cell_signature(row[indexes[column]], policy).sort_key
                for column in columns
            )
            for row in self.rows
        ))


def aggregate_measurement_table_key(
    table: MeasurementTable,
) -> RuntimeMeasurementTableIdentity | None:
    """Return a nominal key for duplicate full-axis measurement tables.

    Grouped execution can materialize an already-aggregated measurement table
    once per group key. Row-local image identity fields carry the actual
    measurement scope, so repeated materializations of the same semantic table
    should only contribute once when one table spans multiple row identities.
    Plane-local aggregate rows remain count-preserving when each runtime record
    carries only one row identity.
    """
    normalized_field_cache: dict[str, str] = {}

    def normalized_field(field_name: str) -> str:
        cached = normalized_field_cache.get(field_name)
        if cached is None:
            cached = normalize_runtime_identifier(field_name)
            normalized_field_cache[field_name] = cached
        return cached

    row_identities: set[tuple[tuple[str, object], ...]] = set()
    for row in iter_measurement_rows((table,)):
        row_mapping = measurement_row_mapping(row)
        row_identity_values: list[tuple[str, object]] = []
        for field_name, value in row_mapping.items():
            normalized_field_name = normalized_field(str(field_name))
            if normalized_field_name in RUNTIME_AGGREGATE_TABLE_IDENTITY_FIELDS:
                row_identity_values.append(
                    (
                        normalized_field_name,
                        measurement_table_cell_payload(value),
                    )
                )
        if row_identity_values:
            row_identities.add(tuple(sorted(row_identity_values)))

    if len(row_identities) <= 1:
        return None
    return RuntimeMeasurementTableIdentity.from_table_rows(
        table,
        iter_measurement_rows((table,)),
    )


def exact_measurement_table_key(
    table: MeasurementTable,
) -> RuntimeMeasurementTableIdentity:
    """Return an exact semantic key for duplicate runtime measurement tables."""
    return RuntimeMeasurementTableIdentity.from_table_rows(
        table,
        iter_measurement_rows((table,)),
    )


RuntimeMeasurementObjectSubtableSet = set[RuntimeMeasurementTableIdentity]


def dedupe_runtime_measurement_table_object_subtable(
    table: MeasurementTable,
    seen_object_subtables: RuntimeMeasurementObjectSubtableSet,
) -> MeasurementTable:
    """Drop duplicate object-identity subtable rows while preserving non-object rows."""
    non_object_rows: list[object] = []
    object_row_fingerprint = RuntimeMeasurementRowFingerprintBuilder()
    object_row_count = 0
    total_row_count = 0
    for row in iter_measurement_rows((table,)):
        total_row_count += 1
        row_mapping = measurement_row_mapping(row)
        if measurement_row_has_object_identity(row_mapping):
            object_row_fingerprint.add_row_mapping(row_mapping)
            object_row_count += 1
        else:
            non_object_rows.append(row)
    if total_row_count == 0 or object_row_count == 0:
        return table

    subtable_key = RuntimeMeasurementTableIdentity.from_table_row_fingerprint(
        table,
        object_row_fingerprint.finish(),
    )
    if subtable_key not in seen_object_subtables:
        seen_object_subtables.add(subtable_key)
        return table
    if not non_object_rows:
        return MeasurementTable(
            name=table.name,
            rows=(),
            object_name=table.object_name,
            fields=table.fields,
            object_id_field=table.object_id_field,
            source_image_name=table.source_image_name,
            subject=table.subject,
            source_provenance=table.source_provenance,
        )
    return MeasurementTable(
        name=table.name,
        rows=tuple(non_object_rows),
        object_name=table.object_name,
        fields=table.fields,
        object_id_field=table.object_id_field,
        source_image_name=table.source_image_name,
        subject=table.subject,
        source_provenance=table.source_provenance,
    )


def aggregate_measurement_table_semantic_key(
    table: MeasurementTable,
) -> RuntimeMeasurementTableIdentity | None:
    """Return duplicate identity after removing aggregate-table transport fields."""
    if aggregate_measurement_table_key(table) is None:
        return None
    builder = RuntimeMeasurementRowFingerprintBuilder()
    for row in iter_measurement_rows((table,)):
        builder.add_row_mapping(
            runtime_aggregate_measurement_row_semantic_mapping(
                measurement_row_mapping(row)
            )
        )
    return RuntimeMeasurementTableIdentity.from_table_row_fingerprint(
        table,
        builder.finish(),
    )


def dedupe_runtime_measurement_table_aggregate_rows(
    table: MeasurementTable,
) -> MeasurementTable:
    """Remove duplicate aggregate rows after transport identity fields are ignored."""
    if aggregate_measurement_table_key(table) is None:
        return table
    rows = tuple(iter_measurement_rows((table,)))
    seen_rows: set[tuple[tuple[str, object], ...]] = set()
    deduped_rows: list[object] = []
    semantic_rows: list[tuple[tuple[str, object], ...]] = []
    for row in rows:
        row_mapping = measurement_row_mapping(row)
        if measurement_row_has_object_identity(row_mapping):
            deduped_rows.append(row)
            continue
        semantic_row = aggregate_measurement_row_semantic_key(
            runtime_aggregate_measurement_row_semantic_mapping(
                row_mapping,
                table=table,
                preserve_multi_value_axes=True,
            )
        )
        semantic_rows.append(semantic_row)
        if semantic_row in seen_rows:
            continue
        seen_rows.add(semantic_row)
        deduped_rows.append(row)
    if len(set(semantic_rows)) <= 1:
        return table
    if len(deduped_rows) == len(rows):
        return table
    return MeasurementTable(
        name=table.name,
        rows=tuple(deduped_rows),
        object_name=table.object_name,
        fields=table.fields,
        object_id_field=table.object_id_field,
        source_image_name=table.source_image_name,
        subject=table.subject,
        source_provenance=table.source_provenance,
    )


def aggregate_measurement_row_semantic_key(
    row_mapping: Mapping[str, object],
) -> tuple[tuple[str, object], ...]:
    """Return a hashable aggregate-row key using table cell semantics."""
    return tuple(
        sorted(
            (
                field_name,
                measurement_table_cell_payload(value),
            )
            for field_name, value in row_mapping.items()
        )
    )


def runtime_aggregate_measurement_row_semantic_mapping(
    row_mapping: Mapping[str, object],
    *,
    table: MeasurementTable | None = None,
    preserve_multi_value_axes: bool = False,
) -> dict[str, object]:
    """Return aggregate-row identity after removing transport-only fields."""
    excluded_fields = RuntimeAggregateMeasurementRowIdentityFields(
        row_mapping,
        table=table,
        preserve_multi_value_axes=preserve_multi_value_axes,
    ).transport_identity_fields()
    return {
        field_name: value
        for field_name, value in row_mapping.items()
        if normalize_runtime_identifier(str(field_name)) not in excluded_fields
    }


@dataclass(frozen=True, slots=True)
class RuntimeAggregateMeasurementRowIdentityFields:
    """Fields excluded from duplicate aggregate-table row identity."""

    row: Mapping[str, object]
    table: MeasurementTable | None = None
    preserve_multi_value_axes: bool = False

    def transport_identity_fields(self) -> frozenset[str]:
        if measurement_row_has_long_form_measurement_fields(self.row):
            return frozenset()
        if measurement_row_identity_role(self.row) is MeasurementObjectRowIdentity.ROW_SEQUENCE:
            return self.row_sequence_transport_identity_fields()
        if self.preserve_multi_value_axes and self.table is not None:
            return self.table_transport_identity_fields()
        return RUNTIME_AGGREGATE_TABLE_IDENTITY_FIELDS

    def table_transport_identity_fields(self) -> frozenset[str]:
        if self.table is None:
            return RUNTIME_AGGREGATE_TABLE_IDENTITY_FIELDS
        row_axes = RuntimeAggregateMeasurementTableRowAxes(
            self.table,
            self.row,
        ).multi_value_axis_fields()
        return RUNTIME_AGGREGATE_TABLE_IDENTITY_FIELDS - row_axes

    @staticmethod
    def row_sequence_transport_identity_fields() -> frozenset[str]:
        return {
            field_name
            for field_name in RUNTIME_AGGREGATE_TABLE_IDENTITY_FIELDS
            if field_name != MeasurementRowAxisField.SLICE_INDEX.value
        }


@dataclass(frozen=True, slots=True)
class RuntimeAggregateMeasurementTableRowAxes:
    """Row-axis fields that identify distinct rows within one aggregate table."""

    table: MeasurementTable
    row: Mapping[str, object]

    def multi_value_axis_fields(self) -> frozenset[str]:
        if not measurement_row_has_object_identity(self.row):
            return frozenset()
        return frozenset(
            axis.value
            for axis in (
                MeasurementRowAxisField.IMAGE_NUMBER,
                MeasurementRowAxisField.SLICE_INDEX,
            )
            if axis.value in self.row
            and len(measurement_table_axis_values(self.table, axis)) > 1
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
