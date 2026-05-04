"""Table snapshot records for runtime equivalence."""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from nominal_refactor_advisor.collection_algebra import sorted_tuple

from openhcs.core.equivalence.cells import runtime_cell_signature
from openhcs.core.equivalence.policy import (
    RuntimeEquivalencePolicy,
    normalize_runtime_identifier,
)
from openhcs.core.runtime_artifact_queries import (
    MEASUREMENT_OBJECT_ID_FIELDS,
    MEASUREMENT_OBJECT_NAME_FIELD,
    MEASUREMENT_SOURCE_IMAGE_NAME_FIELD,
)

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
