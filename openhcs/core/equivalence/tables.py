"""Table snapshot records for runtime equivalence."""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from openhcs.core.equivalence.cells import (
    RuntimeCellSignature,
    measurement_table_cell_payload,
    runtime_cell_signature,
    runtime_measurement_value_is_present,
)
from openhcs.core.equivalence.policy import (
    DEFAULT_RUNTIME_MEASUREMENT_DIALECT,
    RuntimeEquivalencePolicy,
    normalize_runtime_identifier,
)
from openhcs.core.measurement_row_materialization import (
    MEASUREMENT_SPARSE_CELL,
    MeasurementSparseColumnarRows,
)
from openhcs.core.runtime_tabular_values import (
    FieldSpec,
    MeasurementObjectRowIdentity,
)
from openhcs.core.runtime_measurements import (
    MeasurementRowValueField,
    MeasurementRowAxisField,
    MeasurementScope,
    MeasurementSubject,
)
from openhcs.core.runtime_measurements import (
    MeasurementTable,
)


MEASUREMENT_IDENTITY_FIELDS = frozenset(
    {
        "image_id",
        *DEFAULT_RUNTIME_MEASUREMENT_DIALECT.row_identity_contract.image_identity_fields,
        *MeasurementRowAxisField.object_id_field_names(),
        MeasurementRowAxisField.OBJECT_NAME.value,
        MeasurementRowAxisField.OBJECT_ROW_IDENTITY.value,
        MeasurementRowAxisField.SOURCE_IMAGE_NAME.value,
        "group_key",
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
DEFAULT_MEASUREMENT_TABLE_PADDING_GROUP = "measurements"


def measurement_table_padding_group(table_name: str) -> str:
    """Return the semantic padding group for one measurement table name."""
    normalized_name = normalize_runtime_identifier(table_name)
    if normalized_name:
        return normalized_name
    return DEFAULT_MEASUREMENT_TABLE_PADDING_GROUP


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
        rows = tuple(tuple(str(value).strip() for value in row) for row in self.rows)
        malformed_rows = tuple(
            index for index, row in enumerate(rows, start=1) if len(row) != len(header)
        )
        if malformed_rows:
            raise ValueError(
                f"Runtime table {path} rows do not match header width at "
                f"data rows {malformed_rows!r}."
            )
        semantic_columns = tuple(
            zip(
                column_context or (None,) * len(header),
                header,
                strict=True,
            )
        )
        retained_indexes: list[int] = []
        first_index_by_column: dict[tuple[str | None, str], int] = {}
        conflicting_columns: set[tuple[str | None, str]] = set()
        for index, semantic_column in enumerate(semantic_columns):
            first_index = first_index_by_column.get(semantic_column)
            if first_index is None:
                first_index_by_column[semantic_column] = index
                retained_indexes.append(index)
                continue
            if any(row[first_index] != row[index] for row in rows):
                conflicting_columns.add(semantic_column)
        if conflicting_columns:
            raise ValueError(
                f"Runtime table {path} has conflicting duplicate semantic columns "
                f"{tuple(sorted(conflicting_columns, key=repr))!r}."
            )
        retained = tuple(retained_indexes)
        if len(retained) != len(header):
            header = tuple(header[index] for index in retained)
            rows = tuple(tuple(row[index] for index in retained) for row in rows)
            if column_context:
                column_context = tuple(column_context[index] for index in retained)
        self.path = path
        self.header = header
        self.rows = rows
        self.column_context = column_context

    @property
    def schema_key(self) -> tuple[str, ...]:
        """File-order-independent schema identity for this table."""
        return tuple(
            sorted(
                (field_name if context is None else f"{context}:{field_name}")
                for context, field_name in zip(
                    self.column_context or (None,) * len(self.header),
                    self.header,
                    strict=True,
                )
            )
        )

    def content_key(
        self,
        policy: RuntimeEquivalencePolicy,
    ) -> tuple[tuple[tuple[str, str], ...], ...]:
        """File-order-independent row identity for this table."""
        return tuple(
            sorted(
                tuple(signature.sort_key for signature in row)
                for row in self.row_signatures(policy)
            )
        )

    def row_signatures(
        self,
        policy: RuntimeEquivalencePolicy,
    ) -> tuple[tuple[RuntimeCellSignature, ...], ...]:
        """Return rows projected into semantic column order."""
        columns = tuple(
            sorted(
                (
                    (field_name if context is None else f"{context}:{field_name}"),
                    index,
                )
                for index, (context, field_name) in enumerate(
                    zip(
                        self.column_context or (None,) * len(self.header),
                        self.header,
                        strict=True,
                    )
                )
            )
        )
        return tuple(
            tuple(
                runtime_cell_signature(row[index], policy) for _column, index in columns
            )
            for row in self.rows
        )

    def measurement_tables(
        self,
        dialect=DEFAULT_RUNTIME_MEASUREMENT_DIALECT,
    ) -> tuple[MeasurementTable, ...]:
        """Expose exported columns as ordinary subject-owned measurement tables."""
        if not self.column_context:
            subject = self._subject_for_columns(self.header)
            return (
                MeasurementTable(
                    name=self.path.stem,
                    rows=self._columnar_rows(
                        tuple(range(len(self.header))),
                        subject,
                        dialect,
                    ),
                    subject=subject,
                ),
            )

        image_identity_fields = (
            dialect.row_identity_contract.selected_image_identity_fields(
                frozenset(normalize_runtime_identifier(field) for field in self.header)
            )
        )
        image_identity_indexes = tuple(
            index
            for index, field_name in enumerate(self.header)
            if normalize_runtime_identifier(field_name) in image_identity_fields
        )
        contexts = tuple(
            dict.fromkeys(
                context for context in self.column_context if context is not None
            )
        )
        tables: list[MeasurementTable] = []
        for context in contexts:
            context_indexes = tuple(
                index
                for index, column_context in enumerate(self.column_context)
                if column_context == context
            )
            normalized_context = normalize_runtime_identifier(context)
            if normalized_context == MeasurementScope.IMAGE.value:
                indexes = context_indexes
                subject = MeasurementSubject(MeasurementScope.IMAGE, "Image")
            elif normalized_context == MeasurementScope.EXPERIMENT.value:
                indexes = context_indexes
                subject = MeasurementSubject(MeasurementScope.EXPERIMENT)
            elif normalized_context in CSV_HEADER_CONTEXT_STOPWORDS:
                continue
            else:
                indexes = tuple(
                    dict.fromkeys((*image_identity_indexes, *context_indexes))
                )
                subject = self._object_subject(context, indexes, dialect)
            tables.append(
                MeasurementTable(
                    name=f"{self.path.stem}:{context}",
                    rows=self._columnar_rows(indexes, subject, dialect),
                    subject=subject,
                )
            )
        return tuple(tables)

    def _subject_for_columns(
        self,
        header: tuple[str, ...],
    ) -> MeasurementSubject:
        normalized_header = frozenset(
            normalize_runtime_identifier(field_name) for field_name in header
        )
        normalized_name = normalize_runtime_identifier(self.path.stem)
        if normalized_name == MeasurementScope.EXPERIMENT.value:
            return MeasurementSubject(MeasurementScope.EXPERIMENT)
        if (
            normalized_name != MeasurementScope.IMAGE.value
            and normalized_header
            & frozenset(MeasurementRowAxisField.object_id_field_names())
        ):
            return self._object_subject(
                self.path.stem,
                tuple(range(len(header))),
            )
        return MeasurementSubject(MeasurementScope.IMAGE, "Image")

    def _object_subject(
        self,
        name: str,
        indexes: tuple[int, ...],
        dialect=DEFAULT_RUNTIME_MEASUREMENT_DIALECT,
    ) -> MeasurementSubject:
        normalized_fields = {
            normalize_runtime_identifier(self.header[index]): self.header[index]
            for index in indexes
        }
        selected_object_id = (
            dialect.row_identity_contract.selected_object_identity_field(
                frozenset(normalized_fields)
            )
        )
        return MeasurementSubject(
            MeasurementScope.OBJECT,
            name,
            id_field=(
                None
                if selected_object_id is None
                else normalized_fields[selected_object_id]
            ),
        )

    def _columnar_rows(
        self,
        indexes: tuple[int, ...],
        subject: MeasurementSubject,
        dialect=DEFAULT_RUNTIME_MEASUREMENT_DIALECT,
    ) -> MeasurementSparseColumnarRows:
        selected_header = tuple(self.header[index] for index in indexes)
        duplicate_headers = duplicate_values(selected_header)
        if duplicate_headers:
            raise ValueError(
                f"Runtime table {self.path} subject {subject.name!r} has duplicate "
                f"columns {duplicate_headers!r}."
            )
        normalized_fields = {
            normalize_runtime_identifier(field_name): field_name
            for field_name in selected_header
        }
        image_identity_fields = (
            dialect.row_identity_contract.selected_image_identity_fields(
                frozenset(normalized_fields)
            )
        )
        identity_fields = tuple(
            field_name
            for field_name in selected_header
            if normalize_runtime_identifier(field_name) in image_identity_fields
            or field_name == subject.id_field
        )
        superseded_object_identity_fields = frozenset(
            field_name
            for field_name in selected_header
            if subject.id_field is not None
            and field_name != subject.id_field
            and normalize_runtime_identifier(field_name)
            in dialect.row_identity_contract.object_identity_fields
        )
        projected_header = tuple(
            field_name
            for field_name in selected_header
            if field_name not in superseded_object_identity_fields
        )
        identity_is_complete = (
            subject.scope is not MeasurementScope.OBJECT or subject.id_field is not None
        )
        coalesced_rows: dict[object, dict[str, str]] = {}
        for row_index, row in enumerate(self.rows):
            selected_values = {
                field_name: row[index]
                for field_name, index in zip(
                    selected_header,
                    indexes,
                    strict=True,
                )
            }
            if (
                identity_is_complete
                and identity_fields
                and all(
                    runtime_measurement_value_is_present(selected_values[field_name])
                    for field_name in identity_fields
                )
            ):
                identity: object = tuple(
                    (
                        field_name,
                        measurement_table_cell_payload(selected_values[field_name]),
                    )
                    for field_name in identity_fields
                )
            else:
                identity = row_index
            target = coalesced_rows.setdefault(identity, {})
            for field_name, value in selected_values.items():
                if field_name in superseded_object_identity_fields:
                    continue
                if not runtime_measurement_value_is_present(value):
                    continue
                existing = target.get(field_name)
                if existing is not None and existing != value:
                    raise ValueError(
                        f"Runtime table {self.path} subject {subject.name!r} has "
                        "conflicting observed values for row identity "
                        f"{identity!r}, field {field_name!r}: "
                        f"{existing!r} vs {value!r}."
                    )
                target[field_name] = value
        return MeasurementSparseColumnarRows(
            MappingProxyType(
                {
                    field_name: tuple(
                        row.get(field_name, MEASUREMENT_SPARSE_CELL)
                        for row in coalesced_rows.values()
                    )
                    for field_name in projected_header
                }
            ),
            fields=tuple(
                FieldSpec(field_name, str, required=False)
                for field_name in projected_header
            ),
            object_row_identity=(
                MeasurementObjectRowIdentity.LABEL_ID
                if subject.scope is MeasurementScope.OBJECT
                and subject.id_field is not None
                else None
            ),
        )


def is_wide_measurement_table(row: Mapping[str, object]) -> bool:
    """Return whether a table encodes measurements as feature columns."""
    normalized_fields = {normalize_runtime_identifier(field_name) for field_name in row}
    if normalized_fields & frozenset(
        MeasurementRowAxisField.feature_name_field_names()
    ):
        return False
    if normalized_fields & frozenset(MeasurementRowValueField.field_names()):
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
            if _is_contextual_semantic_csv_table_header(header, next_header) and len(
                header
            ) == len(next_header):
                return (
                    next_header,
                    all_rows[index + 2 :],
                    _semantic_csv_column_context(header),
                )
        if _is_semantic_csv_header(header):
            return header, all_rows[index + 1 :], ()
        if _is_contextual_semantic_csv_header(header) and index > 0:
            context = tuple(str(column).strip() for column in all_rows[index - 1])
            if len(context) == len(header):
                return (
                    header,
                    all_rows[index + 1 :],
                    _semantic_csv_column_context(context),
                )
        if _is_contextual_semantic_csv_header(header):
            return header, all_rows[index + 1 :], ()
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
    normalized_context = tuple(
        normalize_runtime_identifier(column) for column in context
    )
    normalized_header = tuple(normalize_runtime_identifier(column) for column in header)
    if normalized_context == normalized_header:
        return False
    if not (frozenset(normalized_header) & MEASUREMENT_IDENTITY_FIELDS):
        return False
    if duplicate_values(normalized_context):
        return True
    return bool(frozenset(normalized_context) & CSV_HEADER_CONTEXT_STOPWORDS)
