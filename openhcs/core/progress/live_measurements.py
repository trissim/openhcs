"""Live measurement table previews carried by progress events."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
import math
from typing import Any

import numpy as np

from openhcs.core.artifacts import MeasurementsArtifactType
from openhcs.core.measurement_row_materialization import (
    is_structural_missing_measurement_cell,
)
from openhcs.core.runtime_stores import RuntimeArtifactAddress, StoredRuntimeValue
from openhcs.core.runtime_values import (
    ColumnarRows,
    MeasurementTable,
)


LIVE_MEASUREMENTS_CONTEXT_KEY = "live_measurements"
DEFAULT_LIVE_MEASUREMENT_ROW_LIMIT = 50
DEFAULT_LIVE_MEASUREMENT_COLUMN_LIMIT = 64
DEFAULT_LIVE_MEASUREMENT_PREVIEW_LIMIT = 8
MAX_CELL_TEXT_LENGTH = 200


class LiveMeasurementPayloadError(ValueError):
    """Raised when a live-measurement progress context is malformed."""


@dataclass(frozen=True, slots=True)
class LiveMeasurementRowPreview:
    """Bounded row preview plus full row count for one measurement table."""

    rows: tuple[Mapping[str, Any], ...]
    row_count: int


@dataclass(frozen=True, slots=True)
class LiveMeasurementTablePreview:
    """Bounded row preview for one measurement artifact write."""

    address: RuntimeArtifactAddress
    columns: tuple[str, ...]
    rows: tuple[Mapping[str, Any], ...]
    row_count: int
    truncated_rows: bool
    truncated_columns: bool
    object_name: str | None = None
    source_image_name: str | None = None

    @classmethod
    def from_record(
        cls,
        record: StoredRuntimeValue,
        *,
        row_limit: int = DEFAULT_LIVE_MEASUREMENT_ROW_LIMIT,
        column_limit: int = DEFAULT_LIVE_MEASUREMENT_COLUMN_LIMIT,
    ) -> "LiveMeasurementTablePreview | None":
        if record.key.artifact_type is not MeasurementsArtifactType:
            return None

        table = MeasurementTable.from_runtime_value(record.value)
        row_preview = _measurement_row_preview(table.rows, row_limit)
        all_columns = _ordered_columns(table=table, rows=row_preview.rows)
        columns = all_columns[:column_limit]
        preview_rows = tuple(
            {column: _json_safe_cell(row.get(column)) for column in columns}
            for row in row_preview.rows
        )
        return cls(
            address=RuntimeArtifactAddress.from_record(record),
            columns=columns,
            rows=preview_rows,
            row_count=row_preview.row_count,
            truncated_rows=row_preview.row_count > row_limit,
            truncated_columns=len(all_columns) > len(columns),
            object_name=table.object_name,
            source_image_name=table.source_image_name,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LiveMeasurementTablePreview":
        try:
            return cls(
                address=RuntimeArtifactAddress.from_dict(data["address"]),
                columns=tuple(str(column) for column in data.get("columns", ())),
                rows=tuple(
                    _decode_row_mapping(row) for row in data.get("rows", ())
                ),
                row_count=int(data.get("row_count", 0)),
                truncated_rows=bool(data.get("truncated_rows", False)),
                truncated_columns=bool(data.get("truncated_columns", False)),
                object_name=_optional_string(data.get("object_name")),
                source_image_name=_optional_string(data.get("source_image_name")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise LiveMeasurementPayloadError(
                f"Malformed live measurement table preview: {exc}"
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "address": self.address.to_dict(),
            "columns": list(self.columns),
            "rows": [dict(row) for row in self.rows],
            "row_count": self.row_count,
            "truncated_rows": self.truncated_rows,
            "truncated_columns": self.truncated_columns,
            "object_name": self.object_name,
            "source_image_name": self.source_image_name,
        }


@dataclass(frozen=True, slots=True)
class LiveMeasurementProgressPayload:
    """Live measurement preview payload stored inside ProgressEvent.context."""

    previews: tuple[LiveMeasurementTablePreview, ...]
    preview_count: int
    truncated_previews: bool

    @classmethod
    def from_records(
        cls,
        records: Sequence[StoredRuntimeValue],
        *,
        row_limit: int = DEFAULT_LIVE_MEASUREMENT_ROW_LIMIT,
        column_limit: int = DEFAULT_LIVE_MEASUREMENT_COLUMN_LIMIT,
        preview_limit: int = DEFAULT_LIVE_MEASUREMENT_PREVIEW_LIMIT,
    ) -> "LiveMeasurementProgressPayload | None":
        previews = tuple(
            preview
            for record in records
            for preview in (
                LiveMeasurementTablePreview.from_record(
                    record,
                    row_limit=row_limit,
                    column_limit=column_limit,
                ),
            )
            if preview is not None and preview.row_count > 0
        )
        if not previews:
            return None
        return cls(
            previews=previews[:preview_limit],
            preview_count=len(previews),
            truncated_previews=len(previews) > preview_limit,
        )

    @classmethod
    def from_context(
        cls,
        context: Mapping[str, Any] | None,
    ) -> "LiveMeasurementProgressPayload | None":
        if not context or LIVE_MEASUREMENTS_CONTEXT_KEY not in context:
            return None
        data = context[LIVE_MEASUREMENTS_CONTEXT_KEY]
        if not isinstance(data, Mapping):
            raise LiveMeasurementPayloadError(
                f"{LIVE_MEASUREMENTS_CONTEXT_KEY!r} must be a mapping."
            )
        raw_previews = data.get("previews")
        if not isinstance(raw_previews, Sequence) or isinstance(
            raw_previews, (str, bytes, bytearray)
        ):
            raise LiveMeasurementPayloadError(
                "Live measurement payload field 'previews' must be a sequence."
            )
        return cls(
            previews=tuple(
                LiveMeasurementTablePreview.from_dict(_require_mapping(raw_preview))
                for raw_preview in raw_previews
            ),
            preview_count=int(data.get("preview_count", len(raw_previews))),
            truncated_previews=bool(data.get("truncated_previews", False)),
        )

    def to_context(self) -> dict[str, Any]:
        return {
            LIVE_MEASUREMENTS_CONTEXT_KEY: {
                "previews": [preview.to_dict() for preview in self.previews],
                "preview_count": self.preview_count,
                "truncated_previews": self.truncated_previews,
            }
        }


def live_measurement_context_for_records(
    records: Sequence[StoredRuntimeValue],
    *,
    row_limit: int = DEFAULT_LIVE_MEASUREMENT_ROW_LIMIT,
    column_limit: int = DEFAULT_LIVE_MEASUREMENT_COLUMN_LIMIT,
    preview_limit: int = DEFAULT_LIVE_MEASUREMENT_PREVIEW_LIMIT,
) -> dict[str, Any] | None:
    """Return a progress context containing bounded measurement previews."""
    payload = LiveMeasurementProgressPayload.from_records(
        records,
        row_limit=row_limit,
        column_limit=column_limit,
        preview_limit=preview_limit,
    )
    return None if payload is None else payload.to_context()


def _measurement_row_preview(
    rows: Any,
    row_limit: int,
) -> LiveMeasurementRowPreview:
    row_limit = max(0, row_limit)
    if isinstance(rows, ColumnarRows):
        return _columnar_row_preview(rows, row_limit)
    if isinstance(rows, Mapping):
        return _mapping_columns_to_row_preview(rows, row_limit)
    if not _is_row_sequence(rows):
        return LiveMeasurementRowPreview(
            rows=({"value": rows},) if row_limit else (),
            row_count=1,
        )
    row_count = len(rows)
    return LiveMeasurementRowPreview(
        rows=tuple(_row_mapping(row) for row in rows[:row_limit]),
        row_count=row_count,
    )


def _columnar_row_preview(
    rows: ColumnarRows,
    row_limit: int,
) -> LiveMeasurementRowPreview:
    columns = tuple(str(column) for column in rows.columns)
    column_values = tuple((column, rows.column_values(column)) for column in columns)
    row_count = rows.row_count()
    return LiveMeasurementRowPreview(
        rows=tuple(
            {
                column: value
                for column, values in column_values
                for value in (
                    values[row_index]
                    if row_index < len(values)
                    else None,
                )
                if not is_structural_missing_measurement_cell(value)
            }
            for row_index in range(min(row_limit, row_count))
        ),
        row_count=row_count,
    )


def _mapping_columns_to_row_preview(
    rows: Mapping[Any, Any],
    row_limit: int,
) -> LiveMeasurementRowPreview:
    columns = {str(column): value for column, value in rows.items()}
    lengths = tuple(len(value) for value in columns.values() if _is_vector(value))
    if not lengths:
        return LiveMeasurementRowPreview(
            rows=(columns,) if row_limit else (),
            row_count=1,
        )
    row_count = max(lengths)
    return LiveMeasurementRowPreview(
        rows=tuple(
            {
                column: (
                    value[row_index]
                    if _is_vector(value) and row_index < len(value)
                    else value
                )
                for column, value in columns.items()
            }
            for row_index in range(min(row_limit, row_count))
        ),
        row_count=row_count,
    )


def _row_mapping(row: Any) -> Mapping[str, Any]:
    if isinstance(row, Mapping):
        return dict(row)
    if is_dataclass(row) and not isinstance(row, type):
        return asdict(row)
    return {"value": row}


def _ordered_columns(
    *,
    table: MeasurementTable,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()

    def add_column(column: Any) -> None:
        name = str(column)
        if name not in seen:
            seen.add(name)
            ordered.append(name)

    for field in table.fields:
        add_column(field.name)
    column_names = table.column_names()
    if column_names is not None:
        for column in column_names:
            add_column(column)
    elif isinstance(table.rows, Mapping):
        for column in table.rows:
            add_column(column)
    for row in rows:
        for column in row:
            add_column(column)
    return tuple(ordered)


def _json_safe_cell(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, np.generic):
        return _json_safe_cell(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe_cell(item) for key, item in value.items()}
    if _is_row_sequence(value):
        return [_json_safe_cell(item) for item in value]
    text = repr(value)
    if len(text) > MAX_CELL_TEXT_LENGTH:
        return f"{text[:MAX_CELL_TEXT_LENGTH]}..."
    return text


def _decode_row_mapping(value: Any) -> Mapping[str, Any]:
    return dict(_require_mapping(value))


def _require_mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LiveMeasurementPayloadError(
            f"Expected live measurement preview mapping, got {type(value).__name__}."
        )
    return value


def _is_row_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    )


def _is_vector(value: Any) -> bool:
    if isinstance(value, Mapping | str | bytes | bytearray):
        return False
    if isinstance(value, np.ndarray):
        return value.ndim > 0
    return isinstance(value, Sequence)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
