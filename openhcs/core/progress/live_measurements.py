"""Live measurement table previews carried by progress events."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
import math
from typing import Any

import numpy as np

from openhcs.core.artifacts import ArtifactKind
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
        if record.key.kind is not ArtifactKind.MEASUREMENTS:
            return None

        table = MeasurementTable.from_runtime_value(record.value)
        row_mappings = _measurement_row_mappings(table.rows)
        all_columns = _ordered_columns(table=table, rows=row_mappings)
        columns = all_columns[:column_limit]
        preview_rows = tuple(
            {column: _json_safe_cell(row.get(column)) for column in columns}
            for row in row_mappings[:row_limit]
        )
        return cls(
            address=RuntimeArtifactAddress.from_record(record),
            columns=columns,
            rows=preview_rows,
            row_count=len(row_mappings),
            truncated_rows=len(row_mappings) > row_limit,
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


def _measurement_row_mappings(rows: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(rows, ColumnarRows):
        return tuple(dict(row) for row in rows.row_mappings())
    if isinstance(rows, Mapping):
        return _mapping_columns_to_rows(rows)
    if not _is_row_sequence(rows):
        return ({"value": rows},)
    return tuple(_row_mapping(row) for row in rows)


def _mapping_columns_to_rows(rows: Mapping[Any, Any]) -> tuple[Mapping[str, Any], ...]:
    columns = {str(column): value for column, value in rows.items()}
    lengths = tuple(len(value) for value in columns.values() if _is_vector(value))
    if not lengths:
        return (columns,)
    row_count = max(lengths)
    return tuple(
        {
            column: (
                value[row_index]
                if _is_vector(value) and row_index < len(value)
                else value
            )
            for column, value in columns.items()
        }
        for row_index in range(row_count)
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
    for field in table.fields:
        name = str(field.name)
        if name not in ordered:
            ordered.append(name)
    for row in rows:
        for column in row:
            name = str(column)
            if name not in ordered:
                ordered.append(name)
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
