"""CellProfiler-style measurement feature lookup over typed measurement tables."""

from __future__ import annotations

from collections.abc import Mapping
import re

import numpy as np

from openhcs.core.runtime_values import MeasurementTable


def measurement_values_for_feature(
    measurement_tables: tuple[MeasurementTable, ...],
    feature_name: str,
    *,
    object_count: int,
) -> np.ndarray:
    """Return object-indexed measurement values for a CellProfiler feature."""

    candidates = measurement_feature_candidates(feature_name)
    values_by_label: dict[int, float] = {}
    positional_values: list[float] = []
    for row in measurement_rows(measurement_tables):
        row_mapping = measurement_row_mapping(row)
        field_name = matching_measurement_field(row_mapping, candidates)
        if field_name is None:
            continue
        value = float(row_mapping[field_name])
        object_label = measurement_object_label(row_mapping)
        if object_label is None:
            positional_values.append(value)
            continue
        values_by_label[object_label] = value
    if values_by_label:
        return np.array(
            [values_by_label.get(index, np.nan) for index in range(1, object_count + 1)]
        )
    if positional_values:
        return np.array(positional_values[:object_count])
    raise ValueError(f"Could not resolve measurement feature {feature_name!r}.")


def measurement_rows(
    measurement_tables: tuple[MeasurementTable, ...],
) -> tuple[object, ...]:
    rows: list[object] = []
    for table in measurement_tables:
        if isinstance(table.rows, list | tuple):
            rows.extend(table.rows)
            continue
        rows.append(table.rows)
    return tuple(rows)


def measurement_row_mapping(row: object) -> Mapping[str, object]:
    if isinstance(row, Mapping):
        return row
    try:
        return vars(row)
    except TypeError as exc:
        raise TypeError(
            f"Unsupported CellProfiler measurement row type {type(row).__name__}."
        ) from exc


def matching_measurement_field(
    row: Mapping[str, object],
    candidates: frozenset[str],
) -> str | None:
    for field_name in row:
        if normalize_measurement_token(field_name) in candidates:
            return field_name
    return None


def measurement_object_label(row: Mapping[str, object]) -> int | None:
    for key in ("object_label", "object_number", "object_id", "label"):
        if key in row:
            return int(row[key])
    return None


def measurement_feature_candidates(feature_name: str) -> frozenset[str]:
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


def count_feature_object_name(feature_name: str | None) -> str | None:
    if feature_name is None:
        return None
    prefix = "Count_"
    if not feature_name.startswith(prefix):
        return None
    object_name = feature_name[len(prefix):].strip()
    return object_name or None


def normalize_measurement_token(value: object) -> str:
    text = str(value)
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
