"""CellProfiler-style measurement feature lookup over typed measurement tables."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import re

import numpy as np

from openhcs.core.runtime_values import MeasurementTable

MEASUREMENT_OBJECT_NAME_FIELD = "object_name"
MEASUREMENT_FEATURE_NAME_FIELDS = ("feature_name", "measurement_name", "output_name")
MEASUREMENT_VALUE_FIELDS = ("result_value", "measurement_value", "value", "mean_value")


def measurement_values_for_feature(
    measurement_tables: tuple[MeasurementTable, ...],
    feature_name: str,
    *,
    object_count: int,
    object_name: str | None = None,
) -> np.ndarray:
    """Return object-indexed measurement values for a CellProfiler feature."""

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


def measurement_values_for_label_slices(
    measurement_tables: tuple[MeasurementTable, ...],
    feature_name: str,
    labels: object,
    *,
    object_name: str | None = None,
) -> tuple[np.ndarray, ...]:
    """Return measurement values aligned to positive label IDs in each label plane."""

    label_array = np.asarray(labels)
    label_planes = (
        (label_array,)
        if label_array.ndim <= 2
        else tuple(label_array[index] for index in range(label_array.shape[0]))
    )
    try:
        values_by_label, positional_values = measurement_value_index(
            measurement_tables,
            feature_name,
            object_name=object_name,
        )
    except ValueError:
        if _label_planes_are_empty(label_planes):
            return tuple(np.array([], dtype=float) for _plane in label_planes)
        raise
    return tuple(
        _measurement_values_for_label_plane(
            label_plane,
            values_by_label,
            positional_values,
            feature_name,
        )
        for label_plane in label_planes
    )


def measurement_value_index(
    measurement_tables: tuple[MeasurementTable, ...],
    feature_name: str,
    *,
    object_name: str | None = None,
) -> tuple[dict[int, float], list[float]]:
    candidates = measurement_feature_candidates(feature_name)
    values_by_label: dict[int, float] = {}
    positional_values: list[float] = []
    for row in measurement_rows(measurement_tables):
        row_mapping = measurement_row_mapping(row)
        row_object_name = measurement_row_object_name(row_mapping)
        if (
            object_name is not None
            and row_object_name is not None
            and row_object_name != object_name
        ):
            continue
        value = measurement_row_value(row_mapping, candidates)
        if value is None:
            continue
        object_label = measurement_object_label(row_mapping)
        if object_label is None:
            positional_values.append(float(value))
            continue
        values_by_label[object_label] = float(value)
    if not values_by_label and not positional_values:
        raise ValueError(f"Could not resolve measurement feature {feature_name!r}.")
    return values_by_label, positional_values


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


def _label_planes_are_empty(label_planes: tuple[np.ndarray, ...]) -> bool:
    return all(not np.any(label_plane > 0) for label_plane in label_planes)


def measurement_row_mapping(row: object) -> Mapping[str, object]:
    if isinstance(row, Mapping):
        return row
    if is_dataclass(row):
        return {field.name: getattr(row, field.name) for field in fields(row)}
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


def measurement_row_value(
    row: Mapping[str, object],
    candidates: frozenset[str],
) -> object | None:
    field_name = matching_measurement_field(row, candidates)
    if field_name is not None:
        return row[field_name]
    if not measurement_row_feature_matches(row, candidates):
        return None
    for value_field in MEASUREMENT_VALUE_FIELDS:
        if value_field in row:
            return row[value_field]
    return None


def measurement_row_feature_matches(
    row: Mapping[str, object],
    candidates: frozenset[str],
) -> bool:
    for field_name in MEASUREMENT_FEATURE_NAME_FIELDS:
        value = row.get(field_name)
        if value is None:
            continue
        if normalize_measurement_token(value) in candidates:
            return True
    return False


def measurement_object_label(row: Mapping[str, object]) -> int | None:
    for key in ("object_label", "object_number", "object_id", "label"):
        if key in row:
            return int(row[key])
    return None


def measurement_row_object_name(row: Mapping[str, object]) -> str | None:
    value = row.get(MEASUREMENT_OBJECT_NAME_FIELD)
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def annotate_measurement_row_object(row: object, object_name: str) -> Mapping[str, object]:
    normalized_object_name = object_name.strip()
    if not normalized_object_name:
        raise ValueError("object_name cannot be empty.")
    return {
        **dict(measurement_row_mapping(row)),
        MEASUREMENT_OBJECT_NAME_FIELD: normalized_object_name,
    }


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


def _measurement_values_for_label_plane(
    label_plane: np.ndarray,
    values_by_label: Mapping[int, float],
    positional_values: list[float],
    feature_name: str,
) -> np.ndarray:
    positive_labels = tuple(
        int(label)
        for label in np.unique(label_plane)
        if int(label) > 0
    )
    if values_by_label:
        return np.array(
            [values_by_label.get(label, np.nan) for label in positive_labels]
        )
    if positional_values:
        return np.array(positional_values[: len(positive_labels)])
    raise ValueError(f"Could not resolve measurement feature {feature_name!r}.")
