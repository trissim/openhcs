"""Measurement row semantics for runtime equivalence."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from types import MappingProxyType

from nominal_refactor_advisor.collection_algebra import sorted_tuple

from openhcs.core.equivalence.policy import (
    RuntimeMeasurementDialect,
    RuntimeMeasurementQualifierValueMode,
    RuntimeMeasurementRowQualifier,
    normalize_runtime_identifier,
)
from openhcs.core.equivalence.tables import measurement_table_cell_payload

IMAGE_IDENTITY_FIELDS = frozenset({"image_number", "image_id", "slice_index"})
_MeasurementQualifierValueRenderer = Callable[[tuple[object, ...]], str | None]
_MEASUREMENT_DIALECT_QUALIFIER_FIELD_NAMES_CACHE: dict[
    int,
    tuple[RuntimeMeasurementDialect, frozenset[str]],
] = {}


def _measurement_qualifier_value_renderers(
    renderers: Mapping[
        RuntimeMeasurementQualifierValueMode,
        _MeasurementQualifierValueRenderer,
    ],
) -> Mapping[
    RuntimeMeasurementQualifierValueMode,
    _MeasurementQualifierValueRenderer,
]:
    renderer_modes = set(renderers)
    value_modes = set(RuntimeMeasurementQualifierValueMode)
    if renderer_modes != value_modes:
        missing = sorted_tuple(mode.value for mode in value_modes - renderer_modes)
        extra = sorted_tuple(mode.value for mode in renderer_modes - value_modes)
        raise ValueError(
            "Measurement qualifier value renderers must cover "
            f"RuntimeMeasurementQualifierValueMode exactly: "
            f"missing={missing!r}, extra={extra!r}."
        )
    return MappingProxyType(dict(renderers))


def _two_digit_integer_measurement_qualifier_value(
    values: tuple[object, ...],
) -> str | None:
    return str(_measurement_qualifier_integer(values[0])).zfill(2)


def _fraction_of_count_measurement_qualifier_value(
    values: tuple[object, ...],
) -> str | None:
    if len(values) != 2:
        return None
    return (
        f"{_measurement_qualifier_integer(values[0])}"
        f"of{_measurement_qualifier_integer(values[1])}"
    )


def _identifier_measurement_qualifier_value(
    values: tuple[object, ...],
) -> str | None:
    return "_".join(_measurement_qualifier_identifier(value) for value in values)


_MEASUREMENT_QUALIFIER_VALUE_RENDERERS = _measurement_qualifier_value_renderers(
    {
        RuntimeMeasurementQualifierValueMode.IDENTIFIER: _identifier_measurement_qualifier_value,
        RuntimeMeasurementQualifierValueMode.TWO_DIGIT_INTEGER: _two_digit_integer_measurement_qualifier_value,
        RuntimeMeasurementQualifierValueMode.FRACTION_OF_COUNT: _fraction_of_count_measurement_qualifier_value,
    }
)


def measurement_row_image_identity_key(
    row: Mapping[str, object],
) -> tuple[tuple[str, object], ...]:
    """Return the image identity carried by a measurement row."""
    identity_values: list[tuple[str, object]] = []
    for field_name, value in row.items():
        normalized_field_name = normalize_runtime_identifier(field_name)
        if normalized_field_name not in IMAGE_IDENTITY_FIELDS:
            continue
        if value is None or not str(value).strip():
            continue
        identity_values.append(
            (
                normalized_field_name,
                measurement_table_cell_payload(value),
            )
        )
    return sorted_tuple(identity_values)


def axis_scoped_measurement_row_identity(
    row: Mapping[str, object],
    axis_key: object | None,
) -> tuple[tuple[str, object], ...]:
    """Return row identity scoped by runtime axis for local image numbering."""
    row_identity = measurement_row_image_identity_key(row)
    if axis_key is None:
        return row_identity
    return (
        ("_runtime_axis", measurement_table_cell_payload(axis_key)),
        *row_identity,
    )


def measurement_row_qualifiers(
    row: Mapping[str, object],
    dialect: RuntimeMeasurementDialect,
    field_name: str,
) -> tuple[str, ...]:
    return _measurement_row_qualifiers_for_field(
        dialect,
        field_name,
        lambda qualifier: _render_measurement_row_qualifier(row, qualifier),
    )


def measurement_row_qualifiers_from_values(
    row_values: Mapping[str, object],
    dialect: RuntimeMeasurementDialect,
    field_name: str,
) -> tuple[str, ...]:
    return _measurement_row_qualifiers_for_field(
        dialect,
        field_name,
        lambda qualifier: _render_measurement_row_qualifier_from_values(
            row_values,
            qualifier,
        ),
    )


def measurement_row_qualifiers_from_indexed_values_cached(
    row_values: tuple[object, ...],
    qualifiers: tuple[tuple[RuntimeMeasurementRowQualifier, tuple[int | None, ...]], ...],
    cache: dict[
        tuple[RuntimeMeasurementRowQualifier, tuple[object | None, ...]],
        str | None,
    ],
) -> tuple[str, ...]:
    rendered_values: list[str] = []
    for qualifier, indexes in qualifiers:
        values = tuple(
            None if index is None else row_values[index]
            for index in indexes
        )
        cache_key = (
            qualifier,
            tuple(
                None if value is None else measurement_table_cell_payload(value)
                for value in values
            ),
        )
        rendered = cache.get(cache_key)
        if cache_key not in cache:
            rendered = _render_measurement_row_qualifier_value_tuple(
                values,
                qualifier,
            )
            cache[cache_key] = rendered
        if rendered is not None:
            rendered_values.append(rendered)
    return tuple(rendered_values)


def row_qualifier_columns(
    normalized_fields: tuple[str, ...],
    dialect: RuntimeMeasurementDialect,
) -> tuple[tuple[str, int], ...]:
    qualifier_fields = measurement_qualifier_field_names(dialect)
    return tuple(
        (field_name, index)
        for index, field_name in enumerate(normalized_fields)
        if field_name in qualifier_fields
    )


def row_qualifier_values(
    row: tuple[object, ...],
    columns: tuple[tuple[str, int], ...],
) -> Mapping[str, object]:
    return MappingProxyType(
        {field_name: row[index] for field_name, index in columns}
    )


def row_qualifier_applies_to_field(
    qualifier: RuntimeMeasurementRowQualifier,
    field_parts: tuple[str, ...],
) -> bool:
    if not qualifier.feature_prefixes:
        return True
    return any(
        len(field_parts) >= len(prefix) and field_parts[: len(prefix)] == prefix
        for prefix in qualifier.feature_prefixes
    )


def measurement_qualifier_field_names(
    dialect: RuntimeMeasurementDialect,
) -> frozenset[str]:
    cached = _MEASUREMENT_DIALECT_QUALIFIER_FIELD_NAMES_CACHE.get(id(dialect))
    if cached is not None and cached[0] is dialect:
        return cached[1]
    field_names = frozenset(
        field_name
        for qualifier in dialect.row_qualifiers
        for field_name in qualifier.field_names
    )
    _MEASUREMENT_DIALECT_QUALIFIER_FIELD_NAMES_CACHE[id(dialect)] = (
        dialect,
        field_names,
    )
    return field_names


def _measurement_row_qualifiers_for_field(
    dialect: RuntimeMeasurementDialect,
    field_name: str,
    render: Callable[[RuntimeMeasurementRowQualifier], str | None],
) -> tuple[str, ...]:
    qualifiers: list[str] = []
    field_parts = tuple(
        part for part in normalize_runtime_identifier(field_name).split("_") if part
    )
    for qualifier in dialect.row_qualifiers:
        if not row_qualifier_applies_to_field(qualifier, field_parts):
            continue
        rendered = render(qualifier)
        if rendered is None:
            continue
        qualifiers.append(rendered)
    return tuple(qualifiers)


def _render_measurement_row_qualifier(
    row: Mapping[str, object],
    qualifier: RuntimeMeasurementRowQualifier,
) -> str | None:
    values = tuple(
        _first_row_value(row, (field_name,))
        for field_name in qualifier.field_names
    )
    return _render_measurement_row_qualifier_value_tuple(values, qualifier)


def _render_measurement_row_qualifier_from_values(
    row_values: Mapping[str, object],
    qualifier: RuntimeMeasurementRowQualifier,
) -> str | None:
    values = tuple(row_values.get(field_name) for field_name in qualifier.field_names)
    return _render_measurement_row_qualifier_value_tuple(values, qualifier)


def _render_measurement_row_qualifier_value_tuple(
    values: tuple[object, ...],
    qualifier: RuntimeMeasurementRowQualifier,
) -> str | None:
    if any(_is_missing_measurement_qualifier_value(value) for value in values):
        return None
    return _MEASUREMENT_QUALIFIER_VALUE_RENDERERS[qualifier.value_mode](values)


def _measurement_qualifier_identifier(value: object) -> str:
    integer_value = _optional_measurement_qualifier_integer(value)
    if integer_value is not None:
        return str(integer_value)
    return normalize_runtime_identifier(value)


def _is_missing_measurement_qualifier_value(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    if not text:
        return True
    try:
        return math.isnan(float(text))
    except (TypeError, ValueError):
        return False


def _measurement_qualifier_integer(value: object) -> int:
    integer_value = _optional_measurement_qualifier_integer(value)
    if integer_value is None:
        raise ValueError(f"Measurement qualifier {value!r} is not integer-like.")
    return integer_value


def _optional_measurement_qualifier_integer(value: object) -> int | None:
    try:
        numeric = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or not numeric.is_integer():
        return None
    return int(numeric)


def _first_row_value(
    row: Mapping[str, object],
    field_names: tuple[str, ...],
) -> object | None:
    normalized_fields = {normalize_runtime_identifier(field): field for field in row}
    for field_name in field_names:
        field = normalized_fields.get(field_name)
        if field is not None:
            return row[field]
    return None
