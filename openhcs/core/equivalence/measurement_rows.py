"""Measurement row semantics for runtime equivalence."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from types import MappingProxyType
from typing import Generic, cast

from openhcs.core.component_group_scope import ComponentGroupScope
from openhcs.core.equivalence.cells import (
    RuntimeCellSignature,
    RuntimeMeasurementCellSignatureProjection,
    cell_signature_numeric_value,
    measurement_numeric_runtime_value,
    measurement_table_cell_payload,
    runtime_cell_signature,
    runtime_measurement_cell_signature,
    runtime_measurement_cell_signature_if_present,
    runtime_measurement_value_is_present,
    runtime_numeric_text_value,
    runtime_value_is_mapping,
)
from openhcs.core.equivalence.keys import (
    RuntimeMeasurementFeatureKey,
    RuntimeMeasurementFeatureKeyProjection,
    RuntimeMeasurementFeatureKeySourceContext,
    RuntimeMeasurementSubjectKey,
    canonical_measurement_feature_name,
)
from openhcs.core.equivalence.measurement_facts import (
    RuntimeDirectionalPairMeasurementDerivationContract,
    RuntimeMeasurementFact,
    RuntimeMeasurementFactProjectionContract,
    RuntimeMeasurementFacts,
    RuntimeMeasurementPaddingGroup,
    RuntimeRequiredMeasurementKeys,
    RuntimeRowProjectionRecord,
    RuntimeRowProjectionRecords,
    RuntimeRowProjectionValueT,
)
from openhcs.core.equivalence.policy import (
    DEFAULT_RUNTIME_MEASUREMENT_DIALECT,
    RuntimeEquivalencePolicy,
    RuntimeMeasurementDialect,
    RuntimeMeasurementQualifierValueMode,
    RuntimeMeasurementRowQualifier,
    normalize_runtime_identifier,
)
from openhcs.core.equivalence.tables import (
    MEASUREMENT_IDENTITY_FIELDS,
)
from openhcs.core.measurement_row_materialization import (
    MeasurementRowDeclaredValue,
    MeasurementRowObjectIdentityRole,
    MeasurementRowObjectLabel,
    MeasurementRowObjectName,
    MeasurementRowSourceImageName,
    measurement_row_has_long_form_measurement_fields,
)
from openhcs.core.runtime_measurements import (
    MeasurementRowValueField,
    MeasurementRowAxisField,
    MeasurementScope,
    MeasurementStatistic,
    MeasuredObjectAnchorFeatureMarker,
    RuntimeMeasurementFeatureOwner,
    RuntimeMeasurementRowIdentityContract,
)
from openhcs.core.runtime_tabular_values import (
    MeasurementObjectRowIdentity,
    measurement_row_mapping,
)
from openhcs.core.runtime_relationships import (
    ObjectInstanceKey,
)

IMAGE_IDENTITY_FIELDS = (
    DEFAULT_RUNTIME_MEASUREMENT_DIALECT.row_identity_contract.image_identity_fields
)
OBJECT_LABEL_ROW_IDENTITY_FIELD = "object_label"
RUNTIME_AXIS_ROW_IDENTITY_FIELD = "_runtime_axis"
RUNTIME_SLICE_ROW_IDENTITY_FIELD = "_runtime_slice"
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
        missing = tuple(sorted(mode.value for mode in value_modes - renderer_modes))
        extra = tuple(sorted(mode.value for mode in renderer_modes - value_modes))
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
    dialect: RuntimeMeasurementDialect = DEFAULT_RUNTIME_MEASUREMENT_DIALECT,
) -> tuple[tuple[str, object], ...]:
    """Return the image identity carried by a measurement row."""
    contract = dialect.row_identity_contract
    normalized_present_fields = frozenset(
        normalize_runtime_identifier(field_name)
        for field_name, value in row.items()
        if value is not None and str(value).strip()
    )
    selected_fields = contract.selected_image_identity_fields(normalized_present_fields)
    identity_values: list[tuple[str, object]] = []
    for field_name, value in row.items():
        normalized_field_name = normalize_runtime_identifier(field_name)
        if normalized_field_name not in selected_fields:
            continue
        identity_values.append(
            (
                normalized_field_name,
                measurement_table_cell_payload(value),
            )
        )
    return tuple(sorted(identity_values))


def axis_scoped_measurement_row_identity(
    row: Mapping[str, object],
    axis_key: str | None,
    dialect: RuntimeMeasurementDialect = DEFAULT_RUNTIME_MEASUREMENT_DIALECT,
) -> tuple[tuple[str, object], ...]:
    """Return row identity scoped by runtime axis for local image numbering."""
    row_identity = measurement_row_image_identity_key(row, dialect)
    if axis_key is None:
        return row_identity
    return (
        (RUNTIME_AXIS_ROW_IDENTITY_FIELD, measurement_table_cell_payload(axis_key)),
        *row_identity,
    )


@dataclass(frozen=True, slots=True)
class RuntimeImageNumberOffset:
    """Compute image-number offset for table rows."""

    value: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", float(self.value))

    @classmethod
    def from_table_rows(
        cls,
        header: tuple[str, ...],
        rows: tuple[tuple[str, ...], ...],
    ) -> "RuntimeImageNumberOffset":
        image_number_indexes = tuple(
            index
            for index, field_name in enumerate(header)
            if normalize_runtime_identifier(field_name) == "image_number"
        )
        if not image_number_indexes:
            return cls()
        image_number_index = image_number_indexes[0]
        return cls(
            cls._offset_from_values(
                row[image_number_index] for row in rows if image_number_index < len(row)
            )
        )

    @classmethod
    def from_runtime_rows(cls, rows: Iterable[object]) -> "RuntimeImageNumberOffset":
        return cls(
            cls._offset_from_values(
                image_number
                for row in rows
                for image_number in (
                    RuntimeMeasurementRowMapping(
                        measurement_row_mapping(row)
                    ).first_value(("image_number",)),
                )
                if image_number is not None
            )
        )

    @classmethod
    def from_measurement_table(cls, table: object) -> "RuntimeImageNumberOffset":
        """Return image-number offset from table schema/columns when available."""
        return cls.from_runtime_rows(table.rows.iter_row_mappings())

    @classmethod
    def _offset_from_values(cls, values: Iterable[object]) -> float:
        image_numbers: list[float] = []
        for value in values:
            try:
                image_number = float(str(value).strip())
            except ValueError:
                continue
            if math.isfinite(image_number) and image_number > 0:
                image_numbers.append(image_number)
        if not image_numbers:
            return 0.0
        return min(image_numbers) - 1.0

    def normalized_reference_value(
        self,
        field_name: str,
        value: object,
    ) -> object:
        """Normalize image-number reference values to axis-local numbering."""
        if self.value == 0:
            return value
        if not image_number_reference_measurement_field(field_name):
            return value
        if isinstance(value, Mapping):
            return {
                key: self.normalized_reference_value(field_name, nested_value)
                for key, nested_value in value.items()
            }
        numeric_value = runtime_numeric_text_value(str(value))
        if numeric_value is None:
            return value
        if not math.isfinite(numeric_value) or numeric_value <= 0:
            return value
        normalized = numeric_value - self.value
        return int(normalized) if normalized.is_integer() else normalized

    def object_instance_key(
        self,
        row: Mapping[str, object],
        object_id: int,
    ) -> ObjectInstanceKey:
        """Normalize an external image-number row into runtime object identity."""
        runtime_key = ObjectInstanceKey.from_measurement_row(row, object_id)
        if runtime_key.slice_index is not None:
            return runtime_key
        image_number = RuntimeMeasurementRowMapping(row).first_value(("image_number",))
        if image_number is None:
            return runtime_key
        numeric_value = runtime_numeric_text_value(str(image_number))
        if numeric_value is None or not math.isfinite(numeric_value):
            return runtime_key
        slice_index = numeric_value - self.value - 1.0
        if slice_index < 0 or not slice_index.is_integer():
            return runtime_key
        return ObjectInstanceKey(object_id, slice_index=int(slice_index))


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


def measurement_qualifier_cache_value(value: object) -> object:
    """Return a cheap exact cache value for scalar row qualifiers."""
    if value is None:
        return None
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        return ("float", "nan" if math.isnan(value) else repr(value))
    return measurement_table_cell_payload(value)


def measurement_row_qualifiers_from_indexed_values_cached(
    row_values: "RuntimeIndexedRowValues",
    qualifiers: tuple[
        tuple[RuntimeMeasurementRowQualifier, tuple[int | None, ...]], ...
    ],
    cache: dict[
        tuple[RuntimeMeasurementRowQualifier, tuple[object | None, ...]],
        str | None,
    ],
) -> tuple[str, ...]:
    rendered_values: list[str] = []
    for qualifier, indexes in qualifiers:
        values = row_values.values_at(indexes)
        cache_key = (
            qualifier,
            tuple(measurement_qualifier_cache_value(value) for value in values),
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
    row: "RuntimeIndexedRowValues",
    columns: tuple[tuple[str, int], ...],
) -> Mapping[str, object]:
    return MappingProxyType(
        {field_name: row.at(index) for field_name, index in columns}
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
        _first_row_value(row, (field_name,)) for field_name in qualifier.field_names
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


_CACHE_MISS = object()

RuntimeMeasurementIndexedQualifier = tuple[
    RuntimeMeasurementRowQualifier,
    tuple[int | None, ...],
]
RuntimeLongFormMeasurementFactValue = RuntimeMeasurementFact | None
RuntimeNumericMeasurementValue = tuple[RuntimeMeasurementFeatureKey, float]
RuntimeNumericMeasurementValues = tuple[RuntimeNumericMeasurementValue, ...]
RuntimeMeasurementQualifierCacheKey = tuple[
    RuntimeMeasurementRowQualifier,
    tuple[object | None, ...],
]
RuntimeMeasurementRowIdentity = tuple[tuple[str, object], ...]
RuntimeMeasurementRowIdentityOrMissing = RuntimeMeasurementRowIdentity | None


def object_label_signature_from_row_identity(
    row_identity: RuntimeMeasurementRowIdentity,
) -> RuntimeCellSignature | None:
    """Return the object-label signature embedded in a runtime row identity."""
    return next(
        (
            field_value
            for field_name, field_value in row_identity
            if (
                field_name == OBJECT_LABEL_ROW_IDENTITY_FIELD
                and isinstance(field_value, RuntimeCellSignature)
            )
        ),
        None,
    )


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementRowSchema:
    """Schema indexes for one runtime measurement table row shape."""

    feature_indexes: tuple[int, ...]
    qualifiers_by_index: dict[int, tuple[RuntimeMeasurementIndexedQualifier, ...]]
    long_form_qualifiers: tuple[RuntimeMeasurementIndexedQualifier, ...]
    long_form_feature_indexes: tuple[int, ...]
    long_form_value_indexes: tuple[int, ...]

    @property
    def has_measurement_features(self) -> bool:
        """Return whether this schema declares wide or long-form measurements."""
        return bool(
            self.feature_indexes
            or (self.long_form_feature_indexes and self.long_form_value_indexes)
        )


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementRowHeaderProjection:
    """Cached normalized-field projection for one row header."""

    normalized_fields: Mapping[str, str]
    normalized_field_names: frozenset[str]


RuntimeMeasurementRowSchemaCache = dict[tuple[str, ...], RuntimeMeasurementRowSchema]
RuntimeMeasurementFeatureKeyCacheKey = tuple[
    MeasurementScope,
    str | None,
    str | None,
    str,
    tuple[str, ...],
]
RuntimeMeasurementProjectedFeatureCacheKey = tuple[
    MeasurementScope,
    str | None,
    str | None,
    str,
    str,
]
RuntimeMeasurementFeatureKeyCache = dict[
    RuntimeMeasurementFeatureKeyCacheKey,
    RuntimeMeasurementFeatureKey | None,
]
RuntimeMeasurementWideFeatureIndexCacheKey = tuple[
    tuple[str, ...],
    MeasurementScope,
    str | None,
    str | None,
    int,
]
RuntimeMeasurementWideFeatureIndexCache = dict[
    RuntimeMeasurementWideFeatureIndexCacheKey,
    tuple[int, ...],
]


@dataclass(frozen=True, slots=True)
class RuntimeWideProjectionColumn:
    """Static semantic projection for one wide measurement-table column."""

    index: int
    field_name: str
    key: RuntimeMeasurementFeatureKey
    projected_key: RuntimeMeasurementProjectedFeatureCacheKey
    padding_group: RuntimeMeasurementPaddingGroup
    measured_object_anchor: bool
    producer_owned_feature: bool


RuntimeMeasurementWideFeaturePlanCacheKey = tuple[
    tuple[str, ...],
    MeasurementScope,
    str | None,
    str | None,
    type[RuntimeMeasurementFeatureOwner] | None,
    str,
    int,
]
RuntimeMeasurementWideFeaturePlanCache = dict[
    RuntimeMeasurementWideFeaturePlanCacheKey,
    tuple[RuntimeWideProjectionColumn, ...],
]


RuntimeMeasurementQualifierRenderCache = dict[
    RuntimeMeasurementQualifierCacheKey,
    str | None,
]
RuntimeMeasurementPaddingGroupCache = dict[
    tuple[str, RuntimeMeasurementProjectedFeatureCacheKey],
    RuntimeMeasurementPaddingGroup,
]
RuntimeMeasurementIndexedQualifierCache = dict[int, tuple[str, ...]]
RuntimeRowQualifierResolutionCache = dict[
    int,
    tuple[tuple[RuntimeMeasurementIndexedQualifier, ...], tuple[str, ...]],
]
RuntimeMeasurementPaddingGroupPresence = dict[RuntimeMeasurementPaddingGroup, bool]
RuntimeMeasurementRequiredNestedBase = RuntimeMeasurementProjectedFeatureCacheKey


def runtime_measurement_row_value_is_present_without_formatting(value: object) -> bool:
    """Return value presence without invoking expensive array/string formatting."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def runtime_measurement_field_indexes(
    normalized_field_indexes: Mapping[str, int],
    field_names: tuple[str, ...],
) -> tuple[int, ...]:
    """Return normalized row indexes for the requested field names."""
    return tuple(
        index
        for field_name in field_names
        if (index := normalized_field_indexes.get(field_name)) is not None
    )


def normalized_runtime_measurement_identity_field_matches_qualifier_names(
    normalized: str,
    qualifier_field_names: frozenset[str],
    non_measurement_field_prefixes: tuple[str, ...],
) -> bool:
    """Return whether a normalized field is structural row metadata."""
    if normalized in MEASUREMENT_IDENTITY_FIELDS:
        return True
    if normalized in qualifier_field_names:
        return True
    if normalized.startswith(non_measurement_field_prefixes):
        return True
    return normalized.startswith("metadata_")


@lru_cache(maxsize=4096)
def runtime_measurement_row_schema_for_header(
    header: tuple[str, ...],
    row_qualifiers: tuple[RuntimeMeasurementRowQualifier, ...],
    row_identity_contract: RuntimeMeasurementRowIdentityContract,
    non_measurement_field_prefixes: tuple[str, ...],
) -> RuntimeMeasurementRowSchema:
    """Return cached projection schema for one header and qualifier declaration."""
    normalized_fields = tuple(normalize_runtime_identifier(field) for field in header)
    object_identity_field = row_identity_contract.selected_object_identity_field(
        frozenset(normalized_fields)
    )
    aggregate_reference_indexes = frozenset(
        index
        for index, field_name in enumerate(header)
        if aggregate_image_number_reference_measurement_field(field_name)
    )
    normalized_field_indexes = {
        field_name: index for index, field_name in enumerate(normalized_fields)
    }
    qualifier_field_names = frozenset(
        field_name
        for qualifier in row_qualifiers
        for field_name in qualifier.field_names
    )
    feature_indexes = tuple(
        index
        for index, field_name in enumerate(normalized_fields)
        if not normalized_runtime_measurement_identity_field_matches_qualifier_names(
            field_name,
            qualifier_field_names,
            non_measurement_field_prefixes,
        )
        and field_name != object_identity_field
        and index not in aggregate_reference_indexes
    )
    qualifier_indexes = {
        qualifier: indexes
        for qualifier in row_qualifiers
        for indexes in (
            tuple(
                normalized_field_indexes.get(field_name)
                for field_name in qualifier.field_names
            ),
        )
        if all(index is not None for index in indexes)
    }
    qualifier_tuple_cache: dict[
        tuple[RuntimeMeasurementIndexedQualifier, ...],
        tuple[RuntimeMeasurementIndexedQualifier, ...],
    ] = {}

    def canonical_qualifiers_for_index(
        index: int,
    ) -> tuple[RuntimeMeasurementIndexedQualifier, ...]:
        qualifiers = tuple(
            (qualifier, indexes)
            for qualifier, indexes in qualifier_indexes.items()
            if row_qualifier_applies_to_field(
                qualifier,
                tuple(part for part in normalized_fields[index].split("_") if part),
            )
        )
        cached = qualifier_tuple_cache.get(qualifiers)
        if cached is not None:
            return cached
        qualifier_tuple_cache[qualifiers] = qualifiers
        return qualifiers

    return RuntimeMeasurementRowSchema(
        feature_indexes,
        {index: canonical_qualifiers_for_index(index) for index in feature_indexes},
        tuple(qualifier_indexes.items()),
        runtime_measurement_field_indexes(
            normalized_field_indexes,
            MeasurementRowAxisField.feature_name_field_names_ordered(),
        ),
        runtime_measurement_field_indexes(
            normalized_field_indexes,
            MeasurementRowValueField.field_names_ordered(),
        ),
    )


@lru_cache(maxsize=16384)
def runtime_measurement_row_header_projection(
    header: tuple[str, ...],
) -> RuntimeMeasurementRowHeaderProjection:
    """Return cached normalized-field metadata for one row header."""
    normalized_fields = MappingProxyType(
        {normalize_runtime_identifier(field_name): field_name for field_name in header}
    )
    return RuntimeMeasurementRowHeaderProjection(
        normalized_fields=normalized_fields,
        normalized_field_names=frozenset(normalized_fields),
    )


def runtime_measurement_feature_key_cache_key(
    subject: RuntimeMeasurementSubjectKey,
    source_name: str | None,
    feature_name: str,
    qualifiers: tuple[str, ...] = (),
) -> RuntimeMeasurementFeatureKeyCacheKey:
    """Return primitive cache identity for a projected measurement feature key."""
    return (
        subject.scope,
        subject.name,
        source_name,
        feature_name,
        qualifiers,
    )


def runtime_measurement_projected_feature_cache_key(
    key: RuntimeMeasurementFeatureKey,
) -> RuntimeMeasurementProjectedFeatureCacheKey:
    """Return primitive cache identity for an already-projected feature key."""
    return runtime_measurement_projected_feature_identity(
        key.subject,
        key.source_name,
        key.feature_name,
        key.statistic,
    )


def runtime_measurement_projected_feature_identity(
    subject: RuntimeMeasurementSubjectKey,
    source_name: str | None,
    feature_name: str,
    statistic: str,
) -> RuntimeMeasurementProjectedFeatureCacheKey:
    """Return primitive identity for one projected measurement feature."""
    return (
        subject.scope,
        subject.name,
        source_name,
        feature_name,
        statistic,
    )


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementRequiredKeyIndex:
    """Precomputed lookup surface for required measurement-key projections."""

    required_keys: RuntimeRequiredMeasurementKeys
    projected_keys: frozenset[RuntimeMeasurementProjectedFeatureCacheKey]
    nested_bases: frozenset[RuntimeMeasurementRequiredNestedBase]

    @classmethod
    def from_required_keys(
        cls,
        required_keys: RuntimeRequiredMeasurementKeys,
    ) -> "RuntimeMeasurementRequiredKeyIndex":
        if required_keys is None:
            return cls(None, frozenset(), frozenset())
        return cls(
            required_keys,
            frozenset(
                runtime_measurement_projected_feature_cache_key(required_key)
                for required_key in required_keys
            ),
            frozenset(
                nested_base
                for required_key in required_keys
                for nested_base in cls.nested_bases_for_key(required_key)
            ),
        )

    @staticmethod
    def nested_bases_for_key(
        key: RuntimeMeasurementFeatureKey,
    ) -> tuple[RuntimeMeasurementRequiredNestedBase, ...]:
        feature_parts = tuple(part for part in key.feature_name.split("_") if part)
        return tuple(
            runtime_measurement_projected_feature_identity(
                key.subject,
                key.source_name,
                "_".join(feature_parts[:index]),
                key.statistic,
            )
            for index in range(1, len(feature_parts))
        )

    @staticmethod
    def nested_base_for_key(
        key: RuntimeMeasurementFeatureKey,
    ) -> RuntimeMeasurementRequiredNestedBase:
        return runtime_measurement_projected_feature_identity(
            key.subject,
            key.source_name,
            key.feature_name,
            key.statistic,
        )

    def requires_key(
        self,
        key: RuntimeMeasurementFeatureKey,
        *,
        projected_key: RuntimeMeasurementProjectedFeatureCacheKey | None = None,
    ) -> bool:
        """Return whether an already-projected key is required."""
        if self.required_keys is None:
            return True
        if projected_key is None:
            projected_key = runtime_measurement_projected_feature_cache_key(key)
        return projected_key in self.projected_keys

    def requires_value(
        self,
        key: RuntimeMeasurementFeatureKey,
        *,
        value_is_mapping: bool,
        projected_key: RuntimeMeasurementProjectedFeatureCacheKey | None = None,
    ) -> bool:
        if self.required_keys is None:
            return True
        if value_is_mapping:
            return self.nested_base_for_key(key) in self.nested_bases
        return self.requires_key(key, projected_key=projected_key)

    def may_require_value(
        self,
        key: RuntimeMeasurementFeatureKey,
        *,
        projected_key: RuntimeMeasurementProjectedFeatureCacheKey | None = None,
    ) -> bool:
        """Return whether a cell under ``key`` could produce a required value."""
        if self.required_keys is None:
            return True
        if self.requires_key(key, projected_key=projected_key):
            return True
        return self.nested_base_for_key(key) in self.nested_bases


@dataclass(slots=True)
class RuntimeIndexedRowValues:
    """Typed accessors for row values indexed by schema positions."""

    row_values: tuple[object, ...] | None = None
    row_mapping: Mapping[str, object] | None = None
    header: tuple[str, ...] = ()
    _value_cache: dict[int, object | None] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    @classmethod
    def from_row(
        cls,
        row: RuntimeMeasurementRowMapping,
    ) -> "RuntimeIndexedRowValues":
        return cls(row_mapping=row.row, header=row.header)

    def at(self, index: int | None) -> object | None:
        if index is None:
            return None
        if self.row_values is not None:
            return self.row_values[index]
        cached = self._value_cache.get(index, _CACHE_MISS)
        if cached is not _CACHE_MISS:
            return cached
        if self.row_mapping is None:
            raise ValueError(
                "RuntimeIndexedRowValues requires row_values or row_mapping."
            )
        value = self.row_mapping.get(self.header[index])
        self._value_cache[index] = value
        return value

    def values_at(self, indexes: tuple[int | None, ...]) -> tuple[object | None, ...]:
        return tuple(self.at(index) for index in indexes)

    def first_at(self, indexes: tuple[int, ...]) -> object | None:
        if not indexes:
            return None
        return self.at(indexes[0])

    def text_at(self, index: int | None) -> str | None:
        value = self.at(index)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def has_text_at(self, index: int | None) -> bool:
        return self.text_at(index) is not None

    def has_text_at_any(self, indexes: tuple[int, ...]) -> bool:
        return any(self.has_text_at(index) for index in indexes)

    def has_present_at_any(self, indexes: tuple[int, ...]) -> bool:
        """Return whether any indexed cell contains a runtime measurement value."""
        return any(
            runtime_measurement_value_is_present(self.at(index)) for index in indexes
        )


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementRowMapping:
    """Nominal row boundary for runtime measurement-row semantics."""

    row: Mapping[str, object]
    object_row_identity: MeasurementObjectRowIdentity | None = None
    _header: tuple[str, ...] = field(init=False, repr=False, compare=False)
    _values: tuple[object, ...] | None = field(init=False, repr=False, compare=False)
    _normalized_fields: Mapping[str, str] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _normalized_field_names: frozenset[str] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _declared_values: dict[type[MeasurementRowDeclaredValue], object | None] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _image_identity_key_cache: dict[
        int,
        tuple[tuple[str, object], ...],
    ] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        header = tuple(self.row)
        header_projection = runtime_measurement_row_header_projection(header)
        object.__setattr__(self, "_header", header)
        object.__setattr__(self, "_values", None)
        object.__setattr__(
            self,
            "_normalized_fields",
            header_projection.normalized_fields,
        )
        object.__setattr__(
            self,
            "_normalized_field_names",
            header_projection.normalized_field_names,
        )

    @property
    def header(self) -> tuple[str, ...]:
        return self._header

    @property
    def values(self) -> tuple[object, ...]:
        if self._values is None:
            object.__setattr__(
                self,
                "_values",
                tuple(self.row.get(field_name) for field_name in self.header),
            )
        return self._values

    @property
    def normalized_fields(self) -> Mapping[str, str]:
        return self._normalized_fields

    @property
    def normalized_field_names(self) -> frozenset[str]:
        return self._normalized_field_names

    def first_value(self, field_names: tuple[str, ...]) -> object | None:
        normalized_fields = self.normalized_fields
        for field_name in field_names:
            field = normalized_fields.get(field_name)
            if field is not None:
                return self.row[field]
        return None

    def has_field(self, field_name: str) -> bool:
        return field_name in self.row

    def has_identity_value(self, field_names: frozenset[str]) -> bool:
        normalized_fields = self.normalized_fields
        for field_name in field_names:
            field = normalized_fields.get(field_name)
            if field is None:
                continue
            value = self.row[field]
            if value is None:
                continue
            if str(value).strip():
                return True
        return False

    def has_image_identity(self) -> bool:
        return self.has_identity_value(IMAGE_IDENTITY_FIELDS)

    def has_object_identity(self) -> bool:
        return self.object_label() is not None

    def has_long_form_measurement_fields(self) -> bool:
        return measurement_row_has_long_form_measurement_fields(self.row)

    def declared_value(
        self,
        declaration_type: type[MeasurementRowDeclaredValue],
    ) -> object | None:
        """Return a declared row value from the nominal declaration cache."""
        if declaration_type not in self._declared_values:
            self._declared_values[declaration_type] = declaration_type.value_from_row(
                self.row,
                normalized_fields=self.normalized_fields,
            )
        return self._declared_values[declaration_type]

    def source_name(self) -> str | None:
        return cast(str | None, self.declared_value(MeasurementRowSourceImageName))

    def object_name(self) -> str | None:
        return cast(str | None, self.declared_value(MeasurementRowObjectName))

    def object_label(self) -> int | None:
        return cast(int | None, self.declared_value(MeasurementRowObjectLabel))

    def object_identity_value(
        self,
        dialect: RuntimeMeasurementDialect,
    ) -> tuple[str, object] | None:
        """Return the object-identity field and value selected by the dialect."""
        selected_field = dialect.row_identity_contract.selected_object_identity_field(
            self.normalized_field_names
        )
        if selected_field is None:
            return None
        field_name = self.normalized_fields[selected_field]
        value = self.row[field_name]
        if not runtime_measurement_value_is_present(value):
            return None
        return field_name, value

    def identity_role(self) -> MeasurementObjectRowIdentity | None:
        return cast(
            MeasurementObjectRowIdentity | None,
            MeasurementRowObjectIdentityRole.resolve_value_from_row(
                self.row,
                carrier_identity=self.object_row_identity,
                normalized_fields=self.normalized_fields,
            ),
        )

    def image_identity_key(
        self,
        dialect: RuntimeMeasurementDialect = DEFAULT_RUNTIME_MEASUREMENT_DIALECT,
    ) -> tuple[tuple[str, object], ...]:
        cache_key = id(dialect)
        cached = self._image_identity_key_cache.get(cache_key)
        if cached is not None:
            return cached
        contract = dialect.row_identity_contract
        normalized_present_fields = frozenset(
            normalized_field_name
            for normalized_field_name, field_name in self.normalized_fields.items()
            for value in (self.row[field_name],)
            if runtime_measurement_row_value_is_present_without_formatting(value)
        )
        selected_fields = contract.selected_image_identity_fields(
            normalized_present_fields
        )
        identity_values = tuple(
            sorted(
                (
                    normalized_field_name,
                    measurement_table_cell_payload(self.row[field_name]),
                )
                for normalized_field_name, field_name in self.normalized_fields.items()
                if normalized_field_name in selected_fields
            )
        )
        self._image_identity_key_cache[cache_key] = identity_values
        return identity_values

    def axis_scoped_identity(
        self,
        axis_key: str | None,
        dialect: RuntimeMeasurementDialect = DEFAULT_RUNTIME_MEASUREMENT_DIALECT,
    ) -> RuntimeMeasurementRowIdentity:
        row_identity = self.image_identity_key(dialect)
        if axis_key is None:
            return row_identity
        return (
            (RUNTIME_AXIS_ROW_IDENTITY_FIELD, measurement_table_cell_payload(axis_key)),
            *row_identity,
        )


@dataclass(frozen=True, slots=True)
class RuntimeObjectMeasurementRowIdentity:
    """Nominal identity for one object measurement row within a runtime axis."""

    row_identity: RuntimeMeasurementRowIdentity
    occurrence_scope: ComponentGroupScope | None = None

    def without_occurrence_scope(self) -> "RuntimeObjectMeasurementRowIdentity":
        """Return this row identity independent of invocation grouping."""
        if self.occurrence_scope is None:
            return self
        return type(self)(self.row_identity)

    @classmethod
    def from_row(
        cls,
        row: RuntimeMeasurementRowMapping,
        axis_key: str | None,
        policy: RuntimeEquivalencePolicy,
    ) -> "RuntimeObjectMeasurementRowIdentity | None":
        object_label = row.object_label()
        if object_label is None:
            return None
        return cls(
            (
                *row.axis_scoped_identity(
                    axis_key,
                    policy.measurement_dialect,
                ),
                (
                    OBJECT_LABEL_ROW_IDENTITY_FIELD,
                    RuntimeMeasurementCellSignatureProjection(
                        object_label,
                        policy,
                    ).signature(),
                ),
            )
        )

    @classmethod
    def from_object_instance(
        cls,
        row: RuntimeMeasurementRowMapping,
        axis_key: str | None,
        policy: RuntimeEquivalencePolicy,
        object_instance_key: ObjectInstanceKey,
        *,
        occurrence_scope: ComponentGroupScope | None = None,
    ) -> "RuntimeObjectMeasurementRowIdentity":
        row_identity = row.axis_scoped_identity(
            axis_key,
            policy.measurement_dialect,
        )
        if object_instance_key.slice_index is not None:
            image_identity_fields = (
                policy.measurement_dialect.row_identity_contract.image_identity_fields
            )
            row_identity = (
                *(
                    field
                    for field in row_identity
                    if field[0] not in image_identity_fields
                ),
                (RUNTIME_SLICE_ROW_IDENTITY_FIELD, object_instance_key.slice_index),
            )
        return cls(
            (
                *row_identity,
                (
                    OBJECT_LABEL_ROW_IDENTITY_FIELD,
                    RuntimeMeasurementCellSignatureProjection(
                        object_instance_key.object_id,
                        policy,
                    ).signature(),
                ),
            ),
            occurrence_scope,
        )

    @property
    def image_identity(self) -> RuntimeMeasurementRowIdentity:
        return tuple(
            field
            for field in self.row_identity
            if field[0] != OBJECT_LABEL_ROW_IDENTITY_FIELD
        )

    @property
    def has_image_identity(self) -> bool:
        return any(
            field[0] in IMAGE_IDENTITY_FIELDS
            or field[0]
            in (
                RUNTIME_AXIS_ROW_IDENTITY_FIELD,
                RUNTIME_SLICE_ROW_IDENTITY_FIELD,
            )
            for field in self.row_identity
        )

    @property
    def object_label_signature(self) -> RuntimeCellSignature | None:
        return object_label_signature_from_row_identity(self.row_identity)


def runtime_metadata_map_row_matches(
    subject: RuntimeMeasurementSubjectKey,
    row: RuntimeMeasurementRowMapping,
) -> bool:
    """Return whether a row is an experiment metadata key/value row."""
    if subject.scope is not MeasurementScope.EXPERIMENT:
        return False
    return row.normalized_field_names == frozenset(("key", "value"))


def runtime_measurement_identity_field_matches(
    field_name: str,
    dialect: RuntimeMeasurementDialect,
) -> bool:
    """Return whether a field is row identity, qualifier, or metadata."""
    return normalized_runtime_measurement_identity_field_matches(
        normalize_runtime_identifier(field_name),
        dialect,
    )


def normalized_runtime_measurement_identity_field_matches(
    normalized: str,
    dialect: RuntimeMeasurementDialect,
) -> bool:
    """Return whether a normalized field is row identity, qualifier, or metadata."""
    if normalized in MEASUREMENT_IDENTITY_FIELDS:
        return True
    if normalized in measurement_qualifier_field_names(dialect):
        return True
    if normalized.startswith(dialect.non_measurement_field_prefixes):
        return True
    return normalized.startswith("metadata_")


RuntimeProjectedCell = tuple[
    RuntimeMeasurementFeatureKey,
    RuntimeRowProjectionValueT,
]
RuntimeProjectedCells = tuple[
    RuntimeProjectedCell[RuntimeRowProjectionValueT],
    ...,
]


@dataclass(frozen=True, slots=True)
class RuntimeRowProjection(Generic[RuntimeRowProjectionValueT]):
    records: RuntimeRowProjectionRecords[RuntimeRowProjectionValueT]
    long_form: bool = False
    declared_anchor_groups: frozenset[RuntimeMeasurementPaddingGroup] = frozenset()


def runtime_row_projection(
    records: Iterable[RuntimeRowProjectionRecord[RuntimeRowProjectionValueT]] = (),
    *,
    long_form: bool = False,
    declared_anchor_groups: frozenset[RuntimeMeasurementPaddingGroup] = frozenset(),
) -> RuntimeRowProjection[RuntimeRowProjectionValueT]:
    """Build a row projection through one normalized record boundary."""
    return RuntimeRowProjection(
        tuple(records),
        long_form=long_form,
        declared_anchor_groups=declared_anchor_groups,
    )


@dataclass(frozen=True, slots=True)
class RuntimeRowProjectionContext:
    row: RuntimeMeasurementRowMapping
    subject: RuntimeMeasurementSubjectKey
    policy: RuntimeEquivalencePolicy
    measurement_feature_owner: type[RuntimeMeasurementFeatureOwner] | None
    source_name: str | None
    known_source_names: tuple[str, ...]
    required_keys: RuntimeRequiredMeasurementKeys
    table_padding_group: str
    image_number_offset: RuntimeImageNumberOffset
    derive_directional_pair_facts: bool
    schema_cache: RuntimeMeasurementRowSchemaCache
    key_cache: RuntimeMeasurementFeatureKeyCache
    wide_feature_index_cache: RuntimeMeasurementWideFeatureIndexCache
    wide_feature_plan_cache: RuntimeMeasurementWideFeaturePlanCache
    qualifier_render_cache: RuntimeMeasurementQualifierRenderCache
    padding_group_cache: RuntimeMeasurementPaddingGroupCache
    required_key_index: RuntimeMeasurementRequiredKeyIndex
    table_schema: RuntimeMeasurementRowSchema

    @classmethod
    def from_row(
        cls,
        row: RuntimeMeasurementRowMapping,
        subject: RuntimeMeasurementSubjectKey,
        policy: RuntimeEquivalencePolicy,
        *,
        measurement_feature_owner: type[RuntimeMeasurementFeatureOwner] | None,
        source_name: str | None,
        known_source_names: tuple[str, ...],
        required_keys: RuntimeRequiredMeasurementKeys,
        table_padding_group: str,
        image_number_offset: RuntimeImageNumberOffset,
        derive_directional_pair_facts: bool,
        schema_cache: RuntimeMeasurementRowSchemaCache,
        key_cache: RuntimeMeasurementFeatureKeyCache,
        wide_feature_index_cache: RuntimeMeasurementWideFeatureIndexCache,
        wide_feature_plan_cache: RuntimeMeasurementWideFeaturePlanCache,
        qualifier_render_cache: RuntimeMeasurementQualifierRenderCache,
        padding_group_cache: RuntimeMeasurementPaddingGroupCache,
        required_key_index: RuntimeMeasurementRequiredKeyIndex,
        table_schema: RuntimeMeasurementRowSchema,
    ) -> "RuntimeRowProjectionContext":
        return cls(
            row=row,
            subject=subject,
            policy=policy,
            measurement_feature_owner=measurement_feature_owner,
            source_name=source_name,
            known_source_names=known_source_names,
            required_keys=required_keys,
            table_padding_group=table_padding_group,
            image_number_offset=image_number_offset,
            derive_directional_pair_facts=derive_directional_pair_facts,
            schema_cache=schema_cache,
            key_cache=key_cache,
            wide_feature_index_cache=wide_feature_index_cache,
            wide_feature_plan_cache=wide_feature_plan_cache,
            qualifier_render_cache=qualifier_render_cache,
            padding_group_cache=padding_group_cache,
            required_key_index=required_key_index,
            table_schema=table_schema,
        )

    def subject_for_field_index(
        self,
        index: int,
    ) -> RuntimeMeasurementSubjectKey:
        del index
        return self.subject

    def padding_indexes(
        self,
        row_schema: RuntimeMeasurementRowSchema,
        row_values: RuntimeIndexedRowValues,
    ) -> frozenset[int]:
        del row_schema, row_values
        return frozenset()

    def supports_static_wide_projection(self) -> bool:
        """Return whether wide-column semantics are invariant across rows."""
        return True

    def project(
        self,
        value_projector: RuntimeRowValueProjection[RuntimeRowProjectionValueT],
        long_form_projector: RuntimeRowLongFormProjection[RuntimeRowProjectionValueT],
    ) -> RuntimeRowProjection[RuntimeRowProjectionValueT]:
        """Project this runtime row through shared schema/key/padding caches."""
        if runtime_metadata_map_row_matches(self.subject, self.row):
            return runtime_row_projection()

        header = self.row.header
        row_schema = self.row_schema(header)
        row_values = RuntimeIndexedRowValues.from_row(self.row)
        long_form_projection = self._long_form_projection(
            row_schema,
            row_values,
            long_form_projector,
        )
        if long_form_projection is not None:
            return long_form_projection
        return self._wide_projection(header, row_schema, row_values, value_projector)

    def fact_projection(
        self,
        *,
        covers_declared_object_measurement_domain: bool = False,
    ) -> RuntimeRowProjection[RuntimeCellSignature]:
        """Project this row while preserving declared alias ownership."""
        projection = self.project(
            RuntimeMeasurementCellFactProjection(),
            RuntimeRowLongFormFactProjection(),
        )
        records = projection.records
        if (
            self.subject.scope is MeasurementScope.OBJECT
            and not self.table_schema.has_measurement_features
        ):
            object_identity = self.row.object_identity_value(
                self.policy.measurement_dialect
            )
            if object_identity is not None:
                field_name, value = object_identity
                key = self._feature_key(field_name, self.subject, ())
                if key is not None and (
                    self.required_keys is None or key in self.required_keys
                ):
                    records = (
                        *records,
                        RuntimeRowProjectionRecord(
                            self._padding_group(field_name, key),
                            key,
                            runtime_measurement_cell_signature(value, self.policy),
                            measured_object_anchor=True,
                        ),
                    )
        if not (
            covers_declared_object_measurement_domain
            and self.subject.scope is MeasurementScope.OBJECT
        ):
            records = RuntimeMeasurementFactProjectionContract.observed_records(
                records,
                self.policy,
                declared_anchor_groups=projection.declared_anchor_groups,
            )
        projection = runtime_row_projection(
            records,
            long_form=projection.long_form,
            declared_anchor_groups=projection.declared_anchor_groups,
        )
        row_facts = RuntimeMeasurementFactProjectionContract.dedupe_records(
            projection.records
        )
        if projection.long_form:
            return projection
        if not self.derive_directional_pair_facts:
            return projection
        derived_facts = RuntimeDirectionalPairMeasurementDerivationContract(
            self.policy,
            self.known_source_names,
        ).derive(row_facts)
        explicit_facts = frozenset(row_facts)
        return runtime_row_projection(
            (
                *projection.records,
                *(
                    RuntimeRowProjectionRecord(
                        RuntimeMeasurementFactProjectionContract.padding_group(
                            self.table_padding_group,
                            key.feature_name,
                            key,
                            self.policy.measurement_dialect,
                        ),
                        key,
                        value,
                    )
                    for key, value in derived_facts
                    if (key, value) not in explicit_facts
                    and (self.required_keys is None or key in self.required_keys)
                ),
            )
        )

    def numeric_values(self) -> RuntimeNumericMeasurementValues:
        """Project numeric runtime row values without building cell signatures."""
        projection = self.project(
            RuntimeMeasurementCellNumericProjection(),
            RuntimeRowLongFormNumericProjection(),
        )
        row_values_by_key = dedupe_numeric_measurement_values(
            (key, value)
            for record in projection.records
            for key, value in ((record.key, record.value),)
        )
        if projection.long_form:
            return row_values_by_key
        if not self.derive_directional_pair_facts:
            return row_values_by_key

        derived_facts = RuntimeDirectionalPairMeasurementDerivationContract(
            self.policy,
            self.known_source_names,
        ).derive(
            tuple(
                (key, runtime_cell_signature(repr(value), self.policy))
                for key, value in row_values_by_key
            ),
        )
        if self.required_keys is not None:
            derived_facts = tuple(
                (key, value)
                for key, value in derived_facts
                if key in self.required_keys
            )
        return tuple(
            (key, numeric_value)
            for key, value in derived_facts
            if (numeric_value := cell_signature_numeric_value(value)) is not None
        )

    def row_schema(self, header: tuple[str, ...]) -> RuntimeMeasurementRowSchema:
        """Return this row header's cached projection schema."""
        cached_schema = self.schema_cache.get(header)
        if cached_schema is not None:
            return cached_schema

        cached_schema = runtime_measurement_row_schema_for_header(
            header,
            self.policy.measurement_dialect.row_qualifiers,
            self.policy.measurement_dialect.row_identity_contract,
            self.policy.measurement_dialect.non_measurement_field_prefixes,
        )
        self.schema_cache[header] = cached_schema
        return cached_schema

    def _long_form_projection(
        self,
        row_schema: RuntimeMeasurementRowSchema,
        row_values: RuntimeIndexedRowValues,
        long_form_projector: RuntimeRowLongFormProjection[RuntimeRowProjectionValueT],
    ) -> RuntimeRowProjection[RuntimeRowProjectionValueT] | None:
        if (
            not row_schema.long_form_feature_indexes
            or not row_schema.long_form_value_indexes
        ):
            return None
        long_form_fact = RuntimeMeasurementLongFormFactProjector(
            CachedRuntimeLongFormMeasurementContext.from_runtime_row_projection(
                self,
                row_values,
                row_schema.long_form_feature_indexes,
                row_schema.long_form_value_indexes,
                row_schema.long_form_qualifiers,
            )
        ).fact()
        if long_form_fact is None:
            return None
        if (
            self.required_keys is not None
            and long_form_fact[0] not in self.required_keys
        ):
            return runtime_row_projection(long_form=True)
        return runtime_row_projection(
            (
                RuntimeRowProjectionRecord(
                    (self.subject, self.source_name, ()),
                    key,
                    value,
                    measured_object_anchor=self._measured_object_anchor(key),
                )
                for key, value in long_form_projector.project(long_form_fact)
                if self.required_keys is None or key in self.required_keys
            ),
            long_form=True,
        )

    def _wide_projection(
        self,
        header: tuple[str, ...],
        row_schema: RuntimeMeasurementRowSchema,
        row_values: RuntimeIndexedRowValues,
        value_projector: RuntimeRowValueProjection[RuntimeRowProjectionValueT],
    ) -> RuntimeRowProjection[RuntimeRowProjectionValueT]:
        records: list[RuntimeRowProjectionRecord[RuntimeRowProjectionValueT]] = []
        padding_group_presence: RuntimeMeasurementPaddingGroupPresence = {}
        row_qualifier_cache: RuntimeRowQualifierResolutionCache = {}
        padding_indexes = self.padding_indexes(row_schema, row_values)
        static_columns = self.static_wide_projection_columns(header, row_schema)
        static_column_indexes = tuple(
            column.index
            for column in static_columns
            if column.index not in padding_indexes
        )
        if static_column_indexes and not row_values.has_present_at_any(
            static_column_indexes
        ):
            return runtime_row_projection()
        for column in static_columns:
            if column.index in padding_indexes:
                continue
            records.extend(
                self._wide_projection_records_for_static_column(
                    row_values,
                    column,
                    padding_group_presence,
                    value_projector,
                )
            )
        for index in self.wide_feature_indexes(header, row_schema):
            if static_columns and not row_schema.qualifiers_by_index[index]:
                continue
            if index in padding_indexes:
                continue
            records.extend(
                self._wide_projection_records_for_index(
                    header,
                    row_schema,
                    row_values,
                    index,
                    row_qualifier_cache,
                    padding_group_presence,
                    value_projector,
                )
            )
        declared_anchor_groups = {
            column.padding_group
            for column in static_columns
            if column.measured_object_anchor
        }
        for index in self.wide_feature_indexes(header, row_schema):
            if not row_schema.qualifiers_by_index[index]:
                continue
            qualifiers = self._qualifiers_for_index(
                row_schema,
                row_values,
                index,
                row_qualifier_cache,
            )
            key = self._feature_key(
                header[index],
                self.subject_for_field_index(index),
                qualifiers,
            )
            if key is None or not self._measured_object_anchor(key):
                continue
            declared_anchor_groups.add(
                self._padding_group(
                    header[index],
                    key,
                    runtime_measurement_projected_feature_cache_key(key),
                )
            )
        return runtime_row_projection(
            (
                record
                for record in records
                if padding_group_presence.get(record.padding_group, True)
            ),
            declared_anchor_groups=frozenset(declared_anchor_groups),
        )

    def static_wide_projection_columns(
        self,
        header: tuple[str, ...],
        row_schema: RuntimeMeasurementRowSchema,
    ) -> tuple[RuntimeWideProjectionColumn, ...]:
        """Return planned wide columns whose semantic keys do not vary by row."""
        if not self.supports_static_wide_projection():
            return ()
        cache_key: RuntimeMeasurementWideFeaturePlanCacheKey = (
            header,
            self.subject.scope,
            self.subject.name,
            self.source_name,
            self.measurement_feature_owner,
            self.table_padding_group,
            id(self.required_key_index),
        )
        cached = self.wide_feature_plan_cache.get(cache_key)
        if cached is not None:
            return cached
        columns: list[RuntimeWideProjectionColumn] = []
        for index in self.wide_feature_indexes(header, row_schema):
            if row_schema.qualifiers_by_index[index]:
                continue
            field_name = header[index]
            subject = self.subject_for_field_index(index)
            key = self._feature_key(field_name, subject, ())
            if key is None:
                continue
            projected_key = runtime_measurement_projected_feature_cache_key(key)
            if not self.required_key_index.may_require_value(
                key,
                projected_key=projected_key,
            ):
                continue
            columns.append(
                RuntimeWideProjectionColumn(
                    index=index,
                    field_name=field_name,
                    key=key,
                    projected_key=projected_key,
                    padding_group=self._padding_group(
                        field_name,
                        key,
                        projected_key,
                    ),
                    measured_object_anchor=self._measured_object_anchor(key),
                    producer_owned_feature=self._producer_owns_feature(field_name),
                )
            )
        planned_columns = tuple(columns)
        self.wide_feature_plan_cache[cache_key] = planned_columns
        return planned_columns

    def wide_feature_indexes(
        self,
        header: tuple[str, ...],
        row_schema: RuntimeMeasurementRowSchema,
    ) -> tuple[int, ...]:
        """Return feature indexes that can emit required semantic keys."""
        if self.required_key_index.required_keys is None:
            return row_schema.feature_indexes
        cache_key: RuntimeMeasurementWideFeatureIndexCacheKey = (
            header,
            self.subject.scope,
            self.subject.name,
            self.source_name,
            id(self.required_key_index),
        )
        cached = self.wide_feature_index_cache.get(cache_key)
        if cached is not None:
            return cached
        indexes = tuple(
            index
            for index in row_schema.feature_indexes
            if self.wide_feature_index_may_emit_required_key(header, row_schema, index)
        )
        self.wide_feature_index_cache[cache_key] = indexes
        return indexes

    def wide_feature_index_may_emit_required_key(
        self,
        header: tuple[str, ...],
        row_schema: RuntimeMeasurementRowSchema,
        index: int,
    ) -> bool:
        """Return whether a wide-table column can satisfy the required-key index."""
        key = self._feature_key(
            header[index],
            self.subject_for_field_index(index),
            (),
        )
        if key is None:
            return False
        projected_key = runtime_measurement_projected_feature_cache_key(key)
        return self.required_key_index.may_require_value(
            key,
            projected_key=projected_key,
        )

    def _wide_projection_records_for_index(
        self,
        header: tuple[str, ...],
        row_schema: RuntimeMeasurementRowSchema,
        row_values: RuntimeIndexedRowValues,
        index: int,
        row_qualifier_cache: RuntimeRowQualifierResolutionCache,
        padding_group_presence: RuntimeMeasurementPaddingGroupPresence,
        value_projector: RuntimeRowValueProjection[RuntimeRowProjectionValueT],
    ) -> RuntimeRowProjectionRecords[RuntimeRowProjectionValueT]:
        field_name = header[index]
        raw_value = row_values.at(index)
        qualifiers = self._qualifiers_for_index(
            row_schema,
            row_values,
            index,
            row_qualifier_cache,
        )
        subject = self.subject_for_field_index(index)
        key = self._feature_key(field_name, subject, qualifiers)
        if key is None:
            return ()
        projected_key = runtime_measurement_projected_feature_cache_key(key)
        if not self.required_key_index.may_require_value(
            key,
            projected_key=projected_key,
        ):
            return ()
        if not runtime_measurement_value_is_present(raw_value):
            return ()
        value = self.image_number_offset.normalized_reference_value(
            field_name,
            raw_value,
        )
        value_is_mapping = isinstance(value, Mapping)
        if not self.required_key_index.requires_value(
            key,
            value_is_mapping=value_is_mapping,
            projected_key=projected_key,
        ):
            return ()
        padding_group = self._padding_group(field_name, key, projected_key)
        padding_group_presence[padding_group] = True
        projected_values = value_projector.project(
            key,
            value,
            self.policy,
            required_keys=self.required_keys,
            required_key_index=self.required_key_index,
            value_is_present=True,
            value_is_mapping=value_is_mapping,
        )
        return tuple(
            RuntimeRowProjectionRecord(
                padding_group,
                cell_key,
                cell_value,
                self._measured_object_anchor(cell_key),
                self._producer_owns_feature(field_name),
            )
            for cell_key, cell_value in projected_values
        )

    def _wide_projection_records_for_static_column(
        self,
        row_values: RuntimeIndexedRowValues,
        column: RuntimeWideProjectionColumn,
        padding_group_presence: RuntimeMeasurementPaddingGroupPresence,
        value_projector: RuntimeRowValueProjection[RuntimeRowProjectionValueT],
    ) -> RuntimeRowProjectionRecords[RuntimeRowProjectionValueT]:
        raw_value = row_values.at(column.index)
        if not runtime_measurement_value_is_present(raw_value):
            return ()
        value = self.image_number_offset.normalized_reference_value(
            column.field_name,
            raw_value,
        )
        value_is_mapping = isinstance(value, Mapping)
        if not self.required_key_index.requires_value(
            column.key,
            value_is_mapping=value_is_mapping,
            projected_key=column.projected_key,
        ):
            return ()
        if isinstance(value_projector, RuntimeMeasurementCellFactProjection):
            if value_is_mapping:
                projected_values = value_projector.project(
                    column.key,
                    value,
                    self.policy,
                    required_keys=self.required_keys,
                    required_key_index=self.required_key_index,
                    value_is_mapping=True,
                )
            else:
                signature = runtime_measurement_cell_signature_if_present(
                    value,
                    self.policy,
                )
                if signature is None:
                    return ()
                padding_group_presence[column.padding_group] = True
                return cast(
                    RuntimeRowProjectionRecords[RuntimeRowProjectionValueT],
                    (
                        RuntimeRowProjectionRecord(
                            column.padding_group,
                            column.key,
                            signature,
                            column.measured_object_anchor,
                            column.producer_owned_feature,
                        ),
                    ),
                )
            padding_group_presence[column.padding_group] = True
            return tuple(
                RuntimeRowProjectionRecord(
                    column.padding_group,
                    cell_key,
                    cell_value,
                    column.measured_object_anchor,
                    column.producer_owned_feature,
                )
                for cell_key, cell_value in projected_values
            )
        if (
            isinstance(value_projector, RuntimeMeasurementCellNumericProjection)
            and not value_is_mapping
        ):
            numeric_value = measurement_numeric_runtime_value(value, self.policy)
            if numeric_value is None:
                return ()
            padding_group_presence[column.padding_group] = True
            return cast(
                RuntimeRowProjectionRecords[RuntimeRowProjectionValueT],
                (
                    RuntimeRowProjectionRecord(
                        column.padding_group,
                        column.key,
                        numeric_value,
                        column.measured_object_anchor,
                        column.producer_owned_feature,
                    ),
                ),
            )
        padding_group_presence[column.padding_group] = True
        projected_values = value_projector.project(
            column.key,
            value,
            self.policy,
            required_keys=self.required_keys,
            required_key_index=self.required_key_index,
            value_is_present=True,
            value_is_mapping=value_is_mapping,
        )
        return tuple(
            RuntimeRowProjectionRecord(
                column.padding_group,
                cell_key,
                cell_value,
                column.measured_object_anchor,
                column.producer_owned_feature,
            )
            for cell_key, cell_value in projected_values
        )

    def _producer_owns_feature(self, field_name: str) -> bool:
        owner = self.measurement_feature_owner
        return (
            owner.owns_primary_measurement_feature_name(field_name)
            if owner is not None
            else self.policy.measurement_dialect.feature_name_has_primary_category(
                field_name
            )
        )

    def _measured_object_anchor(
        self,
        key: RuntimeMeasurementFeatureKey,
    ) -> bool:
        return any(
            issubclass(marker_type, MeasuredObjectAnchorFeatureMarker)
            for marker_type in self.policy.measurement_dialect.measurement_feature_marker_types(
                key
            )
        )

    def _qualifiers_for_index(
        self,
        row_schema: RuntimeMeasurementRowSchema,
        row_values: RuntimeIndexedRowValues,
        index: int,
        row_qualifier_cache: RuntimeRowQualifierResolutionCache,
    ) -> tuple[str, ...]:
        indexed_qualifiers = row_schema.qualifiers_by_index[index]
        if not indexed_qualifiers:
            return ()
        cache_key = id(indexed_qualifiers)
        cached = row_qualifier_cache.get(cache_key)
        if cached is not None and cached[0] is indexed_qualifiers:
            return cached[1]
        qualifiers = measurement_row_qualifiers_from_indexed_values_cached(
            row_values,
            indexed_qualifiers,
            self.qualifier_render_cache,
        )
        row_qualifier_cache[cache_key] = (indexed_qualifiers, qualifiers)
        return qualifiers

    def _feature_key(
        self,
        field_name: str,
        subject: RuntimeMeasurementSubjectKey,
        qualifiers: tuple[str, ...],
    ) -> RuntimeMeasurementFeatureKey | None:
        cache_key = runtime_measurement_feature_key_cache_key(
            subject,
            self.source_name,
            field_name,
            qualifiers,
        )
        key = self.key_cache.get(cache_key, _CACHE_MISS)
        if key is _CACHE_MISS:
            key = RuntimeMeasurementFeatureKeyProjection(
                RuntimeMeasurementFeatureKeySourceContext(
                    field_name,
                    subject,
                    self.policy,
                    qualifiers,
                    self.source_name,
                    self.known_source_names,
                )
            ).key()
            self.key_cache[cache_key] = key
        return key

    def _padding_group(
        self,
        field_name: str,
        key: RuntimeMeasurementFeatureKey,
        projected_key: RuntimeMeasurementProjectedFeatureCacheKey | None = None,
    ) -> RuntimeMeasurementPaddingGroup:
        if projected_key is None:
            projected_key = runtime_measurement_projected_feature_cache_key(key)
        cache_key = (field_name, projected_key)
        padding_group = self.padding_group_cache.get(cache_key)
        if padding_group is None:
            padding_group = RuntimeMeasurementFactProjectionContract.padding_group(
                self.table_padding_group,
                field_name,
                key,
                self.policy.measurement_dialect,
            )
            self.padding_group_cache[cache_key] = padding_group
        return padding_group


class RuntimeRowValueProjection(
    ABC,
    Generic[RuntimeRowProjectionValueT],
):
    """Project wide-form runtime measurement values for row fact extraction."""

    @abstractmethod
    def project(
        self,
        key: RuntimeMeasurementFeatureKey,
        value: object,
        policy: RuntimeEquivalencePolicy,
        *,
        required_keys: RuntimeRequiredMeasurementKeys = None,
        required_key_index: RuntimeMeasurementRequiredKeyIndex | None = None,
        value_is_present: bool | None = None,
        value_is_mapping: bool | None = None,
    ) -> RuntimeProjectedCells[RuntimeRowProjectionValueT]:
        """Project one wide-form cell into semantic values."""


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementCellValue:
    """Runtime measurement cell plus nested value expansion policy."""

    key: RuntimeMeasurementFeatureKey
    value: object
    policy: RuntimeEquivalencePolicy
    required_keys: frozenset[RuntimeMeasurementFeatureKey] | None = None
    required_key_index: RuntimeMeasurementRequiredKeyIndex | None = None
    value_is_mapping: bool | None = None

    def is_mapping(self) -> bool:
        if self.value_is_mapping is not None:
            return self.value_is_mapping
        return runtime_value_is_mapping(self.value)

    def iter_key_values(
        self,
    ) -> Iterable[tuple[RuntimeMeasurementFeatureKey, object]]:
        if not self.is_mapping():
            if not self.requires_key(self.key):
                return ()
            return ((self.key, self.value),)
        return tuple(
            (nested_key, nested_value)
            for name, nested_value in self.value.items()
            for nested_key in (self.nested_key(name),)
            if self.requires_key(nested_key)
        )

    def requires_key(self, key: RuntimeMeasurementFeatureKey) -> bool:
        """Return whether this cell projection should emit ``key``."""
        if self.required_key_index is not None:
            return self.required_key_index.requires_key(key)
        return self.required_keys is None or key in self.required_keys

    def nested_key(self, name: object) -> RuntimeMeasurementFeatureKey:
        return RuntimeMeasurementFeatureKey.from_subject_feature(
            self.key.subject,
            f"{self.key.feature_name}_{canonical_measurement_feature_name(str(name), self.policy)}",
            self.key.statistic,
            source_name=self.key.source_name,
        )


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementCellFactProjection(
    RuntimeRowValueProjection[RuntimeCellSignature],
):
    """Project runtime measurement cells into cell-signature facts."""

    def project(
        self,
        key: RuntimeMeasurementFeatureKey,
        value: object,
        policy: RuntimeEquivalencePolicy,
        *,
        required_keys: RuntimeRequiredMeasurementKeys = None,
        required_key_index: RuntimeMeasurementRequiredKeyIndex | None = None,
        value_is_present: bool | None = None,
        value_is_mapping: bool | None = None,
    ) -> RuntimeMeasurementFacts:
        if value_is_mapping is None:
            value_is_mapping = runtime_value_is_mapping(value)
        if not value_is_mapping:
            if required_key_index is not None:
                if not required_key_index.requires_key(key):
                    return ()
            elif required_keys is not None and key not in required_keys:
                return ()
            if value_is_present is False:
                return ()
            if value_is_present is True:
                signature = runtime_measurement_cell_signature(value, policy)
            else:
                signature = runtime_measurement_cell_signature_if_present(value, policy)
            if signature is None:
                return ()
            return ((key, signature),)
        cell = RuntimeMeasurementCellValue(
            key,
            value,
            policy,
            required_keys,
            required_key_index,
            value_is_mapping,
        )
        return self.project_cell(cell)

    def project_cell(
        self,
        cell: RuntimeMeasurementCellValue,
    ) -> RuntimeMeasurementFacts:
        return tuple(
            (
                cell_key,
                signature,
            )
            for cell_key, cell_value in cell.iter_key_values()
            for signature in (
                runtime_measurement_cell_signature_if_present(
                    cell_value,
                    cell.policy,
                ),
            )
            if signature is not None
        )


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementCellNumericProjection(
    RuntimeRowValueProjection[float],
):
    """Project runtime measurement cells into numeric values."""

    def project(
        self,
        key: RuntimeMeasurementFeatureKey,
        value: object,
        policy: RuntimeEquivalencePolicy,
        *,
        required_keys: RuntimeRequiredMeasurementKeys = None,
        required_key_index: RuntimeMeasurementRequiredKeyIndex | None = None,
        value_is_present: bool | None = None,
        value_is_mapping: bool | None = None,
    ) -> RuntimeNumericMeasurementValues:
        del value_is_present
        if value_is_mapping is None:
            value_is_mapping = runtime_value_is_mapping(value)
        if not value_is_mapping:
            if required_key_index is not None:
                if not required_key_index.requires_key(key):
                    return ()
            elif required_keys is not None and key not in required_keys:
                return ()
            numeric_value = measurement_numeric_runtime_value(value, policy)
            if numeric_value is None:
                return ()
            return ((key, numeric_value),)
        cell = RuntimeMeasurementCellValue(
            key,
            value,
            policy,
            required_keys,
            required_key_index,
            value_is_mapping,
        )
        return self.project_cell(cell)

    def project_cell(
        self,
        cell: RuntimeMeasurementCellValue,
    ) -> RuntimeNumericMeasurementValues:
        return tuple(
            (cell_key, numeric_value)
            for cell_key, cell_value in cell.iter_key_values()
            if (
                numeric_value := measurement_numeric_runtime_value(
                    cell_value,
                    cell.policy,
                )
            )
            is not None
        )


class RuntimeRowLongFormProjection(
    ABC,
    Generic[RuntimeRowProjectionValueT],
):
    """Project normalized long-form measurement facts for row extraction."""

    @abstractmethod
    def project(
        self,
        fact: RuntimeMeasurementFact,
    ) -> RuntimeProjectedCells[RuntimeRowProjectionValueT]:
        """Project one normalized long-form fact into semantic values."""


@dataclass(frozen=True, slots=True)
class RuntimeRowLongFormFactProjection(
    RuntimeRowLongFormProjection[RuntimeCellSignature],
):
    """Preserve long-form cell-signature facts as row facts."""

    def project(self, fact: RuntimeMeasurementFact) -> RuntimeMeasurementFacts:
        return (fact,)


@dataclass(frozen=True, slots=True)
class RuntimeRowLongFormNumericProjection(
    RuntimeRowLongFormProjection[float],
):
    """Project long-form cell-signature facts into numeric values."""

    def project(self, fact: RuntimeMeasurementFact) -> RuntimeNumericMeasurementValues:
        return numeric_long_form_measurement_values(fact)


@dataclass(frozen=True, slots=True)
class RuntimeLongFormMeasurementContext:
    """Runtime row context for long-form measurement extraction."""

    row: RuntimeMeasurementRowMapping
    subject: RuntimeMeasurementSubjectKey
    policy: RuntimeEquivalencePolicy
    source_name: str | None
    known_source_names: tuple[str, ...]
    image_number_offset: RuntimeImageNumberOffset


@dataclass(frozen=True, slots=True)
class CachedRuntimeLongFormMeasurementContext:
    row_values: RuntimeIndexedRowValues
    subject: RuntimeMeasurementSubjectKey
    policy: RuntimeEquivalencePolicy
    source_name: str | None
    known_source_names: tuple[str, ...]
    image_number_offset: RuntimeImageNumberOffset
    feature_indexes: tuple[int, ...]
    value_indexes: tuple[int, ...]
    indexed_qualifiers: tuple[RuntimeMeasurementIndexedQualifier, ...]
    qualifier_render_cache: RuntimeMeasurementQualifierRenderCache
    key_cache: RuntimeMeasurementFeatureKeyCache

    @classmethod
    def from_runtime_row_projection(
        cls,
        context: RuntimeRowProjectionContext,
        row_values: RuntimeIndexedRowValues,
        feature_indexes: tuple[int, ...],
        value_indexes: tuple[int, ...],
        indexed_qualifiers: tuple[RuntimeMeasurementIndexedQualifier, ...],
    ) -> "CachedRuntimeLongFormMeasurementContext":
        return cls(
            row_values,
            context.subject,
            context.policy,
            context.source_name,
            context.known_source_names,
            context.image_number_offset,
            feature_indexes,
            value_indexes,
            indexed_qualifiers,
            context.qualifier_render_cache,
            context.key_cache,
        )


@lru_cache(maxsize=32768)
def aggregate_image_number_reference_measurement_field(field_name: str) -> bool:
    parts = tuple(
        part for part in normalize_runtime_identifier(field_name).split("_") if part
    )
    return (
        bool(parts)
        and parts[0] == MeasurementStatistic.MEAN.value
        and image_number_reference_measurement_field(field_name)
    )


def image_number_reference_measurement_field(field_name: str) -> bool:
    normalized = normalize_runtime_identifier(field_name)
    if normalized in IMAGE_IDENTITY_FIELDS:
        return False
    parts = tuple(part for part in normalized.split("_") if part)
    return parts_contain_adjacent_image_number(parts)


def image_number_reference_feature(key: RuntimeMeasurementFeatureKey) -> bool:
    parts = tuple(part for part in key.feature_name.split("_") if part)
    if parts_contain_adjacent_image_number(parts):
        return True
    return key.source_name == "image" and "parent" in parts and "number" in parts


def parts_contain_adjacent_image_number(parts: tuple[str, ...]) -> bool:
    return any(
        parts[index] == "image" and parts[index + 1] == "number"
        for index in range(len(parts) - 1)
    )


@dataclass(frozen=True, slots=True)
class RuntimeLongFormMeasurementSource:
    """Valid feature/value pair extracted from a long-form measurement row."""

    feature_text: str
    value: object

    @classmethod
    def from_row(
        cls,
        row: RuntimeMeasurementRowMapping,
    ) -> "RuntimeLongFormMeasurementSource | None":
        feature_name = row.first_value(
            MeasurementRowAxisField.feature_name_field_names_ordered()
        )
        value = row.first_value(MeasurementRowValueField.field_names_ordered())
        return cls.from_feature_value(feature_name, value)

    @classmethod
    def from_indexed_values(
        cls,
        row_values: RuntimeIndexedRowValues,
        feature_indexes: tuple[int, ...],
        value_indexes: tuple[int, ...],
    ) -> "RuntimeLongFormMeasurementSource | None":
        return cls.from_feature_value(
            row_values.first_at(feature_indexes),
            row_values.first_at(value_indexes),
        )

    @classmethod
    def from_feature_value(
        cls,
        feature_name: object | None,
        value: object | None,
    ) -> "RuntimeLongFormMeasurementSource | None":
        if feature_name is None or value is None:
            return None
        feature_text = str(feature_name)
        if aggregate_image_number_reference_measurement_field(feature_text):
            return None
        return cls(feature_text, value)

    def cell_signature(
        self,
        image_number_offset: RuntimeImageNumberOffset,
        policy: RuntimeEquivalencePolicy,
    ) -> RuntimeCellSignature:
        normalized_value = image_number_offset.normalized_reference_value(
            self.feature_text,
            self.value,
        )
        return RuntimeMeasurementCellSignatureProjection(
            normalized_value,
            policy,
        ).signature()


@dataclass(frozen=True, slots=True)
class RuntimeLongFormMeasurementFact:
    """Resolved or missing long-form measurement fact."""

    key: RuntimeMeasurementFeatureKey | None
    value: RuntimeCellSignature | None

    @property
    def as_tuple(self) -> RuntimeLongFormMeasurementFactValue:
        if self.key is None or self.value is None:
            return None
        return self.key, self.value


@dataclass(frozen=True, slots=True)
class RuntimeSnapshotLongFormMeasurementFactProjector:
    """Project one snapshot long-form measurement row into a semantic fact."""

    context: RuntimeLongFormMeasurementContext

    def fact(self) -> RuntimeLongFormMeasurementFactValue:
        return self.resolved_fact().as_tuple

    def resolved_fact(self) -> RuntimeLongFormMeasurementFact:
        source = RuntimeLongFormMeasurementSource.from_row(self.context.row)
        if source is None:
            return RuntimeLongFormMeasurementFact(None, None)
        qualifiers = measurement_row_qualifiers(
            self.context.row.row,
            self.context.policy.measurement_dialect,
            source.feature_text,
        )
        key = RuntimeMeasurementFeatureKeyProjection(
            RuntimeMeasurementFeatureKeySourceContext(
                source.feature_text,
                self.context.subject,
                self.context.policy,
                qualifiers,
                self.context.source_name,
                self.context.known_source_names,
            ),
            strip_subject_suffix=False,
        ).key()
        if key is None:
            return RuntimeLongFormMeasurementFact(None, None)
        return RuntimeLongFormMeasurementFact(
            key,
            source.cell_signature(
                self.context.image_number_offset,
                self.context.policy,
            ),
        )


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementLongFormFactProjector:
    """Project one cached long-form measurement row into a semantic fact."""

    context: CachedRuntimeLongFormMeasurementContext

    def fact(self) -> RuntimeLongFormMeasurementFactValue:
        return self.resolved_fact().as_tuple

    def resolved_fact(self) -> RuntimeLongFormMeasurementFact:
        source = RuntimeLongFormMeasurementSource.from_indexed_values(
            self.context.row_values,
            self.context.feature_indexes,
            self.context.value_indexes,
        )
        if source is None:
            return RuntimeLongFormMeasurementFact(None, None)
        feature_parts = tuple(
            part
            for part in normalize_runtime_identifier(source.feature_text).split("_")
            if part
        )
        indexed_qualifiers = tuple(
            indexed_qualifier
            for indexed_qualifier in self.context.indexed_qualifiers
            if row_qualifier_applies_to_field(
                indexed_qualifier[0],
                feature_parts,
            )
        )
        qualifiers = measurement_row_qualifiers_from_indexed_values_cached(
            self.context.row_values,
            indexed_qualifiers,
            self.context.qualifier_render_cache,
        )
        cache_key = runtime_measurement_feature_key_cache_key(
            self.context.subject,
            self.context.source_name,
            source.feature_text,
            qualifiers,
        )
        key = self.context.key_cache.get(cache_key, _CACHE_MISS)
        if key is _CACHE_MISS:
            key = self._feature_key(source.feature_text, qualifiers)
            self.context.key_cache[cache_key] = key
        if key is None:
            return RuntimeLongFormMeasurementFact(None, None)
        return RuntimeLongFormMeasurementFact(
            key,
            source.cell_signature(
                self.context.image_number_offset,
                self.context.policy,
            ),
        )

    def _feature_key(
        self,
        feature_text: str,
        qualifiers: tuple[str, ...],
    ) -> RuntimeMeasurementFeatureKey | None:
        return RuntimeMeasurementFeatureKeyProjection(
            RuntimeMeasurementFeatureKeySourceContext(
                feature_text,
                self.context.subject,
                self.context.policy,
                qualifiers,
                self.context.source_name,
                self.context.known_source_names,
            ),
            strip_subject_suffix=False,
        ).key()


def numeric_long_form_measurement_values(
    fact: RuntimeMeasurementFact,
) -> RuntimeNumericMeasurementValues:
    key, cell_value = fact
    numeric_value = cell_signature_numeric_value(cell_value)
    if numeric_value is None:
        return ()
    return ((key, numeric_value),)


def dedupe_numeric_measurement_values(
    values: Iterable[tuple[RuntimeMeasurementFeatureKey, float]],
) -> RuntimeNumericMeasurementValues:
    values_by_key: dict[RuntimeMeasurementFeatureKey, float] = {}
    for key, value in values:
        values_by_key.setdefault(key, value)
    return tuple(values_by_key.items())


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementRowSubjectProjection:
    """Resolve runtime-row source and subject from nominal row identity."""

    table_subject: RuntimeMeasurementSubjectKey
    table_source_name: str | None
    row: RuntimeMeasurementRowMapping
    dialect: RuntimeMeasurementDialect

    def source_name(self) -> str | None:
        row_source_name = self.row.source_name()
        if row_source_name is not None:
            return row_source_name
        return self.table_source_name

    def subject(self) -> RuntimeMeasurementSubjectKey:
        object_identity = self.row.object_identity_value(self.dialect)
        object_name = self.row.object_name()
        if object_name is not None and object_identity is not None:
            return RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, object_name)
        if self.row.source_name() is not None and object_identity is None:
            return RuntimeMeasurementSubjectKey(
                MeasurementScope.IMAGE,
                MeasurementScope.IMAGE.value,
            )
        return self.table_subject
