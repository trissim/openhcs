"""Measurement row semantics for runtime equivalence."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import ClassVar, Generic

from metaclass_registry import RegistryFamily, RegistryKeyAttribute
from nominal_refactor_advisor.collection_algebra import sorted_tuple

from openhcs.core.equivalence.cells import (
    RuntimeCellSignature,
    RuntimeMeasurementCellPresence,
    RuntimeMeasurementCellSignatureProjection,
    RuntimeMeasurementValuePresence,
    cell_signature_numeric_value,
    measurement_numeric_runtime_value,
    measurement_table_cell_payload,
    runtime_cell_signature,
    runtime_numeric_text_value,
    runtime_value_is_mapping,
)
from openhcs.core.equivalence.keys import (
    RuntimeMeasurementFeatureKey,
    RuntimeMeasurementFeatureKeyProjection,
    RuntimeMeasurementFeatureKeySourceContext,
    RuntimeMeasurementNamePartsProjection,
    RuntimeMeasurementSubjectKey,
    SemanticCoreFeatureAndSourceNameProjection,
    canonical_measurement_feature_name,
)
from openhcs.core.equivalence.measurement_facts import (
    PAIR_REGRESSION_SLOPE_FEATURE,
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
    CSV_HEADER_CONTEXT_STOPWORDS,
    MEASUREMENT_IDENTITY_FIELDS,
)
from openhcs.core.measurement_row_materialization import (
    MEASUREMENT_FEATURE_NAME_FIELDS,
    MEASUREMENT_OBJECT_ID_FIELDS,
    MEASUREMENT_OBJECT_NAME_FIELD,
    MEASUREMENT_SOURCE_IMAGE_NAME_FIELD,
    MEASUREMENT_VALUE_FIELDS,
    measurement_object_label,
    measurement_row_has_long_form_measurement_fields,
    measurement_row_has_object_identity,
    measurement_row_identity_role,
    measurement_row_object_name,
    measurement_row_source_image_name,
)
from openhcs.core.registry_strategies import MostDerivedContextStrategyMixin
from openhcs.core.runtime_semantics import (
    MeasurementObjectRowIdentity,
    MeasurementScope,
    MeasurementStatistic,
    ObjectInstanceKey,
    measurement_row_mapping,
)

IMAGE_IDENTITY_FIELDS = (
    DEFAULT_RUNTIME_MEASUREMENT_DIALECT.row_identity_contract.image_identity_fields
)
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
    dialect: RuntimeMeasurementDialect = DEFAULT_RUNTIME_MEASUREMENT_DIALECT,
) -> tuple[tuple[str, object], ...]:
    """Return the image identity carried by a measurement row."""
    contract = dialect.row_identity_contract
    normalized_present_fields = frozenset(
        normalize_runtime_identifier(field_name)
        for field_name, value in row.items()
        if value is not None and str(value).strip()
    )
    selected_fields = contract.selected_image_identity_fields(
        normalized_present_fields
    )
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
    return sorted_tuple(identity_values)


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
        ("_runtime_axis", measurement_table_cell_payload(axis_key)),
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
                row[image_number_index]
                for row in rows
                if image_number_index < len(row)
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
    row_values: "RuntimeIndexedRowValues",
    qualifiers: tuple[tuple[RuntimeMeasurementRowQualifier, tuple[int | None, ...]], ...],
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

_NON_MEASUREMENT_FIELD_PREFIXES = (
    "channel_",
    "execution_time_",
    "file_name_",
    "frame_",
    "group_",
    "height_",
    "image_quality_scaling_",
    "image_set_",
    "md_5_digest_",
    "md5_digest_",
    "module_error_",
    "path_name_",
    "scaling_",
    "series_",
    "url_",
    "width_",
)
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
RuntimeMeasurementRowSubjectSchemaValue = tuple[
    int | None,
    int | None,
    tuple[int, ...],
    tuple[int, ...],
]
RuntimeMeasurementRowSubjectSchemaCache = dict[
    tuple[str, ...],
    RuntimeMeasurementRowSubjectSchemaValue,
]
RuntimeMeasurementRowIdentity = tuple[tuple[str, object], ...]
RuntimeMeasurementRowIdentityOrMissing = RuntimeMeasurementRowIdentity | None
OBJECT_LABEL_ROW_IDENTITY_FIELD = "object_label"
RUNTIME_SLICE_ROW_IDENTITY_FIELD = "_runtime_slice"


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
    long_form_feature_indexes: tuple[int, ...]
    long_form_value_indexes: tuple[int, ...]


RuntimeMeasurementRowSchemaCache = dict[tuple[str, ...], RuntimeMeasurementRowSchema]
RuntimeMeasurementFeatureKeyCache = dict[
    tuple[RuntimeMeasurementSubjectKey, str | None, str, tuple[str, ...]],
    RuntimeMeasurementFeatureKey | None,
]
RuntimeMeasurementLongFormKeyCache = dict[
    tuple[RuntimeMeasurementSubjectKey, str | None, str],
    RuntimeMeasurementFeatureKey | None,
]


def runtime_measurement_category_priority(
    prefix: tuple[str, ...],
    dialect: RuntimeMeasurementDialect,
) -> int | None:
    """Return the dialect priority for one category prefix."""
    return next(
        (
            index
            for index, category_prefix in enumerate(dialect.category_prefixes)
            if category_prefix == prefix
        ),
        None,
    )


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementFeatureCategoryPriority:
    """Dialect category-prefix priority for runtime measurement feature names."""

    feature_name: str
    dialect: RuntimeMeasurementDialect

    def priority(self) -> int | None:
        normalized = normalize_runtime_identifier(self.feature_name)
        if normalized_runtime_measurement_identity_field_matches(
            normalized,
            self.dialect,
        ):
            return None
        parts = tuple(part for part in normalized.split("_") if part)
        parts_projection = RuntimeMeasurementNamePartsProjection(parts, self.dialect)
        for index, prefix in enumerate(self.dialect.category_prefixes):
            if parts_projection.should_strip_category_prefix(prefix):
                return index
        return -1
RuntimeMeasurementQualifierRenderCache = dict[
    RuntimeMeasurementQualifierCacheKey,
    str | None,
]
RuntimeMeasurementPaddingGroupCache = dict[
    tuple[str, RuntimeMeasurementFeatureKey],
    RuntimeMeasurementPaddingGroup,
]
RuntimeMeasurementIndexedQualifierCache = dict[int, tuple[str, ...]]
RuntimeRowQualifierResolutionCache = dict[
    tuple[RuntimeMeasurementIndexedQualifier, ...],
    tuple[str, ...],
]
RuntimeMeasurementPaddingGroupPresence = dict[RuntimeMeasurementPaddingGroup, bool]


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


@dataclass(frozen=True, slots=True)
class RuntimeIndexedRowValues:
    """Typed accessors for row values indexed by schema positions."""

    row_values: tuple[object, ...]

    def at(self, index: int | None) -> object | None:
        if index is None:
            return None
        return self.row_values[index]

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

@dataclass(frozen=True, slots=True)
class RuntimeMeasurementRowMapping:
    """Nominal row boundary for runtime measurement-row semantics."""

    row: Mapping[str, object]

    @property
    def header(self) -> tuple[str, ...]:
        return tuple(self.row)

    @property
    def values(self) -> tuple[object, ...]:
        return tuple(self.row.get(field_name) for field_name in self.header)

    @property
    def normalized_fields(self) -> Mapping[str, str]:
        return {
            normalize_runtime_identifier(field_name): field_name
            for field_name in self.row
        }

    @property
    def normalized_field_names(self) -> frozenset[str]:
        return frozenset(self.normalized_fields)

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
        return measurement_row_has_object_identity(self.row)

    def has_long_form_measurement_fields(self) -> bool:
        return measurement_row_has_long_form_measurement_fields(self.row)

    def source_name(self) -> str | None:
        return measurement_row_source_image_name(self.row)

    def object_name(self) -> str | None:
        return measurement_row_object_name(self.row)

    def object_label(self) -> int | None:
        return measurement_object_label(self.row)

    def identity_role(self) -> MeasurementObjectRowIdentity | None:
        return measurement_row_identity_role(self.row)

    def image_identity_key(
        self,
        dialect: RuntimeMeasurementDialect = DEFAULT_RUNTIME_MEASUREMENT_DIALECT,
    ) -> tuple[tuple[str, object], ...]:
        return measurement_row_image_identity_key(self.row, dialect)

    def axis_scoped_identity(
        self,
        axis_key: str | None,
        dialect: RuntimeMeasurementDialect = DEFAULT_RUNTIME_MEASUREMENT_DIALECT,
    ) -> RuntimeMeasurementRowIdentity:
        return axis_scoped_measurement_row_identity(self.row, axis_key, dialect)


@dataclass(frozen=True, slots=True)
class RuntimeObjectMeasurementRowIdentity:
    """Nominal identity for one object measurement row within a runtime axis."""

    row_identity: RuntimeMeasurementRowIdentity

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
    ) -> "RuntimeObjectMeasurementRowIdentity":
        row_identity = row.axis_scoped_identity(
            axis_key,
            policy.measurement_dialect,
        )
        if object_instance_key.slice_index is not None:
            row_identity = (
                *row_identity,
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
            )
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
            or field[0] == RUNTIME_SLICE_ROW_IDENTITY_FIELD
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
    if normalized.startswith(_NON_MEASUREMENT_FIELD_PREFIXES):
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


def runtime_row_projection(
    records: Iterable[RuntimeRowProjectionRecord[RuntimeRowProjectionValueT]] = (),
    *,
    long_form: bool = False,
) -> RuntimeRowProjection[RuntimeRowProjectionValueT]:
    """Build a row projection through one normalized record boundary."""
    return RuntimeRowProjection(tuple(records), long_form=long_form)


@dataclass(frozen=True, slots=True)
class RuntimeRowProjectionContext:
    row: RuntimeMeasurementRowMapping
    subject: RuntimeMeasurementSubjectKey
    policy: RuntimeEquivalencePolicy
    source_name: str | None
    known_source_names: tuple[str, ...]
    required_keys: RuntimeRequiredMeasurementKeys
    table_padding_group: str
    image_number_offset: RuntimeImageNumberOffset
    schema_cache: RuntimeMeasurementRowSchemaCache
    key_cache: RuntimeMeasurementFeatureKeyCache
    long_form_key_cache: RuntimeMeasurementLongFormKeyCache
    qualifier_render_cache: RuntimeMeasurementQualifierRenderCache
    padding_group_cache: RuntimeMeasurementPaddingGroupCache

    @classmethod
    def from_row(
        cls,
        row: RuntimeMeasurementRowMapping,
        subject: RuntimeMeasurementSubjectKey,
        policy: RuntimeEquivalencePolicy,
        *,
        source_name: str | None,
        known_source_names: tuple[str, ...],
        required_keys: RuntimeRequiredMeasurementKeys,
        table_padding_group: str,
        image_number_offset: RuntimeImageNumberOffset,
        schema_cache: RuntimeMeasurementRowSchemaCache,
        key_cache: RuntimeMeasurementFeatureKeyCache,
        long_form_key_cache: RuntimeMeasurementLongFormKeyCache,
        qualifier_render_cache: RuntimeMeasurementQualifierRenderCache,
        padding_group_cache: RuntimeMeasurementPaddingGroupCache,
    ) -> "RuntimeRowProjectionContext":
        return cls(
            row=row,
            subject=subject,
            policy=policy,
            source_name=source_name,
            known_source_names=known_source_names,
            required_keys=required_keys,
            table_padding_group=table_padding_group,
            image_number_offset=image_number_offset,
            schema_cache=schema_cache,
            key_cache=key_cache,
            long_form_key_cache=long_form_key_cache,
            qualifier_render_cache=qualifier_render_cache,
            padding_group_cache=padding_group_cache,
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
        row_values = RuntimeIndexedRowValues(self.row.values)
        long_form_projection = self._long_form_projection(
            row_schema,
            row_values,
            long_form_projector,
        )
        if long_form_projection is not None:
            return long_form_projection
        return self._wide_projection(header, row_schema, row_values, value_projector)

    def facts(self) -> RuntimeMeasurementFacts:
        """Project this runtime row into semantic fact views."""
        projection = self.project(
            RuntimeMeasurementCellFactProjection(),
            RuntimeRowLongFormFactProjection(),
        )
        row_facts = RuntimeMeasurementFactProjectionContract.dedupe_observed_qualified_records(
            projection.records,
            self.policy,
        )
        if projection.long_form:
            return row_facts
        derived_facts = RuntimeDirectionalPairMeasurementDerivationContract(
            self.policy,
            self.known_source_names,
        ).derive(row_facts)
        if self.required_keys is not None:
            return tuple(
                (key, value)
                for key, value in derived_facts
                if key in self.required_keys
            )
        return derived_facts

    def numeric_values(self) -> RuntimeNumericMeasurementValues:
        """Project numeric runtime row values without building cell signatures."""
        projection = self.project(
            RuntimeMeasurementCellNumericProjection(),
            RuntimeRowLongFormNumericProjection(),
        )
        row_values_by_key = dedupe_numeric_measurement_values(
            (key, value)
            for _padding_group, key, value, _qualified_observation in projection.records
        )
        if projection.long_form:
            return row_values_by_key
        if not any(
            key.belongs_to_source_qualified_feature_family(
                self.policy.measurement_dialect,
                (PAIR_REGRESSION_SLOPE_FEATURE,),
            )
            for key, _value in row_values_by_key
        ):
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

        normalized_fields = tuple(normalize_runtime_identifier(field) for field in header)
        aggregate_reference_indexes = frozenset(
            index
            for index, field_name in enumerate(header)
            if aggregate_image_number_reference_measurement_field(field_name)
        )
        normalized_field_indexes = {
            field_name: index
            for index, field_name in enumerate(normalized_fields)
        }
        feature_indexes = tuple(
            index
            for index, field_name in enumerate(normalized_fields)
            if not normalized_runtime_measurement_identity_field_matches(
                field_name,
                self.policy.measurement_dialect,
            )
            and index not in aggregate_reference_indexes
        )
        qualifier_indexes = {
            qualifier: tuple(
                normalized_field_indexes.get(field_name)
                for field_name in qualifier.field_names
            )
            for qualifier in self.policy.measurement_dialect.row_qualifiers
        }
        qualifiers_by_index = {
            index: tuple(
                (qualifier, qualifier_indexes[qualifier])
                for qualifier in self.policy.measurement_dialect.row_qualifiers
                if row_qualifier_applies_to_field(
                    qualifier,
                    tuple(
                        part
                        for part in normalized_fields[index].split("_")
                        if part
                    ),
                )
            )
            for index in feature_indexes
        }
        cached_schema = RuntimeMeasurementRowSchema(
            feature_indexes,
            qualifiers_by_index,
            runtime_measurement_field_indexes(
                normalized_field_indexes,
                MEASUREMENT_FEATURE_NAME_FIELDS,
            ),
            runtime_measurement_field_indexes(
                normalized_field_indexes,
                MEASUREMENT_VALUE_FIELDS,
            ),
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
                (
                    (self.subject, self.source_name, ()),
                    key,
                    value,
                    False,
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
        for index in row_schema.feature_indexes:
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
        return runtime_row_projection(
            (
                (padding_group, key, value, qualified_observation)
                for padding_group, key, value, qualified_observation in records
                if padding_group_presence.get(padding_group, True)
            )
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
        value = self.image_number_offset.normalized_reference_value(
            field_name,
            row_values.at(index),
        )
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
        padding_group = self._padding_group(field_name, key)
        padding_group_presence[padding_group] = (
            padding_group_presence.get(padding_group, False)
            or RuntimeMeasurementValuePresence(value).is_present()
        )
        if (
            self.required_keys is not None
            and key not in self.required_keys
            and not runtime_value_is_mapping(value)
        ):
            return ()
        projected_values = value_projector.project(key, value, self.policy)
        if self.required_keys is not None:
            projected_values = tuple(
                (cell_key, cell_value)
                for cell_key, cell_value in projected_values
                if cell_key in self.required_keys
            )
        return tuple(
            (
                padding_group,
                cell_key,
                cell_value,
                measurement_field_has_collapsed_numeric_qualifier(
                    field_name,
                    self.policy.measurement_dialect,
                    known_source_names=self.known_source_names,
                ),
            )
            for cell_key, cell_value in projected_values
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
        qualifiers = row_qualifier_cache.get(indexed_qualifiers)
        if qualifiers is None:
            qualifiers = measurement_row_qualifiers_from_indexed_values_cached(
                row_values,
                indexed_qualifiers,
                self.qualifier_render_cache,
            )
            row_qualifier_cache[indexed_qualifiers] = qualifiers
        return qualifiers

    def _feature_key(
        self,
        field_name: str,
        subject: RuntimeMeasurementSubjectKey,
        qualifiers: tuple[str, ...],
    ) -> RuntimeMeasurementFeatureKey | None:
        cache_key = (subject, self.source_name, field_name, qualifiers)
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
    ) -> RuntimeMeasurementPaddingGroup:
        cache_key = (field_name, key)
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
    ) -> RuntimeProjectedCells[RuntimeRowProjectionValueT]:
        """Project one wide-form cell into semantic values."""


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementCellValue:
    """Runtime measurement cell plus nested value expansion policy."""

    key: RuntimeMeasurementFeatureKey
    value: object
    policy: RuntimeEquivalencePolicy
    required_keys: frozenset[RuntimeMeasurementFeatureKey] | None = None

    def iter_key_values(
        self,
    ) -> Iterable[tuple[RuntimeMeasurementFeatureKey, object]]:
        if not runtime_value_is_mapping(self.value):
            if self.required_keys is not None and self.key not in self.required_keys:
                return ()
            return ((self.key, self.value),)
        return tuple(
            (nested_key, nested_value)
            for name, nested_value in self.value.items()
            for nested_key in (self.nested_key(name),)
            if self.required_keys is None or nested_key in self.required_keys
        )

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
    ) -> RuntimeMeasurementFacts:
        cell = RuntimeMeasurementCellValue(key, value, policy)
        return self.project_cell(cell)

    def project_cell(
        self,
        cell: RuntimeMeasurementCellValue,
    ) -> RuntimeMeasurementFacts:
        return tuple(
            (
                cell_key,
                RuntimeMeasurementCellSignatureProjection(cell_value, cell.policy).signature(),
            )
            for cell_key, cell_value in cell.iter_key_values()
            if RuntimeMeasurementCellPresence(cell_value).is_present()
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
    ) -> RuntimeNumericMeasurementValues:
        cell = RuntimeMeasurementCellValue(key, value, policy)
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
    key_cache: RuntimeMeasurementLongFormKeyCache

    @classmethod
    def from_runtime_row_projection(
        cls,
        context: RuntimeRowProjectionContext,
        row_values: RuntimeIndexedRowValues,
        feature_indexes: tuple[int, ...],
        value_indexes: tuple[int, ...],
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
            context.long_form_key_cache,
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
    return (
        key.source_name == "image"
        and "parent" in parts
        and "number" in parts
    )


def parts_contain_adjacent_image_number(parts: tuple[str, ...]) -> bool:
    return any(
        parts[index] == "image" and parts[index + 1] == "number"
        for index in range(len(parts) - 1)
    )


def measurement_field_has_collapsed_numeric_qualifier(
    field_name: str,
    dialect: RuntimeMeasurementDialect,
    *,
    known_source_names: tuple[str, ...],
) -> bool:
    """Return true when semantic normalization drops a numeric feature qualifier."""
    parts = tuple(part for part in normalize_runtime_identifier(field_name).split("_") if part)
    category_prefix = RuntimeMeasurementNamePartsProjection(
        parts,
        dialect,
    ).category_prefix()
    if category_prefix:
        parts = parts[len(category_prefix) :]
    parts, _source_names = RuntimeMeasurementNamePartsProjection(
        parts,
        dialect,
        known_source_names,
    ).source_qualifier_tokens()
    return (
        RuntimeMeasurementNamePartsProjection(parts, dialect).semantic_core_parts()
        != parts
    )


@dataclass(frozen=True, slots=True)
class ContextualMeasurementPaddingColumn:
    """One contextual wide-table column that may belong to a padding group."""

    context: str | None
    field_name: str
    dialect: RuntimeMeasurementDialect
    known_source_names: tuple[str, ...]

    @property
    def normalized_context(self) -> str | None:
        if self.context is None:
            return None
        normalized = normalize_runtime_identifier(self.context)
        if not normalized or normalized in CSV_HEADER_CONTEXT_STOPWORDS:
            return None
        return normalized

    @property
    def normalized_field_parts(self) -> tuple[str, ...]:
        if runtime_measurement_identity_field_matches(self.field_name, self.dialect):
            return ()
        return tuple(
            part
            for part in normalize_runtime_identifier(self.field_name).split("_")
            if part
        )

    def group(self) -> _ContextualMeasurementPaddingGroup | None:
        normalized_context = self.normalized_context
        parts = self.normalized_field_parts
        if normalized_context is None or not parts:
            return None
        feature_group = RuntimeMeasurementNamePartsProjection(
            parts,
            self.dialect,
        ).category_prefix() or parts[:1]
        _feature_name, source_name = SemanticCoreFeatureAndSourceNameProjection(
            normalize_runtime_identifier(self.field_name),
            self.dialect,
            self.known_source_names,
        ).project()
        return normalized_context, feature_group, source_name


@dataclass(frozen=True, slots=True)
class ContextualMeasurementPaddingProjection:
    """Resolve contextual wide-table cells that represent padding, not facts."""

    column_context: tuple[str | None, ...]
    header: tuple[str, ...]
    feature_indexes: tuple[int, ...]
    dialect: RuntimeMeasurementDialect
    known_source_names: tuple[str, ...]

    def groups_by_index(
        self,
    ) -> Mapping[int, _ContextualMeasurementPaddingGroup | None]:
        if not self.column_context:
            return MappingProxyType({})
        return MappingProxyType(
            {
                index: self.group_for_index(index)
                for index in self.feature_indexes
            }
        )

    def padding_indexes(
        self,
        row_values: RuntimeIndexedRowValues,
        *,
        padding_groups_by_index: Mapping[
            int,
            _ContextualMeasurementPaddingGroup | None,
        ] | None = None,
    ) -> frozenset[int]:
        if not self.column_context:
            return frozenset()
        if padding_groups_by_index is None:
            padding_groups_by_index = self.groups_by_index()

        indexes_by_group: dict[_ContextualMeasurementPaddingGroup, list[int]] = {}
        for index in self.feature_indexes:
            group = padding_groups_by_index.get(index)
            if group is None:
                continue
            indexes_by_group.setdefault(group, []).append(index)

        padding_indexes: set[int] = set()
        for indexes in indexes_by_group.values():
            if any(
                RuntimeMeasurementCellPresence(row_values.at(index)).is_present()
                for index in indexes
            ):
                continue
            padding_indexes.update(indexes)
        return frozenset(padding_indexes)

    def group_for_index(
        self,
        index: int,
    ) -> _ContextualMeasurementPaddingGroup | None:
        if index >= len(self.column_context):
            return None
        return ContextualMeasurementPaddingColumn(
            self.column_context[index],
            self.header[index],
            self.dialect,
            self.known_source_names,
        ).group()


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
        feature_name = row.first_value(MEASUREMENT_FEATURE_NAME_FIELDS)
        value = row.first_value(MEASUREMENT_VALUE_FIELDS)
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
        key = RuntimeMeasurementFeatureKeyProjection(
            RuntimeMeasurementFeatureKeySourceContext(
                source.feature_text,
                self.context.subject,
                self.context.policy,
                (),
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
        cache_key = (
            self.context.subject,
            self.context.source_name,
            source.feature_text,
        )
        key = self.context.key_cache.get(cache_key, _CACHE_MISS)
        if key is _CACHE_MISS:
            key = self._feature_key(source.feature_text)
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
    ) -> RuntimeMeasurementFeatureKey | None:
        return RuntimeMeasurementFeatureKeyProjection(
            RuntimeMeasurementFeatureKeySourceContext(
                feature_text,
                self.context.subject,
                self.context.policy,
                (),
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
class RuntimeMeasurementRowSubjectResolutionContext:
    """Typed subject-resolution facts for one measurement row."""

    table_subject: RuntimeMeasurementSubjectKey
    object_name: str | None
    row_source_name: str | None
    has_object_identity: bool
    has_image_identity: bool


class RuntimeMeasurementRowSubjectResolutionStrategy(
    MostDerivedContextStrategyMixin[RuntimeMeasurementRowSubjectResolutionContext],
    ABC,
):
    """Nominal row-subject resolver for runtime measurement tables."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.STRATEGY_KEY)

    strategy_key: ClassVar[str | None] = None

    @classmethod
    def resolve(
        cls,
        context: RuntimeMeasurementRowSubjectResolutionContext,
    ) -> RuntimeMeasurementSubjectKey:
        strategy = cls.for_context(
            context,
            error_subject="Runtime measurement row subject resolution",
        )
        if strategy is None:
            raise ValueError(
                "Runtime measurement row subject resolution requires a strategy."
            )
        return strategy.subject(context)

    @abstractmethod
    def subject(
        self,
        context: RuntimeMeasurementRowSubjectResolutionContext,
    ) -> RuntimeMeasurementSubjectKey:
        """Return the semantic measurement subject for this row."""


class FallbackTableSubjectResolutionStrategy(
    RuntimeMeasurementRowSubjectResolutionStrategy
):
    """Fallback to the table's declared subject."""

    strategy_key = "fallback_table_subject"

    def matches(
        self,
        context: RuntimeMeasurementRowSubjectResolutionContext,
    ) -> bool:
        return True

    def subject(
        self,
        context: RuntimeMeasurementRowSubjectResolutionContext,
    ) -> RuntimeMeasurementSubjectKey:
        return context.table_subject


class SourceImageRowSubjectResolutionStrategy(FallbackTableSubjectResolutionStrategy):
    """Rows declaring only source-image identity remain image-scoped."""

    strategy_key = "source_image_row"

    def matches(
        self,
        context: RuntimeMeasurementRowSubjectResolutionContext,
    ) -> bool:
        return context.row_source_name is not None

    def subject(
        self,
        context: RuntimeMeasurementRowSubjectResolutionContext,
    ) -> RuntimeMeasurementSubjectKey:
        return RuntimeMeasurementSubjectKey(MeasurementScope.IMAGE, "Image")


class ImageTableSubjectResolutionStrategy(SourceImageRowSubjectResolutionStrategy):
    """Image tables own rows that do not carry a stronger row identity."""

    strategy_key = "image_table_subject"

    def matches(
        self,
        context: RuntimeMeasurementRowSubjectResolutionContext,
    ) -> bool:
        return context.table_subject.scope is MeasurementScope.IMAGE

    def subject(
        self,
        context: RuntimeMeasurementRowSubjectResolutionContext,
    ) -> RuntimeMeasurementSubjectKey:
        return context.table_subject


class ObjectTableSubjectResolutionStrategy(ImageTableSubjectResolutionStrategy):
    """Object tables own rows that do not carry a stronger row identity."""

    strategy_key = "object_table_subject"

    def matches(
        self,
        context: RuntimeMeasurementRowSubjectResolutionContext,
    ) -> bool:
        return context.table_subject.scope is MeasurementScope.OBJECT

    def subject(
        self,
        context: RuntimeMeasurementRowSubjectResolutionContext,
    ) -> RuntimeMeasurementSubjectKey:
        return context.table_subject


class ImageIdentityRowSubjectResolutionStrategy(ObjectTableSubjectResolutionStrategy):
    """Rows with image identity and no object identity are image-scoped."""

    strategy_key = "image_identity"

    def matches(
        self,
        context: RuntimeMeasurementRowSubjectResolutionContext,
    ) -> bool:
        return context.has_image_identity and not context.has_object_identity

    def subject(
        self,
        context: RuntimeMeasurementRowSubjectResolutionContext,
    ) -> RuntimeMeasurementSubjectKey:
        return RuntimeMeasurementSubjectKey(MeasurementScope.IMAGE, "Image")


class ObjectIdentityRowSubjectResolutionStrategy(ImageIdentityRowSubjectResolutionStrategy):
    """Rows with object identity are object-scoped."""

    strategy_key = "object_identity"

    def matches(
        self,
        context: RuntimeMeasurementRowSubjectResolutionContext,
    ) -> bool:
        return context.object_name is not None and context.has_object_identity

    def subject(
        self,
        context: RuntimeMeasurementRowSubjectResolutionContext,
    ) -> RuntimeMeasurementSubjectKey:
        return RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, context.object_name)


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementRowSubjectProjection:
    """Resolve runtime-row source and measurement subject from row values."""

    table_subject: RuntimeMeasurementSubjectKey
    table_source_name: str | None
    row_values: RuntimeIndexedRowValues
    subject_schema: RuntimeMeasurementRowSubjectSchemaValue

    @property
    def indexed_values(self) -> RuntimeIndexedRowValues:
        return self.row_values

    def source_name(self) -> str | None:
        row_source_name = self.row_values.text_at(self.subject_schema[1])
        if row_source_name is not None:
            return row_source_name
        return self.table_source_name

    def subject(self) -> RuntimeMeasurementSubjectKey:
        (
            object_name_index,
            source_name_index,
            object_identity_indexes,
            image_identity_indexes,
        ) = self.subject_schema
        indexed_values = self.indexed_values
        return RuntimeMeasurementRowSubjectResolutionStrategy.resolve(
            RuntimeMeasurementRowSubjectResolutionContext(
                table_subject=self.table_subject,
                object_name=self.row_values.text_at(object_name_index),
                row_source_name=self.row_values.text_at(source_name_index),
                has_object_identity=indexed_values.has_text_at_any(
                    object_identity_indexes
                ),
                has_image_identity=indexed_values.has_text_at_any(image_identity_indexes),
            )
        )


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementRowSubjectSchema:
    """Column indexes used to resolve runtime measurement row subjects."""

    header: tuple[str, ...]

    def schema(self) -> RuntimeMeasurementRowSubjectSchemaValue:
        normalized_fields = tuple(
            normalize_runtime_identifier(field) for field in self.header
        )
        normalized_field_indexes = {
            field_name: index
            for index, field_name in enumerate(normalized_fields)
        }
        return (
            normalized_field_indexes.get(MEASUREMENT_OBJECT_NAME_FIELD),
            normalized_field_indexes.get(MEASUREMENT_SOURCE_IMAGE_NAME_FIELD),
            runtime_measurement_field_indexes(
                normalized_field_indexes,
                MEASUREMENT_OBJECT_ID_FIELDS,
            ),
            runtime_measurement_field_indexes(
                normalized_field_indexes,
                tuple(sorted(IMAGE_IDENTITY_FIELDS)),
            ),
        )


def runtime_measurement_row_subject_schema(
    header: tuple[str, ...],
    cache: RuntimeMeasurementRowSubjectSchemaCache,
) -> RuntimeMeasurementRowSubjectSchemaValue:
    """Return the cached subject schema for a runtime measurement row header."""
    cached_schema = cache.get(header)
    if cached_schema is None:
        cached_schema = RuntimeMeasurementRowSubjectSchema(header).schema()
        cache[header] = cached_schema
    return cached_schema
