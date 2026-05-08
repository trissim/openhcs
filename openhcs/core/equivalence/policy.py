"""Runtime equivalence policy records."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from types import MappingProxyType
from typing import Annotated, get_args, get_origin, get_type_hints

from openhcs.core.runtime_semantics import MeasurementScope


class RuntimeMeasurementFeatureNameMode(str, Enum):
    """How measurement feature names are canonicalized for semantic comparison."""

    FULL = "full"
    SEMANTIC_CORE = "semantic_core"


_EMPTY_FEATURE_ALIASES = MappingProxyType({})
_EMPTY_PAIR_FEATURE_ALIASES = MappingProxyType({})
_EMPTY_NUMBERED_FEATURE_PREFIX_ALIASES = MappingProxyType({})
_IDENTIFIER_INITIALISM_BOUNDARY_RE = re.compile(r"([A-Z]+)([A-Z][a-z])")
_IDENTIFIER_LOWER_UPPER_BOUNDARY_RE = re.compile(r"([a-z0-9])([A-Z])")
_IDENTIFIER_ALPHA_NUMBER_BOUNDARY_RE = re.compile(r"([A-Za-z])([0-9])")
_IDENTIFIER_NUMBER_ALPHA_BOUNDARY_RE = re.compile(r"([0-9])([A-Za-z])")
_IDENTIFIER_NON_ALNUM_RE = re.compile(r"[^A-Za-z0-9]+")


def normalize_runtime_identifier(value: object) -> str:
    """Return OpenHCS' canonical identifier token for runtime comparison."""
    return _normalize_identifier_text(str(value).strip())


def normalize_runtime_source_name(source_name: str | None) -> str | None:
    """Return canonical source-image identity for runtime comparison."""
    if source_name is None:
        return None
    normalized = "__".join(
        part
        for part in (
            normalize_runtime_identifier(part)
            for part in str(source_name).split("__")
        )
        if part
    )
    return normalized or None


@lru_cache(maxsize=32768)
def _normalize_identifier_text(text: str) -> str:
    text = _IDENTIFIER_INITIALISM_BOUNDARY_RE.sub(r"\1_\2", text)
    text = _IDENTIFIER_LOWER_UPPER_BOUNDARY_RE.sub(r"\1_\2", text)
    text = _IDENTIFIER_ALPHA_NUMBER_BOUNDARY_RE.sub(r"\1_\2", text)
    text = _IDENTIFIER_NUMBER_ALPHA_BOUNDARY_RE.sub(r"\1_\2", text)
    text = _IDENTIFIER_NON_ALNUM_RE.sub("_", text)
    return text.strip("_").lower()


class RuntimeMeasurementQualifierValueMode(str, Enum):
    """How a row qualifier value is rendered into a feature suffix."""

    IDENTIFIER = "identifier"
    TWO_DIGIT_INTEGER = "two_digit_integer"
    FRACTION_OF_COUNT = "fraction_of_count"


@dataclass(frozen=True, slots=True)
class _NonNegativeRuntimePolicyField:
    """Type annotation marker for non-negative numeric policy fields."""


NonNegativeFloat = Annotated[float, _NonNegativeRuntimePolicyField()]
NonNegativeInt = Annotated[int, _NonNegativeRuntimePolicyField()]


def _validate_annotated_non_negative_policy_fields(instance: object) -> None:
    """Validate numeric invariants declared directly on dataclass annotations."""
    owner_name = type(instance).__name__
    for field_name, annotation in get_type_hints(
        type(instance),
        include_extras=True,
    ).items():
        if get_origin(annotation) is not Annotated:
            continue
        if not any(
            isinstance(metadata, _NonNegativeRuntimePolicyField)
            for metadata in get_args(annotation)[1:]
        ):
            continue
        if getattr(instance, field_name) < 0:
            raise ValueError(f"{owner_name}.{field_name} cannot be negative.")


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementRowQualifier:
    """Declarative row fields that qualify measurement feature names."""

    field_names: tuple[str, ...]
    value_mode: RuntimeMeasurementQualifierValueMode = (
        RuntimeMeasurementQualifierValueMode.IDENTIFIER
    )
    feature_prefixes: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        field_names = tuple(
            normalize_runtime_identifier(field_name)
            for field_name in self.field_names
            if str(field_name).strip()
        )
        if not field_names:
            raise ValueError(
                "RuntimeMeasurementRowQualifier.field_names cannot be empty."
            )
        object.__setattr__(self, "field_names", field_names)
        object.__setattr__(
            self,
            "value_mode",
            (
                self.value_mode
                if isinstance(
                    self.value_mode,
                    RuntimeMeasurementQualifierValueMode,
                )
                else RuntimeMeasurementQualifierValueMode(self.value_mode)
            ),
        )
        object.__setattr__(
            self,
            "feature_prefixes",
            tuple(
                tuple(
                    normalize_runtime_identifier(part)
                    for part in prefix
                    if str(part).strip()
                )
                for prefix in self.feature_prefixes
            ),
        )


_DEFAULT_MEASUREMENT_ROW_QUALIFIERS = (
    RuntimeMeasurementRowQualifier(("scale",)),
    RuntimeMeasurementRowQualifier(
        ("direction",),
        RuntimeMeasurementQualifierValueMode.TWO_DIGIT_INTEGER,
    ),
    RuntimeMeasurementRowQualifier(("gray_levels",)),
    RuntimeMeasurementRowQualifier(
        ("bin_index", "bin_count"),
        RuntimeMeasurementQualifierValueMode.FRACTION_OF_COUNT,
    ),
)


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementDialect:
    """Policy-provided measurement-name dialect for semantic comparisons."""

    category_prefixes: tuple[tuple[str, ...], ...] = ()
    feature_part_aliases: Mapping[tuple[str, ...], tuple[str, ...]] = (
        _EMPTY_FEATURE_ALIASES
    )
    source_feature_prefixes: tuple[tuple[str, ...], ...] = ()
    directional_pair_feature_aliases: Mapping[str, tuple[str, int]] = (
        _EMPTY_PAIR_FEATURE_ALIASES
    )
    scale_qualified_feature_prefixes: tuple[tuple[str, ...], ...] = ()
    threshold_qualifier_tokens: frozenset[str] = frozenset()
    source_qualifier_prefix_tokens: frozenset[str] = frozenset()
    source_qualifier_suffix_tokens: frozenset[str] = frozenset()
    row_qualifiers: tuple[RuntimeMeasurementRowQualifier, ...] = (
        _DEFAULT_MEASUREMENT_ROW_QUALIFIERS
    )
    numbered_feature_prefix_aliases: Mapping[str, tuple[str, ...]] = (
        _EMPTY_NUMBERED_FEATURE_PREFIX_ALIASES
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "category_prefixes",
            tuple(
                tuple(part for part in prefix if part)
                for prefix in self.category_prefixes
            ),
        )
        object.__setattr__(
            self,
            "feature_part_aliases",
            MappingProxyType(
                {
                    tuple(part for part in parts if part): tuple(
                        part for part in alias if part
                    )
                    for parts, alias in self.feature_part_aliases.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "source_feature_prefixes",
            tuple(
                tuple(part for part in prefix if part)
                for prefix in self.source_feature_prefixes
            ),
        )
        object.__setattr__(
            self,
            "directional_pair_feature_aliases",
            MappingProxyType(
                {
                    str(name): (
                        str(alias[0]),
                        int(alias[1]),
                    )
                    for name, alias in self.directional_pair_feature_aliases.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "scale_qualified_feature_prefixes",
            tuple(
                tuple(part for part in prefix if part)
                for prefix in self.scale_qualified_feature_prefixes
            ),
        )
        object.__setattr__(
            self,
            "threshold_qualifier_tokens",
            frozenset(str(token) for token in self.threshold_qualifier_tokens),
        )
        object.__setattr__(
            self,
            "source_qualifier_prefix_tokens",
            frozenset(
                str(token) for token in self.source_qualifier_prefix_tokens
            ),
        )
        object.__setattr__(
            self,
            "source_qualifier_suffix_tokens",
            frozenset(
                str(token) for token in self.source_qualifier_suffix_tokens
            ),
        )
        object.__setattr__(
            self,
            "row_qualifiers",
            tuple(
                qualifier
                if isinstance(qualifier, RuntimeMeasurementRowQualifier)
                else RuntimeMeasurementRowQualifier(tuple(qualifier))
                for qualifier in self.row_qualifiers
            ),
        )
        object.__setattr__(
            self,
            "numbered_feature_prefix_aliases",
            MappingProxyType(
                {
                    normalize_runtime_identifier(prefix): tuple(
                        normalize_runtime_identifier(part)
                        for part in alias
                        if str(part).strip()
                    )
                    for prefix, alias in self.numbered_feature_prefix_aliases.items()
                    if str(prefix).strip()
                }
            ),
        )


DEFAULT_RUNTIME_MEASUREMENT_DIALECT = RuntimeMeasurementDialect()


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementFeatureNumericTolerance:
    """Numeric tolerance scoped to a semantic measurement feature family."""

    feature_name_prefixes: tuple[str, ...] = ()
    feature_names: frozenset[str] = frozenset()
    subject_scope: MeasurementScope | None = None
    statistic: str | None = None
    numeric_abs_tolerance: NonNegativeFloat = 0.0
    numeric_rel_tolerance: NonNegativeFloat = 0.0
    require_object_count_stability: bool = False

    def __post_init__(self) -> None:
        feature_name_prefixes = tuple(
            str(prefix).strip()
            for prefix in self.feature_name_prefixes
            if str(prefix).strip()
        )
        feature_names = frozenset(
            str(feature_name).strip()
            for feature_name in self.feature_names
            if str(feature_name).strip()
        )
        if not feature_name_prefixes and not feature_names:
            raise ValueError(
                "RuntimeMeasurementFeatureNumericTolerance requires at least "
                "one feature name or feature-name prefix."
            )
        subject_scope = (
            self.subject_scope
            if self.subject_scope is None
            or isinstance(self.subject_scope, MeasurementScope)
            else MeasurementScope(self.subject_scope)
        )
        statistic = (
            normalize_runtime_identifier(self.statistic)
            if self.statistic is not None
            else None
        )
        if statistic == "":
            raise ValueError(
                "RuntimeMeasurementFeatureNumericTolerance.statistic cannot be empty."
            )
        _validate_annotated_non_negative_policy_fields(self)
        object.__setattr__(self, "feature_name_prefixes", feature_name_prefixes)
        object.__setattr__(self, "feature_names", feature_names)
        object.__setattr__(self, "subject_scope", subject_scope)
        object.__setattr__(self, "statistic", statistic)


@dataclass(frozen=True, slots=True)
class RuntimeEquivalencePolicy:
    """Policy controlling semantic output comparison strictness."""

    numeric_decimal_places: NonNegativeInt = 10
    numeric_abs_tolerance: NonNegativeFloat = 0.0
    numeric_rel_tolerance: NonNegativeFloat = 0.0
    allow_tie_sensitive_location_mismatches: bool = False
    allow_unstable_shape_descriptors: bool = False
    shape_descriptor_abs_tolerance: NonNegativeFloat = 0.025
    shape_descriptor_rel_tolerance: NonNegativeFloat = 0.0
    shape_descriptor_max_unstable_values: NonNegativeInt = 2
    shape_descriptor_max_unstable_fraction: NonNegativeFloat = 0.01
    threshold_entropy_abs_tolerance: NonNegativeFloat = 0.0
    threshold_sensitive_pair_abs_tolerance: NonNegativeFloat = 0.0
    threshold_sensitive_pair_rel_tolerance: NonNegativeFloat = 0.0
    allow_sparse_object_boundary_jitter: bool = False
    object_boundary_jitter_abs_tolerance: NonNegativeFloat = 25.0
    object_boundary_jitter_rel_tolerance: NonNegativeFloat = 0.0
    object_boundary_jitter_max_unstable_values: NonNegativeInt = 25
    object_boundary_jitter_max_unstable_fraction: NonNegativeFloat = 0.01
    object_boundary_jitter_aggregate_abs_tolerance: NonNegativeFloat = 0.05
    object_boundary_jitter_aggregate_rel_tolerance: NonNegativeFloat = 0.0
    allow_unstable_zernike_descriptors: bool = False
    zernike_descriptor_magnitude_abs_tolerance: NonNegativeFloat = 1e-6
    zernike_descriptor_phase_abs_tolerance: NonNegativeFloat = 0.35
    zernike_descriptor_rel_tolerance: NonNegativeFloat = 0.0
    compare_table_values: bool = True
    compare_image_pixels: bool = True
    image_abs_tolerance: NonNegativeFloat = 0.0
    image_rel_tolerance: NonNegativeFloat = 0.0
    image_max_different_fraction: NonNegativeFloat = 0.0
    allow_extra_candidate_measurements: bool = True
    measurement_feature_name_mode: RuntimeMeasurementFeatureNameMode = (
        RuntimeMeasurementFeatureNameMode.SEMANTIC_CORE
    )
    measurement_dialect: RuntimeMeasurementDialect = (
        DEFAULT_RUNTIME_MEASUREMENT_DIALECT
    )
    feature_numeric_tolerances: tuple[
        RuntimeMeasurementFeatureNumericTolerance,
        ...
    ] = ()

    def __post_init__(self) -> None:
        _validate_annotated_non_negative_policy_fields(self)
        object.__setattr__(
            self,
            "measurement_feature_name_mode",
            (
                self.measurement_feature_name_mode
                if isinstance(
                    self.measurement_feature_name_mode,
                    RuntimeMeasurementFeatureNameMode,
                )
                else RuntimeMeasurementFeatureNameMode(
                    self.measurement_feature_name_mode
                )
            ),
        )
        if not isinstance(self.measurement_dialect, RuntimeMeasurementDialect):
            raise TypeError(
                "RuntimeEquivalencePolicy.measurement_dialect must be "
                f"RuntimeMeasurementDialect, got {type(self.measurement_dialect).__name__}."
            )
        object.__setattr__(
            self,
            "feature_numeric_tolerances",
            tuple(
                tolerance
                if isinstance(
                    tolerance,
                    RuntimeMeasurementFeatureNumericTolerance,
                )
                else RuntimeMeasurementFeatureNumericTolerance(**tolerance)
                for tolerance in self.feature_numeric_tolerances
            ),
        )
