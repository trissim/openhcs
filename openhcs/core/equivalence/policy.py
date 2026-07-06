"""Runtime equivalence policy records."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from types import MappingProxyType
from typing import Annotated, ClassVar, get_args, get_origin, get_type_hints

from metaclass_registry import AutoRegisterMeta
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.core.runtime_identifier import (
    normalize_runtime_identifier,
    normalize_runtime_source_name,
    runtime_source_name_tokens,
)
from openhcs.core.runtime_semantics import (
    MeasurementScope,
    RuntimeMeasurementFeatureRelation,
    RuntimeMeasurementFeatureRelationDeclaration,
    RuntimeMeasurementFeatureRelationDeclarationCollection,
    RuntimeMeasurementFeatureSemanticMarker,
)


class RuntimeMeasurementFeatureNameMode(str, Enum):
    """How measurement feature names are canonicalized for semantic comparison."""

    FULL = "full"
    SEMANTIC_CORE = "semantic_core"


class RuntimeMeasurementSourceNameEncoding(str, Enum):
    """How a dialect encodes source-image identity in measurement features."""

    SEPARATE_KEY = "separate_key"
    FEATURE_SUFFIX = "feature_suffix"


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementSourceQualifiedFeature:
    """Feature identity after applying a dialect's source-name encoding."""

    feature_name: str
    source_name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "feature_name",
            normalize_runtime_identifier(self.feature_name),
        )
        object.__setattr__(
            self,
            "source_name",
            normalize_runtime_source_name(self.source_name),
        )


_EMPTY_FEATURE_ALIASES = MappingProxyType({})
_EMPTY_PAIR_FEATURE_ALIASES = MappingProxyType({})
_EMPTY_NUMBERED_FEATURE_PREFIX_ALIASES = MappingProxyType({})
_RUNTIME_MEASUREMENT_DIALECTS_BY_ID: dict[int, "RuntimeMeasurementDialect"] = {}


def runtime_measurement_dialect_cache_id(
    dialect: "RuntimeMeasurementDialect",
) -> int:
    """Return a process-local cache key for an immutable measurement dialect."""
    dialect_id = id(dialect)
    _RUNTIME_MEASUREMENT_DIALECTS_BY_ID[dialect_id] = dialect
    return dialect_id


def runtime_measurement_dialect_for_cache_id(
    dialect_id: int,
) -> "RuntimeMeasurementDialect":
    """Return the measurement dialect registered for ``dialect_id``."""
    return _RUNTIME_MEASUREMENT_DIALECTS_BY_ID[dialect_id]


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


class RuntimePolicyNonNegativeFieldValidationMixin:
    """Nominal owner for runtime-policy annotation invariants."""

    def validate_non_negative_policy_fields(self) -> None:
        """Validate numeric invariants declared directly on dataclass annotations."""
        owner_type = type(self)
        for field_name, annotation in get_type_hints(
            owner_type,
            include_extras=True,
        ).items():
            if get_origin(annotation) is not Annotated:
                continue
            if not any(
                isinstance(metadata, _NonNegativeRuntimePolicyField)
                for metadata in get_args(annotation)[1:]
            ):
                continue
            if object.__getattribute__(self, field_name) < 0:
                raise ValueError(
                    f"{owner_type.__name__}.{field_name} cannot be negative."
                )


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


class RuntimeMeasurementQualifierSuffixMatchStrategy(
    EnumKeyedStrategyMixin[RuntimeMeasurementQualifierValueMode],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered suffix parser for one row-qualifier value semantics."""

    __registry_key__ = "strategy_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "value_mode"
    __enum_label_attr__ = "strategy_label"

    value_mode: ClassVar[RuntimeMeasurementQualifierValueMode | None] = None
    strategy_label: ClassVar[str | None] = None

    @abstractmethod
    def matched_token_width(
        self,
        feature_tokens: tuple[str, ...],
        end_index: int,
        qualifier: RuntimeMeasurementRowQualifier,
    ) -> int | None:
        """Return token width owned by ``qualifier`` before ``end_index``."""


class IdentifierQualifierSuffixMatchStrategy(
    RuntimeMeasurementQualifierSuffixMatchStrategy
):
    """Match free-form identifier qualifier suffixes."""

    value_mode = RuntimeMeasurementQualifierValueMode.IDENTIFIER

    def matched_token_width(
        self,
        feature_tokens: tuple[str, ...],
        end_index: int,
        qualifier: RuntimeMeasurementRowQualifier,
    ) -> int | None:
        del feature_tokens, qualifier
        return 1 if end_index > 0 else None


class TwoDigitIntegerQualifierSuffixMatchStrategy(
    RuntimeMeasurementQualifierSuffixMatchStrategy
):
    """Match two-digit integer qualifier suffixes."""

    value_mode = RuntimeMeasurementQualifierValueMode.TWO_DIGIT_INTEGER

    def matched_token_width(
        self,
        feature_tokens: tuple[str, ...],
        end_index: int,
        qualifier: RuntimeMeasurementRowQualifier,
    ) -> int | None:
        del qualifier
        if end_index <= 0:
            return None
        token = feature_tokens[end_index - 1]
        return 1 if len(token) == 2 and token.isdigit() else None


class FractionOfCountQualifierSuffixMatchStrategy(
    RuntimeMeasurementQualifierSuffixMatchStrategy
):
    """Match ``N of M`` fraction-count qualifier suffixes."""

    value_mode = RuntimeMeasurementQualifierValueMode.FRACTION_OF_COUNT

    def matched_token_width(
        self,
        feature_tokens: tuple[str, ...],
        end_index: int,
        qualifier: RuntimeMeasurementRowQualifier,
    ) -> int | None:
        del qualifier
        if end_index < 3:
            return None
        left, of_token, right = feature_tokens[end_index - 3 : end_index]
        return 3 if left.isdigit() and of_token == "of" and right.isdigit() else None


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementRowQualifierSequence:
    """Declared row-qualifier sequence rendered as one feature suffix."""

    field_names_by_qualifier: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        field_names_by_qualifier = tuple(
            tuple(
                normalize_runtime_identifier(field_name)
                for field_name in field_names
                if str(field_name).strip()
            )
            for field_names in self.field_names_by_qualifier
        )
        if not field_names_by_qualifier or any(
            not field_names for field_names in field_names_by_qualifier
        ):
            raise ValueError(
                "RuntimeMeasurementRowQualifierSequence requires non-empty "
                "qualifier field names."
            )
        object.__setattr__(
            self,
            "field_names_by_qualifier",
            field_names_by_qualifier,
        )


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementRowIdentityContract:
    """Declarative identity-field precedence for measurement table rows."""

    primary_image_fields: frozenset[str] = frozenset({"slice_index"})
    fallback_image_fields: frozenset[str] = frozenset({"image_number", "image_id"})

    def __post_init__(self) -> None:
        primary_image_fields = frozenset(
            normalize_runtime_identifier(field_name)
            for field_name in self.primary_image_fields
            if str(field_name).strip()
        )
        fallback_image_fields = frozenset(
            normalize_runtime_identifier(field_name)
            for field_name in self.fallback_image_fields
            if str(field_name).strip()
        )
        overlap = primary_image_fields & fallback_image_fields
        if overlap:
            raise ValueError(
                "RuntimeMeasurementRowIdentityContract fields must be disjoint: "
                f"{sorted(overlap)!r}."
            )
        object.__setattr__(self, "primary_image_fields", primary_image_fields)
        object.__setattr__(self, "fallback_image_fields", fallback_image_fields)

    @property
    def image_identity_fields(self) -> frozenset[str]:
        """Return every field that can identify an image row."""
        return self.primary_image_fields | self.fallback_image_fields

    def selected_image_identity_fields(
        self,
        normalized_present_fields: frozenset[str],
    ) -> frozenset[str]:
        """Return the identity fields that own a row under this contract."""
        primary_fields = normalized_present_fields & self.primary_image_fields
        if primary_fields:
            return primary_fields
        return normalized_present_fields & self.fallback_image_fields


DEFAULT_RUNTIME_MEASUREMENT_ROW_IDENTITY_CONTRACT = (
    RuntimeMeasurementRowIdentityContract()
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
_DEFAULT_MEASUREMENT_ROW_QUALIFIER_SEQUENCES = (
    RuntimeMeasurementRowQualifierSequence(
        (("scale",), ("direction",), ("gray_levels",))
    ),
    RuntimeMeasurementRowQualifierSequence((("bin_index", "bin_count"),)),
)


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementDialect:
    """Policy-provided measurement-name dialect for semantic comparisons."""

    category_prefixes: tuple[tuple[str, ...], ...] = ()
    category_prefixes_provider: Callable[[], Iterable[tuple[str, ...]]] | None = None
    feature_part_aliases: Mapping[tuple[str, ...], tuple[str, ...]] = field(
        default_factory=lambda: _EMPTY_FEATURE_ALIASES
    )
    feature_part_aliases_provider: (
        Callable[[], Mapping[tuple[str, ...], tuple[str, ...]]] | None
    ) = None
    source_feature_prefixes: tuple[tuple[str, ...], ...] = ()
    source_feature_prefixes_provider: (
        Callable[[], Iterable[tuple[str, ...]]] | None
    ) = None
    calculated_feature_prefixes: tuple[tuple[str, ...], ...] = ()
    calculated_feature_prefixes_provider: (
        Callable[[], Iterable[tuple[str, ...]]] | None
    ) = None
    directional_pair_feature_aliases: Mapping[str, tuple[str, int]] = field(
        default_factory=lambda: _EMPTY_PAIR_FEATURE_ALIASES
    )
    directional_pair_feature_aliases_provider: (
        Callable[[], Mapping[str, tuple[str, int]]] | None
    ) = None
    scale_qualified_feature_prefixes: tuple[tuple[str, ...], ...] = ()
    scale_qualified_feature_prefixes_provider: (
        Callable[[], Iterable[tuple[str, ...]]] | None
    ) = None
    pair_correlation_feature_name: str | None = None
    pair_correlation_feature_name_provider: Callable[[], str | None] | None = None
    pair_regression_slope_feature_name: str | None = None
    pair_regression_slope_feature_name_provider: Callable[[], str | None] | None = None
    undirected_pair_feature_names: frozenset[str] = frozenset()
    undirected_pair_feature_names_provider: Callable[[], Iterable[str]] | None = None
    threshold_sensitive_pair_feature_names: frozenset[str] = frozenset()
    threshold_sensitive_pair_feature_names_provider: (
        Callable[[], Iterable[str]] | None
    ) = None
    threshold_qualifier_tokens: frozenset[str] = frozenset()
    source_qualifier_prefix_tokens: frozenset[str] = frozenset()
    source_qualifier_suffix_tokens: frozenset[str] = frozenset()
    row_qualifiers: tuple[RuntimeMeasurementRowQualifier, ...] = (
        _DEFAULT_MEASUREMENT_ROW_QUALIFIERS
    )
    source_suffix_qualifier_sequences: tuple[
        RuntimeMeasurementRowQualifierSequence, ...
    ] = _DEFAULT_MEASUREMENT_ROW_QUALIFIER_SEQUENCES
    numbered_feature_prefix_aliases: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: _EMPTY_NUMBERED_FEATURE_PREFIX_ALIASES
    )
    numbered_feature_prefix_aliases_provider: (
        Callable[[], Mapping[str, tuple[str, ...]]] | None
    ) = None
    source_name_encoding_by_scope: Mapping[
        MeasurementScope,
        RuntimeMeasurementSourceNameEncoding,
    ] = field(default_factory=lambda: MappingProxyType({}))
    row_identity_contract: RuntimeMeasurementRowIdentityContract = (
        DEFAULT_RUNTIME_MEASUREMENT_ROW_IDENTITY_CONTRACT
    )
    measurement_feature_relation_provider: (
        Callable[
            [],
            Iterable[RuntimeMeasurementFeatureRelationDeclaration],
        ]
        | None
    ) = None
    measurement_feature_marker_provider: (
        Callable[
            [object, "RuntimeMeasurementDialect"],
            Iterable[type[RuntimeMeasurementFeatureSemanticMarker]],
        ]
        | None
    ) = None
    indexed_descriptor_suffix_width_provider: (
        Callable[[tuple[str, ...]], int | None] | None
    ) = None

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
            "calculated_feature_prefixes",
            tuple(
                tuple(part for part in prefix if part)
                for prefix in self.calculated_feature_prefixes
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
            "pair_correlation_feature_name",
            (
                None
                if self.pair_correlation_feature_name is None
                else normalize_runtime_identifier(self.pair_correlation_feature_name)
            ),
        )
        object.__setattr__(
            self,
            "pair_regression_slope_feature_name",
            (
                None
                if self.pair_regression_slope_feature_name is None
                else normalize_runtime_identifier(
                    self.pair_regression_slope_feature_name
                )
            ),
        )
        object.__setattr__(
            self,
            "undirected_pair_feature_names",
            frozenset(
                normalize_runtime_identifier(feature_name)
                for feature_name in self.undirected_pair_feature_names
            ),
        )
        object.__setattr__(
            self,
            "threshold_sensitive_pair_feature_names",
            frozenset(
                normalize_runtime_identifier(feature_name)
                for feature_name in self.threshold_sensitive_pair_feature_names
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
            frozenset(str(token) for token in self.source_qualifier_prefix_tokens),
        )
        object.__setattr__(
            self,
            "source_qualifier_suffix_tokens",
            frozenset(str(token) for token in self.source_qualifier_suffix_tokens),
        )
        object.__setattr__(
            self,
            "row_qualifiers",
            tuple(
                (
                    qualifier
                    if isinstance(qualifier, RuntimeMeasurementRowQualifier)
                    else RuntimeMeasurementRowQualifier(tuple(qualifier))
                )
                for qualifier in self.row_qualifiers
            ),
        )
        if self.measurement_feature_marker_provider is not None and not callable(
            self.measurement_feature_marker_provider
        ):
            raise TypeError(
                "RuntimeMeasurementDialect.measurement_feature_marker_provider "
                "must be callable."
            )
        if (
            self.indexed_descriptor_suffix_width_provider is not None
            and not callable(self.indexed_descriptor_suffix_width_provider)
        ):
            raise TypeError(
                "RuntimeMeasurementDialect.indexed_descriptor_suffix_width_provider "
                "must be callable."
            )
        object.__setattr__(
            self,
            "source_suffix_qualifier_sequences",
            tuple(
                (
                    sequence
                    if isinstance(sequence, RuntimeMeasurementRowQualifierSequence)
                    else RuntimeMeasurementRowQualifierSequence(tuple(sequence))
                )
                for sequence in self.source_suffix_qualifier_sequences
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
        object.__setattr__(
            self,
            "source_name_encoding_by_scope",
            MappingProxyType(
                {
                    (
                        scope
                        if isinstance(scope, MeasurementScope)
                        else MeasurementScope(scope)
                    ): (
                        encoding
                        if isinstance(
                            encoding,
                            RuntimeMeasurementSourceNameEncoding,
                        )
                        else RuntimeMeasurementSourceNameEncoding(encoding)
                    )
                    for scope, encoding in self.source_name_encoding_by_scope.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "row_identity_contract",
            (
                self.row_identity_contract
                if isinstance(
                    self.row_identity_contract,
                    RuntimeMeasurementRowIdentityContract,
                )
                else RuntimeMeasurementRowIdentityContract(
                    **self.row_identity_contract  # type: ignore[arg-type]
                )
            ),
        )
        if self.measurement_feature_relation_provider is not None and not callable(
            self.measurement_feature_relation_provider
        ):
            raise TypeError(
                "RuntimeMeasurementDialect.measurement_feature_relation_provider "
                "must be callable."
            )
        for field_name in (
            "category_prefixes_provider",
            "feature_part_aliases_provider",
            "source_feature_prefixes_provider",
            "calculated_feature_prefixes_provider",
            "directional_pair_feature_aliases_provider",
            "scale_qualified_feature_prefixes_provider",
            "pair_correlation_feature_name_provider",
            "pair_regression_slope_feature_name_provider",
            "undirected_pair_feature_names_provider",
            "threshold_sensitive_pair_feature_names_provider",
            "numbered_feature_prefix_aliases_provider",
        ):
            provider = object.__getattribute__(self, field_name)
            if provider is not None and not callable(provider):
                raise TypeError(
                    f"RuntimeMeasurementDialect.{field_name} must be callable."
                )

    def resolved_category_prefixes(self) -> tuple[tuple[str, ...], ...]:
        """Return static and provider-supplied measurement category prefixes."""
        return _resolved_category_prefixes(
            runtime_measurement_dialect_cache_id(self)
        )

    def resolved_feature_part_aliases(self) -> Mapping[tuple[str, ...], tuple[str, ...]]:
        """Return static and provider-supplied direct feature-part aliases."""
        return _resolved_feature_part_aliases(
            runtime_measurement_dialect_cache_id(self)
        )

    def resolved_source_feature_prefixes(self) -> tuple[tuple[str, ...], ...]:
        """Return static and provider-supplied source-feature prefixes."""
        return _resolved_source_feature_prefixes(
            runtime_measurement_dialect_cache_id(self)
        )

    def resolved_calculated_feature_prefixes(self) -> tuple[tuple[str, ...], ...]:
        """Return static and provider-supplied calculated feature prefixes."""
        return _resolved_calculated_feature_prefixes(
            runtime_measurement_dialect_cache_id(self)
        )

    def resolved_numbered_feature_prefix_aliases(
        self,
    ) -> Mapping[str, tuple[str, ...]]:
        """Return static and provider-supplied numbered feature-prefix aliases."""
        return _resolved_numbered_feature_prefix_aliases(
            runtime_measurement_dialect_cache_id(self)
        )

    def resolved_directional_pair_feature_aliases(
        self,
    ) -> Mapping[str, tuple[str, int]]:
        """Return static and provider-supplied directional pair aliases."""
        return _resolved_directional_pair_feature_aliases(
            runtime_measurement_dialect_cache_id(self)
        )

    def resolved_scale_qualified_feature_prefixes(self) -> tuple[tuple[str, ...], ...]:
        """Return static and provider-supplied scale-qualified prefixes."""
        return _resolved_scale_qualified_feature_prefixes(
            runtime_measurement_dialect_cache_id(self)
        )

    def resolved_pair_correlation_feature_name(self) -> str | None:
        """Return static or provider-supplied pair correlation feature name."""
        provider = self.pair_correlation_feature_name_provider
        value = self.pair_correlation_feature_name if provider is None else provider()
        return None if value is None else normalize_runtime_identifier(value)

    def resolved_pair_regression_slope_feature_name(self) -> str | None:
        """Return static or provider-supplied pair regression-slope feature name."""
        provider = self.pair_regression_slope_feature_name_provider
        value = (
            self.pair_regression_slope_feature_name if provider is None else provider()
        )
        return None if value is None else normalize_runtime_identifier(value)

    def resolved_undirected_pair_feature_names(self) -> frozenset[str]:
        """Return static and provider-supplied undirected pair feature names."""
        return _resolved_undirected_pair_feature_names(
            runtime_measurement_dialect_cache_id(self)
        )

    def resolved_threshold_sensitive_pair_feature_names(self) -> frozenset[str]:
        """Return static and provider-supplied threshold-sensitive pair names."""
        return _resolved_threshold_sensitive_pair_feature_names(
            runtime_measurement_dialect_cache_id(self)
        )

    def measurement_feature_relation_declarations(
        self,
    ) -> RuntimeMeasurementFeatureRelationDeclarationCollection:
        """Return producer-declared measurement-feature relations."""
        provider = self.measurement_feature_relation_provider
        return RuntimeMeasurementFeatureRelationDeclarationCollection(
            () if provider is None else provider()
        )

    def measurement_feature_marker_types(
        self,
        key: object,
    ) -> tuple[type[RuntimeMeasurementFeatureSemanticMarker], ...]:
        """Return producer-declared semantic marker types for one feature key."""
        provider = self.measurement_feature_marker_provider
        if provider is None:
            return ()
        marker_types = tuple(provider(key, self))
        for marker_type in marker_types:
            if not isinstance(marker_type, type) or not issubclass(
                marker_type,
                RuntimeMeasurementFeatureSemanticMarker,
            ):
                raise TypeError(
                    "RuntimeMeasurementDialect.measurement_feature_marker_provider "
                    "must return RuntimeMeasurementFeatureSemanticMarker types."
                )
        return marker_types

    def source_feature_family_for_relation(
        self,
        relation_type: type[RuntimeMeasurementFeatureRelation],
        feature_name: str,
        source_name: str | None,
        scope: MeasurementScope,
    ) -> RuntimeMeasurementSourceQualifiedFeature | None:
        """Return the source-qualified family for one declared relation type."""
        return self.source_qualified_feature_family(
            feature_name,
            source_name,
            scope,
            self.measurement_feature_relation_declarations().source_family_names(
                relation_type,
            ),
        )

    def target_family_for_relation_source_family(
        self,
        relation_type: type[RuntimeMeasurementFeatureRelation],
        source_family_name: str,
    ) -> str | None:
        """Return the declared target family for one relation source family."""
        return self.measurement_feature_relation_declarations().target_family_name(
            relation_type,
            source_family_name,
        )

    def source_name_encoding(
        self,
        scope: MeasurementScope,
    ) -> RuntimeMeasurementSourceNameEncoding:
        """Return the source-name encoding declared for a measured scope."""
        return self.source_name_encoding_by_scope.get(
            scope,
            RuntimeMeasurementSourceNameEncoding.SEPARATE_KEY,
        )

    def encode_source_qualified_feature(
        self,
        feature_name: str,
        source_name: str | None,
        scope: MeasurementScope,
        *,
        qualifiers: tuple[str, ...] = (),
    ) -> RuntimeMeasurementSourceQualifiedFeature:
        """Encode source identity into a feature according to this dialect."""
        normalized_feature_name = normalize_runtime_identifier(feature_name)
        normalized_source_name = normalize_runtime_source_name(source_name)
        encoding = self.source_name_encoding(scope)
        if (
            normalized_source_name is None
            or encoding is RuntimeMeasurementSourceNameEncoding.SEPARATE_KEY
        ):
            return RuntimeMeasurementSourceQualifiedFeature(
                normalized_feature_name,
                normalized_source_name,
            )
        if encoding is not RuntimeMeasurementSourceNameEncoding.FEATURE_SUFFIX:
            raise ValueError(
                f"Unsupported measurement source-name encoding: {encoding}."
            )
        source_tokens = runtime_source_name_tokens(normalized_source_name)
        if not source_tokens:
            return RuntimeMeasurementSourceQualifiedFeature(normalized_feature_name)
        feature_tokens = tuple(
            token for token in normalized_feature_name.split("_") if token
        )
        qualifier_tokens = self.source_qualified_feature_qualifier_tokens(qualifiers)
        encoded_tokens = self.place_feature_suffix_source_tokens(
            feature_tokens,
            source_tokens,
            qualifier_tokens,
        )
        return RuntimeMeasurementSourceQualifiedFeature("_".join(encoded_tokens))

    def source_qualified_feature_qualifier_tokens(
        self,
        qualifiers: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Return qualifier tokens used to place feature-suffix source names."""
        return tuple(
            token
            for qualifier in qualifiers
            for token in normalize_runtime_identifier(qualifier).split("_")
            if token
        )

    def place_feature_suffix_source_tokens(
        self,
        feature_tokens: tuple[str, ...],
        source_tokens: tuple[str, ...],
        qualifier_tokens: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        """Place feature-suffix source tokens before declared row qualifiers."""
        if not source_tokens:
            return feature_tokens
        if not qualifier_tokens:
            qualifier_tokens = self.infer_feature_suffix_qualifier_tokens(
                feature_tokens
            )
        suffix_start = self.feature_suffix_source_insertion_index(
            feature_tokens,
            qualifier_tokens,
        )
        if (
            feature_tokens[suffix_start - len(source_tokens) : suffix_start]
            == source_tokens
        ):
            return feature_tokens
        if feature_tokens[-len(source_tokens) :] == source_tokens:
            return feature_tokens
        return (
            *feature_tokens[:suffix_start],
            *source_tokens,
            *feature_tokens[suffix_start:],
        )

    def infer_feature_suffix_qualifier_tokens(
        self,
        feature_tokens: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Infer declared row-qualifier suffix tokens from a flat feature name."""
        descriptor_suffix_width = self.indexed_descriptor_suffix_token_width(
            feature_tokens
        )
        if descriptor_suffix_width is not None:
            return feature_tokens[-descriptor_suffix_width:]
        matches: list[tuple[str, ...]] = []
        for qualifiers in self.source_suffix_qualifier_sequence_qualifiers():
            suffix_length = self.match_feature_suffix_qualifier_sequence(
                feature_tokens,
                qualifiers,
            )
            if suffix_length is None:
                continue
            suffix_tokens = feature_tokens[-suffix_length:]
            if not self.qualifier_sequence_identifies_feature_suffix(
                feature_tokens,
                suffix_tokens,
                qualifiers,
            ):
                continue
            matches.append(suffix_tokens)
        if not matches:
            return ()
        return max(matches, key=len)

    def indexed_descriptor_suffix_token_width(
        self,
        feature_tokens: tuple[str, ...],
    ) -> int | None:
        """Return the trailing descriptor-index token width declared for a feature."""
        provider = self.indexed_descriptor_suffix_width_provider
        if provider is None:
            return None
        suffix_width = provider(tuple(feature_tokens))
        if suffix_width is None:
            return None
        suffix_width = int(suffix_width)
        if suffix_width <= 0 or suffix_width > len(feature_tokens):
            raise ValueError(
                "Indexed descriptor suffix width must be within feature token "
                f"bounds: width={suffix_width!r}, tokens={feature_tokens!r}."
            )
        return suffix_width

    def source_suffix_qualifier_sequence_qualifiers(
        self,
    ) -> tuple[tuple[RuntimeMeasurementRowQualifier, ...], ...]:
        """Return declared source-suffix qualifier sequences as qualifier objects."""
        qualifiers_by_fields = {
            qualifier.field_names: qualifier for qualifier in self.row_qualifiers
        }
        sequences: list[tuple[RuntimeMeasurementRowQualifier, ...]] = []
        for sequence in self.source_suffix_qualifier_sequences:
            qualifiers: list[RuntimeMeasurementRowQualifier] = []
            for field_names in sequence.field_names_by_qualifier:
                qualifier = qualifiers_by_fields.get(field_names)
                if qualifier is None:
                    raise ValueError(
                        "RuntimeMeasurementDialect source suffix qualifier "
                        f"sequence references undeclared qualifier {field_names!r}."
                    )
                qualifiers.append(qualifier)
            sequences.append(tuple(qualifiers))
        return tuple(sequences)

    def match_feature_suffix_qualifier_sequence(
        self,
        feature_tokens: tuple[str, ...],
        qualifiers: tuple[RuntimeMeasurementRowQualifier, ...],
    ) -> int | None:
        """Return matched suffix length for a declared qualifier sequence."""
        cursor = len(feature_tokens)
        for qualifier in reversed(qualifiers):
            token_count = (
                RuntimeMeasurementQualifierSuffixMatchStrategy.for_enum_member(
                    qualifier.value_mode
                ).matched_token_width(feature_tokens, cursor, qualifier)
            )
            if token_count is None:
                return None
            cursor -= token_count
        return len(feature_tokens) - cursor

    def qualifier_sequence_identifies_feature_suffix(
        self,
        feature_tokens: tuple[str, ...],
        suffix_tokens: tuple[str, ...],
        qualifiers: tuple[RuntimeMeasurementRowQualifier, ...],
    ) -> bool:
        """Return whether a qualifier sequence is distinctive enough to infer."""
        if any(
            qualifier.value_mode is not RuntimeMeasurementQualifierValueMode.IDENTIFIER
            for qualifier in qualifiers
        ):
            return True
        base_tokens = feature_tokens[: len(feature_tokens) - len(suffix_tokens)]
        return any(
            base_tokens == prefix
            for prefix in self.resolved_scale_qualified_feature_prefixes()
        )

    def feature_suffix_source_insertion_index(
        self,
        feature_tokens: tuple[str, ...],
        qualifier_tokens: tuple[str, ...] = (),
    ) -> int:
        """Return where feature-suffix source identity belongs."""
        if (
            qualifier_tokens
            and len(feature_tokens) >= len(qualifier_tokens)
            and feature_tokens[-len(qualifier_tokens) :] == qualifier_tokens
        ):
            return len(feature_tokens) - len(qualifier_tokens)
        return len(feature_tokens)

    def source_qualified_feature_family(
        self,
        feature_name: str,
        source_name: str | None,
        scope: MeasurementScope,
        feature_families: Iterable[str],
    ) -> RuntimeMeasurementSourceQualifiedFeature | None:
        """Bind a feature to a declared base family and source identity."""
        normalized_feature_name = normalize_runtime_identifier(feature_name)
        normalized_source_name = normalize_runtime_source_name(source_name)
        normalized_families = tuple(
            sorted(
                {
                    normalize_runtime_identifier(family)
                    for family in feature_families
                    if str(family).strip()
                },
                key=lambda family: (-len(family), family),
            )
        )
        if not normalized_families:
            return None
        encoding = self.source_name_encoding(scope)
        if encoding is RuntimeMeasurementSourceNameEncoding.SEPARATE_KEY:
            if normalized_feature_name in normalized_families:
                return RuntimeMeasurementSourceQualifiedFeature(
                    normalized_feature_name,
                    normalized_source_name,
                )
            return None
        if encoding is not RuntimeMeasurementSourceNameEncoding.FEATURE_SUFFIX:
            raise ValueError(
                f"Unsupported measurement source-name encoding: {encoding}."
            )
        if normalized_feature_name in normalized_families:
            return RuntimeMeasurementSourceQualifiedFeature(normalized_feature_name)
        for family in normalized_families:
            prefix = f"{family}_"
            if normalized_feature_name.startswith(prefix):
                return RuntimeMeasurementSourceQualifiedFeature(
                    family,
                    normalized_feature_name[len(prefix) :],
                )
        return None


@lru_cache(maxsize=64)
def _resolved_category_prefixes(
    dialect_id: int,
) -> tuple[tuple[str, ...], ...]:
    dialect = runtime_measurement_dialect_for_cache_id(dialect_id)
    provider = dialect.category_prefixes_provider
    provided = () if provider is None else provider()
    return tuple(
        dict.fromkeys(
            (
                *dialect.category_prefixes,
                *(tuple(part for part in prefix if part) for prefix in provided),
            )
        )
    )


@lru_cache(maxsize=64)
def _resolved_feature_part_aliases(
    dialect_id: int,
) -> Mapping[tuple[str, ...], tuple[str, ...]]:
    dialect = runtime_measurement_dialect_for_cache_id(dialect_id)
    provider = dialect.feature_part_aliases_provider
    provided = {} if provider is None else provider()
    return MappingProxyType(
        {
            **dialect.feature_part_aliases,
            **{
                tuple(part for part in parts if part): tuple(
                    part for part in alias if part
                )
                for parts, alias in provided.items()
            },
        }
    )


@lru_cache(maxsize=64)
def _resolved_source_feature_prefixes(
    dialect_id: int,
) -> tuple[tuple[str, ...], ...]:
    dialect = runtime_measurement_dialect_for_cache_id(dialect_id)
    provider = dialect.source_feature_prefixes_provider
    provided = () if provider is None else provider()
    return tuple(
        dict.fromkeys(
            (
                *dialect.source_feature_prefixes,
                *(tuple(part for part in prefix if part) for prefix in provided),
            )
        )
    )


@lru_cache(maxsize=64)
def _resolved_calculated_feature_prefixes(
    dialect_id: int,
) -> tuple[tuple[str, ...], ...]:
    dialect = runtime_measurement_dialect_for_cache_id(dialect_id)
    provider = dialect.calculated_feature_prefixes_provider
    provided = () if provider is None else provider()
    return tuple(
        dict.fromkeys(
            (
                *dialect.calculated_feature_prefixes,
                *(tuple(part for part in prefix if part) for prefix in provided),
            )
        )
    )


@lru_cache(maxsize=64)
def _resolved_numbered_feature_prefix_aliases(
    dialect_id: int,
) -> Mapping[str, tuple[str, ...]]:
    dialect = runtime_measurement_dialect_for_cache_id(dialect_id)
    provider = dialect.numbered_feature_prefix_aliases_provider
    provided = {} if provider is None else provider()
    return MappingProxyType(
        {
            **dialect.numbered_feature_prefix_aliases,
            **{
                normalize_runtime_identifier(prefix): tuple(
                    normalize_runtime_identifier(part)
                    for part in alias
                    if str(part).strip()
                )
                for prefix, alias in provided.items()
                if str(prefix).strip()
            },
        }
    )


@lru_cache(maxsize=64)
def _resolved_directional_pair_feature_aliases(
    dialect_id: int,
) -> Mapping[str, tuple[str, int]]:
    dialect = runtime_measurement_dialect_for_cache_id(dialect_id)
    provider = dialect.directional_pair_feature_aliases_provider
    provided = {} if provider is None else provider()
    return MappingProxyType(
        {
            **dialect.directional_pair_feature_aliases,
            **{
                str(name): (str(alias[0]), int(alias[1]))
                for name, alias in provided.items()
            },
        }
    )


@lru_cache(maxsize=64)
def _resolved_scale_qualified_feature_prefixes(
    dialect_id: int,
) -> tuple[tuple[str, ...], ...]:
    dialect = runtime_measurement_dialect_for_cache_id(dialect_id)
    provider = dialect.scale_qualified_feature_prefixes_provider
    provided = () if provider is None else provider()
    return tuple(
        dict.fromkeys(
            (
                *dialect.scale_qualified_feature_prefixes,
                *(tuple(part for part in prefix if part) for prefix in provided),
            )
        )
    )


@lru_cache(maxsize=64)
def _resolved_undirected_pair_feature_names(dialect_id: int) -> frozenset[str]:
    dialect = runtime_measurement_dialect_for_cache_id(dialect_id)
    provider = dialect.undirected_pair_feature_names_provider
    provided = () if provider is None else provider()
    return frozenset(
        normalize_runtime_identifier(feature_name)
        for feature_name in (*dialect.undirected_pair_feature_names, *provided)
    )


@lru_cache(maxsize=64)
def _resolved_threshold_sensitive_pair_feature_names(
    dialect_id: int,
) -> frozenset[str]:
    dialect = runtime_measurement_dialect_for_cache_id(dialect_id)
    provider = dialect.threshold_sensitive_pair_feature_names_provider
    provided = () if provider is None else provider()
    return frozenset(
        normalize_runtime_identifier(feature_name)
        for feature_name in (
            *dialect.threshold_sensitive_pair_feature_names,
            *provided,
        )
    )


DEFAULT_RUNTIME_MEASUREMENT_DIALECT = RuntimeMeasurementDialect()


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementFeatureNumericTolerance(
    RuntimePolicyNonNegativeFieldValidationMixin
):
    """Numeric tolerance scoped to a semantic measurement feature family."""

    feature_name_prefixes: tuple[str, ...] = ()
    feature_name_suffixes: tuple[str, ...] = ()
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
        feature_name_suffixes = tuple(
            str(suffix).strip()
            for suffix in self.feature_name_suffixes
            if str(suffix).strip()
        )
        feature_names = frozenset(
            str(feature_name).strip()
            for feature_name in self.feature_names
            if str(feature_name).strip()
        )
        if (
            not feature_name_prefixes
            and not feature_name_suffixes
            and not feature_names
        ):
            raise ValueError(
                "RuntimeMeasurementFeatureNumericTolerance requires at least "
                "one feature name, prefix, or suffix."
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
        self.validate_non_negative_policy_fields()
        object.__setattr__(self, "feature_name_prefixes", feature_name_prefixes)
        object.__setattr__(self, "feature_name_suffixes", feature_name_suffixes)
        object.__setattr__(self, "feature_names", feature_names)
        object.__setattr__(self, "subject_scope", subject_scope)
        object.__setattr__(self, "statistic", statistic)


@dataclass(frozen=True, slots=True)
class RuntimeEquivalencePolicy(RuntimePolicyNonNegativeFieldValidationMixin):
    """Policy controlling semantic output comparison strictness."""

    numeric_decimal_places: NonNegativeInt = 10
    numeric_abs_tolerance: NonNegativeFloat = 0.0
    numeric_rel_tolerance: NonNegativeFloat = 0.0
    allow_tie_sensitive_location_mismatches: bool = False
    allow_unstable_shape_descriptors: bool = False
    shape_descriptor_abs_tolerance: NonNegativeFloat = 1e-6
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
    measurement_dialect: RuntimeMeasurementDialect = DEFAULT_RUNTIME_MEASUREMENT_DIALECT
    feature_numeric_tolerances: tuple[
        RuntimeMeasurementFeatureNumericTolerance, ...
    ] = ()
    feature_numeric_tolerances_provider: (
        Callable[[], Iterable[RuntimeMeasurementFeatureNumericTolerance]] | None
    ) = None

    def __post_init__(self) -> None:
        self.validate_non_negative_policy_fields()
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
                (
                    tolerance
                    if isinstance(
                        tolerance,
                        RuntimeMeasurementFeatureNumericTolerance,
                    )
                    else RuntimeMeasurementFeatureNumericTolerance(**tolerance)
                )
                for tolerance in self.feature_numeric_tolerances
            ),
        )
        if self.feature_numeric_tolerances_provider is not None and not callable(
            self.feature_numeric_tolerances_provider
        ):
            raise TypeError(
                "RuntimeEquivalencePolicy.feature_numeric_tolerances_provider "
                "must be callable."
            )

    def resolved_feature_numeric_tolerances(
        self,
    ) -> tuple[RuntimeMeasurementFeatureNumericTolerance, ...]:
        """Return static and provider-supplied feature numeric tolerances."""
        provider = self.feature_numeric_tolerances_provider
        provided = () if provider is None else provider()
        return tuple(
            (
                tolerance
                if isinstance(tolerance, RuntimeMeasurementFeatureNumericTolerance)
                else RuntimeMeasurementFeatureNumericTolerance(**tolerance)
            )
            for tolerance in (*self.feature_numeric_tolerances, *provided)
        )
