"""Runtime equivalence policy records."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from types import MappingProxyType
from typing import Annotated, ClassVar, get_args, get_origin, get_type_hints

from metaclass_registry import AutoRegisterMeta
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.core.runtime_semantics import MeasurementScope


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


def runtime_source_name_tokens(source_name: str) -> tuple[str, ...]:
    """Return canonical source-name tokens used by measurement dialects."""
    normalized = normalize_runtime_source_name(source_name)
    if normalized is None:
        return ()
    return tuple(
        token
        for part in normalized.split("__")
        for token in part.split("_")
        if token
    )


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

    primary_image_fields: frozenset[str] = frozenset({"image_number", "image_id"})
    fallback_image_fields: frozenset[str] = frozenset({"slice_index"})

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
    feature_part_aliases: Mapping[tuple[str, ...], tuple[str, ...]] = (
        _EMPTY_FEATURE_ALIASES
    )
    source_feature_prefixes: tuple[tuple[str, ...], ...] = ()
    calculated_feature_prefixes: tuple[tuple[str, ...], ...] = ()
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
    source_suffix_qualifier_sequences: tuple[
        RuntimeMeasurementRowQualifierSequence,
        ...
    ] = _DEFAULT_MEASUREMENT_ROW_QUALIFIER_SEQUENCES
    numbered_feature_prefix_aliases: Mapping[str, tuple[str, ...]] = (
        _EMPTY_NUMBERED_FEATURE_PREFIX_ALIASES
    )
    source_name_encoding_by_scope: Mapping[
        MeasurementScope,
        RuntimeMeasurementSourceNameEncoding,
    ] = MappingProxyType({})
    row_identity_contract: RuntimeMeasurementRowIdentityContract = (
        DEFAULT_RUNTIME_MEASUREMENT_ROW_IDENTITY_CONTRACT
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
            "source_suffix_qualifier_sequences",
            tuple(
                sequence
                if isinstance(sequence, RuntimeMeasurementRowQualifierSequence)
                else RuntimeMeasurementRowQualifierSequence(tuple(sequence))
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
            raise ValueError(f"Unsupported measurement source-name encoding: {encoding}.")
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
        if feature_tokens[suffix_start - len(source_tokens) : suffix_start] == source_tokens:
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

    def source_suffix_qualifier_sequence_qualifiers(
        self,
    ) -> tuple[tuple[RuntimeMeasurementRowQualifier, ...], ...]:
        """Return declared source-suffix qualifier sequences as qualifier objects."""
        qualifiers_by_fields = {
            qualifier.field_names: qualifier
            for qualifier in self.row_qualifiers
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
            token_count = self.match_feature_suffix_qualifier(
                feature_tokens,
                cursor,
                qualifier,
            )
            if token_count is None:
                return None
            cursor -= token_count
        return len(feature_tokens) - cursor

    def match_feature_suffix_qualifier(
        self,
        feature_tokens: tuple[str, ...],
        end_index: int,
        qualifier: RuntimeMeasurementRowQualifier,
    ) -> int | None:
        """Return token width if a qualifier owns the suffix ending at ``end_index``."""
        return (
            RuntimeMeasurementQualifierSuffixMatchStrategy.for_enum_member(
                qualifier.value_mode
            ).matched_token_width(feature_tokens, end_index, qualifier)
        )

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
            for prefix in self.scale_qualified_feature_prefixes
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
            raise ValueError(f"Unsupported measurement source-name encoding: {encoding}.")
        if normalized_feature_name in normalized_families:
            return RuntimeMeasurementSourceQualifiedFeature(normalized_feature_name)
        for family in normalized_families:
            prefix = f"{family}_"
            if normalized_feature_name.startswith(prefix):
                return RuntimeMeasurementSourceQualifiedFeature(
                    family,
                    normalized_feature_name[len(prefix):],
                )
        return None


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
