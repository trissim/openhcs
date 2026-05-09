"""Canonical runtime equivalence identity keys."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from openhcs.core.equivalence.policy import (
    RuntimeMeasurementDialect,
    RuntimeMeasurementSourceQualifiedFeature,
    RuntimeMeasurementSourceNameEncoding,
    normalize_runtime_identifier,
    normalize_runtime_source_name,
    runtime_source_name_tokens,
)
from openhcs.core.runtime_semantics import MeasurementScope, MeasurementSubject


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementSubjectKey:
    """Canonical measured subject for semantic measurement comparison."""

    scope: MeasurementScope
    name: str | None = None

    def __post_init__(self) -> None:
        scope = (
            self.scope
            if isinstance(self.scope, MeasurementScope)
            else MeasurementScope(self.scope)
        )
        name = (
            normalize_runtime_identifier(self.name)
            if self.name is not None
            else None
        )
        if name == "":
            raise ValueError("RuntimeMeasurementSubjectKey.name cannot be empty.")
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "name", name)

    @classmethod
    def from_subject(cls, subject: MeasurementSubject) -> "RuntimeMeasurementSubjectKey":
        """Build a comparison subject key from typed runtime semantics."""
        return cls(scope=subject.scope, name=subject.name)

    @classmethod
    def from_table_subject(
        cls,
        subject: MeasurementSubject,
    ) -> "RuntimeMeasurementSubjectKey":
        """Build the measured table subject, keeping image sources as qualifiers."""
        subject_key = cls.from_subject(subject)
        if subject_key.scope is MeasurementScope.IMAGE:
            return cls(MeasurementScope.IMAGE, MeasurementScope.IMAGE.value)
        return subject_key

    @property
    def sort_key(self) -> tuple[str, str]:
        return (self.scope.value, self.name or "")

    def to_cache_payload(self) -> tuple[str, str | None]:
        """Return a pickle/JSON-stable semantic cache payload."""
        return (self.scope.value, self.name)

    @classmethod
    def from_cache_payload(cls, payload: object) -> "RuntimeMeasurementSubjectKey":
        """Rebuild a measurement subject from a semantic cache payload."""
        scope, name = payload  # type: ignore[misc]
        return cls(MeasurementScope(str(scope)), None if name is None else str(name))

    def bind_row_source_identity(
        self,
        source_name: str | None,
    ) -> "RuntimeMeasurementSourceQualification":
        """Bind row source identity to this subject before feature-key encoding."""
        return RuntimeMeasurementSourceQualification(self, source_name)


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementSourceQualification:
    """Resolved relationship between a measurement subject and row source identity."""

    subject: RuntimeMeasurementSubjectKey
    row_source_name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "row_source_name",
            normalize_runtime_source_name(self.row_source_name),
        )

    @property
    def feature_source_name(self) -> str | None:
        """Return the source qualifier still available to the feature identity."""
        if (
            self.subject.scope is MeasurementScope.IMAGE
            and self.row_source_name == self.subject.name
        ):
            return None
        return self.row_source_name


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementSourcePair:
    """Ordered pair of source-image identities encoded by measurement dialects."""

    first: str
    second: str

    def __post_init__(self) -> None:
        first = normalize_runtime_source_name(self.first)
        second = normalize_runtime_source_name(self.second)
        if first is None or second is None:
            raise ValueError("RuntimeMeasurementSourcePair sources cannot be empty.")
        object.__setattr__(self, "first", first)
        object.__setattr__(self, "second", second)

    @classmethod
    def from_source_name(
        cls,
        source_name: str | None,
        known_source_names: Iterable[str] = (),
    ) -> "RuntimeMeasurementSourcePair | None":
        """Resolve a dialect source identity into an ordered source pair."""
        normalized = normalize_runtime_source_name(source_name)
        if normalized is None:
            return None
        direct_parts = tuple(part for part in normalized.split("__") if part)
        if len(direct_parts) == 2:
            return cls(direct_parts[0], direct_parts[1])

        source_tokens = tuple(token for token in normalized.split("_") if token)
        known_sources = cls.known_single_source_names(known_source_names)
        for split_index in range(1, len(source_tokens)):
            left = "_".join(source_tokens[:split_index])
            right = "_".join(source_tokens[split_index:])
            if left in known_sources and right in known_sources:
                return cls(left, right)
        return None

    @staticmethod
    def known_single_source_names(known_source_names: Iterable[str]) -> frozenset[str]:
        """Return normalized non-pair source names available for pair decoding."""
        return frozenset(
            normalized
            for source_name in known_source_names
            if (normalized := normalize_runtime_source_name(source_name)) is not None
            and "__" not in normalized
        )

    @property
    def source_name(self) -> str:
        """Return the ordered source-pair identity."""
        return f"{self.first}__{self.second}"

    @property
    def reversed_source_name(self) -> str:
        """Return the reversed ordered source-pair identity."""
        return f"{self.second}__{self.first}"

    @property
    def source_token_counter(self) -> Counter[str]:
        """Return order-insensitive tokens for pair companion matching."""
        return Counter(runtime_source_name_tokens(self.source_name))


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementFeatureKey:
    """Canonical measured feature on one semantic subject."""

    subject: RuntimeMeasurementSubjectKey
    feature_name: str
    statistic: str = "value"
    source_name: str | None = None

    def __post_init__(self) -> None:
        feature_name = self.feature_name.strip()
        if not feature_name:
            raise ValueError("RuntimeMeasurementFeatureKey.feature_name cannot be empty.")
        statistic = normalize_runtime_identifier(self.statistic)
        if not statistic:
            raise ValueError("RuntimeMeasurementFeatureKey.statistic cannot be empty.")
        source_name = (
            normalize_runtime_source_name(self.source_name)
            if self.source_name is not None
            else None
        )
        if source_name == "":
            raise ValueError("RuntimeMeasurementFeatureKey.source_name cannot be empty.")
        object.__setattr__(self, "feature_name", feature_name)
        object.__setattr__(self, "statistic", statistic)
        object.__setattr__(self, "source_name", source_name)

    @classmethod
    def from_source_qualified_feature(
        cls,
        subject: RuntimeMeasurementSubjectKey,
        feature_name: str,
        source_name: str | None,
        measurement_dialect: RuntimeMeasurementDialect,
        statistic: str = "value",
        qualifiers: tuple[str, ...] = (),
    ) -> "RuntimeMeasurementFeatureKey":
        """Build a key through the dialect's declared source-name encoding."""
        source_qualified_feature = (
            measurement_dialect.encode_source_qualified_feature(
                feature_name,
                source_name,
                subject.scope,
                qualifiers=qualifiers,
            )
        )
        if (
            subject.scope is MeasurementScope.IMAGE
            and source_qualified_feature.source_name is not None
            and measurement_dialect.source_name_encoding(subject.scope)
            is RuntimeMeasurementSourceNameEncoding.SEPARATE_KEY
        ):
            return cls(
                RuntimeMeasurementSubjectKey(
                    MeasurementScope.IMAGE,
                    source_qualified_feature.source_name,
                ),
                source_qualified_feature.feature_name,
                statistic,
            )
        return cls(
            subject,
            source_qualified_feature.feature_name,
            statistic,
            source_qualified_feature.source_name,
        )

    @property
    def sort_key(self) -> tuple[tuple[str, str], str, str, str]:
        return (
            self.subject.sort_key,
            self.statistic,
            self.feature_name,
            self.source_name or "",
        )

    def source_qualified_feature_family(
        self,
        measurement_dialect: RuntimeMeasurementDialect,
        feature_families: Iterable[str],
    ) -> RuntimeMeasurementSourceQualifiedFeature | None:
        """Bind this key to a source-qualified semantic feature family."""
        return measurement_dialect.source_qualified_feature_family(
            self.feature_name,
            self.source_name,
            self.subject.scope,
            feature_families,
        )

    def belongs_to_source_qualified_feature_family(
        self,
        measurement_dialect: RuntimeMeasurementDialect,
        feature_families: Iterable[str],
    ) -> bool:
        """Return whether this key belongs to a source-qualified family."""
        return (
            self.source_qualified_feature_family(
                measurement_dialect,
                feature_families,
            )
            is not None
        )

    def source_qualified_feature_source_name(
        self,
        measurement_dialect: RuntimeMeasurementDialect,
        feature_families: Iterable[str],
    ) -> str | None:
        """Return source identity encoded in this key for declared families."""
        feature_family = self.source_qualified_feature_family(
            measurement_dialect,
            feature_families,
        )
        if feature_family is not None and feature_family.source_name is not None:
            return feature_family.source_name
        if self.source_name is not None:
            return self.source_name
        if self.subject.scope is MeasurementScope.IMAGE:
            return self.subject.name
        return None

    def source_pair(
        self,
        measurement_dialect: RuntimeMeasurementDialect,
        feature_families: Iterable[str],
        known_source_names: Iterable[str] = (),
    ) -> RuntimeMeasurementSourcePair | None:
        """Return the ordered source-pair identity carried by this key."""
        return RuntimeMeasurementSourcePair.from_source_name(
            self.source_qualified_feature_source_name(
                measurement_dialect,
                feature_families,
            ),
            known_source_names,
        )

    def source_token_counter(
        self,
        measurement_dialect: RuntimeMeasurementDialect,
        feature_families: Iterable[str],
    ) -> Counter[str] | None:
        """Return order-insensitive source tokens carried by this feature key."""
        source_name = self.source_qualified_feature_source_name(
            measurement_dialect,
            feature_families,
        )
        if source_name is None:
            return None
        tokens = runtime_source_name_tokens(source_name)
        if len(tokens) < 2:
            return None
        return Counter(tokens)

    def source_pair_feature_key(
        self,
        source_name: str,
        measurement_dialect: RuntimeMeasurementDialect,
        feature_families: Iterable[str],
    ) -> "RuntimeMeasurementFeatureKey":
        """Return this semantic feature encoded for ``source_name``."""
        feature_family = self.source_qualified_feature_family(
            measurement_dialect,
            feature_families,
        )
        return type(self).from_source_qualified_feature(
            self.subject,
            self.feature_name if feature_family is None else feature_family.feature_name,
            source_name,
            measurement_dialect,
            self.statistic,
        )

    def reversed_source_pair_feature_key(
        self,
        measurement_dialect: RuntimeMeasurementDialect,
        feature_families: Iterable[str],
        known_source_names: Iterable[str] = (),
    ) -> "RuntimeMeasurementFeatureKey | None":
        """Return this feature key with source-pair orientation reversed."""
        pair = self.source_pair(
            measurement_dialect,
            feature_families,
            known_source_names,
        )
        if pair is None:
            return None
        return self.source_pair_feature_key(
            pair.reversed_source_name,
            measurement_dialect,
            feature_families,
        )

    def to_cache_payload(
        self,
    ) -> tuple[tuple[str, str | None], str, str, str | None]:
        """Return a pickle/JSON-stable semantic cache payload."""
        return (
            self.subject.to_cache_payload(),
            self.feature_name,
            self.statistic,
            self.source_name,
        )

    @classmethod
    def from_cache_payload(cls, payload: object) -> "RuntimeMeasurementFeatureKey":
        """Rebuild a measurement feature key from a semantic cache payload."""
        subject, feature_name, statistic, source_name = payload  # type: ignore[misc]
        return cls(
            RuntimeMeasurementSubjectKey.from_cache_payload(subject),
            str(feature_name),
            str(statistic),
            None if source_name is None else str(source_name),
        )
