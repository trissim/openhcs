"""Canonical runtime equivalence identity keys."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from types import MappingProxyType

from openhcs.core.equivalence.policy import (
    RuntimeMeasurementDialect,
    RuntimeEquivalencePolicy,
    RuntimeMeasurementFeatureNameMode,
    RuntimeMeasurementSourceQualifiedFeature,
    RuntimeMeasurementSourceNameEncoding,
    normalize_runtime_identifier,
    normalize_runtime_source_name,
    runtime_measurement_dialect_cache_id,
    runtime_measurement_dialect_for_cache_id,
    runtime_source_name_tokens,
)
from openhcs.core.runtime_measurements import (
    MeasurementScope,
    MeasurementStatistic,
    MeasurementSubject,
    ObjectCoreMeasurementFeature,
)

@dataclass(frozen=True, slots=True)
class RuntimeMeasurementSubjectKey:
    """Canonical measured subject for semantic measurement comparison."""

    scope: MeasurementScope
    name: str | None = None
    _hash_value: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        scope = (
            self.scope
            if isinstance(self.scope, MeasurementScope)
            else MeasurementScope(self.scope)
        )
        name = (
            normalize_runtime_identifier(self.name) if self.name is not None else None
        )
        if name == "":
            raise ValueError("RuntimeMeasurementSubjectKey.name cannot be empty.")
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "_hash_value", hash((scope, name)))

    def __hash__(self) -> int:
        return self._hash_value

    @classmethod
    def from_subject(
        cls, subject: MeasurementSubject
    ) -> "RuntimeMeasurementSubjectKey":
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
    ) -> "RuntimeMeasurementSourcePair | None":
        """Decode the canonical ordered source-pair identity."""
        normalized = normalize_runtime_source_name(source_name)
        if normalized is None:
            return None
        direct_parts = tuple(part for part in normalized.split("__") if part)
        if len(direct_parts) == 2:
            return cls(direct_parts[0], direct_parts[1])
        return None

    @property
    def source_name(self) -> str:
        """Return the ordered source-pair identity."""
        return self.source_pair_name(self.first, self.second)

    @property
    def reversed_source_name(self) -> str:
        """Return the reversed ordered source-pair identity."""
        return self.source_pair_name(self.second, self.first)

    @staticmethod
    def source_pair_name(first: str, second: str) -> str:
        """Return the ordered source-pair table identity."""
        return f"{first}__{second}"

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
    _hash_value: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        feature_name = self.feature_name.strip()
        if not feature_name:
            raise ValueError(
                "RuntimeMeasurementFeatureKey.feature_name cannot be empty."
            )
        statistic = normalize_runtime_identifier(self.statistic)
        if not statistic:
            raise ValueError("RuntimeMeasurementFeatureKey.statistic cannot be empty.")
        source_name = (
            normalize_runtime_source_name(self.source_name)
            if self.source_name is not None
            else None
        )
        if source_name == "":
            raise ValueError(
                "RuntimeMeasurementFeatureKey.source_name cannot be empty."
            )
        object.__setattr__(self, "feature_name", feature_name)
        object.__setattr__(self, "statistic", statistic)
        object.__setattr__(self, "source_name", source_name)
        object.__setattr__(
            self,
            "_hash_value",
            hash((self.subject, feature_name, statistic, source_name)),
        )

    def __hash__(self) -> int:
        return self._hash_value

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
        source_qualified_feature = measurement_dialect.encode_source_qualified_feature(
            feature_name,
            source_name,
            subject.scope,
            qualifiers=qualifiers,
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

    @classmethod
    def from_subject_feature(
        cls,
        subject: RuntimeMeasurementSubjectKey,
        feature_name: str,
        statistic: str = MeasurementStatistic.VALUE.value,
        source_name: str | None = None,
    ) -> "RuntimeMeasurementFeatureKey":
        """Build a key, folding image source identity into the image subject."""
        if subject.scope is MeasurementScope.IMAGE and source_name is not None:
            return cls(
                RuntimeMeasurementSubjectKey(MeasurementScope.IMAGE, source_name),
                feature_name,
                statistic,
            )
        return cls(subject, feature_name, statistic, source_name)

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
    ) -> RuntimeMeasurementSourcePair | None:
        """Return the ordered source-pair identity carried by this key."""
        return RuntimeMeasurementSourcePair.from_source_name(
            self.source_qualified_feature_source_name(
                measurement_dialect,
                feature_families,
            ),
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
            (
                self.feature_name
                if feature_family is None
                else feature_family.feature_name
            ),
            source_name,
            measurement_dialect,
            self.statistic,
        )

    def reversed_source_pair_feature_key(
        self,
        measurement_dialect: RuntimeMeasurementDialect,
        feature_families: Iterable[str],
    ) -> "RuntimeMeasurementFeatureKey | None":
        """Return this feature key with source-pair orientation reversed."""
        pair = self.source_pair(
            measurement_dialect,
            feature_families,
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


_RuntimeMeasurementNameParts = tuple[tuple[str, ...], tuple[str, ...]]
_RuntimeSourceTokenGroups = tuple[tuple[str, tuple[str, ...]], ...]
_MEASUREMENT_AGGREGATE_PREFIXES = frozenset({MeasurementStatistic.MEAN.value})


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementNamePartsProjection:
    """Dialect-aware semantic projection for runtime measurement feature parts."""

    parts: tuple[str, ...]
    dialect: RuntimeMeasurementDialect
    known_source_names: tuple[str, ...] = ()

    @classmethod
    def from_feature_name(
        cls,
        feature_name: str,
        dialect: RuntimeMeasurementDialect,
        *,
        known_source_names: tuple[str, ...] = (),
    ) -> "RuntimeMeasurementNamePartsProjection":
        return cls(
            normalized_measurement_feature_name_parts(feature_name),
            dialect,
            known_source_names,
        )

    def category_prefix(self) -> tuple[str, ...]:
        """Return the longest dialect category prefix matched by these parts."""
        return _category_prefix_for_parts(
            self.parts,
            runtime_measurement_dialect_cache_id(self.dialect),
        )

    def strip_category_prefix_for_core(self) -> "RuntimeMeasurementNamePartsProjection":
        prefix = _core_strip_category_prefix_for_parts(
            self.parts,
            runtime_measurement_dialect_cache_id(self.dialect),
        )
        if prefix:
            return RuntimeMeasurementNamePartsProjection(
                self.parts[len(prefix) :],
                self.dialect,
                self.known_source_names,
            )
        return self

    def should_strip_category_prefix(self, prefix: tuple[str, ...]) -> bool:
        return _should_strip_category_prefix(
            self.parts,
            prefix,
            runtime_measurement_dialect_cache_id(self.dialect),
        )

    def source_qualifier_tokens(self) -> _RuntimeMeasurementNameParts:
        source_token_groups = _source_name_token_groups(self.known_source_names)
        stripped: list[str] = []
        source_names: list[str] = []
        index = 0
        while index < len(self.parts):
            matched_source_name = _matching_source_name_at(
                self.parts,
                index,
                source_token_groups,
            )
            if matched_source_name is not None:
                source_names.append(matched_source_name)
                index += len(matched_source_name.split("_"))
                continue
            if (
                index + 1 < len(self.parts)
                and self.parts[index] in self.dialect.source_qualifier_prefix_tokens
                and self.parts[index + 1] in self.dialect.source_qualifier_suffix_tokens
            ):
                source_names.append(f"{self.parts[index]}_{self.parts[index + 1]}")
                index += 2
                continue
            stripped.append(self.parts[index])
            index += 1
        return tuple(stripped), tuple(source_names)

    def semantic_core_parts(self) -> tuple[str, ...]:
        aliased = self.dialect.resolved_feature_part_aliases().get(self.parts)
        if aliased is not None:
            return aliased
        numbered_alias = _numbered_feature_parts_alias(self.parts, self.dialect)
        if numbered_alias is not None:
            return numbered_alias
        for prefix in self.dialect.resolved_scale_qualified_feature_prefixes():
            if (
                len(self.parts) == len(prefix) + 1
                and self.parts[: len(prefix)] == prefix
                and self.parts[-1].isdigit()
            ):
                return prefix
        if (
            len(self.parts) > 2
            and self.parts[:2] == ("threshold", "otsu")
            and all(
                part.isdigit() or part in self.dialect.threshold_qualifier_tokens
                for part in self.parts[2:]
            )
        ):
            return self.parts[:2]
        if len(self.parts) == 3 and self.parts[:2] == ("center", "mass"):
            return ("center", "mass", "intensity", self.parts[2])
        if (
            len(self.parts) == 2
            and self.parts[0] == "center"
            and self.parts[1] in {"x", "y", "z"}
        ):
            return ("center", self.parts[1])
        return self.parts

    def source_feature_name_and_source(self) -> tuple[str, str | None] | None:
        """Protect dialect-defined source feature phrases from source-name extraction."""
        for prefix in self.dialect.resolved_source_feature_prefixes():
            if self.parts[: len(prefix)] != prefix:
                continue
            source_parts = self.parts[len(prefix) :]
            source_name = "_".join(source_parts) if source_parts else None
            return "_".join(prefix), source_name
        return None


@dataclass(frozen=True, slots=True)
class RuntimeAggregateFeatureIdentity:
    """Parsed aggregate measurement feature identity."""

    aggregate: str
    object_name_parts: tuple[str, ...]
    feature_parts: tuple[str, ...]

    @classmethod
    def from_parts(
        cls,
        parts: tuple[str, ...],
        dialect: RuntimeMeasurementDialect,
    ) -> "RuntimeAggregateFeatureIdentity | None":
        """Parse an aggregate feature using the runtime measurement dialect."""
        if len(parts) < 3 or parts[0] not in _MEASUREMENT_AGGREGATE_PREFIXES:
            return None
        object_name_parts, feature_parts = _aggregate_object_and_feature_parts(
            parts[1:],
            dialect,
        )
        if not object_name_parts or not feature_parts:
            return None
        return cls(
            aggregate=parts[0],
            object_name_parts=object_name_parts,
            feature_parts=feature_parts,
        )

    @classmethod
    def candidates_from_parts(
        cls,
        parts: tuple[str, ...],
    ) -> tuple["RuntimeAggregateFeatureIdentity", ...]:
        """Return all aggregate object/feature splits for family-owned suffixes."""
        if len(parts) < 3 or parts[0] not in _MEASUREMENT_AGGREGATE_PREFIXES:
            return ()
        return tuple(
            cls(
                aggregate=parts[0],
                object_name_parts=parts[1:feature_start_index],
                feature_parts=parts[feature_start_index:],
            )
            for feature_start_index in range(2, len(parts))
        )

    @property
    def object_name(self) -> str:
        return "_".join(self.object_name_parts)

    @property
    def feature_name(self) -> str:
        return "_".join(self.feature_parts)


@dataclass(frozen=True, slots=True)
class SemanticAggregatePrefixedFeatureProjection:
    """Projection of an aggregate-prefixed feature into semantic core form."""

    parts: tuple[str, ...]
    dialect: RuntimeMeasurementDialect
    known_source_names: tuple[str, ...]

    def project(self) -> tuple[str, str | None] | None:
        aggregate_identity = RuntimeAggregateFeatureIdentity.from_parts(
            self.parts,
            self.dialect,
        )
        if aggregate_identity is None:
            return None
        projected_child = SemanticCoreFeatureAndSourceNameProjection(
            aggregate_identity.feature_name,
            self.dialect,
            self.known_source_names,
        ).project()
        return SemanticAggregateFeatureProjection(
            aggregate_identity,
            projected_child,
        ).feature_name_and_source()


@dataclass(frozen=True, slots=True)
class SemanticAggregateFeatureProjection:
    """Resolved aggregate identity and projected child feature."""

    aggregate_identity: RuntimeAggregateFeatureIdentity
    projected_child: tuple[str, str | None]

    def feature_name_and_source(self) -> tuple[str, str | None] | None:
        feature_name, source_name = self.projected_child
        feature_name_parts = tuple(part for part in feature_name.split("_") if part)
        if not feature_name_parts:
            return None
        return (
            "_".join(
                (
                    self.aggregate_identity.aggregate,
                    *self.aggregate_identity.object_name_parts,
                    *feature_name_parts,
                )
            ),
            source_name,
        )


@dataclass(frozen=True, slots=True)
class SemanticCoreFeatureAndSourceNameProjection:
    """Project a runtime feature name to its semantic core and source qualifier."""

    feature_name: str
    dialect: RuntimeMeasurementDialect
    known_source_names: tuple[str, ...] = ()

    def project(self) -> tuple[str, str | None]:
        return semantic_core_feature_and_source_name_projection(
            self.feature_name,
            runtime_measurement_dialect_cache_id(self.dialect),
            self.known_source_names,
        )

    def project_uncached(self) -> tuple[str, str | None]:
        parts_projection = RuntimeMeasurementNamePartsProjection.from_feature_name(
            self.feature_name,
            self.dialect,
            known_source_names=self.known_source_names,
        )
        aggregate_feature = SemanticAggregatePrefixedFeatureProjection(
            parts_projection.parts,
            self.dialect,
            self.known_source_names,
        ).project()
        if aggregate_feature is not None:
            return aggregate_feature
        parts_projection = parts_projection.strip_category_prefix_for_core()

        direct_alias = self.dialect.resolved_feature_part_aliases().get(
            parts_projection.parts
        )
        if direct_alias is not None:
            return "_".join(direct_alias), None

        source_feature = parts_projection.source_feature_name_and_source()
        if source_feature is not None:
            return source_feature

        parts, source_names = parts_projection.source_qualifier_tokens()
        core_parts = RuntimeMeasurementNamePartsProjection(
            parts,
            self.dialect,
            self.known_source_names,
        ).semantic_core_parts()
        source_name = "__".join(source_names) if source_names else None
        return "_".join(core_parts), source_name


@lru_cache(maxsize=65536)
def semantic_core_feature_and_source_name_projection(
    feature_name: str,
    dialect_id: int,
    known_source_names: tuple[str, ...],
) -> tuple[str, str | None]:
    """Return cached semantic core/source projection for one dialect feature."""
    return SemanticCoreFeatureAndSourceNameProjection(
        feature_name,
        runtime_measurement_dialect_for_cache_id(dialect_id),
        known_source_names,
    ).project_uncached()


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementFeatureNameProjection:
    """Canonical feature/source projection under runtime equivalence policy."""

    policy: RuntimeEquivalencePolicy
    source_name: str | None
    semantic_projection: SemanticCoreFeatureAndSourceNameProjection

    @classmethod
    def from_feature_name(
        cls,
        feature_name: str,
        policy: RuntimeEquivalencePolicy,
        source_name: str | None,
        known_source_names: tuple[str, ...],
    ) -> "RuntimeMeasurementFeatureNameProjection":
        return cls(
            policy,
            source_name,
            SemanticCoreFeatureAndSourceNameProjection(
                normalize_runtime_identifier(feature_name),
                policy.measurement_dialect,
                known_source_names,
            ),
        )

    def project(self) -> tuple[str, str | None]:
        normalized = self.semantic_projection.feature_name
        normalized_source_name = normalize_runtime_source_name(self.source_name)
        if (
            self.policy.measurement_feature_name_mode
            is RuntimeMeasurementFeatureNameMode.FULL
        ):
            return normalized, normalized_source_name
        core_feature_name, field_source_name = self.semantic_projection.project()
        return _directional_pair_feature_name_and_source(
            core_feature_name,
            field_source_name or normalized_source_name,
            self.policy.measurement_dialect,
        )


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementFeatureKeySourceContext:
    """Source context for deriving a runtime measurement feature key."""

    field_name: str
    subject: RuntimeMeasurementSubjectKey
    policy: RuntimeEquivalencePolicy
    qualifiers: tuple[str, ...]
    source_name: str | None
    known_source_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _MeasurementScopeAggregatePolicy:
    """Aggregate-feature parsing policy for one closed measurement scope."""

    scope: MeasurementScope
    accepts_aggregate_feature_key: bool


def _measurement_scope_aggregate_policy_by_scope(
    rows: tuple[_MeasurementScopeAggregatePolicy, ...],
) -> Mapping[MeasurementScope, _MeasurementScopeAggregatePolicy]:
    by_scope = {row.scope: row for row in rows}
    if set(by_scope) != set(MeasurementScope):
        missing = tuple(
            sorted(scope.value for scope in set(MeasurementScope) - set(by_scope))
        )
        extra = tuple(
            sorted(scope.value for scope in set(by_scope) - set(MeasurementScope))
        )
        raise ValueError(
            "Measurement scope aggregate policy must cover MeasurementScope "
            f"exactly: missing={missing!r}, extra={extra!r}."
        )
    return MappingProxyType(by_scope)


_MEASUREMENT_SCOPE_AGGREGATE_POLICY_BY_SCOPE = (
    _measurement_scope_aggregate_policy_by_scope(
        (
            _MeasurementScopeAggregatePolicy(MeasurementScope.ARTIFACT, False),
            _MeasurementScopeAggregatePolicy(MeasurementScope.IMAGE, True),
            _MeasurementScopeAggregatePolicy(MeasurementScope.OBJECT, False),
            _MeasurementScopeAggregatePolicy(MeasurementScope.RELATIONSHIP, False),
            _MeasurementScopeAggregatePolicy(MeasurementScope.EXPERIMENT, True),
        )
    )
)


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementFeatureKeyProjection:
    """Project a runtime measurement field into its semantic feature key."""

    context: RuntimeMeasurementFeatureKeySourceContext
    strip_subject_suffix: bool = True

    def key(self) -> RuntimeMeasurementFeatureKey | None:
        aggregate_key = self.aggregate_key()
        if aggregate_key is not None:
            return aggregate_key
        feature_name, feature_source_name = (
            RuntimeMeasurementFeatureNameProjection.from_feature_name(
                self.context.field_name,
                self.context.policy,
                self.context.source_name,
                self.context.known_source_names,
            ).project()
        )
        if self.context.qualifiers:
            qualified_feature_name, feature_source_name = (
                RuntimeMeasurementFeatureNameProjection.from_feature_name(
                    "_".join((feature_name, *self.context.qualifiers)),
                    self.context.policy,
                    feature_source_name,
                    self.context.known_source_names,
                ).project()
            )
            feature_name = qualified_feature_name
        if self.strip_subject_suffix:
            feature_name = _strip_subject_suffix_feature_name(
                feature_name,
                self.context.subject,
            )
        if not feature_name:
            return None
        return RuntimeMeasurementFeatureKey.from_source_qualified_feature(
            self.context.subject,
            feature_name,
            feature_source_name,
            self.context.policy.measurement_dialect,
            qualifiers=self.context.qualifiers,
        )

    def aggregate_key(self) -> RuntimeMeasurementFeatureKey | None:
        subject = self.context.subject
        scope_policy = _MEASUREMENT_SCOPE_AGGREGATE_POLICY_BY_SCOPE[subject.scope]
        if not scope_policy.accepts_aggregate_feature_key:
            return None
        parts = tuple(
            part
            for part in normalize_runtime_identifier(self.context.field_name).split("_")
            if part
        )
        if len(parts) >= 2 and parts[0] == MeasurementStatistic.COUNT.value:
            return RuntimeMeasurementFeatureKey.from_subject_feature(
                RuntimeMeasurementSubjectKey(
                    MeasurementScope.OBJECT,
                    "_".join(parts[1:]),
                ),
                ObjectCoreMeasurementFeature.OBJECT_COUNT.value,
                MeasurementStatistic.COUNT.value,
            )
        aggregate_identity = RuntimeAggregateFeatureIdentity.from_parts(
            parts,
            self.context.policy.measurement_dialect,
        )
        if aggregate_identity is None:
            return None
        feature_name, source_name = (
            RuntimeMeasurementFeatureNameProjection.from_feature_name(
                aggregate_identity.feature_name,
                self.context.policy,
                None,
                self.context.known_source_names,
            ).project()
        )
        return RuntimeMeasurementFeatureKey.from_source_qualified_feature(
            RuntimeMeasurementSubjectKey(
                MeasurementScope.OBJECT,
                aggregate_identity.object_name,
            ),
            feature_name,
            source_name,
            self.context.policy.measurement_dialect,
            aggregate_identity.aggregate,
        )


def canonical_measurement_feature_name(
    feature_name: str,
    policy: RuntimeEquivalencePolicy,
) -> str:
    return RuntimeMeasurementFeatureNameProjection.from_feature_name(
        feature_name,
        policy,
        None,
        (),
    ).project()[0]


def _measurement_qualifier_parts_only(parts: tuple[str, ...]) -> bool:
    return bool(parts) and all(part.isdigit() for part in parts)


@lru_cache(maxsize=65536)
def normalized_measurement_feature_name_parts(feature_name: str) -> tuple[str, ...]:
    """Return normalized runtime feature-name parts for semantic projection."""
    return tuple(
        part
        for part in normalize_runtime_identifier(feature_name).split("_")
        if part
    )


def _numbered_feature_parts_alias(
    parts: tuple[str, ...],
    dialect: RuntimeMeasurementDialect,
) -> tuple[str, ...] | None:
    if len(parts) != 2 or not parts[1].isdigit():
        return None
    prefix_alias = dialect.resolved_numbered_feature_prefix_aliases().get(parts[0])
    if prefix_alias is None:
        return None
    return (*prefix_alias, str(int(parts[1])))


def _directional_pair_feature_name_and_source(
    feature_name: str,
    source_name: str | None,
    dialect: RuntimeMeasurementDialect,
) -> tuple[str, str | None]:
    alias = dialect.resolved_directional_pair_feature_aliases().get(feature_name)
    if (
        source_name is not None
        and feature_name in dialect.resolved_undirected_pair_feature_names()
    ):
        return feature_name, _canonical_pair_source_name(source_name)
    if alias is None or source_name is None:
        return feature_name, source_name

    source_pair = RuntimeMeasurementSourcePair.from_source_name(source_name)
    if source_pair is None:
        return feature_name, source_name

    canonical_feature_name, direction_index = alias
    directed_source_name = (
        source_pair.reversed_source_name if direction_index == 2 else source_pair.source_name
    )
    return canonical_feature_name, directed_source_name


def _canonical_pair_source_name(source_name: str) -> str:
    source_pair = RuntimeMeasurementSourcePair.from_source_name(source_name)
    if source_pair is None:
        return source_name
    return RuntimeMeasurementSourcePair(
        *sorted((source_pair.first, source_pair.second))
    ).source_name


@lru_cache(maxsize=1024)
def _source_name_token_groups(
    known_source_names: tuple[str, ...],
) -> _RuntimeSourceTokenGroups:
    groups = tuple(
        (normalized, runtime_source_name_tokens(normalized))
        for normalized in (
            normalize_runtime_source_name(source_name)
            for source_name in known_source_names
        )
        if normalized
    )
    return tuple(sorted(groups, key=lambda group: (-len(group[1]), group[0])))


def _matching_source_name_at(
    parts: tuple[str, ...],
    index: int,
    source_token_groups: _RuntimeSourceTokenGroups,
) -> str | None:
    for source_name, source_parts in source_token_groups:
        if parts[index : index + len(source_parts)] == source_parts:
            return source_name
    return None


def _strip_subject_suffix_feature_name(
    feature_name: str,
    subject: RuntimeMeasurementSubjectKey,
) -> str:
    """Remove redundant object-table suffixes from object-scoped features."""
    if subject.scope is not MeasurementScope.OBJECT or subject.name is None:
        return feature_name
    subject_parts = tuple(part for part in subject.name.split("_") if part)
    feature_parts = tuple(part for part in feature_name.split("_") if part)
    if (
        subject_parts
        and len(feature_parts) > len(subject_parts)
        and feature_parts[-len(subject_parts) :] == subject_parts
    ):
        return "_".join(feature_parts[: -len(subject_parts)])
    return feature_name


def _aggregate_object_and_feature_parts(
    parts: tuple[str, ...],
    dialect: RuntimeMeasurementDialect,
) -> _RuntimeMeasurementNameParts:
    for index in range(1, len(parts)):
        if _starts_aggregate_feature_parts(parts[index:], dialect):
            return parts[:index], parts[index:]
    return parts[:1], parts[1:]


def _starts_aggregate_feature_parts(
    parts: tuple[str, ...],
    dialect: RuntimeMeasurementDialect,
) -> bool:
    return (
        _starts_with_measurement_category(parts, dialect)
        or parts in dialect.resolved_feature_part_aliases()
    )


def _starts_with_measurement_category(
    parts: tuple[str, ...],
    dialect: RuntimeMeasurementDialect,
) -> bool:
    return bool(
        _category_prefix_for_parts(
            parts,
            runtime_measurement_dialect_cache_id(dialect),
        )
    )


@lru_cache(maxsize=16384)
def _category_prefix_for_parts(
    parts: tuple[str, ...],
    dialect_id: int,
) -> tuple[str, ...]:
    """Return the longest dialect-declared category prefix for ``parts``."""
    matches = tuple(
        prefix
        for prefix in runtime_measurement_dialect_for_cache_id(
            dialect_id
        ).resolved_category_prefixes()
        if len(parts) >= len(prefix) and parts[: len(prefix)] == prefix
    )
    if not matches:
        return ()
    return max(matches, key=len)


@lru_cache(maxsize=16384)
def _core_strip_category_prefix_for_parts(
    parts: tuple[str, ...],
    dialect_id: int,
) -> tuple[str, ...]:
    """Return the longest category prefix stripped from core feature identity."""
    dialect = runtime_measurement_dialect_for_cache_id(dialect_id)
    calculated_prefixes = frozenset(dialect.resolved_calculated_feature_prefixes())
    matching_prefixes = tuple(
        prefix
        for prefix in dialect.resolved_category_prefixes()
        if prefix not in calculated_prefixes
        and _should_strip_category_prefix(parts, prefix, dialect_id)
    )
    if not matching_prefixes:
        return ()
    return max(matching_prefixes, key=len)


def _should_strip_category_prefix(
    parts: tuple[str, ...],
    prefix: tuple[str, ...],
    dialect_id: int,
) -> bool:
    if parts[: len(prefix)] != prefix or len(parts) <= len(prefix):
        return False
    suffix = parts[len(prefix) :]
    pair_correlation_feature_name = (
        runtime_measurement_dialect_for_cache_id(
            dialect_id
        ).resolved_pair_correlation_feature_name()
    )
    if pair_correlation_feature_name is not None and prefix == (
        pair_correlation_feature_name,
    ):
        return not _measurement_qualifier_parts_only(suffix)
    return True
