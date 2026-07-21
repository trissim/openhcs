"""Nominal feature-role semantics for runtime measurement equivalence."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, ClassVar

from metaclass_registry import RegistryFamily, RegistryKeyAttribute

from openhcs.core.equivalence.keys import (
    RuntimeMeasurementFeatureKey,
    RuntimeMeasurementSubjectKey,
)
from openhcs.core.equivalence.policy import (
    RuntimeEquivalencePolicy,
    RuntimeMeasurementDialect,
    normalize_runtime_identifier,
    runtime_measurement_dialect_cache_id,
    runtime_measurement_dialect_for_cache_id,
)
from openhcs.core.registry_strategies import MostDerivedContextStrategyMixin
from openhcs.core.runtime_measurements import (
    MeasurementScope,
    MeasurementStatistic,
    ObjectCoreMeasurementFeature,
    ObjectCalculatedFeatureMarker,
    ObjectCountFeatureMarker,
    ObjectIdentifierFeatureMarker,
    ObjectLocationCoordinateProjectionStrategy,
    ObjectLocationFeatureMarker,
    RuntimeMeasurementFeatureRelation,
    RuntimeMeasurementFeatureSemanticMarker,
    RuntimeMeasurementFeature,
)

if TYPE_CHECKING:
    from openhcs.core.equivalence.measurement_rows import RuntimeMeasurementRowIdentity


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementFeatureSemanticContext:
    """Selection context for runtime measurement feature semantic profiles."""

    key: RuntimeMeasurementFeatureKey
    policy: RuntimeEquivalencePolicy


class RuntimeMeasurementFeatureSemanticProfile(
    MostDerivedContextStrategyMixin[RuntimeMeasurementFeatureSemanticContext],
    ABC,
):
    """Registered semantic behavior for runtime measurement features."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.STRATEGY_KEY)
    strategy_key: ClassVar[str | None] = None

    @classmethod
    def for_feature_key(
        cls,
        key: RuntimeMeasurementFeatureKey,
        policy: RuntimeEquivalencePolicy,
    ) -> "RuntimeMeasurementFeatureSemanticProfile":
        """Return the most-derived semantic profile for ``key``."""
        return cls._for_feature_key_payload(
            key.to_cache_payload(),
            runtime_measurement_dialect_cache_id(policy.measurement_dialect),
        )

    @classmethod
    @lru_cache(maxsize=32768)
    def _for_feature_key_payload(
        cls,
        key_payload: object,
        dialect_id: int,
    ) -> "RuntimeMeasurementFeatureSemanticProfile":
        """Return cached most-derived semantic profile for one key/dialect pair."""
        key = RuntimeMeasurementFeatureKey.from_cache_payload(key_payload)
        context = RuntimeMeasurementFeatureSemanticContext(
            key,
            RuntimeEquivalencePolicy(
                measurement_dialect=runtime_measurement_dialect_for_cache_id(
                    dialect_id
                )
            ),
        )
        strategy = cls.for_context(
            context,
            required=False,
            error_subject=(
                "Runtime measurement feature semantic profile for "
                f"{key!r}"
            ),
        )
        if strategy is None:
            return DefaultRuntimeMeasurementFeatureSemanticProfile()
        return strategy

    @abstractmethod
    def values_equivalent(
        self,
        key: RuntimeMeasurementFeatureKey,
        left: object,
        right: object,
        policy: RuntimeEquivalencePolicy,
    ) -> bool:
        """Return semantic value equivalence for this feature."""

    def row_identity_stable(
        self,
        key: RuntimeMeasurementFeatureKey,
        row_identity: RuntimeMeasurementRowIdentity,
        policy: RuntimeEquivalencePolicy,
    ) -> bool:
        """Return whether this feature's row identity is stable under policy."""
        del key, row_identity, policy
        return True

    def current_object_vector(
        self,
        key: RuntimeMeasurementFeatureKey,
        label_array: object,
    ) -> object | None:
        """Return a current-object vector for ``key`` when this profile owns one."""
        del key, label_array
        return None

    def matches_marker(
        self,
        key: RuntimeMeasurementFeatureKey,
        marker_type: type[RuntimeMeasurementFeatureSemanticMarker],
        policy: RuntimeEquivalencePolicy,
    ) -> bool:
        """Return whether ``key`` carries ``marker_type`` semantics."""
        del key, marker_type, policy
        return False

    def requires_sparse_boundary_object_count_stability(
        self,
        key: RuntimeMeasurementFeatureKey,
        policy: RuntimeEquivalencePolicy,
    ) -> bool:
        """Return whether sparse-boundary comparison is gated by object count."""
        del key, policy
        return True


class MarkerRuntimeMeasurementFeatureSemanticProfile(
    RuntimeMeasurementFeatureSemanticProfile,
    ABC,
):
    """Registered semantic profile for one nominal feature marker."""

    marker_type: ClassVar[type[RuntimeMeasurementFeatureSemanticMarker]]
    subject_scope: ClassVar[MeasurementScope] = MeasurementScope.OBJECT
    statistic: ClassVar[MeasurementStatistic] = MeasurementStatistic.VALUE
    source_name_allowed: ClassVar[bool] = False

    @classmethod
    def declared_marker_types(
        cls,
    ) -> tuple[type[RuntimeMeasurementFeatureSemanticMarker], ...]:
        """Return marker declarations inherited by this semantic profile."""
        return tuple(
            dict.fromkeys(
                marker_type
                for profile_type in cls.__mro__
                if issubclass(profile_type, MarkerRuntimeMeasurementFeatureSemanticProfile)
                and "marker_type" in profile_type.__dict__
                for marker_type in (profile_type.__dict__["marker_type"],)
            )
        )

    def values_equivalent(
        self,
        key: RuntimeMeasurementFeatureKey,
        left: object,
        right: object,
        policy: RuntimeEquivalencePolicy,
    ) -> bool:
        """Return default value equivalence for marker-owned features."""
        del key, policy
        return left == right

    def matches_marker(
        self,
        key: RuntimeMeasurementFeatureKey,
        marker_type: type[RuntimeMeasurementFeatureSemanticMarker],
        policy: RuntimeEquivalencePolicy,
    ) -> bool:
        """Return whether this profile's marker is compatible with ``marker_type``."""
        del key, policy
        if not issubclass(marker_type, RuntimeMeasurementFeatureSemanticMarker):
            raise TypeError(
                "marker_type must inherit RuntimeMeasurementFeatureSemanticMarker."
            )
        return any(
            issubclass(declared_marker_type, marker_type)
            for declared_marker_type in type(self).declared_marker_types()
        )

    def requires_sparse_boundary_object_count_stability(
        self,
        key: RuntimeMeasurementFeatureKey,
        policy: RuntimeEquivalencePolicy,
    ) -> bool:
        """Delegate sparse-boundary gating to the owned marker declaration."""
        del key, policy
        marker_types = type(self).declared_marker_types()
        if not marker_types:
            raise TypeError(
                f"{type(self).__name__} must inherit at least one marker profile."
            )
        return all(
            marker_type.requires_sparse_boundary_object_count_stability()
            for marker_type in marker_types
        )

    def matches(self, context: RuntimeMeasurementFeatureSemanticContext) -> bool:
        """Return whether ``context.key`` carries this marker profile."""
        key = context.key
        if key.subject.scope is not type(self).subject_scope:
            return False
        if key.statistic != type(self).statistic.value:
            return False
        if not type(self).source_name_allowed and key.source_name is not None:
            return False
        return self.matches_feature(context)

    @abstractmethod
    def matches_feature(self, context: RuntimeMeasurementFeatureSemanticContext) -> bool:
        """Return whether the already-shaped key is this marker's feature."""


class RuntimeMeasurementDescriptorSemantics(RuntimeMeasurementFeatureSemanticProfile):
    """Registered equivalence profile for indexed descriptor-like features."""

    @abstractmethod
    def descriptor_identity(
        self,
        key: RuntimeMeasurementFeatureKey,
        dialect: RuntimeMeasurementDialect,
    ) -> object:
        """Return an opaque descriptor identity owned by this profile."""

    def descriptor_snapshots_comparable(
        self,
        key: RuntimeMeasurementFeatureKey,
        reference: object,
        candidate: object,
        policy: RuntimeEquivalencePolicy,
    ) -> bool:
        """Return whether two measurement snapshots can compare this descriptor."""
        del key, reference, candidate, policy
        return True


class RuntimeMeasurementIndexedDescriptorEquivalence(ABC):
    """Equivalence behavior for an indexed descriptor declaration."""

    @classmethod
    @abstractmethod
    def descriptor_values_equivalent(
        cls,
        descriptor: object,
        key: RuntimeMeasurementFeatureKey,
        left: object,
        right: object,
        policy: RuntimeEquivalencePolicy,
    ) -> bool:
        """Return value equivalence for this descriptor."""

    @classmethod
    def descriptor_row_identity_stable(
        cls,
        descriptor: object,
        key: RuntimeMeasurementFeatureKey,
        row_identity: RuntimeMeasurementRowIdentity,
        policy: RuntimeEquivalencePolicy,
    ) -> bool:
        """Return whether descriptor row identity is stable under policy."""
        del descriptor, key, row_identity, policy
        return True

    @classmethod
    def descriptor_snapshots_comparable(
        cls,
        descriptor: object,
        key: RuntimeMeasurementFeatureKey,
        reference: object,
        candidate: object,
        policy: RuntimeEquivalencePolicy,
    ) -> bool:
        """Return whether two measurement snapshots can compare this descriptor."""
        del descriptor, key, reference, candidate, policy
        return True


class DefaultRuntimeMeasurementFeatureSemanticProfile(
    RuntimeMeasurementFeatureSemanticProfile
):
    """Fallback semantic profile for ordinary measurement features."""

    strategy_key = "default"

    def matches(self, context: RuntimeMeasurementFeatureSemanticContext) -> bool:
        del context
        return False

    def values_equivalent(
        self,
        key: RuntimeMeasurementFeatureKey,
        left: object,
        right: object,
        policy: RuntimeEquivalencePolicy,
    ) -> bool:
        del key, policy
        return left == right


class ObjectCountFeatureSemanticProfile(MarkerRuntimeMeasurementFeatureSemanticProfile):
    """Core object-count feature semantics."""

    strategy_key = "object_count"
    marker_type = ObjectCountFeatureMarker
    statistic = MeasurementStatistic.COUNT

    def matches_feature(self, context: RuntimeMeasurementFeatureSemanticContext) -> bool:
        key = context.key
        return key.feature_name == ObjectCoreMeasurementFeature.OBJECT_COUNT.value


class ObjectIdentifierFeatureSemanticProfile(
    MarkerRuntimeMeasurementFeatureSemanticProfile
):
    """Core object-identifier feature semantics."""

    strategy_key = "object_identifier"
    marker_type = ObjectIdentifierFeatureMarker

    def matches_feature(self, context: RuntimeMeasurementFeatureSemanticContext) -> bool:
        key = context.key
        return (
            key.feature_name == ObjectCoreMeasurementFeature.OBJECT_NUMBER.value
            or key.feature_name.endswith(
                f"_{ObjectCoreMeasurementFeature.OBJECT_NUMBER.value}"
            )
        )


class ObjectLocationFeatureSemanticProfile(
    MarkerRuntimeMeasurementFeatureSemanticProfile
):
    """Core object-location feature semantics."""

    strategy_key = "object_location"
    marker_type = ObjectLocationFeatureMarker

    def matches_feature(self, context: RuntimeMeasurementFeatureSemanticContext) -> bool:
        key = context.key
        return any(
            key.feature_name == strategy_type.axis_feature.value
            for strategy_type in (
                ObjectLocationCoordinateProjectionStrategy.registered_strategy_types()
            )
        )


class ObjectCalculatedFeatureSemanticProfile(
    MarkerRuntimeMeasurementFeatureSemanticProfile
):
    """Core calculated object-feature namespace semantics."""

    strategy_key = "object_calculated"
    marker_type = ObjectCalculatedFeatureMarker

    def matches_feature(self, context: RuntimeMeasurementFeatureSemanticContext) -> bool:
        key = context.key
        feature_parts = tuple(part for part in key.feature_name.split("_") if part)
        return any(
            len(feature_parts) > len(prefix)
            and feature_parts[: len(prefix)] == prefix
            for prefix in (
                context.policy.measurement_dialect.resolved_calculated_feature_prefixes()
            )
        )


class ObjectCalculatedIdentifierFeatureSemanticProfile(
    ObjectIdentifierFeatureSemanticProfile,
    ObjectCalculatedFeatureSemanticProfile,
):
    """Identifier semantics for calculated object-identifier features."""

    strategy_key = "object_calculated_identifier"

    def matches_feature(self, context: RuntimeMeasurementFeatureSemanticContext) -> bool:
        return (
            ObjectIdentifierFeatureSemanticProfile.matches_feature(self, context)
            and ObjectCalculatedFeatureSemanticProfile.matches_feature(self, context)
        )


@dataclass(frozen=True, slots=True)
class TieSensitiveLocationValueFeatureRelation(RuntimeMeasurementFeatureRelation):
    """Location-feature relation gated by a stable value feature."""

    target_feature: RuntimeMeasurementFeature
    source_marker: type[RuntimeMeasurementFeatureSemanticMarker]
    target_marker: type[RuntimeMeasurementFeatureSemanticMarker]

    def __post_init__(self) -> None:
        if not self.target_marker.matches_feature(self.target_feature):
            raise ValueError(
                f"{self.target_feature!r} must carry {self.target_marker.__name__}."
            )

    def source_family_names(
        self,
        source_feature: RuntimeMeasurementFeature,
    ) -> tuple[str, ...]:
        """Return bare and marker-qualified source-family names."""
        if not self.source_marker.matches_feature(source_feature):
            raise ValueError(
                f"{source_feature!r} must carry {self.source_marker.__name__}."
            )
        return (
            source_feature.feature_family(),
            self.source_marker.qualified_family(source_feature),
        )

    def target_family_name(
        self,
        source_feature: RuntimeMeasurementFeature,
        source_family_name: str,
        feature_type: type[RuntimeMeasurementFeature],
    ) -> str | None:
        """Return the matching bare or marker-qualified target family."""
        del feature_type
        normalized_source_family = normalize_runtime_identifier(source_family_name)
        if normalized_source_family == source_feature.feature_family():
            return self.target_feature.feature_family()
        if normalized_source_family == self.source_marker.qualified_family(
            source_feature
        ):
            return self.target_marker.qualified_family(self.target_feature)
        return None


def object_measurement_feature_matches_marker(
    key: RuntimeMeasurementFeatureKey,
    marker_type: type[RuntimeMeasurementFeatureSemanticMarker],
    policy: RuntimeEquivalencePolicy,
) -> bool:
    """Return whether a runtime measurement key carries ``marker_type`` semantics."""
    if not issubclass(marker_type, RuntimeMeasurementFeatureSemanticMarker):
        raise TypeError(
            "marker_type must inherit RuntimeMeasurementFeatureSemanticMarker."
        )
    return _object_measurement_feature_matches_marker_cached(
        key.to_cache_payload(),
        marker_type,
        runtime_measurement_dialect_cache_id(policy.measurement_dialect),
    )


@lru_cache(maxsize=65536)
def _object_measurement_feature_matches_marker_cached(
    key_payload: object,
    marker_type: type[RuntimeMeasurementFeatureSemanticMarker],
    dialect_id: int,
) -> bool:
    key = RuntimeMeasurementFeatureKey.from_cache_payload(key_payload)
    policy = RuntimeEquivalencePolicy(
        measurement_dialect=runtime_measurement_dialect_for_cache_id(dialect_id)
    )
    provider_marker_types = policy.measurement_dialect.measurement_feature_marker_types(
        key
    )
    if any(
        issubclass(provider_marker_type, marker_type)
        for provider_marker_type in provider_marker_types
    ):
        return True
    profile = RuntimeMeasurementFeatureSemanticProfile.for_feature_key(key, policy)
    return profile.matches_marker(key, marker_type, policy)


def object_measurement_feature_requires_sparse_boundary_object_count_stability(
    key: RuntimeMeasurementFeatureKey,
    policy: RuntimeEquivalencePolicy | RuntimeMeasurementDialect,
) -> bool:
    """Return whether sparse-boundary equivalence for ``key`` is gated by object count."""
    if isinstance(policy, RuntimeMeasurementDialect):
        policy = RuntimeEquivalencePolicy(measurement_dialect=policy)
    provider_marker_types = policy.measurement_dialect.measurement_feature_marker_types(
        key
    )
    if provider_marker_types and any(
        marker_type.requires_sparse_boundary_object_count_stability()
        for marker_type in provider_marker_types
    ):
        return True
    profile = RuntimeMeasurementFeatureSemanticProfile.for_feature_key(key, policy)
    return profile.requires_sparse_boundary_object_count_stability(key, policy)


def object_measurement_subjects_matching_marker(
    measurement_feature_map: Mapping[RuntimeMeasurementFeatureKey, object],
    marker_type: type[RuntimeMeasurementFeatureSemanticMarker],
    policy: RuntimeEquivalencePolicy,
) -> frozenset[RuntimeMeasurementSubjectKey]:
    """Return measurement subjects with at least one feature carrying ``marker_type``."""
    return frozenset(
        key.subject
        for key in measurement_feature_map
        if object_measurement_feature_matches_marker(key, marker_type, policy)
    )
