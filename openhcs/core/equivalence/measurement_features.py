"""Nominal feature-role semantics for runtime measurement equivalence."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta

from openhcs.core.equivalence.keys import (
    RuntimeMeasurementFeatureKey,
    RuntimeMeasurementSubjectKey,
)
from openhcs.core.equivalence.policy import normalize_runtime_identifier
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.core.runtime_semantics import (
    IndexedObjectZernikeDescriptor,
    MeasurementStatistic,
    MeasurementScope,
    ObjectCoreMeasurementFeature,
    ObjectMeasurementFeatureRole,
    ObjectShapeMeasurementFeature,
)


MeasurementRowIdentity = tuple[tuple[str, object], ...]
MeasurementFeatureRowIdentity = tuple[
    RuntimeMeasurementFeatureKey,
    MeasurementRowIdentity,
]
MeasurementSubjectRowIdentity = tuple[
    RuntimeMeasurementSubjectKey,
    MeasurementRowIdentity,
]


class ObjectMeasurementFeatureRoleStrategy(
    EnumKeyedStrategyMixin[ObjectMeasurementFeatureRole],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Classify object measurement feature keys by semantic role."""

    __registry_key__ = "strategy_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "role"

    role: ClassVar[ObjectMeasurementFeatureRole]
    strategy_label: ClassVar[str | None] = None

    @abstractmethod
    def matches(self, key: RuntimeMeasurementFeatureKey) -> bool:
        """Return whether ``key`` carries this object measurement role."""

    def subjects(
        self,
        values_by_feature: Mapping[RuntimeMeasurementFeatureKey, object],
    ) -> frozenset[RuntimeMeasurementSubjectKey]:
        """Return subjects carrying this role in a feature mapping."""

        return frozenset(
            key.subject
            for key in values_by_feature
            if self.matches(key)
        )

    def subject_row_identities(
        self,
        row_identities_by_feature: Mapping[MeasurementFeatureRowIdentity, object],
    ) -> frozenset[MeasurementSubjectRowIdentity]:
        """Return subject-row identities carrying this role."""

        return frozenset(
            (key.subject, row_identity)
            for key, row_identity in row_identities_by_feature
            if self.matches(key)
        )


class ObjectCountFeatureRoleStrategy(ObjectMeasurementFeatureRoleStrategy):
    """Object-count aggregate features."""

    role = ObjectMeasurementFeatureRole.COUNT

    def matches(self, key: RuntimeMeasurementFeatureKey) -> bool:
        return (
            key.subject.scope is MeasurementScope.OBJECT
            and key.statistic == MeasurementStatistic.COUNT.value
            and key.source_name is None
            and key.feature_name == ObjectCoreMeasurementFeature.OBJECT_COUNT.value
        )


class ObjectIdentifierFeatureRoleStrategy(ObjectMeasurementFeatureRoleStrategy):
    """Object identity features."""

    role = ObjectMeasurementFeatureRole.IDENTIFIER

    def matches(self, key: RuntimeMeasurementFeatureKey) -> bool:
        return (
            key.subject.scope is MeasurementScope.OBJECT
            and key.statistic == MeasurementStatistic.VALUE.value
            and key.source_name is None
            and (
                key.feature_name == ObjectCoreMeasurementFeature.OBJECT_NUMBER.value
                or key.feature_name.endswith(
                    f"_{ObjectCoreMeasurementFeature.OBJECT_NUMBER.value}"
                )
            )
        )


class ObjectLocationFeatureRoleStrategy(ObjectMeasurementFeatureRoleStrategy):
    """Object center-location features."""

    role = ObjectMeasurementFeatureRole.LOCATION
    feature_names: ClassVar[frozenset[str]] = frozenset(
        feature.value
        for feature in (
            ObjectCoreMeasurementFeature.CENTER_X,
            ObjectCoreMeasurementFeature.CENTER_Y,
            ObjectCoreMeasurementFeature.CENTER_Z,
        )
    )

    def matches(self, key: RuntimeMeasurementFeatureKey) -> bool:
        return (
            key.subject.scope is MeasurementScope.OBJECT
            and key.statistic == MeasurementStatistic.VALUE.value
            and key.feature_name in self.feature_names
        )


class ObjectShapeDescriptorFeatureRoleStrategy(ObjectMeasurementFeatureRoleStrategy):
    """Object geometry and shape-descriptor features."""

    role = ObjectMeasurementFeatureRole.SHAPE_DESCRIPTOR
    indexed_feature_prefixes: ClassVar[frozenset[str]] = frozenset(
        normalize_runtime_identifier(feature.value)
        for feature in (
            ObjectShapeMeasurementFeature.SPATIAL_MOMENT,
            ObjectShapeMeasurementFeature.CENTRAL_MOMENT,
            ObjectShapeMeasurementFeature.NORMALIZED_MOMENT,
            ObjectShapeMeasurementFeature.HU_MOMENT,
            ObjectShapeMeasurementFeature.INERTIA_TENSOR,
            ObjectShapeMeasurementFeature.INERTIA_TENSOR_EIGENVALUES,
            ObjectShapeMeasurementFeature.ZERNIKE,
        )
    )
    feature_names: ClassVar[frozenset[str]] = frozenset(
        normalize_runtime_identifier(feature.value)
        for feature in ObjectShapeMeasurementFeature
    )

    def matches(self, key: RuntimeMeasurementFeatureKey) -> bool:
        return (
            key.subject.scope is MeasurementScope.OBJECT
            and key.statistic == MeasurementStatistic.VALUE.value
            and key.source_name is None
            and (
                key.feature_name in self.feature_names
                or any(
                    key.feature_name.startswith(f"{prefix}_")
                    for prefix in self.indexed_feature_prefixes
                )
            )
        )


class ObjectZernikeDescriptorFeatureRoleStrategy(ObjectMeasurementFeatureRoleStrategy):
    """Boundary-sensitive object Zernike descriptor features."""

    role = ObjectMeasurementFeatureRole.ZERNIKE_DESCRIPTOR

    def matches(self, key: RuntimeMeasurementFeatureKey) -> bool:
        return (
            key.subject.scope is MeasurementScope.OBJECT
            and key.statistic == MeasurementStatistic.VALUE.value
            and IndexedObjectZernikeDescriptor.from_feature_name(key.feature_name)
            is not None
        )


def object_measurement_feature_has_role(
    key: RuntimeMeasurementFeatureKey,
    role: ObjectMeasurementFeatureRole,
) -> bool:
    """Return whether a runtime measurement key has ``role``."""

    return ObjectMeasurementFeatureRoleStrategy.for_enum_member(role).matches(key)


def object_measurement_subjects_with_role(
    values_by_feature: Mapping[RuntimeMeasurementFeatureKey, object],
    role: ObjectMeasurementFeatureRole,
) -> frozenset[RuntimeMeasurementSubjectKey]:
    """Return measurement subjects that have at least one feature with ``role``."""

    return ObjectMeasurementFeatureRoleStrategy.for_enum_member(role).subjects(
        values_by_feature
    )


def object_measurement_subject_row_identities_with_role(
    row_identities_by_feature: Mapping[MeasurementFeatureRowIdentity, object],
    role: ObjectMeasurementFeatureRole,
) -> frozenset[MeasurementSubjectRowIdentity]:
    """Return subject-row identities whose feature key has ``role``."""

    return ObjectMeasurementFeatureRoleStrategy.for_enum_member(
        role
    ).subject_row_identities(row_identities_by_feature)
