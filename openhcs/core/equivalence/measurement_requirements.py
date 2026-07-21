"""Projection of requested measurement keys into runtime input requirements."""

from __future__ import annotations

from dataclasses import dataclass

from openhcs.core.equivalence.keys import (
    RuntimeMeasurementFeatureKey,
    RuntimeMeasurementSourcePair,
    RuntimeMeasurementSubjectKey,
)
from openhcs.core.equivalence.measurement_facts import (
    RuntimeDirectionalPairMeasurementDerivationContract,
    RuntimeRequiredMeasurementKeys,
)
from openhcs.core.equivalence.measurement_features import (
    object_measurement_feature_matches_marker,
)
from openhcs.core.equivalence.policy import RuntimeEquivalencePolicy
from openhcs.core.runtime_measurements import (
    MeasurementScope,
    MeasurementStatistic,
    ObjectCoreMeasurementFeature,
    ObjectIdentifierFeatureMarker,
    ObjectLocationFeatureMarker,
)


@dataclass(frozen=True, slots=True)
class RequiredRuntimeMeasurementProjection:
    """Project user-required measurement keys into runtime input domains."""

    required_keys: RuntimeRequiredMeasurementKeys
    policy: RuntimeEquivalencePolicy
    known_source_names: tuple[str, ...] = ()

    def input_keys(self) -> RuntimeRequiredMeasurementKeys:
        if self.required_keys is None:
            return None
        keys: set[RuntimeMeasurementFeatureKey] = set(self.required_keys)
        pair_derivation = RuntimeDirectionalPairMeasurementDerivationContract(
            self.policy,
            self.known_source_names,
        )
        for key in self.required_keys:
            if key.statistic == MeasurementStatistic.MEAN.value:
                keys.add(
                    RuntimeMeasurementFeatureKey(
                        key.subject,
                        key.feature_name,
                        MeasurementStatistic.VALUE.value,
                        key.source_name,
                    )
                )
            keys.update(pair_derivation.required_input_keys(key))
        return frozenset(keys)

    def subjects(self) -> frozenset[RuntimeMeasurementSubjectKey] | None:
        input_keys = self.input_keys()
        if input_keys is None:
            return None
        subjects: set[RuntimeMeasurementSubjectKey] = set()
        for key in input_keys:
            subjects.add(key.subject)
            if (
                key.subject.scope is MeasurementScope.IMAGE
                and key.subject.name is not None
            ):
                source_pair = RuntimeMeasurementSourcePair.from_source_name(
                    key.subject.name
                )
                if source_pair is not None:
                    subjects.add(
                        RuntimeMeasurementSubjectKey(
                            MeasurementScope.IMAGE,
                            source_pair.reversed_source_name,
                        )
                    )
        return frozenset(subjects)

    def object_identifier_keys(
        self,
        subject: RuntimeMeasurementSubjectKey,
    ) -> tuple[RuntimeMeasurementFeatureKey, ...]:
        if self.required_keys is None:
            return (
                RuntimeMeasurementFeatureKey(
                    subject,
                    ObjectCoreMeasurementFeature.OBJECT_NUMBER.value,
                ),
            )
        return tuple(
            key
            for key in sorted(self.required_keys, key=lambda item: item.sort_key)
            if key.subject == subject
            and object_measurement_feature_matches_marker(
                key,
                ObjectIdentifierFeatureMarker,
                self.policy,
            )
        )

    def object_location_feature_names(
        self,
        subject: RuntimeMeasurementSubjectKey,
        *,
        statistic: MeasurementStatistic,
    ) -> frozenset[str] | None:
        if self.required_keys is None:
            return None
        return frozenset(
            key.feature_name
            for key in self.required_keys
            if key.subject == subject
            and key.statistic == statistic.value
            and key.source_name is None
            and object_measurement_feature_matches_marker(
                RuntimeMeasurementFeatureKey(
                    key.subject,
                    key.feature_name,
                    MeasurementStatistic.VALUE.value,
                    source_name=key.source_name,
                ),
                ObjectLocationFeatureMarker,
                self.policy,
            )
        )
