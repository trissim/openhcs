"""CellProfiler-owned runtime measurement semantic profiles."""

from __future__ import annotations

from openhcs.core.equivalence.measurement_features import (
    RuntimeMeasurementDescriptorSemantics,
    RuntimeMeasurementFeatureSemanticContext,
    RuntimeMeasurementIndexedDescriptorEquivalence,
)
from openhcs.core.equivalence.policy import RuntimeEquivalencePolicy
from openhcs.core.equivalence.keys import RuntimeMeasurementFeatureKey
from openhcs.core.runtime_semantics import (
    RuntimeMeasurementIndexedDescriptorDeclaration,
)


class CellProfilerDescriptorSemanticProfile(RuntimeMeasurementDescriptorSemantics):
    """Registered profile backed by feature-member descriptor declarations."""

    strategy_key = "cellprofiler_descriptors"

    @staticmethod
    def _single_descriptor_declaration(
        key: RuntimeMeasurementFeatureKey,
    ) -> tuple[type[RuntimeMeasurementIndexedDescriptorDeclaration], object]:
        matches = RuntimeMeasurementIndexedDescriptorDeclaration.matching_declarations(
            key.feature_name
        )
        if len(matches) != 1:
            raise ValueError(
                "CellProfilerDescriptorSemanticProfile expected exactly one "
                f"descriptor declaration for {key!r}, got {matches!r}."
            )
        return matches[0]

    @staticmethod
    def _equivalence_declaration(
        key: RuntimeMeasurementFeatureKey,
    ) -> tuple[type[RuntimeMeasurementIndexedDescriptorEquivalence], object]:
        declaration_type, descriptor = (
            CellProfilerDescriptorSemanticProfile._single_descriptor_declaration(key)
        )
        if not issubclass(
            declaration_type,
            RuntimeMeasurementIndexedDescriptorEquivalence,
        ):
            raise TypeError(
                f"{declaration_type.__name__} must inherit "
                "RuntimeMeasurementIndexedDescriptorEquivalence."
            )
        return declaration_type, descriptor

    def matches(self, context: RuntimeMeasurementFeatureSemanticContext) -> bool:
        return bool(
            RuntimeMeasurementIndexedDescriptorDeclaration.matching_declarations(
                context.key.feature_name
            )
        )

    def descriptor_identity(
        self,
        key: RuntimeMeasurementFeatureKey,
        dialect: object,
    ) -> object:
        del dialect
        _declaration_type, descriptor = self._single_descriptor_declaration(key)
        return descriptor

    def values_equivalent(
        self,
        key: RuntimeMeasurementFeatureKey,
        left: object,
        right: object,
        policy: RuntimeEquivalencePolicy,
    ) -> bool:
        declaration_type, descriptor = self._equivalence_declaration(key)
        return declaration_type.descriptor_values_equivalent(
            descriptor,
            key,
            left,
            right,
            policy,
        )

    def descriptor_snapshots_comparable(
        self,
        key: RuntimeMeasurementFeatureKey,
        reference: object,
        candidate: object,
        policy: RuntimeEquivalencePolicy,
    ) -> bool:
        """Return whether the owning descriptor declaration can compare snapshots."""
        declaration_type, descriptor = self._equivalence_declaration(key)
        return declaration_type.descriptor_snapshots_comparable(
            descriptor,
            key,
            reference,
            candidate,
            policy,
        )
