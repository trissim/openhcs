"""Artifact observability policies independent of materialization format."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from openhcs.core.artifacts import (
    ArtifactSpec,
    ArtifactType,
    ArtifactTypeStrategyMatchMixin,
    ObjectLabelsArtifactType,
    MeasurementsArtifactType,
)
from openhcs.core.registry_strategies import MostDerivedContextStrategyMixin


class ArtifactObservabilityStrategy(
    ArtifactTypeStrategyMatchMixin,
    MostDerivedContextStrategyMixin[type[ArtifactType]],
    ABC,
):
    """Registered policy for artifacts that remain externally observable.

    Materialization controls files written by OpenHCS. Observability is broader:
    some runtime artifacts induce exported facts even when no artifact file is
    explicitly requested. Object-label counts are the canonical example.
    """

    artifact_type: ClassVar[type[ArtifactType] | None] = None

    @abstractmethod
    def externally_required_outputs(
        self,
        declared_outputs: tuple[ArtifactSpec, ...],
    ) -> tuple[ArtifactSpec, ...]:
        """Return declared outputs that must survive dead-artifact pruning."""


class ObjectLabelArtifactObservabilityStrategy(ArtifactObservabilityStrategy):
    """Object labels induce externally visible object-count measurement facts."""

    artifact_type = ObjectLabelsArtifactType

    def externally_required_outputs(
        self,
        declared_outputs: tuple[ArtifactSpec, ...],
    ) -> tuple[ArtifactSpec, ...]:
        return declared_outputs


class MeasurementArtifactObservabilityStrategy(ArtifactObservabilityStrategy):
    """Measurement artifacts are externally visible semantic facts."""

    artifact_type = MeasurementsArtifactType

    def externally_required_outputs(
        self,
        declared_outputs: tuple[ArtifactSpec, ...],
    ) -> tuple[ArtifactSpec, ...]:
        return declared_outputs


def externally_required_artifact_outputs(
    declared_outputs: tuple[ArtifactSpec, ...],
) -> tuple[ArtifactSpec, ...]:
    """Return declared outputs whose observable facts require runtime retention."""
    required: list[ArtifactSpec] = []
    for output in declared_outputs:
        strategy = ArtifactObservabilityStrategy.for_context(
            output.artifact_type,
            required=False,
        )
        if strategy is None:
            continue
        required.extend(
            strategy.externally_required_outputs((output,))
        )
    return tuple(dict.fromkeys(required))
