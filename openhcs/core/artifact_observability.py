"""Artifact observability policies independent of materialization format."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta

from openhcs.core.artifacts import ArtifactKind, ArtifactSpec
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin


class ArtifactObservabilityStrategy(
    EnumKeyedStrategyMixin[ArtifactKind],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered policy for artifacts that remain externally observable.

    Materialization controls files written by OpenHCS. Observability is broader:
    some runtime artifacts induce exported facts even when no artifact file is
    explicitly requested. Object-label counts are the canonical example.
    """

    strategy_key: ClassVar[ArtifactKind | None] = None
    strategy_label: ClassVar[str | None] = None
    __registry_key__ = "strategy_label"
    __skip_if_no_key__ = True

    @abstractmethod
    def externally_required_outputs(
        self,
        declared_outputs: tuple[ArtifactSpec, ...],
    ) -> tuple[ArtifactSpec, ...]:
        """Return declared outputs that must survive dead-artifact pruning."""


class ObjectLabelArtifactObservabilityStrategy(ArtifactObservabilityStrategy):
    """Object labels induce externally visible object-count measurement facts."""

    strategy_key = ArtifactKind.OBJECT_LABELS

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
    for kind_label, strategy_type in ArtifactObservabilityStrategy.__registry__.items():
        strategy = strategy_type()
        required.extend(
            output
            for output in declared_outputs
            if output.kind.value == kind_label
            for output in strategy.externally_required_outputs((output,))
        )
    return tuple(dict.fromkeys(required))
