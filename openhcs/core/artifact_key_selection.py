"""Nominal artifact-plan key selection shared by compiler declarations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

from openhcs.core.artifacts import ArtifactInputPlan, ArtifactOutputPlan


class ArtifactPlanKeySelector(ABC):
    """Nominal interface for declarations that select compiled artifact plans."""

    @property
    @abstractmethod
    def input_names(self) -> tuple[str, ...]:
        """Declared artifact input names in declaration order."""

    @property
    @abstractmethod
    def output_names(self) -> tuple[str, ...]:
        """Declared artifact output names in declaration order."""

    def select_input_plan_keys(
        self,
        input_plans: Mapping[str, ArtifactInputPlan],
    ) -> tuple[str, ...]:
        """Select compiled artifact inputs consumed by this declaration."""
        declared = set(self.input_names)
        return tuple(key for key in input_plans if key in declared)

    def select_output_plan_keys(
        self,
        output_plans: Mapping[str, ArtifactOutputPlan],
    ) -> tuple[str, ...]:
        """Select compiled artifact outputs produced by this declaration."""
        declared = set(self.output_names)
        return tuple(key for key in output_plans if key in declared)
