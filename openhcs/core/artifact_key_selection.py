"""Nominal artifact-plan key selection shared by compiler declarations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TypeVar

from openhcs.core.artifacts import ArtifactPlan, ArtifactSpecCollection

ArtifactPlanT = TypeVar("ArtifactPlanT", bound=ArtifactPlan)


class ArtifactPlanKeySelector(ABC):
    """Nominal interface for declarations that select compiled artifact plans."""

    @property
    @abstractmethod
    def artifact_specs(self) -> ArtifactSpecCollection:
        """All artifact specs declared by this owner."""

    @property
    def artifact_key_specs(self) -> ArtifactSpecCollection:
        """Artifact specs that participate in compiled plan-key selection."""
        return self.artifact_specs

    def artifact_names_for(
        self,
        plan_type: type[ArtifactPlanT],
    ) -> tuple[str, ...]:
        return self.artifact_key_specs.names_for_plan_type(plan_type)

    def select_plan_keys(
        self,
        plan_type: type[ArtifactPlanT],
        plans: Mapping[str, ArtifactPlanT],
    ) -> tuple[str, ...]:
        """Select compiled artifact plans consumed or produced by this declaration."""
        declared = set(self.artifact_names_for(plan_type))
        return tuple(key for key in plans if key in declared)

    def validate_artifact_relation_refs(self, *, owner_name: str) -> None:
        self.artifact_specs.validate_registered_relation_refs(owner_name=owner_name)
