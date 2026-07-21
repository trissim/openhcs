"""Nominal artifact-plan key selection shared by compiler declarations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TypeVar

from openhcs.core.artifacts import (
    ArtifactOutputPlan,
    ArtifactPlan,
    ArtifactSpecCollection,
    ArtifactSpecRef,
)

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

    def select_plans(
        self,
        plan_type: type[ArtifactPlanT],
        plans: Mapping[ArtifactSpecRef, ArtifactPlanT],
    ) -> tuple[ArtifactPlanT, ...]:
        """Select exact compiled plans in declaration order.

        The path planner owns whether a semantic input requires a runtime
        artifact plan. Inputs satisfied by source bindings, metadata, or main
        flow are therefore absent from ``plans``. Every present plan remains
        indexed by its declaration-owned exact artifact ref.
        """
        plan_type.require_exact_map(
            plans,
            boundary=f"{type(self).__name__} artifact plan",
        )
        declared = self.artifact_key_specs.for_plan_type(plan_type)
        selected: list[ArtifactPlanT] = []
        for spec in declared.specs:
            ref = spec.ref()
            plan = plans.get(ref)
            if plan is None:
                continue
            selected.append(plan)
        return tuple(selected)

    def validate_artifact_relation_refs(self, *, owner_name: str) -> None:
        self.artifact_specs.validate_registered_relation_refs(
            owner_name=owner_name,
            relation_specs=self.artifact_specs.for_plan_type(
                ArtifactOutputPlan
            ).specs,
        )
