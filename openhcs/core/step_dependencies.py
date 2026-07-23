"""Typed step-dependency records for compiled pipeline execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class StepInputDependencyKind(str, Enum):
    """Closed family of main-input dependency kinds."""

    UNRESOLVED = "unresolved"
    NO_MAIN_FLOW = "no_main_flow"
    PIPELINE_START = "pipeline_start"
    STEP_OUTPUT = "step_output"


@dataclass(frozen=True, slots=True)
class StepInputDependency:
    """Authoritative main-input edge for one compiled step."""

    kind: StepInputDependencyKind
    source_step_index: int | None = None
    source_step_scope_id: str | None = None

    @classmethod
    def unresolved(cls) -> "StepInputDependency":
        return cls(StepInputDependencyKind.UNRESOLVED)

    @classmethod
    def pipeline_start(cls) -> "StepInputDependency":
        return cls(StepInputDependencyKind.PIPELINE_START)

    @classmethod
    def no_main_flow(cls) -> "StepInputDependency":
        return cls(StepInputDependencyKind.NO_MAIN_FLOW)

    @classmethod
    def step_output(
        cls,
        *,
        source_step_index: int,
        source_step_scope_id: str,
    ) -> "StepInputDependency":
        return cls(
            kind=StepInputDependencyKind.STEP_OUTPUT,
            source_step_index=source_step_index,
            source_step_scope_id=source_step_scope_id,
        )

    def __post_init__(self) -> None:
        if self.kind is StepInputDependencyKind.STEP_OUTPUT:
            if self.source_step_index is None:
                raise ValueError(
                    "StepInputDependency.step_output requires source_step_index."
                )
            if not self.source_step_scope_id:
                raise ValueError(
                    "StepInputDependency.step_output requires source_step_scope_id."
                )
            return

        if self.source_step_index is not None or self.source_step_scope_id is not None:
            raise ValueError(
                f"StepInputDependency kind {self.kind.value!r} cannot carry a source step."
            )

    @property
    def is_resolved(self) -> bool:
        return self.kind is not StepInputDependencyKind.UNRESOLVED
