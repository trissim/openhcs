"""Axis-scoped compiler session for pipeline compilation stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from openhcs.core.compiled_step_plan import CompiledStepPlan
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.pipeline.step_snapshot import (
    StepSnapshot,
    build_step_snapshots,
)
from openhcs.core.steps.abstract import AbstractStep


@dataclass(slots=True)
class CompilationSession:
    """Compiler boundary for one ProcessingContext.

    The session is not a dict wrapper. It owns the invariants tying together the
    resolved step list, ObjectState map, StepSnapshot tuple, context, and mutable
    compiled-plan map for one axis or sequential-combination context.
    """

    context: ProcessingContext
    steps: Sequence[AbstractStep]
    orchestrator: Any
    step_state_map: Mapping[int, Any]
    snapshots: tuple[StepSnapshot, ...]
    plans: MutableMapping[int, CompiledStepPlan]
    metadata_writer: bool = False
    plate_path: Path | None = None

    @classmethod
    def from_context(
        cls,
        *,
        context: ProcessingContext,
        steps: Sequence[AbstractStep],
        orchestrator: Any,
        step_state_map: Mapping[int, Any],
        snapshots: tuple[StepSnapshot, ...] | None = None,
        metadata_writer: bool = False,
        plate_path: Path | None = None,
    ) -> "CompilationSession":
        if context.step_plans is None:
            raise ValueError("CompilationSession requires context.step_plans.")
        if snapshots is None:
            snapshots = build_step_snapshots(steps, step_state_map)
        return cls(
            context=context,
            steps=steps,
            orchestrator=orchestrator,
            step_state_map=step_state_map,
            snapshots=snapshots,
            plans=context.step_plans,
            metadata_writer=metadata_writer,
            plate_path=plate_path,
        )

    def __post_init__(self) -> None:
        if len(self.steps) != len(self.snapshots):
            raise ValueError(
                "CompilationSession requires one StepSnapshot per step: "
                f"{len(self.snapshots)} snapshots for {len(self.steps)} steps."
            )
        missing_states = [
            index for index in range(len(self.steps)) if index not in self.step_state_map
        ]
        if missing_states:
            raise ValueError(
                f"CompilationSession missing ObjectState entries for steps "
                f"{missing_states}."
            )
        for expected_index, snapshot in enumerate(self.snapshots):
            if snapshot.index != expected_index:
                raise ValueError(
                    f"StepSnapshot index mismatch: expected {expected_index}, "
                    f"got {snapshot.index}."
                )

    @property
    def global_config(self) -> Any:
        return self.context.global_config

    @property
    def axis_id(self) -> str:
        return self.context.axis_id

    def step(self, index: int) -> AbstractStep:
        return self.steps[index]

    def snapshot(self, index: int) -> StepSnapshot:
        return self.snapshots[index]

    def step_state(self, index: int) -> Any:
        try:
            return self.step_state_map[index]
        except KeyError as exc:
            raise ValueError(f"Missing ObjectState for step {index}.") from exc

    def plan(self, index: int) -> CompiledStepPlan:
        try:
            return self.plans[index]
        except KeyError as exc:
            snapshot = self.snapshot(index)
            raise ValueError(
                f"Missing compiled plan for step {index} ({snapshot.name})."
            ) from exc
