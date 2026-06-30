"""Axis-scoped compiler session for pipeline compilation stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Iterator
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping, MutableMapping, Protocol, Sequence, runtime_checkable

from openhcs.core.compiled_step_plan import CompiledStepPlan
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.pipeline.step_snapshot import (
    StepSnapshot,
    build_step_snapshots,
)
from openhcs.core.steps.abstract import AbstractStep

if TYPE_CHECKING:
    from objectstate import ObjectState
    from openhcs.core.config import GlobalPipelineConfig
    from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator


PIPELINE_SOURCE_SCHEMA_METADATA_KEY = "source_schema"


@runtime_checkable
class PipelineMetadataCarrier(Protocol):
    """Structural protocol for pipeline declarations carrying metadata."""

    metadata: Mapping[str, object]


@runtime_checkable
class PipelineIdentityCarrier(Protocol):
    """Structural protocol for pipeline declarations carrying a stable name."""

    name: str


@dataclass(frozen=True, slots=True)
class CompilationPlateScope:
    """Plate-root identity used for compiler ObjectState scopes and paths."""

    path: Path

    @classmethod
    def from_context(cls, context: ProcessingContext) -> "CompilationPlateScope":
        if context.plate_path is None:
            raise ValueError("Compilation plate scope requires context.plate_path.")
        return cls(Path(context.plate_path))

    @classmethod
    def from_path(cls, plate_path: Path | str | None) -> "CompilationPlateScope":
        if plate_path is None:
            raise ValueError("Compilation plate scope requires a plate path.")
        return cls(Path(plate_path))

    @property
    def object_state_scope_id(self) -> str:
        return str(self.path)


@dataclass(slots=True)
class CompilationSession:
    """Compiler boundary for one ProcessingContext.

    The session is not a dict wrapper. It owns the invariants tying together the
    resolved step list, ObjectState map, StepSnapshot tuple, context, and mutable
    compiled-plan map for one axis or sequential-combination context.
    """

    context: ProcessingContext
    steps: Sequence[AbstractStep]
    orchestrator: "PipelineOrchestrator"
    global_config: "GlobalPipelineConfig"
    step_state_map: Mapping[int, "ObjectState"]
    snapshots: tuple[StepSnapshot, ...]
    plans: MutableMapping[int, CompiledStepPlan]
    pipeline_metadata: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )
    metadata_writer: bool = False
    plate_scope: CompilationPlateScope | None = None
    is_zmq_execution: bool = False

    @classmethod
    def from_context(
        cls,
        *,
        context: ProcessingContext,
        steps: Sequence[AbstractStep],
        orchestrator: "PipelineOrchestrator",
        global_config: "GlobalPipelineConfig",
        step_state_map: Mapping[int, "ObjectState"],
        snapshots: tuple[StepSnapshot, ...] | None = None,
        pipeline_metadata: Mapping[str, object] | None = None,
        metadata_writer: bool = False,
        plate_path: Path | None = None,
        is_zmq_execution: bool = False,
    ) -> "CompilationSession":
        if context.step_plans is None:
            raise ValueError("CompilationSession requires context.step_plans.")
        if snapshots is None:
            snapshots = build_step_snapshots(steps, step_state_map)
        return cls(
            context=context,
            steps=steps,
            orchestrator=orchestrator,
            global_config=global_config,
            step_state_map=step_state_map,
            snapshots=snapshots,
            plans=context.step_plans,
            pipeline_metadata=MappingProxyType(dict(pipeline_metadata or {})),
            metadata_writer=metadata_writer,
            plate_scope=(
                CompilationPlateScope.from_path(plate_path)
                if plate_path is not None
                else None
            ),
            is_zmq_execution=is_zmq_execution,
        )

    def __post_init__(self) -> None:
        if self.plate_scope is None and self.context.plate_path is not None:
            self.plate_scope = CompilationPlateScope.from_context(self.context)
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
    def axis_id(self) -> str:
        return self.context.axis_id

    @property
    def plate_path(self) -> Path | None:
        if self.plate_scope is None:
            return None
        return self.plate_scope.path

    def step(self, index: int) -> AbstractStep:
        return self.steps[index]

    def snapshot(self, index: int) -> StepSnapshot:
        return self.snapshots[index]

    @property
    def step_count(self) -> int:
        return len(self.snapshots)

    def indexed_snapshots(self) -> Iterator[tuple[int, StepSnapshot]]:
        return iter(enumerate(self.snapshots))

    def reverse_snapshot_indices(self) -> range:
        return range(self.step_count - 1, -1, -1)

    def step_state(self, index: int) -> "ObjectState":
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


@dataclass(frozen=True, slots=True)
class ResolvedPipelineDefinition:
    """ObjectState-resolved pipeline declaration shared by all axis sessions."""

    steps: Sequence[AbstractStep]
    step_state_map: Mapping[int, "ObjectState"]
    snapshots: tuple[StepSnapshot, ...]
    metadata: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )

    @classmethod
    def metadata_from_steps(
        cls,
        steps: Sequence[AbstractStep],
    ) -> Mapping[str, object]:
        if not isinstance(steps, PipelineMetadataCarrier):
            return MappingProxyType({})
        return MappingProxyType(dict(steps.metadata))
