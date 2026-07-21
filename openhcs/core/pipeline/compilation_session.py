"""Axis-scoped compiler session for pipeline compilation stages."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Mapping, MutableMapping, Sequence, get_type_hints

from objectstate import DataclassFieldAccess

from openhcs.core.compiled_step_plan import CompiledStepPlan
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.pipeline.step_snapshot import (
    StepSnapshot,
    build_step_snapshots,
)
from openhcs.core.steps.abstract import AbstractStep
from openhcs.core.source_metadata import SourceMetadataMapping
from openhcs.core.source_workspace_projection import VirtualWorkspaceSourceProjection
from openhcs.core.vfs_protocol import (
    FileManagerLike,
    PlatePathDeclaration,
)

if TYPE_CHECKING:
    from objectstate import ObjectState
    from openhcs.core.config import GlobalPipelineConfig
    from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator


@dataclass(frozen=True, slots=True)
class CompilationPlateScope:
    """Plate-root identity used for compiler ObjectState scopes and paths."""

    path: Path

    def __post_init__(self) -> None:
        if not self.path.is_absolute():
            raise ValueError(
                f"Compilation plate scope must be absolute, got {self.path}."
            )

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

    def resolve_address(
        self,
        value: str | Path,
        *,
        filemanager: FileManagerLike,
        backend: str,
    ) -> Path:
        """Resolve one address through VFS with this exact plate base."""

        resolved = Path(
            filemanager.resolve_address(
                value,
                backend,
                base_path=self.path,
            )
        )
        if not resolved.is_absolute():
            raise ValueError(
                "VFS path resolution must return an absolute address: "
                f"{value!r} -> {resolved}."
            )
        return resolved


@dataclass(frozen=True, slots=True)
class CompilationPathResolver:
    """Resolve declaration-owned paths for one compilation plate."""

    plate_scope: CompilationPlateScope
    filemanager: FileManagerLike
    backend: str

    def resolve(
        self,
        value: str | Path,
        declaration: PlatePathDeclaration,
        *,
        owner: str,
    ) -> Path:
        try:
            target = self.plate_scope.resolve_address(
                value,
                filemanager=self.filemanager,
                backend=self.backend,
            )
            declaration.validate_target(
                target,
                filemanager=self.filemanager,
                backend=self.backend,
            )
            return target
        except Exception as error:
            error.add_note(
                f"While resolving {owner}: authored={value!r}, "
                f"plate_root={self.plate_scope.path}, backend={self.backend!r}."
            )
            raise


def resolve_declared_dataclass_paths(
    value: Any,
    resolver: CompilationPathResolver,
    *,
    owner: str,
) -> Any:
    """Return an immutable dataclass copy with declared paths resolved."""

    if not is_dataclass(value) or isinstance(value, type):
        return value
    annotations = get_type_hints(type(value), include_extras=True)
    replacements: dict[str, object] = {}
    for dataclass_field in fields(value):
        field_value = DataclassFieldAccess.raw_value(value, dataclass_field.name)
        declaration = PlatePathDeclaration.from_annotation(
            annotations.get(dataclass_field.name)
        )
        if declaration is not None and field_value is not None:
            if not isinstance(field_value, (str, Path)):
                raise TypeError(
                    f"{owner}.{dataclass_field.name} declares a plate path but "
                    f"contains {type(field_value).__name__}."
                )
            resolved_value = resolver.resolve(
                field_value,
                declaration,
                owner=f"{owner}.{dataclass_field.name}",
            )
        else:
            resolved_value = resolve_declared_dataclass_paths(
                field_value,
                resolver,
                owner=f"{owner}.{dataclass_field.name}",
            )
        if resolved_value is not field_value:
            replacements[dataclass_field.name] = resolved_value
    return replace(value, **replacements) if replacements else value


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
    source_workspace_projection: VirtualWorkspaceSourceProjection
    path_resolver: CompilationPathResolver | None = None
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
        source_workspace_projection: VirtualWorkspaceSourceProjection | None = None,
        path_resolver: CompilationPathResolver | None = None,
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
            source_workspace_projection=(
                VirtualWorkspaceSourceProjection.empty(context.plate_path)
                if source_workspace_projection is None
                else source_workspace_projection
            ),
            path_resolver=path_resolver,
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
            index
            for index in range(len(self.steps))
            if index not in self.step_state_map
        ]
        if missing_states:
            raise ValueError(
                f"CompilationSession missing ObjectState entries for steps "
                f"{missing_states}."
            )
        for expected_index, (snapshot, step) in enumerate(
            zip(self.snapshots, self.steps, strict=True)
        ):
            if snapshot.index != expected_index:
                raise ValueError(
                    f"StepSnapshot index mismatch: expected {expected_index}, "
                    f"got {snapshot.index}."
                )
            if snapshot.step is not step:
                raise ValueError(
                    f"StepSnapshot {expected_index} does not reference its resolved step."
                )

    @property
    def axis_id(self) -> str:
        return self.context.axis_id

    @property
    def plate_path(self) -> Path | None:
        if self.plate_scope is None:
            return None
        return self.plate_scope.path

    @property
    def realized_source_metadata(
        self,
    ) -> tuple[SourceMetadataMapping, ...] | None:
        """Return the axis-scoped source metadata realized for this compilation."""

        metadata = tuple(
            self.source_workspace_projection.source_metadata_by_path.values()
        )
        return metadata or None

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
                f"Missing compiled plan for step {index} ({snapshot.step.name})."
            ) from exc


@dataclass(frozen=True, slots=True)
class ResolvedPipelineDefinition:
    """ObjectState-resolved pipeline declaration shared by all axis sessions."""

    steps: Sequence[AbstractStep]
    step_state_map: Mapping[int, "ObjectState"]
    snapshots: tuple[StepSnapshot, ...]
