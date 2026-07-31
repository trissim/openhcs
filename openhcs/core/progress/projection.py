"""OpenHCS runtime projection built on generic zmqruntime projection primitives."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from metaclass_registry import AutoRegisterMeta
from zmqruntime.messages import QueuedExecutionInfo, RunningExecutionInfo
from zmqruntime.progress import (
    GenericAxisProjection,
    GenericExecutionProjection,
    GenericPlateProjection,
    ProgressProjectionAdapterABC,
    build_execution_projection,
)

from .types import (
    ProgressChannel,
    ProgressEvent,
    ProgressPhase,
    is_failure_event,
    is_success_terminal_event,
    phase_channel,
)


class PlateRuntimeState(str, Enum):
    IDLE = "idle"
    QUEUED = "queued"
    COMPILING = "compiling"
    COMPILED = "compiled"
    EXECUTING = "executing"
    COMPLETE = "complete"
    FAILED = "failed"


class PlateRuntimeStateDeclarationBase(ABC, metaclass=AutoRegisterMeta):
    """Nominal semantic declaration for one plate runtime state."""

    __registry_key__ = "state"
    __skip_if_no_key__ = True
    __registry__: ClassVar[
        dict[PlateRuntimeState, type["PlateRuntimeStateDeclarationBase"]]
    ] = {}

    state: ClassVar[PlateRuntimeState | None] = None
    is_terminal: ClassVar[bool] = False
    default_channel: ClassVar[ProgressChannel | None] = None
    status_label: ClassVar[str] = ""
    count_label: ClassVar[str] = ""
    count_sort_order: ClassVar[int] = 0

    @classmethod
    def require_state(cls) -> PlateRuntimeState:
        if cls.state is None:
            raise TypeError(f"{cls.__name__} does not declare a runtime state.")
        return cls.state

    @classmethod
    def for_state(
        cls,
        state: PlateRuntimeState,
    ) -> type["PlateRuntimeStateDeclarationBase"]:
        return cls.__registry__[state]

    @classmethod
    def state_channel_from_events(
        cls,
        events: Iterable[ProgressEvent],
    ) -> ProgressChannel | None:
        del events
        return cls.default_channel

    @classmethod
    def percent_from_generic_projection(
        cls,
        *,
        generic_percent: float,
        axis_progress: Tuple["AxisRuntimeProjection", ...],
    ) -> float:
        del axis_progress
        return generic_percent

    @classmethod
    def status_label_for_plate(cls, plate: "PlateRuntimeProjection") -> str:
        del plate
        return cls.status_label

    @classmethod
    def formatted_status_for_plate(cls, plate: "PlateRuntimeProjection") -> str:
        label = cls.status_label_for_plate(plate)
        if not label:
            return ""
        return f"{label} {plate.percent:.1f}%"

    @classmethod
    def counted_declarations(
        cls,
    ) -> Tuple[type["PlateRuntimeStateDeclarationBase"], ...]:
        return tuple(
            sorted(
                (
                    declaration
                    for declaration in cls.__registry__.values()
                    if declaration.count_label
                ),
                key=lambda declaration: declaration.count_sort_order,
            )
        )

    @classmethod
    def count_status_label(cls, count: int) -> str:
        if count <= 0 or not cls.count_label:
            return ""
        return cls.count_label.format(count=count)

    @classmethod
    def accepts_server_lifecycle_state(cls, state: PlateRuntimeState) -> bool:
        """Return whether a live server snapshot may advance this state."""

        del state
        return False


class TerminalPlateRuntimeState:
    """Trait for plate runtime states that close the current execution."""

    is_terminal: ClassVar[bool] = True


class PendingPlateRuntimeState:
    """Trait for states fully governed by a live server lifecycle snapshot."""

    @classmethod
    def accepts_server_lifecycle_state(cls, state: PlateRuntimeState) -> bool:
        return state in (
            PlateRuntimeState.QUEUED,
            PlateRuntimeState.COMPILING,
            PlateRuntimeState.EXECUTING,
        )


class IdlePlateRuntimeState(PendingPlateRuntimeState, PlateRuntimeStateDeclarationBase):
    state = PlateRuntimeState.IDLE


class QueuedPlateRuntimeState(
    PendingPlateRuntimeState,
    PlateRuntimeStateDeclarationBase,
):
    state = PlateRuntimeState.QUEUED
    status_label = "⏳ Queued"
    count_label = "⏳ {count} queued"
    count_sort_order = 5

    @classmethod
    def formatted_status_for_plate(cls, plate: "PlateRuntimeProjection") -> str:
        status = super().formatted_status_for_plate(plate)
        if plate.queue_position is None:
            return status
        return f"{status} (q#{plate.queue_position})"


class CompilingPlateRuntimeState(PlateRuntimeStateDeclarationBase):
    state = PlateRuntimeState.COMPILING
    default_channel = ProgressChannel.COMPILE
    status_label = "⏳ Compiling"
    count_label = "⏳ {count} compiling"
    count_sort_order = 10

    @classmethod
    def accepts_server_lifecycle_state(cls, state: PlateRuntimeState) -> bool:
        return state in (
            PlateRuntimeState.COMPILING,
            PlateRuntimeState.EXECUTING,
        )


class CompiledPlateRuntimeState(PlateRuntimeStateDeclarationBase):
    state = PlateRuntimeState.COMPILED
    default_channel = ProgressChannel.COMPILE
    status_label = "✅ Compiled"
    count_label = "✓ {count} compiled"
    count_sort_order = 30

    @classmethod
    def accepts_server_lifecycle_state(cls, state: PlateRuntimeState) -> bool:
        return state is PlateRuntimeState.EXECUTING


class ExecutingPlateRuntimeState(PlateRuntimeStateDeclarationBase):
    state = PlateRuntimeState.EXECUTING
    default_channel = ProgressChannel.PIPELINE
    status_label = "⚙️ Executing"
    count_label = "⚙️ {count} executing"
    count_sort_order = 20

    @classmethod
    def accepts_server_lifecycle_state(cls, state: PlateRuntimeState) -> bool:
        return state is PlateRuntimeState.EXECUTING


class CompletePlateRuntimeState(
    TerminalPlateRuntimeState, PlateRuntimeStateDeclarationBase
):
    state = PlateRuntimeState.COMPLETE
    default_channel = ProgressChannel.PIPELINE
    status_label = "✅ Complete"
    count_label = "✅ {count} complete"
    count_sort_order = 40


class FailedPlateRuntimeState(
    TerminalPlateRuntimeState, PlateRuntimeStateDeclarationBase
):
    state = PlateRuntimeState.FAILED
    status_label = "❌ Failed"
    count_label = "❌ {count} failed"
    count_sort_order = 50

    @classmethod
    def state_channel_from_events(
        cls,
        events: Iterable[ProgressEvent],
    ) -> ProgressChannel | None:
        latest_failure_event = max(
            (event for event in events if is_failure_event(event)),
            key=lambda event: event.timestamp,
            default=None,
        )
        if latest_failure_event is None:
            return cls.default_channel
        return phase_channel(latest_failure_event.phase)

    @classmethod
    def percent_from_generic_projection(
        cls,
        *,
        generic_percent: float,
        axis_progress: Tuple["AxisRuntimeProjection", ...],
    ) -> float:
        if generic_percent > 0.0 or not axis_progress:
            return generic_percent
        return sum(axis.percent for axis in axis_progress) / len(axis_progress)

    @classmethod
    def status_label_for_plate(cls, plate: "PlateRuntimeProjection") -> str:
        if plate.state_channel is ProgressChannel.COMPILE:
            return "❌ Compile Failed"
        return cls.status_label


@dataclass(frozen=True)
class AxisRuntimeProjection:
    axis_id: str
    percent: float
    step_name: str
    is_complete: bool
    is_failed: bool

    @classmethod
    def from_generic_axis(
        cls,
        generic_axis: GenericAxisProjection,
    ) -> "AxisRuntimeProjection":
        return cls(
            axis_id=generic_axis.axis_id,
            percent=generic_axis.percent,
            step_name=generic_axis.step_name,
            is_complete=generic_axis.is_complete,
            is_failed=generic_axis.is_failed,
        )


@dataclass(frozen=True)
class PlateRuntimeIdentity:
    execution_id: str
    plate_id: str

    @classmethod
    def from_generic_plate(
        cls,
        generic_plate: GenericPlateProjection[PlateRuntimeState],
    ) -> "PlateRuntimeIdentity":
        return cls(
            execution_id=generic_plate.execution_id,
            plate_id=generic_plate.plate_id,
        )


@dataclass(frozen=True)
class PlateRuntimeProjection:
    identity: PlateRuntimeIdentity
    state: PlateRuntimeState
    percent: float
    axis_progress: Tuple[AxisRuntimeProjection, ...]
    latest_timestamp: float
    state_channel: ProgressChannel | None = None
    queue_position: int | None = None

    @classmethod
    def from_generic_plate(
        cls,
        generic_plate: GenericPlateProjection[PlateRuntimeState],
        *,
        state_channel: ProgressChannel | None = None,
    ) -> "PlateRuntimeProjection":
        axis_progress = tuple(
            AxisRuntimeProjection.from_generic_axis(axis)
            for axis in generic_plate.axis_progress
        )
        state_declaration = PlateRuntimeStateDeclarationBase.for_state(
            generic_plate.state
        )
        return cls(
            identity=PlateRuntimeIdentity.from_generic_plate(generic_plate),
            state=generic_plate.state,
            percent=state_declaration.percent_from_generic_projection(
                generic_percent=generic_plate.percent,
                axis_progress=axis_progress,
            ),
            axis_progress=axis_progress,
            latest_timestamp=generic_plate.latest_timestamp,
            state_channel=state_channel,
        )

    @property
    def execution_id(self) -> str:
        return self.identity.execution_id

    @property
    def plate_id(self) -> str:
        return self.identity.plate_id

    @property
    def active_axes(self) -> Tuple[AxisRuntimeProjection, ...]:
        return tuple(
            axis
            for axis in self.axis_progress
            if not axis.is_complete and not axis.is_failed
        )

    @property
    def is_terminal(self) -> bool:
        return PlateRuntimeStateDeclarationBase.for_state(self.state).is_terminal

    @property
    def status_label(self) -> str:
        return PlateRuntimeStateDeclarationBase.for_state(
            self.state
        ).status_label_for_plate(self)

    @property
    def formatted_status(self) -> str:
        return PlateRuntimeStateDeclarationBase.for_state(
            self.state
        ).formatted_status_for_plate(self)


@dataclass
class ExecutionRuntimeProjection:
    plates: List[PlateRuntimeProjection] = field(default_factory=list)
    by_identity: Dict[PlateRuntimeIdentity, PlateRuntimeProjection] = field(
        default_factory=dict
    )
    by_plate_latest: Dict[str, PlateRuntimeProjection] = field(default_factory=dict)
    state_counts: Dict[PlateRuntimeState, int] = field(default_factory=dict)
    overall_percent: float = 0.0

    @classmethod
    def from_generic_projection(
        cls,
        generic_projection: GenericExecutionProjection[PlateRuntimeState],
        events_by_identity: Mapping[PlateRuntimeIdentity, Sequence[ProgressEvent]]
        | None = None,
    ) -> "ExecutionRuntimeProjection":
        projection = cls()

        for generic_plate in generic_projection.plates:
            identity = PlateRuntimeIdentity.from_generic_plate(generic_plate)
            state_declaration = PlateRuntimeStateDeclarationBase.for_state(
                generic_plate.state
            )
            projection.add_plate(
                PlateRuntimeProjection.from_generic_plate(
                    generic_plate,
                    state_channel=state_declaration.state_channel_from_events(
                        ()
                        if events_by_identity is None
                        else events_by_identity.get(identity, ())
                    ),
                )
            )

        for generic_plate in generic_projection.by_plate_latest.values():
            projection.mark_latest(
                PlateRuntimeIdentity.from_generic_plate(generic_plate)
            )

        projection.recalculate_summary()

        return projection

    def count_for_state(self, state: PlateRuntimeState) -> int:
        return self.state_counts.get(state, 0)

    def count_status_labels(self) -> Tuple[str, ...]:
        labels: List[str] = []
        for declaration in PlateRuntimeStateDeclarationBase.counted_declarations():
            label = declaration.count_status_label(
                self.count_for_state(declaration.require_state())
            )
            if label:
                labels.append(label)
        return tuple(labels)

    def add_plate(self, plate_projection: PlateRuntimeProjection) -> None:
        self.plates.append(plate_projection)
        self.by_identity[plate_projection.identity] = plate_projection

    def upsert_plate(self, plate_projection: PlateRuntimeProjection) -> None:
        existing = self.by_identity.get(plate_projection.identity)
        if existing is None:
            self.add_plate(plate_projection)
            return
        self.plates[self.plates.index(existing)] = plate_projection
        self.by_identity[plate_projection.identity] = plate_projection

    def mark_latest(self, identity: PlateRuntimeIdentity) -> None:
        self.by_plate_latest[identity.plate_id] = self.by_identity[identity]

    def get_plate(
        self, plate_id: str, execution_id: Optional[str] = None
    ) -> Optional[PlateRuntimeProjection]:
        if execution_id is not None:
            return self.by_identity.get(PlateRuntimeIdentity(execution_id, plate_id))
        return self.by_plate_latest.get(plate_id)

    def reconcile_server_executions(
        self,
        *,
        running_executions: Sequence[RunningExecutionInfo],
        queued_executions: Sequence[QueuedExecutionInfo],
    ) -> None:
        """Project the authoritative live server queue over retained event history."""

        running_identities = {
            PlateRuntimeIdentity(entry.execution_id, entry.plate_id)
            for entry in running_executions
        }
        for entry in queued_executions:
            identity = PlateRuntimeIdentity(entry.execution_id, entry.plate_id)
            if identity in running_identities:
                continue
            current = self.by_identity.get(identity)
            if current is None or PlateRuntimeStateDeclarationBase.for_state(
                current.state
            ).accepts_server_lifecycle_state(PlateRuntimeState.QUEUED):
                self.upsert_plate(
                    PlateRuntimeProjection(
                        identity=identity,
                        state=PlateRuntimeState.QUEUED,
                        percent=0.0,
                        axis_progress=(),
                        latest_timestamp=(
                            0.0 if current is None else current.latest_timestamp
                        ),
                        queue_position=entry.queue_position,
                    )
                )
            self.mark_latest(identity)

        for entry in running_executions:
            identity = PlateRuntimeIdentity(entry.execution_id, entry.plate_id)
            current = self.by_identity.get(identity)
            server_state = (
                PlateRuntimeState.COMPILING
                if entry.compile_only
                else PlateRuntimeState.EXECUTING
            )
            if current is None or PlateRuntimeStateDeclarationBase.for_state(
                current.state
            ).accepts_server_lifecycle_state(server_state):
                self.upsert_plate(
                    PlateRuntimeProjection(
                        identity=identity,
                        state=server_state,
                        percent=0.0 if current is None else current.percent,
                        axis_progress=(
                            () if current is None else current.axis_progress
                        ),
                        latest_timestamp=max(
                            entry.start_time,
                            0.0 if current is None else current.latest_timestamp,
                        ),
                    )
                )
            self.mark_latest(identity)

        self.recalculate_summary()

    def recalculate_summary(self) -> None:
        self.state_counts = {
            declaration.require_state(): 0
            for declaration in PlateRuntimeStateDeclarationBase.__registry__.values()
        }
        for plate in self.by_plate_latest.values():
            self.state_counts[plate.state] += 1
        self.overall_percent = (
            sum(plate.percent for plate in self.by_plate_latest.values())
            / len(self.by_plate_latest)
            if self.by_plate_latest
            else 0.0
        )


class _OpenHCSProjectionAdapter(
    ProgressProjectionAdapterABC[ProgressEvent, PlateRuntimeState]
):
    def plate_id(self, event: ProgressEvent) -> str:
        return event.plate_id

    def axis_id(self, event: ProgressEvent) -> str:
        return event.axis_id

    def step_name(self, event: ProgressEvent) -> str:
        return event.step_name

    def percent(self, event: ProgressEvent) -> float:
        return event.percent

    def timestamp(self, event: ProgressEvent) -> float:
        return event.timestamp

    def channel(self, event: ProgressEvent) -> str:
        return phase_channel(event.phase).value

    def known_axes(self, events: Iterable[ProgressEvent]) -> List[str]:
        axes: set[str] = set()
        for event in events:
            if event.total_wells:
                axes.update(event.total_wells)
        return sorted(axes)

    def is_failure_event(self, event: ProgressEvent) -> bool:
        return is_failure_event(event)

    def is_success_terminal_event(self, event: ProgressEvent) -> bool:
        return is_success_terminal_event(event)

    def state_idle(self) -> PlateRuntimeState:
        return PlateRuntimeState.IDLE

    def state_idle_from_events(
        self,
        events: Sequence[ProgressEvent],
    ) -> PlateRuntimeState:
        if any(
            event.worker_assignments is not None or event.phase is ProgressPhase.QUEUED
            for event in events
        ):
            return PlateRuntimeState.QUEUED
        return self.state_idle()

    def state_compiling(self) -> PlateRuntimeState:
        return PlateRuntimeState.COMPILING

    def state_compiled(self) -> PlateRuntimeState:
        return PlateRuntimeState.COMPILED

    def state_executing(self) -> PlateRuntimeState:
        return PlateRuntimeState.EXECUTING

    def state_complete(self) -> PlateRuntimeState:
        return PlateRuntimeState.COMPLETE

    def state_failed(self) -> PlateRuntimeState:
        return PlateRuntimeState.FAILED


_PROJECTION_ADAPTER = _OpenHCSProjectionAdapter()


def build_execution_runtime_projection(
    events_by_execution: Mapping[str, List[ProgressEvent]],
    *,
    running_executions: Sequence[RunningExecutionInfo] = (),
    queued_executions: Sequence[QueuedExecutionInfo] = (),
) -> ExecutionRuntimeProjection:
    events_by_identity: Dict[PlateRuntimeIdentity, List[ProgressEvent]] = {}
    for execution_id, events in events_by_execution.items():
        for event in events:
            events_by_identity.setdefault(
                PlateRuntimeIdentity(execution_id, event.plate_id),
                [],
            ).append(event)

    generic_projection = build_execution_projection(
        events_by_execution,
        adapter=_PROJECTION_ADAPTER,
    )
    projection = ExecutionRuntimeProjection.from_generic_projection(
        generic_projection,
        events_by_identity=events_by_identity,
    )
    projection.reconcile_server_executions(
        running_executions=running_executions,
        queued_executions=queued_executions,
    )
    return projection
