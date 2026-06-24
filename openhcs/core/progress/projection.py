"""OpenHCS runtime projection built on generic zmqruntime projection primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from zmqruntime.progress import (
    GenericAxisProjection,
    GenericExecutionProjection,
    GenericPlateProjection,
    ProgressProjectionAdapterABC,
    build_execution_projection,
)

from .types import (
    ProgressEvent,
    phase_channel,
    is_failure_event,
    is_success_terminal_event,
)


class PlateRuntimeState(str, Enum):
    def __new__(cls, value: str, is_terminal: bool):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj._is_terminal = is_terminal
        return obj

    IDLE = ("idle", False)
    COMPILING = ("compiling", False)
    COMPILED = ("compiled", False)
    EXECUTING = ("executing", False)
    COMPLETE = ("complete", True)
    FAILED = ("failed", True)

    @property
    def is_terminal(self) -> bool:
        return self._is_terminal


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

    @classmethod
    def from_generic_plate(
        cls,
        generic_plate: GenericPlateProjection[PlateRuntimeState],
    ) -> "PlateRuntimeProjection":
        return cls(
            identity=PlateRuntimeIdentity.from_generic_plate(generic_plate),
            state=generic_plate.state,
            percent=generic_plate.percent,
            axis_progress=tuple(
                AxisRuntimeProjection.from_generic_axis(axis)
                for axis in generic_plate.axis_progress
            ),
            latest_timestamp=generic_plate.latest_timestamp,
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


@dataclass
class ExecutionRuntimeProjection:
    plates: List[PlateRuntimeProjection] = field(default_factory=list)
    by_identity: Dict[PlateRuntimeIdentity, PlateRuntimeProjection] = field(
        default_factory=dict
    )
    by_plate_latest: Dict[str, PlateRuntimeProjection] = field(default_factory=dict)
    compiling_count: int = 0
    compiled_count: int = 0
    executing_count: int = 0
    complete_count: int = 0
    failed_count: int = 0
    overall_percent: float = 0.0

    @classmethod
    def from_generic_projection(
        cls,
        generic_projection: GenericExecutionProjection[PlateRuntimeState],
    ) -> "ExecutionRuntimeProjection":
        projection = cls()

        for generic_plate in generic_projection.plates:
            projection.add_plate(
                PlateRuntimeProjection.from_generic_plate(generic_plate)
            )

        for generic_plate in generic_projection.by_plate_latest.values():
            projection.mark_latest(
                PlateRuntimeIdentity.from_generic_plate(generic_plate)
            )

        projection.compiling_count = generic_projection.count_state(
            PlateRuntimeState.COMPILING
        )
        projection.compiled_count = generic_projection.count_state(
            PlateRuntimeState.COMPILED
        )
        projection.executing_count = generic_projection.count_state(
            PlateRuntimeState.EXECUTING
        )
        projection.complete_count = generic_projection.count_state(
            PlateRuntimeState.COMPLETE
        )
        projection.failed_count = generic_projection.count_state(PlateRuntimeState.FAILED)
        projection.overall_percent = generic_projection.overall_percent

        return projection

    def add_plate(self, plate_projection: PlateRuntimeProjection) -> None:
        self.plates.append(plate_projection)
        self.by_identity[plate_projection.identity] = plate_projection

    def mark_latest(self, identity: PlateRuntimeIdentity) -> None:
        self.by_plate_latest[identity.plate_id] = self.by_identity[identity]

    def get_plate(
        self, plate_id: str, execution_id: Optional[str] = None
    ) -> Optional[PlateRuntimeProjection]:
        if execution_id is not None:
            return self.by_identity.get(PlateRuntimeIdentity(execution_id, plate_id))
        return self.by_plate_latest.get(plate_id)


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
) -> ExecutionRuntimeProjection:
    generic_projection = build_execution_projection(
        events_by_execution,
        adapter=_PROJECTION_ADAPTER,
    )
    return ExecutionRuntimeProjection.from_generic_projection(generic_projection)
