"""ZMQ progress event construction for execution-server phases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import os
import time

from zmqruntime.messages import MessageFields

from openhcs.core.progress import (
    ProgressEvent,
    ProgressIdentity,
    ProgressPhase,
    ProgressStatus,
)


@dataclass(frozen=True, slots=True)
class ZMQProgressTarget:
    """Canonical ZMQ progress stream identity."""

    enqueue: Callable[[dict], None]
    plate_id: str


@dataclass(frozen=True, slots=True)
class ImmediateZMQProgressQueue(ZMQProgressTarget):
    """Queue adapter that forwards compiler progress updates immediately."""

    flush: Callable[[], None]

    def put(self, progress_update: dict) -> None:
        canonical_update = dict(progress_update)
        canonical_update[MessageFields.PLATE_ID] = self.plate_id
        self.enqueue(canonical_update)
        self.flush()


@dataclass(frozen=True, slots=True)
class ZMQProgressEmitter(ZMQProgressTarget):
    """Semantic progress-event emitter for one ZMQ execution."""

    execution_id: str

    def compile_started(self, step_count: int) -> None:
        self.emit(
            axis_id="",
            step_name="pipeline",
            total=step_count,
            phase=ProgressPhase.COMPILE,
            status=ProgressStatus.STARTED,
            completed=0,
            percent=0.0,
        )

    def planned_init_started(
        self,
        *,
        wells: list[str],
        step_names: list[str],
    ) -> None:
        self.emit(
            axis_id="",
            step_name="",
            phase=ProgressPhase.INIT,
            status=ProgressStatus.STARTED,
            percent=0.0,
            completed=0,
            total=1,
            total_wells=sorted(wells),
            step_names=step_names,
        )

    def artifact_init_started(
        self,
        *,
        compiled_axis_ids: list[str],
        worker_assignments: dict[str, list[str]],
        step_names: list[str],
    ) -> None:
        self.emit(
            axis_id="",
            step_name="",
            phase=ProgressPhase.INIT,
            status=ProgressStatus.STARTED,
            percent=0.0,
            completed=0,
            total=1,
            total_wells=compiled_axis_ids,
            worker_assignments=worker_assignments,
            step_names=step_names,
        )

    def compiled_init_started(
        self,
        *,
        compiled_axis_ids: list[str],
        worker_assignments: dict[str, list[str]],
    ) -> None:
        self.emit(
            axis_id="",
            step_name="",
            phase=ProgressPhase.INIT,
            status=ProgressStatus.STARTED,
            percent=0.0,
            completed=0,
            total=1,
            total_wells=compiled_axis_ids,
            worker_assignments=worker_assignments,
        )

    def compile_succeeded(
        self,
        *,
        step_count: int,
        compiled_axis_ids: list[str],
        worker_assignments: dict[str, list[str]],
    ) -> None:
        self.emit(
            axis_id="",
            step_name="pipeline",
            total=step_count,
            phase=ProgressPhase.COMPILE,
            status=ProgressStatus.SUCCESS,
            completed=1,
            percent=100.0,
            total_wells=compiled_axis_ids,
            worker_assignments=worker_assignments,
        )

    def axis_compile_succeeded(self, axis_id: str) -> None:
        self.emit(
            axis_id=axis_id,
            step_name="compilation",
            phase=ProgressPhase.COMPILE,
            status=ProgressStatus.SUCCESS,
            completed=1,
            total=1,
            percent=100.0,
        )

    def compile_failed(self, *, axis_ids: list[str], error: str) -> None:
        failed_axis_ids = sorted(axis_ids) if axis_ids else ["pipeline"]
        for axis_id in failed_axis_ids:
            self.emit(
                axis_id=axis_id,
                step_name="compilation",
                phase=ProgressPhase.COMPILE,
                status=ProgressStatus.FAILED,
                completed=0,
                total=1,
                percent=0.0,
                error=error,
                message=error,
            )

    def emit(
        self,
        *,
        axis_id: str,
        step_name: str,
        phase: ProgressPhase,
        status: ProgressStatus,
        percent: float,
        completed: int,
        total: int,
        error: str | None = None,
        message: str | None = None,
        total_wells: list[str] | None = None,
        worker_assignments: dict[str, list[str]] | None = None,
        step_names: list[str] | None = None,
    ) -> None:
        event = ProgressEvent(
            identity=ProgressIdentity(
                execution_id=self.execution_id,
                plate_id=self.plate_id,
                axis_id=axis_id,
                step_name=step_name,
            ),
            phase=phase,
            status=status,
            percent=percent,
            completed=completed,
            total=total,
            timestamp=time.time(),
            pid=os.getpid(),
            error=error,
            message=message,
            total_wells=total_wells,
            worker_assignments=worker_assignments,
            step_names=step_names,
        )
        self.enqueue(event.to_dict())
