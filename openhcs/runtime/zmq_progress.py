"""ZMQ progress event construction for execution-server phases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import os
import time

from openhcs.core.progress import ProgressEvent, ProgressPhase, ProgressStatus


@dataclass(frozen=True, slots=True)
class ZMQProgressEmitter:
    """Semantic progress-event emitter for one ZMQ execution."""

    enqueue: Callable[[dict], None]
    execution_id: str
    plate_id: str

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

    def emit(self, **kwargs) -> None:
        event = ProgressEvent(
            timestamp=time.time(),
            pid=os.getpid(),
            execution_id=self.execution_id,
            plate_id=self.plate_id,
            **kwargs,
        )
        self.enqueue(event.to_dict())

