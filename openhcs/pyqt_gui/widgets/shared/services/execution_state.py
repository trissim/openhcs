"""Execution state enums for Plate Manager workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from openhcs.core.execution_state import (
    TerminalExecutionStatus,
    parse_terminal_status,
)


@dataclass
class ExecutionBatchRuntime:
    """Tracks current run-attempt terminal outcomes and active plates."""

    batch_plates: tuple[str, ...] = ()
    active_plates: set[str] = field(default_factory=set)
    terminal_status_by_plate: dict[str, TerminalExecutionStatus] = field(
        default_factory=dict
    )

    def begin_batch(self, plate_paths: Iterable[str]) -> None:
        ordered = tuple(dict.fromkeys(str(path) for path in plate_paths))
        self.batch_plates = ordered
        self.active_plates = set(ordered)
        self.terminal_status_by_plate.clear()

    def mark_terminal(
        self, plate_path: str, status: str | TerminalExecutionStatus
    ) -> None:
        terminal_status = parse_terminal_status(status)
        self.terminal_status_by_plate[plate_path] = terminal_status
        self.active_plates.discard(plate_path)

    def is_active(self, plate_path: str) -> bool:
        return plate_path in self.active_plates

    def terminal_status(self, plate_path: str) -> TerminalExecutionStatus | None:
        return self.terminal_status_by_plate.get(plate_path)

    def clear_plate(self, plate_path: str, *, clear_terminal: bool = True) -> None:
        self.active_plates.discard(plate_path)
        if clear_terminal:
            self.terminal_status_by_plate.pop(plate_path, None)

    def clear_batch(self) -> None:
        self.batch_plates = ()
        self.active_plates.clear()
        self.terminal_status_by_plate.clear()

    def all_batch_terminal(self) -> bool:
        if not self.batch_plates:
            return True
        return all(
            plate_path in self.terminal_status_by_plate
            for plate_path in self.batch_plates
        )

    def terminal_counts(self) -> tuple[int, int]:
        statuses = [
            self.terminal_status_by_plate[plate_path]
            for plate_path in self.batch_plates
            if plate_path in self.terminal_status_by_plate
        ]
        completed = sum(
            1 for status in statuses if status == TerminalExecutionStatus.COMPLETE
        )
        failed = sum(1 for status in statuses if status.counts_as_failed)
        return completed, failed

    def cancellable_plates(self) -> tuple[str, ...]:
        return tuple(
            plate_path
            for plate_path in self.batch_plates
            if plate_path in self.active_plates
        )
