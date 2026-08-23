"""Execution state owned by the Plate Manager workflow."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

from openhcs.core.execution_state import (
    TerminalExecutionStatus,
    parse_terminal_status,
)


@dataclass(frozen=True, slots=True)
class ExecutionBatchMember:
    """All runtime facts owned for one plate in the current batch."""

    execution_id: str | None = None
    terminal_status: TerminalExecutionStatus | None = None

    @property
    def active(self) -> bool:
        return self.terminal_status is None

    def with_execution(self, execution_id: str) -> ExecutionBatchMember:
        return replace(self, execution_id=execution_id)

    def with_terminal_status(
        self,
        status: str | TerminalExecutionStatus,
    ) -> ExecutionBatchMember:
        return replace(self, terminal_status=parse_terminal_status(status))

    def without_execution(self) -> ExecutionBatchMember:
        return replace(self, execution_id=None)


class ExecutionBatchRuntime:
    """Sole mutable authority for one manager execution batch."""

    def __init__(self) -> None:
        self._members_by_plate: dict[str, ExecutionBatchMember] = {}

    def begin_batch(self, plate_paths: Iterable[str]) -> None:
        self._members_by_plate = {
            plate_path: ExecutionBatchMember()
            for plate_path in dict.fromkeys(str(path) for path in plate_paths)
        }

    def record_execution(self, plate_path: str, execution_id: str) -> None:
        """Associate the runtime identity returned for one batch member."""

        member = self._require_batch_member(plate_path)
        self._members_by_plate[plate_path] = member.with_execution(execution_id)

    def execution_id(self, plate_path: str) -> str | None:
        member = self._members_by_plate.get(plate_path)
        return None if member is None else member.execution_id

    def mark_terminal(
        self,
        plate_path: str,
        status: str | TerminalExecutionStatus,
    ) -> None:
        member = self._require_batch_member(plate_path)
        self._members_by_plate[plate_path] = member.with_terminal_status(status)

    def is_active(self, plate_path: str) -> bool:
        member = self._members_by_plate.get(plate_path)
        return member is not None and member.active

    @property
    def active_plates(self) -> tuple[str, ...]:
        """Derive active members from batch membership and terminal outcomes."""

        return tuple(
            plate_path
            for plate_path, member in self._members_by_plate.items()
            if member.active
        )

    def terminal_status(self, plate_path: str) -> TerminalExecutionStatus | None:
        member = self._members_by_plate.get(plate_path)
        return None if member is None else member.terminal_status

    def terminal_items(self) -> tuple[tuple[str, TerminalExecutionStatus], ...]:
        """Project terminal outcomes in stable batch order."""

        return tuple(
            (plate_path, member.terminal_status)
            for plate_path, member in self._members_by_plate.items()
            if member.terminal_status is not None
        )

    def retire_execution(self, plate_path: str) -> str | None:
        """Retire one runtime identity while preserving its batch outcome."""

        member = self._members_by_plate.get(plate_path)
        if member is None:
            return None
        self._members_by_plate[plate_path] = member.without_execution()
        return member.execution_id

    def remove_plate(self, plate_path: str) -> str | None:
        """Remove one plate and every execution fact owned for it."""

        member = self._members_by_plate.pop(plate_path, None)
        return None if member is None else member.execution_id

    def clear_batch(self) -> None:
        self._members_by_plate.clear()

    def all_batch_terminal(self) -> bool:
        return all(not member.active for member in self._members_by_plate.values())

    def terminal_counts(self) -> tuple[int, int]:
        statuses = tuple(status for _plate_path, status in self.terminal_items())
        completed = sum(
            1 for status in statuses if status is TerminalExecutionStatus.COMPLETE
        )
        failed = sum(1 for status in statuses if status.counts_as_failed)
        return completed, failed

    def cancellable_plates(self) -> tuple[str, ...]:
        return self.active_plates

    def _require_batch_member(self, plate_path: str) -> ExecutionBatchMember:
        try:
            return self._members_by_plate[plate_path]
        except KeyError:
            raise KeyError(
                f"Plate {plate_path!r} is not a member of the active batch"
            ) from None
