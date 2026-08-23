"""Shared execution state vocabulary for workflow status projection."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, fields
from enum import Enum
from typing import Any, ClassVar

from zmqruntime.messages import MessageFields

from openhcs.constants.constants import OrchestratorState


@dataclass(frozen=True, slots=True)
class ExecutionOutputPlateSummary:
    """Typed OpenHCS output-plate fields embedded in a runtime result summary."""

    EXECUTION_RECORD_KEY: ClassVar[str] = "output_plate_summary"

    output_plate_root: str | None = None
    auto_add_output_plate_to_plate_manager: bool | None = None

    def __post_init__(self) -> None:
        if self.output_plate_root is not None and not isinstance(
            self.output_plate_root,
            str,
        ):
            raise TypeError("output_plate_root must be a string or None")
        if self.auto_add_output_plate_to_plate_manager is not None and not isinstance(
            self.auto_add_output_plate_to_plate_manager,
            bool,
        ):
            raise TypeError(
                "auto_add_output_plate_to_plate_manager must be a bool or None"
            )

    @classmethod
    def from_results_summary(
        cls,
        summary: Mapping[str, Any],
    ) -> ExecutionOutputPlateSummary:
        return cls(
            **{
                declared_field.name: summary[declared_field.name]
                for declared_field in fields(cls)
                if declared_field.name in summary
            }
        )

    def results_summary_fields(self) -> dict[str, object]:
        """Project only declared output-plate values for wire enrichment."""

        return {key: value for key, value in asdict(self).items() if value is not None}

    @property
    def is_present(self) -> bool:
        return any(value is not None for value in asdict(self).values())


@dataclass(frozen=True, slots=True)
class ExecutionCompletionPayload:
    """One typed terminal result passed from polling through UI presentation."""

    status: TerminalExecutionStatus
    execution_id: str | None
    results: Mapping[str, Any]
    output_plate: ExecutionOutputPlateSummary
    traceback_text: str
    message: str

    @property
    def output_plate_root(self) -> str | None:
        return self.output_plate.output_plate_root

    @property
    def auto_add_output_plate_to_plate_manager(self) -> bool | None:
        return self.output_plate.auto_add_output_plate_to_plate_manager

    @classmethod
    def completed(
        cls,
        status: TerminalExecutionStatus,
        execution_id: str | None,
        execution_payload: Mapping[str, Any],
    ) -> ExecutionCompletionPayload:
        """Decode the successful runtime payload shape."""

        raw_results = execution_payload.get(MessageFields.RESULTS_SUMMARY, {})
        results = dict(raw_results) if isinstance(raw_results, Mapping) else {}
        return cls(
            status=status,
            execution_id=execution_id,
            results=results,
            output_plate=ExecutionOutputPlateSummary.from_results_summary(results),
            traceback_text="",
            message="Execution completed",
        )

    @classmethod
    def failed(
        cls,
        status: TerminalExecutionStatus,
        execution_id: str | None,
        execution_payload: Mapping[str, Any],
    ) -> ExecutionCompletionPayload:
        """Decode the failed runtime payload shape."""

        return cls(
            status=status,
            execution_id=execution_id,
            results={},
            output_plate=ExecutionOutputPlateSummary(),
            traceback_text=str(execution_payload.get(MessageFields.TRACEBACK, "")),
            message=str(execution_payload.get(MessageFields.ERROR, "Unknown error")),
        )

    @classmethod
    def cancelled(
        cls,
        status: TerminalExecutionStatus,
        execution_id: str | None,
        execution_payload: Mapping[str, Any],
    ) -> ExecutionCompletionPayload:
        """Build the local cancellation completion shape."""

        del execution_payload
        return cls(
            status=status,
            execution_id=execution_id,
            results={},
            output_plate=ExecutionOutputPlateSummary(),
            traceback_text="",
            message="Execution was cancelled",
        )


class ManagerExecutionState(str, Enum):
    """Plate-manager state with member-owned execution policies."""

    suppresses_stop_failure: bool
    stop_pending: bool
    busy: bool
    run_button_text: str
    allows_auto_add_output: bool
    implicit_stop_force: bool | None

    IDLE = (
        "idle",
        True,
        False,
        False,
        "Run",
        lambda compiled: compiled,
        False,
        None,
    )
    RUNNING = (
        "running",
        False,
        False,
        True,
        "Stop",
        lambda _compiled: True,
        True,
        False,
    )
    STOPPING = (
        "stopping",
        True,
        True,
        True,
        "Stop",
        lambda _compiled: False,
        False,
        None,
    )
    FORCE_KILL_READY = (
        "force_kill_ready",
        True,
        True,
        True,
        "Force Kill",
        lambda _compiled: True,
        False,
        True,
    )

    def __new__(
        cls,
        value: str,
        suppresses_stop_failure: bool,
        stop_pending: bool,
        busy: bool,
        run_button_text: str,
        run_button_enabled: Callable[[bool], bool],
        allows_auto_add_output: bool,
        implicit_stop_force: bool | None,
    ) -> ManagerExecutionState:
        member = str.__new__(cls, value)
        member._value_ = value
        member.suppresses_stop_failure = suppresses_stop_failure
        member.stop_pending = stop_pending
        member.busy = busy
        member.run_button_text = run_button_text
        member._run_button_enabled = run_button_enabled
        member.allows_auto_add_output = allows_auto_add_output
        member.implicit_stop_force = implicit_stop_force
        return member

    def run_button_enabled(self, has_compiled_plate: bool) -> bool:
        """Execute this state's run/stop control availability leaf."""

        return self._run_button_enabled(has_compiled_plate)

    def stop_request(
        self,
        force: bool | None = None,
    ) -> tuple[ManagerExecutionState, bool]:
        """Resolve one stop command and its resulting manager state."""

        requested_force = self.implicit_stop_force if force is None else force
        if requested_force is None:
            raise RuntimeError(f"Execution state {self.value!r} does not accept Stop")
        next_state = (
            ManagerExecutionState.STOPPING
            if requested_force
            else ManagerExecutionState.FORCE_KILL_READY
        )
        return next_state, requested_force


class TerminalExecutionStatus(str, Enum):
    """Terminal workflow status shared by UI, MCP, and dev-client polling."""

    counts_as_failed: bool
    orchestrator_state: OrchestratorState
    status_prefix: str
    emit_failure: bool
    auto_add_output_plate: bool
    serialized_aliases: frozenset[str]

    COMPLETE = (
        "complete",
        False,
        OrchestratorState.COMPLETED,
        "✅ Complete",
        False,
        True,
        frozenset(),
        ExecutionCompletionPayload.completed,
    )
    FAILED = (
        "failed",
        True,
        OrchestratorState.EXEC_FAILED,
        "❌ Exec Failed",
        True,
        False,
        frozenset({"error"}),
        ExecutionCompletionPayload.failed,
    )
    CANCELLED = (
        "cancelled",
        True,
        OrchestratorState.READY,
        "✗ Cancelled",
        False,
        False,
        frozenset(),
        ExecutionCompletionPayload.cancelled,
    )

    def __new__(
        cls,
        value: str,
        counts_as_failed: bool,
        orchestrator_state: OrchestratorState,
        status_prefix: str,
        emit_failure: bool,
        auto_add_output_plate: bool,
        serialized_aliases: frozenset[str],
        completion_factory: Callable[
            [TerminalExecutionStatus, str | None, Mapping[str, Any]],
            ExecutionCompletionPayload,
        ],
    ):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.counts_as_failed = counts_as_failed
        obj.orchestrator_state = orchestrator_state
        obj.status_prefix = status_prefix
        obj.emit_failure = emit_failure
        obj.auto_add_output_plate = auto_add_output_plate
        obj.serialized_aliases = serialized_aliases
        obj._completion_factory = completion_factory
        return obj

    @classmethod
    def _missing_(cls, value: object) -> TerminalExecutionStatus | None:
        return next(
            (
                status
                for status in cls
                if isinstance(value, str) and value in status.serialized_aliases
            ),
            None,
        )

    def completion_payload(
        self,
        *,
        execution_id: str | None,
        execution_payload: Mapping[str, Any],
    ) -> ExecutionCompletionPayload:
        """Decode this member's runtime payload into the canonical terminal result."""

        return self._completion_factory(self, execution_id, execution_payload)


def parse_terminal_status(
    status: str | TerminalExecutionStatus,
) -> TerminalExecutionStatus:
    """Return the canonical terminal status for a UI/runtime status value."""
    if isinstance(status, TerminalExecutionStatus):
        return status
    return TerminalExecutionStatus(status)
