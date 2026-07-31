"""Shared execution state vocabulary for workflow status projection."""

from __future__ import annotations

from enum import Enum

from openhcs.constants.constants import OrchestratorState


class BooleanPolicyStringEnum(str, Enum):
    """String enum carrying one boolean policy attribute."""

    def __new__(cls, value: str, policy_value: bool):
        obj = str.__new__(cls, value)
        obj._value_ = value
        setattr(obj, cls.__policy_attribute_name__, policy_value)
        return obj


class ManagerExecutionState(BooleanPolicyStringEnum):
    """Plate-manager execution state shared by UI and agent projections."""

    __policy_attribute_name__ = "suppresses_stop_failure"

    IDLE = ("idle", True)
    RUNNING = ("running", False)
    STOPPING = ("stopping", True)
    FORCE_KILL_READY = ("force_kill_ready", True)


class TerminalExecutionStatus(str, Enum):
    """Terminal workflow status shared by UI, MCP, and dev-client polling."""

    counts_as_failed: bool
    orchestrator_state: OrchestratorState
    status_prefix: str
    emit_failure: bool
    auto_add_output_plate: bool

    COMPLETE = (
        "complete",
        False,
        OrchestratorState.COMPLETED,
        "✅ Complete",
        False,
        True,
    )
    FAILED = (
        "failed",
        True,
        OrchestratorState.EXEC_FAILED,
        "❌ Exec Failed",
        True,
        False,
    )
    CANCELLED = (
        "cancelled",
        True,
        OrchestratorState.READY,
        "✗ Cancelled",
        False,
        False,
    )

    def __new__(
        cls,
        value: str,
        counts_as_failed: bool,
        orchestrator_state: OrchestratorState,
        status_prefix: str,
        emit_failure: bool,
        auto_add_output_plate: bool,
    ):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.counts_as_failed = counts_as_failed
        obj.orchestrator_state = orchestrator_state
        obj.status_prefix = status_prefix
        obj.emit_failure = emit_failure
        obj.auto_add_output_plate = auto_add_output_plate
        return obj


STOP_PENDING_MANAGER_STATES = frozenset(
    {
        ManagerExecutionState.STOPPING,
        ManagerExecutionState.FORCE_KILL_READY,
    }
)
BUSY_MANAGER_STATES = frozenset(
    {
        ManagerExecutionState.RUNNING,
        ManagerExecutionState.STOPPING,
        ManagerExecutionState.FORCE_KILL_READY,
    }
)


_TERMINAL_STATUS_ALIASES: dict[str, TerminalExecutionStatus] = {
    "error": TerminalExecutionStatus.FAILED,
}


def parse_terminal_status(
    status: str | TerminalExecutionStatus,
) -> TerminalExecutionStatus:
    """Return the canonical terminal status for a UI/runtime status value."""
    if isinstance(status, TerminalExecutionStatus):
        return status
    alias = _TERMINAL_STATUS_ALIASES.get(status)
    if alias is not None:
        return alias
    return TerminalExecutionStatus(status)
