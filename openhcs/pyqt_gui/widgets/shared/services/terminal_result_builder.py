"""Terminal execution result records for PyQt batch workflow notifications."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from openhcs.core.execution_state import (
    TerminalExecutionStatus,
    parse_terminal_status,
)


class TerminalExecutionResultBuilder:
    """Builds host-facing result payloads from terminal server status."""

    def build(
        self,
        *,
        terminal_status: str,
        execution_id: str,
        execution_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        status = parse_terminal_status(terminal_status)
        return TERMINAL_RESULT_BUILDERS[status](execution_id, execution_payload)


def build_complete_terminal_result(
    execution_id: str,
    execution_payload: Dict[str, Any],
) -> Dict[str, Any]:
    results_summary = execution_payload.get("results_summary", {}) or {}
    output_plate_root = None
    auto_add_output_plate = None
    if isinstance(results_summary, Mapping):
        output_plate_root = results_summary.get("output_plate_root")
        auto_add_output_plate = results_summary.get(
            "auto_add_output_plate_to_plate_manager"
        )
    return {
        "status": TerminalExecutionStatus.COMPLETE.value,
        "execution_id": execution_id,
        "results": results_summary,
        "output_plate_root": output_plate_root,
        "auto_add_output_plate_to_plate_manager": auto_add_output_plate,
    }


def build_failed_terminal_result(
    execution_id: str,
    execution_payload: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "status": TerminalExecutionStatus.FAILED.value,
        "execution_id": execution_id,
        "message": execution_payload.get("error", "Unknown error"),
        "traceback": execution_payload.get("traceback", ""),
    }


def build_cancelled_terminal_result(
    execution_id: str,
    _execution_payload: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "status": TerminalExecutionStatus.CANCELLED.value,
        "execution_id": execution_id,
        "message": "Execution was cancelled",
    }


TERMINAL_RESULT_BUILDERS = {
    TerminalExecutionStatus.COMPLETE: build_complete_terminal_result,
    TerminalExecutionStatus.FAILED: build_failed_terminal_result,
    TerminalExecutionStatus.CANCELLED: build_cancelled_terminal_result,
}


def is_terminal_result_builder_export(name: str, value: object) -> bool:
    return (
        isinstance(value, type)
        and value.__module__ == __name__
        and not name.startswith("_")
    )


__all__ = tuple(
    name
    for name, value in globals().items()
    if is_terminal_result_builder_export(name, value)
)
