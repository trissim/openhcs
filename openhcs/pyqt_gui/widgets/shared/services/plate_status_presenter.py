"""Plate list status presenter for Plate Manager rows."""

from __future__ import annotations

from typing import Optional

from openhcs.core.orchestrator.orchestrator import OrchestratorState
from openhcs.core.debug_session_projection import (
    DebugSessionPhase,
    DebugSessionPhaseDeclarationBase,
)
from openhcs.core.progress.projection import PlateRuntimeProjection
from openhcs.core.execution_state import (
    TerminalExecutionStatus,
)


class PlateStatusPresenter:
    """Render deterministic plate status text from execution + orchestrator signals."""

    @classmethod
    def build_debug_status_prefix(
        cls,
        *,
        debug_phase: DebugSessionPhase,
    ) -> str:
        return DebugSessionPhaseDeclarationBase.for_phase(debug_phase).status_prefix

    @classmethod
    def build_status_prefix(
        cls,
        *,
        orchestrator_state: Optional[OrchestratorState],
        is_init_pending: bool,
        is_compile_pending: bool,
        is_execution_active: bool,
        terminal_status: Optional[TerminalExecutionStatus],
        runtime_projection: Optional[PlateRuntimeProjection],
    ) -> str:
        # Runtime projection is canonical whenever present.
        if runtime_projection is not None:
            return cls._format_runtime_plate_status(runtime_projection)

        if is_execution_active:
            return "⏳ Pending"

        if is_init_pending:
            return "⏳ Init"
        if is_compile_pending:
            return "⏳ Compile"

        if terminal_status is not None:
            return terminal_status.status_prefix

        if orchestrator_state is None:
            return ""
        return orchestrator_state.status_prefix

    @staticmethod
    def _format_runtime_plate_status(plate: PlateRuntimeProjection) -> str:
        return plate.formatted_status
