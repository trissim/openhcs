"""Shared PlateManager state projection for UI rendering and agent bridge polling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from openhcs.agent.dto.ui_bridge import (
    UiPlateManagerRowState,
    UiPlateManagerState,
    UiStateSurfaceSummary,
)
from openhcs.config_framework.object_state import ObjectStateRegistry
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.core.progress.projection import PlateRuntimeProjection
from openhcs.core.selection import SelectedAllSelectionMode
from openhcs.pyqt_gui.services.plate_manager_row import PlateManagerRow
from openhcs.pyqt_gui.widgets.shared.services.execution_state import (
    TerminalExecutionStatus,
)
from openhcs.pyqt_gui.widgets.shared.services.plate_status_presenter import (
    PlateStatusPresenter,
)

if TYPE_CHECKING:
    from openhcs.pyqt_gui.widgets.plate_manager import PlateManagerWidget


@dataclass(frozen=True, slots=True)
class PlateManagerStateSelectionAuthority:
    """Closed selection-mode table for PlateManager state projection."""

    all_rows: tuple[PlateManagerRow, ...]
    selected_rows: tuple[PlateManagerRow, ...]

    def rows_for_mode(self, selection_mode: str) -> tuple[PlateManagerRow, ...]:
        mode = SelectedAllSelectionMode(selection_mode)
        rows_by_mode = {
            SelectedAllSelectionMode.ALL: self.all_rows,
            SelectedAllSelectionMode.SELECTED: self.selected_rows,
        }
        return rows_by_mode[mode]


class PlateManagerStateProjectionService:
    """Build the single PlateManager state projection used by UI and bridge code."""

    def project(
        self,
        manager: "PlateManagerWidget",
        *,
        schema_version: str,
        summary: UiStateSurfaceSummary,
        selection_mode: str,
    ) -> UiPlateManagerState:
        all_rows = tuple(manager.plates)
        selected_rows = tuple(manager.get_selected_items())
        selected_scope_ids = tuple(row.scope_id for row in selected_rows)
        selected_scope_set = set(selected_scope_ids)
        rows = PlateManagerStateSelectionAuthority(
            all_rows=all_rows,
            selected_rows=selected_rows,
        ).rows_for_mode(selection_mode)
        return UiPlateManagerState(
            schema_version=schema_version,
            summary=summary,
            selection_mode=selection_mode,
            rows=tuple(
                self.project_row(
                    manager,
                    row,
                    selected_scope_ids=selected_scope_set,
                )
                for row in rows
            ),
            selected_scope_ids=selected_scope_ids,
            manager_execution_state=manager.execution_state.value,
            object_state_token=ObjectStateRegistry.get_token(),
        )

    def project_row(
        self,
        manager: "PlateManagerWidget",
        row: PlateManagerRow,
        *,
        selected_scope_ids: set[str],
    ) -> UiPlateManagerRowState:
        plate_key = row.scope_id
        orchestrator = ObjectStateRegistry.get_object(plate_key)
        orchestrator_state = None
        initialized = False
        if isinstance(orchestrator, PipelineOrchestrator):
            orchestrator_state = orchestrator.state
            initialized = orchestrator_state.has_completed_initialization

        execution_id = manager.plate_execution_ids.get(plate_key)
        runtime_projection = manager.runtime_progress_projection.get_plate(
            plate_id=plate_key,
            execution_id=execution_id,
        )
        terminal_status = manager.plate_terminal_activity_status.terminal_status(plate_key)
        execution_active = (
            manager.plate_terminal_activity_status.is_active(plate_key)
            or runtime_projection is not None
        )
        queue_position = self._queued_execution_position_for_plate(manager, plate_key)
        status_prefix = PlateStatusPresenter.build_status_prefix(
            orchestrator_state=orchestrator_state,
            is_init_pending=plate_key in manager.plate_init_pending,
            is_compile_pending=plate_key in manager.plate_compile_pending,
            is_execution_active=execution_active,
            terminal_status=terminal_status,
            queue_position=queue_position,
            runtime_projection=runtime_projection,
        )

        return UiPlateManagerRowState(
            plate_scope_id=plate_key,
            name=row.name,
            plate_root=row.plate_root,
            cppipe_path=row.cppipe_path,
            selected=plate_key in selected_scope_ids,
            initialized=initialized,
            compiled=plate_key in manager.plate_compiled_data,
            init_pending=plate_key in manager.plate_init_pending,
            compile_pending=plate_key in manager.plate_compile_pending,
            execution_active=execution_active,
            status_prefix=status_prefix,
            orchestrator_state=self._orchestrator_state_value(orchestrator_state),
            execution_id=execution_id,
            terminal_status=self._terminal_status_value(terminal_status),
            runtime_state=self._runtime_state_value(runtime_projection),
            runtime_percent=self._runtime_percent(runtime_projection),
            queue_position=queue_position,
        )

    @staticmethod
    def _queued_execution_position_for_plate(
        manager: "PlateManagerWidget",
        plate_id: str,
    ) -> int | None:
        server_info = manager.execution_server_info
        if server_info is None:
            return None
        for queued in server_info.queued_execution_entries:
            if queued.plate_id == plate_id:
                return queued.queue_position
        return None

    @staticmethod
    def _orchestrator_state_value(orchestrator_state) -> str | None:
        if orchestrator_state is None:
            return None
        return orchestrator_state.value

    @staticmethod
    def _terminal_status_value(
        terminal_status: TerminalExecutionStatus | None,
    ) -> str | None:
        if terminal_status is None:
            return None
        return terminal_status.value

    @staticmethod
    def _runtime_state_value(
        runtime_projection: PlateRuntimeProjection | None,
    ) -> str | None:
        if runtime_projection is None:
            return None
        return runtime_projection.state.value

    @staticmethod
    def _runtime_percent(
        runtime_projection: PlateRuntimeProjection | None,
    ) -> float | None:
        if runtime_projection is None:
            return None
        return runtime_projection.percent
