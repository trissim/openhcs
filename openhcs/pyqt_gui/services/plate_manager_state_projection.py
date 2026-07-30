"""Shared PlateManager state projection for UI rendering and agent bridge polling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from openhcs.agent.dto.ui_bridge import (
    UiPlateManagerRowState,
    UiPlateManagerState,
    UiStateSurfaceSummary,
)
from objectstate.object_state import ObjectStateRegistry
from openhcs.core.config import PathPlanningConfig
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.core.pipeline.path_planner import PipelinePathPlanner
from openhcs.core.progress.projection import PlateRuntimeProjection
from openhcs.core.selection import SelectedAllSelectionMode
from openhcs.pyqt_gui.services.plate_manager_row import PlateManagerRow
from openhcs.pyqt_gui.widgets.shared.services.execution_state import (
    TerminalExecutionStatus,
    terminal_ui_policy,
)
from openhcs.pyqt_gui.widgets.shared.services.plate_status_presenter import (
    PlateStatusPresenter,
)
from openhcs.pyqt_gui.widgets.shared.services.debug_session_projection import (
    DebugToolbarActionProjector,
)
from pyqt_reactive.services.scope_color_service import ScopeColorService

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


@dataclass(frozen=True, slots=True)
class PlateManagerOutputPlateRelation:
    """Input/output plate relationship projected from path-planning config."""

    output_plate_scope_id: str | None = None
    output_plate_root: str | None = None
    source_plate_scope_id: str | None = None
    source_plate_root: str | None = None


@dataclass(frozen=True, slots=True)
class PlateManagerOutputPlateRelationAuthority:
    """Build source/output plate relationships from visible PlateManager rows."""

    relations: dict[str, PlateManagerOutputPlateRelation]

    @classmethod
    def from_rows(
        cls,
        rows: tuple[PlateManagerRow, ...],
        default_path_config: PathPlanningConfig,
    ) -> "PlateManagerOutputPlateRelationAuthority":
        row_by_root = {row.plate_root: row for row in rows}
        output_root_by_source: dict[str, str] = {}
        source_by_output_root: dict[str, PlateManagerRow] = {}
        for row in rows:
            output_root = cls._output_plate_root(
                row,
                cls._path_config_for_row(row, default_path_config),
            )
            if output_root is None or output_root == row.plate_root:
                continue
            output_root_by_source[row.scope_id] = output_root
            if output_root in row_by_root:
                source_by_output_root[output_root] = row

        relations: dict[str, PlateManagerOutputPlateRelation] = {}
        for row in rows:
            source_row = source_by_output_root.get(row.plate_root)
            if source_row is not None:
                relations[row.scope_id] = PlateManagerOutputPlateRelation(
                    source_plate_scope_id=source_row.scope_id,
                    source_plate_root=source_row.plate_root,
                )
                continue

            output_root = output_root_by_source.get(row.scope_id)
            if output_root is None:
                continue
            output_row = row_by_root.get(output_root)
            relations[row.scope_id] = PlateManagerOutputPlateRelation(
                output_plate_scope_id=(
                    output_root if output_row is None else output_row.scope_id
                ),
                output_plate_root=(
                    output_root if output_row is None else output_row.plate_root
                ),
            )
        return cls(relations=relations)

    def relation_for(self, row: PlateManagerRow) -> PlateManagerOutputPlateRelation:
        return self.relations.get(row.scope_id, PlateManagerOutputPlateRelation())

    @staticmethod
    def _output_plate_root(
        row: PlateManagerRow,
        path_config: PathPlanningConfig,
    ) -> str | None:
        return str(
            PipelinePathPlanner.build_output_plate_root(
                Path(row.plate_root),
                path_config,
            )
        )

    @staticmethod
    def _path_config_for_row(
        row: PlateManagerRow,
        default_path_config: PathPlanningConfig,
    ) -> PathPlanningConfig:
        orchestrator = ObjectStateRegistry.get_object(row.scope_id)
        if isinstance(orchestrator, PipelineOrchestrator):
            return orchestrator.get_effective_config().path_planning_config
        return default_path_config


class PlateManagerStateProjectionService:
    """Build the single PlateManager state projection used by UI and bridge code."""

    def output_relation_for(
        self,
        manager: "PlateManagerWidget",
        row: PlateManagerRow,
    ) -> PlateManagerOutputPlateRelation:
        """Return the source/output relation for one visible PlateManager row."""
        return PlateManagerOutputPlateRelationAuthority.from_rows(
            tuple(manager.plates),
            manager.global_config.path_planning_config,
        ).relation_for(row)

    @staticmethod
    def scope_accent_color(plate_scope_id: str) -> str:
        """Project the current UI-owned scope accent as an exact Qt color name."""

        return (
            ScopeColorService.instance()
            .get_accent_color(plate_scope_id)
            .name()
            .lower()
        )

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
        output_relations = PlateManagerOutputPlateRelationAuthority.from_rows(
            all_rows,
            manager.global_config.path_planning_config,
        )
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
                    output_relation=output_relations.relation_for(row),
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
        output_relation: PlateManagerOutputPlateRelation,
    ) -> UiPlateManagerRowState:
        plate_key = row.scope_id
        orchestrator = ObjectStateRegistry.get_object(plate_key)
        orchestrator_state = None
        initialized = False
        if isinstance(orchestrator, PipelineOrchestrator):
            orchestrator_state = orchestrator.state
            initialized = orchestrator_state.has_completed_initialization

        execution_id = manager.plate_execution_ids.get(plate_key)
        runtime_projection = None
        if execution_id is not None:
            runtime_projection = manager.runtime_progress_projection.get_plate(
                plate_id=plate_key,
                execution_id=execution_id,
            )
        terminal_status = manager.plate_terminal_activity_status.terminal_status(plate_key)
        effective_orchestrator_state = self._effective_orchestrator_state(
            orchestrator_state,
            terminal_status,
        )
        status_runtime_projection = self._status_runtime_projection(
            runtime_projection,
            terminal_status,
        )
        execution_active = (
            terminal_status is None
            and (
                manager.plate_terminal_activity_status.is_active(plate_key)
                or self._is_active_runtime_projection(runtime_projection)
            )
        )
        queue_position = self._queued_execution_position_for_plate(manager, plate_key)
        status_prefix = PlateStatusPresenter.build_status_prefix(
            orchestrator_state=effective_orchestrator_state,
            is_init_pending=plate_key in manager.plate_init_pending,
            is_compile_pending=plate_key in manager.plate_compile_pending,
            is_execution_active=execution_active,
            terminal_status=terminal_status,
            queue_position=queue_position,
            runtime_projection=status_runtime_projection,
        )
        debug_context = manager.debug_session_context_for_plate(plate_key)
        debug_session = manager.debug_session_for_plate(plate_key)
        terminal_summary = manager.debug_terminal_summary_for_plate(plate_key)
        debug_phase = None
        debug_session_id = None
        if debug_session is not None or terminal_summary is not None:
            projected_debug_phase = DebugToolbarActionProjector.phase(debug_context)
            debug_prefix = PlateStatusPresenter.build_debug_status_prefix(
                debug_phase=projected_debug_phase,
            )
            if debug_prefix:
                status_prefix = debug_prefix
            debug_phase = projected_debug_phase.value
        if debug_session is not None:
            debug_session_id = debug_session.debug_session_id
        elif terminal_summary is not None:
            debug_session_id = terminal_summary.debug_session_id

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
            orchestrator_state=self._orchestrator_state_value(effective_orchestrator_state),
            execution_id=execution_id,
            terminal_status=self._terminal_status_value(terminal_status),
            runtime_state=self._runtime_state_value(status_runtime_projection),
            runtime_percent=self._runtime_percent(status_runtime_projection),
            queue_position=queue_position,
            output_plate_scope_id=output_relation.output_plate_scope_id,
            output_plate_root=output_relation.output_plate_root,
            source_plate_scope_id=output_relation.source_plate_scope_id,
            source_plate_root=output_relation.source_plate_root,
            debug_phase=debug_phase,
            debug_session_id=debug_session_id,
            scope_accent_color=self.scope_accent_color(plate_key),
        )

    @staticmethod
    def _status_runtime_projection(
        runtime_projection: PlateRuntimeProjection | None,
        terminal_status: TerminalExecutionStatus | None,
    ) -> PlateRuntimeProjection | None:
        if terminal_status is not None:
            return None
        return runtime_projection

    @staticmethod
    def _is_active_runtime_projection(
        runtime_projection: PlateRuntimeProjection | None,
    ) -> bool:
        if runtime_projection is None:
            return False
        return not runtime_projection.is_terminal

    @staticmethod
    def _effective_orchestrator_state(
        orchestrator_state,
        terminal_status: TerminalExecutionStatus | None,
    ):
        if terminal_status is None:
            return orchestrator_state
        return terminal_ui_policy(terminal_status).orchestrator_state

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
