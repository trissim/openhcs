"""PlateManager provider set for the PyQt UI bridge."""

from __future__ import annotations

import hashlib
from dataclasses import replace

from openhcs.agent.dto.common import AgentError, AgentWarning, SCHEMA_VERSION
from openhcs.agent.dto.ui_bridge import (
    UiActionCatalog,
    UiActionIdentity,
    UiActionInvocationStatus,
    UiActionInvokeRequest,
    UiActionInvokeResult,
    UiActionSummary,
    UiCodeDocument,
    UiCodeDocumentApplyRequest,
    UiCodeDocumentApplyResult,
    UiCodeDocumentRequest,
    UiCodeDocumentSelectionMode,
    UiCodeDocumentSummary,
    UiCodeDocumentValidationRequest,
    UiCodeDocumentValidationResult,
    UiLiveMeasurementEntryState,
    UiLiveMeasurementsState,
    UiPlateManagerRowState,
    UiPlateManagerState,
    UiLiveOverviewItem,
    UiLiveOverviewMetric,
    UiLiveOverviewSection,
    UiLiveOverviewSeverity,
    UiStateSurfaceDocument,
    UiStateSurfaceRequest,
    UiStateSurfaceSummary,
    UiMutationReceipt,
)
from openhcs.agent.ui_bridge_actions import PlateManagerAction
from openhcs.agent.ui_bridge_identities import (
    PlateManagerLiveMeasurementsStateSurfaceIdentityDeclaration,
    PlateManagerOrchestratorCodeDocumentIdentity,
    PlateManagerStateSurfaceIdentityDeclaration,
    PlateManagerWidgetIdentity,
)
from openhcs.serialization.json import to_jsonable
from objectstate.object_state import ObjectStateRegistry
from openhcs.core.selection import SelectedAllSelectionMode
from openhcs.pyqt_gui.widgets.shared.services.plate_manager_workflows import (
    PlateManagerCodeMutationScope,
)
from openhcs.pyqt_gui.services.plate_manager_state_projection import (
    PlateManagerStateProjectionService,
)
from openhcs.pyqt_gui.services.ui_agent_bridge import (
    UiCodeDocumentApplyLabel,
    UiCodeDocumentExecutionService,
    UiCodeDocumentValidationError,
)
from openhcs.pyqt_gui.services.ui_bridge_contracts import (
    APPLY_TIME_TRAVEL_OPT_IN_GUARD,
    CONFIRMATION_REQUIRED_GUARD,
    STALE_CODE_DOCUMENT_REVISION_ERROR,
    UiBridgeGuardPolicy,
    UiActionProviderABC,
    UiActionProviderIdentity,
    UiBridgeSnapshotProviderABC,
    UiCodeDocumentProviderIdentity,
    UiOwnedStateSurfaceDeclaration,
    UiStateSurfaceProviderABC,
    UiStateSurfaceProviderIdentity,
    SnapshotBackedUiCodeDocumentProviderABC,
    state_surface_declaration_for_identity,
    state_surface_ids_for_action,
)
from openhcs.pyqt_gui.services.ui_bridge_registry import (
    UiBridgeProviderSetABC,
    UiBridgeRegistrationContext,
)
from openhcs.pyqt_gui.widgets.plate_manager import (
    EmptyPlateSelectionPolicy,
    PlateManagerWidget,
    PlateOperationValidator,
)
from openhcs.core.execution_state import (
    TerminalExecutionStatus,
)
from openhcs.pyqt_gui.widgets.shared.services.widget_action_dispatch import (
    dispatch_widget_action,
)
from openhcs.pyqt_gui.widgets.shared.services.qt_widget_edit_commit import (
    commit_focused_widget_edits,
)

ORCHESTRATOR_DOCUMENT_TITLE = "Plate manager orchestrator config"
PLATE_MANAGER_ACTIONS_TITLE = "Plate manager actions"
PYTHON_MIME_TYPE = "text/x-python"

PLATE_MANAGER_ORCHESTRATOR_IDENTITY = UiCodeDocumentProviderIdentity.from_declaration(
    PlateManagerOrchestratorCodeDocumentIdentity,
    title=ORCHESTRATOR_DOCUMENT_TITLE,
)
PLATE_MANAGER_STATE_DECLARATION: UiOwnedStateSurfaceDeclaration = (
    state_surface_declaration_for_identity(
        PlateManagerWidget.UI_STATE_SURFACE_DECLARATIONS,
        PlateManagerStateSurfaceIdentityDeclaration,
    )
)
PLATE_MANAGER_STATE_IDENTITY = UiStateSurfaceProviderIdentity.from_owner(
    PLATE_MANAGER_STATE_DECLARATION,
    widget_declaration=PlateManagerWidget.UI_BRIDGE_WIDGET_IDENTITY,
)
LIVE_MEASUREMENTS_STATE_DECLARATION: UiOwnedStateSurfaceDeclaration = (
    state_surface_declaration_for_identity(
        PlateManagerWidget.UI_STATE_SURFACE_DECLARATIONS,
        PlateManagerLiveMeasurementsStateSurfaceIdentityDeclaration,
    )
)
LIVE_MEASUREMENTS_STATE_IDENTITY = UiStateSurfaceProviderIdentity.from_owner(
    LIVE_MEASUREMENTS_STATE_DECLARATION,
    widget_declaration=PlateManagerWidget.UI_BRIDGE_WIDGET_IDENTITY,
)
PLATE_MANAGER_ACTION_PROVIDER_IDENTITY = (
    UiActionProviderIdentity.from_widget_declaration(
        PlateManagerWidgetIdentity,
        title=PLATE_MANAGER_ACTIONS_TITLE,
    )
)
PLATE_MANAGER_STATE_SURFACE_ID = (
    PLATE_MANAGER_STATE_DECLARATION.surface_id
)
PLATE_MANAGER_ACTION_STATE_SURFACES = (PLATE_MANAGER_STATE_SURFACE_ID,)
PLATE_MANAGER_CODE_DOCUMENT_ID = (
    PlateManagerOrchestratorCodeDocumentIdentity.require_value()
)
PLATE_PATH_CODE_DOCUMENT_HINT = (
    "For autonomous path-based plate setup, read "
    f"{PLATE_MANAGER_CODE_DOCUMENT_ID!r} with openhcs_ui_get_code_document"
    "(selection_mode='all'), then apply source containing plate_paths and "
    "pipeline_data via openhcs_ui_apply_code_document. The add_plate UI action "
    "opens a GUI file dialog."
)
PLATE_SELECTION_REQUIRED_HINT = (
    f"Use openhcs_ui_get_state_surface(surface_id={PLATE_MANAGER_STATE_SURFACE_ID!r}) "
    "to inspect "
    "available rows and selected_scope_ids. If no rows are listed, add a plate "
    f"first. {PLATE_PATH_CODE_DOCUMENT_HINT}"
)
PLATE_ACTION_DISABLED_HINT = (
    f"Inspect openhcs_ui_list_actions and {PLATE_MANAGER_STATE_SURFACE_ID} for the current "
    "selection and workflow preconditions before invoking this action."
)


class PlateManagerBridgeProviderSet(UiBridgeProviderSetABC):
    """Register all PlateManager surfaces with a UI bridge registry."""

    registry_key = PlateManagerWidgetIdentity.require_value()

    def __init__(self, manager) -> None:
        self._manager = manager

    @classmethod
    def for_main_window(cls, main_window) -> "PlateManagerBridgeProviderSet":
        return cls(main_window.plate_manager_widget)

    def register(self, context: UiBridgeRegistrationContext) -> None:
        context.registry.register_code_document_provider(
            PlateManagerOrchestratorCodeDocumentProvider(
                self._manager,
                snapshot_provider=context.snapshot_provider,
            )
        )
        context.registry.register_state_surface_provider(
            PlateManagerStateSurfaceProvider(
                self._manager,
                snapshot_provider=context.snapshot_provider,
            )
        )
        context.registry.register_state_surface_provider(
            LiveMeasurementsStateSurfaceProvider(
                self._manager,
                snapshot_provider=context.snapshot_provider,
            )
        )
        context.registry.register_action_provider(
            PlateManagerActionProvider(
                self._manager,
            )
        )


def bind_snapshot_backed_provider(
    provider,
    manager,
    *,
    snapshot_provider: UiBridgeSnapshotProviderABC,
) -> None:
    """Bind shared state for providers that publish snapshot-backed data."""
    provider._manager = manager
    provider._snapshot_provider = snapshot_provider


class PlateManagerOrchestratorCodeDocumentProvider(
    SnapshotBackedUiCodeDocumentProviderABC,
):
    """Plate-manager orchestrator config provider backed by code mode."""

    identity = PLATE_MANAGER_ORCHESTRATOR_IDENTITY

    def __init__(
        self,
        manager,
        *,
        snapshot_provider: UiBridgeSnapshotProviderABC,
        execution_service: UiCodeDocumentExecutionService | None = None,
    ) -> None:
        bind_snapshot_backed_provider(
            self,
            manager,
            snapshot_provider=snapshot_provider,
        )
        self._execution_service = execution_service or UiCodeDocumentExecutionService()

    def summary(self) -> UiCodeDocumentSummary:
        selected_count = len(self._manager.get_selected_items())
        total_count = len(self._manager.plates)
        return UiCodeDocumentSummary(
            schema_version=SCHEMA_VERSION,
            identity=self.identity.as_document_identity(),
            title=self.identity.title,
            widget_id=self.identity.widget_id,
            readable=True,
            writable=True,
            supported_selection_modes=("selected", "all"),
            current_selection_count=selected_count,
            total_scope_count=total_count,
        )

    def read(self, request: UiCodeDocumentRequest) -> UiCodeDocument:
        selection_mode = request.resolved_selection_mode(
            UiCodeDocumentSelectionMode.SELECTED
        )
        if (
            selection_mode == UiCodeDocumentSelectionMode.SELECTED.value
            and not self._manager.get_selected_items()
        ):
            return self._document_error(
                request,
                (
                    AgentError(
                        code="no_selection",
                        message="No PlateManager rows are selected.",
                        hint=(
                            "Select a plate in the UI or request "
                            "selection_mode='all' explicitly."
                        ),
                    ),
                ),
            )
        try:
            context = self._manager.orchestrator_code_document_context(
                selection_mode=selection_mode,
                empty_selection_policy=EmptyPlateSelectionPolicy.ERROR,
            )
        except Exception as exc:
            return self._document_error(
                request,
                (AgentError.from_exception("ui_code_document_read_failed", exc),),
            )

        source_bytes = context.source.encode("utf-8")
        return UiCodeDocument(
            schema_version=SCHEMA_VERSION,
            summary=self.summary(),
            source=context.source,
            mime_type=PYTHON_MIME_TYPE,
            size_bytes=len(source_bytes),
            sha256=hashlib.sha256(source_bytes).hexdigest(),
            current_revision_token=self._snapshot_provider.revision_token(
                self.identity.revision_key
            ),
            current_snapshot=self._snapshot_provider.current_snapshot(),
            selection_mode=selection_mode,
            selected_scope_ids=context.selected_scope_ids,
        )

    def validate(
        self,
        request: UiCodeDocumentValidationRequest,
    ) -> UiCodeDocumentValidationResult:
        try:
            result = self._execution_service.validate_source(
                request.source,
                self._manager.code_document_execution_operations(),
            )
        except UiCodeDocumentValidationError as exc:
            return UiCodeDocumentValidationResult(
                schema_version=SCHEMA_VERSION,
                document_id=request.document_id,
                valid=False,
                errors=exc.errors,
            )
        except Exception as exc:
            return UiCodeDocumentValidationResult(
                schema_version=SCHEMA_VERSION,
                document_id=request.document_id,
                valid=False,
                errors=(
                    AgentError.from_exception(
                        "ui_code_document_validation_failed", exc
                    ),
                ),
            )

        return UiCodeDocumentValidationResult(
            schema_version=SCHEMA_VERSION,
            document_id=request.document_id,
            valid=True,
            normalized_scope_ids=result.plate_paths,
        )

    def apply(self, request: UiCodeDocumentApplyRequest) -> UiCodeDocumentApplyResult:
        guard_error = self._apply_guard_policy(request).first_error()
        if guard_error is not None:
            return self._apply_error(request, guard_error)

        current_revision = self._snapshot_provider.revision_token(
            self.identity.revision_key
        )
        if request.base_revision_token != current_revision:
            return self._apply_error(
                request,
                STALE_CODE_DOCUMENT_REVISION_ERROR,
            )

        try:
            operations = self._manager.code_document_execution_operations(
                PlateManagerCodeMutationScope.from_carrier(
                    request,
                    default=SelectedAllSelectionMode.ALL,
                )
            )
            result = self._execution_service.validate_source(request.source, operations)
            operations.pre_code_execution()
            ObjectStateRegistry.ensure_baseline_snapshot()
            pre_snapshot = self._snapshot_provider.current_snapshot()
            pre_head_id = self._snapshot_provider.current_branch_head_snapshot_id()
            label = UiCodeDocumentApplyLabel.resolve(request, self.identity).value
            with ObjectStateRegistry.atomic_success(label, result.mutation_scope):
                if not operations.apply_code_namespace(result.apply_namespace()):
                    raise ValueError("Code document payload was not applied.")
            operations.post_code_execution()
            post_head_id = self._snapshot_provider.current_branch_head_snapshot_id()
            if post_head_id == pre_head_id:
                return self._apply_error(
                    request,
                    AgentError(
                        code="snapshot_not_recorded",
                        message=(
                            "Applying the UI document did not record a new "
                            "ObjectState snapshot."
                        ),
                    ),
                )
            post_snapshot = self._snapshot_provider.current_snapshot()
            new_revision_token = self._snapshot_provider.revision_token(
                self.identity.revision_key
            )
            return UiCodeDocumentApplyResult(
                schema_version=SCHEMA_VERSION,
                document_id=request.document_id,
                applied=True,
                base_revision_token=request.base_revision_token,
                receipt=UiMutationReceipt.accepted_for(request.request_token),
                outcome="applied",
                new_revision_token=new_revision_token,
                current_revision_token=new_revision_token,
                current_snapshot=post_snapshot,
                undo_snapshot=pre_snapshot,
                pre_apply_snapshot=pre_snapshot,
                post_apply_snapshot=post_snapshot,
            )
        except UiCodeDocumentValidationError as exc:
            return self._apply_errors(request, exc.errors)
        except Exception as exc:
            return self._apply_error(
                request,
                AgentError.from_exception("ui_code_document_apply_failed", exc),
            )

    def _document_error(
        self,
        request: UiCodeDocumentRequest,
        errors: tuple[AgentError, ...],
    ) -> UiCodeDocument:
        selection_mode = request.resolved_selection_mode(
            UiCodeDocumentSelectionMode.SELECTED
        )
        return UiCodeDocument(
            schema_version=SCHEMA_VERSION,
            summary=self.summary(),
            source="",
            mime_type=PYTHON_MIME_TYPE,
            size_bytes=0,
            sha256=hashlib.sha256(b"").hexdigest(),
            current_revision_token=self._snapshot_provider.revision_token(
                self.identity.revision_key
            ),
            current_snapshot=self._snapshot_provider.current_snapshot(),
            selection_mode=selection_mode,
            selected_scope_ids=(),
            errors=errors,
        )

    @staticmethod
    def _apply_guard_policy(
        request: UiCodeDocumentApplyRequest,
    ) -> UiBridgeGuardPolicy:
        return UiBridgeGuardPolicy(
            rules=(
                CONFIRMATION_REQUIRED_GUARD.bind(
                    lambda: request.confirmation_is_required(),
                ),
                APPLY_TIME_TRAVEL_OPT_IN_GUARD.bind(
                    lambda: (
                        ObjectStateRegistry.is_time_traveling()
                        and not request.apply_if_time_traveling
                    ),
                ),
            )
        )


class PlateManagerStateSurfaceProvider(
    UiStateSurfaceProviderABC,
):
    """Pollable PlateManager state provider backed by the shared UI projection."""

    identity = PLATE_MANAGER_STATE_IDENTITY

    def __init__(
        self,
        manager,
        *,
        snapshot_provider: UiBridgeSnapshotProviderABC,
        projection_service: PlateManagerStateProjectionService | None = None,
    ) -> None:
        bind_snapshot_backed_provider(
            self,
            manager,
            snapshot_provider=snapshot_provider,
        )
        self._projection_service = (
            projection_service or PlateManagerStateProjectionService()
        )

    def summary(self) -> UiStateSurfaceSummary:
        return UiStateSurfaceSummary(
            schema_version=SCHEMA_VERSION,
            identity=self.identity.as_surface_identity(),
            title=self.identity.title,
            widget_id=self.identity.widget_id,
            readable=True,
            supported_selection_modes=("selected", "all"),
            current_selection_count=len(self._manager.get_selected_items()),
            total_scope_count=len(self._manager.plates),
        )

    def read(self, request: UiStateSurfaceRequest) -> UiStateSurfaceDocument:
        selection_mode = request.resolved_selection_mode(
            UiCodeDocumentSelectionMode.ALL
        )
        try:
            state = self._projection_service.project(
                self._manager,
                schema_version=SCHEMA_VERSION,
                summary=self.summary(),
                selection_mode=selection_mode,
            )
        except Exception as exc:
            return self._state_error(
                request,
                (AgentError.from_exception("ui_state_surface_read_failed", exc),),
            )

        revision_token = self._revision_token(state)
        state = replace(
            state,
            current_revision_token=revision_token,
            current_snapshot=self._snapshot_provider.current_snapshot(),
            unchanged=request.base_revision_token == revision_token,
        )
        return self._document_from_state(state)

    def overview_sections(self) -> tuple[UiLiveOverviewSection, ...]:
        state = self._projection_service.project(
            self._manager,
            schema_version=SCHEMA_VERSION,
            summary=self.summary(),
            selection_mode=UiCodeDocumentSelectionMode.ALL.value,
        )
        rows = state.rows
        return (
            UiLiveOverviewSection(
                section_id=self.identity.surface_id,
                title=self.identity.title,
                summary=state.manager_execution_state,
                metrics=(
                    UiLiveOverviewMetric(
                        key="plates",
                        label="plates",
                        value=str(len(rows)),
                    ),
                    UiLiveOverviewMetric(
                        key="selected",
                        label="selected",
                        value=str(sum(1 for row in rows if row.selected)),
                    ),
                    UiLiveOverviewMetric(
                        key="active",
                        label="active",
                        value=str(sum(1 for row in rows if row.execution_active)),
                    ),
                    UiLiveOverviewMetric(
                        key="queued",
                        label="queued",
                        value=str(
                            sum(1 for row in rows if row.queue_position is not None)
                        ),
                    ),
                ),
                items=tuple(self._overview_row_item(row) for row in rows),
            ),
        )

    @classmethod
    def _overview_row_item(cls, row: UiPlateManagerRowState) -> UiLiveOverviewItem:
        return UiLiveOverviewItem(
            label=row.name,
            status=row.status_prefix or row.orchestrator_state,
            detail=cls._overview_row_detail(row),
            severity=cls._overview_row_severity(row).value,
            source_surface_id=PLATE_MANAGER_STATE_SURFACE_ID,
            source_widget_id=PlateManagerWidgetIdentity.require_value(),
        )

    @staticmethod
    def _overview_row_detail(row: UiPlateManagerRowState) -> str:
        parts = [
            f"initialized={row.initialized}",
            f"compiled={row.compiled}",
            f"active={row.execution_active}",
        ]
        if row.runtime_percent is not None:
            parts.append(f"progress={row.runtime_percent:.1f}%")
        if row.terminal_status is not None:
            parts.append(f"terminal={row.terminal_status}")
        if row.debug_phase is not None:
            parts.append(f"debug={row.debug_phase}")
        return " ".join(parts)

    @staticmethod
    def _overview_row_severity(row: UiPlateManagerRowState) -> UiLiveOverviewSeverity:
        if row.terminal_status == TerminalExecutionStatus.FAILED.value:
            return UiLiveOverviewSeverity.ERROR
        if row.execution_active or row.queue_position is not None:
            return UiLiveOverviewSeverity.WARNING
        return UiLiveOverviewSeverity.INFO

    def _state_error(
        self,
        request: UiStateSurfaceRequest,
        errors: tuple[AgentError, ...],
    ) -> UiStateSurfaceDocument:
        selection_mode = request.resolved_selection_mode(
            UiCodeDocumentSelectionMode.ALL
        )
        state = UiPlateManagerState(
            schema_version=SCHEMA_VERSION,
            summary=self.summary(),
            selection_mode=selection_mode,
            selected_scope_ids=(),
            object_state_token=ObjectStateRegistry.get_token(),
            manager_execution_state="unknown",
            rows=(),
            current_revision_token=self._snapshot_provider.revision_token(
                self.identity.revision_key
            ),
            current_snapshot=self._snapshot_provider.current_snapshot(),
            errors=errors,
        )
        return self._document_from_state(state)

    @staticmethod
    def _document_from_state(state: UiPlateManagerState) -> UiStateSurfaceDocument:
        payload = to_jsonable(state)
        if not isinstance(payload, dict):
            raise TypeError(
                "PlateManager state payload did not serialize to an object."
            )
        return UiStateSurfaceDocument(
            schema_version=state.schema_version,
            summary=state.summary,
            payload_schema=PLATE_MANAGER_STATE_DECLARATION.payload_schema,
            payload=payload,
            current_revision_token=state.current_revision_token,
            current_snapshot=state.current_snapshot,
            selection_mode=state.selection_mode,
            selected_scope_ids=state.selected_scope_ids,
            unchanged=state.unchanged,
            warnings=state.warnings,
            errors=state.errors,
        )

    def _revision_token(self, state: UiPlateManagerState) -> str:
        row_parts = tuple(self._row_revision_part(row) for row in state.rows)
        parts = (
            self.identity.revision_key,
            str(state.object_state_token),
            self._snapshot_provider.current_branch_head_snapshot_id(),
            str(ObjectStateRegistry.get_current_snapshot_index()),
            state.selection_mode,
            state.manager_execution_state,
            state.selected_scope_ids,
            row_parts,
        )
        return hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()

    @staticmethod
    def _row_revision_part(row: UiPlateManagerRowState) -> tuple:
        return (
            row.plate_scope_id,
            row.selected,
            row.initialized,
            row.compiled,
            row.init_pending,
            row.compile_pending,
            row.execution_active,
            row.status_prefix,
            row.orchestrator_state,
            row.execution_id,
            row.terminal_status,
            row.runtime_state,
            row.runtime_percent,
            row.queue_position,
        )


class LiveMeasurementsStateSurfaceProvider(UiStateSurfaceProviderABC):
    """Pollable quantitative results backed by Plate Manager's retained model."""

    identity = LIVE_MEASUREMENTS_STATE_IDENTITY

    def __init__(
        self,
        manager,
        *,
        snapshot_provider: UiBridgeSnapshotProviderABC,
    ) -> None:
        self._manager = manager
        self._snapshot_provider = snapshot_provider

    def summary(self) -> UiStateSurfaceSummary:
        all_entries = self._manager.live_measurement_model.semantic_entries()
        selected_plate_ids = self._selected_plate_ids()
        return UiStateSurfaceSummary(
            schema_version=SCHEMA_VERSION,
            identity=self.identity.as_surface_identity(),
            title=self.identity.title,
            widget_id=self.identity.widget_id,
            readable=True,
            supported_selection_modes=(
                UiCodeDocumentSelectionMode.SELECTED.value,
                UiCodeDocumentSelectionMode.ALL.value,
            ),
            current_selection_count=sum(
                entry.plate_id in selected_plate_ids for entry in all_entries
            ),
            total_scope_count=len(all_entries),
        )

    def read(self, request: UiStateSurfaceRequest) -> UiStateSurfaceDocument:
        selection_mode = request.resolved_selection_mode(
            UiCodeDocumentSelectionMode.ALL
        )
        try:
            state = self._project_state(selection_mode)
        except Exception as exc:
            return self._state_error(
                request,
                (AgentError.from_exception("ui_state_surface_read_failed", exc),),
            )

        revision_token = self._revision_token(state)
        state = replace(
            state,
            current_revision_token=revision_token,
            current_snapshot=self._snapshot_provider.current_snapshot(),
            unchanged=request.base_revision_token == revision_token,
        )
        return self._document_from_state(state)

    def overview_sections(self) -> tuple[UiLiveOverviewSection, ...]:
        entries = self._manager.live_measurement_model.semantic_entries()
        return (
            UiLiveOverviewSection(
                section_id=self.identity.surface_id,
                title=self.identity.title,
                summary=(
                    f"{len(entries)} retained table preview(s)"
                    if entries
                    else "No live measurement results yet"
                ),
                metrics=(
                    UiLiveOverviewMetric(
                        key="tables",
                        label="tables",
                        value=str(len(entries)),
                    ),
                    UiLiveOverviewMetric(
                        key="rows",
                        label="rows",
                        value=str(sum(entry.preview.row_count for entry in entries)),
                    ),
                    UiLiveOverviewMetric(
                        key="executions",
                        label="executions",
                        value=str(len({entry.execution_id for entry in entries})),
                    ),
                ),
                items=tuple(
                    self._overview_item(entry)
                    for entry in sorted(
                        entries,
                        key=lambda candidate: candidate.sequence_id,
                        reverse=True,
                    )[:5]
                ),
            ),
        )

    def _project_state(self, selection_mode: str) -> UiLiveMeasurementsState:
        all_entries = self._manager.live_measurement_model.semantic_entries()
        selected_plate_ids = self._selected_plate_ids()
        mode = UiCodeDocumentSelectionMode(selection_mode)
        entries = (
            all_entries
            if mode is UiCodeDocumentSelectionMode.ALL
            else tuple(
                entry for entry in all_entries if entry.plate_id in selected_plate_ids
            )
        )
        entry_states = tuple(self._entry_state(entry) for entry in entries)
        return UiLiveMeasurementsState(
            schema_version=SCHEMA_VERSION,
            summary=self.summary(),
            selection_mode=mode.value,
            selected_scope_ids=tuple(
                row.scope_id for row in self._manager.get_selected_items()
            ),
            object_state_token=ObjectStateRegistry.get_token(),
            retained_entry_count=len(all_entries),
            visible_entry_count=len(entry_states),
            total_row_count=sum(entry.preview.row_count for entry in entries),
            latest_sequence_id=max(
                (entry.sequence_id for entry in entries),
                default=None,
            ),
            entries=entry_states,
        )

    @staticmethod
    def _entry_state(entry) -> UiLiveMeasurementEntryState:
        return UiLiveMeasurementEntryState(
            sequence_id=entry.sequence_id,
            execution_id=entry.execution_id,
            plate_id=entry.plate_id,
            axis_id=entry.axis_id,
            step_name=entry.step_name,
            preview=entry.preview,
            truncated_preview_group=entry.truncated_preview_group,
        )

    def _selected_plate_ids(self) -> frozenset[str]:
        return frozenset(row.scope_id for row in self._manager.get_selected_items())

    def _revision_token(self, state: UiLiveMeasurementsState) -> str:
        parts = (
            self.identity.revision_key,
            str(state.object_state_token),
            self._snapshot_provider.current_branch_head_snapshot_id(),
            str(ObjectStateRegistry.get_current_snapshot_index()),
            state.selection_mode,
            state.selected_scope_ids,
            state.retained_entry_count,
            state.entries,
        )
        return hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()

    def _state_error(
        self,
        request: UiStateSurfaceRequest,
        errors: tuple[AgentError, ...],
    ) -> UiStateSurfaceDocument:
        selection_mode = request.resolved_selection_mode(
            UiCodeDocumentSelectionMode.ALL
        )
        state = UiLiveMeasurementsState(
            schema_version=SCHEMA_VERSION,
            summary=self.summary(),
            selection_mode=selection_mode,
            selected_scope_ids=(),
            object_state_token=ObjectStateRegistry.get_token(),
            retained_entry_count=0,
            visible_entry_count=0,
            total_row_count=0,
            latest_sequence_id=None,
            entries=(),
            current_revision_token=self._snapshot_provider.revision_token(
                self.identity.revision_key
            ),
            current_snapshot=self._snapshot_provider.current_snapshot(),
            errors=errors,
        )
        return self._document_from_state(state)

    @staticmethod
    def _document_from_state(
        state: UiLiveMeasurementsState,
    ) -> UiStateSurfaceDocument:
        payload = to_jsonable(state)
        if not isinstance(payload, dict):
            raise TypeError(
                "Live measurements state payload did not serialize to an object."
            )
        return UiStateSurfaceDocument(
            schema_version=state.schema_version,
            summary=state.summary,
            payload_schema=LIVE_MEASUREMENTS_STATE_DECLARATION.payload_schema,
            payload=payload,
            current_revision_token=state.current_revision_token,
            current_snapshot=state.current_snapshot,
            selection_mode=state.selection_mode,
            selected_scope_ids=state.selected_scope_ids,
            unchanged=state.unchanged,
            warnings=state.warnings,
            errors=state.errors,
        )

    @classmethod
    def _overview_item(cls, entry) -> UiLiveOverviewItem:
        preview = entry.preview
        subject = preview.object_name or preview.source_image_name or "measurements"
        return UiLiveOverviewItem(
            label=f"{entry.step_name}: {preview.address.key.name}",
            status=f"{preview.row_count} row(s)",
            detail=(
                f"{subject}; execution={entry.execution_id}; axis={entry.axis_id}; "
                f"columns={len(preview.columns)}"
            ),
            source_surface_id=cls.identity.surface_id,
            source_widget_id=cls.identity.widget_id,
        )


class PlateManagerActionProvider(
    UiActionProviderABC,
):
    """PlateManager action provider backed by the widget's declared action routes."""

    identity = PLATE_MANAGER_ACTION_PROVIDER_IDENTITY

    def __init__(self, manager) -> None:
        self._manager = manager

    def catalog(self) -> UiActionCatalog:
        return UiActionCatalog(
            schema_version=SCHEMA_VERSION,
            actions=tuple(
                self.summary(action.value) for action in self._manager.ACTION_ROUTES
            ),
            warnings=(
                AgentWarning(
                    code="plate_path_setup_uses_code_document",
                    message=PLATE_PATH_CODE_DOCUMENT_HINT,
                ),
            ),
        )

    def summary(self, action_id: str) -> UiActionSummary:
        action = self._action(action_id)
        selected_scope_ids = self._selected_scope_ids()
        availability_error = self._action_availability_error(action)
        return UiActionSummary(
            schema_version=SCHEMA_VERSION,
            identity=UiActionIdentity(
                widget_id=self.identity.widget_id,
                action_id=action.value,
            ),
            title=self._action_title(action),
            enabled=availability_error is None,
            disabled_error=availability_error,
            invocation_mode=self._invocation_mode(action),
            side_effects=action.side_effects,
            confirmation_required=action.confirmation_required,
            selection_mode="selected",
            current_selection_count=len(selected_scope_ids),
            target_scope_ids=selected_scope_ids,
            selection_revision_token=self._selection_revision_token(),
            related_state_surface_ids=self._related_state_surface_ids(action),
        )

    @staticmethod
    def _related_state_surface_ids(action: PlateManagerAction) -> tuple[str, ...]:
        return state_surface_ids_for_action(
            PlateManagerWidget.UI_STATE_SURFACE_DECLARATIONS,
            action.value,
        )

    def invoke(self, request: UiActionInvokeRequest) -> UiActionInvokeResult:
        try:
            action = self._action(request.action_id)
        except Exception as exc:
            return self._invoke_error(
                request,
                AgentError.from_exception("unknown_ui_action", exc),
            )
        guard_error = self._guard_error(action, request)
        if guard_error is not None:
            return self._invoke_error(request, guard_error)

        try:
            dispatch_result = dispatch_widget_action(
                widget=self._manager,
                action_id=action.value,
                action_enum=PlateManagerAction,
                routes=self._manager.ACTION_ROUTES,
                async_runner=self._manager.service_adapter.execute_async_operation,
                before_dispatch=commit_focused_widget_edits,
            )
        except Exception as exc:
            return self._invoke_error(
                request,
                AgentError.from_exception("ui_action_dispatch_failed", exc),
            )

        return UiActionInvokeResult(
            schema_version=SCHEMA_VERSION,
            identity=UiActionIdentity(
                widget_id=self.identity.widget_id,
                action_id=action.value,
            ),
            status=UiActionInvocationStatus.ACCEPTED.value,
            receipt=UiMutationReceipt.accepted_for(request.request_token),
            target_scope_ids=self._selected_scope_ids(),
            selection_revision_token=self._selection_revision_token(),
            workflow_status_surface_ids=PLATE_MANAGER_ACTION_STATE_SURFACES,
            recommended_poll_interval_ms=500,
            warnings=(
                AgentWarning(
                    code="ui_action_dispatched",
                    message=(
                        f"PlateManager action {action.value!r} was dispatched "
                        f"as {dispatch_result.invocation_mode} work; poll "
                        f"{PLATE_MANAGER_STATE_SURFACE_ID} for workflow status."
                    ),
                ),
            ),
        )

    def _guard_error(
        self,
        action: PlateManagerAction,
        request: UiActionInvokeRequest,
    ) -> AgentError | None:
        selected_scope_ids = self._selected_scope_ids()
        if (
            request.selected_scope_ids
            and request.selected_scope_ids != selected_scope_ids
        ):
            return AgentError(
                code="stale_ui_action_selection",
                message="Requested target scopes do not match current PlateManager selection.",
            )
        observed_revision = request.observed_selection_revision_token
        current_revision = self._selection_revision_token()
        if observed_revision is not None and observed_revision != current_revision:
            return AgentError(
                code="stale_ui_action_revision",
                message="PlateManager selection changed after the action was planned.",
            )
        availability_error = self._action_availability_error(action)
        if availability_error is not None:
            return availability_error
        if action.confirmation_required and request.confirmation_is_required():
            return AgentError(
                code="confirmation_required",
                message=(
                    "This PlateManager action mutates UI state or starts a workflow; "
                    "set require_confirmation=False to dispatch it."
                ),
            )
        return None

    def _action_availability_error(
        self, action: PlateManagerAction
    ) -> AgentError | None:
        operation_error = self._operation_validation_error(action)
        if operation_error is not None:
            return operation_error
        if not self._action_enabled(action):
            return AgentError(
                code="ui_action_disabled",
                message=f"PlateManager action {action.value!r} is disabled.",
                hint=PLATE_ACTION_DISABLED_HINT,
            )
        return None

    def _operation_validation_error(
        self, action: PlateManagerAction
    ) -> AgentError | None:
        selected_rows = tuple(self._manager.get_selected_items())
        if action.plate_operation is not None:
            if not selected_rows:
                return AgentError(
                    code="plate_selection_required",
                    message=(
                        f"PlateManager action {action.value!r} requires a selected plate."
                    ),
                    hint=PLATE_SELECTION_REQUIRED_HINT,
                )

            validator = PlateOperationValidator.for_operation(action.plate_operation)
            invalid_results = []
            for row in selected_rows:
                result = validator.validate(self._manager, row)
                if not result.valid:
                    invalid_results.append(result)
            if invalid_results:
                result = invalid_results[0]
                message = result.message
                if result.recovery_action is not None:
                    message = (
                        f"{message} Next workflow: {result.recovery_action.value}."
                    )
                return AgentError(
                    code=result.reason,
                    message=message,
                    hint=PLATE_ACTION_DISABLED_HINT,
                )
        return None

    def _invoke_error(
        self,
        request: UiActionInvokeRequest,
        error: AgentError,
    ) -> UiActionInvokeResult:
        return UiActionInvokeResult(
            schema_version=SCHEMA_VERSION,
            identity=UiActionIdentity(
                widget_id=request.widget_id,
                action_id=request.action_id,
            ),
            status=UiActionInvocationStatus.REJECTED.value,
            receipt=UiMutationReceipt.rejected_for(request.request_token),
            target_scope_ids=self._selected_scope_ids(),
            selection_revision_token=self._selection_revision_token(),
            workflow_status_surface_ids=PLATE_MANAGER_ACTION_STATE_SURFACES,
            errors=(error,),
        )

    def _action(self, action_id: str) -> PlateManagerAction:
        action = PlateManagerAction(action_id)
        if action not in self._manager.ACTION_ROUTES:
            raise ValueError(f"PlateManager action has no route: {action_id!r}")
        return action

    def _action_enabled(self, action: PlateManagerAction) -> bool:
        button = self._manager.buttons[action.value]
        return button.isEnabled()

    def _invocation_mode(self, action: PlateManagerAction) -> str:
        if action not in self._manager.ACTION_ROUTES:
            raise ValueError(f"PlateManager action has no route: {action.value!r}")
        return "sync"

    def _selected_scope_ids(self) -> tuple[str, ...]:
        return tuple(row.scope_id for row in self._manager.get_selected_items())

    def _selection_revision_token(self) -> str:
        parts = (
            self.identity.widget_id,
            self._selected_scope_ids(),
            ObjectStateRegistry.get_token(),
        )
        return hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()

    def _action_title(self, action: PlateManagerAction) -> str:
        for label, action_id, _tooltip in self._manager.BUTTON_CONFIGS:
            if action_id == action.value:
                return label
        return action.value
