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
    UiCodeDocumentId,
    UiCodeDocumentRequest,
    UiCodeDocumentSelectionMode,
    UiCodeDocumentSummary,
    UiCodeDocumentValidationRequest,
    UiCodeDocumentValidationResult,
    UiPlateManagerRowState,
    UiPlateManagerState,
    UiStateSurfaceId,
    UiStateSurfaceDocument,
    UiStateSurfaceRequest,
    UiStateSurfaceSummary,
    UiMutationReceipt,
)
from openhcs.agent.serialization import to_jsonable
from openhcs.config_framework.object_state import ObjectStateRegistry
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
    UiBridgeGuardPolicy,
    UiActionProviderABC,
    UiActionProviderIdentity,
    UiBridgeSnapshotProviderABC,
    UiCodeDocumentProviderABC,
    UiCodeDocumentProviderIdentity,
    UiStateSurfaceProviderABC,
    UiStateSurfaceProviderIdentity,
)
from openhcs.pyqt_gui.services.ui_bridge_registry import (
    UiBridgeProviderSetABC,
    UiBridgeRegistrationContext,
)
from openhcs.pyqt_gui.widgets.plate_manager import (
    EmptyPlateSelectionPolicy,
    PlateManagerAction,
)
from openhcs.pyqt_gui.widgets.shared.services.widget_action_dispatch import (
    dispatch_widget_action,
)
from openhcs.pyqt_gui.widgets.shared.services.qt_widget_edit_commit import (
    commit_focused_widget_edits,
)


ORCHESTRATOR_DOCUMENT_TITLE = "Plate manager orchestrator config"
PLATE_MANAGER_STATE_TITLE = "Plate manager state"
PLATE_MANAGER_ACTIONS_TITLE = "Plate manager actions"
ORCHESTRATOR_WIDGET_ID = "plate_manager"
PYTHON_MIME_TYPE = "text/x-python"
PLATE_MANAGER_STATE_PAYLOAD_SCHEMA = "openhcs.ui.plate_manager_state.v1"

PLATE_MANAGER_ORCHESTRATOR_IDENTITY = UiCodeDocumentProviderIdentity(
    document_id=UiCodeDocumentId.PLATE_MANAGER_ORCHESTRATOR.value,
    title=ORCHESTRATOR_DOCUMENT_TITLE,
    widget_id=ORCHESTRATOR_WIDGET_ID,
)
PLATE_MANAGER_STATE_IDENTITY = UiStateSurfaceProviderIdentity(
    surface_id=UiStateSurfaceId.PLATE_MANAGER.value,
    title=PLATE_MANAGER_STATE_TITLE,
    widget_id=ORCHESTRATOR_WIDGET_ID,
)
PLATE_MANAGER_ACTION_PROVIDER_IDENTITY = UiActionProviderIdentity(
    action_id="plate_manager.actions",
    widget_id=ORCHESTRATOR_WIDGET_ID,
    title=PLATE_MANAGER_ACTIONS_TITLE,
)
PLATE_MANAGER_ACTION_STATE_SURFACES = (UiStateSurfaceId.PLATE_MANAGER.value,)
PLATE_MANAGER_CONFIRMED_ACTIONS = frozenset(
    (
        PlateManagerAction.ADD_PLATE,
        PlateManagerAction.DELETE_PLATE,
        PlateManagerAction.EDIT_CONFIG,
        PlateManagerAction.INIT_PLATE,
        PlateManagerAction.COMPILE_PLATE,
        PlateManagerAction.RUN_PLATE,
    )
)
PLATE_MANAGER_ACTION_SIDE_EFFECTS = {
    PlateManagerAction.ADD_PLATE: ("opens_file_dialog", "mutates_plate_collection"),
    PlateManagerAction.DELETE_PLATE: ("mutates_plate_collection",),
    PlateManagerAction.EDIT_CONFIG: ("opens_config_window", "may_mutate_plate_config"),
    PlateManagerAction.INIT_PLATE: ("starts_initialization_workflow",),
    PlateManagerAction.COMPILE_PLATE: ("starts_compile_workflow",),
    PlateManagerAction.RUN_PLATE: ("starts_or_stops_execution_workflow",),
    PlateManagerAction.CODE_PLATE: ("opens_code_document_window",),
    PlateManagerAction.VIEW_RESULTS: ("opens_results_window",),
    PlateManagerAction.VIEW_METADATA: ("opens_metadata_window",),
}


class PlateManagerBridgeProviderSet(UiBridgeProviderSetABC):
    """Register all PlateManager surfaces with a UI bridge registry."""

    registry_key = "plate_manager"

    def __init__(self, manager) -> None:
        self._manager = manager

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
    UiCodeDocumentProviderABC,
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
        try:
            context = self._manager.orchestrator_code_document_context(
                selection_mode=selection_mode,
                empty_selection_policy=EmptyPlateSelectionPolicy.FALL_BACK_TO_ALL,
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
                errors=(AgentError.from_exception("ui_code_document_validation_failed", exc),),
            )

        return UiCodeDocumentValidationResult(
            schema_version=SCHEMA_VERSION,
            document_id=request.document_id,
            valid=True,
            normalized_scope_ids=result.payload.plate_paths,
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
                AgentError(
                    code="stale_revision_token",
                    message="The UI document changed after it was read.",
                ),
            )

        try:
            operations = self._manager.code_document_execution_operations()
            result = self._execution_service.validate_source(request.source, operations)
            pre_snapshot = self._snapshot_provider.current_snapshot()
            pre_head_id = self._snapshot_provider.current_branch_head_snapshot_id()
            operations.pre_code_execution()
            ObjectStateRegistry.ensure_baseline_snapshot()
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
            return UiCodeDocumentApplyResult(
                schema_version=SCHEMA_VERSION,
                document_id=request.document_id,
                applied=True,
                base_revision_token=request.base_revision_token,
                outcome="applied",
                new_revision_token=self._snapshot_provider.revision_token(
                    self.identity.revision_key
                ),
                pre_apply_snapshot=pre_snapshot,
                post_apply_snapshot=self._snapshot_provider.current_snapshot(),
            )
        except UiCodeDocumentValidationError as exc:
            return UiCodeDocumentApplyResult(
                schema_version=SCHEMA_VERSION,
                document_id=request.document_id,
                applied=False,
                base_revision_token=request.base_revision_token,
                errors=exc.errors,
            )
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

    @staticmethod
    def apply_error(
        request: UiCodeDocumentApplyRequest,
        error: AgentError,
    ) -> UiCodeDocumentApplyResult:
        return PlateManagerOrchestratorCodeDocumentProvider._apply_error(
            request,
            error,
        )

    @staticmethod
    def _apply_error(
        request: UiCodeDocumentApplyRequest,
        error: AgentError,
    ) -> UiCodeDocumentApplyResult:
        return UiCodeDocumentApplyResult(
            schema_version=SCHEMA_VERSION,
            document_id=request.document_id,
            applied=False,
            base_revision_token=request.base_revision_token,
            errors=(error,),
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
        selection_mode = request.resolved_selection_mode(UiCodeDocumentSelectionMode.ALL)
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

    def _state_error(
        self,
        request: UiStateSurfaceRequest,
        errors: tuple[AgentError, ...],
    ) -> UiStateSurfaceDocument:
        selection_mode = request.resolved_selection_mode(UiCodeDocumentSelectionMode.ALL)
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
            raise TypeError("PlateManager state payload did not serialize to an object.")
        return UiStateSurfaceDocument(
            schema_version=state.schema_version,
            summary=state.summary,
            payload_schema=PLATE_MANAGER_STATE_PAYLOAD_SCHEMA,
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
        row_parts = tuple(
            self._row_revision_part(row)
            for row in state.rows
        )
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
                self.summary(action.value)
                for action in self._manager.ACTION_ROUTES
            ),
        )

    def summary(self, action_id: str) -> UiActionSummary:
        action = self._action(action_id)
        selected_scope_ids = self._selected_scope_ids()
        return UiActionSummary(
            schema_version=SCHEMA_VERSION,
            identity=UiActionIdentity(
                widget_id=self.identity.widget_id,
                action_id=action.value,
            ),
            title=self._action_title(action),
            enabled=self._action_enabled(action),
            invocation_mode=self._invocation_mode(action),
            side_effects=PLATE_MANAGER_ACTION_SIDE_EFFECTS[action],
            confirmation_required=action in PLATE_MANAGER_CONFIRMED_ACTIONS,
            selection_mode="selected",
            current_selection_count=len(selected_scope_ids),
            target_scope_ids=selected_scope_ids,
            selection_revision_token=self._selection_revision_token(),
            related_state_surface_ids=PLATE_MANAGER_ACTION_STATE_SURFACES,
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
            receipt=UiMutationReceipt(
                request_token=request.request_token,
                accepted=True,
            ),
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
                        "plate_manager.state for workflow status."
                    ),
                ),
            ),
        )

    def _guard_error(
        self,
        action: PlateManagerAction,
        request: UiActionInvokeRequest,
    ) -> AgentError | None:
        if not self._action_enabled(action):
            return AgentError(
                code="ui_action_disabled",
                message=f"PlateManager action {action.value!r} is disabled.",
            )
        selected_scope_ids = self._selected_scope_ids()
        if request.selected_scope_ids and request.selected_scope_ids != selected_scope_ids:
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
        if action in PLATE_MANAGER_CONFIRMED_ACTIONS and request.confirmation_is_required():
            return AgentError(
                code="confirmation_required",
                message=(
                    "This PlateManager action mutates UI state or starts a workflow; "
                    "set require_confirmation=False to dispatch it."
                ),
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
            receipt=UiMutationReceipt(
                request_token=request.request_token,
                accepted=False,
            ),
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
        route = self._manager.ACTION_ROUTES[action]
        import inspect

        if inspect.iscoroutinefunction(route.resolve_callable(self._manager)):
            return "async"
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
