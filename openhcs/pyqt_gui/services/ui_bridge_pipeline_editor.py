"""PipelineEditor provider set for the PyQt UI bridge."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import replace
from enum import Enum

from openhcs.agent.dto.common import AgentError, SCHEMA_VERSION
from openhcs.agent.dto.ui_bridge import (
    UiActionCatalog,
    UiActionIdentity,
    UiActionInvocationStatus,
    UiActionInvokeRequest,
    UiActionInvokeResult,
    UiActionSummary,
    UiCodeDocumentSelectionMode,
    UiDebugActionState,
    UiDebugCursorState,
    UiDebugRuntimeFrameState,
    UiDebugTerminalSummaryState,
    UiLiveOverviewItem,
    UiLiveOverviewMetric,
    UiLiveOverviewSection,
    UiLiveOverviewSeverity,
    UiMutationReceipt,
    UiPipelineDebugSessionState,
    UiPipelineEditorState,
    UiPipelineEditorStepState,
    UiProgressIdentityState,
    UiStateSurfaceDocument,
    UiStateSurfaceRequest,
    UiStateSurfaceSummary,
)
from openhcs.agent.ui_bridge_identities import (
    PipelineDebugSessionStateSurfaceIdentityDeclaration,
    PipelineDebugToolbarWidgetIdentity,
    PipelineEditorStateSurfaceIdentityDeclaration,
    PipelineEditorWidgetIdentity,
    PlateManagerStateSurfaceIdentityDeclaration,
)
from openhcs.agent.serialization import to_jsonable
from openhcs.config_framework.object_state import ObjectStateRegistry
from openhcs.core.function_reference import FunctionReferenceTransportAuthority
from openhcs.core.progress.debug_projection import DebugRuntimeFrame
from openhcs.pyqt_gui.services.plate_scope_identity import PipelineScopeIdentity
from openhcs.pyqt_gui.services.ui_bridge_contracts import (
    UiActionProviderABC,
    UiActionProviderIdentity,
    UiBridgeSnapshotProviderABC,
    UiStateSurfaceProviderABC,
    UiStateSurfaceProviderIdentity,
)
from openhcs.pyqt_gui.services.ui_bridge_registry import (
    UiBridgeProviderSetABC,
    UiBridgeRegistrationContext,
)
from openhcs.pyqt_gui.widgets.pipeline_editor import (
    PipelineEditorAction,
    PipelineEditorActionTargetMode,
)
from openhcs.pyqt_gui.widgets.shared.services.debug_session_projection import (
    DebugActionRenderModel,
    DebugToolbarActionProjector,
)
from openhcs.pyqt_gui.widgets.shared.services.pipeline_debug_actions import (
    DebugActionDisabledReason,
)
from openhcs.pyqt_gui.widgets.shared.services.qt_widget_edit_commit import (
    commit_focused_widget_edits,
)
from openhcs.pyqt_gui.widgets.shared.services.widget_action_dispatch import (
    dispatch_widget_action,
)


PIPELINE_EDITOR_STATE_TITLE = "Pipeline editor state"
PIPELINE_EDITOR_ACTIONS_TITLE = "Pipeline editor actions"
PIPELINE_DEBUG_TOOLBAR_ACTIONS_TITLE = "Pipeline debug toolbar actions"
PIPELINE_DEBUG_SESSION_STATE_TITLE = "Pipeline debug session state"
PIPELINE_EDITOR_STATE_PAYLOAD_SCHEMA = "openhcs.ui.pipeline_editor_state.v1"
PIPELINE_DEBUG_SESSION_STATE_PAYLOAD_SCHEMA = (
    "openhcs.ui.pipeline_debug_session_state.v1"
)
PLATE_MANAGER_STATE_SURFACE_ID = PlateManagerStateSurfaceIdentityDeclaration.require_value()
PIPELINE_EDITOR_WIDGET_ID = PipelineEditorWidgetIdentity.require_value()
PIPELINE_DEBUG_TOOLBAR_WIDGET_ID = PipelineDebugToolbarWidgetIdentity.require_value()
PIPELINE_EDITOR_STATE_IDENTITY = UiStateSurfaceProviderIdentity.from_declaration(
    PipelineEditorStateSurfaceIdentityDeclaration,
    title=PIPELINE_EDITOR_STATE_TITLE,
)
PIPELINE_DEBUG_SESSION_STATE_IDENTITY = UiStateSurfaceProviderIdentity.from_declaration(
    PipelineDebugSessionStateSurfaceIdentityDeclaration,
    title=PIPELINE_DEBUG_SESSION_STATE_TITLE,
)


class ManagerWidgetActionProviderABC(UiActionProviderABC, ABC):
    """Base action provider for manager widgets with declared action routes."""

    def __init__(self, manager) -> None:
        self._manager = manager

    def catalog(self) -> UiActionCatalog:
        return UiActionCatalog(
            schema_version=SCHEMA_VERSION,
            actions=tuple(self.summary(action.value) for action in self._actions()),
        )

    def summary(self, action_id: str) -> UiActionSummary:
        action = self._action(action_id)
        target_scope_ids = self._target_scope_ids(action)
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
            invocation_mode="sync",
            side_effects=self._side_effects(action),
            confirmation_required=self._confirmation_required(action),
            selection_mode=self._selection_mode(action),
            current_selection_count=len(target_scope_ids),
            target_scope_ids=target_scope_ids,
            selection_revision_token=self._selection_revision_token(),
            related_state_surface_ids=(PIPELINE_EDITOR_STATE_IDENTITY.surface_id,),
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
            dispatch_widget_action(
                widget=self._manager,
                action_id=action.value,
                action_enum=self._action_enum(),
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
            target_scope_ids=self._target_scope_ids(action),
            selection_revision_token=self._selection_revision_token(),
            recommended_poll_interval_ms=500,
            warnings=(),
        )

    def _guard_error(
        self,
        action: Enum,
        request: UiActionInvokeRequest,
    ) -> AgentError | None:
        target_scope_ids = self._target_scope_ids(action)
        if request.selected_scope_ids and request.selected_scope_ids != target_scope_ids:
            return AgentError(
                code="stale_ui_action_selection",
                message=(
                    f"{self.identity.widget_id} action target scopes changed after "
                    "the action was planned."
                ),
            )
        observed_revision = request.observed_selection_revision_token
        current_revision = self._selection_revision_token()
        if observed_revision is not None and observed_revision != current_revision:
            return AgentError(
                code="stale_ui_action_revision",
                message=(
                    f"{self.identity.widget_id} selection changed after the action "
                    "was planned."
                ),
            )
        availability_error = self._action_availability_error(action)
        if availability_error is not None:
            return availability_error
        if self._confirmation_required(action) and request.confirmation_is_required():
            return AgentError(
                code="confirmation_required",
                message=(
                    f"{self.identity.widget_id} action {action.value!r} mutates UI "
                    "state or opens an editor; set require_confirmation=False to "
                    "dispatch it."
                ),
            )
        return None

    def _action_availability_error(self, action: Enum) -> AgentError | None:
        if self._action_enabled(action):
            return None
        return AgentError(
            code="ui_action_disabled",
            message=(
                f"{self.identity.widget_id} action {action.value!r} is disabled."
            ),
            hint=self._disabled_hint(action),
        )

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
            target_scope_ids=self._target_scope_ids_for_request(request),
            selection_revision_token=self._selection_revision_token(),
            errors=(error,),
        )

    def _actions(self) -> tuple[Enum, ...]:
        return tuple(self._manager.ACTION_ROUTES)

    def _action(self, action_id: str) -> Enum:
        action = self._action_enum()(action_id)
        if action not in self._manager.ACTION_ROUTES:
            raise ValueError(
                f"{self.identity.widget_id} action has no route: {action_id!r}"
            )
        return action

    def _action_enabled(self, action: Enum) -> bool:
        button = self._manager.buttons[action.value]
        return button.isEnabled()

    def _selection_revision_token(self) -> str:
        parts = (
            self.identity.widget_id,
            self._all_target_scope_ids(),
            ObjectStateRegistry.get_token(),
        )
        return hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()

    def _all_target_scope_ids(self) -> tuple[str, ...]:
        scope_ids: list[str] = []
        for action in self._actions():
            scope_ids.extend(self._target_scope_ids(action))
        return tuple(scope_ids)

    def _action_title(self, action: Enum) -> str:
        for label, action_id, _tooltip in self._manager.BUTTON_CONFIGS:
            if action_id == action.value:
                return label
        return action.value

    def _target_scope_ids_for_request(
        self,
        request: UiActionInvokeRequest,
    ) -> tuple[str, ...]:
        try:
            return self._target_scope_ids(self._action(request.action_id))
        except Exception:
            return ()

    @abstractmethod
    def _action_enum(self) -> type[Enum]:
        raise NotImplementedError

    @abstractmethod
    def _target_scope_ids(self, action: Enum) -> tuple[str, ...]:
        raise NotImplementedError

    @abstractmethod
    def _side_effects(self, action: Enum) -> tuple[str, ...]:
        raise NotImplementedError

    @abstractmethod
    def _confirmation_required(self, action: Enum) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _selection_mode(self, action: Enum) -> str:
        raise NotImplementedError

    @abstractmethod
    def _disabled_hint(self, action: Enum) -> str:
        raise NotImplementedError


class PipelineEditorActionProvider(ManagerWidgetActionProviderABC):
    """PipelineEditor action provider backed by the widget's declared routes."""

    identity = UiActionProviderIdentity.from_widget_declaration(
        PipelineEditorWidgetIdentity,
        title=PIPELINE_EDITOR_ACTIONS_TITLE,
    )
    current_pipeline_disabled_hint = (
        "PipelineEditor actions require an initialized current plate. Inspect "
        f"{PLATE_MANAGER_STATE_SURFACE_ID}, initialize the selected plate, then read "
        f"window_code_document:{PIPELINE_EDITOR_WIDGET_ID} or widget-tree "
        f"{PIPELINE_EDITOR_WIDGET_ID}."
    )
    selected_steps_disabled_hint = (
        "PipelineEditor step actions require at least one selected step. Load or "
        "create steps with auto_load_pipeline, add_step, or "
        "window_code_document:pipeline_editor, then select a step row."
    )

    def _action_enum(self) -> type[PipelineEditorAction]:
        return PipelineEditorAction

    def _target_scope_ids(self, action: PipelineEditorAction) -> tuple[str, ...]:
        if action.target_mode is PipelineEditorActionTargetMode.CURRENT_PIPELINE:
            if not self._manager.current_plate:
                return ()
            return (
                PipelineScopeIdentity.from_plate_scope(
                    self._manager.current_plate
                ).scope_id,
            )
        if action.target_mode is PipelineEditorActionTargetMode.SELECTED_STEPS:
            return self._selected_step_scope_ids()
        raise ValueError(f"Unsupported PipelineEditor target mode: {action.target_mode}")

    def _side_effects(self, action: PipelineEditorAction) -> tuple[str, ...]:
        return action.side_effects

    def _confirmation_required(self, action: PipelineEditorAction) -> bool:
        return action.confirmation_required

    def _selection_mode(self, action: PipelineEditorAction) -> str:
        return action.target_mode.value

    def _disabled_hint(self, action: PipelineEditorAction) -> str:
        if action.target_mode is PipelineEditorActionTargetMode.SELECTED_STEPS:
            return self.selected_steps_disabled_hint
        return self.current_pipeline_disabled_hint

    def _selected_step_scope_ids(self) -> tuple[str, ...]:
        return self._manager.selected_step_scope_ids()


class PipelineDebugToolbarActionProvider(UiActionProviderABC):
    """Expose declared debug-toolbar controls through the UI bridge."""

    identity = UiActionProviderIdentity.from_widget_declaration(
        PipelineDebugToolbarWidgetIdentity,
        title=PIPELINE_DEBUG_TOOLBAR_ACTIONS_TITLE,
    )
    _related_state_surface_ids = (
        PLATE_MANAGER_STATE_SURFACE_ID,
        PipelineEditorStateSurfaceIdentityDeclaration.require_value(),
        PipelineDebugSessionStateSurfaceIdentityDeclaration.require_value(),
    )

    def __init__(self, manager) -> None:
        self._manager = manager

    def catalog(self) -> UiActionCatalog:
        return UiActionCatalog(
            schema_version=SCHEMA_VERSION,
            actions=tuple(self.summary(action_id) for action_id in self._action_ids()),
        )

    def summary(self, action_id: str) -> UiActionSummary:
        model = self._model(action_id)
        availability_error = self._availability_error(model)
        return UiActionSummary(
            schema_version=SCHEMA_VERSION,
            identity=UiActionIdentity(
                widget_id=self.identity.widget_id,
                action_id=action_id,
            ),
            title=model.label,
            enabled=availability_error is None,
            disabled_error=availability_error,
            invocation_mode="sync",
            side_effects=model.side_effects,
            confirmation_required=model.confirmation_required,
            selection_mode="current_pipeline",
            current_selection_count=len(model.target_scope_ids),
            target_scope_ids=model.target_scope_ids,
            selection_revision_token=self._selection_revision_token(),
            related_state_surface_ids=self._related_state_surface_ids,
        )

    def invoke(self, request: UiActionInvokeRequest) -> UiActionInvokeResult:
        guard_error = self._guard_error(request)
        if guard_error is not None:
            return self._invoke_error(request, guard_error)

        try:
            model = self._model(request.action_id)
            model.declaration.invoke_workflow(self._manager.debug_workflow)
        except Exception as exc:
            return self._invoke_error(
                request,
                AgentError.from_exception("ui_action_dispatch_failed", exc),
            )

        return UiActionInvokeResult(
            schema_version=SCHEMA_VERSION,
            identity=UiActionIdentity(
                widget_id=self.identity.widget_id,
                action_id=request.action_id,
            ),
            status=UiActionInvocationStatus.ACCEPTED.value,
            receipt=UiMutationReceipt.accepted_for(request.request_token),
            target_scope_ids=self._target_scope_ids(),
            selection_revision_token=self._selection_revision_token(),
            workflow_status_surface_ids=self._related_state_surface_ids,
            recommended_poll_interval_ms=500,
        )

    @staticmethod
    def _action_ids() -> tuple[str, ...]:
        return tuple(
            declaration.action_id()
            for declaration in DebugToolbarActionProjector.declarations()
        )

    def _models(self) -> tuple[DebugActionRenderModel, ...]:
        return DebugToolbarActionProjector.render_models(
            self._manager.debug_session_context()
        )

    def _model(self, action_id: str) -> DebugActionRenderModel:
        for model in self._models():
            if model.action_id == action_id:
                return model
        raise ValueError(f"Debug toolbar action is not declared: {action_id!r}")

    @staticmethod
    def _availability_error(model: DebugActionRenderModel) -> AgentError | None:
        if model.disabled_reason is None:
            return None
        return PipelineDebugToolbarActionProvider._agent_error_from_reason(
            model.disabled_reason
        )

    def _guard_error(self, request: UiActionInvokeRequest) -> AgentError | None:
        try:
            model = self._model(request.action_id)
        except Exception as exc:
            return AgentError.from_exception("unknown_ui_action", exc)

        target_scope_ids = model.target_scope_ids
        if request.selected_scope_ids and request.selected_scope_ids != target_scope_ids:
            return AgentError(
                code="stale_ui_action_selection",
                message=(
                    f"{self.identity.widget_id} action target scopes changed after "
                    "the action was planned."
                ),
            )
        observed_revision = request.observed_selection_revision_token
        current_revision = self._selection_revision_token()
        if observed_revision is not None and observed_revision != current_revision:
            return AgentError(
                code="stale_ui_action_revision",
                message=(
                    f"{self.identity.widget_id} selection changed after the action "
                    "was planned."
                ),
            )
        availability_error = self._availability_error(model)
        if availability_error is not None:
            return availability_error
        if model.confirmation_required and request.confirmation_is_required():
            return AgentError(
                code="confirmation_required",
                message=(
                    f"{self.identity.widget_id} action {request.action_id!r} mutates "
                    "debug execution state; set require_confirmation=False to dispatch it."
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
            receipt=UiMutationReceipt.rejected_for(request.request_token),
            target_scope_ids=self._target_scope_ids(),
            selection_revision_token=self._selection_revision_token(),
            errors=(error,),
        )

    @staticmethod
    def _agent_error_from_reason(reason: DebugActionDisabledReason) -> AgentError:
        return AgentError(
            code=reason.code,
            message=reason.message,
            hint=reason.hint,
        )

    def _target_scope_ids(self) -> tuple[str, ...]:
        return DebugToolbarActionProjector.target_scope_ids(
            self._manager.debug_session_context()
        )

    def _selection_revision_token(self) -> str:
        models = self._models()
        parts = (
            self.identity.widget_id,
            tuple(
                (
                    model.action_id,
                    model.enabled,
                    None
                    if model.disabled_reason is None
                    else model.disabled_reason.code,
                    model.target_scope_ids,
                )
                for model in models
            ),
            ObjectStateRegistry.get_token(),
        )
        return hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()


class PipelineDebugSessionStateSurfaceProvider(UiStateSurfaceProviderABC):
    """Pollable PipelineEditor debug-session state backed by shared projection."""

    identity = PIPELINE_DEBUG_SESSION_STATE_IDENTITY

    def __init__(
        self,
        manager,
        *,
        snapshot_provider: UiBridgeSnapshotProviderABC,
    ) -> None:
        self._manager = manager
        self._snapshot_provider = snapshot_provider

    def summary(self) -> UiStateSurfaceSummary:
        target_scope_ids = self._target_scope_ids()
        return UiStateSurfaceSummary(
            schema_version=SCHEMA_VERSION,
            identity=self.identity.as_surface_identity(),
            title=self.identity.title,
            widget_id=self.identity.widget_id,
            readable=True,
            supported_selection_modes=("all",),
            current_selection_count=len(target_scope_ids),
            total_scope_count=1 if target_scope_ids else 0,
        )

    def read(self, request: UiStateSurfaceRequest) -> UiStateSurfaceDocument:
        selection_mode = request.resolved_selection_mode(UiCodeDocumentSelectionMode.ALL)
        try:
            state = self._state()
        except Exception as exc:
            return self._state_error(
                request,
                (AgentError.from_exception("ui_state_surface_read_failed", exc),),
            )

        revision_token = self._revision_token(state, selection_mode=selection_mode)
        state = replace(
            state,
            current_revision_token=revision_token,
            current_snapshot=self._snapshot_provider.current_snapshot(),
            unchanged=request.base_revision_token == revision_token,
        )
        return self._document_from_state(state, selection_mode=selection_mode)

    def overview_sections(self) -> tuple[UiLiveOverviewSection, ...]:
        state = self._state()
        disabled_actions = tuple(
            action for action in state.actions if action.disabled_error is not None
        )
        items = []
        if state.terminal_summary is not None:
            items.append(
                UiLiveOverviewItem(
                    label="terminal debug session",
                    status=state.terminal_summary.terminal_status,
                    detail=self._terminal_summary_detail(state),
                    severity=UiLiveOverviewSeverity.WARNING.value,
                    source_surface_id=self.identity.surface_id,
                    source_widget_id=PipelineDebugToolbarWidgetIdentity.require_value(),
                )
            )
        items.extend(
            self._disabled_action_item(action)
            for action in disabled_actions
        )
        return (
            UiLiveOverviewSection(
                section_id=self.identity.surface_id,
                title=self.identity.title,
                summary=state.phase,
                metrics=(
                    UiLiveOverviewMetric(
                        key="phase",
                        label="phase",
                        value=state.phase,
                    ),
                    UiLiveOverviewMetric(
                        key="compiled",
                        label="compiled",
                        value=str(state.compiled),
                    ),
                    UiLiveOverviewMetric(
                        key="actions",
                        label="actions",
                        value=str(len(state.actions)),
                    ),
                    UiLiveOverviewMetric(
                        key="disabled",
                        label="disabled",
                        value=str(len(disabled_actions)),
                    ),
                ),
                items=tuple(items),
            ),
        )

    def _state(self) -> UiPipelineDebugSessionState:
        context = self._manager.debug_session_context()
        target = context.target
        session = context.active_session
        actions = tuple(self._action_state(model) for model in self._models())
        debug_projection = self._manager.debug_runtime_projection()
        return UiPipelineDebugSessionState(
            schema_version=SCHEMA_VERSION,
            summary=self.summary(),
            object_state_token=ObjectStateRegistry.get_token(),
            current_plate_scope_id=(
                None if target is None else target.current_plate_scope_id
            ),
            pipeline_scope_id=None if target is None else target.pipeline_scope_id,
            manager_execution_state=context.manager_execution_state.value,
            initialized=False if target is None else target.initialized,
            compiled=False if target is None else target.compiled,
            phase=DebugToolbarActionProjector.phase(context).value,
            active_session_id=None if session is None else session.debug_session_id,
            execution_id=None if session is None else session.execution_id,
            axis_id=None if session is None else session.axis_id,
            selected_source_group=(
                None if session is None else session.selected_source_group
            ),
            snapshot_store_ref=None if session is None else session.snapshot_store_ref,
            snapshot_store_backend=(
                None if session is None else session.snapshot_store_backend
            ),
            terminal_status=None if target is None else target.terminal_status,
            cursor=(
                None
                if session is None
                else self._cursor_state(session.cursor, session.dirty_from_cursor)
            ),
            terminal_summary=self._terminal_summary_state(context),
            actions=actions,
            current_frame=self._runtime_frame_state(debug_projection.current_frame),
            last_frame=self._runtime_frame_state(debug_projection.last_frame),
            selected_scope_ids=self._target_scope_ids(),
            current_revision_token=self._snapshot_provider.revision_token(
                self.identity.revision_key
            ),
            current_snapshot=self._snapshot_provider.current_snapshot(),
        )

    def _models(self) -> tuple[DebugActionRenderModel, ...]:
        return DebugToolbarActionProjector.render_models(
            self._manager.debug_session_context()
        )

    @staticmethod
    def _terminal_summary_detail(state: UiPipelineDebugSessionState) -> str | None:
        summary = state.terminal_summary
        if summary is None:
            return None
        parts = [
            f"plate={summary.plate_scope_id}",
            f"command={summary.command_type}",
        ]
        if summary.step_name is not None:
            parts.append(f"step={summary.step_name}")
        if summary.callable_name is not None:
            parts.append(f"callable={summary.callable_name}")
        return " ".join(parts)

    @staticmethod
    def _disabled_action_item(action: UiDebugActionState) -> UiLiveOverviewItem:
        disabled = action.disabled_error
        return UiLiveOverviewItem(
            label=action.label,
            status=None if disabled is None else disabled.code,
            detail=None if disabled is None else disabled.message,
            severity=UiLiveOverviewSeverity.INFO.value,
            source_surface_id=PIPELINE_DEBUG_SESSION_STATE_IDENTITY.surface_id,
            source_widget_id=PipelineDebugToolbarWidgetIdentity.require_value(),
        )

    @classmethod
    def _action_state(cls, model: DebugActionRenderModel) -> UiDebugActionState:
        return UiDebugActionState(
            action_id=model.action_id,
            label=model.label,
            placement=model.placement.value,
            enabled=model.enabled,
            side_effects=model.side_effects,
            confirmation_required=model.confirmation_required,
            requires_active_debug_session=model.requires_active_debug_session,
            disabled_error=(
                None
                if model.disabled_reason is None
                else PipelineDebugToolbarActionProvider._agent_error_from_reason(
                    model.disabled_reason
                )
            ),
            selected_scope_ids=model.target_scope_ids,
        )

    @staticmethod
    def _cursor_state(cursor, dirty_from_cursor) -> UiDebugCursorState | None:
        if cursor is None:
            return None
        return UiDebugCursorState(
            step_index=cursor.step_index,
            step_scope_id=cursor.step_scope_id,
            group_key=cursor.group_key,
            invocation_key=cursor.invocation_key,
            pattern_group_identity=cursor.pattern_group_identity,
            dirty=dirty_from_cursor == cursor,
        )

    def _terminal_summary_state(
        self,
        context,
    ) -> UiDebugTerminalSummaryState | None:
        summary = context.terminal_summary
        if summary is None:
            return None
        return UiDebugTerminalSummaryState(
            debug_session_id=summary.debug_session_id,
            plate_scope_id=summary.plate_id,
            terminal_status=summary.terminal_status,
            command_type=None if summary.command_type is None else summary.command_type.value,
            axis_id=summary.axis_id,
            snapshot_id=summary.snapshot_id,
            snapshot_store_ref=summary.snapshot_store_ref,
            snapshot_store_backend=summary.snapshot_store_backend,
            step_name=summary.step_name,
            callable_name=summary.callable_name,
            cursor=self._cursor_state(summary.cursor, None),
            completed_at_unix=summary.completed_at_unix,
        )

    @classmethod
    def _runtime_frame_state(
        cls,
        frame: DebugRuntimeFrame | None,
    ) -> UiDebugRuntimeFrameState | None:
        if frame is None:
            return None
        identity = frame.progress_identity
        context = frame.record.context
        cursor_state = cls._cursor_state(frame.cursor, None)
        if cursor_state is None:
            raise ValueError("Debug runtime frame cursor projection is required.")
        return UiDebugRuntimeFrameState(
            debug_session_id=frame.record.session_id,
            snapshot_store_ref=context.snapshot_store_ref,
            snapshot_store_backend=context.snapshot_store_backend,
            progress_identity=UiProgressIdentityState(
                execution_id=identity.execution_id,
                plate_id=identity.plate_id,
                axis_id=identity.axis_id,
                step_name=identity.step_name,
            ),
            cursor=cursor_state,
            event_type=frame.event_type.value,
            step_name=frame.step_name,
            callable_name=frame.callable_name,
            snapshot_id=frame.snapshot_id,
            timestamp=frame.record.event.timestamp,
        )

    def _target_scope_ids(self) -> tuple[str, ...]:
        return DebugToolbarActionProjector.target_scope_ids(
            self._manager.debug_session_context()
        )

    def _revision_token(
        self,
        state: UiPipelineDebugSessionState,
        *,
        selection_mode: str,
    ) -> str:
        action_parts = tuple(
            (
                action.action_id,
                action.enabled,
                None
                if action.disabled_error is None
                else action.disabled_error.code,
                action.selected_scope_ids,
            )
            for action in state.actions
        )
        cursor_parts = None
        if state.cursor is not None:
            cursor_parts = (
                state.cursor.step_index,
                state.cursor.step_scope_id,
                state.cursor.group_key,
                state.cursor.invocation_key,
                state.cursor.pattern_group_identity,
                state.cursor.dirty,
            )
        parts = (
            self.identity.revision_key,
            str(state.object_state_token),
            self._snapshot_provider.current_branch_head_snapshot_id(),
            str(ObjectStateRegistry.get_current_snapshot_index()),
            selection_mode,
            state.current_plate_scope_id,
            state.pipeline_scope_id,
            state.manager_execution_state,
            state.initialized,
            state.compiled,
            state.phase,
            state.active_session_id,
            state.execution_id,
            state.axis_id,
            state.selected_source_group,
            state.snapshot_store_ref,
            state.snapshot_store_backend,
            state.terminal_status,
            cursor_parts,
            None
            if state.terminal_summary is None
            else (
                state.terminal_summary.debug_session_id,
                state.terminal_summary.terminal_status,
                state.terminal_summary.command_type,
                state.terminal_summary.axis_id,
                state.terminal_summary.snapshot_id,
                state.terminal_summary.snapshot_store_ref,
                state.terminal_summary.snapshot_store_backend,
                state.terminal_summary.step_name,
                state.terminal_summary.callable_name,
                state.terminal_summary.completed_at_unix,
            ),
            self._frame_revision_parts(state.current_frame),
            self._frame_revision_parts(state.last_frame),
            action_parts,
        )
        return hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()

    @staticmethod
    def _frame_revision_parts(
        frame: UiDebugRuntimeFrameState | None,
    ) -> tuple | None:
        if frame is None:
            return None
        return (
            frame.debug_session_id,
            frame.progress_identity.execution_id,
            frame.progress_identity.plate_id,
            frame.progress_identity.axis_id,
            frame.progress_identity.step_name,
            frame.cursor.step_index,
            frame.cursor.step_scope_id,
            frame.cursor.group_key,
            frame.cursor.invocation_key,
            frame.cursor.pattern_group_identity,
            frame.event_type,
            frame.step_name,
            frame.callable_name,
            frame.snapshot_id,
            frame.snapshot_store_ref,
            frame.snapshot_store_backend,
            frame.timestamp,
        )

    def _state_error(
        self,
        request: UiStateSurfaceRequest,
        errors: tuple[AgentError, ...],
    ) -> UiStateSurfaceDocument:
        selection_mode = request.resolved_selection_mode(UiCodeDocumentSelectionMode.ALL)
        state = UiPipelineDebugSessionState(
            schema_version=SCHEMA_VERSION,
            summary=self.summary(),
            object_state_token=ObjectStateRegistry.get_token(),
            current_plate_scope_id=None,
            pipeline_scope_id=None,
            manager_execution_state="unknown",
            initialized=False,
            compiled=False,
            phase="unavailable",
            active_session_id=None,
            execution_id=None,
            axis_id=None,
            selected_source_group=None,
            snapshot_store_ref=None,
            snapshot_store_backend=None,
            terminal_status=None,
            cursor=None,
            terminal_summary=None,
            actions=(),
            selected_scope_ids=(),
            current_revision_token=self._snapshot_provider.revision_token(
                self.identity.revision_key
            ),
            current_snapshot=self._snapshot_provider.current_snapshot(),
            errors=errors,
        )
        return self._document_from_state(state, selection_mode=selection_mode)

    @staticmethod
    def _document_from_state(
        state: UiPipelineDebugSessionState,
        *,
        selection_mode: str,
    ) -> UiStateSurfaceDocument:
        payload = to_jsonable(state)
        if not isinstance(payload, dict):
            raise TypeError("Debug session state payload did not serialize to an object.")
        return UiStateSurfaceDocument(
            schema_version=state.schema_version,
            summary=state.summary,
            payload_schema=PIPELINE_DEBUG_SESSION_STATE_PAYLOAD_SCHEMA,
            payload=payload,
            current_revision_token=state.current_revision_token,
            current_snapshot=state.current_snapshot,
            selection_mode=selection_mode,
            selected_scope_ids=state.selected_scope_ids,
            unchanged=state.unchanged,
            warnings=state.warnings,
            errors=state.errors,
        )


class PipelineEditorStateSurfaceProvider(UiStateSurfaceProviderABC):
    """Pollable PipelineEditor state backed by shared manager widget hooks."""

    identity = PIPELINE_EDITOR_STATE_IDENTITY

    def __init__(
        self,
        manager,
        *,
        snapshot_provider: UiBridgeSnapshotProviderABC,
    ) -> None:
        self._manager = manager
        self._snapshot_provider = snapshot_provider

    def summary(self) -> UiStateSurfaceSummary:
        return UiStateSurfaceSummary(
            schema_version=SCHEMA_VERSION,
            identity=self.identity.as_surface_identity(),
            title=self.identity.title,
            widget_id=self.identity.widget_id,
            readable=True,
            supported_selection_modes=("selected", "all"),
            current_selection_count=len(self._manager.get_selected_items()),
            total_scope_count=len(self._manager.STATE_BINDING.items(self._manager)),
        )

    def read(self, request: UiStateSurfaceRequest) -> UiStateSurfaceDocument:
        selection_mode = request.resolved_selection_mode(UiCodeDocumentSelectionMode.ALL)
        try:
            state = self._state(selection_mode=selection_mode)
        except Exception as exc:
            return self._state_error(
                request,
                (AgentError.from_exception("ui_state_surface_read_failed", exc),),
            )

        revision_token = self._revision_token(state, selection_mode=selection_mode)
        state = UiPipelineEditorState(
            schema_version=state.schema_version,
            summary=state.summary,
            object_state_token=state.object_state_token,
            current_plate_scope_id=state.current_plate_scope_id,
            pipeline_scope_id=state.pipeline_scope_id,
            steps=state.steps,
            selected_scope_ids=state.selected_scope_ids,
            current_revision_token=revision_token,
            current_snapshot=self._snapshot_provider.current_snapshot(),
            unchanged=request.base_revision_token == revision_token,
            errors=state.errors,
            warnings=state.warnings,
        )
        return self._document_from_state(state, selection_mode=selection_mode)

    def _state(self, *, selection_mode: str) -> UiPipelineEditorState:
        items = tuple(self._manager.STATE_BINDING.items(self._manager))
        selected_items = tuple(self._manager.get_selected_items())
        selected_identity_ids = frozenset(id(item) for item in selected_items)
        selected_step_scope_ids = self._manager.selected_step_scope_ids()
        pipeline_scope_id = self._pipeline_scope_id()

        steps = tuple(
            self._step_state(
                item,
                index,
                selected=id(item) in selected_identity_ids,
            )
            for index, item in enumerate(items)
        )
        if selection_mode == UiCodeDocumentSelectionMode.SELECTED.value:
            steps = tuple(step for step in steps if step.selected)

        return UiPipelineEditorState(
            schema_version=SCHEMA_VERSION,
            summary=self.summary(),
            object_state_token=ObjectStateRegistry.get_token(),
            current_plate_scope_id=self._manager.current_plate or None,
            pipeline_scope_id=pipeline_scope_id,
            steps=steps,
            selected_scope_ids=selected_step_scope_ids,
            current_revision_token=self._snapshot_provider.revision_token(
                self.identity.revision_key
            ),
            current_snapshot=self._snapshot_provider.current_snapshot(),
        )

    def _step_state(
        self,
        step,
        index: int,
        *,
        selected: bool,
    ) -> UiPipelineEditorStepState:
        scope_id = self._manager._get_item_scope_id(step, index)
        object_state = (
            ObjectStateRegistry.get_by_scope(scope_id)
            if scope_id is not None
            else None
        )
        return UiPipelineEditorStepState(
            step_scope_id=scope_id,
            index=index,
            name=step.name,
            enabled=step.enabled,
            selected=selected,
            dirty=bool(object_state.dirty_fields) if object_state is not None else False,
            default_diff=(
                bool(object_state.signature_diff_fields)
                if object_state is not None
                else False
            ),
            description=step.description,
            debug_pause=step.debug_pause,
            function_names=self._function_names(step.function_spec()),
            function_ids=self._function_ids(step.function_spec()),
        )

    def _pipeline_scope_id(self) -> str | None:
        if not self._manager.current_plate:
            return None
        return PipelineScopeIdentity.from_plate_scope(
            self._manager.current_plate
        ).scope_id

    @classmethod
    def _function_names(cls, function_spec) -> tuple[str, ...]:
        if function_spec is None:
            return ()
        if isinstance(function_spec, list):
            entries = tuple(function_spec)
        else:
            entries = (function_spec,)
        names = []
        for entry in entries:
            function = cls._entry_function(entry)
            if function is not None:
                names.append(cls._function_name(function))
        return tuple(names)

    @staticmethod
    def _entry_function(entry):
        if isinstance(entry, tuple):
            return entry[0]
        if callable(entry):
            return entry
        return None

    @staticmethod
    def _function_name(function) -> str:
        try:
            return function.__name__
        except AttributeError:
            return function.__class__.__name__

    @classmethod
    def _function_ids(cls, function_spec) -> tuple[str, ...]:
        if function_spec is None:
            return ()
        if isinstance(function_spec, list):
            entries = tuple(function_spec)
        else:
            entries = (function_spec,)
        function_ids = []
        for entry in entries:
            function = cls._entry_function(entry)
            if function is None:
                continue
            function_id = cls._function_id(function)
            if function_id is not None:
                function_ids.append(function_id)
        return tuple(function_ids)

    @staticmethod
    def _function_id(function) -> str | None:
        try:
            return FunctionReferenceTransportAuthority.function_reference(
                function
            ).composite_key
        except Exception:
            return None

    def _revision_token(
        self,
        state: UiPipelineEditorState,
        *,
        selection_mode: str,
    ) -> str:
        step_parts = tuple(
            (
                step.step_scope_id,
                step.index,
                step.name,
                step.enabled,
                step.selected,
                step.dirty,
                step.default_diff,
                step.debug_pause,
                step.function_names,
                step.function_ids,
            )
            for step in state.steps
        )
        parts = (
            self.identity.revision_key,
            str(state.object_state_token),
            self._snapshot_provider.current_branch_head_snapshot_id(),
            str(ObjectStateRegistry.get_current_snapshot_index()),
            selection_mode,
            state.current_plate_scope_id,
            state.pipeline_scope_id,
            state.selected_scope_ids,
            step_parts,
        )
        return hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()

    def _state_error(
        self,
        request: UiStateSurfaceRequest,
        errors: tuple[AgentError, ...],
    ) -> UiStateSurfaceDocument:
        selection_mode = request.resolved_selection_mode(UiCodeDocumentSelectionMode.ALL)
        state = UiPipelineEditorState(
            schema_version=SCHEMA_VERSION,
            summary=self.summary(),
            object_state_token=ObjectStateRegistry.get_token(),
            current_plate_scope_id=None,
            pipeline_scope_id=None,
            steps=(),
            selected_scope_ids=(),
            current_revision_token=self._snapshot_provider.revision_token(
                self.identity.revision_key
            ),
            current_snapshot=self._snapshot_provider.current_snapshot(),
            errors=errors,
        )
        return self._document_from_state(state, selection_mode=selection_mode)

    @staticmethod
    def _document_from_state(
        state: UiPipelineEditorState,
        *,
        selection_mode: str,
    ) -> UiStateSurfaceDocument:
        payload = to_jsonable(state)
        if not isinstance(payload, dict):
            raise TypeError("PipelineEditor state payload did not serialize to an object.")
        return UiStateSurfaceDocument(
            schema_version=state.schema_version,
            summary=state.summary,
            payload_schema=PIPELINE_EDITOR_STATE_PAYLOAD_SCHEMA,
            payload=payload,
            current_revision_token=state.current_revision_token,
            current_snapshot=state.current_snapshot,
            selection_mode=selection_mode,
            selected_scope_ids=state.selected_scope_ids,
            unchanged=state.unchanged,
            warnings=state.warnings,
            errors=state.errors,
        )


class PipelineEditorBridgeProviderSet(UiBridgeProviderSetABC):
    """Register PipelineEditor surfaces with a UI bridge registry."""

    registry_key = PIPELINE_EDITOR_WIDGET_ID

    def __init__(self, manager) -> None:
        self._manager = manager

    def register(self, context: UiBridgeRegistrationContext) -> None:
        context.registry.register_state_surface_provider(
            PipelineEditorStateSurfaceProvider(
                self._manager,
                snapshot_provider=context.snapshot_provider,
            )
        )
        context.registry.register_state_surface_provider(
            PipelineDebugSessionStateSurfaceProvider(
                self._manager,
                snapshot_provider=context.snapshot_provider,
            )
        )
        context.registry.register_action_provider(
            PipelineEditorActionProvider(self._manager)
        )
        context.registry.register_action_provider(
            PipelineDebugToolbarActionProvider(self._manager)
        )
