"""PipelineEditor provider set for the PyQt UI bridge."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
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
    UiMutationReceipt,
    UiPipelineEditorState,
    UiPipelineEditorStepState,
    UiStateSurfaceDocument,
    UiStateSurfaceRequest,
    UiStateSurfaceSummary,
)
from openhcs.agent.ui_bridge_identities import (
    PipelineDebugToolbarWidgetIdentity,
    PipelineEditorStateSurfaceIdentityDeclaration,
    PipelineEditorWidgetIdentity,
    PlateManagerStateSurfaceIdentityDeclaration,
)
from openhcs.core.debug import DebugCommand, DebugCommandType
from openhcs.agent.serialization import to_jsonable
from openhcs.config_framework.object_state import ObjectStateRegistry
from openhcs.core.function_reference import FunctionReferenceTransportAuthority
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
from openhcs.pyqt_gui.widgets.debug_toolbar import (
    DebugToolbarAuxiliaryAction,
    DebugToolbarWidget,
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
PIPELINE_EDITOR_STATE_PAYLOAD_SCHEMA = "openhcs.ui.pipeline_editor_state.v1"
PLATE_MANAGER_STATE_SURFACE_ID = PlateManagerStateSurfaceIdentityDeclaration.require_value()
PIPELINE_EDITOR_WIDGET_ID = PipelineEditorWidgetIdentity.require_value()
PIPELINE_DEBUG_TOOLBAR_WIDGET_ID = PipelineDebugToolbarWidgetIdentity.require_value()
PIPELINE_EDITOR_STATE_IDENTITY = UiStateSurfaceProviderIdentity.from_declaration(
    PipelineEditorStateSurfaceIdentityDeclaration,
    title=PIPELINE_EDITOR_STATE_TITLE,
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
    )

    def __init__(self, manager) -> None:
        self._manager = manager

    def catalog(self) -> UiActionCatalog:
        return UiActionCatalog(
            schema_version=SCHEMA_VERSION,
            actions=tuple(self.summary(action_id) for action_id in self._action_ids()),
        )

    def summary(self, action_id: str) -> UiActionSummary:
        title, side_effects, confirmation_required = self._action_declaration(action_id)
        target_scope_ids = self._target_scope_ids()
        availability_error = self._availability_error(action_id)
        return UiActionSummary(
            schema_version=SCHEMA_VERSION,
            identity=UiActionIdentity(
                widget_id=self.identity.widget_id,
                action_id=action_id,
            ),
            title=title,
            enabled=availability_error is None,
            disabled_error=availability_error,
            invocation_mode="sync",
            side_effects=side_effects,
            confirmation_required=confirmation_required,
            selection_mode="current_pipeline",
            current_selection_count=len(target_scope_ids),
            target_scope_ids=target_scope_ids,
            selection_revision_token=self._selection_revision_token(),
            related_state_surface_ids=self._related_state_surface_ids,
        )

    def invoke(self, request: UiActionInvokeRequest) -> UiActionInvokeResult:
        guard_error = self._guard_error(request)
        if guard_error is not None:
            return self._invoke_error(request, guard_error)

        try:
            command_type = self._command_type(request.action_id)
            if command_type is not None:
                self._manager.debug_workflow.handle_command(DebugCommand(command_type))
            else:
                self._auxiliary_action(request.action_id)
                self._manager.debug_workflow.show_runtime_inspection()
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
        command_ids = tuple(
            spec.command_type.value for spec in DebugToolbarWidget.command_specs()
        )
        auxiliary_ids = tuple(
            spec.action_type.value
            for spec in DebugToolbarWidget.AUXILIARY_ACTION_SPECS
        )
        return (*command_ids, *auxiliary_ids)

    @staticmethod
    def _action_declaration(
        action_id: str,
    ) -> tuple[str, tuple[str, ...], bool]:
        command_type = PipelineDebugToolbarActionProvider._command_type(action_id)
        if command_type is not None:
            spec = DebugToolbarWidget.command_spec(command_type)
            return spec.label, spec.side_effects, spec.confirmation_required
        auxiliary_action = PipelineDebugToolbarActionProvider._auxiliary_action(
            action_id
        )
        spec = DebugToolbarWidget.auxiliary_action_spec(auxiliary_action)
        return spec.label, spec.side_effects, spec.confirmation_required

    @staticmethod
    def _command_type(action_id: str) -> DebugCommandType | None:
        try:
            command_type = DebugCommandType(action_id)
        except ValueError:
            return None
        toolbar_command_types = {
            spec.command_type for spec in DebugToolbarWidget.command_specs()
        }
        if command_type not in toolbar_command_types:
            raise ValueError(f"Debug command is not exposed by the toolbar: {action_id}")
        return command_type

    @staticmethod
    def _auxiliary_action(action_id: str) -> DebugToolbarAuxiliaryAction:
        return DebugToolbarAuxiliaryAction(action_id)

    def _availability_error(self, action_id: str) -> AgentError | None:
        if not self._target_scope_ids():
            return AgentError(
                code="debug_target_required",
                message="Debug toolbar actions require an initialized current plate.",
                hint=(
                    "Inspect plate_manager.state and pipeline_editor.state; add and "
                    "initialize a plate before invoking debug controls."
                ),
            )

        toolbar = self._manager.debug_toolbar
        command_type = self._command_type(action_id)
        if command_type is not None:
            spec = DebugToolbarWidget.command_spec(command_type)
            if (
                spec.requires_active_debug_session
                and self._manager.debug_session_state is None
            ):
                return AgentError(
                    code="debug_session_required",
                    message=f"{spec.label} requires an active debug session.",
                    hint=(
                        "Run or step the compiled pipeline in debug mode before "
                        f"invoking {spec.label!r}."
                    ),
                )
            if (
                command_type not in self._manager.DEBUG_COMMAND_ROUTES
                or toolbar is None
                or not toolbar.command_enabled(command_type)
            ):
                return self._disabled_error(action_id)
            return None

        auxiliary_action = self._auxiliary_action(action_id)
        if auxiliary_action is DebugToolbarAuxiliaryAction.RUNTIME_VALUES:
            if self._manager.debug_session_state is None:
                return AgentError(
                    code="debug_session_required",
                    message="Runtime inspection requires an active debug session.",
                    hint="Run or step the pipeline in debug mode before inspecting runtime values.",
                )
        if toolbar is None or not toolbar.auxiliary_action_enabled(auxiliary_action):
            return self._disabled_error(action_id)
        return None

    def _guard_error(self, request: UiActionInvokeRequest) -> AgentError | None:
        try:
            self._action_declaration(request.action_id)
        except Exception as exc:
            return AgentError.from_exception("unknown_ui_action", exc)

        target_scope_ids = self._target_scope_ids()
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
        availability_error = self._availability_error(request.action_id)
        if availability_error is not None:
            return availability_error
        summary = self.summary(request.action_id)
        if summary.confirmation_required and request.confirmation_is_required():
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

    def _disabled_error(self, action_id: str) -> AgentError:
        return AgentError(
            code="ui_action_disabled",
            message=f"{self.identity.widget_id} action {action_id!r} is disabled.",
            hint=(
                "Inspect pipeline_editor.state and plate_manager.state; debug "
                "controls require an initialized and compiled current plate."
            ),
        )

    def _target_scope_ids(self) -> tuple[str, ...]:
        if not self._manager.current_plate:
            return ()
        return (
            PipelineScopeIdentity.from_plate_scope(
                self._manager.current_plate
            ).scope_id,
        )

    def _selection_revision_token(self) -> str:
        session = self._manager.debug_session_state
        session_id = None if session is None else session.debug_session_id
        parts = (
            self.identity.widget_id,
            self._action_ids(),
            self._target_scope_ids(),
            ObjectStateRegistry.get_token(),
            session_id,
        )
        return hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()


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
        context.registry.register_action_provider(
            PipelineEditorActionProvider(self._manager)
        )
        context.registry.register_action_provider(
            PipelineDebugToolbarActionProvider(self._manager)
        )
