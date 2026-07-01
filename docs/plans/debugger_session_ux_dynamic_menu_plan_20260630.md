# Debugger Session UX and Dynamic Action Surface Plan

Date: 2026-06-30

## Status

Historical/superseded plan. Do not implement directly.

The canonical debugger UX/runtime direction now lives in:

- `docs/plans/debugger_session_ux_rework_plan_20260630.md`
- `docs/plans/debugger_runtime_projection_api_plan_20260701.md`

This file is retained only as background for why the action surface moved out of
`DebugToolbarWidget`. Several API drafts below predate the current SSOT rules
and contain patterns that are no longer allowed: module-level
`*_STATE_IDENTITY` constants, auxiliary action enums parallel to command
declarations, widget-specific action protocols, and flat DTO field mirroring.
If this file conflicts with the newer plans, the newer plans win.

## Problem

The current pipeline debugger UI exposes commands, but it does not expose the
debugger as a coherent session. Users see `Debug`, `Step`, `Pause`, `Restart`,
and an `Inspect` menu, but the UI does not make the active `DebugSession`,
cursor, selected source group, runtime store, snapshots, or available command
set obvious.

The recent compiled/session gating fixed one correctness problem, but the UX is
still structurally weak:

- start-debug controls, active-session controls, and inspection controls share
  one flat toolbar;
- `Inspect` is a menu bucket rather than an inspector entry point;
- empty runtime values render as an empty dark region instead of a meaningful
  state;
- MCP can list actions, but there is no first-class debug-session state surface;
- the Qt menu and MCP action provider both project the toolbar, but there is no
  single richer session/action view model for the UI to render.

This plan turns the debug toolbar into a typed session projection. The runtime
authorities remain in `openhcs.core.debug`; the UI bridge and Qt widget consume
one OpenHCS GUI projection instead of each re-deriving availability.

## Existing Authorities

### Runtime

Keep these as runtime/session authorities:

- `openhcs.core.debug.DebugCommandType`
- `openhcs.core.debug.DebugCommandPolicyRow`
- `openhcs.core.debug.DebugSession`
- `openhcs.core.debug.DebugCursor`
- `openhcs.core.debug.DebugPausedWorkerStatus`
- `openhcs.core.debug.DebugExecutionConfig`
- `openhcs.core.debug.DebugInvocationExecutionStrategy`
- `openhcs.core.debug.DebugStepStopStrategy`
- `openhcs.core.debug_views.DebugViewModel`
- `openhcs.core.debug_views.DebugViewSection`
- `openhcs.core.debug_views.DebugViewTable`

Runtime inspection continues to return `DebugViewModel`. Do not add an
MCP-specific runtime-values DTO.

### GUI Lifecycle

Keep these as GUI lifecycle authorities:

- `PlateManagerWidget._active_debug_sessions`
- `PlateManagerWidget.action_run_debug_plate(...)`
- `PlateManagerWidget.action_inspect_debug_runtime(...)`
- `PlateManagerWidget._clear_debug_session_for_plate(...)`
- `PipelineEditorWidget.debug_session_state`
- `PipelineEditorDebugWorkflow`
- `DebugWorkflowService`

The Plate Manager owns active debug sessions and terminal cleanup. The Pipeline
Editor presents and routes the selected plate's current session. The new code
must not create another debug-session registry.

### UI Bridge

Reuse existing bridge infrastructure:

- `UiActionProviderABC`
- `UiStateSurfaceProviderABC`
- `UiActionSummary`
- `UiStateSurfaceDocument`
- `UiStateSurfaceIdentityDeclarationBase`
- `PipelineDebugToolbarWidgetIdentity`
- `PipelineEditorStateSurfaceIdentityDeclaration`
- `UiStateSurfacePayloadRenderer`

Add a debug-session state surface rather than a bespoke MCP tool.

### Toolbar Action Declarations

The current code keeps toolbar action facts on `DebugToolbarWidget` as three
class tuples:

- `BUTTON_SPECS`
- `MENU_ACTION_SPECS`
- `AUXILIARY_ACTION_SPECS`

That works locally, but it creates the wrong dependency direction for the new
projection: the projection cannot import the widget if the widget also imports
the projection. Promote those specs into nominal action declaration classes in a
separate module, then have the widget, bridge, and MCP state surface query the
AutoRegisterMeta-backed declaration registry.

After the migration, `DebugToolbarWidget` owns Qt rendering only. It must not
own the semantic list of debugger actions.

## pyqt-reactive Reuse

Use these directly:

- `pyqt_reactive.widgets.shared.button_panel.ButtonPanel`
  - good for primary command buttons;
  - OpenHCS adapts typed debug declarations into its tuple input.
- `pyqt_reactive.widgets.shared.FormWindowActionHeader`
  - good for a compact debugger header with grouped actions.
- `pyqt_reactive.widgets.shared.DetachableActionBar`
  - useful if the debug inspector can render actions in a parent header or as a
    standalone panel.
- `pyqt_reactive.widgets.shared.action_tabbed_window_body.ActionTabbedWindowBody`
  - best existing primitive for a richer Debug Inspector with tabs and
    per-tab actions.
- `pyqt_reactive.widgets.shared.responsive_layout_widgets.StagedWrapLayout`
  - already used by `FormWindowActionHeader`; use indirectly for responsive
    wrapping.
- `pyqt_reactive.widgets.shared.scope_style_applier` and related scope styling
  helpers
  - useful for visually tying the debugger to the current plate/pipeline scope.

Use cautiously:

- `pyqt_reactive.widgets.status_indicator.StatusIndicator`
  - it is check-function oriented, not state-model oriented. Reuse visual ideas
    only; do not drive debug state through its async check contract.
- `pyqt_reactive.widgets.shared.button_factory.make_accented_button`
  - useful for button visuals, but it must not own debugger semantics.

Do not add a generic menu framework to pyqt-reactive in the first pass. Build a
small OpenHCS projection layer first. Only move it into pyqt-reactive if another
widget needs the same typed-action-to-Qt projection.

## API Draft

### Nominal Pipeline Debug Action Declarations

Add:

`openhcs/pyqt_gui/widgets/shared/services/pipeline_debug_actions.py`

This module owns debugger action declarations. It replaces the existing
`PipelineDebugCommandRoute`/`DEBUG_COMMAND_ROUTES` table rather than sitting next
to it. It is imported by the Pipeline Editor workflow, Qt widget, shared session
projection, and UI bridge. It does not import the widget, so there is no
circular dependency.

Use `AutoRegisterMeta` so adding or changing a debugger action means editing one
declaration class, not updating a command route map, widget list, bridge list,
MCP list, and tests separately.

Move the existing `dispatch_pipeline_debug_*` functions from
`pipeline_editor.py` into this module above the declaration classes. Keep their
behavior unchanged in the first pass; only their owner changes.

Move this exact set:

- `dispatch_pipeline_debug_run_command`
- `dispatch_pipeline_debug_toggle_command`
- `dispatch_pipeline_debug_step_command`
- `dispatch_pipeline_debug_run_to_pause_command`
- `dispatch_pipeline_debug_restart_command`
- `dispatch_pipeline_debug_choose_source_group_command`
- `dispatch_pipeline_debug_random_source_group_command`
- `dispatch_pipeline_debug_stop_command`

```python
from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, ClassVar, Protocol

from metaclass_registry import AutoRegisterMeta

from openhcs.core.debug import DebugCommand, DebugCommandType

if TYPE_CHECKING:
    from openhcs.pyqt_gui.widgets.shared.services.debug_session_projection import (
        PipelineDebugSessionContext,
    )


class DebugToolbarWorkflow(Protocol):
    def handle_command(self, command: DebugCommand) -> None: ...
    def show_runtime_inspection(self) -> None: ...
    def run_command(self, command_type: DebugCommandType) -> None: ...
    def stop_command(self) -> None: ...


class DebugStatusEmitter(Protocol):
    def emit(self, message: str) -> None: ...


class PipelineDebugEditor(Protocol):
    """Minimal editor protocol consumed by debug action dispatch."""

    debug_workflow: DebugToolbarWorkflow
    status_message: DebugStatusEmitter


class DebugActionPlacement(str, Enum):
    PRIMARY = "primary"
    SESSION = "session"
    INSPECTOR = "inspector"


class DebugToolbarAuxiliaryAction(str, Enum):
    RUNTIME_VALUES = "runtime_values"


class PipelineDebugActionDeclarationBase(ABC, metaclass=AutoRegisterMeta):
    """Nominal declaration for one Pipeline Editor debug action."""

    __registry_key__ = "identity"
    __skip_if_no_key__ = True

    identity: ClassVar[DebugCommandType | DebugToolbarAuxiliaryAction | None] = None
    toolbar_placement: ClassVar[DebugActionPlacement | None] = None
    toolbar_order: ClassVar[int | None] = None
    label: ClassVar[str | None] = None
    tooltip: ClassVar[str | None] = None
    side_effects: ClassVar[tuple[str, ...]] = ()
    confirmation_required: ClassVar[bool] = False
    requires_active_debug_session: ClassVar[bool] = False

    @classmethod
    def action_id(cls) -> str:
        if cls.identity is None:
            raise ValueError(f"{cls.__name__} does not declare an identity.")
        return cls.identity.value

    @classmethod
    def is_toolbar_action(cls) -> bool:
        return cls.toolbar_placement is not None and cls.toolbar_order is not None

    @classmethod
    def toolbar_actions(
        cls,
    ) -> tuple[type["PipelineDebugActionDeclarationBase"], ...]:
        return tuple(
            sorted(
                (
                    declaration
                    for declaration in cls.__registry__.values()
                    if declaration.is_toolbar_action()
                ),
                key=lambda declaration: declaration.toolbar_order,
            )
        )

    @classmethod
    def label_for(cls, context: "PipelineDebugSessionContext") -> str:
        if cls.label is None:
            raise ValueError(f"{cls.__name__} does not declare a toolbar label.")
        return cls.label

    @classmethod
    def invoke(cls, workflow: DebugToolbarWorkflow) -> None:
        raise NotImplementedError(f"{cls.__name__} does not implement invoke().")


class PipelineDebugCommandActionDeclaration(PipelineDebugActionDeclarationBase):
    identity: ClassVar[DebugCommandType]
    side_effects: ClassVar[tuple[str, ...]] = (
        "starts_or_controls_debug_execution",
    )
    confirmation_required: ClassVar[bool] = True
    dispatch: ClassVar[Callable[[PipelineDebugEditor], None]]

    @classmethod
    def command_type(cls) -> DebugCommandType:
        return cls.identity

    @classmethod
    def invoke(cls, workflow: DebugToolbarWorkflow) -> None:
        workflow.handle_command(DebugCommand(cls.command_type()))

    @classmethod
    def dispatch_editor(cls, editor: PipelineDebugEditor) -> None:
        cls.dispatch(editor)


class PipelineDebugAuxiliaryActionDeclaration(PipelineDebugActionDeclarationBase):
    identity: ClassVar[DebugToolbarAuxiliaryAction]
    side_effects: ClassVar[tuple[str, ...]] = ("opens_debug_runtime_inspector",)
    confirmation_required: ClassVar[bool] = False


class StartOrContinueDebugAction(PipelineDebugCommandActionDeclaration):
    identity = DebugCommandType.RUN
    toolbar_placement = DebugActionPlacement.PRIMARY
    toolbar_order = 10
    label = "Debug"
    tooltip = "Start or continue debug execution for the selected plate"
    dispatch = dispatch_pipeline_debug_run_command

    @classmethod
    def label_for(cls, context: "PipelineDebugSessionContext") -> str:
        return "Continue" if context.has_active_session else "Start Debug"


class StepDebugAction(PipelineDebugCommandActionDeclaration):
    identity = DebugCommandType.STEP
    toolbar_placement = DebugActionPlacement.PRIMARY
    toolbar_order = 20
    label = "Step"
    tooltip = "Run one debug step"
    dispatch = dispatch_pipeline_debug_step_command


class RunToPauseDebugAction(PipelineDebugCommandActionDeclaration):
    identity = DebugCommandType.RUN_TO_PAUSE
    toolbar_placement = DebugActionPlacement.PRIMARY
    toolbar_order = 30
    label = "Pause"
    tooltip = "Run until the next pause marker"
    dispatch = dispatch_pipeline_debug_run_to_pause_command

    @classmethod
    def label_for(cls, context: "PipelineDebugSessionContext") -> str:
        return "Run to Pause Marker"


class RestartDebugAction(PipelineDebugCommandActionDeclaration):
    identity = DebugCommandType.RESTART
    toolbar_placement = DebugActionPlacement.PRIMARY
    toolbar_order = 40
    label = "Restart"
    tooltip = "Restart the current debug session"
    requires_active_debug_session = True
    dispatch = dispatch_pipeline_debug_restart_command

    @classmethod
    def label_for(cls, context: "PipelineDebugSessionContext") -> str:
        return "Restart From Cursor"


class InspectRuntimeValuesAction(PipelineDebugAuxiliaryActionDeclaration):
    identity = DebugToolbarAuxiliaryAction.RUNTIME_VALUES
    toolbar_placement = DebugActionPlacement.INSPECTOR
    toolbar_order = 50
    label = "Runtime values"
    tooltip = "Inspect live runtime values for the paused debug worker"
    requires_active_debug_session = True

    @classmethod
    def label_for(cls, context: "PipelineDebugSessionContext") -> str:
        return "Inspect Data"

    @classmethod
    def invoke(cls, workflow: DebugToolbarWorkflow) -> None:
        workflow.show_runtime_inspection()


class ChooseSourceGroupDebugAction(PipelineDebugCommandActionDeclaration):
    identity = DebugCommandType.CHOOSE_SOURCE_GROUP
    toolbar_placement = DebugActionPlacement.SESSION
    toolbar_order = 60
    label = "Choose source group"
    tooltip = "Choose a well/image set for debug execution"
    dispatch = dispatch_pipeline_debug_choose_source_group_command


class StopDebugSessionAction(PipelineDebugCommandActionDeclaration):
    identity = DebugCommandType.STOP
    toolbar_placement = DebugActionPlacement.SESSION
    toolbar_order = 70
    label = "Stop debug session"
    tooltip = "Stop the active debug execution"
    requires_active_debug_session = True
    dispatch = dispatch_pipeline_debug_stop_command


class ToggleDebugModeAction(PipelineDebugCommandActionDeclaration):
    identity = DebugCommandType.TOGGLE
    dispatch = dispatch_pipeline_debug_toggle_command


class RandomSourceGroupDebugAction(PipelineDebugCommandActionDeclaration):
    identity = DebugCommandType.RANDOM_SOURCE_GROUP
    dispatch = dispatch_pipeline_debug_random_source_group_command
```

If `AutoRegisterMeta` import or registry typing differs, adapt to the existing
metaclass pattern in `openhcs.agent.ui_bridge_identities`. Do not replace this
with a list of strings.

Move the existing dispatch functions and `PipelineDebugCommandRoute` authority
out of `pipeline_editor.py` during this step. `PipelineEditorDebugWorkflow`
must dispatch through the declaration registry:

```python
class PipelineEditorDebugWorkflow:
    def handle_command(self, command: DebugCommand) -> None:
        declaration = PipelineDebugActionDeclarationBase.__registry__.get(
            command.command_type
        )
        if declaration is None or not issubclass(
            declaration,
            PipelineDebugCommandActionDeclaration,
        ):
            raise RuntimeError(
                f"Unhandled debug command route: {command.command_type.value}"
            )
        declaration.dispatch_editor(self.editor)
```

After this lands, delete `PipelineDebugCommandRoute` and
`PipelineEditorWidget.DEBUG_COMMAND_ROUTES`. They are the route mirror this
refactor is intended to remove.

Update production imports and tests to import moved semantic declarations from
`pipeline_debug_actions.py`. Do not leave re-export shims in
`debug_toolbar.py`; that would keep a second public owner for debugger action
semantics.

### Shared GUI Projection Module

Add:

`openhcs/pyqt_gui/widgets/shared/services/debug_session_projection.py`

This module is Qt-light and bridge-safe. It imports runtime debug state and the
nominal pipeline debug action declarations. It must not import
`DebugToolbarWidget`.
The core projection uses local disabled-reason types so the Qt widget does not
depend on agent DTOs.

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from openhcs.core.debug import DebugSession
from openhcs.pyqt_gui.widgets.shared.services.pipeline_debug_actions import (
    DebugActionPlacement,
    PipelineDebugActionDeclarationBase,
)


class DebugSessionPhase(str, Enum):
    NO_TARGET = "no_target"
    UNINITIALIZED = "uninitialized"
    UNCOMPILED = "uncompiled"
    READY = "ready"
    ACTIVE = "active"
    TERMINAL = "terminal"


class DebugActionDisabledCode(str, Enum):
    TARGET_REQUIRED = "debug_target_required"
    INITIALIZED_PLATE_REQUIRED = "debug_initialized_plate_required"
    COMPILED_PLATE_REQUIRED = "debug_compiled_plate_required"
    ACTIVE_SESSION_REQUIRED = "debug_session_required"
    UNSUPPORTED_COMMAND = "debug_command_unsupported"


@dataclass(frozen=True, slots=True)
class DebugActionDisabledReason:
    code: DebugActionDisabledCode
    message: str
    hint: str


@dataclass(frozen=True, slots=True)
class PipelineDebugTargetState:
    current_plate_scope_id: str | None
    pipeline_scope_id: str | None
    initialized: bool
    compiled: bool
    terminal_status: str | None = None

    @property
    def has_target(self) -> bool:
        return self.current_plate_scope_id is not None


@dataclass(frozen=True, slots=True)
class PipelineDebugSessionContext:
    target: PipelineDebugTargetState
    session: DebugSession | None
    manager_execution_state: str

    @property
    def phase(self) -> DebugSessionPhase:
        if not self.target.has_target:
            return DebugSessionPhase.NO_TARGET
        if not self.target.initialized:
            return DebugSessionPhase.UNINITIALIZED
        if not self.target.compiled:
            return DebugSessionPhase.UNCOMPILED
        if self.session is not None:
            return DebugSessionPhase.ACTIVE
        if self.target.terminal_status is not None:
            return DebugSessionPhase.TERMINAL
        return DebugSessionPhase.READY

    @property
    def has_active_session(self) -> bool:
        return self.session is not None

    @property
    def base_debug_controls_available(self) -> bool:
        return (
            self.target.has_target
            and self.target.initialized
            and self.target.compiled
        )

    @property
    def runtime_inspection_available(self) -> bool:
        return self.base_debug_controls_available and self.has_active_session


@dataclass(frozen=True, slots=True)
class DebugActionRenderModel:
    declaration: type[PipelineDebugActionDeclarationBase]
    label: str
    tooltip: str
    enabled: bool
    disabled_reason: DebugActionDisabledReason | None
    confirmation_required: bool
    side_effects: tuple[str, ...]

    @property
    def action_id(self) -> str:
        return self.declaration.action_id()

    @property
    def placement(self) -> DebugActionPlacement:
        placement = self.declaration.toolbar_placement
        if placement is None:
            raise ValueError(
                f"{self.declaration.__name__} is not exposed in the debug toolbar."
            )
        return placement


class DebugToolbarActionProjector:
    """Project declared debug actions into one Qt/MCP render model."""

    @classmethod
    def declarations(cls) -> tuple[type[PipelineDebugActionDeclarationBase], ...]:
        return PipelineDebugActionDeclarationBase.toolbar_actions()

    @classmethod
    def render_models(
        cls,
        context: PipelineDebugSessionContext,
    ) -> tuple[DebugActionRenderModel, ...]:
        return tuple(cls.render_model(declaration, context) for declaration in cls.declarations())

    @classmethod
    def render_model(
        cls,
        declaration: type[PipelineDebugActionDeclarationBase],
        context: PipelineDebugSessionContext,
    ) -> DebugActionRenderModel:
        if declaration.tooltip is None:
            raise ValueError(
                f"{declaration.__name__} is missing a toolbar tooltip."
            )
        disabled_reason = cls.disabled_reason(declaration, context)
        return DebugActionRenderModel(
            declaration=declaration,
            label=declaration.label_for(context),
            tooltip=declaration.tooltip,
            enabled=disabled_reason is None,
            disabled_reason=disabled_reason,
            confirmation_required=declaration.confirmation_required,
            side_effects=declaration.side_effects,
        )

    @classmethod
    def disabled_reason(
        cls,
        declaration: type[PipelineDebugActionDeclarationBase],
        context: PipelineDebugSessionContext,
    ) -> DebugActionDisabledReason | None:
        if not context.target.has_target:
            return DebugActionDisabledReason(
                DebugActionDisabledCode.TARGET_REQUIRED,
                "Debug controls require a selected plate.",
                "Add or select a plate before invoking debug controls.",
            )
        if not context.target.initialized:
            return DebugActionDisabledReason(
                DebugActionDisabledCode.INITIALIZED_PLATE_REQUIRED,
                "Debug controls require an initialized plate.",
                "Initialize the selected plate before invoking debug controls.",
            )
        if not context.target.compiled:
            return DebugActionDisabledReason(
                DebugActionDisabledCode.COMPILED_PLATE_REQUIRED,
                "Debug controls require a compiled plate.",
                "Compile the selected plate before invoking debug controls.",
            )
        if declaration.requires_active_debug_session and not context.has_active_session:
            return DebugActionDisabledReason(
                DebugActionDisabledCode.ACTIVE_SESSION_REQUIRED,
                f"{declaration.label} requires an active debug session.",
                "Run or step the compiled pipeline in debug mode first.",
            )
        return None
```

Why this is not semantic mirroring:

- command identity still comes from `DebugCommandType`;
- command/menu/auxiliary ownership lives on nominal action declaration classes;
- action iteration is derived from the AutoRegisterMeta registry;
- session existence still comes from `PipelineEditorWidget.debug_session_state`;
- plate readiness still comes from existing Pipeline Editor / Plate Manager
  state;
- this module owns only the projection from those authorities into one render
  model.

### Pipeline Editor Context API

Add one public context producer on `PipelineEditorWidget`:

`openhcs/pyqt_gui/widgets/pipeline_editor.py`

```python
from openhcs.pyqt_gui.widgets.shared.services.debug_session_projection import (
    PipelineDebugSessionContext,
    PipelineDebugTargetState,
)
from openhcs.pyqt_gui.services.plate_scope_identity import PipelineScopeIdentity


class PipelineEditorWidget(...):
    ...

    def debug_session_context(self) -> PipelineDebugSessionContext:
        current_plate_scope_id = self.current_plate or None
        pipeline_scope_id = (
            PipelineScopeIdentity.from_plate_scope(self.current_plate).scope_id
            if self.current_plate
            else None
        )
        target = PipelineDebugTargetState(
            current_plate_scope_id=current_plate_scope_id,
            pipeline_scope_id=pipeline_scope_id,
            initialized=self._is_current_plate_initialized(),
            compiled=self._is_current_plate_compiled(),
            terminal_status=self._current_plate_terminal_status(),
        )
        return PipelineDebugSessionContext(
            target=target,
            session=self.debug_session_state,
            manager_execution_state=(
                self.plate_manager.execution_state.value
                if self.plate_manager is not None
                else "unknown"
            ),
        )

    def _current_plate_terminal_status(self) -> str | None:
        if self.plate_manager is None or self.current_plate is None:
            return None
        terminal_status = self.plate_manager.plate_terminal_activity_status.terminal_status(
            self.current_plate
        )
        return None if terminal_status is None else terminal_status.value
```

This uses existing typed Plate Manager state:

- `execution_state: ManagerExecutionState`
- `plate_terminal_activity_status: ExecutionBatchRuntime`
- `ExecutionBatchRuntime.terminal_status(plate_path)`

Do not scan Qt table strings or duplicate plate-status labels.

Update `PipelineEditorWidget.update_button_states()` to call only:

```python
if self.debug_toolbar is not None:
    self.debug_toolbar.set_debug_session_context(self.debug_session_context())
```

Remove these from the Pipeline Editor once the toolbar consumes the context:

- `set_controls_enabled(...)`
- `set_debug_session_active(...)`
- `set_runtime_inspection_enabled(...)`

Do not leave compatibility shims after the refactor lands. Update tests in the
same commit so old setters are not kept as a second API.

### Toolbar API

Update:

`openhcs/pyqt_gui/widgets/debug_toolbar.py`

```python
from openhcs.core.debug import DebugCommandType
from openhcs.pyqt_gui.widgets.shared.services.debug_session_projection import (
    DebugActionRenderModel,
    DebugToolbarActionProjector,
    PipelineDebugSessionContext,
)
from openhcs.pyqt_gui.widgets.shared.services.pipeline_debug_actions import (
    DebugToolbarAuxiliaryAction,
)


class DebugToolbarWidget(QWidget):
    ...

    def set_debug_session_context(
        self,
        context: PipelineDebugSessionContext,
    ) -> None:
        self._debug_session_context = context
        self.set_action_models(DebugToolbarActionProjector.render_models(context))

    def set_action_models(
        self,
        models: tuple[DebugActionRenderModel, ...],
    ) -> None:
        self._action_models = {model.action_id: model for model in models}
        self._render_primary_buttons(models)
        self._render_inspector_actions(models)
        self._render_session_actions(models)

    def command_enabled(self, command_type: DebugCommandType) -> bool:
        model = self._action_models[command_type.value]
        return model.enabled

    def auxiliary_action_enabled(
        self,
        action_type: DebugToolbarAuxiliaryAction,
    ) -> bool:
        model = self._action_models[action_type.value]
        return model.enabled
```

Rendering rules:

- primary actions render through the existing `ButtonPanel`;
- inspector actions render at the front of the overflow/menu surface and become
  the main `Inspect Data` button in the later UX pass;
- session actions render after a separator;
- disabled tooltips append the typed disabled reason message;
- Qt code does not create MCP `AgentError`.

### Agent DTO Additions

Add typed payload records to:

`openhcs/agent/dto/ui_bridge.py`

```python
@dataclass(frozen=True, slots=True)
class UiDebugActionState:
    action_id: str
    title: str
    placement: str
    enabled: bool
    invocation_mode: str
    side_effects: tuple[str, ...]
    confirmation_required: bool
    disabled_error: AgentError | None = None


@dataclass(frozen=True, slots=True)
class UiDebugCursorState:
    step_index: int
    step_scope_id: str | None
    group_key: str | None
    invocation_key: str | None
    pattern_group_identity: str | None


@dataclass(frozen=True, slots=True)
class UiPipelineDebugSessionState(
    UiStateSurfaceEnvelope,
    UiCodeDocumentCurrentRevision,
    UiCurrentSnapshotState,
    SelectedScopeIdsCarrier,
):
    object_state_token: int
    current_plate_scope_id: str | None
    pipeline_scope_id: str | None
    phase: str
    manager_execution_state: str
    initialized: bool
    compiled: bool
    active_debug_session_id: str | None
    selected_source_group: str | None
    terminal_status: str | None
    runtime_view_available: bool
    cursor: UiDebugCursorState | None
    actions: tuple[UiDebugActionState, ...]
```

`phase` is a string emitted from `DebugSessionPhase.value`; do not duplicate a
separate agent-side phase enum. The agent DTO is a serialization shape, not a
semantic authority.

### State Surface Identity

Add a nominal identity declaration to:

`openhcs/agent/ui_bridge_identities.py`

```python
class PipelineDebugSessionStateSurfaceIdentityDeclaration(
    UiStateSurfaceIdentityDeclarationBase
):
    value = "pipeline_debug_toolbar.session"
    enum_member_name = "PIPELINE_DEBUG_SESSION"
    widget_identity = PipelineDebugToolbarWidgetIdentity
```

This is enough for generated identity enums because the file already uses
`AutoRegisterMeta`. Do not add a manual enum list.

### Pipeline Editor Bridge Provider

Update:

`openhcs/pyqt_gui/services/ui_bridge_pipeline_editor.py`

Add imports:

```python
from dataclasses import replace

from openhcs.agent.dto.ui_bridge import (
    UiDebugActionState,
    UiDebugCursorState,
    UiPipelineDebugSessionState,
)
from openhcs.agent.ui_bridge_identities import (
    PipelineDebugSessionStateSurfaceIdentityDeclaration,
)
from openhcs.pyqt_gui.widgets.shared.services.debug_session_projection import (
    DebugActionDisabledReason,
    DebugActionRenderModel,
    DebugToolbarActionProjector,
    PipelineDebugSessionContext,
)
```

Remove direct bridge imports of `DebugCommand`, `DebugCommandType`,
`DebugToolbarAuxiliaryAction`, and `DebugToolbarWidget` when the provider no
longer uses them.

Add constants:

```python
PIPELINE_DEBUG_SESSION_STATE_PAYLOAD_SCHEMA = (
    "openhcs.ui.pipeline_debug_session_state.v1"
)
PIPELINE_DEBUG_SESSION_STATE_IDENTITY = UiStateSurfaceProviderIdentity.from_declaration(
    PipelineDebugSessionStateSurfaceIdentityDeclaration,
    title="Pipeline debug session state",
)
```

Add a bridge mapper, local to this file or to the projection module if reused by
tests:

```python
def _agent_error_from_disabled_reason(
    reason: DebugActionDisabledReason,
) -> AgentError:
    return AgentError(
        code=reason.code.value,
        message=reason.message,
        hint=reason.hint,
    )


def _action_state_from_model(
    model: DebugActionRenderModel,
) -> UiDebugActionState:
    return UiDebugActionState(
        action_id=model.action_id,
        title=model.label,
        placement=model.placement.value,
        enabled=model.enabled,
        invocation_mode="sync",
        side_effects=model.side_effects,
        confirmation_required=model.confirmation_required,
        disabled_error=(
            None
            if model.disabled_reason is None
            else _agent_error_from_disabled_reason(model.disabled_reason)
        ),
    )
```

Update `PipelineDebugToolbarActionProvider` so it consumes the projector:

```python
class PipelineDebugToolbarActionProvider(UiActionProviderABC):
    _related_state_surface_ids = (
        PLATE_MANAGER_STATE_SURFACE_ID,
        PipelineEditorStateSurfaceIdentityDeclaration.require_value(),
        PipelineDebugSessionStateSurfaceIdentityDeclaration.require_value(),
    )

    def _models(self) -> tuple[DebugActionRenderModel, ...]:
        return DebugToolbarActionProjector.render_models(
            self._manager.debug_session_context()
        )

    def _model(self, action_id: str) -> DebugActionRenderModel:
        for model in self._models():
            if model.action_id == action_id:
                return model
        raise ValueError(f"Unknown debug toolbar action: {action_id}")

    def catalog(self) -> UiActionCatalog:
        return UiActionCatalog(
            schema_version=SCHEMA_VERSION,
            actions=tuple(
                self._summary_from_model(model) for model in self._models()
            ),
        )

    def summary(self, action_id: str) -> UiActionSummary:
        return self._summary_from_model(self._model(action_id))

    def _summary_from_model(self, model: DebugActionRenderModel) -> UiActionSummary:
        return UiActionSummary(
            schema_version=SCHEMA_VERSION,
            identity=UiActionIdentity(
                widget_id=self.identity.widget_id,
                action_id=model.action_id,
            ),
            title=model.label,
            enabled=model.enabled,
            disabled_error=(
                None
                if model.disabled_reason is None
                else _agent_error_from_disabled_reason(model.disabled_reason)
            ),
            invocation_mode="sync",
            side_effects=model.side_effects,
            confirmation_required=model.confirmation_required,
            selection_mode="current_pipeline",
            current_selection_count=len(self._target_scope_ids()),
            target_scope_ids=self._target_scope_ids(),
            selection_revision_token=self._selection_revision_token(),
            related_state_surface_ids=self._related_state_surface_ids,
        )
```

Delete from `PipelineDebugToolbarActionProvider` after this conversion:

- `_action_ids()`
- `_action_declaration(...)`
- `_availability_error(...)`
- `_disabled_error(...)`

Keep dispatch declaration-owned:

```python
def invoke(self, request: UiActionInvokeRequest) -> UiActionInvokeResult:
    model = self._model(request.action_id)
    guard_error = self._guard_error(request, model)
    if guard_error is not None:
        return self._invoke_error(request, guard_error)

    model.declaration.invoke(self._manager.debug_workflow)
    return self._accepted_result(request)

def _accepted_result(self, request: UiActionInvokeRequest) -> UiActionInvokeResult:
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

def _guard_error(
    self,
    request: UiActionInvokeRequest,
    model: DebugActionRenderModel,
) -> AgentError | None:
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
    if model.disabled_reason is not None:
        return _agent_error_from_disabled_reason(model.disabled_reason)
    if model.confirmation_required and request.confirmation_is_required():
        return AgentError(
            code="confirmation_required",
            message=(
                f"{self.identity.widget_id} action {request.action_id!r} mutates "
                "debug execution state; set require_confirmation=False to dispatch it."
            ),
        )
    return None

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
    action_parts = tuple(
        (
            model.action_id,
            model.label,
            model.enabled,
            None if model.disabled_reason is None else model.disabled_reason.code.value,
            model.confirmation_required,
            model.side_effects,
        )
        for model in self._models()
    )
    parts = (
        self.identity.widget_id,
        action_parts,
        self._target_scope_ids(),
        ObjectStateRegistry.get_token(),
        None if session is None else session.debug_session_id,
    )
    return hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()
```

The only string comparison left in the provider is selecting the action
by `action_id` supplied by the bridge request. That is an ABI lookup, not
semantic classification.

Add the state provider:

```python
class PipelineDebugSessionStateSurfaceProvider(UiStateSurfaceProviderABC):
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
            total_scope_count=len(target_scope_ids),
        )

    def read(self, request: UiStateSurfaceRequest) -> UiStateSurfaceDocument:
        selection_mode = request.resolved_selection_mode(UiCodeDocumentSelectionMode.ALL)
        context = self._manager.debug_session_context()
        models = DebugToolbarActionProjector.render_models(context)
        state = self._state(context=context, models=models)
        revision_token = self._revision_token(
            state,
            selection_mode=selection_mode,
        )
        state = replace(
            state,
            current_revision_token=revision_token,
            current_snapshot=self._snapshot_provider.current_snapshot(),
            unchanged=request.base_revision_token == revision_token,
        )
        return self._document_from_state(state, selection_mode=selection_mode)

    def _state(
        self,
        *,
        context: PipelineDebugSessionContext,
        models: tuple[DebugActionRenderModel, ...],
    ) -> UiPipelineDebugSessionState:
        target = context.target
        session = context.session
        return UiPipelineDebugSessionState(
            schema_version=SCHEMA_VERSION,
            summary=self.summary(),
            object_state_token=ObjectStateRegistry.get_token(),
            current_plate_scope_id=target.current_plate_scope_id,
            pipeline_scope_id=target.pipeline_scope_id,
            phase=context.phase.value,
            manager_execution_state=context.manager_execution_state,
            initialized=target.initialized,
            compiled=target.compiled,
            active_debug_session_id=(
                None if session is None else session.debug_session_id
            ),
            selected_source_group=self._selected_source_group(session),
            terminal_status=target.terminal_status,
            runtime_view_available=context.runtime_inspection_available,
            cursor=self._cursor_state(session),
            actions=tuple(_action_state_from_model(model) for model in models),
            selected_scope_ids=self._target_scope_ids(),
            current_revision_token=self._snapshot_provider.revision_token(
                self.identity.revision_key
            ),
            current_snapshot=self._snapshot_provider.current_snapshot(),
        )

    @staticmethod
    def _selected_source_group(session: DebugSession | None) -> str | None:
        return None if session is None else session.selected_source_group

    @staticmethod
    def _cursor_state(session: DebugSession | None) -> UiDebugCursorState | None:
        cursor = None if session is None else session.cursor
        if cursor is None:
            return None
        return UiDebugCursorState(
            step_index=cursor.step_index,
            step_scope_id=cursor.step_scope_id,
            group_key=cursor.group_key,
            invocation_key=cursor.invocation_key,
            pattern_group_identity=cursor.pattern_group_identity,
        )

    def _target_scope_ids(self) -> tuple[str, ...]:
        if not self._manager.current_plate:
            return ()
        return (
            PipelineScopeIdentity.from_plate_scope(
                self._manager.current_plate
            ).scope_id,
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
                action.title,
                action.placement,
                action.enabled,
                None if action.disabled_error is None else action.disabled_error.code,
                action.confirmation_required,
                action.side_effects,
            )
            for action in state.actions
        )
        cursor = state.cursor
        cursor_part = None if cursor is None else (
            cursor.step_index,
            cursor.step_scope_id,
            cursor.group_key,
            cursor.invocation_key,
            cursor.pattern_group_identity,
        )
        parts = (
            self.identity.revision_key,
            state.object_state_token,
            self._snapshot_provider.current_branch_head_snapshot_id(),
            ObjectStateRegistry.get_current_snapshot_index(),
            selection_mode,
            state.current_plate_scope_id,
            state.pipeline_scope_id,
            state.phase,
            state.manager_execution_state,
            state.initialized,
            state.compiled,
            state.active_debug_session_id,
            state.selected_source_group,
            state.terminal_status,
            state.runtime_view_available,
            cursor_part,
            action_parts,
        )
        return hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()

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
```

Register it:

```python
class PipelineEditorBridgeProviderSet(UiBridgeProviderSetABC):
    ...

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
```

### Dev Client Renderer

Add:

`openhcs/mcp/dev_client_renderers/ui_bridge.py`

Add `PipelineDebugSessionStateSurfaceIdentityDeclaration` to the existing
`openhcs.agent.ui_bridge_identities` imports. The renderer registers through
`UiStateSurfacePayloadRenderer`; no explicit renderer registry update is
needed.

```python
class PipelineDebugSessionStateSurfaceRenderer(UiStateSurfacePayloadRenderer):
    """Compact renderer for Pipeline Debug Session state."""

    surface_identity = PipelineDebugSessionStateSurfaceIdentityDeclaration

    @classmethod
    def render(cls, response: JsonObject) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)
        errors = McpDevPayloadProjection.sequence_of_mappings(payload.get("errors"))
        if errors:
            return "\n".join(("Debug session: unavailable", *cls._error_lines(errors)))
        state_payload = McpDevPayloadProjection.nested_mapping(payload, "payload")
        actions = McpDevPayloadProjection.sequence_of_mappings(
            state_payload.get("actions")
        )
        enabled = tuple(action for action in actions if action.get("enabled") is True)
        disabled = tuple(action for action in actions if action.get("enabled") is False)
        cursor = McpDevPayloadProjection.nested_mapping(state_payload, "cursor")
        lines = [
            (
                "Debug session: "
                f"phase={McpDevPayloadProjection.text(state_payload.get('phase'))} "
                f"compiled={McpDevPayloadProjection.text(state_payload.get('compiled'))} "
                f"runtime={McpDevPayloadProjection.text(state_payload.get('runtime_view_available'))}"
            ),
            f"Plate: {McpDevPayloadProjection.text(state_payload.get('current_plate_scope_id'))}",
            f"Pipeline: {McpDevPayloadProjection.text(state_payload.get('pipeline_scope_id'))}",
            f"Active session: {McpDevPayloadProjection.text(state_payload.get('active_debug_session_id'))}",
            f"Source group: {McpDevPayloadProjection.text(state_payload.get('selected_source_group'))}",
            f"Terminal: {McpDevPayloadProjection.text(state_payload.get('terminal_status'))}",
            f"Enabled actions: {cls._action_titles(enabled)}",
            f"Disabled actions: {cls._action_titles(disabled)}",
        ]
        if cursor:
            lines.append(
                "Cursor: "
                f"step={McpDevPayloadProjection.text(cursor.get('step_index'))} "
                f"group={McpDevPayloadProjection.text(cursor.get('group_key'))} "
                f"invocation={McpDevPayloadProjection.text(cursor.get('invocation_key'))}"
            )
        return "\n".join(lines)

    @staticmethod
    def _action_titles(actions: tuple[Mapping[str, JsonValue], ...]) -> str:
        titles = [
            str(action.get("title") or action.get("action_id"))
            for action in actions
        ]
        return ", ".join(titles) if titles else "<none>"

    @staticmethod
    def _error_lines(errors: tuple[Mapping[str, JsonValue], ...]) -> tuple[str, ...]:
        return McpDiagnosticRenderer.error_lines(errors)
```

This uses the existing `UiStateSurfacePayloadRenderer` AutoRegisterMeta path.
Do not add a renderer registry dictionary.

## UX Target

### Idle Compiled Plate

Display:

- status: `Ready to debug`
- selected target: plate name and pipeline scope
- source selector: `Source: A01` or `Choose source`
- primary actions: `Start Debug`, `Step`, `Run to Pause Marker`
- session-only actions disabled with typed reasons
- inspector button disabled with typed reason

### Active or Paused Session

Display:

- status: `Active`
- cursor summary when available: raw `DebugCursor` fields plus source group;
  display code may join `step_scope_id` against `pipeline_editor.state` for a
  step name, but must not parse invocation strings;
- primary actions: `Continue`, `Step`, `Run to Pause Marker`;
- session actions: `Restart From Cursor`, `Stop`;
- inspector entry: `Inspect Data`.

### Running Session

Display:

- status from `manager_execution_state`;
- current progress from Plate Manager / progress registry when available;
- start/step/pause availability from the same action model;
- `Stop` remains visible if an active session exists.

### Completed or Cancelled Session

Display:

- terminal result from `plate_terminal_activity_status`;
- no active session id;
- start commands enabled if the plate remains compiled;
- session-only commands disabled.

### Inspector

Replace the current blank `DebugInspectorWindow` body with an
`ActionTabbedWindowBody`:

- `Runtime Values`
- `Live VFS`
- `Artifacts`
- `Snapshots`
- `Measurements`
- `Events`

Each tab renders existing typed payloads:

- runtime values: `DebugViewModel.from_runtime_value_store(...)`;
- snapshots/artifacts: `DebugSnapshot`, `DebugArtifactRef`;
- events: `ProgressEvent` plus `DebugProgressContext`.

Empty states must be explicit. Example:

`No runtime values recorded yet. Step past an artifact-producing invocation.`

The inspector conversion can be a follow-up after the debug-session state
surface lands, because the state surface is the load-bearing contract MCP and
Qt both need.

## Dry Run

This dry run is the implementation acceptance path. If any step requires a new
lookup table, stop and move the missing fact onto an existing authority.

### 1. No Plate

Inputs:

- `PipelineEditorWidget.current_plate is None`
- `PipelineEditorWidget.debug_session_state is None`

Context:

- `PipelineDebugTargetState.current_plate_scope_id = None`
- `PipelineDebugSessionContext.phase = NO_TARGET`
- `base_debug_controls_available = False`

Projected actions:

- all declared toolbar actions are present;
- all are disabled with `DebugActionDisabledCode.TARGET_REQUIRED`.

Qt:

- toolbar is visible but disabled;
- disabled tooltips explain that a plate is required.

MCP:

- `ui_list_actions` returns the same action ids as the toolbar declaration;
- `state-surface pipeline_debug_toolbar.session` says phase `no_target`.

### 2. Plate Loaded But Not Initialized

Inputs:

- `current_plate` set;
- `_is_current_plate_initialized()` returns `False`;
- `_is_current_plate_compiled()` returns `False`.

Context:

- phase `UNINITIALIZED`;
- target scope ids contain the pipeline scope;
- no active session.

Projected actions:

- all actions disabled with `INITIALIZED_PLATE_REQUIRED`.

No command checks a Qt button's enabled state in the bridge.

### 3. Initialized But Not Compiled

Inputs:

- `_is_current_plate_initialized()` returns `True`;
- `_is_current_plate_compiled()` returns `False`.

Context:

- phase `UNCOMPILED`;
- no active session.

Projected actions:

- all actions disabled with `COMPILED_PLATE_REQUIRED`.

MCP guidance:

- state surface points the agent to compile before invoking debug controls.

### 4. Compiled Idle

Inputs:

- initialized and compiled;
- `debug_session_state is None`;
- no terminal status.

Context:

- phase `READY`;
- `base_debug_controls_available = True`;
- `runtime_inspection_available = False`.

Projected actions:

- `RUN`, `STEP`, `RUN_TO_PAUSE`, and `CHOOSE_SOURCE_GROUP` enabled;
- `RESTART`, `STOP`, and `RUNTIME_VALUES` disabled with
  `ACTIVE_SESSION_REQUIRED`;
- `RUN` label becomes `Start Debug`;
- `RUN_TO_PAUSE` label becomes `Run to Pause Marker`;
- runtime values label becomes `Inspect Data`.

Qt:

- bottom toolbar shows useful start controls immediately after compile.

MCP:

- action catalog and `pipeline_debug_toolbar.session` surface agree on enabled
  actions.

### 5. Agent Invokes Step

Inputs:

- bridge request action id is `DebugCommandType.STEP.value`.

Provider flow:

1. `PipelineDebugToolbarActionProvider._model("step")` selects the projected
   model.
2. `_guard_error(request, model)` checks request scope/revision and
   `model.disabled_reason`.
3. dispatch calls `model.declaration.invoke(debug_workflow)`.
4. `StepDebugAction.invoke(...)` inherits command dispatch from
   `PipelineDebugCommandActionDeclaration` and runs
   `debug_workflow.handle_command(DebugCommand(DebugCommandType.STEP))`.

No bridge code converts `"step"` to a command by string registry. The only
string is the inbound ABI action id.

### 6. Active Session After Step

Inputs:

- Plate Manager creates or updates `DebugSession`;
- Pipeline Editor receives `debug_session_state`.

Context:

- phase `ACTIVE`;
- active debug session id set;
- cursor may be `None` until the first pause/snapshot event.

Projected actions:

- all compiled-plate actions remain available;
- session-only actions become enabled;
- runtime inspection becomes enabled.

State surface:

- `active_debug_session_id` is populated;
- `runtime_view_available = True`;
- `cursor` is present only when `DebugSession.cursor` is present.

### 7. Runtime Inspection

Inputs:

- bridge request action id is `DebugToolbarAuxiliaryAction.RUNTIME_VALUES.value`.

Provider flow:

1. projected model selected;
2. guard validates active session;
3. dispatch calls `model.declaration.invoke(debug_workflow)`;
4. `InspectRuntimeValuesAction.invoke(...)` calls
   `debug_workflow.show_runtime_inspection()`;
5. the workflow calls
   `PlateManagerWidget.action_inspect_debug_runtime(debug_session_id=...)`;
6. runtime returns `DebugViewModel`.

Inspector:

- `DebugInspectorWindow.set_inspection_view_model(view_model)` renders runtime
  values;
- if `view_model.sections` is empty, show an explicit empty state.

### 8. Stop Session

Inputs:

- bridge or Qt dispatches `DebugCommandType.STOP`.

Runtime flow:

- `PipelineEditorDebugWorkflow.stop_command()` calls
  `PlateManagerWidget.action_stop_execution()`;
- Plate Manager terminal cleanup calls
  `_clear_debug_session_for_plate(...)`;
- Pipeline Editor `debug_session_state` is cleared.

Context:

- compiled target remains true;
- phase becomes `TERMINAL` if terminal status is available, otherwise `READY`;
- active-session controls disable;
- start controls remain enabled.

This is the observed desired behavior from the current live UI smoke test.

## Implementation Steps

1. Add `pipeline_debug_actions.py`.
   - Move toolbar action semantics out of `DebugToolbarWidget`.
   - Move existing debug dispatch functions and route authority out of
     `pipeline_editor.py`.
   - Declare one nominal action class per debug command or auxiliary action.
   - Use AutoRegisterMeta for action iteration.
   - Put command route dispatch and auxiliary dispatch on declaration classes.
   - Delete `PipelineDebugCommandRoute` and
     `PipelineEditorWidget.DEBUG_COMMAND_ROUTES`.
   - Update internal imports/tests to use `pipeline_debug_actions.py`; do not
     re-export moved semantic declarations from `debug_toolbar.py`.

2. Add `debug_session_projection.py`.
   - Implement projection dataclasses/enums above.
   - Iterate `PipelineDebugActionDeclarationBase.toolbar_actions()`.
   - Unit test no-target, uninitialized, uncompiled, compiled-idle, active.

3. Add `PipelineEditorWidget.debug_session_context()`.
   - Use existing `_is_current_plate_initialized()` and
     `_is_current_plate_compiled()`.
   - Read terminal status from
     `plate_manager.plate_terminal_activity_status.terminal_status(current_plate)`.
   - Update `update_button_states()` to feed context into the toolbar.

4. Rework `DebugToolbarWidget` to consume action models.
   - Keep current visual structure initially.
   - Replace independent `_controls_enabled`, `_debug_session_active`, and
     `_runtime_inspection_enabled` decisions with projected models.
   - Remove old public setters and update tests in the same commit.

5. Add agent DTO records.
   - Add `UiDebugActionState`, `UiDebugCursorState`,
     `UiPipelineDebugSessionState`.
   - Keep semantic values as strings emitted by projection enums; do not create
     separate MCP enums.

6. Add debug-session state-surface identity and provider.
   - Add `PipelineDebugSessionStateSurfaceIdentityDeclaration`.
   - Add `PipelineDebugSessionStateSurfaceProvider`.
   - Register it in `PipelineEditorBridgeProviderSet`.

7. Rework `PipelineDebugToolbarActionProvider`.
   - Catalog from `DebugToolbarActionProjector.render_models(...)`.
   - Summary from the selected model.
   - Guard from the model disabled reason.
   - Dispatch by calling `model.declaration.invoke(debug_workflow)`.
   - Delete provider-local action id, declaration, and availability helpers.

8. Add dev-client renderer.
   - Add `PipelineDebugSessionStateSurfaceRenderer`.
   - Rely on `UiStateSurfacePayloadRenderer` AutoRegisterMeta.
   - Make output concise: phase, target, active session, cursor, runtime
     available, enabled/disabled action names.

9. Inspector follow-up.
   - Convert `DebugInspectorWindow` body to `ActionTabbedWindowBody`.
   - Keep `Runtime Values` backed by `DebugViewModel`.
   - Add empty-state labels for zero sections/tables.
   - Add placeholder tabs only if they are backed by existing typed debug
     objects, not invented MCP DTOs.

## Semantic Boundary Rules

- Do not add action-id string lists outside debug action declarations.
- Do not add a second debug-session registry.
- Do not add MCP-specific runtime-value DTOs.
- Do not make the inspector infer semantics from labels; consume
  `DebugViewModel`, `DebugSnapshot`, `DebugArtifactRef`, and progress context.
- Do not make pyqt-reactive own OpenHCS debugger semantics. It can provide
  layout/action chrome only.
- Do not use Qt widget enabled state as the bridge authority. The bridge and Qt
  must both consume the shared projection.
- Do not add renderer dictionaries. Use `UiStateSurfacePayloadRenderer` and
  `UiStateSurfaceIdentityDeclarationBase` AutoRegisterMeta.

## AST / Search Gates

Before and after the implementation, use AST/search to keep the refactor
bounded.

### Ownership Search

```bash
source .venv/bin/activate
python - <<'PY'
import ast
from pathlib import Path

targets = [
    Path("openhcs/pyqt_gui/services/ui_bridge_pipeline_editor.py"),
    Path("openhcs/pyqt_gui/widgets/debug_toolbar.py"),
    Path("openhcs/pyqt_gui/widgets/pipeline_editor.py"),
]
for path in targets:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if "debug" in node.name.lower() or "action" in node.name.lower():
                print(f"{path}:{node.lineno}:def {node.name}")
PY
```

Use this to verify which methods are being replaced, not to generate new
registries.

### No Semantic Mirrors

These commands must return no matches outside tests and the new projection module:

```bash
rg -n "debug_session_required" openhcs/pyqt_gui/services/ui_bridge_pipeline_editor.py
rg -n "_action_ids|_action_declaration|_availability_error|_disabled_error" openhcs/pyqt_gui/services/ui_bridge_pipeline_editor.py
rg -n "set_controls_enabled|set_debug_session_active|set_runtime_inspection_enabled" openhcs/pyqt_gui/widgets/pipeline_editor.py
rg -n "BUTTON_SPECS|MENU_ACTION_SPECS|AUXILIARY_ACTION_SPECS" openhcs/pyqt_gui/widgets/debug_toolbar.py
rg -n "PipelineDebugCommandRoute|DEBUG_COMMAND_ROUTES" openhcs/pyqt_gui/widgets openhcs/pyqt_gui/services
rg -n "DebugCommandType\\.(RUN|STEP|RUN_TO_PAUSE|RESTART|STOP)" openhcs/pyqt_gui/services/ui_bridge_pipeline_editor.py
rg -n "Start Debug|Inspect Data|Run to Pause Marker|Restart From Cursor" openhcs/mcp
```

Expected:

- dynamic labels appear on declaration classes and tests only;
- bridge dispatch must not mention leaf debug commands or auxiliary actions;
- MCP renderers do not hardcode debug command labels.

### State Surface Registration

```bash
rg -n "PipelineDebugActionDeclarationBase|StartOrContinueDebugAction|InspectRuntimeValuesAction" openhcs/pyqt_gui/widgets/shared/services/pipeline_debug_actions.py
rg -n "PipelineDebugSessionStateSurfaceIdentityDeclaration" openhcs
rg -n "PipelineDebugSessionStateSurfaceProvider" openhcs
rg -n "PipelineDebugSessionStateSurfaceRenderer" openhcs/mcp/dev_client_renderers/ui_bridge.py
```

Expected:

- action declaration classes live in `pipeline_debug_actions.py`;
- identity declaration in `openhcs/agent/ui_bridge_identities.py`;
- provider in `openhcs/pyqt_gui/services/ui_bridge_pipeline_editor.py`;
- renderer in `openhcs/mcp/dev_client_renderers/ui_bridge.py`;
- no manual enum or renderer dictionaries.

## Test Plan

Focused unit tests:

```bash
source .venv/bin/activate
python -m pytest \
  tests/unit/pyqt_gui/test_debug_toolbar.py \
  tests/unit/pyqt_gui/test_ui_agent_bridge.py \
  tests/unit/pyqt_gui/test_pipeline_editor_widget.py \
  -q
```

Add or update tests:

- `PipelineDebugActionDeclarationBase` routes every `DebugCommandType` that the
  old `DEBUG_COMMAND_ROUTES` table routed;
- `TOGGLE` and `RANDOM_SOURCE_GROUP` remain command declarations but are not
  returned by `toolbar_actions()`;
- `DebugToolbarActionProjector` no-target/uninitialized/uncompiled/ready/active
  projections;
- `DebugToolbarWidget` uses projected enabled state for command and auxiliary
  queries;
- debug-session state surface appears in `ui_list_state_surfaces`;
- compiled idle state exposes start actions and disables session-only actions;
- active session exposes session id and enables inspector actions;
- terminal cleanup clears session-only actions;
- action catalog and debug-session state surface agree on enabled actions.

Live smoke:

```bash
source .venv/bin/activate
python -m openhcs.mcp.dev_client ui-status --timeout-ms 1000
python -m openhcs.mcp.dev_client state-surfaces --timeout-ms 2000
python -m openhcs.mcp.dev_client state-surface pipeline_debug_toolbar.session --timeout-ms 2000
python -m openhcs.mcp.dev_client actions pipeline_debug_toolbar --timeout-ms 2000
python -m openhcs.mcp.dev_client window-snapshot pipeline_editor --timeout-ms 2000
```

Live expected behavior:

- no plate: all debug actions disabled with target-required reasons;
- compiled idle plate: `Start Debug`, `Step`, `Run to Pause Marker`, and
  `Choose source group` enabled;
- after step: active session id appears and `Inspect Data`, `Restart From
  Cursor`, and `Stop debug session` become enabled;
- after stop: active session id clears, start controls remain enabled, and
  session-only controls disable.
