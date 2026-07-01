"""Nominal declarations for PipelineEditor debug-session actions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, ClassVar, Protocol

from metaclass_registry import AutoRegisterMeta
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QPushButton

from openhcs.core.debug import DebugCommand, DebugCommandType

if TYPE_CHECKING:
    from openhcs.pyqt_gui.widgets.shared.services.debug_session_projection import (
        PipelineDebugSessionContext,
    )


@dataclass(frozen=True, slots=True)
class DebugActionDisabledReason:
    """Agent/UI facing disabled reason for one debug action."""

    code: str
    message: str
    hint: str


class DebugToolbarWorkflow(Protocol):
    """Workflow surface consumed by declared debug toolbar actions."""

    def handle_command(self, command: DebugCommand) -> None:
        """Dispatch a typed debug command."""

    def show_runtime_inspection(self) -> None:
        """Open the runtime-value inspector."""


class DebugStatusEmitter(Protocol):
    """Signal-like status surface owned by PipelineEditorWidget."""

    def emit(self, message: str) -> None:
        """Emit a user-facing status message."""


class PipelineDebugEditorWorkflow(Protocol):
    """PipelineEditor debug workflow methods invoked by action declarations."""

    def run_command(self, command_type: DebugCommandType = DebugCommandType.RUN) -> None:
        """Run or continue a debug command."""

    def stop_command(self) -> None:
        """Stop an active debug command."""

    def show_runtime_inspection(self) -> None:
        """Open the runtime-value inspector."""


class PipelineDebugEditor(Protocol):
    """PipelineEditor surface consumed by debug action declarations."""

    status_message: DebugStatusEmitter
    debug_workflow: PipelineDebugEditorWorkflow


class DebugSignal(Protocol):
    """Signal-like Qt surface used by debug toolbar declarations."""

    def emit(self) -> None:
        """Emit the signal."""


class DebugToolbarSurface(Protocol):
    """Toolbar attributes mutated by declared debug actions during rendering."""

    buttons: dict[DebugCommandType, QPushButton]
    auxiliary_buttons: dict[DebugToolbarAuxiliaryAction, QPushButton]
    menu_actions: dict[DebugCommandType, QAction]
    auxiliary_actions: dict[DebugToolbarAuxiliaryAction, QAction]
    runtime_inspection_button: QPushButton | None
    runtime_inspection_action: QAction | None
    runtime_inspection_requested: DebugSignal

    def emit_debug_command(self, action_id: str) -> None:
        """Emit a declared debug command by action id."""


class DebugActionPlacement(Enum):
    """Toolbar placement declared by one debug action."""

    PRIMARY = "primary"
    INSPECTOR = "inspector"
    SESSION = "session"
    HIDDEN = "hidden"


class DebugToolbarAuxiliaryAction(str, Enum):
    """Non-command debug toolbar actions."""

    RUNTIME_VALUES = "runtime_values"


class PipelineDebugActionDeclarationBase(metaclass=AutoRegisterMeta):
    """Single semantic owner for one PipelineEditor debug action."""

    __registry__: ClassVar[
        dict[
            DebugCommandType | DebugToolbarAuxiliaryAction,
            type["PipelineDebugActionDeclarationBase"],
        ]
    ] = {}
    __registry_key__ = "identity"
    __skip_if_no_key__ = True

    identity: ClassVar[DebugCommandType | DebugToolbarAuxiliaryAction | None] = None
    toolbar_placement: ClassVar[DebugActionPlacement] = DebugActionPlacement.HIDDEN
    toolbar_order: ClassVar[int] = 0
    label: ClassVar[str]
    tooltip: ClassVar[str]
    side_effects: ClassVar[tuple[str, ...]] = ("starts_or_controls_debug_execution",)
    confirmation_required: ClassVar[bool] = True
    requires_active_debug_session: ClassVar[bool] = False
    requires_active_or_pending_execution: ClassVar[bool] = False
    enabled_during_pending_execution: ClassVar[bool] = False

    @classmethod
    def action_id(cls) -> str:
        identity = cls.require_identity()
        return identity.value

    @classmethod
    def require_identity(cls) -> DebugCommandType | DebugToolbarAuxiliaryAction:
        if cls.identity is None:
            raise ValueError(f"{cls.__name__} does not declare a debug action identity.")
        return cls.identity

    @classmethod
    def is_toolbar_action(cls) -> bool:
        return cls.toolbar_placement is not DebugActionPlacement.HIDDEN

    @classmethod
    def toolbar_actions(cls) -> tuple[type["PipelineDebugActionDeclarationBase"], ...]:
        return tuple(
            sorted(
                (
                    declaration
                    for declaration in cls.__registry__.values()
                    if declaration.is_toolbar_action()
                ),
                key=lambda declaration: (
                    declaration.toolbar_placement.value,
                    declaration.toolbar_order,
                    declaration.action_id(),
                ),
            )
        )

    @classmethod
    def for_action_id(cls, action_id: str) -> type["PipelineDebugActionDeclarationBase"]:
        for declaration in cls.__registry__.values():
            if declaration.action_id() == action_id:
                return declaration
        raise ValueError(f"Unknown PipelineEditor debug action: {action_id!r}")

    @classmethod
    def for_command_type(
        cls,
        command_type: DebugCommandType,
    ) -> type["PipelineDebugCommandActionDeclaration"]:
        declaration = cls.__registry__.get(command_type)
        if declaration is None:
            raise ValueError(f"Unhandled debug command route: {command_type.value}")
        if not issubclass(declaration, PipelineDebugCommandActionDeclaration):
            raise TypeError(
                f"Debug action {command_type.value!r} is not a command declaration."
            )
        return declaration

    @classmethod
    def label_for(cls, context: "PipelineDebugSessionContext") -> str:
        del context
        return cls.label

    @classmethod
    def tooltip_for(cls, context: "PipelineDebugSessionContext") -> str:
        del context
        return cls.tooltip

    @classmethod
    def availability_override(
        cls,
        context: "PipelineDebugSessionContext",
    ) -> DebugActionDisabledReason | None:
        del context
        return None

    @classmethod
    def dispatch_editor(cls, editor: PipelineDebugEditor) -> None:
        raise NotImplementedError

    @classmethod
    def invoke_workflow(cls, workflow: DebugToolbarWorkflow) -> None:
        raise NotImplementedError

    @classmethod
    def connect_qaction(cls, toolbar: DebugToolbarSurface, action: QAction) -> None:
        raise NotImplementedError

    @classmethod
    def register_widget_button(
        cls,
        toolbar: DebugToolbarSurface,
        button: QPushButton,
    ) -> None:
        raise NotImplementedError

    @classmethod
    def register_widget_action(cls, toolbar: DebugToolbarSurface, action: QAction) -> None:
        raise NotImplementedError


class PipelineDebugCommandActionDeclaration(PipelineDebugActionDeclarationBase):
    """Debug action backed by a core DebugCommandType."""

    identity: ClassVar[DebugCommandType | None] = None

    @classmethod
    def command_type(cls) -> DebugCommandType:
        identity = cls.require_identity()
        if not isinstance(identity, DebugCommandType):
            raise TypeError(f"{cls.__name__} is not backed by DebugCommandType.")
        return identity

    @classmethod
    def invoke_workflow(cls, workflow: DebugToolbarWorkflow) -> None:
        workflow.handle_command(DebugCommand(cls.command_type()))

    @classmethod
    def connect_qaction(cls, toolbar: DebugToolbarSurface, action: QAction) -> None:
        action.triggered.connect(
            lambda checked, action_id=cls.action_id(): toolbar.emit_debug_command(
                action_id
            )
        )

    @classmethod
    def register_widget_button(
        cls,
        toolbar: DebugToolbarSurface,
        button: QPushButton,
    ) -> None:
        toolbar.buttons[cls.command_type()] = button

    @classmethod
    def register_widget_action(cls, toolbar: DebugToolbarSurface, action: QAction) -> None:
        toolbar.menu_actions[cls.command_type()] = action


class PipelineDebugAuxiliaryActionDeclaration(PipelineDebugActionDeclarationBase):
    """Debug action owned by the toolbar but not represented by DebugCommandType."""

    identity: ClassVar[DebugToolbarAuxiliaryAction | None] = None
    side_effects: ClassVar[tuple[str, ...]] = ("opens_debug_runtime_inspector",)
    confirmation_required: ClassVar[bool] = False

    @classmethod
    def auxiliary_action_type(cls) -> DebugToolbarAuxiliaryAction:
        identity = cls.require_identity()
        if not isinstance(identity, DebugToolbarAuxiliaryAction):
            raise TypeError(f"{cls.__name__} is not backed by DebugToolbarAuxiliaryAction.")
        return identity

    @classmethod
    def invoke_workflow(cls, workflow: DebugToolbarWorkflow) -> None:
        workflow.show_runtime_inspection()

    @classmethod
    def dispatch_editor(cls, editor: PipelineDebugEditor) -> None:
        editor.debug_workflow.show_runtime_inspection()

    @classmethod
    def connect_qaction(cls, toolbar: DebugToolbarSurface, action: QAction) -> None:
        action.triggered.connect(toolbar.runtime_inspection_requested.emit)

    @classmethod
    def register_widget_button(
        cls,
        toolbar: DebugToolbarSurface,
        button: QPushButton,
    ) -> None:
        toolbar.auxiliary_buttons[cls.auxiliary_action_type()] = button

    @classmethod
    def register_widget_action(cls, toolbar: DebugToolbarSurface, action: QAction) -> None:
        toolbar.auxiliary_actions[cls.auxiliary_action_type()] = action


class ToggleDebugModeAction(PipelineDebugCommandActionDeclaration):
    identity = DebugCommandType.TOGGLE
    label = "Debug"
    tooltip = "Debug toolbar active. Use Debug, Step, Pause, Restart, or Inspect."

    @classmethod
    def dispatch_editor(cls, editor: PipelineDebugEditor) -> None:
        editor.status_message.emit(cls.tooltip)


class StartOrContinueDebugAction(PipelineDebugCommandActionDeclaration):
    identity = DebugCommandType.RUN
    toolbar_placement = DebugActionPlacement.PRIMARY
    toolbar_order = 10
    label = "Debug"
    tooltip = "Start or continue debug execution for the selected plate"

    @classmethod
    def label_for(cls, context: "PipelineDebugSessionContext") -> str:
        if context.active_session is not None:
            return "Continue"
        target = context.target
        if (
            target is not None
            and target.initialized
            and target.compiled
        ):
            return "Start Debug"
        return cls.label

    @classmethod
    def dispatch_editor(cls, editor: PipelineDebugEditor) -> None:
        editor.debug_workflow.run_command(DebugCommandType.RUN)


class StepDebugAction(PipelineDebugCommandActionDeclaration):
    identity = DebugCommandType.STEP
    toolbar_placement = DebugActionPlacement.PRIMARY
    toolbar_order = 20
    label = "Step"
    tooltip = "Run one debug step"

    @classmethod
    def dispatch_editor(cls, editor: PipelineDebugEditor) -> None:
        editor.debug_workflow.run_command(DebugCommandType.STEP)


class RunToPauseDebugAction(PipelineDebugCommandActionDeclaration):
    identity = DebugCommandType.RUN_TO_PAUSE
    toolbar_placement = DebugActionPlacement.PRIMARY
    toolbar_order = 30
    label = "Run to Pause"
    tooltip = "Run until the next pause marker"

    @classmethod
    def availability_override(
        cls,
        context: "PipelineDebugSessionContext",
    ) -> DebugActionDisabledReason | None:
        if not context.pause_boundaries.has_pause_boundaries:
            return DebugActionDisabledReason(
                code="debug_pause_boundary_required",
                message="Run to Pause requires at least one debug-pause step.",
                hint=(
                    "Enable debug_pause on a pipeline step before invoking "
                    "Run to Pause."
                ),
            )
        return None

    @classmethod
    def dispatch_editor(cls, editor: PipelineDebugEditor) -> None:
        editor.debug_workflow.run_command(DebugCommandType.RUN_TO_PAUSE)


class RestartDebugAction(PipelineDebugCommandActionDeclaration):
    identity = DebugCommandType.RESTART
    toolbar_placement = DebugActionPlacement.SESSION
    toolbar_order = 40
    label = "Restart"
    tooltip = "Restart the current debug session"
    requires_active_debug_session = True

    @classmethod
    def dispatch_editor(cls, editor: PipelineDebugEditor) -> None:
        editor.debug_workflow.run_command(DebugCommandType.RESTART)


class InspectRuntimeValuesAction(PipelineDebugAuxiliaryActionDeclaration):
    identity = DebugToolbarAuxiliaryAction.RUNTIME_VALUES
    toolbar_placement = DebugActionPlacement.INSPECTOR
    toolbar_order = 10
    label = "Inspect Runtime"
    tooltip = "Inspect live runtime values for the paused debug worker"
    requires_active_debug_session = True

    @classmethod
    def register_widget_action(cls, toolbar: DebugToolbarSurface, action: QAction) -> None:
        super().register_widget_action(toolbar, action)
        toolbar.runtime_inspection_action = action

    @classmethod
    def register_widget_button(
        cls,
        toolbar: DebugToolbarSurface,
        button: QPushButton,
    ) -> None:
        super().register_widget_button(toolbar, button)
        toolbar.runtime_inspection_button = button


class ChooseSourceGroupDebugAction(PipelineDebugCommandActionDeclaration):
    identity = DebugCommandType.CHOOSE_SOURCE_GROUP
    toolbar_placement = DebugActionPlacement.SESSION
    toolbar_order = 10
    label = "Choose source group"
    tooltip = "Choose a well/image set for debug execution"

    @classmethod
    def dispatch_editor(cls, editor: PipelineDebugEditor) -> None:
        editor.debug_workflow.run_command(DebugCommandType.CHOOSE_SOURCE_GROUP)


class StopDebugSessionAction(PipelineDebugCommandActionDeclaration):
    identity = DebugCommandType.STOP
    toolbar_placement = DebugActionPlacement.SESSION
    toolbar_order = 20
    label = "Stop debug session"
    tooltip = "Stop the active debug execution"
    requires_active_or_pending_execution = True
    enabled_during_pending_execution = True

    @classmethod
    def dispatch_editor(cls, editor: PipelineDebugEditor) -> None:
        editor.debug_workflow.stop_command()


class RandomSourceGroupDebugAction(PipelineDebugCommandActionDeclaration):
    identity = DebugCommandType.RANDOM_SOURCE_GROUP
    label = "Random source group"
    tooltip = "Choose a random well/image set for debug execution"

    @classmethod
    def dispatch_editor(cls, editor: PipelineDebugEditor) -> None:
        editor.debug_workflow.run_command(DebugCommandType.RANDOM_SOURCE_GROUP)


__all__ = (
    "ChooseSourceGroupDebugAction",
    "DebugActionDisabledReason",
    "DebugActionPlacement",
    "DebugToolbarAuxiliaryAction",
    "InspectRuntimeValuesAction",
    "PipelineDebugActionDeclarationBase",
    "PipelineDebugAuxiliaryActionDeclaration",
    "PipelineDebugCommandActionDeclaration",
    "RandomSourceGroupDebugAction",
    "RestartDebugAction",
    "RunToPauseDebugAction",
    "StartOrContinueDebugAction",
    "StepDebugAction",
    "StopDebugSessionAction",
    "ToggleDebugModeAction",
)
