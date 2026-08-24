"""Nominal declarations for PipelineEditor debug-session actions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, ClassVar

from metaclass_registry import AutoRegisterMeta

from openhcs.core.debug import DebugCommandType

if TYPE_CHECKING:
    from openhcs.pyqt_gui.widgets.shared.services.debug_session_projection import (
        PipelineDebugSessionContext,
    )
    from openhcs.pyqt_gui.widgets.shared.services.pipeline_editor_workflows import (
        PipelineEditorDebugWorkflow,
    )


@dataclass(frozen=True, slots=True)
class DebugActionDisabledReason:
    """Agent/UI facing disabled reason for one debug action."""

    code: str
    message: str
    hint: str


class DebugActionPlacement(Enum):
    """Toolbar placement declared by one debug action."""

    PRIMARY = "primary"
    INSPECTOR = "inspector"
    SESSION = "session"
    HIDDEN = "hidden"


class DebugToolbarAuxiliaryAction(str, Enum):
    """Non-command debug toolbar actions."""

    RUNTIME_VALUES = "runtime_values"


class PipelineDebugActionDeclarationBase(ABC, metaclass=AutoRegisterMeta):
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
            raise ValueError(
                f"{cls.__name__} does not declare a debug action identity."
            )
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
    def for_action_id(
        cls, action_id: str
    ) -> type["PipelineDebugActionDeclarationBase"]:
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
    @abstractmethod
    def invoke(cls, workflow: "PipelineEditorDebugWorkflow") -> None:
        """Invoke this action through the nominal editor workflow."""


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
    def invoke(cls, workflow: "PipelineEditorDebugWorkflow") -> None:
        workflow.run_command(cls.command_type())


class PipelineDebugAuxiliaryActionDeclaration(PipelineDebugActionDeclarationBase):
    """Debug action owned by the toolbar but not represented by DebugCommandType."""

    identity: ClassVar[DebugToolbarAuxiliaryAction | None] = None
    side_effects: ClassVar[tuple[str, ...]] = ("opens_debug_runtime_inspector",)
    confirmation_required: ClassVar[bool] = False

    @classmethod
    def auxiliary_action_type(cls) -> DebugToolbarAuxiliaryAction:
        identity = cls.require_identity()
        if not isinstance(identity, DebugToolbarAuxiliaryAction):
            raise TypeError(
                f"{cls.__name__} is not backed by DebugToolbarAuxiliaryAction."
            )
        return identity

    @classmethod
    def invoke(cls, workflow: "PipelineEditorDebugWorkflow") -> None:
        workflow.show_runtime_inspection()


class ToggleDebugModeAction(PipelineDebugCommandActionDeclaration):
    identity = DebugCommandType.TOGGLE
    label = "Debug"
    tooltip = "Debug toolbar active. Use Debug, Step, Pause, Restart, or Inspect."

    @classmethod
    def invoke(cls, workflow: "PipelineEditorDebugWorkflow") -> None:
        workflow.show_status(cls.tooltip)


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
        if target is not None and target.initialized and target.compiled:
            return "Start Debug"
        return cls.label


class StepDebugAction(PipelineDebugCommandActionDeclaration):
    identity = DebugCommandType.STEP
    toolbar_placement = DebugActionPlacement.PRIMARY
    toolbar_order = 20
    label = "Step"
    tooltip = "Run one debug step"


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


class RestartDebugAction(PipelineDebugCommandActionDeclaration):
    identity = DebugCommandType.RESTART
    toolbar_placement = DebugActionPlacement.SESSION
    toolbar_order = 40
    label = "Restart"
    tooltip = "Restart the current debug session"
    requires_active_debug_session = True


class InspectRuntimeValuesAction(PipelineDebugAuxiliaryActionDeclaration):
    identity = DebugToolbarAuxiliaryAction.RUNTIME_VALUES
    toolbar_placement = DebugActionPlacement.INSPECTOR
    toolbar_order = 10
    label = "Inspect Runtime"
    tooltip = "Inspect live runtime values for the paused debug worker"
    requires_active_debug_session = True


class ChooseSourceGroupDebugAction(PipelineDebugCommandActionDeclaration):
    identity = DebugCommandType.CHOOSE_SOURCE_GROUP
    toolbar_placement = DebugActionPlacement.SESSION
    toolbar_order = 10
    label = "Choose source group"
    tooltip = "Choose a well/image set for debug execution"


class StopDebugSessionAction(PipelineDebugCommandActionDeclaration):
    identity = DebugCommandType.STOP
    toolbar_placement = DebugActionPlacement.SESSION
    toolbar_order = 20
    label = "Stop debug session"
    tooltip = "Stop the active debug execution"
    requires_active_or_pending_execution = True
    enabled_during_pending_execution = True

    @classmethod
    def invoke(cls, workflow: "PipelineEditorDebugWorkflow") -> None:
        workflow.stop_command()


class RandomSourceGroupDebugAction(PipelineDebugCommandActionDeclaration):
    identity = DebugCommandType.RANDOM_SOURCE_GROUP
    label = "Random source group"
    tooltip = "Choose a random well/image set for debug execution"


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
