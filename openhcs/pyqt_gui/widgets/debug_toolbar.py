"""PyQt debug/test-mode toolbar built on core debug commands."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QWidget

from openhcs.core.debug import DebugCommand, DebugCommandType
from pyqt_reactive.widgets.shared.button_panel import ButtonPanel


@dataclass(frozen=True, slots=True)
class DebugToolbarButtonSpec:
    """One visible debug toolbar button."""

    label: str
    command_type: DebugCommandType
    tooltip: str
    side_effects: tuple[str, ...] = ("starts_or_controls_debug_execution",)
    confirmation_required: bool = True
    requires_active_debug_session: bool = False


@dataclass(frozen=True, slots=True)
class DebugToolbarMenuActionSpec:
    """One debug command exposed from the overflow menu."""

    label: str
    command_type: DebugCommandType
    tooltip: str
    side_effects: tuple[str, ...] = ("starts_or_controls_debug_execution",)
    confirmation_required: bool = True
    requires_active_debug_session: bool = False


class DebugToolbarAuxiliaryAction(str, Enum):
    """Non-command debug toolbar actions."""

    RUNTIME_VALUES = "runtime_values"


@dataclass(frozen=True, slots=True)
class DebugToolbarAuxiliaryActionSpec:
    """One debug toolbar action not represented as a runtime command."""

    label: str
    action_type: DebugToolbarAuxiliaryAction
    tooltip: str
    side_effects: tuple[str, ...] = ("opens_debug_runtime_inspector",)
    confirmation_required: bool = False
    requires_active_debug_session: bool = False


class DebugToolbarWidget(QWidget):
    """Compact command surface for bounded debug/test-mode runs."""

    command_requested = pyqtSignal(object)
    runtime_inspection_requested = pyqtSignal()

    BUTTON_SPECS = (
        DebugToolbarButtonSpec(
            "Debug",
            DebugCommandType.RUN,
            "Start or continue debug execution for the selected plate",
        ),
        DebugToolbarButtonSpec("Step", DebugCommandType.STEP, "Run one debug step"),
        DebugToolbarButtonSpec(
            "Pause",
            DebugCommandType.RUN_TO_PAUSE,
            "Run until the next pause marker",
        ),
        DebugToolbarButtonSpec(
            "Restart",
            DebugCommandType.RESTART,
            "Restart the current debug session",
            requires_active_debug_session=True,
        ),
    )
    MENU_ACTION_SPECS = (
        DebugToolbarMenuActionSpec(
            "Choose source group",
            DebugCommandType.CHOOSE_SOURCE_GROUP,
            "Choose a well/image set for debug execution",
        ),
        DebugToolbarMenuActionSpec(
            "Stop debug session",
            DebugCommandType.STOP,
            "Stop the active debug execution",
            requires_active_debug_session=True,
        ),
    )
    AUXILIARY_ACTION_SPECS = (
        DebugToolbarAuxiliaryActionSpec(
            "Runtime values",
            DebugToolbarAuxiliaryAction.RUNTIME_VALUES,
            "Inspect live runtime values for the paused debug worker",
            requires_active_debug_session=True,
        ),
    )

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        style_generator=None,
    ) -> None:
        super().__init__(parent)
        self.button_panel: ButtonPanel | None = None
        self.buttons: dict[DebugCommandType, QPushButton] = {}
        self._controls_enabled = True
        self._debug_session_active = False
        self._runtime_inspection_enabled = True
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.button_panel = ButtonPanel(
            button_configs=[
                (spec.label, spec.command_type.value, spec.tooltip)
                for spec in self.BUTTON_SPECS
            ],
            on_action=self._emit_command,
            style_generator=style_generator,
            parent=self,
        )
        self.buttons = {
            DebugCommandType(action_id): button
            for action_id, button in self.button_panel.buttons.items()
        }
        self.menu_actions: dict[DebugCommandType, QAction] = {}
        self.auxiliary_actions: dict[DebugToolbarAuxiliaryAction, QAction] = {}
        self.runtime_inspection_action: QAction | None = None
        self.menu_button = QPushButton("Inspect")
        self.menu_button.setToolTip("Open debug inspection and session commands")
        if style_generator:
            self.menu_button.setStyleSheet(style_generator.generate_button_style())
        self._setup_menu_button()
        layout.addWidget(self.button_panel, 1)
        layout.addWidget(self.menu_button)
        self._apply_enabled_state()

    def _setup_menu_button(self) -> None:
        menu = self.menu_button.menu()
        if menu is None:
            from PyQt6.QtWidgets import QMenu

            menu = QMenu(self.menu_button)
            self.menu_button.setMenu(menu)
        for spec in self.AUXILIARY_ACTION_SPECS:
            action = QAction(spec.label, self.menu_button)
            action.setToolTip(spec.tooltip)
            if spec.action_type is DebugToolbarAuxiliaryAction.RUNTIME_VALUES:
                action.triggered.connect(self.runtime_inspection_requested.emit)
                self.runtime_inspection_action = action
            menu.addAction(action)
            self.auxiliary_actions[spec.action_type] = action
        menu.addSeparator()
        for spec in self.MENU_ACTION_SPECS:
            action = QAction(spec.label, self.menu_button)
            action.setToolTip(spec.tooltip)
            action.triggered.connect(
                lambda checked, command_type=spec.command_type: self._emit_command(
                    command_type.value
                )
            )
            menu.addAction(action)
            self.menu_actions[spec.command_type] = action

    def _emit_command(self, action_id: str) -> None:
        self.command_requested.emit(DebugCommand(DebugCommandType(action_id)))

    @classmethod
    def command_specs(
        cls,
    ) -> tuple[DebugToolbarButtonSpec | DebugToolbarMenuActionSpec, ...]:
        return (*cls.BUTTON_SPECS, *cls.MENU_ACTION_SPECS)

    @classmethod
    def command_spec(
        cls,
        command_type: DebugCommandType,
    ) -> DebugToolbarButtonSpec | DebugToolbarMenuActionSpec:
        for spec in cls.command_specs():
            if spec.command_type is command_type:
                return spec
        raise ValueError(f"Debug command is not exposed by the toolbar: {command_type}")

    @classmethod
    def auxiliary_action_spec(
        cls,
        action_type: DebugToolbarAuxiliaryAction,
    ) -> DebugToolbarAuxiliaryActionSpec:
        for spec in cls.AUXILIARY_ACTION_SPECS:
            if spec.action_type is action_type:
                return spec
        raise ValueError(f"Debug auxiliary action is not exposed by the toolbar: {action_type}")

    def command_enabled(self, command_type: DebugCommandType) -> bool:
        button = self.buttons.get(command_type)
        if button is not None:
            return button.isEnabled()
        action = self.menu_actions.get(command_type)
        if action is not None:
            return self.menu_button.isEnabled() and action.isEnabled()
        raise ValueError(f"Debug command is not exposed by the toolbar: {command_type}")

    def auxiliary_action_enabled(
        self,
        action_type: DebugToolbarAuxiliaryAction,
    ) -> bool:
        action = self.auxiliary_actions.get(action_type)
        if action is None:
            raise ValueError(
                f"Debug auxiliary action is not exposed by the toolbar: {action_type}"
            )
        return self.menu_button.isEnabled() and action.isEnabled()

    def set_controls_enabled(self, enabled: bool) -> None:
        """Set base debug availability for the selected plate."""

        self._controls_enabled = enabled
        self._apply_enabled_state()

    def set_runtime_inspection_enabled(self, enabled: bool) -> None:
        """Enable the live-runtime inspector action independently of commands."""

        self._runtime_inspection_enabled = enabled
        self._apply_enabled_state()

    def set_debug_session_active(self, active: bool) -> None:
        """Set whether an active debug session exists for session-only controls."""

        self._debug_session_active = active
        self._apply_enabled_state()

    def _apply_enabled_state(self) -> None:
        for spec in self.BUTTON_SPECS:
            self.buttons[spec.command_type].setEnabled(
                self._spec_enabled(spec)
            )
        for spec in self.MENU_ACTION_SPECS:
            self.menu_actions[spec.command_type].setEnabled(
                self._spec_enabled(spec)
            )
        for spec in self.AUXILIARY_ACTION_SPECS:
            enabled = self._spec_enabled(spec)
            if spec.action_type is DebugToolbarAuxiliaryAction.RUNTIME_VALUES:
                enabled = enabled and self._runtime_inspection_enabled
            self.auxiliary_actions[spec.action_type].setEnabled(enabled)
        self.menu_button.setEnabled(self._controls_enabled)

    def _spec_enabled(
        self,
        spec: (
            DebugToolbarButtonSpec
            | DebugToolbarMenuActionSpec
            | DebugToolbarAuxiliaryActionSpec
        ),
    ) -> bool:
        if not self._controls_enabled:
            return False
        if spec.requires_active_debug_session and not self._debug_session_active:
            return False
        return True


__all__ = (
    "DebugToolbarAuxiliaryAction",
    "DebugToolbarAuxiliaryActionSpec",
    "DebugToolbarButtonSpec",
    "DebugToolbarMenuActionSpec",
    "DebugToolbarWidget",
)
