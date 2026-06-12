"""PyQt debug/test-mode toolbar built on core debug commands."""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QPushButton, QVBoxLayout, QWidget

from openhcs.core.debug import DebugCommand, DebugCommandType
from pyqt_reactive.widgets.shared.button_panel import ButtonPanel


@dataclass(frozen=True, slots=True)
class DebugToolbarButtonSpec:
    """One visible debug toolbar button."""

    label: str
    command_type: DebugCommandType
    tooltip: str


class DebugToolbarWidget(QWidget):
    """Compact command surface for bounded debug/test-mode runs."""

    command_requested = pyqtSignal(object)

    BUTTON_SPECS = (
        DebugToolbarButtonSpec(
            "Debug",
            DebugCommandType.TOGGLE,
            "Toggle debug/test mode for the current pipeline",
        ),
        DebugToolbarButtonSpec("Step", DebugCommandType.STEP, "Run one debug step"),
        DebugToolbarButtonSpec("Run", DebugCommandType.RUN, "Run debug execution"),
        DebugToolbarButtonSpec(
            "Pause",
            DebugCommandType.RUN_TO_PAUSE,
            "Run until the next pause marker",
        ),
        DebugToolbarButtonSpec(
            "Restart",
            DebugCommandType.RESTART,
            "Restart the current debug session",
        ),
        DebugToolbarButtonSpec(
            "Choose",
            DebugCommandType.CHOOSE_SOURCE_GROUP,
            "Choose a well/image set for debug execution",
        ),
        DebugToolbarButtonSpec(
            "Random",
            DebugCommandType.RANDOM_SOURCE_GROUP,
            "Choose a random well/image set for debug execution",
        ),
        DebugToolbarButtonSpec("Stop", DebugCommandType.STOP, "Stop debug mode"),
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
        layout = QVBoxLayout(self)
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
        layout.addWidget(self.button_panel)

    def _emit_command(self, action_id: str) -> None:
        self.command_requested.emit(DebugCommand(DebugCommandType(action_id)))

    def set_controls_enabled(self, enabled: bool) -> None:
        """Enable or disable all debug controls together."""

        for button in self.buttons.values():
            button.setEnabled(enabled)


__all__ = ("DebugToolbarButtonSpec", "DebugToolbarWidget")
