"""PyQt debug/test-mode toolbar built on core debug commands."""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QWidget

from openhcs.core.debug import DebugCommand, DebugCommandType


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

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.buttons: dict[DebugCommandType, QPushButton] = {}
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 0, 2, 0)
        layout.setSpacing(4)

        for spec in self.BUTTON_SPECS:
            button = QPushButton(spec.label)
            button.setToolTip(spec.tooltip)
            button.clicked.connect(
                lambda _checked=False, command_type=spec.command_type: (
                    self.command_requested.emit(DebugCommand(command_type))
                )
            )
            self.buttons[spec.command_type] = button
            layout.addWidget(button)
        layout.addStretch(1)

    def set_controls_enabled(self, enabled: bool) -> None:
        """Enable or disable all debug controls together."""

        for button in self.buttons.values():
            button.setEnabled(enabled)


__all__ = ("DebugToolbarButtonSpec", "DebugToolbarWidget")
