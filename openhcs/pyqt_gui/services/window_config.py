"""
Window Configuration - Declarative specs for application windows.

Centralized window definitions replacing hardcoded creation code.
"""

from dataclasses import dataclass
from enum import Enum, nonmember
from collections.abc import Callable

from PyQt6.QtWidgets import QDialog, QWidget


WindowPresentationLeaf = Callable[[QWidget], None]


class StartupWindowPresentation(Enum):
    """Closed startup-presentation policies owned by window declarations."""

    _keep_visible = nonmember(lambda _window: None)
    _hide = nonmember(lambda window: window.hide())

    def __new__(
        cls,
        value: str,
        presenter: WindowPresentationLeaf,
    ) -> "StartupWindowPresentation":
        member = object.__new__(cls)
        member._value_ = value
        member._presenter = presenter
        return member

    KEEP_VISIBLE = ("keep_visible", _keep_visible)
    HIDE = ("hide", _hide)

    def present(self, window: QWidget) -> None:
        """Execute this declaration member's presentation leaf."""

        self._presenter(window)


@dataclass(frozen=True)
class WindowSpec:
    """
    Declarative specification for an application window.

    Centralizes window configuration (widget, title, size) in one place.
    """

    window_id: str
    title: str
    window_class: type[QDialog]
    initialize_on_startup: bool = False
    startup_presentation: StartupWindowPresentation = (
        StartupWindowPresentation.KEEP_VISIBLE
    )

    def apply_startup_presentation(
        self,
        window: QWidget,
        *,
        requested: bool,
    ) -> None:
        """Apply declaration-owned startup presentation when requested."""

        if requested and self.initialize_on_startup:
            self.startup_presentation.present(window)
