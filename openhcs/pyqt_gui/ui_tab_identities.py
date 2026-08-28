"""Lightweight nominal tab identities for OpenHCS desktop windows."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Self

from pyqt_reactive.services.tab_identity import TabLabelDeclarationMixin

if TYPE_CHECKING:
    from pyqt_reactive.widgets.shared import ActionTabbedWindowBody


class DualEditorTab(TabLabelDeclarationMixin, str, Enum):
    """Dual-editor tabs with declaration-owned public labels."""

    def __new__(cls, value: str, label: str) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member.label = label
        return member

    STEP_SETTINGS = ("step_settings", "Step Settings")
    FUNCTION_PATTERN = ("function_pattern", "Function Pattern")
    ARTIFACTS = ("artifacts", "Artifacts")

    def select(self, body: ActionTabbedWindowBody) -> None:
        """Select this declaration in one live dual-editor body."""

        labels = tuple(
            body.tab_bar.tabText(index) for index in range(body.tab_bar.count())
        )
        body.setCurrentIndex(self.index_in(labels))


class PlateViewerTab(TabLabelDeclarationMixin, str, Enum):
    """Plate-viewer tabs with declaration-owned public labels."""

    def __new__(cls, value: str, label: str) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member.label = label
        return member

    IMAGE_BROWSER = ("image_browser", "Image Browser")
    METADATA = ("metadata", "Metadata")
