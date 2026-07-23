"""Shared OpenHCS manager-widget declaration mixins."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from openhcs.agent.dto.knowledge import KnowledgeBaseDocumentTarget
    from pyqt_reactive.widgets.shared.responsive_groupbox_title import (
        ResponsiveGroupBoxTitle,
    )


class OpenHCSSingleRowActionManagerMixin:
    """Manager widgets whose button bar is rendered as one row."""

    BUTTON_GRID_COLUMNS = 0
    ACTION_REGISTRY = {}
    HELP_KNOWLEDGE_TARGET: ClassVar["KnowledgeBaseDocumentTarget | None"] = None

    def install_context_help_button(
        self,
        *,
        title_layout: "ResponsiveGroupBoxTitle",
        object_name: str,
    ):
        """Place the managed knowledge-browser action beside the manager title."""
        from pyqt_reactive.widgets.shared.clickable_help_components import (
            HelpButton,
            HelpContext,
        )

        help_button = HelpButton(
            help_context=HelpContext(
                color_scheme=self.color_scheme,
                parent=title_layout,
            ),
            text="Help",
        )
        help_button.clicked.disconnect(help_button.show_help)
        help_button.clicked.connect(self.show_managed_help)
        help_button.setObjectName(object_name)
        help_button.setToolTip("Open the OpenHCS knowledge base")
        title_layout.set_help_widget(help_button)
        return help_button

    def show_managed_help(self) -> None:
        """Open the canonical managed knowledge window at this manager's target."""
        from openhcs.pyqt_gui.services.ui_window_ids import OpenHCSUiWindowId
        from openhcs.pyqt_gui.windows.help_window import HelpWindow
        from pyqt_reactive.services.window_manager import WindowManager

        self.service_adapter.main_window.show_window(
            OpenHCSUiWindowId.knowledge_base,
            hide_if_startup=False,
        )
        help_window = WindowManager.get_window(OpenHCSUiWindowId.knowledge_base)
        if not isinstance(help_window, HelpWindow):
            raise RuntimeError("Managed knowledge-base window is unavailable")
        if self.HELP_KNOWLEDGE_TARGET is not None:
            help_window.open_target(self.HELP_KNOWLEDGE_TARGET)
