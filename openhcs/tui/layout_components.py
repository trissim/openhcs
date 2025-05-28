"""
Layout component builders for OpenHCS TUI.

This module contains modular layout component builders that reduce
complexity in the main TUI architecture by separating layout concerns.

🔒 Clause 3: Declarative Primacy
Layout components are built declaratively with clear separation of concerns.

🔒 Clause 245: Modular Architecture
Layout building is separated from main TUI logic for better modularity.
"""

from typing import TYPE_CHECKING, Optional, Any
from prompt_toolkit.layout import (
    Container, HSplit, VSplit, Window,
    DynamicContainer, FloatContainer, Float, Dimension
)
from prompt_toolkit.widgets import Frame, Label, Button, Box
from .components.framed_button import FramedButton
from prompt_toolkit.application import get_app
import logging

if TYPE_CHECKING:
    from .tui_state import TUIState

logger = logging.getLogger(__name__)




class LayoutComponentBuilder:
    """
    Builder for modular TUI layout components.

    This class provides methods to build individual layout components
    in a modular way, reducing complexity in the main TUI architecture.
    """

    def __init__(self, state: "TUIState", context: Any):
        """Initialize the layout component builder."""
        self.state = state
        self.context = context

    def build_top_bar(self, show_global_settings_handler, show_help_handler) -> Frame:
        """
        Build the top bar with global settings and help buttons.

        Args:
            show_global_settings_handler: Handler for global settings button
            show_help_handler: Handler for help button

        Returns:
            Frame containing the top bar layout
        """
        return Frame(
            VSplit([
                FramedButton(
                    "Global Settings",
                    handler=lambda: get_app().create_background_task(show_global_settings_handler()),
                    width=18
                ),
                Window(width=1, char=' '),  # Small spacer
                FramedButton(
                    "Help",
                    handler=lambda: get_app().create_background_task(show_help_handler()),
                    width=8
                ),
                Window(width=Dimension(weight=1), char=' '),  # Flexible spacer
                Label("OpenHCS V1.0", style="class:app-title", dont_extend_width=True)
            ], padding=0),
            height=Dimension.exact(3),
            style="class:top-bar-frame"
        )

    def build_pane_frame(self, title: str, buttons_container: Container,
                        content_container: Container, style_class: str) -> Frame:
        """
        Build a standardized pane frame with title, buttons, and content.

        Args:
            title: Title for the pane
            buttons_container: Container with pane buttons
            content_container: Container with pane content
            style_class: CSS style class for the frame

        Returns:
            Frame containing the pane layout
        """
        return Frame(
            HSplit([
                # Title bar with buttons directly underneath
                HSplit([
                    # Title bar
                    VSplit([
                        Label(f" {title} ", style="class:frame.title"),
                        Window(width=Dimension(weight=1), char=' '),  # Flexible spacer
                    ], height=1, style="class:frame.title"),
                    # Buttons bar directly under title
                    buttons_container
                ]),
                # Content area
                content_container
            ]),
            height=Dimension(weight=1),
            width=Dimension(weight=1),
            style=style_class
        )

    def build_main_content_area(self, left_pane: Container, right_pane: Container) -> VSplit:
        """
        Build the main content area with left and right panes.

        Args:
            left_pane: Left pane container (Plate Manager)
            right_pane: Right pane container (Pipeline Editor)

        Returns:
            VSplit containing the main content layout
        """
        return VSplit([
            left_pane,
            right_pane
        ], height=Dimension(weight=1), padding=0)

    def build_status_bar(self, status_bar_component: Container) -> Frame:
        """
        Build the status bar frame.

        Args:
            status_bar_component: Status bar component

        Returns:
            Frame containing the status bar
        """
        return Frame(
            status_bar_component,
            height=Dimension.exact(3),
            style="class:status-bar-frame"
        )

    def build_root_layout(self, top_bar: Container, main_content: Container,
                         status_bar: Container, loading_screen: Container) -> FloatContainer:
        """
        Build the complete root layout.

        Args:
            top_bar: Top bar container
            main_content: Main content area container
            status_bar: Status bar container
            loading_screen: Loading screen component

        Returns:
            FloatContainer with the complete layout
        """
        main_layout = HSplit([
            top_bar,
            main_content,
            status_bar
        ], padding=0)

        return FloatContainer(
            content=main_layout,
            floats=[
                Float(
                    content=loading_screen,
                    transparent=False,
                )
            ]
        )

    def build_dynamic_buttons_container(self, component: Any, method_name: str,
                                      placeholder_text: str) -> DynamicContainer:
        """
        Build a dynamic container for component buttons.

        Args:
            component: Component that provides buttons
            method_name: Method name to call for buttons
            placeholder_text: Placeholder text when component not available

        Returns:
            DynamicContainer for buttons
        """
        def get_buttons():
            if component and hasattr(component, method_name):
                return getattr(component, method_name)()
            else:
                return Box(Label(f"[{placeholder_text}]"), padding_left=1)

        return DynamicContainer(get_buttons)
