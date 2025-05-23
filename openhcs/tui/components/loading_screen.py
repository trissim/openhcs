"""
Loading screen component for the OpenHCS TUI.

This module provides a loading screen that can be displayed while the TUI is initializing.
"""
import asyncio
import logging
from typing import Optional, Callable, List

from prompt_toolkit.application import get_app
from prompt_toolkit.filters import Condition
from prompt_toolkit.layout import (
    Container, HSplit, VSplit, Window, ConditionalContainer,
    FormattedTextControl, Dimension
)
from prompt_toolkit.layout.mouse_handlers import MouseHandlers
from prompt_toolkit.layout.screen import Screen, WritePosition
from prompt_toolkit.key_binding.key_bindings import KeyBindingsBase
from prompt_toolkit.widgets import Box, Label, Frame

from .spinner import Spinner

logger = logging.getLogger(__name__)


class LoadingScreen(Container):
    """A loading screen with a spinner and message."""

    def __init__(
        self,
        message: str = "Loading OpenHCS TUI...",
        on_complete: Optional[Callable[[], None]] = None
    ):
        """
        Initialize the loading screen.

        Args:
            message: The message to display.
            on_complete: Callback to execute when loading is complete.
        """
        self.message = message
        self.on_complete = on_complete
        self.is_loading = True
        self.spinner = Spinner(style="class:spinner")

        # Create the loading screen layout
        self.loading_container = HSplit([
            Window(height=1),  # Top padding
            VSplit([
                Window(width=1),  # Left padding
                HSplit([
                    Label(f"{message}"),
                    VSplit([
                        Label(" "),
                        self.spinner,
                    ]),
                ]),
                Window(width=1),  # Right padding
            ]),
            Window(height=1),  # Bottom padding
        ])

        # Wrap in a box with a border
        self.loading_box = Box(
            self.loading_container,
            padding=1,
            style="class:loading-box"
        )

        # Create a frame around the box
        self.loading_frame = Frame(
            self.loading_box,
            title="OpenHCS V1.0",
            style="class:loading-frame"
        )

        # Create a full-screen window with the loading frame centered
        self.full_screen = Box(
            self.loading_frame,
            padding_top=10,  # Center vertically
            style="class:loading-background"
        )

        # Create a conditional container that shows the loading screen only when loading
        self.container = ConditionalContainer(
            self.full_screen,
            filter=Condition(lambda: self.is_loading)
        )

    def __pt_container__(self):
        """Return the container to be rendered."""
        return self.container

    def start(self):
        """Start the loading animation."""
        self.is_loading = True
        self.spinner.start()
        get_app().invalidate()
        logger.info("Loading screen started")

    def complete(self):
        """Mark loading as complete and hide the loading screen."""
        self.is_loading = False
        self.spinner.stop()
        get_app().invalidate()
        logger.info("Loading screen completed")

        if self.on_complete:
            self.on_complete()

    # Implement abstract methods by delegating to the container
    def get_children(self):
        return self.container.get_children()

    def preferred_width(self, max_available_width):
        # Use the container's preferred width, but ensure it's at least 40 characters
        container_width = self.container.preferred_width(max_available_width)
        if container_width.min is not None and container_width.min > 0:
            return container_width
        return Dimension(min=40, preferred=40)

    def preferred_height(self, width, max_available_height):
        # Use the container's preferred height, but ensure it's at least 5 lines
        container_height = self.container.preferred_height(width, max_available_height)
        if container_height.min is not None and container_height.min > 0:
            return container_height
        return Dimension(min=5, preferred=5)

    def reset(self):
        self.container.reset()

    def write_to_screen(self, screen, mouse_handlers, write_position,
                        parent_style, erase_bg, z_index):
        self.container.write_to_screen(screen, mouse_handlers, write_position,
                                      parent_style, erase_bg, z_index)

    def mouse_handler(self, mouse_event):
        """Handle mouse events."""
        return self.container.mouse_handler(mouse_event)

    def is_modal(self):
        """Return whether this container is modal."""
        return False

    def get_key_bindings(self):
        """Return key bindings for this container."""
        return None
