"""
Spinner component for the OpenHCS TUI.

This module provides a simple spinner component that can be used to indicate loading.
"""
import asyncio
import time
from typing import List, Optional, Callable

from prompt_toolkit.application import get_app
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.layout import Container, Dimension
from prompt_toolkit.layout.mouse_handlers import MouseHandlers
from prompt_toolkit.layout.screen import Screen, WritePosition
from prompt_toolkit.key_binding.key_bindings import KeyBindingsBase
from prompt_toolkit.widgets import Label


class Spinner(Container):
    """A spinner component that rotates through a sequence of characters."""

    def __init__(
        self,
        spinner_chars: Optional[List[str]] = None,
        interval: float = 0.1,
        style: str = "",
        done_callback: Optional[Callable[[], None]] = None
    ):
        """
        Initialize the spinner.

        Args:
            spinner_chars: List of characters to cycle through. Defaults to a simple spinner.
            interval: Time in seconds between spinner updates.
            style: Style string for the spinner.
            done_callback: Optional callback to call when the spinner is stopped.
        """
        self.spinner_chars = spinner_chars or ['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷']
        self.interval = interval
        self.style = style
        self.done_callback = done_callback
        self.current_index = 0
        self.running = False
        self.task = None

        self.label = Label(
            text=lambda: FormattedText([(self.style, self.spinner_chars[self.current_index])]),
            dont_extend_width=True
        )

    def __pt_container__(self):
        """Return the container to be rendered."""
        return self.label

    async def spin(self):
        """Animate the spinner."""
        self.running = True
        while self.running:
            self.current_index = (self.current_index + 1) % len(self.spinner_chars)
            get_app().invalidate()
            await asyncio.sleep(self.interval)

    def start(self):
        """Start the spinner animation."""
        if not self.running:
            self.running = True
            app = get_app()
            self.task = app.create_background_task(self.spin())

    def stop(self):
        """Stop the spinner animation."""
        self.running = False
        if self.done_callback:
            self.done_callback()

    # Implement abstract methods
    def get_children(self):
        # Label doesn't have get_children, so return an empty list
        return []

    def _get_text(self):
        """Get the current spinner character."""
        return self.spinner_chars[self.current_index]

    def preferred_width(self, max_available_width):
        # Container should return Dimension, not int
        return Dimension.exact(3)  # Exact width of 3 characters

    def preferred_height(self, width, max_available_height):
        # Always exactly 1 line high
        return Dimension.exact(1)

    def reset(self):
        # Label doesn't have a reset method, so we do nothing here
        pass

    def write_to_screen(self, screen, mouse_handlers, write_position,
                        parent_style, erase_bg, z_index):
        """Write the spinner to the screen."""
        # Label doesn't implement write_to_screen, so we need to create a Window with the label's content
        from prompt_toolkit.layout import Window
        from prompt_toolkit.layout.controls import FormattedTextControl

        # Create a control with the current spinner character
        spinner_text = [(self.style, self.spinner_chars[self.current_index])]
        control = FormattedTextControl(text=spinner_text)

        # Create a window with the control
        window = Window(content=control)

        # Write the window to the screen
        window.write_to_screen(screen, mouse_handlers, write_position,
                              parent_style, erase_bg, z_index)

    def mouse_handler(self, mouse_event):
        """Handle mouse events."""
        return self.label.mouse_handler(mouse_event)

    def is_modal(self):
        """Return whether this container is modal."""
        return False

    def get_key_bindings(self):
        """Return key bindings for this container."""
        return None
