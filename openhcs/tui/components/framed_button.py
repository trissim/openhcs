"""
Framed Button Component for OpenHCS TUI.

This module provides a custom button component with a frame around it.
"""
from typing import Callable, Optional

from prompt_toolkit.layout import Container, HSplit, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.widgets import Button, Frame, Box
from prompt_toolkit.mouse_events import MouseEventType


class FramedButton:
    """
    A custom button with a frame around it.

    This component creates a button with a proper frame around it,
    rather than just angle brackets.
    """
    def __init__(
        self,
        text: str,
        handler: Optional[Callable[[], None]] = None,
        width: Optional[int] = None,
        style: str = "class:button.frame"
    ):
        """
        Initialize a framed button.

        Args:
            text: The text to display on the button
            handler: The function to call when the button is clicked
            width: The width of the button (None for auto-width)
            style: The style to apply to the frame
        """
        self.text = text
        self.handler = handler
        self.width = width
        self.style = style

        # Create a custom button with fixed text (no formatting)
        # This avoids the center-alignment issue in prompt_toolkit's Button
        control = FormattedTextControl([("class:button.text", f" {text} ")])

        # Add mouse handler to the control
        def mouse_handler(mouse_event):
            if mouse_event.event_type == MouseEventType.MOUSE_UP and self.handler:
                self.handler()
                return True
            return False

        control.mouse_handler = mouse_handler

        button_window = Window(
            content=control,
            # If width is specified, use it; otherwise, adapt to content
            width=width if width is not None else Dimension(min=len(text) + 2, preferred=len(text) + 2),
            dont_extend_width=True
        )

        # Create the frame around the button
        self.container = Frame(
            Box(
                button_window,
                padding=0,
                padding_left=1,
                padding_right=1
            ),
            style=style
        )

    def __pt_container__(self) -> Container:
        """Return the container for prompt_toolkit."""
        return self.container
