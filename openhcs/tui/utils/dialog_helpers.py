"""
Dialog helper utilities for the TUI.

This module provides utilities for creating and showing dialogs.
"""

import logging
import asyncio
from typing import Any, Optional, Callable

from prompt_toolkit.application import get_app
from prompt_toolkit.layout.containers import HSplit
from prompt_toolkit.widgets import Button, Dialog, Label, TextArea

logger = logging.getLogger(__name__)

# Define SafeButton locally to avoid circular imports
class SafeButton(Button):
    """Safe wrapper around Button that handles formatting errors."""

    def __init__(self, text="", handler=None, width=None, **kwargs):
        # Sanitize text before passing to parent
        if text is not None:
            text = str(text).replace('{', '{{').replace('}', '}}').replace(':', ' ')
        super().__init__(text=text, handler=handler, width=width, **kwargs)

    def _get_text_fragments(self):
        """Safe version that handles formatting errors gracefully."""
        try:
            return super()._get_text_fragments()
        except (ValueError, TypeError, AttributeError):
            # Fallback to simple text formatting without centering
            text = str(self.text) if self.text is not None else ""
            safe_text = text.replace('{', '{{').replace('}', '}}')
            return [("class:button", f" {safe_text} ")]

async def show_error_dialog(title: str, message: str, app_state: Optional[Any] = None):
    """
    Displays a modal error dialog.

    Args:
        title: The title of the dialog.
        message: The error message to display.
        app_state: The TUIState or equivalent application state object,
                   if needed for more complex dialog interactions in the future.
                   Currently used to ensure the dialog is shown in the main app event loop.
    """
    # Ensure we have an application instance
    app = get_app()

    # Create a future to track the dialog result
    future = asyncio.Future()

    # Define button handlers
    def ok_handler():
        future.set_result(True)

    error_dialog = Dialog(
        title=title,
        body=HSplit([
            Label(message),
        ]),
        buttons=[
            SafeButton("OK", handler=ok_handler)
        ],
        width=80,  # Standard width
        modal=True
    )

    # If app_state has a show_dialog method, use it
    if hasattr(app_state, 'show_dialog') and callable(getattr(app_state, 'show_dialog')):
        await app_state.show_dialog(error_dialog, result_future=future)
    else:
        # Fallback if app_state doesn't have a show_dialog or if app_state is None
        logger.error("show_error_dialog: app_state does not have show_dialog method")
        future.set_result(False)

    # Wait for the dialog to complete
    return await future

async def prompt_for_path_dialog(title: str, prompt_message: str, app_state: Any, initial_value: str = "") -> Optional[str]:
    """
    Displays a modal dialog prompting the user for a file path.

    Args:
        title: The title of the dialog.
        prompt_message: The message to display above the input field.
        app_state: The TUIState or equivalent, expected to have a `show_dialog` method.
        initial_value: Optional initial value for the path input field.

    Returns:
        The entered path string if OK is pressed, otherwise None.
    """
    app = get_app()
    future = asyncio.Future()

    def accept_path(path_text: str):
        future.set_result(path_text)

    def cancel_dialog():
        future.set_result(None)

    path_text_area = TextArea(
        text=initial_value,
        multiline=False,
        height=1,
        prompt="Path: ",
        accept_handler=lambda buff: accept_path(buff.text)  # Accept on Enter
    )

    dialog = Dialog(
        title=title,
        body=HSplit([
            Label(prompt_message),
            path_text_area,
        ]),
        buttons=[
            SafeButton("OK", handler=lambda: accept_path(path_text_area.text)),
            SafeButton("Cancel", handler=cancel_dialog),
        ],
        width=80,
        modal=True
    )

    # If app_state has a show_dialog method, use it
    if hasattr(app_state, 'show_dialog') and callable(getattr(app_state, 'show_dialog')):
        # Schedule a task to focus the text area after the dialog is shown
        async def focus_text_area():
            await asyncio.sleep(0.1)  # Short delay to ensure dialog is rendered
            app.layout.focus(path_text_area)

        app.create_background_task(focus_text_area())

        # Show the dialog
        await app_state.show_dialog(dialog, result_future=future)
    else:
        # Fallback if app_state doesn't have a show_dialog
        logger.error("prompt_for_path_dialog: app_state does not have show_dialog method")
        future.set_result(None)

    return await future

def ok_handler(future: asyncio.Future) -> Callable[[], None]:
    """
    Create a handler for OK button that sets the future result to True.

    Args:
        future: The future to set the result on

    Returns:
        A handler function
    """
    def handler():
        future.set_result(True)
    return handler

def cancel_dialog(future: asyncio.Future) -> Callable[[], None]:
    """
    Create a handler for Cancel button that sets the future result to None.

    Args:
        future: The future to set the result on

    Returns:
        A handler function
    """
    def handler():
        future.set_result(None)
    return handler

async def focus_text_area(text_area: TextArea) -> None:
    """
    Focus a text area after a short delay.

    Args:
        text_area: The text area to focus

    Returns:
        None
    """
    await asyncio.sleep(0.1)  # Short delay to ensure dialog is rendered
    get_app().layout.focus(text_area)
