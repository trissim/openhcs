from typing import Any, Optional # Added Optional
import logging
from prompt_toolkit.application import get_app
from prompt_toolkit.layout.containers import HSplit
from prompt_toolkit.widgets import Button, Dialog, Label, TextArea # Added TextArea
import asyncio # Added for Future

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
            Button("OK", handler=ok_handler)
        ],
        width=80, # Standard width
        modal=True
    )

    # If app_state has a show_dialog method, use it
    if hasattr(app_state, 'show_dialog') and callable(getattr(app_state, 'show_dialog')):
        await app_state.show_dialog(error_dialog, result_future=future)
    else:
        # Fallback if app_state doesn't have a show_dialog or if app_state is None
        logger = logging.getLogger(__name__)
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
        accept_handler=lambda buff: accept_path(buff.text) # Accept on Enter
    )

    dialog = Dialog(
        title=title,
        body=HSplit([
            Label(prompt_message),
            path_text_area,
        ]),
        buttons=[
            Button("OK", handler=lambda: accept_path(path_text_area.text)),
            Button("Cancel", handler=cancel_dialog),
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
        logger = logging.getLogger(__name__)
        logger.error("prompt_for_path_dialog: app_state does not have show_dialog method")
        future.set_result(None)

    return await future