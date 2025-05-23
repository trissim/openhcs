from typing import Any, Optional # Added Optional
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

    error_dialog = Dialog(
        title=title,
        body=HSplit([
            Label(message),
        ]),
        buttons=[
            Button("OK", handler=lambda: app.exit_dialog()) # Use app.exit_dialog()
        ],
        width=80, # Standard width
        modal=True
    )

    # If app_state has a specific method to show dialogs, prefer that.
    # Otherwise, use the general prompt_toolkit way.
    if hasattr(app_state, 'show_dialog_from_util') and callable(getattr(app_state, 'show_dialog_from_util')):
        # This assumes TUIState might have a preferred way to manage dialogs
        # For now, we'll make it compatible with direct prompt_toolkit usage too.
        # await app_state.show_dialog_from_util(error_dialog) # Example if TUIState handles it
        pass # Fall through to direct display if not handled by app_state

    # Standard way to show a dialog if not handled by a specific app_state method
    # This part might need adjustment based on how TUIState manages its layout and dialogs.
    # For a simple utility, directly manipulating app.layout might be too intrusive.
    # A better approach for a generic utility might be to return the dialog
    # and let the caller handle displaying it, or use a more robust dialog manager service.

    # For now, let's assume a simple direct display for demonstration,
    # but acknowledge this might need refinement for a complex TUI.
    # The ExternalEditorService used `await self.state.show_dialog(dialog)`
    # which implies the 'state' object (passed as app_state here) has a show_dialog method.

    if hasattr(app_state, 'show_dialog') and callable(getattr(app_state, 'show_dialog')):
         await app_state.show_dialog(error_dialog)
    else:
        # Fallback if app_state doesn't have a show_dialog or if app_state is None
        # This direct manipulation is generally not recommended for a library utility
        # but included for completeness if no app_state context is available.
        # In a real scenario, this utility would likely require 'app_state' or similar context.
        # For TUIState, it's expected to have a `show_dialog` method.
        # If not, this utility might not function correctly without direct app access.
        pass # The app_state.show_dialog should handle display.

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
        # app.exit_dialog() # Handled by TUIState.show_dialog typically

    def cancel_dialog():
        future.set_result(None)
        # app.exit_dialog() # Handled by TUIState.show_dialog

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
        modal=True,
        # Ensure the TextArea is focused when the dialog appears
        # This might require the dialog to be shown via a method that can handle focus,
        # like app_state.show_dialog if it sets focus.
    )
    
    # Focus the text area after the dialog is part of the layout.
    # This is tricky with a utility function. The caller (TUIState.show_dialog)
    # would ideally handle focusing the first focusable element in the dialog.
    # If TUIState.show_dialog doesn't handle focus, we might need to schedule it.
    # async def set_focus_task():
    #     app.layout.focus(path_text_area)
    # get_app().create_background_task(set_focus_task())


    if hasattr(app_state, 'show_dialog') and callable(getattr(app_state, 'show_dialog')):
        # The show_dialog method is responsible for displaying and managing the dialog lifecycle
        # including focus and how the result is obtained (e.g., via the future).
        # It should also handle app.exit_dialog() internally when a button is pressed.
        await app_state.show_dialog(dialog, result_future=future)
    else:
        # Fallback: This part is problematic for a generic utility as it assumes direct control
        # over the app loop and layout, which is not ideal.
        # A robust TUI framework would have a dialog manager.
        logger.warning("prompt_for_path_dialog: app_state does not have show_dialog. Dialog may not function correctly.")
        future.set_result(None) # Cannot display dialog properly

    return await future