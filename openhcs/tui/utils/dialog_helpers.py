"""
Dialog helper utilities for the TUI.

This module provides utilities for creating and showing dialogs.
"""

import logging
import asyncio
from pathlib import Path
from typing import Any, Optional, Callable, List

from prompt_toolkit.application import get_app
from prompt_toolkit.layout.containers import HSplit
from prompt_toolkit.widgets import Button, Dialog, Label, TextArea
from prompt_toolkit.key_binding import merge_key_bindings

logger = logging.getLogger(__name__)

# SafeButton eliminated - use Button directly

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
            Button("OK", handler=ok_handler, width=len("OK") + 2)
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

async def prompt_for_file_dialog(
    title: str,
    prompt_message: str,
    app_state: Any,
    filemanager=None,
    selection_mode: str = "files",  # "files", "directories", or "both"
    filter_extensions: Optional[List[str]] = None
) -> Optional[str]:
    """
    Show a dialog that allows the user to select a single file or directory using FileManagerBrowser.

    Args:
        title: The dialog title
        prompt_message: The message to show to the user
        app_state: The TUIState or equivalent, expected to have a `show_dialog` method
        filemanager: Shared FileManager instance from context (required)
        selection_mode: "files", "directories", or "both"
        filter_extensions: Optional list of file extensions to filter (e.g., [".pipeline", ".step"])

    Returns:
        Selected file/directory path as string if selected, otherwise None.
    """
    app = get_app()
    future = asyncio.Future()

    # Import FileManagerBrowser and required types
    from openhcs.tui.editors.file_browser import FileManagerBrowser, SelectionMode
    from openhcs.io.base import Backend

    # Validate required filemanager parameter
    if not filemanager:
        logger.error("prompt_for_file_dialog: filemanager parameter is required")
        return None

    # Map selection mode string to enum
    mode_map = {
        "files": SelectionMode.FILES_ONLY,
        "directories": SelectionMode.DIRECTORIES_ONLY,
        "both": SelectionMode.BOTH
    }

    if selection_mode not in mode_map:
        logger.error(f"Invalid selection_mode: {selection_mode}")
        return None

    selection_mode_enum = mode_map[selection_mode]

    # Track selected path
    selected_path = None

    async def on_path_selected(paths: list):
        """Handle path selection from FileManagerBrowser."""
        nonlocal selected_path
        if paths:
            selected_path = str(paths[0])  # Take first path for single selection
            future.set_result(selected_path)

    async def on_browser_cancel():
        """Handle cancel from FileManagerBrowser."""
        future.set_result(None)

    # Create FileManagerBrowser for file/directory selection
    file_browser = FileManagerBrowser(
        file_manager=filemanager,
        backend=Backend.DISK,  # Explicit backend in correct position
        on_path_selected=on_path_selected,
        on_cancel=on_browser_cancel,
        initial_path=Path.home(),  # Start from user home directory
        selection_mode=selection_mode_enum,
        allow_multiple=False,  # Add missing parameter
        show_hidden_files=False,
        filter_extensions=filter_extensions
    )

    # Create dialog body
    dialog_body = HSplit([
        Label(prompt_message),
        Label(""),
        file_browser,  # The file browser component
    ])

    dialog = Dialog(
        title=title,
        body=dialog_body,
        buttons=[
            Button("Cancel", handler=cancel_dialog(future), width=len("Cancel") + 2),
        ],
        width=100,  # Wide enough for file browser
        modal=True
    )

    # Initialize the file browser by loading initial directory
    file_browser.start_load()

    # If app_state has a show_dialog method, use it
    if hasattr(app_state, 'show_dialog') and callable(getattr(app_state, 'show_dialog')):
        # Show the dialog
        await app_state.show_dialog(dialog, result_future=future)
    else:
        # Fallback if app_state doesn't have a show_dialog
        logger.error("prompt_for_file_dialog: app_state does not have show_dialog method")
        future.set_result(None)

    return await future

async def prompt_for_multi_folder_dialog(title: str, prompt_message: str, app_state: Any, filemanager=None) -> Optional[list]:
    """
    Displays a modal dialog with FileManagerBrowser for selecting multiple folders.

    Args:
        title: The title of the dialog.
        prompt_message: The message to display above the file browser.
        app_state: The TUIState or equivalent, expected to have a `show_dialog` method.
        filemanager: Shared FileManager instance from context (required).

    Returns:
        List of folder paths if folders are selected, otherwise None.
    """
    app = get_app()
    future = asyncio.Future()

    # Import FileManagerBrowser and required types
    from openhcs.tui.editors.file_browser import FileManagerBrowser, SelectionMode
    from openhcs.io.base import Backend

    # Validate required filemanager parameter
    if not filemanager:
        logger.error("prompt_for_multi_folder_dialog: filemanager parameter is required")
        return None

    # Track selected folders
    selected_folders = []

    async def on_folder_selected(paths: list):
        """Handle folder selection from FileManagerBrowser."""
        nonlocal selected_folders
        # Add selected paths to our list (allowing multiple selections)
        for path in paths:
            path_str = str(path)
            if path_str not in selected_folders:
                selected_folders.append(path_str)

        # Update the selected folders display
        update_selected_display()

    async def on_browser_cancel():
        """Handle cancel from FileManagerBrowser."""
        # Don't close the main dialog, just clear browser
        pass

    def update_selected_display():
        """Update the display of selected folders."""
        if selected_folders:
            selected_text = "\n".join(f"• {folder}" for folder in selected_folders)
        else:
            selected_text = "No folders selected yet"

        selected_display.text = selected_text
        app.invalidate()

    def accept_selection():
        """Accept the current selection."""
        if selected_folders:
            future.set_result(selected_folders)
        else:
            future.set_result(None)

    def clear_selection():
        """Clear all selected folders."""
        nonlocal selected_folders
        selected_folders = []
        update_selected_display()

    # Create FileManagerBrowser for directory selection
    file_browser = FileManagerBrowser(
        file_manager=filemanager,
        backend=Backend.DISK,  # Explicit backend in correct position
        on_path_selected=on_folder_selected,
        on_cancel=on_browser_cancel,
        initial_path=Path.home(),  # Start from user home directory
        selection_mode=SelectionMode.DIRECTORIES_ONLY,  # Select directories only
        allow_multiple=True,  # Add missing parameter for multi-folder selection
        show_hidden_files=False
    )

    # Create display for selected folders
    selected_display = Label("No folders selected yet")

    # Create dialog body
    dialog_body = HSplit([
        Label(prompt_message),
        Label(""),
        Label("Browse and select folders (you can select multiple folders one at a time):"),
        file_browser,  # The file browser component
        Label(""),
        Label("Selected folders:"),
        selected_display,
    ])

    dialog = Dialog(
        title=title,
        body=dialog_body,
        buttons=[
            Button("Add More", handler=lambda: None, width=len("Add More") + 2),  # Browser handles this
            Button("Clear All", handler=clear_selection, width=len("Clear All") + 2),
            Button("OK", handler=accept_selection, width=len("OK") + 2),
            Button("Cancel", handler=cancel_dialog(future), width=len("Cancel") + 2),
        ],
        width=140,  # Wider for file browser
        modal=True
    )

    # Attach file browser key bindings to the dialog's frame
    file_browser_kb = file_browser.get_key_bindings()
    if file_browser_kb and hasattr(dialog.container, 'body') and hasattr(dialog.container.body, 'key_bindings'):
        # Merge the file browser key bindings with the dialog's existing key bindings
        dialog.container.body.key_bindings = merge_key_bindings([
            dialog.container.body.key_bindings,
            file_browser_kb
        ])

    # Initialize the file browser by loading initial directory
    file_browser.start_load()

    # If app_state has a show_dialog method, use it
    if hasattr(app_state, 'show_dialog') and callable(getattr(app_state, 'show_dialog')):
        # Show the dialog
        await app_state.show_dialog(dialog, result_future=future)
    else:
        # Fallback if app_state doesn't have a show_dialog
        logger.error("prompt_for_multi_folder_dialog: app_state does not have show_dialog method")
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
        if not future.done():
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
