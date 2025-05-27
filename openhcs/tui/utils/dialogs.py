"""
Dialog utilities ported from TUI with async/await support.

Provides error dialogs, file selection dialogs, and other UI interaction utilities
with proper async/await patterns for the hybrid TUI architecture.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Any
from prompt_toolkit.application import get_app
from prompt_toolkit.widgets import Dialog, Label, Button, TextArea
from prompt_toolkit.layout import HSplit, VSplit, Layout, FloatContainer, Float, Dimension
from prompt_toolkit.formatted_text import HTML

logger = logging.getLogger(__name__)

async def show_error_dialog(title: str, message: str) -> None:
    """
    Show error dialog with OK button.

    Args:
        title: Dialog title
        message: Error message to display
    """
    try:
        app = get_app()

        # Create dialog content
        dialog_content = HSplit([
            Label(HTML(f"<style fg='ansired'>{message}</style>")),
            Label(""),  # Spacer
        ])

        # Result future for dialog completion
        result_future = asyncio.Future()

        def ok_handler():
            if not result_future.done():
                result_future.set_result(True)

        # Create dialog
        dialog = Dialog(
            title=title,
            body=dialog_content,
            buttons=[
                Button("OK", handler=ok_handler)
            ],
            width=60,
            modal=True
        )

        # Show dialog
        previous_layout = app.layout
        float_container = FloatContainer(
            content=previous_layout.container,
            floats=[
                Float(content=dialog)
            ]
        )

        app.layout = Layout(float_container)
        app.layout.focus(dialog)

        # Wait for dialog completion
        await result_future

        # Restore previous layout
        app.layout = previous_layout

    except Exception as e:
        logger.error(f"Failed to show error dialog: {e}")

async def prompt_for_path_dialog(
    title: str,
    initial_path: Optional[Path] = None,
    file_types: Optional[List[str]] = None,
    save_mode: bool = False
) -> Optional[Path]:
    """
    Show file selection dialog using FileManagerBrowser.

    Args:
        title: Dialog title
        initial_path: Initial path to display
        file_types: List of allowed file extensions (e.g., ['.func', '.step'])
        save_mode: If True, allows creating new files

    Returns:
        Selected path or None if cancelled
    """
    try:
        from ..components.file_browser import FileManagerBrowser
        from openhcs.io.filemanager import FileManager
        from openhcs.constants.constants import Backend

        app = get_app()

        # Create FileManager instance
        file_manager = FileManager()

        # Result future
        result_future = asyncio.Future()

        async def on_path_selected(selected_paths: List[Path]):
            """Handle path selection."""
            if selected_paths:
                selected_path = selected_paths[0]  # Take first selection

                # Validate file type if specified
                if file_types and selected_path.suffix not in file_types:
                    await show_error_dialog(
                        "Invalid File Type",
                        f"Please select a file with extension: {', '.join(file_types)}"
                    )
                    return

                # Check file exists for load mode
                if not save_mode and not selected_path.exists():
                    await show_error_dialog(
                        "File Not Found",
                        f"File does not exist: {selected_path}"
                    )
                    return

                if not result_future.done():
                    result_future.set_result(selected_path)

        async def on_cancel():
            """Handle cancellation."""
            if not result_future.done():
                result_future.set_result(None)

        # Create file browser
        file_browser = FileManagerBrowser(
            file_manager=file_manager,
            on_path_selected=on_path_selected,
            on_cancel=on_cancel,
            initial_path=initial_path,
            backend=Backend.DISK,  # Default to disk backend
            select_files=True,
            select_multiple=False,
            filter_extensions=file_types
        )

        # Create dialog
        dialog = Dialog(
            title=title,
            body=file_browser,
            buttons=[],  # File browser has its own buttons
            width=Dimension(preferred=100, max=120),
            height=Dimension(preferred=30, max=40),
            modal=True,
            key_bindings=file_browser.get_key_bindings()
        )

        # Show dialog
        previous_layout = app.layout
        float_container = FloatContainer(
            content=previous_layout.container,
            floats=[
                Float(content=dialog)
            ]
        )

        app.layout = Layout(float_container)
        app.layout.focus(file_browser.get_initial_focus_target())

        # Wait for result
        result = await result_future

        # Restore layout
        app.layout = previous_layout

        return result

    except Exception as e:
        logger.error(f"Failed to show path dialog: {e}")
        return None

async def prompt_for_directory_dialog(
    title: str,
    initial_path: Optional[Path] = None
) -> Optional[Path]:
    """
    Show directory selection dialog using FileManagerBrowser.

    Args:
        title: Dialog title
        initial_path: Initial path to display

    Returns:
        Selected directory path or None if cancelled
    """
    try:
        from ..components.file_browser import FileManagerBrowser
        from openhcs.io.filemanager import FileManager
        from openhcs.constants.constants import Backend

        app = get_app()

        # Create FileManager instance
        file_manager = FileManager()

        # Result future
        result_future = asyncio.Future()

        async def on_path_selected(selected_paths: List[Path]):
            """Handle directory selection."""
            if selected_paths:
                selected_path = selected_paths[0]  # Take first selection

                # Ensure it's a directory
                if not selected_path.is_dir():
                    await show_error_dialog(
                        "Invalid Selection",
                        f"Please select a directory, not a file: {selected_path}"
                    )
                    return

                if not result_future.done():
                    result_future.set_result(selected_path)

        async def on_cancel():
            """Handle cancellation."""
            if not result_future.done():
                result_future.set_result(None)

        # Create file browser for directory selection
        file_browser = FileManagerBrowser(
            file_manager=file_manager,
            on_path_selected=on_path_selected,
            on_cancel=on_cancel,
            initial_path=initial_path or Path.home(),
            backend=Backend.DISK,
            select_files=False,  # Select directories only
            select_multiple=False,
            show_hidden_files=False
        )

        # Create dialog
        dialog = Dialog(
            title=title,
            body=file_browser,
            buttons=[],  # File browser has its own buttons
            width=Dimension(preferred=100, max=120),
            height=Dimension(preferred=30, max=40),
            modal=True,
            key_bindings=file_browser.get_key_bindings()
        )

        # Show dialog
        previous_layout = app.layout
        float_container = FloatContainer(
            content=previous_layout.container,
            floats=[
                Float(content=dialog)
            ]
        )

        app.layout = Layout(float_container)
        app.layout.focus(file_browser.get_initial_focus_target())

        # Wait for result
        result = await result_future

        # Restore layout
        app.layout = previous_layout

        return result

    except Exception as e:
        logger.error(f"Failed to show directory dialog: {e}")
        return None

async def show_confirmation_dialog(title: str, message: str) -> bool:
    """
    Show confirmation dialog with Yes/No buttons.

    Args:
        title: Dialog title
        message: Confirmation message

    Returns:
        True if Yes selected, False if No/Cancel
    """
    try:
        app = get_app()

        # Result future
        result_future = asyncio.Future()

        def yes_handler():
            if not result_future.done():
                result_future.set_result(True)

        def no_handler():
            if not result_future.done():
                result_future.set_result(False)

        # Create dialog
        dialog = Dialog(
            title=title,
            body=HSplit([Label(message)]),
            buttons=[
                Button("Yes", handler=yes_handler),
                Button("No", handler=no_handler)
            ],
            width=60,
            modal=True
        )

        # Show dialog
        previous_layout = app.layout
        float_container = FloatContainer(
            content=previous_layout.container,
            floats=[
                Float(content=dialog)
            ]
        )

        app.layout = Layout(float_container)
        app.layout.focus(dialog)

        # Wait for result
        result = await result_future

        # Restore layout
        app.layout = previous_layout

        return result

    except Exception as e:
        logger.error(f"Failed to show confirmation dialog: {e}")
        return False
