"""
Dialog helper utilities for the TUI.

This module provides utilities for creating and showing dialogs.
"""

import logging
import asyncio
import traceback
import re
from pathlib import Path
from typing import Any, Optional, Callable, List

from prompt_toolkit.application import get_app
from prompt_toolkit.layout.containers import HSplit
from prompt_toolkit.layout import Window, Dimension, ScrollablePane
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.widgets import Button, Dialog, Label, TextArea
from prompt_toolkit.key_binding import merge_key_bindings

logger = logging.getLogger(__name__)

# SafeButton eliminated - use Button directly

def _syntax_highlight_traceback(traceback_text: str) -> List[tuple]:
    """
    Apply syntax highlighting to traceback text using Pygments.

    Args:
        traceback_text: The raw traceback text

    Returns:
        List of (style, text) tuples for FormattedTextControl
    """
    # Import Pygments components
    from pygments.lexers import get_lexer_by_name

    # Use Python traceback lexer for syntax highlighting
    lexer = get_lexer_by_name('pytb')  # Python traceback lexer

    # Get tokens from Pygments
    tokens = list(lexer.get_tokens(traceback_text))

    # Convert Pygments tokens to prompt_toolkit format
    formatted_tokens = []
    for token_type, text in tokens:
        # Map Pygments token types to simple ANSI colors (fallback if styles not defined)
        token_str = str(token_type)
        if 'Error' in token_str or 'Exception' in token_str:
            style = 'ansired bold'  # Red and bold for exceptions
        elif 'Generic.Traceback' in token_str:
            style = 'ansibrightblue'  # Bright blue for traceback headers
        elif 'Name.Builtin' in token_str or 'Keyword' in token_str:
            style = 'ansimagenta'  # Magenta for keywords
        elif 'String' in token_str:
            style = 'ansigreen'  # Green for strings
        elif 'Number' in token_str:
            style = 'ansicyan'  # Cyan for numbers
        elif 'Comment' in token_str:
            style = 'ansibrightblack'  # Gray for comments
        elif 'Name.Function' in token_str:
            style = 'ansiyellow'  # Yellow for functions
        elif 'Literal.String.Doc' in token_str:
            style = 'ansibrightgreen'  # Bright green for file paths
        else:
            style = ''  # Default color

        formatted_tokens.append((style, text))

    return formatted_tokens



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
        # Restore focus before setting result
        if hasattr(ok_handler, 'previous_focus') and ok_handler.previous_focus:
            try:
                from prompt_toolkit.application import get_app
                get_app().layout.focus(ok_handler.previous_focus)
            except Exception as e:
                pass  # Silently handle focus restoration errors
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

async def show_scrollable_error_dialog(
    title: str = "Error",
    message: str = "An error occurred",
    exception: Exception | None = None,
    app_state: object | None = None,
):
    """
    Syntax-highlighted, scrollable error dialog that mirrors FileManagerBrowser
    behaviour: wheel scroll anywhere in the pane, single-click OK to close.
    """
    # ───────────────────────── imports ───────────────────────────────────────
    import asyncio, traceback, logging
    from prompt_toolkit.application import get_app
    from prompt_toolkit.layout import Dimension, HSplit, Window
    from prompt_toolkit.layout.containers import ScrollablePane
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.widgets import Button, Dialog, Label
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.mouse_events import MouseEventType
    from prompt_toolkit.data_structures import Point

    log = logging.getLogger(__name__)
    app = get_app()
    done: asyncio.Future[bool] = asyncio.Future()

    # ─────────────────── prepare highlighted fragments ──────────────────────
    frags: list[tuple[str, str]] = [("", f"{message}\n\n")]

    if exception:
        tb_text = "".join(
            traceback.format_exception(
                type(exception), exception, exception.__traceback__
            )
        )
        frags += _syntax_highlight_traceback(tb_text)

    total_lines = sum(txt.count("\n") for _, txt in frags) + 1
    cursor_y = 0  # virtual cursor row that keeps the current view centred

    # ───────────────────────── key bindings ─────────────────────────────────
    kb = KeyBindings()

    def _scroll(dy: int) -> None:
        nonlocal cursor_y
        cursor_y = max(0, min(total_lines - 1, cursor_y + dy))
        ft_control.get_cursor_position = lambda: Point(x=0, y=cursor_y)
        app.invalidate()

    for k, dy in [("up", -1), ("k", -1), ("down", 1), ("j", 1)]:
        kb.add(k)(lambda e, d=dy: _scroll(d))
    kb.add("pageup")(lambda e: _scroll(-10))
    kb.add("pagedown")(lambda e: _scroll(10))
    kb.add("home")(lambda e: _scroll(-10_000))
    kb.add("end")(lambda e: _scroll(10_000))

    # ───────────────────── mouse handler (at control level) ─────────────────
    def mouse_handler(mev):
        if mev.event_type is MouseEventType.SCROLL_UP:
            _scroll(-3)
            return True                       # we handled the wheel
        if mev.event_type is MouseEventType.SCROLL_DOWN:
            _scroll(3)
            return True
        return NotImplemented                 # propagate *all* clicks!

    # ────────────────────────── text control & pane ─────────────────────────
    ft_control = FormattedTextControl(
        text=frags,
        focusable=True,
        key_bindings=kb,
        mouse_handler=mouse_handler,
        show_cursor=False,
    )

    scrollable = ScrollablePane(
        Window(ft_control),
        height=Dimension(min=10, max=30),
        show_scrollbar=True,
        display_arrows=True,
    )

    # ────────────────────────────── dialog ----------------------------------
    previous_focus = app.layout.current_window

    def _close() -> None:
        if previous_focus is not None:
            try:
                app.layout.focus(previous_focus)
            except Exception:  # noqa: BLE001
                pass
        if not done.done():
            done.set_result(True)

    dlg = Dialog(
        title=title,
        body=HSplit(
            [
                Label(message, dont_extend_height=True),
                scrollable,
            ]
        ),
        buttons=[Button(text="OK", handler=_close, width=6)],
        width=100,
        modal=True,
    )

    # ─────────────────────────── show modal ─────────────────────────────────
    if hasattr(app_state, "show_dialog") and callable(app_state.show_dialog):
        await app_state.show_dialog(dlg, result_future=done)
    else:
        log.error("app_state missing .show_dialog; cannot open error dialog")
        done.set_result(False)

    return await done

async def prompt_for_file_dialog(title: str, prompt_message: str, app_state: Any, filemanager=None) -> Optional[str]:
    """
    Displays a modal dialog with FileManagerBrowser for selecting a single file.

    Args:
        title: The title of the dialog.
        prompt_message: The message to display above the file browser.
        app_state: The TUIState or equivalent, expected to have a `show_dialog` method.
        filemanager: Shared FileManager instance from context (required).

    Returns:
        File path if a file is selected, otherwise None.
    """
    from prompt_toolkit.application import get_app
    app = get_app()
    future = asyncio.Future()

    # Import FileManagerBrowser and required types
    from openhcs.tui.editors.file_browser import FileManagerBrowser, SelectionMode
    from openhcs.io.base import Backend

    # Validate required filemanager parameter
    if not filemanager:
        logger.error("prompt_for_file_dialog: filemanager parameter is required")
        return None

    async def on_file_selected(paths: list):
        """Handle file selection from FileManagerBrowser."""
        if paths:
            # Single file selection - take the first path
            future.set_result(str(paths[0]))

    async def on_browser_cancel():
        """Handle cancel from FileManagerBrowser."""
        future.set_result(None)

    # Create FileManagerBrowser for file selection
    file_browser = FileManagerBrowser(
        file_manager=filemanager,
        backend=Backend.DISK,  # Explicit backend in correct position
        on_path_selected=on_file_selected,
        on_cancel=on_browser_cancel,
        initial_path=Path.home(),  # Start from user home directory
        selection_mode=SelectionMode.FILES_AND_DIRECTORIES,  # Allow files and directories
        allow_multiple=False,  # Single file selection
        show_hidden_files=False
    )

    # Create dialog body
    dialog_body = HSplit([
        Label(prompt_message),
        Label(""),
        file_browser,
    ])

    # Initialize the file browser by loading initial directory
    file_browser.start_load()

    # Create cancel handler
    cancel_handler = cancel_dialog(future)

    # If app_state has a show_dialog method, use it
    if hasattr(app_state, 'show_dialog') and callable(getattr(app_state, 'show_dialog')):
        # Capture current focus before showing dialog
        previous_focus = None
        try:
            previous_focus = get_app().layout.current_window
            logger.info(f"💾 Captured previous focus: {previous_focus}")
        except Exception as e:
            logger.warning(f"Could not capture previous focus: {e}")

        # Store previous focus on the handler for restoration
        cancel_handler.previous_focus = previous_focus

        # Create dialog with simple buttons for single file selection
        dialog = Dialog(
            title=title,
            body=dialog_body,
            buttons=[
                Button("Cancel", handler=cancel_handler, width=len("Cancel") + 2),
            ],
            width=100,  # Standard width
            modal=True
        )

        # Show the dialog
        await app_state.show_dialog(dialog, result_future=future)
    else:
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
    from prompt_toolkit.application import get_app
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
        # Restore focus before setting result
        if hasattr(accept_selection, 'previous_focus') and accept_selection.previous_focus:
            try:
                from prompt_toolkit.application import get_app
                get_app().layout.focus(accept_selection.previous_focus)
                logger.info(f"🔄 Restored focus to: {accept_selection.previous_focus}")
            except Exception as e:
                logger.warning(f"Could not restore previous focus: {e}")

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

    # Create dialog body with explicit padding using Box
    from prompt_toolkit.widgets import Box

    dialog_content = HSplit([
        Label(prompt_message),
        Label(""),
        Label("Browse and select folders (you can select multiple folders one at a time):"),
        file_browser,
        Label(""),
        Label("Selected folders:"),
        selected_display,
    ])

    # Wrap entire content in Box with padding
    dialog_body = Box(
        dialog_content,
        padding_left=2,
        padding_right=2
    )

    # Initialize the file browser by loading initial directory
    file_browser.start_load()

    # If app_state has a show_dialog method, use it
    if hasattr(app_state, 'show_dialog') and callable(getattr(app_state, 'show_dialog')):
        # Capture current focus before showing dialog
        previous_focus = None
        try:
            previous_focus = get_app().layout.current_window
            logger.info(f"💾 Captured previous focus: {previous_focus}")
        except Exception as e:
            logger.warning(f"Could not capture previous focus: {e}")

        # Store previous focus on the handlers for restoration
        accept_selection.previous_focus = previous_focus
        cancel_handler = cancel_dialog(future)
        cancel_handler.previous_focus = previous_focus

        # Create dialog with properly configured handlers
        dialog = Dialog(
            title=title,
            body=dialog_body,
            buttons=[
                Button("Add More", handler=lambda: None, width=len("Add More") + 2),  # Browser handles this
                Button("Clear All", handler=clear_selection, width=len("Clear All") + 2),
                Button("OK", handler=accept_selection, width=len("OK") + 2),
                Button("Cancel", handler=cancel_handler, width=len("Cancel") + 2),
            ],
            width=140,  # Wider for file browser
            modal=True
        )

        # Create a task to set focus after dialog is shown
        async def set_focus_after_delay():
            await asyncio.sleep(0.1)  # Small delay to ensure dialog is rendered
            try:
                get_app().layout.focus(file_browser.item_list_control)
                logger.info("🎯 Focus set to file browser")
            except Exception as e:
                logger.warning(f"Could not set focus to file browser: {e}")

        # Start the focus task
        focus_task = asyncio.create_task(set_focus_after_delay())

        # Show the dialog
        await app_state.show_dialog(dialog, result_future=future)

        # Cancel focus task if still running
        if not focus_task.done():
            focus_task.cancel()
    else:
        # Fallback if app_state doesn't have a show_dialog
        # Create a basic dialog without focus management
        dialog = Dialog(
            title=title,
            body=dialog_body,
            buttons=[
                Button("Add More", handler=lambda: None, width=len("Add More") + 2),
                Button("Clear All", handler=clear_selection, width=len("Clear All") + 2),
                Button("OK", handler=accept_selection, width=len("OK") + 2),
                Button("Cancel", handler=cancel_dialog(future), width=len("Cancel") + 2),
            ],
            width=140,
            modal=True
        )
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
        # Restore focus before setting result
        if hasattr(handler, 'previous_focus') and handler.previous_focus:
            try:
                from prompt_toolkit.application import get_app
                get_app().layout.focus(handler.previous_focus)
                logger.info(f"🔄 Restored focus to: {handler.previous_focus}")
            except Exception as e:
                logger.warning(f"Could not restore previous focus: {e}")

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

# Removed unused catch_and_show_error decorator to eliminate potential unawaited coroutines

async def show_global_error(exception: Exception, context: str = "operation", app_state: Optional[Any] = None):
    """
    Global error handler - shows any error in a scrollable dialog with syntax highlighting.
    Takes focus immediately and restores it when dismissed.

    NOTE: This function is *async*. You must either:
        • `await show_global_error(...)` from async code, or
        • schedule it with `get_app().create_background_task(...)`
          from synchronous code, or
        • use `show_global_error_sync(...)` from synchronous code.

    Args:
        exception: The exception to display
        context: Context description for the error
        app_state: The TUIState for dialog management
    """
    logger.error(f"Global error in {context}: {exception}", exc_info=True)

    # Show the error dialog - it handles focus management internally
    await show_scrollable_error_dialog(
        title=f"Error in {context}",
        message=f"An error occurred during {context}",
        exception=exception,
        app_state=app_state
    )

def show_global_error_sync(exception: Exception, context: str = "operation", app_state: Optional[Any] = None) -> None:
    """
    Schedule the global-error dialog on the running PTK event-loop and return
    immediately. Use this from synchronous code.

    Args:
        exception: The exception to display
        context: Context description for the error
        app_state: The TUIState for dialog management
    """
    from prompt_toolkit.application import get_app
    get_app().create_background_task(show_global_error(exception, context, app_state))

def setup_global_exception_handler(app_state):
    """
    Setup COMPREHENSIVE global exception handler that catches ALL unhandled exceptions
    from any source and shows them in the error dialog. Call this once at app startup.

    Catches:
    - Main thread exceptions
    - Thread exceptions
    - Async task exceptions
    - Event loop exceptions
    - Background coroutine exceptions
    - run_in_executor exceptions

    Args:
        app_state: The TUIState for dialog management
    """
    import sys
    import threading
    import asyncio

    def handle_exception(exc_type, exc_value, exc_traceback):
        """Handle any unhandled exception by showing error dialog."""
        if issubclass(exc_type, KeyboardInterrupt):
            # Don't catch Ctrl+C
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        # Create exception object with traceback
        exception = exc_value
        if exception.__traceback__ is None:
            exception.__traceback__ = exc_traceback

        # Show error dialog using sync wrapper
        try:
            show_global_error_sync(exception, "application", app_state)
        except Exception:
            # Fallback to default handler if we can't show dialog
            sys.__excepthook__(exc_type, exc_value, exc_traceback)

    def handle_thread_exception(args):
        """Handle exceptions in threads."""
        handle_exception(args.exc_type, args.exc_value, args.exc_traceback)

    def handle_asyncio_exception(loop, context):
        """Handle asyncio exceptions (background tasks, etc)."""
        exception = context.get('exception')
        if exception:
            if isinstance(exception, KeyboardInterrupt):
                return

            # Show error dialog for async exceptions using sync wrapper
            try:
                show_global_error_sync(exception, "async task", app_state)
            except Exception:
                # Fallback to logging if we can't show dialog
                logger.error(f"Asyncio exception: {exception}", exc_info=exception)

    # Set ALL exception handlers
    sys.excepthook = handle_exception
    threading.excepthook = handle_thread_exception

    # Set asyncio exception handler for current and future event loops
    try:
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(handle_asyncio_exception)
    except RuntimeError:
        # No running loop yet, set default policy
        pass

    # Note: Cannot set default exception handler for new event loops
    # Each loop must set its own exception handler when created

    # Create a global wrapper for run_in_executor that ensures exceptions are caught
    def create_safe_run_in_executor():
        """Create a wrapper for run_in_executor that ensures exceptions show in dialogs."""
        async def safe_run_in_executor(executor, func, *args):
            """Safe wrapper for run_in_executor that catches exceptions."""
            try:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(executor, func, *args)
            except Exception as e:
                # Manually trigger the asyncio exception handler
                context = {
                    'message': 'Exception in run_in_executor',
                    'exception': e,
                    'future': None
                }
                # Call the handler properly - it schedules its own background task
                try:
                    handle_asyncio_exception(loop, context)
                except Exception:
                    # If handler fails, just log the original exception
                    logger.error(f"Exception in run_in_executor: {e}", exc_info=e)
                raise  # Re-raise so the calling code can handle it too
        return safe_run_in_executor

    # Make the safe wrapper available globally
    import builtins
    builtins.safe_run_in_executor = create_safe_run_in_executor()
    logger.info("Created safe_run_in_executor wrapper for better exception handling")

    # Also patch prompt_toolkit Application to catch internal exceptions
    try:
        from prompt_toolkit.application import Application
        original_run_async = Application.run_async

        async def patched_run_async(self, *args, **kwargs):
            """Patched run_async that catches prompt_toolkit internal exceptions."""
            try:
                return await original_run_async(self, *args, **kwargs)
            except Exception as e:
                if not isinstance(e, KeyboardInterrupt):
                    # Show error dialog for prompt_toolkit exceptions using sync wrapper
                    try:
                        show_global_error_sync(e, "prompt_toolkit", app_state)
                    except Exception:
                        logger.error(f"Prompt_toolkit exception: {e}", exc_info=e)
                raise

        Application.run_async = patched_run_async
        logger.info("Patched prompt_toolkit Application.run_async for global error handling")
    except Exception as e:
        logger.warning(f"Failed to patch prompt_toolkit: {e}")

# Global error handler replaces handle_error_with_dialog - use show_global_error instead
