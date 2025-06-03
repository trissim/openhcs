"""
Utilities for the TUI.

This package contains utility functions and classes for the TUI.
"""

from openhcs.tui.utils.error_handling import (
    handle_async_errors,
    handle_async_errors_decorator,
    show_error_dialog
)

from openhcs.tui.utils.dialog_helpers import (
    prompt_for_file_dialog,
    ok_handler,
    cancel_dialog,
    focus_text_area
)

__all__ = [
    'handle_async_errors',
    'handle_async_errors_decorator',
    'show_error_dialog',
    'prompt_for_file_dialog',
    'ok_handler',
    'cancel_dialog',
    'focus_text_area'
]
