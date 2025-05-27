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
    prompt_for_path_dialog,
    ok_handler,
    cancel_dialog,
    focus_text_area
)

from openhcs.tui.utils.safe_formatting import (
    safe_format,
    safe_text,
    SafeLabel,
    safe_error_label,
    safe_info_label,
    safe_status_label,
    error_building_component,
    unsupported_type_label,
    field_label
)

__all__ = [
    'handle_async_errors',
    'handle_async_errors_decorator',
    'show_error_dialog',
    'prompt_for_path_dialog',
    'ok_handler',
    'cancel_dialog',
    'focus_text_area',
    'safe_format',
    'safe_text',
    'SafeLabel',
    'safe_error_label',
    'safe_info_label',
    'safe_status_label',
    'error_building_component',
    'unsupported_type_label',
    'field_label'
]
