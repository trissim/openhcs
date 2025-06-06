"""
Views for the OpenHCS TUI.

This module provides view components that coordinate between services
and UI components with clean separation of concerns.
"""

from openhcs.tui.views.function_pattern_view import FunctionPatternView

__all__ = [
    'FunctionPatternView'
]
