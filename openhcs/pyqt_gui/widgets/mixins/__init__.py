"""
PyQt6 Widget Utilities

Shared utility functions for PyQt6 widgets, mirroring patterns from the Textual TUI.
"""

from openhcs.pyqt_gui.widgets.mixins.selection_preservation_mixin import (
    preserve_selection_during_update,
    restore_selection_by_id,
    handle_selection_change_with_prevention
)

__all__ = [
    "preserve_selection_during_update",
    "restore_selection_by_id",
    "handle_selection_change_with_prevention",
]
