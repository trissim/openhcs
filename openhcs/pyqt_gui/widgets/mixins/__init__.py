"""
PyQt6 Widget Mixins

Shared functionality for PyQt6 widgets, mirroring patterns from the Textual TUI.
"""

from .selection_preservation_mixin import SelectionPreservationMixin

__all__ = [
    "SelectionPreservationMixin",
]
