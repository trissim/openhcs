"""
Dialogs for the TUI.

This package contains dialog implementations for the TUI.
"""

from openhcs.tui.dialogs.base import BaseDialog
from openhcs.tui.dialogs.manager import DialogManager, dialog_manager, initialize_dialog_manager

__all__ = [
    'BaseDialog',
    'DialogManager',
    'dialog_manager',
    'initialize_dialog_manager'
]
