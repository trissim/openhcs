"""
Dialog manager for the TUI.

This module provides a manager for dialogs in the TUI.
"""

import logging
from typing import Dict, Optional, Any, TypeVar, Generic

from openhcs.tui.dialogs.base import BaseDialog

logger = logging.getLogger(__name__)

T = TypeVar('T')

class DialogManager:
    """
    Manager for dialogs.

    This class provides a centralized way to create and show dialogs.
    """

    def __init__(self, state: Any):
        """
        Initialize the dialog manager.

        Args:
            state: The application state
        """
        self.state = state
        self._dialogs: Dict[str, BaseDialog] = {}

    def register(self, dialog_id: str, dialog: BaseDialog) -> None:
        """
        Register a dialog.

        Args:
            dialog_id: The ID to register the dialog under
            dialog: The dialog to register
        """
        if dialog_id in self._dialogs:
            logger.warning(f"Dialog {dialog_id} already registered, overwriting")
            # Clean up the old dialog
            self._dialogs[dialog_id].cleanup()
        self._dialogs[dialog_id] = dialog

    def get(self, dialog_id: str) -> Optional[BaseDialog]:
        """
        Get a dialog.

        Args:
            dialog_id: The ID of the dialog to get

        Returns:
            The dialog, or None if not found
        """
        if dialog_id in self._dialogs:
            return self._dialogs[dialog_id]
        else:
            logger.warning(f"Dialog {dialog_id} not found")
            return None

    async def show(self, dialog_id: str) -> Any:
        """
        Show a dialog.

        Args:
            dialog_id: The ID of the dialog to show

        Returns:
            The result of the dialog, or None if cancelled or not found
        """
        dialog = self.get(dialog_id)
        if dialog is None:
            return None

        try:
            return await dialog.show()
        except Exception as e:
            logger.error(f"Error showing dialog {dialog_id}: {e}", exc_info=True)
            return None

    def cleanup(self) -> None:
        """
        Clean up all dialogs.

        This method should be called when the application is shutting down.
        """
        for dialog in self._dialogs.values():
            dialog.cleanup()
        self._dialogs.clear()

# Create a global dialog manager instance
dialog_manager = None

def initialize_dialog_manager(state: Any) -> None:
    """
    Initialize the dialog manager.

    Args:
        state: The application state
    """
    global dialog_manager
    dialog_manager = DialogManager(state)
