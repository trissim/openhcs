"""
Base dialog class for the TUI.

This module provides a base class for all dialogs in the TUI.
"""

import logging
from typing import TypeVar, Generic, Optional, Dict, Any, List, Callable, Awaitable, Union, Type

from prompt_toolkit.application import Application
from prompt_toolkit.layout import Layout
from prompt_toolkit.key_binding import KeyBindings

logger = logging.getLogger(__name__)

T = TypeVar('T')

class BaseDialog(Generic[T]):
    """
    Base class for all dialogs.

    This class provides a consistent interface for all dialogs, with standard
    methods for showing, handling results, and cleanup.
    """

    def __init__(self, title: str = "Dialog"):
        """
        Initialize the dialog.

        Args:
            title: The title of the dialog
        """
        self.title = title
        self._application: Optional[Application] = None
        self._result: Optional[T] = None
        self._is_showing = False

    def _create_application(self) -> Application:
        """
        Create the prompt_toolkit Application for the dialog.

        This method should be overridden by subclasses to create the specific
        Application for the dialog.

        Returns:
            The prompt_toolkit Application
        """
        raise NotImplementedError("Subclasses must implement _create_application")

    async def show(self) -> Optional[T]:
        """
        Show the dialog and wait for a result.

        Returns:
            The result of the dialog, or None if cancelled
        """
        if self._is_showing:
            logger.warning("Dialog is already showing")
            return None

        self._is_showing = True
        self._result = None

        try:
            # Create the application if it doesn't exist
            if self._application is None:
                self._application = self._create_application()

            # Run the application and wait for a result
            await self._application.run_async()

            # Return the result
            return self._result
        except Exception as e:
            logger.error(f"Error showing dialog: {e}", exc_info=True)
            return None
        finally:
            self._is_showing = False

    def set_result(self, result: Optional[T]) -> None:
        """
        Set the result of the dialog and exit.

        Args:
            result: The result to set
        """
        self._result = result

        # Exit the application if it exists
        if self._application is not None:
            self._application.exit()

    async def run_async(self) -> Optional[T]:
        """
        Alias for show() to match prompt_toolkit's interface.

        Returns:
            The result of the dialog, or None if cancelled
        """
        return await self.show()

    def cleanup(self) -> None:
        """
        Clean up resources used by the dialog.

        This method should be called when the dialog is no longer needed.
        """
        self._application = None
        self._result = None
        self._is_showing = False
