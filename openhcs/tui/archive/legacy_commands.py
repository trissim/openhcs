"""
Legacy TUI commands that have been superseded by new functionality.

These commands were removed from the active codebase but preserved here for reference.
They were marked as "deleted" in commands/__init__.py but still existed in pipeline_commands.py.

Commands in this file:
- DeleteSelectedPlatesCommand: Legacy plate deletion command
- ShowEditPlateConfigDialogCommand: Legacy plate config dialog command
"""

from typing import Any, TYPE_CHECKING
import logging
from prompt_toolkit.shortcuts import message_dialog

if TYPE_CHECKING:
    from openhcs.tui.state import TUIState
    from openhcs.core.processing_context import ProcessingContext

logger = logging.getLogger(__name__)


class DeleteSelectedPlatesCommand:
    """
    Command to delete selected plates.
    
    LEGACY: This command was superseded by new plate management functionality.
    """
    def __init__(self, state: "TUIState" = None, context: "ProcessingContext" = None):
        """
        Initialize the command.

        Args:
            state: The TUIState instance (optional, can be provided during execute)
            context: The ProcessingContext instance (optional, can be provided during execute)
        """
        self.state = state
        self.context = context

    async def execute(self, state: "TUIState" = None, context: "ProcessingContext" = None, **kwargs: Any) -> None:
        """
        Execute the command to delete selected plates.

        Args:
            state: The TUIState instance
            context: The ProcessingContext instance
            **kwargs: Additional arguments
                selected_plates_data: List of plate data dictionaries to delete
        """
        # Use provided state/context or the ones from initialization
        state = state or self.state
        context = context or self.context
        
        # Get the selected plates data from kwargs
        selected_plates_data = kwargs.get('selected_plates_data', [])
        
        if not selected_plates_data:
            await message_dialog(
                title="Error",
                text="No plates selected to delete."
            ).run_async()
            return
            
        # Notify the plate manager to delete the selected plates
        await state.notify('delete_plates_requested', {'plates_to_delete': selected_plates_data})

    def can_execute(self, state: "TUIState") -> bool:
        """
        Determine if the command can be executed.

        Args:
            state: The TUIState instance

        Returns:
            True if the command can be executed, False otherwise
        """
        return state.selected_plate is not None


class ShowEditPlateConfigDialogCommand:
    """
    Command to show the Edit Plate Config dialog.
    
    LEGACY: This command was superseded by new configuration management functionality.
    """
    def __init__(self, state: "TUIState" = None, context: "ProcessingContext" = None):
        """
        Initialize the command.

        Args:
            state: The TUIState instance (optional, can be provided during execute)
            context: The ProcessingContext instance (optional, can be provided during execute)
        """
        self.state = state
        self.context = context

    async def execute(self, state: "TUIState" = None, context: "ProcessingContext" = None, **kwargs: Any) -> None:
        """
        Execute the command to show the Edit Plate Config dialog.

        Args:
            state: The TUIState instance
            context: The ProcessingContext instance
            **kwargs: Additional arguments
        """
        # Use provided state/context or the ones from initialization
        state = state or self.state
        context = context or self.context
        
        # Check if there's an active orchestrator
        if not state.active_orchestrator:
            await message_dialog(
                title="Error",
                text="No active plate selected to edit configuration."
            ).run_async()
            return
            
        # Notify to show the edit plate config dialog
        await state.notify('show_edit_plate_config_requested', {'orchestrator': state.active_orchestrator})

    def can_execute(self, state: "TUIState") -> bool:
        """
        Determine if the command can be executed.

        Args:
            state: The TUIState instance

        Returns:
            True if the command can be executed, False otherwise
        """
        return state.active_orchestrator is not None
