"""
Plate-related commands for the TUI.

This module contains commands for plate management in the TUI.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from prompt_toolkit.application import get_app
from prompt_toolkit.shortcuts import message_dialog

if TYPE_CHECKING:
    from openhcs.tui.tui_architecture import TUIState
    from openhcs.core.context.processing_context import ProcessingContext
    from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator

logger = logging.getLogger(__name__)

class ShowAddPlateDialogCommand:
    """
    Command to show the Add Plate dialog.
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
        Execute the command to show the Add Plate dialog.

        Args:
            state: The TUIState instance
            context: The ProcessingContext instance
            **kwargs: Additional arguments
        """
        # Use provided state/context or the ones from initialization
        state = state or self.state
        context = context or self.context

        # Import here to avoid circular imports
        from openhcs.tui.dialogs.plate_dialog_manager import PlateDialogManager

        # Get the plate dialog manager
        plate_dialog_manager = PlateDialogManager.get_instance(state)
        
        # Show the add plate dialog
        await plate_dialog_manager.show_add_plate_dialog()

    def can_execute(self, state: "TUIState") -> bool:
        """
        Determine if the command can be executed.

        Args:
            state: The TUIState instance

        Returns:
            True if the command can be executed, False otherwise
        """
        return True


class ShowEditPlateDialogCommand:
    """
    Command to show the Edit Plate dialog.
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
        Execute the command to show the Edit Plate dialog.

        Args:
            state: The TUIState instance
            context: The ProcessingContext instance
            **kwargs: Additional arguments
        """
        # Use provided state/context or the ones from initialization
        state = state or self.state
        context = context or self.context

        # Import here to avoid circular imports
        from openhcs.tui.dialogs.plate_dialog_manager import PlateDialogManager

        # Get the plate dialog manager
        plate_dialog_manager = PlateDialogManager.get_instance(state)
        
        # Show the edit plate dialog for the selected plate
        if state.selected_plate:
            await plate_dialog_manager.show_edit_plate_dialog(state.selected_plate)
        else:
            await message_dialog(
                title="Error",
                text="No plate selected to edit."
            ).run_async()

    def can_execute(self, state: "TUIState") -> bool:
        """
        Determine if the command can be executed.

        Args:
            state: The TUIState instance

        Returns:
            True if the command can be executed, False otherwise
        """
        return state.selected_plate is not None


class DeletePlateCommand:
    """
    Command to delete a plate.
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
        Execute the command to delete a plate.

        Args:
            state: The TUIState instance
            context: The ProcessingContext instance
            **kwargs: Additional arguments
        """
        # Use provided state/context or the ones from initialization
        state = state or self.state
        context = context or self.context

        # Import here to avoid circular imports
        from openhcs.tui.dialogs.plate_dialog_manager import PlateDialogManager
        from openhcs.tui.dialogs.confirm_dialog import confirm_dialog

        # Get the plate dialog manager
        plate_dialog_manager = PlateDialogManager.get_instance(state)
        
        # Show confirmation dialog
        if state.selected_plate:
            plate_name = state.selected_plate.get('name', 'Unknown')
            result = await confirm_dialog(
                title="Confirm Delete",
                text=f"Are you sure you want to delete plate '{plate_name}'?",
                app_state=state
            )
            
            if result:
                # Delete the plate
                await plate_dialog_manager.delete_plate(state.selected_plate)
        else:
            await message_dialog(
                title="Error",
                text="No plate selected to delete."
            ).run_async()

    def can_execute(self, state: "TUIState") -> bool:
        """
        Determine if the command can be executed.

        Args:
            state: The TUIState instance

        Returns:
            True if the command can be executed, False otherwise
        """
        return state.selected_plate is not None
