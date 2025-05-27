"""
Plate Actions Toolbar Component for OpenHCS TUI.

This module defines the PlateActionsToolbar class, responsible for
displaying and handling action buttons related to plate management.
"""
import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from prompt_toolkit.application import get_app
from prompt_toolkit.layout import VSplit, Window
from prompt_toolkit.widgets import Button # Using prompt_toolkit Button directly or Button

# Using standard Button widgets for now

if TYPE_CHECKING:
    from openhcs.tui.tui_architecture import TUIState
    from openhcs.tui.interfaces import CoreApplicationAdapterInterface, CoreOrchestratorAdapterInterface
    from openhcs.tui.commands import Command # Base command for type hinting

logger = logging.getLogger(__name__)

class PlateActionsToolbar:
    """
    Displays and handles action buttons for the plate list.
    """
    def __init__(self,
                 ui_state: 'TUIState',
                 app_adapter: 'CoreApplicationAdapterInterface',
                 # Callback to get the current plate_adapter for the active plate
                 get_current_plate_adapter: Callable[[], Coroutine[Any, Any, Optional['CoreOrchestratorAdapterInterface']]]
                ):
        self.ui_state = ui_state
        self.app_adapter = app_adapter
        self.get_current_plate_adapter = get_current_plate_adapter

        # Button instances - to be created in _create_buttons
        self.add_button: Optional[Button] = None
        self.remove_button: Optional[Button] = None
        self.edit_config_button: Optional[Button] = None # Renamed from 'edit_button' for clarity
        self.init_button: Optional[Button] = None
        self.compile_button: Optional[Button] = None
        self.run_button: Optional[Button] = None

        self._create_buttons()
        self.container = self._build_container()

    def _create_buttons(self):
        """Creates the action buttons."""
        # Dynamically import commands to avoid circular dependencies at module level
        from openhcs.tui.commands import (
            ShowAddPlateDialogCommand, DeleteSelectedPlatesCommand,
            ShowEditPlateConfigDialogCommand, InitializePlatesCommand,
            CompilePlatesCommand, RunPlatesCommand
        )

        async def dispatch_command_no_plate_adapter(command_instance: 'Command', **kwargs):
            # For commands that don't need a plate_adapter (e.g., ShowAddPlateDialog)
            await command_instance.execute(
                app_adapter=self.app_adapter,
                plate_adapter=None,
                ui_state=self.ui_state,
                **kwargs
            )

        async def dispatch_command_with_plate_adapter(command_instance: 'Command', **kwargs):
            plate_adapter = await self.get_current_plate_adapter()
            # Commands should ideally check if plate_adapter is None if they require it.
            # Or can_execute should prevent the button from being clickable.
            await command_instance.execute(
                app_adapter=self.app_adapter,
                plate_adapter=plate_adapter,
                ui_state=self.ui_state,
                **kwargs
            )

        # Add Button - Does not require an active plate
        self.add_button = Button(
            "Add",
            handler=lambda: get_app().create_background_task(
                dispatch_command_no_plate_adapter(ShowAddPlateDialogCommand())
                # If PlateDialogManager is needed by command, it should be passed via kwargs
                # For ShowAddPlateDialogCommand, it might trigger a TUIState event which PlateManagerController handles.
            ),
            width=6
        )

        # Remove Button - Operates on selected plates (app_adapter)
        self.remove_button = Button(
            "Del",
            handler=lambda: get_app().create_background_task(
                dispatch_command_no_plate_adapter(DeleteSelectedPlatesCommand()) # Delete uses app_adapter
            ),
            width=6,
            # Enablement based on selection (checked in command's can_execute)
        )

        # Edit Config Button - Requires an active plate
        self.edit_config_button = Button(
            "Edit Cfg", # Changed label for clarity
            handler=lambda: get_app().create_background_task(
                dispatch_command_with_plate_adapter(ShowEditPlateConfigDialogCommand())
            ),
            width=9,
        )

        # Initialize Button - Requires an active plate
        self.init_button = Button(
            "Init",
            handler=lambda: get_app().create_background_task(
                dispatch_command_with_plate_adapter(InitializePlatesCommand())
            ),
            width=6
        )

        # Compile Button - Requires an active plate
        self.compile_button = Button(
            "Compile",
            handler=lambda: get_app().create_background_task(
                dispatch_command_with_plate_adapter(CompilePlatesCommand())
            ),
            width=9
        )

        # Run Button - Requires an active plate
        self.run_button = Button(
            "Run",
            handler=lambda: get_app().create_background_task(
                dispatch_command_with_plate_adapter(RunPlatesCommand())
            ),
            width=5
        )

    def _build_container(self) -> VSplit:
        """Builds the VSplit container for the toolbar buttons."""
        buttons = [
            self.add_button, Window(width=1, char=' '),
            self.remove_button, Window(width=1, char=' '),
            self.edit_config_button, Window(width=1, char=' '),
            self.init_button, Window(width=1, char=' '),
            self.compile_button, Window(width=1, char=' '),
            self.run_button,
        ]
        # Filter out None if any button failed to initialize (should not happen with Button)
        valid_buttons = [b for b in buttons if b is not None]
        return VSplit(valid_buttons, height=1, padding=0) # Ensure fixed height for toolbar

    def update_button_states(self):
        """
        Updates the enabled/disabled state of buttons based on TUIState.
        This method would be called by the controller when relevant state changes.
        It relies on the `can_execute` method of the commands.
        """
        from openhcs.tui.commands import ( # Re-import for can_execute checks
            DeleteSelectedPlatesCommand, ShowEditPlateConfigDialogCommand,
            InitializePlatesCommand, CompilePlatesCommand, RunPlatesCommand
        )

        if self.remove_button:
            self.remove_button.disabled = not DeleteSelectedPlatesCommand().can_execute(self.ui_state)
        if self.edit_config_button:
            self.edit_config_button.disabled = not ShowEditPlateConfigDialogCommand().can_execute(self.ui_state)
        if self.init_button:
            self.init_button.disabled = not InitializePlatesCommand().can_execute(self.ui_state)
        if self.compile_button:
            self.compile_button.disabled = not CompilePlatesCommand().can_execute(self.ui_state)
        if self.run_button:
            self.run_button.disabled = not RunPlatesCommand().can_execute(self.ui_state)

        if get_app().is_running:
            get_app().invalidate()


    def __pt_container__(self) -> VSplit:
        return self.container

