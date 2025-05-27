"""
Pipeline Actions Toolbar Component for OpenHCS TUI.

This module defines the PipelineActionsToolbar class, responsible for
displaying and handling action buttons related to pipeline step management.
"""
import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING, Coroutine

from prompt_toolkit.application import get_app
from prompt_toolkit.layout import VSplit, Window, Container
from prompt_toolkit.widgets import Button # Using prompt_toolkit Button or Button

# Using standard Button widgets

if TYPE_CHECKING:
    from openhcs.tui.tui_architecture import TUIState
    from openhcs.tui.interfaces import CoreApplicationAdapterInterface, CoreOrchestratorAdapterInterface
    from openhcs.tui.commands import Command # Base command for type hinting

logger = logging.getLogger(__name__)

class PipelineActionsToolbar:
    """
    Displays and handles action buttons for the pipeline editor.
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

        # Button instances
        self.add_button: Optional[Button] = None
        self.remove_button: Optional[Button] = None
        self.edit_button: Optional[Button] = None
        self.load_button: Optional[Button] = None
        self.save_button: Optional[Button] = None

        self._create_buttons()
        self.container = self._build_container()

    def _create_buttons(self):
        """Creates the action buttons."""
        from openhcs.tui.commands import ( # Local import
            AddStepCommand, DeleteSelectedStepsCommand, LoadPipelineCommand,
            SavePipelineCommand, ShowEditStepDialogCommand
        )

        async def dispatch_command_with_plate_adapter(command_instance: 'Command', **kwargs):
            plate_adapter = await self.get_current_plate_adapter()
            # Commands should check if plate_adapter is None if they require it.
            # Their can_execute methods should prevent UI elements from being active if prereqs not met.
            if not plate_adapter and getattr(command_instance, "requires_plate_adapter", True): # Default to requiring if attr missing
                logger.warning(f"Command {command_instance.__class__.__name__} requires an active plate, but none found.")
                # Optionally show a user message via TUIState notification
                await self.ui_state.notify("error", {"message": "This action requires an active plate."})
                return

            await command_instance.execute(
                app_adapter=self.app_adapter,
                plate_adapter=plate_adapter,
                ui_state=self.ui_state,
                **kwargs
            )

        self.add_button = Button(
            "Add",
            handler=lambda: get_app().create_background_task(
                dispatch_command_with_plate_adapter(AddStepCommand())
            ),
            width=6
        )
        self.remove_button = Button(
            "Del",
            handler=lambda: get_app().create_background_task(
                # DeleteSelectedStepsCommand might get selected step from ui_state.selected_step
                dispatch_command_with_plate_adapter(DeleteSelectedStepsCommand())
            ),
            width=6
        )
        self.edit_button = Button(
            "Edit",
            handler=lambda: get_app().create_background_task(
                dispatch_command_with_plate_adapter(ShowEditStepDialogCommand()) # Relies on ui_state.selected_step
            ),
            width=6
        )
        self.load_button = Button(
            "Load",
            handler=lambda: get_app().create_background_task(
                dispatch_command_with_plate_adapter(LoadPipelineCommand())
            ),
            width=6
        )
        self.save_button = Button(
            "Save",
            handler=lambda: get_app().create_background_task(
                dispatch_command_with_plate_adapter(SavePipelineCommand())
            ),
            width=6
        )

    def _build_container(self) -> VSplit:
        """Builds the VSplit container for the toolbar buttons."""
        buttons = [
            self.add_button, Window(width=1, char=' '),
            self.remove_button, Window(width=1, char=' '),
            self.edit_button, Window(width=1, char=' '),
            self.load_button, Window(width=1, char=' '),
            self.save_button,
        ]
        valid_buttons = [b for b in buttons if b is not None]
        return VSplit(valid_buttons, height=1, padding=0)

    def update_button_states(self):
        """
        Updates the enabled/disabled state of buttons based on TUIState.
        Relies on the `can_execute` method of the commands.
        """
        from openhcs.tui.commands import ( # Re-import for can_execute checks
            AddStepCommand, DeleteSelectedStepsCommand, LoadPipelineCommand,
            SavePipelineCommand, ShowEditStepDialogCommand
        )
        # Note: Commands are instantiated here just to call can_execute.
        # This is acceptable if commands are lightweight or if can_execute is a static/classmethod.
        # If commands have significant construction cost, consider alternative for state checking.

        active_plate_exists = self.ui_state.active_plate_id is not None
        step_selected = self.ui_state.selected_step is not None

        if self.add_button:
            self.add_button.disabled = not AddStepCommand().can_execute(self.ui_state)
        if self.remove_button:
            # DeleteSelectedStepsCommand might depend on a step being selected
            cmd = DeleteSelectedStepsCommand()
            # Pass necessary context to can_execute if it needs more than just ui_state
            # For now, assume can_execute(ui_state) is sufficient and checks ui_state.selected_step
            self.remove_button.disabled = not cmd.can_execute(self.ui_state)
        if self.edit_button:
            self.edit_button.disabled = not ShowEditStepDialogCommand().can_execute(self.ui_state)
        if self.load_button:
            self.load_button.disabled = not LoadPipelineCommand().can_execute(self.ui_state)
        if self.save_button:
            self.save_button.disabled = not SavePipelineCommand().can_execute(self.ui_state)

        if get_app().is_running:
            get_app().invalidate()

    def __pt_container__(self) -> Container:
        return self.container

