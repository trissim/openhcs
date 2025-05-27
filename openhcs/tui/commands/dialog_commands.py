"""
Dialog commands for the TUI.

This module contains commands for showing dialogs in the TUI.
"""

import logging
from typing import Any, Optional, TYPE_CHECKING

from prompt_toolkit.application import get_app
from prompt_toolkit.shortcuts import message_dialog

if TYPE_CHECKING:
    from openhcs.tui.tui_architecture import TUIState
    from openhcs.core.context.processing_context import ProcessingContext
    from openhcs.core.config import GlobalPipelineConfig

# Import the dialog classes
try:
    from openhcs.tui.dialogs.global_settings_editor import GlobalSettingsEditorDialog
except ImportError:
    # This might happen if commands.py is imported by one of these modules before they are fully defined.
    GlobalSettingsEditorDialog = Any

logger = logging.getLogger(__name__)

class ShowGlobalSettingsDialogCommand:
    """
    Command to show the Global Settings Editor dialog.
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
        Execute the command to show the Global Settings Editor dialog.

        Args:
            state: The TUIState instance
            context: The ProcessingContext instance
            **kwargs: Additional arguments
        """
        # Use provided state/context or the ones from initialization
        state = state or self.state
        context = context or self.context
        if state.global_config is None:
            # This should ideally not happen if TUIState is properly initialized
            await message_dialog(
                title="Error",
                text="Global configuration is not available."
            ).run_async()
            return

        dialog = GlobalSettingsEditorDialog(
            initial_config=state.global_config,
            state=state # Pass state if dialog needs to notify on changes directly or access other state
        )
        # The dialog's show() method should be async and return the result
        # or handle notifications internally.
        # For now, assuming it updates state.global_config if changes are saved
        # and notifies 'global_config_needs_update'
        result_config: "GlobalPipelineConfig" | None = await dialog.show()

        if result_config and isinstance(result_config, GlobalPipelineConfig):
            # If dialog returns the new config, update state and notify
            # This logic aligns with V4 plan (Phase 4.1)
            # where the dialog's save handler notifies 'global_config_changed' and 'global_config_needs_update'.
            # If the dialog itself handles notifications, this command might just show it.
            # For now, let's assume the command is responsible for the notification if dialog returns data.
            # This part might be refined when GlobalSettingsEditorDialog is fully implemented with commands.

            # As per plan 4.1: "MenuBar._on_settings handler correctly instantiates and shows this dialog,
            # then passes the result to OpenHCSTUILauncher via self.state.notify('global_config_needs_update', ...)"
            # The command now replaces the _on_settings handler.

            # The dialog itself (its _save_settings) should notify 'global_config_changed'.
            # This command ensures 'global_config_needs_update' is notified for the launcher.
            await state.notify('global_config_needs_update', result_config)
            # The TUIState's global_config reference should also be updated by the dialog or its save command.
            # state.global_config = result_config # This should be handled by the dialog's save logic / command

    def can_execute(self, state: "TUIState") -> bool:
        """
        Determine if the command can be executed.

        Args:
            state: The TUIState instance

        Returns:
            True if the command can be executed, False otherwise
        """
        return state.global_config is not None


class ShowHelpCommand:
    """
    Command to show a simple help message/dialog.
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
        Execute the command to show the help dialog.

        Args:
            state: The TUIState instance
            context: The ProcessingContext instance
            **kwargs: Additional arguments
        """
        # Use provided state/context or the ones from initialization
        state = state or self.state
        context = context or self.context
        # For now, a simple message dialog. This can be expanded later.
        # The content of the help message should be defined elsewhere.
        help_text = (
            "OpenHCS TUI - Help\n\n"
            "Controls:\n"
            "- Use mouse or Tab/Shift-Tab to navigate.\n"
            "- Arrow keys for lists and menus.\n"
            "- Enter to activate buttons/menu items.\n"
            "- Ctrl-C to Exit.\n\n"
            "Workflow:\n"
            "1. Add Plate(s) using Plate Manager [add] button.\n"
            "2. Select a plate, then [init] it.\n"
            "3. Add steps to its pipeline using Pipeline Editor [add] button.\n"
            "4. [edit] steps to configure them.\n"
            "5. [compile] the plate's pipeline.\n"
            "6. [run] the compiled pipeline.\n\n"
            "Global Settings: Access via top menu to change default configurations."
        )
        await message_dialog(
            title="OpenHCS Help",
            text=help_text
        ).run_async()

    def can_execute(self, state: "TUIState") -> bool:
        """
        Determine if the command can be executed.

        Args:
            state: The TUIState instance

        Returns:
            True if the command can be executed, False otherwise
        """
        return True
