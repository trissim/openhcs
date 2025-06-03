"""
Commands for the TUI.

This package contains Command implementations for the TUI.
"""

import logging
from openhcs.tui.commands.registry import CommandRegistry, command_registry
from openhcs.tui.commands.base_command import BaseCommand
# Create Command alias for backward compatibility
Command = BaseCommand

# Removed imports from deleted files: dialog_commands, plate_commands
from openhcs.tui.commands.pipeline_commands import (
    InitializePlatesCommand, CompilePlatesCommand, RunPlatesCommand
)
from openhcs.tui.commands.pipeline_step_commands import (
    AddStepCommand, RemoveStepCommand, ValidatePipelineCommand
)
# pipeline_commands.py contains the COMPREHENSIVE implementations with direct orchestrator integration

# Create minimal stub commands for menu_bar.py compatibility
logger = logging.getLogger(__name__)

class ShowGlobalSettingsDialogCommand(BaseCommand):
    """Minimal stub for global settings dialog command."""
    async def execute(self, **kwargs):
        logger.info("Global Settings dialog not implemented yet")

class ShowHelpCommand(BaseCommand):
    """Minimal stub for help dialog command."""
    async def execute(self, **kwargs):
        logger.info("Help dialog not implemented yet")

__all__ = [
    'CommandRegistry',
    'command_registry',
    'BaseCommand',
    'Command',  # Alias for BaseCommand
    'ShowGlobalSettingsDialogCommand',  # Minimal stub
    'ShowHelpCommand',  # Minimal stub
    # Removed deleted commands: ShowAddPlateDialogCommand, ShowEditPlateDialogCommand, DeletePlateCommand,
    # DeleteSelectedPlatesCommand, ShowEditPlateConfigDialogCommand
    'InitializePlatesCommand',
    'CompilePlatesCommand',
    'RunPlatesCommand',
    'AddStepCommand',
    'RemoveStepCommand',
    'ValidatePipelineCommand'
]
