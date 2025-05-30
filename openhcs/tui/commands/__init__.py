"""
Commands for the TUI.

This package contains Command implementations for the TUI.
"""

from openhcs.tui.commands.registry import CommandRegistry, command_registry
from openhcs.tui.commands.dialog_commands import ShowGlobalSettingsDialogCommand, ShowHelpCommand
from openhcs.tui.commands.plate_commands import ShowAddPlateDialogCommand, ShowEditPlateDialogCommand, DeletePlateCommand
from openhcs.tui.commands.pipeline_commands import (
    DeleteSelectedPlatesCommand, ShowEditPlateConfigDialogCommand,
    InitializePlatesCommand, CompilePlatesCommand, RunPlatesCommand
)
from openhcs.tui.commands.pipeline_step_commands import (
    AddStepCommand, RemoveStepCommand, ValidatePipelineCommand
)
# pipeline_commands.py contains the COMPREHENSIVE implementations with direct orchestrator integration

__all__ = [
    'CommandRegistry',
    'command_registry',
    'ShowGlobalSettingsDialogCommand',
    'ShowHelpCommand',
    'ShowAddPlateDialogCommand',
    'ShowEditPlateDialogCommand',
    'DeletePlateCommand',
    'DeleteSelectedPlatesCommand',
    'ShowEditPlateConfigDialogCommand',
    'InitializePlatesCommand',
    'CompilePlatesCommand',
    'RunPlatesCommand',
    'AddStepCommand',
    'RemoveStepCommand',
    'ValidatePipelineCommand'
]
