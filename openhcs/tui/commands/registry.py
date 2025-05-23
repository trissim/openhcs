"""
Command registry for the TUI.

This module provides a registry for Command objects, allowing them to be
registered, retrieved, and executed in a consistent way.
"""

import logging
from typing import Dict, Type, Optional, Any, TypeVar, Generic, List, Callable, Awaitable, Union

logger = logging.getLogger(__name__)

# Forward reference for Command type
if False:  # TYPE_CHECKING equivalent without circular imports
    from openhcs.tui.commands import Command

T = TypeVar('T')

class CommandRegistry:
    """
    Registry for Command objects.

    This class provides a central registry for Command objects, allowing them to be
    registered, retrieved, and executed in a consistent way.
    """

    def __init__(self):
        """Initialize the command registry."""
        self._commands: Dict[str, 'Command'] = {}
        self._command_types: Dict[str, Type['Command']] = {}

    def register(self, command_id: str, command: 'Command') -> None:
        """
        Register a Command instance.

        Args:
            command_id: The ID to register the command under
            command: The Command instance to register
        """
        if command_id in self._commands:
            logger.warning(f"Command {command_id} already registered, overwriting")
        self._commands[command_id] = command

    def register_type(self, command_id: str, command_type: Type['Command']) -> None:
        """
        Register a Command type.

        Args:
            command_id: The ID to register the command type under
            command_type: The Command type to register
        """
        if command_id in self._command_types:
            logger.warning(f"Command type {command_id} already registered, overwriting")
        self._command_types[command_id] = command_type

    def get(self, command_id: str) -> Optional['Command']:
        """
        Get a Command instance.

        Args:
            command_id: The ID of the command to get

        Returns:
            The Command instance, or None if not found
        """
        if command_id in self._commands:
            return self._commands[command_id]
        elif command_id in self._command_types:
            # Create a new instance of the command type
            command = self._command_types[command_id]()
            self._commands[command_id] = command
            return command
        else:
            logger.warning(f"Command {command_id} not found")
            return None

    def create_handler(self, command_id: str, state: Any) -> Optional[Callable[[], Awaitable[None]]]:
        """
        Create a handler function for a command.

        Args:
            command_id: The ID of the command to create a handler for
            state: The application state to pass to the command

        Returns:
            A handler function that executes the command, or None if the command is not found
        """
        command = self.get(command_id)
        if command is None:
            return None

        async def handler() -> None:
            try:
                await command.execute(state)
            except Exception as e:
                logger.error(f"Error executing command {command_id}: {e}", exc_info=True)
                # Optionally show an error dialog to the user
                if hasattr(state, 'notify'):
                    await state.notify('operation_status_changed', {
                        'operation': 'command_execution',
                        'status': 'error',
                        'message': f"Error executing {command_id}: {str(e)}",
                        'source': 'CommandRegistry'
                    })

        return handler

    async def execute(self, command_id: str, state: Any, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Execute a command.

        Args:
            command_id: The ID of the command to execute
            state: The application state to pass to the command
            context: Optional context information for the command

        Returns:
            None
        """
        command = self.get(command_id)
        if command is None:
            logger.warning(f"Command {command_id} not found, no-op")
            return

        try:
            await command.execute(state, context)
        except Exception as e:
            logger.error(f"Error executing command {command_id}: {e}", exc_info=True)
            # Optionally show an error dialog to the user
            if hasattr(state, 'notify'):
                await state.notify('operation_status_changed', {
                    'operation': 'command_execution',
                    'status': 'error',
                    'message': f"Error executing {command_id}: {str(e)}",
                    'source': 'CommandRegistry'
                })

# Create a global command registry instance
command_registry = CommandRegistry()
