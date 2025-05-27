# plan_04_command_registration.md
## Component: Command Registration and Usage

### Objective
Improve the way commands are registered and used throughout the TUI codebase. Create a more consistent pattern for command creation, registration, and execution.

### Findings
Based on analysis of the codebase:

1. **Command Registration Issues**:
   - Inconsistent patterns for command creation and registration
   - Some commands are created during initialization, others on-the-fly
   - No central registry for commands

2. **Command Usage Issues**:
   - Inconsistent patterns for command execution
   - Some commands are executed directly, others through handlers
   - No consistent pattern for passing state and context to commands

3. **Command Lifecycle Issues**:
   - No clear lifecycle for commands (creation, registration, execution, cleanup)
   - Commands are sometimes recreated unnecessarily

4. **Affected Components**:
   - `openhcs/tui/commands.py`: Command implementations
   - `openhcs/tui/menu_bar.py`: Command usage in menu system
   - `openhcs/tui/tui_architecture.py`: Command creation and registration

### Plan
1. **Create Command Registry**:
   - Create a central registry for commands
   - Add methods for registering and retrieving commands

2. **Update Command Creation**:
   - Create a consistent pattern for command creation
   - Use factory methods or dependency injection for command creation

3. **Update Command Usage**:
   - Create a consistent pattern for command execution
   - Use the command registry for retrieving commands

4. **Update Menu Bar**:
   - Use the command registry for retrieving commands
   - Create a consistent pattern for command execution in menu handlers

5. **Update TUI Architecture**:
   - Register commands during initialization
   - Use the command registry for retrieving commands

6. **Static Analysis**:
   - Use the meta tool to analyze command registration and usage patterns
   - Apply call graph analysis to verify command execution flows
   - Verify architectural compliance through static analysis

### Implementation Draft
```python
# In openhcs/tui/commands/registry.py

import logging
from typing import Dict, Type, Optional, Any, TypeVar, Generic, List, Callable, Awaitable

from openhcs.tui.commands.base import Command

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=Command)

class CommandRegistry:
    """
    Registry for Command objects.

    This class provides a central registry for Command objects, allowing them to be
    registered, retrieved, and executed in a consistent way.
    """

    def __init__(self):
        """Initialize the command registry."""
        self._commands: Dict[str, Command] = {}
        self._command_types: Dict[str, Type[Command]] = {}

    def register(self, command_id: str, command: Command) -> None:
        """
        Register a Command instance.

        Args:
            command_id: The ID to register the command under
            command: The Command instance to register
        """
        if command_id in self._commands:
            logger.warning(f"Command {command_id} already registered, overwriting")
        self._commands[command_id] = command

    def register_type(self, command_id: str, command_type: Type[Command]) -> None:
        """
        Register a Command type.

        Args:
            command_id: The ID to register the command type under
            command_type: The Command type to register
        """
        if command_id in self._command_types:
            logger.warning(f"Command type {command_id} already registered, overwriting")
        self._command_types[command_id] = command_type

    def get(self, command_id: str) -> Optional[Command]:
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

    def create_handler(self, command_id: str, state: 'AppState') -> Optional[Callable[[], Awaitable[None]]]:
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
                    state.notify('operation_status_changed', {
                        'operation': 'command_execution',
                        'status': 'error',
                        'message': f"Error executing {command_id}: {str(e)}",
                        'source': 'CommandRegistry'
                    })

        return handler

    def execute(self, command_id: str, state: 'AppState', context: Optional[Dict[str, Any]] = None) -> Awaitable[None]:
        """
        Execute a command.

        Args:
            command_id: The ID of the command to execute
            state: The application state to pass to the command
            context: Optional context information for the command

        Returns:
            A coroutine that executes the command
        """
        command = self.get(command_id)
        if command is None:
            # Return a coroutine that does nothing
            async def noop() -> None:
                logger.warning(f"Command {command_id} not found, no-op")
            return noop()

        return command.execute(state, context)

# In openhcs/tui/tui_architecture.py

# Create a global command registry
command_registry = CommandRegistry()

# Register commands during initialization
def register_commands() -> None:
    """Register all commands."""
    from openhcs.tui.commands.file_commands import (
        NewPipelineCommand,
        OpenPipelineCommand,
        SavePipelineCommand,
        SavePipelineAsCommand,
        ExitCommand
    )
    from openhcs.tui.commands.edit_commands import (
        AddStepCommand,
        EditStepCommand,
        RemoveStepCommand,
        MoveStepUpCommand,
        MoveStepDownCommand
    )
    from openhcs.tui.commands.view_commands import (
        ToggleLogDrawerCommand,
        ToggleVimModeCommand,
        SetThemeCommand
    )
    from openhcs.tui.commands.pipeline_commands import (
        PreCompileCommand,
        CompileCommand,
        RunCommand,
        TestCommand
    )
    from openhcs.tui.commands.help_commands import (
        ShowHelpCommand,
        ShowGlobalSettingsDialogCommand
    )

    # Register file commands
    command_registry.register("new_pipeline", NewPipelineCommand())
    command_registry.register("open_pipeline", OpenPipelineCommand())
    command_registry.register("save_pipeline", SavePipelineCommand())
    command_registry.register("save_pipeline_as", SavePipelineAsCommand())
    command_registry.register("exit", ExitCommand())

    # Register edit commands
    command_registry.register("add_step", AddStepCommand())
    command_registry.register("edit_step", EditStepCommand())
    command_registry.register("remove_step", RemoveStepCommand())
    command_registry.register("move_step_up", MoveStepUpCommand())
    command_registry.register("move_step_down", MoveStepDownCommand())

    # Register view commands
    command_registry.register("toggle_log_drawer", ToggleLogDrawerCommand())
    command_registry.register("toggle_vim_mode", ToggleVimModeCommand())
    command_registry.register_type("set_theme", SetThemeCommand)

    # Register pipeline commands
    command_registry.register("pre_compile", PreCompileCommand())
    command_registry.register("compile", CompileCommand())
    command_registry.register("run", RunCommand())
    command_registry.register("test", TestCommand())

    # Register help commands
    command_registry.register("show_help", ShowHelpCommand())
    command_registry.register("show_global_settings", ShowGlobalSettingsDialogCommand())
```
