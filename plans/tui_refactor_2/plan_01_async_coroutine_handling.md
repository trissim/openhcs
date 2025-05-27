# plan_01_async_coroutine_handling.md
## Component: Async Coroutine Handling in TUI

### Objective
Fix the RuntimeWarning about coroutines never being awaited in the TUI, specifically focusing on the `ShowGlobalSettingsDialogCommand.execute` method and other similar async patterns. Ensure all coroutines are properly awaited throughout the codebase, particularly in the menu system and command handlers.

### Findings
Based on analysis of the prompt_toolkit interfaces and our implementation:

1. **Root Cause**: The warning occurs because Command objects with async `execute()` methods are being used directly as handlers in the menu system, but their coroutines are not being properly awaited.

2. **prompt_toolkit Patterns**:
   - prompt_toolkit's Application class has a complex async/await pattern with `run_async()` method
   - Dialogs like `message_dialog()` return an Application object with a `run_async()` method that must be awaited
   - The `get_app().create_background_task()` is used for scheduling async tasks

3. **Current Implementation Issues**:
   - In `menu_bar.py`, Command objects are assigned directly as handlers: `"_on_settings": ShowGlobalSettingsDialogCommand()`
   - In `_handle_menu_item()`, the handler is called with `await item.handler()` without checking if it's a Command object
   - The MenuItem class doesn't properly handle Command objects with async execute() methods

4. **Affected Components**:
   - `openhcs/tui/menu_bar.py`: MenuItem class and handler management
   - `openhcs/tui/commands.py`: Command implementations with async execute() methods
   - `openhcs/tui/dialogs/global_settings_editor.py`: Dialog with async show() method

### Plan
1. **Update MenuItem Class in menu_bar.py**:
   - Modify the `__init__` method to accept Command objects in the handler parameter
   - Update type annotations to include Command objects

2. **Add Command Handler Wrapper**:
   - Create a `_create_command_handler` method in MenuBar class that wraps Command objects in an async function
   - The wrapper should properly await the execute() method and handle exceptions

3. **Update Handler Map Creation**:
   - Modify `_create_handler_map` to use the wrapper for Command objects
   - Update the handler map to use the wrapper for ShowGlobalSettingsDialogCommand and ShowHelpCommand

4. **Update Menu Item Handling**:
   - Modify `_handle_menu_item` to check if the handler is a Command object and call execute() with proper error handling
   - Ensure all coroutines are properly awaited

5. **Update from_dict Method**:
   - Update the `from_dict` method to handle Command objects in the handler_map

6. **Static Analysis**:
   - Use the meta tool to analyze the updated code for potential coroutine issues
   - Verify type correctness of async/await patterns through static analysis
   - Ensure interface compliance with prompt_toolkit patterns

### Implementation Draft
```python
# In openhcs/tui/menu_bar.py

# 1. Update MenuItem class to handle Command objects
class MenuItem:
    def __init__(
        self,
        type: MenuItemType,
        label: str = "",
        handler: Optional[Union[Callable[[], Awaitable[None]], Command]] = None,
        shortcut: Optional[str] = None,
        enabled: Union[bool, Condition] = True,
        checked: Union[bool, Condition] = False,
        children: Optional[List['MenuItem']] = None
    ):
        # ... rest of the method remains the same

# 2. Add Command handler wrapper
def _create_command_handler(self, command: Command) -> Callable[[], Awaitable[None]]:
    """
    Create an async handler function for a Command object.

    Args:
        command: The Command object to wrap

    Returns:
        An async function that executes the command with the current state and context
    """
    async def handler() -> None:
        try:
            await command.execute(self.state, self.context)
        except Exception as e:
            logger.error(f"Error executing command {command.__class__.__name__}: {e}", exc_info=True)
            # Optionally show an error dialog to the user
            if hasattr(self.state, 'notify'):
                self.state.notify('operation_status_changed', {
                    'operation': 'command_execution',
                    'status': 'error',
                    'message': f"Error executing {command.__class__.__name__}: {str(e)}",
                    'source': 'MenuBar'
                })
    return handler

# 3. Update handler map creation to use the wrapper
def _create_handler_map(self) -> Dict[str, Union[Callable[[], Awaitable[None]], Command]]:
    """
    Create a map of handler names to handler functions or Command objects.

    Returns:
        Dictionary mapping handler names to callables or Commands
    """
    return {
        # ... other handlers
        "_on_settings": self._create_command_handler(self.show_global_settings_command),
        "_on_show_help": self._create_command_handler(self.show_help_command)
    }

# 4. Update menu item handling
async def _handle_menu_item(self, item: MenuItem) -> None:
    """
    Handle selection of a menu item.

    Args:
        item: The selected menu item
    """
    # Close menu
    await self._close_menu()

    # Toggle checkbox items
    if item.type == MenuItemType.CHECKBOX:
        item.set_checked(not item.is_checked())

    # Call handler
    if item.handler:
        if hasattr(item.handler, 'execute') and callable(item.handler.execute):
            # This is a Command object
            try:
                await item.handler.execute(self.state, self.context)
            except Exception as e:
                logger.error(f"Error executing command: {e}", exc_info=True)
                # Optionally show an error dialog to the user
                if hasattr(self.state, 'notify'):
                    await self.state.notify('operation_status_changed', {
                        'operation': 'command_execution',
                        'status': 'error',
                        'message': f"Error executing command: {str(e)}",
                        'source': 'MenuBar'
                    })
        else:
            # This is a regular callable
            await item.handler()

# 5. Update mouse handler creation to properly await coroutines
def create_mouse_handler(menu_item):
    def item_mouse_handler(mouse_event):
        if mouse_event.event_type == MouseEventType.MOUSE_UP:
            if menu_item.handler:
                # Create a background task to handle the menu item
                # This ensures the coroutine is properly awaited
                get_app().create_background_task(self._handle_menu_item(menu_item))
            return True
        return original_mouse_handler(mouse_event)
    return item_mouse_handler
```
