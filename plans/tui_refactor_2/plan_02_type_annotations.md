# plan_02_type_annotations.md
## Component: Type Annotations and Interface Compliance

### Objective
Improve type annotations throughout the codebase, particularly in the TUI components, to ensure better static type checking and IDE support. Focus on Command objects, async functions, and prompt_toolkit interfaces.

### Findings
Based on analysis of the codebase and prompt_toolkit interfaces:

1. **Type Annotation Issues**:
   - Inconsistent use of type annotations for Command objects
   - Some places use string literals for forward references, others use proper imports
   - Missing or incorrect return type annotations for async functions
   - Inconsistent use of Optional, Union, and other typing constructs

2. **Interface Compliance Issues**:
   - Some classes implement multiple interfaces, leading to confusion
   - ShowGlobalSettingsDialogCommand implements many interfaces beyond just Command
   - Some methods don't properly implement the interfaces they claim to

3. **Affected Components**:
   - `openhcs/tui/commands.py`: Command implementations and type annotations
   - `openhcs/tui/menu_bar.py`: MenuItem class and handler type annotations
   - `openhcs/tui/dialogs/*.py`: Dialog implementations and async patterns

### Plan
1. **Update Command Protocol**:
   - Ensure the Command Protocol in commands.py has proper type annotations
   - Add proper return type annotations for async methods

2. **Update Command Implementations**:
   - Update all Command implementations to use consistent type annotations
   - Ensure all async methods have proper return type annotations
   - Use proper imports for forward references where possible

3. **Update MenuItem Class**:
   - Update type annotations in MenuItem class to properly handle Command objects
   - Ensure all methods have proper return type annotations

4. **Update Dialog Implementations**:
   - Ensure all async methods in dialogs have proper return type annotations
   - Use consistent patterns for async/await

5. **Simplify Interface Implementations**:
   - Identify classes that implement too many interfaces
   - Refactor to simplify interface implementations where appropriate

6. **Static Analysis**:
   - Use the meta tool to analyze type annotations and interface compliance
   - Leverage the interface_classifier component to verify interface implementations
   - Apply static analysis to identify potential type mismatches

### Implementation Draft
```python
# In openhcs/tui/commands.py

from typing import Protocol, Awaitable, Optional, Dict, Any, TypeVar, Generic, List, Union

# 1. Update Command Protocol with proper type annotations
class Command(Protocol):
    """Protocol for command objects that can be executed."""

    async def execute(self, state: 'AppState', context: Optional[Dict[str, Any]] = None) -> None:
        """
        Execute the command.

        Args:
            state: The application state
            context: Optional context information for the command

        Returns:
            None
        """
        ...

# 2. Update Command implementations with proper type annotations
class ShowGlobalSettingsDialogCommand:
    """Command to show the global settings dialog."""

    async def execute(self, state: 'AppState', context: Optional[Dict[str, Any]] = None) -> None:
        """
        Show the global settings dialog.

        Args:
            state: The application state
            context: Optional context information for the command

        Returns:
            None
        """
        from openhcs.tui.dialogs.global_settings_editor import GlobalSettingsEditorDialog

        dialog = GlobalSettingsEditorDialog(state.config)
        result = await dialog.run_async()

        if result:
            # Handle dialog result
            state.config.update(result)
            state.notify("config_updated", result)

# In openhcs/tui/menu_bar.py

# 3. Update MenuItem class with proper type annotations
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
        """
        Initialize a menu item.

        Args:
            type: The type of menu item
            label: The label to display
            handler: The handler function or Command object to call when the item is selected
            shortcut: The keyboard shortcut for the item
            enabled: Whether the item is enabled
            checked: Whether the item is checked (for checkbox items)
            children: Child menu items (for submenu items)
        """
        # ... rest of the method remains the same

# 4. Update from_dict method with proper type annotations
@classmethod
def from_dict(cls, item_dict: Dict[str, Any], handler_map: Dict[str, Union[Callable[[], Awaitable[None]], Command]]) -> 'MenuItem':
    """
    Create a MenuItem from a dictionary.

    Args:
        item_dict: Dictionary with menu item data
        handler_map: Map of handler names to handler functions or Command objects

    Returns:
        MenuItem instance
    """
    # ... rest of the method remains the same
```
