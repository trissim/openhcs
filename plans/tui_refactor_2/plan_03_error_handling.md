# plan_03_error_handling.md
## Component: Error Handling in Async Functions

### Objective
Improve error handling in async functions throughout the TUI codebase, particularly in Command implementations and dialog interactions. Ensure errors are properly caught, logged, and displayed to the user.

### Findings
Based on analysis of the codebase:

1. **Error Handling Issues**:
   - Inconsistent error handling in async functions
   - Some errors are caught and logged, others are not
   - Some errors are displayed to the user, others are silently ignored
   - Missing try/except blocks in some async functions

2. **Dialog Error Handling**:
   - Some dialogs don't properly handle errors during show() or other async methods
   - Error messages are not consistently displayed to the user

3. **Command Error Handling**:
   - Some Command implementations have proper error handling, others don't
   - Errors in Command.execute() methods are not consistently handled

4. **Affected Components**:
   - `openhcs/tui/commands.py`: Command implementations and error handling
   - `openhcs/tui/menu_bar.py`: MenuItem handling and error handling
   - `openhcs/tui/dialogs/*.py`: Dialog implementations and error handling

### Plan
1. **Create Consistent Error Handling Pattern**:
   - Define a consistent pattern for error handling in async functions
   - Include try/except blocks, logging, and user feedback

2. **Update Command Implementations**:
   - Add proper error handling to all Command.execute() methods
   - Ensure errors are logged and displayed to the user

3. **Update Dialog Implementations**:
   - Add proper error handling to all dialog async methods
   - Ensure errors during dialog operations are properly handled

4. **Update Menu Bar**:
   - Improve error handling in _handle_menu_item and other async methods
   - Ensure errors in Command objects are properly caught and handled

5. **Create Helper Functions**:
   - Create helper functions for common error handling patterns
   - Use these helpers consistently throughout the codebase

6. **Static Analysis**:
   - Use the meta tool to analyze error handling patterns
   - Apply static analysis to identify potential uncaught exceptions
   - Verify exception flow through call graphs

### Implementation Draft
```python
# In openhcs/tui/utils/error_handling.py

import logging
import traceback
from typing import Callable, Awaitable, TypeVar, Optional, Dict, Any, Union, Type

logger = logging.getLogger(__name__)

T = TypeVar('T')

# 1. Create consistent error handling pattern
async def handle_async_errors(
    coro: Awaitable[T],
    error_message: str = "An error occurred",
    notify_state: Optional['AppState'] = None,
    operation_name: str = "operation",
    source: str = "unknown",
    error_types: Optional[Union[Type[Exception], Tuple[Type[Exception], ...]]] = None
) -> Optional[T]:
    """
    Handle errors in async functions.

    Args:
        coro: The coroutine to execute
        error_message: The error message to log
        notify_state: Optional AppState to notify of the error
        operation_name: The name of the operation for notification
        source: The source of the error for notification
        error_types: Specific error types to catch, or None to catch all exceptions

    Returns:
        The result of the coroutine, or None if an error occurred
    """
    try:
        return await coro
    except Exception as e:
        if error_types is None or isinstance(e, error_types):
            logger.error(f"{error_message}: {e}", exc_info=True)

            # Notify state if provided
            if notify_state is not None and hasattr(notify_state, 'notify'):
                notify_state.notify('operation_status_changed', {
                    'operation': operation_name,
                    'status': 'error',
                    'message': f"{error_message}: {str(e)}",
                    'source': source
                })
            return None
        else:
            # Re-raise if not a specified error type
            raise

# 2. Create decorator for async error handling
def handle_async_errors_decorator(
    error_message: str = "An error occurred",
    operation_name: str = "operation",
    source: str = "unknown",
    error_types: Optional[Union[Type[Exception], Tuple[Type[Exception], ...]]] = None
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[Optional[T]]]]:
    """
    Decorator for handling errors in async functions.

    Args:
        error_message: The error message to log
        operation_name: The name of the operation for notification
        source: The source of the error for notification
        error_types: Specific error types to catch, or None to catch all exceptions

    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[Optional[T]]]:
        async def wrapper(*args, **kwargs) -> Optional[T]:
            # Find state argument if present
            state = None
            if args and hasattr(args[0], 'state'):
                state = args[0].state
            elif 'state' in kwargs:
                state = kwargs['state']

            return await handle_async_errors(
                func(*args, **kwargs),
                error_message=error_message,
                notify_state=state,
                operation_name=operation_name,
                source=source,
                error_types=error_types
            )
        return wrapper
    return decorator

# In openhcs/tui/commands.py

# 3. Update Command implementations with error handling
class ShowGlobalSettingsDialogCommand:
    """Command to show the global settings dialog."""

    @handle_async_errors_decorator(
        error_message="Error showing global settings dialog",
        operation_name="show_global_settings",
        source="GlobalSettingsCommand"
    )
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
```
