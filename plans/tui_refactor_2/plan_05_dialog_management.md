# plan_05_dialog_management.md
## Component: Dialog Management

### Objective
Improve dialog management throughout the TUI codebase. Create a more consistent pattern for dialog creation, showing, and handling results. Ensure proper async/await patterns are used for all dialog operations.

### Findings
Based on analysis of the codebase and prompt_toolkit interfaces:

1. **Dialog Management Issues**:
   - Inconsistent patterns for dialog creation and showing
   - Some dialogs use async show() methods, others don't
   - No consistent pattern for handling dialog results

2. **Dialog Async Patterns**:
   - Some dialogs properly use async/await, others don't
   - Inconsistent use of run_async() method from prompt_toolkit
   - Missing await statements in some dialog operations

3. **Dialog Lifecycle Issues**:
   - No clear lifecycle for dialogs (creation, showing, handling results, cleanup)
   - Dialogs are sometimes recreated unnecessarily

4. **Affected Components**:
   - `openhcs/tui/dialogs/*.py`: Dialog implementations
   - `openhcs/tui/commands.py`: Command implementations that show dialogs
   - `openhcs/tui/tui_architecture.py`: Dialog creation and management

### Plan
1. **Create Dialog Base Class**:
   - Create a base class for all dialogs with consistent async patterns
   - Include standard methods for showing, handling results, and cleanup

2. **Update Dialog Implementations**:
   - Update all dialog implementations to use the base class
   - Ensure consistent async/await patterns in all dialog methods

3. **Update Dialog Usage**:
   - Create a consistent pattern for dialog creation and showing
   - Ensure proper async/await patterns when showing dialogs

4. **Update Command Implementations**:
   - Update commands that show dialogs to use consistent patterns
   - Ensure proper async/await patterns when showing dialogs from commands

5. **Update TUI Architecture**:
   - Create a dialog manager for centralized dialog management
   - Use the dialog manager for all dialog operations

6. **Static Analysis**:
   - Use the meta tool to analyze dialog management patterns
   - Apply static analysis to verify async/await patterns in dialog operations
   - Verify interface compliance with prompt_toolkit patterns

### Implementation Draft
```python
# In openhcs/tui/dialogs/base.py

import logging
from typing import TypeVar, Generic, Optional, Dict, Any, List, Callable, Awaitable, Union, Type

from prompt_toolkit.application import Application
from prompt_toolkit.layout import Layout
from prompt_toolkit.key_binding import KeyBindings

logger = logging.getLogger(__name__)

T = TypeVar('T')

class BaseDialog(Generic[T]):
    """
    Base class for all dialogs.

    This class provides a consistent interface for all dialogs, with standard
    methods for showing, handling results, and cleanup.
    """

    def __init__(self, title: str = "Dialog"):
        """
        Initialize the dialog.

        Args:
            title: The title of the dialog
        """
        self.title = title
        self._application: Optional[Application] = None
        self._result: Optional[T] = None
        self._is_showing = False

    def _create_application(self) -> Application:
        """
        Create the prompt_toolkit Application for the dialog.

        This method should be overridden by subclasses to create the specific
        Application for the dialog.

        Returns:
            The prompt_toolkit Application
        """
        raise NotImplementedError("Subclasses must implement _create_application")

    async def show(self) -> Optional[T]:
        """
        Show the dialog and wait for a result.

        Returns:
            The result of the dialog, or None if cancelled
        """
        if self._is_showing:
            logger.warning("Dialog is already showing")
            return None

        self._is_showing = True
        self._result = None

        try:
            # Create the application if it doesn't exist
            if self._application is None:
                self._application = self._create_application()

            # Run the application and wait for a result
            await self._application.run_async()

            # Return the result
            return self._result
        except Exception as e:
            logger.error(f"Error showing dialog: {e}", exc_info=True)
            return None
        finally:
            self._is_showing = False

    def set_result(self, result: Optional[T]) -> None:
        """
        Set the result of the dialog and exit.

        Args:
            result: The result to set
        """
        self._result = result

        # Exit the application if it exists
        if self._application is not None:
            self._application.exit()

    async def run_async(self) -> Optional[T]:
        """
        Alias for show() to match prompt_toolkit's interface.

        Returns:
            The result of the dialog, or None if cancelled
        """
        return await self.show()

    def cleanup(self) -> None:
        """
        Clean up resources used by the dialog.

        This method should be called when the dialog is no longer needed.
        """
        self._application = None
        self._result = None
        self._is_showing = False

# In openhcs/tui/dialogs/manager.py

class DialogManager:
    """
    Manager for dialogs.

    This class provides a centralized way to create and show dialogs.
    """

    def __init__(self, state: 'AppState'):
        """
        Initialize the dialog manager.

        Args:
            state: The application state
        """
        self.state = state
        self._dialogs: Dict[str, BaseDialog] = {}

    def register(self, dialog_id: str, dialog: BaseDialog) -> None:
        """
        Register a dialog.

        Args:
            dialog_id: The ID to register the dialog under
            dialog: The dialog to register
        """
        if dialog_id in self._dialogs:
            logger.warning(f"Dialog {dialog_id} already registered, overwriting")
            # Clean up the old dialog
            self._dialogs[dialog_id].cleanup()
        self._dialogs[dialog_id] = dialog

    def get(self, dialog_id: str) -> Optional[BaseDialog]:
        """
        Get a dialog.

        Args:
            dialog_id: The ID of the dialog to get

        Returns:
            The dialog, or None if not found
        """
        if dialog_id in self._dialogs:
            return self._dialogs[dialog_id]
        else:
            logger.warning(f"Dialog {dialog_id} not found")
            return None

    async def show(self, dialog_id: str) -> Any:
        """
        Show a dialog.

        Args:
            dialog_id: The ID of the dialog to show

        Returns:
            The result of the dialog, or None if cancelled or not found
        """
        dialog = self.get(dialog_id)
        if dialog is None:
            return None

        try:
            return await dialog.show()
        except Exception as e:
            logger.error(f"Error showing dialog {dialog_id}: {e}", exc_info=True)
            return None

    def cleanup(self) -> None:
        """
        Clean up all dialogs.

        This method should be called when the application is shutting down.
        """
        for dialog in self._dialogs.values():
            dialog.cleanup()
        self._dialogs.clear()

# In openhcs/tui/tui_architecture.py

# Create a global dialog manager
dialog_manager = None

def initialize_dialog_manager(state: 'AppState') -> None:
    """
    Initialize the dialog manager.

    Args:
        state: The application state
    """
    global dialog_manager
    dialog_manager = DialogManager(state)

    # Register dialogs
    from openhcs.tui.dialogs.global_settings_editor import GlobalSettingsEditorDialog
    from openhcs.tui.dialogs.help_dialog import HelpDialog

    dialog_manager.register("global_settings", GlobalSettingsEditorDialog(state.config))
    dialog_manager.register("help", HelpDialog())
```
