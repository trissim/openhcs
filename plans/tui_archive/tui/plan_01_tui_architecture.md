# plan_01_tui_architecture.md
## Component: TUI Architecture and Core Principles

### Objective
Define the architectural foundation and core principles for the OpenHCS TUI, ensuring alignment with OpenHCS's declarative nature while providing an intuitive interface for users.

### Plan

1. **Core Architectural Principles**
   - Implement a strict Model-View-Controller (MVC) pattern with clear separation of concerns
   - Use prompt-toolkit with asyncio for responsive, non-blocking UI operations
   - Maintain complete separation between UI state and OpenHCS state
   - Ensure all UI components are testable in isolation
   - Follow OpenHCS's declarative principles in the TUI design

2. **TUI Layout Structure**
   - Implement a three-pane layout with fixed top menu and bottom status bars
   - Make panes independently resizable with minimum size constraints (≥20 cols/5 rows)
   - Support both mouse and Vim-style keyboard navigation
   - Implement a drawer system for expandable components (logs, help)

3. **State Management**
   - Implement a centralized TUIState manager with observer pattern
   - Use event-driven communication between components
   - Ensure UI state changes are atomic and predictable
   - Maintain clear boundaries between UI state and ProcessingContext

4. **Error Handling and Validation**
   - Implement consistent error display in dedicated UI regions
   - Validate user input before passing to OpenHCS (Clause 92)
   - Provide clear feedback for validation failures
   - Log all errors to the status bar with expandable detailed log

5. **Integration with OpenHCS**
   - Use FUNC_REGISTRY for dynamic component generation
   - Leverage static introspection for parameter validation
   - Connect to the orchestrator for pipeline operations
   - Respect OpenHCS's memory type system and backend decorators

### Findings

#### Key Rules for Building a TUI for a Declarative System

1. **🔒 Static Reflection Over Configuration**
   - Rule: Use static reflection to discover and present capabilities
   - Example: Dynamically populate function dropdowns from FUNC_REGISTRY
   - Rationale: UI automatically stays in sync with available functionality

2. **🔒 Contract Preservation**
   - Rule: UI must preserve all contracts enforced by OpenHCS
   - Example: Prevent saving invalid function patterns that would fail compilation
   - Rationale: Makes illegal states unrepresentable in the UI

3. **🔒 Explicit State Transitions**
   - Rule: All state changes must be explicit and visible to the user
   - Example: Clearly indicate compilation status with visual indicators
   - Rationale: Maintains the declarative nature of OpenHCS in the UI

4. **🔒 No Hidden Mutations**
   - Rule: UI must not perform hidden mutations of underlying data
   - Example: Function pattern changes applied only when explicitly saved
   - Rationale: Prevents unexpected side effects and maintains predictability

5. **🔒 Validation Before Action (Clause 92)**
   - Rule: Validate all inputs before performing actions
   - Example: Pre-compile before enabling the run button
   - Rationale: Ensures early failure for invalid configurations

6. **🔒 Declarative UI Components**
   - Rule: UI components should be declaratively defined
   - Example: Define layouts using prompt-toolkit's container system
   - Rationale: Makes UI structure clear and maintainable

7. **🔒 Separation of Concerns**
   - Rule: Strictly separate UI logic from business logic
   - Example: UI components communicate with ProcessingContext through events
   - Rationale: Maintains clean architecture and testability

8. **🔒 Immutable Display, Mutable Edit**
   - Rule: Display components should be immutable; editing in dedicated editors
   - Example: Function patterns displayed as read-only until explicitly edited
   - Rationale: Prevents accidental modifications and clarifies intent

9. **🔒 Progressive Disclosure**
   - Rule: Show only what's relevant to the current context
   - Example: Only show parameter fields for the selected function
   - Rationale: Reduces cognitive load and focuses attention

10. **🔒 Consistent Feedback**
    - Rule: Provide consistent feedback for all operations
    - Example: Show compilation status in the same location with consistent formatting
    - Rationale: Builds user confidence and reduces confusion

#### prompt-toolkit with asyncio Integration

1. **⚡ Non-blocking UI**
   - Benefit: UI remains responsive during long-running operations
   - Implementation: `get_app().create_background_task(self._compile_pipeline())`
   - Rationale: Prevents UI freezing during compilation, execution, or I/O

2. **⚡ Event-driven Architecture**
   - Benefit: Components react to events without tight coupling
   - Implementation: `self.state.add_observer('plate_selected', self._on_plate_selected)`
   - Rationale: Maintains separation of concerns and simplifies component interaction

3. **⚡ Vim Keybindings**
   - Benefit: Familiar navigation for power users
   - Implementation: `kb = KeyBindings(); @kb.add('j', filter=vi_navigation_mode)`
   - Rationale: Improves efficiency for experienced users

4. **⚡ Mouse Support**
   - Benefit: Intuitive interaction for casual users
   - Implementation: `Application(mouse_support=True)`
   - Rationale: Provides multiple interaction methods for different user preferences

5. **⚡ Dynamic Layout Management**
   - Benefit: Flexible layout that adapts to user needs
   - Implementation: `DynamicContainer(lambda: self._get_current_layout())`
   - Rationale: Supports context-switching between different editing modes

6. **⚡ External Editor Integration**
   - Benefit: Leverage familiar editing tools for complex tasks
   - Implementation: Secure subprocess with validation (Clause 92)
   - Rationale: Combines power of external editors with OpenHCS validation

### Implementation Draft

```python
"""
OpenHCS Terminal User Interface (TUI) Architecture.

This module implements the core architecture for the OpenHCS TUI,
following the principles of declarative programming and static reflection.

🔒 Clause 5: Disallows any UI layout logic from proceeding without validation
All UI components must be validated before use to prevent silent failures.

🔒 Clause 92: Structural Validation First
All user inputs are validated before processing to prevent invalid states.

🔒 Clause 234: Pattern Type Conversion Requires Structural Truth
When converting function patterns, None is used as the key for unnamed groups.

🔒 Clause 299: Explicitly prohibits false instantiation stubs
All component placeholders must raise NotImplementedError when used.

🔒 Clause 503: Makes invisible work visible — restores load transfer
All unimplemented components must be clearly marked with TODO references.
"""
import asyncio
import os
from typing import Dict, List, Any, Optional, Callable, Union, Container


class Clause5Violation(Exception):
    """
    Exception raised when a UI component is used before it is implemented.

    🔒 Clause 5: Disallows any UI layout logic from proceeding without validation
    """
    pass


# Placeholder component classes that enforce Clause 299
# These will be replaced by actual implementations in their respective plans

# TODO(plan_02_plate_manager.md)
class PlateManagerPane:
    """Placeholder for the Plate Manager pane."""

    def __init__(self, *args, **kwargs):
        """
        Placeholder initializer that raises NotImplementedError.

        🔒 Clause 299: Explicitly prohibits false instantiation stubs
        """
        raise NotImplementedError("PlateManagerPane is not implemented yet — see plan_02_plate_manager.md")


# TODO(plan_03_step_viewer.md)
class StepViewerPane:
    """Placeholder for the Step Viewer pane."""

    def __init__(self, *args, **kwargs):
        """
        Placeholder initializer that raises NotImplementedError.

        🔒 Clause 299: Explicitly prohibits false instantiation stubs
        """
        raise NotImplementedError("StepViewerPane is not implemented yet — see plan_03_step_viewer.md")


# TODO(plan_04_action_menu.md)
class ActionMenuPane:
    """Placeholder for the Action Menu pane."""

    def __init__(self, *args, **kwargs):
        """
        Placeholder initializer that raises NotImplementedError.

        🔒 Clause 299: Explicitly prohibits false instantiation stubs
        """
        raise NotImplementedError("ActionMenuPane is not implemented yet — see plan_04_action_menu.md")


# TODO(plan_05a_function_pattern_editor_core.md)
class FunctionPatternEditor:
    """Placeholder for the Function Pattern Editor."""

    def __init__(self, *args, **kwargs):
        """
        Placeholder initializer that raises NotImplementedError.

        🔒 Clause 299: Explicitly prohibits false instantiation stubs
        """
        raise NotImplementedError("FunctionPatternEditor is not implemented yet — see plan_05a_function_pattern_editor_core.md")


# TODO(plan_06_status_bar.md)
class StatusBar:
    """Placeholder for the Status Bar."""

    def __init__(self, *args, **kwargs):
        """
        Placeholder initializer that raises NotImplementedError.

        🔒 Clause 299: Explicitly prohibits false instantiation stubs
        """
        raise NotImplementedError("StatusBar is not implemented yet — see plan_06_status_bar.md")


# TODO(plan_07_menu_bar.md)
class MenuBar:
    """Placeholder for the Menu Bar."""

    def __init__(self, *args, **kwargs):
        """
        Placeholder initializer that raises NotImplementedError.

        🔒 Clause 299: Explicitly prohibits false instantiation stubs
        """
        raise NotImplementedError("MenuBar is not implemented yet — see plan_07_menu_bar.md")

from prompt_toolkit import Application
from prompt_toolkit.layout import Layout, HSplit, VSplit, Window, Container
from prompt_toolkit.layout.containers import DynamicContainer, FloatContainer, Float
from prompt_toolkit.widgets import Frame, TextArea, Button, Box
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.filters import has_focus, Condition
from prompt_toolkit.application import get_app

from ezstitcher.core.context.processing_context import ProcessingContext
from ezstitcher.processing.function_registry import FUNC_REGISTRY


class TUIState:
    """
    Centralized state manager for the OpenHCS TUI.

    Implements the observer pattern for event-driven communication
    between UI components while maintaining separation of concerns.
    """
    def __init__(self):
        """Initialize the TUI state."""
        # Core state
        self.selected_plate: Optional[Dict[str, Any]] = None
        self.selected_step: Optional[Dict[str, Any]] = None
        self.compilation_status: Optional[str] = None
        self.is_compiled: bool = False
        self.is_running: bool = False
        self.error_message: Optional[str] = None

        # Observer pattern implementation
        self.observers: Dict[str, List[Callable]] = {}

    def add_observer(self, event_type: str, callback: Callable) -> None:
        """
        Register an observer for a specific event type.

        Args:
            event_type: The event type to observe
            callback: The callback function to call when the event occurs
        """
        if event_type not in self.observers:
            self.observers[event_type] = []
        self.observers[event_type].append(callback)

    def notify(self, event_type: str, data: Any = None) -> None:
        """
        Notify all observers of an event.

        Args:
            event_type: The event type that occurred
            data: Optional data to pass to observers
        """
        if event_type in self.observers:
            for callback in self.observers[event_type]:
                callback(data)

    def set_selected_plate(self, plate: Dict[str, Any]) -> None:
        """
        Set the selected plate and notify observers.

        Args:
            plate: The plate to select
        """
        self.selected_plate = plate
        self.notify('plate_selected', plate)

        # Reset compilation state when plate changes
        self.is_compiled = False
        self.error_message = None

    def set_selected_step(self, step: Dict[str, Any]) -> None:
        """
        Set the selected step and notify observers.

        Args:
            step: The step to select
        """
        self.selected_step = step
        self.notify('step_selected', step)

        # Reset compilation state when step changes
        self.is_compiled = False


class OpenHCSTUI:
    """
    Core TUI application for OpenHCS.

    Implements a three-pane layout with Vim keybindings and mouse support,
    following OpenHCS's declarative principles.
    """
    def __init__(self, context: ProcessingContext):
        """
        Initialize the OpenHCS TUI application.

        Args:
            context: The OpenHCS ProcessingContext
        """
        # Initialize state and context
        self.state = TUIState()
        self.context = context

        # Create key bindings
        self.kb = self._create_key_bindings()

        # Component validation - enforces Clause 92 and Clause 299
        self._validate_components_present()

        # Create root container
        self.root_container = self._create_root_container()

        # Create application
        self.application = Application(
            layout=Layout(self.root_container),
            key_bindings=self.kb,
            mouse_support=True,
            full_screen=True
        )

    def _create_key_bindings(self) -> KeyBindings:
        """
        Create key bindings for the application.

        Returns:
            KeyBindings object with global and Vim navigation bindings
        """
        kb = KeyBindings()

        # Global key bindings
        @kb.add('c-c')
        def _(event):
            """Exit the application."""
            event.app.exit()

        # Add Vim navigation bindings
        @kb.add('j', filter=Condition(lambda: self.vim_mode))
        def _(event):
            """Move down (Vim style)."""
            event.current_buffer.cursor_down()

        @kb.add('k', filter=Condition(lambda: self.vim_mode))
        def _(event):
            """Move up (Vim style)."""
            event.current_buffer.cursor_up()

        return kb

    def _validate_components_present(self):
        """
        Validate that all required components are present and properly implemented.

        🔒 Clause 92: Structural Validation First
        Prevents interface fraud by disallowing fake instantiations.

        🔒 Clause 299: Explicitly prohibits false instantiation stubs

        🔒 Clause 5: Disallows any UI layout logic from proceeding without validation

        🔒 Clause 503: Makes invisible work visible — restores load transfer

        Raises:
            Clause5Violation: If any required component is missing or unimplemented
        """
        # Initialize components with proper stubs that will raise errors if used
        # TODO(plan_02_plate_manager.md)
        self.plate_manager = PlateManagerPane(self.state, self.context)

        # TODO(plan_03_step_viewer.md)
        self.step_viewer = StepViewerPane(self.state, self.context)

        # TODO(plan_04_action_menu.md)
        self.action_menu = ActionMenuPane(self.state, self.context)

        # TODO(plan_05a_function_pattern_editor_core.md)
        self.function_pattern_editor = None  # Will be created on demand

        # TODO(plan_06_status_bar.md)
        self.status_bar = StatusBar(self.state)

        # TODO(plan_07_menu_bar.md)
        self.menu_bar = MenuBar(self.state)

    def _get_left_pane(self) -> Container:
        """
        Get the current left pane based on state.

        Returns:
            Either the Plate Manager or Function Pattern Editor

        Raises:
            Clause5Violation: If the requested pane is not implemented
        """
        if getattr(self.state, 'editing_pattern', False):
            if not hasattr(self, "function_pattern_editor") or self.function_pattern_editor is None:
                raise Clause5Violation("Function Pattern Editor is unimplemented.")
            return Frame(self.function_pattern_editor.container, title="Function Pattern Editor")
        else:
            if not hasattr(self, "plate_manager"):
                raise Clause5Violation("Plate Manager pane is unimplemented.")
            return Frame(self.plate_manager.container, title="Plate Manager")

    def _get_step_viewer(self) -> Container:
        """
        Get the Step Viewer pane.

        Returns:
            The Step Viewer container

        Raises:
            Clause5Violation: If the Step Viewer is not implemented
        """
        if not hasattr(self, "step_viewer"):
            raise Clause5Violation("Step Viewer pane is unimplemented.")
        return self.step_viewer.container

    def _get_action_menu(self) -> Container:
        """
        Get the Action Menu pane.

        Returns:
            The Action Menu container

        Raises:
            Clause5Violation: If the Action Menu is not implemented
        """
        if not hasattr(self, "action_menu"):
            raise Clause5Violation("Action Menu pane is unimplemented.")
        return self.action_menu.container

    def _get_status_bar(self) -> Container:
        """
        Get the Status Bar.

        Returns:
            The Status Bar container

        Raises:
            Clause5Violation: If the Status Bar is not implemented
        """
        if not hasattr(self, "status_bar"):
            raise Clause5Violation("Status Bar is unimplemented.")
        return self.status_bar.container

    def _get_menu_bar(self) -> Container:
        """
        Get the Menu Bar.

        Returns:
            The Menu Bar container

        Raises:
            Clause5Violation: If the Menu Bar is not implemented
        """
        if not hasattr(self, "menu_bar"):
            raise Clause5Violation("Menu Bar is unimplemented.")
        return self.menu_bar.container

    def _create_root_container(self) -> Container:
        """
        Create the root container for the application.

        Returns:
            Container with the three-pane layout

        Raises:
            Clause5Violation: If any required component is not implemented
        """
        return FloatContainer(
            content=HSplit([
                # Top menu bar
                Box(self._get_menu_bar(), height=1),

                # Main content area with three panes
                VSplit([
                    # Left pane: Plate Manager or Function Pattern Editor
                    DynamicContainer(lambda: self._get_left_pane()),

                    # Middle pane: Step Viewer
                    Frame(self._get_step_viewer(), title="Step Viewer"),

                    # Right pane: Action Menu
                    Frame(self._get_action_menu(), title="Action Menu")
                ]),

                # Bottom status bar
                Box(self._get_status_bar(), height=1)
            ]),
            floats=[]  # Will contain dialogs
        )

    # _get_left_pane method is now defined above with proper validation

    async def run(self) -> None:
        """Run the application asynchronously."""
        await self.application.run_async()


# Entry point
if __name__ == "__main__":
    # Create ProcessingContext
    context = ProcessingContext()

    # Create and run TUI
    tui = OpenHCSTUI(context)
    asyncio.run(tui.run())
```
