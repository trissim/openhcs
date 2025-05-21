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
from typing import Any, Callable, Container, Dict, List, Optional, Union


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
from prompt_toolkit.application import get_app
from prompt_toolkit.filters import Condition, has_focus
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Container, HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.containers import (DynamicContainer, Float,
                                              FloatContainer)
from prompt_toolkit.widgets import Box, Button, Frame, TextArea

from openhcs.core.context.processing_context import ProcessingContext
# Import from func_registry instead of function_registry to avoid circular imports
from openhcs.processing.func_registry import FUNC_REGISTRY


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