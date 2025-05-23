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
import logging # ADDED for logging config updates
from typing import Any, Callable, Container, Dict, List, Optional, Union

from openhcs.core.config import GlobalPipelineConfig
from openhcs.io.filemanager import FileManager # ADDED
# ProcessingContext is already imported below

# Import the new ActionMenuPane
from openhcs.tui.action_menu_pane import ActionMenuPane
# Import the actual PlateManagerPane
from openhcs.tui.plate_manager_core import PlateManagerPane as ActualPlateManagerPane
# Import the actual MenuBar
from openhcs.tui.menu_bar import MenuBar as ActualMenuBar
# Import the actual StepViewerPane
from openhcs.tui.step_viewer import StepViewerPane as ActualStepViewerPane
# Import the actual FunctionPatternEditor
from openhcs.tui.function_pattern_editor import FunctionPatternEditor as ActualFunctionPatternEditor
# Import the actual StatusBar
from openhcs.tui.status_bar import StatusBar as ActualStatusBar
# Import main strage registry
from openhcs.io.base import storage_registry

logger = logging.getLogger(__name__) # ADDED module-level logger


class Clause5Violation(Exception):
    """
    Exception raised when a UI component is used before it is implemented.

    🔒 Clause 5: Disallows any UI layout logic from proceeding without validation
    """
    pass


# Placeholder component classes that enforce Clause 299
# These will be replaced by actual implementations in their respective plans

# PlateManagerPane stub removed, will be imported from openhcs.tui.plate_manager_core


# StepViewerPane stub removed, will be imported from openhcs.tui.step_viewer
# Its initialization will be handled asynchronously.


# Placeholder for ActionMenuPane removed, as it's now imported from action_menu_pane.py


# FunctionPatternEditor stub removed, will be imported from openhcs.tui.function_pattern_editor
# It's created on demand by _get_left_pane.
# FunctionPatternEditor stub class definition removed.


# StatusBar stub removed, will be imported from openhcs.tui.status_bar


# MenuBar stub removed, will be imported from openhcs.tui.menu_bar


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
        self.is_running: bool = False # Generic flag for any long operation
        self.error_message: Optional[str] = None
        
        # Specific states for compile/run cycle
        self.compiled_contexts: Optional[Dict[str, ProcessingContext]] = None
        self.execution_status: Optional[str] = None

        # State for FunctionPatternEditor interaction
        self.editing_pattern: bool = False
        self.selected_step_for_editing: Optional[Dict[str, Any]] = None


        # TUI-specific settings (as per plans/tui_final/plan01_phase1.md)
        self.vim_mode: bool = False
        self.tui_log_level: str = "INFO"
        # Default editor from environment, fallback to vim
        self.editor_path: str = os.environ.get('EDITOR', 'vim')
        # Add other TUI specific state as needed, e.g. for active orchestrator
        self.active_orchestrator: Optional[Any] = None # Placeholder for actual orchestrator type

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

    async def notify(self, event_type: str, data: Any = None) -> None:
        """
        Notify all observers of an event.

        Args:
            event_type: The event type that occurred
            data: Optional data to pass to observers
        """
        if event_type in self.observers:
            for callback in self.observers[event_type]:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)

    async def set_selected_plate(self, plate: Dict[str, Any]) -> None:
        """
        Set the selected plate and notify observers.

        Args:
            plate: The plate to select
        """
        self.selected_plate = plate
        await self.notify('plate_selected', plate)

        # Reset compilation state when plate changes
        self.is_compiled = False
        self.error_message = None

    async def set_selected_step(self, step: Dict[str, Any]) -> None:
        """
        Set the selected step and notify observers.

        Args:
            step: The step to select
        """
        self.selected_step = step
        await self.notify('step_selected', step)

        # Reset compilation state when step changes
        self.is_compiled = False


class OpenHCSTUI:
    """
    Core TUI application for OpenHCS.

    Implements a three-pane layout with Vim keybindings and mouse support,
    following OpenHCS's declarative principles.
    """
    def __init__(self,
                 initial_context: ProcessingContext,
                 state: TUIState,
                 global_config: GlobalPipelineConfig):
        """
        Initialize the OpenHCS TUI application.

        Args:
            initial_context: A pre-configured ProcessingContext (contains global_config & filemanager).
            state: The shared TUIState instance.
            global_config: The shared GlobalPipelineConfig instance.
        """
        # Initialize state and context
        self.state = state # Use passed-in state
        self.context = initial_context # Use passed-in context (this is the TUI's initial/default context)
        # OpenHCSTUI is the main initialization class for the TUI, so it creates the storage registry.
        self.storage_registry = storage_registry()
        self.global_config = global_config # Store shared global_config
        
        self.step_viewer: Optional[ActualStepViewerPane] = None # For async initialization
        self.function_pattern_editor: Optional[ActualFunctionPatternEditor] = None # Created on demand

        # Create key bindings
        self.kb = self._create_key_bindings()

        # Component validation - enforces Clause 92 and Clause 299
        self._validate_components_present()

        # Register observer for launcher's core config updates
        self.state.add_observer('launcher_core_config_rebound', self._on_launcher_config_rebound)

        # Create root container
        self.root_container = self._create_root_container()

        # Create application
        self.application = Application(
            layout=Layout(self.root_container),
            key_bindings=self.kb,
            mouse_support=True,
            full_screen=True
        )
        
        # Schedule asynchronous initialization of components that require it
        # Ensure get_app() is valid here, or defer to a point where it is.
        # If called too early, get_app() might not yet return the created application.
        # A common pattern is to do this in an app.startup event if available,
        # or ensure this __init__ is called when an event loop is already running.
        # For now, assuming get_app() will work when this task is scheduled.
        if self.application: # Ensure application object exists
             self.application.create_background_task(self._async_initialize_step_viewer())

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
        # These components would now receive state, and can access filemanager
        # and global_config via self.filemanager, self.global_config, or self.context.
        # For now, their instantiation signature is kept simple as they are stubs.
        # When implemented, they'll need to be passed the correct dependencies.
        
        # TODO(plan_02_plate_manager.md)
        # Example: self.plate_manager = PlateManagerPane(state=self.state, filemanager=self.filemanager)
        # Instantiate the actual PlateManagerPane from plate_manager_core.py
        self.plate_manager = ActualPlateManagerPane(
            state=self.state,
            context=self.context,
            storage_registry=self.storage_registry, # Pass the shared registry
        )

        # TODO(plan_03_step_viewer.md)
        # self.step_viewer is now initialized asynchronously in __init__ via _async_initialize_step_viewer
        # No direct instantiation here anymore.

        # TODO(plan_04_action_menu.md)
        # Now instantiates the ActionMenuPane from openhcs.tui.action_menu_pane
        # The arguments self.state (TUIState) and self.context (initial_tui_context)
        # match the __init__ signature of the new ActionMenuPane.
        self.action_menu = ActionMenuPane(state=self.state, initial_tui_context=self.context)

        # TODO(plan_05a_function_pattern_editor_core.md)
        # self.function_pattern_editor is initialized to None in __init__ and created by _get_left_pane.

        # TODO(plan_06_status_bar.md)
        # Instantiate the actual StatusBar from status_bar.py
        self.status_bar = ActualStatusBar(self.state)

        # TODO(plan_07_menu_bar.md)
        # Instantiate the actual MenuBar from menu_bar.py
        self.menu_bar = ActualMenuBar(self.state)

    def _get_left_pane(self) -> Container:
        """
        Get the current left pane based on state.
        Dynamically shows PlateManagerPane or FunctionPatternEditor.
        """
        # Check TUIState for 'editing_pattern' and 'selected_step_for_editing'
        # These attributes need to be managed by TUIState based on user actions
        # (e.g., clicking "Edit Step" in StepViewerPane sets these).
        is_editing_pattern = getattr(self.state, 'editing_pattern', False)
        step_to_edit = getattr(self.state, 'selected_step_for_editing', None)

        if is_editing_pattern:
            if step_to_edit is None:
                logger.warning("OpenHCSTUI: 'editing_pattern' is true, but 'selected_step_for_editing' is not set in TUIState.")
                # Fallback: Show an error message or the PlateManagerPane
                return Frame(Box(Label("Error: No step selected for pattern editing.")), title="Error")

            # Logic to create or reuse FunctionPatternEditor instance
            # Recreate if the step to edit has changed or if it's the first time
            if self.function_pattern_editor is None or \
               not hasattr(self.function_pattern_editor, 'step') or \
               self.function_pattern_editor.step != step_to_edit:
                logger.info(f"OpenHCSTUI: Instantiating FunctionPatternEditor for step: {step_to_edit.get('name', 'N/A')}")
                try:
                    self.function_pattern_editor = ActualFunctionPatternEditor(state=self.state, step=step_to_edit)
                except Exception as e:
                    logger.error(f"OpenHCSTUI: Failed to instantiate FunctionPatternEditor: {e}", exc_info=True)
                    self.function_pattern_editor = None # Ensure it's None on failure
                    return Frame(Box(Label(f"Error creating FPE: {e}")), title="FPE Error")
            
            if self.function_pattern_editor and hasattr(self.function_pattern_editor, 'container'):
                return Frame(self.function_pattern_editor.container, title="Function Pattern Editor")
            else:
                logger.error("OpenHCSTUI: FunctionPatternEditor instance available but has no container, or failed instantiation.")
                return Frame(Box(Label("Error: Function Pattern Editor could not be loaded.")), title="FPE Error")
        else:
            # Not editing a pattern, show PlateManagerPane
            # If FPE was active, clear its instance so it's recreated fresh next time
            if self.function_pattern_editor is not None:
                logger.debug("OpenHCSTUI: Clearing FunctionPatternEditor instance as 'editing_pattern' is false.")
                # TODO: If FPE has a close/cleanup method, call it here.
                # For now, just dereferencing.
                self.function_pattern_editor = None
            
            if self.plate_manager and hasattr(self.plate_manager, 'container'):
                return Frame(self.plate_manager.container, title="Plate Manager")
            else:
                logger.error("OpenHCSTUI: PlateManagerPane not available or has no container.")
                # This should ideally not happen if PlateManagerPane initializes correctly.
                return Frame(Box(Label("Error: Plate Manager not available.")), title="Error")

    def _get_step_viewer(self) -> Container:
        """
        Get the Step Viewer pane.
        Returns a placeholder if the pane is not yet initialized.
        """
        if self.step_viewer is None:
            # Return a temporary placeholder if not yet initialized or init failed
            logger.debug("OpenHCSTUI: StepViewerPane not yet initialized, returning placeholder.")
            return Box(Label("Step Viewer - Initializing..."), padding=1)
        
        # Ensure the initialized step_viewer has a container
        if not hasattr(self.step_viewer, 'container') or self.step_viewer.container is None:
            logger.error("OpenHCSTUI: StepViewerPane is initialized but has no container.")
            return Box(Label("Step Viewer - Error: No container"), padding=1)
            
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

    def _on_launcher_config_rebound(self, new_core_config: GlobalPipelineConfig) -> None:
        """
        Callback for when the OpenHCSTUILauncher updates the core_global_config.
        Updates this TUI instance's reference to the global configuration.
        """
        if isinstance(new_core_config, GlobalPipelineConfig):
            self.global_config = new_core_config
            logger.info(
                "OpenHCSTUI: Internal GlobalPipelineConfig reference updated. "
                f"New num_workers: {self.global_config.num_workers}"
            )
            # Potentially trigger a UI refresh or notify child components if they
            # cache or depend directly on aspects of global_config that affect rendering.
            # For now, just updating the reference.
            get_app().invalidate() # Request a redraw, as some UI might depend on config
        else:
            logger.warning(
                "OpenHCSTUI: Received invalid data for 'launcher_core_config_rebound' event. "
                f"Expected GlobalPipelineConfig, got {type(new_core_config)}."
            )

    async def shutdown_components(self):
        """
        Gracefully shut down all managed TUI components that have a shutdown method.
        """
        logger.info("OpenHCSTUI: Initiating shutdown sequence for components...")
        
        component_attributes = [
            'plate_manager',
            'step_viewer',
            'action_menu',
            'status_bar',
            'menu_bar',
            'function_pattern_editor' # This might be None if never activated
        ]
 
        for attr_name in component_attributes:
            component = getattr(self, attr_name, None)
            if component and hasattr(component, 'shutdown') and callable(component.shutdown):
                try:
                    logger.info(f"Attempting to shut down {attr_name} ({component.__class__.__name__})...")
                    # Check if shutdown is an async method
                    if asyncio.iscoroutinefunction(component.shutdown):
                        await component.shutdown()
                    else:
                        component.shutdown() # Call synchronously if not async
                    logger.info(f"Successfully shut down {attr_name}.")
                except Exception as e:
                    logger.error(f"Error during shutdown of {attr_name} ({component.__class__.__name__}): {e}", exc_info=True)
            elif component:
                logger.debug(f"Component {attr_name} ({component.__class__.__name__}) does not have a callable 'shutdown' method.")
            # else:
                # logger.debug(f"Component attribute {attr_name} not found or is None.")
 
        logger.info("OpenHCSTUI: Component shutdown sequence complete.")
 
    async def _async_initialize_step_viewer(self):
        """
        Asynchronously initializes the StepViewerPane.
        This is called as a background task after OpenHCSTUI is instantiated.
        """
        if self.step_viewer is not None:
            logger.info("OpenHCSTUI: StepViewerPane already initialized or initialization in progress.")
            return
 
        logger.info("OpenHCSTUI: Asynchronously initializing StepViewerPane...")
        try:
            # Use the async factory 'create' from step_viewer.py
            self.step_viewer = await ActualStepViewerPane.create(self.state, self.context)
            logger.info("OpenHCSTUI: StepViewerPane initialized and setup complete.")
        except Exception as e:
            logger.error(f"OpenHCSTUI: Error initializing StepViewerPane: {e}", exc_info=True)
            # self.step_viewer remains None, _get_step_viewer will show "Initializing..." or an error.

    # Implement abstract methods by delegating to the root_container
    def get_children(self):
        return self.root_container.get_children()

    def preferred_width(self, max_available_width):
        return self.root_container.preferred_width(max_available_width)

    def preferred_height(self, max_available_height, width):
        return self.root_container.preferred_height(max_available_height, width)

    def reset(self):
        self.root_container.reset()

    def write_to_screen(self, screen, mouse_handlers, write_position,
                        parent_style, erase_bg, z_index):
        self.root_container.write_to_screen(screen, mouse_handlers, write_position,
                                           parent_style, erase_bg, z_index)

    def mouse_handler(self, mouse_event):
        """Handle mouse events by delegating to the root container."""
        return self.root_container.mouse_handler(mouse_event)