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
from typing import Any, Callable, Container, Dict, List, Optional, Union, TYPE_CHECKING

from openhcs.core.config import GlobalPipelineConfig
from openhcs.io.filemanager import FileManager # ADDED
# ProcessingContext is already imported below

# Import the actual PlateManagerPane
from openhcs.tui.plate_manager_core import PlateManagerPane as ActualPlateManagerPane
# Import the actual MenuBar
from openhcs.tui.menu_bar import MenuBar as ActualMenuBar
# Import the actual StepViewerPane
from openhcs.tui.step_viewer import StepViewerPane as ActualStepViewerPane
# Import the actual FunctionPatternEditor (old) and new DualStepFuncEditorPane
from openhcs.tui.function_pattern_editor import FunctionPatternEditor as ActualFunctionPatternEditor # Keep for now if any logic is reused
from openhcs.tui.dual_step_func_editor import DualStepFuncEditorPane # New editor
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


from prompt_toolkit import Application
from prompt_toolkit.application import get_app
from prompt_toolkit.filters import Condition, has_focus
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Container, HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.containers import (DynamicContainer, Float,
                                              FloatContainer)
from prompt_toolkit.widgets import Box, Button, Frame, Label, TextArea

from openhcs.core.context.processing_context import ProcessingContext
if TYPE_CHECKING: # Add for PipelineOrchestrator type hint
    from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
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

        # State for DualStepFuncEditorPane interaction
        self.editing_step_config: bool = False # Renamed from editing_pattern
        self.step_to_edit_config: Optional[Dict[str, Any]] = None # Renamed from selected_step_for_editing

        # State for PlateConfigEditorPane interaction
        self.editing_plate_config: bool = False
        self.orchestrator_for_plate_config_edit: Optional["PipelineOrchestrator"] = None

        # Global configuration (rebound from launcher)
        self.global_config: Optional[GlobalPipelineConfig] = None # Added for global config access

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


# Added import for FunctionStep type hint
from openhcs.core.steps.function_step import FunctionStep

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
        self.state.global_config = global_config # Also set in TUIState for broader access

        self.step_viewer: Optional[ActualStepViewerPane] = None # For async initialization
        self.function_pattern_editor: Optional[ActualFunctionPatternEditor] = None # Old editor, might be removed later
        self.dual_step_func_editor: Optional[DualStepFuncEditorPane] = None # New editor, created on demand
        self.plate_config_editor: Optional[Any] = None # Placeholder for PlateConfigEditorPane instance

        # Create key bindings
        self.kb = self._create_key_bindings()

        # Component instantiation (formerly _validate_components_present)
        self._initialize_components() # Renamed for clarity

        # Register observer for launcher's core config updates
        self.state.add_observer('launcher_core_config_rebound', self._on_launcher_config_rebound)

        # Observers for PlateConfigEditorPane
        self.state.add_observer('show_edit_plate_config_requested', self._handle_show_edit_plate_config_request)
        self.state.add_observer('plate_config_editing_cancelled', self._handle_plate_config_editing_cancelled)
        self.state.add_observer('plate_config_saved', self._handle_plate_config_saved)

        # Observer for DualStepFuncEditorPane close/cancel
        self.state.add_observer('step_editing_cancelled', self._handle_step_editing_cancelled)
        # Note: 'step_pattern_saved' is handled by PipelineEditorPane to update its internal step list.
        # OpenHCSTUI only needs to know when to stop showing the editor.
        # 'edit_step_dialog_requested' is handled by PipelineEditorPane to set TUIState.

        # Create application
        self.application = Application(
            layout=Layout(self._create_root_container()), # Call _create_root_container here
            key_bindings=self.kb,
            mouse_support=True,
            full_screen=True
        )

        # Schedule asynchronous initialization of components that require it
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

    def _initialize_components(self): # Renamed from _validate_components_present
        """
        Instantiate core TUI components.
        """
        # Instantiate the actual PlateManagerPane from plate_manager_core.py
        self.plate_manager = ActualPlateManagerPane(
            state=self.state,
            context=self.context,
            storage_registry=self.storage_registry, # Pass the shared registry
        )

        # self.step_viewer is initialized asynchronously.

        # ActionMenuPane is no longer used and its instantiation is removed.
        # Functionality integrated into PlateManagerPane and PipelineEditorPane via Commands.

        # self.function_pattern_editor (old) is created on demand by _get_left_pane if needed.
        # self.dual_step_func_editor is also created on demand by _get_left_pane.

        # Instantiate the actual StatusBar from status_bar.py
        self.status_bar = ActualStatusBar(self.state)

        # Instantiate the actual MenuBar from menu_bar.py
        self.menu_bar = ActualMenuBar(self.state)

    def _create_root_container(self) -> Container:
        """
        Creates the root container for the TUI application with the new 3-horizontal-bar layout.
        This structure aligns with V4 Plan 1.2.
        """
        return HSplit([
            # Top Bar (MenuBar and Version Label)
            VSplit([
                self._get_menu_bar(),
                # Use a Window with flexible width to push the version label to the right
                Window(width=0, char=' '), # Flexible spacer
                Label("OpenHCS_V1.0", style="class:app-title", dont_extend_width=True)
            ], height=1),
            # 2nd Bar (Titles)
            VSplit([
                Frame(Label("1 Plate Manager"), style="class:pane-title"),
                Frame(Label("2 Pipeline Editor"), style="class:pane-title"),
            ], height=1),
            # 3rd Bar (Contextual Buttons)
            VSplit([
                DynamicContainer(
                    lambda: self.plate_manager.get_buttons_container()
                    if self.plate_manager and hasattr(self.plate_manager, 'get_buttons_container')
                    else Box(Label("Plate Buttons Loading..."), padding_left=1)
                ),
                DynamicContainer(
                    lambda: self.step_viewer.get_buttons_container()
                    if self.step_viewer and hasattr(self.step_viewer, 'get_buttons_container')
                    else Box(Label("Pipeline Buttons Loading..."), padding_left=1)
                ),
            ], height=1),
            # Main Panes (Plate Manager | Pipeline Editor) - This should be a VSplit
            VSplit([
                self._get_left_pane(),
                self._get_step_viewer(),
            ]), # Main content area takes remaining space
            # Bottom Bar (StatusBar)
            self._get_status_bar(), # This should have a fixed height, e.g., height=1
        ])

    def _get_left_pane(self) -> Container:
        """
        Get the current left pane based on state.
        Dynamically shows PlateManagerPane or DualStepFuncEditorPane.
        """
        is_editing_step_config = getattr(self.state, 'editing_step_config', False)
        is_editing_plate_config = getattr(self.state, 'editing_plate_config', False)
        step_to_edit_config = getattr(self.state, 'step_to_edit_config', None)
        orchestrator_to_edit_config = getattr(self.state, 'orchestrator_for_plate_config_edit', None)

        if is_editing_plate_config:
            if orchestrator_to_edit_config is None:
                logger.warning("OpenHCSTUI: 'editing_plate_config' is true, but 'orchestrator_for_plate_config_edit' is not set.")
                self.state.editing_plate_config = False # Reset state
                return self.plate_manager.container if self.plate_manager else Frame(Label("Error")) # Fallback

            from openhcs.tui.dialogs.plate_config_editor import PlateConfigEditorPane # Import here

            # Instantiate or re-instantiate if orchestrator changed or editor was cleared
            if self.plate_config_editor is None or \
               getattr(self.plate_config_editor, 'orchestrator', None) != orchestrator_to_edit_config:
                logger.info(f"OpenHCSTUI: Instantiating PlateConfigEditorPane for orchestrator of plate: {getattr(orchestrator_to_edit_config, 'plate_id', 'N/A')}")
                try:
                    self.plate_config_editor = PlateConfigEditorPane(state=self.state, orchestrator=orchestrator_to_edit_config)
                except Exception as e:
                    logger.error(f"OpenHCSTUI: Failed to instantiate PlateConfigEditorPane: {e}", exc_info=True)
                    self.plate_config_editor = None # Ensure it's None on failure
                    self.state.editing_plate_config = False # Reset state on error
                    return Frame(Box(Label(f"Error creating Plate Config Editor: {e}")), title="Editor Error") # Show error

            if self.plate_config_editor and hasattr(self.plate_config_editor, 'container'):
                return self.plate_config_editor.container
            else: # Should not happen if instantiation was successful
                logger.error("OpenHCSTUI: PlateConfigEditorPane instance available but has no container, or failed instantiation.")
                self.state.editing_plate_config = False # Reset state
                return Frame(Box(Label("Error: Plate Config Editor could not be loaded.")), title="Editor Error") # Fallback

        elif is_editing_step_config:
            if step_to_edit_config is None:
                logger.warning("OpenHCSTUI: 'editing_step_config' is true, but 'step_to_edit_config' is not set in TUIState.")
                return Frame(Box(Label("Error: No step selected for editing.")), title="Error")

            # Ensure step_to_edit_config is a FuncStep instance
            if not isinstance(step_to_edit_config, FunctionStep): # Requires FunctionStep import
                 # Attempt to convert if it's a dict, otherwise log error
                if isinstance(step_to_edit_config, dict):
                    try:
                        step_to_edit_config = FunctionStep(**step_to_edit_config)
                    except Exception as e:
                        logger.error(f"OpenHCSTUI: Failed to convert step_to_edit_config dict to FuncStep: {e}", exc_info=True)
                        return Frame(Box(Label(f"Error: Invalid step data for editor: {e}")), title="Editor Error")
                else:
                    logger.error(f"OpenHCSTUI: 'step_to_edit_config' is not a FuncStep or dict, but {type(step_to_edit_config)}.")
                    return Frame(Box(Label("Error: Invalid step data type for editor.")), title="Editor Error")


            if self.dual_step_func_editor is None or \
               self.dual_step_func_editor.original_func_step.id != step_to_edit_config.id: # Compare by ID or a unique attribute
                logger.info(f"OpenHCSTUI: Instantiating DualStepFuncEditorPane for step: {step_to_edit_config.name or step_to_edit_config.id}")
                try:
                    self.dual_step_func_editor = DualStepFuncEditorPane(state=self.state, func_step=step_to_edit_config)
                except Exception as e:
                    logger.error(f"OpenHCSTUI: Failed to instantiate DualStepFuncEditorPane: {e}", exc_info=True)
                    self.dual_step_func_editor = None
                    return Frame(Box(Label(f"Error creating Step/Func Editor: {e}")), title="Editor Error")

            if self.dual_step_func_editor and hasattr(self.dual_step_func_editor, 'container'):
                # The title is now handled within DualStepFuncEditorPane itself or by its Frame
                return self.dual_step_func_editor.container
            else:
                logger.error("OpenHCSTUI: DualStepFuncEditorPane instance available but has no container, or failed instantiation.")
                return Frame(Box(Label("Error: Step/Func Editor could not be loaded.")), title="Editor Error")
        else: # Neither step config nor plate config is being edited, show PlateManagerPane
            # Clear editor instances if they exist and we are switching away from them
            if self.dual_step_func_editor is not None:
                logger.debug("OpenHCSTUI: Clearing DualStepFuncEditorPane instance.")
                # await self.dual_step_func_editor.shutdown() # if it has one
                self.dual_step_func_editor = None
            if self.plate_config_editor is not None:
                logger.debug("OpenHCSTUI: Clearing PlateConfigEditorPane instance.")
                # await self.plate_config_editor.shutdown() # if it has one
                self.plate_config_editor = None

            if self.plate_manager and hasattr(self.plate_manager, 'container'):
                return self.plate_manager.container # PlateManagerPane now includes its own Frame
            else:
                logger.error("OpenHCSTUI: PlateManagerPane not available or has no container.")
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

    async def _handle_show_edit_plate_config_request(self, data: Dict[str, Any]):
        """Handles request to show the plate-specific configuration editor."""
        orchestrator = data.get('orchestrator')
        if orchestrator:
            self.state.orchestrator_for_plate_config_edit = orchestrator
            self.state.editing_plate_config = True
            self.state.editing_step_config = False # Ensure not also editing step
            logger.info(f"OpenHCSTUI: Switched to editing plate config for orchestrator: {getattr(orchestrator, 'plate_id', 'N/A')}")
            get_app().invalidate() # Trigger layout refresh
        else:
            logger.warning("OpenHCSTUI: _handle_show_edit_plate_config_request called without orchestrator data.")

    async def _handle_plate_config_editing_cancelled(self, data: Any = None):
        """Handles cancellation of plate config editing."""
        self.state.editing_plate_config = False
        self.state.orchestrator_for_plate_config_edit = None
        self.plate_config_editor = None # Clear the editor instance
        logger.info("OpenHCSTUI: Plate config editing cancelled/closed.")
        get_app().invalidate()

    async def _handle_plate_config_saved(self, data: Any = None):
        """Handles successful save of plate config."""
        # The actual save logic is in PlateConfigEditorPane.
        # This handler just resets the TUI state to switch views.
        self.state.editing_plate_config = False
        self.state.orchestrator_for_plate_config_edit = None
        self.plate_config_editor = None # Clear the editor instance
        logger.info("OpenHCSTUI: Plate config saved, closing editor.")
        get_app().invalidate()

    async def _handle_step_editing_cancelled(self, data: Any = None):
        """Handles cancellation of step editing from DualStepFuncEditorPane."""
        self.state.editing_step_config = False
        self.state.step_to_edit_config = None
        self.dual_step_func_editor = None # Clear the editor instance
        logger.info("OpenHCSTUI: Step editing cancelled/closed.")
        get_app().invalidate()


    async def shutdown_components(self):
        """
        Gracefully shut down all managed TUI components that have a shutdown method.
        """
        logger.info("OpenHCSTUI: Initiating shutdown sequence for components...")

        component_attributes = [
            'plate_manager',
            'step_viewer',
            'status_bar',
            'menu_bar',
            'function_pattern_editor', # Old editor, might be removed later
            'dual_step_func_editor', # New editor
            'plate_config_editor' # New editor
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