import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from prompt_toolkit.application import get_app
from prompt_toolkit.filters import has_focus
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, VSplit, DynamicContainer, Container, Dimension, Window
from prompt_toolkit.widgets import Button, Frame, Label, TextArea, Box, Dialog

from .components import FramedButton

logger = logging.getLogger(__name__)

# Define SafeButton locally to avoid circular imports
class SafeButton(Button):
    """Safe wrapper around Button that handles formatting errors."""

    def __init__(self, text="", handler=None, width=None, **kwargs):
        # Sanitize text before passing to parent
        if text is not None:
            text = str(text).replace('{', '{{').replace('}', '}}').replace(':', ' ')
        super().__init__(text=text, handler=handler, width=width, **kwargs)

    def _get_text_fragments(self):
        """Safe version that handles formatting errors gracefully."""
        try:
            return super()._get_text_fragments()
        except (ValueError, TypeError, AttributeError):
            # Fallback to simple text formatting without centering
            text = str(self.text) if self.text is not None else ""
            safe_text = text.replace('{', '{{').replace('}', '}}')
            return [("class:button", f" {safe_text} ")]
import pickle # Ensure pickle is imported if not already (it was added for save)
from pathlib import Path # Ensure Path is imported

from .components import InteractiveListItem
from .utils import show_error_dialog, prompt_for_path_dialog # Ensure these are imported

from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.steps.function_step import FunctionStep
from openhcs.core.steps.abstract import AbstractStep # Import AbstractStep for type checking
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator


class PipelineEditorPane: # Renamed from StepViewerPane
    """
    Right pane for editing pipelines (sequences of steps) in the OpenHCS TUI.

    This pane displays a list of FuncStep declarations for the selected plate's
    active pipeline. It supports keyboard navigation, selection, and reordering.
    """
    def __init__(self, state, context: ProcessingContext):
        """
        Initialize the Pipeline Editor pane.

        🔒 Clause 12: Explicit Error Handling
        Constructor must be synchronous in Python.

        Args:
            state: The TUI state manager
            context: The OpenHCS ProcessingContext
        """
        self.state = state
        self.context = context
        self.steps: List[Dict[str, Any]] = []
        self.pipelines: List[Dict[str, Any]] = [] # This might be simplified if only one active pipeline is shown
        self.selected_index = 0
        self.steps_lock = asyncio.Lock()  # Thread safety for step operations

        # Placeholder for UI components that will be created in setup()
        self.step_items_container_widget: Optional[HSplit] = None # Will hold HSplit of InteractiveListItems
        self.edit_button: Optional[FramedButton] = None
        self.add_button: Optional[FramedButton] = None
        self.remove_button: Optional[FramedButton] = None
        self.load_button: Optional[FramedButton] = None
        self.save_button: Optional[FramedButton] = None
        self.kb: Optional[KeyBindings] = None
        self._dynamic_step_list_wrapper: Optional[DynamicContainer] = None
        self._container: Optional[Frame] = None # This will be the Frame around step_items_container_widget

        # Flag to track initialization state
        self._ui_initialized = False

    @classmethod
    async def create(cls, state, context: ProcessingContext):
        """
        Factory method to create and initialize a PipelineEditorPane asynchronously.

        🔒 Clause 12: Explicit Error Handling
        Use factory pattern for async initialization.

        Args:
            state: The TUI state manager
            context: The OpenHCS ProcessingContext

        Returns:
            Initialized PipelineEditorPane instance
        """
        # Create instance with synchronous constructor
        instance = cls(state, context)

        # Initialize UI components asynchronously
        await instance.setup()

        return instance

    async def setup(self):
        """
        Set up UI components asynchronously.

        This method must be called after instantiation to initialize
        UI components that require async operations.
        """
        if self._ui_initialized:
            logger.info("PipelineEditorPane: UI already initialized, skipping initialization.")
            return

        if not hasattr(self, 'context') or self.context is None:
            logger.error("PipelineEditorPane: ProcessingContext not available for command execution.")
            # Fallback or disable buttons if context is critical and missing
            # For now, assuming context is initialized.

        # Create UI components asynchronously - handlers now use Commands
        # Import commands here to avoid circular imports
        try:
            from .commands import (
                AddStepCommand, DeleteSelectedStepsCommand, LoadPipelineCommand,
                SavePipelineCommand, ShowEditStepDialogCommand
            )
        except ImportError as e:
            logger.error(f"PipelineEditorPane: Failed to import commands: {e}")
            # Create placeholder commands
            class PlaceholderCommand:
                async def execute(self, *args, **kwargs):
                    pass
            AddStepCommand = PlaceholderCommand
            DeleteSelectedStepsCommand = PlaceholderCommand
            LoadPipelineCommand = PlaceholderCommand
            SavePipelineCommand = PlaceholderCommand
            ShowEditStepDialogCommand = PlaceholderCommand

        self.add_button = FramedButton("Add", handler=lambda: get_app().create_background_task(
            AddStepCommand().execute(self.state, self.context)
        ), width=6)
        self.remove_button = FramedButton("Del", handler=lambda: get_app().create_background_task(
            DeleteSelectedStepsCommand().execute(self.state, self.context, steps_to_delete=self._get_selected_steps_for_action())
        ), width=6)
        self.edit_button = FramedButton("Edit", handler=lambda: get_app().create_background_task(
            ShowEditStepDialogCommand().execute(self.state, self.context) # Relies on state.selected_step
        ), width=6)
        self.load_button = FramedButton("Load", handler=lambda: get_app().create_background_task(
            LoadPipelineCommand().execute(self.state, self.context) # Relies on state.active_orchestrator
        ), width=6)
        self.save_button = FramedButton("Save", handler=lambda: get_app().create_background_task(
            SavePipelineCommand().execute(self.state, self.context) # Relies on state.active_orchestrator
        ), width=6)

        # This will now build a container of interactive items
        try:
            self.step_items_container_widget = await self._build_step_items_container()
            logger.info("PipelineEditorPane: Successfully built step items container.")
        except Exception as e:
            logger.error(f"PipelineEditorPane: Error building step items container: {e}", exc_info=True)
            self.step_items_container_widget = HSplit([Label(f"Error building step list: {e}")])

        # Create key bindings
        self.kb = self._create_key_bindings()

        # The main container for this pane will be a Frame around the step list items container
        # We use a DynamicContainer to allow replacing the HSplit of items easily
        self.get_current_step_list_container = lambda: self.step_items_container_widget or HSplit([Label("Loading...")])
        self._dynamic_step_list_wrapper = DynamicContainer(self.get_current_step_list_container)
        self._container = Frame(self._dynamic_step_list_wrapper, title="Steps")

        # Register for events - use wrapper for async methods
        try:
            self.state.add_observer('plate_selected', lambda plate: get_app().create_background_task(self._on_plate_selected(plate)))
            self.state.add_observer('steps_updated', lambda _: get_app().create_background_task(self._refresh_steps()))
            self.state.add_observer('step_pattern_saved', lambda data: get_app().create_background_task(self._handle_step_pattern_saved(data)))
            # Add observer for when ShowEditStepDialogCommand requests to edit a step
            self.state.add_observer('edit_step_dialog_requested', lambda data: get_app().create_background_task(self._handle_edit_step_request(data)))
            logger.info("PipelineEditorPane: Successfully registered observers.")
        except Exception as e:
            logger.error(f"PipelineEditorPane: Error registering observers: {e}", exc_info=True)

        # Mark as initialized
        self._ui_initialized = True
        logger.info("PipelineEditorPane: UI initialization complete.")

        # Invalidate the application to refresh the UI
        get_app().invalidate()

    def _get_selected_steps_for_action(self) -> List[Dict[str, Any]]:
        """Helper to get data of the currently selected step(s) for commands."""
        if self.steps and 0 <= self.selected_index < len(self.steps):
            return [self.steps[self.selected_index]]
        return []

    @property
    def container(self) -> Container:
        """Return the main container for the PipelineEditorPane (the step list)."""
        return self._container

    def get_buttons_container(self) -> Container:
        """Returns a container with the Pipeline Editor's action buttons."""
        # Ensure buttons are initialized before creating the container
        if not all([self.add_button, self.remove_button, self.edit_button, self.load_button, self.save_button]):
             # This can happen if get_buttons_container is called before setup completes
             return VSplit([Label("Pipeline Buttons not ready.")])

        # Return a simple VSplit with all buttons in a row
        # Each button is already framed by FramedButton
        return VSplit([
            self.add_button,
            Window(width=1, char=' '),  # Small spacer
            self.remove_button,
            Window(width=1, char=' '),  # Small spacer
            self.edit_button,
            Window(width=1, char=' '),  # Small spacer
            self.load_button,
            Window(width=1, char=' '),  # Small spacer
            self.save_button,
        ], height=1)

    async def _build_step_items_container(self) -> HSplit:
        """
        Builds the HSplit container holding individual InteractiveStepItem widgets.
        """
        async with self.steps_lock:
            if not self.steps:
                return self._create_empty_steps_container()

            item_widgets = []
            for i, step_data in enumerate(self.steps):
                item_widget = self._create_step_item_widget(i, step_data)
                item_widgets.append(item_widget)

            return self._create_steps_container(item_widgets)

    def _create_empty_steps_container(self) -> HSplit:
        """Create container for when no steps are available."""
        return HSplit([Label("No steps available. Select a plate first.")],
                     width=Dimension(weight=1), height=Dimension(weight=1))

    def _create_step_item_widget(self, index: int, step_data: Dict[str, Any]) -> InteractiveListItem:
        """Create a widget for a single step item."""
        is_selected = (index == self.selected_index)
        can_move_up = self._can_move_step_up(index, step_data)
        can_move_down = self._can_move_step_down(index, step_data)

        return InteractiveListItem(
            item_data=step_data,
            item_index=index,
            is_selected=is_selected,
            display_text_func=self._get_step_display_text,
            on_select=self._handle_item_select,
            on_move_up=self._handle_item_move_up,
            on_move_down=self._handle_item_move_down,
            can_move_up=can_move_up,
            can_move_down=can_move_down
        )

    def _can_move_step_up(self, index: int, step_data: Dict[str, Any]) -> bool:
        """Check if step can be moved up."""
        if index <= 0:
            return False

        current_pipeline_id = step_data.get('pipeline_id')
        prev_pipeline_id = self.steps[index - 1].get('pipeline_id')
        return current_pipeline_id == prev_pipeline_id

    def _can_move_step_down(self, index: int, step_data: Dict[str, Any]) -> bool:
        """Check if step can be moved down."""
        if index >= len(self.steps) - 1:
            return False

        current_pipeline_id = step_data.get('pipeline_id')
        next_pipeline_id = self.steps[index + 1].get('pipeline_id')
        return current_pipeline_id == next_pipeline_id

    def _create_steps_container(self, item_widgets: List[InteractiveListItem]) -> HSplit:
        """Create the final steps container."""
        widgets = item_widgets if item_widgets else [Label("No steps to display.")]
        return HSplit(widgets, width=Dimension(weight=1), height=Dimension(weight=1))

    def _get_step_display_text(self, step_data: Dict[str, Any], is_selected: bool) -> str:
        """
        Generates the display text for a single step item.
        Format: symbol | name | func_name -> output_memory_type
        The InteractiveListItem is assumed to handle the '^/v index:' part.
        """
        status_icon = self._get_status_icon(step_data.get('status', 'unknown'))
        name = step_data.get('name', 'Unknown Step')
        func_name = self._get_function_name(step_data) # This already returns a string
        output_memory_type = step_data.get('output_memory_type', '[N/A]')

        # Ensure func_name is just the name, not complex structure for this display line
        if not isinstance(func_name, str): # Should be string from _get_function_name
            func_name = str(func_name)

        # Format to match the image
        return f"{status_icon} {name} | {func_name} → {output_memory_type}"

    # _format_step_list is obsolete.

    # --- New callback handlers for InteractiveListItem ---
    async def _handle_item_select(self, index: int):
        """Handles selection of an item from the list."""
        if 0 <= index < len(self.steps):
            self.selected_index = index
            await self._update_selection() # Rebuilds list to reflect selection, calls _select_step
            # Optionally, directly trigger edit on select, or rely on 'Enter' keybinding
            # For now, selection updates the UI, 'Enter' or 'Edit' button triggers editing.
            # await self._edit_step() # If direct edit on click is desired

    async def _handle_item_move_up(self, index: int):
        """Handles 'move up' button click for an item."""
        if 0 <= index < len(self.steps):
            self.selected_index = index # Ensure the correct item is targeted by _move_step_up
            await self._move_step_up()

    async def _handle_item_move_down(self, index: int):
        """Handles 'move down' button click for an item."""
        if 0 <= index < len(self.steps):
            self.selected_index = index # Ensure the correct item is targeted by _move_step_down
            await self._move_step_down()
    # --- End of new callback handlers ---

    def _get_status_icon(self, status: str) -> str:
        """
        Get the status icon for a step.

        Args:
            status: The step status

        Returns:
            A status icon character
        """
        # Canonical status symbols from tui_final.md
        icons = {
            'pending': "?",           # Uninitialized
            'validated': "o",         # Compiled/ready
            'error': "!",            # Error (same as initialized for now)
            'not_initialized': "?",   # Red - uninitialized
            'initialized': "!",       # Yellow - initialized but not compiled
            'ready': "!",            # Yellow - initialized but not compiled
            'compiled_ok': "o",      # Green - compiled/ready
            'compiled': "o",         # Green - compiled/ready
            'running': "o"           # Green - running
        }
        if status not in icons:
            # Instead of raising an error, return a default icon
            return "o"
        return icons[status]

    def _get_function_name(self, step: Dict[str, Any]) -> str:
        """
        Get a display name for the function pattern.

        Args:
            step: The step dictionary

        Returns:
            A display name for the function pattern
        """
        # 🔒 Clause 88: No Inferred Capabilities
        # 🔒 Clause 92: Structural Validation First
        # Safe access to func field with fallback for rendering
        if 'func' not in step:
            return '[MISSING FUNC]'

        func = step['func']
        if func is None:
            return '[NULL FUNC]'

        # Handle different function pattern types
        if callable(func):
            return func.__name__
        elif isinstance(func, tuple) and len(func) == 2 and callable(func[0]):
            return f"{func[0].__name__}(...)"
        elif isinstance(func, list):
            return f"[{len(func)} functions]"
        elif isinstance(func, dict):
            return f"{{{len(func)} components}}"
        else:
            return str(func)

    # Old _add_step, _remove_step, _edit_step, _load_pipeline, _save_pipeline methods
    # are removed as their core logic is moved to Command classes.
    # _edit_step is replaced by _handle_edit_step_request.

    async def _handle_edit_step_request(self, data: Optional[Dict[str, Any]]) -> None:
        """
        Handles the 'edit_step_dialog_requested' event from ShowEditStepDialogCommand.
        Prepares TUIState for showing the DualStepFuncEditorPane.
        """
        if not self._validate_edit_request_data(data):
            return

        selected_step_data = data.get('step_data')
        active_orchestrator = getattr(self.state, 'active_orchestrator', None)

        if not self._validate_edit_prerequisites(active_orchestrator, selected_step_data):
            return

        actual_step_instance = self._find_step_instance(selected_step_data, active_orchestrator)
        if not actual_step_instance:
            return

        if not self._validate_step_type(actual_step_instance):
            return

        await self._activate_step_editor(actual_step_instance)

    def _validate_edit_request_data(self, data: Optional[Dict[str, Any]]) -> bool:
        """Validate the edit request data."""
        if data is None:
            logger.warning("PipelineEditorPane: _handle_edit_step_request received no data.")
            return False
        return True

    async def _validate_edit_prerequisites(self, active_orchestrator, selected_step_data) -> bool:
        """Validate prerequisites for editing a step."""
        if not active_orchestrator or not selected_step_data:
            await show_error_dialog("Error", "No step selected or no active pipeline to edit a step from.", app_state=self.state)
            return False
        return True

    def _find_step_instance(self, selected_step_data: Dict[str, Any], active_orchestrator) -> Optional[AbstractStep]:
        """Find the actual step instance in the pipeline."""
        step_id_to_edit = selected_step_data.get('id')

        if not step_id_to_edit or not active_orchestrator.pipeline_definition:
            return None

        for step_in_pipeline in active_orchestrator.pipeline_definition:
            if (isinstance(step_in_pipeline, AbstractStep) and
                step_in_pipeline.id == step_id_to_edit):
                return step_in_pipeline

        logger.error(f"PipelineEditorPane: Could not find actual step instance for ID: {step_id_to_edit}")
        return None

    async def _validate_step_type(self, step_instance: AbstractStep) -> bool:
        """Validate that the step is a FunctionStep."""
        if not isinstance(step_instance, FunctionStep):
            await show_error_dialog("Info", "Selected step is not a FunctionStep and cannot be edited with this editor.", app_state=self.state)
            return False
        return True

    async def _activate_step_editor(self, step_instance: FunctionStep) -> None:
        """Activate the step editor for the given step."""
        logger.info(f"PipelineEditorPane: Requesting edit for step: {step_instance.name or step_instance.id}")
        self.state.step_to_edit_config = step_instance
        self.state.editing_step_config = True
        await self.state.notify('editing_step_config_changed', True)

    def _create_key_bindings(self) -> KeyBindings:
        """
        Create key bindings for the step list.

        Returns:
            KeyBindings object
        """
        kb = KeyBindings()

        # The filter for has_focus will need to target the new container or individual items
        # For now, let's assume the Frame around step_items_container_widget can get focus.
        list_container_focused = has_focus(self._container) if self._container else has_focus(self.step_items_container_widget)


        @kb.add('up', filter=list_container_focused)
        def _(event):
            """Move selection up."""
            if self.steps:
                self.selected_index = max(0, self.selected_index - 1)
                get_app().create_background_task(self._update_selection())

        @kb.add('down', filter=list_container_focused)
        def _(event):
            """Move selection down."""
            if self.steps:
                self.selected_index = min(len(self.steps) - 1, self.selected_index + 1)
                get_app().create_background_task(self._update_selection())

        @kb.add('enter', filter=list_container_focused)
        def _(event):
            """Edit the current step."""
            if self.steps and 0 <= self.selected_index < len(self.steps):
                # If on_select is implemented in InteractiveListItem to call _edit_step,
                # this might be redundant or could directly call _edit_step.
                # For now, assume _edit_step is the primary action for 'enter'.
                get_app().create_background_task(self._edit_step())

        @kb.add('s-up', filter=list_container_focused)
        def _(event):
            """Move the selected step up."""
            get_app().create_background_task(self._move_step_up())

        @kb.add('s-down', filter=list_container_focused)
        def _(event):
            """Move the selected step down."""
            get_app().create_background_task(self._move_step_down())

        return kb

    async def _update_selection(self):
        """Rebuilds the step items container to reflect current selection and data."""
        if not self._ui_initialized:
            return

        # Rebuild the HSplit with new InteractiveListItem instances
        self.step_items_container_widget = await self._build_step_items_container()
        # The DynamicContainer (self._dynamic_step_list_wrapper) will pick up the change
        # to self.step_items_container_widget when get_app().invalidate() is called.

        if self.steps and 0 <= self.selected_index < len(self.steps):
            # Notify about selection, which might trigger other UI updates (e.g. details pane).
            # This is important if other parts of the UI depend on the selected step.
            get_app().create_background_task(self._select_step(self.selected_index)) # Ensure this is called

        get_app().invalidate() # Crucial to trigger a redraw of the DynamicContainer

    async def _select_step(self, index: int):
        """
        Select a step and emit selection event.

        Args:
            index: The index of the step to select
        """
        async with self.steps_lock:
            if 0 <= index < len(self.steps):
                step_data = self.steps[index]
                self.state.set_selected_step(step_data)



    async def _save_pipeline(self):
        """Saves the current pipeline definition for the active plate."""
        if not await self._validate_save_prerequisites():
            return

        file_path_str = await self._prompt_for_save_path()
        if not file_path_str:
            await self._notify_save_cancelled()
            return

        file_path = Path(file_path_str)
        await self._perform_pipeline_save(file_path)

    async def _validate_save_prerequisites(self) -> bool:
        """Validate prerequisites for saving pipeline."""
        if not self.state.active_orchestrator:
            await self.state.notify('error', {
                'message': "No active plate selected to save pipeline for.",
                'source': self.__class__.__name__
            })
            return False

        if not self.steps:
            await self.state.notify('error', {
                'message': "No steps in the current pipeline to save.",
                'source': self.__class__.__name__
            })
            return False

        return True

    async def _prompt_for_save_path(self) -> Optional[str]:
        """Prompt user for save path."""
        initial_filename = self._get_default_filename()

        return await prompt_for_path_dialog(
            title="Save Pipeline As",
            prompt_message="Enter path to save .pipeline file:",
            app_state=self.state,
            initial_value=initial_filename
        )

    def _get_default_filename(self) -> str:
        """Get default filename for pipeline save."""
        if self.state.selected_plate:
            plate_name = self.state.selected_plate.get('name', 'default_pipeline')
            return f"{plate_name}.pipeline"
        return "pipeline.pipeline"

    async def _notify_save_cancelled(self):
        """Notify that save operation was cancelled."""
        await self.state.notify('info', {
            'message': "Save pipeline operation cancelled.",
            'source': self.__class__.__name__
        })

    async def _perform_pipeline_save(self, file_path: Path):
        """Perform the actual pipeline save operation."""
        try:
            pipeline_to_save = self.state.active_orchestrator.pipeline_definition

            if not self._validate_pipeline_for_save(pipeline_to_save):
                return

            await self._save_pipeline_to_file(pipeline_to_save, file_path)
            await self._notify_save_success(file_path)

        except pickle.PicklingError as e:
            await self._handle_pickle_error(e, file_path)
        except Exception as e:
            await self._handle_save_error(e, file_path)

    def _validate_pipeline_for_save(self, pipeline_to_save) -> bool:
        """Validate pipeline data before saving."""
        if not isinstance(pipeline_to_save, list):
            self.state.notify('error', {
                'message': "No valid pipeline definition to save.",
                'source': self.__class__.__name__
            })
            return False
        return True

    async def _save_pipeline_to_file(self, pipeline_to_save: list, file_path: Path):
        """Save pipeline data to file."""
        with open(file_path, "wb") as f:
            pickle.dump(pipeline_to_save, f)

    async def _notify_save_success(self, file_path: Path):
        """Notify successful save."""
        await self.state.notify('operation_status_changed', {
            'message': f"Pipeline saved to {file_path}",
            'status': 'success',
            'source': self.__class__.__name__
        })

    async def _handle_pickle_error(self, error: pickle.PicklingError, file_path: Path):
        """Handle pickle errors during save."""
        logger.error(f"Error pickling pipeline to {file_path}: {error}", exc_info=True)
        await show_error_dialog("Save Pipeline Error", f"Error pickling pipeline: {error}", app_state=self.state)

    async def _handle_save_error(self, error: Exception, file_path: Path):
        """Handle general save errors."""
        logger.error(f"Failed to save pipeline to {file_path}: {error}", exc_info=True)
        await show_error_dialog("Save Pipeline Error", f"Failed to save pipeline: {error}", app_state=self.state)

    async def _handle_step_pattern_saved(self, data: Dict[str, Any]):
        """Handles the 'step_pattern_saved' event from DualStepFuncEditorPane."""
        saved_step = data.get('step')
        if not saved_step:
            return

        async with self.steps_lock:
            for i, step in enumerate(self.steps):
                if step.get('id') == saved_step.get('id'):
                    self.steps[i] = saved_step
                    break
            else: # If loop didn't break, step not found (should not happen if editing existing)
                self.steps.append(saved_step) # Or handle as an error

            # Refresh display and notify
            await self._update_selection() # This will re-render the list
            self.state.notify('steps_updated', {'steps': self.steps})


    async def _on_plate_selected(self, plate):
        """
        Handle plate selection event.

        Args:
            plate: The selected plate
        """
        if plate:
            # In a real implementation, this would query the ProcessingContext
            # for the steps associated with the plate
            # For now, we'll use placeholder data
            await self._load_steps_for_plate(plate['id'])

    async def _load_steps_for_plate(self, plate_id: str):
        """
        Load steps for the specified plate.

        Args:
            plate_id: The ID of the plate
        """
        self._initialize_pipeline_data(plate_id)

        async with self.steps_lock:
            raw_step_objects = self._get_orchestrator_steps()

            if raw_step_objects:
                await self._load_from_orchestrator(raw_step_objects)
            else:
                await self._load_from_context_fallback(plate_id)

    def _initialize_pipeline_data(self, plate_id: str):
        """Initialize pipeline data for the plate."""
        self.pipelines = self.context.list_pipelines_for_plate(plate_id)
        # 🔒 Clause 24: Performance Optimization
        # Build pipeline lookup dict for O(1) access instead of O(N) search
        self.pipeline_lookup = {p['id']: p for p in self.pipelines}

    def _get_orchestrator_steps(self) -> List[Any]:
        """Get step objects from the active orchestrator."""
        if (hasattr(self.state, 'active_orchestrator') and
            self.state.active_orchestrator and
            hasattr(self.state.active_orchestrator, 'pipeline_definition') and
            self.state.active_orchestrator.pipeline_definition is not None):
            return self.state.active_orchestrator.pipeline_definition
        return []

    async def _load_from_orchestrator(self, raw_step_objects: List[Any]):
        """Load steps from orchestrator step objects."""
        transformed_steps = []

        for step_obj in raw_step_objects:
            if isinstance(step_obj, FunctionStep):
                step_dict = self._transform_step_object_to_dict(step_obj)
                transformed_steps.append(step_dict)

        self.steps = transformed_steps
        await self._finalize_step_loading()

    def _transform_step_object_to_dict(self, step_obj: FunctionStep) -> Dict[str, Any]:
        """Transform a FunctionStep object to a dictionary for display."""
        temp_func_dict = {'func': step_obj.func}
        func_display_name = self._get_function_name(temp_func_dict)

        return {
            'id': step_obj.step_id,
            'name': step_obj.name,
            'func': step_obj.func,
            'func_display_name': func_display_name,
            'status': 'pending',
            'pipeline_id': getattr(step_obj, 'pipeline_id', None)
        }

    async def _load_from_context_fallback(self, plate_id: str):
        """Load steps from context as fallback."""
        self.steps = self.context.list_steps_for_plate(plate_id)
        self._validate_context_steps()
        await self._finalize_step_loading()

    def _validate_context_steps(self):
        """Validate steps loaded from context."""
        invalid_steps = []
        required_fields = ['func', 'output_memory_type', 'name', 'status', 'id']

        for i, step_dict in enumerate(self.steps):
            step_id = step_dict.get('id', f'ctx index {i}')

            missing_fields = [field for field in required_fields if field not in step_dict]
            if missing_fields:
                invalid_steps.append((step_id, missing_fields))

            if 'func' in step_dict and step_dict['func'] is None:
                invalid_steps.append((step_id, ['func is None']))

        if invalid_steps:
            # TODO: Handle/report errors for steps from context
            pass

    async def _finalize_step_loading(self):
        """Finalize the step loading process."""
        self.selected_index = 0
        get_app().create_background_task(self._update_selection())

    async def _edit_step(self):
        """
        Edit the currently selected step.
        This will trigger showing the DualStepFuncEditorPane in the left pane.
        """
        async with self.steps_lock:
            if not self.steps or self.selected_index >= len(self.steps):
                return

            # Get the selected step
            step = self.steps[self.selected_index]

            # Set the state to trigger showing the DualStepFuncEditorPane
            self.state.step_to_edit_config = step
            self.state.editing_step_config = True

            # Notify observers of the change
            await self.state.notify('edit_step_dialog_requested', step)

    async def _refresh_steps(self, _=None):
        """
        Refresh the step list, typically after the underlying pipeline definition changes.
        It re-loads the steps for the currently selected plate.
        """
        if hasattr(self.state, 'selected_plate') and self.state.selected_plate and 'id' in self.state.selected_plate:
            # Re-load steps from the orchestrator/context for the current plate
            await self._load_steps_for_plate(self.state.selected_plate['id'])
        else:
            # If no plate is selected, or plate_id is missing, clear steps and refresh UI
            async with self.steps_lock:
                self.steps = []
                self.selected_index = 0
            await self._update_selection() # This will show "No steps available" or similar

    async def _move_step_up(self):
        """Move the selected step up in the pipeline."""
        await self._move_step(-1)

    async def _move_step_down(self):
        """Move the selected step down in the pipeline."""
        await self._move_step(1)

    async def _move_step(self, direction: int):
        """
        Move the selected step in the specified direction.

        Args:
            direction: -1 for up, 1 for down
        """
        if not self._can_move_step(direction):
            return

        async with self.steps_lock:
            if not self._validate_pipeline_state():
                return

            pipeline = self.state.active_orchestrator.pipeline_definition
            target_index = self.selected_index + direction

            # Get step data and validate
            current_step_dict = self.steps[self.selected_index]
            target_step_dict = self.steps[target_index]

            step_ids = self._extract_step_ids(current_step_dict, target_step_dict)
            if not step_ids:
                return

            # Find orchestrator indices
            orchestrator_indices = self._find_orchestrator_indices(pipeline, step_ids)
            if not self._validate_orchestrator_indices(orchestrator_indices, direction):
                await self._refresh_steps()
                return

            # Perform the swap
            self._swap_steps_in_pipeline(pipeline, orchestrator_indices)
            self.selected_index = target_index
            await self.state.notify('steps_updated', {'action': 'reorder'})

    def _can_move_step(self, direction: int) -> bool:
        """Check if step can be moved in the specified direction."""
        if not self.steps:
            return False

        if direction == -1:  # Moving up
            return self.selected_index > 0
        else:  # Moving down
            return self.selected_index < len(self.steps) - 1

    def _validate_pipeline_state(self) -> bool:
        """Validate that pipeline state is ready for step movement."""
        if not (hasattr(self.state, 'active_orchestrator') and
                self.state.active_orchestrator and
                self.state.active_orchestrator.pipeline_definition):
            self.state.notify('error', {
                'message': "No active pipeline to reorder steps in.",
                'source': 'PipelineEditorPane'
            })
            return False
        return True

    def _extract_step_ids(self, current_step: Dict[str, Any], target_step: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        """Extract and validate step IDs."""
        current_id = current_step.get('id')
        target_id = target_step.get('id')

        if not current_id or not target_id:
            self.state.notify('error', {
                'message': "Selected steps for reorder are missing IDs.",
                'source': 'PipelineEditorPane'
            })
            return None

        return current_id, target_id

    def _find_orchestrator_indices(self, pipeline: List[FunctionStep], step_ids: Tuple[str, str]) -> Tuple[int, int]:
        """Find indices of steps in the orchestrator pipeline."""
        current_id, target_id = step_ids
        current_idx = target_idx = -1

        for i, step_obj in enumerate(pipeline):
            step_id = getattr(step_obj, 'step_id', None)
            if step_id == current_id:
                current_idx = i
            elif step_id == target_id:
                target_idx = i

        return current_idx, target_idx

    def _validate_orchestrator_indices(self, indices: Tuple[int, int], direction: int) -> bool:
        """Validate that orchestrator indices are adjacent and valid."""
        current_idx, target_idx = indices

        if current_idx == -1 or target_idx == -1:
            self.state.notify('error', {
                'message': "Could not find adjacent steps in orchestrator pipeline for reorder.",
                'source': 'PipelineEditorPane'
            })
            return False

        # Check adjacency based on direction
        expected_target = current_idx + direction
        if target_idx != expected_target:
            self.state.notify('error', {
                'message': "Could not find adjacent steps in orchestrator pipeline for reorder.",
                'source': 'PipelineEditorPane'
            })
            return False

        return True

    def _swap_steps_in_pipeline(self, pipeline: List[FunctionStep], indices: Tuple[int, int]) -> None:
        """Swap steps in the pipeline."""
        current_idx, target_idx = indices
        pipeline[current_idx], pipeline[target_idx] = pipeline[target_idx], pipeline[current_idx]

    async def shutdown(self):
        """
        Explicit cleanup method for deterministic resource release.
        Unregisters observers from TUIState.
        """
        logger.info("StepViewerPane: Shutting down...")
        # Unregister observers
        self.state.remove_observer('plate_selected', self._on_plate_selected)
        self.state.remove_observer('steps_updated', self._refresh_steps)
        logger.info("StepViewerPane: Observers unregistered.")
