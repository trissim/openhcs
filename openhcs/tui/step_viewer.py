import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional

from prompt_toolkit.application import get_app
from prompt_toolkit.filters import has_focus
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, VSplit, DynamicContainer, Container, Dimension
from prompt_toolkit.widgets import Button, Frame, Label, TextArea, Box, Dialog

logger = logging.getLogger(__name__)
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
        self.edit_button: Optional[Button] = None
        self.add_button: Optional[Button] = None
        self.remove_button: Optional[Button] = None
        self.load_button: Optional[Button] = None
        self.save_button: Optional[Button] = None
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
            return

        if not hasattr(self, 'context') or self.context is None:
            logger.error("PipelineEditorPane: ProcessingContext not available for command execution.")
            # Fallback or disable buttons if context is critical and missing
            # For now, assuming context is initialized.

        # Create UI components asynchronously - handlers now use Commands
        # Import commands here to avoid circular imports
        from .commands import (
            AddStepCommand, DeleteSelectedStepsCommand, LoadPipelineCommand,
            SavePipelineCommand, ShowEditStepDialogCommand
        )

        self.add_button = Button("Add", handler=lambda: get_app().create_background_task(
            AddStepCommand().execute(self.state, self.context)
        ))
        self.remove_button = Button("Del", handler=lambda: get_app().create_background_task(
            DeleteSelectedStepsCommand().execute(self.state, self.context, steps_to_delete=self._get_selected_steps_for_action())
        ))
        self.edit_button = Button("Edit", handler=lambda: get_app().create_background_task(
            ShowEditStepDialogCommand().execute(self.state, self.context) # Relies on state.selected_step
        ))
        self.load_button = Button("Load", handler=lambda: get_app().create_background_task(
            LoadPipelineCommand().execute(self.state, self.context) # Relies on state.active_orchestrator
        ))
        self.save_button = Button("Save", handler=lambda: get_app().create_background_task(
            SavePipelineCommand().execute(self.state, self.context) # Relies on state.active_orchestrator
        ))

        # This will now build a container of interactive items
        self.step_items_container_widget = await self._build_step_items_container()

        # Create key bindings
        self.kb = self._create_key_bindings()

        # The main container for this pane will be a Frame around the step list items container
        # We use a DynamicContainer to allow replacing the HSplit of items easily
        self.get_current_step_list_container = lambda: self.step_items_container_widget or HSplit([Label("Loading...")])
        self._dynamic_step_list_wrapper = DynamicContainer(self.get_current_step_list_container)
        self._container = Frame(self._dynamic_step_list_wrapper, title="Steps")


        # Register for events - use wrapper for async methods
        self.state.add_observer('plate_selected', lambda plate: get_app().create_background_task(self._on_plate_selected(plate)))
        self.state.add_observer('steps_updated', lambda _: get_app().create_background_task(self._refresh_steps()))
        self.state.add_observer('step_pattern_saved', lambda data: get_app().create_background_task(self._handle_step_pattern_saved(data)))
        # Add observer for when ShowEditStepDialogCommand requests to edit a step
        self.state.add_observer('edit_step_dialog_requested', lambda data: get_app().create_background_task(self._handle_edit_step_request(data)))

        # Mark as initialized
        self._ui_initialized = True

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

        return VSplit([ # Use VSplit for horizontal button layout
            Box(self.add_button, padding_left=1, padding_right=1),
            Box(self.remove_button, padding_right=1),
            Box(self.edit_button, padding_right=1),
            Box(self.load_button, padding_right=1),
            Box(self.save_button, padding_right=1),
        ])

    async def _build_step_items_container(self) -> HSplit:
        """
        Builds the HSplit container holding individual InteractiveStepItem widgets.
        """
        item_widgets = []
        async with self.steps_lock:
            if not self.steps:
                item_widgets.append(Label("No steps available. Select a plate first."))
            else:
                for i, step_data in enumerate(self.steps):
                    is_selected = (i == self.selected_index)

                    can_move_up = False
                    if i > 0:
                        current_step_pipeline_id = step_data.get('pipeline_id')
                        prev_step_pipeline_id = self.steps[i - 1].get('pipeline_id')
                        if current_step_pipeline_id == prev_step_pipeline_id:
                            can_move_up = True

                    can_move_down = False
                    if i < len(self.steps) - 1:
                        current_step_pipeline_id = step_data.get('pipeline_id')
                        next_step_pipeline_id = self.steps[i + 1].get('pipeline_id')
                        if current_step_pipeline_id == next_step_pipeline_id:
                            can_move_down = True

                    item_widget = InteractiveListItem(
                        item_data=step_data,
                        item_index=i,
                        is_selected=is_selected,
                        display_text_func=self._get_step_display_text,
                        on_select=self._handle_item_select,
                        on_move_up=self._handle_item_move_up,
                        on_move_down=self._handle_item_move_down,
                        can_move_up=can_move_up,
                        can_move_down=can_move_down
                    )
                    item_widgets.append(item_widget)
        return HSplit(item_widgets if item_widgets else [Label("No steps to display.")], width=Dimension(weight=1), height=Dimension(weight=1)) # Ensure it fills space

    def _get_step_display_text(self, step_data: Dict[str, Any], is_selected: bool) -> str:
        """
        Generates the display text for a single step item.
        Reordering symbols (^/v) should be handled by InteractiveListItem.
        """
        status_icon = self._get_status_icon(step_data.get('status', 'unknown'))
        name = step_data.get('name', 'Unknown Step')
        func_name = self._get_function_name(step_data)
        memory_type = step_data.get('output_memory_type', '[N/A]')

        # Format the display text to match the desired layout
        # The ^/v symbols and item index are handled by InteractiveListItem
        return f"{status_icon} {name}: {func_name} → {memory_type}"

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
        icons = {
            'pending': "o",
            'validated': "✓",
            'error': "!"
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
        if data is None: # Should not happen if command sends data
            logger.warning("PipelineEditorPane: _handle_edit_step_request received no data.")
            return

        selected_step_data = data.get('step_data') # This is the dict from self.steps
        active_orchestrator: Optional["PipelineOrchestrator"] = getattr(self.state, 'active_orchestrator', None)

        if not active_orchestrator or not selected_step_data:
            await show_error_dialog("Error", "No step selected or no active pipeline to edit a step from.", app_state=self.state)
            return

        step_id_to_edit = selected_step_data.get('id')
        actual_step_instance: Optional[AbstractStep] = None # AbstractStep for type hint

        if step_id_to_edit and active_orchestrator.pipeline_definition:
            for step_in_pipeline in active_orchestrator.pipeline_definition:
                if isinstance(step_in_pipeline, AbstractStep) and step_in_pipeline.id == step_id_to_edit:
                    actual_step_instance = step_in_pipeline
                    break

        if not actual_step_instance:
            logger.error(f"PipelineEditorPane: Could not find actual step instance for ID: {step_id_to_edit}")
            await show_error_dialog("Error", "Could not find step instance to edit.", app_state=self.state)
            return

        if not isinstance(actual_step_instance, FunctionStep): # Ensure it's a FunctionStep
            await show_error_dialog("Info", "Selected step is not a FunctionStep and cannot be edited with this editor.", app_state=self.state)
            return

        logger.info(f"PipelineEditorPane: Requesting edit for step: {actual_step_instance.name or actual_step_instance.id}")
        self.state.step_to_edit_config = actual_step_instance # Store the actual FunctionStep instance
        self.state.editing_step_config = True
        await self.state.notify('editing_step_config_changed', True) # Notify UI to switch views

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
        # This is a simplified version. A real implementation would use a file dialog.
        if not self.state.active_orchestrator:
            await self.state.notify('error', {'message': "No active plate selected to save pipeline for.", 'source': self.__class__.__name__})
            return

        if not self.steps:
            await self.state.notify('error', {'message': "No steps in the current pipeline to save.", 'source': self.__class__.__name__})
            return

        # Simulate asking for a path (in a real app, use a dialog)
        # path_to_save = await self.show_file_dialog_for_save()
        # if not path_to_save: return

        file_path_str = await prompt_for_path_dialog(
            title="Save Pipeline As",
            prompt_message="Enter path to save .pipeline file:",
            app_state=self.state,
            initial_value=f"{self.state.selected_plate.get('name', 'default_pipeline')}.pipeline" if self.state.selected_plate else "pipeline.pipeline"
        )

        if not file_path_str:
            await self.state.notify('info', {'message': "Save pipeline operation cancelled.", 'source': self.__class__.__name__})
            return

        file_path = Path(file_path_str)

        try:
            pipeline_to_save = self.state.active_orchestrator.pipeline_definition
            if not isinstance(pipeline_to_save, list): # Basic check
                await self.state.notify('error', {'message': "No valid pipeline definition to save.", 'source': self.__class__.__name__})
                return

            with open(file_path, "wb") as f:
                pickle.dump(pipeline_to_save, f)

            await self.state.notify('operation_status_changed', {'message': f"Pipeline saved to {file_path}", 'status': 'success', 'source': self.__class__.__name__})

        except pickle.PicklingError as e:
            logger.error(f"Error pickling pipeline to {file_path}: {e}", exc_info=True)
            await show_error_dialog("Save Pipeline Error", f"Error pickling pipeline: {e}", app_state=self.state)
        except Exception as e:
            logger.error(f"Failed to save pipeline to {file_path}: {e}", exc_info=True)
            await show_error_dialog("Save Pipeline Error", f"Failed to save pipeline: {e}", app_state=self.state)

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
        # Get pipelines and build lookup dict for O(1) access
        self.pipelines = self.context.list_pipelines_for_plate(plate_id)

        # 🔒 Clause 24: Performance Optimization
        # Build pipeline lookup dict for O(1) access instead of O(N) search
        self.pipeline_lookup = {p['id']: p for p in self.pipelines}

        async with self.steps_lock:
            raw_step_objects = []
            if hasattr(self.state, 'active_orchestrator') and \
               self.state.active_orchestrator and \
               hasattr(self.state.active_orchestrator, 'pipeline_definition') and \
               self.state.active_orchestrator.pipeline_definition is not None:
                raw_step_objects = self.state.active_orchestrator.pipeline_definition
            else:
                # Fallback or if orchestrator is not the primary source for this view's initial load
                # This part might need reconciliation with how ProcessingContext is populated
                # For now, if no orchestrator pipeline, assume context is the source of dicts
                self.steps = self.context.list_steps_for_plate(plate_id)
                # Perform validation on these dicts as before
                # (The existing validation block for dicts from context can remain here for this fallback)
                invalid_steps_from_context = []
                for i, step_dict in enumerate(self.steps):
                    required_fields = ['func', 'output_memory_type', 'name', 'status', 'id'] # Added 'id'
                    missing_fields = [field for field in required_fields if field not in step_dict]
                    if missing_fields:
                        invalid_steps_from_context.append((step_dict.get('id', f'ctx index {i}'), missing_fields))
                    if 'func' in step_dict and step_dict['func'] is None:
                         invalid_steps_from_context.append((step_dict.get('id', f'ctx index {i}'), ['func is None']))

                if invalid_steps_from_context:
                    # Handle/report errors for steps from context
                    pass # Placeholder for error reporting logic for context-sourced steps

                self.selected_index = 0
                get_app().create_background_task(self._update_selection())
                return # Exit if we used context data

            # If we have raw_step_objects from the orchestrator, transform them
            transformed_steps = []
            for step_obj in raw_step_objects:
                if not isinstance(step_obj, FunctionStep): # Assuming we primarily edit FunctionSteps here
                    # Handle or log other step types if necessary
                    continue

                # Basic representation for display
                temp_func_dict = {'func': step_obj.func}
                func_display_name = self._get_function_name(temp_func_dict)

                transformed_steps.append({
                    'id': step_obj.step_id,
                    'name': step_obj.name,
                    'func': step_obj.func, # Store the actual func pattern for editing
                    'func_display_name': func_display_name, # For _get_step_display_text
                    'status': 'pending', # Default status for display
                    'pipeline_id': getattr(step_obj, 'pipeline_id', None) # If steps are grouped by pipeline
                })

            self.steps = transformed_steps
            self.selected_index = 0
            get_app().create_background_task(self._update_selection())

    async def _edit_step(self):
        """Edit the selected step."""
        async with self.steps_lock:
            if self.steps and 0 <= self.selected_index < len(self.steps):
                step = self.steps[self.selected_index]
                self.state.notify('edit_step', step)

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
        """
        Move the selected step up in the pipeline.

        This reorders steps within the same pipeline but does not
        allow moving steps across pipeline boundaries.
        """
        if not self.steps or self.selected_index <= 0:
            return

        # Thread-safe step reordering
        async with self.steps_lock:
            if not hasattr(self.state, 'active_orchestrator') or \
               not self.state.active_orchestrator or \
               not self.state.active_orchestrator.pipeline_definition:
                self.state.notify('error', {'message': "No active pipeline to reorder steps in.", 'source': 'PipelineEditorPane'})
                return

            pipeline: List[FunctionStep] = self.state.active_orchestrator.pipeline_definition

            if not (0 < self.selected_index < len(self.steps)):
                 return

            current_step_dict = self.steps[self.selected_index]
            prev_step_dict = self.steps[self.selected_index - 1]

            current_step_id = current_step_dict.get('id')
            prev_step_id = prev_step_dict.get('id')

            if not current_step_id or not prev_step_id:
                self.state.notify('error', {'message': "Selected steps for reorder are missing IDs.", 'source': 'PipelineEditorPane'})
                return

            current_orchestrator_idx = -1
            prev_orchestrator_idx = -1

            for i, step_obj in enumerate(pipeline):
                if getattr(step_obj, 'step_id', None) == current_step_id:
                    current_orchestrator_idx = i
                if getattr(step_obj, 'step_id', None) == prev_step_id:
                    prev_orchestrator_idx = i

            if current_orchestrator_idx == -1 or prev_orchestrator_idx == -1 or \
               current_orchestrator_idx != prev_orchestrator_idx + 1 : # Ensure they are adjacent in orchestrator list
                self.state.notify('error', {'message': "Could not find adjacent steps in orchestrator pipeline for reorder.", 'source': 'PipelineEditorPane'})
                await self._refresh_steps()
                return

            pipeline[current_orchestrator_idx], pipeline[prev_orchestrator_idx] = \
                pipeline[prev_orchestrator_idx], pipeline[current_orchestrator_idx]

            self.selected_index -= 1
            await self.state.notify('steps_updated', {'action': 'reorder'})

    async def _move_step_down(self):
        """
        Move the selected step down in the pipeline.

        This reorders steps within the same pipeline but does not
        allow moving steps across pipeline boundaries.
        """
        if not self.steps or self.selected_index >= len(self.steps) - 1:
            return

        # Thread-safe step reordering
        async with self.steps_lock:
            if not hasattr(self.state, 'active_orchestrator') or \
               not self.state.active_orchestrator or \
               not self.state.active_orchestrator.pipeline_definition:
                self.state.notify('error', {'message': "No active pipeline to reorder steps in.", 'source': 'PipelineEditorPane'})
                return

            pipeline: List[FunctionStep] = self.state.active_orchestrator.pipeline_definition

            if not (0 <= self.selected_index < len(self.steps) - 1):
                return

            current_step_dict = self.steps[self.selected_index]
            next_step_dict = self.steps[self.selected_index + 1]

            current_step_id = current_step_dict.get('id')
            next_step_id = next_step_dict.get('id')

            if not current_step_id or not next_step_id:
                self.state.notify('error', {'message': "Selected steps for reorder are missing IDs.", 'source': 'PipelineEditorPane'})
                return

            current_orchestrator_idx = -1
            next_orchestrator_idx = -1

            for i, step_obj in enumerate(pipeline):
                if getattr(step_obj, 'step_id', None) == current_step_id:
                    current_orchestrator_idx = i
                if getattr(step_obj, 'step_id', None) == next_step_id:
                    next_orchestrator_idx = i

            if current_orchestrator_idx == -1 or next_orchestrator_idx == -1 or \
               next_orchestrator_idx != current_orchestrator_idx + 1: # Ensure they are adjacent
                self.state.notify('error', {'message': "Could not find adjacent steps in orchestrator pipeline for reorder.", 'source': 'PipelineEditorPane'})
                await self._refresh_steps()
                return

            pipeline[current_orchestrator_idx], pipeline[next_orchestrator_idx] = \
                pipeline[next_orchestrator_idx], pipeline[current_orchestrator_idx]

            self.selected_index += 1
            await self.state.notify('steps_updated', {'action': 'reorder'})

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
