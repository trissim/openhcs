"""
Clean pipeline editor using unified list management architecture.
"""
import asyncio
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import copy

from prompt_toolkit.application import get_app
from prompt_toolkit.layout import Container, VSplit, Window
from prompt_toolkit.widgets import Label, Dialog

from openhcs.tui.components import ListManagerPane, ListConfig, ButtonConfig
from openhcs.tui.utils.dialog_helpers import show_error_dialog, prompt_for_file_dialog
from openhcs.tui.interfaces.swappable_pane import SwappablePaneInterface
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.steps.function_step import FunctionStep
from openhcs.constants.constants import Backend
from openhcs.core.pipeline import Pipeline
from openhcs.tui.services.visual_programming_dialog_service import VisualProgrammingDialogService

logger = logging.getLogger(__name__)


class PipelineEditorPane:
    """Clean pipeline editor using unified list management architecture."""

    def __init__(self, state, context: ProcessingContext):
        """Initialize with clean architecture."""
        self.state = state
        self.context = context
        self.steps_lock = asyncio.Lock()

        # EXACT: Pipeline-plate association storage
        self.plate_pipelines: Dict[str, Pipeline] = {}  # {plate_path: Pipeline}
        self.current_selected_plates: List[str] = []    # Currently selected plate paths
        self.pipeline_differs_across_plates: bool = False

        # EXACT: Visual programming dialog service
        self.visual_programming_service = VisualProgrammingDialogService(
            state=self.state,
            context=self.context
        )

        # Create unified list manager with declarative config (without enabled_func initially)
        config = ListConfig(
            title="Pipeline Editor",
            frame_title="Pipeline Editor",
            button_configs=[
                ButtonConfig("Add", self._handle_add_step, width=len("Add") + 2),
                ButtonConfig("Del", self._handle_delete_step, width=len("Del") + 2),
                ButtonConfig("Edit", self._handle_edit_step, width=len("Edit") + 2),
                ButtonConfig("Load", self._handle_load_pipeline, width=len("Load") + 2),
                ButtonConfig("Save", self._handle_save_pipeline, width=len("Save") + 2),
            ],
            display_func=self._get_display_text,
            can_move_up_func=self._can_move_up,
            can_move_down_func=self._can_move_down,
            empty_message="No steps available.\n\nSelect a plate first."
        )

        self.list_manager = ListManagerPane(config, context)

        # Now set enabled functions after list_manager exists
        config.button_configs[1].enabled_func = lambda: self._has_items()  # Del
        config.button_configs[2].enabled_func = lambda: self._has_selection()  # Edit
        config.button_configs[4].enabled_func = lambda: self._has_items()  # Save
        self.list_manager._on_model_changed = self._on_selection_changed

        logger.info("PipelineEditorPane: Initialized with clean architecture")

        # EXACT: Register state observer for real-time updates
        self._register_plate_selection_observer()

    @classmethod
    async def create(cls, state, context: ProcessingContext):
        """Factory for backward compatibility - simplified."""
        instance = cls(state, context)
        await instance._register_observers()
        
        # Load initial steps if plate already selected
        if state.selected_plate:
            await instance._load_steps_for_plate(state.selected_plate.get('id'))
        
        return instance

    @property
    def container(self) -> Container:
        """Get the UI container."""
        return self.list_manager.container

    def get_buttons_container(self) -> Container:
        """Return the button bar from list manager."""
        return self.list_manager.view._create_button_bar()

    # Helper methods for button state
    def _has_items(self) -> bool:
        """Check if list has items."""
        return len(self.list_manager.model.items) > 0

    def _has_selection(self) -> bool:
        """Check if there's a valid selection."""
        return self.list_manager.get_selected_item() is not None

    # Display and validation functions
    def _get_display_text(self, step_data: Dict[str, Any], is_selected: bool) -> str:
        """Generate display text for a step."""
        status_icon = self._get_status_icon(step_data.get('status', 'unknown'))
        name = step_data.get('name', 'Unknown Step')
        func_name = self._get_function_name(step_data)
        output_type = step_data.get('output_memory_type', '[N/A]')
        return f"{status_icon} {name} | {func_name} → {output_type}"

    def _can_move_up(self, index: int, step_data: Dict[str, Any]) -> bool:
        """Check if step can be moved up (within same pipeline)."""
        if index <= 0:
            return False
        items = self.list_manager.model.items
        current_pipeline = step_data.get('pipeline_id')
        prev_pipeline = items[index - 1].get('pipeline_id')
        return current_pipeline == prev_pipeline

    def _can_move_down(self, index: int, step_data: Dict[str, Any]) -> bool:
        """Check if step can be moved down (within same pipeline)."""
        items = self.list_manager.model.items
        if index >= len(items) - 1:
            return False
        current_pipeline = step_data.get('pipeline_id')
        next_pipeline = items[index + 1].get('pipeline_id')
        return current_pipeline == next_pipeline

    def _get_status_icon(self, status: str) -> str:
        """Get status icon for a step."""
        icons = {
            'pending': "?", 'validated': "o", 'error': "!",
            'not_initialized': "?", 'initialized': "!", 'ready': "!",
            'compiled_ok': "o", 'compiled': "o", 'running': "o"
        }
        return icons.get(status, "o")

    def _get_function_name(self, step: Dict[str, Any]) -> str:
        """Get display name for function pattern."""
        if 'func' not in step:
            return '[MISSING FUNC]'
        
        func = step['func']
        if func is None:
            return '[NULL FUNC]'
        
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

    # Event handling
    async def _register_observers(self):
        """Register observers for state integration."""
        try:
            self.state.add_observer('plate_selected', 
                lambda plate: get_app().create_background_task(self._on_plate_selected(plate)))
            self.state.add_observer('steps_updated', 
                lambda _: get_app().create_background_task(self._refresh_steps()))
            self.state.add_observer('step_pattern_saved', 
                lambda data: get_app().create_background_task(self._handle_step_pattern_saved(data)))
            logger.info("PipelineEditorPane: Observers registered")
        except Exception as e:
            logger.error(f"Error registering observers: {e}", exc_info=True)

    def _on_selection_changed(self):
        """Handle selection changes."""
        # CRITICAL: Call the original method to trigger UI invalidation
        from prompt_toolkit.application import get_app
        get_app().invalidate()

        # Then handle our business logic
        selected_item = self.list_manager.get_selected_item()
        if selected_item:
            self.state.set_selected_step(selected_item)

    async def _on_plate_selected(self, plate):
        """Handle plate selection event."""
        if plate:
            await self._load_steps_for_plate(plate['id'])

    async def _refresh_steps(self, _=None):
        """Refresh the step list."""
        if self.state.selected_plate:
            await self._load_steps_for_plate(self.state.selected_plate['id'])
        else:
            self.list_manager.load_items([])

    # Step loading
    async def _load_steps_for_plate(self, plate_id: str):
        """Load steps for the specified plate."""
        async with self.steps_lock:
            # Try orchestrator first, fallback to context
            raw_steps = self._get_orchestrator_steps()
            if raw_steps:
                steps = [self._transform_step_to_dict(step) for step in raw_steps 
                        if isinstance(step, FunctionStep)]
            else:
                steps = self.context.list_steps_for_plate(plate_id)
            
            self.list_manager.load_items(steps)

    def _get_orchestrator_steps(self) -> List[Any]:
        """Get step objects from active orchestrator."""
        if self.state.active_orchestrator and self.state.active_orchestrator.pipeline_definition:
            return self.state.active_orchestrator.pipeline_definition
        return []

    def _transform_step_to_dict(self, step_obj: FunctionStep) -> Dict[str, Any]:
        """Transform FunctionStep to display dictionary."""
        return {
            'id': step_obj.step_id,
            'name': step_obj.name,
            'func': step_obj.func,
            'status': 'pending',
            'pipeline_id': getattr(step_obj, 'pipeline_id', None),
            'output_memory_type': getattr(step_obj, 'output_memory_type', '[N/A]')
        }

    # Action handlers
    async def _handle_add_step(self):
        """EXACT: Add step handler with visual programming integration."""
        if not self.current_selected_plates:
            await show_error_dialog(
                "No Plate Selected",
                "Please select a plate to add steps to its pipeline.",
                self.state
            )
            return

        if self.pipeline_differs_across_plates:
            await show_error_dialog(
                "Multiple Different Pipelines",
                "Cannot add steps when selected plates have different pipelines. Please select plates with identical pipelines or edit them individually.",
                self.state
            )
            return

        # EXACT: Get target pipeline(s)
        target_pipelines = []
        for plate_path in self.current_selected_plates:
            pipeline = self.plate_pipelines.get(plate_path)
            if not pipeline:
                # EXACT: Create new pipeline if none exists
                pipeline = Pipeline(name=f"Pipeline for {Path(plate_path).name}")
                self.plate_pipelines[plate_path] = pipeline
            target_pipelines.append(pipeline)

        # EXACT: Launch visual programming dialog using service
        created_step = await self.visual_programming_service.show_add_step_dialog(target_pipelines)

        if created_step:
            # EXACT: Add step to all target pipelines
            for pipeline in target_pipelines:
                pipeline.append(copy.deepcopy(created_step))

            # EXACT: Refresh display
            await self._update_pipeline_display_for_selection(self.current_selected_plates)

    async def _handle_delete_step(self):
        """EXACT: Delete step handler with multi-plate support."""
        if not self.current_selected_plates:
            await show_error_dialog(
                "No Plate Selected",
                "Please select a plate to delete steps from its pipeline.",
                self.state
            )
            return

        if self.pipeline_differs_across_plates:
            await show_error_dialog(
                "Multiple Different Pipelines",
                "Cannot delete steps when selected plates have different pipelines. Please select plates with identical pipelines or edit them individually.",
                self.state
            )
            return

        selected_step = self.list_manager.get_selected_item()
        if not selected_step:
            await show_error_dialog("No Selection", "Please select a step to delete.", self.state)
            return

        step_id = selected_step.get('id')
        if not step_id:
            await show_error_dialog("Delete Error", "Selected step has no ID.", self.state)
            return

        # EXACT: Remove from all target pipelines
        removed_count = 0
        for plate_path in self.current_selected_plates:
            pipeline = self.plate_pipelines.get(plate_path)
            if pipeline:
                # EXACT: Find and remove step by object ID
                for i, step in enumerate(pipeline):
                    if str(id(step)) == step_id:
                        pipeline.pop(i)
                        removed_count += 1
                        break

        if removed_count > 0:
            # EXACT: Refresh display
            await self._update_pipeline_display_for_selection(self.current_selected_plates)
            logger.info(f"Removed step {step_id} from {removed_count} pipeline(s)")
        else:
            await show_error_dialog("Delete Error", f"Step {step_id} not found in any selected pipeline.", self.state)

    async def _handle_edit_step(self):
        """EXACT: Edit step handler with visual programming integration."""
        if not self.current_selected_plates:
            await show_error_dialog(
                "No Plate Selected",
                "Please select a plate to edit steps in its pipeline.",
                self.state
            )
            return

        if self.pipeline_differs_across_plates:
            await show_error_dialog(
                "Multiple Different Pipelines",
                "Cannot edit steps when selected plates have different pipelines. Please select plates with identical pipelines or edit them individually.",
                self.state
            )
            return

        selected_step = self.list_manager.get_selected_item()
        if not selected_step:
            await show_error_dialog("No Selection", "Please select a step to edit.", self.state)
            return

        step_id = selected_step.get('id')
        if not step_id:
            await show_error_dialog("Edit Error", "Selected step has no ID.", self.state)
            return

        # EXACT: Find step instance in first pipeline (they're identical)
        first_plate_path = self.current_selected_plates[0]
        pipeline = self.plate_pipelines.get(first_plate_path)

        if not pipeline:
            await show_error_dialog("Edit Error", "No pipeline found for selected plate.", self.state)
            return

        target_step = None
        for step in pipeline:
            if str(id(step)) == step_id:
                target_step = step
                break

        if not target_step:
            await show_error_dialog("Edit Error", f"Step {step_id} not found in pipeline.", self.state)
            return

        # EXACT: Launch visual programming dialog for editing using service
        edited_step = await self.visual_programming_service.show_edit_step_dialog(target_step)

        if edited_step:
            # EXACT: Update step in all target pipelines
            for plate_path in self.current_selected_plates:
                pipeline = self.plate_pipelines.get(plate_path)
                if pipeline:
                    # EXACT: Find and replace step by object ID
                    for i, step in enumerate(pipeline):
                        if step is target_step:
                            pipeline[i] = copy.deepcopy(edited_step)
                            break

            # EXACT: Refresh display
            await self._update_pipeline_display_for_selection(self.current_selected_plates)

    async def _handle_load_pipeline(self):
        """EXACT: Load pipeline handler with multi-plate support."""
        if not self.current_selected_plates:
            await show_error_dialog(
                "No Plate Selected",
                "Please select plates to load pipeline into.",
                self.state
            )
            return

        file_path = await prompt_for_file_dialog(
            title="Load Pipeline",
            prompt_message="Select pipeline file:",
            app_state=self.state,
            filemanager=self.context.filemanager,
            selection_mode="files",
            filter_extensions=[".pipeline"]
        )

        if not file_path:
            return

        try:
            # EXACT: Load pipeline data
            loaded_data = self.context.filemanager.load(file_path, Backend.DISK.value)

            if not isinstance(loaded_data, list):
                await show_error_dialog("Load Error", "Invalid pipeline format - must be a list of steps.", self.state)
                return

            # EXACT: Create Pipeline object from loaded data
            loaded_pipeline = Pipeline(name=f"Loaded from {Path(file_path).name}")
            loaded_pipeline.extend(loaded_data)

            # EXACT: Apply to all selected plates
            for plate_path in self.current_selected_plates:
                self.plate_pipelines[plate_path] = copy.deepcopy(loaded_pipeline)

            # EXACT: Refresh display
            await self._update_pipeline_display_for_selection(self.current_selected_plates)
            logger.info(f"Loaded pipeline from {file_path} into {len(self.current_selected_plates)} plate(s)")

        except Exception as e:
            await show_error_dialog("Load Error", f"Failed to load pipeline: {str(e)}", self.state)

    async def _handle_save_pipeline(self):
        """EXACT: Save pipeline handler with multi-plate support."""
        if not self.current_selected_plates:
            await show_error_dialog(
                "No Plate Selected",
                "Please select a plate to save its pipeline.",
                self.state
            )
            return

        if self.pipeline_differs_across_plates:
            await show_error_dialog(
                "Multiple Different Pipelines",
                "Cannot save when selected plates have different pipelines. Please select plates with identical pipelines or save them individually.",
                self.state
            )
            return

        # EXACT: Get pipeline to save (from first selected plate)
        first_plate_path = self.current_selected_plates[0]
        pipeline = self.plate_pipelines.get(first_plate_path)

        if not pipeline or len(pipeline) == 0:
            await show_error_dialog("Save Error", "No pipeline steps to save.", self.state)
            return

        file_path = await prompt_for_file_dialog(
            title="Save Pipeline",
            prompt_message="Select save location:",
            app_state=self.state,
            filemanager=self.context.filemanager,
            selection_mode="files",
            filter_extensions=[".pipeline"]
        )

        if not file_path:
            return

        try:
            # EXACT: Save pipeline as list (Pipeline IS a list)
            pipeline_data = list(pipeline)
            self.context.filemanager.save(pipeline_data, file_path, Backend.DISK.value)
            logger.info(f"Saved pipeline to {file_path} ({len(pipeline_data)} steps)")

        except Exception as e:
            await show_error_dialog("Save Error", f"Failed to save pipeline: {str(e)}", self.state)

    # EXACT: Legacy cleanup - these methods are no longer needed

    def _register_plate_selection_observer(self):
        """EXACT: Register observer for plate selection changes."""
        self.state.add_observer('plate_selected',
            lambda plate: get_app().create_background_task(self._on_plate_selection_changed(plate)))

    async def _on_plate_selection_changed(self, plate_data):
        """EXACT: Handle plate selection changes in real-time."""
        try:
            # EXACT: Get current selection state from PlateManager
            selected_plates = self._get_current_selected_plates()
            self.current_selected_plates = selected_plates

            # EXACT: Update pipeline display based on selection
            await self._update_pipeline_display_for_selection(selected_plates)

        except Exception as e:
            await show_error_dialog(
                "Selection Update Error",
                f"Failed to update pipeline display: {str(e)}",
                self.state
            )

    def _get_current_selected_plates(self) -> List[str]:
        """EXACT: Get currently selected plate paths from PlateManager."""
        # EXACT: Access PlateManager selection state
        if hasattr(self.state, 'selected_plate') and self.state.selected_plate:
            return [self.state.selected_plate['path']]
        return []

    async def _update_pipeline_display_for_selection(self, selected_plates: List[str]):
        """EXACT: Update pipeline display based on selected plates."""
        if not selected_plates:
            # EXACT: No plates selected - show empty
            self.pipeline_differs_across_plates = False
            self.list_manager.load_items([])
            return

        if len(selected_plates) == 1:
            # EXACT: Single plate selected - show its pipeline
            plate_path = selected_plates[0]
            pipeline = self.plate_pipelines.get(plate_path)

            if not pipeline:
                # EXACT: Create default empty pipeline for new plate
                pipeline = Pipeline(name=f"Pipeline for {Path(plate_path).name}")
                self.plate_pipelines[plate_path] = pipeline

            self.pipeline_differs_across_plates = False
            self._refresh_step_list_for_pipeline(pipeline)

        else:
            # EXACT: Multiple plates selected - check if pipelines match
            pipelines = [self.plate_pipelines.get(plate_path) for plate_path in selected_plates]

            if self._all_pipelines_identical(pipelines):
                # EXACT: All pipelines identical - show common pipeline
                common_pipeline = pipelines[0] if pipelines[0] else Pipeline(name="Common Pipeline")
                self.pipeline_differs_across_plates = False
                self._refresh_step_list_for_pipeline(common_pipeline)
            else:
                # EXACT: Pipelines differ - show "differs" message
                self.pipeline_differs_across_plates = True
                self._show_pipeline_differs_message()

    def _all_pipelines_identical(self, pipelines: List[Optional[Pipeline]]) -> bool:
        """EXACT: Check if all pipelines are identical."""
        # EXACT: Handle None pipelines (treat as empty)
        normalized_pipelines = []
        for p in pipelines:
            if p is None:
                normalized_pipelines.append([])  # Empty pipeline
            else:
                normalized_pipelines.append(list(p))  # Pipeline IS a list

        # EXACT: Check if all normalized pipelines are identical
        if not normalized_pipelines:
            return True

        first_pipeline = normalized_pipelines[0]
        return all(pipeline == first_pipeline for pipeline in normalized_pipelines[1:])

    def _show_pipeline_differs_message(self):
        """EXACT: Show 'pipeline differs across plates' message."""
        self.list_manager.load_items([{
            'id': 'differs_message',
            'name': 'Pipeline differs across plates',
            'func': 'Cannot show pipeline - selected plates have different pipelines',
            'variable_components': '',
            'group_by': ''
        }])

    def _refresh_step_list_for_pipeline(self, pipeline: Pipeline):
        """EXACT: Refresh step list for specific pipeline."""
        step_items = []
        for i, step in enumerate(pipeline):
            step_items.append({
                'id': str(id(step)),  # Use object ID as unique identifier
                'name': getattr(step, 'name', f'Step {i+1}'),
                'func': self._format_func_display(step.func),
                'variable_components': ', '.join(step.variable_components or []),
                'group_by': step.group_by
            })

        self.list_manager.load_items(step_items)

    def _format_func_display(self, func) -> str:
        """EXACT: Format function for display."""
        if func is None:
            return "No function"
        elif callable(func):
            return getattr(func, '__name__', str(func))
        elif isinstance(func, tuple) and len(func) >= 1:
            return getattr(func[0], '__name__', str(func[0]))
        elif isinstance(func, list) and len(func) > 0:
            first_func = func[0]
            if isinstance(first_func, tuple):
                return f"{getattr(first_func[0], '__name__', str(first_func[0]))} + {len(func)-1} more"
            else:
                return f"{getattr(first_func, '__name__', str(first_func))} + {len(func)-1} more"
        else:
            return str(func)

    # EXACT: Dialog methods removed - now handled by VisualProgrammingDialogService





    async def shutdown(self):
        """EXACT: Cleanup observers."""
        logger.info("PipelineEditorPane: Shutting down")

        # EXACT: Remove observers (updated for new architecture)
        try:
            self.state.remove_observer('plate_selected', self._on_plate_selection_changed)
        except (AttributeError, ValueError):
            pass  # Observer may not exist or already removed

        logger.info("PipelineEditorPane: Observers unregistered and cleanup complete")
