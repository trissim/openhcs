"""
Clean pipeline editor using unified list management architecture.
"""
import asyncio
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from prompt_toolkit.application import get_app
from prompt_toolkit.layout import Container, VSplit, Window
from prompt_toolkit.widgets import Label

from openhcs.tui.components import ListManagerPane, ListConfig, ButtonConfig
from openhcs.tui.utils.dialog_helpers import show_error_dialog, prompt_for_file_dialog
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.steps.function_step import FunctionStep

logger = logging.getLogger(__name__)


class PipelineEditorPane:
    """Clean pipeline editor using unified list management architecture."""

    def __init__(self, state, context: ProcessingContext):
        """Initialize with clean architecture."""
        self.state = state
        self.context = context
        self.steps_lock = asyncio.Lock()

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
            empty_message="No steps available. Select a plate first."
        )

        self.list_manager = ListManagerPane(config, context)

        # Now set enabled functions after list_manager exists
        config.button_configs[1].enabled_func = lambda: self._has_items()  # Del
        config.button_configs[2].enabled_func = lambda: self._has_selection()  # Edit
        config.button_configs[4].enabled_func = lambda: self._has_items()  # Save
        self.list_manager._on_model_changed = self._on_selection_changed

        logger.info("PipelineEditorPane: Initialized with clean architecture")

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
        """Add step handler."""
        logger.info("PipelineEditor: _handle_add_step called!")
        if not self._validate_orchestrator():
            return
        
        # Placeholder for step addition UI
        await show_error_dialog(
            "Add Step",
            "Step addition interface will be implemented.",
            self.state
        )

    async def _handle_delete_step(self):
        """Delete step handler."""
        if not self._validate_orchestrator() or not self._validate_selection():
            return

        selected_step = self.list_manager.get_selected_item()
        step_id = selected_step.get('id')
        
        if not step_id:
            await show_error_dialog("Delete Error", "Selected step has no ID.", self.state)
            return

        # Remove from orchestrator
        pipeline = self.state.active_orchestrator.pipeline_definition
        for i, step in enumerate(pipeline):
            if getattr(step, 'step_id', None) == step_id:
                pipeline.pop(i)
                logger.info(f"Removed step {step_id}")
                await self._refresh_steps()
                return
        
        await show_error_dialog("Delete Error", f"Step {step_id} not found.", self.state)

    async def _handle_edit_step(self):
        """Edit step handler."""
        if not self._validate_orchestrator() or not self._validate_selection():
            return

        selected_step = self.list_manager.get_selected_item()
        step_id = selected_step.get('id')

        # Find step instance in orchestrator
        for step in self.state.active_orchestrator.pipeline_definition:
            if getattr(step, 'step_id', None) == step_id:
                if isinstance(step, FunctionStep):
                    # Activate step editor
                    self.state.step_to_edit_config = step
                    self.state.editing_step_config = True
                    await self.state.notify('editing_step_config_changed', True)
                    return
        
        await show_error_dialog("Edit Error", f"Step {step_id} not found.", self.state)

    async def _handle_load_pipeline(self):
        """Load pipeline handler."""
        if not self._validate_orchestrator():
            return

        file_path = await prompt_for_file_dialog(
            title="Load Pipeline",
            prompt_message="Select pipeline file:",
            app_state=self.state,
            filemanager=getattr(self.context, 'filemanager', None),
            selection_mode="files",
            filter_extensions=[".pipeline"]
        )

        if not file_path:
            return

        try:
            backend = getattr(self.context.global_config, 'backend', 'disk')
            loaded_data = self.context.filemanager.load(file_path, backend)
            
            if not isinstance(loaded_data, list):
                await show_error_dialog("Load Error", "Invalid pipeline format.", self.state)
                return

            self.state.active_orchestrator.pipeline_definition = loaded_data
            await self._refresh_steps()
            logger.info(f"Loaded pipeline from {file_path}")
            
        except Exception as e:
            await show_error_dialog("Load Error", f"Failed to load: {e}", self.state)

    async def _handle_save_pipeline(self):
        """Save pipeline handler."""
        if not self._validate_orchestrator():
            return

        pipeline = self.state.active_orchestrator.pipeline_definition
        if not pipeline:
            await show_error_dialog("Save Error", "No pipeline to save.", self.state)
            return

        file_path = await prompt_for_file_dialog(
            title="Save Pipeline",
            prompt_message="Select save location:",
            app_state=self.state,
            filemanager=getattr(self.context, 'filemanager', None),
            selection_mode="files",
            filter_extensions=[".pipeline"]
        )

        if not file_path:
            return

        try:
            backend = getattr(self.context.global_config, 'backend', 'disk')
            self.context.filemanager.save(pipeline, file_path, backend)
            logger.info(f"Saved pipeline to {file_path}")
            
        except Exception as e:
            await show_error_dialog("Save Error", f"Failed to save: {e}", self.state)

    # Validation helpers
    def _validate_orchestrator(self) -> bool:
        """Validate orchestrator is available."""
        if not self.state.active_orchestrator:
            get_app().create_background_task(
                show_error_dialog("No Orchestrator", "No active plate selected.", self.state)
            )
            return False
        return True

    def _validate_selection(self) -> bool:
        """Validate item is selected."""
        if not self.list_manager.get_selected_item():
            get_app().create_background_task(
                show_error_dialog("No Selection", "Please select a step.", self.state)
            )
            return False
        return True

    # Event handlers for step updates
    async def _handle_step_pattern_saved(self, data: Dict[str, Any]):
        """Handle step pattern saved event."""
        saved_step = data.get('step')
        if not saved_step:
            return

        async with self.steps_lock:
            items = self.list_manager.model.items
            for i, step in enumerate(items):
                if step.get('id') == saved_step.get('id'):
                    items[i] = saved_step
                    break
            else:
                items.append(saved_step)

            self.list_manager.load_items(items)
            self.state.notify('steps_updated', {'steps': items})

    async def shutdown(self):
        """Cleanup observers."""
        logger.info("PipelineEditorPane: Shutting down")
        self.state.remove_observer('plate_selected', self._on_plate_selected)
        self.state.remove_observer('steps_updated', self._refresh_steps)
        logger.info("PipelineEditorPane: Observers unregistered")
