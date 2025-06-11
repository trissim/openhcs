"""
PipelineEditorWidget for OpenHCS Textual TUI

Pipeline editing widget with complete button set and reactive state management.
Matches the functionality from the current prompt-toolkit TUI.
"""

import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.reactive import reactive
from textual.widgets import Button, Static, SelectionList
from textual.widget import Widget
from .button_list_widget import ButtonListWidget, ButtonConfig
from textual import work

from openhcs.core.config import GlobalPipelineConfig
from openhcs.io.filemanager import FileManager
from openhcs.core.steps.function_step import FunctionStep

logger = logging.getLogger(__name__)


class PipelineEditorWidget(ButtonListWidget):
    """
    Pipeline editing widget using Textual reactive state.
    
    Features:
    - Complete button set: Add, Del, Edit, Load, Save
    - Reactive state management for automatic UI updates
    - Scrollable content area
    - Integration with plate selection from PlateManager
    """
    
    # Textual reactive state
    pipeline_steps = reactive([])
    current_plate = reactive("")
    current_plate_status = reactive("?")  # Track plate initialization status
    selected_step = reactive("")
    plate_pipelines = reactive({})  # {plate_path: List[FunctionStep]} - per-plate pipeline storage
    
    def __init__(self, filemanager: FileManager, global_config: GlobalPipelineConfig):
        """
        Initialize the pipeline editor widget.

        Args:
            filemanager: FileManager instance for file operations
            global_config: Global configuration
        """
        # Define button configuration
        button_configs = [
            ButtonConfig("Add", "add_step", disabled=True),
            ButtonConfig("Del", "del_step", disabled=True),
            ButtonConfig("Edit", "edit_step", disabled=True),
            ButtonConfig("Load", "load_pipeline", disabled=True),
            ButtonConfig("Save", "save_pipeline", disabled=True),
        ]

        super().__init__(
            button_configs=button_configs,
            list_id="pipeline_content",
            container_id="pipeline_list",
            on_button_pressed=self._handle_button_press,
            on_selection_changed=self._handle_selection_change,
            on_item_moved=self._handle_item_moved
        )

        self.filemanager = filemanager
        # Note: We don't store global_config as it can become stale
        # Always use self.app.global_config to get the current config

        # Reference to plate manager (set by MainContent)
        self.plate_manager = None

        logger.debug("PipelineEditorWidget initialized")
    
    def format_item_for_display(self, step: FunctionStep) -> Tuple[str, str]:
        """Format step for display in the list."""
        step_name = getattr(step, 'name', 'Unknown Step')
        step_type = 'function'  # All steps are FunctionStep objects
        display_text = f"📋 {step_name} ({step_type})"
        return display_text, step_name

    def _handle_button_press(self, button_id: str) -> None:
        """Handle button presses from ButtonListWidget."""
        if button_id == "add_step":
            self.action_add_step()
        elif button_id == "del_step":
            self.action_delete_step()
        elif button_id == "edit_step":
            self.action_edit_step()
        elif button_id == "load_pipeline":
            self.action_load_pipeline()
        elif button_id == "save_pipeline":
            self.action_save_pipeline()

    def _handle_selection_change(self, selected_values: List[str]) -> None:
        """Handle selection changes from ButtonListWidget."""
        # Update selected_step - use first selected item if any
        if selected_values:
            self.selected_step = selected_values[0]  # This is the step name
        else:
            self.selected_step = ""

    def _handle_item_moved(self, from_index: int, to_index: int) -> None:
        """Handle item movement from ButtonListWidget."""
        current_steps = list(self.pipeline_steps)

        # Move the step
        step = current_steps.pop(from_index)
        current_steps.insert(to_index, step)

        # Update pipeline steps
        self.pipeline_steps = current_steps

        step_name = getattr(step, 'name', 'Unknown Step')
        direction = "up" if to_index < from_index else "down"
        self.app.current_status = f"Moved step '{step_name}' {direction}"
    
    def on_selection_list_selected_changed(self, event: SelectionList.SelectedChanged) -> None:
        """Handle selection changes from SelectionList."""
        selected_values = event.selection_list.selected

        # Update selected_step - use first selected item if any
        if selected_values:
            self.selected_step = selected_values[0]  # This is the step name/id
        else:
            self.selected_step = ""

        # Update button states based on selection
        self._update_button_states_for_selection(selected_values)

    def _update_button_states_for_selection(self, selected_values: List[str]) -> None:
        """Update button states based on current selection and mathematical constraints."""
        try:
            has_plate = bool(self.current_plate)
            is_initialized = self.current_plate_status in ['-', 'o']  # Initialized or compiled
            has_steps = len(self.pipeline_steps) > 0
            has_selection = len(selected_values) > 0

            # Mathematical constraints:
            # - Pipeline editing requires initialization
            # - Edit requires exactly one selection
            self.query_one("#add_step").disabled = not (has_plate and is_initialized)
            self.query_one("#del_step").disabled = not has_selection
            self.query_one("#edit_step").disabled = not (len(selected_values) == 1)  # Edit requires exactly one selection
            self.query_one("#load_pipeline").disabled = not (has_plate and is_initialized)
            self.query_one("#save_pipeline").disabled = not has_steps

        except Exception:
            # Buttons might not be mounted yet
            pass

    def get_selection_state(self) -> tuple[List[FunctionStep], str]:
        """Get current selection state from SelectionList."""
        try:
            selection_list = self.query_one("#pipeline_content", SelectionList)
            selected_values = selection_list.selected

            # Convert selected values back to step objects
            selected_items = []
            for step in self.pipeline_steps:
                step_name = getattr(step, 'name', '')
                if step_name in selected_values:
                    selected_items.append(step)

            # Determine selection mode
            if not selected_items:
                selection_mode = "empty"
            elif len(selected_items) == len(self.pipeline_steps):
                selection_mode = "all"
            else:
                selection_mode = "checkbox"  # SelectionList is always checkbox-based

            return selected_items, selection_mode
        except Exception:
            # Fallback if widget not mounted
            return [], "empty"
    
    def watch_current_plate(self, plate_path: str) -> None:
        """Automatically update UI when current_plate changes."""
        logger.debug(f"Current plate changed: {plate_path}")

        # Load pipeline for the new plate WITHOUT triggering save/invalidation
        if plate_path:
            # Get pipeline for this plate (or empty if none exists)
            plate_pipeline = self.plate_pipelines.get(plate_path, [])
            # Set pipeline_steps directly without triggering reactive save
            self._set_pipeline_steps_without_save(plate_pipeline)
        else:
            # No plate selected - clear steps
            self._set_pipeline_steps_without_save([])

        # Clear selection when plate changes
        self.selected_step = ""

        # Update button states
        self._update_button_states()

    def _set_pipeline_steps_without_save(self, steps: List[FunctionStep]) -> None:
        """Set pipeline steps without triggering save/invalidation (for loading existing data)."""
        # Temporarily disable the reactive watcher to prevent save cascade
        self._loading_existing_pipeline = True
        self.pipeline_steps = steps
        # Sync with ButtonListWidget's items property
        self.items = list(steps)
        self._loading_existing_pipeline = False

    def watch_pipeline_steps(self, steps: List[FunctionStep]) -> None:
        """Automatically update UI when pipeline_steps changes."""
        # Sync with ButtonListWidget's items property to trigger its reactive system
        self.items = list(steps)

        logger.debug(f"Pipeline steps updated: {len(steps)} steps")

        # Only save/invalidate if this is a real change, not loading existing data
        if not getattr(self, '_loading_existing_pipeline', False):
            # Save pipeline changes to plate storage
            self._save_pipeline_to_plate_storage()

    def _save_pipeline_to_plate_storage(self) -> None:
        """Save current pipeline steps to plate storage and invalidate compilation."""
        if self.current_plate:
            # Update plate pipelines storage
            current_pipelines = dict(self.plate_pipelines)
            current_pipelines[self.current_plate] = list(self.pipeline_steps)
            self.plate_pipelines = current_pipelines
            logger.debug(f"Saved {len(self.pipeline_steps)} steps for plate: {self.current_plate}")

            # Invalidate compilation status when pipeline changes
            self._invalidate_compilation_status()

    def get_pipeline_for_plate(self, plate_path: str) -> List[FunctionStep]:
        """Get pipeline for specific plate."""
        return self.plate_pipelines.get(plate_path, [])

    def save_pipeline_for_plate(self, plate_path: str, pipeline: List[FunctionStep]) -> None:
        """Save pipeline for specific plate."""
        current_pipelines = dict(self.plate_pipelines)
        current_pipelines[plate_path] = pipeline
        self.plate_pipelines = current_pipelines

    def clear_pipeline_for_plate(self, plate_path: str) -> None:
        """Clear pipeline for specific plate."""
        current_pipelines = dict(self.plate_pipelines)
        if plate_path in current_pipelines:
            del current_pipelines[plate_path]
            self.plate_pipelines = current_pipelines

    def _invalidate_compilation_status(self) -> None:
        """Reset compilation status when pipeline definition changes."""
        if not self.plate_manager or not self.current_plate:
            return

        # Clear compiled data from simple state
        if self.current_plate in self.plate_manager.plate_compiled_data:
            del self.plate_manager.plate_compiled_data[self.current_plate]

        # Find the current plate and reset status
        for plate in self.plate_manager.plates:
            if plate.get('path') == self.current_plate:
                plate['status'] = '-'  # Reset to initialized

                # Trigger reactive update
                self.plate_manager.mutate_reactive(self.plate_manager.__class__.plates)

                # Update our own status
                self.current_plate_status = '-'
                break

        # Update plate manager button states immediately
        if self.plate_manager:
            self.plate_manager._update_button_states()
    
    def watch_current_plate_status(self, status: str) -> None:
        """Automatically update UI when plate status changes."""
        self._update_button_states()
        logger.debug(f"Plate status changed: {status}")

    def watch_selected_step(self, step_id: str) -> None:
        """Automatically update UI when selected_step changes."""
        self._update_button_states()
        logger.debug(f"Selected step: {step_id}")
    
    def _update_button_states(self) -> None:
        """Update button enabled/disabled states based on mathematical constraints."""
        try:
            has_plate = bool(self.current_plate)
            is_initialized = self.current_plate_status in ['-', 'o']  # Initialized or compiled
            has_steps = len(self.pipeline_steps) > 0
            has_valid_selection = bool(self.selected_step) and self._find_step_index_by_selection() is not None

            # Mathematical constraints:
            # - Pipeline editing requires initialization
            # - Step operations require steps to exist
            # - Edit requires valid selection that maps to actual step
            self.query_one("#add_step").disabled = not (has_plate and is_initialized)
            self.query_one("#del_step").disabled = not has_steps
            self.query_one("#edit_step").disabled = not (has_steps and has_valid_selection)
            self.query_one("#load_pipeline").disabled = not (has_plate and is_initialized)
            self.query_one("#save_pipeline").disabled = not has_steps
        except Exception:
            # Buttons might not be mounted yet
            pass
    

    
    def action_add_step(self) -> None:
        """Handle Add Step button - now triggers modal."""

        def handle_result(result: Optional[FunctionStep]) -> None:
            if result:  # User saved new step
                # Store the actual FunctionStep object directly (preserves memory type decorators)
                new_steps = self.pipeline_steps + [result]
                self.pipeline_steps = new_steps
                self.app.current_status = f"Added step: {result.name}"
            else:
                self.app.current_status = "Add step cancelled"

        # LAZY IMPORT to avoid circular import
        from openhcs.textual_tui.screens.dual_editor import DualEditorScreen

        # Launch modal
        self.app.push_screen(DualEditorScreen(is_new=True), handle_result)
    
    def action_delete_step(self) -> None:
        """Handle Delete Step button - delete selected steps."""

        # Get current selection state
        selected_items, selection_mode = self.get_selection_state()

        if selection_mode == "empty":
            self.app.current_status = "No steps available for deletion"
            return

        # Generate description and perform deletion
        count = len(selected_items)
        if selection_mode == "empty":
            desc = "No items available for deletion"
        elif selection_mode == "all":
            desc = f"Delete ALL {count} items"
        elif count == 1:
            item_name = getattr(selected_items[0], 'name', 'Unknown')
            desc = f"Delete selected item: {item_name}"
        else:
            desc = f"Delete {count} selected items"

        # Remove selected steps
        current_steps = list(self.pipeline_steps)
        steps_to_remove = set(getattr(item, 'name', '') for item in selected_items)

        # Filter out selected steps
        new_steps = [step for step in current_steps if getattr(step, 'name', '') not in steps_to_remove]

        # Update pipeline steps (this will trigger save to plate storage)
        self.pipeline_steps = new_steps

        deleted_count = len(current_steps) - len(new_steps)
        self.app.current_status = f"Deleted {deleted_count} steps"
    
    def _dict_to_function_step(self, step_dict: Dict) -> FunctionStep:
        """Convert step dict to FunctionStep object with proper data preservation."""
        # Extract function - handle both callable and registry lookup
        func = step_dict.get("func")
        if func is None:
            # Fallback to default function if missing
            from openhcs.textual_tui.services.function_registry_service import FunctionRegistryService
            registry = FunctionRegistryService()
            func = registry.find_default_function()
            logger.warning(f"Step '{step_dict.get('name', 'Unknown')}' missing function, using default")

        # Extract variable components - handle both list and string formats
        var_components = step_dict.get("variable_components", [])
        if isinstance(var_components, str):
            var_components = [var_components]
        elif not isinstance(var_components, list):
            var_components = []

        return FunctionStep(
            func=func,
            name=step_dict.get("name", "Unknown Step"),
            variable_components=var_components,
            group_by=step_dict.get("group_by", "")
        )

    def _function_step_to_dict(self, step: FunctionStep) -> Dict:
        """Convert FunctionStep object to dict with complete data preservation."""
        return {
            "name": step.name,
            "type": "function",
            "func": step.func,
            "variable_components": step.variable_components,
            "group_by": step.group_by
        }

    def _find_step_index_by_selection(self) -> Optional[int]:
        """Find the index of the currently selected step."""
        if not self.selected_step:
            return None

        # selected_step contains the step name/id
        for i, step in enumerate(self.pipeline_steps):
            # Now step is a FunctionStep object, not a dict
            step_name = getattr(step, 'name', f"Step {i+1}")
            if step_name == self.selected_step:
                return i
        return None

    def action_edit_step(self) -> None:
        """Handle Edit Step button with proper selection and data preservation."""

        if not self.pipeline_steps:
            self.app.current_status = "No steps to edit"
            return

        # Find selected step index
        step_index = self._find_step_index_by_selection()
        if step_index is None:
            self.app.current_status = "No step selected for editing"
            return

        step_to_edit = self.pipeline_steps[step_index]

        def handle_result(result: Optional[FunctionStep]) -> None:
            if result:  # User saved changes
                # Store the actual FunctionStep object directly (preserves memory type decorators)
                updated_steps = self.pipeline_steps.copy()
                updated_steps[step_index] = result
                self.pipeline_steps = updated_steps
                self.app.current_status = f"Updated step: {result.name}"
            else:
                self.app.current_status = "Edit step cancelled"

        # Use the actual FunctionStep object directly (no conversion needed)
        edit_step = step_to_edit

        # LAZY IMPORT to avoid circular import
        from openhcs.textual_tui.screens.dual_editor import DualEditorScreen

        # Launch modal
        self.app.push_screen(DualEditorScreen(edit_step), handle_result)
    
    def action_load_pipeline(self) -> None:
        """Handle Load Pipeline button - load pipeline from file."""

        if not self.current_plate:
            self.app.current_status = "No plate selected for loading pipeline"
            return

        # Launch enhanced file browser for .pipeline files
        def handle_result(result):
            if result and isinstance(result, Path):
                self._load_pipeline_from_file(result)
            else:
                self.app.current_status = "Load pipeline cancelled"

        # Create enhanced file browser for .pipeline files
        from openhcs.textual_tui.screens.enhanced_file_browser import EnhancedFileBrowserScreen, BrowserMode, SelectionMode
        from openhcs.constants.constants import Backend
        from openhcs.textual_tui.utils.path_cache import get_cached_browser_path, PathCacheKey

        browser = EnhancedFileBrowserScreen(
            file_manager=self.filemanager,
            initial_path=get_cached_browser_path(PathCacheKey.FILE_SELECTION),
            backend=Backend.DISK,
            title="Load Pipeline (.pipeline)",
            mode=BrowserMode.LOAD,
            selection_mode=SelectionMode.FILES_ONLY,
            filter_extensions=['.pipeline']
        )

        self.app.push_screen(browser, handle_result)

    def _load_pipeline_from_file(self, file_path: Path) -> None:
        """Load pipeline from .pipeline file."""
        import pickle
        try:
            with open(file_path, 'rb') as f:
                pattern = pickle.load(f)

            if isinstance(pattern, list):
                self.pipeline_steps = pattern
                self.app.current_status = f"Loaded {len(pattern)} steps from {file_path.name}"
            else:
                self.app.current_status = f"Invalid pipeline format in {file_path.name}"
                logger.error(f"Invalid pipeline format: expected list, got {type(pattern)}")
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            self.app.current_status = f"Failed to load pipeline: {e}"

    def action_save_pipeline(self) -> None:
        """Handle Save Pipeline button - save pipeline to file."""

        if not self.current_plate:
            self.app.current_status = "No plate selected for saving pipeline"
            return

        if not self.pipeline_steps:
            self.app.current_status = "No pipeline steps to save"
            return

        # Launch enhanced file browser for saving pipeline
        def handle_result(result):
            if result and isinstance(result, Path):
                self._save_pipeline_to_file(result)
            else:
                self.app.current_status = "Save pipeline cancelled"

        # Create enhanced file browser for saving .pipeline files
        from openhcs.textual_tui.screens.enhanced_file_browser import EnhancedFileBrowserScreen, BrowserMode, SelectionMode
        from openhcs.constants.constants import Backend
        from openhcs.textual_tui.utils.path_cache import get_cached_browser_path, PathCacheKey

        # Generate default filename from plate name
        plate_name = Path(self.current_plate).name if self.current_plate else "pipeline"
        default_filename = f"{plate_name}.pipeline"

        browser = EnhancedFileBrowserScreen(
            file_manager=self.filemanager,
            initial_path=get_cached_browser_path(PathCacheKey.FILE_SELECTION),
            backend=Backend.DISK,
            title="Save Pipeline (.pipeline)",
            mode=BrowserMode.SAVE,
            selection_mode=SelectionMode.FILES_ONLY,
            filter_extensions=['.pipeline'],
            default_filename=default_filename
        )

        self.app.push_screen(browser, handle_result)

    def _save_pipeline_to_file(self, file_path: Path) -> None:
        """Save pipeline to .pipeline file."""
        import pickle
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(list(self.pipeline_steps), f)
            self.app.current_status = f"Saved pipeline to {file_path.name}"
        except Exception as e:
            logger.error(f"Failed to save pipeline: {e}")
            self.app.current_status = f"Failed to save pipeline: {e}"
