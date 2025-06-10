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
    plate_pipelines = reactive({})  # {plate_path: List[Dict]} - per-plate pipeline storage
    
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
            list_id="step_content",
            container_id="step_list",
            on_button_pressed=self._handle_button_press,
            on_selection_changed=self._handle_selection_change,
            on_item_moved=self._handle_item_moved
        )

        self.filemanager = filemanager
        self.global_config = global_config

        # Reference to plate manager (set by MainContent)
        self.plate_manager = None

        logger.debug("PipelineEditorWidget initialized")
    
    def format_item_for_display(self, step: Dict) -> Tuple[str, str]:
        """Format step for display in the list."""
        step_name = step.get('name', 'Unknown Step')
        step_type = step.get('type', 'function')
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

        step_name = step['name']
        direction = "up" if to_index < from_index else "down"
        self.app.current_status = f"Moved step '{step_name}' {direction}"
        logger.info(f"Moved step '{step_name}' from index {from_index} to {to_index}")
    
    def on_selection_list_selected_changed(self, event: SelectionList.SelectedChanged) -> None:
        """Handle selection changes from SelectionList."""
        selected_values = event.selection_list.selected

        logger.info(f"Step selection changed: {len(selected_values)} items selected")

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

    def get_selection_state(self) -> tuple[List[Dict], str]:
        """Get current selection state from SelectionList."""
        try:
            selection_list = self.query_one("#pipeline_content", SelectionList)
            selected_values = selection_list.selected

            # Convert selected values back to step dictionaries
            selected_items = []
            for step in self.pipeline_steps:
                step_name = step.get('name', '')
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

        # Load pipeline for the new plate
        if plate_path:
            # Get pipeline for this plate (or empty if none exists)
            plate_pipeline = self.plate_pipelines.get(plate_path, [])
            self.pipeline_steps = plate_pipeline
            logger.info(f"Loaded {len(plate_pipeline)} steps for plate: {plate_path}")
        else:
            # No plate selected - clear steps
            self.pipeline_steps = []
            logger.info("No plate selected - cleared pipeline steps")

        # Clear selection when plate changes
        self.selected_step = ""

        # Update SelectionList content
        try:
            selection_list = self.query_one("#pipeline_content", SelectionList)

            # Clear existing options
            selection_list.clear_options()

            # Add steps as selection options
            step_options = []
            for step in self.pipeline_steps:
                # Format: (display_text, value)
                step_name = step.get('name', 'Unknown Step')
                step_type = step.get('type', 'function')
                display_text = f"📋 {step_name} ({step_type})"
                step_options.append((display_text, step_name))

            selection_list.add_options(step_options)
        except Exception:
            pass

        # Update button states
        self._update_button_states()

    def watch_pipeline_steps(self, steps: List[Dict]) -> None:
        """Automatically update UI when pipeline_steps changes."""
        # Update SelectionList content
        try:
            selection_list = self.query_one("#pipeline_content", SelectionList)

            # Clear existing options
            selection_list.clear_options()

            # Add steps as selection options
            step_options = []
            for step in steps:
                # Format: (display_text, value)
                step_name = step.get('name', 'Unknown Step')
                step_type = step.get('type', 'function')
                display_text = f"📋 {step_name} ({step_type})"
                step_options.append((display_text, step_name))

            selection_list.add_options(step_options)
        except Exception:
            pass

        # Update button states
        self._update_button_states()

        logger.debug(f"Pipeline steps updated: {len(steps)} steps")

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

    def get_pipeline_for_plate(self, plate_path: str) -> List[Dict]:
        """Get pipeline for specific plate."""
        return self.plate_pipelines.get(plate_path, [])

    def save_pipeline_for_plate(self, plate_path: str, pipeline: List[Dict]) -> None:
        """Save pipeline for specific plate."""
        current_pipelines = dict(self.plate_pipelines)
        current_pipelines[plate_path] = pipeline
        self.plate_pipelines = current_pipelines
        logger.info(f"Saved pipeline with {len(pipeline)} steps for plate: {plate_path}")

    def clear_pipeline_for_plate(self, plate_path: str) -> None:
        """Clear pipeline for specific plate."""
        current_pipelines = dict(self.plate_pipelines)
        if plate_path in current_pipelines:
            del current_pipelines[plate_path]
            self.plate_pipelines = current_pipelines
            logger.info(f"Cleared pipeline for plate: {plate_path}")

    def _invalidate_compilation_status(self) -> None:
        """Reset plate status from compiled to initialized when pipeline changes."""
        if not self.plate_manager or not self.current_plate:
            return

        # Find the current plate and reset status if compiled
        for plate in self.plate_manager.plates:
            if plate.get('path') == self.current_plate and plate.get('status') == 'o':
                plate['status'] = '-'  # Reset from compiled to initialized
                logger.info(f"Reset plate {plate.get('name')} from compiled to initialized due to pipeline change")

                # Trigger reactive update
                self.plate_manager.mutate_reactive(self.plate_manager.__class__.plates)

                # Update our own status
                self.current_plate_status = '-'
                break
    
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
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
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
    
    def action_add_step(self) -> None:
        """Handle Add Step button - now triggers modal."""
        logger.info("Add Step button pressed")

        def handle_result(result: Optional[FunctionStep]) -> None:
            if result:  # User saved new step
                # Convert to dict using consistent conversion method
                new_step_dict = self._function_step_to_dict(result)
                new_steps = self.pipeline_steps + [new_step_dict]
                self.pipeline_steps = new_steps
                self.app.current_status = f"Added step: {result.name}"
                logger.info(f"Added step '{result.name}' to plate '{self.current_plate}'")
            else:
                self.app.current_status = "Add step cancelled"

        # LAZY IMPORT to avoid circular import
        from openhcs.textual_tui.screens.dual_editor import DualEditorScreen

        # Launch modal
        self.app.push_screen(DualEditorScreen(is_new=True), handle_result)
    
    def action_delete_step(self) -> None:
        """Handle Delete Step button - delete selected steps."""
        logger.info("Delete Step button pressed")

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
            item_name = selected_items[0].get('name', 'Unknown')
            desc = f"Delete selected item: {item_name}"
        else:
            desc = f"Delete {count} selected items"

        logger.info(f"Deleting steps: {desc}")

        # Remove selected steps
        current_steps = list(self.pipeline_steps)
        steps_to_remove = set(item.get('name', '') for item in selected_items)

        # Filter out selected steps
        new_steps = [step for step in current_steps if step.get('name', '') not in steps_to_remove]

        # Update pipeline steps (this will trigger save to plate storage)
        self.pipeline_steps = new_steps

        deleted_count = len(current_steps) - len(new_steps)
        self.app.current_status = f"Deleted {deleted_count} steps"
        logger.info(f"Deleted {deleted_count} steps from plate '{self.current_plate}'")
    
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
            step_name = step.get("name", f"Step {i+1}")
            if step_name == self.selected_step:
                return i
        return None

    def action_edit_step(self) -> None:
        """Handle Edit Step button with proper selection and data preservation."""
        logger.info("Edit Step button pressed")

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
                # Convert back to dict with complete data preservation
                updated_step_dict = self._function_step_to_dict(result)

                # Update step in pipeline at correct index
                updated_steps = self.pipeline_steps.copy()
                updated_steps[step_index] = updated_step_dict
                self.pipeline_steps = updated_steps
                self.app.current_status = f"Updated step: {result.name}"
                logger.info(f"Updated step at index {step_index}: {result.name}")
            else:
                self.app.current_status = "Edit step cancelled"

        # Convert dict to FunctionStep with proper data preservation
        edit_step = self._dict_to_function_step(step_to_edit)

        # LAZY IMPORT to avoid circular import
        from openhcs.textual_tui.screens.dual_editor import DualEditorScreen

        # Launch modal
        self.app.push_screen(DualEditorScreen(edit_step), handle_result)
    
    def action_load_pipeline(self) -> None:
        """Handle Load Pipeline button - load pipeline from file."""
        logger.info("Load Pipeline button pressed")

        if not self.current_plate:
            self.app.current_status = "No plate selected for loading pipeline"
            return

        # Launch enhanced file browser for .func files
        def handle_result(result):
            if result and isinstance(result, Path):
                self._load_pipeline_from_file(result)
            else:
                self.app.current_status = "Load pipeline cancelled"

        # Create enhanced file browser for .func files
        from openhcs.textual_tui.screens.enhanced_file_browser import EnhancedFileBrowserScreen, BrowserMode, SelectionMode
        from openhcs.constants.constants import Backend

        browser = EnhancedFileBrowserScreen(
            file_manager=self.filemanager,
            initial_path=Path.home(),
            backend=Backend.DISK,
            title="Load Pipeline (.func)",
            mode=BrowserMode.LOAD,
            selection_mode=SelectionMode.FILES_ONLY,
            filter_extensions=['.func']
        )

        self.app.push_screen(browser, handle_result)

    def _load_pipeline_from_file(self, file_path: Path) -> None:
        """Load pipeline from .func file."""
        import pickle
        try:
            with open(file_path, 'rb') as f:
                pattern = pickle.load(f)

            if isinstance(pattern, list):
                self.pipeline_steps = pattern
                self.app.current_status = f"Loaded {len(pattern)} steps from {file_path.name}"
                logger.info(f"Loaded pipeline with {len(pattern)} steps")
            else:
                self.app.current_status = f"Invalid pipeline format in {file_path.name}"
                logger.error(f"Invalid pipeline format: expected list, got {type(pattern)}")
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            self.app.current_status = f"Failed to load pipeline: {e}"

    def action_save_pipeline(self) -> None:
        """Handle Save Pipeline button - save pipeline to file."""
        logger.info("Save Pipeline button pressed")

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

        # Create enhanced file browser for saving .func files
        from openhcs.textual_tui.screens.enhanced_file_browser import EnhancedFileBrowserScreen, BrowserMode, SelectionMode
        from openhcs.constants.constants import Backend

        # Generate default filename from plate name
        plate_name = Path(self.current_plate).name if self.current_plate else "pipeline"
        default_filename = f"{plate_name}.func"

        browser = EnhancedFileBrowserScreen(
            file_manager=self.filemanager,
            initial_path=Path.home(),
            backend=Backend.DISK,
            title="Save Pipeline (.func)",
            mode=BrowserMode.SAVE,
            selection_mode=SelectionMode.FILES_ONLY,
            filter_extensions=['.func'],
            default_filename=default_filename
        )

        self.app.push_screen(browser, handle_result)

    def _save_pipeline_to_file(self, file_path: Path) -> None:
        """Save pipeline to .func file."""
        import pickle
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(list(self.pipeline_steps), f)
            self.app.current_status = f"Saved pipeline to {file_path.name}"
            logger.info(f"Saved pipeline with {len(self.pipeline_steps)} steps")
        except Exception as e:
            logger.error(f"Failed to save pipeline: {e}")
            self.app.current_status = f"Failed to save pipeline: {e}"
