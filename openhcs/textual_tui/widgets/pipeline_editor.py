"""
PipelineEditorWidget for OpenHCS Textual TUI

Pipeline editing widget with complete button set and reactive state management.
Matches the functionality from the current prompt-toolkit TUI.
"""

import logging
from typing import Dict, List

from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer
from textual.reactive import reactive
from textual.widgets import Button, Static
from textual.widget import Widget
from textual import work

from openhcs.core.config import GlobalPipelineConfig
from openhcs.io.filemanager import FileManager
from openhcs.core.steps.function_step import FunctionStep
from .selectable_list_view import SelectableListView

logger = logging.getLogger(__name__)


class PipelineEditorWidget(Widget):
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
    selected_step = reactive("")
    plate_pipelines = reactive({})  # {plate_path: List[Dict]} - per-plate pipeline storage
    
    def __init__(self, filemanager: FileManager, global_config: GlobalPipelineConfig):
        """
        Initialize the pipeline editor widget.
        
        Args:
            filemanager: FileManager instance for file operations
            global_config: Global configuration
        """
        super().__init__()
        self.filemanager = filemanager
        self.global_config = global_config
        
        logger.debug("PipelineEditorWidget initialized")
    
    def compose(self) -> ComposeResult:
        """Compose the pipeline editor layout."""
        # Complete button set with compact uniform styling
        with Horizontal():
            yield Button("Add", id="add_step", disabled=True, compact=True)
            yield Button("Del", id="del_step", disabled=True, compact=True)
            yield Button("Edit", id="edit_step", disabled=True, compact=True)
            yield Button("Load", id="load_pipeline", disabled=True, compact=True)
            yield Button("Save", id="save_pipeline", disabled=True, compact=True)

        # Scrollable content area with SelectableListView
        with ScrollableContainer(id="pipeline_steps"):
            yield SelectableListView(items=self.pipeline_steps, id="pipeline_content")
    
    def on_selectable_list_view_selection_changed(self, event: SelectableListView.SelectionChanged) -> None:
        """Handle selection changes from SelectableListView."""
        selected_items = event.selected_items
        selection_mode = event.selection_mode

        logger.info(f"Step selection changed: {len(selected_items)} items in '{selection_mode}' mode")

        # Update selected_step based on selection mode
        if selection_mode == "cursor" and selected_items:
            self.selected_step = selected_items[0].get('name', '')
        elif selection_mode == "checkbox" and len(selected_items) == 1:
            self.selected_step = selected_items[0].get('name', '')
        else:
            self.selected_step = ""

        # Update button states based on selection
        self._update_button_states_for_selection(selected_items, selection_mode)

    def _update_button_states_for_selection(self, selected_items: List[Dict], selection_mode: str) -> None:
        """Update button states based on current selection."""
        try:
            has_plate = bool(self.current_plate)
            has_steps = len(self.pipeline_steps) > 0
            has_selection = len(selected_items) > 0

            self.query_one("#add_step").disabled = not has_plate
            self.query_one("#del_step").disabled = not has_selection
            self.query_one("#edit_step").disabled = not (selection_mode in ["cursor", "checkbox"] and len(selected_items) == 1)
            self.query_one("#load_pipeline").disabled = not has_plate
            self.query_one("#save_pipeline").disabled = not has_steps

        except Exception:
            # Buttons might not be mounted yet
            pass

    def get_selection_state(self) -> tuple[List[Dict], str]:
        """Get current selection state from SelectableListView."""
        try:
            list_view = self.query_one("#pipeline_content", SelectableListView)
            return list_view.get_selection_state()
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

        # Update SelectableListView content
        try:
            list_view = self.query_one("#pipeline_content", SelectableListView)
            list_view.load_items(self.pipeline_steps)
        except Exception:
            pass

        # Update button states
        self._update_button_states()

    def watch_pipeline_steps(self, steps: List[Dict]) -> None:
        """Automatically update UI when pipeline_steps changes."""
        # Update SelectableListView content
        try:
            list_view = self.query_one("#pipeline_content", SelectableListView)
            list_view.load_items(steps)
        except Exception:
            pass

        # Update button states
        self._update_button_states()

        logger.debug(f"Pipeline steps updated: {len(steps)} steps")

        # Save pipeline changes to plate storage
        self._save_pipeline_to_plate_storage()

    def _save_pipeline_to_plate_storage(self) -> None:
        """Save current pipeline steps to plate storage."""
        if self.current_plate:
            # Update plate pipelines storage
            current_pipelines = dict(self.plate_pipelines)
            current_pipelines[self.current_plate] = list(self.pipeline_steps)
            self.plate_pipelines = current_pipelines
            logger.debug(f"Saved {len(self.pipeline_steps)} steps for plate: {self.current_plate}")

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
    
    def watch_selected_step(self, step_id: str) -> None:
        """Automatically update UI when selected_step changes."""
        self._update_button_states()
        logger.debug(f"Selected step: {step_id}")
    
    def _update_button_states(self) -> None:
        """Update button enabled/disabled states based on current state."""
        try:
            has_plate = bool(self.current_plate)
            has_steps = len(self.pipeline_steps) > 0
            has_selection = bool(self.selected_step)
            
            self.query_one("#add_step").disabled = not has_plate
            self.query_one("#del_step").disabled = not has_steps
            self.query_one("#edit_step").disabled = not has_selection
            self.query_one("#load_pipeline").disabled = not has_plate
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

        def handle_result(result):
            if result:  # User saved new step
                # Add to pipeline steps (this will trigger save to plate storage)
                new_step = {
                    "name": result.name,
                    "type": "function",
                    "func": result.func,
                    "variable_components": result.variable_components,
                    "group_by": result.group_by
                }
                new_steps = self.pipeline_steps + [new_step]
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
        from .selectable_list_view import SelectableListView
        list_view = self.query_one("#pipeline_content", SelectableListView)
        desc = list_view.get_operation_description(selected_items, selection_mode, "delete")
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
    
    def action_edit_step(self) -> None:
        """Handle Edit Step button - now triggers modal."""
        logger.info("Edit Step button pressed")

        if not self.pipeline_steps:
            self.app.current_status = "No steps to edit"
            return

        # For now, edit first step (TODO: implement selection)
        step_to_edit = self.pipeline_steps[0]

        def handle_result(result):
            if result:  # User saved changes
                # Update step in pipeline
                updated_steps = self.pipeline_steps.copy()
                updated_steps[0] = {"name": result.name, "type": "function"}
                self.pipeline_steps = updated_steps
                self.app.current_status = f"Updated step: {result.name}"
            else:
                self.app.current_status = "Edit step cancelled"

        # Create a basic FunctionStep for editing
        # TODO: Convert from dict to proper FunctionStep
        from openhcs.textual_tui.services.function_registry_service import FunctionRegistryService
        registry = FunctionRegistryService()
        default_func = registry.find_default_function()

        edit_step = FunctionStep(
            func=default_func,
            name=step_to_edit.get("name", "Unknown Step"),
            variable_components=[],
            group_by=""
        )

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

        # Launch file browser for .func files
        def handle_result(selected_paths):
            if selected_paths and len(selected_paths) > 0:
                self._load_pipeline_from_file(selected_paths[0])
            else:
                self.app.current_status = "Load pipeline cancelled"

        # Create file browser for .func files
        from openhcs.textual_tui.screens.enhanced_file_browser import EnhancedFileBrowserScreen
        from openhcs.constants.constants import Backend
        from pathlib import Path

        browser = EnhancedFileBrowserScreen(
            file_manager=self.filemanager,
            initial_path=Path.home(),
            backend=Backend.DISK,
            title="Load Pipeline File (.func)"
        )

        self.app.push_screen(browser, handle_result)

    def _load_pipeline_from_file(self, file_path: Path) -> None:
        """Load pipeline from .func file."""
        try:
            # Use pattern file service to load
            from openhcs.textual_tui.services.pattern_file_service import PatternFileService
            pattern_service = PatternFileService(None)  # No state needed for loading

            # Load pattern from file (async operation)
            from textual import work
            self._load_pipeline_worker(file_path, pattern_service)

        except Exception as e:
            logger.error(f"Failed to start pipeline loading: {e}")
            self.app.current_status = f"Failed to load pipeline: {e}"

    @work(exclusive=True)
    async def _load_pipeline_worker(self, file_path: Path, pattern_service) -> None:
        """Background worker for pipeline loading."""
        try:
            # Load pattern from file
            pattern = await pattern_service.load_pattern_from_file(file_path)

            # Convert pattern to pipeline steps
            if isinstance(pattern, list):
                # Convert list of function steps to dict format
                pipeline_steps = []
                for i, step in enumerate(pattern):
                    if hasattr(step, 'name'):
                        step_dict = {
                            "name": step.name,
                            "type": "function",
                            "func": step.func,
                            "variable_components": getattr(step, 'variable_components', []),
                            "group_by": getattr(step, 'group_by', "")
                        }
                        pipeline_steps.append(step_dict)
                    else:
                        # Handle dict format
                        pipeline_steps.append(step)

                # Update pipeline steps
                self.pipeline_steps = pipeline_steps
                self.app.current_status = f"Loaded {len(pipeline_steps)} steps from {file_path.name}"
                logger.info(f"Loaded pipeline with {len(pipeline_steps)} steps for plate '{self.current_plate}'")

            else:
                self.app.current_status = f"Invalid pipeline format in {file_path.name}"
                logger.error(f"Invalid pipeline format: expected list, got {type(pattern)}")

        except Exception as e:
            logger.error(f"Failed to load pipeline from {file_path}: {e}")
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

        # TODO: Implement file save dialog
        # For now, save to default location
        default_filename = f"pipeline_{self.current_plate.replace('/', '_')}.func"
        self._save_pipeline_to_file(Path(default_filename))

    def _save_pipeline_to_file(self, file_path: Path) -> None:
        """Save pipeline to .func file."""
        try:
            # Use pattern file service to save
            from openhcs.textual_tui.services.pattern_file_service import PatternFileService
            pattern_service = PatternFileService(None)  # No state needed for saving

            # Convert pipeline steps to pattern format
            pattern = list(self.pipeline_steps)

            # Save pattern to file (async operation)
            from textual import work
            self._save_pipeline_worker(file_path, pattern, pattern_service)

        except Exception as e:
            logger.error(f"Failed to start pipeline saving: {e}")
            self.app.current_status = f"Failed to save pipeline: {e}"

    @work(exclusive=True)
    async def _save_pipeline_worker(self, file_path: Path, pattern, pattern_service) -> None:
        """Background worker for pipeline saving."""
        try:
            # Save pattern to file
            await pattern_service.save_pattern_to_file(pattern, file_path)

            self.app.current_status = f"Saved pipeline to {file_path.name}"
            logger.info(f"Saved pipeline with {len(pattern)} steps to {file_path}")

        except Exception as e:
            logger.error(f"Failed to save pipeline to {file_path}: {e}")
            self.app.current_status = f"Failed to save pipeline: {e}"
