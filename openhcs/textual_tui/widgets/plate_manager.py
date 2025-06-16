"""
PlateManagerWidget for OpenHCS Textual TUI

Plate management widget with complete button set and reactive state management.
Matches the functionality from the current prompt-toolkit TUI.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Tuple

from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer
from textual.reactive import reactive
from textual.widgets import Button, Static, SelectionList
from textual.widget import Widget
from .button_list_widget import ButtonListWidget, ButtonConfig
from textual import work, on

from openhcs.core.config import GlobalPipelineConfig
from openhcs.io.filemanager import FileManager

logger = logging.getLogger(__name__)


class PlateManagerWidget(ButtonListWidget):
    """
    Plate management widget using Textual reactive state.
    
    Features:
    - Complete button set: Add, Del, Edit, Init, Compile, Run
    - Reactive state management for automatic UI updates
    - Scrollable content area
    - Integration with FileManager backend
    """
    
    # Textual reactive state (automatic UI updates!)
    plates = reactive([])
    selected_plate = reactive("")
    orchestrators = reactive({})  # {plate_path: PipelineOrchestrator}
    plate_configs = reactive({})  # {plate_path: GlobalPipelineConfig}
    
    def __init__(self, filemanager: FileManager, global_config: GlobalPipelineConfig):
        """
        Initialize the plate manager widget.

        Args:
            filemanager: FileManager instance for file operations
            global_config: Global configuration
        """
        # Define button configuration
        button_configs = [
            ButtonConfig("Add", "add_plate"),
            ButtonConfig("Del", "del_plate", disabled=True),
            ButtonConfig("Edit", "edit_plate", disabled=True),
            ButtonConfig("Init", "init_plate", disabled=True),
            ButtonConfig("Compile", "compile_plate", disabled=True),
            ButtonConfig("Run", "run_plate", disabled=True),
        ]

        super().__init__(
            button_configs=button_configs,
            list_id="plate_content",
            container_id="plate_list",
            on_button_pressed=self._handle_button_press,
            on_selection_changed=self._handle_selection_change,
            on_item_moved=self._handle_item_moved
        )

        self.filemanager = filemanager
        # Note: We don't store global_config as it can become stale
        # Always use self.app.global_config to get the current config

        # Simple state management - no reactive property issues
        self.plate_compiled_data = {}  # {plate_path: (execution_pipeline, compiled_contexts)}

        # Callback for plate selection (set by MainContent)
        self.on_plate_selected: Optional[Callable[[str], None]] = None

        # Reference to pipeline editor (set by MainContent)
        self.pipeline_editor: Optional['PipelineEditorWidget'] = None

        # Worker tracking for cancellation support
        self.current_run_worker: Optional[Any] = None

        logger.debug("PlateManagerWidget initialized")
    
    def format_item_for_display(self, plate: Dict) -> Tuple[str, str]:
        """Format plate for display in the list."""
        # Status symbols: ? = added, - = initialized, o = compiled, ! = running, X = error
        status_symbols = {"?": "➕", "-": "✅", "o": "⚡", "!": "🔄", "X": "❌"}
        status_icon = status_symbols.get(plate.get("status", "?"), "❓")
        plate_name = plate.get('name', 'Unknown')
        plate_path = plate.get('path', '')
        display_text = f"{status_icon} {plate_name} - {plate_path}"
        return display_text, plate_path

    def _handle_button_press(self, button_id: str) -> None:
        """Handle button presses from ButtonListWidget."""
        if button_id == "add_plate":
            self.action_add_plate()
        elif button_id == "del_plate":
            self.action_delete_plate()
        elif button_id == "edit_plate":
            self.action_edit_plate()
        elif button_id == "init_plate":
            self.action_init_plate()
        elif button_id == "compile_plate":
            self.action_compile_plate()
        elif button_id == "run_plate":
            # Check if we should run or stop based on current state
            if self._is_any_plate_running():
                self.action_stop_execution()
            else:
                self.action_run_plate()

    def _handle_selection_change(self, selected_values: List[str]) -> None:
        """Handle selection changes from ButtonListWidget (checkmarks only)."""
        # This handles multi-selection (checkmarks) for operations like Init/Compile/Run
        # Pipeline editor responds to blue highlight via watch_highlighted_item()
        logger.debug(f"Checkmarks changed: {len(selected_values)} items selected")

    def _handle_item_moved(self, from_index: int, to_index: int) -> None:
        """Handle item movement from ButtonListWidget."""
        current_plates = list(self.plates)

        # Move the plate
        plate = current_plates.pop(from_index)
        current_plates.insert(to_index, plate)

        # Update plates list
        self.plates = current_plates

        plate_name = plate['name']
        direction = "up" if to_index < from_index else "down"
        self.app.current_status = f"Moved plate '{plate_name}' {direction}"





    def on_mount(self) -> None:
        """Called when the widget is mounted - ensure display is up to date."""
        # Schedule multiple update attempts to ensure it works
        self.call_later(self._delayed_update_display)
        self.set_timer(0.1, self._delayed_update_display)
        self.call_later(self._update_button_states)
    

    
    def watch_plates(self, plates: List[Dict]) -> None:
        """Automatically update UI when plates reactive property changes."""
        try:
            logger.info(f"watch_plates called with {len(plates)} plates")
            for plate in plates:
                logger.info(f"  - {plate['name']}: status={plate.get('status', '?')}")

            # Update ButtonListWidget items - this will trigger the parent's watch_items
            # Force a new list to ensure reactive update is triggered
            self.items = list(plates)
            # Also explicitly trigger the reactive update for items
            self.mutate_reactive(ButtonListWidget.items)


        except Exception as e:
            # Show global error for any unexpected exceptions
            self.app.show_error(f"Error in watch_plates: {str(e)}", e)

        # Update button states
        self._update_button_states()

    
    def watch_highlighted_item(self, plate_path: str) -> None:
        """Automatically update pipeline editor when highlighted_item changes (blue highlight)."""
        # Update selected_plate for pipeline editor
        self.selected_plate = plate_path
        logger.debug(f"Highlighted plate: {plate_path}")

    def watch_selected_plate(self, plate_path: str) -> None:
        """Automatically update UI when selected_plate changes."""
        self._update_button_states()

        # Notify parent about selection
        if self.on_plate_selected and plate_path:
            self.on_plate_selected(plate_path)

        logger.debug(f"Selected plate: {plate_path}")





    def get_selection_state(self) -> tuple[List[Dict], str]:
        """Get current selection state - supports both single and multi-selection."""
        try:
            # Get the SelectionList widget to check for multi-selection (checkmarks)
            selection_list = self.query_one(f"#{self.list_id}")
            multi_selected_values = selection_list.selected

            if multi_selected_values:
                # Multi-selection mode (checkmarks) - return all checked items
                selected_items = []
                for plate in self.plates:
                    if plate.get('path') in multi_selected_values:
                        selected_items.append(plate)
                return selected_items, "checkbox"

            elif self.selected_plate:
                # Single selection mode (blue highlight) - return highlighted item
                selected_items = []
                for plate in self.plates:
                    if plate.get('path') == self.selected_plate:
                        selected_items.append(plate)
                        break
                return selected_items, "cursor"

            else:
                return [], "empty"

        except Exception as e:
            logger.warning(f"Failed to get selection state: {e}")
            # Fallback to single selection mode
            if self.selected_plate:
                selected_items = []
                for plate in self.plates:
                    if plate.get('path') == self.selected_plate:
                        selected_items.append(plate)
                        break
                return selected_items, "cursor"
            else:
                return [], "empty"

    def get_operation_description(self, selected_items: List[Dict], selection_mode: str, operation: str) -> str:
        """Generate human-readable description of what will be operated on."""
        count = len(selected_items)
        if selection_mode == "empty":
            return f"No items available for {operation}"
        elif selection_mode == "all":
            return f"{operation.title()} ALL {count} items"
        elif selection_mode == "checkbox":
            if count == 1:
                item_name = selected_items[0].get('name', 'Unknown')
                return f"{operation.title()} selected item: {item_name}"
            else:
                return f"{operation.title()} {count} selected items"
        else:
            return f"{operation.title()} {count} items"

    def _delayed_update_display(self) -> None:
        """Update the plate display - called when widget is mounted or as fallback."""
        try:
            # Trigger the ButtonListWidget's watch_items method
            self.mutate_reactive(PlateManagerWidget.items)
        except Exception as e:
            logger.warning(f"Delayed update failed (widget may not be ready): {e}")
            # Try again in a moment using proper Textual API
            self.set_timer(0.1, self._delayed_update_display)

    def _update_button_states(self) -> None:
        """Update button enabled/disabled states based on mathematical constraints."""
        try:
            has_plates = len(self.plates) > 0
            has_selection = bool(self.selected_plate)

            # Get selected plate for constraint checking
            selected_plate = None
            if self.selected_plate:
                for plate in self.plates:
                    if plate.get('path') == self.selected_plate:
                        selected_plate = plate
                        break

            # Check if any plates are currently running
            is_running = self._is_any_plate_running()

            # Mathematical constraints (like test_main.py workflow)
            # Init: available for plates that need initialization (not blocked by failures)
            can_init = selected_plate and selected_plate.get('status') in ['?', '-', 'F']

            # Compile: available for plates that are not compiled and have pipelines
            can_compile = (selected_plate and
                          selected_plate.get('path') not in self.plate_compiled_data and
                          self._has_pipelines([selected_plate]))

            # Run/Stop button: available if ANY plates are compiled OR if execution is running
            can_run_or_stop = len(self.plate_compiled_data) > 0 or is_running

            # Update button labels and states
            run_button = self.query_one("#run_plate")
            if is_running:
                run_button.label = "Stop"
                run_button.disabled = False  # Always enabled when running
            else:
                run_button.label = "Run"
                run_button.disabled = not (has_selection and can_run_or_stop)

            self.query_one("#del_plate").disabled = not has_plates
            self.query_one("#edit_plate").disabled = not has_selection
            self.query_one("#init_plate").disabled = not (has_selection and can_init)
            self.query_one("#compile_plate").disabled = not (has_selection and can_compile)
        except Exception as e:
            # Buttons might not be mounted yet
            logger.warning(f"Failed to update button states: {e}")

    def _is_any_plate_running(self) -> bool:
        """Check if any plates are currently running."""
        return any(plate.get('status') == '!' for plate in self.plates)

    def action_stop_execution(self) -> None:
        """Handle Stop button - cancel running execution."""
        logger.info("🛑 STOP BUTTON PRESSED")

        if self.current_run_worker and not self.current_run_worker.is_finished:
            # Cancel the running worker
            self.current_run_worker.cancel()
            logger.info("🛑 Cancelled running worker")

            # Update status of running plates to indicate cancellation
            current_plates = list(self.plates)
            cancelled_count = 0
            for plate in current_plates:
                if plate.get('status') == '!':  # Running
                    plate['status'] = 'F'  # Mark as failed (cancelled)
                    cancelled_count += 1

            if cancelled_count > 0:
                self.plates = current_plates
                self.app.current_status = f"Cancelled execution of {cancelled_count} running plates"
            else:
                self.app.current_status = "No running plates to cancel"
        else:
            self.app.current_status = "No active execution to cancel"

        # Clear the worker reference
        self.current_run_worker = None

    def action_add_plate(self) -> None:
        """Handle Add Plate button."""
        self.app.push_screen(self._create_file_browser_screen(), self._on_plate_directory_selected)

    def _create_file_browser_screen(self) -> Any:
        """Create enhanced file browser screen for plate selection with path caching."""
        from openhcs.textual_tui.screens.enhanced_file_browser import EnhancedFileBrowserScreen
        from openhcs.constants.constants import Backend
        from openhcs.textual_tui.utils.path_cache import get_path_cache, PathCacheKey
        from pathlib import Path

        # Get cached path for better UX - remembers last used directory
        path_cache = get_path_cache()
        initial_path = path_cache.get_initial_path(PathCacheKey.PLATE_IMPORT, Path.home())

        # Create enhanced file browser for directory selection
        return EnhancedFileBrowserScreen(
            file_manager=self.filemanager,
            initial_path=initial_path,
            backend=Backend.DISK,
            title="Select Plate Directory"
        )

    def _on_plate_directory_selected(self, selected_paths: Any) -> None:
        """Handle directory selection from file browser."""
        if selected_paths is None:
            self.app.current_status = "Plate selection cancelled"
            return

        # Handle both single path and list of paths
        if not isinstance(selected_paths, list):
            selected_paths = [selected_paths]

        added_plates = []
        current_plates = list(self.plates)

        for selected_path in selected_paths:
            # Ensure selected_path is a Path object
            if isinstance(selected_path, str):
                selected_path = Path(selected_path)
            elif not isinstance(selected_path, Path):
                selected_path = Path(str(selected_path))


            # Check if plate already exists
            if any(plate['path'] == str(selected_path) for plate in current_plates):
                continue

            # Create orchestrator for the plate
            plate_path = str(selected_path)
            plate_config = self.plate_configs.get(plate_path, self.app.global_config)

            try:
                # Import orchestrator
                from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator

                # Create orchestrator (construction only, no initialization)
                orchestrator = PipelineOrchestrator(
                    plate_path=plate_path,
                    global_config=plate_config,
                    storage_registry=self.filemanager.registry
                )

                # Store orchestrator
                current_orchestrators = dict(self.orchestrators)
                current_orchestrators[plate_path] = orchestrator
                self.orchestrators = current_orchestrators

                # Add the plate to the list
                plate_name = selected_path.name
                plate_entry = {
                    'name': plate_name,
                    'path': plate_path,
                    'status': '?'  # Added but not initialized
                }

                current_plates.append(plate_entry)
                added_plates.append(plate_name)

            except Exception as e:
                logger.error(f"Failed to create orchestrator for {selected_path}: {e}")
                # Still add the plate but with error status
                plate_name = selected_path.name
                plate_entry = {
                    'name': plate_name,
                    'path': str(selected_path),
                    'status': 'X',  # Error status
                    'error': str(e)
                }
                current_plates.append(plate_entry)
                added_plates.append(f"{plate_name} (ERROR)")
                logger.warning(f"Added plate without orchestrator: {plate_name}")

        # Cache the parent directory for next time (save user navigation time)
        if selected_paths:
            from openhcs.textual_tui.utils.path_cache import get_path_cache, PathCacheKey
            # Use parent of first selected path as the cached directory
            first_path = selected_paths[0] if isinstance(selected_paths[0], Path) else Path(selected_paths[0])
            parent_dir = first_path.parent
            get_path_cache().set_cached_path(PathCacheKey.PLATE_IMPORT, parent_dir)

        # Update plates list using reactive property (triggers automatic UI update)
        self.plates = current_plates

        if added_plates:
            if len(added_plates) == 1:
                logger.info(f"Added plate: {added_plates[0]}")
            else:
                logger.info(f"Added {len(added_plates)} plates: {', '.join(added_plates)}")
        else:
            logger.info("No new plates added (duplicates skipped)")
    
    def action_delete_plate(self) -> None:
        """Handle Delete Plate button - delete selected plates with orchestrator cleanup."""

        # Get current selection state
        selected_items, selection_mode = self.get_selection_state()

        if selection_mode == "empty":
            logger.warning("No plates available for deletion")
            return

        # Generate description and perform deletion
        desc = self.get_operation_description(selected_items, selection_mode, "delete")

        # Clean up orchestrators and remove items
        current_plates = list(self.plates)
        current_orchestrators = dict(self.orchestrators)
        current_configs = dict(self.plate_configs)

        for item in selected_items:
            plate_path = item['path']

            # Remove orchestrator if it exists
            if plate_path in current_orchestrators:
                del current_orchestrators[plate_path]

            # Remove plate-specific config if it exists
            if plate_path in current_configs:
                del current_configs[plate_path]

            # Remove from plates list
            current_plates = [p for p in current_plates if p['path'] != plate_path]

        # Update reactive properties
        self.plates = current_plates
        self.orchestrators = current_orchestrators
        self.plate_configs = current_configs

        self.app.current_status = f"Deleted {len(selected_items)} plates"
    
    def action_edit_plate(self) -> None:
        """Handle Edit Plate button - edit configuration for selected plate."""

        # Get current selection state
        selected_items, selection_mode = self.get_selection_state()

        if selection_mode == "empty":
            self.app.current_status = "No plates available for editing"
            return

        if selection_mode != "cursor" or len(selected_items) != 1:
            self.app.current_status = "Please select exactly one plate to edit"
            return

        plate_data = selected_items[0]
        plate_path = plate_data['path']

        # Get current config for this plate (or global default)
        current_config = self.plate_configs.get(plate_path, self.app.global_config)

        # Launch configuration form
        def handle_result(result_config: Any) -> None:
            if result_config:
                # Save plate-specific config
                current_configs = dict(self.plate_configs)
                current_configs[plate_path] = result_config
                self.plate_configs = current_configs

                self.app.current_status = f"Updated configuration for {plate_data['name']}"
            else:
                self.app.current_status = "Configuration edit cancelled"

        # Create and show config form
        from openhcs.textual_tui.screens.config_form import ConfigFormScreen

        config_form = ConfigFormScreen(
            plate_path=plate_path,
            current_config=current_config,
            title="Plate Configuration"
        )

        self.app.push_screen(config_form, handle_result)
    
    def action_init_plate(self) -> None:
        """Handle Init Plate button - initialize selected plates."""

        # Get current selection state
        selected_items, selection_mode = self.get_selection_state()

        if selection_mode == "empty":
            logger.warning("No plates available for initialization")
            return

        # Validate all selected plates can be initialized (allow failed plates to be re-initialized)
        invalid_plates = [item for item in selected_items if item.get('status') not in ['?', '-', 'F']]
        if invalid_plates:
            names = [item['name'] for item in invalid_plates]
            logger.warning(f"Cannot initialize plates with invalid status: {', '.join(names)}")
            return

        # Start async initialization
        self._start_async_init(selected_items, selection_mode)

    def _start_async_init(self, selected_items: List[Dict], selection_mode: str) -> None:
        """Start async initialization of selected plates."""
        from textual import work

        # Generate operation description
        desc = self.get_operation_description(selected_items, selection_mode, "initialize")
        logger.info(f"Initializing: {desc}")

        # Start background worker
        self._init_plates_worker(selected_items)

    @work(exclusive=True)
    async def _init_plates_worker(self, selected_items: List[Dict]) -> None:
        """Background worker for plate initialization."""
        for plate_data in selected_items:
            plate_path = plate_data['path']

            # Find the actual plate in self.plates (not the copy from get_selection_state)
            actual_plate = None
            for plate in self.plates:
                if plate['path'] == plate_path:
                    actual_plate = plate
                    break

            if not actual_plate:
                logger.error(f"Plate not found in plates list: {plate_path}")
                continue

            # Get orchestrator
            orchestrator = self.orchestrators.get(plate_path)
            if not orchestrator:
                logger.error(f"No orchestrator found for {plate_path}")
                actual_plate['status'] = 'X'
                actual_plate['error'] = "No orchestrator found"
                # Force UI update immediately for error state
                self.mutate_reactive(PlateManagerWidget.plates)
                continue

            # Check if already initialized
            if orchestrator.is_initialized():
                actual_plate['status'] = '-'
                # Force UI update immediately
                self.mutate_reactive(PlateManagerWidget.plates)
                continue

            # Initialize orchestrator (heavy operation - run in executor to avoid blocking UI)
            await asyncio.get_event_loop().run_in_executor(None, orchestrator.initialize)
            actual_plate['status'] = '-'  # Initialized
            logger.info(f"Set plate {actual_plate['name']} status to '-' (initialized)")

            # Force UI update immediately after each plate
            self.mutate_reactive(PlateManagerWidget.plates)
            # Update button states immediately
            self._update_button_states()
            # Notify pipeline editor of status change
            self._notify_pipeline_editor_status_change(actual_plate['path'], actual_plate['status'])
            logger.info(f"Called mutate_reactive for plate {actual_plate['name']}")

        # Final UI update
        self.mutate_reactive(PlateManagerWidget.plates)
        # Update button states after all plates processed
        self._update_button_states()

        # Update status
        success_count = len([p for p in selected_items if p.get('status') == '-'])
        error_count = len([p for p in selected_items if p.get('status') == 'X'])

        if error_count == 0:
            logger.info(f"Successfully initialized {success_count} plates")
        else:
            logger.warning(f"Initialized {success_count} plates, {error_count} errors")


    
    def action_compile_plate(self) -> None:
        """Handle Compile Plate button - compile pipelines for selected plates."""

        # Get current selection state
        selected_items, selection_mode = self.get_selection_state()

        if selection_mode == "empty":
            logger.warning("No plates available for compilation")
            return

        # Validate all selected plates are initialized (allow failed plates to be re-compiled)
        uninitialized = [item for item in selected_items if item.get('status') not in ['-', 'F']]
        if uninitialized:
            names = [item['name'] for item in uninitialized]
            logger.warning(f"Cannot compile uninitialized plates: {', '.join(names)}")
            return

        # Validate all selected plates have pipelines
        no_pipeline = []
        for item in selected_items:
            pipeline = self._get_current_pipeline_definition(item['path'])
            if not pipeline:
                no_pipeline.append(item)

        if no_pipeline:
            names = [item['name'] for item in no_pipeline]
            self.app.current_status = f"Cannot compile plates without pipelines: {', '.join(names)}"
            return

        # Start async compilation
        self._start_async_compile(selected_items, selection_mode)

    def _start_async_compile(self, selected_items: List[Dict], selection_mode: str) -> None:
        """Start async compilation of selected plates."""
        from textual import work

        # Generate operation description
        desc = self.get_operation_description(selected_items, selection_mode, "compile")
        logger.info(f"Compiling: {desc}")

        # Start background worker
        self._compile_plates_worker(selected_items)

    @work(exclusive=True)
    async def _compile_plates_worker(self, selected_items: List[Dict]) -> None:
        """Background worker for plate compilation."""
        for plate_data in selected_items:
            plate_path = plate_data['path']

            # Find the actual plate in self.plates (not the copy from get_selection_state)
            actual_plate = None
            for plate in self.plates:
                if plate['path'] == plate_path:
                    actual_plate = plate
                    break

            if not actual_plate:
                logger.error(f"Plate not found in plates list: {plate_path}")
                continue

            # Get orchestrator
            orchestrator = self.orchestrators.get(plate_path)
            if not orchestrator:
                logger.error(f"No orchestrator found for {plate_path}")
                actual_plate['status'] = 'X'
                actual_plate['error'] = "No orchestrator found"
                # Force UI update immediately for error state
                self.mutate_reactive(PlateManagerWidget.plates)
                continue

            # Get definition pipeline and make fresh copy
            definition_pipeline = self._get_current_pipeline_definition(plate_path)
            if not definition_pipeline:
                logger.warning(f"No pipeline defined for {actual_plate['name']}, using empty pipeline")
                definition_pipeline = []

            try:
                # Make fresh copy for compilation
                import copy
                from openhcs.constants.constants import VariableComponents
                execution_pipeline = copy.deepcopy(definition_pipeline)

                # Fix step IDs after deep copy to match new object IDs
                for step in execution_pipeline:
                    step.step_id = str(id(step))
                    # Debug: Check for None variable_components
                    if step.variable_components is None:
                        logger.warning(f"🔥 Step '{step.name}' has None variable_components, setting default")
                        step.variable_components = [VariableComponents.SITE]

                # Get wells and compile (async - run in executor to avoid blocking UI)
                # Wrap in Pipeline object like test_main.py does
                from openhcs.core.pipeline import Pipeline
                pipeline_obj = Pipeline(steps=execution_pipeline)

                # Run heavy operations in executor to avoid blocking UI
                wells = await asyncio.get_event_loop().run_in_executor(None, orchestrator.get_wells)
                compiled_contexts = await asyncio.get_event_loop().run_in_executor(
                    None, orchestrator.compile_pipelines, pipeline_obj.steps, wells
                )

                # Store state simply - no reactive property issues
                step_ids_in_pipeline = [id(step) for step in execution_pipeline]
                # Get step IDs from contexts (ProcessingContext objects)
                first_well_key = list(compiled_contexts.keys())[0] if compiled_contexts else None
                step_ids_in_contexts = list(compiled_contexts[first_well_key].step_plans.keys()) if first_well_key and hasattr(compiled_contexts[first_well_key], 'step_plans') else []
                logger.info(f"🔥 Storing compiled data for {plate_path}: pipeline={type(execution_pipeline)}, contexts={type(compiled_contexts)}")
                logger.info(f"🔥 Step IDs in pipeline: {step_ids_in_pipeline}")
                logger.info(f"🔥 Step IDs in contexts: {step_ids_in_contexts}")
                self.plate_compiled_data[plate_path] = (execution_pipeline, compiled_contexts)
                logger.info(f"🔥 Stored! Available compiled plates: {list(self.plate_compiled_data.keys())}")

                # Update plate status ONLY on successful compilation
                actual_plate['status'] = 'o'  # Compiled
                logger.info(f"🔥 Successfully compiled {plate_path}")

            except Exception as e:
                logger.error(f"🔥 Compilation failed for {plate_path}: {e}")
                actual_plate['status'] = 'F'  # Failed
                actual_plate['error'] = str(e)
                # Don't store anything in plate_compiled_data on failure

            # Force UI update immediately after each plate
            self.mutate_reactive(PlateManagerWidget.plates)
            # Update button states immediately
            self._update_button_states()
            # Notify pipeline editor of status change
            self._notify_pipeline_editor_status_change(actual_plate['path'], actual_plate['status'])

        # Final UI update
        self.mutate_reactive(PlateManagerWidget.plates)
        # Update button states after all plates processed
        self._update_button_states()

        # Update status
        success_count = len([p for p in selected_items if p.get('status') == 'o'])
        error_count = len([p for p in selected_items if p.get('status') == 'X'])

        if error_count == 0:
            logger.info(f"Successfully compiled {success_count} plates")
        else:
            logger.warning(f"Compiled {success_count} plates, {error_count} errors")



    def get_plate_status(self, plate_path: str) -> str:
        """Get status for specific plate."""
        for plate in self.plates:
            if plate.get('path') == plate_path:
                return plate.get('status', '?')
        return '?'  # Default to added status

    def _has_pipelines(self, plates: List[Dict]) -> bool:
        """Check if all plates have pipeline definitions."""
        if not self.pipeline_editor:
            return False

        for plate in plates:
            pipeline = self.pipeline_editor.get_pipeline_for_plate(plate['path'])
            if not pipeline:
                return False
        return True

    def _notify_pipeline_editor_status_change(self, plate_path: str, new_status: str) -> None:
        """Notify pipeline editor when plate status changes (enables Add button immediately)."""
        if self.pipeline_editor and self.pipeline_editor.current_plate == plate_path:
            # Update pipeline editor's status and trigger button state update
            self.pipeline_editor.current_plate_status = new_status



    def _get_current_pipeline_definition(self, plate_path: str = None) -> List:
        """Get current pipeline definition from PipelineEditor (now returns FunctionStep objects directly)."""
        if not self.pipeline_editor:
            logger.warning("No pipeline editor reference - using empty pipeline")
            return []

        # Get pipeline for specific plate or current plate
        target_plate = plate_path or self.pipeline_editor.current_plate
        if not target_plate:
            logger.warning("No plate specified - using empty pipeline")
            return []

        # Get pipeline from editor (now returns List[FunctionStep] directly)
        pipeline_steps = self.pipeline_editor.get_pipeline_for_plate(target_plate)

        # No conversion needed - pipeline_steps are already FunctionStep objects with memory type decorators
        return pipeline_steps
    
    def action_run_plate(self) -> None:
        """Handle Run Plate button - execute compiled plates (like test_main.py)."""
        logger.info("🔥 RUN BUTTON PRESSED")

        # Get current selection state
        selected_items, selection_mode = self.get_selection_state()
        logger.info(f"🔥 Selection mode: {selection_mode}, items: {len(selected_items)}")

        if selection_mode == "empty":
            # No selection - run ALL ready plates (like test_main.py batch execution)
            ready_plates = [p for p in self.plates
                           if p.get('status') in ['o', 'C', 'F'] and p.get('compiled_contexts') and p.get('compiled_pipeline_definition')]
            logger.info(f"🔥 No selection - found {len(ready_plates)} ready plates out of {len(self.plates)} total")
            for plate in self.plates:
                logger.info(f"🔥 Plate {plate.get('name')}: status={plate.get('status')}, "
                           f"has_contexts={bool(plate.get('compiled_contexts'))}, "
                           f"has_compiled_pipeline={bool(plate.get('compiled_pipeline_definition'))}")

            if ready_plates:
                logger.info(f"Running all {len(ready_plates)} ready plates")
                logger.info(f"🔥 Starting async run for {len(ready_plates)} plates")
                self._start_async_run(ready_plates, "all_ready")
            else:
                logger.warning("No plates ready for execution")
                logger.warning("🔥 No plates ready for execution")
            return

        # Selection exists - filter to only compiled plates
        ready_items = [item for item in selected_items if item.get('path') in self.plate_compiled_data]

        logger.info(f"🔥 Selection exists - {len(ready_items)} ready out of {len(selected_items)} selected")
        for item in selected_items:
            has_compiled_data = item.get('path') in self.plate_compiled_data
            logger.info(f"🔥 Selected {item.get('name')}: status={item.get('status')}, "
                       f"has_compiled_data={has_compiled_data}")

        if not ready_items:
            logger.warning("No selected plates are ready for execution")
            logger.warning("🔥 No selected plates ready for execution")
            return

        # Run only the ready selected plates
        skipped_count = len(selected_items) - len(ready_items)
        if skipped_count > 0:
            logger.info(f"Running {len(ready_items)} ready plates (skipped {skipped_count} unready)")
        else:
            logger.info(f"Running {len(ready_items)} selected plates")

        logger.info(f"🔥 Starting async run for {len(ready_items)} selected plates")
        self._start_async_run(ready_items, selection_mode)

    def _start_async_run(self, selected_items: List[Dict], selection_mode: str) -> None:
        """Start async execution of selected plates."""
        from textual import work

        logger.info(f"🔥 _start_async_run called with {len(selected_items)} items, mode: {selection_mode}")

        # Generate operation description
        desc = self.get_operation_description(selected_items, selection_mode, "run")
        logger.info(f"Running: {desc}")
        logger.info(f"🔥 Status set to: Running: {desc}")

        # Start background worker and track it for cancellation
        logger.info("🔥 Starting background worker _run_plates_worker")
        self.current_run_worker = self._run_plates_worker(selected_items)

        # Update button states immediately to show Stop button
        self._update_button_states()

    @work(exclusive=True)
    async def _run_plates_worker(self, selected_items: List[Dict]) -> None:
        """Run plates using orchestrator's execute method."""
        from textual.worker import get_current_worker
        import asyncio

        worker = get_current_worker()

        try:
            for plate_data in selected_items:
                # Check for cancellation before processing each plate
                if worker.is_cancelled:
                    logger.info("🛑 Worker cancelled, stopping execution")
                    break
            plate_path = plate_data['path']

            # Get orchestrator
            orchestrator = self.orchestrators.get(plate_path)
            if not orchestrator:
                continue

            # Find actual plate for status updates
            actual_plate = None
            for plate in self.plates:
                if plate['path'] == plate_path:
                    actual_plate = plate
                    break
            if not actual_plate:
                continue

            # Get compiled data from simple state
            logger.info(f"🔥 Checking compiled data for {plate_path}")
            logger.info(f"🔥 Available compiled plates: {list(self.plate_compiled_data.keys())}")

            if plate_path not in self.plate_compiled_data:
                logger.warning(f"🔥 No compiled data found for {plate_path}")
                continue  # Skip - no compiled data

            execution_pipeline, compiled_contexts = self.plate_compiled_data[plate_path]
            step_ids_in_pipeline = [id(step) for step in execution_pipeline]
            # Get step IDs from contexts (ProcessingContext objects)
            first_well_key = list(compiled_contexts.keys())[0] if compiled_contexts else None
            step_ids_in_contexts = list(compiled_contexts[first_well_key].step_plans.keys()) if first_well_key and hasattr(compiled_contexts[first_well_key], 'step_plans') else []
            logger.info(f"🔥 Retrieved compiled data for {plate_path}: pipeline={type(execution_pipeline)}, contexts={type(compiled_contexts)}")
            logger.info(f"🔥 Step IDs in retrieved pipeline: {step_ids_in_pipeline}")
            logger.info(f"🔥 Step IDs in retrieved contexts: {step_ids_in_contexts}")

                # Run it
                actual_plate['status'] = '!'  # Running
                self.mutate_reactive(PlateManagerWidget.plates)
                # Update button states to show Stop button
                self._update_button_states()

                try:
                    # Check for cancellation before starting execution
                    if worker.is_cancelled:
                        logger.info("🛑 Worker cancelled before execution")
                        actual_plate['status'] = 'F'  # Cancelled
                        break

                    # Execute like test_main.py (async - run in executor to avoid blocking UI)
                    results = await asyncio.get_event_loop().run_in_executor(
                        None, orchestrator.execute_compiled_plate, execution_pipeline, compiled_contexts
                    )

                    # Check for cancellation after execution
                    if worker.is_cancelled:
                        logger.info("🛑 Worker cancelled after execution")
                        actual_plate['status'] = 'F'  # Cancelled
                        break

                    # Update status based on results
                    if results and all(r.get('status') != 'error' for r in results.values()):
                        actual_plate['status'] = 'C'  # Success
                    else:
                        actual_plate['status'] = 'F'  # Failed

                except asyncio.CancelledError:
                    logger.info("🛑 Execution cancelled via CancelledError")
                    actual_plate['status'] = 'F'  # Cancelled
                    break
                except Exception:
                    actual_plate['status'] = 'F'  # Failed

            finally:
                # 🔥 COMPREHENSIVE GPU CLEANUP: Clear all GPU memory after plate execution
                try:
                    from openhcs.core.memory.gpu_cleanup import cleanup_all_gpu_frameworks
                    cleanup_all_gpu_frameworks()
                    logger.info(f"🔥 COMPREHENSIVE GPU CLEANUP: Cleared all GPU frameworks after plate execution: {plate_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to perform comprehensive GPU cleanup: {cleanup_error}")

                # Reset memory backend after each plate execution to prevent key collisions
                # TEMPORARILY DISABLED FOR TESTING - checking if reset timing is causing pattern detection issues
                from openhcs.io.base import reset_memory_backend
                reset_memory_backend()
                logger.info(f"🔥 Memory backend reset DISABLED for testing: {plate_path}")

                self.mutate_reactive(PlateManagerWidget.plates)

        except asyncio.CancelledError:
            logger.info("🛑 Worker cancelled at top level")
            # Update any running plates to cancelled state
            current_plates = list(self.plates)
            for plate in current_plates:
                if plate.get('status') == '!':  # Running
                    plate['status'] = 'F'  # Mark as cancelled
            self.plates = current_plates
            raise  # Re-raise to properly handle cancellation

        finally:
            # Clear the worker reference and update button states
            self.current_run_worker = None
            self._update_button_states()
            logger.info("🔥 Worker finished, button states updated")
