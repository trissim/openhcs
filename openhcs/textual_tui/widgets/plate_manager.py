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
from textual import work

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
        self.global_config = global_config

        # Callback for plate selection (set by MainContent)
        self.on_plate_selected: Optional[Callable[[str], None]] = None

        # Reference to pipeline editor (set by MainContent)
        self.pipeline_editor: Optional['PipelineEditorWidget'] = None

        logger.debug("PlateManagerWidget initialized")
    
    def format_item_for_display(self, plate: Dict) -> Tuple[str, str]:
        """Format plate for display in the list."""
        # Status symbols: ? = added, - = initialized, o = compiled, X = error
        status_symbols = {"?": "➕", "-": "✅", "o": "⚡", "X": "❌"}
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
            self.action_run_plate()

    def _handle_selection_change(self, selected_values: List[str]) -> None:
        """Handle selection changes from ButtonListWidget."""
        # Update selected_plate - use first selected item if any
        if selected_values:
            self.selected_plate = selected_values[0]  # This is the plate path
        else:
            self.selected_plate = ""

        # Notify parent about selection
        if self.on_plate_selected and self.selected_plate:
            self.on_plate_selected(self.selected_plate)

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
        logger.info(f"Moved plate '{plate_name}' from index {from_index} to {to_index}")

    def on_mount(self) -> None:
        """Called when the widget is mounted - ensure display is up to date."""
        logger.info("PlateManagerWidget mounted, updating display")
        # Schedule multiple update attempts to ensure it works
        self.call_later(self._delayed_update_display)
        self.set_timer(0.1, self._delayed_update_display)
        self.call_later(self._update_button_states)
    

    
    def watch_plates(self, plates: List[Dict]) -> None:
        """Automatically update UI when plates reactive property changes."""
        try:
            logger.info(f"watch_plates called with {len(plates)} plates: {[p.get('name', 'Unknown') for p in plates]}")

            # Update ButtonListWidget items - this will trigger the parent's watch_items
            self.items = plates

            logger.info(f"✅ Updated plate list successfully - now showing {len(plates)} plates")

        except Exception as e:
            # Show global error for any unexpected exceptions
            self.app.show_error(f"Error in watch_plates: {str(e)}", e)

        # Update button states
        self._update_button_states()

        logger.info(f"Plates UI update complete: {len(plates)} plates")
    
    def watch_selected_plate(self, plate_path: str) -> None:
        """Automatically update UI when selected_plate changes."""
        self._update_button_states()
        
        # Notify parent about selection
        if self.on_plate_selected and plate_path:
            self.on_plate_selected(plate_path)
        
        logger.debug(f"Selected plate: {plate_path}")





    def get_selection_state(self) -> tuple[List[Dict], str]:
        """Get current selection state."""
        # Use the selected_plate from ButtonListWidget
        if self.selected_plate:
            # Find the selected plate
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
            logger.info(f"✅ Delayed plate list update successful - showing {len(self.plates)} plates")
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

            # Mathematical constraints
            can_init = selected_plate and selected_plate.get('status') in ['?', '-']
            can_compile = selected_plate and selected_plate.get('status') == '-' and self._has_pipelines([selected_plate])
            can_run = selected_plate and selected_plate.get('status') == 'o'

            self.query_one("#del_plate").disabled = not has_plates
            self.query_one("#edit_plate").disabled = not has_selection
            self.query_one("#init_plate").disabled = not (has_selection and can_init)
            self.query_one("#compile_plate").disabled = not (has_selection and can_compile)
            self.query_one("#run_plate").disabled = not (has_selection and can_run)
        except Exception as e:
            # Buttons might not be mounted yet
            logger.warning(f"Failed to update button states: {e}")
    

    
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

            logger.info(f"Processing plate directory: {selected_path}")

            # Check if plate already exists
            if any(plate['path'] == str(selected_path) for plate in current_plates):
                logger.info(f"Plate {selected_path.name} already exists, skipping")
                continue

            # Create orchestrator for the plate
            plate_path = str(selected_path)
            plate_config = self.plate_configs.get(plate_path, self.global_config)

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
                logger.info(f"Added plate with orchestrator: {plate_name} at {selected_path}")

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
                self.app.current_status = f"Added plate: {added_plates[0]}"
            else:
                self.app.current_status = f"Added {len(added_plates)} plates: {', '.join(added_plates)}"
        else:
            self.app.current_status = "No new plates added (duplicates skipped)"
    
    def action_delete_plate(self) -> None:
        """Handle Delete Plate button - delete selected plates with orchestrator cleanup."""

        # Get current selection state
        selected_items, selection_mode = self.get_selection_state()

        if selection_mode == "empty":
            self.app.current_status = "No plates available for deletion"
            return

        # Generate description and perform deletion
        desc = self.get_operation_description(selected_items, selection_mode, "delete")
        logger.info(f"Deleting: {desc}")

        # Clean up orchestrators and remove items
        current_plates = list(self.plates)
        current_orchestrators = dict(self.orchestrators)
        current_configs = dict(self.plate_configs)

        for item in selected_items:
            plate_path = item['path']

            # Remove orchestrator if it exists
            if plate_path in current_orchestrators:
                del current_orchestrators[plate_path]
                logger.info(f"Cleaned up orchestrator for: {item['name']}")

            # Remove plate-specific config if it exists
            if plate_path in current_configs:
                del current_configs[plate_path]
                logger.info(f"Cleaned up config for: {item['name']}")

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
        current_config = self.plate_configs.get(plate_path, self.global_config)

        # Launch configuration form
        def handle_result(result_config: Any) -> None:
            if result_config:
                # Save plate-specific config
                current_configs = dict(self.plate_configs)
                current_configs[plate_path] = result_config
                self.plate_configs = current_configs

                self.app.current_status = f"Updated configuration for {plate_data['name']}"
                logger.info(f"Saved configuration for plate: {plate_path}")
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
            self.app.current_status = "No plates available for initialization"
            return

        # Validate all selected plates can be initialized
        invalid_plates = [item for item in selected_items if item.get('status') not in ['?', '-']]
        if invalid_plates:
            names = [item['name'] for item in invalid_plates]
            self.app.current_status = f"Cannot initialize plates with invalid status: {', '.join(names)}"
            return

        # Start async initialization
        self._start_async_init(selected_items, selection_mode)

    def _start_async_init(self, selected_items: List[Dict], selection_mode: str) -> None:
        """Start async initialization of selected plates."""
        from textual import work

        # Generate operation description
        desc = self.get_operation_description(selected_items, selection_mode, "initialize")
        logger.info(f"Starting initialization: {desc}")
        self.app.current_status = f"Initializing: {desc}"

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
                continue

            # Check if already initialized
            if orchestrator.is_initialized():
                logger.info(f"Orchestrator for {actual_plate['name']} already initialized")
                actual_plate['status'] = '-'
                continue

            # Initialize orchestrator (heavy operation - run in executor to avoid blocking UI)
            await asyncio.get_event_loop().run_in_executor(None, orchestrator.initialize)
            actual_plate['status'] = '-'  # Initialized
            logger.info(f"Initialized orchestrator for plate: {actual_plate['name']}")

        # Update UI - trigger reactive update for mutable list
        self.mutate_reactive(PlateManagerWidget.plates)

        # Update status
        success_count = len([p for p in selected_items if p.get('status') == '-'])
        error_count = len([p for p in selected_items if p.get('status') == 'X'])

        if error_count == 0:
            self.app.current_status = f"Successfully initialized {success_count} plates"
        else:
            self.app.current_status = f"Initialized {success_count} plates, {error_count} errors"


    
    def action_compile_plate(self) -> None:
        """Handle Compile Plate button - compile pipelines for selected plates."""

        # Get current selection state
        selected_items, selection_mode = self.get_selection_state()

        if selection_mode == "empty":
            self.app.current_status = "No plates available for compilation"
            return

        # Validate all selected plates are initialized
        uninitialized = [item for item in selected_items if item.get('status') == '?']
        if uninitialized:
            names = [item['name'] for item in uninitialized]
            self.app.current_status = f"Cannot compile uninitialized plates: {', '.join(names)}"
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
        logger.info(f"Starting compilation: {desc}")
        self.app.current_status = f"Compiling: {desc}"

        # Start background worker
        self._compile_plates_worker(selected_items)

    @work(exclusive=True)
    async def _compile_plates_worker(self, selected_items: List[Dict]) -> None:
        """Background worker for plate compilation."""
        for plate_data in selected_items:
            plate_path = plate_data['path']

            # Get orchestrator
            orchestrator = self.orchestrators.get(plate_path)
            if not orchestrator:
                logger.error(f"No orchestrator found for {plate_path}")
                plate_data['status'] = 'X'
                plate_data['error'] = "No orchestrator found"
                continue

            # Get pipeline definition for this specific plate
            pipeline_definition = self._get_current_pipeline_definition(plate_path)
            if not pipeline_definition:
                logger.warning(f"No pipeline defined for {plate_data['name']}, using empty pipeline")
                pipeline_definition = []

            # Get wells (heavy operation)
            wells = await orchestrator.get_wells()

            # Compile pipelines (heavy operation)
            compiled_contexts = await orchestrator.compile_pipelines(
                pipeline_definition=pipeline_definition,
                well_filter=wells
            )

            # Store compiled data
            plate_data['compiled_contexts'] = compiled_contexts
            plate_data['pipeline_definition'] = pipeline_definition
            plate_data['status'] = 'o'  # Compiled
            logger.info(f"Compiled pipeline for plate: {plate_data['name']}")

        # Update UI - trigger reactive update
        self.plates = list(self.plates)  # Force reactive update

        # Update status
        success_count = len([p for p in selected_items if p.get('status') == 'o'])
        error_count = len([p for p in selected_items if p.get('status') == 'X'])

        if error_count == 0:
            self.app.current_status = f"Successfully compiled {success_count} plates"
        else:
            self.app.current_status = f"Compiled {success_count} plates, {error_count} errors"



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

    def _get_current_pipeline_definition(self, plate_path: str = None) -> List:
        """Get current pipeline definition from PipelineEditor."""
        if not self.pipeline_editor:
            logger.warning("No pipeline editor reference - using empty pipeline")
            return []

        # Get pipeline for specific plate or current plate
        target_plate = plate_path or self.pipeline_editor.current_plate
        if not target_plate:
            logger.warning("No plate specified - using empty pipeline")
            return []

        # Get pipeline from editor
        pipeline = self.pipeline_editor.get_pipeline_for_plate(target_plate)
        logger.info(f"Retrieved pipeline with {len(pipeline)} steps for plate: {target_plate}")
        return pipeline
    
    def action_run_plate(self) -> None:
        """Handle Run Plate button - execute compiled plates."""

        # Get current selection state
        selected_items, selection_mode = self.get_selection_state()

        if selection_mode == "empty":
            self.app.current_status = "No plates available for execution"
            return

        # Validate all selected plates are compiled
        uncompiled = [item for item in selected_items if item.get('status') != 'o']
        if uncompiled:
            names = [item['name'] for item in uncompiled]
            self.app.current_status = f"Cannot run uncompiled plates: {', '.join(names)}"
            return

        # Start async execution
        self._start_async_run(selected_items, selection_mode)

    def _start_async_run(self, selected_items: List[Dict], selection_mode: str) -> None:
        """Start async execution of selected plates."""
        from textual import work

        # Generate operation description
        desc = self.get_operation_description(selected_items, selection_mode, "run")
        logger.info(f"Starting execution: {desc}")
        self.app.current_status = f"Running: {desc}"

        # Start background worker
        self._run_plates_worker(selected_items)

    @work(exclusive=True)
    async def _run_plates_worker(self, selected_items: List[Dict]) -> None:
        """Background worker for plate execution."""
        for plate_data in selected_items:
            plate_path = plate_data['path']

            # Get orchestrator and compiled data
            orchestrator = self.orchestrators.get(plate_path)
            if not orchestrator:
                logger.error(f"No orchestrator found for {plate_path}")
                plate_data['status'] = 'X'
                plate_data['error'] = "No orchestrator found"
                continue

            compiled_contexts = plate_data.get('compiled_contexts')
            pipeline_definition = plate_data.get('pipeline_definition')

            if not compiled_contexts or not pipeline_definition:
                logger.error(f"No compiled data found for {plate_data['name']}")
                plate_data['status'] = 'X'
                plate_data['error'] = "No compiled data found"
                continue

            # Set status to running
            plate_data['status'] = '!'  # Running
            self.plates = list(self.plates)  # Force UI update

            # Execute compiled plate (heavy operation)
            results = await orchestrator.execute_compiled_plate(
                pipeline_definition=pipeline_definition,
                compiled_contexts=compiled_contexts
            )

            # Store results and update status
            plate_data['execution_results'] = results
            if results and all(r.get('status') != 'error' for r in results.values()):
                plate_data['status'] = 'o'  # Completed successfully
                logger.info(f"Successfully executed plate: {plate_data['name']}")
            else:
                plate_data['status'] = 'X'  # Execution error
                plate_data['error'] = "Execution failed - check logs"
                logger.error(f"Execution failed for {plate_data['name']}")

        # Update UI - trigger reactive update
        self.plates = list(self.plates)  # Force reactive update

        # Update status
        success_count = len([p for p in selected_items if p.get('status') == 'o'])
        error_count = len([p for p in selected_items if p.get('status') == 'X'])

        if error_count == 0:
            self.app.current_status = f"Successfully executed {success_count} plates"
        else:
            self.app.current_status = f"Executed {success_count} plates, {error_count} errors"
