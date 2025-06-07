"""
PlateManagerWidget for OpenHCS Textual TUI

Plate management widget with complete button set and reactive state management.
Matches the functionality from the current prompt-toolkit TUI.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable

from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer
from textual.reactive import reactive
from textual.widgets import Button, Static
from textual.widget import Widget
from textual import work

from openhcs.core.config import GlobalPipelineConfig
from openhcs.io.filemanager import FileManager
from .selectable_list_view import SelectableListView

logger = logging.getLogger(__name__)


class PlateManagerWidget(Widget):
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
        super().__init__()
        self.filemanager = filemanager
        self.global_config = global_config

        # Callback for plate selection (set by MainContent)
        self.on_plate_selected: Optional[Callable[[str], None]] = None

        # Reference to pipeline editor (set by MainContent)
        self.pipeline_editor: Optional['PipelineEditorWidget'] = None

        logger.debug("PlateManagerWidget initialized")
    
    def compose(self) -> ComposeResult:
        """Compose the plate manager layout."""
        # Complete button set with compact uniform styling
        with Horizontal():
            yield Button("Add", id="add_plate", compact=True)
            yield Button("Del", id="del_plate", disabled=True, compact=True)
            yield Button("Edit", id="edit_plate", disabled=True, compact=True)
            yield Button("Init", id="init_plate", disabled=True, compact=True)
            yield Button("Compile", id="compile_plate", disabled=True, compact=True)
            yield Button("Run", id="run_plate", disabled=True, compact=True)

        # Scrollable content area with SelectableListView
        with ScrollableContainer(id="plate_list"):
            yield SelectableListView(items=self.plates, id="plate_content")

    def on_mount(self) -> None:
        """Called when the widget is mounted - ensure display is up to date."""
        logger.info("PlateManagerWidget mounted, updating display")
        # Schedule multiple update attempts to ensure it works
        self.call_later(self._delayed_update_display)
        self.set_timer(0.1, self._delayed_update_display)
        self.call_later(self._update_button_states)
    

    
    def watch_plates(self, plates: List[Dict]) -> None:
        """Automatically update UI when plates reactive property changes."""
        logger.info(f"watch_plates called with {len(plates)} plates: {[p.get('name', 'Unknown') for p in plates]}")

        # Update SelectableListView content
        try:
            list_view = self.query_one("#plate_content", SelectableListView)
            list_view.load_items(plates)
            logger.info(f"✅ Updated SelectableListView successfully - now showing {len(plates)} plates")

        except Exception as e:
            logger.warning(f"Could not update plate content (widget may not be mounted): {e}")
            # Schedule delayed update attempts using proper Textual API
            self.call_later(self._delayed_update_display)
            self.set_timer(0.1, self._delayed_update_display)
            self.set_timer(0.5, self._delayed_update_display)

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

    def on_selectable_list_view_selection_changed(self, event: SelectableListView.SelectionChanged) -> None:
        """Handle selection changes from SelectableListView."""
        selected_items = event.selected_items
        selection_mode = event.selection_mode

        logger.info(f"Selection changed: {len(selected_items)} items in '{selection_mode}' mode")

        # Update selected_plate based on selection mode
        if selection_mode == "cursor" and selected_items:
            self.selected_plate = selected_items[0]['path']
        elif selection_mode == "checkbox" and len(selected_items) == 1:
            self.selected_plate = selected_items[0]['path']
        else:
            self.selected_plate = ""

        # Update button states based on selection
        self._update_button_states_for_selection(selected_items, selection_mode)

    def _update_button_states_for_selection(self, selected_items: List[Dict], selection_mode: str) -> None:
        """Update button states based on current selection."""
        try:
            has_plates = len(self.plates) > 0
            has_selection = len(selected_items) > 0

            # Basic button states
            self.query_one("#del_plate").disabled = not has_selection
            self.query_one("#edit_plate").disabled = not (selection_mode in ["cursor", "checkbox"] and len(selected_items) == 1)
            self.query_one("#init_plate").disabled = not has_selection
            self.query_one("#compile_plate").disabled = not has_selection
            self.query_one("#run_plate").disabled = not has_selection

        except Exception:
            # Buttons might not be mounted yet
            pass

    def get_selection_state(self) -> tuple[List[Dict], str]:
        """Get current selection state from SelectableListView."""
        try:
            list_view = self.query_one("#plate_content", SelectableListView)
            return list_view.get_selection_state()
        except Exception:
            # Fallback if widget not mounted
            return [], "empty"

    def _delayed_update_display(self) -> None:
        """Update the plate display - called when widget is mounted or as fallback."""
        try:
            list_view = self.query_one("#plate_content", SelectableListView)
            list_view.load_items(self.plates)
            logger.info(f"✅ Delayed SelectableListView update successful - showing {len(self.plates)} plates")
        except Exception as e:
            logger.warning(f"Delayed update failed (widget may not be ready): {e}")
            # Try again in a moment using proper Textual API
            self.set_timer(0.1, self._delayed_update_display)

    def _update_button_states(self) -> None:
        """Update button enabled/disabled states based on current state."""
        try:
            has_plates = len(self.plates) > 0
            has_selection = bool(self.selected_plate)
            
            self.query_one("#del_plate").disabled = not has_plates
            self.query_one("#edit_plate").disabled = not has_selection
            self.query_one("#init_plate").disabled = not has_selection
            self.query_one("#compile_plate").disabled = not has_selection
            self.query_one("#run_plate").disabled = not has_selection
        except Exception:
            # Buttons might not be mounted yet
            pass
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
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
    
    def action_add_plate(self) -> None:
        """Handle Add Plate button."""
        logger.info("Add Plate button pressed")
        self.app.push_screen(self._create_file_browser_screen(), self._on_plate_directory_selected)

    def _create_file_browser_screen(self):
        """Create enhanced file browser screen for plate selection."""
        from openhcs.textual_tui.screens.enhanced_file_browser import EnhancedFileBrowserScreen
        from openhcs.constants.constants import Backend
        from pathlib import Path

        # Create enhanced file browser for directory selection
        return EnhancedFileBrowserScreen(
            file_manager=self.filemanager,
            initial_path=Path.home(),
            backend=Backend.DISK,
            title="Select Plate Directory"
        )

    def _on_plate_directory_selected(self, selected_paths) -> None:
        """Handle directory selection from file browser."""
        if selected_paths is None:
            logger.info("Plate directory selection cancelled")
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
        logger.info("Delete Plate button pressed")

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
        """Handle Edit Plate button."""
        logger.info("Edit Plate button pressed")
        # TODO: Implement edit functionality in Sprint 2
        self.app.current_status = "Edit Plate not yet implemented"
    
    def action_init_plate(self) -> None:
        """Handle Init Plate button - initialize selected plates."""
        logger.info("Init Plate button pressed")

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
        try:
            for plate_data in selected_items:
                plate_path = plate_data['path']

                # Get orchestrator
                orchestrator = self.orchestrators.get(plate_path)
                if not orchestrator:
                    logger.error(f"No orchestrator found for {plate_path}")
                    plate_data['status'] = 'X'
                    plate_data['error'] = "No orchestrator found"
                    continue

                # Check if already initialized
                if orchestrator.is_initialized():
                    logger.info(f"Orchestrator for {plate_data['name']} already initialized")
                    plate_data['status'] = '-'
                    continue

                # Initialize orchestrator (heavy operation)
                try:
                    await orchestrator.initialize()
                    plate_data['status'] = '-'  # Initialized
                    logger.info(f"Initialized orchestrator for plate: {plate_data['name']}")

                except Exception as e:
                    logger.error(f"Failed to initialize {plate_data['name']}: {e}")
                    plate_data['status'] = 'X'  # Error
                    plate_data['error'] = str(e)

            # Update UI - trigger reactive update
            self.plates = list(self.plates)  # Force reactive update

            # Update status
            success_count = len([p for p in selected_items if p.get('status') == '-'])
            error_count = len([p for p in selected_items if p.get('status') == 'X'])

            if error_count == 0:
                self.app.current_status = f"Successfully initialized {success_count} plates"
            else:
                self.app.current_status = f"Initialized {success_count} plates, {error_count} errors"

        except Exception as e:
            logger.error(f"Initialization worker failed: {e}")
            self.app.current_status = f"Initialization failed: {e}"
    
    def action_compile_plate(self) -> None:
        """Handle Compile Plate button - compile pipelines for selected plates."""
        logger.info("Compile Plate button pressed")

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
        try:
            for plate_data in selected_items:
                plate_path = plate_data['path']

                # Get orchestrator
                orchestrator = self.orchestrators.get(plate_path)
                if not orchestrator:
                    logger.error(f"No orchestrator found for {plate_path}")
                    plate_data['status'] = 'X'
                    plate_data['error'] = "No orchestrator found"
                    continue

                # Get pipeline definition (TODO: get from PipelineEditor)
                pipeline_definition = self._get_current_pipeline_definition()
                if not pipeline_definition:
                    logger.warning(f"No pipeline defined for {plate_data['name']}, using empty pipeline")
                    pipeline_definition = []

                try:
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

                except Exception as e:
                    logger.error(f"Failed to compile {plate_data['name']}: {e}")
                    plate_data['status'] = 'X'  # Error
                    plate_data['error'] = str(e)

            # Update UI - trigger reactive update
            self.plates = list(self.plates)  # Force reactive update

            # Update status
            success_count = len([p for p in selected_items if p.get('status') == 'o'])
            error_count = len([p for p in selected_items if p.get('status') == 'X'])

            if error_count == 0:
                self.app.current_status = f"Successfully compiled {success_count} plates"
            else:
                self.app.current_status = f"Compiled {success_count} plates, {error_count} errors"

        except Exception as e:
            logger.error(f"Compilation worker failed: {e}")
            self.app.current_status = f"Compilation failed: {e}"

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
        logger.info("Run Plate button pressed")

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
        try:
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

                try:
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

                except Exception as e:
                    logger.error(f"Failed to execute {plate_data['name']}: {e}")
                    plate_data['status'] = 'X'  # Error
                    plate_data['error'] = str(e)

            # Update UI - trigger reactive update
            self.plates = list(self.plates)  # Force reactive update

            # Update status
            success_count = len([p for p in selected_items if p.get('status') == 'o'])
            error_count = len([p for p in selected_items if p.get('status') == 'X'])

            if error_count == 0:
                self.app.current_status = f"Successfully executed {success_count} plates"
            else:
                self.app.current_status = f"Executed {success_count} plates, {error_count} errors"

        except Exception as e:
            logger.error(f"Execution worker failed: {e}")
            self.app.current_status = f"Execution failed: {e}"
