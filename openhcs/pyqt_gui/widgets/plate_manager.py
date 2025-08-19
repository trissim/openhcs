"""
Plate Manager Widget for PyQt6

Manages plate selection, initialization, and execution with full feature parity
to the Textual TUI version. Uses hybrid approach: extracted business logic + clean PyQt6 UI.
"""

import logging
import asyncio
import inspect
import copy
import sys
import subprocess
import tempfile
from typing import List, Dict, Optional, Callable
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget,
    QListWidgetItem, QLabel, QMessageBox, QFileDialog, QProgressBar,
    QCheckBox, QFrame, QSplitter
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QThread
from PyQt6.QtGui import QFont

from openhcs.core.config import GlobalPipelineConfig, PipelineConfig
from openhcs.io.filemanager import FileManager
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator, OrchestratorState
from openhcs.core.pipeline import Pipeline
from openhcs.constants.constants import VariableComponents, GroupBy
from openhcs.pyqt_gui.widgets.mixins import (
    preserve_selection_during_update,
    handle_selection_change_with_prevention
)
from openhcs.pyqt_gui.shared.style_generator import StyleSheetGenerator
from openhcs.pyqt_gui.shared.color_scheme import PyQt6ColorScheme

logger = logging.getLogger(__name__)


class PlateManagerWidget(QWidget):
    """
    PyQt6 Plate Manager Widget.
    
    Manages plate selection, initialization, compilation, and execution.
    Preserves all business logic from Textual version with clean PyQt6 UI.
    """
    
    # Signals
    plate_selected = pyqtSignal(str)  # plate_path
    status_message = pyqtSignal(str)  # status message
    orchestrator_state_changed = pyqtSignal(str, str)  # plate_path, state
    orchestrator_config_changed = pyqtSignal(str, object)  # plate_path, effective_config

    # Log viewer integration signals
    subprocess_log_started = pyqtSignal(str)  # base_log_path
    subprocess_log_stopped = pyqtSignal()
    clear_subprocess_logs = pyqtSignal()

    # Progress update signals (thread-safe UI updates)
    progress_started = pyqtSignal(int)  # max_value
    progress_updated = pyqtSignal(int)  # current_value
    progress_finished = pyqtSignal()

    # Error handling signals (thread-safe error reporting)
    compilation_error = pyqtSignal(str, str)  # plate_name, error_message
    initialization_error = pyqtSignal(str, str)  # plate_name, error_message
    
    def __init__(self, file_manager: FileManager, service_adapter,
                 color_scheme: Optional[PyQt6ColorScheme] = None, parent=None):
        """
        Initialize the plate manager widget.

        Args:
            file_manager: FileManager instance for file operations
            service_adapter: PyQt service adapter for dialogs and operations
            color_scheme: Color scheme for styling (optional, uses service adapter if None)
            parent: Parent widget
        """
        super().__init__(parent)

        # Core dependencies
        self.file_manager = file_manager
        self.service_adapter = service_adapter
        self.global_config = service_adapter.get_global_config()
        self.pipeline_editor = None  # Will be set by main window

        # Initialize color scheme and style generator
        self.color_scheme = color_scheme or service_adapter.get_current_color_scheme()
        self.style_generator = StyleSheetGenerator(self.color_scheme)
        
        # Business logic state (extracted from Textual version)
        self.plates: List[Dict] = []  # List of plate dictionaries
        self.selected_plate_path: str = ""
        self.orchestrators: Dict[str, PipelineOrchestrator] = {}
        self.plate_configs: Dict[str, Dict] = {}
        self.plate_compiled_data: Dict[str, tuple] = {}  # Store compiled pipeline data
        self.current_process = None
        self.execution_state = "idle"
        self.log_file_path: Optional[str] = None
        self.log_file_position: int = 0
        
        # UI components
        self.plate_list: Optional[QListWidget] = None
        self.buttons: Dict[str, QPushButton] = {}
        self.status_label: Optional[QLabel] = None
        self.progress_bar: Optional[QProgressBar] = None
        
        # Setup UI
        self.setup_ui()
        self.setup_connections()
        self.update_button_states()
        
        logger.debug("Plate manager widget initialized")

    # ========== UI Setup ==========

    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Title
        title_label = QLabel("Plate Manager")
        title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        title_label.setStyleSheet(f"color: {self.color_scheme.to_hex(self.color_scheme.text_accent)}; padding: 5px;")
        layout.addWidget(title_label)
        
        # Main content splitter
        splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(splitter)
        
        # Plate list
        self.plate_list = QListWidget()
        self.plate_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        # Apply centralized styling to plate list
        self.setStyleSheet(self.style_generator.generate_plate_manager_style())
        splitter.addWidget(self.plate_list)
        
        # Button panel
        button_panel = self.create_button_panel()
        splitter.addWidget(button_panel)
        
        # Status section
        status_frame = self.create_status_section()
        layout.addWidget(status_frame)
        
        # Set splitter proportions
        splitter.setSizes([300, 150])
    
    def create_button_panel(self) -> QWidget:
        """
        Create the button panel with all plate management actions.
        
        Returns:
            Widget containing action buttons
        """
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.Box)
        # Frame styling is handled by the main widget stylesheet
        
        layout = QVBoxLayout(panel)
        
        # Button configurations (extracted from Textual version)
        button_configs = [
            ("Add", "add_plate", "Add new plate directory"),
            ("Del", "del_plate", "Delete selected plates"),
            ("Edit", "edit_config", "Edit plate configuration"),
            ("Init", "init_plate", "Initialize selected plates"),
            ("Compile", "compile_plate", "Compile plate pipelines"),
            ("Run", "run_plate", "Run/Stop plate execution"),
            ("Code", "code_plate", "Generate Python code"),
            ("Save", "save_python_script", "Save Python script"),
        ]
        
        # Create buttons in rows
        for i in range(0, len(button_configs), 4):
            row_layout = QHBoxLayout()
            
            for j in range(4):
                if i + j < len(button_configs):
                    name, action, tooltip = button_configs[i + j]
                    
                    button = QPushButton(name)
                    button.setToolTip(tooltip)
                    button.setMinimumHeight(30)
                    # Button styling is handled by the main widget stylesheet
                    
                    # Connect button to action
                    button.clicked.connect(lambda checked, a=action: self.handle_button_action(a))
                    
                    self.buttons[action] = button
                    row_layout.addWidget(button)
                else:
                    row_layout.addStretch()
            
            layout.addLayout(row_layout)
        
        return panel
    
    def create_status_section(self) -> QWidget:
        """
        Create the status section with progress bar and status label.
        
        Returns:
            Widget containing status information
        """
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.Box)
        # Frame styling is handled by the main widget stylesheet
        
        layout = QVBoxLayout(frame)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(f"color: {self.color_scheme.to_hex(self.color_scheme.status_success)}; font-weight: bold;")
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        # Progress bar styling is handled by the main widget stylesheet
        layout.addWidget(self.progress_bar)
        
        return frame
    
    def setup_connections(self):
        """Setup signal/slot connections."""
        # Plate list selection
        self.plate_list.itemSelectionChanged.connect(self.on_selection_changed)
        self.plate_list.itemDoubleClicked.connect(self.on_item_double_clicked)
        
        # Internal signals
        self.status_message.connect(self.update_status)
        self.orchestrator_state_changed.connect(self.on_orchestrator_state_changed)

        # Progress signals for thread-safe UI updates
        self.progress_started.connect(self._on_progress_started)
        self.progress_updated.connect(self._on_progress_updated)
        self.progress_finished.connect(self._on_progress_finished)

        # Error handling signals for thread-safe error reporting
        self.compilation_error.connect(self._handle_compilation_error)
        self.initialization_error.connect(self._handle_initialization_error)
    
    def handle_button_action(self, action: str):
        """
        Handle button actions (extracted from Textual version).

        Args:
            action: Action identifier
        """
        # Action mapping (preserved from Textual version)
        action_map = {
            "add_plate": self.action_add_plate,
            "del_plate": self.action_delete_plate,
            "edit_config": self.action_edit_config,
            "init_plate": self.action_init_plate,
            "compile_plate": self.action_compile_plate,
            "code_plate": self.action_code_plate,
            "save_python_script": self.action_save_python_script,
        }

        if action in action_map:
            action_func = action_map[action]

            # Handle async actions
            if inspect.iscoroutinefunction(action_func):
                self.run_async_action(action_func)
            else:
                action_func()
        elif action == "run_plate":
            if self.is_any_plate_running():
                self.run_async_action(self.action_stop_execution)
            else:
                self.run_async_action(self.action_run_plate)
        else:
            logger.warning(f"Unknown action: {action}")
    
    def run_async_action(self, async_func: Callable):
        """
        Run async action using service adapter.

        Args:
            async_func: Async function to execute
        """
        self.service_adapter.execute_async_operation(async_func)

    def _update_orchestrator_global_config(self, orchestrator, new_global_config):
        """Update orchestrator's global config reference and rebuild pipeline config if needed."""
        from openhcs.core.lazy_config import rebuild_lazy_config_with_new_global_reference
        from openhcs.core.config import GlobalPipelineConfig, set_current_global_config

        # Update orchestrator's global config reference
        orchestrator.global_config = new_global_config

        # Rebuild orchestrator-specific config if it exists
        if orchestrator.pipeline_config is not None:
            orchestrator.pipeline_config = rebuild_lazy_config_with_new_global_reference(
                orchestrator.pipeline_config,
                new_global_config,
                GlobalPipelineConfig
            )
            logger.info(f"Rebuilt orchestrator-specific config for plate: {orchestrator.plate_path}")

        # Get effective config and emit signal for UI refresh
        effective_config = orchestrator.get_effective_config()
        self.orchestrator_config_changed.emit(str(orchestrator.plate_path), effective_config)
    
    # ========== Business Logic Methods (Extracted from Textual) ==========
    
    def action_add_plate(self):
        """Handle Add Plate button (adapted from Textual version)."""
        from openhcs.core.path_cache import PathCacheKey

        # Use cached directory dialog (mirrors Textual TUI pattern)
        directory_path = self.service_adapter.show_cached_directory_dialog(
            cache_key=PathCacheKey.PLATE_IMPORT,
            title="Select Plate Directory",
            fallback_path=Path.home()
        )

        if directory_path:
            self.add_plate_callback([directory_path])
    
    def add_plate_callback(self, selected_paths: List[Path]):
        """
        Handle plate directory selection (extracted from Textual version).
        
        Args:
            selected_paths: List of selected directory paths
        """
        if not selected_paths:
            self.status_message.emit("Plate selection cancelled")
            return
        
        added_plates = []
        
        for selected_path in selected_paths:
            # Check if plate already exists
            if any(plate['path'] == str(selected_path) for plate in self.plates):
                continue
            
            # Add the plate to the list
            plate_name = selected_path.name
            plate_path = str(selected_path)
            plate_entry = {
                'name': plate_name,
                'path': plate_path,
            }
            
            self.plates.append(plate_entry)
            added_plates.append(plate_name)
        
        if added_plates:
            self.update_plate_list()
            self.status_message.emit(f"Added {len(added_plates)} plate(s): {', '.join(added_plates)}")
        else:
            self.status_message.emit("No new plates added (duplicates skipped)")
    
    def action_delete_plate(self):
        """Handle Delete Plate button (extracted from Textual version)."""
        selected_items = self.get_selected_plates()
        if not selected_items:
            self.service_adapter.show_error_dialog("No plate selected to delete.")
            return
        
        paths_to_delete = {p['path'] for p in selected_items}
        self.plates = [p for p in self.plates if p['path'] not in paths_to_delete]
        
        # Clean up orchestrators for deleted plates
        for path in paths_to_delete:
            if path in self.orchestrators:
                del self.orchestrators[path]
        
        if self.selected_plate_path in paths_to_delete:
            self.selected_plate_path = ""
            # Notify pipeline editor that no plate is selected (mirrors Textual TUI)
            self.plate_selected.emit("")

        self.update_plate_list()
        self.status_message.emit(f"Deleted {len(paths_to_delete)} plate(s)")
    
    def _validate_plates_for_operation(self, plates, operation_type):
        """Unified functional validator for all plate operations."""
        # Functional validation mapping
        validators = {
            'init': lambda p: True,  # Init can work on any plates
            'compile': lambda p: (
                self.orchestrators.get(p['path']) and
                self._get_current_pipeline_definition(p['path'])
            ),
            'run': lambda p: (
                self.orchestrators.get(p['path']) and
                self.orchestrators[p['path']].state in ['COMPILED', 'COMPLETED']
            )
        }

        # Functional pattern: filter invalid plates in one pass
        validator = validators.get(operation_type, lambda p: True)
        return [p for p in plates if not validator(p)]

    async def action_init_plate(self):
        """Handle Initialize Plate button with unified validation."""
        selected_items = self.get_selected_plates()

        # Unified validation - let it fail if no plates
        invalid_plates = self._validate_plates_for_operation(selected_items, 'init')

        self.progress_started.emit(len(selected_items))

        # Functional pattern: async map with enumerate
        async def init_single_plate(i, plate):
            plate_path = plate['path']
            orchestrator = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: PipelineOrchestrator(
                    plate_path=plate_path,
                    global_config=self.global_config,
                    storage_registry=self.file_manager.registry
                ).initialize()
            )

            self.orchestrators[plate_path] = orchestrator
            self.orchestrator_state_changed.emit(plate_path, "READY")

            if not self.selected_plate_path:
                self.selected_plate_path = plate_path
                self.plate_selected.emit(plate_path)

            self.progress_updated.emit(i + 1)

        # Process all plates functionally
        await asyncio.gather(*[
            init_single_plate(i, plate)
            for i, plate in enumerate(selected_items)
        ])

        self.progress_finished.emit()
        self.status_message.emit(f"Initialized {len(selected_items)} plate(s)")
    
    # Additional action methods would be implemented here following the same pattern...
    # (compile_plate, run_plate, code_plate, save_python_script, edit_config)
    
    def action_edit_config(self):
        """
        Handle Edit Config button - create per-orchestrator PipelineConfig instances.

        This enables per-orchestrator configuration without affecting global configuration.
        Shows resolved defaults from GlobalPipelineConfig with "Pipeline default: {value}" placeholders.
        """
        selected_items = self.get_selected_plates()

        if not selected_items:
            self.service_adapter.show_error_dialog("No plates selected for configuration.")
            return

        # Get selected orchestrators
        selected_orchestrators = [
            self.orchestrators[item['path']] for item in selected_items
            if item['path'] in self.orchestrators
        ]

        if not selected_orchestrators:
            self.service_adapter.show_error_dialog("No initialized orchestrators selected.")
            return

        # Load existing config or create new one for editing
        representative_orchestrator = selected_orchestrators[0]

        # Set up thread-local context for pipeline config editing
        from openhcs.core.config import set_current_global_config, GlobalPipelineConfig
        set_current_global_config(GlobalPipelineConfig, self.global_config)

        # Create config for editing based on whether orchestrator has existing config
        if representative_orchestrator.pipeline_config is not None:
            # Existing config - use the editing function that preserves user values
            from openhcs.core.pipeline_config import create_editing_config_from_existing_lazy_config
            current_plate_config = create_editing_config_from_existing_lazy_config(
                representative_orchestrator.pipeline_config,
                None  # Use thread-local context set up above
            )
        else:
            # Create new config with placeholders using current global config
            from openhcs.core.pipeline_config import create_pipeline_config_for_editing
            current_plate_config = create_pipeline_config_for_editing(self.global_config)

        def handle_config_save(new_config: PipelineConfig) -> None:
            """Apply per-orchestrator configuration without global side effects."""
            for orchestrator in selected_orchestrators:
                # Direct synchronous call - no async needed
                orchestrator.apply_pipeline_config(new_config)
                # Emit signal for UI components to refresh
                effective_config = orchestrator.get_effective_config()
                self.orchestrator_config_changed.emit(str(orchestrator.plate_path), effective_config)
            count = len(selected_orchestrators)
            # Success message dialog removed for test automation compatibility

        # Open configuration window using PipelineConfig (not GlobalPipelineConfig)
        # PipelineConfig already imported from openhcs.core.config
        # CRITICAL FIX: Pass orchestrator reference for context persistence
        self._open_config_window(
            config_class=PipelineConfig,
            current_config=current_plate_config,
            on_save_callback=handle_config_save,
            orchestrator=representative_orchestrator  # Pass orchestrator for context persistence
        )

    def _open_config_window(self, config_class, current_config, on_save_callback, orchestrator=None):
        """
        Open configuration window with specified config class and current config.

        Args:
            config_class: Configuration class type (PipelineConfig or GlobalPipelineConfig)
            current_config: Current configuration instance
            on_save_callback: Function to call when config is saved
            orchestrator: Optional orchestrator reference for context persistence
        """
        from openhcs.pyqt_gui.windows.config_window import ConfigWindow

        config_window = ConfigWindow(
            config_class,           # config_class
            current_config,         # current_config
            on_save_callback,       # on_save_callback
            self.color_scheme,      # color_scheme
            self,                   # parent
            orchestrator=orchestrator  # Pass orchestrator for context persistence
        )
        # Show as non-modal window (like main window configuration)
        config_window.show()
        config_window.raise_()
        config_window.activateWindow()

    def action_edit_global_config(self):
        """
        Handle global configuration editing - affects all orchestrators.

        Uses concrete GlobalPipelineConfig for direct editing with static placeholder defaults.
        """
        from openhcs.core.config import get_default_global_config, GlobalPipelineConfig

        # Get current global config from service adapter or use default
        current_global_config = self.service_adapter.get_global_config() or get_default_global_config()

        def handle_global_config_save(new_config: GlobalPipelineConfig) -> None:
            """Apply global configuration to all orchestrators and save to cache."""
            self.service_adapter.set_global_config(new_config)  # Update app-level config

            # Update thread-local storage for MaterializationPathConfig defaults
            from openhcs.core.config import set_current_global_config, GlobalPipelineConfig
            set_current_global_config(GlobalPipelineConfig, new_config)

            # Save to cache for persistence between sessions
            self._save_global_config_to_cache(new_config)

            for orchestrator in self.orchestrators.values():
                self._update_orchestrator_global_config(orchestrator, new_config)

            # Update thread-local storage for currently selected orchestrator
            # This ensures step forms resolve against the updated orchestrator defaults
            if self.selected_plate_path and self.selected_plate_path in self.orchestrators:
                current_orchestrator = self.orchestrators[self.selected_plate_path]
                effective_config = current_orchestrator.get_effective_config()
                from openhcs.core.config import set_current_global_config, GlobalPipelineConfig
                set_current_global_config(GlobalPipelineConfig, effective_config)
                logger.debug(f"Updated thread-local storage for currently selected orchestrator: {self.selected_plate_path}")

            # Refresh placeholder text in any open parameter forms
            self._refresh_all_parameter_form_placeholders()

            self.service_adapter.show_info_dialog("Global configuration applied to all orchestrators")

        # Open configuration window using concrete GlobalPipelineConfig
        self._open_config_window(
            config_class=GlobalPipelineConfig,
            current_config=current_global_config,
            on_save_callback=handle_global_config_save
        )

    def _save_global_config_to_cache(self, config: GlobalPipelineConfig):
        """Save global config to cache for persistence between sessions."""
        try:
            # Use synchronous saving to ensure it completes
            from openhcs.core.config_cache import _sync_save_config
            from openhcs.core.xdg_paths import get_config_file_path

            cache_file = get_config_file_path("global_config.config")
            success = _sync_save_config(config, cache_file)

            if success:
                logger.info("Global config saved to cache for session persistence")
            else:
                logger.error("Failed to save global config to cache - sync save returned False")
        except Exception as e:
            logger.error(f"Failed to save global config to cache: {e}")
            # Don't show error dialog as this is not critical for immediate functionality

    async def action_compile_plate(self):
        """Handle Compile Plate button - compile pipelines for selected plates."""
        selected_items = self.get_selected_plates()

        if not selected_items:
            logger.warning("No plates available for compilation")
            return

        # Unified validation using functional validator
        invalid_plates = self._validate_plates_for_operation(selected_items, 'compile')

        # Let validation failures bubble up as status messages
        if invalid_plates:
            invalid_names = [p['name'] for p in invalid_plates]
            self.status_message.emit(f"Cannot compile invalid plates: {', '.join(invalid_names)}")
            return

        # Start async compilation
        await self._compile_plates_worker(selected_items)

    async def _compile_plates_worker(self, selected_items: List[Dict]) -> None:
        """Background worker for plate compilation."""
        # Use signals for thread-safe UI updates
        self.progress_started.emit(len(selected_items))

        for i, plate_data in enumerate(selected_items):
            plate_path = plate_data['path']

            # Get definition pipeline and make fresh copy
            definition_pipeline = self._get_current_pipeline_definition(plate_path)
            if not definition_pipeline:
                logger.warning(f"No pipeline defined for {plate_data['name']}, using empty pipeline")
                definition_pipeline = []

            try:
                # Get or create orchestrator for compilation (run in executor to avoid blocking)
                def get_or_create_orchestrator():
                    if plate_path in self.orchestrators:
                        orchestrator = self.orchestrators[plate_path]
                        if not orchestrator.is_initialized():
                            orchestrator.initialize()
                        return orchestrator
                    else:
                        return PipelineOrchestrator(
                            plate_path=plate_path,
                            global_config=self.global_config,
                            storage_registry=self.file_manager.registry
                        ).initialize()

                # Run in executor (works in Qt thread)
                import asyncio
                loop = asyncio.get_event_loop()
                orchestrator = await loop.run_in_executor(None, get_or_create_orchestrator)
                self.orchestrators[plate_path] = orchestrator

                # Make fresh copy for compilation
                execution_pipeline = copy.deepcopy(definition_pipeline)

                # Fix step IDs after deep copy to match new object IDs
                for step in execution_pipeline:
                    step.step_id = str(id(step))
                    # Ensure variable_components is never None - use FunctionStep default
                    if step.variable_components is None:
                        logger.warning(f"Step '{step.name}' has None variable_components, setting FunctionStep default")
                        step.variable_components = [VariableComponents.SITE]
                    # Also ensure it's not an empty list
                    elif not step.variable_components:
                        logger.warning(f"Step '{step.name}' has empty variable_components, setting FunctionStep default")
                        step.variable_components = [VariableComponents.SITE]

                # Get wells and compile (async - run in executor to avoid blocking UI)
                # Wrap in Pipeline object like test_main.py does
                pipeline_obj = Pipeline(steps=execution_pipeline)

                # Run heavy operations in executor to avoid blocking UI (works in Qt thread)
                import asyncio
                loop = asyncio.get_event_loop()
                wells = await loop.run_in_executor(None, lambda: orchestrator.get_component_keys(GroupBy.WELL))
                compiled_contexts = await loop.run_in_executor(
                    None, orchestrator.compile_pipelines, pipeline_obj.steps, wells
                )

                # Store compiled data
                self.plate_compiled_data[plate_path] = (execution_pipeline, compiled_contexts)
                logger.info(f"Successfully compiled {plate_path}")

                # Update orchestrator state change signal
                self.orchestrator_state_changed.emit(plate_path, "COMPILED")

            except Exception as e:
                logger.error(f"COMPILATION ERROR: Pipeline compilation failed for {plate_path}: {e}", exc_info=True)
                plate_data['error'] = str(e)
                # Don't store anything in plate_compiled_data on failure
                self.orchestrator_state_changed.emit(plate_path, "COMPILE_FAILED")
                # Use signal for thread-safe error reporting instead of direct dialog call
                self.compilation_error.emit(plate_data['name'], str(e))

            # Use signal for thread-safe progress update
            self.progress_updated.emit(i + 1)

        # Use signal for thread-safe progress completion
        self.progress_finished.emit()
        self.status_message.emit(f"Compilation completed for {len(selected_items)} plate(s)")
        self.update_button_states()
    
    async def action_run_plate(self):
        """Handle Run Plate button - execute compiled plates."""
        selected_items = self.get_selected_plates()
        if not selected_items:
            self.service_adapter.show_error_dialog("No plates selected to run.")
            return

        ready_items = [item for item in selected_items if item.get('path') in self.plate_compiled_data]
        if not ready_items:
            self.service_adapter.show_error_dialog("Selected plates are not compiled. Please compile first.")
            return

        try:
            # Use subprocess approach like Textual TUI
            logger.debug("Using subprocess approach for clean isolation")

            plate_paths_to_run = [item['path'] for item in ready_items]

            # Pass definition pipeline steps - subprocess will make fresh copy and compile
            pipeline_data = {}
            for plate_path in plate_paths_to_run:
                definition_pipeline = self._get_current_pipeline_definition(plate_path)
                pipeline_data[plate_path] = definition_pipeline

            logger.info(f"Starting subprocess for {len(plate_paths_to_run)} plates")

            # Clear subprocess logs before starting new execution
            self.clear_subprocess_logs.emit()

            # Create data file for subprocess
            data_file = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl')

            # Generate unique ID for this subprocess
            import time
            subprocess_timestamp = int(time.time())
            plate_names = [Path(path).name for path in plate_paths_to_run]
            unique_id = f"plates_{'_'.join(plate_names[:2])}_{subprocess_timestamp}"

            # Build subprocess log name using log utilities
            from openhcs.core.log_utils import get_current_log_file_path
            try:
                tui_log_path = get_current_log_file_path()
                if tui_log_path.endswith('.log'):
                    tui_base = tui_log_path[:-4]  # Remove .log extension
                else:
                    tui_base = tui_log_path
                log_file_base = f"{tui_base}_subprocess_{subprocess_timestamp}"
            except RuntimeError:
                # Fallback if no main log found
                log_dir = Path.home() / ".local" / "share" / "openhcs" / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file_base = str(log_dir / f"pyqt_gui_subprocess_{subprocess_timestamp}")

            # Pickle data for subprocess
            subprocess_data = {
                'plate_paths': plate_paths_to_run,
                'pipeline_data': pipeline_data,
                'global_config': self.global_config
            }

            # Resolve all lazy configurations to concrete values before pickling
            from openhcs.core.lazy_config import resolve_lazy_configurations_for_serialization
            resolved_subprocess_data = resolve_lazy_configurations_for_serialization(subprocess_data)

            # Write pickle data
            def _write_pickle_data():
                import dill as pickle
                with open(data_file.name, 'wb') as f:
                    pickle.dump(resolved_subprocess_data, f)
                data_file.close()

            # Write pickle data in executor (works in Qt thread)
            import asyncio
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _write_pickle_data)

            logger.debug(f"Created data file: {data_file.name}")

            # Create subprocess
            subprocess_script = Path(__file__).parent.parent.parent / "textual_tui" / "subprocess_runner.py"

            # Generate actual log file path that subprocess will create
            actual_log_file_path = f"{log_file_base}_{unique_id}.log"
            logger.debug(f"Log file base: {log_file_base}")
            logger.debug(f"Unique ID: {unique_id}")
            logger.debug(f"Actual log file: {actual_log_file_path}")

            # Store log file path for monitoring
            self.log_file_path = actual_log_file_path
            self.log_file_position = 0

            logger.debug(f"Subprocess command: {sys.executable} {subprocess_script} {data_file.name} {log_file_base} {unique_id}")

            # Create subprocess
            def _create_subprocess():
                return subprocess.Popen([
                    sys.executable, str(subprocess_script),
                    data_file.name, log_file_base, unique_id
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                )

            # Create subprocess in executor (works in Qt thread)
            import asyncio
            loop = asyncio.get_event_loop()
            self.current_process = await loop.run_in_executor(None, _create_subprocess)

            logger.info(f"Subprocess started with PID: {self.current_process.pid}")

            # Emit signal for log viewer to start monitoring
            self.subprocess_log_started.emit(log_file_base)

            # Update orchestrator states to show running state
            for plate in ready_items:
                plate_path = plate['path']
                if plate_path in self.orchestrators:
                    self.orchestrators[plate_path]._state = OrchestratorState.EXECUTING

            self.execution_state = "running"
            self.status_message.emit(f"Running {len(ready_items)} plate(s) in subprocess...")
            self.update_button_states()

            # Start monitoring
            await self._start_monitoring()

        except Exception as e:
            logger.error(f"Failed to start plate execution: {e}", exc_info=True)
            self.service_adapter.show_error_dialog(f"Failed to start execution: {e}")
            self.execution_state = "idle"
            self.update_button_states()
    
    async def action_stop_execution(self):
        """Handle Stop Execution - terminate running subprocess (matches TUI implementation)."""
        logger.info("🛑 Stop button pressed. Terminating subprocess.")
        self.status_message.emit("Terminating execution...")

        if self.current_process and self.current_process.poll() is None:  # Still running
            try:
                # Kill the entire process group, not just the parent process (matches TUI)
                # The subprocess creates its own process group, so we need to kill that group
                logger.info(f"🛑 Killing process group for PID {self.current_process.pid}...")

                # Get the process group ID (should be same as PID since subprocess calls os.setpgrp())
                process_group_id = self.current_process.pid

                # Kill entire process group (negative PID kills process group)
                import os
                import signal
                os.killpg(process_group_id, signal.SIGTERM)

                # Give processes time to exit gracefully
                import asyncio
                await asyncio.sleep(1)

                # Force kill if still alive
                try:
                    os.killpg(process_group_id, signal.SIGKILL)
                    logger.info(f"🛑 Force killed process group {process_group_id}")
                except ProcessLookupError:
                    logger.info(f"🛑 Process group {process_group_id} already terminated")

                # Reset execution state
                self.execution_state = "idle"
                self.current_process = None

                # Update orchestrator states
                for orchestrator in self.orchestrators.values():
                    if orchestrator.state == OrchestratorState.EXECUTING:
                        orchestrator._state = OrchestratorState.COMPILED

                self.status_message.emit("Execution terminated by user")
                self.update_button_states()

                # Emit signal for log viewer
                self.subprocess_log_stopped.emit()

            except Exception as e:
                logger.warning(f"🛑 Error killing process group: {e}, falling back to single process kill")
                # Fallback to killing just the main process (original behavior)
                self.current_process.terminate()
                try:
                    self.current_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.current_process.kill()
                    self.current_process.wait()

                # Reset state even on fallback
                self.execution_state = "idle"
                self.current_process = None
                self.status_message.emit("Execution terminated by user")
                self.update_button_states()
                self.subprocess_log_stopped.emit()
        else:
            self.service_adapter.show_info_dialog("No execution is currently running.")
    
    def action_code_plate(self):
        """Handle Code Generation button (placeholder)."""
        self.service_adapter.show_info_dialog("Code generation not yet implemented in PyQt6 version.")
    
    def action_save_python_script(self):
        """Handle Save Python Script button (placeholder)."""
        self.service_adapter.show_info_dialog("Script saving not yet implemented in PyQt6 version.")
    
    # ========== UI Helper Methods ==========
    
    def update_plate_list(self):
        """Update the plate list widget using selection preservation mixin."""
        def format_plate_item(plate):
            """Format plate item for display."""
            display_text = f"{plate['name']} ({plate['path']})"

            # Add status indicators
            status_indicators = []
            if plate['path'] in self.orchestrators:
                orchestrator = self.orchestrators[plate['path']]
                if orchestrator.state == OrchestratorState.READY:
                    status_indicators.append("✓ Init")
                elif orchestrator.state == OrchestratorState.COMPILED:
                    status_indicators.append("✓ Compiled")
                elif orchestrator.state == OrchestratorState.EXECUTING:
                    status_indicators.append("🔄 Running")
                elif orchestrator.state == OrchestratorState.COMPLETED:
                    status_indicators.append("✅ Complete")
                elif orchestrator.state == OrchestratorState.COMPILE_FAILED:
                    status_indicators.append("❌ Compile Failed")
                elif orchestrator.state == OrchestratorState.EXEC_FAILED:
                    status_indicators.append("❌ Exec Failed")

            if status_indicators:
                display_text = f"[{', '.join(status_indicators)}] {display_text}"

            return display_text, plate

        def update_func():
            """Update function that clears and rebuilds the list."""
            self.plate_list.clear()

            for plate in self.plates:
                display_text, plate_data = format_plate_item(plate)
                item = QListWidgetItem(display_text)
                item.setData(Qt.ItemDataRole.UserRole, plate_data)

                # Add tooltip
                if plate['path'] in self.orchestrators:
                    orchestrator = self.orchestrators[plate['path']]
                    item.setToolTip(f"Status: {orchestrator.state.value}")

                self.plate_list.addItem(item)

            # Auto-select first plate if no selection and plates exist
            if self.plates and not self.selected_plate_path:
                self.plate_list.setCurrentRow(0)

        # Use utility to preserve selection during update
        preserve_selection_during_update(
            self.plate_list,
            lambda item_data: item_data['path'] if isinstance(item_data, dict) and 'path' in item_data else str(item_data),
            lambda: bool(self.orchestrators),
            update_func
        )
        self.update_button_states()
    
    def get_selected_plates(self) -> List[Dict]:
        """
        Get currently selected plates.
        
        Returns:
            List of selected plate dictionaries
        """
        selected_items = []
        for item in self.plate_list.selectedItems():
            plate_data = item.data(Qt.ItemDataRole.UserRole)
            if plate_data:
                selected_items.append(plate_data)
        return selected_items
    
    def update_button_states(self):
        """Update button enabled/disabled states based on selection."""
        selected_plates = self.get_selected_plates()
        has_selection = len(selected_plates) > 0
        has_initialized = any(plate['path'] in self.orchestrators for plate in selected_plates)
        has_compiled = any(plate['path'] in self.plate_compiled_data for plate in selected_plates)
        is_running = self.is_any_plate_running()

        # Update button states (logic extracted from Textual version)
        self.buttons["del_plate"].setEnabled(has_selection and not is_running)
        self.buttons["edit_config"].setEnabled(has_initialized and not is_running)
        self.buttons["init_plate"].setEnabled(has_selection and not is_running)
        self.buttons["compile_plate"].setEnabled(has_initialized and not is_running)
        self.buttons["code_plate"].setEnabled(has_initialized and not is_running)
        self.buttons["save_python_script"].setEnabled(has_initialized and not is_running)

        # Run button - enabled if plates are compiled or if currently running (for stop)
        if is_running:
            self.buttons["run_plate"].setEnabled(True)
            self.buttons["run_plate"].setText("Stop")
        else:
            self.buttons["run_plate"].setEnabled(has_compiled)
            self.buttons["run_plate"].setText("Run")
    
    def is_any_plate_running(self) -> bool:
        """
        Check if any plate is currently running.
        
        Returns:
            True if any plate is running, False otherwise
        """
        return self.execution_state == "running"
    
    def update_status(self, message: str):
        """
        Update status label.
        
        Args:
            message: Status message to display
        """
        self.status_label.setText(message)
    
    def on_selection_changed(self):
        """Handle plate list selection changes using utility."""
        def on_selected(selected_plates):
            self.selected_plate_path = selected_plates[0]['path']
            self.plate_selected.emit(self.selected_plate_path)

        def on_cleared():
            self.selected_plate_path = ""

        # Use utility to handle selection with prevention
        handle_selection_change_with_prevention(
            self.plate_list,
            self.get_selected_plates,
            lambda item_data: item_data['path'] if isinstance(item_data, dict) and 'path' in item_data else str(item_data),
            lambda: bool(self.orchestrators),
            lambda: self.selected_plate_path,
            on_selected,
            on_cleared
        )

        self.update_button_states()





    def on_item_double_clicked(self, item: QListWidgetItem):
        """Handle double-click on plate item."""
        plate_data = item.data(Qt.ItemDataRole.UserRole)
        if plate_data:
            # Double-click could trigger initialization or configuration
            if plate_data['path'] not in self.orchestrators:
                self.run_async_action(self.action_init_plate)
    
    def on_orchestrator_state_changed(self, plate_path: str, state: str):
        """
        Handle orchestrator state changes.
        
        Args:
            plate_path: Path of the plate
            state: New orchestrator state
        """
        self.update_plate_list()
        logger.debug(f"Orchestrator state changed: {plate_path} -> {state}")
    
    def on_config_changed(self, new_config: GlobalPipelineConfig):
        """
        Handle global configuration changes.

        Args:
            new_config: New global configuration
        """
        self.global_config = new_config

        # Apply new global config to all existing orchestrators
        # This rebuilds their pipeline configs preserving concrete values
        for orchestrator in self.orchestrators.values():
            self._update_orchestrator_global_config(orchestrator, new_config)

        # Update thread-local storage for currently selected orchestrator
        # This ensures step forms resolve against the updated orchestrator defaults
        if self.selected_plate_path and self.selected_plate_path in self.orchestrators:
            current_orchestrator = self.orchestrators[self.selected_plate_path]
            effective_config = current_orchestrator.get_effective_config()
            from openhcs.core.config import set_current_global_config, GlobalPipelineConfig
            set_current_global_config(GlobalPipelineConfig, effective_config)
            logger.debug(f"Updated thread-local storage for currently selected orchestrator: {self.selected_plate_path}")

        # Update thread-local storage for currently selected orchestrator
        if self.selected_plate_path and self.selected_plate_path in self.orchestrators:
            current_orchestrator = self.orchestrators[self.selected_plate_path]
            effective_config = current_orchestrator.get_effective_config()
            from openhcs.core.config import set_current_global_config, GlobalPipelineConfig
            set_current_global_config(GlobalPipelineConfig, effective_config)
            logger.debug(f"Updated thread-local storage for currently selected orchestrator: {self.selected_plate_path}")

        logger.info(f"Applied new global config to {len(self.orchestrators)} orchestrators")

        # Refresh placeholder text in any open parameter forms
        self._refresh_all_parameter_form_placeholders()

    def _refresh_all_parameter_form_placeholders(self) -> None:
        """
        Refresh placeholder text in all open parameter form windows.

        This ensures that lazy dataclass forms show updated placeholder text
        when the GlobalPipelineConfig changes.
        """
        # Check if there are any floating windows with parameter forms
        if hasattr(self, 'service_adapter') and hasattr(self.service_adapter, 'app'):
            app = self.service_adapter.app
            if hasattr(app, 'floating_windows'):
                for window in app.floating_windows.values():
                    # Get the widget from the window's layout
                    layout = window.layout()
                    if layout and layout.count() > 0:
                        widget = layout.itemAt(0).widget()
                        # Look for parameter form managers in the widget
                        self._refresh_widget_parameter_forms(widget)

    def _refresh_widget_parameter_forms(self, widget) -> None:
        """Recursively refresh parameter forms using functional patterns."""
        # Check if this widget has a parameter form manager
        if hasattr(widget, 'form_manager') and hasattr(widget.form_manager, 'refresh_placeholder_text'):
            widget.form_manager.refresh_placeholder_text()

        # Functional pattern: process children with filter and map
        if hasattr(widget, 'children'):
            # Direct refresh for widgets with refresh_placeholder_text
            [child.refresh_placeholder_text()
             for child in widget.children()
             if hasattr(child, 'refresh_placeholder_text')]

            # Recursive refresh for other widgets
            [self._refresh_widget_parameter_forms(child)
             for child in widget.children()
             if not hasattr(child, 'refresh_placeholder_text')]

    # ========== Helper Methods ==========

    def _get_current_pipeline_definition(self, plate_path: str) -> List:
        """
        Get the current pipeline definition for a plate.

        Args:
            plate_path: Path to the plate

        Returns:
            List of pipeline steps or empty list if no pipeline
        """
        if not self.pipeline_editor:
            logger.warning("No pipeline editor reference - using empty pipeline")
            return []

        # Get pipeline for specific plate or current plate
        target_plate = plate_path or getattr(self.pipeline_editor, 'current_plate', None)
        if not target_plate:
            logger.warning("No plate specified - using empty pipeline")
            return []

        # Get pipeline from editor (should return List[FunctionStep] directly)
        if hasattr(self.pipeline_editor, 'get_pipeline_for_plate'):
            pipeline_steps = self.pipeline_editor.get_pipeline_for_plate(target_plate)
        elif hasattr(self.pipeline_editor, 'pipeline_steps'):
            # Fallback to current pipeline steps if get_pipeline_for_plate not available
            pipeline_steps = getattr(self.pipeline_editor, 'pipeline_steps', [])
        else:
            logger.warning("Pipeline editor doesn't have expected methods - using empty pipeline")
            return []

        return pipeline_steps or []

    def set_pipeline_editor(self, pipeline_editor):
        """
        Set the pipeline editor reference.

        Args:
            pipeline_editor: Pipeline editor widget instance
        """
        self.pipeline_editor = pipeline_editor
        logger.debug("Pipeline editor reference set in plate manager")

    async def _start_monitoring(self):
        """Start monitoring subprocess execution."""
        if not self.current_process:
            return

        # Simple monitoring - check if process is still running
        def check_process():
            if self.current_process and self.current_process.poll() is not None:
                # Process has finished
                return_code = self.current_process.returncode
                logger.info(f"Subprocess finished with return code: {return_code}")

                # Reset execution state
                self.execution_state = "idle"
                self.current_process = None

                # Update orchestrator states based on return code
                for orchestrator in self.orchestrators.values():
                    if orchestrator.state == OrchestratorState.EXECUTING:
                        if return_code == 0:
                            orchestrator._state = OrchestratorState.COMPLETED
                        else:
                            orchestrator._state = OrchestratorState.EXEC_FAILED

                if return_code == 0:
                    self.status_message.emit("Execution completed successfully")
                else:
                    self.status_message.emit(f"Execution failed with code {return_code}")

                self.update_button_states()

                # Emit signal for log viewer
                self.subprocess_log_stopped.emit()

                return False  # Stop monitoring
            return True  # Continue monitoring

        # Monitor process in background
        while check_process():
            await asyncio.sleep(1)  # Check every second

    def _on_progress_started(self, max_value: int):
        """Handle progress started signal (main thread)."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(max_value)
        self.progress_bar.setValue(0)

    def _on_progress_updated(self, value: int):
        """Handle progress updated signal (main thread)."""
        self.progress_bar.setValue(value)

    def _on_progress_finished(self):
        """Handle progress finished signal (main thread)."""
        self.progress_bar.setVisible(False)

    def _handle_compilation_error(self, plate_name: str, error_message: str):
        """Handle compilation error on main thread (slot)."""
        self.service_adapter.show_error_dialog(f"Compilation failed for {plate_name}: {error_message}")

    def _handle_initialization_error(self, plate_name: str, error_message: str):
        """Handle initialization error on main thread (slot)."""
        self.service_adapter.show_error_dialog(f"Failed to initialize {plate_name}: {error_message}")
