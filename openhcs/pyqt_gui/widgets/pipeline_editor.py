"""
Pipeline Editor Widget for PyQt6

Pipeline step management with full feature parity to Textual TUI version.
Uses hybrid approach: extracted business logic + clean PyQt6 UI.
"""

import logging
import asyncio
import inspect
import contextlib
from typing import List, Dict, Optional, Callable, Tuple
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget,
    QListWidgetItem, QLabel, QMessageBox, QFileDialog, QFrame,
    QSplitter, QTextEdit, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal, QMimeData
from PyQt6.QtGui import QFont, QDrag

from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.core.config import GlobalPipelineConfig, set_current_global_config, get_current_global_config
from openhcs.io.filemanager import FileManager
from openhcs.core.steps.function_step import FunctionStep
from openhcs.pyqt_gui.widgets.mixins import (
    preserve_selection_during_update,
    handle_selection_change_with_prevention
)
from openhcs.pyqt_gui.shared.style_generator import StyleSheetGenerator
from openhcs.pyqt_gui.shared.color_scheme import PyQt6ColorScheme

logger = logging.getLogger(__name__)


class ReorderableListWidget(QListWidget):
    """
    Custom QListWidget that properly handles drag and drop reordering.
    Emits a signal when items are moved so the parent can update the data model.
    """

    items_reordered = pyqtSignal(int, int)  # from_index, to_index

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragDropMode(QListWidget.DragDropMode.InternalMove)

    def dropEvent(self, event):
        """Handle drop events and emit reorder signal."""
        # Get the item being dropped and its original position
        source_item = self.currentItem()
        if not source_item:
            super().dropEvent(event)
            return

        source_index = self.row(source_item)

        # Let the default drop behavior happen first
        super().dropEvent(event)

        # Find the new position of the item
        target_index = self.row(source_item)

        # Only emit signal if position actually changed
        if source_index != target_index:
            self.items_reordered.emit(source_index, target_index)


class PipelineEditorWidget(QWidget):
    """
    PyQt6 Pipeline Editor Widget.
    
    Manages pipeline steps with add, edit, delete, load, save functionality.
    Preserves all business logic from Textual version with clean PyQt6 UI.
    """
    
    # Signals
    pipeline_changed = pyqtSignal(list)  # List[FunctionStep]
    step_selected = pyqtSignal(object)  # FunctionStep
    status_message = pyqtSignal(str)  # status message
    
    def __init__(self, file_manager: FileManager, service_adapter,
                 color_scheme: Optional[PyQt6ColorScheme] = None, parent=None):
        """
        Initialize the pipeline editor widget.

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

        # Initialize color scheme and style generator
        self.color_scheme = color_scheme or service_adapter.get_current_color_scheme()
        self.style_generator = StyleSheetGenerator(self.color_scheme)
        
        # Business logic state (extracted from Textual version)
        self.pipeline_steps: List[FunctionStep] = []
        self.current_plate: str = ""
        self.selected_step: str = ""
        self.plate_pipelines: Dict[str, List[FunctionStep]] = {}  # Per-plate pipeline storage
        
        # UI components
        self.step_list: Optional[QListWidget] = None
        self.buttons: Dict[str, QPushButton] = {}
        self.status_label: Optional[QLabel] = None
        
        # Reference to plate manager (set externally)
        self.plate_manager = None
        
        # Setup UI
        self.setup_ui()
        self.setup_connections()
        self.update_button_states()
        
        logger.debug("Pipeline editor widget initialized")

    # ========== UI Setup ==========

    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Title
        title_label = QLabel("Pipeline Editor")
        title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        title_label.setStyleSheet(f"color: {self.color_scheme.to_hex(self.color_scheme.text_accent)}; padding: 5px;")
        layout.addWidget(title_label)
        
        # Main content splitter
        splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(splitter)
        
        # Pipeline steps list
        self.step_list = ReorderableListWidget()
        self.step_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.step_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.panel_bg)};
                color: {self.color_scheme.to_hex(self.color_scheme.text_primary)};
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.border_color)};
                border-radius: 3px;
                padding: 5px;
            }}
            QListWidget::item {{
                padding: 8px;
                border-bottom: 1px solid {self.color_scheme.to_hex(self.color_scheme.separator_color)};
                border-radius: 3px;
                margin: 2px;
            }}
            QListWidget::item:selected {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.selection_bg)};
            }}
            QListWidget::item:hover {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.hover_bg)};
            }}
        """)
        splitter.addWidget(self.step_list)
        
        # Button panel
        button_panel = self.create_button_panel()
        splitter.addWidget(button_panel)
        
        # Status section
        status_frame = self.create_status_section()
        layout.addWidget(status_frame)
        
        # Set splitter proportions
        splitter.setSizes([400, 120])
    
    def create_button_panel(self) -> QWidget:
        """
        Create the button panel with all pipeline actions.
        
        Returns:
            Widget containing action buttons
        """
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.Box)
        panel.setStyleSheet(f"""
            QFrame {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.frame_bg)};
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.border_color)};
                border-radius: 3px;
                padding: 5px;
            }}
        """)
        
        layout = QVBoxLayout(panel)
        
        # Button configurations (extracted from Textual version)
        button_configs = [
            ("Add", "add_step", "Add new pipeline step"),
            ("Del", "del_step", "Delete selected steps"),
            ("Edit", "edit_step", "Edit selected step"),
            ("Load", "load_pipeline", "Load pipeline from file"),
            ("Save", "save_pipeline", "Save pipeline to file"),
            ("Code", "code_pipeline", "Edit pipeline as Python code"),
        ]
        
        # Create buttons in rows
        for i in range(0, len(button_configs), 3):
            row_layout = QHBoxLayout()
            
            for j in range(3):
                if i + j < len(button_configs):
                    name, action, tooltip = button_configs[i + j]
                    
                    button = QPushButton(name)
                    button.setToolTip(tooltip)
                    button.setMinimumHeight(30)
                    button.setStyleSheet(self.style_generator.generate_button_style())
                    
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
        Create the status section.
        
        Returns:
            Widget containing status information
        """
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.Box)
        frame.setStyleSheet(f"""
            QFrame {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.frame_bg)};
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.border_color)};
                border-radius: 3px;
                padding: 5px;
            }}
        """)
        
        layout = QVBoxLayout(frame)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(f"color: {self.color_scheme.to_hex(self.color_scheme.status_success)}; font-weight: bold;")
        layout.addWidget(self.status_label)
        
        return frame
    
    def setup_connections(self):
        """Setup signal/slot connections."""
        # Step list selection
        self.step_list.itemSelectionChanged.connect(self.on_selection_changed)
        self.step_list.itemDoubleClicked.connect(self.on_item_double_clicked)

        # Step list reordering
        self.step_list.items_reordered.connect(self.on_steps_reordered)

        # Internal signals
        self.status_message.connect(self.update_status)
        self.pipeline_changed.connect(self.on_pipeline_changed)
    
    def handle_button_action(self, action: str):
        """
        Handle button actions (extracted from Textual version).
        
        Args:
            action: Action identifier
        """
        # Action mapping (preserved from Textual version)
        action_map = {
            "add_step": self.action_add_step,
            "del_step": self.action_delete_step,
            "edit_step": self.action_edit_step,
            "load_pipeline": self.action_load_pipeline,
            "save_pipeline": self.action_save_pipeline,
            "code_pipeline": self.action_code_pipeline,
        }
        
        if action in action_map:
            action_func = action_map[action]
            
            # Handle async actions
            if inspect.iscoroutinefunction(action_func):
                # Run async action in thread
                self.run_async_action(action_func)
            else:
                action_func()
    
    def run_async_action(self, async_func: Callable):
        """
        Run async action using service adapter.

        Args:
            async_func: Async function to execute
        """
        self.service_adapter.execute_async_operation(async_func)
    
    # ========== Business Logic Methods (Extracted from Textual) ==========
    
    def format_item_for_display(self, step: FunctionStep) -> Tuple[str, str]:
        """
        Format step for display in the list (extracted from Textual version).
        
        Args:
            step: FunctionStep to format
            
        Returns:
            Tuple of (display_text, step_name)
        """
        step_name = getattr(step, 'name', 'Unknown Step')
        display_text = f"📋 {step_name}"
        return display_text, step_name
    
    def action_add_step(self):
        """Handle Add Step button (adapted from Textual version)."""

        from openhcs.core.steps.function_step import FunctionStep
        from openhcs.pyqt_gui.windows.dual_editor_window import DualEditorWindow

        # Get orchestrator for step creation
        orchestrator = self._get_current_orchestrator()

        # Create new step
        step_name = f"Step_{len(self.pipeline_steps) + 1}"
        new_step = FunctionStep(
            func=[],  # Start with empty function list
            name=step_name
        )



        def handle_save(edited_step):
            """Handle step save from editor."""
            self.pipeline_steps.append(edited_step)
            self.update_step_list()
            self.pipeline_changed.emit(self.pipeline_steps)
            self.status_message.emit(f"Added new step: {edited_step.name}")

        # Create and show editor dialog within the correct config context
        orchestrator = self._get_current_orchestrator()
        with self._scoped_orchestrator_context():
            editor = DualEditorWindow(
                step_data=new_step,
                is_new=True,
                on_save_callback=handle_save,
                orchestrator=orchestrator,
                parent=self
            )
            # Set original step for change detection within the scoped context
            editor.set_original_step_for_change_detection()
        editor.show()
        editor.raise_()
        editor.activateWindow()
    
    def action_delete_step(self):
        """Handle Delete Step button (extracted from Textual version)."""
        selected_items = self.get_selected_steps()
        if not selected_items:
            self.service_adapter.show_error_dialog("No steps selected to delete.")
            return
        
        # Remove selected steps
        steps_to_remove = set(getattr(item, 'name', '') for item in selected_items)
        new_steps = [step for step in self.pipeline_steps if getattr(step, 'name', '') not in steps_to_remove]
        
        self.pipeline_steps = new_steps
        self.update_step_list()
        self.pipeline_changed.emit(self.pipeline_steps)
        
        deleted_count = len(selected_items)
        self.status_message.emit(f"Deleted {deleted_count} steps")
    
    def action_edit_step(self):
        """Handle Edit Step button (adapted from Textual version)."""
        selected_items = self.get_selected_steps()
        if not selected_items:
            self.service_adapter.show_error_dialog("No step selected to edit.")
            return

        step_to_edit = selected_items[0]

        # Open step editor dialog
        from openhcs.pyqt_gui.windows.dual_editor_window import DualEditorWindow

        def handle_save(edited_step):
            """Handle step save from editor."""
            # Find and replace the step in the pipeline
            for i, step in enumerate(self.pipeline_steps):
                if step is step_to_edit:
                    self.pipeline_steps[i] = edited_step
                    break

            # Update the display
            self.update_step_list()
            self.pipeline_changed.emit(self.pipeline_steps)
            self.status_message.emit(f"Updated step: {edited_step.name}")

        # Create and show editor dialog within the correct config context
        orchestrator = self._get_current_orchestrator()
        with self._scoped_orchestrator_context():
            editor = DualEditorWindow(
                step_data=step_to_edit,
                is_new=False,
                on_save_callback=handle_save,
                orchestrator=orchestrator,
                parent=self
            )
            # Set original step for change detection within the scoped context
            editor.set_original_step_for_change_detection()
        editor.show()
        editor.raise_()
        editor.activateWindow()
    
    def action_load_pipeline(self):
        """Handle Load Pipeline button (adapted from Textual version)."""

        from openhcs.core.path_cache import PathCacheKey

        # Use cached file dialog (mirrors Textual TUI pattern)
        file_path = self.service_adapter.show_cached_file_dialog(
            cache_key=PathCacheKey.PIPELINE_FILES,
            title="Load Pipeline",
            file_filter="Pipeline Files (*.pipeline);;All Files (*)",
            mode="open"
        )

        if file_path:
            self.load_pipeline_from_file(file_path)
    
    def action_save_pipeline(self):
        """Handle Save Pipeline button (adapted from Textual version)."""
        if not self.pipeline_steps:
            self.service_adapter.show_error_dialog("No pipeline steps to save.")
            return

        from openhcs.core.path_cache import PathCacheKey

        # Use cached file dialog (mirrors Textual TUI pattern)
        file_path = self.service_adapter.show_cached_file_dialog(
            cache_key=PathCacheKey.PIPELINE_FILES,
            title="Save Pipeline",
            file_filter="Pipeline Files (*.pipeline);;All Files (*)",
            mode="save"
        )

        if file_path:
            self.save_pipeline_to_file(file_path)
    
    def action_code_pipeline(self):
        """Handle Code Pipeline button - edit pipeline as Python code."""
        logger.debug("Code button pressed - opening code editor")

        if not self.pipeline_steps:
            self.service_adapter.show_error_dialog("No pipeline steps to edit")
            return

        if not self.current_plate:
            self.service_adapter.show_error_dialog("No plate selected")
            return

        try:
            # Use complete pipeline steps code generation
            from openhcs.debug.pickle_to_python import generate_complete_pipeline_steps_code

            # Generate complete pipeline steps code with imports
            python_code = generate_complete_pipeline_steps_code(
                pipeline_steps=list(self.pipeline_steps),
                clean_mode=False
            )

            # Create simple code editor service
            from openhcs.pyqt_gui.services.simple_code_editor import SimpleCodeEditorService
            editor_service = SimpleCodeEditorService(self)

            # Check if user wants external editor (check environment variable)
            import os
            use_external = os.environ.get('OPENHCS_USE_EXTERNAL_EDITOR', '').lower() in ('1', 'true', 'yes')

            # Launch editor with callback
            editor_service.edit_code(
                initial_content=python_code,
                title="Edit Pipeline Steps",
                callback=self._handle_edited_pipeline_code,
                use_external=use_external
            )

        except Exception as e:
            logger.error(f"Failed to open pipeline code editor: {e}")
            self.service_adapter.show_error_dialog(f"Failed to open code editor: {str(e)}")

    def _handle_edited_pipeline_code(self, edited_code: str) -> None:
        """Handle the edited pipeline code from code editor."""
        logger.debug("Pipeline code edited, processing changes...")
        try:
            # Ensure we have a string
            if not isinstance(edited_code, str):
                logger.error(f"Expected string, got {type(edited_code)}: {edited_code}")
                self.service_adapter.show_error_dialog("Invalid code format received from editor")
                return

            # Execute the code (it has all necessary imports)
            namespace = {}
            exec(edited_code, namespace)

            # Get the pipeline_steps from the namespace
            if 'pipeline_steps' in namespace:
                new_pipeline_steps = namespace['pipeline_steps']
                # Update the pipeline with new steps
                self.pipeline_steps = new_pipeline_steps
                self.update_step_list()
                self.pipeline_changed.emit(self.pipeline_steps)
                self.status_message.emit(f"Pipeline updated with {len(new_pipeline_steps)} steps")
            else:
                self.service_adapter.show_error_dialog("No 'pipeline_steps = [...]' assignment found in edited code")

        except SyntaxError as e:
            self.service_adapter.show_error_dialog(f"Invalid Python syntax: {e}")
        except Exception as e:
            logger.error(f"Failed to parse edited pipeline code: {e}")
            self.service_adapter.show_error_dialog(f"Failed to parse pipeline code: {str(e)}")
    
    def load_pipeline_from_file(self, file_path: Path):
        """
        Load pipeline from file (extracted from Textual version).
        
        Args:
            file_path: Path to pipeline file
        """
        try:
            import dill as pickle
            with open(file_path, 'rb') as f:
                steps = pickle.load(f)
            
            if isinstance(steps, list):
                self.pipeline_steps = steps
                self.update_step_list()
                self.pipeline_changed.emit(self.pipeline_steps)
                self.status_message.emit(f"Loaded {len(steps)} steps from {file_path.name}")
            else:
                self.status_message.emit(f"Invalid pipeline format in {file_path.name}")
                
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            self.service_adapter.show_error_dialog(f"Failed to load pipeline: {e}")
    
    def save_pipeline_to_file(self, file_path: Path):
        """
        Save pipeline to file (extracted from Textual version).
        
        Args:
            file_path: Path to save pipeline
        """
        try:
            import dill as pickle
            with open(file_path, 'wb') as f:
                pickle.dump(list(self.pipeline_steps), f)
            self.status_message.emit(f"Saved pipeline to {file_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to save pipeline: {e}")
            self.service_adapter.show_error_dialog(f"Failed to save pipeline: {e}")
    
    def save_pipeline_for_plate(self, plate_path: str, pipeline: List[FunctionStep]):
        """
        Save pipeline for specific plate (extracted from Textual version).
        
        Args:
            plate_path: Path of the plate
            pipeline: Pipeline steps to save
        """
        self.plate_pipelines[plate_path] = pipeline
        logger.debug(f"Saved pipeline for plate: {plate_path}")
    
    def set_current_plate(self, plate_path: str):
        """
        Set current plate and load its pipeline (extracted from Textual version).

        Args:
            plate_path: Path of the current plate
        """
        self.current_plate = plate_path

        # Load pipeline for the new plate
        if plate_path:
            plate_pipeline = self.plate_pipelines.get(plate_path, [])
            self.pipeline_steps = plate_pipeline
        else:
            self.pipeline_steps = []

        self.update_step_list()
        self.update_button_states()
        logger.debug(f"Current plate changed: {plate_path}")

    def on_orchestrator_config_changed(self, plate_path: str, effective_config):
        """
        Handle orchestrator configuration changes for placeholder refresh.

        Args:
            plate_path: Path of the plate whose orchestrator config changed
            effective_config: The orchestrator's new effective configuration
        """
        # Only refresh if this is for the current plate
        if plate_path == self.current_plate:
            logger.debug(f"Refreshing placeholders for orchestrator config change: {plate_path}")

            # Refresh any open step forms within the orchestrator's scoped context
            # This ensures step forms resolve against the updated effective config
            with self._scoped_orchestrator_context():
                # Trigger refresh of any open configuration windows or step forms
                # The scoped context ensures they resolve against the updated orchestrator config
                logger.debug(f"Step forms will now resolve against updated orchestrator config for: {plate_path}")
    
    # ========== UI Helper Methods ==========
    
    def update_step_list(self):
        """Update the step list widget using selection preservation mixin."""
        def format_step_item(step):
            """Format step item for display."""
            display_text, step_name = self.format_item_for_display(step)
            return display_text, step

        def update_func():
            """Update function that clears and rebuilds the list."""
            self.step_list.clear()

            for step in self.pipeline_steps:
                display_text, step_data = format_step_item(step)
                item = QListWidgetItem(display_text)
                item.setData(Qt.ItemDataRole.UserRole, step_data)
                item.setToolTip(f"Step: {getattr(step, 'name', 'Unknown')}")
                self.step_list.addItem(item)

        # Use utility to preserve selection during update
        preserve_selection_during_update(
            self.step_list,
            lambda item_data: getattr(item_data, 'name', str(item_data)),
            lambda: bool(self.pipeline_steps),
            update_func
        )
        self.update_button_states()
    
    def get_selected_steps(self) -> List[FunctionStep]:
        """
        Get currently selected steps.
        
        Returns:
            List of selected FunctionStep objects
        """
        selected_items = []
        for item in self.step_list.selectedItems():
            step_data = item.data(Qt.ItemDataRole.UserRole)
            if step_data:
                selected_items.append(step_data)
        return selected_items
    
    def update_button_states(self):
        """Update button enabled/disabled states based on mathematical constraints (mirrors Textual TUI)."""
        has_plate = bool(self.current_plate)
        is_initialized = self._is_current_plate_initialized()
        has_steps = len(self.pipeline_steps) > 0
        has_selection = len(self.get_selected_steps()) > 0

        # Mathematical constraints (mirrors Textual TUI logic):
        # - Pipeline editing requires initialization
        # - Step operations require steps to exist
        # - Edit requires valid selection
        self.buttons["add_step"].setEnabled(has_plate and is_initialized)
        self.buttons["load_pipeline"].setEnabled(has_plate and is_initialized)
        self.buttons["del_step"].setEnabled(has_steps)
        self.buttons["edit_step"].setEnabled(has_steps and has_selection)
        self.buttons["save_pipeline"].setEnabled(has_steps)
        self.buttons["code_pipeline"].setEnabled(has_steps)
    
    def update_status(self, message: str):
        """
        Update status label.
        
        Args:
            message: Status message to display
        """
        self.status_label.setText(message)
    
    def on_selection_changed(self):
        """Handle step list selection changes using utility."""
        def on_selected(selected_steps):
            self.selected_step = getattr(selected_steps[0], 'name', '')
            self.step_selected.emit(selected_steps[0])

        def on_cleared():
            self.selected_step = ""

        # Use utility to handle selection with prevention
        handle_selection_change_with_prevention(
            self.step_list,
            self.get_selected_steps,
            lambda item_data: getattr(item_data, 'name', str(item_data)),
            lambda: bool(self.pipeline_steps),
            lambda: self.selected_step,
            on_selected,
            on_cleared
        )

        self.update_button_states()

    def on_item_double_clicked(self, item: QListWidgetItem):
        """Handle double-click on step item."""
        step_data = item.data(Qt.ItemDataRole.UserRole)
        if step_data:
            # Double-click triggers edit
            self.action_edit_step()

    def on_steps_reordered(self, from_index: int, to_index: int):
        """
        Handle step reordering from drag and drop.

        Args:
            from_index: Original position of the moved step
            to_index: New position of the moved step
        """
        # Update the underlying pipeline_steps list to match the visual order
        current_steps = list(self.pipeline_steps)

        # Move the step in the data model
        step = current_steps.pop(from_index)
        current_steps.insert(to_index, step)

        # Update pipeline steps
        self.pipeline_steps = current_steps

        # Emit pipeline changed signal to notify other components
        self.pipeline_changed.emit(self.pipeline_steps)

        # Update status message
        step_name = getattr(step, 'name', 'Unknown Step')
        direction = "up" if to_index < from_index else "down"
        self.status_message.emit(f"Moved step '{step_name}' {direction}")

        logger.debug(f"Reordered step '{step_name}' from index {from_index} to {to_index}")

    def on_pipeline_changed(self, steps: List[FunctionStep]):
        """
        Handle pipeline changes.
        
        Args:
            steps: New pipeline steps
        """
        # Save pipeline to current plate if one is selected
        if self.current_plate:
            self.save_pipeline_for_plate(self.current_plate, steps)
        
        logger.debug(f"Pipeline changed: {len(steps)} steps")

    def _is_current_plate_initialized(self) -> bool:
        """Check if current plate has an initialized orchestrator (mirrors Textual TUI)."""
        if not self.current_plate:
            return False

        # Get plate manager from main window
        main_window = self._find_main_window()
        if not main_window:
            return False

        # Get plate manager widget from floating windows
        plate_manager_window = main_window.floating_windows.get("plate_manager")
        if not plate_manager_window:
            return False

        layout = plate_manager_window.layout()
        if not layout or layout.count() == 0:
            return False

        plate_manager_widget = layout.itemAt(0).widget()
        if not hasattr(plate_manager_widget, 'orchestrators'):
            return False

        orchestrator = plate_manager_widget.orchestrators.get(self.current_plate)
        if orchestrator is None:
            return False

        # Check if orchestrator is in an initialized state (mirrors Textual TUI logic)
        from openhcs.constants.constants import OrchestratorState
        return orchestrator.state in [OrchestratorState.READY, OrchestratorState.COMPILED,
                                     OrchestratorState.COMPLETED, OrchestratorState.COMPILE_FAILED,
                                     OrchestratorState.EXEC_FAILED]



    def _get_current_orchestrator(self) -> Optional[PipelineOrchestrator]:
        """Get the orchestrator for the currently selected plate."""
        if not self.current_plate:
            return None
        main_window = self._find_main_window()
        if not main_window:
            return None
        plate_manager_window = main_window.floating_windows.get("plate_manager")
        if not plate_manager_window:
            return None
        layout = plate_manager_window.layout()
        if not layout or layout.count() == 0:
            return None
        plate_manager_widget = layout.itemAt(0).widget()
        if not hasattr(plate_manager_widget, 'orchestrators'):
            return None
        return plate_manager_widget.orchestrators.get(self.current_plate)

    @contextlib.contextmanager
    def _scoped_orchestrator_context(self):
        """A context manager to temporarily set the thread-local config for the current orchestrator."""
        original_config = get_current_global_config(GlobalPipelineConfig)
        orchestrator = self._get_current_orchestrator()
        if orchestrator:
            effective_config = orchestrator.get_effective_config()
            set_current_global_config(GlobalPipelineConfig, effective_config)
            logger.debug(f"Set scoped config context for orchestrator: {orchestrator.plate_path}")
        
        try:
            yield
        finally:
            set_current_global_config(GlobalPipelineConfig, original_config)
            logger.debug("Restored original config context.")

    def _find_main_window(self):
        """Find the main window by traversing parent hierarchy."""
        widget = self
        while widget:
            if hasattr(widget, 'floating_windows'):
                return widget
            widget = widget.parent()
        return None

    def on_config_changed(self, new_config: GlobalPipelineConfig):
        """
        Handle global configuration changes.

        Args:
            new_config: New global configuration
        """
        self.global_config = new_config


