"""
Pipeline Editor Widget for PyQt6

Pipeline step management with full feature parity to Textual TUI version.
Uses hybrid approach: extracted business logic + clean PyQt6 UI.
"""

import logging
import asyncio
import inspect
from typing import List, Dict, Optional, Callable, Tuple
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget, 
    QListWidgetItem, QLabel, QMessageBox, QFileDialog, QFrame,
    QSplitter, QTextEdit, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal, QMimeData
from PyQt6.QtGui import QFont, QDrag

from openhcs.core.config import GlobalPipelineConfig
from openhcs.io.filemanager import FileManager
from openhcs.core.steps.function_step import FunctionStep
from openhcs.pyqt_gui.widgets.mixins import (
    preserve_selection_during_update,
    handle_selection_change_with_prevention
)

logger = logging.getLogger(__name__)


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
    
    def __init__(self, file_manager: FileManager, service_adapter, parent=None):
        """
        Initialize the pipeline editor widget.
        
        Args:
            file_manager: FileManager instance for file operations
            service_adapter: PyQt service adapter for dialogs and operations
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Core dependencies
        self.file_manager = file_manager
        self.service_adapter = service_adapter
        self.global_config = service_adapter.get_global_config()
        
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
        title_label.setStyleSheet("color: #00aaff; padding: 5px;")
        layout.addWidget(title_label)
        
        # Main content splitter
        splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(splitter)
        
        # Pipeline steps list
        self.step_list = QListWidget()
        self.step_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.step_list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.step_list.setStyleSheet("""
            QListWidget {
                background-color: #1e1e1e;
                color: white;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #333333;
                border-radius: 3px;
                margin: 2px;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
            }
            QListWidget::item:hover {
                background-color: #333333;
            }
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
        panel.setStyleSheet("""
            QFrame {
                background-color: #2b2b2b;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 5px;
            }
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
                    button.setStyleSheet("""
                        QPushButton {
                            background-color: #404040;
                            color: white;
                            border: 1px solid #666666;
                            border-radius: 3px;
                            padding: 5px;
                        }
                        QPushButton:hover {
                            background-color: #505050;
                        }
                        QPushButton:pressed {
                            background-color: #303030;
                        }
                        QPushButton:disabled {
                            background-color: #2a2a2a;
                            color: #666666;
                        }
                    """)
                    
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
        frame.setStyleSheet("""
            QFrame {
                background-color: #2b2b2b;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 5px;
            }
        """)
        
        layout = QVBoxLayout(frame)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #00ff00; font-weight: bold;")
        layout.addWidget(self.status_label)
        
        return frame
    
    def setup_connections(self):
        """Setup signal/slot connections."""
        # Step list selection
        self.step_list.itemSelectionChanged.connect(self.on_selection_changed)
        self.step_list.itemDoubleClicked.connect(self.on_item_double_clicked)
        
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
        # Validate orchestrator is selected before allowing step creation
        if not self._validate_orchestrator_selected():
            return

        from openhcs.core.steps.function_step import FunctionStep
        from openhcs.pyqt_gui.windows.dual_editor_window import DualEditorWindow

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

        # Create and show editor dialog
        editor = DualEditorWindow(
            step_data=new_step,
            is_new=True,
            on_save_callback=handle_save,
            parent=self
        )
        editor.exec()
    
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
        # Validate orchestrator is selected before allowing step editing
        if not self._validate_orchestrator_selected():
            return

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

        # Create and show editor dialog
        editor = DualEditorWindow(
            step_data=step_to_edit,
            is_new=False,
            on_save_callback=handle_save,
            parent=self
        )
        editor.exec()
    
    def action_load_pipeline(self):
        """Handle Load Pipeline button (adapted from Textual version)."""
        # Validate orchestrator is selected before allowing pipeline load
        if not self._validate_orchestrator_selected():
            return

        from openhcs.pyqt_gui.utils.path_cache import PathCacheKey

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

        from openhcs.pyqt_gui.utils.path_cache import PathCacheKey

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
        """Handle Code Pipeline button (placeholder)."""
        self.service_adapter.show_info_dialog("Pipeline code editing not yet implemented in PyQt6 version.")
    
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
        """Update button enabled/disabled states based on selection and orchestrator."""
        has_steps = len(self.pipeline_steps) > 0
        has_selection = len(self.get_selected_steps()) > 0
        has_orchestrator = bool(self.current_plate)  # Only allow operations when orchestrator is selected

        # Update button states - require orchestrator for editing operations
        self.buttons["add_step"].setEnabled(has_orchestrator)
        self.buttons["load_pipeline"].setEnabled(has_orchestrator)
        self.buttons["del_step"].setEnabled(has_selection and has_orchestrator)
        self.buttons["edit_step"].setEnabled(has_selection and has_orchestrator)
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

    def _validate_orchestrator_selected(self) -> bool:
        """Validate that an orchestrator is selected before allowing pipeline operations."""
        if not self.current_plate:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "No Orchestrator Selected",
                "Please select and initialize a plate in the Plate Manager before editing pipelines.\n\n"
                "Pipeline editing requires an active orchestrator to provide component information."
            )
            return False
        return True

    def on_config_changed(self, new_config: GlobalPipelineConfig):
        """
        Handle global configuration changes.
        
        Args:
            new_config: New global configuration
        """
        self.global_config = new_config
