"""
Step Parameter Editor Widget for PyQt6 GUI.

Mirrors the Textual TUI StepParameterEditorWidget with type hint-based form generation.
Handles FunctionStep parameter editing with nested dataclass support.
"""

import logging
from typing import Any
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QScrollArea, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal

from openhcs.core.steps.function_step import FunctionStep
from openhcs.textual_tui.widgets.shared.signature_analyzer import SignatureAnalyzer
from openhcs.pyqt_gui.widgets.shared.parameter_form_manager import ParameterFormManager

logger = logging.getLogger(__name__)


class StepParameterEditorWidget(QScrollArea):
    """
    Step parameter editor using dynamic form generation.
    
    Mirrors Textual TUI implementation - builds forms based on FunctionStep 
    constructor signature with nested dataclass support.
    """
    
    # Signals
    step_parameter_changed = pyqtSignal()
    
    def __init__(self, step: FunctionStep, service_adapter=None, parent=None):
        super().__init__(parent)
        
        self.step = step
        self.service_adapter = service_adapter
        
        # Analyze FunctionStep signature (mirrors Textual TUI)
        param_info = SignatureAnalyzer.analyze(FunctionStep.__init__)
        
        # Get current parameter values from step instance
        parameters = {}
        parameter_types = {}
        param_defaults = {}
        
        for name, info in param_info.items():
            if name in ('func',):  # Skip func parameter
                continue
            current_value = getattr(self.step, name)
            parameters[name] = current_value
            parameter_types[name] = info.param_type
            param_defaults[name] = info.default_value
        
        # Create parameter form manager (reuses Textual TUI logic)
        self.form_manager = ParameterFormManager(
            parameters, parameter_types, "step", param_info
        )
        self.param_defaults = param_defaults
        
        self.setup_ui()
        self.setup_connections()
        
        logger.debug(f"Step parameter editor initialized for step: {getattr(step, 'name', 'Unknown')}")
    
    def setup_ui(self):
        """Setup the user interface."""
        self.setWidgetResizable(True)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Main content widget
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)
        
        # Header
        header_label = QLabel("Step Parameters")
        header_label.setStyleSheet("color: #4a9eff; font-weight: bold; font-size: 14px;")
        layout.addWidget(header_label)
        
        # Parameter form (using shared form manager)
        form_frame = QFrame()
        form_frame.setFrameStyle(QFrame.Shape.Box)
        form_frame.setStyleSheet("""
            QFrame {
                background-color: #1e1e1e;
                border: 1px solid #555555;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        
        form_layout = QVBoxLayout(form_frame)
        
        # Add parameter form manager
        form_layout.addWidget(self.form_manager)
        
        layout.addWidget(form_frame)
        
        # Action buttons (mirrors Textual TUI)
        button_layout = QHBoxLayout()
        
        load_btn = QPushButton("Load .step")
        load_btn.setMaximumWidth(100)
        load_btn.setStyleSheet(self._get_button_style())
        load_btn.clicked.connect(self.load_step_settings)
        button_layout.addWidget(load_btn)
        
        save_btn = QPushButton("Save .step As")
        save_btn.setMaximumWidth(120)
        save_btn.setStyleSheet(self._get_button_style())
        save_btn.clicked.connect(self.save_step_settings)
        button_layout.addWidget(save_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        layout.addStretch()
        
        self.setWidget(content_widget)
    
    def _get_button_style(self) -> str:
        """Get consistent button styling."""
        return """
            QPushButton {
                background-color: #404040;
                color: white;
                border: 1px solid #666666;
                border-radius: 3px;
                padding: 6px 12px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #505050;
            }
            QPushButton:pressed {
                background-color: #303030;
            }
        """
    
    def setup_connections(self):
        """Setup signal/slot connections."""
        # Connect form manager parameter changes
        self.form_manager.parameter_changed.connect(self._handle_parameter_change)
    
    def _handle_parameter_change(self, param_name: str, value: Any):
        """Handle parameter change from form manager (mirrors Textual TUI)."""
        try:
            # Convert value to appropriate type
            if param_name == 'force_disk_output':
                value = bool(value)
            elif param_name in ('input_dir', 'output_dir') and value:
                value = Path(value)
            
            # Update step attribute
            setattr(self.step, param_name, value)
            logger.debug(f"Updated step parameter {param_name}={value}")
            self.step_parameter_changed.emit()
            
        except Exception as e:
            logger.error(f"Error updating step parameter {param_name}: {e}")
    
    def load_step_settings(self):
        """Load step settings from .step file (mirrors Textual TUI)."""
        if not self.service_adapter:
            logger.warning("No service adapter available for file dialog")
            return
        
        from openhcs.pyqt_gui.utils.path_cache import PathCacheKey
        
        file_path = self.service_adapter.show_cached_file_dialog(
            cache_key=PathCacheKey.STEP_SETTINGS,
            title="Load Step Settings (.step)",
            file_filter="Step Files (*.step);;All Files (*)",
            mode="open"
        )
        
        if file_path:
            self._load_step_settings_from_file(file_path)
    
    def save_step_settings(self):
        """Save step settings to .step file (mirrors Textual TUI)."""
        if not self.service_adapter:
            logger.warning("No service adapter available for file dialog")
            return
        
        from openhcs.pyqt_gui.utils.path_cache import PathCacheKey
        
        file_path = self.service_adapter.show_cached_file_dialog(
            cache_key=PathCacheKey.STEP_SETTINGS,
            title="Save Step Settings (.step)",
            file_filter="Step Files (*.step);;All Files (*)",
            mode="save"
        )
        
        if file_path:
            self._save_step_settings_to_file(file_path)
    
    def _load_step_settings_from_file(self, file_path: Path):
        """Load step settings from file."""
        try:
            # TODO: Implement step settings loading
            logger.debug(f"Load step settings from {file_path} - TODO: implement")
            
        except Exception as e:
            logger.error(f"Failed to load step settings from {file_path}: {e}")
            if self.service_adapter:
                self.service_adapter.show_error_dialog(f"Failed to load step settings: {e}")
    
    def _save_step_settings_to_file(self, file_path: Path):
        """Save step settings to file."""
        try:
            # TODO: Implement step settings saving
            logger.debug(f"Save step settings to {file_path} - TODO: implement")
            
        except Exception as e:
            logger.error(f"Failed to save step settings to {file_path}: {e}")
            if self.service_adapter:
                self.service_adapter.show_error_dialog(f"Failed to save step settings: {e}")
    
    def reset_parameter(self, param_name: str):
        """Reset parameter to default value."""
        if param_name in self.param_defaults:
            default_value = self.param_defaults[param_name]
            setattr(self.step, param_name, default_value)
            
            # Update form manager
            self.form_manager.update_parameter(param_name, default_value)
            
            self.step_parameter_changed.emit()
            logger.debug(f"Reset parameter {param_name} to default: {default_value}")
    
    def reset_all_parameters(self):
        """Reset all parameters to default values."""
        for param_name, default_value in self.param_defaults.items():
            setattr(self.step, param_name, default_value)
            self.form_manager.update_parameter(param_name, default_value)
        
        self.step_parameter_changed.emit()
        logger.debug("Reset all step parameters to defaults")
    
    def get_current_step(self) -> FunctionStep:
        """Get the current step with all parameter values."""
        return self.step
    
    def update_step(self, step: FunctionStep):
        """Update the step and refresh the form."""
        self.step = step
        
        # Update form manager with new values
        for param_name in self.form_manager.parameters.keys():
            current_value = getattr(self.step, param_name, None)
            self.form_manager.update_parameter(param_name, current_value)
        
        logger.debug(f"Updated step parameter editor for step: {getattr(step, 'name', 'Unknown')}")
