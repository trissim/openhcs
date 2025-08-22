"""
Step Parameter Editor Widget for PyQt6 GUI.

Mirrors the Textual TUI StepParameterEditorWidget with type hint-based form generation.
Handles FunctionStep parameter editing with nested dataclass support.
"""

import logging
from typing import Any, Optional
from pathlib import Path
from contextlib import contextmanager

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QScrollArea, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal

from openhcs.core.steps.function_step import FunctionStep
from openhcs.textual_tui.widgets.shared.signature_analyzer import SignatureAnalyzer
from openhcs.pyqt_gui.widgets.shared.parameter_form_manager import ParameterFormManager
from openhcs.pyqt_gui.shared.color_scheme import PyQt6ColorScheme
from openhcs.pyqt_gui.shared.color_scheme import PyQt6ColorScheme

logger = logging.getLogger(__name__)


class StepParameterEditorWidget(QScrollArea):
    """
    Step parameter editor using dynamic form generation.
    
    Mirrors Textual TUI implementation - builds forms based on FunctionStep 
    constructor signature with nested dataclass support.
    """
    
    # Signals
    step_parameter_changed = pyqtSignal()
    
    def __init__(self, step: FunctionStep, service_adapter=None, color_scheme: Optional[PyQt6ColorScheme] = None, orchestrator=None, parent=None):
        super().__init__(parent)

        # Initialize color scheme
        self.color_scheme = color_scheme or PyQt6ColorScheme()

        self.step = step
        self.service_adapter = service_adapter
        self.orchestrator = orchestrator  # Store orchestrator reference for context management
        
        # Analyze AbstractStep signature to get all inherited parameters (mirrors Textual TUI)
        from openhcs.core.steps.abstract import AbstractStep
        # Auto-detection correctly identifies constructors and includes all parameters
        param_info = SignatureAnalyzer.analyze(AbstractStep.__init__)
        
        # Get current parameter values from step instance
        parameters = {}
        parameter_types = {}
        param_defaults = {}

        for name, info in param_info.items():
            # All AbstractStep parameters are relevant for editing
            # ParameterFormManager will automatically route lazy dataclass parameters to LazyDataclassEditor
            current_value = getattr(self.step, name, info.default_value)

            # Generic handling for any optional lazy dataclass parameter that exists in PipelineConfig
            if current_value is None and self._is_optional_lazy_dataclass_in_pipeline(info.param_type, name):
                # Create step-level config for proper inheritance hierarchy
                step_level_config = self._create_step_level_config(name, info.param_type)
                current_value = step_level_config
                param_defaults[name] = step_level_config
                # Mark this as a step-level config for special handling
                if not hasattr(self, '_step_level_configs'):
                    self._step_level_configs = {}
                self._step_level_configs[name] = True
            else:
                param_defaults[name] = info.default_value

            parameters[name] = current_value
            parameter_types[name] = info.param_type
        
        # Create parameter form manager for function parameters
        # Note: Step editor needs special context setup to show step-level inheritance

        self.form_manager = ParameterFormManager(
            parameters, parameter_types, "step", None,
            param_info,
            parent=self,  # Pass self as parent so form manager can access _step_level_configs
            color_scheme=self.color_scheme,
            placeholder_prefix="Pipeline default",
            param_defaults=param_defaults
        )
        
        self.setup_ui()
        self.setup_connections()

        logger.debug(f"Step parameter editor initialized for step: {getattr(step, 'name', 'Unknown')}")

    def _is_optional_lazy_dataclass_in_pipeline(self, param_type, param_name):
        """
        Check if parameter is an optional lazy dataclass that exists in PipelineConfig.

        This enables automatic step-level config creation for any parameter that:
        1. Is Optional[SomeDataclass]
        2. SomeDataclass exists as a field type in PipelineConfig (type-based matching)
        3. The dataclass has lazy resolution capabilities

        No manual mappings needed - uses type-based discovery.
        """
        from openhcs.core.pipeline_config import PipelineConfig
        from openhcs.ui.shared.parameter_type_utils import ParameterTypeUtils
        import dataclasses

        # Check if parameter is Optional[dataclass]
        if not ParameterTypeUtils.is_optional_dataclass(param_type):
            return False

        # Get the inner dataclass type
        inner_type = ParameterTypeUtils.get_optional_inner_type(param_type)

        # Find if this type exists as a field in PipelineConfig (type-based matching)
        pipeline_field_name = self._find_pipeline_field_by_type(inner_type)
        if not pipeline_field_name:
            return False

        # Check if the dataclass has lazy resolution capabilities
        try:
            # Try to create an instance to see if it's a lazy dataclass
            test_instance = inner_type()
            # Check for lazy dataclass methods
            return hasattr(test_instance, '_resolve_field_value') or hasattr(test_instance, '_lazy_resolution_config')
        except:
            return False

    def _find_pipeline_field_by_type(self, target_type):
        """
        Find the field in PipelineConfig that matches the target type.

        This is type-based discovery - no manual mappings needed.
        """
        from openhcs.core.pipeline_config import PipelineConfig
        import dataclasses

        for field in dataclasses.fields(PipelineConfig):
            # Use string comparison to handle type identity issues
            if str(field.type) == str(target_type):
                return field.name
        return None

    def _create_step_level_config(self, param_name, param_type):
        """
        Generic method to create step-level config for any lazy dataclass parameter.

        Uses type-based discovery to find the corresponding pipeline field as defaults source.
        """
        from openhcs.core.lazy_config import LazyDataclassFactory
        from openhcs.core.config import GlobalPipelineConfig, get_current_global_config
        from openhcs.ui.shared.parameter_type_utils import ParameterTypeUtils

        # Get the inner dataclass type
        inner_type = ParameterTypeUtils.get_optional_inner_type(param_type)

        # Find the corresponding pipeline field by type (no manual mapping needed)
        pipeline_field_name = self._find_pipeline_field_by_type(inner_type)
        if not pipeline_field_name:
            # Fallback to standard lazy config if no matching type found
            return inner_type()

        # Get pipeline's corresponding field as defaults source
        pipeline_config = get_current_global_config(GlobalPipelineConfig)
        if pipeline_config and hasattr(pipeline_config, pipeline_field_name):
            pipeline_field_value = getattr(pipeline_config, pipeline_field_name)
            if pipeline_field_value:
                # Create step-level config that inherits from pipeline's field
                StepLevelConfig = LazyDataclassFactory.create_lazy_dataclass(
                    defaults_source=pipeline_field_value,
                    lazy_class_name=f"StepLevel{inner_type.__name__}",
                    use_recursive_resolution=False
                )
                return StepLevelConfig()

        # Fallback to standard lazy config if no pipeline context
        return inner_type()




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
        header_label.setStyleSheet(f"color: {self.color_scheme.to_hex(self.color_scheme.text_accent)}; font-weight: bold; font-size: 14px;")
        layout.addWidget(header_label)
        
        # Parameter form (using shared form manager)
        # ParameterFormManager automatically routes lazy dataclass parameters to LazyDataclassEditor
        form_frame = QFrame()
        form_frame.setFrameStyle(QFrame.Shape.Box)
        form_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.panel_bg)};
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.border_color)};
                border-radius: 5px;
                padding: 10px;
            }}
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
                background-color: {self.color_scheme.to_hex(self.color_scheme.input_bg)};
                color: white;
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.border_light)};
                border-radius: 3px;
                padding: 6px 12px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: {self.color_scheme.to_hex(self.color_scheme.button_hover_bg)};
            }
            QPushButton:pressed {
                background-color: {self.color_scheme.to_hex(self.color_scheme.button_pressed_bg)};
            }
        """
    
    def setup_connections(self):
        """Setup signal/slot connections."""
        # Connect form manager parameter changes
        self.form_manager.parameter_changed.connect(self._handle_parameter_change)
    
    def _handle_parameter_change(self, param_name: str, value: Any):
        """Handle parameter change from form manager (mirrors Textual TUI)."""
        try:
            # Get the properly converted value from the form manager
            # The form manager handles all type conversions including List[Enum]
            final_value = self.form_manager.get_current_values().get(param_name, value)

            # Debug: Check what we're actually saving
            if param_name == 'materialization_config':
                print(f"DEBUG: Saving materialization_config, type: {type(final_value)}")
                print(f"DEBUG: Raw value from form manager: {value}")
                print(f"DEBUG: Final value from get_current_values(): {final_value}")
                if hasattr(final_value, '__dataclass_fields__'):
                    from dataclasses import fields
                    for field_obj in fields(final_value):
                        raw_value = object.__getattribute__(final_value, field_obj.name)
                        print(f"DEBUG: Field {field_obj.name} = {raw_value}")

            # Update step attribute
            setattr(self.step, param_name, final_value)
            logger.debug(f"Updated step parameter {param_name}={final_value}")
            self.step_parameter_changed.emit()

        except Exception as e:
            logger.error(f"Error updating step parameter {param_name}: {e}")

    def load_step_settings(self):
        """Load step settings from .step file (mirrors Textual TUI)."""
        if not self.service_adapter:
            logger.warning("No service adapter available for file dialog")
            return
        
        from openhcs.core.path_cache import PathCacheKey
        
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
        
        from openhcs.core.path_cache import PathCacheKey
        
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
