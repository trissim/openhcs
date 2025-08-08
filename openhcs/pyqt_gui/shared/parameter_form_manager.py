"""
PyQt6 Parameter Form Manager

Parameter form management for PyQt6 widgets.
Adapted from Textual TUI version with PyQt6 widget integration.
"""

import logging
from typing import Dict, Any, Type, Optional
from pathlib import Path

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QFormLayout, QLabel
from PyQt6.QtCore import pyqtSignal, QObject

from openhcs.textual_tui.widgets.shared.parameter_form_manager import ParameterFormManager
from openhcs.textual_tui.widgets.shared.signature_analyzer import ParameterInfo
from openhcs.pyqt_gui.shared.typed_widget_factory import TypedWidgetFactory
from openhcs.pyqt_gui.shared.color_scheme import PyQt6ColorScheme

logger = logging.getLogger(__name__)


class PyQtParameterFormManager(QObject):
    """
    PyQt6 Parameter Form Manager.
    
    Manages parameter forms for PyQt6 widgets, adapting the Textual TUI
    ParameterFormManager to work with PyQt6 widgets.
    """
    
    # Signals
    parameter_changed = pyqtSignal(str, object)  # param_name, value
    form_updated = pyqtSignal()
    
    def __init__(self, parameters: Dict[str, Any], parameter_types: Dict[str, Type],
                 form_id: str, param_info: Optional[Dict] = None,
                 color_scheme: Optional[PyQt6ColorScheme] = None):
        """
        Initialize the PyQt6 parameter form manager.

        Args:
            parameters: Dictionary of parameter names to values
            parameter_types: Dictionary of parameter names to types
            form_id: Unique identifier for this form
            param_info: Additional parameter information
            color_scheme: Color scheme for styling (optional, uses default if None)
        """
        super().__init__()

        # Initialize color scheme
        self.color_scheme = color_scheme or PyQt6ColorScheme()

        # Use the original ParameterFormManager for business logic
        self.core_manager = ParameterFormManager(parameters, parameter_types, form_id, param_info)

        # PyQt6 specific state
        self.parameter_widgets: Dict[str, QWidget] = {}
        self.widget_factory = TypedWidgetFactory(self.color_scheme)
        
        logger.debug(f"PyQt6 parameter form manager initialized: {form_id}")
    
    def create_form_widget(self, parent: Optional[QWidget] = None) -> QWidget:
        """
        Create a PyQt6 widget containing the parameter form.
        
        Args:
            parent: Parent widget
            
        Returns:
            Widget containing the parameter form
        """
        form_widget = QWidget(parent)
        layout = QFormLayout(form_widget)
        
        # Create widgets for each parameter
        for param_name, param_type in self.core_manager.parameter_types.items():
            current_value = self.core_manager.parameters[param_name]

            # Get parameter info for enhanced widget creation
            param_info = self.core_manager.parameter_info.get(param_name) if hasattr(self.core_manager, 'parameter_info') else None

            # Use enhanced widget creation for Path types only
            if self.widget_factory._is_path_type(param_type):
                widget = self.widget_factory.create_enhanced_widget(param_name, param_type, current_value, param_info)
            else:
                widget = self.widget_factory.create_widget(param_name, param_type, current_value)
            if widget:
                # Connect widget changes to parameter updates
                self.connect_widget_signals(widget, param_name)
                
                # Create label
                label = QLabel(param_name.replace('_', ' ').title())
                label.setStyleSheet(f"color: {self.color_scheme.to_hex(self.color_scheme.text_secondary)}; font-weight: normal;")
                
                # Add to form
                layout.addRow(label, widget)
                self.parameter_widgets[param_name] = widget
        
        return form_widget
    
    def connect_widget_signals(self, widget: QWidget, param_name: str):
        """
        Connect widget signals to parameter change handling.
        
        Args:
            widget: Widget to connect
            param_name: Parameter name
        """
        # Handle EnhancedPathWidget FIRST (duck typing)
        if hasattr(widget, 'path_changed'):
            widget.path_changed.connect(lambda text: self.handle_parameter_change(param_name, text))
            return

        # Connect appropriate signal based on widget type
        if hasattr(widget, 'textChanged'):
            widget.textChanged.connect(lambda text: self.handle_parameter_change(param_name, text))
        elif hasattr(widget, 'valueChanged'):
            widget.valueChanged.connect(lambda value: self.handle_parameter_change(param_name, value))
        elif hasattr(widget, 'toggled'):
            widget.toggled.connect(lambda checked: self.handle_parameter_change(param_name, checked))
        elif hasattr(widget, 'currentTextChanged'):
            widget.currentTextChanged.connect(lambda text: self.handle_parameter_change(param_name, text))
        elif hasattr(widget, 'currentIndexChanged'):
            widget.currentIndexChanged.connect(
                lambda index: self.handle_parameter_change(param_name, widget.itemData(index))
            )
    
    def handle_parameter_change(self, param_name: str, value: Any):
        """
        Handle parameter value changes.
        
        Args:
            param_name: Name of the parameter
            value: New parameter value
        """
        # Update core manager
        self.core_manager.update_parameter(param_name, value)
        
        # Emit signal
        self.parameter_changed.emit(param_name, value)
        
        logger.debug(f"Parameter changed: {param_name} = {value}")
    
    def update_parameter(self, param_name: str, value: Any):
        """
        Update parameter value programmatically.
        
        Args:
            param_name: Parameter name
            value: New value
        """
        # Update core manager
        self.core_manager.update_parameter(param_name, value)
        
        # Update widget if it exists
        if param_name in self.parameter_widgets:
            widget = self.parameter_widgets[param_name]
            self.widget_factory.update_widget_value(widget, value)
        
        # Emit signal
        self.parameter_changed.emit(param_name, value)
    
    def reset_parameter(self, param_name: str, default_value: Any):
        """
        Reset parameter to default value.
        
        Args:
            param_name: Parameter name
            default_value: Default value
        """
        self.update_parameter(param_name, default_value)
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current parameter values.
        
        Returns:
            Dictionary of parameter values
        """
        return self.core_manager.parameters.copy()
    
    def get_parameter_types(self) -> Dict[str, Type]:
        """
        Get parameter types.
        
        Returns:
            Dictionary of parameter types
        """
        return self.core_manager.parameter_types.copy()
    
    def validate_parameters(self) -> bool:
        """
        Validate all parameters.
        
        Returns:
            True if all parameters are valid, False otherwise
        """
        # Use core manager validation
        return self.core_manager.validate_all_parameters()
    
    def get_validation_errors(self) -> Dict[str, str]:
        """
        Get validation errors for parameters.
        
        Returns:
            Dictionary of parameter names to error messages
        """
        # Use core manager validation
        return self.core_manager.get_validation_errors()
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """Get current parameters."""
        return self.core_manager.parameters
    
    @property
    def parameter_types(self) -> Dict[str, Type]:
        """Get parameter types."""
        return self.core_manager.parameter_types


