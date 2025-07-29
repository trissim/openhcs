"""
Configuration Window for PyQt6

Configuration editing dialog with full feature parity to Textual TUI version.
Uses hybrid approach: extracted business logic + clean PyQt6 UI.
"""

import logging
import dataclasses
from typing import Type, Any, Callable, Optional, Dict

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QScrollArea, QWidget, QFormLayout, QGroupBox, QFrame,
    QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from openhcs.textual_tui.widgets.shared.signature_analyzer import SignatureAnalyzer
from openhcs.pyqt_gui.widgets.shared.parameter_form_manager import ParameterFormManager

# Import PyQt6 help components
from openhcs.pyqt_gui.widgets.shared.clickable_help_components import GroupBoxWithHelp, LabelWithHelp

logger = logging.getLogger(__name__)


class ConfigWindow(QDialog):
    """
    PyQt6 Configuration Window.
    
    Configuration editing dialog with parameter forms and validation.
    Preserves all business logic from Textual version with clean PyQt6 UI.
    """
    
    # Signals
    config_saved = pyqtSignal(object)  # saved config
    config_cancelled = pyqtSignal()
    
    def __init__(self, config_class: Type, current_config: Any, 
                 on_save_callback: Optional[Callable] = None, parent=None):
        """
        Initialize the configuration window.
        
        Args:
            config_class: Configuration class type
            current_config: Current configuration instance
            on_save_callback: Function to call when config is saved
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Business logic state (extracted from Textual version)
        self.config_class = config_class
        self.current_config = current_config
        self.on_save_callback = on_save_callback

        # Create config form using shared parameter form manager (mirrors Textual TUI)
        param_info = SignatureAnalyzer.analyze(config_class)

        # Get current parameter values from config instance
        parameters = {}
        parameter_types = {}

        for name, info in param_info.items():
            current_value = getattr(current_config, name, info.default_value)
            parameters[name] = current_value
            parameter_types[name] = info.param_type

        # Create parameter form manager (reuses Textual TUI logic)
        self.form_manager = ParameterFormManager(
            parameters, parameter_types, "config", param_info
        )

        # Setup UI
        self.setup_ui()
        self.setup_connections()

        logger.debug(f"Config window initialized for {config_class.__name__}")

    def _should_use_scroll_area(self) -> bool:
        """Determine if scroll area should be used based on config complexity."""
        # For simple dataclasses with few fields, don't use scroll area
        # This ensures dataclass fields show in full as requested
        if dataclasses.is_dataclass(self.config_class):
            field_count = len(dataclasses.fields(self.config_class))
            # Use scroll area only for complex configs with many fields
            return field_count > 15

        # For non-dataclass configs, use scroll area
        return True

    def setup_ui(self):
        """Setup the user interface."""
        self.setWindowTitle(f"Configuration - {self.config_class.__name__}")
        self.setModal(True)
        self.setMinimumSize(600, 400)
        self.resize(800, 600)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Header with help functionality for dataclass
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(10, 10, 10, 10)

        header_label = QLabel(f"Configure {self.config_class.__name__}")
        header_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        header_label.setStyleSheet("color: #00aaff;")
        header_layout.addWidget(header_label)

        # Add help button for the dataclass itself
        if dataclasses.is_dataclass(self.config_class):
            from openhcs.pyqt_gui.widgets.shared.clickable_help_components import HelpButton
            help_btn = HelpButton(help_target=self.config_class, text="Help")
            help_btn.setMaximumWidth(80)
            header_layout.addWidget(help_btn)

        header_layout.addStretch()
        layout.addWidget(header_widget)
        
        # Parameter form - use scroll area only for complex configs, not simple dataclasses
        if self._should_use_scroll_area():
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll_area.setWidget(self.form_manager)
            layout.addWidget(scroll_area)
        else:
            # For simple dataclasses, show form directly without scrolling
            layout.addWidget(self.form_manager)
        
        # Button panel
        button_panel = self.create_button_panel()
        layout.addWidget(button_panel)
        
        # Set styling
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
                color: white;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #00aaff;
            }
            QLabel {
                color: #cccccc;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #404040;
                color: white;
                border: 1px solid #666666;
                border-radius: 3px;
                padding: 5px;
            }
            QCheckBox {
                color: white;
            }
        """)
    
    def create_parameter_form(self) -> QWidget:
        """
        Create the parameter form using extracted business logic.
        
        Returns:
            Widget containing parameter form
        """
        form_widget = QWidget()
        main_layout = QVBoxLayout(form_widget)
        
        # Group parameters by category (simplified grouping)
        basic_params = {}
        advanced_params = {}
        
        for param_name, param_info in self.parameter_info.items():
            # Simple categorization based on parameter name
            if any(keyword in param_name.lower() for keyword in ['debug', 'verbose', 'advanced', 'experimental']):
                advanced_params[param_name] = param_info
            else:
                basic_params[param_name] = param_info
        
        # Create basic parameters group
        if basic_params:
            basic_group = self.create_parameter_group("Basic Settings", basic_params)
            main_layout.addWidget(basic_group)
        
        # Create advanced parameters group
        if advanced_params:
            advanced_group = self.create_parameter_group("Advanced Settings", advanced_params)
            main_layout.addWidget(advanced_group)
        
        main_layout.addStretch()
        return form_widget
    
    def create_parameter_group(self, group_name: str, parameters: Dict) -> QGroupBox:
        """
        Create a parameter group.
        
        Args:
            group_name: Name of the parameter group
            parameters: Dictionary of parameters
            
        Returns:
            QGroupBox containing the parameters
        """
        group_box = QGroupBox(group_name)
        layout = QFormLayout(group_box)
        
        for param_name, param_info in parameters.items():
            # Get current value
            current_value = getattr(self.current_config, param_name, param_info.default_value)
            
            # Create parameter widget
            widget = self.create_parameter_widget(param_name, param_info.param_type, current_value)
            if widget:
                # Parameter label with help functionality
                label_text = param_name.replace('_', ' ').title()
                param_description = param_info.description

                # Use LabelWithHelp for parameter help
                label_with_help = LabelWithHelp(
                    text=label_text,
                    param_name=param_name,
                    param_description=param_description,
                    param_type=param_info.param_type
                )
                label_with_help.setStyleSheet("color: #cccccc; font-weight: normal;")
                
                # Add to form
                layout.addRow(label_with_help, widget)
                self.parameter_widgets[param_name] = widget
        
        return group_box
    
    def create_parameter_widget(self, param_name: str, param_type: type, current_value: Any) -> Optional[QWidget]:
        """
        Create parameter widget based on type.
        
        Args:
            param_name: Parameter name
            param_type: Parameter type
            current_value: Current parameter value
            
        Returns:
            Widget for parameter editing or None
        """
        try:
            # Boolean parameters
            if param_type == bool:
                widget = QCheckBox()
                widget.setChecked(bool(current_value))
                widget.toggled.connect(lambda checked: self.handle_parameter_change(param_name, checked))
                return widget
            
            # Integer parameters
            elif param_type == int:
                widget = QSpinBox()
                widget.setRange(-999999, 999999)
                widget.setValue(int(current_value) if current_value is not None else 0)
                widget.valueChanged.connect(lambda value: self.handle_parameter_change(param_name, value))
                return widget
            
            # Float parameters
            elif param_type == float:
                widget = QDoubleSpinBox()
                widget.setRange(-999999.0, 999999.0)
                widget.setDecimals(6)
                widget.setValue(float(current_value) if current_value is not None else 0.0)
                widget.valueChanged.connect(lambda value: self.handle_parameter_change(param_name, value))
                return widget
            
            # Enum parameters
            elif any(base.__name__ == 'Enum' for base in param_type.__bases__):
                widget = QComboBox()
                for enum_value in param_type:
                    widget.addItem(str(enum_value.value), enum_value)
                
                # Set current value
                if current_value is not None:
                    for i in range(widget.count()):
                        if widget.itemData(i) == current_value:
                            widget.setCurrentIndex(i)
                            break
                
                widget.currentIndexChanged.connect(
                    lambda index: self.handle_parameter_change(param_name, widget.itemData(index))
                )
                return widget
            
            # String and other parameters
            else:
                widget = QLineEdit()
                widget.setText(str(current_value) if current_value is not None else "")
                widget.textChanged.connect(lambda text: self.handle_parameter_change(param_name, text))
                return widget
                
        except Exception as e:
            logger.warning(f"Failed to create widget for parameter {param_name}: {e}")
            return None
    
    def create_button_panel(self) -> QWidget:
        """
        Create the button panel.
        
        Returns:
            Widget containing action buttons
        """
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.Box)
        panel.setStyleSheet("""
            QFrame {
                background-color: #1e1e1e;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 10px;
            }
        """)
        
        layout = QHBoxLayout(panel)
        layout.addStretch()
        
        # Reset button
        reset_button = QPushButton("Reset to Defaults")
        reset_button.setMinimumWidth(120)
        reset_button.clicked.connect(self.reset_to_defaults)
        reset_button.setStyleSheet("""
            QPushButton {
                background-color: #666666;
                color: white;
                border: 1px solid #888888;
                border-radius: 3px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #777777;
            }
        """)
        layout.addWidget(reset_button)
        
        layout.addSpacing(10)
        
        # Cancel button
        cancel_button = QPushButton("Cancel")
        cancel_button.setMinimumWidth(80)
        cancel_button.clicked.connect(self.reject)
        cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #cc0000;
                color: white;
                border: 1px solid #ff0000;
                border-radius: 3px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #dd0000;
            }
        """)
        layout.addWidget(cancel_button)
        
        # Save button
        save_button = QPushButton("Save")
        save_button.setMinimumWidth(80)
        save_button.clicked.connect(self.save_config)
        save_button.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: 1px solid #106ebe;
                border-radius: 3px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
        """)
        layout.addWidget(save_button)
        
        return panel
    
    def setup_connections(self):
        """Setup signal/slot connections."""
        self.config_saved.connect(self.on_config_saved)
        self.config_cancelled.connect(self.on_config_cancelled)

        # Connect form manager parameter changes
        self.form_manager.parameter_changed.connect(self._handle_parameter_change)

    def _handle_parameter_change(self, param_name: str, value):
        """Handle parameter change from form manager (mirrors Textual TUI)."""
        # DON'T mutate the original config - just log the change
        # The form manager keeps the values internally like Textual TUI
        logger.debug(f"Config parameter changed: {param_name} = {value}")
    
    def load_current_values(self):
        """Load current configuration values into widgets."""
        for param_name, widget in self.parameter_widgets.items():
            current_value = getattr(self.current_config, param_name)
            self.update_widget_value(widget, current_value)
    
    def handle_parameter_change(self, param_name: str, value: Any):
        """
        Handle parameter value changes.
        
        Args:
            param_name: Name of the parameter
            value: New parameter value
        """
        self.modified_values[param_name] = value
        logger.debug(f"Parameter changed: {param_name} = {value}")
    
    def update_widget_value(self, widget: QWidget, value: Any):
        """
        Update widget value without triggering signals.
        
        Args:
            widget: Widget to update
            value: New value
        """
        # Temporarily block signals to avoid recursion
        widget.blockSignals(True)
        
        try:
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(value) if value is not None else 0)
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(value) if value is not None else 0.0)
            elif isinstance(widget, QComboBox):
                for i in range(widget.count()):
                    if widget.itemData(i) == value:
                        widget.setCurrentIndex(i)
                        break
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value) if value is not None else "")
        finally:
            widget.blockSignals(False)
    
    def reset_to_defaults(self):
        """Reset all parameters to default values."""
        for param_name, param_info in self.parameter_info.items():
            default_value = param_info.default_value
            
            # Update widget
            if param_name in self.parameter_widgets:
                widget = self.parameter_widgets[param_name]
                self.update_widget_value(widget, default_value)
            
            # Update modified values
            self.modified_values[param_name] = default_value
        
        logger.debug("Reset all parameters to defaults")
    
    def save_config(self):
        """Save the configuration using form manager values (mirrors Textual TUI)."""
        try:
            # Get current values from form manager
            form_values = self.form_manager.get_current_values()

            # Create new config instance
            new_config = self.config_class(**form_values)

            # Emit signal and call callback
            self.config_saved.emit(new_config)

            if self.on_save_callback:
                self.on_save_callback(new_config)

            self.accept()
            logger.debug("Configuration saved successfully")

        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Save Error", f"Failed to save configuration:\n{e}")
    
    def on_config_saved(self, config):
        """Handle config saved signal."""
        logger.debug(f"Config saved: {config}")
    
    def on_config_cancelled(self):
        """Handle config cancelled signal."""
        logger.debug("Config cancelled")
    
    def reject(self):
        """Handle dialog rejection (Cancel button)."""
        self.config_cancelled.emit()
        super().reject()
