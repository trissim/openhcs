"""
Configuration Window for PyQt6

Configuration editing dialog with full feature parity to Textual TUI version.
Uses hybrid approach: extracted business logic + clean PyQt6 UI.
"""

import logging
import dataclasses
from dataclasses import fields
from typing import Type, Any, Callable, Optional, Dict, Protocol, Union
from functools import partial
from abc import ABC, abstractmethod

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QScrollArea, QWidget, QFormLayout, QGroupBox, QFrame,
    QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

# SignatureAnalyzer import removed - no longer needed with ResetOperation approach
from openhcs.pyqt_gui.widgets.shared.parameter_form_manager import ParameterFormManager
from openhcs.pyqt_gui.shared.style_generator import StyleSheetGenerator
from openhcs.pyqt_gui.shared.color_scheme import PyQt6ColorScheme
from openhcs.core.config import GlobalPipelineConfig

# Import PyQt6 help components
from openhcs.pyqt_gui.widgets.shared.clickable_help_components import GroupBoxWithHelp, LabelWithHelp

logger = logging.getLogger(__name__)


# ========== FUNCTIONAL ABSTRACTIONS FOR CONFIG RESET ==========

class FormManagerProtocol(Protocol):
    """Protocol defining the interface for form managers."""
    def update_parameter(self, param_name: str, value: Any) -> None: ...
    def get_current_values(self) -> Dict[str, Any]: ...


class DataclassIntrospector:
    """Pure functional dataclass introspection and analysis."""

    @staticmethod
    def is_lazy_dataclass(instance: Any) -> bool:
        """Check if an instance is a lazy dataclass."""
        return hasattr(instance, '_resolve_field_value')

    @staticmethod
    def get_static_defaults(config_class: Type) -> Dict[str, Any]:
        """Get static default values from dataclass definition."""
        return {
            field.name: field.default if field.default is not dataclasses.MISSING
                       else field.default_factory() if field.default_factory is not dataclasses.MISSING
                       else None
            for field in fields(config_class)
        }

    @staticmethod
    def get_lazy_reset_values(config_class: Type) -> Dict[str, Any]:
        """Get reset values for lazy dataclass (all None for lazy loading)."""
        return {field.name: None for field in fields(config_class)}

    @staticmethod
    def extract_field_values(dataclass_instance: Any) -> Dict[str, Any]:
        """Extract field values from a dataclass instance."""
        return {
            field.name: getattr(dataclass_instance, field.name)
            for field in fields(dataclass_instance)
        }


class ResetStrategy(ABC):
    """Abstract base class for reset strategies."""

    @abstractmethod
    def generate_reset_values(self, config_class: Type, current_config: Any) -> Dict[str, Any]:
        """Generate the values to reset to."""
        pass


class LazyAwareResetStrategy(ResetStrategy):
    """Strategy that respects lazy dataclass architecture."""

    def generate_reset_values(self, config_class: Type, current_config: Any) -> Dict[str, Any]:
        if DataclassIntrospector.is_lazy_dataclass(current_config):
            # For lazy dataclasses, we need to resolve to actual static defaults
            # instead of trying to create a new lazy instance with None values

            # Get the base class that the lazy dataclass is based on
            base_class = self._get_base_class_from_lazy(config_class)

            # Create a fresh instance of the base class to get static defaults
            static_defaults_instance = base_class()

            # Extract the field values from the static defaults
            resolved_values = {}
            for field in fields(config_class):
                resolved_values[field.name] = getattr(static_defaults_instance, field.name)

            return resolved_values
        else:
            # Regular dataclass: reset to static default values
            return DataclassIntrospector.get_static_defaults(config_class)

    def _get_base_class_from_lazy(self, lazy_class: Type) -> Type:
        """Extract the base class from a lazy dataclass."""
        # For PipelineConfig, the base class is GlobalPipelineConfig
        # We can determine this from the to_base_config method
        if hasattr(lazy_class, 'to_base_config'):
            # Create a dummy instance to inspect the to_base_config method
            dummy_instance = lazy_class()
            base_instance = dummy_instance.to_base_config()
            return type(base_instance)

        # Fallback: assume the lazy class name pattern and import the base class
        from openhcs.core.config import GlobalPipelineConfig
        return GlobalPipelineConfig


class FormManagerUpdater:
    """Pure functional form manager update operations."""

    @staticmethod
    def apply_values_to_form_manager(
        form_manager: FormManagerProtocol,
        values: Dict[str, Any],
        modified_values_tracker: Optional[Dict[str, Any]] = None
    ) -> None:
        """Apply values to form manager and optionally track modifications."""
        for param_name, value in values.items():
            form_manager.update_parameter(param_name, value)
            if modified_values_tracker is not None:
                modified_values_tracker[param_name] = value

    @staticmethod
    def apply_nested_reset_recursively(
        form_manager: Any,
        config_class: Type,
        current_config: Any
    ) -> None:
        """Apply reset values to nested form managers recursively."""
        if not hasattr(form_manager, 'nested_managers'):
            return

        for nested_param_name, nested_manager in form_manager.nested_managers.items():
            # Get the nested dataclass type and current instance
            nested_field = next(
                (f for f in fields(config_class) if f.name == nested_param_name),
                None
            )

            if nested_field and dataclasses.is_dataclass(nested_field.type):
                nested_config_class = nested_field.type
                nested_current_config = getattr(current_config, nested_param_name, None) if current_config else None

                # Generate reset values for nested dataclass with mixed state support
                if nested_current_config and DataclassIntrospector.is_lazy_dataclass(nested_current_config):
                    # Lazy dataclass: support mixed states - preserve individual field lazy behavior
                    nested_reset_values = {}
                    for field in fields(nested_config_class):
                        # For lazy dataclasses, always reset to None to preserve lazy behavior
                        # This allows individual fields to maintain placeholder behavior
                        nested_reset_values[field.name] = None
                else:
                    # Regular concrete dataclass: reset to static defaults
                    nested_reset_values = DataclassIntrospector.get_static_defaults(nested_config_class)

                # Apply reset values to nested manager
                FormManagerUpdater.apply_values_to_form_manager(nested_manager, nested_reset_values)

                # Recurse for deeper nesting
                FormManagerUpdater.apply_nested_reset_recursively(
                    nested_manager, nested_config_class, nested_current_config
                )
            else:
                # Fallback: reset using parameter info
                FormManagerUpdater._reset_manager_to_parameter_defaults(nested_manager)

    @staticmethod
    def _reset_manager_to_parameter_defaults(manager: Any) -> None:
        """Reset a manager to its parameter defaults."""
        if (hasattr(manager, 'textual_form_manager') and
            hasattr(manager.textual_form_manager, 'parameter_info')):
            default_values = {
                param_name: param_info.default_value
                for param_name, param_info in manager.textual_form_manager.parameter_info.items()
            }
            FormManagerUpdater.apply_values_to_form_manager(manager, default_values)


class ResetOperation:
    """Immutable reset operation that respects lazy dataclass architecture."""

    def __init__(self, strategy: ResetStrategy, config_class: Type, current_config: Any):
        self.strategy = strategy
        self.config_class = config_class
        self.current_config = current_config
        self._reset_values = None

    @property
    def reset_values(self) -> Dict[str, Any]:
        """Lazy computation of reset values."""
        if self._reset_values is None:
            self._reset_values = self.strategy.generate_reset_values(
                self.config_class, self.current_config
            )
        return self._reset_values

    def apply_to_form_manager(
        self,
        form_manager: FormManagerProtocol,
        modified_values_tracker: Optional[Dict[str, Any]] = None
    ) -> None:
        """Apply this reset operation to a form manager."""
        # Apply top-level reset values
        FormManagerUpdater.apply_values_to_form_manager(
            form_manager, self.reset_values, modified_values_tracker
        )

        # Apply nested reset values recursively
        FormManagerUpdater.apply_nested_reset_recursively(
            form_manager, self.config_class, self.current_config
        )

    @classmethod
    def create_lazy_aware_reset(cls, config_class: Type, current_config: Any) -> 'ResetOperation':
        """Factory method for lazy-aware reset operations."""
        return cls(LazyAwareResetStrategy(), config_class, current_config)

    @classmethod
    def create_custom_reset(cls, strategy: ResetStrategy, config_class: Type, current_config: Any) -> 'ResetOperation':
        """Factory method for custom reset operations."""
        return cls(strategy, config_class, current_config)


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
                 on_save_callback: Optional[Callable] = None,
                 color_scheme: Optional[PyQt6ColorScheme] = None, parent=None,
                 orchestrator=None):
        """
        Initialize the configuration window.

        Args:
            config_class: Configuration class type
            current_config: Current configuration instance
            on_save_callback: Function to call when config is saved
            color_scheme: Color scheme for styling (optional, uses default if None)
            parent: Parent widget
            orchestrator: Optional orchestrator reference for context persistence
        """
        super().__init__(parent)

        # Business logic state (extracted from Textual version)
        self.config_class = config_class
        self.current_config = current_config
        self.on_save_callback = on_save_callback
        self.orchestrator = orchestrator  # Store orchestrator reference for context persistence

        # Initialize color scheme and style generator
        self.color_scheme = color_scheme or PyQt6ColorScheme()
        self.style_generator = StyleSheetGenerator(self.color_scheme)

        # Determine placeholder prefix based on dataclass type
        from openhcs.core.config import LazyDefaultPlaceholderService
        is_lazy_dataclass = LazyDefaultPlaceholderService.has_lazy_resolution(config_class)
        placeholder_prefix = "Pipeline default" if is_lazy_dataclass else "Default"

        # Always use ParameterFormManager with dataclass editing mode - unified approach
        self.form_manager = ParameterFormManager.from_dataclass_instance(
            dataclass_instance=current_config,
            field_id="config",
            placeholder_prefix=placeholder_prefix,
            color_scheme=self.color_scheme,
            use_scroll_area=True
        )

        # No config_editor needed - everything goes through form_manager
        self.config_editor = None

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
            # Use scroll area for configs with more than 8 fields (PipelineConfig has ~12 fields)
            return field_count > 8

        # For non-dataclass configs, use scroll area
        return True

    def setup_ui(self):
        """Setup the user interface."""
        self.setWindowTitle(f"Configuration - {self.config_class.__name__}")
        self.setModal(False)  # Non-modal like plate manager and pipeline editor
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
        header_label.setStyleSheet(f"color: {self.color_scheme.to_hex(self.color_scheme.text_accent)};")
        header_layout.addWidget(header_label)

        # Add help button for the dataclass itself
        if dataclasses.is_dataclass(self.config_class):
            from openhcs.pyqt_gui.widgets.shared.clickable_help_components import HelpButton
            help_btn = HelpButton(help_target=self.config_class, text="Help", color_scheme=self.color_scheme)
            help_btn.setMaximumWidth(80)
            header_layout.addWidget(help_btn)

        header_layout.addStretch()
        layout.addWidget(header_widget)
        
        # Parameter form - always use form_manager (unified approach)
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
        
        # Apply centralized styling
        self.setStyleSheet(self.style_generator.generate_config_window_style())
    
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
            # Get current value - preserve None values for lazy dataclasses
            if hasattr(self.current_config, '_resolve_field_value'):
                current_value = object.__getattribute__(self.current_config, param_name) if hasattr(self.current_config, param_name) else param_info.default_value
            else:
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
                    param_type=param_info.param_type,
                    color_scheme=self.color_scheme
                )
                label_with_help.setStyleSheet(f"color: {self.color_scheme.to_hex(self.color_scheme.text_secondary)}; font-weight: normal;")
                
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
        panel.setStyleSheet(f"""
            QFrame {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.panel_bg)};
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.border_color)};
                border-radius: 3px;
                padding: 10px;
            }}
        """)
        
        layout = QHBoxLayout(panel)
        layout.addStretch()
        
        # Reset button
        reset_button = QPushButton("Reset to Defaults")
        reset_button.setMinimumWidth(120)
        reset_button.clicked.connect(self.reset_to_defaults)
        button_styles = self.style_generator.generate_config_button_styles()
        reset_button.setStyleSheet(button_styles["reset"])
        layout.addWidget(reset_button)
        
        layout.addSpacing(10)
        
        # Cancel button
        cancel_button = QPushButton("Cancel")
        cancel_button.setMinimumWidth(80)
        cancel_button.clicked.connect(self.reject)
        cancel_button.setStyleSheet(button_styles["cancel"])
        layout.addWidget(cancel_button)
        
        # Save button
        save_button = QPushButton("Save")
        save_button.setMinimumWidth(80)
        save_button.clicked.connect(self.save_config)
        save_button.setStyleSheet(button_styles["save"])
        layout.addWidget(save_button)
        
        return panel
    
    def setup_connections(self):
        """Setup signal/slot connections."""
        self.config_saved.connect(self.on_config_saved)
        self.config_cancelled.connect(self.on_config_cancelled)

        # Always use form manager parameter changes (unified approach)
        self.form_manager.parameter_changed.connect(self._handle_parameter_change)

    def _handle_parameter_change(self, param_name: str, value):
        """Handle parameter change from form manager (mirrors Textual TUI)."""
        # No need to track modifications - form manager maintains state correctly
        pass
    
    def load_current_values(self):
        """Load current configuration values into widgets."""
        # The form manager already loads current values during initialization
        # This method is kept for compatibility but doesn't need to do anything
        # since the form manager handles widget initialization with current values
        pass
    
    def handle_parameter_change(self, param_name: str, value: Any):
        """
        Handle parameter value changes.

        Args:
            param_name: Name of the parameter
            value: New parameter value
        """
        # Form manager handles state correctly - no tracking needed
        pass
    
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
        """Reset all parameters including nested dataclass fields using proper reset operation."""
        # Use the sophisticated ResetOperation that handles nested dataclass fields properly
        reset_operation = ResetOperation.create_lazy_aware_reset(self.config_class, self.current_config)

        # Apply the reset operation to the form manager - this handles both top-level and nested parameters
        reset_operation.apply_to_form_manager(self.form_manager)

        # Refresh placeholder text to ensure UI shows correct defaults
        if hasattr(self.form_manager, 'refresh_placeholder_text'):
            self.form_manager.refresh_placeholder_text()

        logger.debug("Reset all parameters including nested fields using ResetOperation")

    def save_config(self):
        """Save the configuration preserving lazy behavior for unset fields."""
        try:
            # Always use form manager to get dataclass instance (unified approach)
            new_config = self.form_manager.get_dataclass_instance()

            # Emit signal and call callback
            self.config_saved.emit(new_config)

            if self.on_save_callback:
                self.on_save_callback(new_config)

            self.accept()

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
