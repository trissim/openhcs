"""
Parameter form manager for PyQt6 GUI.

REUSES the Textual TUI parameter form generation logic for consistent UX.
This is a PyQt6 adapter that uses the actual working Textual TUI services.
"""

import dataclasses
import logging
from typing import Any, Dict, get_origin, get_args, Union, Optional
from pathlib import Path
from enum import Enum

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QSpinBox,
    QDoubleSpinBox, QCheckBox, QComboBox, QPushButton, QGroupBox,
    QScrollArea, QFrame
)
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtCore import Qt, pyqtSignal

from openhcs.pyqt_gui.shared.color_scheme import PyQt6ColorScheme

# No-scroll widget classes to prevent accidental value changes
# Import no-scroll widgets from separate module
from .no_scroll_spinbox import NoScrollSpinBox, NoScrollDoubleSpinBox, NoScrollComboBox

# REUSE the actual working Textual TUI services
from openhcs.textual_tui.widgets.shared.signature_analyzer import SignatureAnalyzer, ParameterInfo
from openhcs.textual_tui.widgets.shared.parameter_form_manager import ParameterFormManager as TextualParameterFormManager
from openhcs.textual_tui.widgets.shared.typed_widget_factory import TypedWidgetFactory

# Import PyQt6 help components (using same pattern as Textual TUI)
from openhcs.pyqt_gui.widgets.shared.clickable_help_components import LabelWithHelp, GroupBoxWithHelp

# Import simplified abstraction layer
from openhcs.ui.shared.parameter_form_abstraction import (
    ParameterFormAbstraction, apply_lazy_default_placeholder
)
from openhcs.ui.shared.widget_creation_registry import create_pyqt6_registry
from openhcs.ui.shared.pyqt6_widget_strategies import PyQt6WidgetEnhancer

logger = logging.getLogger(__name__)


class ParameterFormManager(QWidget):
    """
    PyQt6 adapter for Textual TUI ParameterFormManager.

    REUSES the actual working Textual TUI parameter form logic by creating
    a PyQt6 UI that mirrors the Textual TUI behavior exactly.
    """

    parameter_changed = pyqtSignal(str, object)  # param_name, value

    def __init__(self, parameters: Dict[str, Any], parameter_types: Dict[str, type],
                 field_id: str, parameter_info: Dict = None, parent=None, use_scroll_area: bool = True,
                 function_target=None, color_scheme: Optional[PyQt6ColorScheme] = None):
        super().__init__(parent)

        # Initialize color scheme
        self.color_scheme = color_scheme or PyQt6ColorScheme()

        # Store function target for docstring fallback
        self._function_target = function_target

        # Initialize simplified abstraction layer
        self.form_abstraction = ParameterFormAbstraction(
            parameters, parameter_types, field_id, create_pyqt6_registry(), parameter_info
        )

        # Create the actual Textual TUI form manager (reuse the working logic for compatibility)
        self.textual_form_manager = TextualParameterFormManager(
            parameters, parameter_types, field_id, parameter_info
        )

        # Store field_id for PyQt6 widget creation
        self.field_id = field_id

        # Control whether to use scroll area (disable for nested dataclasses)
        self.use_scroll_area = use_scroll_area

        # Track PyQt6 widgets for value updates
        self.widgets = {}
        self.nested_managers = {}

        self.setup_ui()
    
    def setup_ui(self):
        """Setup the parameter form UI using Textual TUI logic."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Content widget
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)

        # Build form fields using Textual TUI parameter types and logic
        # Initialize logger for debug logging
        import logging
        logger = logging.getLogger(__name__)

        for param_name, param_type in self.textual_form_manager.parameter_types.items():
            current_value = self.textual_form_manager.parameters[param_name]

            # Handle Optional[dataclass] types with checkbox wrapper
            if self._is_optional_dataclass(param_type):
                # DEBUG: Log Optional dataclass detection
                logger.info(f"=== OPTIONAL DATACLASS DETECTED === {param_name}: {param_type}")

                inner_dataclass_type = self._get_optional_inner_type(param_type)
                field_widget = self._create_optional_dataclass_field(param_name, inner_dataclass_type, current_value)
            # Handle nested dataclasses (reuse Textual TUI logic)
            elif dataclasses.is_dataclass(param_type):
                # DEBUG: Log regular dataclass detection
                logger.info(f"=== REGULAR DATACLASS DETECTED === {param_name}: {param_type}")
                field_widget = self._create_nested_dataclass_field(param_name, param_type, current_value)
            else:
                field_widget = self._create_regular_parameter_field(param_name, param_type, current_value)

            if field_widget:
                content_layout.addWidget(field_widget)

        # Only use scroll area if requested (not for nested dataclasses)
        if self.use_scroll_area:
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            scroll_area.setWidget(content_widget)
            layout.addWidget(scroll_area)
        else:
            # Add content widget directly without scroll area
            layout.addWidget(content_widget)
    
    def _create_nested_dataclass_field(self, param_name: str, param_type: type, current_value: Any) -> QWidget:
        """Create a collapsible group for nested dataclass with help functionality."""
        # Use GroupBoxWithHelp to show dataclass documentation
        group_box = GroupBoxWithHelp(
            title=f"{param_name.replace('_', ' ').title()}",
            help_target=param_type,  # Show help for the dataclass type
            color_scheme=self.color_scheme
        )

        # Use the content layout from GroupBoxWithHelp
        layout = group_box.content_layout

        # Check if we need to create a lazy version of the nested dataclass
        nested_dataclass_for_form = self._create_lazy_nested_dataclass_if_needed(param_name, param_type, current_value)

        # Analyze nested dataclass
        nested_param_info = SignatureAnalyzer.analyze(param_type)

        # Get current values from nested dataclass instance
        nested_parameters = {}
        nested_parameter_types = {}

        for nested_name, nested_info in nested_param_info.items():
            if nested_dataclass_for_form:
                # For lazy dataclasses, preserve None values for storage but use resolved values for initialization
                if hasattr(nested_dataclass_for_form, '_resolve_field_value'):
                    # Get stored value (None if not explicitly set)
                    stored_value = object.__getattribute__(nested_dataclass_for_form, nested_name) if hasattr(nested_dataclass_for_form, nested_name) else None
                    if stored_value is not None:
                        # User has explicitly set this value, use it
                        nested_current_value = stored_value
                    else:
                        # No explicit value, use resolved value from parent for initialization
                        # This allows the nested manager to show parent values while keeping None for unchanged fields
                        nested_current_value = getattr(nested_dataclass_for_form, nested_name, nested_info.default_value)
                else:
                    nested_current_value = getattr(nested_dataclass_for_form, nested_name, nested_info.default_value)
            else:
                nested_current_value = nested_info.default_value
            nested_parameters[nested_name] = nested_current_value
            nested_parameter_types[nested_name] = nested_info.param_type
        
        # Create nested form manager without scroll area (dataclasses should show in full)
        nested_manager = ParameterFormManager(
            nested_parameters,
            nested_parameter_types,
            f"{self.field_id}_{param_name}",
            nested_param_info,
            use_scroll_area=False  # Disable scroll area for nested dataclasses
        )

        # Store the parent dataclass type for proper lazy resolution detection
        nested_manager._parent_dataclass_type = param_type
        # Also store the lazy dataclass instance we created for this nested field
        nested_manager._lazy_dataclass_instance = nested_dataclass_for_form
        
        # Connect nested parameter changes
        nested_manager.parameter_changed.connect(
            lambda name, value, parent_name=param_name: self._handle_nested_parameter_change(parent_name, name, value)
        )
        
        self.nested_managers[param_name] = nested_manager
        layout.addWidget(nested_manager)
        
        return group_box

    def _create_lazy_nested_dataclass_if_needed(self, param_name: str, param_type: type, current_value: Any) -> Any:
        """
        Create a lazy version of any nested dataclass for consistent lazy loading behavior.

        This ensures that all nested dataclasses automatically get lazy loading behavior
        without needing fragile context detection logic.
        """
        import dataclasses

        # Only process actual dataclass types
        if not dataclasses.is_dataclass(param_type):
            return current_value

        # Create lazy version of the dataclass
        try:
            from openhcs.core.lazy_config import LazyDataclassFactory

            # Create lazy version with field path pointing to this nested field
            lazy_nested_class = LazyDataclassFactory.make_lazy_thread_local(
                base_class=param_type,
                field_path=param_name,  # e.g., "vfs", "zarr", "path_planning"
                lazy_class_name=f"Lazy{param_type.__name__}"
            )

            # Create instance with all None values for placeholder behavior
            return lazy_nested_class()

        except Exception as e:
            # If lazy creation fails, fall back to current value
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Failed to create lazy nested dataclass for {param_name}: {e}")
            return current_value

    def _is_optional_dataclass(self, param_type: type) -> bool:
        """Check if parameter type is Optional[dataclass]."""
        if get_origin(param_type) is Union:
            args = get_args(param_type)
            if len(args) == 2 and type(None) in args:
                inner_type = next(arg for arg in args if arg is not type(None))
                return dataclasses.is_dataclass(inner_type)
        return False

    def _get_optional_inner_type(self, param_type: type) -> type:
        """Extract the inner type from Optional[T]."""
        if get_origin(param_type) is Union:
            args = get_args(param_type)
            if len(args) == 2 and type(None) in args:
                return next(arg for arg in args if arg is not type(None))
        return param_type

    def _create_optional_dataclass_field(self, param_name: str, dataclass_type: type, current_value: Any) -> QWidget:
        """Create a checkbox + dataclass widget for Optional[dataclass] parameters."""
        from PyQt6.QtWidgets import QWidget, QVBoxLayout, QCheckBox

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # Checkbox and dataclass widget
        checkbox = QCheckBox(f"Enable {param_name.replace('_', ' ').title()}")
        checkbox.setChecked(current_value is not None)
        dataclass_widget = self._create_nested_dataclass_field(param_name, dataclass_type, current_value)
        dataclass_widget.setEnabled(current_value is not None)

        # Toggle logic
        def toggle_dataclass(checked: bool):
            dataclass_widget.setEnabled(checked)
            value = (dataclass_type() if checked and current_value is None
                    else self.nested_managers[param_name].get_current_values()
                         and dataclass_type(**self.nested_managers[param_name].get_current_values())
                    if checked and param_name in self.nested_managers else None)
            self.textual_form_manager.update_parameter(param_name, value)
            self.parameter_changed.emit(param_name, value)

        checkbox.stateChanged.connect(toggle_dataclass)

        layout.addWidget(checkbox)
        layout.addWidget(dataclass_widget)

        # Store reference
        if not hasattr(self, 'optional_checkboxes'):
            self.optional_checkboxes = {}
        self.optional_checkboxes[param_name] = checkbox

        return container

    def _create_regular_parameter_field(self, param_name: str, param_type: type, current_value: Any) -> QWidget:
        """Create a field for regular (non-dataclass) parameter."""
        container = QFrame()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(5, 2, 5, 2)
        
        # Parameter label with help (reuses Textual TUI parameter info)
        param_info = self.textual_form_manager.parameter_info.get(param_name) if hasattr(self.textual_form_manager, 'parameter_info') else None
        param_description = param_info.description if param_info else f"Parameter: {param_name}"

        label_with_help = LabelWithHelp(
            text=f"{param_name.replace('_', ' ').title()}:",
            param_name=param_name,
            param_description=param_description,
            param_type=param_type,
            color_scheme=self.color_scheme
        )
        label_with_help.setMinimumWidth(150)
        layout.addWidget(label_with_help)

        # Create widget using registry and apply placeholder
        widget = self.form_abstraction.create_widget_for_parameter(param_name, param_type, current_value)
        if widget:
            apply_lazy_default_placeholder(widget, param_name, current_value,
                                         self.form_abstraction.parameter_types, 'pyqt6')
            PyQt6WidgetEnhancer.connect_change_signal(widget, param_name, self._emit_parameter_change)

            self.widgets[param_name] = widget
            layout.addWidget(widget)

            # Add reset button
            reset_btn = QPushButton("Reset")
            reset_btn.setMaximumWidth(60)
            reset_btn.clicked.connect(lambda: self._reset_parameter(param_name))
            layout.addWidget(reset_btn)
        
        return container
    
    # _create_typed_widget method removed - functionality moved inline


    
    def _emit_parameter_change(self, param_name: str, value: Any):
        """Emit parameter change signal."""
        # Update the Textual TUI form manager (which holds the actual parameters)
        self.textual_form_manager.update_parameter(param_name, value)
        self.parameter_changed.emit(param_name, value)
    
    def _handle_nested_parameter_change(self, parent_name: str, nested_name: str, value: Any):
        """Handle parameter change in nested dataclass."""
        if parent_name in self.nested_managers:
            # Update nested manager's parameters
            nested_manager = self.nested_managers[parent_name]
            nested_manager.textual_form_manager.update_parameter(nested_name, value)

            # Rebuild nested dataclass instance
            nested_type = self.textual_form_manager.parameter_types[parent_name]

            # Resolve Union types (like Optional[DataClass]) to the actual dataclass type
            if self._is_optional_dataclass(nested_type):
                nested_type = self._get_optional_inner_type(nested_type)

            # Get current values from nested manager
            nested_values = nested_manager.get_current_values()

            # Get the original nested dataclass instance to preserve unchanged values
            original_instance = self.textual_form_manager.parameters.get(parent_name)

            # Create new instance, preserving original values for None fields (lazy loading pattern)
            if original_instance and hasattr(original_instance, '__dataclass_fields__'):
                # Merge: use nested_values for changed fields, original values for None fields
                merged_values = {}
                for field_name, field_value in nested_values.items():
                    if field_value is not None:
                        # User has explicitly set this value
                        merged_values[field_name] = field_value
                    else:
                        # Preserve original value for unchanged field
                        merged_values[field_name] = getattr(original_instance, field_name)
                new_instance = nested_type(**merged_values)
            else:
                # Fallback: create with nested values as-is
                new_instance = nested_type(**nested_values)

            # Update parent parameter in textual form manager
            self.textual_form_manager.update_parameter(parent_name, new_instance)

            # Emit change for parent parameter
            self.parameter_changed.emit(parent_name, new_instance)
    
    def _reset_parameter(self, param_name: str):
        """Reset parameter to default value."""
        # Use textual form manager's parameter info and reset functionality
        if hasattr(self.textual_form_manager, 'parameter_info') and param_name in self.textual_form_manager.parameter_info:
            default_value = self.textual_form_manager.parameter_info[param_name].default_value

            # Update textual form manager
            self.textual_form_manager.update_parameter(param_name, default_value)

            # Update widget
            if param_name in self.widgets:
                widget = self.widgets[param_name]
                self._update_widget_value(widget, default_value)

            self.parameter_changed.emit(param_name, default_value)
    
    def _update_widget_value(self, widget: QWidget, value: Any):
        """Update widget value without triggering signals."""
        # Handle EnhancedPathWidget FIRST (duck typing)
        if hasattr(widget, 'set_path'):
            widget.set_path(value)
            return

        if isinstance(widget, QCheckBox):
            widget.blockSignals(True)
            widget.setChecked(bool(value) if value is not None else False)
            widget.blockSignals(False)
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            widget.blockSignals(True)
            widget.setValue(value if value is not None else 0)
            widget.blockSignals(False)
        elif isinstance(widget, QLineEdit):
            widget.blockSignals(True)
            # Handle literal "None" string - should display as empty
            if isinstance(value, str) and value == "None":
                widget.setText("")
            else:
                widget.setText(str(value) if value is not None else "")
            widget.blockSignals(False)
        elif isinstance(widget, QComboBox):
            widget.blockSignals(True)
            index = widget.findData(value)
            if index >= 0:
                widget.setCurrentIndex(index)
            widget.blockSignals(False)

    def update_parameter(self, param_name: str, value: Any):
        """Update parameter value programmatically."""
        self.textual_form_manager.update_parameter(param_name, value)
        if param_name in self.widgets:
            self._update_widget_value(self.widgets[param_name], value)

    def get_current_values(self) -> Dict[str, Any]:
        """Get current parameter values (mirrors Textual TUI)."""
        return self.textual_form_manager.parameters.copy()

    # Old placeholder methods removed - now using centralized abstraction layer
