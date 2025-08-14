"""
Parameter form manager for PyQt6 GUI.

REUSES the Textual TUI parameter form generation logic for consistent UX.
This is a PyQt6 adapter that uses the actual working Textual TUI services.
"""

import dataclasses
import logging
from typing import Any, Dict, get_origin, get_args, Union, Optional, Type
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


class NoneAwareLineEdit(QLineEdit):
    """QLineEdit that properly handles None values for lazy dataclass contexts."""

    def get_value(self):
        """Get value, returning None for empty text instead of empty string."""
        text = self.text().strip()
        return None if text == "" else text

    def set_value(self, value):
        """Set value, handling None properly."""
        self.setText("" if value is None else str(value))


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
                 function_target=None, color_scheme: Optional[PyQt6ColorScheme] = None,
                 is_global_config_editing: bool = False, global_config_type: Optional[Type] = None,
                 placeholder_prefix: str = "Pipeline default"):
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
            parameters, parameter_types, field_id, parameter_info, is_global_config_editing=is_global_config_editing
        )

        # Store field_id for PyQt6 widget creation
        self.field_id = field_id
        self.is_global_config_editing = is_global_config_editing
        self.global_config_type = global_config_type
        self.placeholder_prefix = placeholder_prefix

        # Control whether to use scroll area (disable for nested dataclasses)
        self.use_scroll_area = use_scroll_area

        # Track PyQt6 widgets for value updates
        self.widgets = {}
        self.nested_managers = {}

        # Optional lazy dataclass for placeholder generation in nested static forms
        self.lazy_dataclass_for_placeholders = None

        self.setup_ui()
    
    def setup_ui(self):
        """Setup the parameter form UI using Textual TUI logic."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Content widget
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)

        # Build form fields using Textual TUI parameter types and logic
        for param_name, param_type in self.textual_form_manager.parameter_types.items():
            current_value = self.textual_form_manager.parameters[param_name]

            # Handle Optional[dataclass] types with checkbox wrapper
            if self._is_optional_dataclass(param_type):
                inner_dataclass_type = self._get_optional_inner_type(param_type)
                field_widget = self._create_optional_dataclass_field(param_name, inner_dataclass_type, current_value)
            # Handle nested dataclasses (reuse Textual TUI logic)
            elif dataclasses.is_dataclass(param_type):
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
            if self.is_global_config_editing:
                # Global config editing: use concrete values
                if nested_dataclass_for_form:
                    nested_current_value = getattr(nested_dataclass_for_form, nested_name, nested_info.default_value)
                else:
                    nested_current_value = nested_info.default_value
            else:
                # Lazy context: check if field has a concrete value, otherwise use None for placeholder behavior
                if nested_dataclass_for_form:
                    # Extract the actual value from the nested dataclass
                    # For both lazy and regular dataclasses, use getattr to get the resolved value
                    nested_current_value = getattr(nested_dataclass_for_form, nested_name, None)

                    # If this is a lazy dataclass and we got a resolved value, check if it's actually stored
                    if hasattr(nested_dataclass_for_form, '_resolve_field_value') and nested_current_value is not None:
                        # Check if this field has a concrete stored value vs lazy resolved value
                        try:
                            stored_value = object.__getattribute__(nested_dataclass_for_form, nested_name)
                            # If stored value is None, this field is lazy (use None for placeholder)
                            # If stored value is not None, this field is concrete (use the value)
                            nested_current_value = stored_value
                        except AttributeError:
                            # Field doesn't exist as stored attribute, so it's lazy (use None for placeholder)
                            nested_current_value = None
                else:
                    # No nested dataclass instance - use None for placeholder behavior
                    nested_current_value = None

            nested_parameters[nested_name] = nested_current_value
            nested_parameter_types[nested_name] = nested_info.param_type
        
        # Create nested form manager without scroll area (dataclasses should show in full)
        nested_field_id = f"{self.field_id}_{param_name}"

        # For lazy contexts where we need placeholder generation, create a lazy dataclass
        lazy_dataclass_for_placeholders = None
        if not self._should_use_concrete_nested_values(nested_dataclass_for_form):
            # We're in a lazy context - create lazy dataclass for placeholder generation
            lazy_dataclass_for_placeholders = self._create_static_lazy_dataclass_for_placeholders(param_type)
            # Use special field_id to signal nested forms should not use thread-local resolution
            nested_field_id = f"nested_static_{param_name}"

        # Create nested form manager without scroll area (dataclasses should show in full)
        nested_manager = ParameterFormManager(
            nested_parameters,
            nested_parameter_types,
            nested_field_id,
            nested_param_info,
            use_scroll_area=False,  # Disable scroll area for nested dataclasses
            is_global_config_editing=self.is_global_config_editing  # Pass through the global config editing flag
        )

        # For nested static forms, provide the lazy dataclass for placeholder generation
        if lazy_dataclass_for_placeholders:
            nested_manager.lazy_dataclass_for_placeholders = lazy_dataclass_for_placeholders

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

    def _get_field_path_for_nested_type(self, nested_type: Type) -> Optional[str]:
        """
        Automatically determine the field path for a nested dataclass type using type inspection.

        This method examines the GlobalPipelineConfig fields and their type annotations
        to find which field corresponds to the given nested_type. This eliminates the need
        for hardcoded string mappings and automatically works with new nested dataclass fields.

        Args:
            nested_type: The dataclass type to find the field path for

        Returns:
            The field path string (e.g., 'path_planning', 'vfs') or None if not found
        """
        try:
            from openhcs.core.config import GlobalPipelineConfig
            from dataclasses import fields
            import typing

            # Get all fields from GlobalPipelineConfig
            global_config_fields = fields(GlobalPipelineConfig)

            for field in global_config_fields:
                field_type = field.type

                # Handle Optional types (Union[Type, None])
                if hasattr(typing, 'get_origin') and typing.get_origin(field_type) is typing.Union:
                    # Get the non-None type from Optional[Type]
                    args = typing.get_args(field_type)
                    if len(args) == 2 and type(None) in args:
                        field_type = args[0] if args[1] is type(None) else args[1]

                # Check if the field type matches our nested type
                if field_type == nested_type:
                    return field.name



            return None

        except Exception as e:
            # Fallback to None if type inspection fails
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Failed to determine field path for {nested_type.__name__}: {e}")
            return None

    def _should_use_concrete_nested_values(self, current_value: Any) -> bool:
        """
        Determine if nested dataclass fields should use concrete values or None for placeholders.

        Returns True if:
        1. Global config editing (always concrete)
        2. Regular concrete dataclass (always concrete)

        Returns False if:
        1. Lazy dataclass (supports mixed lazy/concrete states per field)
        2. None values (show placeholders)

        Note: This method now supports mixed states within nested dataclasses.
        Individual fields can be lazy (None) or concrete within the same dataclass.
        """
        # Global config editing always uses concrete values
        if self.is_global_config_editing:
            return True

        # If current_value is None, use placeholders
        if current_value is None:
            return False

        # If current_value is a concrete dataclass instance, use its values
        if hasattr(current_value, '__dataclass_fields__') and not hasattr(current_value, '_resolve_field_value'):
            return True

        # For lazy dataclasses, always return False to enable mixed lazy/concrete behavior
        # Individual field values will be checked separately in the nested form creation
        if hasattr(current_value, '_resolve_field_value'):
            return False

        # Default to placeholder behavior for lazy contexts
        return False

    def _should_use_concrete_for_placeholder_rendering(self, current_value: Any) -> bool:
        """
        Determine if nested dataclass should use concrete values for PLACEHOLDER RENDERING specifically.

        This is separate from _should_use_concrete_nested_values which is used for saving/rebuilding.
        For placeholder rendering, we want field-level logic in lazy contexts.
        """
        # Global config editing always uses concrete values
        if self.is_global_config_editing:
            return True

        # In lazy contexts, ALWAYS return False to enable field-level placeholder logic
        # This allows mixed states: some fields can be None (placeholders) while others have values
        return False

    def _create_lazy_nested_dataclass_if_needed(self, param_name: str, param_type: type, current_value: Any) -> Any:
        """
        Create a lazy version of any nested dataclass for consistent lazy loading behavior.

        Returns the appropriate nested dataclass instance based on context:
        - Concrete contexts: return the actual nested dataclass instance
        - Lazy contexts: return None for placeholder behavior or preserve explicit values
        """
        import dataclasses

        # Only process actual dataclass types
        if not dataclasses.is_dataclass(param_type):
            return current_value

        # Use the new robust logic to determine behavior
        if self._should_use_concrete_nested_values(current_value):
            return current_value
        else:
            return None

    def _create_static_lazy_dataclass_for_placeholders(self, param_type: type) -> Any:
        """
        Create a lazy dataclass that resolves from current global config for placeholder generation.

        This is used in nested static forms to provide placeholder behavior that reflects
        the current global config values (not static defaults) while avoiding thread-local conflicts.
        """
        try:
            from openhcs.core.lazy_config import LazyDataclassFactory
            from openhcs.core.config import _current_pipeline_config

            # Check if we have a current thread-local pipeline config context
            if hasattr(_current_pipeline_config, 'value') and _current_pipeline_config.value:
                # Use the current global config instance as the defaults source
                # This ensures placeholders show current global config values, not static defaults
                current_global_config = _current_pipeline_config.value

                # Find the specific nested dataclass instance from the global config
                nested_dataclass_instance = self._extract_nested_dataclass_from_global_config(
                    current_global_config, param_type
                )

                if nested_dataclass_instance:
                    # Create lazy version that resolves from the specific nested dataclass instance
                    lazy_class = LazyDataclassFactory.create_lazy_dataclass(
                        defaults_source=nested_dataclass_instance,  # Use current nested instance
                        lazy_class_name=f"GlobalContextLazy{param_type.__name__}"
                    )

                    # Create instance for placeholder resolution
                    return lazy_class()
                else:
                    # Fallback to static resolution if nested instance not found
                    lazy_class = LazyDataclassFactory.create_lazy_dataclass(
                        defaults_source=param_type,  # Use class defaults as fallback
                        lazy_class_name=f"StaticLazy{param_type.__name__}"
                    )

                    # Create instance for placeholder resolution
                    return lazy_class()
            else:
                # Fallback to static resolution if no thread-local context
                lazy_class = LazyDataclassFactory.create_lazy_dataclass(
                    defaults_source=param_type,  # Use class defaults as fallback
                    lazy_class_name=f"StaticLazy{param_type.__name__}"
                )

                # Create instance for placeholder resolution
                return lazy_class()

        except Exception as e:
            # If lazy creation fails, return None
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Failed to create lazy dataclass for {param_type.__name__}: {e}")
            return None

    def _extract_nested_dataclass_from_global_config(self, global_config: Any, param_type: type) -> Any:
        """Extract the specific nested dataclass instance from the global config."""
        try:
            import dataclasses

            # Get all fields from the global config
            if dataclasses.is_dataclass(global_config):
                for field in dataclasses.fields(global_config):
                    field_value = getattr(global_config, field.name)
                    if isinstance(field_value, param_type):
                        return field_value

            return None

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Failed to extract nested dataclass {param_type.__name__} from global config: {e}")
            return None

    def _apply_placeholder_with_lazy_context(self, widget: Any, param_name: str, current_value: Any) -> None:
        """Apply placeholder using lazy dataclass context when available."""
        from openhcs.ui.shared.parameter_form_abstraction import apply_lazy_default_placeholder

        # If we have a lazy dataclass for placeholders (nested static forms), use it directly
        if hasattr(self, 'lazy_dataclass_for_placeholders') and self.lazy_dataclass_for_placeholders:
            self._apply_placeholder_from_lazy_dataclass(widget, param_name, current_value, self.lazy_dataclass_for_placeholders)
        # For nested static forms, create lazy dataclass on-demand
        elif self.field_id.startswith("nested_static_"):
            # Extract the dataclass type from the field_id and create lazy dataclass
            lazy_dataclass = self._create_lazy_dataclass_for_nested_static_form()
            if lazy_dataclass:
                self._apply_placeholder_from_lazy_dataclass(widget, param_name, current_value, lazy_dataclass)
            else:
                # Fallback to standard placeholder application
                apply_lazy_default_placeholder(widget, param_name, current_value,
                                             self.form_abstraction.parameter_types, 'pyqt6',
                                             is_global_config_editing=self.is_global_config_editing,
                                             global_config_type=self.global_config_type,
                                             placeholder_prefix=self.placeholder_prefix)
        else:
            # Use the standard placeholder application
            apply_lazy_default_placeholder(widget, param_name, current_value,
                                         self.form_abstraction.parameter_types, 'pyqt6',
                                         is_global_config_editing=self.is_global_config_editing,
                                         global_config_type=self.global_config_type,
                                         placeholder_prefix=self.placeholder_prefix)

    def _apply_placeholder_from_lazy_dataclass(self, widget: Any, param_name: str, current_value: Any, lazy_dataclass: Any) -> None:
        """Apply placeholder using a specific lazy dataclass instance."""
        if current_value is not None:
            return

        try:
            from openhcs.core.config import LazyDefaultPlaceholderService

            # Get the lazy dataclass type
            lazy_dataclass_type = type(lazy_dataclass)

            # Generate placeholder using the lazy dataclass
            placeholder_text = LazyDefaultPlaceholderService.get_lazy_resolved_placeholder(
                lazy_dataclass_type, param_name
            )

            if placeholder_text:
                from openhcs.ui.shared.pyqt6_widget_strategies import PyQt6WidgetEnhancer
                PyQt6WidgetEnhancer.apply_placeholder_text(widget, placeholder_text)

        except Exception:
            pass

    def _create_lazy_dataclass_for_nested_static_form(self) -> Any:
        """Create lazy dataclass for nested static form based on parameter types."""
        try:
            # For nested static forms, we need to determine the dataclass type from the parameter types
            # The parameter types should all belong to the same dataclass
            import dataclasses
            from openhcs.core import config

            # Get all parameter names
            param_names = set(self.form_abstraction.parameter_types.keys())

            # Find the dataclass that matches these parameter names
            for name, obj in vars(config).items():
                if (dataclasses.is_dataclass(obj) and
                    hasattr(obj, '__dataclass_fields__')):
                    dataclass_fields = {field.name for field in dataclasses.fields(obj)}
                    if param_names == dataclass_fields:
                        # Found the matching dataclass, create lazy version
                        return self._create_static_lazy_dataclass_for_placeholders(obj)

            return None

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Failed to create lazy dataclass for nested static form: {e}")
            return None

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
            self._apply_placeholder_with_lazy_context(widget, param_name, current_value)
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
        # For nested fields, also update the nested manager to keep it in sync
        parent_nested_name = self._find_parent_nested_manager(param_name)

        # Debug: Check why nested manager isn't being found
        if param_name == 'output_dir_suffix':
            logger.info(f"*** NESTED DEBUG *** param_name={param_name}, parent_nested_name={parent_nested_name}")
            if hasattr(self, 'nested_managers'):
                logger.info(f"*** NESTED DEBUG *** Available nested managers: {list(self.nested_managers.keys())}")
                for name, manager in self.nested_managers.items():
                    param_types = manager.textual_form_manager.parameter_types.keys()
                    logger.info(f"*** NESTED DEBUG *** {name} contains: {list(param_types)}")
            else:
                logger.info(f"*** NESTED DEBUG *** No nested_managers attribute")

        if parent_nested_name and hasattr(self, 'nested_managers'):
            logger.info(f"*** NESTED UPDATE *** Updating nested manager {parent_nested_name}.{param_name} = {value}")
            nested_manager = self.nested_managers[parent_nested_name]
            nested_manager.textual_form_manager.update_parameter(param_name, value)

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

            # Create new instance using nested_values as-is (respecting explicit None values)
            # Don't preserve original values for None fields - None means user explicitly cleared the field
            new_instance = nested_type(**nested_values)

            # Update parent parameter in textual form manager
            self.textual_form_manager.update_parameter(parent_name, new_instance)

            # Emit change for parent parameter
            self.parameter_changed.emit(parent_name, new_instance)
    
    def _reset_parameter(self, param_name: str):
        """Reset parameter to appropriate default value based on lazy vs concrete dataclass context."""
        if not (hasattr(self.textual_form_manager, 'parameter_info') and param_name in self.textual_form_manager.parameter_info):
            return

        # For nested fields, reset the parent nested manager first to prevent old values
        parent_nested_name = self._find_parent_nested_manager(param_name)
        logger.info(f"*** RESET DEBUG *** param_name={param_name}, parent_nested_name={parent_nested_name}")
        if parent_nested_name and hasattr(self, 'nested_managers'):
            logger.info(f"*** RESET FIX *** Resetting parent nested manager {parent_nested_name} for field {param_name}")
            nested_manager = self.nested_managers[parent_nested_name]
            nested_manager.reset_all_parameters()
        else:
            logger.info(f"*** RESET DEBUG *** No parent nested manager found or no nested_managers attribute")

        # Determine the correct reset value based on context
        reset_value = self._get_reset_value_for_parameter(param_name)

        # Update textual form manager
        self.textual_form_manager.update_parameter(param_name, reset_value)

        # Update widget with context-aware behavior
        if param_name in self.widgets:
            widget = self.widgets[param_name]
            self._update_widget_value_with_context(widget, reset_value, param_name)

        self.parameter_changed.emit(param_name, reset_value)

    def _find_parent_nested_manager(self, param_name: str) -> str:
        """Find which nested manager contains the given parameter."""
        if hasattr(self, 'nested_managers'):
            for nested_name, nested_manager in self.nested_managers.items():
                if param_name in nested_manager.textual_form_manager.parameter_types:
                    return nested_name
        return None

    def reset_all_parameters(self):
        """Reset all parameters using individual field reset logic for consistency."""
        # Reset each parameter individually using the same logic as individual reset buttons
        # This ensures consistent behavior between individual resets and reset all
        for param_name in self.textual_form_manager.parameter_types.keys():
            self._reset_parameter(param_name)

        # Also reset all nested form parameters
        if hasattr(self, 'nested_managers'):
            for nested_name, nested_manager in self.nested_managers.items():
                nested_manager.reset_all_parameters()

    def reset_parameter_by_path(self, parameter_path: str):
        """Reset a parameter by its full path (supports nested parameters).

        Args:
            parameter_path: Either a simple parameter name (e.g., 'num_workers')
                          or a nested path (e.g., 'path_planning.output_dir_suffix')
        """
        if '.' in parameter_path:
            # Handle nested parameter
            parts = parameter_path.split('.', 1)
            nested_name = parts[0]
            nested_param = parts[1]

            if hasattr(self, 'nested_managers') and nested_name in self.nested_managers:
                nested_manager = self.nested_managers[nested_name]
                if '.' in nested_param:
                    # Further nesting
                    nested_manager.reset_parameter_by_path(nested_param)
                else:
                    # Direct nested parameter
                    nested_manager._reset_parameter(nested_param)

                # Rebuild the parent dataclass instance with the updated nested values
                self._rebuild_nested_dataclass_from_manager(nested_name)
            else:
                logger.warning(f"Nested manager '{nested_name}' not found for parameter path '{parameter_path}'")
        else:
            # Handle top-level parameter
            self._reset_parameter(parameter_path)

    def _get_reset_value_for_parameter(self, param_name: str) -> Any:
        """
        Get the appropriate reset value for a parameter based on lazy vs concrete dataclass context.

        For concrete dataclasses (like GlobalPipelineConfig):
        - Reset to static class defaults

        For lazy dataclasses (like PipelineConfig for orchestrator configs):
        - Reset to None to preserve placeholder behavior and inheritance hierarchy
        """
        param_info = self.textual_form_manager.parameter_info[param_name]
        param_type = param_info.param_type

        # For global config editing, always use static defaults
        if self.is_global_config_editing:
            return param_info.default_value

        # For nested dataclass fields, check if we should use concrete values
        if hasattr(param_type, '__dataclass_fields__'):
            # This is a dataclass field - determine if it should be concrete or None
            current_value = self.textual_form_manager.parameters.get(param_name)
            if self._should_use_concrete_nested_values(current_value):
                # Use static default for concrete nested dataclass
                return param_info.default_value
            else:
                # Use None for lazy nested dataclass to preserve placeholder behavior
                return None

        # For non-dataclass fields in lazy context, use None to preserve placeholder behavior
        # This allows the field to inherit from the parent config hierarchy
        if not self.is_global_config_editing:
            return None

        # Fallback to static default
        return param_info.default_value

    def _update_widget_value_with_context(self, widget: QWidget, value: Any, param_name: str):
        """Update widget value with context-aware placeholder handling."""
        # For static contexts (global config editing), set actual values and clear placeholder styling
        if self.is_global_config_editing or value is not None:
            # Clear any existing placeholder state
            self._clear_placeholder_state(widget)
            # Set the actual value
            self._update_widget_value_direct(widget, value)
        else:
            # For lazy contexts with None values, apply placeholder styling directly
            # Don't call _update_widget_value_direct with None as it breaks combobox selection
            # and doesn't properly handle placeholder text for string fields
            self._reapply_placeholder_if_needed(widget, param_name)

    def _clear_placeholder_state(self, widget: QWidget):
        """Clear placeholder state from a widget."""
        if widget.property("is_placeholder_state"):
            widget.setStyleSheet("")
            widget.setProperty("is_placeholder_state", False)
            # Clean tooltip
            current_tooltip = widget.toolTip()
            if "Pipeline default:" in current_tooltip:
                widget.setToolTip("")

    def _update_widget_value_direct(self, widget: QWidget, value: Any):
        """Update widget value without triggering signals or applying placeholder styling."""
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
        elif isinstance(widget, NoneAwareLineEdit):
            widget.blockSignals(True)
            widget.set_value(value)
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

    def _update_widget_value(self, widget: QWidget, value: Any):
        """Update widget value without triggering signals (legacy method for compatibility)."""
        self._update_widget_value_direct(widget, value)

    def _reapply_placeholder_if_needed(self, widget: QWidget, param_name: str = None):
        """Re-apply placeholder styling to a widget when its value is set to None."""
        # If param_name not provided, find it by searching widgets
        if param_name is None:
            for name, w in self.widgets.items():
                if w is widget:
                    param_name = name
                    break

        if param_name is None:
            return

        # Re-apply placeholder using the same logic as initial widget creation
        self._apply_placeholder_with_lazy_context(widget, param_name, None)

    def update_parameter(self, param_name: str, value: Any):
        """Update parameter value programmatically with recursive nested parameter support."""
        # Handle nested parameters with dot notation (e.g., 'path_planning.output_dir_suffix')
        if '.' in param_name:
            parts = param_name.split('.', 1)
            parent_name = parts[0]
            remaining_path = parts[1]

            # Update nested manager if it exists
            if hasattr(self, 'nested_managers') and parent_name in self.nested_managers:
                nested_manager = self.nested_managers[parent_name]

                # Recursively handle the remaining path (supports unlimited nesting levels)
                nested_manager.update_parameter(remaining_path, value)

                # Now rebuild the parent dataclass from the nested manager's current values
                self._rebuild_nested_dataclass_from_manager(parent_name)
                return

        # Handle regular parameters
        self.textual_form_manager.update_parameter(param_name, value)
        if param_name in self.widgets:
            self._update_widget_value(self.widgets[param_name], value)

    def get_current_values(self) -> Dict[str, Any]:
        """Get current parameter values (mirrors Textual TUI)."""
        return self.textual_form_manager.parameters.copy()

    def _rebuild_nested_dataclass_from_manager(self, parent_name: str):
        """Rebuild the nested dataclass instance from the nested manager's current values."""
        if not (hasattr(self, 'nested_managers') and parent_name in self.nested_managers):
            return

        nested_manager = self.nested_managers[parent_name]
        nested_values = nested_manager.get_current_values()
        nested_type = self.textual_form_manager.parameter_types[parent_name]

        # Resolve Union types (like Optional[DataClass]) to the actual dataclass type
        if self._is_optional_dataclass(nested_type):
            nested_type = self._get_optional_inner_type(nested_type)

        # Get the original nested dataclass instance to preserve unchanged values
        original_instance = self.textual_form_manager.parameters.get(parent_name)

        # SIMPLIFIED APPROACH: In lazy contexts, don't create concrete dataclasses for mixed states
        # This preserves the nested manager's None values for placeholder behavior

        if self.is_global_config_editing:
            # Global config editing: always create concrete dataclass with all values
            merged_values = {}
            for field_name, field_value in nested_values.items():
                if field_value is not None:
                    merged_values[field_name] = field_value
                else:
                    # Use default value for None fields in global config editing
                    from dataclasses import fields
                    for field in fields(nested_type):
                        if field.name == field_name:
                            merged_values[field_name] = field.default if field.default != field.default_factory else field.default_factory()
                            break
            new_instance = nested_type(**merged_values)
        else:
            # Lazy context: always create lazy dataclass instance with mixed concrete/lazy fields
            # Even if all values are None (especially after reset), we want lazy resolution
            from openhcs.core.lazy_config import LazyDataclassFactory

            # Determine the correct field path using type inspection
            field_path = self._get_field_path_for_nested_type(nested_type)

            lazy_nested_type = LazyDataclassFactory.make_lazy_thread_local(
                base_class=nested_type,
                field_path=field_path,  # Use correct field path for nested resolution
                lazy_class_name=f"Mixed{nested_type.__name__}"
            )

            # Create instance with mixed concrete/lazy field values
            # Pass ALL fields to constructor: concrete values for edited fields, None for lazy fields
            # The lazy __getattribute__ will resolve None values via _resolve_field_value
            new_instance = lazy_nested_type(**nested_values)

        # Update parent parameter in textual form manager
        self.textual_form_manager.update_parameter(parent_name, new_instance)

        # Emit change for parent parameter
        self.parameter_changed.emit(parent_name, new_instance)

    # Old placeholder methods removed - now using centralized abstraction layer
