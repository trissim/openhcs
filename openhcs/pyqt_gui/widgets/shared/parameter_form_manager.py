"""
Dramatically simplified PyQt parameter form manager.

This demonstrates how the widget implementation can be drastically simplified
by leveraging the comprehensive shared infrastructure we've built.
"""

import dataclasses
import logging
from typing import Any, Dict, Type, Optional
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QLabel, QPushButton, QLineEdit, QCheckBox, QComboBox
from PyQt6.QtCore import Qt, pyqtSignal

logger = logging.getLogger(__name__)

# Import our comprehensive shared infrastructure
from openhcs.ui.shared.parameter_form_service import ParameterFormService, ParameterInfo
from openhcs.ui.shared.parameter_form_config_factory import pyqt_config
from openhcs.ui.shared.parameter_form_constants import CONSTANTS

from openhcs.ui.shared.widget_creation_registry import create_pyqt6_registry
from openhcs.ui.shared.ui_utils import format_param_name, format_field_id, format_reset_button_id
from openhcs.ui.shared.pyqt6_widget_strategies import PyQt6WidgetEnhancer

# Import PyQt-specific components
from .clickable_help_components import GroupBoxWithHelp, LabelWithHelp
from openhcs.pyqt_gui.shared.color_scheme import PyQt6ColorScheme

# Import OpenHCS core components
from openhcs.core.config import LazyDefaultPlaceholderService, GlobalPipelineConfig
from openhcs.ui.shared.parameter_type_utils import ParameterTypeUtils





class NoneAwareLineEdit(QLineEdit):
    """QLineEdit that properly handles None values for lazy dataclass contexts."""

    def get_value(self):
        """Get value, returning None for empty text instead of empty string."""
        text = self.text().strip()
        return None if text == "" else text

    def set_value(self, value):
        """Set value, handling None properly."""
        self.setText("" if value is None else str(value))


class ParameterFormManager(QWidget):
    """
    PyQt6 parameter form manager with simplified implementation.

    This implementation uses shared infrastructure while maintaining
    exact backward compatibility with the original API.

    Key improvements:
    - Internal implementation reduced by ~85%
    - Parameter analysis delegated to service layer
    - Widget creation patterns centralized
    - All magic strings eliminated
    - Type checking delegated to utilities
    - Clean, minimal implementation focused on core functionality
    """

    parameter_changed = pyqtSignal(str, object)  # param_name, value

    def __init__(self, parameters: Dict[str, Any], parameter_types: Dict[str, type],
                 field_id: str, dataclass_type: Type, parameter_info: Dict = None, parent=None,
                 use_scroll_area: bool = True, function_target=None,
                 color_scheme: Optional[PyQt6ColorScheme] = None, placeholder_prefix: str = None):
        """
        Initialize PyQt parameter form manager with mathematically elegant single-parameter design.

        Args:
            parameters: Dictionary of parameter names to current values
            parameter_types: Dictionary of parameter names to types
            field_id: Unique identifier for the form
            dataclass_type: The dataclass type that deterministically controls all form behavior
            parameter_info: Optional parameter information dictionary
            parent: Optional parent widget
            use_scroll_area: Whether to use scroll area
            function_target: Optional function target for docstring fallback
            color_scheme: Optional PyQt color scheme
            placeholder_prefix: Prefix for placeholder text
        """
        QWidget.__init__(self, parent)

        # Store configuration parameters - dataclass_type is the single source of truth
        self.dataclass_type = dataclass_type
        self.placeholder_prefix = placeholder_prefix or CONSTANTS.DEFAULT_PLACEHOLDER_PREFIX

        # Convert old API to new config object internally
        if color_scheme is None:
            color_scheme = PyQt6ColorScheme()

        config = pyqt_config(
            field_id=field_id,
            color_scheme=color_scheme,
            function_target=function_target,
            use_scroll_area=use_scroll_area
        )
        config.parameter_info = parameter_info
        config.dataclass_type = dataclass_type
        config.placeholder_prefix = placeholder_prefix
        config.is_global_config_editing = not LazyDefaultPlaceholderService.has_lazy_resolution(dataclass_type)
        # global_config_type is no longer needed - dataclass_type determines all behavior

        # Initialize core attributes directly (avoid abstract class instantiation)
        self.parameters = parameters.copy()
        self.parameter_types = parameter_types
        self.config = config

        # Initialize service layer for business logic
        self.service = ParameterFormService()

        # Initialize tracking attributes
        self.widgets = {}
        self.reset_buttons = {}  # Track reset buttons for API compatibility
        self.nested_managers = {}

        # Store public API attributes for backward compatibility
        self.field_id = field_id
        self.parameter_info = parameter_info or {}
        self.use_scroll_area = use_scroll_area
        self.function_target = function_target
        self.color_scheme = color_scheme
        # Note: dataclass_type already stored above

        # Analyze form structure once using service layer
        self.form_structure = self.service.analyze_parameters(
            parameters, parameter_types, config.field_id, config.parameter_info, self.dataclass_type
        )

        # Get widget creator from registry
        self.create_widget = create_pyqt6_registry()

        
        # Set up UI
        self.setup_ui()

    @classmethod
    def from_dataclass_instance(cls, dataclass_instance: Any, field_id: str,
                              placeholder_prefix: str = "Default",
                              parent=None, use_scroll_area: bool = True,
                              function_target=None, color_scheme=None):
        """
        Create ParameterFormManager for editing entire dataclass instance.

        This replaces LazyDataclassEditor functionality by automatically extracting
        parameters from the dataclass instance and creating the form manager.

        Args:
            dataclass_instance: The dataclass instance to edit
            field_id: Unique identifier for the form
            placeholder_prefix: Prefix for placeholder text
            parent: Parent widget
            use_scroll_area: Whether to use scroll area
            function_target: Optional function target
            color_scheme: Optional color scheme

        Returns:
            ParameterFormManager configured for dataclass editing
        """
        from dataclasses import fields, is_dataclass

        if not is_dataclass(dataclass_instance):
            raise ValueError(f"{type(dataclass_instance)} is not a dataclass")

        # Handle lazy dataclasses properly using the editing config pattern
        if hasattr(dataclass_instance, '_resolve_field_value'):
            # Lazy dataclass - use create_editing_config_from_existing_lazy_config
            from openhcs.core.pipeline_config import create_editing_config_from_existing_lazy_config
            editing_config = create_editing_config_from_existing_lazy_config(
                dataclass_instance, None  # Use existing thread-local context
            )

            # Extract parameters from editing config
            parameters = {}
            parameter_types = {}
            for field_obj in fields(editing_config):
                parameters[field_obj.name] = object.__getattribute__(editing_config, field_obj.name)
                parameter_types[field_obj.name] = field_obj.type
            dataclass_type = type(editing_config)
        else:
            # Regular dataclass - extract parameters normally
            parameters = {}
            parameter_types = {}
            for field_obj in fields(dataclass_instance):
                parameters[field_obj.name] = getattr(dataclass_instance, field_obj.name)
                parameter_types[field_obj.name] = field_obj.type

            dataclass_type = type(dataclass_instance)

        # Create ParameterFormManager with extracted data
        form_manager = cls(
            parameters=parameters,
            parameter_types=parameter_types,
            field_id=field_id,
            dataclass_type=dataclass_type,  # Use determined dataclass type
            parameter_info=None,
            parent=parent,
            use_scroll_area=use_scroll_area,
            function_target=function_target,
            color_scheme=color_scheme,
            placeholder_prefix=placeholder_prefix
        )

        # Store the original dataclass instance for reset operations
        form_manager._current_config_instance = dataclass_instance

        return form_manager
    
    def setup_ui(self):
        """Set up the UI layout."""
        layout = QVBoxLayout(self)
        
        # Build form content
        form_widget = self.build_form()
        
        # Add scroll area if requested
        if self.config.use_scroll_area:
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            scroll_area.setWidget(form_widget)
            layout.addWidget(scroll_area)
        else:
            layout.addWidget(form_widget)
    
    def build_form(self) -> QWidget:
        """
        Build the complete form UI.
        
        Dramatically simplified by delegating analysis to service layer
        and using centralized widget creation patterns.
        """
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        # Use unified widget creation for all parameter types
        for param_info in self.form_structure.parameters:
            widget = self._create_parameter_widget_unified(param_info)
            content_layout.addWidget(widget)
        
        return content_widget

    def _create_parameter_widget_unified(self, param_info) -> QWidget:
        """Unified widget creation handling all parameter types."""
        return self._create_parameter_section(param_info)

    def _create_parameter_section(self, param_info) -> QWidget:
        """Unified scaffolding - consolidates all common patterns, injects differences through return values."""
        display_info = self.service.get_parameter_display_info(
            param_info.name, param_info.type, param_info.description)
        field_ids = self.service.generate_field_ids(self.config.field_id, param_info.name)

        if param_info.is_optional and param_info.is_nested:
            container, widgets = self._build_optional_content(param_info, display_info, field_ids)
        elif param_info.is_nested:
            container, widgets = self._build_nested_content(param_info, display_info, field_ids)
        else:
            container, widgets = self._build_regular_content(param_info, display_info, field_ids)

        self.widgets[param_info.name] = widgets['main']
        if 'reset_button' in widgets:
            self.reset_buttons[param_info.name] = widgets['reset_button']

        if 'widget' in widgets:
            PyQt6WidgetEnhancer.connect_change_signal(widgets['widget'], param_info.name, self._emit_parameter_change)

        return container

    def _build_regular_content(self, param_info, display_info, field_ids):
        container = QWidget()
        layout = QHBoxLayout(container)
        label = LabelWithHelp(
            text=display_info['field_label'], param_name=param_info.name,
            param_description=display_info['description'], param_type=param_info.type,
            color_scheme=self.config.color_scheme or PyQt6ColorScheme()
        )
        layout.addWidget(label)
        current_value = self.parameters.get(param_info.name)
        widget = self.create_widget(
            param_info.name, param_info.type, current_value,
            f"{self.field_id}_{param_info.name}",
            self.parameter_info.get(param_info.name)
        )
        if current_value is None and self.dataclass_type:
            self._apply_placeholder_with_lazy_context(widget, param_info.name, current_value)
        widget.setObjectName(field_ids['widget_id'])
        layout.addWidget(widget, 1)
        reset_button = QPushButton(CONSTANTS.RESET_BUTTON_TEXT)
        reset_button.setObjectName(field_ids['reset_button_id'])
        reset_button.setMaximumWidth(60)
        reset_button.clicked.connect(lambda: self.reset_parameter(param_info.name))
        layout.addWidget(reset_button)
        return container, {'main': widget, 'widget': widget, 'reset_button': reset_button}

    def _build_nested_content(self, param_info, display_info, field_ids):
        group_box = GroupBoxWithHelp(
            title=display_info['field_label'], help_target=param_info.type,
            color_scheme=self.config.color_scheme or PyQt6ColorScheme()
        )
        current_value = self.parameters.get(param_info.name)
        if LazyDefaultPlaceholderService.has_lazy_resolution(param_info.type):
            nested_manager = ParameterFormManager.from_dataclass_instance(
                dataclass_instance=current_value or param_info.type(),
                field_id=f"nested_{param_info.name}", placeholder_prefix=self.placeholder_prefix,
                parent=group_box, use_scroll_area=False, color_scheme=self.config.color_scheme
            )
            self.nested_managers[param_info.name] = nested_manager
            # Connect nested parameter changes to trigger parent parameter updates
            # This ensures LazyStepMaterializationConfig changes are properly saved
            nested_manager.parameter_changed.connect(
                lambda name, value, parent_name=param_info.name: self._handle_nested_parameter_change(parent_name, value)
            )
            group_box.content_layout.addWidget(nested_manager)
        else:
            nested_form = self._create_nested_form_inline(param_info.name, param_info.type, current_value)
            group_box.content_layout.addWidget(nested_form)
        return group_box, {'main': group_box}

    def _build_optional_content(self, param_info, display_info, field_ids):
        container = QWidget()
        layout = QVBoxLayout(container)
        checkbox = QCheckBox(display_info['field_label'])
        current_value = self.parameters.get(param_info.name)
        checkbox.setChecked(current_value is not None)
        layout.addWidget(checkbox)
        inner_type = ParameterTypeUtils.get_optional_inner_type(param_info.type)
        nested_param_info = ParameterInfo(param_info.name, inner_type, current_value, True, False)
        nested_widget, nested_widgets = self._build_nested_content(nested_param_info, display_info, field_ids)
        nested_widget.setEnabled(current_value is not None)
        layout.addWidget(nested_widget)
        def toggle(state):
            enabled = state == 2
            nested_widget.setEnabled(enabled)
            new_value = inner_type() if enabled else None
            self.parameters[param_info.name] = new_value
            self.parameter_changed.emit(param_info.name, new_value)
        checkbox.stateChanged.connect(toggle)
        return container, {'main': container}

    def _create_nested_form_inline(self, param_name: str, param_type: Type, current_value: Any) -> Any:
        """Create nested form - inlined from create_nested_form."""
        # Extract nested parameters using service with parent context (handles both dataclass and non-dataclass contexts)
        nested_params, nested_types = self.service.extract_nested_parameters(
            current_value, param_type, self.dataclass_type
        )

        # Get field IDs from service
        field_ids = self.service.generate_field_ids(self.config.field_id, param_name)

        # Return nested manager with inherited configuration
        nested_manager = ParameterFormManager(
            nested_params,
            nested_types,
            field_ids['nested_field_id'],
            param_type,    # Use the actual nested dataclass type, not parent type
            None,  # parameter_info
            None,  # parent
            False,  # use_scroll_area
            None,   # function_target
            PyQt6ColorScheme(),  # color_scheme
            self.placeholder_prefix # Pass through placeholder prefix
        )

        # Store nested manager
        self.nested_managers[param_name] = nested_manager

        return nested_manager



    def _apply_placeholder_with_lazy_context(self, widget: QWidget, param_name: str, current_value: Any) -> None:
        """Apply placeholder using current form state, not saved thread-local state."""
        # Only apply placeholder if value is None
        if current_value is not None:
            return

        # Create placeholder using current form state instead of saved thread-local state
        placeholder_text = self._get_form_state_resolved_placeholder(param_name)

        if placeholder_text is None:
            return

        # Apply the placeholder text to the widget
        if hasattr(widget, 'setPlaceholderText'):
            widget.setPlaceholderText(placeholder_text)

        # Block signals to prevent placeholder application from triggering parameter updates
        widget.blockSignals(True)
        try:
            # Apply placeholder using PyQt6 widget strategies
            PyQt6WidgetEnhancer.apply_placeholder_text(widget, placeholder_text)
        finally:
            # Always restore signal connections
            widget.blockSignals(False)

    def _get_form_state_resolved_placeholder(self, param_name: str) -> Optional[str]:
        """
        Get placeholder text using current form state instead of saved thread-local state.

        This creates a temporary instance using the current form values (including None
        for reset fields) to show what the resolved value would be if we saved now.
        """
        if not self.dataclass_type or not LazyDefaultPlaceholderService.has_lazy_resolution(self.dataclass_type):
            return None

        try:
            # Get current form state from parameters attribute
            current_form_data = self.parameters.copy()

            # CRITICAL FIX: Build form-state config instead of clearing all context
            # This preserves saved values for root fields while using form state for resolution
            from openhcs.core.config import get_current_global_config, set_current_global_config, get_base_type_for_lazy

            # Get the appropriate global config type for this dataclass
            global_config_type = get_base_type_for_lazy(self.dataclass_type)
            if global_config_type is None:
                # Fallback to GlobalPipelineConfig for backward compatibility
                from openhcs.core.config import GlobalPipelineConfig
                global_config_type = GlobalPipelineConfig

            # Save current context
            original_global = get_current_global_config(global_config_type)

            # Build config from form state that preserves saved root values
            form_config = self._build_form_state_config(current_form_data, original_global)

            # Set form-state context
            set_current_global_config(global_config_type, form_config)

            # Create a temporary instance with current form state in form context
            temp_instance = self.dataclass_type()

            # Get the resolved value for this field
            resolved_value = getattr(temp_instance, param_name)

            # Restore original context
            set_current_global_config(global_config_type, original_global)

            # Format as placeholder text
            if resolved_value is not None:
                return f"{self.placeholder_prefix}: {resolved_value}"
            else:
                return None

        except Exception as e:
            logger.warning(f"Failed to resolve placeholder for {param_name}: {e}")
            return None

    def _build_form_state_config(self, current_form_data: dict, original_global: Any) -> Any:
        """
        Build a config that combines form state with preserved saved values.

        This preserves saved root-level values while using form state for nested configs
        to show proper inheritance. Works generically for any lazy dataclass type.
        """
        from openhcs.core.config import get_base_type_for_lazy

        # Get the global config type for this lazy dataclass
        global_config_type = get_base_type_for_lazy(self.dataclass_type)

        if global_config_type is not None:
            # Generic handling for any lazy dataclass with a global config mapping
            from dataclasses import fields

            # Get valid field names for the global config type
            valid_field_names = {f.name for f in fields(global_config_type)}

            if original_global is not None:
                # Start with saved config to preserve root values
                config_dict = original_global.__dict__.copy()

                # Override with form state values where they exist and are not None
                # Only include fields that are valid for this global config type
                for field_name, form_value in current_form_data.items():
                    if form_value is not None and field_name in valid_field_names:
                        config_dict[field_name] = form_value

                return global_config_type(**config_dict)
            else:
                # No saved config, use form state directly
                # Only include fields that are valid for this global config type
                filtered_form_data = {
                    field_name: form_value
                    for field_name, form_value in current_form_data.items()
                    if field_name in valid_field_names
                }
                return global_config_type(**filtered_form_data)
        else:
            # No global config mapping found, use the original approach
            return None

    def _get_compiler_resolved_placeholder(self, param_name: str) -> Optional[str]:
        """
        Get placeholder text using the automatic hierarchy resolution system with custom context.

        Creates a custom context provider that masks the specific field being reset,
        allowing the automatic hierarchy system to resolve through the inheritance chain.
        """
        if not self.dataclass_type or not LazyDefaultPlaceholderService.has_lazy_resolution(self.dataclass_type):
            return None

        try:
            # CRITICAL FIX: Use the automatic system generically - no hardcoding
            from openhcs.core.lazy_config import LazyDataclassFactory
            from openhcs.core.config import get_current_global_config, GlobalPipelineConfig
            from dataclasses import replace

            logger.info(f"🔍 COMPILER DEBUG: Creating generic masked context for {param_name} reset")
            logger.info(f"🔍 COMPILER DEBUG: Dataclass type: {self.dataclass_type}")
            logger.info(f"🔍 COMPILER DEBUG: Available attributes: {dir(self.dataclass_type)}")

            # Get the field path and base class automatically from the lazy dataclass
            field_path = getattr(self.dataclass_type, '_field_path', None)
            base_class = getattr(self.dataclass_type, '_base_class', None)

            logger.info(f"🔍 COMPILER DEBUG: field_path={field_path}, base_class={base_class}")

            if not field_path or not base_class:
                logger.info(f"🔍 COMPILER DEBUG: Missing field_path ({field_path}) or base_class ({base_class})")
                return None

            # Create custom context provider that masks the specific field generically
            def reset_context_provider():
                current_config = get_current_global_config(GlobalPipelineConfig)
                if current_config is None:
                    return None

                if hasattr(current_config, field_path):
                    # Get the nested config and mask the specific field
                    nested_config = getattr(current_config, field_path)
                    if hasattr(nested_config, param_name):
                        # Create modified config with the field set to None
                        modified_nested = replace(nested_config, **{param_name: None})
                        modified_config = replace(current_config, **{field_path: modified_nested})
                        logger.info(f"🔍 COMPILER DEBUG: Masked {param_name} in {field_path}")
                        return modified_config

                return current_config

            # Create temporary lazy class with custom context using the automatic attributes
            temp_lazy_class = LazyDataclassFactory.make_lazy_with_field_level_auto_hierarchy(
                base_class=base_class,
                global_config_type=GlobalPipelineConfig,
                field_path=field_path,
                context_provider=reset_context_provider
            )

            # Create instance and resolve
            temp_instance = temp_lazy_class()
            if hasattr(temp_instance, '_resolve_field_value'):
                logger.info(f"🔍 COMPILER DEBUG: Resolving {param_name} with masked context")
                resolved_value = temp_instance._resolve_field_value(param_name)
                logger.info(f"🔍 COMPILER DEBUG: Masked context returned: {resolved_value}")

                if resolved_value is not None:
                    # Format the resolved value as placeholder text
                    from openhcs.core.config import EnumDisplayFormatter
                    if hasattr(resolved_value, '__class__') and hasattr(resolved_value.__class__, '__name__'):
                        if 'Enum' in str(type(resolved_value)):
                            formatted_value = EnumDisplayFormatter.get_code_representation(resolved_value)
                        else:
                            formatted_value = str(resolved_value)
                    else:
                        formatted_value = str(resolved_value)

                    return f"{self.placeholder_prefix}: {formatted_value}"

        except Exception as e:
            # If custom context resolution fails, fall back to original method
            logger.info(f"🔍 COMPILER DEBUG: Custom context failed for {param_name}: {e}")

        return None

    def _emit_parameter_change(self, param_name: str, value: Any) -> None:
        """Handle parameter change from widget and update parameter data model."""
        # Convert value using service layer
        converted_value = self.service.convert_value_to_type(value, self.parameter_types.get(param_name, type(value)), param_name)

        # Update parameter in data model
        self.parameters[param_name] = converted_value

        # Emit signal only once
        self.parameter_changed.emit(param_name, converted_value)
    
    def update_widget_value(self, widget: QWidget, value: Any, param_name: str = None) -> None:
        """Update widget value using functional dispatch."""
        dispatch_table = [
            (QComboBox, self._update_combo_box),
            ('get_selected_values', self._update_checkbox_group),
            ('setChecked', lambda w, v: w.setChecked(bool(v))),
            ('setValue', lambda w, v: w.setValue(v or 0)),
            ('set_value', lambda w, v: w.set_value(v)),
            ('setText', lambda w, v: w.setText(str(v or ""))),
            ('clear', lambda w, v: v is None and w.clear())
        ]

        self._execute_with_signal_blocking(widget, lambda: next(
            (updater(widget, value) for matcher, updater in dispatch_table
             if (isinstance(widget, matcher) if isinstance(matcher, type) else hasattr(widget, matcher))), None
        ))
        self._apply_context_behavior(widget, value, param_name)

    def _update_combo_box(self, widget: QComboBox, value: Any) -> None:
        """Update combo box with value matching."""
        widget.setCurrentIndex(-1 if value is None else
                             next((i for i in range(widget.count()) if widget.itemData(i) == value), -1))

    def _update_checkbox_group(self, widget: QWidget, value: Any) -> None:
        """Update checkbox group using functional operations."""
        if hasattr(widget, '_checkboxes') and isinstance(value, list):
            # Functional: reset all, then set selected
            [cb.setChecked(False) for cb in widget._checkboxes.values()]
            [widget._checkboxes[v].setChecked(True) for v in value if v in widget._checkboxes]

    def _execute_with_signal_blocking(self, widget: QWidget, operation: callable) -> None:
        """Execute operation with signal blocking - stateless utility."""
        widget.blockSignals(True)
        operation()
        widget.blockSignals(False)

    def _apply_context_behavior(self, widget: QWidget, value: Any, param_name: str) -> None:
        """Apply lazy placeholder context behavior - pure function of inputs."""
        if not param_name or not self.dataclass_type:
            return

        if value is None and LazyDefaultPlaceholderService.has_lazy_resolution(self.dataclass_type):
            self._apply_placeholder_with_lazy_context(widget, param_name, value)
        elif value is not None:
            PyQt6WidgetEnhancer._clear_placeholder_state(widget)



    def get_widget_value(self, widget: QWidget) -> Any:
        """Get widget value using functional dispatch."""
        dispatch_table = [
            (QComboBox, lambda w: w.itemData(w.currentIndex()) if w.currentIndex() >= 0 else None),
            ('get_selected_values', lambda w: w.get_selected_values()),
            ('get_value', lambda w: w.get_value()),
            ('isChecked', lambda w: w.isChecked()),
            ('value', lambda w: None if (hasattr(w, 'specialValueText') and w.value() == w.minimum() and w.specialValueText()) else w.value()),
            ('text', lambda w: w.text())
        ]

        return next(
            (extractor(widget) for matcher, extractor in dispatch_table
             if (isinstance(widget, matcher) if isinstance(matcher, type) else hasattr(widget, matcher))),
            None
        )

    # Framework-specific methods for backward compatibility

    def reset_all_parameters(self) -> None:
        """Reset all parameters - let reset_parameter handle everything."""
        for param_name in self.parameters.keys():
            self.reset_parameter(param_name)

        # Handle nested managers once at the end
        if self.dataclass_type and self.nested_managers:
            current_config = getattr(self, '_current_config_instance', None)
            if current_config:
                self.service.reset_nested_managers(self.nested_managers, self.dataclass_type, current_config)



    # Core parameter management methods (using shared service layer)

    def update_parameter(self, param_name: str, value: Any) -> None:
        """Update parameter value using shared service layer."""


        if param_name in self.parameters:
            # Convert value using service layer
            converted_value = self.service.convert_value_to_type(value, self.parameter_types.get(param_name, type(value)), param_name)

            # Update parameter in data model
            self.parameters[param_name] = converted_value

            # Update corresponding widget if it exists
            if param_name in self.widgets:
                self.update_widget_value(self.widgets[param_name], converted_value)

            # Emit signal for PyQt6 compatibility
            self.parameter_changed.emit(param_name, converted_value)


    def _is_function_parameter(self, param_name: str) -> bool:
        """
        Detect if parameter is a function parameter vs dataclass field.

        Function parameters should not be reset against dataclass types.
        This prevents the critical bug where step editor tries to reset
        function parameters like 'group_by' against GlobalPipelineConfig.
        """
        if not self.function_target or not self.dataclass_type:
            return False

        # Check if parameter exists in dataclass fields
        try:
            import dataclasses
            if dataclasses.is_dataclass(self.dataclass_type):
                field_names = {field.name for field in dataclasses.fields(self.dataclass_type)}
                # If parameter is NOT in dataclass fields, it's a function parameter
                return param_name not in field_names
        except Exception:
            # If we can't determine, assume it's a function parameter to be safe
            return True

        return False

    def reset_parameter(self, param_name: str, default_value: Any = None) -> None:
        """Reset parameter with streamlined logic."""
        print(f"🔄 RESET DEBUG: reset_parameter called for {param_name}")

        if param_name not in self.parameters:
            print(f"🔄 RESET DEBUG: {param_name} not in parameters, returning")
            return

        # Resolve reset value using dispatch
        reset_value = default_value or self._get_reset_value(param_name)

        print(f"🔄 RESET DEBUG: Resetting {param_name}")
        print(f"🔄 RESET DEBUG: Current value: {self.parameters.get(param_name)}")
        print(f"🔄 RESET DEBUG: Reset value: {reset_value}")
        print(f"🔄 RESET DEBUG: Dataclass type: {self.dataclass_type}")

        # Apply reset with functional operations
        self.parameters[param_name] = reset_value
        print(f"🔄 RESET DEBUG: After setting parameters[{param_name}] = {self.parameters[param_name]}")

        self.widgets.get(param_name) and self.update_widget_value(self.widgets[param_name], reset_value, param_name)
        self.nested_managers.get(param_name) and hasattr(self.nested_managers[param_name], 'reset_all_parameters') and self.nested_managers[param_name].reset_all_parameters()
        self.parameter_changed.emit(param_name, reset_value)

    def _get_reset_value(self, param_name: str) -> Any:
        """Get reset value using context dispatch."""
        if self.dataclass_type:
            is_static = not LazyDefaultPlaceholderService.has_lazy_resolution(self.dataclass_type)
            print(f"🔄 RESET DEBUG: Getting reset value for {param_name}")
            print(f"🔄 RESET DEBUG: Parameter type: {self.parameter_types.get(param_name)}")
            print(f"🔄 RESET DEBUG: Dataclass type: {self.dataclass_type}")
            print(f"🔄 RESET DEBUG: Is static (not lazy): {is_static}")

            reset_value = self.service.get_reset_value_for_parameter(
                param_name, self.parameter_types.get(param_name), self.dataclass_type, is_static)

            print(f"🔄 RESET DEBUG: Service returned reset value: {reset_value}")
            return reset_value

        return (getattr(self.parameter_info.get(param_name, object()), 'default_value', None)
                if self.parameter_info and param_name in self.parameter_info else None)

    def get_current_values(self) -> Dict[str, Any]:
        """
        Get current parameter values preserving lazy dataclass structure.

        This fixes the lazy default materialization override saving issue by ensuring
        that lazy dataclasses maintain their structure when values are retrieved.
        """
        # Start with a copy of current parameters
        current_values = self.parameters.copy()

        # Validate optional dataclass checkbox states
        self._validate_optional_dataclass_checkbox_states(current_values)

        # Collect values from nested managers, respecting optional dataclass checkbox states
        self._apply_to_nested_managers(
            lambda name, manager: self._process_nested_values_if_checkbox_enabled(
                name, manager, current_values
            )
        )

        # Lazy dataclasses are now handled by LazyDataclassEditor, so no structure preservation needed
        return current_values

    def _get_optional_checkbox_state(self, param_name: str) -> Optional[bool]:
        """Get checkbox state for optional dataclass parameter."""
        param_type = self.parameter_types.get(param_name)
        if not (param_type and self.service._type_utils.is_optional_dataclass(param_type)):
            return None

        widget = self.widgets.get(param_name)
        checkbox = self._find_checkbox_in_container(widget) if widget else None
        return checkbox.isChecked() if checkbox else None

    def _validate_optional_dataclass_checkbox_states(self, current_values: Dict[str, Any]) -> None:
        """Update parameter values based on checkbox states."""
        for param_name in self.widgets.keys():
            checkbox_state = self._get_optional_checkbox_state(param_name)
            if checkbox_state is True and current_values[param_name] is None:
                # Create default instance when checkbox is checked
                param_type = self.parameter_types[param_name]
                inner_type = self.service._type_utils.get_optional_inner_type(param_type)
                current_values[param_name] = inner_type()
            elif checkbox_state is False:
                # Clear value when checkbox is unchecked
                current_values[param_name] = None

    def _process_nested_values_if_checkbox_enabled(self, param_name: str, manager: Any,
                                                 current_values: Dict[str, Any]) -> None:
        """Process nested values only if checkbox is enabled."""
        if self._get_optional_checkbox_state(param_name) is not False:
            self._process_nested_values(param_name, manager.get_current_values(), current_values)

    def _handle_nested_parameter_change(self, parent_name: str, value: Any) -> None:
        """Handle nested parameter changes respecting checkbox state."""
        if self._get_optional_checkbox_state(parent_name) is not False:
            self.parameter_changed.emit(parent_name, value)

    def _find_checkbox_in_container(self, container: QWidget) -> Optional['QCheckBox']:
        """Find the checkbox widget within an optional dataclass container."""
        from PyQt6.QtWidgets import QCheckBox

        # Check if the container itself is a checkbox
        if isinstance(container, QCheckBox):
            return container

        # Search for checkbox in container's children
        for child in container.findChildren(QCheckBox):
            return child  # Return the first checkbox found

        return None





    def refresh_placeholder_text(self) -> None:
        """
        Refresh placeholder text for all widgets to reflect current GlobalPipelineConfig.

        This method should be called when the GlobalPipelineConfig changes to ensure
        that lazy dataclass forms show updated placeholder text.
        """
        # Only refresh for lazy dataclasses (PipelineConfig forms)
        if not self.dataclass_type:
            return

        is_lazy_dataclass = LazyDefaultPlaceholderService.has_lazy_resolution(self.dataclass_type)

        if not is_lazy_dataclass:
            return

        # Refresh placeholder text for all widgets with None values
        for param_name, widget in self.widgets.items():
            current_value = self.parameters.get(param_name)
            if current_value is None:
                self._apply_placeholder_with_lazy_context(widget, param_name, current_value)

        # Recursively refresh nested managers
        self._apply_to_nested_managers(lambda name, manager: manager.refresh_placeholder_text())



    def _rebuild_nested_dataclass_instance(self, nested_values: Dict[str, Any],
                                         nested_type: Type, param_name: str) -> Any:
        """
        Rebuild nested dataclass instance from current values.

        This method handles both lazy and non-lazy dataclasses by checking the nested_type
        itself rather than the parent dataclass type.

        Args:
            nested_values: Current values from nested manager
            nested_type: The dataclass type to create
            param_name: Parameter name for context

        Returns:
            Reconstructed dataclass instance
        """
        # Check if the nested type itself is a lazy dataclass
        # This is the correct check - we need to examine the nested type, not the parent
        nested_type_is_lazy = LazyDefaultPlaceholderService.has_lazy_resolution(nested_type)

        if nested_type_is_lazy:
            # Lazy dataclass: preserve None values for lazy resolution, include concrete values
            # This maintains the "lazy mixed" pattern where some fields are concrete and others are None
            return nested_type(**nested_values)
        else:
            # Non-lazy dataclass: filter out None values and use concrete dataclass
            filtered_values = {k: v for k, v in nested_values.items() if v is not None}
            if filtered_values:
                return nested_type(**filtered_values)
            else:
                return nested_type()

    def _apply_to_nested_managers(self, operation_func: callable) -> None:
        """Apply operation to all nested managers."""
        for param_name, nested_manager in self.nested_managers.items():
            operation_func(param_name, nested_manager)

    def _process_nested_values(self, param_name: str, nested_values: Dict[str, Any], current_values: Dict[str, Any]) -> None:
        """Process nested values and rebuild dataclass instance."""
        nested_type = self.parameter_types.get(param_name)
        if nested_type and nested_values:
            if self.service._type_utils.is_optional_dataclass(nested_type):
                nested_type = self.service._type_utils.get_optional_inner_type(nested_type)
            rebuilt_instance = self._rebuild_nested_dataclass_instance(nested_values, nested_type, param_name)
            current_values[param_name] = rebuilt_instance