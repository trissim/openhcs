# File: openhcs/textual_tui/widgets/shared/parameter_form_manager.py

import dataclasses
import ast
import logging
from enum import Enum
from typing import Any, Dict, get_origin, get_args, Union, Optional, Type

logger = logging.getLogger(__name__)
from textual.containers import Vertical, Horizontal
from textual.widgets import Static, Button, Collapsible
from textual.app import ComposeResult

from .typed_widget_factory import TypedWidgetFactory
from .signature_analyzer import SignatureAnalyzer
from .clickable_help_label import ClickableParameterLabel, HelpIndicator
from ..different_values_input import DifferentValuesInput

# Import simplified abstraction layer
from openhcs.ui.shared.parameter_form_abstraction import (
    ParameterFormAbstraction, apply_lazy_default_placeholder
)
from openhcs.ui.shared.widget_creation_registry import create_textual_registry
from openhcs.ui.shared.textual_widget_strategies import create_different_values_widget

# Import new shared utilities
from openhcs.ui.shared.parameter_form_constants import CONSTANTS
from openhcs.ui.shared.field_id_generator import FieldIdGenerator
from openhcs.ui.shared.parameter_type_utils import ParameterTypeUtils
from openhcs.ui.shared.parameter_name_formatter import ParameterNameFormatter
from openhcs.ui.shared.debug_config import get_debugger, DebugConfig

class ParameterFormManager:
    """Mathematical: (parameters, types, field_id) → parameter form

    Refactored implementation using shared infrastructure while maintaining
    exact backward compatibility with the original API.
    """

    def __init__(self, parameters: Dict[str, Any], parameter_types: Dict[str, type], field_id: str, parameter_info: Dict = None, is_global_config_editing: bool = False, global_config_type: Optional[Type] = None, placeholder_prefix: str = None):
        # Convert old constructor arguments to config object
        from openhcs.ui.shared.parameter_form_config_factory import textual_config

        if placeholder_prefix is None:
            placeholder_prefix = CONSTANTS.DEFAULT_PLACEHOLDER_PREFIX

        config = textual_config(
            field_id=field_id,
            parameter_info=parameter_info,
            is_global_config_editing=is_global_config_editing,
            global_config_type=global_config_type,
            placeholder_prefix=placeholder_prefix
        )

        # Store public API attributes for backward compatibility
        self.parameters = parameters.copy()
        self.parameter_types = parameter_types
        self.field_id = field_id
        self.parameter_info = parameter_info or {}
        self.is_global_config_editing = is_global_config_editing
        self.global_config_type = global_config_type
        self.placeholder_prefix = placeholder_prefix

        # Initialize using the simplified implementation
        self._simplified_manager = SimplifiedParameterFormManager(parameters, parameter_types, config)

        # Expose simplified manager's attributes for compatibility
        self.debugger = self._simplified_manager.debugger
        self.service = self._simplified_manager.service
        self.form_structure = self._simplified_manager.form_structure
        self.nested_managers = self._simplified_manager.nested_managers
        self.widgets = self._simplified_manager.widgets

        # Initialize legacy abstraction layer for compatibility with existing widget creation
        self.form_abstraction = ParameterFormAbstraction(
            parameters, parameter_types, field_id, create_textual_registry(), parameter_info
        )

        # Initialize optional checkboxes for compatibility
        self.optional_checkboxes = {}
    
    def build_form(self) -> ComposeResult:
        """Build parameter form - simplified using service layer analysis."""
        with Vertical() as form:
            form.styles.height = CONSTANTS.AUTO_SIZE

            # Use pre-analyzed form structure instead of re-analyzing
            for param_info in self._form_structure.parameters:
                if param_info.is_optional and param_info.is_nested:
                    yield from self._build_optional_dataclass_form(
                        param_info.name,
                        ParameterTypeUtils.get_optional_inner_type(param_info.type),
                        param_info.current_value
                    )
                elif param_info.is_nested:
                    yield from self._build_nested_dataclass_form(
                        param_info.name,
                        param_info.type,
                        param_info.current_value
                    )
                else:
                    yield from self._build_regular_parameter_form(
                        param_info.name,
                        param_info.type,
                        param_info.current_value
                    )

    def _build_nested_dataclass_form(self, param_name: str, param_type: type, current_value: Any) -> ComposeResult:
        """Build form for nested dataclass parameter - simplified using service layer."""
        # Create collapsible widget (no ID - just structure)
        collapsible = TypedWidgetFactory.create_widget(param_type, current_value, None)

        # Use service layer to extract nested parameters (eliminates manual analysis)
        nested_parameters, nested_parameter_types = self._service.extract_nested_parameters(current_value, param_type)

        # Get nested parameter info from signature analyzer (maintain compatibility)
        nested_param_info = SignatureAnalyzer.analyze(param_type)

        # Create nested form manager with simplified constructor
        nested_field_id = FieldIdGenerator.nested_field_id(self.field_id, param_name)
        nested_form_manager = ParameterFormManager(
            nested_parameters,
            nested_parameter_types,
            nested_field_id,
            nested_param_info,
            self.is_global_config_editing,
            self.global_config_type,
            self.placeholder_prefix
        )

        # Store the parent dataclass type for proper lazy resolution detection
        nested_form_manager._parent_dataclass_type = param_type

        # Store reference to nested form manager for updates
        self.nested_managers[param_name] = nested_form_manager

        # Build nested form and add to collapsible
        with collapsible:
            yield from nested_form_manager.build_form()

        yield collapsible

    def _is_optional_dataclass(self, param_type: type) -> bool:
        """Check if parameter type is Optional[dataclass]."""
        return ParameterTypeUtils.is_optional_dataclass(param_type)

    def _get_optional_inner_type(self, param_type: type) -> type:
        """Extract the inner type from Optional[T]."""
        return ParameterTypeUtils.get_optional_inner_type(param_type)

    def _build_optional_dataclass_form(self, param_name: str, dataclass_type: type, current_value: Any) -> ComposeResult:
        """Build form for Optional[dataclass] parameter with checkbox toggle."""
        from textual.widgets import Checkbox

        # Checkbox
        checkbox_id = FieldIdGenerator.optional_checkbox_id(self.field_id, param_name)
        checkbox = Checkbox(
            value=current_value is not None,
            label=ParameterNameFormatter.to_checkbox_label(param_name),
            id=checkbox_id,
            compact=CONSTANTS.COMPACT_WIDGET
        )
        yield checkbox

        # Collapsible dataclass widget
        collapsible = TypedWidgetFactory.create_widget(dataclass_type, current_value, None)
        collapsible.collapsed = (current_value is None)

        # Setup nested form
        nested_param_info = SignatureAnalyzer.analyze(dataclass_type)
        nested_parameters = {}
        for name, info in nested_param_info.items():
            if current_value:
                # For lazy dataclasses, preserve None values for placeholder behavior
                if ParameterTypeUtils.has_resolve_field_value(current_value):
                    value = object.__getattribute__(current_value, name) if hasattr(current_value, name) else info.default_value
                else:
                    value = getattr(current_value, name, info.default_value)
            else:
                value = info.default_value
            nested_parameters[name] = value
        nested_parameter_types = {name: info.param_type for name, info in nested_param_info.items()}

        nested_field_id = FieldIdGenerator.nested_field_id(self.field_id, param_name)
        nested_form_manager = ParameterFormManager(
            nested_parameters, nested_parameter_types, nested_field_id, nested_param_info,
            is_global_config_editing=self.is_global_config_editing
        )

        # Store the parent dataclass type for proper lazy resolution detection
        nested_form_manager._parent_dataclass_type = dataclass_type

        # Store references
        if not hasattr(self, CONSTANTS.NESTED_MANAGERS_ATTR):
            self.nested_managers = {}
        if not hasattr(self, CONSTANTS.OPTIONAL_CHECKBOXES_ATTR):
            self.optional_checkboxes = {}
        self.nested_managers[param_name] = nested_form_manager
        self.optional_checkboxes[param_name] = checkbox

        with collapsible:
            yield from nested_form_manager.build_form()
        yield collapsible

    def _build_regular_parameter_form(self, param_name: str, param_type: type, current_value: Any) -> ComposeResult:
        """Build form for regular (non-dataclass) parameter - simplified using service layer."""
        # Get display information and field IDs from service layer
        display_info = self._service.get_parameter_display_info(param_name, param_type)
        field_ids = self._service.generate_field_ids(self.field_id, param_name)

        # Check if this field has different values across orchestrators
        config_analysis = getattr(self, 'config_analysis', {})
        field_analysis = config_analysis.get(param_name, {})

        # Handle different values or create normal widget
        if field_analysis.get(CONSTANTS.DIFFERENT_VALUES_TYPE) == CONSTANTS.DIFFERENT_VALUES_TYPE:
            default_value = field_analysis.get('default')
            input_widget = create_different_values_widget(param_name, param_type, default_value, field_ids['widget_id'])
        else:
            # Use registry for widget creation and apply placeholder
            widget_value = ParameterTypeUtils.extract_value_attribute(current_value)
            input_widget = self.form_abstraction.create_widget_for_parameter(param_name, param_type, widget_value)
            apply_lazy_default_placeholder(input_widget, param_name, current_value, self.parameter_types, CONSTANTS.TEXTUAL_FRAMEWORK,
                                          is_global_config_editing=self.is_global_config_editing,
                                          global_config_type=self.global_config_type,
                                          placeholder_prefix=self.placeholder_prefix)

        # 3-column layout: label + input + reset
        with Horizontal() as row:
            row.styles.height = CONSTANTS.AUTO_SIZE

            # Parameter label with help - use service-generated display info
            label = ClickableParameterLabel(
                param_name,
                display_info['description'],
                param_type,
                classes=CONSTANTS.PARAM_LABEL_CLASS
            )

            label.styles.width = CONSTANTS.AUTO_SIZE
            label.styles.text_align = CONSTANTS.LEFT_ALIGN
            label.styles.height = "1"
            yield label

            # Input widget (flexible width, left aligned, with left margin for spacing)
            input_widget.styles.width = CONSTANTS.FLEXIBLE_WIDTH
            input_widget.styles.text_align = CONSTANTS.LEFT_ALIGN
            input_widget.styles.margin = CONSTANTS.LEFT_MARGIN_ONLY
            yield input_widget

            # Reset button (auto width)
            reset_btn_id = FieldIdGenerator.reset_button_id(widget_id)
            reset_btn = Button(CONSTANTS.RESET_BUTTON_TEXT, id=reset_btn_id, compact=CONSTANTS.COMPACT_WIDGET)
            reset_btn.styles.width = CONSTANTS.AUTO_SIZE
            yield reset_btn
    
    def update_parameter(self, param_name: str, value: Any):
        """Update parameter value with centralized enum conversion and nested dataclass support."""
        # Debug logging using the new debug system
        self.debugger.log_parameter_update(param_name, value, "textual_update")

        # Parse hierarchical parameter name (e.g., "path_planning_global_output_folder")
        # Split and check if this is a nested parameter
        parts = param_name.split(CONSTANTS.FIELD_ID_SEPARATOR)
        if len(parts) >= 2:  # nested_field format
            # Try to find nested manager by checking all possible prefixes
            for i in range(1, len(parts)):
                potential_nested = CONSTANTS.FIELD_ID_SEPARATOR.join(parts[:i])
                if potential_nested in self.parameters and hasattr(self, CONSTANTS.NESTED_MANAGERS_ATTR) and potential_nested in self.nested_managers:
                    # Reconstruct the nested field name from remaining parts
                    nested_field = CONSTANTS.FIELD_ID_SEPARATOR.join(parts[i:])

                    # Update nested form manager
                    self.debugger.log_nested_update(potential_nested, nested_field, value)
                    self.nested_managers[potential_nested].update_parameter(nested_field, value)

                    # Rebuild nested dataclass instance with lazy/concrete mixed behavior
                    nested_values = self.nested_managers[potential_nested].get_current_values()

                    # Debug: Check what values the nested manager is returning
                    self.debugger.log_form_manager_operation("nested_values_retrieved", {
                        "parent": potential_nested,
                        "nested_values": nested_values
                    })

                    nested_type = self.parameter_types[potential_nested]

                    # Resolve Union types (like Optional[DataClass]) to the actual dataclass type
                    if self._is_optional_dataclass(nested_type):
                        nested_type = self._get_optional_inner_type(nested_type)

                    # Create lazy dataclass instance with mixed concrete/lazy fields
                    if self.is_global_config_editing:
                        # Global config editing: use concrete dataclass
                        self.parameters[potential_nested] = nested_type(**nested_values)
                    else:
                        # Lazy context: create lazy instance using shared utility
                        self.parameters[potential_nested] = self._convert_to_lazy_dataclass(nested_values, nested_type)
                    return

        # Handle regular parameters (direct match)
        if param_name in self.parameters:
            # Handle literal "None" string - convert back to Python None
            if isinstance(value, str) and value == CONSTANTS.NONE_STRING_LITERAL:
                value = None

            # Convert string back to proper type (comprehensive conversion)
            # Skip type conversion for None values (preserve for lazy placeholder behavior)
            if param_name in self.parameter_types and value is not None:
                param_type = self.parameter_types[param_name]
                if ParameterTypeUtils.is_enum_type(param_type):
                    value = param_type(value)  # Convert string → enum
                elif ParameterTypeUtils.is_list_of_enums(param_type):
                    # Handle List[Enum] types (like List[VariableComponents])
                    enum_type = ParameterTypeUtils.get_enum_from_list_type(param_type)
                    if enum_type:
                        # Convert string value to enum, then wrap in list
                        enum_value = enum_type(value)
                        value = [enum_value]
                elif param_type == float:
                    # Convert string → float, handle empty based on parameter requirements
                    try:
                        if value == CONSTANTS.EMPTY_STRING:
                            # For empty values, we need to check if parameter is required
                            # This requires access to parameter info, but we don't have it here
                            # For now, convert empty to None (safer than 0.0)
                            value = None
                        else:
                            value = float(value)
                    except (ValueError, TypeError):
                        value = None  # Use None instead of 0.0 for failed conversions
                elif param_type == int:
                    # Convert string → int, handle empty based on parameter requirements
                    try:
                        if value == CONSTANTS.EMPTY_STRING:
                            # For empty values, convert to None (safer than 0)
                            value = None
                        else:
                            value = int(value)
                    except (ValueError, TypeError):
                        value = None  # Use None instead of 0 for failed conversions
                elif param_type == bool:
                    # Convert string → bool
                    if isinstance(value, str):
                        value = ParameterTypeUtils.convert_string_to_bool(value)
                # Add more type conversions as needed

            self.parameters[param_name] = value

            # FALLBACK: If this is a nested field that bypassed the nested logic, update the nested manager
            self.debugger.log_form_manager_operation("fallback_check", {"param_name": param_name})
            if hasattr(self, CONSTANTS.NESTED_MANAGERS_ATTR):
                for nested_name, nested_manager in self.nested_managers.items():
                    if param_name in nested_manager.parameter_types:
                        self.debugger.log_form_manager_operation("fallback_update", {
                            "nested_name": nested_name,
                            "param_name": param_name,
                            "value": value
                        })
                        nested_manager.parameters[param_name] = value
                        break

            # Debug: Check what was actually stored
            self.debugger.log_parameter_update(param_name, self.parameters.get(param_name), "stored_value")
    
    def reset_parameter(self, param_name: str, default_value: Any = None):
        """Reset parameter to appropriate default value based on lazy vs concrete dataclass context."""
        # Determine the correct reset value if not provided
        if default_value is None:
            default_value = self._get_reset_value_for_parameter(param_name)

        # Parse hierarchical parameter name for nested parameters
        parts = param_name.split(CONSTANTS.FIELD_ID_SEPARATOR)
        if len(parts) >= 2:  # nested_field format
            # Try to find nested manager by checking all possible prefixes
            for i in range(1, len(parts)):
                potential_nested = CONSTANTS.FIELD_ID_SEPARATOR.join(parts[:i])
                if potential_nested in self.parameters and hasattr(self, CONSTANTS.NESTED_MANAGERS_ATTR) and potential_nested in self.nested_managers:
                    # Reconstruct the nested field name
                    nested_field = CONSTANTS.FIELD_ID_SEPARATOR.join(parts[i:])

                    # Get appropriate reset value for nested field
                    nested_reset_value = self._get_reset_value_for_nested_parameter(potential_nested, nested_field)

                    # Reset in nested form manager
                    self.debugger.log_reset_operation(f"{potential_nested}.{nested_field}", None, nested_reset_value)
                    self.nested_managers[potential_nested].reset_parameter(nested_field, nested_reset_value)

                    # Rebuild nested dataclass instance
                    nested_values = self.nested_managers[potential_nested].get_current_values()

                    # Resolve Union types (like Optional[DataClass]) to the actual dataclass type
                    if self._is_optional_dataclass(self.parameter_types[potential_nested]):
                        nested_type = self._get_optional_inner_type(self.parameter_types[potential_nested])
                    else:
                        nested_type = self.parameter_types[potential_nested]

                    # Create lazy dataclass instance with mixed concrete/lazy fields
                    if self.is_global_config_editing:
                        # Global config editing: use concrete dataclass
                        self.parameters[potential_nested] = nested_type(**nested_values)
                    else:
                        # Lazy context: create lazy instance using shared utility
                        self.parameters[potential_nested] = self._convert_to_lazy_dataclass(nested_values, nested_type)
                    return

        # Handle regular parameters
        if param_name in self.parameters:
            old_value = self.parameters[param_name]
            self.parameters[param_name] = default_value
            self.debugger.log_reset_operation(param_name, old_value, default_value)

            # Handle special reset behavior for DifferentValuesInput widgets
            self._handle_different_values_reset(param_name)

            # Re-apply placeholder styling if value is None (for reset functionality)
            if default_value is None:
                self._reapply_placeholder_if_needed(param_name)

    def _reapply_placeholder_if_needed(self, param_name: str):
        """Re-apply placeholder styling to a widget when its value is set to None."""
        # For Textual, we need to find the widget and re-apply placeholder
        # This is more complex than PyQt since Textual widgets are reactive
        # For now, we'll rely on the reactive nature of Textual widgets
        # The placeholder should be re-applied automatically when the value changes to None
        pass

    def _get_reset_value_for_parameter(self, param_name: str) -> Any:
        """
        Get the appropriate reset value for a parameter based on lazy vs concrete dataclass context.

        For concrete dataclasses (like GlobalPipelineConfig):
        - Reset to static class defaults

        For lazy dataclasses (like PipelineConfig for orchestrator configs):
        - Reset to None to preserve placeholder behavior and inheritance hierarchy
        """
        if param_name not in self.parameter_info:
            return None

        param_info = self.parameter_info[param_name]
        param_type = self.parameter_types[param_name]

        # For global config editing, always use static defaults
        if self.is_global_config_editing:
            return param_info.default_value

        # For nested dataclass fields, check if we should use concrete values
        if ParameterTypeUtils.has_dataclass_fields(param_type):
            # This is a dataclass field - determine if it should be concrete or None
            current_value = self.parameters.get(param_name)
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

    def _get_reset_value_for_nested_parameter(self, nested_param_name: str, nested_field_name: str) -> Any:
        """Get appropriate reset value for a nested parameter field."""
        nested_type = self.parameter_types[nested_param_name]
        nested_param_info = SignatureAnalyzer.analyze(nested_type)

        if nested_field_name not in nested_param_info:
            return None

        nested_field_info = nested_param_info[nested_field_name]

        # For global config editing, always use static defaults
        if self.is_global_config_editing:
            return nested_field_info.default_value

        # For lazy context, check if nested dataclass should use concrete values
        current_nested_value = self.parameters.get(nested_param_name)
        if self._should_use_concrete_nested_values(current_nested_value):
            return nested_field_info.default_value
        else:
            return None

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
        This mirrors the logic from the PyQt form manager.

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
        if ParameterTypeUtils.is_concrete_dataclass(current_value):
            return True

        # For lazy dataclasses, always return False to enable mixed lazy/concrete behavior
        # Individual field values will be checked separately in the nested form creation
        if ParameterTypeUtils.is_lazy_dataclass(current_value):
            return False

        # Default to placeholder behavior for lazy contexts
        return False

    def handle_optional_checkbox_change(self, param_name: str, enabled: bool):
        """Handle checkbox change for Optional[dataclass] parameters."""
        if param_name in self.parameter_types and self._is_optional_dataclass(self.parameter_types[param_name]):
            dataclass_type = self._get_optional_inner_type(self.parameter_types[param_name])
            nested_managers = getattr(self, 'nested_managers', {})
            self.parameters[param_name] = (
                dataclass_type(**nested_managers[param_name].get_current_values())
                if enabled and param_name in nested_managers
                else dataclass_type() if enabled
                else None
            )

    def _handle_different_values_reset(self, param_name: str):
        """Handle reset behavior for DifferentValuesInput widgets."""
        # Check if this field has different values across orchestrators
        config_analysis = getattr(self, 'config_analysis', {})
        field_analysis = config_analysis.get(param_name, {})

        if field_analysis.get('type') == CONSTANTS.DIFFERENT_VALUES_TYPE:
            # For different values fields, reset means go back to "DIFFERENT VALUES" state
            # We need to find the widget and call its reset method
            widget_id = FieldIdGenerator.widget_id(self.field_id, param_name)

            # This will be handled by the screen/container that manages the widgets
            # The widget itself will handle the reset via its reset_to_different() method
            # We just need to ensure the parameter value reflects the "different" state
            pass  # Widget-level reset will be handled by the containing screen

    def reset_all_parameters(self, defaults: Dict[str, Any] = None):
        """Reset all parameters to appropriate defaults based on lazy vs concrete dataclass context."""
        # If no defaults provided, generate them based on context
        if defaults is None:
            defaults = {}
            for param_name in self.parameters.keys():
                defaults[param_name] = self._get_reset_value_for_parameter(param_name)

        for param_name, default_value in defaults.items():
            if param_name in self.parameters:
                # Handle nested dataclasses
                if dataclasses.is_dataclass(self.parameter_types.get(param_name)):
                    if hasattr(self, 'nested_managers') and param_name in self.nested_managers:
                        # Generate appropriate reset values for nested parameters
                        nested_type = self.parameter_types[param_name]
                        nested_param_info = SignatureAnalyzer.analyze(nested_type)

                        # Use lazy-aware reset logic for nested parameters with mixed state support
                        nested_defaults = {}
                        for nested_field_name in nested_param_info.keys():
                            # For nested fields in lazy contexts, always reset to None to preserve lazy behavior
                            # This ensures individual fields can maintain placeholder behavior regardless of other field states
                            if not self.is_global_config_editing:
                                nested_defaults[nested_field_name] = None
                            else:
                                nested_defaults[nested_field_name] = self._get_reset_value_for_nested_parameter(param_name, nested_field_name)

                        self.nested_managers[param_name].reset_all_parameters(nested_defaults)

                        # Rebuild nested dataclass instance
                        nested_values = self.nested_managers[param_name].get_current_values()

                        # Resolve Union types (like Optional[DataClass]) to the actual dataclass type
                        if self._is_optional_dataclass(nested_type):
                            nested_type = self._get_optional_inner_type(nested_type)

                        # Create lazy dataclass instance with mixed concrete/lazy fields
                        if self.is_global_config_editing:
                            # Global config editing: use concrete dataclass
                            self.parameters[param_name] = nested_type(**nested_values)
                        else:
                            # Lazy context: create lazy instance using shared utility
                            self.parameters[param_name] = self._convert_to_lazy_dataclass(nested_values, nested_type)
                    else:
                        self.parameters[param_name] = default_value
                else:
                    self.parameters[param_name] = default_value

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
                    nested_manager.reset_parameter(nested_param)
            else:
                logger.warning(f"Nested manager '{nested_name}' not found for parameter path '{parameter_path}'")
        else:
            # Handle top-level parameter
            self.reset_parameter(parameter_path)

    def _is_list_of_enums(self, param_type) -> bool:
        """Check if parameter type is List[Enum]."""
        return ParameterTypeUtils.is_list_of_enums(param_type)

    def _get_enum_from_list(self, param_type):
        """Extract enum type from List[Enum] type."""
        return ParameterTypeUtils.get_enum_from_list_type(param_type)

    @staticmethod
    def _is_list_of_enums_static(param_type) -> bool:
        """Static version of _is_list_of_enums for use in convert_string_to_type."""
        return ParameterTypeUtils.is_list_of_enums(param_type)

    @staticmethod
    def _get_enum_from_list_static(param_type):
        """Static version of _get_enum_from_list for use in convert_string_to_type."""
        return ParameterTypeUtils.get_enum_from_list_type(param_type)

    def get_current_values(self) -> Dict[str, Any]:
        """
        Get current parameter values preserving lazy dataclass structure.

        This fixes the lazy default materialization override saving issue by ensuring
        that lazy dataclasses maintain their structure when values are retrieved.
        """
        return {
            param_name: self._preserve_lazy_structure_if_needed(param_name, param_value)
            for param_name, param_value in self.parameters.items()
        }

    def _preserve_lazy_structure_if_needed(self, param_name: str, param_value: Any) -> Any:
        """Preserve lazy dataclass structure for dataclass parameters in lazy contexts."""
        # Early returns for simple cases
        if param_value is None or ParameterTypeUtils.is_lazy_dataclass(param_value):
            return param_value
        if self.is_global_config_editing:
            return param_value

        # Check if this should be converted to lazy dataclass
        param_type = self._get_dataclass_type_for_param(param_name)
        if param_type is None:
            return param_value

        return self._convert_to_lazy_dataclass(param_value, param_type)

    def _get_dataclass_type_for_param(self, param_name: str) -> Optional[type]:
        """Get the dataclass type for a parameter, handling Optional types."""
        return ParameterTypeUtils.get_dataclass_type_for_param(param_name, self.parameter_types)

    def _convert_to_lazy_dataclass(self, param_value: Any, param_type: type) -> Any:
        """Convert concrete dataclass or dict to lazy dataclass preserving field values."""
        from openhcs.core.lazy_config import LazyDataclassFactory

        field_path = self._get_field_path_for_nested_type(param_type)
        lazy_type = LazyDataclassFactory.make_lazy_thread_local(
            base_class=param_type,
            field_path=field_path,
            lazy_class_name=f"Mixed{param_type.__name__}"
        )

        # Extract field values based on input type
        if ParameterTypeUtils.has_dataclass_fields(param_value):
            # Concrete dataclass - extract field values
            import dataclasses
            field_values = {
                field.name: getattr(param_value, field.name)
                for field in dataclasses.fields(param_value)
            }
        elif isinstance(param_value, dict):
            # Dict from nested form manager
            field_values = param_value
        else:
            # Fallback: return value as-is
            return param_value

        return lazy_type(**field_values)

    def _get_parameter_info(self, param_name: str):
        """Get parameter info for help functionality."""
        return self.parameter_info.get(param_name)

    # Old placeholder methods removed - now using centralized abstraction layer

    @staticmethod
    def convert_string_to_type(string_value: str, param_type: type, strict: bool = False) -> Any:
        """
        Convert string input to expected type using existing type conversion logic.

        Args:
            string_value: The string value from user input
            param_type: The expected type from function signature
            strict: If True, raise errors on conversion failure. If False, return None.

        Returns:
            Converted value of the expected type

        Raises:
            ValueError: If strict=True and conversion fails with specific error message
        """
        # Handle empty/None values - let compiler validate if required
        if string_value == CONSTANTS.EMPTY_STRING or string_value is None:
            return None

        try:
            # Handle Union types (like Optional[List[float]] which is Union[List[float], None])
            origin = get_origin(param_type)
            if origin is Union:
                # Try each type in the Union until one works
                union_args = get_args(param_type)
                last_error = None

                for union_type in union_args:
                    # Skip NoneType - we handle None separately
                    if union_type is type(None):
                        continue

                    try:
                        # Recursively try to convert to this union member type
                        return ParameterFormManager.convert_string_to_type(string_value, union_type, strict=True)
                    except (ValueError, TypeError, SyntaxError) as e:
                        last_error = e
                        continue

                # If no union type worked, raise the last error
                if last_error:
                    raise last_error
                else:
                    raise ValueError(CONSTANTS.NO_VALID_CONVERSION_MSG.format(param_type))

            # Use existing type conversion logic from update_parameter
            elif ParameterTypeUtils.is_enum_type(param_type):
                return param_type(string_value)  # Convert string → enum
            elif ParameterTypeUtils.is_list_of_enums(param_type):
                # Handle List[Enum] types (like List[VariableComponents])
                enum_type = ParameterTypeUtils.get_enum_from_list_type(param_type)
                if enum_type:
                    # Convert string value to enum, then wrap in list
                    enum_value = enum_type(string_value)
                    return [enum_value]
            elif param_type == float:
                return float(string_value)
            elif param_type == int:
                return int(string_value)
            elif param_type == bool:
                # Convert string → bool
                return ParameterTypeUtils.convert_string_to_bool(string_value)
            elif param_type in (list, tuple, dict):
                # Use ast.literal_eval for complex types like [1,2,3], (1,2), {"a":1}
                return ast.literal_eval(string_value)
            elif get_origin(param_type) in (list, tuple, dict):
                # Handle generic types like List[float], Tuple[int, int], Dict[str, int]
                # Use ast.literal_eval since List("[1]") doesn't work, but ast.literal_eval("[1]") does
                return ast.literal_eval(string_value)
            elif param_type is Any:
                # No type hints available - try ast.literal_eval for Python literals
                try:
                    return ast.literal_eval(string_value)
                except (ValueError, SyntaxError):
                    # If literal_eval fails, return as string
                    return string_value
            else:
                # For everything else, try calling the type directly
                return param_type(string_value)

        except (ValueError, TypeError, SyntaxError) as e:
            if strict:
                # Provide specific error message for user
                raise ValueError(f"Cannot convert '{string_value}' to {param_type.__name__}: {e}")
            else:
                # Silent failure - return None (existing behavior)
                return None

    def _create_nested_managers_for_testing(self):
        """Create nested managers without building widgets (for testing)."""
        for param_name, param_type in self.parameter_types.items():
            current_value = self.parameters[param_name]

            # Handle nested dataclasses
            if dataclasses.is_dataclass(param_type):
                # Analyze nested dataclass
                nested_param_info = SignatureAnalyzer.analyze(param_type)

                # Get current values from nested dataclass instance
                nested_parameters = {}
                nested_parameter_types = {}

                for nested_name, nested_info in nested_param_info.items():
                    if current_value:
                        # For lazy dataclasses, preserve None values for placeholder behavior
                        if ParameterTypeUtils.has_resolve_field_value(current_value):
                            nested_current_value = object.__getattribute__(current_value, nested_name) if hasattr(current_value, nested_name) else nested_info.default_value
                        else:
                            nested_current_value = getattr(current_value, nested_name, nested_info.default_value)
                    else:
                        nested_current_value = nested_info.default_value
                    nested_parameters[nested_name] = nested_current_value
                    nested_parameter_types[nested_name] = nested_info.param_type

                # Create nested form manager with hierarchical underscore notation and parameter info
                nested_field_id = f"{self.field_id}_{param_name}"
                nested_form_manager = ParameterFormManager(nested_parameters, nested_parameter_types, nested_field_id, nested_param_info)

                # Store reference to nested form manager for updates
                if not hasattr(self, 'nested_managers'):
                    self.nested_managers = {}
                self.nested_managers[param_name] = nested_form_manager
