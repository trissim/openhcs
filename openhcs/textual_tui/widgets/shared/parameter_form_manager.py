# File: openhcs/textual_tui/widgets/shared/parameter_form_manager.py

import dataclasses
import ast
from enum import Enum
from typing import Any, Dict, get_origin, get_args, Union, Optional
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

class ParameterFormManager:
    """Mathematical: (parameters, types, field_id) → parameter form"""

    def __init__(self, parameters: Dict[str, Any], parameter_types: Dict[str, type], field_id: str, parameter_info: Dict = None):
        # Initialize simplified abstraction layer
        self.form_abstraction = ParameterFormAbstraction(
            parameters, parameter_types, field_id, create_textual_registry(), parameter_info
        )

        # Maintain backward compatibility
        self.parameters = parameters.copy()
        self.parameter_types = parameter_types
        self.field_id = field_id
        self.parameter_info = parameter_info or {}
    
    def build_form(self) -> ComposeResult:
        """Build parameter form - pure function with recursive dataclass support."""
        with Vertical() as form:
            form.styles.height = "auto"

            for param_name, param_type in self.parameter_types.items():
                current_value = self.parameters[param_name]

                # Handle Optional[dataclass] types with checkbox wrapper
                if self._is_optional_dataclass(param_type):
                    inner_dataclass_type = self._get_optional_inner_type(param_type)
                    yield from self._build_optional_dataclass_form(param_name, inner_dataclass_type, current_value)
                # Handle nested dataclasses recursively
                elif dataclasses.is_dataclass(param_type):
                    yield from self._build_nested_dataclass_form(param_name, param_type, current_value)
                else:
                    yield from self._build_regular_parameter_form(param_name, param_type, current_value)

    def _build_nested_dataclass_form(self, param_name: str, param_type: type, current_value: Any) -> ComposeResult:
        """Build form for nested dataclass parameter."""
        # Create collapsible widget (no ID - just structure)
        collapsible = TypedWidgetFactory.create_widget(param_type, current_value, None)

        # Analyze nested dataclass
        nested_param_info = SignatureAnalyzer.analyze(param_type)

        # Get current values from nested dataclass instance
        nested_parameters = {}
        nested_parameter_types = {}

        for nested_name, nested_info in nested_param_info.items():
            nested_current_value = getattr(current_value, nested_name, nested_info.default_value) if current_value else nested_info.default_value
            nested_parameters[nested_name] = nested_current_value
            nested_parameter_types[nested_name] = nested_info.param_type

        # Create nested form manager with hierarchical underscore notation and parameter info
        nested_field_id = f"{self.field_id}_{param_name}"
        nested_form_manager = ParameterFormManager(nested_parameters, nested_parameter_types, nested_field_id, nested_param_info)

        # Store the parent dataclass type for proper lazy resolution detection
        nested_form_manager._parent_dataclass_type = param_type

        # Store reference to nested form manager for updates
        if not hasattr(self, 'nested_managers'):
            self.nested_managers = {}
        self.nested_managers[param_name] = nested_form_manager

        # Build nested form and add to collapsible
        with collapsible:
            yield from nested_form_manager.build_form()

        yield collapsible

    def _is_optional_dataclass(self, param_type: type) -> bool:
        """Check if parameter type is Optional[dataclass]."""
        from typing import get_origin, get_args, Union
        if get_origin(param_type) is Union:
            args = get_args(param_type)
            if len(args) == 2 and type(None) in args:
                inner_type = next(arg for arg in args if arg is not type(None))
                return dataclasses.is_dataclass(inner_type)
        return False

    def _get_optional_inner_type(self, param_type: type) -> type:
        """Extract the inner type from Optional[T]."""
        from typing import get_origin, get_args, Union
        if get_origin(param_type) is Union:
            args = get_args(param_type)
            if len(args) == 2 and type(None) in args:
                return next(arg for arg in args if arg is not type(None))
        return param_type

    def _build_optional_dataclass_form(self, param_name: str, dataclass_type: type, current_value: Any) -> ComposeResult:
        """Build form for Optional[dataclass] parameter with checkbox toggle."""
        from textual.widgets import Checkbox

        # Checkbox
        checkbox_id = f"{self.field_id}_{param_name}_enabled"
        checkbox = Checkbox(
            value=current_value is not None,
            label=f"Enable {param_name.replace('_', ' ').title()}",
            id=checkbox_id,
            compact=True
        )
        yield checkbox

        # Collapsible dataclass widget
        collapsible = TypedWidgetFactory.create_widget(dataclass_type, current_value, None)
        collapsible.collapsed = (current_value is None)

        # Setup nested form
        nested_param_info = SignatureAnalyzer.analyze(dataclass_type)
        nested_parameters = {name: getattr(current_value, name, info.default_value) if current_value else info.default_value
                           for name, info in nested_param_info.items()}
        nested_parameter_types = {name: info.param_type for name, info in nested_param_info.items()}

        nested_form_manager = ParameterFormManager(
            nested_parameters, nested_parameter_types, f"{self.field_id}_{param_name}", nested_param_info
        )

        # Store the parent dataclass type for proper lazy resolution detection
        nested_form_manager._parent_dataclass_type = dataclass_type

        # Store references
        if not hasattr(self, 'nested_managers'):
            self.nested_managers = {}
        if not hasattr(self, 'optional_checkboxes'):
            self.optional_checkboxes = {}
        self.nested_managers[param_name] = nested_form_manager
        self.optional_checkboxes[param_name] = checkbox

        with collapsible:
            yield from nested_form_manager.build_form()
        yield collapsible

    def _build_regular_parameter_form(self, param_name: str, param_type: type, current_value: Any) -> ComposeResult:
        """Build form for regular (non-dataclass) parameter."""
        # Check if this field has different values across orchestrators
        config_analysis = getattr(self, 'config_analysis', {})
        field_analysis = config_analysis.get(param_name, {})

        # Create widget using hierarchical underscore notation
        widget_id = f"{self.field_id}_{param_name}"

        # Handle different values or create normal widget
        if field_analysis.get('type') == 'different':
            default_value = field_analysis.get('default')
            input_widget = create_different_values_widget(param_name, param_type, default_value, widget_id)
        else:
            # Use registry for widget creation and apply placeholder
            widget_value = current_value.value if hasattr(current_value, 'value') else current_value
            input_widget = self.form_abstraction.create_widget_for_parameter(param_name, param_type, widget_value)
            apply_lazy_default_placeholder(input_widget, param_name, current_value, self.parameter_types, 'textual')

        # Get parameter info for help functionality
        param_info = self._get_parameter_info(param_name)
        param_description = param_info.description if param_info else None

        # 3-column layout: label + input + reset
        with Horizontal() as row:
            row.styles.height = "auto"

            # Parameter label with help (auto width - sizes to content)
            # Always use clickable label with help - provide default description if none exists
            description = param_description or f"Parameter: {param_name.replace('_', ' ')}"
            label = ClickableParameterLabel(
                param_name,
                description,
                param_type,
                classes="param-label clickable"
            )

            label.styles.width = "auto"
            label.styles.text_align = "left"
            label.styles.height = "1"
            yield label

            # Input widget (flexible width, left aligned, with left margin for spacing)
            input_widget.styles.width = "1fr"
            input_widget.styles.text_align = "left"
            input_widget.styles.margin = (0, 0, 0, 1)  # top, right, bottom, left margin
            yield input_widget

            # Reset button (auto width)
            reset_btn = Button("Reset", id=f"reset_{widget_id}", compact=True)
            reset_btn.styles.width = "auto"
            yield reset_btn
    
    def update_parameter(self, param_name: str, value: Any):
        """Update parameter value with centralized enum conversion and nested dataclass support."""
        # Parse hierarchical parameter name (e.g., "path_planning_global_output_folder")
        # Split and check if this is a nested parameter
        parts = param_name.split('_')
        if len(parts) >= 2:  # nested_field format
            # Try to find nested manager by checking all possible prefixes
            for i in range(1, len(parts)):
                potential_nested = '_'.join(parts[:i])
                if potential_nested in self.parameters and hasattr(self, 'nested_managers') and potential_nested in self.nested_managers:
                    # Reconstruct the nested field name from remaining parts
                    nested_field = '_'.join(parts[i:])

                    # Update nested form manager
                    self.nested_managers[potential_nested].update_parameter(nested_field, value)

                    # Rebuild nested dataclass instance
                    nested_values = self.nested_managers[potential_nested].get_current_values()
                    nested_type = self.parameter_types[potential_nested]

                    # Resolve Union types (like Optional[DataClass]) to the actual dataclass type
                    if self._is_optional_dataclass(nested_type):
                        nested_type = self._get_optional_inner_type(nested_type)

                    self.parameters[potential_nested] = nested_type(**nested_values)
                    return

        # Handle regular parameters (direct match)
        if param_name in self.parameters:
            # Convert string back to proper type (comprehensive conversion)
            if param_name in self.parameter_types:
                param_type = self.parameter_types[param_name]
                if hasattr(param_type, '__bases__') and Enum in param_type.__bases__:
                    value = param_type(value)  # Convert string → enum
                elif self._is_list_of_enums(param_type):
                    # Handle List[Enum] types (like List[VariableComponents])
                    enum_type = self._get_enum_from_list(param_type)
                    if enum_type:
                        # Convert string value to enum, then wrap in list
                        enum_value = enum_type(value)
                        value = [enum_value]
                elif param_type == float:
                    # Convert string → float, handle empty based on parameter requirements
                    try:
                        if value == "":
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
                        if value == "":
                            # For empty values, convert to None (safer than 0)
                            value = None
                        else:
                            value = int(value)
                    except (ValueError, TypeError):
                        value = None  # Use None instead of 0 for failed conversions
                elif param_type == bool:
                    # Convert string → bool
                    if isinstance(value, str):
                        value = value.lower() in ('true', '1', 'yes', 'on')
                # Add more type conversions as needed

            self.parameters[param_name] = value
    
    def reset_parameter(self, param_name: str, default_value: Any):
        """Reset parameter to default value with nested dataclass support."""
        # Parse hierarchical parameter name for nested parameters
        parts = param_name.split('_')
        if len(parts) >= 2:  # nested_field format
            # Try to find nested manager by checking all possible prefixes
            for i in range(1, len(parts)):
                potential_nested = '_'.join(parts[:i])
                if potential_nested in self.parameters and hasattr(self, 'nested_managers') and potential_nested in self.nested_managers:
                    # Reconstruct the nested field name
                    nested_field = '_'.join(parts[i:])

                    # Get default value for nested field
                    nested_type = self.parameter_types[potential_nested]
                    nested_param_info = SignatureAnalyzer.analyze(nested_type)
                    nested_default = nested_param_info[nested_field].default_value

                    # Reset in nested form manager
                    self.nested_managers[potential_nested].reset_parameter(nested_field, nested_default)

                    # Rebuild nested dataclass instance
                    nested_values = self.nested_managers[potential_nested].get_current_values()

                    # Resolve Union types (like Optional[DataClass]) to the actual dataclass type
                    if self._is_optional_dataclass(nested_type):
                        nested_type = self._get_optional_inner_type(nested_type)

                    self.parameters[potential_nested] = nested_type(**nested_values)
                    return

        # Handle regular parameters
        if param_name in self.parameters:
            self.parameters[param_name] = default_value

            # Handle special reset behavior for DifferentValuesInput widgets
            self._handle_different_values_reset(param_name)

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

        if field_analysis.get('type') == 'different':
            # For different values fields, reset means go back to "DIFFERENT VALUES" state
            # We need to find the widget and call its reset method
            widget_id = f"{self.field_id}_{param_name}"

            # This will be handled by the screen/container that manages the widgets
            # The widget itself will handle the reset via its reset_to_different() method
            # We just need to ensure the parameter value reflects the "different" state
            pass  # Widget-level reset will be handled by the containing screen

    def reset_all_parameters(self, defaults: Dict[str, Any]):
        """Reset all parameters to defaults with nested dataclass support."""
        for param_name, default_value in defaults.items():
            if param_name in self.parameters:
                # Handle nested dataclasses
                if dataclasses.is_dataclass(self.parameter_types.get(param_name)):
                    if hasattr(self, 'nested_managers') and param_name in self.nested_managers:
                        # Reset all nested parameters
                        nested_type = self.parameter_types[param_name]
                        nested_param_info = SignatureAnalyzer.analyze(nested_type)
                        nested_defaults = {name: info.default_value for name, info in nested_param_info.items()}
                        self.nested_managers[param_name].reset_all_parameters(nested_defaults)

                        # Rebuild nested dataclass instance
                        nested_values = self.nested_managers[param_name].get_current_values()

                        # Resolve Union types (like Optional[DataClass]) to the actual dataclass type
                        if self._is_optional_dataclass(nested_type):
                            nested_type = self._get_optional_inner_type(nested_type)

                        self.parameters[param_name] = nested_type(**nested_values)
                    else:
                        self.parameters[param_name] = default_value
                else:
                    self.parameters[param_name] = default_value
    
    def _is_list_of_enums(self, param_type) -> bool:
        """Check if parameter type is List[Enum]."""
        try:
            # Check if it's a generic type (like List[Something])
            origin = get_origin(param_type)
            if origin is list:
                # Get the type arguments (e.g., VariableComponents from List[VariableComponents])
                args = get_args(param_type)
                if args and len(args) > 0:
                    inner_type = args[0]
                    # Check if the inner type is an enum
                    return hasattr(inner_type, '__bases__') and Enum in inner_type.__bases__
            return False
        except Exception:
            return False

    def _get_enum_from_list(self, param_type):
        """Extract enum type from List[Enum] type."""
        return self._get_enum_from_list_static(param_type)

    @staticmethod
    def _is_list_of_enums_static(param_type) -> bool:
        """Static version of _is_list_of_enums for use in convert_string_to_type."""
        try:
            # Check if it's a generic type (like List[Something])
            origin = get_origin(param_type)
            if origin is list:
                # Get the type arguments (e.g., VariableComponents from List[VariableComponents])
                args = get_args(param_type)
                if args and len(args) > 0:
                    inner_type = args[0]
                    # Check if the inner type is an enum
                    return hasattr(inner_type, '__bases__') and Enum in inner_type.__bases__
            return False
        except Exception:
            return False

    @staticmethod
    def _get_enum_from_list_static(param_type):
        """Static version of _get_enum_from_list for use in convert_string_to_type."""
        try:
            args = get_args(param_type)
            if args and len(args) > 0:
                return args[0]  # Return the enum type (e.g., VariableComponents)
            return None
        except Exception:
            return None

    def get_current_values(self) -> Dict[str, Any]:
        """Get current parameter values."""
        return self.parameters.copy()

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
        if string_value == "" or string_value is None:
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
                    raise ValueError(f"No valid conversion found for Union type {param_type}")

            # Use existing type conversion logic from update_parameter
            elif hasattr(param_type, '__bases__') and Enum in param_type.__bases__:
                return param_type(string_value)  # Convert string → enum
            elif ParameterFormManager._is_list_of_enums_static(param_type):
                # Handle List[Enum] types (like List[VariableComponents])
                enum_type = ParameterFormManager._get_enum_from_list_static(param_type)
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
                return string_value.lower() in ('true', '1', 'yes', 'on')
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
                    nested_current_value = getattr(current_value, nested_name, nested_info.default_value) if current_value else nested_info.default_value
                    nested_parameters[nested_name] = nested_current_value
                    nested_parameter_types[nested_name] = nested_info.param_type

                # Create nested form manager with hierarchical underscore notation and parameter info
                nested_field_id = f"{self.field_id}_{param_name}"
                nested_form_manager = ParameterFormManager(nested_parameters, nested_parameter_types, nested_field_id, nested_param_info)

                # Store reference to nested form manager for updates
                if not hasattr(self, 'nested_managers'):
                    self.nested_managers = {}
                self.nested_managers[param_name] = nested_form_manager
