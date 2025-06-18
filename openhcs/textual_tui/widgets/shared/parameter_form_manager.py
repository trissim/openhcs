# File: openhcs/textual_tui/widgets/shared/parameter_form_manager.py

import dataclasses
from enum import Enum
from typing import Any, Dict, get_origin, get_args
from textual.containers import Vertical, Horizontal
from textual.widgets import Static, Button, Collapsible
from textual.app import ComposeResult

from .typed_widget_factory import TypedWidgetFactory
from .signature_analyzer import SignatureAnalyzer
from .clickable_help_label import ClickableParameterLabel, HelpIndicator

class ParameterFormManager:
    """Mathematical: (parameters, types, field_id) → parameter form"""

    def __init__(self, parameters: Dict[str, Any], parameter_types: Dict[str, type], field_id: str, parameter_info: Dict = None):
        self.parameters = parameters.copy()  # Current values
        self.parameter_types = parameter_types  # Types (immutable)
        self.field_id = field_id
        self.parameter_info = parameter_info or {}  # Store parameter info for help
    
    def build_form(self) -> ComposeResult:
        """Build parameter form - pure function with recursive dataclass support."""
        with Vertical() as form:
            form.styles.height = "auto"

            for param_name, param_type in self.parameter_types.items():
                current_value = self.parameters[param_name]

                # Handle nested dataclasses recursively
                if dataclasses.is_dataclass(param_type):
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

        # Create nested form manager with hierarchical underscore notation
        nested_field_id = f"{self.field_id}_{param_name}"
        nested_form_manager = ParameterFormManager(nested_parameters, nested_parameter_types, nested_field_id)

        # Store reference to nested form manager for updates
        if not hasattr(self, 'nested_managers'):
            self.nested_managers = {}
        self.nested_managers[param_name] = nested_form_manager

        # Build nested form and add to collapsible
        with collapsible:
            yield from nested_form_manager.build_form()

        yield collapsible

    def _build_regular_parameter_form(self, param_name: str, param_type: type, current_value: Any) -> ComposeResult:
        """Build form for regular (non-dataclass) parameter."""
        # Convert enum to string for widget (centralized conversion)
        widget_value = current_value.value if hasattr(current_value, 'value') else current_value

        # Create widget using hierarchical underscore notation
        widget_id = f"{self.field_id}_{param_name}"
        input_widget = TypedWidgetFactory.create_widget(param_type, widget_value, widget_id)

        # Get parameter info for help functionality
        param_info = self._get_parameter_info(param_name)
        param_description = param_info.description if param_info else None

        # 3-column layout: label + input + reset
        with Horizontal() as row:
            row.styles.height = "auto"

            # Parameter label with help (auto width - sizes to content)
            if param_description:
                # Clickable label with help
                label = ClickableParameterLabel(
                    param_name,
                    param_description,
                    param_type,
                    classes="param-label clickable"
                )
            else:
                # Regular label
                label = Static(f"{param_name}:", classes="param-label")

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
                    self.parameters[potential_nested] = nested_type(**nested_values)
                    return

        # Handle regular parameters
        if param_name in self.parameters:
            self.parameters[param_name] = default_value

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

                # Create nested form manager with hierarchical underscore notation
                nested_field_id = f"{self.field_id}_{param_name}"
                nested_form_manager = ParameterFormManager(nested_parameters, nested_parameter_types, nested_field_id)

                # Store reference to nested form manager for updates
                if not hasattr(self, 'nested_managers'):
                    self.nested_managers = {}
                self.nested_managers[param_name] = nested_form_manager
