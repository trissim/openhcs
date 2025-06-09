# File: openhcs/textual_tui/widgets/shared/parameter_form_manager.py

import dataclasses
from enum import Enum
from typing import Any, Dict
from textual.containers import Vertical, Horizontal
from textual.widgets import Static, Button, Collapsible
from textual.app import ComposeResult

from .typed_widget_factory import TypedWidgetFactory
from .signature_analyzer import SignatureAnalyzer

class ParameterFormManager:
    """Mathematical: (parameters, types, field_id) → parameter form"""
    
    def __init__(self, parameters: Dict[str, Any], parameter_types: Dict[str, type], field_id: str):
        self.parameters = parameters.copy()  # Current values
        self.parameter_types = parameter_types  # Types (immutable)
        self.field_id = field_id
    
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

        # 3-column layout: label + input + reset
        with Horizontal() as row:
            row.styles.height = "auto"

            # Parameter label (fixed width)
            label = Static(f"{param_name}:", classes="param-label")
            label.styles.width = "20"
            label.styles.text_align = "left"
            label.styles.height = "1"
            yield label

            # Input widget (flexible width)
            input_widget.styles.width = "1fr"
            yield input_widget

            # Reset button (auto width)
            reset_btn = Button("Reset", id=f"reset_{widget_id}", compact=True)
            reset_btn.styles.width = "auto"
            yield reset_btn
    
    def update_parameter(self, param_name: str, value: Any):
        """Update parameter value with centralized enum conversion and nested dataclass support."""
        # Parse hierarchical parameter name (e.g., "config_path_planning_output_dir_suffix")
        # Split and check if this is a nested parameter
        parts = param_name.split('_')
        if len(parts) >= 3:  # config_nested_field format
            # Try to find nested manager by checking if parts[1] is a nested parameter
            potential_nested = parts[1]
            if potential_nested in self.parameters and hasattr(self, 'nested_managers') and potential_nested in self.nested_managers:
                # Reconstruct the nested field name from remaining parts
                nested_field = '_'.join(parts[2:])

                # Update nested form manager
                self.nested_managers[potential_nested].update_parameter(nested_field, value)

                # Rebuild nested dataclass instance
                nested_values = self.nested_managers[potential_nested].get_current_values()
                nested_type = self.parameter_types[potential_nested]
                self.parameters[potential_nested] = nested_type(**nested_values)
                return

        # Handle regular parameters (direct match)
        if param_name in self.parameters:
            # Convert string back to enum if needed (centralized conversion)
            if param_name in self.parameter_types:
                param_type = self.parameter_types[param_name]
                if hasattr(param_type, '__bases__') and Enum in param_type.__bases__:
                    value = param_type(value)  # Convert string → enum

            self.parameters[param_name] = value
    
    def reset_parameter(self, param_name: str, default_value: Any):
        """Reset parameter to default value with nested dataclass support."""
        # Parse hierarchical parameter name for nested parameters
        parts = param_name.split('_')
        if len(parts) >= 3:  # config_nested_field format
            potential_nested = parts[1]
            if potential_nested in self.parameters and hasattr(self, 'nested_managers') and potential_nested in self.nested_managers:
                # Reconstruct the nested field name
                nested_field = '_'.join(parts[2:])

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
    
    def get_current_values(self) -> Dict[str, Any]:
        """Get current parameter values."""
        return self.parameters.copy()
