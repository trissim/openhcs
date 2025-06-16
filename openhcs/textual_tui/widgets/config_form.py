"""Config form widget with reactive properties."""

from typing import List, Dict, Any, Callable, Optional
from textual.containers import Container, Vertical, ScrollableContainer
from textual.widgets import Static
from textual.app import ComposeResult
from textual.reactive import reactive

from openhcs.textual_tui.services.config_reflection_service import FieldSpec
from .shared.parameter_form_manager import ParameterFormManager
from .shared.signature_analyzer import SignatureAnalyzer


class ConfigFormWidget(ScrollableContainer):
    """Reactive form widget for config editing."""
    
    field_values = reactive(dict, recompose=False)  # Prevent automatic recomposition during typing
    
    def __init__(self, field_specs: List[FieldSpec], **kwargs):
        super().__init__(**kwargs)
        self.field_specs = field_specs

        # Convert FieldSpec to shared component format
        parameters = {}
        parameter_types = {}
        param_defaults = {}

        for spec in field_specs:
            parameters[spec.name] = spec.current_value
            parameter_types[spec.name] = spec.actual_type
            param_defaults[spec.name] = spec.default_value

        # Create shared form manager
        self.form_manager = ParameterFormManager(parameters, parameter_types, "config")
        self.param_defaults = param_defaults

        # Initialize field values for reactive updates
        self.field_values = parameters.copy()
    
    def compose(self) -> ComposeResult:
        """Compose the config form using shared form manager."""
        try:
            # Use shared form manager to build form
            yield from self.form_manager.build_form()
        except Exception as e:
            yield Static(f"[red]Error building config form: {e}[/red]")
    
    def _on_field_change(self, field_name: str, value: Any) -> None:
        """Handle field value changes."""
        if self.form_manager:
            self.form_manager.update_parameter(field_name, value)
            # Update internal field values without triggering reactive update
            # This prevents recomposition and focus loss during typing
            if not hasattr(self, '_internal_field_values'):
                self._internal_field_values = self.field_values.copy()

            # For nested parameters like "path_planning_output_dir_suffix",
            # update the top-level "path_planning" parameter
            parts = field_name.split('_')
            if len(parts) >= 2:
                top_level_param = parts[0]
                if top_level_param in self.form_manager.parameters:
                    self._internal_field_values[top_level_param] = self.form_manager.parameters[top_level_param]
            else:
                # Regular parameter
                if field_name in self.form_manager.parameters:
                    self._internal_field_values[field_name] = self.form_manager.parameters[field_name]

    def on_input_changed(self, event) -> None:
        """Handle input changes from shared components."""
        if event.input.id.startswith("config_"):
            field_name = event.input.id.split("_", 1)[1]
            self._on_field_change(field_name, event.value)

    def on_checkbox_changed(self, event) -> None:
        """Handle checkbox changes from shared components."""
        if event.checkbox.id.startswith("config_"):
            field_name = event.checkbox.id.split("_", 1)[1]
            self._on_field_change(field_name, event.value)

    def on_radio_set_changed(self, event) -> None:
        """Handle RadioSet changes from shared components."""
        if event.radio_set.id.startswith("config_"):
            field_name = event.radio_set.id.split("_", 1)[1]
            if event.pressed and event.pressed.id:
                enum_value = event.pressed.id[5:]  # Remove "enum_" prefix
                self._on_field_change(field_name, enum_value)

    def on_button_pressed(self, event) -> None:
        """Handle reset button presses from shared components."""
        if event.button.id.startswith("reset_config_"):
            field_name = event.button.id.split("_", 2)[2]
            self._reset_field(field_name)

    def _reset_field(self, field_name: str) -> None:
        """Reset a field to its default value."""
        if field_name in self.param_defaults:
            default_value = self.param_defaults[field_name]
            if self.form_manager:
                self.form_manager.reset_parameter(field_name, default_value)
                self._on_field_change(field_name, default_value)
    
    def _sync_field_values(self) -> None:
        """Sync internal field values to reactive property when safe to do so."""
        if hasattr(self, '_internal_field_values'):
            self.field_values = self._internal_field_values.copy()

    def get_config_values(self) -> Dict[str, Any]:
        """Get current config values from form manager."""
        if self.form_manager:
            return self.form_manager.get_current_values()
        else:
            # Fallback to internal field values if available, otherwise reactive field_values
            if hasattr(self, '_internal_field_values'):
                return self._internal_field_values.copy()
            return self.field_values.copy()


