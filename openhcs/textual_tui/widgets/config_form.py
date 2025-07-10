"""Config form widget with reactive properties."""

from typing import List, Dict, Any, Callable, Optional
from textual.containers import Container, Vertical, ScrollableContainer
from textual.widgets import Static
from textual.app import ComposeResult
from textual.reactive import reactive


from .shared.parameter_form_manager import ParameterFormManager
from .shared.signature_analyzer import SignatureAnalyzer


class ConfigFormWidget(ScrollableContainer):
    """Reactive form widget for config editing."""

    field_values = reactive(dict, recompose=False)  # Prevent automatic recomposition during typing


    def __init__(self, dataclass_type: type, instance: Any = None, **kwargs):
        super().__init__(**kwargs)
        self.dataclass_type = dataclass_type
        self.instance = instance or dataclass_type()

        # Analyze dataclass using unified parameter analysis
        param_info = SignatureAnalyzer.analyze(dataclass_type)

        # Convert to form manager format
        parameters = {}
        parameter_types = {}
        param_defaults = {}

        for name, info in param_info.items():
            current_value = getattr(self.instance, name, info.default_value)
            parameters[name] = current_value
            parameter_types[name] = info.param_type
            param_defaults[name] = info.default_value

        # Create shared form manager with parameter info for help functionality
        self.form_manager = ParameterFormManager(parameters, parameter_types, "config", param_info)
        self.param_defaults = param_defaults

        # Initialize field values for reactive updates
        self.field_values = parameters.copy()

    @classmethod
    def from_dataclass(cls, dataclass_type: type, instance: Any = None, **kwargs):
        """Create ConfigFormWidget from dataclass type and instance."""
        return cls(dataclass_type, instance, **kwargs)
    
    def compose(self) -> ComposeResult:
        """Compose the config form using shared form manager."""
        try:
            # Use shared form manager to build form
            yield from self.form_manager.build_form()
        except Exception as e:
            yield Static(f"[red]Error building config form: {e}[/red]")

    def on_mount(self) -> None:
        """Called when the form is mounted - fix scroll position to top."""
        # Force scroll to top after mounting to prevent automatic scrolling to bottom
        self.call_after_refresh(self._fix_scroll_position)

    def _fix_scroll_position(self) -> None:
        """Fix scroll position to top of form."""
        try:
            # Force scroll to top (0, 0)
            self.scroll_to(0, 0, animate=False)
        except Exception:
            # If anything goes wrong, just continue
            pass


    
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
        if not self.form_manager:
            return

        # Handle both top-level and nested parameters
        if field_name in self.param_defaults:
            # Top-level parameter
            default_value = self.param_defaults[field_name]
            self.form_manager.reset_parameter(field_name, default_value)
            self._on_field_change(field_name, default_value)
            # Refresh the UI widget to show the reset value
            self._refresh_field_widget(field_name, default_value)
        else:
            # Check if it's a nested parameter (e.g., "path_planning_output_dir_suffix" or "nested_nested_bool")
            parts = field_name.split('_')
            if len(parts) >= 2:
                # Try to find the nested parameter by checking prefixes from longest to shortest
                # This handles cases like "nested_nested_bool" where "nested" is the parent
                for i in range(len(parts) - 1, 0, -1):  # Start from longest prefix
                    potential_nested = '_'.join(parts[:i])
                    if potential_nested in self.param_defaults:
                        # Found the nested parent, get the nested field name
                        nested_field = '_'.join(parts[i:])

                        # Get the nested default value from the default dataclass instance
                        nested_parent_default = self.param_defaults[potential_nested]
                        if hasattr(nested_parent_default, nested_field):
                            nested_default = getattr(nested_parent_default, nested_field)

                            # The form manager's reset_parameter method handles nested parameters automatically
                            # Just pass the full hierarchical name and it will find the right nested manager
                            self.form_manager.reset_parameter(field_name, nested_default)
                            self._on_field_change(field_name, nested_default)
                            # Refresh the UI widget to show the reset value
                            self._refresh_field_widget(field_name, nested_default)
                            return

            # If we get here, the parameter wasn't found
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Could not reset field {field_name}: not found in defaults")
    
    def _sync_field_values(self) -> None:
        """Sync internal field values to reactive property when safe to do so."""
        if hasattr(self, '_internal_field_values'):
            self.field_values = self._internal_field_values.copy()

    def _refresh_field_widget(self, field_name: str, value: Any) -> None:
        """Refresh a specific field widget to show the new value."""
        try:
            widget_id = f"config_{field_name}"

            # Try to find the widget
            try:
                widget = self.query_one(f"#{widget_id}")
            except Exception:
                # Widget not found with exact ID, try searching more broadly
                widgets = self.query(f"[id$='{field_name}']")  # Find widgets ending with field_name
                if widgets:
                    widget = widgets[0]
                else:
                    return  # Widget not found

            # Update widget based on type
            from textual.widgets import Input, Checkbox, RadioSet, Collapsible
            from .shared.enum_radio_set import EnumRadioSet

            if isinstance(widget, Input):
                # Input widget (int, float, str) - set value as string
                display_value = value.value if hasattr(value, 'value') else value
                widget.value = str(display_value) if display_value is not None else ""

            elif isinstance(widget, Checkbox):
                # Checkbox widget (bool) - set boolean value
                widget.value = bool(value)

            elif isinstance(widget, (RadioSet, EnumRadioSet)):
                # RadioSet/EnumRadioSet widget (Enum, List[Enum]) - find and press the correct radio button
                # Handle both enum values and string values
                if hasattr(value, 'value'):
                    # Enum value - use the .value attribute
                    target_value = value.value
                elif isinstance(value, list) and len(value) > 0:
                    # List[Enum] - get first item's value
                    first_item = value[0]
                    target_value = first_item.value if hasattr(first_item, 'value') else str(first_item)
                else:
                    # String value or other
                    target_value = str(value)

                # Find and press the correct radio button
                target_id = f"enum_{target_value}"
                for radio in widget.query("RadioButton"):
                    if radio.id == target_id:
                        radio.value = True
                        break
                    else:
                        # Unpress other radio buttons
                        radio.value = False

            elif isinstance(widget, Collapsible):
                # Collapsible widget (nested dataclass) - cannot be reset directly
                # The nested parameters are handled by their own reset buttons
                pass

            elif hasattr(widget, 'value'):
                # Generic widget with value attribute - fallback
                display_value = value.value if hasattr(value, 'value') else value
                widget.value = str(display_value) if display_value is not None else ""

        except Exception as e:
            # Widget not found or update failed - this is expected for some field types
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Could not refresh widget for field {field_name}: {e}")

    def get_config_values(self) -> Dict[str, Any]:
        """Get current config values from form manager."""
        if self.form_manager:
            return self.form_manager.get_current_values()
        else:
            # Fallback to internal field values if available, otherwise reactive field_values
            if hasattr(self, '_internal_field_values'):
                return self._internal_field_values.copy()
            return self.field_values.copy()