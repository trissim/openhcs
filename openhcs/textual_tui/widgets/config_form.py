"""Config form widget with reactive properties."""

from typing import List, Dict, Any, Callable
from enum import Enum
from pathlib import Path
from textual.containers import Container, Vertical
from textual.widgets import Static, Input, Select, Checkbox
from textual.app import ComposeResult
from textual.reactive import reactive

from openhcs.textual_tui.services.config_reflection_service import FieldSpec


class TextualWidgetFactory:
    """Creates Textual widgets for config field types."""
    
    @staticmethod
    def create_widget_for_spec(spec: FieldSpec, on_change: Callable) -> Any:
        """Create appropriate Textual widget for field spec."""
        field_id = f"field_{spec.name}"
        
        if spec.actual_type == bool:
            return Checkbox(
                label=spec.label,
                value=bool(spec.current_value) if spec.current_value is not None else False,
                id=field_id
            )
        elif spec.actual_type in (str, Path):
            return Input(
                value=str(spec.current_value) if spec.current_value is not None else "",
                id=field_id
            )
        elif spec.actual_type in (int, float):
            return Input(
                value=str(spec.current_value) if spec.current_value is not None else "",
                type="number",
                id=field_id
            )
        elif isinstance(spec.actual_type, type) and issubclass(spec.actual_type, Enum):
            options = [(member.name, member.value) for member in spec.actual_type]
            current_value = spec.current_value.value if spec.current_value else None
            return Select(
                options=options,
                value=current_value,
                id=field_id
            )
        else:
            # Fallback to string input
            return Input(
                value=str(spec.current_value) if spec.current_value is not None else "",
                id=field_id
            )


class ConfigFormWidget(Container):
    """Reactive form widget for config editing."""
    
    field_values = reactive(dict)  # Automatic UI updates
    
    def __init__(self, field_specs: List[FieldSpec], **kwargs):
        super().__init__(**kwargs)
        self.field_specs = field_specs
        self.widget_factory = TextualWidgetFactory()
        
        # Initialize field values
        initial_values = {}
        for spec in field_specs:
            initial_values[spec.name] = spec.current_value
        self.field_values = initial_values
    
    def compose(self) -> ComposeResult:
        """Compose the config form."""
        with Vertical():
            for spec in self.field_specs:
                yield Static(f"{spec.label}:")
                widget = self.widget_factory.create_widget_for_spec(spec, self._on_field_change)
                yield widget
    
    def _on_field_change(self, field_name: str, value: Any) -> None:
        """Handle field value changes."""
        current_values = self.field_values.copy()
        current_values[field_name] = value
        self.field_values = current_values
    
    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input widget changes."""
        if event.input.id and event.input.id.startswith("field_"):
            field_name = event.input.id.replace("field_", "")
            self._on_field_change(field_name, event.value)
    
    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select widget changes."""
        if event.select.id and event.select.id.startswith("field_"):
            field_name = event.select.id.replace("field_", "")
            self._on_field_change(field_name, event.value)
    
    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Handle checkbox widget changes."""
        if event.checkbox.id and event.checkbox.id.startswith("field_"):
            field_name = event.checkbox.id.replace("field_", "")
            self._on_field_change(field_name, event.value)
    
    def get_config_values(self) -> Dict[str, Any]:
        """Get current config values with type conversion."""
        values = {}
        for spec in self.field_specs:
            raw_value = self.field_values.get(spec.name, spec.current_value)
            converted_value = self._convert_value(raw_value, spec)
            values[spec.name] = converted_value
        return values
    
    def _convert_value(self, value: Any, spec: FieldSpec) -> Any:
        """Convert form value to appropriate type."""
        if value is None or value == "":
            return None if spec.is_optional else spec.default_value
        
        if spec.actual_type == bool:
            return bool(value)
        elif spec.actual_type == int:
            try:
                return int(value)
            except (ValueError, TypeError):
                return spec.default_value
        elif spec.actual_type == float:
            try:
                return float(value)
            except (ValueError, TypeError):
                return spec.default_value
        elif spec.actual_type == Path:
            return Path(str(value))
        elif isinstance(spec.actual_type, type) and issubclass(spec.actual_type, Enum):
            # Find enum member by value
            for member in spec.actual_type:
                if member.value == value:
                    return member
            return spec.default_value
        else:
            return str(value)
