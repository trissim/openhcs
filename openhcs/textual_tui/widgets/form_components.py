"""Reusable form components for Textual TUI."""

from typing import Any, List, Dict, Optional, Callable
from textual.containers import Container, Vertical
from textual.widgets import Static, Input, Select, Checkbox
from textual.app import ComposeResult
from textual.reactive import reactive


class FormField(Container):
    """Individual form field with label and widget."""
    
    def __init__(self, label: str, widget: Any, **kwargs):
        super().__init__(**kwargs)
        self.label = label
        self.widget = widget
    
    def compose(self) -> ComposeResult:
        """Compose the form field."""
        yield Static(f"{self.label}:")
        yield self.widget


class FormContainer(Container):
    """Container for multiple form fields with validation."""
    
    field_values = reactive(dict)
    validation_errors = reactive(dict)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fields = []
        self.field_values = {}
        self.validation_errors = {}
    
    def add_field(self, name: str, label: str, widget: Any) -> None:
        """Add a field to the form."""
        field = FormField(label, widget)
        self.fields.append((name, field))
    
    def compose(self) -> ComposeResult:
        """Compose all form fields."""
        with Vertical():
            for name, field in self.fields:
                yield field
    
    def get_field_values(self) -> Dict[str, Any]:
        """Get current values of all fields."""
        return self.field_values.copy()
    
    def set_field_value(self, name: str, value: Any) -> None:
        """Set value for a specific field."""
        current_values = self.field_values.copy()
        current_values[name] = value
        self.field_values = current_values
    
    def validate_fields(self) -> bool:
        """Validate all fields. Override in subclasses."""
        return True
    
    def clear_validation_errors(self) -> None:
        """Clear all validation errors."""
        self.validation_errors = {}
    
    def set_validation_error(self, field_name: str, error: str) -> None:
        """Set validation error for a field."""
        current_errors = self.validation_errors.copy()
        current_errors[field_name] = error
        self.validation_errors = current_errors
