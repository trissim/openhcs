"""Simple checkbox widget for handling different values across multiple configurations."""

from typing import Any
from textual.widgets import Checkbox
from textual.events import Click


class DifferentValuesCheckbox(Checkbox):
    """Simple checkbox that shows empty state for different values.
    
    This widget appears as an empty square (no X, no label) when values differ
    across configurations. Click sets it to the default value.
    """
    
    def __init__(
        self, 
        default_value: bool, 
        field_name: str = "",
        **kwargs
    ):
        """Initialize the DifferentValuesCheckbox widget.
        
        Args:
            default_value: The default value to use when clicked
            field_name: Name of the field (for debugging/logging)
            **kwargs: Additional arguments passed to Checkbox
        """
        # Create empty checkbox with no label
        kwargs.setdefault("value", False)
        kwargs.setdefault("label", "")  # No label
        
        super().__init__(**kwargs)
        
        self.default_value = default_value
        self.field_name = field_name
        self.is_different_state = True

        # Add CSS class to appear disabled but still receive clicks
        self.add_class("different-values-disabled")

        # Set tooltip to explain the state
        self.tooltip = f"DIFFERENT VALUES - Click to set to default: {self.default_value}"
    
    def on_click(self, event: Click) -> None:
        """Message handler for Click events - set to default value if in different state."""
        if self.is_different_state:
            # Set to default value and remove disabled appearance
            self.value = bool(self.default_value)
            self.is_different_state = False
            self.remove_class("different-values-disabled")

            # Clear tooltip since we now have a value
            self.tooltip = ""

            # Focus the checkbox
            self.focus()

            # Prevent the event from bubbling up
            event.stop()
    
    def reset_to_different(self) -> None:
        """Reset the checkbox back to 'DIFFERENT VALUES' state."""
        self.value = False  # Empty state
        self.is_different_state = True

        # Add CSS class to appear disabled
        self.add_class("different-values-disabled")

        # Restore tooltip
        self.tooltip = f"DIFFERENT VALUES - Click to set to default: {self.default_value}"
    
    def set_value(self, value: bool) -> None:
        """Set a specific value (not from click-to-default)."""
        self.value = bool(value)
        self.is_different_state = False
        self.tooltip = ""
    
    @property
    def current_state(self) -> str:
        """Get the current state of the widget."""
        if self.is_different_state:
            return "different"
        elif self.value == bool(self.default_value):
            return "default"
        else:
            return "modified"
