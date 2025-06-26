"""Special input widget for handling different values across multiple configurations."""

from typing import Any, Optional
from textual.widgets import Input
from textual.events import Click


class DifferentValuesInput(Input):
    """Special input widget that shows 'DIFFERENT VALUES' and handles click-to-default behavior.
    
    This widget is used in multi-orchestrator configuration editing when a field has
    different values across the selected orchestrators. It displays "DIFFERENT VALUES"
    as a placeholder and allows the user to click to set it to the default value.
    """
    
    def __init__(
        self,
        default_value: Any,
        field_name: str = "",
        **kwargs
    ):
        """Initialize the DifferentValuesInput widget.

        Args:
            default_value: The default value to use when clicked
            field_name: Name of the field (for debugging/logging)
            **kwargs: Additional arguments passed to Input
        """
        # Set "DIFFERENT" as placeholder, empty value
        kwargs.setdefault("placeholder", "DIFFERENT")
        kwargs.setdefault("value", "")

        super().__init__(**kwargs)

        self.default_value = default_value
        self.field_name = field_name
        self.is_different_state = True

        # No special styling - looks like normal input
    
    def on_click(self, event: Click) -> None:
        """Handle click event - set to default value if in different state."""
        if self.is_different_state:
            # Convert default value to string for display
            self.value = str(self.default_value) if self.default_value is not None else ""
            self.is_different_state = False

            # Clear placeholder since we now have a value
            self.placeholder = ""

            # Focus the input for immediate editing
            self.focus()
    
    def reset_to_different(self) -> None:
        """Reset the widget back to 'DIFFERENT' placeholder state."""
        self.value = ""
        self.placeholder = "DIFFERENT"
        self.is_different_state = True
    
    def set_value(self, value: Any) -> None:
        """Set a specific value (not from click-to-default)."""
        self.value = str(value) if value is not None else ""
        self.is_different_state = False
        self.placeholder = ""
    
    @property
    def current_state(self) -> str:
        """Get the current state of the widget."""
        if self.is_different_state:
            return "different"
        elif self.value == str(self.default_value):
            return "default"
        else:
            return "modified"


class DifferentValuesInputCSS:
    """CSS styles for DifferentValuesInput widget."""
    
    CSS = """
    .different-values-input {
        border: dashed $warning;
        color: $warning;
        background: $surface;
    }
    
    .different-values-input:hover {
        border: solid $warning;
        background: $warning 20%;
    }
    
    .different-values-input:focus {
        border: solid $warning;
        background: $warning 30%;
    }
    
    .modified-from-different {
        border: solid $success;
        color: $text;
        background: $success 10%;
    }
    
    .user-modified {
        border: solid $primary;
        color: $text;
        background: $primary 10%;
    }
    """
