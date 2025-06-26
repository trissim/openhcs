"""Universal wrapper for handling 'DIFFERENT VALUES' state across all widget types."""

from typing import Any, Optional, Type
from textual.widget import Widget
from textual.events import Click
from textual import on


class DifferentValuesWrapper:
    """
    Universal wrapper that adds 'DIFFERENT VALUES' functionality to ANY Textual widget.
    
    Uses the common disabled interface that all widgets inherit from Widget base class.
    Provides consistent behavior: disabled + hover feedback + click-to-default.
    """
    
    def __init__(
        self, 
        widget: Widget,
        default_value: Any, 
        field_name: str = "",
        on_value_set_callback: Optional[callable] = None
    ):
        """Initialize the wrapper around any widget.
        
        Args:
            widget: Any Textual widget (Input, Checkbox, RadioSet, etc.)
            default_value: The default value to use when clicked
            field_name: Name of the field (for debugging/logging)
            on_value_set_callback: Optional callback when value is set from different state
        """
        self.widget = widget
        self.default_value = default_value
        self.field_name = field_name
        self.on_value_set_callback = on_value_set_callback
        self.is_different_state = True
        
        # Set initial disabled state and styling
        self._apply_different_state()
        
        # Hook into widget's click event
        self._setup_click_handler()
    
    def _apply_different_state(self) -> None:
        """Apply 'DIFFERENT VALUES' visual state to the widget."""
        # DON'T disable the widget - disabled widgets don't receive click events!
        # Instead, use CSS styling to make it appear disabled
        self.widget.disabled = False

        # Add CSS class for styling (will make it look disabled)
        self.widget.add_class("different-values-widget")

        # Set tooltip to explain the state
        self.widget.tooltip = f"DIFFERENT VALUES - Click to set to default: {self.default_value}"
    
    def _setup_click_handler(self) -> None:
        """Set up click handler for the wrapped widget using Textual's event system."""
        # Store original click handler if it exists
        self._original_on_click = getattr(self.widget, 'on_click', None)

        # Use Textual's proper event handling approach
        # We'll add a message handler to the widget
        original_on_click = getattr(self.widget, 'on_click', None)

        async def wrapped_on_click(event: Click) -> None:
            # Handle our different values logic first
            self._handle_click(event)
            # Then call original handler if it exists
            if original_on_click:
                if hasattr(original_on_click, '__call__'):
                    if hasattr(original_on_click, '__code__') and original_on_click.__code__.co_flags & 0x80:
                        # It's an async function
                        await original_on_click(event)
                    else:
                        # It's a sync function
                        original_on_click(event)

        # Replace the widget's on_click method
        self.widget.on_click = wrapped_on_click
    
    def _handle_click(self, event: Click) -> None:
        """Handle click event - set to default value if in different state."""
        if self.is_different_state:
            # Set to default value using widget-specific method
            self._set_widget_value(self.default_value)
            
            # Update state
            self.is_different_state = False
            
            # Remove different state styling
            self._remove_different_state()
            
            # Call callback if provided
            if self.on_value_set_callback:
                self.on_value_set_callback(self.field_name, self.default_value)
        else:
            # If not in different state, call original handler
            if self._original_on_click:
                self._original_on_click(event)
    
    def _set_widget_value(self, value: Any) -> None:
        """Set value on the wrapped widget using appropriate method."""
        widget_type = type(self.widget).__name__

        if hasattr(self.widget, 'value'):
            # Most widgets (Input, Checkbox, etc.)
            if widget_type == 'Checkbox':
                self.widget.value = bool(value)
            else:
                self.widget.value = str(value) if value is not None else ""
        elif hasattr(self.widget, 'pressed_button') or widget_type in ['RadioSet', 'EnumRadioSet']:
            # RadioSet widgets (use pressed_button, not pressed)
            target_value = value.value if hasattr(value, 'value') else str(value)

            # Find the radio button with matching value and set it
            for button in self.widget.query('RadioButton'):
                if hasattr(button, 'label') and str(button.label).upper() == target_value.upper():
                    button.value = True
                    break
                elif hasattr(button, 'id') and target_value.lower() in str(button.id).lower():
                    button.value = True
                    break
        else:
            # Fallback for custom widgets
            if hasattr(self.widget, 'set_value'):
                self.widget.set_value(value)
    
    def _remove_different_state(self) -> None:
        """Remove 'DIFFERENT VALUES' state and restore normal widget."""
        # Re-enable the widget
        self.widget.disabled = False
        
        # Remove styling
        self.widget.remove_class("different-values-widget")
        self.widget.add_class("modified-from-different")
        
        # Update tooltip
        self.widget.tooltip = f"Set to default value: {self.default_value}"
    
    def reset_to_different(self) -> None:
        """Reset the widget back to 'DIFFERENT VALUES' state."""
        self.is_different_state = True
        self._apply_different_state()

        # Clear widget value
        widget_type = type(self.widget).__name__
        if hasattr(self.widget, 'value'):
            if widget_type == 'Checkbox':
                self.widget.value = False
            else:
                self.widget.value = ""
        elif hasattr(self.widget, 'pressed_button') or widget_type in ['RadioSet', 'EnumRadioSet']:
            # Clear RadioSet selection by setting all buttons to False
            for button in self.widget.query('RadioButton'):
                button.value = False
    
    def set_value(self, value: Any) -> None:
        """Set a specific value (not from click-to-default)."""
        self._set_widget_value(value)
        self.is_different_state = False
        
        # Update styling
        self.widget.disabled = False
        self.widget.remove_class("different-values-widget")
        self.widget.add_class("user-modified")
        self.widget.tooltip = f"User modified value: {value}"
    
    @property
    def current_state(self) -> str:
        """Get the current state of the wrapper."""
        if self.is_different_state:
            return "different"
        elif self._is_default_value():
            return "default"
        else:
            return "modified"
    
    def _is_default_value(self) -> bool:
        """Check if current widget value matches default."""
        widget_type = type(self.widget).__name__

        if hasattr(self.widget, 'value'):
            current = self.widget.value
            if widget_type == 'Checkbox':
                return bool(current) == bool(self.default_value)
            else:
                return str(current) == str(self.default_value)
        elif hasattr(self.widget, 'pressed_button') or widget_type in ['RadioSet', 'EnumRadioSet']:
            # Check RadioSet selection
            pressed_button = self.widget.pressed_button
            if pressed_button is None:
                return False

            current_label = str(pressed_button.label).upper()
            default_str = str(self.default_value.value if hasattr(self.default_value, 'value') else self.default_value).upper()
            return current_label == default_str
        return False


def create_different_values_widget(
    widget_type: Type[Widget],
    default_value: Any,
    field_name: str = "",
    widget_kwargs: Optional[dict] = None,
    on_value_set_callback: Optional[callable] = None
) -> Widget:
    """
    Factory function to create any widget with 'DIFFERENT VALUES' functionality.
    
    Args:
        widget_type: The widget class to create (Input, Checkbox, RadioSet, etc.)
        default_value: Default value for click-to-set behavior
        field_name: Field name for debugging
        widget_kwargs: Additional kwargs for widget creation
        on_value_set_callback: Callback when value is set from different state
        
    Returns:
        Widget instance with DifferentValuesWrapper attached
    """
    widget_kwargs = widget_kwargs or {}
    
    # Create the widget
    widget = widget_type(**widget_kwargs)
    
    # Wrap it with different values functionality
    wrapper = DifferentValuesWrapper(
        widget=widget,
        default_value=default_value,
        field_name=field_name,
        on_value_set_callback=on_value_set_callback
    )
    
    # Attach wrapper to widget for later access
    widget._different_values_wrapper = wrapper
    
    return widget
