"""Simple radio set widget for handling different values across multiple configurations."""

from typing import Any, Type
from enum import Enum
from textual.widgets import RadioSet, RadioButton
from textual.events import Click


class DifferentValuesRadioSet(RadioSet):
    """Simple radio set that shows no selection for different values.
    
    This widget appears with no radio button selected when values differ
    across configurations. Click sets it to the default value.
    """
    
    def __init__(
        self, 
        enum_type: Type[Enum],
        default_value: Enum, 
        field_name: str = "",
        **kwargs
    ):
        """Initialize the DifferentValuesRadioSet widget.
        
        Args:
            enum_type: The enum type for the radio options
            default_value: The default value to use when clicked
            field_name: Name of the field (for debugging/logging)
            **kwargs: Additional arguments passed to RadioSet
        """
        # Create radio buttons for each enum value
        from openhcs.ui.shared.ui_utils import format_enum_display

        radio_buttons = []
        for enum_value in enum_type:
            radio_buttons.append(format_enum_display(enum_value))
        
        super().__init__(*radio_buttons, **kwargs)
        
        self.enum_type = enum_type
        self.default_value = default_value
        self.field_name = field_name
        self.is_different_state = True
        
        # Clear any initial selection and add disabled appearance
        self._clear_selection()
        self.add_class("different-values-disabled")

        # Set tooltip to explain the state
        self.tooltip = f"DIFFERENT VALUES - Click to set to default: {self.default_value.value}"
    
    def _clear_selection(self) -> None:
        """Clear all radio button selections."""
        for button in self.query('RadioButton'):
            button.value = False


    
    def on_click(self, event: Click) -> None:
        """Message handler for Click events - set to default value if in different state."""
        if self.is_different_state:
            # Set to default value by finding and selecting the matching button
            default_label = self.default_value.value.upper()

            for button in self.query('RadioButton'):
                if hasattr(button, 'label') and str(button.label).upper() == default_label:
                    button.value = True
                    break

            self.is_different_state = False
            self.remove_class("different-values-disabled")

            # Clear tooltip since we now have a value
            self.tooltip = ""

            # Focus the radio set
            self.focus()

            # Prevent the event from bubbling up
            event.stop()
    
    def reset_to_different(self) -> None:
        """Reset the radio set back to 'DIFFERENT VALUES' state."""
        self._clear_selection()
        self.add_class("different-values-disabled")
        self.is_different_state = True

        # Restore tooltip
        self.tooltip = f"DIFFERENT VALUES - Click to set to default: {self.default_value.value}"
    
    def set_value(self, value: Enum) -> None:
        """Set a specific value (not from click-to-default)."""
        from openhcs.ui.shared.enum_display_formatter import EnumDisplayFormatter

        # Clear current selection
        self._clear_selection()

        # Set the new value
        target_label = EnumDisplayFormatter.get_display_text(value)
        for button in self.query('RadioButton'):
            if hasattr(button, 'label') and str(button.label).upper() == target_label:
                button.value = True
                break
        
        self.is_different_state = False
        self.tooltip = ""
    
    @property
    def current_state(self) -> str:
        """Get the current state of the widget."""
        if self.is_different_state:
            return "different"
        
        # Check if current selection matches default
        pressed_button = self.pressed_button
        if pressed_button and hasattr(pressed_button, 'label'):
            current_label = str(pressed_button.label).upper()
            default_label = self.default_value.value.upper()
            if current_label == default_label:
                return "default"
        
        return "modified"
    
    @property
    def selected_enum_value(self) -> Enum:
        """Get the currently selected enum value."""
        pressed_button = self.pressed_button
        if pressed_button and hasattr(pressed_button, 'label'):
            label = str(pressed_button.label).upper()
            # Find matching enum value
            for enum_value in self.enum_type:
                if enum_value.value.upper() == label:
                    return enum_value
        return None
