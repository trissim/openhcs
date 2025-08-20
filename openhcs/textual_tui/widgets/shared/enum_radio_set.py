# File: openhcs/textual_tui/widgets/shared/enum_radio_set.py

from enum import Enum
from typing import Optional
from textual.widgets import RadioSet, RadioButton
from textual.app import ComposeResult

class EnumRadioSet(RadioSet):
    """RadioSet for enum parameters. Simple enum → radio buttons mapping."""

    def __init__(self, enum_class: type, current_value: Optional[str] = None, **kwargs):
        """Create RadioSet from enum class.

        Args:
            enum_class: The enum class (e.g., VariableComponents)
            current_value: Current string value (e.g., "site")
            **kwargs: Additional RadioSet arguments
        """
        # Force compact mode and pass to parent
        kwargs['compact'] = True
        super().__init__(**kwargs)
        self.enum_class = enum_class
        self.current_value = current_value

        # Set height to exactly fit the number of enum options
        num_options = len(list(enum_class))
        self.styles.height = num_options
        self.styles.max_height = num_options
    
    def compose(self) -> ComposeResult:
        """Create radio buttons for each enum option."""
        from openhcs.ui.shared.ui_utils import format_enum_display

        for enum_member in self.enum_class:
            # Create button with enum value as ID
            button_id = f"enum_{enum_member.value}"
            is_pressed = (self.current_value == enum_member.value)

            yield RadioButton(
                label=format_enum_display(enum_member),
                value=is_pressed,
                id=button_id
            )
