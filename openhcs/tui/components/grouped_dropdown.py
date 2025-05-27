"""
Grouped Dropdown component for Hybrid TUI.

Ported from TUI's grouped_dropdown.py with adaptations for:
- Component interface compliance
- Simplified API for hybrid architecture
- Better type handling

This provides a dropdown component that groups options by category,
with category headers that cannot be selected.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from prompt_toolkit.application import get_app
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.layout import Container, HSplit, VSplit, Window, Dimension
from prompt_toolkit.widgets import Box, Button, Label, RadioList

from ..interfaces.component_interfaces import ComponentInterface

logger = logging.getLogger(__name__)

class GroupedDropdown(ComponentInterface):
    """
    A dropdown component that groups options by category.

    This component displays a dropdown with options grouped by category,
    with category headers that cannot be selected.
    """

    def __init__(
        self,
        values: Union[List[Tuple[Any, str]], Dict[str, List[Tuple[Any, str]]]],
        default: Any = None,
        on_change: Optional[Callable[[Any], None]] = None
    ):
        """
        Initialize the grouped dropdown.

        Args:
            values: Either a flat list of (value, label) tuples, or a dict mapping
                   group names to lists of (value, label) tuples
            default: The default selected value
            on_change: Callback function when selection changes
        """
        self.on_change = on_change
        self.default = default
        self.current_value = default

        # Normalize input to grouped format
        if isinstance(values, dict):
            self.options_by_group = values
        else:
            # Convert flat list to single group
            self.options_by_group = {"Options": values}

        # Flatten options with group headers
        self.all_options = []
        self.selectable_values = set()
        
        for group_name, options in self.options_by_group.items():
            # Add group header as a disabled option
            header_key = f"__header_{group_name}"
            self.all_options.append((header_key, HTML(f"<b>{group_name}</b>")))
            
            # Add options for this group
            for value, label in options:
                self.all_options.append((value, label))
                self.selectable_values.add(value)

        # Create the dropdown
        self.dropdown = RadioList(
            values=self.all_options,
            default=default
        )

        # Set handler for dropdown
        def on_selection_change(value):
            # Ignore selection of headers
            if isinstance(value, str) and value.startswith("__header_"):
                # Reset to previous selection
                self.dropdown.current_value = self.current_value
                return

            # Update current value
            self.current_value = value

            # Call handler
            if self.on_change:
                try:
                    self.on_change(value)
                except Exception as e:
                    logger.error(f"Error in dropdown change handler: {e}")

        self.dropdown.handler = on_selection_change

        # Create container
        self._container = Box(self.dropdown)

    @property
    def container(self) -> Container:
        """Return prompt_toolkit container for this component."""
        return self._container

    def update_data(self, data: Any) -> None:
        """Update component with new selected value."""
        if data in self.selectable_values:
            self.current_value = data
            self.dropdown.current_value = data

    def get_current_value(self) -> Any:
        """Get currently selected value."""
        return self.current_value

    def set_values(self, values: Union[List[Tuple[Any, str]], Dict[str, List[Tuple[Any, str]]]]):
        """Update the dropdown with new values."""
        # Store current selection
        current_selection = self.current_value

        # Update options
        if isinstance(values, dict):
            self.options_by_group = values
        else:
            self.options_by_group = {"Options": values}

        # Rebuild options list
        self.all_options = []
        self.selectable_values = set()
        
        for group_name, options in self.options_by_group.items():
            header_key = f"__header_{group_name}"
            self.all_options.append((header_key, HTML(f"<b>{group_name}</b>")))
            
            for value, label in options:
                self.all_options.append((value, label))
                self.selectable_values.add(value)

        # Update dropdown
        self.dropdown.values = self.all_options

        # Restore selection if still valid
        if current_selection in self.selectable_values:
            self.current_value = current_selection
            self.dropdown.current_value = current_selection
        else:
            # Select first available option
            if self.selectable_values:
                first_value = next(iter(self.selectable_values))
                self.current_value = first_value
                self.dropdown.current_value = first_value
            else:
                self.current_value = None
                self.dropdown.current_value = None

    def set_change_callback(self, callback: Callable[[Any], None]) -> None:
        """Set callback for value changes."""
        self.on_change = callback

    def is_empty(self) -> bool:
        """Check if dropdown has no selectable options."""
        return len(self.selectable_values) == 0

    def get_group_names(self) -> List[str]:
        """Get list of group names."""
        return list(self.options_by_group.keys())

    def get_options_for_group(self, group_name: str) -> List[Tuple[Any, str]]:
        """Get options for a specific group."""
        return self.options_by_group.get(group_name, [])

    def add_group(self, group_name: str, options: List[Tuple[Any, str]]):
        """Add a new group of options."""
        self.options_by_group[group_name] = options
        self.set_values(self.options_by_group)

    def remove_group(self, group_name: str):
        """Remove a group of options."""
        if group_name in self.options_by_group:
            del self.options_by_group[group_name]
            self.set_values(self.options_by_group)

    def clear(self):
        """Clear all options."""
        self.options_by_group = {}
        self.set_values(self.options_by_group)

    def focus(self):
        """Focus the dropdown."""
        get_app().layout.focus(self.dropdown)

    def __pt_container__(self):
        """Return the container to be rendered."""
        return self._container
