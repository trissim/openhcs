"""
Grouped Dropdown component for the OpenHCS TUI.

This module provides a dropdown component that groups options by category.
"""
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from prompt_toolkit.application import get_app
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding.key_bindings import KeyBindingsBase
from prompt_toolkit.layout import Container, HSplit, VSplit, Window, Dimension
from prompt_toolkit.layout.mouse_handlers import MouseHandlers
from prompt_toolkit.layout.screen import Screen, WritePosition
from prompt_toolkit.widgets import Box, Button, Label, RadioList

logger = logging.getLogger(__name__)


class GroupedDropdown(Container):
    """
    A dropdown component that groups options by category.

    This component displays a dropdown with options grouped by category,
    with category headers that cannot be selected.
    """

    def __init__(
        self,
        options_by_group: Dict[str, List[Tuple[Any, str]]],
        default: Any = None,
        handler: Optional[Callable[[Any], None]] = None
    ):
        """
        Initialize the grouped dropdown.

        Args:
            options_by_group: Dictionary mapping group names to lists of (value, label) tuples
            default: The default selected value
            handler: Callback function when selection changes
        """
        self.options_by_group = options_by_group
        self.default = default
        self.handler = handler

        # Flatten options with group headers
        self.all_options = []
        for group_name, options in options_by_group.items():
            # Add group header as a disabled option
            self.all_options.append((f"__header_{group_name}", HTML(f"<b>{group_name}</b>")))
            # Add options for this group
            self.all_options.extend(options)

        # Create the dropdown
        self.dropdown = RadioList(
            values=[(value, label) for value, label in self.all_options],
            default=default
        )

        # Set handler for dropdown
        def on_change(value):
            # Ignore selection of headers
            if isinstance(value, str) and value.startswith("__header_"):
                # Reset to previous selection
                self.dropdown.current_value = self.default
                return

            # Update default
            self.default = value

            # Call handler
            if self.handler:
                self.handler(value)

        self.dropdown.handler = on_change

        # Create container - HSplit IS a Container with preferred_height
        self.container = HSplit([self.dropdown])

    def __pt_container__(self):
        """Return the container to be rendered."""
        return self.container

    # Implement abstract methods - delegate to HSplit
    def get_children(self):
        # HSplit has proper get_children() method
        return self.container.get_children()

    def preferred_width(self, max_available_width):
        # Container should return Dimension, not int
        return Dimension(min=30, preferred=max(30, max_available_width // 2))

    def preferred_height(self, width, max_available_height):
        # Delegate to the container but ensure a minimum height
        container_height = self.container.preferred_height(width, max_available_height)

        # If the container has a valid preferred height, use it
        if container_height.preferred is not None and container_height.preferred > 0:
            return container_height

        # Otherwise, use a reasonable default based on number of options
        # At least 3 lines, or more if we have many options
        min_height = min(max(3, len(self.all_options) // 2), max_available_height)
        return Dimension(min=min_height, preferred=min_height)

    def reset(self):
        self.container.reset()

    def write_to_screen(self, screen, mouse_handlers, write_position,
                        parent_style, erase_bg, z_index):
        self.container.write_to_screen(screen, mouse_handlers, write_position,
                                      parent_style, erase_bg, z_index)

    def mouse_handler(self, mouse_event):
        """Handle mouse events."""
        return self.container.mouse_handler(mouse_event)

    def is_modal(self):
        """Return whether this container is modal."""
        return False

    def get_key_bindings(self):
        """Return key bindings for this container."""
        return None
