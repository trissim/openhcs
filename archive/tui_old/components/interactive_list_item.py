"""
Interactive List Item component for the OpenHCS TUI.

This module provides a component for displaying interactive list items
that can be selected, moved up/down, and clicked.
"""
import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

from prompt_toolkit.application import get_app
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_bindings import KeyBindingsBase
from prompt_toolkit.layout import Container, HSplit, VSplit, Window, Dimension
from prompt_toolkit.layout.mouse_handlers import MouseHandlers
from prompt_toolkit.layout.screen import Screen, WritePosition
from prompt_toolkit.mouse_events import MouseEventType
from prompt_toolkit.widgets import Box, Button, Label

logger = logging.getLogger(__name__)


class InteractiveListItem(Container):
    """
    An interactive list item component that can be selected and reordered.

    This component displays a list item with optional up/down buttons for reordering
    and handles selection via mouse clicks.
    """

    def __init__(
        self,
        item_data: Dict[str, Any],
        item_index: int,
        is_selected: bool = False,
        display_text_func: Optional[Callable[[Dict[str, Any], bool], str]] = None,
        on_select: Optional[Callable[[int], Any]] = None,
        on_move_up: Optional[Callable[[int], Any]] = None,
        on_move_down: Optional[Callable[[int], Any]] = None,
        can_move_up: bool = True,
        can_move_down: bool = True
    ):
        """
        Initialize the interactive list item.

        Args:
            item_data: The data for this list item
            item_index: The index of this item in the list
            is_selected: Whether this item is currently selected
            display_text_func: Function to generate display text from item_data
            on_select: Callback when item is selected
            on_move_up: Callback when item is moved up
            on_move_down: Callback when item is moved down
            can_move_up: Whether this item can be moved up
            can_move_down: Whether this item can be moved down
        """
        self.item_data = item_data
        self.item_index = item_index
        self.is_selected = is_selected
        self.display_text_func = display_text_func
        self.on_select = on_select
        self.on_move_up = on_move_up
        self.on_move_down = on_move_down
        self.can_move_up = can_move_up
        self.can_move_down = can_move_down

        # Create the item label with safe formatting
        from openhcs.tui.utils.safe_formatting import SafeLabel
        self.item_label = SafeLabel(
            text=self._get_display_text,
            style=""  # Empty style, we'll handle styling in the container
        )

        # Create up/down buttons if needed
        # Create up button (style will be applied via Box container)
        if on_move_up:
            self.up_button = Button(
                text="^",
                handler=self._handle_move_up,
                width=1
            )
        else:
            self.up_button = None

        # Create down button (style will be applied via Box container)
        if on_move_down:
            self.down_button = Button(
                text="v",
                handler=self._handle_move_down,
                width=1
            )
        else:
            self.down_button = None

        # Create the container
        self._create_container()

    def _create_container(self):
        """Create the container for this list item."""
        # Create the main content with the label
        # Use a simple string for style to avoid concatenation issues
        style = "class:selected-item-box" if self.is_selected else "class:list-item-box"
        main_content = Box(
            self.item_label,
            padding_left=1,
            padding_right=1,
            style=style
        )

        # If we have move buttons, add them
        if self.up_button or self.down_button:
            # Create boxes for buttons with appropriate styles
            up_style = "class:move-button" if self.can_move_up else "class:disabled-button"
            down_style = "class:move-button" if self.can_move_down else "class:disabled-button"

            up_box = Box(self.up_button, style=up_style) if self.up_button else Window(width=1)
            down_box = Box(self.down_button, style=down_style) if self.down_button else Window(width=1)

            buttons_container = VSplit([up_box, down_box])

            self.container = HSplit([
                VSplit([
                    main_content,
                    buttons_container
                ])
            ])
        else:
            self.container = HSplit([main_content])

    def _get_display_text(self) -> str:
        """Get the display text for this item."""
        if self.display_text_func:
            return self.display_text_func(self.item_data, self.is_selected)
        return str(self.item_data)

    def _handle_move_up(self):
        """Handle move up button click."""
        if self.on_move_up and self.can_move_up:
            if asyncio.iscoroutinefunction(self.on_move_up):
                get_app().create_background_task(self.on_move_up(self.item_index))
            else:
                self.on_move_up(self.item_index)

    def _handle_move_down(self):
        """Handle move down button click."""
        if self.on_move_down and self.can_move_down:
            if asyncio.iscoroutinefunction(self.on_move_down):
                get_app().create_background_task(self.on_move_down(self.item_index))
            else:
                self.on_move_down(self.item_index)

    def __pt_container__(self):
        """Return the container to be rendered."""
        return self.container

    def mouse_handler(self, mouse_event):
        """Handle mouse events for this list item."""
        # If the mouse event is a click and we have an on_select callback
        if mouse_event.event_type == MouseEventType.MOUSE_UP and self.on_select:
            # Create a background task for the on_select callback
            if asyncio.iscoroutinefunction(self.on_select):
                get_app().create_background_task(self.on_select(self.item_index))
            else:
                self.on_select(self.item_index)
            return True

        # Otherwise, delegate to the container
        return self.container.mouse_handler(mouse_event)

    # Implement abstract methods by delegating to the container
    def get_children(self):
        return self.container.get_children()

    def preferred_width(self, max_available_width):
        # Delegate to the container but ensure a minimum width
        container_width = self.container.preferred_width(max_available_width)

        # If the container has a valid preferred width, use it
        if container_width.preferred is not None and container_width.preferred > 0:
            return container_width

        # Otherwise, use a reasonable default
        return Dimension(min=20, preferred=max(20, max_available_width // 2))

    def preferred_height(self, width, max_available_height):
        # Delegate to the container but ensure a minimum height
        container_height = self.container.preferred_height(width, max_available_height)

        # If the container has a valid preferred height, use it
        if container_height.preferred is not None and container_height.preferred > 0:
            return container_height

        # Otherwise, use a reasonable default (1 line)
        return Dimension.exact(1)

    def reset(self):
        self.container.reset()

    def write_to_screen(self, screen, mouse_handlers, write_position,
                        parent_style, erase_bg, z_index):
        """Write the list item to the screen."""
        self.container.write_to_screen(screen, mouse_handlers, write_position,
                                      parent_style, erase_bg, z_index)

    def is_modal(self):
        """Return whether this container is modal."""
        return False

    def get_key_bindings(self):
        """Return key bindings for this container."""
        return None
