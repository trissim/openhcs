"""
Interactive List Item component for the OpenHCS TUI.

Fixed internal architecture while maintaining compatible interface for list_manager.py.
"""
import asyncio
import logging
from typing import Any, Callable, Dict, Optional

from prompt_toolkit.application import get_app
from prompt_toolkit.layout import Container, HSplit, VSplit, Window, Dimension
from prompt_toolkit.widgets import Box, Button, Label
from prompt_toolkit.mouse_events import MouseEventType

logger = logging.getLogger(__name__)

# SafeButton eliminated - use Button directly

class InteractiveListItem:
    """
    Interactive list item component - no inheritance abuse, clean architecture.
    
    Maintains exact same interface as original for list_manager.py compatibility.
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
        can_move_down: bool = True,
        selected_style: str = "class:selected-item-box",
        unselected_style: str = "class:list-item-box"
    ):
        """Initialize - pure setup, no work in constructor."""
        self.item_data = item_data
        self.item_index = item_index
        self.is_selected = is_selected
        self.display_text_func = display_text_func
        self.on_select = on_select
        self.on_move_up = on_move_up
        self.on_move_down = on_move_down
        self.can_move_up = can_move_up
        self.can_move_down = can_move_down
        self.selected_style = selected_style
        self.unselected_style = unselected_style

        # No work in constructor - just pure setup

    def _get_display_text(self) -> str:
        """Get display text for this item."""
        if self.display_text_func:
            return self.display_text_func(self.item_data, self.is_selected)
        return str(self.item_data)

    def _build_container(self) -> Container:
        """Build container - pure function, called when needed."""
        # Create main text with constrained width (don't expand to fit content)
        from prompt_toolkit.layout.controls import FormattedTextControl

        # Create control with mouse handler (following FramedButton pattern)
        control = FormattedTextControl(
            self._get_display_text(),
            focusable=False,
        )

        # Assign mouse handler directly to the control
        def mouse_handler(mouse_event):
            if mouse_event.event_type == MouseEventType.MOUSE_UP and self.on_select:
                self._run_callback(self.on_select, self.item_index)
                return True
            return False

        control.mouse_handler = mouse_handler

        item_text = Window(
            control,
            width=Dimension(weight=1),  # Take proportional space
            wrap_lines=True,  # Wrap long lines instead of expanding
            dont_extend_width=True,  # Don't expand beyond allocated width
        )

        # Style based on selection
        style = self.selected_style if self.is_selected else self.unselected_style
        main_content = Box(
            item_text,
            padding_left=1,
            padding_right=1,
            style=style
        )

        # Create move buttons if callbacks provided
        buttons = []
        if self.on_move_up:
            up_button = Button(
                text="^",
                handler=self._handle_move_up,
                width=3  # Minimum width: 2 (symbols) + 1 (text) = 3
            )
            up_style = "class:move-button" if self.can_move_up else "class:disabled-button"
            up_box = Box(up_button, style=up_style)
            buttons.append(up_box)

        if self.on_move_down:
            down_button = Button(
                text="v",
                handler=self._handle_move_down,
                width=3  # Minimum width: 2 (symbols) + 1 (text) = 3
            )
            down_style = "class:move-button" if self.can_move_down else "class:disabled-button"
            down_box = Box(down_button, style=down_style)
            buttons.append(down_box)

        # Combine main content with buttons with proper width constraints
        if buttons:
            buttons_container = VSplit(buttons)
            # Use VSplit with width constraints to prevent expansion
            content_row = VSplit([
                main_content,  # Box already has the content
                buttons_container  # Buttons have fixed width
            ], width=Dimension(weight=1))
            return HSplit([content_row])
        else:
            # Just return the main content directly
            return HSplit([main_content])

    def _handle_move_up(self):
        """Handle move up button click."""
        if self.on_move_up and self.can_move_up:
            self._run_callback(self.on_move_up, self.item_index)

    def _handle_move_down(self):
        """Handle move down button click."""
        if self.on_move_down and self.can_move_down:
            self._run_callback(self.on_move_down, self.item_index)

    def _run_callback(self, callback: Callable, *args):
        """Run callback with proper async handling."""
        if asyncio.iscoroutinefunction(callback):
            get_app().create_background_task(callback(*args))
        else:
            callback(*args)

    # Prompt_toolkit Container interface - no inheritance, just duck typing
    def __pt_container__(self):
        """Return container for prompt_toolkit."""
        return self._build_container()

    def get_children(self):
        return self._build_container().get_children()



    def preferred_height(self, width, max_available_height):
        container = self._build_container()
        container_height = container.preferred_height(width, max_available_height)
        
        if container_height.preferred is not None and container_height.preferred > 0:
            return container_height
        
        return Dimension.exact(1)

    def reset(self):
        self._build_container().reset()

    def write_to_screen(self, screen, mouse_handlers, write_position,
                        parent_style, erase_bg, z_index):
        """Write to screen."""
        self._build_container().write_to_screen(
            screen, mouse_handlers, write_position, parent_style, erase_bg, z_index
        )

    def mouse_handler(self, mouse_event):
        """Handle mouse events."""
        # Delegate to container
        return self._build_container().mouse_handler(mouse_event)

    def is_modal(self):
        """Return whether this is modal."""
        return False

    def get_key_bindings(self):
        """Return key bindings."""
        return None
