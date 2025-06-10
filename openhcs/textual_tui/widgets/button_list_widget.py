"""
Reusable ButtonListWidget - Enhanced SelectionList with integrated buttons.

This widget implements two key patterns:
1. Top button bar for global actions
2. SelectionList with left-aligned Up/Down buttons for item reordering

Used by PlateManager and PipelineEditor for consistent behavior.
"""

from typing import List, Dict, Any, Callable, Optional, Tuple, cast, Iterable
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, ScrollableContainer
from textual.widget import Widget
from textual.widgets import Button, SelectionList, Static
from textual.widgets._selection_list import Selection
from textual.reactive import reactive
from textual.strip import Strip
from textual.events import Click
from rich.segment import Segment
from rich.style import Style

import logging
from textual import on

logger = logging.getLogger(__name__)


class ButtonConfig:
    """Configuration for a button in the button row."""
    
    def __init__(
        self,
        label: str,
        button_id: str,
        disabled: bool = False,
        compact: bool = True
    ):
        self.label = label
        self.button_id = button_id
        self.disabled = disabled
        self.compact = compact


class InlineButtonSelectionList(SelectionList):
    """SelectionList with ↑↓ buttons rendered directly in each line."""

    def __init__(self, on_item_moved_callback=None, **kwargs):
        super().__init__(**kwargs)
        self.on_item_moved_callback = on_item_moved_callback

    def render_line(self, y: int) -> Strip:
        """Override to add ↑↓ buttons at start of each line."""
        # Get original line from SelectionList
        original = super().render_line(y)

        # Only add buttons if we have options and this is a valid line
        if y < self.option_count:
            # Add button text with proper styling
            from textual.strip import Strip
            from rich.segment import Segment
            from rich.style import Style

            # Create styled button segments
            button_style = Style(bgcolor="blue", color="white", bold=True)
            up_segment = Segment(" ↑ ", button_style)
            down_segment = Segment(" ↓ ", button_style)
            space_segment = Segment(" ")

            buttons = Strip([up_segment, down_segment, space_segment])

            # Combine strips
            return Strip.join([buttons, original])
        else:
            return original

    @on(Click)
    def handle_click(self, event: Click) -> None:
        """Handle clicks on ↑↓ buttons, pass other clicks to SelectionList."""
        # Get content offset to account for padding/borders
        content_offset = event.get_content_offset(self)
        if content_offset is None:
            logger.info(f"Click outside content area at ({event.x}, {event.y})")
            return

        # Use content-relative coordinates
        x, y = content_offset.x, content_offset.y
        logger.info(f"Click at content offset ({x}, {y}), option_count={self.option_count}")

        # Check if click is in button area (first 7 characters: " ↑  ↓  ")
        if x < 7 and y < self.option_count:
            row = int(y)  # Ensure integer row
            if 0 <= x <= 2:  # ↑ button area
                logger.info(f"UP button clicked: moving item {row} to {row - 1}")
                if row > 0 and self.on_item_moved_callback:
                    self.on_item_moved_callback(row, row - 1)
                    event.stop()
                    return
            elif 3 <= x <= 5:  # ↓ button area
                logger.info(f"DOWN button clicked: moving item {row} to {row + 1}")
                if row < self.option_count - 1 and self.on_item_moved_callback:
                    self.on_item_moved_callback(row, row + 1)
                    event.stop()
                    return

        # Not a button click - let normal SelectionList behavior continue
        logger.info(f"Click outside button area at content ({x}, {y}) - letting SelectionList handle it")





class ButtonListWidget(Widget):
    """
    A widget that combines a button row with an enhanced SelectionList.

    Layout:
    - Button row at top (height: auto)
    - Enhanced SelectionList with inline up/down buttons on each item
    """
    
    # Reactive properties for data and selection
    items: reactive[List[Dict]] = reactive([])
    selected_item: reactive[str] = reactive("")
    
    def __init__(
        self,
        button_configs: List[ButtonConfig],
        list_id: str = "content_list",
        container_id: str = "content_container",
        on_button_pressed: Optional[Callable[[str], None]] = None,
        on_selection_changed: Optional[Callable[[List[str]], None]] = None,
        on_item_moved: Optional[Callable[[int, int], None]] = None,
        **kwargs
    ):
        """
        Initialize ButtonListWidget.
        
        Args:
            button_configs: List of ButtonConfig objects defining the buttons
            list_id: ID for the SelectionList widget
            container_id: ID for the ScrollableContainer
            on_button_pressed: Callback for button press events (button_id)
            on_selection_changed: Callback for selection changes (selected_values)
            on_item_moved: Callback for item reordering (from_index, to_index)
        """
        super().__init__(**kwargs)
        self.button_configs = button_configs
        self.list_id = list_id
        self.container_id = container_id
        self.on_button_pressed_callback = on_button_pressed
        self.on_selection_changed_callback = on_selection_changed
        self.on_item_moved_callback = on_item_moved
    
    def compose(self) -> ComposeResult:
        """Compose the button-list layout."""
        with Vertical():
            # Button row - takes minimal height needed for buttons
            with Horizontal() as button_row:
                button_row.styles.height = "auto"  # CRITICAL: Take only needed height

                for config in self.button_configs:
                    yield Button(
                        config.label,
                        id=config.button_id,
                        disabled=config.disabled,
                        compact=config.compact
                    )

            # Use SelectionList with overlaid buttons
            selection_list = InlineButtonSelectionList(
                id=self.list_id,
                on_item_moved_callback=self.on_item_moved_callback
            )
            selection_list.styles.height = "1fr"  # CRITICAL: Fill remaining space
            yield selection_list
    
    def on_mount(self) -> None:
        """Called when widget is mounted."""
        logger.info(f"ButtonListWidget mounted with {len(self.button_configs)} buttons")
        self._update_button_states()
        # Update the SelectionList when mounted
        if self.items:
            self.call_later(self._update_selection_list)
    
    def watch_items(self, items: List[Dict]) -> None:
        """Automatically update UI when items reactive property changes."""
        logger.info(f"watch_items called with {len(items)} items")
        # Update the SelectionList
        self._update_selection_list()
        # Update button states
        self._update_button_states()
    
    def watch_selected_item(self, item_value: str) -> None:
        """Automatically update UI when selected_item changes."""
        self._update_button_states()
        logger.debug(f"Selected item: {item_value}")
    
    def format_item_for_display(self, item: Dict) -> Tuple[str, str]:
        """
        Format an item for display in the SelectionList.

        Subclasses should override this method.

        Args:
            item: Item dictionary

        Returns:
            Tuple of (display_text, value)
        """
        # Default implementation - subclasses should override
        name = item.get('name', 'Unknown')
        value = item.get('path', item.get('id', str(item)))
        return name, value

    def _sanitize_id(self, value: str) -> str:
        """
        Sanitize a value for use as a Textual widget ID.

        Textual IDs must contain only letters, numbers, underscores, or hyphens,
        and must not begin with a number.
        """
        import re
        # Replace invalid characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', value)
        # Ensure it doesn't start with a number
        if sanitized and sanitized[0].isdigit():
            sanitized = f"item_{sanitized}"
        # Ensure it's not empty
        if not sanitized:
            sanitized = "item_unknown"
        return sanitized
    
    @on(SelectionList.OptionSelected)
    def handle_option_selected(self, event: SelectionList.OptionSelected) -> None:
        """Handle SelectionList selection changes."""
        self.selected_item = event.option.value

        # Notify callback if provided
        if self.on_selection_changed_callback:
            self.on_selection_changed_callback([event.option.value])

        # Update button states
        self._update_button_states()

        logger.debug(f"Selected item: {event.option.value}")
    
    @on(Button.Pressed)
    def handle_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses from the top button bar."""
        button_id = event.button.id
        logger.info(f"Button pressed: {button_id}")

        # CRITICAL: Stop event propagation
        event.stop()

        # Notify callback if provided
        if self.on_button_pressed_callback:
            self.on_button_pressed_callback(button_id)
    
    def get_selection_state(self) -> Tuple[List[Dict], str]:
        """
        Get current selection state.

        Returns:
            Tuple of (selected_items, selection_mode)
        """
        # Use the selected_item from our custom list
        if self.selected_item:
            # Find the selected item
            selected_items = []
            for item in self.items:
                _, value = self.format_item_for_display(item)
                if value == self.selected_item:
                    selected_items.append(item)
                    break

            return selected_items, "cursor"
        else:
            return [], "empty"
    
    def _update_selection_list(self) -> None:
        """Update the InlineButtonSelectionList with current items."""
        if not self.is_mounted:
            logger.debug("Widget not mounted yet, skipping list update")
            return

        try:
            # Get the InlineButtonSelectionList instance
            selection_list = self.query_one(f"#{self.list_id}", InlineButtonSelectionList)

            # Clear existing options
            selection_list.clear_options()

            # Add options for each item - SelectionList uses simple tuples (text, value)
            options = []
            for item in self.items:
                display_text, value = self.format_item_for_display(item)
                options.append((display_text, value))

            selection_list.add_options(options)

            # Set selection if we have a selected item
            if self.selected_item:
                try:
                    selection_list.highlighted = self.selected_item
                except Exception:
                    pass  # Item not found, ignore

            logger.info(f"✅ Updated InlineButtonSelectionList with {len(self.items)} items")

        except Exception as e:
            logger.error(f"Failed to update InlineButtonSelectionList: {e}")
            import traceback
            traceback.print_exc()

    def _delayed_update_display(self) -> None:
        """Update the display - called when widget is mounted or as fallback."""
        try:
            self._update_selection_list()
            logger.info(f"✅ Delayed list update successful - showing {len(self.items)} items with inline buttons")
        except Exception as e:
            logger.warning(f"Delayed update failed (widget may not be ready): {e}")
            self.set_timer(0.1, self._delayed_update_display)
            
    def action_add_item_buttons(self) -> None:
        """Add buttons to list items - not needed with InlineButtonSelectionList."""
        # This is a no-op since the InlineButtonSelectionList handles button rendering
        pass
    
    def _update_button_states(self) -> None:
        """
        Update button enabled/disabled states.
        
        Subclasses should override this method to implement specific button logic.
        """
        # Default implementation - subclasses should override
        has_items = len(self.items) > 0
        has_selection = bool(self.selected_item)
        
        # Basic logic - enable buttons based on data availability
        for config in self.button_configs:
            try:
                button = self.query_one(f"#{config.button_id}", Button)
                # Default: disable if no items, enable if has selection
                if "add" in config.button_id.lower():
                    button.disabled = False  # Add always enabled
                else:
                    button.disabled = not has_selection  # Others need selection
            except Exception:
                # Button might not be mounted yet
                pass
