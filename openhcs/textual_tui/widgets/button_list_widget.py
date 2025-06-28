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
            #button_style = Style(bgcolor="blue", color="white", bold=True)
            button_style = Style( color="white", bold=True)
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
            # Click outside content area - let it bubble up normally
            return

        # Use content-relative coordinates
        x, y = content_offset.x, content_offset.y

        # Check if click is in button area (first 7 characters: " ↑  ↓  ")
        if x < 7 and y < self.option_count:
            row = int(y)  # Ensure integer row
            if 0 <= x <= 2:  # ↑ button area
                if row > 0 and self.on_item_moved_callback:
                    self.on_item_moved_callback(row, row - 1)
                    event.stop()  # Stop event - this was a button click
                    return
            elif 3 <= x <= 5:  # ↓ button area
                if row < self.option_count - 1 and self.on_item_moved_callback:
                    self.on_item_moved_callback(row, row + 1)
                    event.stop()  # Stop event - this was a button click
                    return

        # Not a button click - let normal SelectionList behavior continue
        # DO NOT call event.stop() here - let the event bubble to SelectionList





class ButtonListWidget(Widget):
    """
    A widget that combines a button row with an enhanced SelectionList.

    Layout:
    - Button row at top (height: auto)
    - Enhanced SelectionList with inline up/down buttons on each item
    """
    
    # Reactive properties for data and selection
    items: reactive[List[Dict]] = reactive([])
    selected_item: reactive[str] = reactive("")  # First selected item (for backward compatibility)
    highlighted_item: reactive[str] = reactive("")  # Currently highlighted item (blue highlight)
    selected_items: reactive[List[str]] = reactive([])  # All selected items (checkmarks)
    
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
    
    @on(SelectionList.SelectedChanged)
    def handle_selected_changed(self, event: SelectionList.SelectedChanged) -> None:
        """Handle SelectionList selection changes (checkmarks)."""
        # Get all selected items (checkmarks)
        selected_values = event.selection_list.selected
        self.selected_items = list(selected_values)

        # Update selected_item for backward compatibility (first selected item)
        if selected_values:
            self.selected_item = selected_values[0]
        else:
            self.selected_item = ""

        # Notify callback if provided
        if self.on_selection_changed_callback:
            self.on_selection_changed_callback(selected_values)

        # Update button states
        self._update_button_states()

    @on(SelectionList.SelectionHighlighted)
    def handle_highlight_changed(self, event: SelectionList.SelectionHighlighted) -> None:
        """Handle SelectionList highlight changes (blue highlight)."""
        try:
            # Get the highlighted item using the selection_list's highlighted property
            selection_list = event.selection_list
            highlighted_index = selection_list.highlighted

            if highlighted_index is not None and 0 <= highlighted_index < len(self.items):
                # Get the value for the highlighted item
                highlighted_item = self.items[highlighted_index]
                _, value = self.format_item_for_display(highlighted_item)
                self.highlighted_item = value
                logger.debug(f"Highlight changed to index {highlighted_index}: {value}")
            else:
                self.highlighted_item = ""
                logger.debug("No highlight (cleared)")
        except Exception as e:
            logger.warning(f"Failed to handle highlight change: {e}")
            self.highlighted_item = ""

        # Update button states
        self._update_button_states()
    
    @on(Button.Pressed)
    async def handle_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses from the top button bar (supports both sync and async callbacks)."""
        button_id = event.button.id

        # CRITICAL: Stop event propagation
        event.stop()

        # Notify callback if provided (support both sync and async callbacks)
        if self.on_button_pressed_callback:
            import asyncio
            import inspect

            if inspect.iscoroutinefunction(self.on_button_pressed_callback):
                # Async callback - await it
                await self.on_button_pressed_callback(button_id)
            else:
                # Sync callback - call normally
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
                logger.info(f"Adding option: {display_text}")

            selection_list.add_options(options)
            logger.info(f"Added {len(options)} options to SelectionList")

            # Force refresh the SelectionList display
            selection_list.refresh()
            logger.info("Called refresh() on SelectionList")

            # Force refresh the SelectionList display
            try:
                selection_list.refresh()
                logger.info("Called refresh() on SelectionList")
            except Exception as e:
                logger.warning(f"Failed to refresh SelectionList: {e}")

            # Set selection if we have a selected item
            if self.selected_item:
                try:
                    selection_list.highlighted = self.selected_item
                except Exception:
                    pass  # Item not found, ignore

        except Exception as e:
            logger.error(f"Failed to update InlineButtonSelectionList: {e}", exc_info=True)

    def _delayed_update_display(self, retry_count: int = 0) -> None:
        """Update the display - called when widget is mounted or as fallback."""
        try:
            self._update_selection_list()
        except Exception as e:
            # Limit retries to prevent infinite loops
            if retry_count < 5:  # Max 5 retries
                logger.warning(f"Delayed update failed (widget may not be ready), retry {retry_count + 1}/5: {e}")
                self.call_later(lambda: self._delayed_update_display(retry_count + 1))
            else:
                logger.error(f"Delayed update failed after 5 retries, giving up: {e}")
            
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
