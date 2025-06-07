"""
SelectableListView - Multi-selection ListView with checkbox support for Textual TUI.

This widget provides the core multi-selection functionality that replaces Static widgets
in PlateManager and PipelineEditor, implementing the exact selection state logic from
the prompt-toolkit ListManagerPane.
"""

from typing import Dict, List, Set, Tuple, Any, Optional, Callable
import logging

from textual.widget import Widget
from textual.widgets import ListView, ListItem, Static
from textual.reactive import reactive
from textual.message import Message
from textual import events
from textual.containers import Horizontal, Vertical
from textual.app import ComposeResult

logger = logging.getLogger(__name__)


class SelectableListItem(ListItem):
    """ListItem with checkbox support and proper rendering."""
    
    def __init__(self, item_data: Dict[str, Any], index: int, is_checked: bool = False):
        self.item_data = item_data
        self.index = index
        self.is_checked = is_checked
        
        # Create display text with checkbox
        checkbox = "☑" if is_checked else "☐"
        display_text = self._format_item_display(item_data)
        super().__init__(Static(f"{checkbox} {display_text}"))
    
    def _format_item_display(self, item_data: Dict[str, Any]) -> str:
        """Format item data for display. Override in subclasses for custom formatting."""
        name = item_data.get('name', 'Unknown')
        status = item_data.get('status', '?')
        return f"{name} ({status})"
    
    def update_checkbox(self, is_checked: bool):
        """Update checkbox state and re-render."""
        self.is_checked = is_checked
        checkbox = "☑" if is_checked else "☐"
        display_text = self._format_item_display(self.item_data)
        
        # Update the Static widget content
        static_widget = self.children[0]
        static_widget.update(f"{checkbox} {display_text}")


class SelectableListView(Widget):
    """
    ListView with checkbox multi-selection support.
    
    Implements the exact selection state logic from prompt-toolkit ListManagerPane:
    - Priority-based selection: Checkbox > Cursor > All > Empty
    - Bulletproof validation with edge case handling
    - Mouse and keyboard interaction support
    """
    
    # Reactive state
    items = reactive(list)
    checked_indices = reactive(set)
    highlighted_index = reactive(0)
    
    class SelectionChanged(Message):
        """Message sent when selection state changes."""
        def __init__(self, selected_items: List[Dict], selection_mode: str):
            self.selected_items = selected_items
            self.selection_mode = selection_mode
            super().__init__()
    
    def __init__(self, items: List[Dict[str, Any]] = None, **kwargs):
        super().__init__(**kwargs)
        self._list_view = None
        self._initializing = True

        # Initialize reactive properties carefully to avoid circular watchers
        self.checked_indices = set()
        self.highlighted_index = 0
        self.items = items or []

        self._initializing = False
    
    def compose(self) -> ComposeResult:
        """Create the ListView component."""
        self._list_view = ListView()
        yield self._list_view
        
        # Populate initial items
        self._rebuild_list_items()
    
    def _rebuild_list_items(self):
        """Rebuild ListView items with current state."""
        if not self._list_view:
            return
            
        # Clear existing items
        self._list_view.clear()
        
        # Add items with current checkbox state
        for index, item_data in enumerate(self.items):
            is_checked = index in self.checked_indices
            list_item = SelectableListItem(item_data, index, is_checked)
            self._list_view.append(list_item)
        
        # Set highlighted item
        if self.items and 0 <= self.highlighted_index < len(self.items):
            self._list_view.index = self.highlighted_index
    
    def watch_items(self, new_items: List[Dict[str, Any]]):
        """React to items changes."""
        if getattr(self, '_initializing', False):
            return

        # Validate checked indices are still valid
        max_index = len(new_items) - 1
        if max_index >= 0:
            self.checked_indices = {i for i in self.checked_indices if i <= max_index}
        else:
            self.checked_indices = set()

        # Validate highlighted index
        if self.highlighted_index >= len(new_items):
            self.highlighted_index = max(0, len(new_items) - 1)

        # Rebuild list
        self._rebuild_list_items()

        # Emit selection change
        self._emit_selection_changed()
    
    def watch_checked_indices(self, new_checked: Set[int]):
        """React to checked indices changes."""
        if getattr(self, '_initializing', False):
            return
        self._update_checkbox_display()
        self._emit_selection_changed()

    def watch_highlighted_index(self, new_index: int):
        """React to highlighted index changes."""
        if getattr(self, '_initializing', False):
            return
        if self._list_view and 0 <= new_index < len(self.items):
            self._list_view.index = new_index
        self._emit_selection_changed()
    
    def _update_checkbox_display(self):
        """Update checkbox display for all items."""
        if not self._list_view:
            return
            
        for index, list_item in enumerate(self._list_view.children):
            if isinstance(list_item, SelectableListItem):
                is_checked = index in self.checked_indices
                list_item.update_checkbox(is_checked)
    
    def get_selection_state(self) -> Tuple[List[Dict], str]:
        """
        Get current selection state with bulletproof validation.
        
        Implements exact logic from prompt-toolkit ListManagerPane:
        Priority: Checkbox > Cursor > All > Empty
        
        Returns:
            Tuple[List[Dict], str]: (selected_items, selection_mode)
            - selected_items: List of item dictionaries to operate on
            - selection_mode: "empty" | "all" | "cursor" | "checkbox"
        """
        # VALIDATION 1: Check if list has items
        if not self.items:
            return [], "empty"
        
        # VALIDATION 2: Check if any checkboxes are currently checked (highest priority)
        if self.checked_indices:
            # Validate checked indices are still in the list
            valid_checked_indices = {i for i in self.checked_indices if 0 <= i < len(self.items)}
            if not valid_checked_indices:
                # Checked items were removed - clear checks and fall through to cursor/all logic
                self.checked_indices = set()
            else:
                # At least one checkbox is currently checked - use checkbox mode
                selected_items = [self.items[i] for i in valid_checked_indices]
                return selected_items, "checkbox"
        
        # VALIDATION 3: Check for cursor selection
        if 0 <= self.highlighted_index < len(self.items):
            highlighted_item = self.items[self.highlighted_index]
            return [highlighted_item], "cursor"
        
        # VALIDATION 4: No specific selection - default to all items
        return self.items.copy(), "all"
    
    def get_operation_description(self, selected_items: List[Dict], selection_mode: str, operation: str) -> str:
        """Generate human-readable description of what will be operated on."""
        count = len(selected_items)
        if selection_mode == "empty":
            return f"No items available for {operation}"
        elif selection_mode == "all":
            return f"{operation.title()} ALL {count} items"
        elif selection_mode == "cursor":
            item_name = selected_items[0].get('name', 'Unknown')
            return f"{operation.title()} highlighted item: {item_name}"
        elif selection_mode == "checkbox":
            if count == 1:
                item_name = selected_items[0].get('name', 'Unknown')
                return f"{operation.title()} checked item: {item_name}"
            else:
                return f"{operation.title()} {count} checked items"
        else:
            raise ValueError(f"Unknown selection_mode: {selection_mode}")
    
    def toggle_checkbox(self, index: int):
        """Toggle checkbox state for item at index."""
        if 0 <= index < len(self.items):
            if index in self.checked_indices:
                self.checked_indices = self.checked_indices - {index}
            else:
                self.checked_indices = self.checked_indices | {index}
    
    def clear_all_checks(self):
        """Clear all checkbox selections."""
        self.checked_indices = set()
    
    def check_all(self):
        """Check all items."""
        self.checked_indices = set(range(len(self.items)))
    
    def _emit_selection_changed(self):
        """Emit selection changed message."""
        selected_items, selection_mode = self.get_selection_state()
        self.post_message(self.SelectionChanged(selected_items, selection_mode))
    
    async def on_list_view_selected(self, event: ListView.Selected):
        """Handle ListView selection (cursor movement)."""
        self.highlighted_index = event.list_view.index
        event.stop()
    
    async def on_key(self, event: events.Key):
        """Handle keyboard input."""
        if event.key == "space":
            # Toggle checkbox for highlighted item
            self.toggle_checkbox(self.highlighted_index)
            event.prevent_default()
        elif event.key == "ctrl+a":
            # Select all
            self.check_all()
            event.prevent_default()
        elif event.key == "ctrl+d":
            # Deselect all
            self.clear_all_checks()
            event.prevent_default()
    
    def load_items(self, new_items: List[Dict[str, Any]]):
        """Load new items into the list."""
        self.items = new_items.copy()
    
    def get_checked_items(self) -> List[Dict[str, Any]]:
        """Get all checked items."""
        return [self.items[i] for i in self.checked_indices if 0 <= i < len(self.items)]
    
    def get_highlighted_item(self) -> Optional[Dict[str, Any]]:
        """Get currently highlighted item."""
        if 0 <= self.highlighted_index < len(self.items):
            return self.items[self.highlighted_index]
        return None
