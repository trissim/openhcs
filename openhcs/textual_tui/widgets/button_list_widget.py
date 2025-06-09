"""
Reusable ButtonListWidget - Perfect layout pattern for button row + expandable list.

This widget implements the perfected layout pattern:
- Button row at top (height: auto) 
- List area filling remaining space (height: 1fr)
- SelectionList expanding to fill container (height: 100%)

Used by PlateManager and PipelineEditor for consistent behavior.
"""

from typing import List, Dict, Any, Callable, Optional, Tuple
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, ScrollableContainer
from textual.widget import Widget
from textual.widgets import Button, SelectionList
from textual.reactive import reactive

import logging

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


class ButtonListWidget(Widget):
    """
    Reusable widget with button row + expandable list.
    
    Perfect layout pattern:
    - Vertical container
    - Button row (height: auto) 
    - ScrollableContainer (height: 1fr) containing SelectionList (height: 100%)
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
        """
        super().__init__(**kwargs)
        self.button_configs = button_configs
        self.list_id = list_id
        self.container_id = container_id
        self.on_button_pressed_callback = on_button_pressed
        self.on_selection_changed_callback = on_selection_changed
    
    def compose(self) -> ComposeResult:
        """Compose the perfect button-list layout."""
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
            
            # List area - expands to fill ALL remaining vertical space
            with ScrollableContainer(id=self.container_id) as container:
                container.styles.height = "1fr"  # CRITICAL: Fill remaining space
                
                selection_list = SelectionList(id=self.list_id)
                # Make SelectionList fill the entire container
                selection_list.styles.width = "100%"
                selection_list.styles.height = "100%"  # CRITICAL: Fill container
                yield selection_list
    
    def on_mount(self) -> None:
        """Called when widget is mounted."""
        logger.info(f"ButtonListWidget mounted with {len(self.button_configs)} buttons")
        self._update_button_states()
    
    def watch_items(self, items: List[Dict]) -> None:
        """Automatically update UI when items reactive property changes."""
        try:
            logger.info(f"watch_items called with {len(items)} items")
            
            # Update SelectionList content
            try:
                selection_list = self.query_one(f"#{self.list_id}", SelectionList)
                
                # Clear existing options
                selection_list.clear_options()
                
                # Add items as selection options - subclasses override format_item_for_display
                item_options = []
                for item in items:
                    display_text, value = self.format_item_for_display(item)
                    item_options.append((display_text, value))
                
                selection_list.add_options(item_options)
                logger.info(f"✅ Updated SelectionList successfully - now showing {len(items)} items")
                
            except Exception as e:
                logger.warning(f"Could not update list content (widget may not be mounted): {e}")
                self.call_later(self._delayed_update_display)
        
        except Exception as e:
            logger.error(f"Error in watch_items: {e}")
            self.call_later(self._delayed_update_display)
        
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
    
    def on_selection_list_selected_changed(self, event: SelectionList.SelectedChanged) -> None:
        """Handle selection changes from SelectionList."""
        selected_values = event.selection_list.selected
        
        logger.info(f"Selection changed: {len(selected_values)} items selected")
        
        # Update selected_item - use first selected item if any
        if selected_values:
            self.selected_item = selected_values[0]
        else:
            self.selected_item = ""
        
        # Notify callback if provided
        if self.on_selection_changed_callback:
            self.on_selection_changed_callback(selected_values)
        
        # Update button states
        self._update_button_states()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        logger.info(f"Button pressed: {button_id}")
        
        # Notify callback if provided
        if self.on_button_pressed_callback:
            self.on_button_pressed_callback(button_id)
    
    def get_selection_state(self) -> Tuple[List[Dict], str]:
        """
        Get current selection state from SelectionList.
        
        Returns:
            Tuple of (selected_items, selection_mode)
        """
        try:
            selection_list = self.query_one(f"#{self.list_id}", SelectionList)
            selected_values = selection_list.selected
            
            # Convert selected values back to item dictionaries
            selected_items = []
            for item in self.items:
                _, value = self.format_item_for_display(item)
                if value in selected_values:
                    selected_items.append(item)
            
            # Determine selection mode
            if not selected_items:
                selection_mode = "empty"
            elif len(selected_items) == len(self.items):
                selection_mode = "all"
            else:
                selection_mode = "checkbox"  # SelectionList is always checkbox-based
            
            return selected_items, selection_mode
        except Exception:
            # Fallback if widget not mounted
            return [], "empty"
    
    def _delayed_update_display(self) -> None:
        """Update the display - called when widget is mounted or as fallback."""
        try:
            selection_list = self.query_one(f"#{self.list_id}", SelectionList)
            
            # Clear and rebuild options
            selection_list.clear_options()
            item_options = []
            for item in self.items:
                display_text, value = self.format_item_for_display(item)
                item_options.append((display_text, value))
            
            selection_list.add_options(item_options)
            logger.info(f"✅ Delayed SelectionList update successful - showing {len(self.items)} items")
        except Exception as e:
            logger.warning(f"Delayed update failed (widget may not be ready): {e}")
            self.set_timer(0.1, self._delayed_update_display)
    
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
