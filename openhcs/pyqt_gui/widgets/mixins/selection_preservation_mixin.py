"""
Selection Preservation Mixin for PyQt6 List Widgets

Provides shared logic for preserving selection when updating QListWidget contents,
similar to the ButtonListWidget pattern used in the Textual TUI.
"""

import logging
from typing import Optional, Callable, Any
from PyQt6.QtWidgets import QListWidget
from PyQt6.QtCore import Qt

logger = logging.getLogger(__name__)


class SelectionPreservationMixin:
    """
    Mixin that provides selection preservation for QListWidget updates.
    
    This mirrors the selection management logic from the Textual TUI's ButtonListWidget,
    ensuring consistent behavior across plate manager and pipeline editor.
    
    Requirements for using this mixin:
    1. The class must have a QListWidget attribute (specify via list_widget_attr)
    2. The class must implement get_item_identifier() to extract unique IDs from list items
    3. The class must implement should_preserve_selection() to determine preservation policy
    """
    
    def __init__(self, list_widget_attr: str = "list_widget", **kwargs):
        """
        Initialize the selection preservation mixin.
        
        Args:
            list_widget_attr: Name of the QListWidget attribute on the class
        """
        super().__init__(**kwargs)
        self.list_widget_attr = list_widget_attr
        self._preserved_selection: Optional[str] = None
        self._preserved_selections: list = []  # For multi-selection support
    
    def get_list_widget(self) -> QListWidget:
        """Get the QListWidget instance."""
        return getattr(self, self.list_widget_attr)
    
    def get_item_identifier(self, item_data: Any) -> str:
        """
        Extract unique identifier from item data.
        
        Must be implemented by the using class.
        
        Args:
            item_data: Data stored in the list item
            
        Returns:
            Unique identifier string for the item
        """
        raise NotImplementedError("Subclass must implement get_item_identifier()")
    
    def should_preserve_selection(self) -> bool:
        """
        Determine if selection should be preserved during list updates.
        
        Must be implemented by the using class.
        
        Returns:
            True if selection should be preserved, False otherwise
        """
        raise NotImplementedError("Subclass must implement should_preserve_selection()")
    
    def preserve_selection_during_update(self, update_func: Callable[[], None]):
        """
        Execute a list update function while preserving selection.
        
        This is the main method that wraps list updates to preserve selection.
        
        Args:
            update_func: Function that updates the list widget contents
        """
        if not self.should_preserve_selection():
            # No preservation needed, just update
            update_func()
            return
        
        # Save current selection
        self._save_current_selection()
        
        # Execute the update
        update_func()
        
        # Restore selection
        self._restore_saved_selection()
    
    def _save_current_selection(self):
        """Save the current selection state."""
        list_widget = self.get_list_widget()
        
        # Save single selection (highlighted item)
        current_item = list_widget.currentItem()
        if current_item:
            item_data = current_item.data(Qt.ItemDataRole.UserRole)
            if item_data:
                self._preserved_selection = self.get_item_identifier(item_data)
            else:
                self._preserved_selection = None
        else:
            self._preserved_selection = None
        
        # Save multi-selection (for future use)
        self._preserved_selections = []
        for item in list_widget.selectedItems():
            item_data = item.data(Qt.ItemDataRole.UserRole)
            if item_data:
                identifier = self.get_item_identifier(item_data)
                self._preserved_selections.append(identifier)
        
        logger.debug(f"Saved selection: {self._preserved_selection}, multi: {self._preserved_selections}")
    
    def _restore_saved_selection(self):
        """Restore the previously saved selection."""
        if not self._preserved_selection:
            return
        
        list_widget = self.get_list_widget()
        
        # Find and restore the primary selection
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            item_data = item.data(Qt.ItemDataRole.UserRole)
            if item_data:
                identifier = self.get_item_identifier(item_data)
                if identifier == self._preserved_selection:
                    list_widget.setCurrentRow(i)
                    logger.debug(f"Restored selection to: {identifier}")
                    break
        
        # Clear saved state
        self._preserved_selection = None
        self._preserved_selections = []
    
    def update_list_with_preservation(self, items: list, format_item_func: Callable[[Any], tuple]):
        """
        Update list widget contents with automatic selection preservation.
        
        This is a convenience method that handles the common pattern of:
        1. Clear list
        2. Add new items
        3. Preserve selection
        
        Args:
            items: List of items to add to the widget
            format_item_func: Function that takes an item and returns (display_text, item_data)
        """
        def update_func():
            list_widget = self.get_list_widget()
            list_widget.clear()
            
            for item in items:
                display_text, item_data = format_item_func(item)
                from PyQt6.QtWidgets import QListWidgetItem
                list_item = QListWidgetItem(display_text)
                list_item.setData(Qt.ItemDataRole.UserRole, item_data)
                list_widget.addItem(list_item)

        self.preserve_selection_during_update(update_func)

    def handle_selection_change_with_prevention(self, get_selected_func: Callable, on_selected_func: Callable, on_cleared_func: Callable):
        """
        Handle selection changes with automatic deselection prevention.

        This should be called from the widget's on_selection_changed method.

        Args:
            get_selected_func: Function that returns currently selected items
            on_selected_func: Function to call when items are selected (receives selected items)
            on_cleared_func: Function to call when selection is cleared (no args)
        """
        selected_items = get_selected_func()

        if selected_items:
            # Normal selection - call the handler and save selection ID
            on_selected_func(selected_items)
            # Save the selection ID for prevention
            first_item = selected_items[0]
            if hasattr(first_item, 'data') and callable(first_item.data):
                item_data = first_item.data(Qt.ItemDataRole.UserRole)
            else:
                item_data = first_item
            if item_data:
                self.set_current_selection_id(self.get_item_identifier(item_data))
        else:
            # No selection - check if we should prevent deselection
            if self.should_preserve_selection() and hasattr(self, '_current_selection_id') and self._current_selection_id:
                # Re-select the previously selected item
                self._reselect_by_id(self._current_selection_id)
                return

            # Allow clearing selection
            on_cleared_func()

    def _reselect_by_id(self, item_id: str):
        """
        Re-select an item by its identifier to prevent deselection.

        Args:
            item_id: Identifier of the item to re-select
        """
        list_widget = self.get_list_widget()

        # Find the item in the list and re-select it
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            item_data = item.data(Qt.ItemDataRole.UserRole)
            if item_data:
                identifier = self.get_item_identifier(item_data)
                if identifier == item_id:
                    # Temporarily block signals to avoid recursion
                    list_widget.blockSignals(True)
                    list_widget.setCurrentRow(i)
                    list_widget.blockSignals(False)
                    break

    def set_current_selection_id(self, item_id: str):
        """
        Set the current selection ID for deselection prevention.

        Args:
            item_id: Identifier of the currently selected item
        """
        self._current_selection_id = item_id
