"""
Selection Preservation Utilities for PyQt6 List Widgets

Simple utility functions for preserving selection when updating QListWidget contents,
similar to the ButtonListWidget pattern used in the Textual TUI.
"""

import logging
from typing import Callable, Any
from PyQt6.QtWidgets import QListWidget
from PyQt6.QtCore import Qt

logger = logging.getLogger(__name__)


def preserve_selection_during_update(
    list_widget: QListWidget,
    get_identifier_func: Callable[[Any], str],
    should_preserve_func: Callable[[], bool],
    update_func: Callable[[], None]
):
    """
    Execute a list update function while preserving selection.

    Args:
        list_widget: The QListWidget to update
        get_identifier_func: Function to extract unique ID from item data
        should_preserve_func: Function that returns True if selection should be preserved
        update_func: Function that updates the list widget contents
    """
    if not should_preserve_func():
        # No preservation needed, just update
        update_func()
        return

    # Save current selection
    current_item = list_widget.currentItem()
    current_id = None
    if current_item:
        item_data = current_item.data(Qt.ItemDataRole.UserRole)
        if item_data:
            current_id = get_identifier_func(item_data)

    # Execute the update
    update_func()

    # Restore selection
    if current_id:
        restore_selection_by_id(list_widget, current_id, get_identifier_func)


def restore_selection_by_id(
    list_widget: QListWidget,
    item_id: str,
    get_identifier_func: Callable[[Any], str]
):
    """
    Restore selection to an item by its identifier.

    Args:
        list_widget: The QListWidget to update
        item_id: Identifier of the item to select
        get_identifier_func: Function to extract unique ID from item data
    """
    for i in range(list_widget.count()):
        item = list_widget.item(i)
        item_data = item.data(Qt.ItemDataRole.UserRole)
        if item_data and get_identifier_func(item_data) == item_id:
            list_widget.setCurrentRow(i)
            logger.debug(f"Restored selection to: {item_id}")
            break


def handle_selection_change_with_prevention(
    list_widget: QListWidget,
    get_selected_func: Callable,
    get_identifier_func: Callable[[Any], str],
    should_preserve_func: Callable[[], bool],
    get_current_id_func: Callable[[], str],
    on_selected_func: Callable,
    on_cleared_func: Callable
):
    """
    Handle selection changes with automatic deselection prevention.

    Args:
        list_widget: The QListWidget to manage
        get_selected_func: Function that returns currently selected items
        get_identifier_func: Function to extract unique ID from item data
        should_preserve_func: Function that returns True if deselection should be prevented
        get_current_id_func: Function that returns the current selection ID
        on_selected_func: Function to call when items are selected (receives selected items)
        on_cleared_func: Function to call when selection is cleared (no args)
    """
    selected_items = get_selected_func()

    if selected_items:
        # Normal selection
        on_selected_func(selected_items)
    elif should_preserve_func():
        # Prevent deselection - re-select the current item
        current_id = get_current_id_func()
        if current_id:
            restore_selection_by_id(list_widget, current_id, get_identifier_func)
        return
    else:
        # Allow clearing selection
        on_cleared_func()
