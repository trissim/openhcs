"""
Plate List View Component for OpenHCS TUI.

This module defines the PlateListView class, responsible for displaying
the list of available plates.
"""
import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING, Coroutine

from prompt_toolkit.application import get_app
from prompt_toolkit.layout import HSplit, DynamicContainer, Dimension, ScrollablePane, Container
from prompt_toolkit.widgets import Label

from .interactive_list_item import InteractiveListItem

if TYPE_CHECKING:
    from openhcs.tui.interfaces import CorePlateData # For type hinting plate data

logger = logging.getLogger(__name__)

class PlateListView:
    """
    Displays a scrollable list of plates.
    """
    def __init__(self,
                 on_plate_selected: Callable[[Optional[Dict[str, Any]]], Coroutine[Any, Any, None]], # Callback when a plate is selected by user
                 on_plate_activated: Optional[Callable[[Dict[str, Any]], Coroutine[Any, Any, None]]] = None # Callback for activation (e.g. Enter)
                ):
        self.plates_display_data: List[Dict[str, Any]] = [] # Holds CorePlateData-like dicts
        self.selected_index: int = 0 # Index in self.plates_display_data
        self._internal_list_items_container: HSplit = HSplit([Label("Loading plates...")])

        self.on_plate_selected_callback = on_plate_selected
        self.on_plate_activated_callback = on_plate_activated

        # The main container for this view will be a DynamicContainer wrapping a ScrollablePane
        # which in turn wraps the HSplit of InteractiveListItems.
        self.container = DynamicContainer(lambda: ScrollablePane(self._internal_list_items_container))
        self._lock = asyncio.Lock() # To protect access to plates_display_data and selected_index

    async def update_plate_list(self, new_plates_data: List[Dict[str, Any]]):
        """
        Updates the list of plates to display.
        new_plates_data should be a list of CorePlateData-like dictionaries.
        """
        async with self._lock:
            self.plates_display_data = new_plates_data
            # Try to preserve selection or reset
            if not self.plates_display_data:
                self.selected_index = 0 # Or -1 to indicate no selection possible
            elif self.selected_index >= len(self.plates_display_data):
                self.selected_index = len(self.plates_display_data) - 1
            # If an item was previously selected, try to find it by ID and re-select
            # This logic might be better handled by the controller, which knows the active_plate_id

            await self._rebuild_ui_items()
            # After rebuilding, if a valid item is selected, notify controller.
            # This is important if the list update itself should confirm/change selection.
            await self._notify_selection_to_controller()


    async def set_selected_plate_by_id(self, plate_id: Optional[str]):
        """
        Sets the selected plate based on its ID.
        Called by the controller when TUIState.active_plate_id changes.
        """
        async with self._lock:
            if plate_id is None:
                if self.selected_index != 0 or not self.plates_display_data: # Check if change is needed
                    self.selected_index = 0 # Default to first or handle no selection
            else:
                found = False
                for i, plate_data in enumerate(self.plates_display_data):
                    if plate_data.get('id') == plate_id:
                        if self.selected_index != i:
                            self.selected_index = i
                        found = True
                        break
                if not found:
                    logger.warning(f"PlateListView: Plate ID '{plate_id}' not found in current list. Cannot select.")
                    # Optionally deselect or select first: self.selected_index = 0

            await self._rebuild_ui_items() # Rebuild to reflect selection change visually
            # No need to call _notify_selection_to_controller here as this is reacting to controller/TUIState


    async def _rebuild_ui_items(self):
        """
        Rebuilds the HSplit container with new InteractiveListItem widgets.
        This should be called whenever plates_display_data or selected_index changes.
        """
        item_widgets = []
        if not self.plates_display_data:
            item_widgets.append(Label("No plates available."))
        else:
            for i, plate_data_dict in enumerate(self.plates_display_data):
                is_selected = (i == self.selected_index)
                item_widget = InteractiveListItem(
                    item_data=plate_data_dict,
                    item_index=i,
                    is_selected=is_selected,
                    display_text_func=self._get_plate_display_text_for_item,
                    on_select=self._handle_list_item_selected, # Internal handler for when item is clicked
                    on_activate=self._handle_list_item_activated, # Internal handler for Enter
                    # Reordering not handled by PlateListView directly; controller would manage.
                    can_move_up=False,
                    can_move_down=False
                )
                item_widgets.append(item_widget)

        self._internal_list_items_container = HSplit(
            item_widgets if item_widgets else [Label(" ")],
            width=Dimension(weight=1), # Fill width
            height=Dimension(min=len(item_widgets) if item_widgets else 1) # Adjust height dynamically
        )
        if get_app().is_running: # Avoid invalidation if app not fully up
            get_app().invalidate()

    def _get_plate_display_text_for_item(self, plate_core_data_dict: Dict[str, Any], is_selected: bool) -> str:
        """
        Generates the display text for a single plate item.
        plate_core_data_dict is a dictionary conforming to CorePlateData.
        """
        plate_status = plate_core_data_dict.get('status', 'unknown')

        # Status symbols according to TUI specification
        status_symbols = {
            'uninitialized': '?',    # Not initialized yet
            'initialized': '-',      # Initialized but not compiled
            'compiled': '✓',         # Compiled and ready
            'completed': '✓',        # Successfully completed
            'running': 'o',          # Currently running
            'error': '!',            # Error state
            'unknown': '?'           # Default fallback
        }

        status_symbol = status_symbols.get(plate_status, '?')
        name = plate_core_data_dict.get('name', 'Unknown Plate')

        # Simplified path display with ^/v navigation symbols
        path_or_id = plate_core_data_dict.get('path', plate_core_data_dict.get('id', '[No Path/ID]'))
        if len(str(path_or_id)) > 30:
            path_str = "(...)" + str(path_or_id)[-25:]
        else:
            path_str = str(path_or_id)

        # Add navigation symbols as per specification
        nav_symbols = "^/v" if len(self.plates_display_data) > 1 else "   "

        return f"{status_symbol}| {nav_symbols} {name} | {path_str}"

    async def _handle_list_item_selected(self, index: int):
        """Internal handler when an InteractiveListItem is selected (e.g., by mouse click)."""
        async with self._lock:
            if 0 <= index < len(self.plates_display_data):
                if self.selected_index != index:
                    self.selected_index = index
                    await self._rebuild_ui_items() # Update visual selection
                    await self._notify_selection_to_controller() # Notify controller of user-driven change

    async def _handle_list_item_activated(self, index: int):
        """Internal handler when an InteractiveListItem is activated (e.g., by Enter key)."""
        async with self._lock:
            if 0 <= index < len(self.plates_display_data):
                if self.selected_index != index: # If activation also implies selection
                    self.selected_index = index
                    await self._rebuild_ui_items()
                    # Selection notification is primary, activation is secondary
                    await self._notify_selection_to_controller()

                if self.on_plate_activated_callback: # If a specific activation handler exists
                    selected_plate_data = self.plates_display_data[self.selected_index]
                    await self.on_plate_activated_callback(selected_plate_data)


    async def _notify_selection_to_controller(self):
        """Notifies the controller about the current selection."""
        selected_plate_data: Optional[Dict[str, Any]] = None
        if self.plates_display_data and 0 <= self.selected_index < len(self.plates_display_data):
            selected_plate_data = self.plates_display_data[self.selected_index]

        if self.on_plate_selected_callback:
            await self.on_plate_selected_callback(selected_plate_data)

    def get_selected_item_data(self) -> Optional[Dict[str, Any]]:
        """Returns the data of the currently selected item, if any."""
        if self.plates_display_data and 0 <= self.selected_index < len(self.plates_display_data):
            return self.plates_display_data[self.selected_index]
        return None

    def __pt_container__(self) -> Container:
        return self.container

