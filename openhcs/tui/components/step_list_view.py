"""
Step List View Component for OpenHCS TUI.

This module defines the StepListView class, responsible for displaying
the list of steps in the active pipeline.
"""
import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING, Coroutine

from prompt_toolkit.application import get_app
from prompt_toolkit.layout import HSplit, DynamicContainer, Dimension, ScrollablePane, Container
from prompt_toolkit.widgets import Label

from .interactive_list_item import InteractiveListItem

if TYPE_CHECKING:
    from openhcs.tui.interfaces import CoreStepData # For type hinting step data

logger = logging.getLogger(__name__)

class StepListView:
    """
    Displays a scrollable list of pipeline steps.
    """
    def __init__(self,
                 on_step_selected: Callable[[Optional[Dict[str, Any]]], Coroutine[Any, Any, None]],
                 on_step_activated: Optional[Callable[[Dict[str, Any]], Coroutine[Any, Any, None]]] = None,
                 # Callback for reordering requests (index, direction "up" or "down")
                 on_step_reorder_requested: Optional[Callable[[int, str], Coroutine[Any, Any, None]]] = None
                ):
        self.steps_display_data: List[Dict[str, Any]] = [] # Holds CoreStepData-like dicts
        self.selected_index: int = 0
        self._internal_list_items_container: HSplit = HSplit([Label("Loading steps...")])

        self.on_step_selected_callback = on_step_selected
        self.on_step_activated_callback = on_step_activated
        self.on_step_reorder_requested_callback = on_step_reorder_requested

        self.container = DynamicContainer(lambda: ScrollablePane(self._internal_list_items_container))
        self._lock = asyncio.Lock()

    async def update_step_list(self, new_steps_data: List[Dict[str, Any]]):
        """
        Updates the list of steps to display.
        new_steps_data should be a list of CoreStepData-like dictionaries.
        """
        async with self._lock:
            self.steps_display_data = new_steps_data
            if not self.steps_display_data:
                self.selected_index = 0
            elif self.selected_index >= len(self.steps_display_data):
                self.selected_index = len(self.steps_display_data) - 1

            await self._rebuild_ui_items()
            await self._notify_selection_to_controller()

    async def set_selected_step_by_id(self, step_id: Optional[str]):
        """
        Sets the selected step based on its ID.
        Called by the controller when TUIState.selected_step changes.
        """
        async with self._lock:
            if step_id is None:
                if self.selected_index != 0 or not self.steps_display_data: # Default to first or handle no selection
                    self.selected_index = 0
            else:
                found = False
                for i, step_data in enumerate(self.steps_display_data):
                    if step_data.get('id') == step_id:
                        if self.selected_index != i:
                            self.selected_index = i
                        found = True
                        break
                if not found:
                    logger.warning(f"StepListView: Step ID '{step_id}' not found. Cannot select.")

            await self._rebuild_ui_items() # Update visual selection

    async def _rebuild_ui_items(self):
        """Rebuilds the HSplit container with InteractiveListItem widgets for steps."""
        item_widgets = []
        if not self.steps_display_data:
            item_widgets.append(Label("No steps in pipeline. Use 'Add' or 'Load'."))
        else:
            for i, step_data_dict in enumerate(self.steps_display_data):
                is_selected = (i == self.selected_index)
                item_widget = InteractiveListItem(
                    item_data=step_data_dict,
                    item_index=i,
                    is_selected=is_selected,
                    display_text_func=self._get_step_display_text_for_item,
                    on_select=self._handle_list_item_selected,
                    on_activate=self._handle_list_item_activated,
                    on_move_up=self._handle_list_item_move_up if self.on_step_reorder_requested_callback else None,
                    on_move_down=self._handle_list_item_move_down if self.on_step_reorder_requested_callback else None,
                    can_move_up=(i > 0), # Basic condition, controller might add more logic
                    can_move_down=(i < len(self.steps_display_data) - 1)
                )
                item_widgets.append(item_widget)

        self._internal_list_items_container = HSplit(
            item_widgets if item_widgets else [Label(" ")],
            width=Dimension(weight=1),
            height=Dimension(min=len(item_widgets) if item_widgets else 1)
        )
        if get_app().is_running:
            get_app().invalidate()

    def _get_step_display_text_for_item(self, step_core_data_dict: Dict[str, Any], is_selected: bool) -> str:
        """Generates display text for a step from its CoreStepData-like dict."""
        status = step_core_data_dict.get('status', 'unknown')

        # Status symbols according to TUI specification
        status_symbols = {
            'pending': '?',          # Not yet executed
            'ready': 'o',            # Ready to execute
            'running': 'o',          # Currently running
            'completed': '✓',        # Successfully completed
            'error': '!',            # Error state
            'unknown': '?'           # Default fallback
        }

        status_symbol = status_symbols.get(status, '?')
        name = step_core_data_dict.get('name', 'Unknown Step')

        # Get step type or function info
        step_type = step_core_data_dict.get('type', step_core_data_dict.get('function_identifier', 'Unknown'))

        # Add navigation symbols as per specification
        nav_symbols = "^/v" if len(self.steps_display_data) > 1 else "   "

        return f"{status_symbol}| {nav_symbols} {name}: {step_type}"

    async def _handle_list_item_selected(self, index: int):
        """Internal handler when an item is selected (e.g., by mouse click)."""
        async with self._lock:
            if 0 <= index < len(self.steps_display_data):
                if self.selected_index != index:
                    self.selected_index = index
                    await self._rebuild_ui_items()
                    await self._notify_selection_to_controller()

    async def _handle_list_item_activated(self, index: int):
        """Internal handler when an item is activated (e.g., by Enter key)."""
        async with self._lock:
            if 0 <= index < len(self.steps_display_data):
                if self.selected_index != index: # If activation also implies selection
                    self.selected_index = index
                    await self._rebuild_ui_items()
                    await self._notify_selection_to_controller()

                if self.on_step_activated_callback:
                    selected_step_data = self.steps_display_data[self.selected_index]
                    await self.on_step_activated_callback(selected_step_data)

    async def _handle_list_item_move_up(self, index: int):
        if self.on_step_reorder_requested_callback:
            await self.on_step_reorder_requested_callback(index, "up")

    async def _handle_list_item_move_down(self, index: int):
        if self.on_step_reorder_requested_callback:
            await self.on_step_reorder_requested_callback(index, "down")

    async def _notify_selection_to_controller(self):
        """Notifies the controller about the current selection."""
        selected_step_data: Optional[Dict[str, Any]] = None
        if self.steps_display_data and 0 <= self.selected_index < len(self.steps_display_data):
            selected_step_data = self.steps_display_data[self.selected_index]

        if self.on_step_selected_callback:
            await self.on_step_selected_callback(selected_step_data)

    def get_selected_item_data(self) -> Optional[Dict[str, Any]]:
        if self.steps_display_data and 0 <= self.selected_index < len(self.steps_display_data):
            return self.steps_display_data[self.selected_index]
        return None

    def __pt_container__(self) -> Container:
        return self.container