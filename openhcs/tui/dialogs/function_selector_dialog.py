"""
Function selector dialog with search functionality.

Provides a searchable interface for selecting functions from the registry.
"""

import time
from typing import Dict, List, Tuple, Callable, Optional

from prompt_toolkit.application import get_app
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import HSplit, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout import ScrollablePane
from prompt_toolkit.mouse_events import MouseEventType
from prompt_toolkit.widgets import TextArea, Dialog, Button, Box, Frame
import logging

logger = logging.getLogger(__name__)


class FunctionSelectorDialog:
    """
    Dialog for selecting functions with search functionality.
    
    Provides a searchable list of all available functions organized by backend.
    """
    
    def __init__(self, functions_by_backend: Dict[str, List[Tuple[Callable, str]]], 
                 current_func: Optional[Callable], 
                 on_selection: Callable):
        """
        Initialize function selector dialog.

        Args:
            functions_by_backend: Dict of backend -> [(func, name), ...]
            current_func: Currently selected function
            on_selection: Callback when function is selected
        """
        self.functions_by_backend = functions_by_backend
        self.current_func = current_func
        self.on_selection = on_selection
        
        # Build flat function list for selection
        self.function_list = self._build_function_list()
        self.filtered_list = self.function_list.copy()
        
        # UI state
        self.focused_index = 0
        self.search_text = ""
        self.last_click_time = 0
        self.last_click_index = -1
        
        # Find current selection
        self._find_current_selection()
        
        # Build dialog components
        self.search_field = TextArea(
            text="",
            prompt="Search functions...",
            height=1,
            multiline=False,
            focusable=True
        )
        
        # Bind search field changes
        self.search_field.buffer.on_text_changed += self._on_search_changed
        
        self.dialog = None
    
    def _build_function_list(self) -> List[Tuple[Callable, str, str]]:
        """Build flat list of (func, name, backend) for selection."""
        function_list = []
        for backend, funcs in sorted(self.functions_by_backend.items()):
            for func, name in funcs:
                function_list.append((func, name, backend))
        return function_list

    def _find_current_selection(self):
        """Find the index of the currently selected function."""
        for i, (func, name, backend) in enumerate(self.filtered_list):
            if func == self.current_func:
                self.focused_index = i
                break
    
    def _on_search_changed(self, buffer):
        """Handle search text changes."""
        self.search_text = buffer.text.lower()
        self._filter_functions()
        self.focused_index = 0  # Reset to top
        self._find_current_selection()  # Try to keep current selection visible
        get_app().invalidate()
    
    def _filter_functions(self):
        """Filter function list based on search text."""
        if not self.search_text:
            self.filtered_list = self.function_list.copy()
        else:
            self.filtered_list = [
                (func, name, backend) for func, name, backend in self.function_list
                if self.search_text in name.lower() or self.search_text in backend.lower()
            ]
    
    def _build_function_list_container(self):
        """Build the scrollable function list."""
        list_items = []
        
        for i, (func, name, backend) in enumerate(self.filtered_list):
            is_focused = (i == self.focused_index)
            is_current = (func == self.current_func)
            
            # Style based on state
            if is_current:
                style = "class:dialog-list-selected"
                prefix = "● "
            elif is_focused:
                style = "class:dialog-list-focused"
                prefix = "▶ "
            else:
                style = "class:dialog-list-item"
                prefix = "  "
            
            display_text = f"{prefix}{name} ({backend})"
            
            # Create control with mouse handler
            control = FormattedTextControl(
                display_text,
                focusable=False,
            )
            
            # Mouse handler for this item
            def make_handler(item_index):
                def handler(mouse_event):
                    if mouse_event.event_type == MouseEventType.MOUSE_UP:
                        current_time = time.time()
                        
                        # Check for double-click
                        is_double_click = (
                            item_index == self.last_click_index and
                            current_time - self.last_click_time < 0.5
                        )
                        
                        # Update click tracking
                        self.last_click_time = current_time
                        self.last_click_index = item_index
                        
                        if is_double_click:
                            # Double click - select and close
                            self._select_function(item_index)
                        else:
                            # Single click - just focus
                            self.focused_index = item_index
                            get_app().invalidate()
                        
                        return True
                    return False
                return handler
            
            control.mouse_handler = make_handler(i)
            
            list_items.append(
                Box(
                    Window(control, height=1),
                    style=style
                )
            )
        
        return ScrollablePane(HSplit(list_items))
    
    def _select_function(self, item_index: int):
        """Select a function and close the dialog."""
        if 0 <= item_index < len(self.filtered_list):
            func, name, backend = self.filtered_list[item_index]

            # Call selection callback
            self.on_selection(func)

            # Close dialog by completing the future
            if self.result_future and not self.result_future.done():
                self.result_future.set_result(func)
    
    async def show(self, app_state):
        """Show the function selector dialog using proper app_state integration."""
        import asyncio
        from prompt_toolkit.layout.dimension import Dimension

        # Create result future for dialog completion
        self.result_future = asyncio.Future()

        # Build dialog content
        content = HSplit([
            # Search field
            Frame(self.search_field, title="Search"),
            # Function list with fixed height for testing
            Frame(
                self._build_function_list_container(),
                title="Functions",
                style="class:dialog-content",
                height=Dimension(min=10, preferred=15)
            )
        ])

        # Create dialog with buttons - let prompt-toolkit size naturally
        self.dialog = Dialog(
            title="Select Function",
            body=content,
            buttons=[
                Button("Select", handler=self._handle_select),
                Button("Cancel", handler=self._handle_cancel)
            ],
            modal=True
        )

        # Show dialog using proper app_state method
        if hasattr(app_state, 'show_dialog') and callable(app_state.show_dialog):
            await app_state.show_dialog(self.dialog, result_future=self.result_future)
        else:
            logger.error("app_state does not have show_dialog method")
            self.result_future.set_result(None)

        # Wait for dialog completion
        return await self.result_future
    
    def _handle_select(self):
        """Handle select button click."""
        if 0 <= self.focused_index < len(self.filtered_list):
            self._select_function(self.focused_index)
        else:
            # No selection, just close
            if self.result_future and not self.result_future.done():
                self.result_future.set_result(None)

    def _handle_cancel(self):
        """Handle cancel button click."""
        if self.result_future and not self.result_future.done():
            self.result_future.set_result(None)
