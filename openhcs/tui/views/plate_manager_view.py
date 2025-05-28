"""
Plate Manager View Component.

Pure UI component responsible only for rendering and user input handling.
Delegates all business logic to the controller.

🔒 Clause 295: Component Boundaries
Clear separation between view and controller layers.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from prompt_toolkit.application import get_app
from prompt_toolkit.layout import Container, HSplit, VSplit, Window, Dimension
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.widgets import Box, Label
from prompt_toolkit.mouse_events import MouseEventType

from openhcs.tui.components import InteractiveListItem, FramedButton

logger = logging.getLogger(__name__)


class PlateManagerView:
    """
    View component for the plate manager.
    
    Handles:
    - UI rendering
    - User input events
    - Visual state updates
    """
    
    def __init__(self, controller):
        self.controller = controller
        
        # UI components
        self.title_label = None
        self.add_button = None
        self.remove_button = None
        self.refresh_button = None
        self.plate_list_container = None
        self.container = None
        
        # UI state
        self.plates = []
        self.selected_index = 0
        self.is_loading = False
        
        # Initialize UI
        self._build_ui()
        
        # Register for UI updates
        self.controller.state.add_observer('plate_manager_ui_update', self._handle_ui_update)
    
    def _build_ui(self):
        """Build the UI components."""
        # Title bar
        self.title_label = Label("📁 Plate Manager", style="class:title")
        
        # Action buttons
        self.add_button = FramedButton(
            text="Add",
            handler=self._handle_add_button,
            style="class:button"
        )
        
        self.remove_button = FramedButton(
            text="Remove",
            handler=self._handle_remove_button,
            style="class:button"
        )
        
        self.refresh_button = FramedButton(
            text="Refresh",
            handler=self._handle_refresh_button,
            style="class:button"
        )
        
        # Button bar
        button_bar = VSplit([
            self.add_button,
            Window(width=1),  # Spacer
            self.remove_button,
            Window(width=1),  # Spacer
            self.refresh_button,
        ])
        
        # Plate list (initially empty)
        self.plate_list_container = HSplit([])
        
        # Main container
        self.container = Box(
            HSplit([
                self.title_label,
                Window(height=1),  # Spacer
                button_bar,
                Window(height=1),  # Spacer
                self.plate_list_container,
            ]),
            style="class:plate-manager"
        )
    
    async def _handle_ui_update(self, data):
        """Handle UI update events from the controller."""
        self.plates = data.get('plates', [])
        self.selected_index = data.get('selected_index', 0)
        self.is_loading = data.get('is_loading', False)
        
        await self._update_plate_list()
    
    async def _update_plate_list(self):
        """Update the plate list display."""
        if self.is_loading:
            self.plate_list_container.children = [
                Window(
                    content=FormattedTextControl("Loading plates..."),
                    height=1
                )
            ]
            return
        
        if not self.plates:
            self.plate_list_container.children = [
                Window(
                    content=FormattedTextControl("No plates found. Use [Add] to create a new plate."),
                    height=1
                )
            ]
            return
        
        # Create list items
        list_items = []
        for i, plate in enumerate(self.plates):
            is_selected = (i == self.selected_index)
            item = self._create_plate_list_item(plate, i, is_selected)
            list_items.append(item)
        
        self.plate_list_container.children = list_items
        
        # Invalidate to trigger redraw
        app = get_app()
        app.invalidate()
    
    def _create_plate_list_item(self, plate: Dict[str, Any], index: int, is_selected: bool) -> Container:
        """Create a list item for a plate."""
        # Format plate display text
        name = plate.get('name', 'Unknown')
        status = plate.get('status', 'unknown')
        path = plate.get('path', '')
        
        # Create status indicator
        status_indicator = self._get_status_indicator(status)
        
        # Format path for display
        display_path = self._format_path_for_display(path)
        
        # Create the display text
        display_text = f"{status_indicator} {name} ({display_path})"
        
        # Create interactive list item
        item = InteractiveListItem(
            text=display_text,
            selected=is_selected,
            on_click=lambda: self._handle_plate_click(index),
            style="class:plate-item"
        )
        
        return item
    
    def _get_status_indicator(self, status: str) -> str:
        """Get a visual indicator for the plate status."""
        status_indicators = {
            'ready': '✅',
            'not_initialized': '⚪',
            'initializing': '🔄',
            'error': '❌',
            'running': '🔄',
            'completed': '✅',
            'info': 'ℹ️'
        }
        return status_indicators.get(status, '❓')
    
    def _format_path_for_display(self, path: str) -> str:
        """Format a path for display in the UI."""
        if not path or path == 'N/A':
            return "[No Path]"
        
        # Truncate long paths
        max_length = 40
        if len(path) > max_length:
            return f"...{path[-(max_length-3):]}"
        
        return path
    
    async def _handle_plate_click(self, index: int):
        """Handle clicking on a plate item."""
        await self.controller.select_plate(index)
    
    async def _handle_add_button(self):
        """Handle add button click."""
        await self.controller.show_add_plate_dialog()
    
    async def _handle_remove_button(self):
        """Handle remove button click."""
        selected_plate = await self.controller.get_selected_plate()
        if selected_plate:
            plate_id = selected_plate.get('id')
            if plate_id:
                await self.controller.remove_selected_plates([plate_id])
    
    async def _handle_refresh_button(self):
        """Handle refresh button click."""
        await self.controller.refresh_plates()
    
    def get_container(self) -> Container:
        """Get the main UI container."""
        return self.container
    
    def handle_key(self, key_event):
        """Handle keyboard input."""
        key = key_event.key
        
        if key == 'up':
            self._move_selection(-1)
        elif key == 'down':
            self._move_selection(1)
        elif key == 'enter':
            app = get_app()
            app.create_background_task(self._handle_plate_click(self.selected_index))
        elif key == 'a':
            app = get_app()
            app.create_background_task(self._handle_add_button())
        elif key == 'r':
            app = get_app()
            app.create_background_task(self._handle_refresh_button())
        elif key == 'delete':
            app = get_app()
            app.create_background_task(self._handle_remove_button())
    
    def _move_selection(self, delta: int):
        """Move the selection up or down."""
        if not self.plates:
            return
        
        new_index = max(0, min(len(self.plates) - 1, self.selected_index + delta))
        if new_index != self.selected_index:
            app = get_app()
            app.create_background_task(self.controller.select_plate(new_index))
