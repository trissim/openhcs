"""
Simple channel selection dialog with dual lists.

Mathematical approach: move items between available and selected lists.
"""

import logging
from typing import List, Callable, Optional

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static, ListView, ListItem, Label

logger = logging.getLogger(__name__)


class ChannelSelectionDialog(ModalScreen):
    """
    Simple dual-list channel selection dialog.

    Mathematical operations:
    - Move →: selected += highlighted_available
    - ← Move: selected -= highlighted_selected
    - Select All: selected = available
    - None: selected = []
    """

    CSS = """
    ChannelSelectionDialog {
        align: center middle;
    }

    #channel_dialog {
        width: 60;
        height: 18;
    }

    #button_row {
        height: auto;
        align: center middle;
    }

    #button_row Button {
        width: auto;
        min-width: 8;
        margin: 0 1;
    }

    #lists_row {
        height: 1fr;
    }

    #available_panel, #selected_panel {
        width: 1fr;
    }

    #available_list, #selected_list {
        height: 1fr;
        width: 1fr;
    }

    #action_row {
        height: auto;
        align: center middle;
    }

    #action_row Button {
        width: auto;
        min-width: 8;
        margin: 0 1;
    }
    """
    
    def __init__(
        self, 
        available_channels: List[int], 
        selected_channels: List[int],
        callback: Callable[[Optional[List[int]]], None]
    ):
        super().__init__()
        self.available_channels = available_channels.copy()
        self.selected_channels = selected_channels.copy()
        self.callback = callback
        
        # Calculate initial lists
        self.current_available = [ch for ch in self.available_channels if ch not in self.selected_channels]
        self.current_selected = self.selected_channels.copy()
        
        logger.debug(f"Channel dialog: available={self.current_available}, selected={self.current_selected}")

    def compose(self) -> ComposeResult:
        """Compose the simple dual-list dialog."""
        with Container(id="channel_dialog") as container:
            container.styles.border = ("solid", "white")
            yield Static("Select Channels", id="dialog_title")
            
            # Button row
            with Horizontal(id="button_row"):
                yield Button("Move →", id="move_right", compact=True)
                yield Button("← Move", id="move_left", compact=True)
                yield Button("Select All", id="select_all", compact=True)
                yield Button("None", id="select_none", compact=True)
            
            # Dual lists
            with Horizontal(id="lists_row"):
                with Vertical(id="available_panel"):
                    yield Static("Available", id="available_label")
                    yield ListView(id="available_list")
                
                with Vertical(id="selected_panel"):
                    yield Static("Selected", id="selected_label")
                    yield ListView(id="selected_list")
            
            # Action buttons
            with Horizontal(id="action_row"):
                yield Button("OK", id="ok_btn", compact=True)
                yield Button("Cancel", id="cancel_btn", compact=True)

    def on_mount(self) -> None:
        """Initialize the lists."""
        self._update_lists()

    def _update_lists(self) -> None:
        """Update both list views with current data."""
        # Update available list
        available_list = self.query_one("#available_list", ListView)
        available_list.clear()
        for channel in sorted(self.current_available):
            available_list.append(ListItem(Label(f"Channel {channel}")))
        
        # Update selected list
        selected_list = self.query_one("#selected_list", ListView)
        selected_list.clear()
        for channel in sorted(self.current_selected):
            selected_list.append(ListItem(Label(f"Channel {channel}")))
        
        logger.debug(f"Updated lists: available={self.current_available}, selected={self.current_selected}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses with simple mathematical operations."""
        if event.button.id == "move_right":
            self._move_right()
        elif event.button.id == "move_left":
            self._move_left()
        elif event.button.id == "select_all":
            self._select_all()
        elif event.button.id == "select_none":
            self._select_none()
        elif event.button.id == "ok_btn":
            self._handle_ok()
        elif event.button.id == "cancel_btn":
            self._handle_cancel()

    def _move_right(self) -> None:
        """Move highlighted available channels to selected."""
        available_list = self.query_one("#available_list", ListView)
        if available_list.index is not None and 0 <= available_list.index < len(self.current_available):
            channel = self.current_available[available_list.index]
            self.current_available.remove(channel)
            self.current_selected.append(channel)
            self._update_lists()
            logger.debug(f"Moved channel {channel} to selected")

    def _move_left(self) -> None:
        """Move highlighted selected channels to available."""
        selected_list = self.query_one("#selected_list", ListView)
        if selected_list.index is not None and 0 <= selected_list.index < len(self.current_selected):
            channel = self.current_selected[selected_list.index]
            self.current_selected.remove(channel)
            self.current_available.append(channel)
            self._update_lists()
            logger.debug(f"Moved channel {channel} to available")

    def _select_all(self) -> None:
        """Select all available channels."""
        self.current_selected = self.available_channels.copy()
        self.current_available = []
        self._update_lists()
        logger.debug("Selected all channels")

    def _select_none(self) -> None:
        """Deselect all channels."""
        self.current_available = self.available_channels.copy()
        self.current_selected = []
        self._update_lists()
        logger.debug("Deselected all channels")

    def _handle_ok(self) -> None:
        """Handle OK button - return selected channels."""
        self.callback(self.current_selected)
        self.dismiss()

    def _handle_cancel(self) -> None:
        """Handle Cancel button - return None."""
        self.callback(None)
        self.dismiss()
