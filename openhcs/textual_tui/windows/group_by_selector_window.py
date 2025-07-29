"""
Group-by selector window with dual lists for component selection.

Mathematical approach: move items between available and selected lists.
"""

import logging
from typing import List, Callable, Optional

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Static, ListView, ListItem, Label

from openhcs.textual_tui.windows.base_window import BaseOpenHCSWindow

logger = logging.getLogger(__name__)


class GroupBySelectorWindow(BaseOpenHCSWindow):
    """
    Window for selecting components with dual-list interface.

    Mathematical operations:
    - Move →: selected += highlighted_available
    - ← Move: selected -= highlighted_selected
    - Select All: selected = available
    - None: selected = []
    """

    DEFAULT_CSS = """
    GroupBySelectorWindow {
        width: 50; height: 20;
        min-width: 50; min-height: 20;
    }

    /* Content area takes most space (like ConfigWindow) */
    GroupBySelectorWindow .dialog-content {
        height: 1fr;
        width: 100%;
    }

    /* Top buttons row */
    GroupBySelectorWindow #top_buttons {
        height: auto;
        align: center middle;
        width: 100%;
    }

    /* Lists container fills remaining content space */
    GroupBySelectorWindow #lists_container {
        height: 1fr;
        width: 100%;
    }

    /* Bottom buttons row - same as top buttons */
    GroupBySelectorWindow #bottom_buttons {
        height: auto;
        align: center middle;
        width: 100%;
    }

    /* All buttons styling */
    GroupBySelectorWindow Button {
        width: auto;
        margin: 0 1;
    }

    /* Top buttons more compact */
    GroupBySelectorWindow #top_buttons Button {
        min-width: 4;
    }

    /* Bottom buttons standard size */
    GroupBySelectorWindow #bottom_buttons Button {
        min-width: 8;
    }

    /* List views fill container */
    GroupBySelectorWindow ListView {
        height: 1fr;
        min-height: 3;
    }

    /* Static labels */
    GroupBySelectorWindow Static {
        height: auto;
        padding: 0;
        margin: 0;
    }
    """

    def __init__(
        self,
        available_channels: List[str],
        selected_channels: List[str],
        on_result_callback: Callable[[Optional[List[str]]], None],
        component_type: str = "channel",  # Parameter for dynamic component type
        orchestrator=None,  # Add orchestrator for metadata access
        **kwargs
    ):
        self.available_channels = available_channels.copy()
        self.selected_channels = selected_channels.copy()
        self.on_result_callback = on_result_callback
        self.component_type = component_type  # Store for dynamic labels
        self.orchestrator = orchestrator  # Store for metadata access

        # Calculate initial lists
        self.current_available = [ch for ch in self.available_channels if ch not in self.selected_channels]
        self.current_selected = self.selected_channels.copy()

        logger.debug(f"{component_type.title()} dialog: available={self.current_available}, selected={self.current_selected}")

        # Dynamic title based on component type
        title = f"Select {self.component_type.title()}s"
        
        super().__init__(
            window_id="group_by_selector",
            title=title,
            mode="temporary",
            **kwargs
        )

    def _format_component_display(self, component_key: str) -> str:
        """
        Format component key for display with metadata if available.

        Args:
            component_key: Component key (e.g., "1", "2", "A01")

        Returns:
            Formatted display string (e.g., "Channel 1 | HOECHST 33342" or "Channel 1")
        """
        base_text = f"{self.component_type.title()} {component_key}"

        # Get metadata name if orchestrator is available
        if self.orchestrator:
            # Convert component_type string back to GroupBy enum
            from openhcs.constants.constants import GroupBy
            group_by = GroupBy(self.component_type)
            metadata_name = self.orchestrator.get_component_metadata(group_by, component_key)

            if metadata_name:
                return f"{base_text} | {metadata_name}"

        return base_text

    def _extract_channel_from_display(self, display_text: str) -> Optional[str]:
        """Extract the channel key from formatted display text.

        Display text format: "Channel 1 | HOECHST 33342" or "Channel 1"
        Returns: "1"
        """
        try:
            # Split by the first space to get "Channel" and "1 | ..." or "1"
            parts = display_text.split(' ', 2)
            if len(parts) >= 2:
                # Get the part after "Channel" which is "1" or "1 | metadata"
                key_part = parts[1]
                # Split by " | " to separate key from metadata
                key = key_part.split(' | ')[0]
                return key
        except Exception as e:
            logger.debug(f"Could not extract channel from display text '{display_text}': {e}")
        return None

    def compose(self) -> ComposeResult:
        """Compose the dual-list selector window - follow working window pattern."""
        # Content area (like ConfigWindow does)
        with Container(classes="dialog-content"):
            # Top button row
            with Horizontal(id="top_buttons"):
                yield Button("→", id="move_right", compact=True)
                yield Button("←", id="move_left", compact=True)
                yield Button("All", id="select_all", compact=True)
                yield Button("None", id="select_none", compact=True)

            # Dual lists
            with Horizontal(id="lists_container"):
                with Vertical():
                    yield Static("Available")
                    yield ListView(id="available_list")

                with Vertical():
                    yield Static("Selected")
                    yield ListView(id="selected_list")

        # Bottom buttons in horizontal row (like top buttons)
        with Horizontal(id="bottom_buttons"):
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
            # Use enhanced formatting with metadata
            label_text = self._format_component_display(channel)
            available_list.append(ListItem(Label(label_text)))

        # Update selected list
        selected_list = self.query_one("#selected_list", ListView)
        selected_list.clear()
        for channel in sorted(self.current_selected):
            # Use enhanced formatting with metadata
            label_text = self._format_component_display(channel)
            selected_list.append(ListItem(Label(label_text)))

        # Clear selections to prevent stale index issues
        available_list.index = None
        selected_list.index = None

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
        if available_list.index is not None and 0 <= available_list.index < len(available_list.children):
            # Get the actual selected item from the ListView, not from our sorted list
            selected_item = available_list.children[available_list.index]
            if hasattr(selected_item, 'children') and selected_item.children:
                label = selected_item.children[0]  # Get the Label widget
                if hasattr(label, 'renderable'):
                    # Extract the channel key from the formatted display text
                    display_text = str(label.renderable)
                    channel = self._extract_channel_from_display(display_text)

                    if channel and channel in self.current_available:
                        self.current_available.remove(channel)
                        self.current_selected.append(channel)
                        self._update_lists()
                        logger.debug(f"Moved {self.component_type} {channel} to selected")

    def _move_left(self) -> None:
        """Move highlighted selected channels to available."""
        selected_list = self.query_one("#selected_list", ListView)
        if selected_list.index is not None and 0 <= selected_list.index < len(selected_list.children):
            # Get the actual selected item from the ListView, not from our sorted list
            selected_item = selected_list.children[selected_list.index]
            if hasattr(selected_item, 'children') and selected_item.children:
                label = selected_item.children[0]  # Get the Label widget
                if hasattr(label, 'renderable'):
                    # Extract the channel key from the formatted display text
                    display_text = str(label.renderable)
                    channel = self._extract_channel_from_display(display_text)

                    if channel and channel in self.current_selected:
                        self.current_selected.remove(channel)
                        self.current_available.append(channel)
                        self._update_lists()
                        logger.debug(f"Moved {self.component_type} {channel} to available")

    def _select_all(self) -> None:
        """Select all available channels."""
        self.current_selected = self.available_channels.copy()
        self.current_available = []
        self._update_lists()
        logger.debug(f"Selected all {self.component_type}s")

    def _select_none(self) -> None:
        """Deselect all channels."""
        self.current_available = self.available_channels.copy()
        self.current_selected = []
        self._update_lists()
        logger.debug(f"Deselected all {self.component_type}s")

    def _handle_ok(self) -> None:
        """Handle OK button - return selected channels."""
        if self.on_result_callback:
            self.on_result_callback(self.current_selected)
        self.close_window()

    def _handle_cancel(self) -> None:
        """Handle Cancel button - return None."""
        if self.on_result_callback:
            self.on_result_callback(None)
        self.close_window()
