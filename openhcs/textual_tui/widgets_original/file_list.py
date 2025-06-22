"""File list widget for directory browsing."""

import logging
from pathlib import Path
from typing import List, Optional, Callable
from textual.containers import ScrollableContainer
from textual.widgets import Static
from textual.app import ComposeResult
from textual.reactive import reactive
from textual import events

from openhcs.constants.constants import Backend
from openhcs.textual_tui.services.file_browser_service import FileItem, FileBrowserService, SelectionMode

logger = logging.getLogger(__name__)


class FileListWidget(ScrollableContainer):
    """Widget for displaying and navigating file listings."""
    
    current_path = reactive(Path)
    listing = reactive(list)
    focused_index = reactive(0)
    backend = reactive(Backend.DISK)  # Use Backend enum, not string
    
    def __init__(self, 
                 file_browser_service: FileBrowserService,
                 initial_path: Path,
                 backend: Backend = Backend.DISK,
                 selection_mode: SelectionMode = SelectionMode.FILES_ONLY,
                 on_selection: Optional[Callable] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.file_browser_service = file_browser_service
        self.current_path = initial_path
        self.backend = backend
        self.selection_mode = selection_mode
        self.on_selection = on_selection
        self.listing = []
        self.focused_index = 0
        
        # Load initial directory (display will be updated on mount)
        self.listing = self.file_browser_service.load_directory(self.current_path, self.backend)
        self.focused_index = 0
        logger.debug(f"FileListWidget: Loaded {len(self.listing)} items from {self.current_path}")
    
    def compose(self) -> ComposeResult:
        """Compose the file list."""
        # Create a single static widget that we'll update dynamically
        yield Static(self._get_listing_display(), id="file_listing")

    def on_mount(self) -> None:
        """Update display when widget is mounted."""
        self._update_display()
    
    def _get_listing_display(self) -> str:
        """Get the formatted listing display."""
        if not self.listing:
            return "[center](empty directory)[/center]"

        lines = []
        for i, item in enumerate(self.listing):
            icon = "📁" if item.is_dir else "📄"
            prefix = ">" if i == self.focused_index else " "
            style = "[reverse]" if i == self.focused_index else ""
            end_style = "[/reverse]" if i == self.focused_index else ""
            lines.append(f"{prefix} {style}{icon} {item.name}{end_style}")

        return "\n".join(lines)

    def load_current_directory(self) -> None:
        """Load the current directory contents."""
        self.listing = self.file_browser_service.load_directory(self.current_path, self.backend)
        self.focused_index = 0
        self._update_display()
    
    def navigate_to(self, path: Path) -> None:
        """Navigate to a new directory."""
        self.current_path = path
        self.load_current_directory()
    
    def navigate_up(self) -> None:
        """Navigate to parent directory."""
        parent = self.current_path.parent
        if parent != self.current_path:
            self.navigate_to(parent)
    
    def get_focused_item(self) -> Optional[FileItem]:
        """Get the currently focused item."""
        if 0 <= self.focused_index < len(self.listing):
            return self.listing[self.focused_index]
        return None
    
    def can_select_focused(self) -> bool:
        """Check if focused item can be selected."""
        item = self.get_focused_item()
        if item is None:
            return False
        return self.file_browser_service.can_select_item(item, self.selection_mode)
    
    def on_key(self, event: events.Key) -> None:
        """Handle keyboard navigation."""
        if event.key == "up":
            self.focused_index = max(0, self.focused_index - 1)
            self._update_display()
        elif event.key == "down":
            self.focused_index = min(len(self.listing) - 1, self.focused_index + 1)
            self._update_display()
        elif event.key == "enter":
            self._handle_selection()
        elif event.key == "backspace":
            self.navigate_up()
    
    def _handle_selection(self) -> None:
        """Handle item selection."""
        item = self.get_focused_item()
        if item is None:
            return
        
        if item.is_dir:
            # Navigate into directory
            self.navigate_to(item.path)
        elif self.can_select_focused():
            # Select file
            if self.on_selection:
                self.on_selection(item.path)
    
    def _update_display(self) -> None:
        """Update the file listing display."""
        try:
            listing_widget = self.query_one("#file_listing", Static)
            display_text = self._get_listing_display()
            listing_widget.update(display_text)
            logger.debug(f"FileListWidget: Updated display with {len(self.listing)} items")
        except Exception as e:
            logger.debug(f"FileListWidget: Failed to update display: {e}")
            # Widget might not be mounted yet
            pass

    def watch_current_path(self, new_path: Path) -> None:
        """React to path changes."""
        self.load_current_directory()

    def watch_listing(self, new_listing: List[FileItem]) -> None:
        """React to listing changes."""
        self._update_display()

    def watch_focused_index(self, new_index: int) -> None:
        """React to focus changes."""
        self._update_display()
