"""File browser modal screen for OpenHCS Textual TUI."""

from pathlib import Path
from typing import Optional, Callable, List
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, Static

from openhcs.constants.constants import Backend
from openhcs.io.filemanager import FileManager
from openhcs.textual_tui.services.file_browser_service import FileBrowserService, SelectionMode
from openhcs.textual_tui.widgets.file_list import FileListWidget
from openhcs.textual_tui.widgets.floating_window import BaseFloatingWindow


class FileBrowserScreen(BaseFloatingWindow):
    """File browser modal for selecting files or directories using global floating window system."""

    def __init__(self,
                 file_manager: FileManager,
                 initial_path: Path,
                 backend: Backend = Backend.DISK,
                 selection_mode: SelectionMode = SelectionMode.FILES_ONLY,
                 title: str = "Select File",
                 **kwargs):
        self.file_manager = file_manager
        self.initial_path = initial_path
        self.backend = backend
        self.selection_mode = selection_mode
        self.browser_title = title
        self.selected_path: Optional[Path] = None

        # Create services
        self.file_browser_service = FileBrowserService(file_manager)

        # Create file list widget
        self.file_list = FileListWidget(
            file_browser_service=self.file_browser_service,
            initial_path=initial_path,
            backend=backend,
            selection_mode=selection_mode,
            on_selection=self._on_file_selected
        )

        super().__init__(title=title, **kwargs)
    
    def compose_content(self) -> ComposeResult:
        """Compose the file browser content."""
        # Path display
        yield Static(f"Path: {self.file_list.current_path}", id="path_display")

        # File list in a container that can scroll if needed
        from textual.containers import ScrollableContainer
        with ScrollableContainer():
            yield self.file_list

    def compose_buttons(self) -> ComposeResult:
        """Provide Up/Select/Cancel buttons."""
        yield Button("Up", id="up", compact=True)
        yield Button("Select", id="select", compact=True)
        yield Button("Cancel", id="cancel", compact=True)

    def handle_button_action(self, button_id: str, button_text: str):
        """Handle button actions - Up/Select/Cancel logic."""
        if button_text == 'Up':
            self.file_list.navigate_up()
            self._update_path_display()
            return False  # Don't dismiss dialog
        elif button_text == 'Select':
            return self._handle_select()
        elif button_text == 'Cancel':
            return None  # Dismiss with None
        return False  # Don't dismiss by default
    
    def _on_file_selected(self, path: Path) -> None:
        """Handle file selection from file list."""
        self.selected_path = path
        self.dismiss(path)

    def _handle_select(self):
        """Handle select button."""
        if self.selection_mode == SelectionMode.DIRECTORIES_ONLY:
            # Select current directory
            return self.file_list.current_path
        else:
            # Select focused item if valid
            item = self.file_list.get_focused_item()
            if item and self.file_list.can_select_focused():
                return item.path
            else:
                # No valid selection - don't dismiss
                return False

    def _update_path_display(self) -> None:
        """Update the path display."""
        try:
            path_widget = self.query_one("#path_display", Static)
            path_widget.update(f"Path: {self.file_list.current_path}")
        except Exception:
            # Widget might not be mounted yet
            pass

    def watch_file_list_current_path(self, new_path: Path) -> None:
        """React to path changes in file list."""
        self._update_path_display()

    DEFAULT_CSS = """
    FileBrowserScreen #path_display {
        margin-bottom: 1;
        text-style: italic;
    }
    """
