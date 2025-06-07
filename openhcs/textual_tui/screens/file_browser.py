"""File browser modal screen for OpenHCS Textual TUI."""

from pathlib import Path
from typing import Optional, Callable, List
from textual.screen import ModalScreen
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, Center, Middle
from textual.widgets import Button, Static

from openhcs.constants.constants import Backend
from openhcs.io.filemanager import FileManager
from openhcs.textual_tui.services.file_browser_service import FileBrowserService, SelectionMode
from openhcs.textual_tui.widgets.file_list import FileListWidget


class FileBrowserScreen(ModalScreen):
    """File browser modal for selecting files or directories - floating window style."""

    DEFAULT_CSS = """
    FileBrowserScreen {
        align: center middle;
        background: $background 60%;
    }

    #file_browser {
        /* Inherit app's dialog style */
        max-width: 90%;
        max-height: 80%;
        min-width: 70;
        min-height: 30;
    }

    #path_display {
        margin-bottom: 1;
        text-style: italic;
    }
    """
    
    def __init__(self, 
                 file_manager: FileManager,
                 initial_path: Path,
                 backend: Backend = Backend.DISK,
                 selection_mode: SelectionMode = SelectionMode.FILES_ONLY,
                 title: str = "Select File",
                 **kwargs):
        super().__init__(**kwargs)
        self.file_manager = file_manager
        self.initial_path = initial_path
        self.backend = backend
        self.selection_mode = selection_mode
        self.title = title
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
    
    def compose(self) -> ComposeResult:
        """Compose the floating file browser dialog."""
        # Use app's dialog class for consistent styling
        with Container(id="file_browser", classes="dialog"):
            yield Static(self.title, classes="dialog-title")

            # Path display
            yield Static(f"Path: {self.file_list.current_path}", id="path_display")

            # File list
            yield self.file_list

            # Buttons
            with Horizontal(classes="dialog-buttons"):
                yield Button("Up", id="up_btn", compact=True)
                yield Button("Select", id="select_btn", variant="primary", compact=True)
                yield Button("Cancel", id="cancel_btn", compact=True)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "up_btn":
            self.file_list.navigate_up()
            self._update_path_display()
        elif event.button.id == "select_btn":
            self._handle_select()
        elif event.button.id == "cancel_btn":
            self.dismiss(None)
    
    def _on_file_selected(self, path: Path) -> None:
        """Handle file selection from file list."""
        self.selected_path = path
        self.dismiss(path)
    
    def _handle_select(self) -> None:
        """Handle select button."""
        if self.selection_mode == SelectionMode.DIRECTORIES_ONLY:
            # Select current directory
            self.dismiss(self.file_list.current_path)
        else:
            # Select focused item if valid
            item = self.file_list.get_focused_item()
            if item and self.file_list.can_select_focused():
                self.dismiss(item.path)
            else:
                # No valid selection
                pass
    
    def _update_path_display(self) -> None:
        """Update the path display."""
        path_widget = self.query_one("#path_display", Static)
        path_widget.update(f"Path: {self.file_list.current_path}")
    
    def watch_file_list_current_path(self, new_path: Path) -> None:
        """React to path changes in file list."""
        self._update_path_display()
