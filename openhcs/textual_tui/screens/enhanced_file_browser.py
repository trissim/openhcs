"""
Enhanced file browser using textual-universal-directorytree with OpenHCS FileManager.

This provides a more robust file browser experience using the mature
textual-universal-directorytree widget adapted for OpenHCS backends.
"""

import logging
from pathlib import Path
from typing import Optional, Set, List

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DirectoryTree, Static, Checkbox

from openhcs.constants.constants import Backend
from openhcs.io.filemanager import FileManager
from openhcs.textual_tui.adapters.universal_directorytree import OpenHCSDirectoryTree

logger = logging.getLogger(__name__)


class EnhancedFileBrowserScreen(ModalScreen):
    """
    Enhanced file browser dialog using OpenHCS DirectoryTree adapter.
    
    This provides a more robust file browsing experience using the mature
    textual-universal-directorytree widget adapted to work with OpenHCS's
    FileManager backend system.
    """
    
    DEFAULT_CSS = """
    EnhancedFileBrowserScreen {
        align: center middle;
        background: $background 60%;
    }
    
    #file_browser {
        /* Inherit app's dialog style */
        max-width: 90%;
        max-height: 80%;
        min-width: 80;
        min-height: 35;
    }
    
    #path_display {
        margin-bottom: 1;
        text-style: italic;
        color: $text-muted;
    }

    #selection_panel {
        width: 30%;
        height: 1fr;
        border: solid $primary;
        padding: 1;
    }

    #tree_panel {
        width: 70%;
        height: 1fr;
    }

    OpenHCSDirectoryTree {
        height: 1fr;
        border: solid $primary;
    }

    #selected_list {
        height: 1fr;
        border: solid $accent;
        padding: 1;
    }
    """
    
    def __init__(
        self,
        file_manager: FileManager,
        initial_path: Path,
        backend: Backend = Backend.DISK,
        title: str = "Select Directory",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.file_manager = file_manager
        self.initial_path = initial_path
        self.backend = backend
        self.title = title
        self.selected_path: Optional[Path] = None
        self.selected_paths: Set[Path] = set()  # For multi-selection
        
        # Create OpenHCS DirectoryTree
        self.directory_tree = OpenHCSDirectoryTree(
            filemanager=file_manager,
            backend=backend,
            path=initial_path
        )
        
        logger.debug(f"EnhancedFileBrowserScreen created for {backend.value} at {initial_path}")
    
    def compose(self) -> ComposeResult:
        """Compose the enhanced file browser dialog."""
        # Use app's dialog class for consistent styling
        with Container(id="file_browser", classes="dialog"):
            yield Static(self.title, classes="dialog-title")

            # Path display
            yield Static(f"Path: {self.initial_path}", id="path_display")

            # Main content area with tree and selection panel
            with Horizontal():
                # Directory tree panel
                with Container(id="tree_panel"):
                    yield self.directory_tree

                # Selection panel for multi-select
                with Vertical(id="selection_panel"):
                    yield Static("Selected Directories:", classes="dialog-title")
                    with Container(id="selected_list"):
                        yield Static("(none selected)", id="selected_display")

            # Buttons
            with Horizontal(classes="dialog-buttons"):
                yield Button("Add Current", id="add_current_btn", compact=True)
                yield Button("Remove Selected", id="remove_selected_btn", compact=True)
                yield Button("Select All", id="select_all_btn", variant="primary", compact=True)
                yield Button("Cancel", id="cancel_btn", compact=True)
    
    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        # Focus the directory tree for keyboard navigation
        self.directory_tree.focus()
    
    @on(DirectoryTree.DirectorySelected)
    def on_directory_selected(self, event: DirectoryTree.DirectorySelected) -> None:
        """Handle directory selection from tree."""
        # Update path display
        path_widget = self.query_one("#path_display", Static)
        path_widget.update(f"Path: {event.path}")

        # Store selected path - ensure it's always a Path object
        if hasattr(event.path, '_path'):
            # OpenHCSPathAdapter
            self.selected_path = Path(event.path._path)
        elif isinstance(event.path, Path):
            self.selected_path = event.path
        else:
            # Convert string or other types to Path
            self.selected_path = Path(str(event.path))

        logger.debug(f"Directory selected: {self.selected_path} (type: {type(self.selected_path)})")
    
    @on(DirectoryTree.FileSelected)
    def on_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        """Handle file selection from tree."""
        # For directory selection, we ignore file selections
        # But we could extend this for file selection modes
        logger.debug(f"File selected: {event.path}")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "add_current_btn":
            self._handle_add_current()
        elif event.button.id == "remove_selected_btn":
            self._handle_remove_selected()
        elif event.button.id == "select_all_btn":
            self._handle_select_all()
        elif event.button.id == "cancel_btn":
            self.dismiss(None)

    def _handle_add_current(self) -> None:
        """Add current directory to selection."""
        if self.selected_path and self.selected_path.is_dir():
            self.selected_paths.add(self.selected_path)
            self._update_selected_display()
            logger.debug(f"Added {self.selected_path} to selection")

    def _handle_remove_selected(self) -> None:
        """Remove current directory from selection."""
        if self.selected_path and self.selected_path in self.selected_paths:
            self.selected_paths.remove(self.selected_path)
            self._update_selected_display()
            logger.debug(f"Removed {self.selected_path} from selection")

    def _handle_select_all(self) -> None:
        """Return all selected directories."""
        if self.selected_paths:
            # Return list of selected paths
            self.dismiss(list(self.selected_paths))
        else:
            # No selection, return current path if it's a directory
            if self.selected_path and self.selected_path.is_dir():
                self.dismiss([self.selected_path])
            else:
                self.dismiss([self.initial_path])

    def _update_selected_display(self) -> None:
        """Update the selected directories display."""
        try:
            display_widget = self.query_one("#selected_display", Static)
            if self.selected_paths:
                paths_text = "\n".join([f"📁 {path.name}" for path in sorted(self.selected_paths)])
                display_widget.update(paths_text)
            else:
                display_widget.update("(none selected)")
        except Exception:
            # Widget might not be mounted yet
            pass
    
    def _update_path_display(self, path: Path) -> None:
        """Update the path display."""
        try:
            path_widget = self.query_one("#path_display", Static)
            path_widget.update(f"Path: {path}")
        except Exception:
            # Widget might not be mounted yet
            pass
