"""
Enhanced file browser using textual-universal-directorytree with OpenHCS FileManager.

This provides a more robust file browser experience using the mature
textual-universal-directorytree widget adapted for OpenHCS backends.
"""

import logging
from pathlib import Path
from typing import Optional, Set, List, Dict

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer, VerticalScroll
from textual.widgets import Button, DirectoryTree, Static, Checkbox

from openhcs.constants.constants import Backend
from openhcs.io.filemanager import FileManager
from openhcs.textual_tui.adapters.universal_directorytree import OpenHCSDirectoryTree
from openhcs.textual_tui.widgets.floating_window import BaseFloatingWindow

logger = logging.getLogger(__name__)


class EnhancedFileBrowserScreen(BaseFloatingWindow):
    """
    Enhanced file browser dialog using OpenHCS DirectoryTree adapter with global floating window system.

    This provides a more robust file browsing experience using the mature
    textual-universal-directorytree widget adapted to work with OpenHCS's
    FileManager backend system.
    """

    def __init__(
        self,
        file_manager: FileManager,
        initial_path: Path,
        backend: Backend = Backend.DISK,
        title: str = "Select Directory",
        **kwargs
    ):
        self.file_manager = file_manager
        self.initial_path = initial_path
        self.backend = backend
        self.browser_title = title
        self.selected_path: Optional[Path] = None
        self.selected_paths: Set[Path] = set()  # For multi-selection

        # Path caching for performance
        self.path_cache: Dict[str, List[Path]] = {}

        # Hidden files toggle
        self.show_hidden_files = False

        # Create OpenHCS DirectoryTree
        self.directory_tree = OpenHCSDirectoryTree(
            filemanager=file_manager,
            backend=backend,
            path=initial_path,
            show_hidden=self.show_hidden_files,
            id='tree_panel'
        )

        super().__init__(title=title, **kwargs)
        logger.debug(f"EnhancedFileBrowserScreen created for {backend.value} at {initial_path}")
    
    def get_content_info(self) -> dict:
        """Provide content information for dynamic sizing."""
        return {
            'title': self.browser_title,
            'content_text': '',  # No text content, complex layout
            'button_texts': ['🏠 Home', '⬆️ Up', 'Add Current', 'Remove Selected', 'Select All', 'Cancel'],
            'extra_content_width': 80  # Wide for tree + selection panels
        }

    def compose_content(self) -> ComposeResult:
        """Compose the enhanced file browser content."""
        # Path display (always visible)
        yield Static(f"Path: {self.initial_path}", id="path_display")

        # Directory tree - scrollable area
        #with ScrollableContainer(id="tree_panel"):
        #with VerticalScroll(id="tree_panel"):
        yield self.directory_tree

        # Bottom area: buttons left, selected panel right (always visible)
        with Horizontal(id="bottom_area"):
            # Buttons on left (auto width)
            with Vertical(id="buttons_panel"):
                yield Button("🏠 Home", id="go_home", compact=True)
                yield Button("⬆️ Up", id="go_up", compact=True)
                yield Button("Add Current", id="add_current", compact=True)
                yield Button("Remove Selected", id="remove_selected", compact=True)
                yield Button("Select All", id="select_all", compact=True)
                yield Checkbox(
                    label="Show Hidden Files",
                    value=self.show_hidden_files,
                    id="show_hidden_checkbox",
                    compact=True
                )
                yield Button("Cancel", id="cancel", compact=True)

            # Selection panel on right (remaining width)
            with Vertical(id="selection_panel"):
                yield Static("Selected:", classes="dialog-title")
                with ScrollableContainer(id="selected_list"):
                    yield Static("(none)", id="selected_display")

    def compose_buttons(self) -> ComposeResult:
        """Buttons are now part of content layout - return empty."""
        return
        yield  # This line will never execute, but satisfies the generator requirement

    def handle_button_action(self, button_id: str, button_text: str):
        """Handle button actions - Navigation/Add/Remove/Select/Cancel logic."""
        if button_text == '🏠 Home':
            self._handle_go_home()
            return False  # Don't dismiss dialog
        elif button_text == '⬆️ Up':
            self._handle_go_up()
            return False  # Don't dismiss dialog
        elif button_text == 'Add Current':
            self._handle_add_current()
            return False  # Don't dismiss dialog
        elif button_text == 'Remove Selected':
            self._handle_remove_selected()
            return False  # Don't dismiss dialog
        elif button_text == 'Select All':
            return self._handle_select_all()
        elif button_text == 'Cancel':
            return None  # Dismiss with None
        return False  # Don't dismiss by default
    
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
    
    # Button handling now done through handle_button_action method

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Handle checkbox changes."""
        if event.checkbox.id == "show_hidden_checkbox":
            self.show_hidden_files = event.value
            # Use call_after_refresh to ensure proper async context
            self.call_after_refresh(self._refresh_directory_tree)
            logger.debug(f"Hidden files toggle: {self.show_hidden_files}")

    def _refresh_directory_tree(self) -> None:
        """Refresh directory tree with current settings."""
        # Clear path cache when settings change
        self.path_cache.clear()

        # Recreate directory tree with new settings
        current_path = self.selected_path or self.initial_path
        self.directory_tree = OpenHCSDirectoryTree(
            filemanager=self.file_manager,
            backend=self.backend,
            path=current_path,
            show_hidden=self.show_hidden_files
        )

        # Replace the tree in the UI
        try:
            tree_container = self.query_one("#tree_panel", ScrollableContainer)
            tree_container.remove_children()
            tree_container.mount(self.directory_tree)
            self.directory_tree.focus()
        except Exception as e:
            logger.warning(f"Failed to refresh directory tree: {e}")

    def _get_cached_paths(self, path: Path) -> Optional[List[Path]]:
        """Get cached directory contents."""
        cache_key = f"{path}:{self.show_hidden_files}"
        return self.path_cache.get(cache_key)

    def _cache_paths(self, path: Path, paths: List[Path]) -> None:
        """Cache directory contents."""
        cache_key = f"{path}:{self.show_hidden_files}"
        self.path_cache[cache_key] = paths

        # Limit cache size to prevent memory issues
        if len(self.path_cache) > 100:
            # Remove oldest entries (simple FIFO)
            oldest_key = next(iter(self.path_cache))
            del self.path_cache[oldest_key]

    def _handle_go_home(self) -> None:
        """Navigate to home directory."""
        home_path = Path.home()
        self._navigate_to_path(home_path)
        logger.debug(f"Navigated to home: {home_path}")

    def _handle_go_up(self) -> None:
        """Navigate to parent directory."""
        current_path = self.selected_path or self.initial_path
        parent_path = current_path.parent

        # Don't go above root
        if parent_path != current_path:
            self._navigate_to_path(parent_path)
            logger.debug(f"Navigated up from {current_path} to {parent_path}")
        else:
            logger.debug(f"Already at root directory: {current_path}")

    def _navigate_to_path(self, new_path: Path) -> None:
        """Navigate to a new path and refresh the tree."""
        self.selected_path = new_path
        self._update_path_display(new_path)

        # Recreate directory tree for new path
        self.directory_tree = OpenHCSDirectoryTree(
            filemanager=self.file_manager,
            backend=self.backend,
            path=new_path,
            show_hidden=self.show_hidden_files
        )

        # Replace the tree in the UI
        try:
            tree_container = self.query_one("#tree_panel", ScrollableContainer)
            tree_container.remove_children()
            tree_container.mount(self.directory_tree)
            self.directory_tree.focus()
        except Exception as e:
            logger.warning(f"Failed to navigate to {new_path}: {e}")

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

    def _handle_select_all(self):
        """Return all selected directories."""
        if self.selected_paths:
            # Return list of selected paths
            return list(self.selected_paths)
        else:
            # No selection, return current path if it's a directory
            if self.selected_path and self.selected_path.is_dir():
                return [self.selected_path]
            else:
                return [self.initial_path]

    def _update_selected_display(self) -> None:
        """Update the selected directories display."""
        try:
            display_widget = self.query_one("#selected_display", Static)
            if self.selected_paths:
                # Show just the directory names for compact display
                paths_text = "\n".join([f"📁 {path.name}" for path in sorted(self.selected_paths)])
                display_widget.update(paths_text)
            else:
                display_widget.update("(none)")
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

    CSS_PATH = "enhanced_browser.css"
#    DEFAULT_CSS = """
#    EnhancedFileBrowserScreen {
#        align: center middle;
#        background: $background 60%;
#    }
#
#    #floating_window {
#        background: $surface;
#        border: solid $primary;
#        padding: 2;
#        height: auto;
#    }
#
#    .dialog-title {
#        text-style: bold;
#        text-align: center;
#        margin-bottom: 1;
#    }
#
#    #path_display {
#        margin-bottom: 1;
#        text-style: italic;
#        color: $text-muted;
#    }
#
#
#
#    #tree_panel {
#        width: 100%;
#        height: 3fr;
#        min-height: 10;
#        margin-bottom: 1;
#    }
#
#    #bottom_area {
#        height: 1fr;
#        margin-top: 1;
#    }
#
#    #buttons_panel {
#        width: auto;
#        height: 8;
#        margin-right: 2;
#    }
#
#    #buttons_panel Button {
#        margin-bottom: 0;
#        width: 15;
#        height: 1;
#    }
#
#    #buttons_panel Checkbox {
#        margin-bottom: 0;
#        width: 15;
#        height: 1;
#    }
#
#    #selection_panel {
#        width: 1fr;
#        height: auto;
#        max-height: 8;
#        border: solid $primary;
#        padding: 1;
#    }
#
#    #selected_list {
#        height: auto;
#        min-height: 2;
#        max-height: 5;
#        border: solid $accent;
#        padding: 1;
#    }
#
#    OpenHCSDirectoryTree {
#        height: 1fr;
#        border: solid $primary;
#    }
#
#    .dialog-content {
#        height: auto;
#        margin: 0;
#    }
#
#    .dialog-buttons {
#        height: auto;
#        align: center middle;
#        margin-top: 1;
#    }
#
#    .dialog-buttons Button {
#        margin: 0 1;
#    }
#    """
#