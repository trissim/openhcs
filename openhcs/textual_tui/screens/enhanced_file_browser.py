"""
Enhanced file browser using textual-universal-directorytree with OpenHCS FileManager.

This provides a more robust file browser experience using the mature
textual-universal-directorytree widget adapted for OpenHCS backends.
"""

import logging
from pathlib import Path
from typing import Optional, Set, List, Dict
from enum import Enum

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer, VerticalScroll
from textual.widgets import Button, DirectoryTree, Static, Checkbox, Input

from openhcs.constants.constants import Backend
from openhcs.io.filemanager import FileManager
from openhcs.textual_tui.adapters.universal_directorytree import OpenHCSDirectoryTree
from openhcs.textual_tui.widgets.floating_window import BaseFloatingWindow

logger = logging.getLogger(__name__)


class BrowserMode(Enum):
    """Browser operation mode."""
    LOAD = "load"
    SAVE = "save"


class SelectionMode(Enum):
    """File selection mode."""
    FILES_ONLY = "files_only"
    DIRECTORIES_ONLY = "directories_only"
    FILES_AND_DIRECTORIES = "files_and_directories"


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
        mode: BrowserMode = BrowserMode.LOAD,
        selection_mode: SelectionMode = SelectionMode.DIRECTORIES_ONLY,
        filter_extensions: Optional[List[str]] = None,
        default_filename: str = "",
        **kwargs
    ):
        self.file_manager = file_manager
        self.initial_path = initial_path
        self.backend = backend
        self.browser_title = title
        self.mode = mode
        self.selection_mode = selection_mode
        self.filter_extensions = filter_extensions
        self.default_filename = default_filename
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
            filter_extensions=self.filter_extensions,
            id='tree_panel'
        )

        super().__init__(title=title, **kwargs)
        logger.debug(f"EnhancedFileBrowserScreen created for {backend.value} at {initial_path}")
    
    def get_content_info(self) -> dict:
        """Provide content information for dynamic sizing."""
        if self.mode == BrowserMode.LOAD:
            button_texts = ['🏠 Home', '⬆️ Up', 'Add', 'Remove', 'Select', 'Cancel']
        else:  # SAVE mode
            button_texts = ['🏠 Home', '⬆️ Up', 'Save', 'Cancel']

        return {
            'title': self.browser_title,
            'content_text': '',  # No text content, complex layout
            'button_texts': button_texts,
            'extra_content_width': 80  # Wide for tree + selection panels
        }

    def compose_content(self) -> ComposeResult:
        """Compose the enhanced file browser content."""
        # Path display (always visible)
        yield Static(f"Path: {self.initial_path}", id="path_display")

        # Directory tree - scrollable area
        yield self.directory_tree

        # Filename input for save mode - horizontal layout
        if self.mode == BrowserMode.SAVE:
            with Horizontal(id="filename_area"):
                yield Static("Filename:", classes="filename-label")
                yield Input(
                    placeholder="Enter filename...",
                    value=self.default_filename,
                    id="filename_input",
                    compact=True
                )

        # Bottom area: buttons left, selected panel right (always visible)
        with Horizontal(id="bottom_area"):
            # Buttons on left (auto width)
            with Vertical(id="buttons_panel"):
                yield Button("🏠 Home", id="go_home", compact=True)
                yield Button("⬆️ Up", id="go_up", compact=True)

                # Mode-specific buttons
                if self.mode == BrowserMode.LOAD:
                    yield Button("Add", id="add_current", compact=True)
                    yield Button("Remove", id="remove_selected", compact=True)
                    yield Button("Select", id="select_all", compact=True)
                else:  # SAVE mode
                    yield Button("Save", id="save_file", compact=True)

                yield Checkbox(
                    label="Hidden",
                    value=self.show_hidden_files,
                    id="show_hidden_checkbox",
                    compact=True
                )
                yield Button("Cancel", id="cancel", compact=True)

            # Selection panel on right (remaining width)
            with Vertical(id="selection_panel"):
                if self.mode == BrowserMode.LOAD:
                    yield Static("Selected:", classes="dialog-title")
                    with ScrollableContainer(id="selected_list"):
                        yield Static("(none)", id="selected_display")
                else:  # SAVE mode
                    yield Static("Save Info:", classes="dialog-title")
                    with ScrollableContainer(id="save_info"):
                        info_text = "Select directory and enter filename"
                        if self.filter_extensions:
                            info_text += f"\nAllowed extensions: {', '.join(self.filter_extensions)}"
                        yield Static(info_text, id="save_info_display")

    def compose_buttons(self) -> ComposeResult:
        """Buttons are now part of content layout - return empty."""
        return
        yield  # This line will never execute, but satisfies the generator requirement

    def handle_button_action(self, button_id: str, button_text: str):
        """Handle button actions - Navigation/Add/Remove/Select/Save/Cancel logic."""
        if button_text == '🏠 Home':
            self._handle_go_home()
            return False  # Don't dismiss dialog
        elif button_text == '⬆️ Up':
            self._handle_go_up()
            return False  # Don't dismiss dialog
        elif button_text == 'Add':
            self._handle_add_current()
            return False  # Don't dismiss dialog
        elif button_text == 'Remove':
            self._handle_remove_selected()
            return False  # Don't dismiss dialog
        elif button_text == 'Select':
            return self._handle_select_all()
        elif button_text == 'Save':
            return self._handle_save_file()
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
        if self.selection_mode in [SelectionMode.FILES_ONLY, SelectionMode.FILES_AND_DIRECTORIES]:
            # Store selected file path
            if hasattr(event.path, '_path'):
                # OpenHCSPathAdapter
                self.selected_path = Path(event.path._path)
            elif isinstance(event.path, Path):
                self.selected_path = event.path
            else:
                # Convert string or other types to Path
                self.selected_path = Path(str(event.path))

            # For save mode, populate filename input with selected file name
            if self.mode == BrowserMode.SAVE:
                try:
                    filename_input = self.query_one("#filename_input", Input)
                    filename_input.value = self.selected_path.name
                except Exception:
                    pass  # Input might not be mounted yet

            logger.debug(f"File selected: {self.selected_path}")
        else:
            logger.debug(f"File selection ignored in directory-only mode: {event.path}")
    
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

        # Update tree settings and reload instead of recreating
        self.directory_tree.show_hidden = self.show_hidden_files
        self.directory_tree.filter_extensions = self.filter_extensions

        try:
            # Reload the tree to apply new settings
            self.directory_tree.reload()
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

        # Update tree path and reload
        try:
            self.directory_tree.path = new_path
            self.directory_tree.reload()
            self.directory_tree.focus()
        except Exception as e:
            logger.warning(f"Failed to navigate to {new_path}: {e}")

    def _handle_add_current(self) -> None:
        """Add current directory to selection."""
        if self.selected_path:
            try:
                # Use FileManager to check if it's a directory (respects backend abstraction)
                is_dir = self.file_manager.is_dir(self.selected_path, self.backend.value)
                if is_dir:
                    self.selected_paths.add(self.selected_path)
                    self._update_selected_display()
                    logger.debug(f"Added {self.selected_path} to selection")
            except Exception as e:
                logger.warning(f"Could not verify if {self.selected_path} is a directory: {e}")

    def _handle_remove_selected(self) -> None:
        """Remove current directory from selection."""
        if self.selected_path and self.selected_path in self.selected_paths:
            self.selected_paths.remove(self.selected_path)
            self._update_selected_display()
            logger.debug(f"Removed {self.selected_path} from selection")

    def _handle_select_all(self):
        """Return all selected directories or files based on mode."""
        if self.mode == BrowserMode.LOAD:
            # For FILES_ONLY mode, return single Path object (not list)
            if self.selection_mode == SelectionMode.FILES_ONLY:
                if self.selected_path:
                    try:
                        # Use FileManager to check if it's a file
                        is_dir = self.file_manager.is_dir(self.selected_path, self.backend.value)
                        if not is_dir:  # It's a file
                            return self.selected_path  # Return single Path object
                    except Exception:
                        pass
                # No valid file selected, return None to cancel
                return None

            # For other modes (DIRECTORIES_ONLY, FILES_AND_DIRECTORIES), return list
            if self.selected_paths:
                # Return list of selected paths
                return list(self.selected_paths)
            else:
                # No selection, return current path if it's a directory
                if self.selected_path:
                    try:
                        # Use FileManager to check if it's a directory (respects backend abstraction)
                        is_dir = self.file_manager.is_dir(self.selected_path, self.backend.value)
                        if is_dir:
                            return [self.selected_path]
                        elif self.selection_mode == SelectionMode.FILES_AND_DIRECTORIES:
                            return [self.selected_path]
                    except Exception:
                        # If we can't determine type, fall back to initial path
                        pass
                return [self.initial_path]
        else:
            # Save mode - should use _handle_save_file instead
            return self._handle_save_file()

    def _handle_save_file(self):
        """Handle save file operation with overwrite confirmation."""
        try:
            # Get filename from input
            filename_input = self.query_one("#filename_input", Input)
            filename = filename_input.value.strip()

            if not filename:
                logger.warning("No filename provided for save operation")
                return False  # Don't dismiss, show error

            # Validate filename
            if not self._validate_filename(filename):
                logger.warning(f"Invalid filename: {filename}")
                return False  # Don't dismiss, show error

            # Ensure proper extension
            if self.filter_extensions:
                filename = self._ensure_extension(filename, self.filter_extensions[0])

            # Get current directory (use selected_path if it's a directory, otherwise its parent)
            if self.selected_path:
                try:
                    # Use FileManager to check if it's a directory (respects backend abstraction)
                    is_dir = self.file_manager.is_dir(self.selected_path, self.backend.value)
                    if is_dir:
                        save_dir = self.selected_path
                    else:
                        save_dir = self.selected_path.parent
                except Exception:
                    # If we can't determine type, use parent directory
                    save_dir = self.selected_path.parent
            else:
                save_dir = self.initial_path

            # Construct full save path
            save_path = save_dir / filename

            # Check if file already exists and show confirmation dialog
            if self._file_exists(save_path):
                self._show_overwrite_confirmation(save_path)
                return False  # Don't dismiss yet, wait for confirmation

            logger.debug(f"Save file operation: {save_path}")
            return save_path

        except Exception as e:
            logger.error(f"Error in save file operation: {e}")
            return False  # Don't dismiss, show error

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

    def _ensure_extension(self, filename: str, extension: str) -> str:
        """Ensure filename has the correct extension."""
        if not extension.startswith('.'):
            extension = f'.{extension}'
        path = Path(filename)
        if path.suffix.lower() != extension.lower():
            return str(path.with_suffix(extension))
        return filename

    def _validate_filename(self, filename: str) -> bool:
        """Validate filename for save operations."""
        if not filename.strip():
            return False

        # Check for invalid characters (basic validation)
        invalid_chars = '<>:"/\\|?*'
        if any(char in filename for char in invalid_chars):
            return False

        # Check extension if filter is specified
        if self.filter_extensions:
            path = Path(filename)
            if path.suffix:
                # Has extension, check if it's allowed
                return any(path.suffix.lower() == ext.lower() for ext in self.filter_extensions)
            # No extension, will be added by _ensure_extension

        return True

    def _file_exists(self, file_path: Path) -> bool:
        """Check if file exists using FileManager."""
        try:
            return self.file_manager.exists(file_path, self.backend.value)
        except Exception:
            return False

    def _show_overwrite_confirmation(self, save_path: Path) -> None:
        """Show confirmation dialog for overwriting existing file."""
        from openhcs.textual_tui.widgets.floating_window import ConfirmationWindow

        message = f"File '{save_path.name}' already exists.\nDo you want to overwrite it?"
        confirmation = ConfirmationWindow(
            title="Confirm Overwrite",
            message=message
        )

        def handle_confirmation(result):
            """Handle the confirmation dialog result."""
            if result:  # User clicked Yes
                logger.debug(f"User confirmed overwrite for: {save_path}")
                self.dismiss(save_path)  # Dismiss with the save path
            # If result is False/None (No/Cancel), do nothing - stay in dialog

        self.app.push_screen(confirmation, handle_confirmation)

    CSS_PATH = "enhanced_browser.css"
