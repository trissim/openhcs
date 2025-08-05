"""
Enhanced file browser using textual-universal-directorytree with OpenHCS FileManager.

This provides a more robust file browser experience using the mature
textual-universal-directorytree widget adapted for OpenHCS backends.
"""

import logging
from pathlib import Path
from typing import Optional, Set, List, Dict, Callable
from enum import Enum

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer, VerticalScroll
from textual.widgets import Button, DirectoryTree, Static, Checkbox, Input

from openhcs.constants.constants import Backend
from openhcs.io.filemanager import FileManager
from openhcs.textual_tui.adapters.universal_directorytree import OpenHCSDirectoryTree
from openhcs.textual_tui.windows.base_window import BaseOpenHCSWindow
from openhcs.core.path_cache import PathCacheKey
from openhcs.textual_tui.services.file_browser_service import SelectionMode

logger = logging.getLogger(__name__)


class BrowserMode(Enum):
    """Browser operation mode."""
    LOAD = "load"
    SAVE = "save"


class FileBrowserWindow(BaseOpenHCSWindow):
    """
    Enhanced file browser window using OpenHCS DirectoryTree adapter with textual-window system.

    This provides a more robust file browsing experience using the mature
    textual-universal-directorytree widget adapted to work with OpenHCS's
    FileManager backend system.

    Features:
    - Single and multi-selection modes (multi-selection is opt-in)
    - Inline folder creation and editing
    - Backend-agnostic file operations through FileManager
    - Custom click behaviors (left: select, right: multi-select if enabled, double: navigate)
    """

    DEFAULT_CSS = """
    FileBrowserWindow {
        width: 80; height: 30;
        min-width: 60; min-height: 25;
    }
    FileBrowserWindow #content_pane {
        padding: 0;  /* Remove padding for compact layout */
    }

    /* Bottom area should have minimal height */
    FileBrowserWindow #bottom_area {
        height: auto;
        max-height: 10;  /* Slightly more space for horizontal buttons + selection area */
    }

    /* Buttons panel - single horizontal row */
    FileBrowserWindow #buttons_panel {
        height: 1;      /* Exactly 1 row */
        width: 100%;
        align: center middle;
    }

    /* Buttons should be very compact */
    FileBrowserWindow #buttons_panel Button {
        width: auto;    /* Auto-size buttons to content */
        min-width: 4;   /* Very small minimum width */
        padding: 0;     /* No padding for maximum compactness */
    }

    /* Checkbox should also be very compact */
    FileBrowserWindow #buttons_panel Checkbox {
        width: auto;    /* Auto-size checkbox */
        padding: 0;     /* No padding */
    }

    /* Selection panel - starts at 2 rows (label + 1 for content), expands as needed (LOAD mode only) */
    FileBrowserWindow #selection_panel {
        width: 100%;  /* Full width */
        height: 2;    /* Start at 2 rows (1 for label + 1 for content) */
        max-height: 5; /* Maximum 5 rows (1 for label + 4 for list) */
    }

    /* Selections label - compact and left-aligned */
    FileBrowserWindow #selections_label {
        height: 1;    /* Exactly 1 row for label */
        text-align: left;
        padding: 0;
        margin: 0;
    }

    /* Selection list should start at 1 row and expand when needed */
    FileBrowserWindow #selected_list {
        height: 1;    /* Start at exactly 1 row */
        max-height: 4; /* Maximum 4 rows for the list itself */
        text-align: left; /* Ensure container is left-aligned */
        content-align: left top; /* Force content alignment to left */
        align: left top; /* Additional alignment property */
    }

    /* Selected display text should be left-aligned */
    FileBrowserWindow #selected_display {
        text-align: left;
        content-align: left top; /* Force content alignment to left */
        align: left top; /* Additional alignment property */
        padding: 0;
        margin: 0;
        width: 100%; /* Ensure full width */
    }

    /* Path area - horizontal layout for label + input */
    FileBrowserWindow #path_area {
        height: 1;
        width: 100%;
        margin: 0;
        padding: 0;
    }

    /* Path label styling */
    FileBrowserWindow .path-label {
        width: auto;
        min-width: 6;
        text-align: left;
        padding: 0 1 0 0;
        margin: 0;
    }

    /* Path input should be minimal and editable */
    FileBrowserWindow #path_input {
        height: 1;
        width: 1fr;
        margin: 0;
        padding: 0;
    }

    /* Filename area should have explicit height and styling */
    FileBrowserWindow #filename_area {
        height: 3;
        width: 100%;
        margin: 0;
        padding: 0;
    }

    /* Filename label styling */
    FileBrowserWindow .filename-label {
        width: auto;
        min-width: 10;
        text-align: left;
        padding: 0 1 0 0;
    }

    /* Filename input styling */
    FileBrowserWindow #filename_input {
        width: 1fr;
        height: 1;
    }

    /* Folder editing container - inline style aligned with tree folders */
    FileBrowserWindow #folder_edit_container {
        layer: overlay;
        width: 50;
        height: 1;
        background: $surface;
        align: left top;
        offset: 4 6;  /* Align with tree folder indentation */
        padding: 0;
    }

    FileBrowserWindow .edit-help {
        text-align: left;
        text-style: dim;
        height: 1;
    }

    FileBrowserWindow #folder_edit_input {
        width: 1fr;
        height: 1;
    }
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
        cache_key: Optional[PathCacheKey] = None,
        on_result_callback: Optional[Callable] = None,
        caller_id: str = "unknown",
        enable_multi_selection: bool = False,
        **kwargs
    ):
        # Create unique window ID based on caller to avoid conflicts
        unique_window_id = f"file_browser_{caller_id}"

        # Use unique window ID - textual-window expects consistent IDs per caller
        super().__init__(
            window_id=unique_window_id,
            title=title,
            mode="temporary",
            **kwargs
        )

        self.file_manager = file_manager
        self.initial_path = initial_path
        self.backend = backend
        self.browser_title = title
        self.mode = mode
        self.selection_mode = selection_mode
        self.filter_extensions = filter_extensions
        self.default_filename = default_filename
        self.cache_key = cache_key
        self.on_result_callback = on_result_callback
        self.enable_multi_selection = enable_multi_selection
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
            enable_multi_selection=self.enable_multi_selection,
            id='tree_panel'
        )

        logger.debug(f"FileBrowserWindow created for {backend.value} at {initial_path}")
    


    def compose(self) -> ComposeResult:
        """Compose the enhanced file browser content."""
        with Vertical():
            # Path input with label (fixed height at top) - editable path field
            with Horizontal(id="path_area"):
                yield Static("Path:", classes="path-label")
                yield Input(
                    value=str(self.initial_path),
                    placeholder="Enter path...",
                    id="path_input",
                    compact=True
                )

            # Directory tree - scrollable area (this should expand to fill remaining space)
            with ScrollableContainer(id="tree_area"):
                yield self.directory_tree

            # Filename input for save mode - horizontal layout (fixed height)
            if self.mode == BrowserMode.SAVE:
                logger.debug(f"🔍 SAVE MODE: Rendering filename input area with default: '{self.default_filename}'")
                with Horizontal(id="filename_area"):
                    yield Static("Filename:", classes="filename-label")
                    yield Input(
                        placeholder="Enter filename...",
                        value=self.default_filename,
                        id="filename_input",
                        compact=True
                    )

            # Bottom area: buttons on top, selection area below (fixed height at bottom)
            with Vertical(id="bottom_area"):
                # All buttons in single horizontal row with compact spacing
                with Horizontal(id="buttons_panel", classes="dialog-buttons"):
                    yield Button("🏠 Home", id="go_home", compact=True)
                    yield Button("⬆️ Up", id="go_up", compact=True)

                    # Mode-specific buttons
                    if self.mode == BrowserMode.LOAD:
                        # Only show Add/Remove buttons if multi-selection is enabled
                        if self.enable_multi_selection:
                            yield Button("Add", id="add_current", compact=True)
                            yield Button("Remove", id="remove_selected", compact=True)
                        yield Button("📁 New", id="new_folder", compact=True)
                        yield Button("Select", id="select_all", compact=True)
                    else:  # SAVE mode
                        yield Button("Save", id="save_file", compact=True)
                        yield Button("📁 New", id="new_folder", compact=True)

                    yield Checkbox(
                        label="Hidden",
                        value=self.show_hidden_files,
                        id="show_hidden_checkbox",
                        compact=True
                    )
                    yield Button("Cancel", id="cancel", compact=True)

                # Selection panel below buttons (only for LOAD mode with multi-selection enabled)
                if self.mode == BrowserMode.LOAD and self.enable_multi_selection:
                    with Vertical(id="selection_panel"):
                        # Add "Selections:" label
                        yield Static("Selections:", id="selections_label")
                        with ScrollableContainer(id="selected_list"):
                            yield Static("(none)", id="selected_display")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        logger.debug(f"🔍 BUTTON PRESSED: {button_id}")

        if button_id == "go_home":
            self._handle_go_home()
        elif button_id == "go_up":
            self._handle_go_up()
        elif button_id == "add_current":
            logger.debug("🔍 ADD BUTTON: Calling _handle_add_current")
            self._handle_add_current()
        elif button_id == "remove_selected":
            self._handle_remove_selected()
        elif button_id == "new_folder":
            self._handle_new_folder()
        elif button_id == "select_all":
            result = self._handle_select_all()
            self._finish_with_result(result)
        elif button_id == "save_file":
            result = self._handle_save_file()
            if result is not False:  # False means don't dismiss
                self._finish_with_result(result)
        elif button_id == "cancel":
            self._finish_with_result(None)

    def _finish_with_result(self, result):
        """Finish the dialog with a result."""
        import logging
        logger = logging.getLogger(__name__)

        # Log result for debugging (only when result exists)
        if result is not None:
            logger.debug(f"File browser returning: {result}")

        # Cache the path if successful
        if result is not None and self.cache_key is not None:
            self._cache_successful_path(result)

        # Call the callback if provided
        if self.on_result_callback:
            self.on_result_callback(result)
        else:
            logger.debug("No callback provided to file browser")

        # Close the window
        self.close_window()

    def _cache_successful_path(self, result):
        """Cache the successful path selection."""
        from openhcs.core.path_cache import get_path_cache

        try:
            path_cache = get_path_cache()
            cache_path = None

            if isinstance(result, Path):
                # Single path result - cache its parent directory
                cache_path = result.parent if result.is_file() else result
            elif isinstance(result, list) and result:
                # List of paths - cache the parent of the first path
                first_path = result[0]
                if isinstance(first_path, Path):
                    cache_path = first_path.parent if first_path.is_file() else first_path

            # Cache the path if we determined one
            if cache_path and cache_path.exists():
                path_cache.set_cached_path(self.cache_key, cache_path)
                logger.debug(f"Cached path for {self.cache_key.value}: {cache_path}")

        except Exception as e:
            logger.warning(f"Failed to cache path: {e}")

    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        # Set initial border title
        self.directory_tree.border_title = f"Path: {self.initial_path}"

        # Set initial border title for selected panel (LOAD mode only)
        if self.mode == BrowserMode.LOAD:
            try:
                selection_panel = self.query_one("#selection_panel", Vertical)
                selection_panel.border_title = "Selected:"
            except Exception:
                pass  # Widget might not be mounted yet

        # Focus the directory tree for keyboard navigation
        self.directory_tree.focus()
    
    @on(DirectoryTree.DirectorySelected)
    def on_directory_selected(self, event: DirectoryTree.DirectorySelected) -> None:
        """Handle directory selection from tree."""
        logger.debug(f"🔍 DIRECTORY SELECTED: {event.path}")

        # Store selected path - ensure it's always a Path object
        if hasattr(event.path, '_path'):
            # OpenHCSPathAdapter
            self.selected_path = Path(event.path._path)
        elif isinstance(event.path, Path):
            self.selected_path = event.path
        else:
            # Convert string or other types to Path
            self.selected_path = Path(str(event.path))

        # Update path input
        self._update_path_display(self.selected_path)

        # For save mode, update filename input when directory is selected
        # This allows users to see the directory name as a potential filename base
        if self.mode == BrowserMode.SAVE:
            self._update_filename_from_selection(self.selected_path)

        logger.debug(f"🔍 STORED selected_path: {self.selected_path} (type: {type(self.selected_path)})")

    @on(Input.Submitted)
    def on_path_input_submitted(self, event: Input.Submitted) -> None:
        """Handle path input submission to navigate to entered path."""
        if event.input.id == "path_input":
            entered_path = event.input.value.strip()
            if entered_path:
                try:
                    new_path = Path(entered_path).resolve()

                    # Check if path exists and is accessible
                    if self.file_manager.exists(new_path, self.backend.value):
                        # Check if it's a directory
                        if self.file_manager.is_dir(new_path, self.backend.value):
                            self._navigate_to_path(new_path)
                            logger.debug(f"🔍 NAVIGATED via path input to: {new_path}")
                        else:
                            # It's a file, navigate to its parent directory
                            parent_path = new_path.parent
                            self._navigate_to_path(parent_path)
                            logger.debug(f"🔍 NAVIGATED via path input to parent of file: {parent_path}")
                    else:
                        # Path doesn't exist, revert to current path
                        current_path = self.selected_path or self.initial_path
                        event.input.value = str(current_path)
                        logger.warning(f"🔍 PATH NOT FOUND: {new_path}, reverted to {current_path}")

                except Exception as e:
                    # Invalid path, revert to current path
                    current_path = self.selected_path or self.initial_path
                    event.input.value = str(current_path)
                    logger.warning(f"🔍 INVALID PATH: {entered_path}, error: {e}, reverted to {current_path}")

    @on(DirectoryTree.FileSelected)
    def on_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        """Handle file selection from tree."""
        logger.debug(f"File selected event: {event.path} (selection_mode: {self.selection_mode})")

        # Always store the selected file path for save mode filename updates
        if hasattr(event.path, '_path'):
            # OpenHCSPathAdapter
            selected_file_path = Path(event.path._path)
        elif isinstance(event.path, Path):
            selected_file_path = event.path
        else:
            # Convert string or other types to Path
            selected_file_path = Path(str(event.path))

        # For save mode, always update filename input with selected file name
        # This works regardless of selection_mode since we want filename suggestions
        if self.mode == BrowserMode.SAVE:
            self._update_filename_from_selection(selected_file_path)

        # Store selected path only if selection mode allows files
        if self.selection_mode in [SelectionMode.FILES_ONLY, SelectionMode.BOTH]:
            self.selected_path = selected_file_path
            logger.info(f"✅ FILE STORED: {self.selected_path} (type: {type(self.selected_path)})")
        else:
            logger.info(f"❌ FILE IGNORED: selection_mode {self.selection_mode} only allows directories")

    @on(OpenHCSDirectoryTree.AddToSelectionList)
    def on_add_to_selection_list(self, event: OpenHCSDirectoryTree.AddToSelectionList) -> None:
        """Handle adding multiple folders to selection list via double-click."""
        logger.info(f"🔍 ADD TO SELECTION LIST: {len(event.paths)} folders")

        for path in event.paths:
            try:
                # Check if path is compatible with selection mode
                is_dir = self.file_manager.is_dir(path, self.backend.value)

                if self.selection_mode == SelectionMode.DIRECTORIES_ONLY and not is_dir:
                    logger.info(f"❌ SKIPPED: Cannot add file in DIRECTORIES_ONLY mode: {path}")
                    continue
                if self.selection_mode == SelectionMode.FILES_ONLY and is_dir:
                    logger.info(f"❌ SKIPPED: Cannot add directory in FILES_ONLY mode: {path}")
                    continue

                # Add to selection list if not already present
                if path not in self.selected_paths:
                    self.selected_paths.add(path)
                    logger.info(f"✅ ADDED TO LIST: {path}")
                else:
                    logger.info(f"⚠️ ALREADY IN LIST: {path}")

            except Exception as e:
                logger.error(f"❌ ERROR adding path to list: {path}, error: {e}")

        # Update the selection display
        self._update_selected_display()
        logger.debug(f"🔍 SELECTION LIST UPDATED: Total {len(self.selected_paths)} items")

    @on(OpenHCSDirectoryTree.NavigateToFolder)
    def on_navigate_to_folder(self, event: OpenHCSDirectoryTree.NavigateToFolder) -> None:
        """Handle double-click navigation into a folder."""
        logger.debug(f"🔍 NAVIGATE TO FOLDER: {event.path}")

        try:
            # Verify the path is a directory
            if not self.file_manager.is_dir(event.path, self.backend.value):
                logger.warning(f"❌ Cannot navigate to non-directory: {event.path}")
                return

            # Navigate to the new folder
            self._navigate_to_path(event.path)

        except Exception as e:
            logger.error(f"❌ ERROR navigating to {event.path}: {e}")

    @on(OpenHCSDirectoryTree.SelectFile)
    def on_select_file(self, event: OpenHCSDirectoryTree.SelectFile) -> None:
        """Handle double-click file selection - equivalent to highlight + Select button."""
        logger.debug(f"🔍 SELECT FILE: {event.path}")

        try:
            # Verify the path is a file (not a directory)
            if self.file_manager.is_dir(event.path, self.backend.value):
                logger.warning(f"❌ Cannot select directory as file: {event.path}")
                return

            # Set the selected path (this highlights it)
            self.selected_path = event.path
            self._update_path_display(event.path.parent)  # Update path display to parent directory

            # For save mode, update filename input with selected file name
            if self.mode == BrowserMode.SAVE:
                self._update_filename_from_selection(event.path)
                # In save mode, don't auto-close on double-click - just populate filename
                logger.debug(f"✅ FILENAME POPULATED: {event.path.name}")
                return

            # For load mode, immediately trigger the select action (equivalent to clicking Select button)
            result = self._handle_select_all()
            if result is not None:
                logger.debug(f"✅ FILE SELECTED: {event.path}")
                self._finish_with_result(result)
            else:
                logger.warning(f"❌ FILE SELECTION FAILED: {event.path}")

        except Exception as e:
            logger.error(f"❌ ERROR selecting file {event.path}: {e}")

    @on(Input.Submitted)
    def on_folder_edit_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key during folder name editing."""
        if event.input.id == "folder_edit_input":
            self._finish_folder_editing(event.value)

    @on(Input.Blurred)
    def on_folder_edit_blurred(self, event: Input.Blurred) -> None:
        """Handle focus loss during folder name editing."""
        if event.input.id == "folder_edit_input":
            self._finish_folder_editing(event.value)

    def _finish_folder_editing(self, new_name: str) -> None:
        """Complete the folder editing process."""
        logger.debug(f"🔍 FINISH EDIT: Completing folder edit with name '{new_name}'")

        try:
            # Remove the editing container
            edit_container = self.query_one("#folder_edit_container", Container)
            edit_container.remove()

            # Check if we have editing state
            if not hasattr(self, 'editing_folder_path'):
                logger.warning("❌ No editing state found")
                return

            old_path = self.editing_folder_path
            original_name = self.editing_original_name

            # Clean up editing state
            delattr(self, 'editing_folder_path')
            delattr(self, 'editing_original_name')

            # Validate new name
            new_name = new_name.strip()
            if not new_name or new_name == original_name:
                logger.debug(f"📝 EDIT CANCELLED: No change or empty name")
                return

            # Create new path
            new_path = old_path.parent / new_name

            # Check if new name already exists
            if self.file_manager.exists(new_path, self.backend.value):
                logger.warning(f"❌ RENAME FAILED: {new_name} already exists")
                return

            # Rename the folder using FileManager move operation
            self.file_manager.move(old_path, new_path, self.backend.value)

            logger.debug(f"✅ FOLDER RENAMED: {original_name} -> {new_name}")

            # Refresh the tree to show the renamed folder
            tree = self.query_one("#tree_panel", OpenHCSDirectoryTree)
            tree.reload()

        except Exception as e:
            logger.error(f"❌ ERROR finishing folder edit: {e}")

    def on_key(self, event) -> None:
        """Handle key events, including Escape to cancel folder editing."""
        if event.key == "escape" and hasattr(self, 'editing_folder_path'):
            # Cancel folder editing
            logger.debug("🔍 EDIT CANCELLED: Escape key pressed")
            try:
                edit_container = self.query_one("#folder_edit_container", Container)
                edit_container.remove()
                # Clean up editing state
                delattr(self, 'editing_folder_path')
                delattr(self, 'editing_original_name')
            except Exception as e:
                logger.error(f"❌ ERROR cancelling edit: {e}")

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
        """Add selected folders to selection list (multi-selection if enabled, otherwise single selection)."""
        # Get the directory tree
        tree = self.query_one("#tree_panel", OpenHCSDirectoryTree)

        if self.enable_multi_selection:
            # Multi-selection mode: add all folders with checkmarks
            selected_folders = tree.multi_selected_paths
            logger.debug(f"🔍 ADD BUTTON: Adding {len(selected_folders)} multi-selected folders, selection_mode={self.selection_mode}")

            if not selected_folders:
                logger.warning("❌ ADD FAILED: No folders selected (use right-click to select multiple folders)")
                return
        else:
            # Single selection mode: add only the current cursor selection
            if not self.selected_path:
                logger.warning("❌ ADD FAILED: No folder selected")
                return
            selected_folders = {self.selected_path}
            logger.debug(f"🔍 ADD BUTTON: Adding single selected folder {self.selected_path}, selection_mode={self.selection_mode}")

        added_count = 0
        for path in selected_folders:
            try:
                is_dir = self.file_manager.is_dir(path, self.backend.value)
                item_type = "directory" if is_dir else "file"
                logger.debug(f"🔍 ADD CHECK: {item_type} '{path.name}' in {self.selection_mode} mode")

                # Check if this type is allowed
                if self.selection_mode == SelectionMode.DIRECTORIES_ONLY and not is_dir:
                    logger.info(f"❌ SKIPPED: Cannot add {item_type} in DIRECTORIES_ONLY mode: {path}")
                    continue
                if self.selection_mode == SelectionMode.FILES_ONLY and is_dir:
                    logger.info(f"❌ SKIPPED: Cannot add {item_type} in FILES_ONLY mode: {path}")
                    continue

                # Add if not already present
                if path not in self.selected_paths:
                    self.selected_paths.add(path)
                    added_count += 1
                    logger.info(f"✅ ADDED: {item_type} '{path.name}'")
                else:
                    logger.info(f"⚠️ ALREADY ADDED: {item_type} '{path.name}'")

            except Exception as e:
                logger.error(f"❌ ERROR adding path {path}: {e}")

        # Update the selection display
        self._update_selected_display()
        logger.debug(f"✅ ADD COMPLETE: Added {added_count} new items (Total: {len(self.selected_paths)})")

    def _handle_new_folder(self) -> None:
        """Create a new folder with in-place editable name."""
        logger.info("🔍 NEW FOLDER: Creating new folder")

        try:
            # Get current directory from the tree
            tree = self.query_one("#tree_panel", OpenHCSDirectoryTree)
            current_dir = tree.path  # This should be the current directory being viewed

            # Generate a unique folder name
            base_name = "New Folder"
            counter = 1
            new_folder_name = base_name

            while True:
                new_folder_path = Path(current_dir) / new_folder_name
                if not self.file_manager.exists(new_folder_path, self.backend.value):
                    break
                counter += 1
                new_folder_name = f"{base_name} {counter}"

            # Create the folder using FileManager
            self.file_manager.ensure_directory(new_folder_path, self.backend.value)
            logger.info(f"✅ CREATED FOLDER: {new_folder_path}")

            # Refresh the tree to show the new folder
            tree.reload()

            # Start in-place editing of the new folder name
            self._start_folder_editing(new_folder_path, new_folder_name)

        except Exception as e:
            logger.error(f"❌ ERROR creating new folder: {e}")

    def _start_folder_editing(self, folder_path: Path, current_name: str) -> None:
        """Start editing of a folder name using a simple modal approach."""
        logger.debug(f"🔍 EDIT FOLDER: Starting folder name editing for {folder_path}")

        try:
            # For now, implement a simple approach - create a temporary input area
            # Store editing state
            self.editing_folder_path = folder_path
            self.editing_original_name = current_name

            # Create a simple editing container with proper Textual pattern
            from textual.containers import Container
            from textual.widgets import Input, Static

            # Create widgets first - compact inline editing with folder icon
            edit_input = Input(
                value=current_name,
                id="folder_edit_input",
                placeholder="Folder name",
                compact=True
            )
            # Add folder icon prefix to make it look like a tree node
            edit_input.prefix = "📁 "

            # Create container and mount it with just the input
            edit_container = Container(
                edit_input,
                id="folder_edit_container"
            )

            # Mount the complete container
            self.mount(edit_container)

            # Focus the input and select all text
            edit_input.focus()
            edit_input.action_home(select=True)
            edit_input.action_end(select=True)

            logger.debug(f"✅ EDIT FOLDER: Folder name editing started for {current_name}")

        except Exception as e:
            logger.error(f"❌ ERROR starting folder editing: {e}")

    def _handle_remove_selected(self) -> None:
        """Remove current directory from selection."""
        if self.selected_path and self.selected_path in self.selected_paths:
            self.selected_paths.remove(self.selected_path)
            self._update_selected_display()
            logger.info(f"Removed {self.selected_path} from selection")

    def _handle_select_all(self):
        """
        Return selected paths with intelligent priority system:
        1. Selection area (explicit Add button usage) - highest priority
        2. Multi-selected folders from tree (green checkmarks) - medium priority
        3. Current cursor selection - lowest priority fallback
        """
        if self.mode == BrowserMode.SAVE:
            return self._handle_save_file()

        # Priority 1: Return selected paths if any exist in selections area (explicit Add button usage)
        if self.selected_paths:
            return list(self.selected_paths)

        # Priority 2: Return multi-selected folders from tree (green checkmarks) if any exist
        tree = self.query_one("#tree_panel", OpenHCSDirectoryTree)
        if tree.multi_selected_paths:
            # Filter multi-selected paths based on selection mode
            valid_paths = []
            for path in tree.multi_selected_paths:
                try:
                    is_dir = self.file_manager.is_dir(path, self.backend.value)

                    # Check compatibility with selection mode
                    if self.selection_mode == SelectionMode.DIRECTORIES_ONLY and is_dir:
                        valid_paths.append(path)
                    elif self.selection_mode == SelectionMode.FILES_ONLY and not is_dir:
                        valid_paths.append(path)
                    elif self.selection_mode == SelectionMode.BOTH:
                        valid_paths.append(path)

                except Exception:
                    # Skip paths we can't validate
                    continue

            if valid_paths:
                logger.info(f"🔍 SELECT: Using {len(valid_paths)} multi-selected folders from tree")
                return valid_paths

        # Priority 3: Fallback to current tree cursor selection if nothing else is selected
        if self.selected_path:
            try:
                is_dir = self.file_manager.is_dir(self.selected_path, self.backend.value)

                # Check compatibility with selection mode
                if self.selection_mode == SelectionMode.DIRECTORIES_ONLY and is_dir:
                    logger.debug(f"🔍 SELECT: Using cursor selection {self.selected_path}")
                    return [self.selected_path]
                elif self.selection_mode == SelectionMode.FILES_ONLY and not is_dir:
                    logger.debug(f"🔍 SELECT: Using cursor selection {self.selected_path}")
                    return [self.selected_path]
                elif self.selection_mode == SelectionMode.BOTH:
                    logger.debug(f"🔍 SELECT: Using cursor selection {self.selected_path}")
                    return [self.selected_path]

            except Exception:
                pass

        # No valid selection
        logger.debug("🔍 SELECT: No valid selection found")
        return None

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
        """Update the selected directories display and adjust height."""
        try:
            display_widget = self.query_one("#selected_display", Static)
            selection_panel = self.query_one("#selection_panel", Vertical)
            selected_list = self.query_one("#selected_list", ScrollableContainer)

            # Force left alignment programmatically
            display_widget.styles.text_align = "left"
            display_widget.styles.content_align = ("left", "top")
            selected_list.styles.text_align = "left"
            selected_list.styles.content_align = ("left", "top")

            if self.selected_paths:
                # Show files and directories with appropriate icons
                paths_list = []
                for path in sorted(self.selected_paths):
                    try:
                        is_dir = self.file_manager.is_dir(path, self.backend.value)
                        icon = "📁" if is_dir else "📄"
                        paths_list.append(f"{icon} {path.name}")
                    except Exception:
                        # Fallback if we can't determine type
                        paths_list.append(f"📄 {path.name}")

                paths_text = "\n".join(paths_list)
                display_widget.update(paths_text)

                # Dynamically adjust height based on number of items (1-4 rows for list + 1 for label)
                num_items = len(self.selected_paths)
                list_height = min(max(num_items, 1), 4)  # Clamp between 1 and 4 for the list
                panel_height = list_height + 1  # Add 1 for the "Selections:" label

                # Update the height of the selection components
                selection_panel.styles.height = panel_height
                selected_list.styles.height = list_height
            else:
                display_widget.update("(none)")
                # Reset to minimum height when no items (1 for list + 1 for label)
                selection_panel.styles.height = 2  # 1 for label + 1 for "(none)"
                selected_list.styles.height = 1

        except Exception:
            # Widget might not be mounted yet
            pass
    
    def _update_path_display(self, path: Path) -> None:
        """Update the path input field and tree border title."""
        try:
            # Update the path input field
            path_input = self.query_one("#path_input", Input)
            path_input.value = str(path)

            # Set the border title on the directory tree
            self.directory_tree.border_title = f"Path: {path}"
        except Exception:
            # Widget might not be mounted yet
            pass

    def _update_filename_from_selection(self, selected_path: Path) -> None:
        """Update the filename input field based on the selected file or directory.

        This provides intelligent filename suggestions in save mode:
        - For files: Use the filename directly
        - For directories: Use the directory name as a base filename
        """
        if self.mode != BrowserMode.SAVE:
            return

        try:
            filename_input = self.query_one("#filename_input", Input)

            # Determine if the selected path is a file or directory
            try:
                is_dir = self.file_manager.is_dir(selected_path, self.backend.value)
            except Exception:
                # If we can't determine, assume it's a file based on extension
                is_dir = not selected_path.suffix

            if is_dir:
                # For directories, use the directory name as filename base
                suggested_name = selected_path.name

                # Add appropriate extension if filter is specified
                if self.filter_extensions and suggested_name:
                    suggested_name = self._ensure_extension(suggested_name, self.filter_extensions[0])

                filename_input.value = suggested_name
                logger.debug(f"🔍 FILENAME UPDATED from directory: {suggested_name}")
            else:
                # For files, use the filename directly
                filename_input.value = selected_path.name
                logger.debug(f"🔍 FILENAME UPDATED from file: {selected_path.name}")

        except Exception as e:
            logger.debug(f"Failed to update filename input: {e}")
            # Input might not be mounted yet or other error

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
        # Create a simple confirmation window using BaseOpenHCSWindow
        from openhcs.textual_tui.windows.base_window import BaseOpenHCSWindow
        from textual.widgets import Static, Button
        from textual.containers import Container, Horizontal
        from textual.app import ComposeResult

        class OverwriteConfirmationWindow(BaseOpenHCSWindow):
            def __init__(self, save_path: Path, on_result_callback):
                super().__init__(
                    window_id="overwrite_confirmation",
                    title="Confirm Overwrite",
                    mode="temporary"
                )
                self.save_path = save_path
                self.on_result_callback = on_result_callback

            def compose(self) -> ComposeResult:
                message = f"File '{self.save_path.name}' already exists.\nDo you want to overwrite it?"
                yield Static(message, classes="dialog-content")

                with Horizontal(classes="dialog-buttons"):
                    yield Button("Yes", id="yes", compact=True)
                    yield Button("No", id="no", compact=True)

            def on_button_pressed(self, event: Button.Pressed) -> None:
                result = event.button.id == "yes"
                if self.on_result_callback:
                    self.on_result_callback(result)
                self.close_window()

        def handle_confirmation(result):
            """Handle the confirmation dialog result."""
            if result:  # User clicked Yes
                logger.debug(f"User confirmed overwrite for: {save_path}")
                self._finish_with_result(save_path)  # Finish with the save path (will auto-cache)
            # If result is False/None (No/Cancel), do nothing - stay in dialog

        # Create and mount confirmation window
        confirmation = OverwriteConfirmationWindow(save_path, handle_confirmation)
        self.app.run_worker(self._mount_confirmation(confirmation))

    async def _mount_confirmation(self, confirmation):
        """Mount confirmation dialog."""
        await self.app.mount(confirmation)
        confirmation.open_state = True


async def open_file_browser_window(
    app,
    file_manager: FileManager,
    initial_path: Path,
    backend: Backend = Backend.DISK,
    title: str = "Select Directory",
    mode: BrowserMode = BrowserMode.LOAD,
    selection_mode: SelectionMode = SelectionMode.DIRECTORIES_ONLY,
    filter_extensions: Optional[List[str]] = None,
    default_filename: str = "",
    cache_key: Optional[PathCacheKey] = None,
    on_result_callback: Optional[Callable] = None,
    caller_id: str = "unknown",
    enable_multi_selection: bool = False,
) -> FileBrowserWindow:
    """
    Convenience function to open a file browser window.

    This replaces the old push_screen pattern with proper textual-window mounting.

    Args:
        app: The Textual app instance
        file_manager: FileManager instance
        initial_path: Starting directory path
        backend: Storage backend to use
        title: Window title
        mode: LOAD or SAVE mode
        selection_mode: What can be selected (files/dirs/both)
        filter_extensions: File extensions to filter (e.g., ['.pipeline'])
        default_filename: Default filename for save mode
        cache_key: Path cache key for remembering location
        on_result_callback: Callback function for when selection is made
        caller_id: Unique identifier for the calling window/widget (e.g., "plate_manager")

    Returns:
        The created FileBrowserWindow instance
    """
    from textual.css.query import NoMatches

    # Follow ConfigWindow pattern exactly - check if file browser already exists for this caller
    unique_window_id = f"file_browser_{caller_id}"
    try:
        window = app.query_one(f"#{unique_window_id}")
        # Window exists, update its parameters and open it
        window.file_manager = file_manager
        window.initial_path = initial_path
        window.backend = backend
        window.mode = mode
        window.selection_mode = selection_mode
        window.filter_extensions = filter_extensions
        window.default_filename = default_filename
        window.cache_key = cache_key
        window.on_result_callback = on_result_callback
        window.title = title
        # Refresh the window content with new parameters
        window._navigate_to_path(initial_path)
        window.open_state = True
    except NoMatches:
        # Expected case: window doesn't exist yet, create new one
        window = FileBrowserWindow(
            file_manager=file_manager,
            initial_path=initial_path,
            backend=backend,
            title=title,
            mode=mode,
            selection_mode=selection_mode,
            filter_extensions=filter_extensions,
            default_filename=default_filename,
            cache_key=cache_key,
            on_result_callback=on_result_callback,
            caller_id=caller_id,
            enable_multi_selection=enable_multi_selection,
        )
        await app.mount(window)  # Properly await mounting like ConfigWindow
        window.open_state = True

    return window


