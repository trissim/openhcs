"""
OpenHCS adapter for textual-universal-directorytree.

This module provides a simplified adapter that creates OpenHCS-aware DirectoryTree
widgets by using standard UPath but with OpenHCS FileManager integration.
"""

import logging
from pathlib import Path
from typing import Union, Iterable, Optional, List, Set

from textual import events
from textual_universal_directorytree import UniversalDirectoryTree
from upath import UPath

from openhcs.constants.constants import Backend
from openhcs.io.filemanager import FileManager

logger = logging.getLogger(__name__)
class OpenHCSDirectoryTree(UniversalDirectoryTree):
    """
    DirectoryTree widget that uses OpenHCS FileManager backend.

    This is a simplified version that uses standard UPath but stores
    OpenHCS FileManager reference for potential future integration.
    For now, it works with local filesystem through UPath.

    Custom click behavior:
    - Left click: Select single folder (no expansion, clears other selections)
    - Right click: Toggle folder in multi-selection
    - Double click: Navigate into folder (change view root)
    """

    def __init__(
        self,
        filemanager: FileManager,
        backend: Backend,
        path: Union[str, Path],
        show_hidden: bool = False,
        filter_extensions: Optional[List[str]] = None,
        enable_multi_selection: bool = False,
        **kwargs
    ):
        """
        Initialize OpenHCS DirectoryTree.

        Args:
            filemanager: OpenHCS FileManager instance
            backend: Backend to use (DISK, MEMORY, etc.)
            path: Initial path to display
            show_hidden: Whether to show hidden files (default: False)
            filter_extensions: Optional list of file extensions to show (e.g., ['.txt', '.py'])
            enable_multi_selection: Whether to enable multi-selection with right-click (default: False)
            **kwargs: Additional arguments passed to UniversalDirectoryTree
        """
        self.filemanager = filemanager
        self.backend = backend
        self.show_hidden = show_hidden
        self.filter_extensions = filter_extensions
        self.enable_multi_selection = enable_multi_selection

        # Track multi-selection state
        self.multi_selected_paths: Set[Path] = set()
        self.last_click_time = 0
        self.double_click_threshold = 0.25  # seconds

        # For now, use standard UPath for local filesystem
        # TODO: Future enhancement could integrate FileManager more deeply
        if backend == Backend.DISK:
            upath = UPath(path)
        else:
            # For non-disk backends, fall back to local path for now
            # This could be enhanced to support other backends
            upath = UPath(path)
            logger.warning(f"Backend {backend.value} not fully supported yet, using local filesystem")

        # Initialize parent with UPath
        super().__init__(path=upath, **kwargs)

        logger.debug(f"OpenHCSDirectoryTree initialized with {backend.value} backend at {path}, show_hidden={show_hidden}")

    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        """Filter paths to optionally hide hidden files and filter by extensions.

        Args:
            paths: The paths to be filtered.

        Returns:
            The filtered paths.
        """
        filtered_paths = []

        for path in paths:
            # Filter hidden files
            if not self.show_hidden and path.name.startswith('.'):
                continue

            # Filter by extension (only for files, not directories)
            if self.filter_extensions:
                try:
                    # Use FileManager to check if it's a directory (respects backend abstraction)
                    is_dir = self.filemanager.is_dir(path, self.backend.value)
                    if not is_dir:  # It's a file, apply extension filter
                        if not any(path.name.lower().endswith(ext.lower()) for ext in self.filter_extensions):
                            continue
                except Exception:
                    # If we can't determine file type, skip extension filtering for this item
                    # This preserves the item rather than breaking the entire operation
                    pass

            filtered_paths.append(path)

        return filtered_paths

    def on_click(self, event: events.Click) -> None:
        """Handle custom click behavior for folder selection."""
        import time

        # Always stop the event first to prevent tree expansion
        event.stop()
        event.prevent_default()

        # Try to get the node at the click position
        try:
            # Adjust click coordinate for scroll offset
            # event.y is screen coordinate, but get_node_at_line expects content line number
            scroll_offset = self.scroll_offset
            adjusted_y = event.y + scroll_offset.y

            logger.debug(f"Click at screen y={event.y}, scroll_offset={scroll_offset.y}, adjusted_y={adjusted_y}")

            # Get the node at the adjusted click position
            clicked_node = self.get_node_at_line(adjusted_y)
            if not clicked_node:
                # Try using cursor_node as fallback
                clicked_node = getattr(self, 'cursor_node', None)
                if not clicked_node:
                    logger.debug("Could not determine clicked node")
                    return
        except Exception as e:
            logger.debug(f"Error getting clicked node: {e}")
            return

        # Get the path from the node
        if not hasattr(clicked_node, 'data') or not clicked_node.data:
            logger.debug("Clicked node has no data")
            return

        # Handle different data types (DirEntry, Path, str)
        if hasattr(clicked_node.data, 'path'):
            # It's a DirEntry object
            node_path = Path(clicked_node.data.path)
        else:
            # It's a Path or string
            node_path = Path(str(clicked_node.data))
        current_time = time.time()

        # Check if it's a double click
        is_double_click = (current_time - self.last_click_time) < self.double_click_threshold
        self.last_click_time = current_time

        # Handle different click types
        if event.button == 3:  # Right click
            self._handle_right_click(node_path)
        elif is_double_click:
            self._handle_double_click(node_path)
        else:  # Left click
            # Set cursor to clicked node for visual feedback
            self.move_cursor(clicked_node)
            # Handle left click selection
            self._handle_left_click(node_path)

    def _handle_left_click(self, path: Path) -> None:
        """Handle left click - select single folder without expansion."""
        logger.info(f"🔍 LEFT CLICK: Selecting {path}")

        # Clear multi-selection and select this path
        self.multi_selected_paths.clear()
        self.multi_selected_paths.add(path)

        # Force complete UI update by invalidating and refreshing
        self._force_ui_update()

        # Post directory selected event using the standard DirectoryTree message
        from textual.widgets import DirectoryTree
        self.post_message(DirectoryTree.DirectorySelected(self, path))

    def _handle_right_click(self, path: Path) -> None:
        """Handle right click - toggle folder in multi-selection if enabled, otherwise treat as left click."""
        if not self.enable_multi_selection:
            # If multi-selection is disabled, treat right-click as left-click
            logger.info(f"🔍 RIGHT CLICK: Multi-selection disabled, treating as left click for {path}")
            self._handle_left_click(path)
            return

        # Multi-selection is enabled - toggle selection
        if path in self.multi_selected_paths:
            logger.info(f"🔍 RIGHT CLICK: Removing {path} from multi-selection")
            self.multi_selected_paths.remove(path)
        else:
            logger.info(f"🔍 RIGHT CLICK: Adding {path} to multi-selection")
            self.multi_selected_paths.add(path)

        # Force complete UI update by invalidating and refreshing
        self._force_ui_update()

        # Also try to refresh the specific node that was clicked
        self._refresh_specific_path(path)

        # Post directory selected event for the toggled path using the standard DirectoryTree message
        from textual.widgets import DirectoryTree
        self.post_message(DirectoryTree.DirectorySelected(self, path))

    def _handle_double_click(self, path: Path) -> None:
        """Handle double click - navigate into folder or select file."""
        try:
            # Check if the path is a directory or file
            is_directory = self.filemanager.is_dir(path, self.backend.value)

            if is_directory:
                # Navigate into the folder
                logger.info(f"🔍 DOUBLE CLICK: Navigating into directory {path}")
                self.post_message(self.NavigateToFolder(self, path))
            else:
                # Select the file (equivalent to highlight + Select button)
                logger.info(f"🔍 DOUBLE CLICK: Selecting file {path}")
                self.post_message(self.SelectFile(self, path))

        except Exception as e:
            # Fallback: treat as directory navigation if we can't determine type
            logger.warning(f"🔍 DOUBLE CLICK: Could not determine type for {path}, treating as directory: {e}")
            self.post_message(self.NavigateToFolder(self, path))

    def _force_ui_update(self) -> None:
        """Force the tree UI to update by trying multiple aggressive refresh strategies."""
        # Strategy 1: Clear internal caches that might prevent re-rendering
        try:
            # Clear line cache if it exists (common in Tree widgets)
            if hasattr(self, '_clear_line_cache'):
                self._clear_line_cache()
            # Clear any other caches
            if hasattr(self, '_clear_cache'):
                self._clear_cache()
            # Increment updates counter if it exists
            if hasattr(self, '_updates'):
                self._updates += 1
        except Exception:
            pass

        # Strategy 2: Immediate comprehensive refresh
        self.refresh(layout=True, repaint=True)

        # Strategy 3: Force complete re-render
        try:
            # Try to invalidate the widget's cache
            if hasattr(self, '_invalidate'):
                self._invalidate()
            elif hasattr(self, 'invalidate'):
                self.invalidate()
        except Exception:
            pass

        # Strategy 4: Multiple refresh calls
        self.refresh()
        self.refresh(layout=True)
        self.refresh(repaint=True)

        # Strategy 5: Force parent container refresh
        if self.parent:
            self.parent.refresh(layout=True, repaint=True)

        # Strategy 6: Refresh all visible lines using Tree-specific methods
        try:
            # Get the number of visible lines and refresh them all
            if hasattr(self, 'last_line') and self.last_line >= 0:
                self.refresh_lines(0, self.last_line + 1)
        except Exception:
            pass

        # Strategy 7: Schedule immediate and delayed refreshes
        self.call_next(lambda: self.refresh(layout=True, repaint=True))
        self.set_timer(0.001, lambda: self.refresh(layout=True, repaint=True))
        self.set_timer(0.01, lambda: self.refresh(layout=True, repaint=True))

        # Strategy 8: Force app-level refresh if available
        if hasattr(self, 'app') and self.app:
            self.app.refresh()

    def _refresh_specific_path(self, path: Path) -> None:
        """Try to refresh the specific tree node for the given path."""
        try:
            # Try to find the node for this path and refresh it specifically
            # This is more targeted than refreshing the entire tree

            # Method 1: Try to find the node by walking the tree
            def find_node_for_path(node, target_path):
                if hasattr(node, 'data') and node.data:
                    # Handle different data types (DirEntry, Path, str)
                    if hasattr(node.data, 'path'):
                        node_path = Path(node.data.path)
                    else:
                        node_path = Path(str(node.data))

                    if node_path == target_path:
                        return node

                # Recursively check children
                for child in getattr(node, 'children', []):
                    result = find_node_for_path(child, target_path)
                    if result:
                        return result
                return None

            # Find the node for this path
            if hasattr(self, 'root') and self.root:
                target_node = find_node_for_path(self.root, path)
                if target_node:
                    # Refresh this specific node
                    if hasattr(self, '_refresh_node'):
                        self._refresh_node(target_node)
                    # Also refresh its line if we can find it
                    if hasattr(target_node, 'line') and target_node.line >= 0:
                        if hasattr(self, 'refresh_line'):
                            self.refresh_line(target_node.line)
                        if hasattr(self, '_refresh_line'):
                            self._refresh_line(target_node.line)
        except Exception:
            # If targeted refresh fails, fall back to general refresh
            pass

    def render_label(self, node, base_style, style):
        """Override label rendering to show multi-selection state."""
        # Get the default rendered label from parent
        label = super().render_label(node, base_style, style)

        # Check if this node's path is in multi-selection
        if hasattr(node, 'data') and node.data:
            # Handle different data types (DirEntry, Path, str)
            if hasattr(node.data, 'path'):
                # It's a DirEntry object
                node_path = Path(node.data.path)
            else:
                # It's a Path or string
                node_path = Path(str(node.data))

            if node_path in self.multi_selected_paths:
                # Add visual indicator for selected items
                from rich.text import Text
                # Create a new label with selection styling
                selected_label = Text()
                selected_label.append("✓ ", style="bold green")  # Checkmark prefix
                selected_label.append(label)
                selected_label.stylize("bold")  # Make the whole label bold
                return selected_label

        return label

    class AddToSelectionList(events.Message):
        """Message posted when folders should be added to selection list."""

        def __init__(self, sender, paths: List[Path]) -> None:
            super().__init__()
            self.sender = sender
            self.paths = paths

    class NavigateToFolder(events.Message):
        """Message posted when user double-clicks to navigate into a folder."""

        def __init__(self, sender, path: Path) -> None:
            super().__init__()
            self.sender = sender
            self.path = path

    class SelectFile(events.Message):
        """Message posted when user double-clicks a file to select it."""

        def __init__(self, sender, path: Path) -> None:
            super().__init__()
            self.sender = sender
            self.path = path
