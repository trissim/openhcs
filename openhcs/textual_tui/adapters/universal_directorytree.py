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
    - Left click: Select folder (no expansion)
    - Right click: Add to multi-selection
    - Double click: Add selected folders to selection list
    """

    def __init__(
        self,
        filemanager: FileManager,
        backend: Backend,
        path: Union[str, Path],
        show_hidden: bool = False,
        filter_extensions: Optional[List[str]] = None,
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
            **kwargs: Additional arguments passed to UniversalDirectoryTree
        """
        self.filemanager = filemanager
        self.backend = backend
        self.show_hidden = show_hidden
        self.filter_extensions = filter_extensions

        # Track multi-selection state
        self.multi_selected_paths: Set[Path] = set()
        self.last_click_time = 0
        self.double_click_threshold = 0.5  # seconds

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

        # Get the cursor position and find the node
        try:
            # Use the tree's cursor to get the currently highlighted node
            if hasattr(self, 'cursor_node') and self.cursor_node:
                clicked_node = self.cursor_node
            elif hasattr(self, 'highlighted_node') and self.highlighted_node:
                clicked_node = self.highlighted_node
            else:
                # Fallback: let the parent handle the click to set cursor, then get it
                super().on_click(event)
                if hasattr(self, 'cursor_node') and self.cursor_node:
                    clicked_node = self.cursor_node
                else:
                    return
        except Exception:
            # If we can't get the node, fall back to default behavior
            super().on_click(event)
            return

        # Get the path from the node
        if not hasattr(clicked_node, 'data') or not clicked_node.data:
            return

        node_path = Path(str(clicked_node.data))
        current_time = time.time()

        # Check if it's a double click
        is_double_click = (current_time - self.last_click_time) < self.double_click_threshold
        self.last_click_time = current_time

        # Handle different click types
        if event.button == 3:  # Right click
            self._handle_right_click(node_path)
            event.stop()  # Prevent default behavior
        elif is_double_click:
            self._handle_double_click(node_path)
            event.stop()  # Prevent default behavior
        else:  # Left click
            self._handle_left_click(node_path)
            # Don't stop event for left click - let it set cursor/selection

    def _handle_left_click(self, path: Path) -> None:
        """Handle left click - select folder without expansion."""
        logger.info(f"🔍 LEFT CLICK: Selecting {path}")

        # Clear multi-selection and select this path
        self.multi_selected_paths.clear()
        self.multi_selected_paths.add(path)

        # Post directory selected event
        self.post_message(self.DirectorySelected(self, path))

    def _handle_right_click(self, path: Path) -> None:
        """Handle right click - add to multi-selection."""
        logger.info(f"🔍 RIGHT CLICK: Adding {path} to multi-selection")

        # Add to multi-selection
        self.multi_selected_paths.add(path)

        # Post directory selected event for the newly added path
        self.post_message(self.DirectorySelected(self, path))

    def _handle_double_click(self, path: Path) -> None:
        """Handle double click - add all selected folders to selection list."""
        logger.info(f"🔍 DOUBLE CLICK: Adding selected folders to selection list")

        # If the double-clicked folder is not in multi-selection, add it
        if path not in self.multi_selected_paths:
            self.multi_selected_paths.add(path)

        # Post a custom message to add all selected paths to the selection list
        self.post_message(self.AddToSelectionList(self, list(self.multi_selected_paths)))

        # Clear multi-selection after adding to list
        self.multi_selected_paths.clear()

    class AddToSelectionList(events.Message):
        """Message posted when folders should be added to selection list."""

        def __init__(self, sender, paths: List[Path]) -> None:
            super().__init__()
            self.sender = sender
            self.paths = paths
