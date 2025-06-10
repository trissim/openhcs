"""
OpenHCS adapter for textual-universal-directorytree.

This module provides a simplified adapter that creates OpenHCS-aware DirectoryTree
widgets by using standard UPath but with OpenHCS FileManager integration.
"""

import logging
from pathlib import Path
from typing import Union, Iterable, Optional, List

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
