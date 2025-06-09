"""
OpenHCS adapter for textual-universal-directorytree.

This module provides a simplified adapter that creates OpenHCS-aware DirectoryTree
widgets by using standard UPath but with OpenHCS FileManager integration.
"""

import logging
from pathlib import Path
from typing import Union, Iterable

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
        **kwargs
    ):
        """
        Initialize OpenHCS DirectoryTree.

        Args:
            filemanager: OpenHCS FileManager instance
            backend: Backend to use (DISK, MEMORY, etc.)
            path: Initial path to display
            show_hidden: Whether to show hidden files (default: False)
            **kwargs: Additional arguments passed to UniversalDirectoryTree
        """
        self.filemanager = filemanager
        self.backend = backend
        self.show_hidden = show_hidden

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
        """Filter paths to optionally hide hidden files.

        Args:
            paths: The paths to be filtered.

        Returns:
            The filtered paths.
        """
        if self.show_hidden:
            # Show all paths
            return paths
        else:
            # Filter out hidden files (starting with .)
            return [path for path in paths if not path.name.startswith('.')]
