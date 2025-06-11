# openhcs/io/storage/backends/base.py
"""
Abstract base classes for storage backends.

This module defines the fundamental interfaces for storage backends,
independent of specific implementations. It establishes the contract
that all storage backends must fulfill.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, Union, Callable
from functools import wraps
from openhcs.constants.constants import Backend
from openhcs.io.exceptions import StorageResolutionError

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """
    Abstract base class for basic storage operations.

    Defines the fundamental operations required for interacting with a storage system,
    independent of specific data types like microscopy images.
    """

    @abstractmethod
    def load(self, file_path: Union[str, Path], **kwargs) -> Any:
        """
        Load data from a file.

        Args:
            file_path: Path to the file to load
            **kwargs: Additional arguments for the load operation

        Returns:
            The loaded data

        Raises:
            FileNotFoundError: If the file does not exist
            TypeError: If the file_path is not a valid path type
            ValueError: If the file cannot be loaded
        """
        pass

    @abstractmethod
    def save(self, data: Any, output_path: Union[str, Path], **kwargs) -> None:
        """
        Save data to a file.

        Args:
            data: The data to save
            output_path: Path where the data should be saved
            **kwargs: Additional arguments for the save operation

        Raises:
            TypeError: If the output_path is not a valid path type
            ValueError: If the data cannot be saved
        """
        pass

    @abstractmethod
    def list_files(self, directory: Union[str, Path], pattern: Optional[str] = None,
                  extensions: Optional[Set[str]] = None, recursive: bool = False) -> List[Path]:
        """
        List files in a directory, optionally filtering by pattern and extensions.

        Args:
            directory: Directory to search.
            pattern: Optional glob pattern to match filenames.
            extensions: Optional set of file extensions to filter by (e.g., {'.tif', '.png'}).
                        Extensions should include the dot and are case-insensitive.
            recursive: Whether to search recursively.

        Returns:
            List of paths to matching files.

        Raises:
            TypeError: If the directory is not a valid path type
            FileNotFoundError: If the directory does not exist
        """
        pass

    @abstractmethod
    def list_dir(self, path: Union[str, Path]) -> List[str]:
        """
        List the names of immediate entries in a directory.

        Args:
            path: Directory path to list.

        Returns:
            List of entry names (not full paths) in the directory.

        Raises:
            FileNotFoundError: If the path does not exist.
            NotADirectoryError: If the path is not a directory.
            TypeError: If the path is not a valid path type.
        """
        pass

    @abstractmethod
    def delete(self, file_path: Union[str, Path]) -> None:
        """
        Delete a file.

        Args:
            file_path: Path to the file to delete

        Raises:
            TypeError: If the file_path is not a valid path type
            FileNotFoundError: If the file does not exist
            ValueError: If the file cannot be deleted
        """
        pass

    @abstractmethod
    def delete_all(self, file_path: Union[str, Path]) -> None:
        """
        Deletes a file or a folder in full.

        Args:
            file_path: Path to the file to delete

        Raises:
            TypeError: If the file_path is not a valid path type
            ValueError: If the file cannot be deleted
        """
        pass


    @abstractmethod
    def ensure_directory(self, directory: Union[str, Path]) -> Path:
        """
        Ensure a directory exists, creating it if necessary.

        Args:
            directory: Path to the directory to ensure exists

        Returns:
            The path to the directory

        Raises:
            TypeError: If the directory is not a valid path type
            ValueError: If the directory cannot be created
        """
        pass


    @abstractmethod
    def create_symlink(self, source: Union[str, Path], link_name: Union[str, Path]):
        """
        Creates a symlink from source to link_name.

        Args:
            source: Path to the source file
            link_name: Path where the symlink should be created

        Raises:
            TypeError: If the path is not a valid path type
        """
        pass

    @abstractmethod
    def is_symlink(self, source: Union[str, Path]) -> bool:
        """
        Checks if a path is a symlink.

        Args:
            source: Path to the source file

        Raises:
            TypeError: If the path is not a valid path type
        """
    
    @abstractmethod
    def is_file(self, source: Union[str, Path]) -> bool:
        """
        Checks if a path is a file.

        Args:
            source: Path to the source file

        Raises:
            TypeError: If the path is not a valid path type
        """
    @abstractmethod
    def is_dir(self, source: Union[str, Path]) -> bool:
        """
        Checks if a path is a symlink.

        Args:
            source: Path to the source file

        Raises:
            TypeError: If the path is not a valid path type
        """
    
    @abstractmethod
    def move(self, src: Union[str, Path], dst: Union[str, Path]) -> None:
        """ 
        Move a file or directory from src to dst.

        Args:
            src: Path to the source file
            dst: Path to the destination file

        Raises:
            TypeError: If the path is not a valid path type
            FileNotFoundError: If the source file does not exist
            FileExistsError: If the destination file already exists
            ValueError: If the file cannot be moved
        """
        pass

    @abstractmethod
    def copy(self, src: Union[str, Path], dst: Union[str, Path]) -> None:
        """
        Copy a file or directory from src to dst.

        Args:
            src: Path to the source file
            dst: Path to the destination file

        Raises:
            TypeError: If the path is not a valid path type
            FileNotFoundError: If the source file does not exist
            FileExistsError: If the destination file already exists
            ValueError: If the file cannot be copied
        """
        pass

    @abstractmethod
    def stat(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get metadata for a file or directory.

        Args:
            src: Path to the source file

        Raises:
            TypeError: If the path is not a valid path type
            FileNotFoundError: If the source file does not exist
        """
        pass

    def exists(self, path: Union[str, Path]) -> bool:
        """
        Declarative truth test: does the path resolve to a valid object?

        A path only 'exists' if:
        - it is a valid file or directory
        - or it is a symlink that resolves to a valid file or directory

        Returns:
            bool: True if path structurally resolves to a real object
        """
        try:
            return self.is_file(path)
        except (FileNotFoundError, NotADirectoryError, StorageResolutionError):
            pass
        except IsADirectoryError:
            # Path exists but is a directory, so check if it's a valid directory
            try:
                return self.is_dir(path)
            except (FileNotFoundError, NotADirectoryError, StorageResolutionError):
                return False

        # If is_file failed for other reasons, try is_dir
        try:
            return self.is_dir(path)
        except (FileNotFoundError, NotADirectoryError, StorageResolutionError):
            return False


def _create_storage_registry() -> Dict[str, StorageBackend]:
    """
    Create a new storage registry.

    This function creates a dictionary mapping backend names to their respective
    storage backend instances. It is the canonical factory for creating backend
    registries in the system.

    Returns:
        A dictionary mapping backend names to backend instances

    Note:
        This is an internal factory function. Use the global storage_registry
        instance instead of calling this directly.
    """
    # Import here to avoid circular imports
    from openhcs.io.disk import DiskStorageBackend
    from openhcs.io.memory import MemoryStorageBackend
    from openhcs.io.zarr import ZarrStorageBackend

    return {
        Backend.DISK.value: DiskStorageBackend(),
        Backend.MEMORY.value: MemoryStorageBackend(),
        Backend.ZARR.value: ZarrStorageBackend()
    }


# Global singleton storage registry - created once at module import time
# This is the shared registry instance that all components should use
storage_registry = _create_storage_registry()


def reset_memory_backend() -> None:
    """
    Clear files from the memory backend while preserving directory structure.

    This function clears all file entries from the existing memory backend but preserves
    directory entries (None values). This prevents key collisions between plate executions
    while maintaining the directory structure needed for subsequent operations.

    Benefits over full reset:
    - Preserves directory structure created by path planner
    - Prevents "Parent path does not exist" errors on subsequent runs
    - Avoids key collisions for special inputs/outputs
    - Maintains performance by not recreating directory hierarchy

    Note:
        This only affects the memory backend. Other backends (disk, zarr) are not modified.
    """
    from openhcs.constants.constants import Backend

    # Clear files from existing memory backend while preserving directories
    memory_backend = storage_registry[Backend.MEMORY.value]
    memory_backend.clear_files_only()
    logger.info("Memory backend reset - files cleared, directories preserved")