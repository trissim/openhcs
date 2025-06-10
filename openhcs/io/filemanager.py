"""
FileManager directory operations.

This module contains the directory-related methods of the FileManager class,
including directory listing, existence checking, mkdir, symlink, and mirror operations.
"""

import logging
import os
from pathlib import Path
from typing import List, Set, Union, Tuple, Optional, Any

from openhcs.constants.constants import DEFAULT_IMAGE_EXTENSIONS
from openhcs.io.base import StorageBackend
from openhcs.io.exceptions import PathMismatchError, StorageResolutionError
from openhcs.validation import validate_path_types, validate_backend_parameter
import traceback

logger = logging.getLogger(__name__)

class FileManager:

    def __init__(self, registry):
        """
        Initialize the file manager.

        Args:
            registry: Registry for storage backends. Must be provided.

        Raises:
            ValueError: If registry is not provided.

        Note:
            This class is a backend-agnostic router. It maintains no default backend
            or fallback behavior, and all state is instance-local and declarative.
            Each operation must explicitly specify which backend to use.

        Thread Safety:
            Each FileManager instance must be scoped to a single execution context.
            Do NOT share FileManager instances across pipelines or threads.
            For isolation, create a dedicated registry for each FileManager.
        """
        # Validate registry parameter
        if registry is None:
            raise ValueError("Registry must be provided to FileManager. Default fallback has been removed.")

        # Store registry
        self.registry = registry



        logger.debug("FileManager initialized with registry")

    def _get_backend(self, backend_name: str) -> StorageBackend:
        """
        Get a backend by name.

        This method uses the instance registry to get the backend instance directly.
        All FileManagers that use the same registry share the same backend instances.

        Args:
            backend_name: Name of the backend to get (e.g., "disk", "memory", "zarr")

        Returns:
            The backend instance

        Raises:
            StorageResolutionError: If the backend is not found in the registry

        Thread Safety:
            Backend instances are shared across all FileManager instances that use
            the same registry. This ensures shared state (especially for memory backend).
        """
        # Normalize backend name
        backend_name = backend_name.lower()

        if backend_name is None:
            raise StorageResolutionError(f"Backend '{backend_name}' not found in registry")

        try:
            # Get the backend instance from the registry dictionary
            if backend_name not in self.registry:
                raise KeyError(f"Backend '{backend_name}' not found in registry")

            # Return the backend instance directly
            return self.registry[backend_name]
        except Exception as e:
            raise StorageResolutionError(f"Failed to get backend '{backend_name}': {e}") from e

    def load(self, file_path: Union[str, Path], backend: str, **kwargs) -> Any:
        """
        Load data from a file using the specified backend.

        This method assumes the file path is already backend-compatible and performs no inference or fallback.
        All semantic validation and file format decoding must occur within the backend.

        Args:
        file_path: Path to the file to load (str or Path)
        backend: Backend enum to use for loading (StorageBackendType.DISK, etc.) — POSITIONAL argument
        **kwargs: Additional keyword arguments passed to the backend's load method

        Returns:
        Any: The loaded data object

        Raises:
        StorageResolutionError: If the backend is not supported or load fails
        """

        try:
            backend_instance = self._get_backend(backend)
            return backend_instance.load(file_path, **kwargs)
        except StorageResolutionError: # Allow specific backend errors to propagate
            raise
        except Exception as e:
            logger.error(f"Unexpected error during load from {file_path} with backend {backend}: {e}", exc_info=True)
            raise StorageResolutionError(
                f"Failed to load file at {file_path} using backend '{backend}'"
            ) from e

    def save(self, data: Any, output_path: Union[str, Path], backend: str, **kwargs) -> None:
        """
        Save data to a file using the specified backend.

        This method performs no semantic transformation, format inference, or fallback logic.
        It assumes the output path and data are valid and structurally aligned with the backend’s expectations.

        Args:
        data: The data object to save (e.g., np.ndarray, torch.Tensor, dict, etc.)
        output_path: Destination path to write to (str or Path)
        backend: Backend enum to use for saving (StorageBackendType.DISK, etc.) — POSITIONAL argument
        **kwargs: Additional keyword arguments passed to the backend's save method

        Raises:
        StorageResolutionError: If the backend is not supported or save fails
        """

        try:
            backend_instance = self._get_backend(backend)
            backend_instance.save(data, output_path, **kwargs)
        except StorageResolutionError: # Allow specific backend errors to propagate if they are StorageResolutionError
            raise
        except Exception as e:
            logger.error(f"Unexpected error during save to {output_path} with backend {backend}: {e}", exc_info=True)
            raise StorageResolutionError(
                f"Failed to save data to {output_path} using backend '{backend}'"
            ) from e

    def list_image_files(self, directory: Union[str, Path], backend: str,
                         pattern: str = None, extensions: Set[str] = DEFAULT_IMAGE_EXTENSIONS, recursive: bool = False) -> List[str]:
        """
        List all image files in a directory using the specified backend.

        This method performs no semantic validation, normalization, or naming enforcement on the input path.
        It assumes the caller has provided a valid, backend-compatible path and merely dispatches it for execution.

        Note: ONLY backend is a POSITIONAL argument. Other parameters may remain as kwargs.

        Args:
            directory: Directory to search (str or Path)
            backend: Backend to use for listing ('disk', 'memory', 'zarr') - POSITIONAL
            pattern: Pattern to filter files (e.g., "*.tif") - can be keyword arg
            extensions: Set of file extensions to filter by - can be keyword arg
            recursive: Whether to search recursively - can be keyword arg

        Returns:
            List of string paths for image files found

        Raises:
            StorageResolutionError: If the backend is not supported
            TypeError: If directory is not a valid path type
            PathMismatchError: If the path scheme doesn't match the expected scheme for the backend
        """
        # Get backend instance
        backend_instance = self._get_backend(backend)

        # List image files and apply natural sorting
        from openhcs.core.utils import natural_sort
        files = backend_instance.list_files(str(directory), pattern, extensions, recursive)
        return natural_sort(files)


    def list_files(self, directory: Union[str, Path], backend: str,
                   pattern: str = None, extensions: Set[str] = None, recursive: bool = False) -> List[str]:
        """
        List all files in a directory using the specified backend.

        This method performs no semantic validation, normalization, or naming enforcement on the input path.
        It assumes the caller has provided a valid, backend-compatible path and merely dispatches it for execution.

        Note: ONLY backend is a POSITIONAL argument. Other parameters may remain as kwargs.

        Args:
            directory: Directory to search (str or Path)
            backend: Backend to use for listing ('disk', 'memory', 'zarr') - POSITIONAL
            pattern: Pattern to filter files (e.g., "*.txt") - can be keyword arg
            extensions: Set of file extensions to filter by - can be keyword arg
            recursive: Whether to search recursively - can be keyword arg

        Returns:
            List of string paths for files found

        Raises:
            StorageResolutionError: If the backend is not supported
            TypeError: If directory is not a valid path type
            PathMismatchError: If the path scheme doesn't match the expected scheme for the backend
        """
        # Get backend instance
        backend_instance = self._get_backend(backend)

        # List files and apply natural sorting
        from openhcs.core.utils import natural_sort
        files = backend_instance.list_files(str(directory), pattern, extensions, recursive)
        return natural_sort(files)


    def find_file_recursive(self, directory: Union[str, Path], backend: str, filename: str) -> Union[str, None]:
        """
        Find a file recursively in a directory using the specified backend.

        This is a convenience method that uses list_files with recursive=True and filters for the specific filename.

        Args:
            directory: Directory to search (str or Path)
            backend: Backend to use for listing ('disk', 'memory', 'zarr') - POSITIONAL
            filename: Name of the file to find

        Returns:
            String path to the file if found, None otherwise

        Raises:
            StorageResolutionError: If the backend is not supported
            TypeError: If directory is not a valid path type
            PathMismatchError: If the path scheme doesn't match the expected scheme for the backend
        """
        # List all files recursively
        all_files = self.list_files(directory, backend, recursive=True)

        # Filter for the specific filename
        for file_path in all_files:
            if Path(file_path).name == filename:
                return file_path

        # File not found
        return None


    def list_dir(self, path: Union[str, Path], backend: str) -> List[str]:
        if not isinstance(path, (str, Path)):
            raise TypeError(f"Expected str or Path, got {type(path)}")

        path = str(path)
        backend_instance = self._get_backend(backend)

        try:
            # Get directory listing and apply natural sorting
            from openhcs.core.utils import natural_sort
            entries = backend_instance.list_dir(str(path))
            return natural_sort(entries)
        except (FileNotFoundError, NotADirectoryError):
            # Let these bubble up for structural truth-checking
            raise
        except Exception as e:
            # Optional trace wrapper, no type mutation
            raise RuntimeError(f"Unexpected failure in list_dir({path}) for backend {backend}") from e

    def ensure_directory(self, directory: Union[str, Path], backend: str) -> str:
        """
        Ensure a directory exists, creating it if necessary.

        This method performs no semantic validation, normalization, or naming enforcement on the input path.
        It assumes the caller has provided a valid, backend-compatible path and merely dispatches it for execution.

        Note: ONLY backend is a POSITIONAL argument. All parameters are required.

        Args:
            directory: Directory to ensure exists (str or Path)
            backend: Backend to use for directory operations ('disk', 'memory', 'zarr') - POSITIONAL

        Returns:
            String path to the directory

        Raises:
            StorageResolutionError: If the backend is not supported
            TypeError: If directory is not a valid path type
            PathMismatchError: If the path scheme doesn't match the expected scheme for the backend
        """
        # Get backend instance
        backend_instance = self._get_backend(backend)

        # Ensure directory
        return backend_instance.ensure_directory(str(directory))



    def exists(self, path: Union[str, Path], backend: str) -> bool:
        """
        Check if a path exists.

        This method performs no semantic validation, normalization, or naming enforcement on the input path.
        It assumes the caller has provided a valid, backend-compatible path and merely dispatches it for execution.

        Note: ONLY backend is a POSITIONAL argument. All parameters are required.

        Args:
            path: Path to check (str or Path)
            backend: Backend to use for checking ('disk', 'memory', 'zarr') - POSITIONAL

        Returns:
            True if the path exists, False otherwise

        Raises:
            StorageResolutionError: If the backend is not supported
            TypeError: If path is not a valid path type
            PathMismatchError: If the path scheme doesn't match the expected scheme for the backend
        """
        # Get backend instance
        backend_instance = self._get_backend(backend)

        # Check if path exists
        return backend_instance.exists(str(path))


    def mirror_directory_with_symlinks(
        self,
        source_dir: Union[str, Path],
        target_dir: Union[str, Path],
        backend: str,
        recursive: bool = True,
        overwrite_symlinks_only: bool = False
    ) -> int:
        """
        Mirror a directory structure from source to target and create symlinks to all files.

        This method performs no semantic validation, normalization, or naming enforcement on the input paths.
        It assumes the caller has provided valid, backend-compatible paths and merely dispatches them for execution.

        By default, this method will NOT overwrite existing files. Use overwrite_symlinks_only=True to allow
        overwriting existing symlinks (but not regular files).

        Note: ONLY backend is a POSITIONAL argument. Other parameters may remain as kwargs.

        Args:
            source_dir: Path to the source directory to mirror (str or Path)
            target_dir: Path to the target directory where the mirrored structure will be created (str or Path)
            backend: Backend to use for mirroring ('disk', 'memory', 'zarr') - POSITIONAL
            recursive: Whether to recursively mirror subdirectories - can be keyword arg
            overwrite_symlinks_only: If True, allows overwriting existing symlinks but blocks overwriting regular files.
                                    If False (default), no overwriting is allowed. - can be keyword arg

        Returns:
            int: Number of symlinks created

        Raises:
            StorageResolutionError: If the backend is not supported
            FileExistsError: If target files exist and overwrite_symlinks_only=False, or if trying to overwrite regular files
            TypeError: If source_dir or target_dir is not a valid path type
            PathMismatchError: If the path scheme doesn't match the expected scheme for the backend
        """
        # Get backend instance
        backend_instance = self._get_backend(backend)
        # Mirror the directory structure and create symlinks for files recursively
        self.ensure_directory(target_dir, backend)
        try:
            # Ensure target directory exists
            
            # Count symlinks
            symlink_count = 0
            
            # Get all directories under source_dir (including source_dir itself)

            _, all_files = self.collect_dirs_and_files(source_dir, backend, recursive=True)

            # 1. Ensure base target exists
            self.ensure_directory(target_dir, backend)

            # 2. Symlink all file paths
            for file_path in all_files:
                rel_path = Path(file_path).relative_to(Path(source_dir))
                symlink_path = Path(target_dir) / rel_path
                self.create_symlink(file_path, str(symlink_path), backend, overwrite_symlinks_only=overwrite_symlinks_only)
                symlink_count += 1

            return symlink_count

        except Exception as e:
            raise StorageResolutionError(f"Failed to mirror directory {source_dir} to {target_dir} with backend {backend}") from e

    def create_symlink(
        self,
        source_path: Union[str, Path],
        symlink_path: Union[str, Path],
        backend: str,
        overwrite_symlinks_only: bool = False
    ) -> bool:
        """
        Create a symbolic link from source_path to symlink_path.

        This method performs no semantic validation, normalization, or naming enforcement on the input paths.
        It assumes the caller has provided valid, backend-compatible paths and merely dispatches them for execution.

        Note: ONLY backend is a POSITIONAL argument. All parameters are required.

        Args:
            source_path: Path to the source file or directory (str or Path)
            symlink_path: Path where the symlink should be created (str or Path)
            backend: Backend to use for symlink creation ('disk', 'memory', 'zarr') - POSITIONAL
            overwrite_symlinks_only: If True, only allow overwriting existing symlinks (not regular files)

        Returns:
            bool: True if successful, False otherwise

        Raises:
            StorageResolutionError: If the backend is not supported
            FileExistsError: If target exists and is not a symlink when overwrite_symlinks_only=True
            VFSTypeError: If source_path or symlink_path cannot be converted to internal path format
            PathMismatchError: If the path scheme doesn't match the expected scheme for the backend
        """
        # Get backend instance
        backend_instance = self._get_backend(backend)

        # Check if target exists and handle overwrite policy
        try:
            if backend_instance.exists(str(symlink_path)):
                if overwrite_symlinks_only:
                    # Check if existing target is a symlink
                    if not backend_instance.is_symlink(str(symlink_path)):
                        raise FileExistsError(
                            f"Target exists and is not a symlink (overwrite_symlinks_only=True): {symlink_path}"
                        )
                    # Target is a symlink, allow overwrite
                    backend_instance.create_symlink(str(source_path), str(symlink_path), overwrite=True)
                else:
                    # No overwrite allowed
                    raise FileExistsError(f"Target already exists: {symlink_path}")
            else:
                # Target doesn't exist, create new symlink
                backend_instance.create_symlink(str(source_path), str(symlink_path), overwrite=False)

            return True
        except FileExistsError:
            # Re-raise FileExistsError from our check or from backend
            raise
        except Exception as e:
            raise StorageResolutionError(
                f"Failed to create symlink from {source_path} to {symlink_path} with backend {backend}"
            ) from e

    def delete(self, path: Union[str, Path], backend: str, recursive: bool = False) -> bool:
        """
        Delete a file or directory.

        This method performs no semantic validation, normalization, or naming enforcement on the input path.
        It assumes the caller has provided a valid, backend-compatible path and merely dispatches it for execution.

        Note: ONLY backend is a POSITIONAL argument. All parameters are required.

        Args:
            path: Path to the file or directory to delete (str or Path)
            backend: Backend to use for deletion ('disk', 'memory', 'zarr') - POSITIONAL

        Returns:
            True if successful, False otherwise

        Raises:
            StorageResolutionError: If the backend is not supported
            FileNotFoundError: If the file does not exist
            TypeError: If the path is not a valid path type
        """
        # Get backend instance
        backend_instance = self._get_backend(backend)

        # Delete the file or directory
        try:
            # No virtual path conversion needed
            return backend_instance.delete(str(path))
        except Exception as e:
            raise StorageResolutionError(
                f"Failed to delete {path} with backend {backend}"
            ) from e

    def delete_all(self, path: Union[str, Path], backend: str) -> bool:
        """
        Recursively delete a file, symlink, or directory at the given path.
    
        This method performs no fallback, coercion, or resolution — it dispatches to the backend.
        All resolution and deletion behavior must be encoded in the backend's `delete_all()` method.
    
        Args:
            path: The path to delete
            backend: The backend key (e.g., 'disk', 'memory', 'zarr')
    
        Returns:
            True if successful
    
        Raises:
            StorageResolutionError: If the backend operation fails
            FileNotFoundError: If the path does not exist
            TypeError: If the path is not a str or Path
        """
        backend_instance = self._get_backend(backend)
        path_str = str(path)
    
        try:
            backend_instance.delete_all(path_str)
            return True
        except Exception as e:
            raise StorageResolutionError(
                f"Failed to delete_all({path_str}) using backend '{backend}'"
            ) from e


    def copy(self, source_path: Union[str, Path], dest_path: Union[str, Path], backend: str) -> bool:
        """
        Copy a file, directory, or symlink from source_path to dest_path using the given backend.

        - Will NOT overwrite existing files/directories.
        - Handles symlinks as first-class objects (not dereferenced).
        - Raises on broken links or mismatched structure.

        Raises:
            FileExistsError: If destination exists
            FileNotFoundError: If source does not exist
            StorageResolutionError: On backend failure
        """
        backend_instance = self._get_backend(backend)

        try:
            # Prevent overwriting
            if backend_instance.exists(dest_path):
                raise FileExistsError(f"Destination already exists: {dest_path}")

            # Ensure destination parent exists
            dest_parent = Path(dest_path).parent
            self.ensure_directory(dest_parent, backend)

            # Delegate to backend-native copy
            return backend_instance.copy(str(source_path), str(dest_path))
        except Exception as e:
            raise StorageResolutionError(
                f"Failed to copy from {source_path} to {dest_path} on backend {backend}"
            ) from e


    def move(self, source_path: Union[str, Path], dest_path: Union[str, Path], backend: str) -> bool:
        """
        Move a file, directory, or symlink from source_path to dest_path.

        - Will NOT overwrite.
        - Preserves symbolic identity (moves links as links).
        - Uses backend-native move if available.

        Raises:
            FileExistsError: If destination exists
            FileNotFoundError: If source is missing
            StorageResolutionError: On backend failure
        """
        backend_instance = self._get_backend(backend)

        try:
            if backend_instance.exists(dest_path):
                raise FileExistsError(f"Destination already exists: {dest_path}")

            dest_parent = Path(dest_path).parent
            self.ensure_directory(dest_parent, backend)
            return backend_instance.move(str(source_path), str(dest_path))

        except Exception as e:
            raise StorageResolutionError(
                f"Failed to move from {source_path} to {dest_path} on backend {backend}"
            ) from e
    
    def collect_dirs_and_files(
        self,
        base_dir: Union[str, Path],
        backend: str,
        recursive: bool = True
    ) -> Tuple[List[str], List[str]]:
        """
        Collect all valid directories and files starting from base_dir.
    
        Returns:
            (dirs, files): Lists of string paths for directories and files
        """
        base_dir = str(base_dir)
        stack = [base_dir]
        dirs: List[str] = []
        files: List[str] = []
    
        while stack:
            current_path = stack.pop()
    
            try:
                entries = self.list_dir(current_path, backend)
                dirs.append(current_path)
            except (NotADirectoryError, FileNotFoundError):
                files.append(current_path)
                continue
            except Exception as e:
                print(f"[collect_dirs_and_files] Unexpected error at {current_path}: {type(e).__name__} — {e}")
                continue  # Fail-safe: skip unexpected issues
            
            if entries is None:
                # Defensive fallback — entries must be iterable
                print(f"[collect_dirs_and_files] WARNING: list_dir() returned None at {current_path}")
                continue
            
            for entry in entries:
                full_path = str(Path(current_path) / entry)
                try:
                    self.list_dir(full_path, backend)
                    dirs.append(full_path)
                    if recursive:
                        stack.append(full_path)
                except (NotADirectoryError, FileNotFoundError):
                    files.append(full_path)
                except Exception as e:
                    print(f"[collect_dirs_and_files] Skipping {full_path}: {type(e).__name__} — {e}")
                    continue

        # Apply natural sorting to both dirs and files before returning
        from openhcs.core.utils import natural_sort
        return natural_sort(dirs), natural_sort(files)
    
    def is_file(self, path: Union[str, Path], backend: str) -> bool:
        """
        Check if a given path is a file using the specified backend.

        Args:
            path: Path to check (raw string or Path)
            backend: Backend key ('disk', 'memory', 'zarr') — must be positional

        Returns:
            bool: True if the path is a file, False otherwise

        Raises:
            StorageResolutionError: If resolution fails or backend misbehaves
            FileNotFoundError: If path does not exist
            IsADirectoryError: If path is a directory
        """
        try:
            backend_instance = self._get_backend(backend)
            return backend_instance.is_file(path)
        except (FileNotFoundError, IsADirectoryError):
            raise  # propagate known semantics
        except Exception as e:
            raise StorageResolutionError(
                f"Failed to check if {path} is a file with backend '{backend}'"
            ) from e

    def is_dir(self, path: Union[str, Path], backend: str) -> bool:
        """
        Check if a given path is a directory using the specified backend.

        Args:
            path: Path to check (raw string or Path)
            backend: Backend key ('disk', 'memory', 'zarr') — must be positional

        Returns:
            bool: True if the path is a directory, False if it's a file or doesn't exist

        Raises:
            StorageResolutionError: If resolution fails or backend misbehaves
        """
        try:
            backend_instance = self._get_backend(backend)
            return backend_instance.is_dir(path)
        except (FileNotFoundError, NotADirectoryError):
            # Return False for files or non-existent paths instead of raising
            return False
        except Exception as e:
            raise StorageResolutionError(
                f"Failed to check if {path} is a directory with backend '{backend}'"
            ) from e

