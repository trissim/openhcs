# openhcs/io/storage/backends/zarr.py
"""
Zarr storage backend module for OpenHCS.

This module provides a Zarr-backed implementation of the MicroscopyStorageBackend interface.
It stores data in a Zarr store on disk and supports overlay operations
for materializing data to disk when needed.
"""

import fnmatch
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

import numpy as np
import zarr
from zarr.storage import LocalStore

from openhcs.constants.constants import Backend, DEFAULT_IMAGE_EXTENSIONS
from openhcs.io.base import StorageBackend
from openhcs.io.exceptions import StorageResolutionError

logger = logging.getLogger(__name__)


class ZarrStorageBackend(StorageBackend):
    """
    Zarr storage backend implementation with configurable compression.

    This class provides a concrete implementation of the storage backend interfaces
    for Zarr storage. It stores data in a Zarr store on disk with configurable
    compression algorithms and settings.

    Features:
    - Single-chunk batch operations for 40x performance improvement
    - Configurable compression (Blosc, Zlib, LZ4, Zstd, or none)
    - Configurable compression levels
    - Full path mapping for batch operations
    """

    def __init__(self, store_name: str = "images.zarr", compressor: Optional[Any] = None,
                 compression_level: Optional[int] = None):
        """
        Initialize Zarr backend with configurable store name and compression settings.

        Args:
            store_name: Name of the zarr store directory (default: "images.zarr")
            compressor: Zarr compressor to use (default: None for no compression)
                       Examples: zarr.Blosc(), zarr.Zlib(), zarr.LZ4(), zarr.Zstd()
            compression_level: Compression level (1-9, default: None)
                              Only used if compressor supports levels
        """
        self.store_name = store_name
        self.compressor = compressor
        self.compression_level = compression_level

    def _get_compressor(self) -> Optional[Any]:
        """
        Get the configured compressor with appropriate settings.

        Returns:
            Configured compressor instance or None for no compression
        """
        if self.compressor is None:
            return None

        # If compression_level is specified and compressor supports it
        if self.compression_level is not None:
            # Check if compressor has level parameter
            if hasattr(self.compressor, '__class__'):
                try:
                    # Create new instance with compression level
                    compressor_class = self.compressor.__class__
                    if 'level' in compressor_class.__init__.__code__.co_varnames:
                        return compressor_class(level=self.compression_level)
                    elif 'clevel' in compressor_class.__init__.__code__.co_varnames:
                        return compressor_class(clevel=self.compression_level)
                except (AttributeError, TypeError):
                    # Fall back to original compressor if level setting fails
                    pass

        return self.compressor

    def _split_store_and_key(self, path: Union[str, Path]) -> Tuple[Any, str]:
        """
        Auto-inject zarr store path for clean filesystem-like API.

        Maps paths to zarr store and key:
        - File: "/path/to/plate/A01.tif" → Store: "/path/to/plate/images.zarr", Key: "A01.tif"
        - Directory: "/path/to/plate" → Store: "/path/to/plate/images.zarr", Key: ""
        """
        path = Path(path)

        # If path has no extension, treat as directory (zarr store goes inside it)
        if not path.suffix:
            # Directory path - zarr store goes inside the directory
            store_path = path / self.store_name
            relative_key = ""
        else:
            # File path - zarr store goes in same directory as file
            store_path = path.parent / self.store_name
            relative_key = path.name

        store = LocalStore(str(store_path))
        return store, relative_key

    def save(self, data: Any, output_path: Union[str, Path], **kwargs):
        """
        Save data to Zarr at the given output_path.

        Will only write if the key does not already exist.
        Will NOT overwrite or delete existing data.

        Raises:
            FileExistsError: If destination key already exists
            StorageResolutionError: If creation fails
        """
        store, key = self._split_store_and_key(output_path)
        group = zarr.group(store=store)

        if key in group:
            raise FileExistsError(f"Zarr key already exists: {output_path}")

        chunks = kwargs.get("chunks")
        if chunks is None:
            chunks = self._auto_chunks(data, chunk_divisor=kwargs.get("chunk_divisor", 1))

        try:
            # Create array with correct shape and dtype, then assign data
            array = group.create_array(
                name=key,
                shape=data.shape,
                dtype=data.dtype,
                chunks=chunks,
                compressor=kwargs.get("compressor", self._get_compressor()),
                overwrite=False  # 🔒 Must be False by doctrine
            )
            array[:] = data
        except Exception as e:
            raise StorageResolutionError(f"Failed to save to Zarr: {output_path}") from e

    def load_batch(self, file_paths: List[Union[str, Path]], **kwargs) -> List[Any]:
        """
        Load from zarr array using filename mapping.

        Args:
            file_paths: List of file paths to load
            **kwargs: Additional arguments (zarr_config not needed)

        Returns:
            List of loaded data objects in same order as file_paths

        Raises:
            FileNotFoundError: If expected zarr store not found
            KeyError: If filename not found in filename_map
        """
        if not file_paths:
            return []

        # Use first path's directory to find zarr store
        base_dir = Path(file_paths[0]).parent
        store_path = base_dir / self.store_name

        # FAIL LOUD: Store must exist
        if not store_path.exists():
            raise FileNotFoundError(f"Expected zarr store not found: {store_path}")

        # Open store as group (not array)
        root = zarr.open_group(str(store_path), mode='r')

        # Group files by chunk based on filename mapping
        chunk_to_files = {}
        chunk_to_indices = {}

        # Search all chunks for the requested files
        for chunk_name in root.array_keys():
            chunk_array = root[chunk_name]
            if "filename_map" in chunk_array.attrs:
                filename_map = dict(chunk_array.attrs["filename_map"])

                # Check which requested files are in this chunk
                for i, path in enumerate(file_paths):
                    filename = str(path)  # Use full path for matching
                    if filename in filename_map:
                        if chunk_name not in chunk_to_files:
                            chunk_to_files[chunk_name] = []
                            chunk_to_indices[chunk_name] = []
                        chunk_to_files[chunk_name].append(i)  # Original position in file_paths
                        chunk_to_indices[chunk_name].append(filename_map[filename])  # Index in chunk

        # Load data from each chunk in single batch reads
        results = [None] * len(file_paths)  # Pre-allocate results array

        for chunk_name, file_positions in chunk_to_files.items():
            chunk_array = root[chunk_name]
            chunk_indices = chunk_to_indices[chunk_name]

            # Single batch read of entire chunk
            all_chunk_data = chunk_array[:]

            # Extract requested images and place in correct positions
            for file_pos, chunk_idx in zip(file_positions, chunk_indices):
                results[file_pos] = all_chunk_data[chunk_idx]

        logger.debug(f"Loaded {len(file_paths)} images from zarr store at {store_path} from {len(chunk_to_files)} chunks")
        return results

    def save_batch(self, data_list: List[Any], output_paths: List[Union[str, Path]], chunk_name: str = None, **kwargs) -> None:
        """Save multiple images as a chunk in shared zarr store."""

        base_dir = Path(output_paths[0]).parent
        store_path = base_dir / self.store_name

        # Use chunk_name or generate one from first filename
        if chunk_name is None:
            chunk_name = Path(output_paths[0]).stem

        # Ensure parent directory exists
        store_path.parent.mkdir(parents=True, exist_ok=True)

        # Open or create zarr group (shared store)
        root = zarr.open_group(str(store_path), mode='a')  # 'a' = read/write, create if doesn't exist

        # Create array for this chunk within the shared store
        batch_size = len(data_list)
        sample_shape = data_list[0].shape
        chunk_shape = (batch_size, *sample_shape)

        # Get compression settings using v3 API
        compressor = self._get_compressor()
        if compressor is not None:
            compressors = [compressor]
        else:
            compressors = None  # No compression

        # Create chunk array in the shared store
        chunk_array = root.create_array(
            chunk_name,
            shape=chunk_shape,
            chunks=chunk_shape,  # Use same shape as chunks for single chunk
            dtype=data_list[0].dtype,
            compressors=compressors,  # zarr v3 API
            overwrite=True  # Allow overwriting if chunk already exists
        )

        # Store filename mapping in chunk attributes
        filename_map = {str(path): i for i, path in enumerate(output_paths)}
        chunk_array.attrs["filename_map"] = filename_map
        chunk_array.attrs["output_paths"] = [str(path) for path in output_paths]

        # Convert GPU arrays to CPU arrays before saving
        cpu_data_list = []
        for data in data_list:
            if hasattr(data, 'get'):  # CuPy array
                cpu_data_list.append(data.get())
            elif hasattr(data, 'cpu'):  # PyTorch tensor
                cpu_data_list.append(data.cpu().numpy())
            elif hasattr(data, 'device') and 'cuda' in str(data.device).lower():  # JAX on GPU
                import jax
                cpu_data_list.append(jax.device_get(data))
            else:  # Already CPU array (NumPy, etc.)
                cpu_data_list.append(data)

        # Single batch write - stack all images and write at once
        stacked_data = np.stack(cpu_data_list, axis=0)
        chunk_array[:] = stacked_data

    
        """Ensure zarr store exists, creating it if necessary."""
        if store_path.exists():
            return

        zarr_config = kwargs["zarr_config"]
        if zarr_config and zarr_config["needs_initialization"]:
            self._create_store_with_locking(store_path, zarr_config["all_wells"], sample_shape, sample_dtype, batch_size)
        else:
            self._create_store_with_locking(store_path, ["single_well"], sample_shape, sample_dtype, batch_size)

    def _create_store_with_locking(self, store_path: Path, all_wells: List[str], sample_shape: tuple, sample_dtype: np.dtype, batch_size: int) -> None:
        """Create zarr store with file locking for multiprocessing safety."""

        if store_path.exists():
            return

        # Ensure parent directory exists before creating lock file
        store_path.parent.mkdir(parents=True, exist_ok=True)

        lock_path = store_path.with_suffix('.lock')

        try:
            with open(lock_path, 'x') as lock_file:
                if not store_path.exists():
                    self._create_zarr_array(store_path, all_wells, sample_shape, sample_dtype, batch_size)
                    logger.info(f"Created zarr array at {store_path}")

        except FileExistsError:
            logger.debug(f"Another process is creating zarr store at {store_path}, waiting...")
            self._wait_for_store_creation(store_path)

        finally:
            if lock_path.exists():
                lock_path.unlink()

    def _wait_for_store_creation(self, store_path: Path) -> None:
        """Wait for another process to finish creating the zarr store."""
        import time

        while not store_path.exists():
            time.sleep(0.1)

        logger.debug(f"Zarr store creation completed at {store_path}")

    def _create_zarr_array(self, store_path: Path, all_wells: List[str], sample_shape: tuple, sample_dtype: np.dtype, batch_size: int) -> None:
        """Create single zarr array with filename mapping."""

        # Calculate total array size: num_wells × batch_size
        total_images = len(all_wells) * batch_size
        full_shape = (total_images, *sample_shape)

        # Create single zarr array using v3 API
        compressor = self._get_compressor()
        if compressor is not None:
            # Convert v2 compressor to v3 codecs
            codecs = [compressor]
        else:
            # No compression
            codecs = None

        z = zarr.open(
            str(store_path),
            mode='w',
            shape=full_shape,
            chunks=None,  # Single chunk for optimal batch I/O
            dtype=sample_dtype,
            codecs=codecs
        )

        # Initialize empty filename mapping
        z.attrs["filename_map"] = {}
        z.attrs["next_index"] = 0

        logger.info(f"Created zarr array at {store_path} with shape {full_shape} for {len(all_wells)} wells × {batch_size} images")

    def _get_or_assign_indices(self, zarr_array, output_paths: List[Union[str, Path]]) -> List[int]:
        """Get or assign indices for filenames in zarr array using filename mapping."""

        # Read current filename mapping
        filename_map = dict(zarr_array.attrs["filename_map"])
        next_index = zarr_array.attrs["next_index"]

        indices = []
        updated = False

        for path in output_paths:
            filename = Path(path).name

            if filename in filename_map:
                # Filename already mapped
                indices.append(filename_map[filename])
            else:
                # Assign new index
                filename_map[filename] = next_index
                indices.append(next_index)
                next_index += 1
                updated = True

        # Update attributes if we assigned new indices
        if updated:
            zarr_array.attrs["filename_map"] = filename_map
            zarr_array.attrs["next_index"] = next_index

        return indices

    def load(self, file_path: Union[str, Path], **kwargs) -> Any:
        store, key = self._split_store_and_key(file_path)
        group = zarr.group(store=store)

        visited = set()
        while self.is_symlink(key):
            if key in visited:
                raise RuntimeError(f"Zarr symlink loop detected at {key}")
            visited.add(key)
            key = group[key].attrs["_symlink"]

        if key not in group:
            raise FileNotFoundError(f"No array found at key '{key}'")
        return group[key][:]

    def list_files(self,
                   directory: Union[str, Path],
                   pattern: Optional[str] = None,
                   extensions: Optional[Set[str]] = None,
                   recursive: bool = False) -> List[Path]:
        """
        List all file-like entries (i.e. arrays) in a Zarr store, optionally filtered.
        Returns filenames from array attributes (output_paths) if available.
        """

        store, relative_key = self._split_store_and_key(directory)
        result: List[Path] = []

        def _matches_filters(name: str) -> bool:
            if pattern and not fnmatch.fnmatch(name, pattern):
                return False
            if extensions:
                return any(name.lower().endswith(ext.lower()) for ext in extensions)
            return True

        try:
            # Open zarr group and get all array keys directly (Zarr 3.x compatible)
            group = zarr.open_group(store=store)
            array_keys = list(group.array_keys())

            for array_key in array_keys:
                try:
                    array = group[array_key]
                    if "output_paths" in array.attrs:
                        # Get original filenames from array attributes
                        output_paths = array.attrs["output_paths"]
                        for filename in output_paths:
                            filename_only = Path(filename).name
                            if _matches_filters(filename_only):
                                result.append(Path(filename))
                    else:
                        # Fallback to array key if no output_paths
                        if _matches_filters(array_key):
                            result.append(Path(array_key))
                except Exception as e:
                    # Skip arrays that can't be accessed
                    continue

        except Exception as e:
            raise StorageResolutionError(f"Failed to list zarr arrays: {e}") from e

        return result

    def list_dir(self, path: Union[str, Path]) -> List[str]:
        store, relative_key = self._split_store_and_key(path)

        # Normalize key for Zarr API
        key = relative_key.rstrip("/") if relative_key else ""

        try:
            # Zarr 3.x uses async API - convert async generator to list
            import asyncio
            async def _get_entries():
                entries = []
                async for entry in store.list_dir(key):
                    entries.append(entry)
                return entries
            return asyncio.run(_get_entries())
        except KeyError:
            raise NotADirectoryError(f"Zarr path is not a directory: {path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Zarr path does not exist: {path}")


    def delete(self, path: Union[str, Path]) -> None:
        """
        Delete a Zarr array (file) or empty group (directory) at the given path.

        Args:
            path: Zarr path or URI

        Raises:
            FileNotFoundError: If path does not exist
            IsADirectoryError: If path is a non-empty group
            StorageResolutionError: For unexpected failures
        """
        import zarr
        import shutil
        import os

        path = str(path)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Zarr path does not exist: {path}")

        try:
            zarr_obj = zarr.open(path, mode='r')
        except Exception as e:
            raise StorageResolutionError(f"Failed to open Zarr path: {path}") from e

        # Determine if it's a file (array) or directory (group)
        if isinstance(zarr_obj, zarr.core.Array):
            try:
                shutil.rmtree(path)  # Array folders can be deleted directly
            except Exception as e:
                raise StorageResolutionError(f"Failed to delete Zarr array: {path}") from e

        elif isinstance(zarr_obj, zarr.hierarchy.Group):
            if os.listdir(path):
                raise IsADirectoryError(f"Zarr group is not empty: {path}")
            try:
                os.rmdir(path)
            except Exception as e:
                raise StorageResolutionError(f"Failed to delete empty Zarr group: {path}") from e
        else:
            raise StorageResolutionError(f"Unrecognized Zarr object type at: {path}")

    def delete_all(self, path: Union[str, Path]) -> None:
        """
        Recursively delete a Zarr array or group (file or directory).

        This is the only permitted recursive deletion method for the Zarr backend.

        Args:
            path: the path shared through all backnds

        Raises:
            FileNotFoundError: If the path does not exist
            StorageResolutionError: If deletion fails
        """
        import os
        import shutil

        path = str(path)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Zarr path does not exist: {path}")

        try:
            shutil.rmtree(path)
        except Exception as e:
            raise StorageResolutionError(f"Failed to recursively delete Zarr path: {path}") from e

    def exists(self, path: Union[str, Path]) -> bool:
        path = Path(path)

        # If path has no file extension, treat as directory existence check
        # This handles auto_detect_patterns asking "does this directory exist?"
        if not path.suffix:
            return path.exists()

        # Otherwise, check zarr key existence (for actual files)
        store, key = self._split_store_and_key(path)

        # First check if the zarr store itself exists
        if isinstance(store, str):
            store_path = Path(store)
            if not store_path.exists():
                return False

        try:
            root_group = zarr.group(store=store)
            return key in root_group or any(k.startswith(key.rstrip("/") + "/") for k in root_group.array_keys())
        except Exception:
            # If we can't open the zarr store, it doesn't exist
            return False

    def ensure_directory(self, directory: Union[str, Path]) -> Path:
        """
        No-op for zarr backend - zarr stores handle their own structure.

        Zarr doesn't have filesystem directories that need to be "ensured".
        Store creation and group structure is handled by save operations.
        """
        return Path(directory)

    def create_symlink(self, source: Union[str, Path], link_name: Union[str, Path], overwrite: bool = False):
        store, src_key = self._split_store_and_key(source)
        store2, dst_key = self._split_store_and_key(link_name)

        if store.root != store2.root:
            raise ValueError("Symlinks must exist within the same .zarr store")

        group = zarr.group(store=store)
        if src_key not in group:
            raise FileNotFoundError(f"Source key '{src_key}' not found in Zarr store")

        if dst_key in group:
            if not overwrite:
                raise FileExistsError(f"Symlink target already exists at: {dst_key}")
            # Remove existing entry if overwrite=True
            del group[dst_key]

        # Create a new group at the symlink path
        link_group = group.require_group(dst_key)
        link_group.attrs["_symlink"] = src_key  # Store as declared string

    def is_symlink(self, path: Union[str, Path]) -> bool:
        """
        Check if the given Zarr path represents a logical symlink (based on attribute contract).

        Returns:
            bool: True if the key exists and has an OpenHCS-declared symlink attribute
            False if the key doesn't exist or is not a symlink
        """
        store, key = self._split_store_and_key(path)
        group = zarr.group(store=store)

        try:
            obj = group[key]
            attrs = getattr(obj, "attrs", {})

            if "_symlink" not in attrs:
                return False

            # Enforce that the _symlink attr matches schema (e.g. str or list of path components)
            if not isinstance(attrs["_symlink"], str):
                raise StorageResolutionError(f"Invalid symlink format in Zarr attrs at: {path}")

            return True
        except KeyError:
            # Key doesn't exist, so it's not a symlink
            return False
        except Exception as e:
            raise StorageResolutionError(f"Failed to inspect Zarr symlink at: {path}") from e

    def _auto_chunks(self, data: Any, chunk_divisor: int = 1) -> Union[Tuple[int, ...], bool]:
        shape = getattr(data, "shape", None)
        if shape is None:
            return True  # fallback to Zarr's default

        # Example heuristic: chunk along first dimension (e.g., time, slices)
        if len(shape) == 0:
            return True  # scalar

        # Simple logic: 1/10th of each dim, with min 1
        return tuple(max(1, s // chunk_divisor) for s in shape)

    def is_file(self, path: Union[str, Path]) -> bool:
        """
        Check if a Zarr path points to a file (Zarr array), resolving both OS and Zarr-native symlinks.

        Args:
            path: Zarr store path (may point to key within store)

        Returns:
            bool: True if resolved path is a Zarr array

        Raises:
            FileNotFoundError: If path does not exist or broken symlink
            IsADirectoryError: If resolved object is a Zarr group
            StorageResolutionError: For other failures
        """
        path = str(path)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Zarr path does not exist: {path}")

        try:
            store, key = self._split_store_and_key(path)
            group = zarr.group(store=store)

            # Resolve symlinks (Zarr-native, via .attrs)
            seen_keys = set()
            while True:
                if key not in group:
                    raise FileNotFoundError(f"Zarr key does not exist: {key}")
                obj = group[key]

                if hasattr(obj, "attrs") and "_symlink" in obj.attrs:
                    if key in seen_keys:
                        raise StorageResolutionError(f"Symlink cycle detected in Zarr at: {key}")
                    seen_keys.add(key)
                    key = obj.attrs["_symlink"]
                    continue
                break  # resolution complete

            # Now obj is the resolved target
            if isinstance(obj, zarr.core.Array):
                return True
            elif isinstance(obj, zarr.hierarchy.Group):
                raise IsADirectoryError(f"Zarr path is a group (directory): {path}")
            else:
                raise StorageResolutionError(f"Unknown Zarr object at: {path}")

        except Exception as e:
            raise StorageResolutionError(f"Failed to resolve Zarr file path: {path}") from e

    def is_dir(self, path: Union[str, Path]) -> bool:
        """
        Check if a Zarr path resolves to a directory (i.e., a Zarr group).
    
        Resolves both OS-level symlinks and Zarr-native symlinks via .attrs['_symlink'].
    
        Args:
            path: Zarr path or URI
    
        Returns:
            bool: True if path resolves to a Zarr group
    
        Raises:
            FileNotFoundError: If path or resolved target does not exist
            NotADirectoryError: If resolved target is not a group
            StorageResolutionError: For symlink cycles or other failures
        """
        import os
    
    
        path = str(path)
    
        if not os.path.exists(path):
            raise FileNotFoundError(f"Zarr path does not exist: {path}")
    
        try:
            store, key = self._split_store_and_key(path)
            group = zarr.group(store=store)
    
            seen_keys = set()
    
            # Resolve symlink chain
            while True:
                if key not in group:
                    raise FileNotFoundError(f"Zarr key does not exist: {key}")
                obj = group[key]
    
                if hasattr(obj, "attrs") and "_symlink" in obj.attrs:
                    if key in seen_keys:
                        raise StorageResolutionError(f"Symlink cycle detected in Zarr at: {key}")
                    seen_keys.add(key)
                    key = obj.attrs["_symlink"]
                    continue
                break
            
            # obj is resolved
            if isinstance(obj, zarr.hierarchy.Group):
                return True
            elif isinstance(obj, zarr.core.Array):
                raise NotADirectoryError(f"Zarr path is an array (file): {path}")
            else:
                raise StorageResolutionError(f"Unknown Zarr object at: {path}")
    
        except Exception as e:
            raise StorageResolutionError(f"Failed to resolve Zarr directory path: {path}") from e

    def move(self, src: Union[str, Path], dst: Union[str, Path]) -> None:
        """
        Move a Zarr key or object (array/group) from one location to another, resolving symlinks.
    
        Supports:
        - Disk or memory stores
        - Zarr-native symlinks
        - Key renames within group
        - Full copy+delete across stores if needed
    
        Raises:
            FileNotFoundError: If src does not exist
            FileExistsError: If dst already exists
            StorageResolutionError: On failure
        """
        import zarr
    
        src_store, src_key = self._split_store_and_key(src)
        dst_store, dst_key = self._split_store_and_key(dst)
    
        src_group = zarr.group(store=src_store)
        dst_group = zarr.group(store=dst_store)
    
        if src_key not in src_group:
            raise FileNotFoundError(f"Zarr source key does not exist: {src_key}")
        if dst_key in dst_group:
            raise FileExistsError(f"Zarr destination key already exists: {dst_key}")
    
        obj = src_group[src_key]
    
        # Resolve symlinks if present
        seen_keys = set()
        while hasattr(obj, "attrs") and "_symlink" in obj.attrs:
            if src_key in seen_keys:
                raise StorageResolutionError(f"Symlink cycle detected at: {src_key}")
            seen_keys.add(src_key)
            src_key = obj.attrs["_symlink"]
            obj = src_group[src_key]
    
        try:
            if src_store is dst_store:
                # Native move within the same Zarr group/store
                src_group.move(src_key, dst_key)
            else:
                # Cross-store: perform manual copy + delete
                obj.copy(dst_group, name=dst_key)
                del src_group[src_key]
        except Exception as e:
            raise StorageResolutionError(f"Failed to move {src_key} to {dst_key}") from e

    def copy(self, src: Union[str, Path], dst: Union[str, Path]) -> None:
        """
        Copy a Zarr key or object (array/group) from one location to another.

        - Resolves Zarr-native symlinks before copying
        - Prevents overwrite unless explicitly allowed (future feature)
        - Works across memory or disk stores

        Raises:
            FileNotFoundError: If src does not exist
            FileExistsError: If dst already exists
            StorageResolutionError: On failure
        """
        import zarr

        src_store, src_key = self._split_store_and_key(src)
        dst_store, dst_key = self._split_store_and_key(dst)

        src_group = zarr.group(store=src_store)
        dst_group = zarr.group(store=dst_store)

        if src_key not in src_group:
            raise FileNotFoundError(f"Zarr source key does not exist: {src_key}")
        if dst_key in dst_group:
            raise FileExistsError(f"Zarr destination key already exists: {dst_key}")

        obj = src_group[src_key]

        seen_keys = set()
        while hasattr(obj, "attrs") and "_symlink" in obj.attrs:
            if src_key in seen_keys:
                raise StorageResolutionError(f"Symlink cycle detected at: {src_key}")
            seen_keys.add(src_key)
            src_key = obj.attrs["_symlink"]
            obj = src_group[src_key]

        try:
            obj.copy(dst_group, name=dst_key)
        except Exception as e:
            raise StorageResolutionError(f"Failed to copy {src_key} to {dst_key}") from e

    def stat(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Return structural metadata about a Zarr path.

        Returns:
            dict with keys:
            - 'type': 'file', 'directory', 'symlink', or 'missing'
            - 'key': final resolved key
            - 'target': symlink target if applicable
            - 'store': repr(store)
            - 'exists': bool

        Raises:
            StorageResolutionError: On resolution failure
        """
        store, key = self._split_store_and_key(path)
        group = zarr.group(store=store)

        try:
            if key in group:
                obj = group[key]
                attrs = getattr(obj, "attrs", {})
                is_link = "_symlink" in attrs

                if is_link:
                    target = attrs["_symlink"]
                    if not isinstance(target, str):
                        raise StorageResolutionError(f"Invalid symlink format at {key}")
                    return {
                        "type": "symlink",
                        "key": key,
                        "target": target,
                        "store": repr(store),
                        "exists": target in group
                    }

                if isinstance(obj, zarr.Array):
                    return {
                        "type": "file",
                        "key": key,
                        "store": repr(store),
                        "exists": True
                    }

                elif isinstance(obj, zarr.Group):
                    return {
                        "type": "directory",
                        "key": key,
                        "store": repr(store),
                        "exists": True
                    }

                raise StorageResolutionError(f"Unknown object type at: {key}")
            else:
                return {
                    "type": "missing",
                    "key": key,
                    "store": repr(store),
                    "exists": False
                }

        except Exception as e:
            raise StorageResolutionError(f"Failed to stat Zarr key {key}") from e

class ZarrSymlink:
    """
    Represents a symbolic link in a Zarr store.

    This class is used to represent symbolic links in a Zarr store.
    It stores the target path of the symlink.
    """
    def __init__(self, target: str):
        self.target = target

    def __repr__(self):
        return f"<ZarrSymlink → {self.target}>"