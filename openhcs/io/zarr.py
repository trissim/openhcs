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

import zarr
from zarr.storage import LocalStore

from openhcs.io.base import StorageBackend

logger = logging.getLogger(__name__)


class ZarrStorageBackend(StorageBackend):
    """
    Zarr storage backend implementation.

    This class provides a concrete implementation of the storage backend interfaces
    for Zarr storage. It stores data in a Zarr store on disk.
    """

    def __init__(self, store_name: str = "images.zarr"):
        """
        Initialize Zarr backend with configurable store name.

        Args:
            store_name: Name of the zarr store directory (default: "images.zarr")
        """
        self.store_name = store_name
    def _split_store_and_key(self, path: Union[str, Path]) -> Tuple[Any, str]:
        """
        Auto-inject zarr store path for clean filesystem-like API.

        Maps clean paths like "/path/to/plate/A01.tif" to:
        - Store: "/path/to/plate/{self.store_name}"
        - Key: "A01.tif"
        """
        path = Path(path)

        # Store is always self.store_name in the same directory as the file
        store_path = path.parent / self.store_name
        store = LocalStore(str(store_path))

        # Key is just the filename
        relative_key = path.name

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
            group.create_dataset(
                name=key,
                data=data,
                chunks=chunks,
                compressor=kwargs.get("compressor", None),
                overwrite=False  # 🔒 Must be False by doctrine
            )
        except Exception as e:
            raise StorageResolutionError(f"Failed to save to Zarr: {output_path}") from e

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
        Recursively list all file-like entries (i.e. leaf arrays) in a Zarr store, optionally filtered.
        """

        store, relative_key = self._split_store_and_key(directory)
        prefix = relative_key.rstrip("/") if relative_key else ""

        result: List[Path] = []

        def is_array(key: str) -> bool:
            # Zarr arrays have a `.zarray` marker
            return f"{key}/.zarray" in store

        def visit_group(group_prefix: str):
            try:
                entries = store.listdir(group_prefix)
            except KeyError:
                raise NotADirectoryError(f"Zarr path is not a directory: {group_prefix}")
            for name in entries:
                full_key = f"{group_prefix}/{name}".strip("/")
                if is_array(full_key):
                    if _matches_filters(name):
                        result.append(Path(full_key))
                elif recursive:
                    visit_group(full_key)

        def _matches_filters(name: str) -> bool:
            if pattern and not fnmatch.fnmatch(name, pattern):
                return False
            if extensions:
                return any(name.lower().endswith(ext.lower()) for ext in extensions)
            return True

        visit_group(prefix)

        return result

    def list_dir(self, path: Union[str, Path]) -> List[str]:
        store, relative_key = self._split_store_and_key(path)

        # Normalize key for Zarr API
        key = relative_key.rstrip("/") if relative_key else ""

        try:
            return store.listdir(key)
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
        store, key = self._split_store_and_key(path)
        root_group = zarr.group(store=store)
        return key in root_group or any(k.startswith(key.rstrip("/") + "/") for k in root_group.array_keys())

    def ensure_directory(self, directory: Union[str, Path]) -> Path:
        store, key = self._split_store_and_key(directory)
        group = zarr.group(store=store)
        group.require_group(key)
        return Path(store.dir_path) / key

    def create_symlink(self, source: Union[str, Path], link_name: Union[str, Path], overwrite: bool = False):
        store, src_key = self._split_store_and_key(source)
        store2, dst_key = self._split_store_and_key(link_name)

        if store.dir_path != store2.dir_path:
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
    
        Raises:
            FileNotFoundError: If the path doesn't exist in the group
            StorageResolutionError: For backend access issues
        """
        store, key = self._split_store_and_key(path)
        group = zarr.group(store=store)
    
        try:
            obj = group[key]
            attrs = getattr(obj, "attrs", {})
            target = attrs.get("_symlink", None)

            if "_symlink" not in attrs:
                return False
    
            # Enforce that the _symlink attr matches schema (e.g. str or list of path components)
            if not isinstance(attrs["_symlink"], str):
                raise StorageResolutionError(f"Invalid symlink format in Zarr attrs at: {path}")
    
            return True
        except KeyError:
            raise FileNotFoundError(f"No such key in Zarr group: {path}")
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