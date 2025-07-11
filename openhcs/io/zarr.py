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

try:
    from ome_zarr.writer import write_image, write_plate_metadata, write_well_metadata
    from ome_zarr.io import parse_url
    OME_ZARR_AVAILABLE = True
except ImportError:
    OME_ZARR_AVAILABLE = False

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

    def __init__(self, zarr_config: Optional['ZarrConfig'] = None):
        """
        Initialize Zarr backend with ZarrConfig.

        Args:
            zarr_config: ZarrConfig dataclass with all zarr settings (uses defaults if None)
        """
        # Import here to avoid circular imports
        from openhcs.core.config import ZarrConfig

        if zarr_config is None:
            zarr_config = ZarrConfig()

        self.config = zarr_config

        # Convenience attributes
        self.store_name = zarr_config.store_name
        self.compression_level = zarr_config.compression_level

        # Create actual compressor from config
        self.compressor = self.config.compressor.create_compressor(
            self.config.compression_level,
            self.config.shuffle
        )

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

        # In zarr v2, we just use the path string directly
        return str(store_path), relative_key

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

        # Group files by well based on OME-ZARR structure
        well_to_files = {}
        well_to_indices = {}

        # Search OME-ZARR structure for requested files
        for row_name in root.group_keys():
            if len(row_name) == 1 and row_name.isalpha():  # Row directory (A, B, etc.)
                row_group = root[row_name]
                for col_name in row_group.group_keys():
                    if col_name.isdigit():  # Column directory (01, 02, etc.)
                        well_group = row_group[col_name]
                        well_name = f"{row_name}{col_name}"

                        # Check if this well has our filename mapping in the field array
                        if "0" in well_group.group_keys():
                            field_group = well_group["0"]
                            if "0" in field_group.array_keys():
                                field_array = field_group["0"]
                                if "openhcs_filename_map" in field_array.attrs:
                                    filename_map = dict(field_array.attrs["openhcs_filename_map"])

                                    # Check which requested files are in this well
                                    for i, path in enumerate(file_paths):
                                        filename = str(path)  # Use full path for matching
                                        if filename in filename_map:
                                            if well_name not in well_to_files:
                                                well_to_files[well_name] = []
                                                well_to_indices[well_name] = []
                                            well_to_files[well_name].append(i)  # Original position in file_paths
                                            well_to_indices[well_name].append(filename_map[filename])  # 5D coordinates (field, channel, z)

        # Load data from each well using single well chunk
        results = [None] * len(file_paths)  # Pre-allocate results array

        for well_name, file_positions in well_to_files.items():
            row, col = well_name[0], well_name[1:]
            well_group = root[row][col]
            well_indices = well_to_indices[well_name]

            # Load entire well field array in single operation (well chunking)
            field_group = well_group["0"]
            field_array = field_group["0"]
            all_well_data = field_array[:]  # Single I/O operation for entire well

            # Extract requested 2D slices using 5D coordinates
            for file_pos, coords_5d in zip(file_positions, well_indices):
                field_idx, channel_idx, z_idx = coords_5d
                # Extract 2D slice: (field, channel, z, y, x) -> (y, x)
                results[file_pos] = all_well_data[field_idx, channel_idx, z_idx, :, :]  # 2D slice

        logger.debug(f"Loaded {len(file_paths)} images from zarr store at {store_path} from {len(well_to_files)} wells")
        return results

    def save_batch(self, data_list: List[Any], output_paths: List[Union[str, Path]], **kwargs) -> None:
        """Save multiple images using ome-zarr-py for proper OME-ZARR compliance with multi-dimensional support.

        Args:
            data_list: List of image data to save
            output_paths: List of output file paths
            **kwargs: Must include chunk_name, n_channels, n_z, n_fields, row, col
        """

        # Extract required parameters from kwargs
        chunk_name = kwargs.get('chunk_name')
        n_channels = kwargs.get('n_channels')
        n_z = kwargs.get('n_z')
        n_fields = kwargs.get('n_fields')
        row = kwargs.get('row')
        col = kwargs.get('col')

        # Validate required parameters
        if chunk_name is None:
            raise ValueError("chunk_name must be provided")
        if n_channels is None:
            raise ValueError("n_channels must be provided")
        if n_z is None:
            raise ValueError("n_z must be provided")
        if n_fields is None:
            raise ValueError("n_fields must be provided")
        if row is None:
            raise ValueError("row must be provided")
        if col is None:
            raise ValueError("col must be provided")

        if not data_list:
            logger.warning(f"Empty data list for chunk {chunk_name}")
            return

        if not OME_ZARR_AVAILABLE:
            raise ImportError("ome-zarr package is required. Install with: pip install ome-zarr")

        base_dir = Path(output_paths[0]).parent
        store_path = base_dir / self.store_name

        logger.debug(f"Saving batch for chunk {chunk_name} with {len(data_list)} images to row={row}, col={col}")

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

        # Ensure zarr store exists before opening (thread-safe)
        if not store_path.exists():
            # Ensure parent directory exists before creating lock file
            store_path.parent.mkdir(parents=True, exist_ok=True)
            # Use file locking to prevent race condition during store creation
            lock_path = store_path.with_suffix('.init.lock')
            try:
                with open(lock_path, 'x') as lock_file:
                    if not store_path.exists():
                        # Create initial zarr group structure
                        store = parse_url(str(store_path), mode="w").store
                        root = zarr.group(store=store)
                        root.attrs["ome"] = {"version": "0.4"}
                        logger.debug(f"Created initial zarr store at {store_path}")
            except FileExistsError:
                # Another process is creating the store, wait for it
                import time
                while not store_path.exists():
                    time.sleep(0.01)
                logger.debug(f"Waited for zarr store creation at {store_path}")
            finally:
                if lock_path.exists():
                    lock_path.unlink()

        # Initialize ome-zarr store (allow overwriting)
        # Use 'a' mode to allow appending/overwriting individual arrays
        store = parse_url(str(store_path), mode="a").store
        root = zarr.group(store=store)

        # Write plate metadata with locking to prevent concurrent corruption (if enabled)
        should_write_plate_metadata = kwargs.get('write_plate_metadata', self.config.write_plate_metadata)
        if should_write_plate_metadata:
            self._ensure_plate_metadata_with_lock(root, row, col, store_path)

        # Create HCS-compliant structure: plate/row/col/field/resolution
        # Create row group if it doesn't exist
        if row not in root:
            row_group = root.create_group(row)
        else:
            row_group = root[row]

        # Create well group (remove existing if present to allow overwrite)
        if col in row_group:
            del row_group[col]
        well_group = row_group.create_group(col)

        # Add HCS well metadata
        well_metadata = {
            "images": [
                {
                    "path": "0",  # Single image containing all fields
                    "acquisition": 0
                }
            ],
            "version": "0.5"
        }
        well_group.attrs["ome"] = {"version": "0.5", "well": well_metadata}

        # Create field group (single field "0" containing all field data)
        field_group = well_group.require_group("0")

        # Always use full 5D structure: (fields, channels, z, y, x)
        # Define OME-NGFF compliant axes
        axes = [
            {'name': 'field', 'type': 'field'},  # Custom field type - allowed before space
            {'name': 'c', 'type': 'channel'},
            {'name': 'z', 'type': 'space'},
            {'name': 'y', 'type': 'space'},
            {'name': 'x', 'type': 'space'}
        ]

        # Get image dimensions
        sample_image = cpu_data_list[0]
        height, width = sample_image.shape[-2:]

        # Always reshape to full 5D: (n_fields, n_channels, n_z, y, x)
        target_shape = [n_fields, n_channels, n_z, height, width]

        # Stack and reshape data
        stacked_data = np.stack(cpu_data_list, axis=0)

        # Calculate total expected images for validation
        total_expected = n_fields * n_channels * n_z
        if len(data_list) != total_expected:
            logger.warning(f"Data count mismatch: got {len(data_list)}, expected {total_expected} "
                         f"(fields={n_fields}, channels={n_channels}, z={n_z})")

        # Always reshape to 5D structure
        reshaped_data = stacked_data.reshape(target_shape)

        logger.info(f"Zarr save_batch: {len(data_list)} images → {stacked_data.shape} → {reshaped_data.shape}")
        axes_names = [ax['name'] for ax in axes]
        logger.info(f"Dimensions: fields={n_fields}, channels={n_channels}, z={n_z}, axes={''.join(axes_names)}")

        # Create field group (single field "0" containing all field data)
        if "0" in well_group:
            field_group = well_group["0"]
        else:
            field_group = well_group.create_group("0")

        # Write OME-ZARR well metadata with single field (well-chunked approach)
        write_well_metadata(well_group, ['0'])

        # Use single chunk approach for optimal performance
        storage_options = {
            "chunks": reshaped_data.shape,  # Single chunk for entire well
            "compressor": self._get_compressor()
        }

        # Write as single well-chunked array with proper multi-dimensional axes
        write_image(
            image=reshaped_data,
            group=field_group,
            axes=axes,
            storage_options=storage_options,
            scaler=None,  # Single scale only for performance
            compute=True
        )

        # Axes are already correctly set by write_image function

        # Store filename mapping with 5D coordinates (field, channel, z, y, x)
        # Convert flat index to 5D coordinates for proper zarr slicing
        filename_map = {}
        for i, path in enumerate(output_paths):
            # Calculate 5D coordinates from flat index
            field_idx = i // (n_channels * n_z)
            remaining = i % (n_channels * n_z)
            channel_idx = remaining // n_z
            z_idx = remaining % n_z

            # Store as tuple (field, channel, z) - y,x are full slices
            filename_map[str(path)] = (field_idx, channel_idx, z_idx)

        field_array = field_group['0']
        field_array.attrs["openhcs_filename_map"] = filename_map
        field_array.attrs["openhcs_output_paths"] = [str(path) for path in output_paths]
        field_array.attrs["openhcs_dimensions"] = {
            "n_fields": n_fields,
            "n_channels": n_channels,
            "n_z": n_z
        }

        logger.debug(f"Successfully saved batch for chunk {chunk_name}")

        # Aggressive memory cleanup
        del cpu_data_list
        import gc
        gc.collect()

    def _ensure_plate_metadata_with_lock(self, root: zarr.Group, row: str, col: str, store_path: Path) -> None:
        """Ensure plate-level metadata includes ALL existing wells with file locking."""
        import fcntl

        lock_path = store_path.with_suffix('.metadata.lock')

        try:
            with open(lock_path, 'w') as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                self._ensure_plate_metadata(root, row, col)
        except Exception as e:
            logger.error(f"Failed to update plate metadata with lock: {e}")
            raise
        finally:
            if lock_path.exists():
                lock_path.unlink()

    def _ensure_plate_metadata(self, root: zarr.Group, row: str, col: str) -> None:
        """Ensure plate-level metadata includes ALL existing wells in the store."""

        # Scan the store for all existing wells
        all_rows = set()
        all_cols = set()
        all_wells = []

        for row_name in root.group_keys():
            if isinstance(root[row_name], zarr.Group):  # Ensure it's a row group
                row_group = root[row_name]
                all_rows.add(row_name)

                for col_name in row_group.group_keys():
                    if isinstance(row_group[col_name], zarr.Group):  # Ensure it's a well group
                        all_cols.add(col_name)
                        well_path = f"{row_name}/{col_name}"
                        all_wells.append(well_path)

        # Include the current well being added (might not exist yet)
        all_rows.add(row)
        all_cols.add(col)
        current_well_path = f"{row}/{col}"
        if current_well_path not in all_wells:
            all_wells.append(current_well_path)

        # Sort for consistent ordering
        sorted_rows = sorted(all_rows)
        sorted_cols = sorted(all_cols)
        sorted_wells = sorted(all_wells)

        # Build wells metadata with proper indices
        wells_metadata = []
        for well_path in sorted_wells:
            well_row, well_col = well_path.split("/")
            row_index = sorted_rows.index(well_row)
            col_index = sorted_cols.index(well_col)
            wells_metadata.append({
                "path": well_path,
                "rowIndex": row_index,
                "columnIndex": col_index
            })

        # Add acquisition metadata for HCS compliance
        acquisitions = [
            {
                "id": 0,
                "name": "default_acquisition",
                "maximumfieldcount": 1  # Single field containing all field data
            }
        ]

        # Write complete HCS plate metadata
        write_plate_metadata(
            root,
            sorted_rows,
            sorted_cols,
            wells_metadata,
            acquisitions=acquisitions,
            field_count=1,
            name="OpenHCS_Plate"
        )






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
            # Open zarr group and traverse OME-ZARR structure
            group = zarr.open_group(store=store)

            # Check if this is OME-ZARR structure (has plate metadata)
            if "plate" in group.attrs:
                # OME-ZARR structure: traverse A/01/ wells
                for row_name in group.group_keys():
                    if len(row_name) == 1 and row_name.isalpha():  # Row directory (A, B, etc.)
                        row_group = group[row_name]
                        for col_name in row_group.group_keys():
                            if col_name.isdigit():  # Column directory (01, 02, etc.)
                                well_group = row_group[col_name]

                                # Get filenames from field array metadata
                                if "0" in well_group.group_keys():
                                    field_group = well_group["0"]
                                    if "0" in field_group.array_keys():
                                        field_array = field_group["0"]
                                        if "openhcs_output_paths" in field_array.attrs:
                                            output_paths = field_array.attrs["openhcs_output_paths"]
                                            for filename in output_paths:
                                                filename_only = Path(filename).name
                                                if _matches_filters(filename_only):
                                                    result.append(Path(filename))
            else:
                # Legacy flat structure: get array keys directly
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

                    except Exception as e:
                        # Skip arrays that can't be accessed
                        continue

        except Exception as e:
            raise StorageResolutionError(f"Failed to list zarr arrays: {e}") from e

        return result

    def list_dir(self, path: Union[str, Path]) -> List[str]:
        store, relative_key = self._split_store_and_key(path)

        # Normalize key for Zarr API
        key = relative_key.rstrip("/")

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

    def _auto_chunks(self, data: Any, chunk_divisor: int = 1) -> Tuple[int, ...]:
        shape = data.shape

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