"""
Atomic metadata writer for OpenHCS with concurrency safety.

Provides specialized atomic operations for OpenHCS metadata files with proper
locking and merging to prevent race conditions in multiprocessing environments.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from .atomic import atomic_update_json, FileLockError, LOCK_CONFIG

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetadataConfig:
    """Configuration constants for metadata operations."""
    METADATA_FILENAME: str = "openhcs_metadata.json"
    SUBDIRECTORIES_KEY: str = "subdirectories"
    AVAILABLE_BACKENDS_KEY: str = "available_backends"
    DEFAULT_TIMEOUT: float = LOCK_CONFIG.DEFAULT_TIMEOUT


METADATA_CONFIG = MetadataConfig()


@dataclass(frozen=True)
class MetadataUpdateRequest:
    """Parameter object for metadata update operations."""
    metadata_path: Union[str, Path]
    sub_dir: str
    metadata: Dict[str, Any]
    available_backends: Optional[Dict[str, bool]] = None


class MetadataWriteError(Exception):
    """Raised when metadata write operations fail."""
    pass


class AtomicMetadataWriter:
    """Atomic metadata writer with file locking for concurrent safety."""

    def __init__(self, timeout: float = METADATA_CONFIG.DEFAULT_TIMEOUT):
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

    def _execute_update(self, metadata_path: Union[str, Path], update_func: Callable, default_data: Optional[Dict] = None) -> None:
        """Execute atomic update with error handling."""
        try:
            atomic_update_json(metadata_path, update_func, self.timeout, default_data)
        except FileLockError as e:
            raise MetadataWriteError(f"Failed to update metadata: {e}") from e

    def _ensure_subdirectories_structure(self, data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Ensure metadata has proper subdirectories structure."""
        data = data or {}
        data.setdefault(METADATA_CONFIG.SUBDIRECTORIES_KEY, {})
        return data

    def _create_subdirectory_update(self, sub_dir: str, metadata: Dict[str, Any]) -> Callable:
        """Create update function for subdirectory operations."""
        def update_func(data):
            data = self._ensure_subdirectories_structure(data)
            data[METADATA_CONFIG.SUBDIRECTORIES_KEY][sub_dir] = metadata
            return data
        return update_func
    
    def update_subdirectory_metadata(self, metadata_path: Union[str, Path], sub_dir: str, metadata: Dict[str, Any]) -> None:
        """Atomically update metadata for a specific subdirectory."""
        update_func = self._create_subdirectory_update(sub_dir, metadata)
        self._execute_update(metadata_path, update_func, {METADATA_CONFIG.SUBDIRECTORIES_KEY: {}})
        self.logger.debug(f"Updated subdirectory '{sub_dir}' in {metadata_path}")
    
    def update_available_backends(self, metadata_path: Union[str, Path], available_backends: Dict[str, bool]) -> None:
        """Atomically update available backends in metadata."""
        def update_func(data):
            if data is None:
                raise MetadataWriteError("Cannot update backends: metadata file does not exist")
            data[METADATA_CONFIG.AVAILABLE_BACKENDS_KEY] = available_backends
            return data

        self._execute_update(metadata_path, update_func)
        self.logger.debug(f"Updated available backends in {metadata_path}")
    
    def merge_subdirectory_metadata(self, metadata_path: Union[str, Path], subdirectory_updates: Dict[str, Dict[str, Any]]) -> None:
        """Atomically merge multiple subdirectory metadata updates."""
        def update_func(data):
            data = self._ensure_subdirectories_structure(data)
            data[METADATA_CONFIG.SUBDIRECTORIES_KEY].update(subdirectory_updates)
            return data

        self._execute_update(metadata_path, update_func, {METADATA_CONFIG.SUBDIRECTORIES_KEY: {}})
        self.logger.debug(f"Merged {len(subdirectory_updates)} subdirectories in {metadata_path}")
    
    def create_or_update_metadata(self, request: MetadataUpdateRequest) -> None:
        """Atomically create or update metadata file with subdirectory and backend info."""
        update_func = self._create_subdirectory_update(request.sub_dir, request.metadata)

        if request.available_backends is not None:
            # Compose with backend update
            original_func = update_func
            def update_func(data):
                data = original_func(data)
                data[METADATA_CONFIG.AVAILABLE_BACKENDS_KEY] = request.available_backends
                return data

        self._execute_update(request.metadata_path, update_func, {METADATA_CONFIG.SUBDIRECTORIES_KEY: {}})
        self.logger.debug(f"Created/updated metadata for '{request.sub_dir}' in {request.metadata_path}")


def get_metadata_path(plate_root: Union[str, Path]) -> Path:
    """
    Get the standard metadata file path for a plate root directory.
    
    Args:
        plate_root: Path to the plate root directory
        
    Returns:
        Path to the metadata file
    """
    return Path(plate_root) / METADATA_CONFIG.METADATA_FILENAME
