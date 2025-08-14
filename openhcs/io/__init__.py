"""
Storage backends package for openhcs.

This package contains the storage backend implementations for openhcs.
"""

from .atomic import file_lock, atomic_write_json, atomic_update_json, FileLockError, FileLockTimeoutError
from .base import StorageBackend, storage_registry, reset_memory_backend
from .disk import DiskStorageBackend
from .filemanager import FileManager
from .memory import MemoryStorageBackend
from .metadata_writer import AtomicMetadataWriter, MetadataWriteError, MetadataUpdateRequest, get_metadata_path
from .metadata_migration import detect_legacy_format, migrate_legacy_metadata, migrate_plate_metadata
from .zarr import ZarrStorageBackend

__all__ = [
    'StorageBackend',
    'storage_registry',
    'reset_memory_backend',
    'DiskStorageBackend',
    'MemoryStorageBackend',
    'ZarrStorageBackend',
    'FileManager',
    'file_lock',
    'atomic_write_json',
    'atomic_update_json',
    'FileLockError',
    'FileLockTimeoutError',
    'AtomicMetadataWriter',
    'MetadataWriteError',
    'MetadataUpdateRequest',
    'get_metadata_path',
    'detect_legacy_format',
    'migrate_legacy_metadata',
    'migrate_plate_metadata'
]
