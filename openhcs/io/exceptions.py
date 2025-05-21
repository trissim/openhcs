"""Doctrinal Exceptions for Structural Enforcement.

This module defines exceptions that enforce structural integrity
and prevent runtime flexibility across the VFS boundary.
"""

class StorageResolutionError(ValueError):
    """Raised when a storage key cannot be resolved to a valid VirtualPath."""
    pass

class ImageLoadError(RuntimeError):
    """Raised when image loading fails through the VFS boundary."""
    pass

class ImageSaveError(RuntimeError):
    """Raised when image saving fails through the VFS boundary."""
    pass

class StorageWriteError(RuntimeError):
    """Raised when writing to storage fails."""
    pass

class MetadataNotFoundError(ValueError):
    """Raised when required metadata files cannot be found."""
    pass

class PathMismatchError(ValueError):
    """Raised when a path scheme doesn't match the expected scheme for a backend."""
    pass

class VFSTypeError(TypeError):
    """Raised when a type error occurs in the VFS boundary."""
    pass