"""
Pattern resolution utilities for OpenHCS.

Provides functions for resolving image patterns from microscope data.

Doctrinal Clauses Enforced:
- Clause 42 — Ambiguity Resolution
- Clause 65 — No Fallback Logic
- Clause 87 — VFS Abstraction Purpose
- Clause 88 — No Inferred Capabilities
- Clause 92 — No Interface Fraud
- Clause 231 — Deferred Default Enforcement
"""

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from polystore.filemanager import FileManager
from polystore.base import StorageBackend

logger = logging.getLogger(__name__)


class PatternDetector(ABC):
    """Nominal interface compatible with microscope pattern detectors."""

    @abstractmethod
    def auto_detect_patterns(
        self,
        directory: Union[str, Path],
        variable_components: List[str],
        backend: str,
        group_by=None,  # Accept GroupBy enum or None
        recursive: bool = False,
        **kwargs  # Dynamic filter parameters (e.g., well_filter, site_filter)
    ) -> Dict[str, Any]:
        """Detect patterns in the given directory."""
        ...


class PathListProvider(ABC):
    """Nominal interface for objects that can list paths from a pattern."""

    @abstractmethod
    def path_list_from_pattern(
        self,
        directory: Union[str, Path],
        pattern: str,
        backend: str
    ) -> List[Union[str, Path]]:
        """List paths matching a pattern in a directory."""
        ...


class DirectoryLister(ABC):
    """Nominal interface for objects that can list files in a directory."""

    @abstractmethod
    def list_files(
        self,
        directory: Union[str, Path],
        backend: str,
        recursive: bool = False,
        pattern: Optional[str] = None,
        extensions: Optional[Set[str]] = None
    ) -> List[Union[str, Path]]:
        """List files in a directory."""
        ...

    @abstractmethod
    def is_dir(self, path: Union[str, Path], backend: str) -> bool:
        """Check if a path is a directory."""
        ...


class ManualRecursivePatternDetector(PatternDetector, ABC):
    """
    Nominal interface for detectors supporting manual recursive scanning.

    This interface defines the contract for pattern detectors that support
    manual recursive scanning of directories. It extends the PatternDetector
    interface with additional attributes for path listing and file management.
    """

    @property
    @abstractmethod
    def parser(self) -> PathListProvider:
        ...

    @property
    @abstractmethod
    def filemanager(self) -> DirectoryLister:
        ...

def _validate_filename_pattern(filename_pattern: str) -> None:
    """
    Validate a filename pattern string.

    Args:
        filename_pattern: The pattern string to validate

    Raises:
        ValueError: If the pattern is invalid
    """
    # Check for balanced braces
    if filename_pattern.count('{') != filename_pattern.count('}'):
        raise ValueError(f"Unbalanced braces in pattern: {filename_pattern}")

    # Check for valid characters
    if not re.match(r'^[a-zA-Z0-9_\-.*?{}/]+$', filename_pattern):
        raise ValueError(f"Invalid characters in pattern: {filename_pattern}")

    # Check for .tif or .tiff extension
    if not filename_pattern.endswith('.tif') and not filename_pattern.endswith('.tiff'):
        raise ValueError(f"Pattern must end with .tif or .tiff: {filename_pattern}")


def _extract_patterns_from_data(
    pattern_data: Any,
    filemanager: FileManager,
    backend: str
) -> List[str]:
    """
    Extract patterns from detector data.

    Args:
        pattern_data: Data from pattern detector (dict or list)
        filemanager: FileManager instance for path operations
        backend: Storage backend to use

    Returns:
        List of standardized pattern strings

    Raises:
        TypeError: If pattern data is of unexpected type
    """
    result: List[str] = []

    # Process dictionary of pattern lists
    if isinstance(pattern_data, dict):
        for _, patterns_list in pattern_data.items():
            if isinstance(patterns_list, list):
                result.extend(_process_pattern_list(patterns_list, filemanager, backend))

    # Process flat list of patterns
    elif isinstance(pattern_data, list):
        result.extend(_process_pattern_list(pattern_data, filemanager, backend))

    # Handle unexpected pattern format
    else:
        raise TypeError(f"Unexpected pattern format: {type(pattern_data).__name__}")

    return result


def _process_pattern_list(
    patterns: List[Any],
    filemanager: FileManager,
    backend: str
) -> List[str]:
    """
    Process a list of patterns.

    Args:
        patterns: List of pattern objects (str or Path)
        filemanager: FileManager instance for path operations
        backend: Storage backend to use

    Returns:
        List of standardized pattern strings

    Raises:
        TypeError: If a pattern is of unexpected type
    """
    result: List[str] = []

    for pattern in patterns:
        if isinstance(pattern, str):
            result.append(convert_pattern_string(pattern, filemanager, backend))
        elif isinstance(pattern, Path):
            result.append(str(pattern))
        else:
            raise TypeError(
                f"Pattern must be a string or Path, got {type(pattern).__name__}"
            )

    return result


def convert_filename_pattern(
    filename_pattern: str,
    filemanager: FileManager
) -> str:
    """
    Convert a pattern string to a standardized format.

    Args:
        filename_pattern: The pattern string to convert
        filemanager: FileManager instance for path operations
        backend: Storage backend to use (required)

    Returns:
        Standardized pattern string

    Raises:
        ValueError: If the pattern is invalid
        TypeError: If inputs are of incorrect types
    """
    # Validate input structure
    if not isinstance(filename_pattern, str):
        raise InvalidPatternError(f"Pattern must be string, got {type(filename_pattern).__name__}")
    
    if not filemanager.backend:
        raise ValueError("FileManager must be initialized with a backend")

    # Validate pattern format
    _validate_filename_pattern(filename_pattern)

    # Get concrete path from backend
    return filemanager.get_standard_path(filename_pattern)


def get_patterns_for_well(
    well: str,
    filemanager: FileManager,
    directory: Union[str, Path],
    backend: str,
    detector: PatternDetector,
    variable_components: List[str],
    recursive: bool = False
) -> List[str]:
    """
    Get flattened list of patterns for a specific well.

    Args:
        well: Well identifier (e.g., 'A01')
        filemanager: FileManager instance for path operations
        directory: Directory to search for patterns (str or Path)
        backend: Storage backend to use (required)
        detector: Object implementing PatternDetector
        variable_components: Components that vary across files
        recursive: Whether to scan subdirectories recursively

    Returns:
        List of path patterns for the well (as strings)

    Raises:
        ValueError: If pattern validation fails
        TypeError: If inputs are of incorrect types
    """
    # Validate input types
    if not isinstance(well, str):
        raise TypeError(f"well must be a string, got {type(well).__name__}")

    if not isinstance(directory, (str, Path)):
        raise TypeError(f"directory must be a string or Path, got {type(directory).__name__}")

    if not isinstance(variable_components, list):
        raise TypeError("variable_components must be a list, "
                       f"got {type(variable_components).__name__}")

    if not isinstance(filemanager, FileManager):
        raise TypeError("filemanager must be a FileManager instance, "
                       f"got {type(filemanager).__name__}")

    if not isinstance(backend, str):
        raise TypeError(f"backend must be a string, got {type(backend).__name__}")

    # Use public methods when available, but we need to use protected methods
    # for now as the public API doesn't provide the functionality we need.
    # This is a known limitation and will be addressed in a future refactoring.

    # pylint: disable=protected-access
    # Validate backend is properly initialized
    backend_instance = filemanager._get_backend(backend)
    assert isinstance(backend_instance, StorageBackend), \
        "Backend must be a StorageBackend instance"
    # pylint: enable=protected-access

    # Get patterns from detector using dynamic filter parameter
    from openhcs.constants import MULTIPROCESSING_AXIS
    axis_name = MULTIPROCESSING_AXIS.value
    filter_kwargs = {f"{axis_name}_filter": [well]}

    patterns_by_well = detector.auto_detect_patterns(
        directory,
        variable_components=variable_components,
        backend=backend,
        recursive=recursive,
        **filter_kwargs
    )

    all_patterns: List[str] = []

    # Process patterns if found for the specified well
    if patterns_by_well and well in patterns_by_well:
        well_patterns_data = patterns_by_well[well]

        # Extract patterns based on data type
        all_patterns = _extract_patterns_from_data(
            well_patterns_data, filemanager, backend
        )

    # Recursive scanning prohibited per Clause 65
    if recursive:
        raise NotImplementedError("Recursive scanning requires explicit pattern declaration")

    return all_patterns
