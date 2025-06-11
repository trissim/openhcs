"""
Microscope base implementations for openhcs.

This module provides the base implementations for microscope-specific functionality,
including filename parsing and metadata handling.
"""

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Import constants
from openhcs.constants.constants import Backend
# Import PatternDiscoveryEngine for MicroscopeHandler initialization
from openhcs.formats.pattern.pattern_discovery import PatternDiscoveryEngine
from openhcs.io.filemanager import FileManager
# Import interfaces from the base interfaces module
from openhcs.microscopes.microscope_interfaces_base import (FilenameParser,
                                                               MetadataHandler)

logger = logging.getLogger(__name__)

# Dictionary to store registered microscope handlers
MICROSCOPE_HANDLERS = {}


class MicroscopeHandler(ABC):
    """Composed class for handling microscope-specific functionality."""

    DEFAULT_MICROSCOPE = 'auto'
    _handlers_cache = None

    def __init__(self, parser: FilenameParser,
                 metadata_handler: MetadataHandler):
        """
        Initialize the microscope handler.

        Args:
            parser: Parser for microscopy filenames.
            metadata_handler: Handler for microscope metadata.
        """
        self.parser = parser
        self.metadata_handler = metadata_handler
        self.plate_folder: Optional[Path] = None # Store workspace path if needed by methods

        # Pattern discovery engine will be created on demand with the provided filemanager

    @property
    @abstractmethod
    def common_dirs(self) -> List[str]:
        """
        Canonical subdirectory names where image data may reside.
        Example: ['Images', 'TimePoint', 'Data']
        """
        pass

    def post_workspace(self, workspace_path: Union[str, Path], filemanager: FileManager, width: int = 3):
        """
        Hook called after workspace symlink creation.
        Applies normalization logic followed by consistent filename padding.

        This method requires a disk-backed path and should only be called
        from steps with requires_fs_input=True.

        Args:
            workspace_path: Path to the workspace (string or Path object)
            filemanager: FileManager instance for file operations
            width: Width for padding (default: 3)

        Returns:
            Path to the normalized image directory

        Raises:
            FileNotFoundError: If workspace_path does not exist
        """
        # Ensure workspace_path is a Path object
        if isinstance(workspace_path, str):
            workspace_path = Path(workspace_path)

        # Ensure the path exists
        if not workspace_path.exists():
            raise FileNotFoundError(f"Workspace path does not exist: {workspace_path}")

        # Apply microscope-specific preparation logic
        prepared_dir = self._prepare_workspace(workspace_path, filemanager)

        # Deterministically resolve the image directory based on common_dirs
        # Clause 245: Workspace operations are disk-only by design
        # This call is structurally hardcoded to use the "disk" backend
        entries = filemanager.list_dir(workspace_path, Backend.DISK.value)

        # Filter entries to get only directories
        subdirs = []
        for entry in entries:
            entry_path = Path(workspace_path) / entry
            if entry_path.is_dir():
                subdirs.append(entry_path)

        # Look for a directory matching any of the common_dirs patterns
        image_dir = None
        for item in subdirs:
            # FileManager should return strings, but handle Path objects too
            if isinstance(item, str):
                item_name = os.path.basename(item)
            elif isinstance(item, Path):
                item_name = item.name
            else:
                # Skip any unexpected types
                logger.warning("Unexpected directory path type: %s", type(item).__name__)
                continue

            if any(dir_name.lower() in item_name.lower() for dir_name in self.common_dirs):
                # Found a matching directory
                logger.info("Found directory matching common_dirs pattern: %s", item)
                image_dir = item
                break

        # If no matching directory found, use the prepared directory
        if image_dir is None:
            logger.info("No directory matching common_dirs found, using prepared directory: %s", prepared_dir)
            image_dir = prepared_dir

        # Ensure parser is provided
        parser = self.parser

        # Get all image files in the directory
        # Clause 245: Workspace operations are disk-only by design
        # This call is structurally hardcoded to use the "disk" backend
        image_files = filemanager.list_image_files(image_dir, Backend.DISK.value)

        # Map original filenames to reconstructed filenames
        rename_map = {}

        for file_path in image_files:
            # FileManager should return strings, but handle Path objects too
            if isinstance(file_path, str):
                original_name = os.path.basename(file_path)
            elif isinstance(file_path, Path):
                original_name = file_path.name
            else:
                # Skip any unexpected types
                logger.warning("Unexpected file path type: %s", type(file_path).__name__)
                continue

            # Parse the filename components
            metadata = parser.parse_filename(original_name)
            if not metadata:
                logger.warning("Could not parse filename: %s", original_name)
                continue

            # Validate required components
            if metadata['site'] is None:
                logger.warning("Missing 'site' component in filename: %s", original_name)
                continue

            if metadata['channel'] is None:
                logger.warning("Missing 'channel' component in filename: %s", original_name)
                continue

            # z_index is optional - default to 1 if not present
            site = metadata['site']
            channel = metadata['channel']
            z_index = metadata['z_index'] if metadata['z_index'] is not None else 1

            # Log the components for debugging
            logger.debug(
                "Parsed components for %s: site=%s, channel=%s, z_index=%s",
                original_name, site, channel, z_index
            )

            # Reconstruct the filename with proper padding
            new_name = parser.construct_filename(
                well=metadata['well'],
                site=site,
                channel=channel,
                z_index=z_index,
                extension=metadata['extension'],
                site_padding=width,
                z_padding=width
            )

            # Add to rename map if different
            if original_name != new_name:
                rename_map[original_name] = new_name

        # Perform the renaming
        for original_name, new_name in rename_map.items():
            # Create paths for the source and destination
            if isinstance(image_dir, str):
                original_path = os.path.join(image_dir, original_name)
                new_path = os.path.join(image_dir, new_name)
            else:  # Path object
                original_path = image_dir / original_name
                new_path = image_dir / new_name

            try:
                # Ensure the parent directory exists
                # Clause 245: Workspace operations are disk-only by design
                # This call is structurally hardcoded to use the "disk" backend
                parent_dir = os.path.dirname(new_path) if isinstance(new_path, str) else new_path.parent
                filemanager.ensure_directory(parent_dir, Backend.DISK.value)

                # Rename the file using move operation
                # Clause 245: Workspace operations are disk-only by design
                # This call is structurally hardcoded to use the "disk" backend
                filemanager.move(original_path, new_path, Backend.DISK.value)
                logger.debug("Renamed %s to %s", original_path, new_path)
            except (OSError, FileNotFoundError) as e:
                logger.error("Filesystem error renaming %s to %s: %s", original_path, new_path, e)
            except TypeError as e:
                logger.error("Type error renaming %s to %s: %s", original_path, new_path, e)
            except Exception as e:
                logger.error("Unexpected error renaming %s to %s: %s", original_path, new_path, e)

        return image_dir

    @abstractmethod
    def _prepare_workspace(self, workspace_path: Path, filemanager: FileManager):
        """
        Microscope-specific preparation logic before image directory resolution.

        This method performs any necessary preprocessing on the workspace but does NOT
        determine the final image directory. It may return a suggested directory, but
        the final image directory will be determined by post_workspace() based on
        common_dirs matching.

        Override in subclasses. Default implementation just returns the workspace path.

        This method requires a disk-backed path and should only be called
        from steps with requires_fs_input=True.

        Args:
            workspace_path: Path to the symlinked workspace
            filemanager: FileManager instance for file operations

        Returns:
            Path: A suggested directory for further processing (not necessarily the final image directory)

        Raises:
            FileNotFoundError: If workspace_path does not exist
        """
        return workspace_path


    # Delegate methods to parser
    def parse_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """Delegate to parser."""
        return self.parser.parse_filename(filename)

    def construct_filename(self, well: str, site: Optional[Union[int, str]] = None,
                          channel: Optional[int] = None,
                          z_index: Optional[Union[int, str]] = None,
                          extension: str = '.tif',
                          site_padding: int = 3, z_padding: int = 3) -> str:
        """Delegate to parser."""
        return self.parser.construct_filename(
            well, site, channel, z_index, extension, site_padding, z_padding
        )

    def auto_detect_patterns(self, folder_path: Union[str, Path], filemanager: FileManager, backend: str,
                           well_filter=None, extensions=None, group_by='channel', variable_components=None):
        """
        Delegate to pattern engine.

        Args:
            folder_path: Path to the folder (string or Path object)
            filemanager: FileManager instance for file operations
            backend: Backend to use for file operations (required)
            well_filter: Optional list of wells to include
            extensions: Optional list of file extensions to include
            group_by: Component to group patterns by (e.g., 'channel', 'z_index', 'well')
            variable_components: List of components to make variable (e.g., ['site', 'z_index'])

        Returns:
            Dict[str, Any]: Dictionary mapping wells to patterns
        """
        # Ensure folder_path is a valid path
        if isinstance(folder_path, str):
            folder_path = Path(folder_path)
        elif not isinstance(folder_path, Path):
            raise TypeError(f"Expected string or Path object, got {type(folder_path).__name__}")

        # Ensure the path exists using FileManager abstraction
        if not filemanager.exists(str(folder_path), backend):
            raise ValueError(f"Folder path does not exist: {folder_path}")

        # Create pattern engine on demand with the provided filemanager
        pattern_engine = PatternDiscoveryEngine(self.parser, filemanager)

        # Get patterns from the pattern engine
        patterns_by_well = pattern_engine.auto_detect_patterns(
            folder_path,
            well_filter=well_filter,
            extensions=extensions,
            group_by=group_by,
            variable_components=variable_components,
            backend=backend
        )

        # 🔒 Clause 74 — Runtime Behavior Variation
        # Ensure we always return a dictionary, not a generator
        if not isinstance(patterns_by_well, dict):
            # Convert to dictionary if it's not already one
            return dict(patterns_by_well)

        return patterns_by_well

    def path_list_from_pattern(self, directory: Union[str, Path], pattern, filemanager: FileManager, backend: str, variable_components: Optional[List[str]] = None):
        """
        Delegate to pattern engine.

        Args:
            directory: Directory to search (string or Path object)
            pattern: Pattern to match (str for literal filenames)
            filemanager: FileManager instance for file operations
            backend: Backend to use for file operations (required)
            variable_components: List of components that can vary (will be ignored during matching)

        Returns:
            List of matching filenames

        Raises:
            TypeError: If a string with braces is passed (pattern paths are no longer supported)
            ValueError: If directory does not exist
        """
        # Ensure directory is a valid path using FileManager abstraction
        if isinstance(directory, str):
            directory_path = Path(directory)
            if not filemanager.exists(str(directory_path), backend):
                raise ValueError(f"Directory does not exist: {directory}")
        elif isinstance(directory, Path):
            directory_path = directory
            if not filemanager.exists(str(directory_path), backend):
                raise ValueError(f"Directory does not exist: {directory}")
        else:
            raise TypeError(f"Expected string or Path object, got {type(directory).__name__}")

        # Allow string patterns with braces - they are used for template matching
        # The pattern engine will handle template expansion to find matching files

        # Create pattern engine on demand with the provided filemanager
        pattern_engine = PatternDiscoveryEngine(self.parser, filemanager)

        # Delegate to the pattern engine
        return pattern_engine.path_list_from_pattern(directory_path, pattern, backend=backend, variable_components=variable_components)

    # Delegate metadata handling methods to metadata_handler with context

    def find_metadata_file(self, plate_path: Union[str, Path]) -> Optional[Path]:
        """Delegate to metadata handler."""
        return self.metadata_handler.find_metadata_file(plate_path)

    def get_grid_dimensions(self, plate_path: Union[str, Path]) -> Tuple[int, int]:
        """Delegate to metadata handler."""
        return self.metadata_handler.get_grid_dimensions(plate_path)

    def get_pixel_size(self, plate_path: Union[str, Path]) -> float:
        """Delegate to metadata handler."""
        return self.metadata_handler.get_pixel_size(plate_path)


# Import handler classes at module level with explicit mapping
# No aliases or legacy compatibility layers (Clause 77)

# Factory function
def create_microscope_handler(microscope_type: str = 'auto',
                              plate_folder: [Union[str, Path]] = None,
                              filemanager: FileManager = None,
                              pattern_format: Optional[str] = None) -> MicroscopeHandler:
    """
    Factory function to create a microscope handler.

    This function enforces explicit dependency injection by requiring a FileManager
    instance to be provided. This ensures that all components requiring file operations
    receive their dependencies explicitly, eliminating runtime fallbacks and enforcing
    declarative configuration.

    Args:
        microscope_type: 'auto', 'imagexpress', 'opera_phenix'.
        plate_folder: Required for 'auto' detection.
        filemanager: FileManager instance. Must be provided.
        pattern_format: Name of the pattern format to use.

    Returns:
        An initialized MicroscopeHandler instance.

    Raises:
        ValueError: If filemanager is None or if microscope_type cannot be determined.
    """
    if filemanager is None:
        raise ValueError(
            "FileManager must be provided to create_microscope_handler. "
            "Default fallback has been removed."
        )

    logger.info("Using provided FileManager for microscope handler.")

    # Auto-detect microscope type if needed
    if microscope_type == 'auto':
        if not plate_folder:
            raise ValueError("plate_folder is required for auto-detection")

        plate_folder = Path(plate_folder) if isinstance(plate_folder, str) else plate_folder
        microscope_type = _auto_detect_microscope_type(plate_folder, filemanager)
        logger.info("Auto-detected microscope type: %s", microscope_type)

    # Get the appropriate handler class from the constant mapping
    # No dynamic imports or fallbacks (Clause 77: Rot Intolerance)
    handler_class = MICROSCOPE_HANDLERS.get(microscope_type.lower())
    if not handler_class:
        raise ValueError(
            f"Unsupported microscope type: {microscope_type}. "
            f"Supported types: {list(MICROSCOPE_HANDLERS.keys())}"
        )

    # Create and configure the handler
    logger.info(f"Creating {handler_class.__name__}")

    # Create the handler with the parser and metadata handler
    # The filemanager will be passed to methods that need it
    handler = handler_class(filemanager, pattern_format=pattern_format)

    return handler


def _auto_detect_microscope_type(plate_folder: Path, filemanager: FileManager) -> str:
    """
    Auto-detect microscope type based on files in the plate folder.

    Args:
        plate_folder: Path to the plate folder
        filemanager: FileManager instance

    Returns:
        Detected microscope type as string

    Raises:
        ValueError: If microscope type cannot be determined
    """
    try:
        # Check for Opera Phenix (Index.xml)
        # Clause 245: Workspace operations are disk-only by design
        # This call is structurally hardcoded to use the "disk" backend
        if filemanager.find_file_recursive(path=plate_folder, filename="Index.xml",
                                           backend=Backend.DISK.value):
            logger.info("Auto-detected Opera Phenix microscope type.")
            # Use consistent key from MICROSCOPE_HANDLERS (Clause 77: Rot Intolerance)
            return 'opera_phenix'

        # Check for ImageXpress (.htd files)
        # Clause 245: Workspace operations are disk-only by design
        # This call is structurally hardcoded to use the "disk" backend
        if filemanager.list_files(
            path=plate_folder, extensions={'.htd','.HTD'}, recursive=True,
            backend=Backend.DISK.value
        ):
            logger.info("Auto-detected ImageXpress microscope type.")
            # Use consistent key from MICROSCOPE_HANDLERS (Clause 77: Rot Intolerance)
            return 'imagexpress'

        # No known microscope type detected - fail deterministically (Clause 12: Smell Intolerance)
        supported_types = list(MICROSCOPE_HANDLERS.keys())
        msg = (f"Could not auto-detect microscope type in {plate_folder}. "
               f"Supported types: {supported_types}")
        logger.error(msg)
        raise ValueError(msg)

    except Exception as e:
        # Wrap exception with clear context (Clause 12: Smell Intolerance)
        raise ValueError(f"Error during microscope type auto-detection in {plate_folder}: {e}") from e
