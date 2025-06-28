"""
Microscope base implementations for openhcs.

This module provides the base implementations for microscope-specific functionality,
including filename parsing and metadata handling.
"""

import logging
import os
from abc import ABC, ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Type

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

# Dictionary to store registered metadata handlers for auto-detection
METADATA_HANDLERS = {}


class MicroscopeHandlerMeta(ABCMeta):
    """Metaclass for automatic registration of microscope handlers."""

    def __new__(cls, name, bases, attrs):
        new_class = super().__new__(cls, name, bases, attrs)

        # Only register concrete handler classes (not the abstract base class)
        if bases and not getattr(new_class, '__abstractmethods__', None):
            # Use explicit microscope type if provided, otherwise extract from class name
            microscope_type = getattr(new_class, '_microscope_type', None)
            if not microscope_type:
                if name.endswith('Handler'):
                    microscope_type = name[:-7].lower()  # ImageXpressHandler -> imagexpress
                else:
                    microscope_type = name.lower()

            # Auto-register in MICROSCOPE_HANDLERS
            MICROSCOPE_HANDLERS[microscope_type] = new_class

            # Store the microscope type as the standard class attribute
            new_class._microscope_type = microscope_type

            # Auto-register metadata handler if the class has one
            metadata_handler_class = getattr(new_class, '_metadata_handler_class', None)
            if metadata_handler_class:
                METADATA_HANDLERS[microscope_type] = metadata_handler_class

            logger.debug(f"Auto-registered {name} as '{microscope_type}'")

        return new_class


def register_metadata_handler(handler_class, metadata_handler_class):
    """
    Register a metadata handler for a microscope handler class.

    This function is called when _metadata_handler_class is set after class definition.
    """
    microscope_type = getattr(handler_class, '_microscope_type', None)
    if microscope_type:
        METADATA_HANDLERS[microscope_type] = metadata_handler_class
        logger.debug(f"Registered metadata handler {metadata_handler_class.__name__} for '{microscope_type}'")
    else:
        logger.warning(f"Could not register metadata handler for {handler_class.__name__} - no microscope type found")




class MicroscopeHandler(ABC, metaclass=MicroscopeHandlerMeta):
    """Composed class for handling microscope-specific functionality."""

    DEFAULT_MICROSCOPE = 'auto'
    _handlers_cache = None

    # Optional class attribute for explicit metadata handler registration
    _metadata_handler_class: Optional[Type[MetadataHandler]] = None

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

    @property
    @abstractmethod
    def microscope_type(self) -> str:
        """Microscope type identifier (for interface enforcement only)."""
        pass

    @property
    @abstractmethod
    def metadata_handler_class(self) -> Type[MetadataHandler]:
        """Metadata handler class (for interface enforcement only)."""
        pass

    @property
    @abstractmethod
    def compatible_backends(self) -> List[Backend]:
        """
        List of storage backends this microscope handler is compatible with, in priority order.

        Must be explicitly declared by each handler implementation.
        The first backend in the list is the preferred/highest priority backend.
        The compiler will use the first backend for initial step materialization.

        Common patterns:
        - [Backend.DISK] - Basic handlers (ImageXpress, Opera Phenix)
        - [Backend.ZARR, Backend.DISK] - Advanced handlers (OpenHCS: zarr preferred, disk fallback)

        Returns:
            List of Backend enum values this handler can work with, in priority order
        """
        pass

    @abstractmethod
    def get_available_backends(self, plate_path: Union[str, Path]) -> List[Backend]:
        """
        Get available storage backends for this specific plate.

        Args:
            plate_path: Path to the plate folder

        Returns:
            List of Backend enums that are available for this plate.
            For most handlers, this will be based on compatible_backends.
            For OpenHCS, this reads from metadata.
        """
        pass

    def initialize_workspace(self, plate_path: Path, workspace_path: Optional[Path], filemanager: FileManager) -> Path:
        """
        Default workspace initialization: create workspace in plate folder and mirror with symlinks.

        Most microscope handlers need workspace mirroring. Override this method only if different behavior is needed.

        Args:
            plate_path: Path to the original plate directory
            workspace_path: Optional workspace path (creates default in plate folder if None)
            filemanager: FileManager instance for file operations

        Returns:
            Path to the actual directory containing images to process
        """
        from openhcs.constants.constants import Backend

        # Create workspace path in plate folder if not provided
        if workspace_path is None:
            workspace_path = plate_path / "workspace"

        # Check if workspace already exists - skip mirroring if it does
        if workspace_path.exists():
            logger.info(f"📁 EXISTING WORKSPACE FOUND: {workspace_path} - skipping mirror operation")
            num_links = 0  # No new links created
        else:
            # Ensure workspace directory exists
            filemanager.ensure_directory(str(workspace_path), Backend.DISK.value)

            # Mirror plate directory with symlinks
            logger.info(f"Mirroring plate directory {plate_path} to workspace {workspace_path}...")
            try:
                num_links = filemanager.mirror_directory_with_symlinks(
                    source_dir=str(plate_path),
                    target_dir=str(workspace_path),
                    backend=Backend.DISK.value,
                    recursive=True,
                    overwrite_symlinks_only=True,
                )
                logger.info(f"Created {num_links} symlinks in workspace.")
            except Exception as mirror_error:
                # If mirroring fails, clean up and try again with fail-loud
                logger.warning(f"⚠️ MIRROR FAILED: {mirror_error}. Cleaning workspace and retrying...")
                try:
                    import shutil
                    shutil.rmtree(workspace_path)
                    logger.info(f"🧹 Cleaned up failed workspace: {workspace_path}")

                    # Recreate directory and try mirroring again
                    filemanager.ensure_directory(str(workspace_path), Backend.DISK.value)
                    num_links = filemanager.mirror_directory_with_symlinks(
                        source_dir=str(plate_path),
                        target_dir=str(workspace_path),
                        backend=Backend.DISK.value,
                        recursive=True,
                        overwrite_symlinks_only=True,
                    )
                    logger.info(f"✅ RETRY SUCCESS: Created {num_links} symlinks in workspace.")
                except Exception as retry_error:
                    # Fail loud on second attempt
                    error_msg = f"Failed to mirror plate directory to workspace after cleanup: {retry_error}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from retry_error

        # Set plate_folder for this handler
        self.plate_folder = workspace_path

        # Prepare workspace and return final image directory
        return self.post_workspace(workspace_path, filemanager)

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
                # Use replace_symlinks=True to allow overwriting existing symlinks
                filemanager.move(original_path, new_path, Backend.DISK.value, replace_symlinks=True)
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
                              plate_folder: Optional[Union[str, Path]] = None,
                              filemanager: Optional[FileManager] = None,
                              pattern_format: Optional[str] = None,
                              allowed_auto_types: Optional[List[str]] = None) -> MicroscopeHandler:
    """
    Factory function to create a microscope handler.

    This function enforces explicit dependency injection by requiring a FileManager
    instance to be provided. This ensures that all components requiring file operations
    receive their dependencies explicitly, eliminating runtime fallbacks and enforcing
    declarative configuration.

    Args:
        microscope_type: 'auto', 'imagexpress', 'opera_phenix', 'openhcs'.
        plate_folder: Required for 'auto' detection.
        filemanager: FileManager instance. Must be provided.
        pattern_format: Name of the pattern format to use.
        allowed_auto_types: For 'auto' mode, limit detection to these types.
                           'openhcs' is always included and tried first.

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
        microscope_type = _auto_detect_microscope_type(plate_folder, filemanager, allowed_types=allowed_auto_types)
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

    # If the handler is OpenHCSMicroscopeHandler, set its plate_folder attribute.
    # This is crucial for its dynamic parser loading mechanism.
    # Use string comparison to avoid circular import
    if handler.__class__.__name__ == 'OpenHCSMicroscopeHandler':
        if plate_folder:
            handler.plate_folder = Path(plate_folder) if isinstance(plate_folder, str) else plate_folder
            logger.info(f"Set plate_folder for OpenHCSMicroscopeHandler: {handler.plate_folder}")
        else:
            # This case should ideally not happen if auto-detection or explicit type setting
            # implies a plate_folder is known.
            logger.warning("OpenHCSMicroscopeHandler created without an initial plate_folder. "
                           "Parser will load upon first relevant method call with a path e.g. post_workspace.")

    return handler


def validate_backend_compatibility(handler: MicroscopeHandler, backend: Backend) -> bool:
    """
    Validate that a microscope handler supports a given storage backend.

    Args:
        handler: MicroscopeHandler instance to check
        backend: Backend to validate compatibility with

    Returns:
        bool: True if the handler supports the backend, False otherwise

    Example:
        >>> handler = ImageXpressHandler(filemanager)
        >>> validate_backend_compatibility(handler, Backend.ZARR)
        False
        >>> validate_backend_compatibility(handler, Backend.DISK)
        True
    """
    return backend in handler.supported_backends


def _try_metadata_detection(handler_class, filemanager: FileManager, plate_folder: Path) -> Optional[Path]:
    """
    Try metadata detection with a handler, normalizing return types and exceptions.

    Args:
        handler_class: MetadataHandler class to try
        filemanager: FileManager instance
        plate_folder: Path to plate directory

    Returns:
        Path if metadata found, None if not found (regardless of handler's native behavior)
    """
    try:
        handler = handler_class(filemanager)
        result = handler.find_metadata_file(plate_folder)

        # Normalize return type: convert any truthy result to Path, falsy to None
        return Path(result) if result else None

    except (FileNotFoundError, Exception) as e:
        # Expected exceptions for "not found" - convert to None
        # Note: Using broad Exception catch for now, can be refined based on actual handler exceptions
        logger.debug(f"Metadata detection failed for {handler_class.__name__}: {e}")
        return None


def _auto_detect_microscope_type(plate_folder: Path, filemanager: FileManager,
                                allowed_types: Optional[List[str]] = None) -> str:
    """
    Auto-detect microscope type using registry iteration.

    Args:
        plate_folder: Path to plate directory
        filemanager: FileManager instance
        allowed_types: Optional list of microscope types to try.
                      If None, tries all registered types.
                      'openhcs' is always included and tried first.

    Returns:
        Detected microscope type string

    Raises:
        ValueError: If microscope type cannot be determined
    """
    try:
        # Build detection order: openhcsdata first, then filtered/ordered list
        detection_order = ['openhcsdata']  # Always first, always included (correct registration name)

        if allowed_types is None:
            # Use all registered handlers in registration order
            detection_order.extend([name for name in METADATA_HANDLERS.keys() if name != 'openhcsdata'])
        else:
            # Use filtered list, but ensure openhcsdata is first
            filtered_types = [name for name in allowed_types if name != 'openhcsdata' and name in METADATA_HANDLERS]
            detection_order.extend(filtered_types)

        # Try detection in order
        for handler_name in detection_order:
            handler_class = METADATA_HANDLERS.get(handler_name)
            if handler_class and _try_metadata_detection(handler_class, filemanager, plate_folder):
                logger.info(f"Auto-detected {handler_name} microscope type")
                return handler_name

        # No handler succeeded - provide detailed error message
        available_types = list(METADATA_HANDLERS.keys())
        msg = (f"Could not auto-detect microscope type in {plate_folder}. "
               f"Tried: {detection_order}. "
               f"Available types: {available_types}. "
               f"Ensure metadata files are present for supported formats.")
        logger.error(msg)
        raise ValueError(msg)

    except Exception as e:
        # Wrap exception with clear context
        raise ValueError(f"Error during microscope type auto-detection in {plate_folder}: {e}") from e
