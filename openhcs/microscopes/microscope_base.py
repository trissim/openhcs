"""
Microscope base implementations for openhcs.

This module provides the base implementations for microscope-specific functionality,
including filename parsing and metadata handling.
"""

import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Union, Type, TYPE_CHECKING

# Import constants
from openhcs.constants.constants import Backend, Microscope
from openhcs.core.source_metadata import source_metadata_dict
# Import generic metaclass infrastructure from external package
from metaclass_registry import (
    AutoRegisterMeta,
    SecondaryRegistry,
    extract_key_from_handler_suffix,
    PRIMARY_KEY
)
# PatternDiscoveryEngine imported locally to avoid circular imports
from polystore.filemanager import FileManager
from polystore.streaming.viewer_transport import ViewerMicroscopeHandlerABC
from polystore.virtual_workspace import SourcePixelRef
# Import interfaces from the base interfaces module
from openhcs.microscopes.microscope_interfaces import (
    FilenameParseResult,
    FilenameParser,
    MetadataHandler,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from openhcs.core.config import MaterializationBackend
    from openhcs.core.source_bindings import SourceBindingsConfig

# Dictionary to store registered metadata handlers for auto-detection
# This will be auto-wrapped with SecondaryRegistryDict by the metaclass
METADATA_HANDLERS = {}
OPENHCS_DATA_MICROSCOPE_TYPE = Microscope.OPENHCS.value


def register_metadata_handler(handler_class, metadata_handler_class):
    """
    Register a metadata handler for a microscope handler class.

    This function is called when _metadata_handler_class is set after class definition.
    """
    microscope_type = handler_class._microscope_type
    METADATA_HANDLERS[microscope_type] = metadata_handler_class
    logger.debug(f"Registered metadata handler {metadata_handler_class.__name__} for '{microscope_type}'")




class MicroscopeSourceSelectionRole(str, Enum):
    """How a registered handler should participate in source selection."""

    FORMAT_SPECIFIC = "format_specific"
    BROAD_STRUCTURED_STORE = "broad_structured_store"
    DECLARED_FILE_FALLBACK = "declared_file_fallback"
    REMOTE_SERVICE = "remote_service"
    PREPARED_WORKSPACE = "prepared_workspace"


class BroadMicroscopeDetector:
    """Generic store detector that must run after format-specific handlers."""

    @classmethod
    def source_selection_role(cls) -> MicroscopeSourceSelectionRole:
        """Declare broad structured-store ownership through the existing marker."""

        return MicroscopeSourceSelectionRole.BROAD_STRUCTURED_STORE

    @classmethod
    def source_selection_guidance(cls) -> str:
        """Explain when a broad structured-store decoder is the right owner."""

        return (
            "Use for a supported structured or rich container when no registered "
            "format-specific parser recognizes the source layout. A missing native "
            "metadata file alone does not make the broad decoder semantically "
            "preferable when a format-specific parser recognizes the filenames."
        )


class MicroscopeHandler(ViewerMicroscopeHandlerABC, ABC, metaclass=AutoRegisterMeta):
    """
    Composed class for handling microscope-specific functionality.

    Registry auto-created and stored as MicroscopeHandler.__registry__.
    Subclasses auto-register by setting _microscope_type class attribute.
    Secondary registry METADATA_HANDLERS populated via _metadata_handler_class.
    """
    __registry_key__ = '_microscope_type'
    __key_extractor__ = extract_key_from_handler_suffix
    __skip_if_no_key__ = False
    __secondary_registries__ = [
        SecondaryRegistry(
            registry_dict=METADATA_HANDLERS,
            key_source=PRIMARY_KEY,
            attr_name='_metadata_handler_class'
        )
    ]

    DEFAULT_MICROSCOPE = 'auto'
    _handlers_cache = None

    # Optional class attribute for explicit metadata handler registration
    _metadata_handler_class: Optional[Type[MetadataHandler]] = None

    def __init__(self, parser: Optional[FilenameParser],
                 metadata_handler: MetadataHandler):
        """
        Initialize the microscope handler.

        Args:
            parser: Parser for microscopy filenames. Can be None for virtual backends
                   that don't need workspace preparation or pattern discovery.
            metadata_handler: Handler for microscope metadata.
        """
        self.parser = parser
        self.metadata_handler = metadata_handler
        self.plate_folder: Optional[Path] = None # Store workspace path if needed by methods

        # Pattern discovery engine will be created on demand with the provided filemanager

    @classmethod
    def create(
        cls,
        *,
        filemanager: FileManager,
        pattern_format: Optional[str] = None,
        source_bindings_config: Optional["SourceBindingsConfig"] = None,
    ) -> "MicroscopeHandler":
        """Construct a registered microscope handler from factory inputs."""
        del source_bindings_config
        return cls(filemanager, pattern_format=pattern_format)

    @classmethod
    def projects_declared_source_bindings(cls) -> bool:
        """Return whether this handler projects named bindings onto its sources."""

        return False

    @classmethod
    def source_selection_role(cls) -> MicroscopeSourceSelectionRole:
        """Declare this handler as the owner of a format-specific source layout."""

        return MicroscopeSourceSelectionRole.FORMAT_SPECIFIC

    @classmethod
    def supports_explicit_incomplete_export(cls) -> bool:
        """Return whether parser-recognized subsets are valid explicit inputs.

        Format-specific leaves override this when their metadata detection contract
        is required for a dataset to have valid native semantics.
        """

        return True

    @classmethod
    def source_selection_guidance(cls) -> str:
        """Explain format-specific auto-detection and explicit partial selection."""

        if not cls.supports_explicit_incomplete_export():
            return (
                "Use only when this handler's complete detection contract succeeds. "
                "Parser recognition without the required vendor metadata is filename "
                "evidence, not a valid native dataset. Obtain the complete export, or "
                "treat independently decodable ordinary files through the declared-file "
                "fallback with explicit source semantics."
            )
        return (
            "Use when this handler's parser and layout describe the dataset. Prefer a "
            "complete source layout/export so the handler detection contract and "
            "metadata owner can supply the available plate facts. If the parser "
            "recognizes an intentionally incomplete export, select this handler "
            "explicitly and expect metadata-derived fields to remain unavailable."
        )

    @classmethod
    def detect(cls, plate_folder: Path, filemanager: FileManager) -> bool:
        """Return whether this handler can initialize the plate."""
        from polystore.exceptions import MetadataNotFoundError

        metadata_handler_class = cls._metadata_handler_class
        if metadata_handler_class is None:
            raise RuntimeError(f"{cls.__name__} missing _metadata_handler_class for detection")

        metadata_handler = metadata_handler_class(filemanager)
        try:
            metadata_handler.find_metadata_file(plate_folder)
        except (MetadataNotFoundError, FileNotFoundError, TypeError):
            return False
        return True

    @property
    @abstractmethod
    def root_dir(self) -> str:
        """
        Root directory where virtual workspace preparation starts.

        This defines:
        1. The starting point for virtual workspace operations (flattening, remapping)
        2. The subdirectory key used when saving virtual workspace metadata

        Examples:
        - ImageXpress: "." (plate root - TimePoint/ZStep folders are flattened from plate root)
        - OperaPhenix: "Images" (field remapping applied to Images/ subdirectory)
        - OpenHCS: Determined from metadata (e.g., "zarr", "images")
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
        - [Backend.OMERO_LOCAL] - Virtual backends (OMERO: single required backend)

        Returns:
            List of Backend enum values this handler can work with, in priority order
        """
        pass

    def get_required_backend(self) -> Optional['MaterializationBackend']:
        """
        Get the required materialization backend if this microscope has only one compatible backend.

        For microscopes with a single compatible backend (e.g., OMERO with OMERO_LOCAL),
        this returns the required backend for auto-correction. For microscopes with multiple
        compatible backends, returns None (user must choose explicitly).

        Returns:
            MaterializationBackend if microscope requires a specific backend, None otherwise
        """
        from openhcs.core.config import MaterializationBackend

        if len(self.compatible_backends) == 1:
            backend_value = self.compatible_backends[0].value
            materialization_values = {
                materialization_backend.value
                for materialization_backend in MaterializationBackend
            }
            if backend_value not in materialization_values:
                raise RuntimeError(
                    f"{self.microscope_type} declares single backend "
                    f"{backend_value!r}, but it is not a materialization backend."
                )
            return MaterializationBackend(backend_value)
        return None

    def get_available_backends(self, plate_path: Union[str, Path]) -> List[Backend]:
        """
        Get available storage backends for this specific plate.

        Default implementation returns all compatible backends.
        Override this method only if you need to check actual disk state
        (like OpenHCS which reads from metadata).

        Args:
            plate_path: Path to the plate folder

        Returns:
            List of Backend enums that are available for this plate.
        """
        return self.compatible_backends

    def get_primary_backend(self, plate_path: Union[str, Path], filemanager: 'FileManager') -> str:
        """
        Get the primary backend name for this plate.

        Checks FileManager registry first for registered backends (like virtual_workspace),
        then falls back to compatible backends.

        Override this method only if you need custom backend selection logic
        (like OpenHCS which reads from metadata).

        Args:
            plate_path: Path to the plate folder (or subdirectory)
            filemanager: FileManager instance to check for registered backends

        Returns:
            Backend name string (e.g., 'disk', 'zarr', 'virtual_workspace')
        """
        # Check if virtual workspace backend is registered in FileManager
        # This takes priority over compatible backends
        if Backend.VIRTUAL_WORKSPACE.value in filemanager.registry:
            logger.info(f"✅ Using backend '{Backend.VIRTUAL_WORKSPACE.value}' from FileManager registry")
            return Backend.VIRTUAL_WORKSPACE.value

        # Fall back to compatible backends
        available_backends = self.get_available_backends(plate_path)
        if not available_backends:
            raise RuntimeError(f"No available backends for {self.microscope_type} microscope at {plate_path}")
        logger.info(f"⚠️ Using backend '{available_backends[0].value}' from compatible backends (virtual workspace not registered)")
        return available_backends[0].value

    def save_virtual_workspace_metadata(
        self,
        plate_path: Path,
        workspace_mapping: Mapping[str, SourcePixelRef],
    ) -> None:
        """
        Save virtual workspace mapping and handler metadata to openhcs_metadata.json.

        This is a helper method for subclasses to call from _build_virtual_mapping().
        It writes all available metadata fields including handler name and parser name.

        Args:
            plate_path: Path to plate directory
            workspace_mapping: Virtual paths mapped to backend-owned source refs.
        """
        from openhcs.core.virtual_workspace_metadata import (
            AtomicMetadataWriter,
            FIELDS,
            get_metadata_path,
        )

        metadata_path = get_metadata_path(plate_path)
        writer = AtomicMetadataWriter()

        # Build metadata dict with all available fields
        metadata_dict = {
            FIELDS.WORKSPACE_MAPPING: {
                virtual_path: source_ref.to_workspace_mapping()
                for virtual_path, source_ref in workspace_mapping.items()
            },
            FIELDS.SOURCE_METADATA: {
                virtual_path: source_metadata_dict(parsed)
                for virtual_path in workspace_mapping
                if (
                    parsed := self.parser.parse_filename(
                        Path(virtual_path).name
                    )
                )
                is not None
            },
            FIELDS.AVAILABLE_BACKENDS: {
                Backend.DISK.value: True,
                Backend.VIRTUAL_WORKSPACE.value: True,
            },
            FIELDS.MICROSCOPE_HANDLER_NAME: self.microscope_type,
            FIELDS.SOURCE_FILENAME_PARSER_NAME: self.parser.__class__.__name__,
        }

        metadata_dict[FIELDS.GRID_DIMENSIONS] = (
            self.metadata_handler.get_grid_dimensions(plate_path)
        )
        metadata_dict[FIELDS.PIXEL_SIZE] = self.metadata_handler.get_pixel_size(
            plate_path
        )

        writer.merge_subdirectory_metadata(metadata_path, {self.root_dir: metadata_dict})
        logger.info(f"✅ Saved virtual workspace metadata to {metadata_path}")

    def _register_virtual_workspace_backend(self, plate_path: Union[str, Path], filemanager: FileManager) -> None:
        """
        Register virtual workspace backend for this plate.

        VirtualWorkspace backends are plate-specific (each has a plate_root),
        so we always create a new backend for each plate, replacing any existing one.
        This ensures each plate uses the correct virtual workspace mapping.

        Args:
            plate_path: Path to plate directory
            filemanager: FileManager instance
        """
        from polystore.virtual_workspace import VirtualWorkspaceBackend
        from openhcs.constants.constants import Backend

        # Always create a new backend for this plate (VirtualWorkspace is plate-specific)
        backend = VirtualWorkspaceBackend(plate_root=Path(plate_path))
        filemanager.register_backend(Backend.VIRTUAL_WORKSPACE.value, backend)
        logger.info(f"Registered virtual workspace backend for {plate_path}")

    def register_workspace_backends(
        self,
        plate_path: Union[str, Path],
        filemanager: FileManager,
    ) -> None:
        """Register the backends required to replay this handler's workspace."""

        self._register_virtual_workspace_backend(plate_path, filemanager)

    def register_source_backends(self, filemanager: FileManager) -> None:
        """Register direct source backends needed before workspace materialization."""

        del filemanager

    def initialize_workspace(self, plate_path: Path, filemanager: FileManager) -> Path:
        """
        Initialize plate by creating virtual mapping in metadata.

        No physical workspace directory is created - mapping is purely metadata-based.
        All paths are relative to plate_path.

        Args:
            plate_path: Path to plate directory
            filemanager: FileManager instance

        Returns:
            Path to image directory (determined from plate structure)
        """
        plate_path = Path(plate_path)

        # Set plate_folder for this handler
        self.plate_folder = plate_path

        # Call microscope-specific virtual mapping builder
        # This builds the plate-relative mapping dict and saves to metadata
        self._build_virtual_mapping(plate_path, filemanager)

        # Register virtual workspace backend
        self._register_virtual_workspace_backend(plate_path, filemanager)

        # Return image directory using post_workspace
        # skip_preparation=True because _build_virtual_mapping() already ran
        return self.post_workspace(plate_path, filemanager, skip_preparation=True)

    def post_workspace(self, plate_path: Union[str, Path], filemanager: FileManager, skip_preparation: bool = False) -> Path:
        """
        Apply post-workspace processing using virtual mapping.

        All operations use plate_path - no workspace directory concept.

        Args:
            plate_path: Path to plate directory
            filemanager: FileManager instance
            skip_preparation: Skip microscope-specific preparation (default: False)

        Returns:
            Path to image directory
        """
        # Ensure plate_path is a Path object
        if isinstance(plate_path, str):
            plate_path = Path(plate_path)

        # NO existence check - virtual workspaces are metadata-only

        # Apply microscope-specific preparation logic
        # _build_virtual_mapping() returns the correct image directory for each microscope:
        # - ImageXpress: plate_path (images are at plate root with TimePoint_X/ prefix in mapping)
        # - OperaPhenix: plate_path/Images (images are in Images/ subdirectory)
        if skip_preparation:
            logger.info("📁 SKIPPING PREPARATION: Virtual mapping already built")
            # When skipping, we need to determine image_dir from metadata
            # Read metadata to get the subdirectory key
            from openhcs.core.virtual_workspace_metadata import FIELDS
            from openhcs.microscopes.openhcs import (
                OpenHCSMetadataHandler,
                resolve_subdirectory_path,
            )
            from polystore.exceptions import MetadataNotFoundError

            openhcs_metadata_handler = OpenHCSMetadataHandler(filemanager)
            metadata = openhcs_metadata_handler._load_metadata_dict(plate_path)
            if FIELDS.SUBDIRECTORIES not in metadata:
                raise MetadataNotFoundError(
                    f"skip_preparation=True but '{FIELDS.SUBDIRECTORIES}' is missing "
                    f"from metadata for {plate_path}."
                )
            subdirs = metadata[FIELDS.SUBDIRECTORIES]

            # Find the subdirectory with the virtual workspace mapping.
            subdir_with_mapping = None
            for name, data in subdirs.items():
                if FIELDS.WORKSPACE_MAPPING in data:
                    subdir_with_mapping = name
                    break

            # Fail if no virtual workspace mapping was found.
            if subdir_with_mapping is None:
                raise MetadataNotFoundError(
                    "skip_preparation=True but no "
                    f"{FIELDS.WORKSPACE_MAPPING} found in metadata for {plate_path}. "
                    "Virtual workspace must be prepared before skipping."
                )

            # Convert subdirectory name to path (handles "." -> plate_path)
            image_dir = resolve_subdirectory_path(subdir_with_mapping, plate_path)
        else:
            logger.info("🔄 APPLYING PREPARATION: Building virtual mapping")
            # _build_virtual_mapping() returns the image directory
            image_dir = self._build_virtual_mapping(plate_path, filemanager)

        # Determine backend - check if virtual workspace backend is registered
        if Backend.VIRTUAL_WORKSPACE.value in filemanager.registry:
            backend_type = Backend.VIRTUAL_WORKSPACE.value
        else:
            backend_type = Backend.DISK.value

        # Ensure parser is provided
        parser = self.parser

        # Get all image files in the directory
        image_files = filemanager.list_image_files(image_dir, backend_type)

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
            metadata['site'] = site
            metadata['channel'] = channel
            metadata['z_index'] = z_index
            new_name = parser.construct_filename(**metadata)

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

    def _build_virtual_mapping(self, plate_path: Path, filemanager: FileManager) -> Path:
        """
        Build microscope-specific virtual workspace mapping.

        This method creates a plate-relative mapping dict and saves it to metadata.
        All paths in the mapping are relative to plate_path.

        Override in subclasses that need virtual workspace mapping (e.g., ImageXpress, Opera Phenix).
        Handlers that override initialize_workspace() completely (e.g., OMERO, OpenHCS) don't need
        to implement this method.

        Args:
            plate_path: Path to plate directory
            filemanager: FileManager instance for file operations

        Returns:
            Path: Suggested directory for further processing

        Raises:
            NotImplementedError: If called on a handler that doesn't support virtual workspace mapping
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement _build_virtual_mapping(). "
            f"This method is only needed for handlers that use the base class initialize_workspace(). "
            f"Handlers that override initialize_workspace() completely (like OMERO, OpenHCS) don't need this."
        )

    # Delegate methods to parser
    def parse_filename(self, filename: str) -> Optional[FilenameParseResult]:
        """Delegate to parser."""
        return self.parser.parse_filename(filename)

    def construct_filename(self, extension: str = '.tif', **component_values) -> str:
        """
        Delegate to parser using pure generic interface.
        """
        return self.parser.construct_filename(extension=extension, **component_values)

    def auto_detect_patterns(self, folder_path: Union[str, Path], filemanager: FileManager, backend: str,
                           extensions=None, group_by=None, variable_components=None, **kwargs):
        """
        Delegate to pattern engine.

        Args:
            folder_path: Path to the folder (string or Path object)
            filemanager: FileManager instance for file operations
            backend: Backend to use for file operations (required)
            extensions: Optional list of file extensions to include
            group_by: GroupBy enum to group patterns by (e.g., GroupBy.CHANNEL, GroupBy.Z_INDEX)
            variable_components: List of components to make variable (e.g., ['site', 'z_index'])
            **kwargs: Dynamic filter parameters (e.g., well_filter, site_filter, channel_filter)

        Returns:
            Dict[str, Any]: Dictionary mapping axis values to patterns
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
        from openhcs.formats.pattern.pattern_discovery import PatternDiscoveryEngine
        pattern_engine = PatternDiscoveryEngine(self.parser, filemanager)

        # Get patterns from the pattern engine
        patterns_by_well = pattern_engine.auto_detect_patterns(
            folder_path,
            extensions=extensions,
            group_by=group_by,
            variable_components=variable_components,
            backend=backend,
            **kwargs  # Pass through dynamic filter parameters
        )

        # Ensure we always return a dictionary, not a generator
        if not isinstance(patterns_by_well, dict):
            # Convert to dictionary if it's not already one
            return dict(patterns_by_well)

        return patterns_by_well

    def path_list_from_pattern(
        self,
        directory: Union[str, Path],
        pattern,
        filemanager: FileManager,
        backend: str,
        variable_components: Optional[List[str]] = None,
    ):
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
        from openhcs.formats.pattern.pattern_discovery import PatternDiscoveryEngine
        pattern_engine = PatternDiscoveryEngine(self.parser, filemanager)

        # Delegate to the pattern engine
        return pattern_engine.path_list_from_pattern(directory_path, pattern, backend=backend, variable_components=variable_components)

    # Delegate metadata handling methods to metadata_handler with context

    def find_metadata_file(self, plate_path: Union[str, Path]) -> Path:
        """Delegate to metadata handler."""
        return self.metadata_handler.find_metadata_file(plate_path)

    def get_grid_dimensions(self, plate_path: Union[str, Path]) -> Tuple[int, int]:
        """Delegate to metadata handler."""
        return self.metadata_handler.get_grid_dimensions(plate_path)

    def can_resolve_metadata_artifact(self, artifact_name: str) -> bool:
        """Return whether this microscope can provide a metadata artifact."""
        return self.metadata_handler.can_resolve_metadata_artifact(artifact_name)

    def resolve_metadata_artifact(
        self,
        artifact_name: str,
        plate_path: Union[str, Path],
    ) -> object:
        """Resolve a metadata artifact through this microscope's metadata handler."""
        return self.metadata_handler.resolve_metadata_artifact(
            artifact_name,
            plate_path,
        )

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
                              source_bindings_config: Optional["SourceBindingsConfig"] = None,
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
        source_bindings_config: Source-binding declarations for explicit source-binding mode.
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

    source_bindings = None
    if source_bindings_config is not None:
        from openhcs.core.source_bindings import source_bindings_defaults_to_base

        source_bindings = source_bindings_defaults_to_base(source_bindings_config)

    if microscope_type == 'auto':
        if not plate_folder:
            raise ValueError("plate_folder is required for auto-detection")
        plate_folder = Path(plate_folder) if isinstance(plate_folder, str) else plate_folder
        detected_microscope_type = _auto_detect_microscope_type(
            plate_folder,
            filemanager,
            allowed_types=allowed_auto_types,
        )
        if detected_microscope_type is None:
            if source_bindings is None or source_bindings.is_empty:
                raise ValueError(
                    f"Could not auto-detect microscope type in {plate_folder}."
                )
            detected_microscope_type = source_bindings.microscope_handler_name
        microscope_type = detected_microscope_type
        logger.info("Auto-detected microscope type: %s", microscope_type)

    # Source bindings use a store handler when that detected/declared owner projects them.
    if source_bindings is not None:
        declared_handler_class = MICROSCOPE_HANDLERS.get(microscope_type)
        if (
            not source_bindings.is_empty
            and (
                declared_handler_class is None
                or not declared_handler_class.projects_declared_source_bindings()
            )
        ):
            microscope_type = source_bindings.microscope_handler_name
            logger.info(
                "Selected %s microscope from source-binding declarations",
                microscope_type,
            )

    # Handlers auto-discovered on first access to MICROSCOPE_HANDLERS
    from openhcs.microscopes.handler_registry_service import get_all_handler_types

    # Get the appropriate handler class from the registry
    # No dynamic imports or fallbacks (Clause 77: Rot Intolerance)
    handler_class = MICROSCOPE_HANDLERS.get(microscope_type)
    if not handler_class:
        available_types = get_all_handler_types()
        raise ValueError(
            f"Unsupported microscope type: {microscope_type}. "
            f"Available types: {available_types}"
        )

    # Create and configure the handler
    logger.info(f"Creating {handler_class.__name__}")

    handler = handler_class.create(
        filemanager=filemanager,
        pattern_format=pattern_format,
        source_bindings_config=source_bindings_config,
    )

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


def _auto_detect_microscope_type(plate_folder: Path, filemanager: FileManager,
                                allowed_types: Optional[List[str]] = None) -> Optional[str]:
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

    Returns ``None`` when no registered physical handler declares the path.
    """
    # Build detection order: OpenHCS data first, then filtered/ordered handlers.
    detection_order = [OPENHCS_DATA_MICROSCOPE_TYPE]

    if allowed_types is None:
        # Use all registered handlers in registration order
        detection_order.extend(
            name
            for name in MICROSCOPE_HANDLERS.keys()
            if name != OPENHCS_DATA_MICROSCOPE_TYPE
        )
    else:
        # Use filtered list, but ensure OpenHCS data is first.
        filtered_types = [
            name
            for name in allowed_types
            if name != OPENHCS_DATA_MICROSCOPE_TYPE and name in MICROSCOPE_HANDLERS
        ]
        detection_order.extend(filtered_types)
    detection_order = _detector_contract_order(detection_order)

    # Try detection in order - only catch expected "not found" exceptions
    for handler_name in detection_order:
        handler_class = MICROSCOPE_HANDLERS[handler_name]

        if handler_class.detect(plate_folder, filemanager):
            logger.info(f"Auto-detected {handler_name} microscope type")
            return handler_name
        logger.debug(f"{handler_name} metadata not found in {plate_folder}")

    logger.debug(
        "No registered physical microscope handler declared %s after trying %s",
        plate_folder,
        detection_order,
    )
    return None


# ============================================================================
# Registry Export
# ============================================================================
# Auto-created registry from MicroscopeHandler base class
MICROSCOPE_HANDLERS = MicroscopeHandler.__registry__


def _detector_contract_order(handler_names: List[str]) -> List[str]:
    """Order metadata-backed detectors before filename/path-only detectors."""

    keyed_handlers = []
    for index, name in enumerate(handler_names):
        handler_class = MICROSCOPE_HANDLERS[name]
        if issubclass(handler_class, BroadMicroscopeDetector):
            contract_rank = 2
        elif "detect" in handler_class.__dict__:
            contract_rank = 1
        else:
            contract_rank = 0
        keyed_handlers.append((contract_rank, index, name))
    return [name for _contract_rank, _index, name in sorted(keyed_handlers)]
