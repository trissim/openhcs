"""
OpenHCS microscope handler implementation for openhcs.

This module provides the OpenHCSMicroscopeHandler, which reads plates
that have been pre-processed and standardized into the OpenHCS format.
The metadata for such plates is defined in an 'openhcs_metadata.json' file.
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Type

from openhcs.constants.constants import Backend, GroupBy, DEFAULT_IMAGE_EXTENSIONS
from openhcs.io.exceptions import MetadataNotFoundError
from openhcs.io.filemanager import FileManager
from openhcs.io.metadata_writer import AtomicMetadataWriter, MetadataWriteError, get_metadata_path, METADATA_CONFIG
from openhcs.microscopes.microscope_interfaces import MetadataHandler
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenHCSMetadataFields:
    """Centralized constants for OpenHCS metadata field names."""
    # Core metadata structure - use centralized constants
    SUBDIRECTORIES: str = METADATA_CONFIG.SUBDIRECTORIES_KEY
    IMAGE_FILES: str = "image_files"
    AVAILABLE_BACKENDS: str = METADATA_CONFIG.AVAILABLE_BACKENDS_KEY

    # Required metadata fields
    GRID_DIMENSIONS: str = "grid_dimensions"
    PIXEL_SIZE: str = "pixel_size"
    SOURCE_FILENAME_PARSER_NAME: str = "source_filename_parser_name"
    MICROSCOPE_HANDLER_NAME: str = "microscope_handler_name"

    # Optional metadata fields
    CHANNELS: str = "channels"
    WELLS: str = "wells"
    SITES: str = "sites"
    Z_INDEXES: str = "z_indexes"
    OBJECTIVES: str = "objectives"
    ACQUISITION_DATETIME: str = "acquisition_datetime"
    PLATE_NAME: str = "plate_name"

    # Default values
    DEFAULT_SUBDIRECTORY: str = "."
    DEFAULT_SUBDIRECTORY_LEGACY: str = "images"

    # Microscope type identifier
    MICROSCOPE_TYPE: str = "openhcsdata"


# Global instance for easy access
FIELDS = OpenHCSMetadataFields()

def _get_available_filename_parsers():
    """
    Lazy import of filename parsers to avoid circular imports.

    Returns:
        Dict mapping parser class names to parser classes
    """
    # Import parsers only when needed to avoid circular imports
    from openhcs.microscopes.imagexpress import ImageXpressFilenameParser
    from openhcs.microscopes.opera_phenix import OperaPhenixFilenameParser

    return {
        "ImageXpressFilenameParser": ImageXpressFilenameParser,
        "OperaPhenixFilenameParser": OperaPhenixFilenameParser,
        # Add other parsers to this dictionary as they are implemented/imported.
        # Example: "MyOtherParser": MyOtherParser,
    }


class OpenHCSMetadataHandler(MetadataHandler):
    """
    Metadata handler for the OpenHCS pre-processed format.

    This handler reads metadata from an 'openhcs_metadata.json' file
    located in the root of the plate folder.
    """
    METADATA_FILENAME = METADATA_CONFIG.METADATA_FILENAME

    def __init__(self, filemanager: FileManager):
        """
        Initialize the metadata handler.

        Args:
            filemanager: FileManager instance for file operations.
        """
        super().__init__()
        self.filemanager = filemanager
        self.atomic_writer = AtomicMetadataWriter()
        self._metadata_cache: Optional[Dict[str, Any]] = None
        self._plate_path_cache: Optional[Path] = None

    def _load_metadata(self, plate_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Loads the JSON metadata file if not already cached or if plate_path changed.

        Args:
            plate_path: Path to the plate folder.

        Returns:
            A dictionary containing the parsed JSON metadata.

        Raises:
            MetadataNotFoundError: If the metadata file cannot be found or parsed.
            FileNotFoundError: If plate_path does not exist.
        """
        current_path = Path(plate_path)
        if self._metadata_cache is not None and self._plate_path_cache == current_path:
            return self._metadata_cache

        metadata_file_path = self.find_metadata_file(current_path)
        if not self.filemanager.exists(str(metadata_file_path), Backend.DISK.value):
            raise MetadataNotFoundError(f"Metadata file '{self.METADATA_FILENAME}' not found in {plate_path}")

        try:
            content = self.filemanager.load(str(metadata_file_path), Backend.DISK.value)
            metadata_dict = json.loads(content.decode('utf-8') if isinstance(content, bytes) else content)

            # Handle subdirectory-keyed format
            if subdirs := metadata_dict.get(FIELDS.SUBDIRECTORIES):
                if not subdirs:
                    raise MetadataNotFoundError(f"Empty subdirectories in metadata file '{metadata_file_path}'")

                # Merge all subdirectories: use first as base, combine all image_files
                base_metadata = next(iter(subdirs.values())).copy()
                base_metadata[FIELDS.IMAGE_FILES] = [
                    file for subdir in subdirs.values()
                    for file in subdir.get(FIELDS.IMAGE_FILES, [])
                ]
                self._metadata_cache = base_metadata
            else:
                # Legacy format not supported - use migration script
                raise MetadataNotFoundError(
                    f"Legacy metadata format detected in '{metadata_file_path}'. "
                    f"Please run the migration script: python scripts/migrate_legacy_metadata.py {current_path}"
                )

            self._plate_path_cache = current_path
            return self._metadata_cache

        except json.JSONDecodeError as e:
            raise MetadataNotFoundError(f"Error decoding JSON from '{metadata_file_path}': {e}") from e



    def determine_main_subdirectory(self, plate_path: Union[str, Path]) -> str:
        """Determine main input subdirectory from metadata."""
        metadata_dict = self._load_metadata_dict(plate_path)
        subdirs = metadata_dict.get(FIELDS.SUBDIRECTORIES)

        # Legacy format not supported - should have been caught by _load_metadata_dict
        if not subdirs:
            raise MetadataNotFoundError(f"No subdirectories found in metadata for {plate_path}")

        # Single subdirectory - use it
        if len(subdirs) == 1:
            return next(iter(subdirs.keys()))

        # Multiple subdirectories - find main or fallback
        main_subdir = next((name for name, data in subdirs.items() if data.get("main")), None)
        if main_subdir:
            return main_subdir

        # Fallback hierarchy: legacy default -> first available
        if FIELDS.DEFAULT_SUBDIRECTORY_LEGACY in subdirs:
            return FIELDS.DEFAULT_SUBDIRECTORY_LEGACY
        else:
            return next(iter(subdirs.keys()))

    def _load_metadata_dict(self, plate_path: Union[str, Path]) -> Dict[str, Any]:
        """Load and parse metadata JSON, fail-loud on errors."""
        metadata_file_path = self.find_metadata_file(plate_path)
        if not self.filemanager.exists(str(metadata_file_path), Backend.DISK.value):
            raise MetadataNotFoundError(f"Metadata file '{self.METADATA_FILENAME}' not found in {plate_path}")

        try:
            content = self.filemanager.load(str(metadata_file_path), Backend.DISK.value)
            return json.loads(content.decode('utf-8') if isinstance(content, bytes) else content)
        except json.JSONDecodeError as e:
            raise MetadataNotFoundError(f"Error decoding JSON from '{metadata_file_path}': {e}") from e

    def find_metadata_file(self, plate_path: Union[str, Path], context: Optional[Any] = None) -> Optional[Path]:
        """Find the OpenHCS JSON metadata file."""
        plate_p = Path(plate_path)
        if not self.filemanager.is_dir(str(plate_p), Backend.DISK.value):
            return None

        expected_file = plate_p / self.METADATA_FILENAME
        if self.filemanager.exists(str(expected_file), Backend.DISK.value):
            return expected_file

        # Fallback: recursive search
        try:
            if found_files := self.filemanager.find_file_recursive(plate_p, self.METADATA_FILENAME, Backend.DISK.value):
                if isinstance(found_files, list):
                    # Prioritize root location, then first found
                    return next((Path(f) for f in found_files if Path(f).parent == plate_p), Path(found_files[0]))
                return Path(found_files)
        except Exception as e:
            logger.error(f"Error searching for {self.METADATA_FILENAME} in {plate_path}: {e}")

        return None


    def get_grid_dimensions(self, plate_path: Union[str, Path], context: Optional[Any] = None) -> Tuple[int, int]:
        """Get grid dimensions from OpenHCS metadata."""
        dims = self._load_metadata(plate_path).get(FIELDS.GRID_DIMENSIONS)
        if not (isinstance(dims, list) and len(dims) == 2 and all(isinstance(d, int) for d in dims)):
            raise ValueError(f"'{FIELDS.GRID_DIMENSIONS}' must be a list of two integers in {self.METADATA_FILENAME}")
        return tuple(dims)

    def get_pixel_size(self, plate_path: Union[str, Path], context: Optional[Any] = None) -> float:
        """Get pixel size from OpenHCS metadata."""
        pixel_size = self._load_metadata(plate_path).get(FIELDS.PIXEL_SIZE)
        if not isinstance(pixel_size, (float, int)):
            raise ValueError(f"'{FIELDS.PIXEL_SIZE}' must be a number in {self.METADATA_FILENAME}")
        return float(pixel_size)

    def get_source_filename_parser_name(self, plate_path: Union[str, Path]) -> str:
        """Get source filename parser name from OpenHCS metadata."""
        parser_name = self._load_metadata(plate_path).get(FIELDS.SOURCE_FILENAME_PARSER_NAME)
        if not (isinstance(parser_name, str) and parser_name):
            raise ValueError(f"'{FIELDS.SOURCE_FILENAME_PARSER_NAME}' must be a non-empty string in {self.METADATA_FILENAME}")
        return parser_name

    def get_image_files(self, plate_path: Union[str, Path]) -> List[str]:
        """Get image files list from OpenHCS metadata."""
        image_files = self._load_metadata(plate_path).get(FIELDS.IMAGE_FILES)
        if not (isinstance(image_files, list) and all(isinstance(f, str) for f in image_files)):
            raise ValueError(f"'{FIELDS.IMAGE_FILES}' must be a list of strings in {self.METADATA_FILENAME}")
        return image_files

    # Optional metadata getters
    def _get_optional_metadata_dict(self, plate_path: Union[str, Path], key: str) -> Optional[Dict[str, str]]:
        """Helper to get optional dictionary metadata."""
        value = self._load_metadata(plate_path).get(key)
        return {str(k): str(v) for k, v in value.items()} if isinstance(value, dict) else None

    def get_channel_values(self, plate_path: Union[str, Path], context: Optional[Any] = None) -> Optional[Dict[str, Optional[str]]]:
        return self._get_optional_metadata_dict(plate_path, FIELDS.CHANNELS)

    def get_well_values(self, plate_path: Union[str, Path], context: Optional[Any] = None) -> Optional[Dict[str, Optional[str]]]:
        return self._get_optional_metadata_dict(plate_path, FIELDS.WELLS)

    def get_site_values(self, plate_path: Union[str, Path], context: Optional[Any] = None) -> Optional[Dict[str, Optional[str]]]:
        return self._get_optional_metadata_dict(plate_path, FIELDS.SITES)

    def get_z_index_values(self, plate_path: Union[str, Path], context: Optional[Any] = None) -> Optional[Dict[str, Optional[str]]]:
        return self._get_optional_metadata_dict(plate_path, FIELDS.Z_INDEXES)

    def get_objective_values(self, plate_path: Union[str, Path], context: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """Get objective lens information if available."""
        return self._get_optional_metadata_dict(plate_path, FIELDS.OBJECTIVES)

    def get_plate_acquisition_datetime(self, plate_path: Union[str, Path], context: Optional[Any] = None) -> Optional[str]:
        """Get plate acquisition datetime if available."""
        return self._get_optional_metadata_str(plate_path, FIELDS.ACQUISITION_DATETIME)

    def get_plate_name(self, plate_path: Union[str, Path], context: Optional[Any] = None) -> Optional[str]:
        """Get plate name if available."""
        return self._get_optional_metadata_str(plate_path, FIELDS.PLATE_NAME)

    def _get_optional_metadata_str(self, plate_path: Union[str, Path], field: str) -> Optional[str]:
        """Helper to get optional string metadata field."""
        value = self._load_metadata(plate_path).get(field)
        return value if isinstance(value, str) and value else None

    def get_available_backends(self, input_dir: Union[str, Path]) -> Dict[str, bool]:
        """
        Get available storage backends for the input directory.

        This method resolves the plate root from the input directory,
        loads the OpenHCS metadata, and returns the available backends.

        Args:
            input_dir: Path to the input directory (may be plate root or subdirectory)

        Returns:
            Dictionary mapping backend names to availability (e.g., {"disk": True, "zarr": False})

        Raises:
            MetadataNotFoundError: If metadata file cannot be found or parsed
        """
        # Resolve plate root from input directory
        plate_root = self._resolve_plate_root(input_dir)

        # Load metadata using existing infrastructure
        metadata = self._load_metadata(plate_root)

        # Extract available backends, defaulting to empty dict if not present
        available_backends = metadata.get(FIELDS.AVAILABLE_BACKENDS, {})

        if not isinstance(available_backends, dict):
            logger.warning(f"Invalid available_backends format in metadata: {available_backends}")
            return {}

        return available_backends

    def _resolve_plate_root(self, input_dir: Union[str, Path]) -> Path:
        """
        Resolve the plate root directory from an input directory.

        The input directory may be the plate root itself or a subdirectory.
        This method walks up the directory tree to find the directory containing
        the OpenHCS metadata file.

        Args:
            input_dir: Path to resolve

        Returns:
            Path to the plate root directory

        Raises:
            MetadataNotFoundError: If no metadata file is found
        """
        current_path = Path(input_dir)

        # Walk up the directory tree looking for metadata file
        for path in [current_path] + list(current_path.parents):
            metadata_file = path / self.METADATA_FILENAME
            if self.filemanager.exists(str(metadata_file), Backend.DISK.value):
                return path

        # If not found, raise an error
        raise MetadataNotFoundError(
            f"Could not find {self.METADATA_FILENAME} in {input_dir} or any parent directory"
        )

    def update_available_backends(self, plate_path: Union[str, Path], available_backends: Dict[str, bool]) -> None:
        """Update available storage backends in metadata and save to disk."""
        metadata_file_path = get_metadata_path(plate_path)

        try:
            self.atomic_writer.update_available_backends(metadata_file_path, available_backends)
            # Clear cache to force reload on next access
            self._metadata_cache = None
            self._plate_path_cache = None
            logger.info(f"Updated available backends to {available_backends} in {metadata_file_path}")
        except MetadataWriteError as e:
            raise ValueError(f"Failed to update available backends: {e}") from e


@dataclass(frozen=True)
class OpenHCSMetadata:
    """
    Declarative OpenHCS metadata structure.

    Fail-loud: All fields are required, no defaults, no fallbacks.
    """
    microscope_handler_name: str
    source_filename_parser_name: str
    grid_dimensions: List[int]
    pixel_size: float
    image_files: List[str]
    channels: Optional[Dict[str, str]]
    wells: Optional[Dict[str, str]]
    sites: Optional[Dict[str, str]]
    z_indexes: Optional[Dict[str, str]]
    available_backends: Dict[str, bool]
    main: Optional[bool] = None  # Indicates if this subdirectory is the primary/input subdirectory


@dataclass(frozen=True)
class SubdirectoryKeyedMetadata:
    """
    Subdirectory-keyed metadata structure for OpenHCS.

    Organizes metadata by subdirectory to prevent conflicts when multiple
    steps write to the same plate folder with different subdirectories.

    Structure: {subdirectory_name: OpenHCSMetadata}
    """
    subdirectories: Dict[str, OpenHCSMetadata]

    def get_subdirectory_metadata(self, sub_dir: str) -> Optional[OpenHCSMetadata]:
        """Get metadata for specific subdirectory."""
        return self.subdirectories.get(sub_dir)

    def add_subdirectory_metadata(self, sub_dir: str, metadata: OpenHCSMetadata) -> 'SubdirectoryKeyedMetadata':
        """Add or update metadata for subdirectory (immutable operation)."""
        new_subdirs = {**self.subdirectories, sub_dir: metadata}
        return SubdirectoryKeyedMetadata(subdirectories=new_subdirs)

    @classmethod
    def from_single_metadata(cls, sub_dir: str, metadata: OpenHCSMetadata) -> 'SubdirectoryKeyedMetadata':
        """Create from single OpenHCSMetadata (migration helper)."""
        return cls(subdirectories={sub_dir: metadata})

    @classmethod
    def from_legacy_dict(cls, legacy_dict: Dict[str, Any], default_sub_dir: str = FIELDS.DEFAULT_SUBDIRECTORY_LEGACY) -> 'SubdirectoryKeyedMetadata':
        """Create from legacy single-subdirectory metadata dict."""
        return cls.from_single_metadata(default_sub_dir, OpenHCSMetadata(**legacy_dict))


class OpenHCSMetadataGenerator:
    """
    Generator for OpenHCS metadata files.

    Handles creation of openhcs_metadata.json files for processed plates,
    extracting information from processing context and output directories.

    Design principle: Generate metadata that accurately reflects what exists on disk
    after processing, not what was originally intended or what the source contained.
    """

    def __init__(self, filemanager: FileManager):
        """
        Initialize the metadata generator.

        Args:
            filemanager: FileManager instance for file operations
        """
        self.filemanager = filemanager
        self.atomic_writer = AtomicMetadataWriter()
        self.logger = logging.getLogger(__name__)

    def create_metadata(
        self,
        context: 'ProcessingContext',
        output_dir: str,
        write_backend: str,
        is_main: bool = False,
        plate_root: str = None,
        sub_dir: str = None
    ) -> None:
        """Create or update subdirectory-keyed OpenHCS metadata file."""
        plate_root_path = Path(plate_root)
        metadata_path = get_metadata_path(plate_root_path)

        current_metadata = self._extract_metadata_from_disk_state(context, output_dir, write_backend, is_main, sub_dir)
        metadata_dict = asdict(current_metadata)

        self.atomic_writer.update_subdirectory_metadata(metadata_path, sub_dir, metadata_dict)



    def _extract_metadata_from_disk_state(self, context: 'ProcessingContext', output_dir: str, write_backend: str, is_main: bool, sub_dir: str) -> OpenHCSMetadata:
        """Extract metadata reflecting current disk state after processing."""
        handler = context.microscope_handler
        cache = context.metadata_cache or {}

        actual_files = self.filemanager.list_image_files(output_dir, write_backend)
        relative_files = [f"{sub_dir}/{Path(f).name}" for f in actual_files]

        return OpenHCSMetadata(
            microscope_handler_name=handler.microscope_type,
            source_filename_parser_name=handler.parser.__class__.__name__,
            grid_dimensions=handler.metadata_handler._get_with_fallback('get_grid_dimensions', context.input_dir),
            pixel_size=handler.metadata_handler._get_with_fallback('get_pixel_size', context.input_dir),
            image_files=relative_files,
            channels=cache.get(GroupBy.CHANNEL),
            wells=cache.get(GroupBy.WELL),
            sites=cache.get(GroupBy.SITE),
            z_indexes=cache.get(GroupBy.Z_INDEX),
            available_backends={write_backend: True},
            main=is_main if is_main else None
        )





from openhcs.microscopes.microscope_base import MicroscopeHandler
from openhcs.microscopes.microscope_interfaces import FilenameParser


class OpenHCSMicroscopeHandler(MicroscopeHandler):
    """
    MicroscopeHandler for OpenHCS pre-processed format.

    This handler reads plates that have been standardized, with metadata
    provided in an 'openhcs_metadata.json' file. It dynamically loads the
    appropriate FilenameParser based on the metadata.
    """

    # Class attributes for automatic registration
    _microscope_type = FIELDS.MICROSCOPE_TYPE  # Override automatic naming
    _metadata_handler_class = None  # Set after class definition

    def __init__(self, filemanager: FileManager, pattern_format: Optional[str] = None):
        """
        Initialize the OpenHCSMicroscopeHandler.

        Args:
            filemanager: FileManager instance for file operations.
            pattern_format: Optional pattern format string, passed to dynamically loaded parser.
        """
        self.filemanager = filemanager
        self.metadata_handler = OpenHCSMetadataHandler(filemanager)
        self._parser: Optional[FilenameParser] = None
        self.plate_folder: Optional[Path] = None # Will be set by factory or post_workspace
        self.pattern_format = pattern_format # Store for parser instantiation

        # Initialize super with a None parser. The actual parser is loaded dynamically.
        # The `parser` property will handle on-demand loading.
        super().__init__(parser=None, metadata_handler=self.metadata_handler)

    def _load_and_get_parser(self) -> FilenameParser:
        """
        Ensures the dynamic filename parser is loaded based on metadata from plate_folder.
        This method requires self.plate_folder to be set.
        """
        if self._parser is None:
            if self.plate_folder is None:
                raise RuntimeError(
                    "OpenHCSHandler: plate_folder not set. Cannot determine and load the source filename parser."
                )

            parser_name = self.metadata_handler.get_source_filename_parser_name(self.plate_folder)
            available_parsers = _get_available_filename_parsers()
            ParserClass = available_parsers.get(parser_name)

            if not ParserClass:
                raise ValueError(
                    f"Unknown or unsupported filename parser '{parser_name}' specified in "
                    f"{OpenHCSMetadataHandler.METADATA_FILENAME} for plate {self.plate_folder}. "
                    f"Available parsers: {list(available_parsers.keys())}"
                )

            try:
                # Attempt to instantiate with filemanager and pattern_format
                self._parser = ParserClass(filemanager=self.filemanager, pattern_format=self.pattern_format)
                logger.info(f"OpenHCSHandler for plate {self.plate_folder} loaded source filename parser: {parser_name} with filemanager and pattern_format.")
            except TypeError:
                try:
                    # Attempt with filemanager only
                    self._parser = ParserClass(filemanager=self.filemanager)
                    logger.info(f"OpenHCSHandler for plate {self.plate_folder} loaded source filename parser: {parser_name} with filemanager.")
                except TypeError:
                    # Attempt with default constructor
                    self._parser = ParserClass()
                    logger.info(f"OpenHCSHandler for plate {self.plate_folder} loaded source filename parser: {parser_name} with default constructor.")

        return self._parser

    @property
    def parser(self) -> FilenameParser:
        """
        Provides the dynamically loaded FilenameParser.
        The actual parser is determined from the 'openhcs_metadata.json' file.
        Requires `self.plate_folder` to be set prior to first access.
        """
        # If plate_folder is not set here, it means it wasn't set by the factory
        # nor by a method like post_workspace before parser access.
        if self.plate_folder is None:
             # This situation should ideally be avoided by ensuring plate_folder is set appropriately.
            raise RuntimeError("OpenHCSHandler: plate_folder must be set before accessing the parser property.")

        return self._load_and_get_parser()

    @parser.setter
    def parser(self, value: Optional[FilenameParser]):
        """
        Allows setting the parser instance. Used by base class __init__ if it attempts to set it,
        though our dynamic loading means we primarily manage it internally.
        """
        # If the base class __init__ tries to set it (e.g. to None as we passed),
        # this setter will be called. We want our dynamic loading to take precedence.
        # If an actual parser is passed, we could use it, but it would override dynamic logic.
        # For now, if None is passed (from our super call), _parser remains None until dynamically loaded.
        # If a specific parser is passed, it will be set.
        if value is not None:
            logger.debug(f"OpenHCSMicroscopeHandler.parser being explicitly set to: {type(value).__name__}")
        self._parser = value


    @property
    def common_dirs(self) -> List[str]:
        """
        OpenHCS format expects images in the root of the plate folder.
        No common subdirectories are applicable.
        """
        return []

    @property
    def microscope_type(self) -> str:
        """Microscope type identifier (for interface enforcement only)."""
        return FIELDS.MICROSCOPE_TYPE

    @property
    def metadata_handler_class(self) -> Type[MetadataHandler]:
        """Metadata handler class (for interface enforcement only)."""
        return OpenHCSMetadataHandler

    @property
    def compatible_backends(self) -> List[Backend]:
        """
        OpenHCS is compatible with ZARR (preferred) and DISK (fallback) backends.

        ZARR: Advanced chunked storage for large datasets (preferred)
        DISK: Standard file operations for compatibility (fallback)
        """
        return [Backend.ZARR, Backend.DISK]

    def get_available_backends(self, plate_path: Union[str, Path]) -> List[Backend]:
        """
        Get available storage backends for OpenHCS plates.

        OpenHCS plates can support multiple backends based on what actually exists on disk.
        This method checks the metadata to see what backends are actually available.
        """
        try:
            # Get available backends from metadata as Dict[str, bool]
            available_backends_dict = self.metadata_handler.get_available_backends(plate_path)

            # Convert to List[Backend] by filtering compatible backends that are available
            available_backends = []
            for backend_enum in self.compatible_backends:
                backend_name = backend_enum.value
                if available_backends_dict.get(backend_name, False):
                    available_backends.append(backend_enum)

            # If no backends are available from metadata, fall back to compatible backends
            # This handles cases where metadata might not have the available_backends field
            if not available_backends:
                logger.warning(f"No available backends found in metadata for {plate_path}, using all compatible backends")
                return self.compatible_backends

            return available_backends

        except Exception as e:
            logger.warning(f"Failed to get available backends from metadata for {plate_path}: {e}")
            # Fall back to all compatible backends if metadata reading fails
            return self.compatible_backends

    def initialize_workspace(self, plate_path: Path, workspace_path: Optional[Path], filemanager: FileManager) -> Path:
        """
        OpenHCS format doesn't need workspace - determines the correct input subdirectory from metadata.

        Args:
            plate_path: Path to the original plate directory
            workspace_path: Optional workspace path (ignored for OpenHCS)
            filemanager: FileManager instance for file operations

        Returns:
            Path to the main subdirectory containing input images (e.g., plate_path/images)
        """
        logger.info(f"OpenHCS format: Determining input subdirectory from metadata in {plate_path}")

        # Set plate_folder for this handler
        self.plate_folder = plate_path
        logger.debug(f"OpenHCSHandler: plate_folder set to {self.plate_folder}")

        # Determine the main subdirectory from metadata - fail-loud on errors
        main_subdir = self.metadata_handler.determine_main_subdirectory(plate_path)
        input_dir = plate_path / main_subdir

        # Verify the subdirectory exists - fail-loud if missing
        if not filemanager.is_dir(str(input_dir), Backend.DISK.value):
            raise FileNotFoundError(
                f"Main subdirectory '{main_subdir}' does not exist at {input_dir}. "
                f"Expected directory structure: {plate_path}/{main_subdir}/"
            )

        logger.info(f"OpenHCS input directory determined: {input_dir} (subdirectory: {main_subdir})")
        return input_dir

    def _prepare_workspace(self, workspace_path: Path, filemanager: FileManager) -> Path:
        """
        OpenHCS format assumes the workspace is already prepared (e.g., flat structure).
        This method is a no-op.
        Args:
            workspace_path: Path to the symlinked workspace.
            filemanager: FileManager instance for file operations.
        Returns:
            The original workspace_path.
        """
        logger.info(f"OpenHCSHandler._prepare_workspace: No preparation needed for {workspace_path} as it's pre-processed.")
        # Ensure plate_folder is set if this is the first relevant operation knowing the path
        if self.plate_folder is None:
            self.plate_folder = Path(workspace_path)
            logger.debug(f"OpenHCSHandler: plate_folder set to {self.plate_folder} during _prepare_workspace.")
        return workspace_path

    def post_workspace(self, workspace_path: Union[str, Path], filemanager: FileManager, width: int = 3):
        """
        Hook called after workspace symlink creation.
        For OpenHCS, this ensures the plate_folder is set (if not already) which allows
        the parser to be loaded using this workspace_path. It then calls the base
        implementation which handles filename normalization using the loaded parser.
        """
        current_plate_folder = Path(workspace_path)
        if self.plate_folder is None:
            logger.info(f"OpenHCSHandler.post_workspace: Setting plate_folder to {current_plate_folder}.")
            self.plate_folder = current_plate_folder
            self._parser = None # Reset parser if plate_folder changes or is set for the first time
        elif self.plate_folder != current_plate_folder:
            logger.warning(
                f"OpenHCSHandler.post_workspace: plate_folder was {self.plate_folder}, "
                f"now processing {current_plate_folder}. Re-initializing parser."
            )
            self.plate_folder = current_plate_folder
            self._parser = None # Force re-initialization for the new path

        # Accessing self.parser here will trigger _load_and_get_parser() if not already loaded
        _ = self.parser

        logger.info(f"OpenHCSHandler (plate: {self.plate_folder}): Files are expected to be pre-normalized. "
                     "Superclass post_workspace will run with the dynamically loaded parser.")
        return super().post_workspace(workspace_path, filemanager, width)

    # The following methods from MicroscopeHandler delegate to `self.parser`.
    # The `parser` property will ensure the correct, dynamically loaded parser is used.
    # No explicit override is needed for them unless special behavior for OpenHCS is required
    # beyond what the dynamically loaded original parser provides.
    # - parse_filename(self, filename: str)
    # - construct_filename(self, well: str, ...)
    # - auto_detect_patterns(self, folder_path: Union[str, Path], ...)
    # - path_list_from_pattern(self, directory: Union[str, Path], ...)

    # Metadata handling methods are delegated to `self.metadata_handler` by the base class.
    # - find_metadata_file(self, plate_path: Union[str, Path])
    # - get_grid_dimensions(self, plate_path: Union[str, Path])
    # - get_pixel_size(self, plate_path: Union[str, Path])
    # These will use our OpenHCSMetadataHandler correctly.


# Set metadata handler class after class definition for automatic registration
from openhcs.microscopes.microscope_base import register_metadata_handler
OpenHCSMicroscopeHandler._metadata_handler_class = OpenHCSMetadataHandler
register_metadata_handler(OpenHCSMicroscopeHandler, OpenHCSMetadataHandler)
