"""
OpenHCS microscope handler implementation for openhcs.

This module provides the OpenHCSMicroscopeHandler, which reads plates
that have been pre-processed and standardized into the OpenHCS format.
The metadata for such plates is defined in an 'openhcs_metadata.json' file.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Type

from openhcs.constants.constants import Backend
from openhcs.io.exceptions import MetadataNotFoundError
from openhcs.io.filemanager import FileManager
from openhcs.microscopes.microscope_interfaces_base import MetadataHandler
from openhcs.microscopes.imagexpress import ImageXpressFilenameParser # Placeholder for dynamic loading
from openhcs.microscopes.opera_phenix import OperaPhenixFilenameParser # Placeholder for dynamic loading

logger = logging.getLogger(__name__)

# Import known filename parsers for dynamic loading
from openhcs.microscopes.imagexpress import ImageXpressFilenameParser
from openhcs.microscopes.opera_phenix import OperaPhenixFilenameParser
# Import other FilenameParser implementations here if they exist and are needed.

AVAILABLE_FILENAME_PARSERS = {
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
    METADATA_FILENAME = "openhcs_metadata.json"

    def __init__(self, filemanager: FileManager):
        """
        Initialize the metadata handler.

        Args:
            filemanager: FileManager instance for file operations.
        """
        super().__init__()
        self.filemanager = filemanager
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
        if not metadata_file_path or not self.filemanager.exists(str(metadata_file_path), 'disk'):
            raise MetadataNotFoundError(
                f"Metadata file '{self.METADATA_FILENAME}' not found in {plate_path}."
            )

        try:
            # Use filemanager to load file content - returns string content
            content = self.filemanager.load(str(metadata_file_path), Backend.DISK.value)
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            self._metadata_cache = json.loads(content)
            self._plate_path_cache = current_path
            return self._metadata_cache
        except json.JSONDecodeError as e:
            raise MetadataNotFoundError(
                f"Error decoding JSON from '{metadata_file_path}': {e}"
            ) from e
        except Exception as e:
            raise MetadataNotFoundError(
                f"Could not read or parse metadata file '{metadata_file_path}': {e}"
            ) from e

    def find_metadata_file(self, plate_path: Union[str, Path],
                           context: Optional[Any] = None) -> Optional[Path]:
        """
        Find the OpenHCS JSON metadata file.

        Args:
            plate_path: Path to the plate folder.
            context: Optional context (not used).

        Returns:
            Path to the 'openhcs_metadata.json' file if found, else None.
        """
        plate_p = Path(plate_path)
        if not self.filemanager.is_dir(str(plate_p), 'disk'):
            logger.warning(f"Plate path {plate_p} is not a directory.")
            return None

        expected_file = plate_p / self.METADATA_FILENAME
        if self.filemanager.exists(str(expected_file), 'disk') and self.filemanager.is_file(str(expected_file), 'disk'):
            return expected_file

        logger.debug(f"Metadata file {self.METADATA_FILENAME} not found directly in {plate_path}.")

        # Attempt to find it recursively, though it's expected to be in the root.
        # This uses the filemanager's find_file_recursive method.
        try:
            # Use correct signature: find_file_recursive(directory, filename, backend)
            # Use disk backend for metadata file search
            found_files = self.filemanager.find_file_recursive(plate_p, self.METADATA_FILENAME, 'disk')
            if found_files:
                # find_file_recursive might return a list or a single path string/Path
                if isinstance(found_files, list):
                    if not found_files:
                        return None
                    # Prioritize file in root if multiple found (though unlikely for this specific filename)
                    for f_path_str in found_files:
                        f_path = Path(f_path_str)
                        if f_path.name == self.METADATA_FILENAME and f_path.parent == plate_p:
                            return f_path
                    return Path(found_files[0]) # Return the first one found
                else: # Assuming it's a single path string or Path object
                    return Path(found_files)
        except Exception as e:
            logger.error(f"Error while searching for {self.METADATA_FILENAME} in {plate_path} using filemanager: {e}")

        return None


    def get_grid_dimensions(self, plate_path: Union[str, Path],
                             context: Optional[Any] = None) -> Tuple[int, int]:
        """
        Get grid dimensions from the OpenHCS JSON metadata.

        Args:
            plate_path: Path to the plate folder.
            context: Optional context (not used).

        Returns:
            Tuple (rows, cols).
        """
        metadata = self._load_metadata(plate_path)
        dims = metadata.get("grid_dimensions")
        if not isinstance(dims, list) or len(dims) != 2 or \
           not all(isinstance(d, int) for d in dims):
            raise ValueError(
                f"'grid_dimensions' is missing, malformed, or not a list of two integers in {self.METADATA_FILENAME}"
            )
        return tuple(dims)

    def get_pixel_size(self, plate_path: Union[str, Path],
                       context: Optional[Any] = None) -> float:
        """
        Get pixel size from the OpenHCS JSON metadata.

        Args:
            plate_path: Path to the plate folder.
            context: Optional context (not used).

        Returns:
            Pixel size in micrometers.
        """
        metadata = self._load_metadata(plate_path)
        pixel_size = metadata.get("pixel_size")
        if not isinstance(pixel_size, (float, int)):
            raise ValueError(
                f"'pixel_size' is missing or not a number in {self.METADATA_FILENAME}"
            )
        return float(pixel_size)

    def get_source_filename_parser_name(self, plate_path: Union[str, Path]) -> str:
        """
        Get the name of the source filename parser from the OpenHCS JSON metadata.

        Args:
            plate_path: Path to the plate folder.

        Returns:
            The class name of the source filename parser.
        """
        metadata = self._load_metadata(plate_path)
        parser_name = metadata.get("source_filename_parser_name")
        if not isinstance(parser_name, str) or not parser_name:
            raise ValueError(
                f"'source_filename_parser_name' is missing or not a string in {self.METADATA_FILENAME}"
            )
        return parser_name

    def get_image_files(self, plate_path: Union[str, Path]) -> List[str]:
        """
        Get the list of image files from the OpenHCS JSON metadata.

        Args:
            plate_path: Path to the plate folder.

        Returns:
            A list of image filenames.
        """
        metadata = self._load_metadata(plate_path)
        image_files = metadata.get("image_files")
        if not isinstance(image_files, list) or not all(isinstance(f, str) for f in image_files):
            raise ValueError(
                f"'image_files' is missing or not a list of strings in {self.METADATA_FILENAME}"
            )
        return image_files

    # Optional metadata getters
    def _get_optional_metadata_dict(self, plate_path: Union[str, Path], key: str) -> Optional[Dict[str, str]]:
        """Helper to get optional dictionary metadata."""
        metadata = self._load_metadata(plate_path)
        value = metadata.get(key)
        if value is None:
            return None
        if not isinstance(value, dict):
            logger.warning(f"Optional metadata '{key}' is not a dictionary in {self.METADATA_FILENAME}. Ignoring.")
            return None
        # Ensure keys and values are strings, as expected by some interfaces, though JSON naturally supports string keys.
        return {str(k): str(v) for k, v in value.items()}

    def get_channel_values(self, plate_path: Union[str, Path],
                           context: Optional[Any] = None) -> Optional[Dict[str, Optional[str]]]:
        return self._get_optional_metadata_dict(plate_path, "channels")

    def get_well_values(self, plate_path: Union[str, Path],
                        context: Optional[Any] = None) -> Optional[Dict[str, Optional[str]]]:
        return self._get_optional_metadata_dict(plate_path, "wells")

    def get_site_values(self, plate_path: Union[str, Path],
                        context: Optional[Any] = None) -> Optional[Dict[str, Optional[str]]]:
        return self._get_optional_metadata_dict(plate_path, "sites")

    def get_z_index_values(self, plate_path: Union[str, Path],
                           context: Optional[Any] = None) -> Optional[Dict[str, Optional[str]]]:
        return self._get_optional_metadata_dict(plate_path, "z_indexes")

    def get_objective_values(self, plate_path: Union[str, Path],
                             context: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieves objective lens information if available in the metadata.
        The structure within the JSON for this is not strictly defined by the initial plan,
        so this is a placeholder implementation.
        """
        metadata = self._load_metadata(plate_path)
        # Assuming 'objectives' might be a key in the JSON if this data is stored
        objectives_data = metadata.get("objectives")
        if objectives_data and isinstance(objectives_data, dict):
            return objectives_data
        logger.debug("No 'objectives' data found in OpenHCS metadata.")
        return None

    def get_plate_acquisition_datetime(self, plate_path: Union[str, Path],
                                       context: Optional[Any] = None) -> Optional[str]:
        """
        Retrieves plate acquisition date/time if available.
        The JSON field for this is not strictly defined by the initial plan.
        """
        metadata = self._load_metadata(plate_path)
        # Assuming 'acquisition_datetime' might be a key
        acq_datetime = metadata.get("acquisition_datetime")
        if acq_datetime and isinstance(acq_datetime, str):
            return acq_datetime
        logger.debug("No 'acquisition_datetime' data found in OpenHCS metadata.")
        return None

    def get_plate_name(self, plate_path: Union[str, Path],
                       context: Optional[Any] = None) -> Optional[str]:
        """
        Retrieves plate name if available.
        The JSON field for this is not strictly defined by the initial plan.
        """
        metadata = self._load_metadata(plate_path)
        # Assuming 'plate_name' might be a key
        plate_name = metadata.get("plate_name")
        if plate_name and isinstance(plate_name, str):
            return plate_name
        logger.debug("No 'plate_name' data found in OpenHCS metadata.")
        return None

    def get_available_backends(self, plate_path: Union[str, Path]) -> Dict[str, bool]:
        """
        Get available storage backends from metadata in priority order.

        Args:
            plate_path: Path to the plate folder.

        Returns:
            Ordered dictionary mapping backend names to availability flags.
            Order represents selection priority (first available backend is used).
            Defaults to {"zarr": False, "disk": True} if not specified.
        """
        metadata = self._load_metadata(plate_path)
        return metadata.get("available_backends", {"zarr": False, "disk": True})

    def update_available_backends(self, plate_path: Union[str, Path], available_backends: Dict[str, bool]) -> None:
        """
        Update available storage backends in metadata and save to disk.

        Args:
            plate_path: Path to the plate folder.
            available_backends: Ordered dict mapping backend names to availability flags.
        """
        # Load current metadata
        metadata = self._load_metadata(plate_path)

        # Update the available backends
        metadata["available_backends"] = available_backends

        # Save back to file
        metadata_file_path = Path(plate_path) / self.METADATA_FILENAME
        content = json.dumps(metadata, indent=2)
        self.filemanager.save(content, str(metadata_file_path), Backend.DISK.value)

        # Update cache
        self._metadata_cache = metadata
        logger.info(f"Updated available backends to {available_backends} in {metadata_file_path}")
from openhcs.microscopes.microscope_base import MicroscopeHandler
from openhcs.microscopes.microscope_interfaces_base import FilenameParser


class OpenHCSMicroscopeHandler(MicroscopeHandler):
    """
    MicroscopeHandler for OpenHCS pre-processed format.

    This handler reads plates that have been standardized, with metadata
    provided in an 'openhcs_metadata.json' file. It dynamically loads the
    appropriate FilenameParser based on the metadata.
    """

    # Class attributes for automatic registration
    _microscope_type = 'openhcsdata'  # Override automatic naming
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
            ParserClass = AVAILABLE_FILENAME_PARSERS.get(parser_name)

            if not ParserClass:
                raise ValueError(
                    f"Unknown or unsupported filename parser '{parser_name}' specified in "
                    f"{OpenHCSMetadataHandler.METADATA_FILENAME} for plate {self.plate_folder}. "
                    f"Available parsers: {list(AVAILABLE_FILENAME_PARSERS.keys())}"
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
        return 'openhcsdata'

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
        Get available storage backends from metadata.

        Only returns backends that this handler supports AND are available in metadata.
        """
        backend_dict = self.metadata_handler.get_available_backends(plate_path)
        available_backends = []
        for backend in self.compatible_backends:
            if backend_dict.get(backend.value, False):
                available_backends.append(backend)
        return available_backends

    def initialize_workspace(self, plate_path: Path, workspace_path: Optional[Path], filemanager: FileManager) -> Path:
        """
        OpenHCS format doesn't need workspace - images are already processed and ready.

        Args:
            plate_path: Path to the original plate directory
            workspace_path: Optional workspace path (ignored for OpenHCS)
            filemanager: FileManager instance for file operations

        Returns:
            The plate path directly (no workspace needed)
        """
        logger.info(f"OpenHCS format: Using plate directory directly {plate_path} (no workspace needed)")

        # Set plate_folder for this handler
        self.plate_folder = plate_path
        logger.debug(f"OpenHCSHandler: plate_folder set to {self.plate_folder}")

        return plate_path

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
