"""
OMERO microscope implementations for openhcs.

This module provides concrete implementations of FilenameParser and MetadataHandler
for OMERO microscopes using native OMERO metadata.
"""

import logging
import re
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Tuple, Type, Union

from openhcs.constants.constants import Backend, Microscope
from polystore.exceptions import MetadataNotFoundError
from polystore.filemanager import FileManager
from openhcs.microscopes.microscope_base import (
    MicroscopeHandler,
    MicroscopeSourceSelectionRole,
)
from openhcs.microscopes.microscope_interfaces import (
    FilenameParseResult,
    FilenameParser,
    MetadataHandler,
)

logger = logging.getLogger(__name__)


class OMEROMetadataHandler(MetadataHandler):
    GRID_DIMENSIONS_METADATA_KEY: ClassVar[str] = "openhcs.grid_dimensions"

    """
    Metadata handler that queries OMERO API for native metadata with caching.

    Does NOT read OpenHCS metadata files - uses OMERO's native metadata.
    Caches metadata per plate to avoid repeated OMERO queries.
    
    Retrieves OMERO connection from backend registry via filemanager (Option B pattern).
    """

    def __init__(self, filemanager: FileManager):
        super().__init__()
        self.filemanager = filemanager
        self._metadata_cache: Dict[int, Dict[str, Dict[int, str]]] = {}  # plate_id → metadata

    def _get_omero_conn(self):
        """
        Get OMERO connection from backend registry.

        Retrieves the connection from the OMERO backend instance in the registry,
        using the backend's _get_connection() method which handles lazy connection
        management for multiprocessing compatibility.

        Returns:
            BlitzGateway connection instance

        Raises:
            RuntimeError: If OMERO backend cannot provide a connection
        """
        # Get OMERO backend from registry (via filemanager)
        omero_backend = self.filemanager.registry[Backend.OMERO_LOCAL.value]

        # Use the backend's connection retrieval method
        # This handles lazy connection management and multiprocessing
        try:
            conn = omero_backend._get_connection()
            return conn
        except Exception as e:
            raise RuntimeError(
                f"Failed to get OMERO connection from backend: {e}. "
                "Ensure OMERO backend is properly initialized with connection."
            ) from e

    def _load_plate_metadata(self, plate_id: int) -> Dict[str, Dict[int, str]]:
        """
        Load all metadata for a plate in one OMERO query (cached).

        Queries OMERO once per plate and caches results.
        Subsequent calls return cached data.
        """
        if plate_id in self._metadata_cache:
            return self._metadata_cache[plate_id]

        # Get connection from backend registry
        conn = self._get_omero_conn()
        
        plate = conn.getObject("Plate", plate_id)
        if not plate:
            raise ValueError(f"OMERO Plate not found: {plate_id}")

        # Query OMERO once for all metadata
        metadata = {}

        # Get metadata from all wells to ensure complete coverage
        # (different wells might have different dimensions)
        all_channels = {}
        max_z = 0
        max_t = 0

        for well in plate.listChildren():
            well_sample = well.getWellSample(0)
            if well_sample:
                image = well_sample.getImage()

                # Collect channel names
                for i, channel in enumerate(image.getChannels()):
                    channel_idx = i + 1
                    if channel_idx not in all_channels:
                        all_channels[channel_idx] = channel.getLabel() or f"Channel {channel_idx}"

                # Track max dimensions
                max_z = max(max_z, image.getSizeZ())
                max_t = max(max_t, image.getSizeT())

        # Build metadata dict
        metadata['channel'] = all_channels
        metadata['z_index'] = {z + 1: f"Z{z + 1}" for z in range(max_z)}
        metadata['timepoint'] = {t + 1: f"T{t + 1}" for t in range(max_t)}

        # Cache it
        self._metadata_cache[plate_id] = metadata
        return metadata

    def find_metadata_file(self, plate_path: Union[str, Path]) -> Path:
        """
        OMERO doesn't use metadata files, but detects based on /omero/ path pattern.

        Returns the plate_path itself if it matches the OMERO virtual path pattern,
        otherwise raises MetadataNotFoundError.
        """
        plate_path = Path(plate_path)
        # OMERO plates use virtual paths like /omero/plate_123
        if str(plate_path).startswith('/omero/plate_'):
            return plate_path
        raise MetadataNotFoundError(
            f"OMERO plate path must use /omero/plate_<id>, got {plate_path}"
        )

    def _extract_plate_id(self, plate_path: Union[str, Path, int]) -> int:
        """
        Extract plate_id from various path formats.

        Handles:
        - int: plate_id directly
        - Path('/omero/plate_57'): extracts 57 from 'plate_57'
        - Path('/omero/plate_57_outputs'): extracts 57 from 'plate_57_outputs'

        Args:
            plate_path: Plate identifier (int or Path)

        Returns:
            Plate ID as integer

        Raises:
            ValueError: If path format is invalid
        """
        if isinstance(plate_path, int):
            return plate_path

        # Extract from path: /omero/plate_57 or /omero/plate_57_outputs
        path_str = str(Path(plate_path).name)  # Get just the filename part

        # Match 'plate_<id>' or 'plate_<id>_<suffix>'
        match = re.match(r'plate_(\d+)', path_str)
        if not match:
            raise ValueError(f"Invalid OMERO path format: {plate_path}. Expected /omero/plate_<id> or /omero/plate_<id>_<suffix>")

        return int(match.group(1))

    def get_channel_values(self, plate_path: Union[str, Path, int]) -> Dict[int, str]:
        """Get channel metadata (cached)."""
        plate_id = self._extract_plate_id(plate_path)
        metadata = self._load_plate_metadata(plate_id)
        return metadata['channel']

    def get_z_index_values(self, plate_path: Union[str, Path, int]) -> Dict[int, str]:
        """Get Z-index metadata (cached)."""
        plate_id = self._extract_plate_id(plate_path)
        metadata = self._load_plate_metadata(plate_id)
        return metadata['z_index']

    def get_timepoint_values(self, plate_path: Union[str, Path, int]) -> Dict[int, str]:
        """Get timepoint metadata (cached)."""
        plate_id = self._extract_plate_id(plate_path)
        metadata = self._load_plate_metadata(plate_id)
        return metadata['timepoint']

    # Other component methods return empty dicts (not applicable for OMERO)
    def get_site_values(self, plate_path: Union[str, Path, int]) -> Dict[int, str]:
        return {}

    def get_well_values(self, plate_path: Union[str, Path, int]) -> Dict[str, str]:
        """
        Get well metadata (cached).
        
        Raises if the OMERO plate cannot be loaded.
        """
        plate_id = self._extract_plate_id(plate_path)
        conn = self._get_omero_conn()
        
        plate = conn.getObject("Plate", plate_id)
        if not plate:
            raise ValueError(f"OMERO Plate not found: {plate_id}")
        
        # Extract well IDs from the plate
        well_values = {}
        for well in plate.listChildren():
            # Get well label (e.g., "A01", "B02")
            well_label = f"{chr(ord('A') + well.getRow())}{well.getColumn() + 1:02d}"
            well_values[well_label] = well_label
        
        return well_values

    def parse_metadata(self, plate_path: Union[str, Path, int]) -> Dict[str, Dict[str, Optional[str]]]:
        """
        Parse all metadata for OMERO plate.
        
        OMERO doesn't use metadata files - it queries OMERO API directly.
        This implementation returns metadata for all components that OMERO supports.
        
        Raises if the OMERO plate cannot be loaded.
        """
        # Use the base class's parse_metadata method which dynamically
        # calls the appropriate get_*_values methods
        return super().parse_metadata(plate_path)

    def get_grid_dimensions(self, plate_path: Union[str, Path, int]) -> Tuple[int, int]:
        """
        Extract grid dimensions from OMERO plate metadata.

        Grid dimensions should be stored in the plate's MapAnnotation under
        the key declared by ``GRID_DIMENSIONS_METADATA_KEY`` as ``rows,cols``.

        Returns:
            Tuple of (rows, cols) representing the grid dimensions
        """
        plate_id = self._extract_plate_id(plate_path)
        conn = self._get_omero_conn()
        plate = conn.getObject("Plate", plate_id)

        if not plate:
            raise ValueError(f"OMERO Plate not found: {plate_id}")

        import omero.model  # Lazy import - only needed when OMERO is actually used

        for ann in plate.listAnnotations():
            if ann.OMERO_TYPE != omero.model.MapAnnotationI:
                continue
            for nv in ann.getMapValue():
                if nv.name == self.GRID_DIMENSIONS_METADATA_KEY:
                    rows, cols = map(int, nv.value.split(','))
                    logger.info(
                        "Found grid_dimensions (%s, %s) in OMERO metadata",
                        rows,
                        cols,
                    )
                    return (rows, cols)

        raise ValueError(
            f"OMERO Plate {plate_id} metadata does not declare "
            f"{self.GRID_DIMENSIONS_METADATA_KEY}."
        )

    def get_pixel_size(self, plate_path: Union[str, Path, int]) -> float:
        """
        Get pixel size from OMERO image metadata.

        Queries the first image in the plate for pixel size.
        """
        plate_id = self._extract_plate_id(plate_path)
        conn = self._get_omero_conn()
        plate = conn.getObject("Plate", plate_id)

        if not plate:
            raise ValueError(f"OMERO Plate not found: {plate_id}")

        for well in plate.listChildren():
            well_sample = well.getWellSample(0)
            if well_sample:
                image = well_sample.getImage()
                pixels = image.getPrimaryPixels()
                pixel_size_x = pixels.getPhysicalSizeX()
                if pixel_size_x:
                    return float(pixel_size_x)
                break

        raise ValueError(f"OMERO Plate {plate_id} does not declare pixel size.")

    def get_image_files(self, plate_path: Union[str, Path, int], all_subdirs: bool = False) -> List[str]:
        """
        Get list of virtual filenames from OMERO backend.

        Delegates to OMEROLocalBackend.list_files() to generate virtual filenames.

        Args:
            plate_path: Path to the plate folder or plate ID
            all_subdirs: Unused for OMERO (no subdirectories), kept for interface compatibility
        """
        plate_id = plate_path if isinstance(plate_path, int) else int(Path(plate_path).name)

        # Get OMERO backend from registry
        omero_backend = self.filemanager.registry[Backend.OMERO_LOCAL.value]

        # Call list_files with plate_id to get virtual filenames
        virtual_files = omero_backend.list_files(str(plate_id), plate_id=plate_id)

        # Return just the basenames (they're already basenames from backend)
        return [Path(f).name for f in virtual_files]


class OMEROFilenameParser(FilenameParser):
    """
    Parser for OMERO virtual filenames.

    OMERO backend generates filenames in standard format with ALL components:
    A01_s001_w1_z001_t001.tif

    This is compatible with ImageXpress format, but OMERO always includes
    all components (well, site, channel, z_index, timepoint) since it knows
    the full plate structure from OMERO metadata.

    For now, this just uses the ImageXpress pattern since they're compatible.
    In the future, this could enforce that all components are present.
    """

    # Use ImageXpress pattern - it's compatible
    from openhcs.microscopes.imagexpress import ImageXpressFilenameParser
    _pattern = ImageXpressFilenameParser._pattern

    @classmethod
    def can_parse(cls, filename: str) -> bool:
        """Check if this parser can parse the given filename."""
        return cls._pattern.match(filename) is not None

    def parse_filename(self, filename: str) -> Optional[FilenameParseResult]:
        """Parse OMERO virtual filename using ImageXpress pattern."""
        from openhcs.microscopes.imagexpress import ImageXpressFilenameParser
        parser = ImageXpressFilenameParser()
        return parser.parse_filename(filename)

    def construct_filename(self, well, site, channel, z_index, timepoint, extension='.tif', **kwargs) -> str:
        """
        Construct OMERO virtual filename.

        OMERO always generates complete filenames with all components.
        """
        from openhcs.microscopes.imagexpress import ImageXpressFilenameParser
        parser = ImageXpressFilenameParser()
        return parser.construct_filename(
            well=well,
            site=site,
            channel=channel,
            z_index=z_index,
            timepoint=timepoint,
            extension=extension,
            **kwargs
        )

    def extract_component_coordinates(self, component_value: str) -> Tuple[str, str]:
        """Extract coordinates from well identifier (e.g., 'A01' → ('A', '01'))."""
        from openhcs.microscopes.imagexpress import ImageXpressFilenameParser
        parser = ImageXpressFilenameParser()
        return parser.extract_component_coordinates(component_value)


class OMEROHandler(MicroscopeHandler):
    """OMERO microscope handler - uses OMERO native metadata."""

    _microscope_type = Microscope.OMERO.value
    _metadata_handler_class = None  # Set after class definition

    @classmethod
    def source_selection_role(cls) -> MicroscopeSourceSelectionRole:
        """Declare OMERO as a remote-service owner, not a disk filename format."""

        return MicroscopeSourceSelectionRole.REMOTE_SERVICE

    @classmethod
    def source_selection_guidance(cls) -> str:
        """Explain when the OMERO virtual source owner should be selected."""

        return (
            "Use for plates addressed through an OMERO connection. Its filename "
            "parser describes projected virtual names, so local disk filenames are "
            "not evidence that OMERO owns a folder."
        )

    def __init__(self, filemanager: FileManager, pattern_format: Optional[str] = None):
        """
        Initialize OMERO handler.

        OMERO uses OMEROFilenameParser to parse virtual filenames generated by the backend.
        The orchestrator needs the parser to extract components from filenames.

        Args:
            filemanager: FileManager with OMERO backend in registry
            pattern_format: Unused for OMERO (included for interface compatibility)
        """
        parser = OMEROFilenameParser()
        metadata_handler = OMEROMetadataHandler(filemanager)
        super().__init__(parser=parser, metadata_handler=metadata_handler)

    @property
    def root_dir(self) -> str:
        """Root directory for OMERO virtual workspace.

        Returns "Images" for path compatibility with OMERO virtual paths.
        """
        return "Images"

    @property
    def microscope_type(self) -> str:
        return self._microscope_type

    @property
    def metadata_handler_class(self) -> Type[MetadataHandler]:
        """Metadata handler class (for interface enforcement only)."""
        return OMEROMetadataHandler

    @property
    def compatible_backends(self) -> List[Backend]:
        """OMERO is only compatible with OMERO_LOCAL backend."""
        return [Backend.OMERO_LOCAL]

    def _prepare_workspace(self, workspace_path: Path, filemanager: FileManager) -> Path:
        """
        OMERO doesn't need workspace preparation - it's a virtual filesystem.

        Args:
            workspace_path: Path to workspace (unused for OMERO)
            filemanager: FileManager instance (unused for OMERO)

        Returns:
            workspace_path unchanged
        """
        return workspace_path

    def initialize_workspace(self, plate_path: Union[int, Path], filemanager: FileManager) -> Path:
        """
        OMERO creates a virtual path for the plate.

        For OMERO, plate_path is an int (plate_id). We convert it to a virtual
        Path like "/omero/plate_54/Images" that can be used throughout the system.
        The backend extracts the plate_id from this virtual path when needed.

        Args:
            plate_path: OMERO plate_id (int) or Path (for compatibility)
            filemanager: Unused for OMERO

        Returns:
            Virtual Path for OMERO plate (e.g., "/omero/plate_54/Images")
        """
        # Convert int plate_id to virtual path
        if isinstance(plate_path, int):
            # Create virtual path: /omero/plate_{id}/Images
            virtual_path = Path(f"/omero/plate_{plate_path}/Images")
            return virtual_path
        else:
            # Already a Path (shouldn't happen for OMERO, but handle it)
            return plate_path


# Set metadata handler class after class definition for automatic registration
from openhcs.microscopes.microscope_base import register_metadata_handler  # noqa: E402
OMEROHandler._metadata_handler_class = OMEROMetadataHandler
register_metadata_handler(OMEROHandler, OMEROMetadataHandler)
