"""
Opera Phenix microscope implementations for openhcs.

This module provides concrete implementations of FilenameParser and MetadataHandler
for Opera Phenix microscopes.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type, Tuple

from openhcs.constants.constants import Backend, Microscope
from openhcs.core.components.parser_metaprogramming import (
    format_filename_component,
    require_filename_component,
)
from openhcs.microscopes.opera_phenix_xml_parser import OperaPhenixXmlParser
from polystore.filemanager import FileManager
from polystore.virtual_workspace import SourcePixelRef
from polystore.exceptions import MetadataNotFoundError
from openhcs.microscopes.microscope_base import MicroscopeHandler
from openhcs.microscopes.microscope_interfaces import (
    DiskImageFileListingMetadataHandler,
    FilenameParseResult,
    FilenameParser,
    MetadataHandler,
)

logger = logging.getLogger(__name__)



class OperaPhenixHandler(MicroscopeHandler):
    """
    MicroscopeHandler implementation for Opera Phenix systems.

    This handler combines the OperaPhenix filename parser with its
    corresponding metadata handler. It guarantees aligned behavior
    for plate structure parsing, metadata extraction, and any optional
    post-processing steps required after workspace setup.
    """

    # Explicit microscope type for proper registration
    _microscope_type = Microscope.OPERAPHENIX.value

    # Class attribute for automatic metadata handler registration (set after class definition)
    _metadata_handler_class = None
    # metadata handler class assigned post-definition

    @classmethod
    def supports_explicit_incomplete_export(cls) -> bool:
        """Require Index.xml before Opera Phenix owns native plate semantics."""

        return False

    def __init__(self, filemanager: FileManager, pattern_format: Optional[str] = None):
        self.parser = OperaPhenixFilenameParser(filemanager, pattern_format=pattern_format)
        self.metadata_handler = OperaPhenixMetadataHandler(filemanager)
        super().__init__(parser=self.parser, metadata_handler=self.metadata_handler)

    @property
    def root_dir(self) -> str:
        """
        Root directory for Opera Phenix virtual workspace preparation.

        Returns "Images" because Opera Phenix field remapping is applied
        to images in the Images/ subdirectory, and virtual paths include Images/ prefix.
        """
        return "Images"

    @property
    def microscope_type(self) -> str:
        """Microscope type identifier (for interface enforcement only)."""
        return self._microscope_type

    @property
    def metadata_handler_class(self) -> Type[MetadataHandler]:
        """Metadata handler class (for interface enforcement only)."""
        return OperaPhenixMetadataHandler

    @property
    def compatible_backends(self) -> List[Backend]:
        """
        Opera Phenix is compatible with DISK backend only.

        Legacy microscope format with standard file operations.
        """
        return [Backend.DISK]

    # Uses default workspace initialization from base class

    def _build_virtual_mapping(self, plate_path: Path, filemanager: FileManager) -> Path:
        """
        Build Opera Phenix virtual workspace mapping using plate-relative paths.

        Args:
            plate_path: Path to plate directory
            filemanager: FileManager instance for file operations

        Returns:
            Path to image directory
        """
        plate_path = Path(plate_path)  # Ensure Path object

        logger.info(f"🔄 BUILDING VIRTUAL MAPPING: Opera Phenix field remapping for {plate_path}")

        # Opera Phenix images are always in Images/ subdirectory
        image_dir = plate_path / self.root_dir

        # Default to empty field mapping (no remapping)
        field_mapping = {}

        # Try to load field mapping from Index.xml if available
        xml_parser = None
        try:
            index_xml = filemanager.find_file_recursive(plate_path, "Index.xml", Backend.DISK.value)
            if index_xml:
                xml_parser = OperaPhenixXmlParser(index_xml)
                field_mapping = xml_parser.get_field_id_mapping()
                logger.debug("Loaded field mapping from Index.xml: %s", field_mapping)
            else:
                logger.debug("Index.xml not found. Using default field mapping.")
        except Exception as e:
            logger.error("Error loading Index.xml: %s", e)
            logger.debug("Using default field mapping due to error.")

        # Fill missing images BEFORE building virtual mapping
        # This handles autofocus failures by creating black placeholder images
        if xml_parser:
            num_filled = self._fill_missing_images(image_dir, xml_parser, filemanager)
            if num_filled > 0:
                logger.info(f"Created {num_filled} placeholder images for autofocus failures")

        # Get all image files in the directory (including newly created placeholders)
        image_files = filemanager.list_image_files(image_dir, Backend.DISK.value)

        # Initialize mapping dict (PLATE-RELATIVE paths)
        workspace_mapping = {}

        # Process each file
        for file_path in image_files:
            # FileManager should return strings, but handle Path objects too
            if isinstance(file_path, str):
                file_name = os.path.basename(file_path)
            elif isinstance(file_path, Path):
                file_name = file_path.name
            else:
                # Skip any unexpected types
                logger.warning("Unexpected file path type: %s", type(file_path).__name__)
                continue

            # Parse file metadata
            metadata = self.parser.parse_filename(file_name)
            if not metadata or 'site' not in metadata or metadata['site'] is None:
                continue

            # Remap the field ID using the spatial layout
            original_field_id = metadata['site']
            new_field_id = field_mapping.get(original_field_id, original_field_id)

            # Construct the new filename with proper padding
            metadata['site'] = new_field_id  # Update site with remapped value
            new_name = self.parser.construct_filename(**metadata)

            # Build PLATE-RELATIVE mapping (no workspace directory)
            # Use .as_posix() to ensure forward slashes on all platforms (Windows uses backslashes with str())
            virtual_relative = (Path("Images") / new_name).as_posix()
            real_relative = (Path("Images") / file_name).as_posix()
            workspace_mapping[virtual_relative] = SourcePixelRef(
                backend=Backend.DISK.value,
                backend_address=real_relative,
            )

        logger.info(f"Built {len(workspace_mapping)} virtual path mappings for Opera Phenix")

        # Save virtual workspace mapping and all available metadata
        self.save_virtual_workspace_metadata(plate_path, workspace_mapping)

        return image_dir

    def _fill_missing_images(
        self,
        image_dir: Path,
        xml_parser: OperaPhenixXmlParser,
        filemanager: FileManager
    ) -> int:
        """
        Fill in missing images with black pixels by detecting gaps in continuous sequences.

        This method:
        1. Parses all existing files to extract dimension indices
        2. Finds all unique values for each dimension (wells, channels, sites, z-planes, timepoints)
        3. Generates all expected combinations (cartesian product)
        4. Creates black placeholder images for missing combinations

        Args:
            image_dir: Path to the image directory
            xml_parser: Parsed Index.xml (unused, kept for signature compatibility)
            filemanager: FileManager for file operations

        Returns:
            Number of missing images created
        """
        import numpy as np
        from itertools import product

        logger.debug("Checking for missing images in Opera Phenix workspace using continuous sequence detection")

        # 1. Get actual files
        # Clause 245: Workspace operations are disk-only by design
        actual_file_paths = filemanager.list_image_files(image_dir, Backend.DISK.value)

        if not actual_file_paths:
            logger.debug("No existing images found, skipping missing image detection")
            return 0

        # 2. Parse all files and collect dimension values
        wells = set()
        channels = set()
        sites = set()
        z_indices = set()
        timepoints = set()
        actual_combinations = set()  # Store parsed component tuples, not filenames

        # Track filename format from first file
        has_timepoint = None
        sample_metadata = None

        for file_path in actual_file_paths:
            filename = os.path.basename(file_path)

            # Parse filename
            metadata = self.parser.parse_filename(filename)
            if not metadata:
                logger.warning(f"Could not parse filename: {filename}")
                continue

            # Store first valid metadata as sample
            if sample_metadata is None:
                sample_metadata = metadata

            # Collect dimension values
            well = metadata.get('well')
            channel = metadata.get('channel')
            site = metadata.get('site')
            z_index = metadata.get('z_index')
            timepoint = metadata.get('timepoint')

            if well:
                wells.add(well)
            if channel is not None:
                channels.add(channel)
            if site is not None:
                sites.add(site)
            if z_index is not None:
                z_indices.add(z_index)
            if timepoint is not None:
                timepoints.add(timepoint)
                has_timepoint = True
            elif has_timepoint is None:
                has_timepoint = False

            # Store the actual combination tuple for comparison
            actual_combinations.add((well, channel, site, z_index, timepoint))

        if not wells or not channels or not sites:
            logger.warning("Could not extract sufficient dimension information from filenames")
            return 0

        # Default z_index to 1 if not present in any file
        if not z_indices:
            z_indices.add(1)

        # Handle timepoint dimension
        if not timepoints and has_timepoint:
            timepoints.add(1)

        logger.info(f"Detected dimensions: {len(wells)} wells, {len(channels)} channels, "
                   f"{len(sites)} sites, {len(z_indices)} z-planes" +
                   (f", {len(timepoints)} timepoints" if timepoints else ""))

        # 3. Generate all expected combinations
        expected_combinations = set()
        for well, channel, site, z_index in product(wells, channels, sites, z_indices):
            if timepoints:
                for timepoint in timepoints:
                    expected_combinations.add((well, channel, site, z_index, timepoint))
            else:
                expected_combinations.add((well, channel, site, z_index, None))

        logger.debug(f"Expected total images: {len(expected_combinations)}")
        logger.debug(f"Actual images found: {len(actual_combinations)}")

        # 4. Find missing combinations by comparing component tuples (not filenames)
        missing_combinations = expected_combinations - actual_combinations

        if not missing_combinations:
            logger.debug("No missing images detected in continuous sequence")
            return 0

        logger.info(f"Found {len(missing_combinations)} missing images in continuous sequence")

        # 5. Construct filenames for missing combinations
        missing_files = []
        for combination in missing_combinations:
            well, channel, site, z_index, timepoint = combination

            # Construct filename using standardized format
            filename = self.parser.construct_filename(
                well=well,
                site=site,
                channel=channel,
                z_index=z_index,
                timepoint=timepoint,
                extension=sample_metadata.get('extension', '.tiff'),
                site_padding=3,  # Virtual workspace uses standardized 3-digit padding
                z_padding=3
            )

            missing_files.append(filename)

        # 6. Get image dimensions from first existing image
        try:
            first_image_path = actual_file_paths[0]
            # Clause 245: Workspace operations are disk-only by design
            first_image = filemanager.load(first_image_path, Backend.DISK.value)
            height, width = first_image.shape
            dtype = first_image.dtype
            logger.debug(f"Using dimensions from existing image: {height}x{width}, dtype={dtype}")
        except Exception as e:
            logger.warning(f"Could not load existing image for dimensions: {e}")
            # Default dimensions for Opera Phenix
            height, width = 2160, 2160
            dtype = np.uint16
            logger.debug(f"Using default dimensions: {height}x{width}, dtype={dtype}")

        # 7. Create black images for missing files
        black_image = np.zeros((height, width), dtype=dtype)

        created_count = 0
        skipped_count = 0
        
        for filename in missing_files:
            output_path = image_dir / filename
            
            # CRITICAL SAFETY CHECK: Never overwrite existing files
            if output_path.exists():
                # File already exists - DO NOT OVERWRITE to prevent data loss
                logger.warning(f"Image already exists, skipping to prevent data loss: {filename}")
                skipped_count += 1
                continue
            
            # Clause 245: Workspace operations are disk-only by design
            filemanager.save(black_image, output_path, Backend.DISK.value)
            logger.debug(f"Created missing image: {filename}")
            created_count += 1

        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} existing files to prevent overwriting")
        logger.info(f"Successfully created {created_count} missing images with black pixels")
        return created_count


class OperaPhenixFilenameParser(FilenameParser):
    """Parser for Opera Phenix microscope filenames.

    Handles Opera Phenix format filenames like:
    - r01c01f001p01-ch1sk1fk1fl1.tiff
    - r01c01f001p01-ch1.tiff
    """

    # Regular expression pattern for Opera Phenix filenames
    # Supports: row, column, site (field), z_index (plane), channel, timepoint (sk=stack)
    # sk = stack/timepoint, fk = field stack, fl = focal level
    # Also supports result files with suffixes like: r01c01f001p01-ch1_cell_counts_step7.json
    _pattern = re.compile(r"r(\d{1,2})c(\d{1,2})f(\d+|\{[^\}]*\})p(\d+|\{[^\}]*\})-ch(\d+|\{[^\}]*\})(?:sk(\d+|\{[^\}]*\}))?(?:fk\d+)?(?:fl\d+)?(?:_.*?)?(\.\w+)$", re.I)

    # Pattern for extracting row and column from Opera Phenix well format
    _well_pattern = re.compile(r"R(\d{2})C(\d{2})", re.I)

    def __init__(self, filemanager=None, pattern_format=None):
        """
        Initialize the parser.

        Args:
            filemanager: FileManager instance (not used, but required for interface compatibility)
            pattern_format: Optional pattern format (not used, but required for interface compatibility)
        """
        super().__init__()  # Initialize the generic parser interface

        # These parameters are not used by this parser, but are required for interface compatibility
        self.filemanager = filemanager
        self.pattern_format = pattern_format

    @classmethod
    def can_parse(cls, filename: str) -> bool:
        """
        Check if this parser can parse the given filename.

        Args:
            filename (str): Filename to check

        Returns:
            bool: True if this parser can parse the filename, False otherwise
        """
        # This is a string operation that doesn't perform actual file I/O
        # Extract just the basename
        basename = os.path.basename(filename)
        # Check if the filename matches the Opera Phenix pattern
        return bool(cls._pattern.match(basename))

    def parse_filename(self, filename: str) -> Optional[FilenameParseResult]:
        """
        Parse an Opera Phenix filename to extract all components.
        Supports placeholders like {iii} which will return None for that field.

        Args:
            filename (str): Filename to parse

        Returns:
            dict or None: Dictionary with extracted components or None if parsing fails.
        """
        # This is a string operation that doesn't perform actual file I/O
        basename = os.path.basename(filename)
        logger.debug("OperaPhenixFilenameParser attempting to parse basename: '%s'", basename)

        # Try parsing using the Opera Phenix pattern
        match = self._pattern.match(basename)
        if match:
            logger.debug("Regex match successful for '%s'", basename)
            row, col, site_str, z_str, channel_str, sk_str, ext = match.groups()

            # Helper function to parse component strings
            def parse_comp(s):
                """Parse component string to int or None if it's a placeholder."""
                if not s or '{' in s:
                    return None
                return int(s)

            # Create well ID from row and column
            well = f"R{int(row):02d}C{int(col):02d}"

            # Parse components
            site = parse_comp(site_str)
            channel = parse_comp(channel_str)
            z_index = parse_comp(z_str)
            timepoint = parse_comp(sk_str)  # sk = stack/timepoint

            result = FilenameParseResult({
                'well': well,
                'site': site,
                'channel': channel,
                'wavelength': channel,  # For backward compatibility
                'z_index': z_index,
                'timepoint': timepoint,  # sk = stack/timepoint
                'extension': ext if ext else '.tif'
            })
            return result

        logger.warning("Regex match failed for basename: '%s'", basename)
        return None

    def construct_filename(self, extension: str = '.tiff', site_padding: int = 3, z_padding: int = 3, **component_values) -> str:
        """
        Construct an Opera Phenix filename from components.

        This method now uses **kwargs to accept any component values dynamically,
        making it compatible with the generic parser interface.

        Note: Opera Phenix uses 'sk' (stack) for timepoint in filenames.

        Args:
            extension (str, optional): File extension (default: '.tiff')
            site_padding (int, optional): Width to pad site numbers to (default: 3)
            z_padding (int, optional): Width to pad Z-index numbers to (default: 3)
            **component_values: Component values as keyword arguments.
                               Expected keys: well, site, channel, z_index, timepoint

        Returns:
            str: Constructed filename
        """
        well = require_filename_component(component_values, 'well')
        site = require_filename_component(component_values, 'site')
        channel = require_filename_component(component_values, 'channel')
        z_index = require_filename_component(component_values, 'z_index')
        timepoint = require_filename_component(component_values, 'timepoint')

        # Extract row and column from well name
        # Check if well is in Opera Phenix format (e.g., 'R01C03')
        match = self._well_pattern.match(well)
        if match:
            # Extract row and column from Opera Phenix format
            row = int(match.group(1))
            col = int(match.group(2))
        else:
            raise ValueError(f"Invalid well format: {well}. Expected format: 'R01C03'")

        # Construct filename in Opera Phenix format
        site_part = f"f{format_filename_component(site, site_padding)}"
        z_part = f"p{format_filename_component(z_index, z_padding)}"
        sk_part = f"sk{format_filename_component(timepoint)}"

        return (
            f"r{row:02d}c{col:02d}{site_part}{z_part}"
            f"-ch{format_filename_component(channel)}{sk_part}fk1fl1{extension}"
        )

    def remap_field_in_filename(self, filename: str, xml_parser: Optional[OperaPhenixXmlParser] = None) -> str:
        """
        Remap the field ID in a filename to follow a top-left to bottom-right pattern.

        Args:
            filename: Original filename
            xml_parser: Parser with XML data

        Returns:
            str: New filename with remapped field ID
        """
        if xml_parser is None:
            return filename

        # Parse the filename
        metadata = self.parse_filename(filename)
        if not metadata or 'site' not in metadata or metadata['site'] is None:
            return filename

        # Get the mapping and remap the field ID
        mapping = xml_parser.get_field_id_mapping()
        new_field_id = xml_parser.remap_field_id(metadata['site'], mapping)

        # Always create a new filename with the remapped field ID and consistent padding
        # This ensures all filenames have the same format, even if the field ID didn't change
        metadata['site'] = new_field_id  # Update site with remapped value
        return self.construct_filename(**metadata)

    def extract_component_coordinates(self, component_value: str) -> Tuple[str, str]:
        """
        Extract coordinates from component identifier (typically well).

        Args:
            component_value (str): Component identifier (e.g., 'R03C04' or 'A01')

        Returns:
            Tuple[str, str]: (row, column) where row is like 'A', 'B' and column is like '01', '04'

        Raises:
            ValueError: If component format is invalid
        """
        if not component_value:
            raise ValueError(f"Invalid component format: {component_value}")

        # Check if component is in Opera Phenix format (e.g., 'R01C03')
        match = self._well_pattern.match(component_value)
        if match:
            # Extract row and column from Opera Phenix format
            row_num = int(match.group(1))
            col_num = int(match.group(2))
            # Convert to letter-number format: R01C03 -> A, 03
            row = chr(ord('A') + row_num - 1)  # R01 -> A, R02 -> B, etc.
            col = f"{col_num:02d}"  # Ensure 2-digit padding
            return row, col
        else:
            # Assume simple format like 'A01', 'C04'
            if len(component_value) < 2:
                raise ValueError(f"Invalid component format: {component_value}")
            row = component_value[0]
            col = component_value[1:]
            if not row.isalpha() or not col.isdigit():
                raise ValueError(f"Invalid Opera Phenix component format: {component_value}. Expected 'R01C03' or 'A01' format")
            return row, col


class OperaPhenixMetadataHandler(DiskImageFileListingMetadataHandler):
    """
    Metadata handler for Opera Phenix microscopes.

    Handles finding and parsing Index.xml files for Opera Phenix microscopes.
    """

    def __init__(self, filemanager: FileManager):
        """
        Initialize the metadata handler.

        Args:
            filemanager: FileManager instance for file operations.
        """
        super().__init__()
        self.filemanager = filemanager

    # Legacy mode has been completely purged

    def find_metadata_file(self, plate_path: Union[str, Path]):
        """
        Find the Index.xml file in the plate directory.

        Args:
            plate_path: Path to the plate directory

        Returns:
            Path to the Index.xml file

        Raises:
            FileNotFoundError: If no Index.xml file is found
        """
        # Ensure plate_path is a Path object
        if isinstance(plate_path, str):
            plate_path = Path(plate_path)

        # Ensure the path exists
        if not plate_path.exists():
            raise FileNotFoundError(f"Plate path does not exist: {plate_path}")

        # Check for Index.xml in the plate directory
        index_xml = plate_path / "Index.xml"
        if index_xml.exists():
            return index_xml

        # Check for Index.xml in the Images directory
        images_dir = plate_path / "Images"
        if images_dir.exists():
            index_xml = images_dir / "Index.xml"
            if index_xml.exists():
                return index_xml

        # No recursive search - only check root and Images directories
        raise FileNotFoundError(
            f"Index.xml not found in {plate_path} or {plate_path}/Images. "
            "Opera Phenix metadata requires Index.xml file."
        )

        # Ensure result is a Path object
        if isinstance(result, str):
            return Path(result)
        if isinstance(result, Path):
            return result
        # This should not happen if FileManager is properly implemented
        logger.warning("Unexpected result type from find_file_recursive: %s", type(result).__name__)
        return Path(str(result))

    def get_grid_dimensions(self, plate_path: Union[str, Path]):
        """
        Get grid dimensions for stitching from Index.xml file.

        Args:
            plate_path: Path to the plate folder

        Returns:
            Tuple of (grid_rows, grid_cols) - UPDATED: Now returns (rows, cols) for MIST compatibility

        Raises:
            FileNotFoundError: If no Index.xml file is found
            OperaPhenixXmlParseError: If the XML cannot be parsed
            OperaPhenixXmlContentError: If grid dimensions cannot be determined
        """
        # Ensure plate_path is a Path object
        if isinstance(plate_path, str):
            plate_path = Path(plate_path)

        # Ensure the path exists
        if not plate_path.exists():
            raise FileNotFoundError(f"Plate path does not exist: {plate_path}")

        # Find the Index.xml file - this will raise FileNotFoundError if not found
        index_xml = self.find_metadata_file(plate_path)

        # Use the OperaPhenixXmlParser to get the grid size
        # This will raise appropriate exceptions if parsing fails
        xml_parser = self.create_xml_parser(index_xml)
        grid_size = xml_parser.get_grid_size()

        # Validate the grid size
        if grid_size[0] <= 0 or grid_size[1] <= 0:
            raise ValueError(
                f"Invalid grid dimensions: {grid_size[0]}x{grid_size[1]}. "
                "Grid dimensions must be positive integers."
            )

        logger.info("Grid size from Index.xml: %dx%d (cols x rows)", grid_size[0], grid_size[1])
        # FIXED: Return (rows, cols) for MIST compatibility instead of (cols, rows)
        return (grid_size[1], grid_size[0])

    def get_pixel_size(self, plate_path: Union[str, Path]):
        """
        Get the pixel size from Index.xml file.

        Args:
            plate_path: Path to the plate folder

        Returns:
            Pixel size in micrometers

        Raises:
            FileNotFoundError: If no Index.xml file is found
            OperaPhenixXmlParseError: If the XML cannot be parsed
            OperaPhenixXmlContentError: If pixel size cannot be determined
        """
        # Ensure plate_path is a Path object
        if isinstance(plate_path, str):
            plate_path = Path(plate_path)

        # Ensure the path exists
        if not plate_path.exists():
            raise FileNotFoundError(f"Plate path does not exist: {plate_path}")

        # Find the Index.xml file - this will raise FileNotFoundError if not found
        index_xml = self.find_metadata_file(plate_path)

        # Use the OperaPhenixXmlParser to get the pixel size
        # This will raise appropriate exceptions if parsing fails
        xml_parser = self.create_xml_parser(index_xml)
        pixel_size = xml_parser.get_pixel_size()

        # Validate the pixel size
        if pixel_size <= 0:
            raise ValueError(
                f"Invalid pixel size: {pixel_size}. "
                "Pixel size must be a positive number."
            )

        logger.info("Pixel size from Index.xml: %.4f μm", pixel_size)
        return pixel_size

    def get_channel_values(self, plate_path: Union[str, Path]) -> Optional[Dict[str, Optional[str]]]:
        """
        Get channel key→name mapping from Opera Phenix Index.xml.

        Args:
            plate_path: Path to the plate folder (str or Path)

        Returns:
            Dict mapping channel IDs to channel names from metadata
            Example: {"1": "HOECHST 33342", "2": "Calcein", "3": "Alexa 647"}
        """
        try:
            # Ensure plate_path is a Path object
            if isinstance(plate_path, str):
                plate_path = Path(plate_path)

            # Find and parse Index.xml
            index_xml = self.find_metadata_file(plate_path)
            xml_parser = self.create_xml_parser(index_xml)

            # Extract channel information
            channel_mapping = {}

            # Look for channel entries in the XML
            # Opera Phenix stores channel info in multiple places, try the most common
            root = xml_parser.root
            namespace = xml_parser.namespace

            # Find channel entries with ChannelName elements
            channel_entries = root.findall(f".//{namespace}Entry[@ChannelID]")
            for entry in channel_entries:
                channel_id = entry.get('ChannelID')
                channel_name_elem = entry.find(f"{namespace}ChannelName")

                if channel_id and channel_name_elem is not None:
                    channel_name = channel_name_elem.text
                    if channel_name:
                        channel_mapping[channel_id] = channel_name

            return channel_mapping if channel_mapping else None

        except Exception as e:
            logger.debug(f"Could not extract channel names from Opera Phenix metadata: {e}")
            return None

    def get_well_values(self, plate_path: Union[str, Path]) -> Optional[Dict[str, Optional[str]]]:
        """
        Get well key→name mapping from Opera Phenix metadata.

        Args:
            plate_path: Path to the plate folder (str or Path)

        Returns:
            None - Opera Phenix doesn't provide rich well names in metadata
        """
        return None

    def get_site_values(self, plate_path: Union[str, Path]) -> Optional[Dict[str, Optional[str]]]:
        """
        Get site key→name mapping from Opera Phenix metadata.

        Args:
            plate_path: Path to the plate folder (str or Path)

        Returns:
            None - Opera Phenix doesn't provide rich site names in metadata
        """
        return None

    def get_z_index_values(self, plate_path: Union[str, Path]) -> Optional[Dict[str, Optional[str]]]:
        """
        Get z_index key→name mapping from Opera Phenix metadata.

        Args:
            plate_path: Path to the plate folder (str or Path)

        Returns:
            None - Opera Phenix doesn't provide rich z_index names in metadata
        """
        return None

    def get_timepoint_values(self, plate_path: Union[str, Path]) -> Optional[Dict[str, Optional[str]]]:
        """
        Get timepoint key→name mapping from Opera Phenix metadata.

        Args:
            plate_path: Path to the plate folder (str or Path)

        Returns:
            None - Opera Phenix doesn't provide rich timepoint names in metadata
        """
        return None

    def create_xml_parser(self, xml_path: Union[str, Path]):
        """
        Create an OperaPhenixXmlParser for the given XML file.

        Args:
            xml_path: Path to the XML file

        Returns:
            OperaPhenixXmlParser: Parser for the XML file

        Raises:
            FileNotFoundError: If the XML file does not exist
        """
        # Ensure xml_path is a Path object
        if isinstance(xml_path, str):
            xml_path = Path(xml_path)

        # Ensure the path exists
        if not xml_path.exists():
            raise FileNotFoundError(f"XML file does not exist: {xml_path}")

        # Create the parser
        return OperaPhenixXmlParser(xml_path)


# Set metadata handler class after class definition for automatic registration
from openhcs.microscopes.microscope_base import register_metadata_handler
OperaPhenixHandler._metadata_handler_class = OperaPhenixMetadataHandler
register_metadata_handler(OperaPhenixHandler, OperaPhenixMetadataHandler)
