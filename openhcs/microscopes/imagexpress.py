"""
ImageXpress microscope implementations for openhcs.

This module provides concrete implementations of FilenameParser and MetadataHandler
for ImageXpress microscopes.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Type

import tifffile

from openhcs.constants.constants import Backend
from polystore.exceptions import MetadataNotFoundError
from polystore.filemanager import FileManager
from openhcs.microscopes.microscope_base import MicroscopeHandler
from openhcs.microscopes.microscope_interfaces import (FilenameParser,
                                                            MetadataHandler)

logger = logging.getLogger(__name__)

class ImageXpressHandler(MicroscopeHandler):
    """
    MicroscopeHandler implementation for Molecular Devices ImageXpress systems.

    This handler binds the ImageXpress filename parser and metadata handler,
    enforcing semantic alignment between file layout parsing and metadata resolution.
    """

    # Explicit microscope type for proper registration
    _microscope_type = 'imagexpress'

    # Class attribute for automatic metadata handler registration (set after class definition)
    _metadata_handler_class = None

    def __init__(self, filemanager: FileManager, pattern_format: Optional[str] = None):
        # Initialize parser with filemanager, respecting its interface
        self.parser = ImageXpressFilenameParser(filemanager, pattern_format)
        self.metadata_handler = ImageXpressMetadataHandler(filemanager)
        super().__init__(parser=self.parser, metadata_handler=self.metadata_handler)

    @property
    def root_dir(self) -> str:
        """
        Root directory for ImageXpress virtual workspace preparation.

        Returns "." (plate root) because ImageXpress TimePoint/ZStep folders
        are flattened starting from the plate root, and virtual paths have no prefix.
        """
        return "."

    @property
    def microscope_type(self) -> str:
        """Microscope type identifier (for interface enforcement only)."""
        return 'imagexpress'

    @property
    def metadata_handler_class(self) -> Type[MetadataHandler]:
        """Metadata handler class (for interface enforcement only)."""
        return ImageXpressMetadataHandler

    @property
    def compatible_backends(self) -> List[Backend]:
        """
        ImageXpress is compatible with DISK backend only.

        Legacy microscope format with standard file operations.
        """
        return [Backend.DISK]

    # Uses default workspace initialization from base class

    def _build_virtual_mapping(self, plate_path: Path, filemanager: FileManager) -> Path:
        """
        Build ImageXpress virtual workspace mapping using plate-relative paths.

        Flattens TimePoint and Z-step folder structures virtually by building a mapping dict.

        Args:
            plate_path: Path to plate directory
            filemanager: FileManager instance for file operations

        Returns:
            Path to image directory
        """
        plate_path = Path(plate_path)  # Ensure Path object

        logger.info(f"🔄 BUILDING VIRTUAL MAPPING: ImageXpress folder flattening for {plate_path}")

        # Initialize mapping dict (PLATE-RELATIVE paths)
        workspace_mapping = {}

        # Flatten TimePoint and ZStep folders virtually (starting from plate root)
        self._flatten_timepoints(plate_path, filemanager, workspace_mapping, plate_path)
        self._flatten_zsteps(plate_path, filemanager, workspace_mapping, plate_path)

        logger.info(f"Built {len(workspace_mapping)} virtual path mappings for ImageXpress")

        # Save virtual workspace mapping and all available metadata
        self._save_virtual_workspace_metadata(plate_path, workspace_mapping)

        # Return the image directory
        return plate_path

    def _flatten_zsteps(self, directory: Path, fm: FileManager, mapping_dict: Dict[str, str], plate_path: Path):
        """
        Process Z-step folders virtually by building plate-relative mapping dict.

        Args:
            directory: Path to directory that might contain Z-step folders
            fm: FileManager instance for file operations
            mapping_dict: Dict to populate with virtual → real mappings
            plate_path: Plate root path for computing relative paths
        """
        zstep_pattern = re.compile(r"ZStep[_-]?(\d+)", re.IGNORECASE)

        # Find and process Z-step folders
        self._flatten_indexed_folders(
            directory=directory,
            fm=fm,
            folder_pattern=zstep_pattern,
            component_name='z_index',
            folder_type="ZStep",
            mapping_dict=mapping_dict,
            plate_path=plate_path
        )

    def _flatten_timepoints(self, directory: Path, fm: FileManager, mapping_dict: Dict[str, str], plate_path: Path):
        """
        Process TimePoint folders virtually by building plate-relative mapping dict.

        Args:
            directory: Path to directory that might contain TimePoint folders
            fm: FileManager instance for file operations
            mapping_dict: Dict to populate with virtual → real mappings
            plate_path: Plate root path for computing relative paths
        """
        timepoint_pattern = re.compile(r"TimePoint[_-]?(\d+)", re.IGNORECASE)

        # First flatten Z-steps within each timepoint folder (if they exist)
        entries = fm.list_dir(directory, Backend.DISK.value)
        subdirs = [Path(directory) / entry for entry in entries
                   if (Path(directory) / entry).is_dir()]

        for subdir in subdirs:
            if timepoint_pattern.search(subdir.name):
                self._flatten_zsteps(subdir, fm, mapping_dict, plate_path)

        # Then flatten timepoint folders themselves
        self._flatten_indexed_folders(
            directory=directory,
            fm=fm,
            folder_pattern=timepoint_pattern,
            component_name='timepoint',
            folder_type="TimePoint",
            mapping_dict=mapping_dict,
            plate_path=plate_path
        )

    def _flatten_indexed_folders(self, directory: Path, fm: FileManager,
                                 folder_pattern: re.Pattern, component_name: str,
                                 folder_type: str, mapping_dict: Dict[str, str], plate_path: Path):
        """
        Generic helper to flatten indexed folders virtually (TimePoint_N, ZStep_M, etc.).

        Builds plate-relative mapping dict instead of moving files.

        Args:
            directory: Path to directory that might contain indexed folders
            fm: FileManager instance for file operations
            folder_pattern: Regex pattern to match folder names (must have one capture group for index)
            component_name: Component to update in metadata (e.g., 'z_index', 'timepoint')
            folder_type: Human-readable folder type name (for logging)
            mapping_dict: Dict to populate with virtual → real mappings
            plate_path: Plate root path for computing relative paths
        """
        # List all subdirectories
        entries = fm.list_dir(directory, Backend.DISK.value)
        subdirs = [Path(directory) / entry for entry in entries
                   if (Path(directory) / entry).is_dir()]

        # Find indexed folders
        indexed_folders = []
        for d in subdirs:
            match = folder_pattern.search(d.name)
            if match:
                index = int(match.group(1))
                indexed_folders.append((index, d))

        if not indexed_folders:
            return

        # Sort by index
        indexed_folders.sort(key=lambda x: x[0])

        logger.info(f"Found {len(indexed_folders)} {folder_type} folders. Building virtual mapping...")

        # Process each folder
        for index, folder in indexed_folders:
            logger.debug(f"Processing {folder.name} ({folder_type}={index})")

            # List all files in the folder
            img_files = fm.list_files(str(folder), Backend.DISK.value)

            for img_file in img_files:
                if not fm.is_file(img_file, Backend.DISK.value):
                    continue

                filename = Path(img_file).name

                # Parse existing filename
                metadata = self.parser.parse_filename(filename)
                if not metadata:
                    continue

                # Update the component
                metadata[component_name] = index

                # Reconstruct filename
                new_filename = self.parser.construct_filename(**metadata)

                # Build PLATE-RELATIVE virtual flattened path (at plate root, not in subdirectory)
                # This makes images appear at plate root in virtual workspace
                virtual_relative = new_filename

                # Build PLATE-RELATIVE real path (in subfolder)
                real_relative = Path(img_file).relative_to(plate_path).as_posix()

                # Add to mapping (both plate-relative)
                mapping_dict[virtual_relative] = real_relative
                logger.debug(f"  Mapped: {virtual_relative} → {real_relative}")




class ImageXpressFilenameParser(FilenameParser):
    """
    Parser for ImageXpress microscope filenames.

    Handles standard ImageXpress format filenames like:
    - A01_s001_w1.tif
    - A01_s1_w1_z1.tif
    """

    # Regular expression pattern for ImageXpress filenames
    # Supports: well, site, channel, z_index, timepoint
    # Also supports result files with suffixes like: A01_s001_w1_z001_t001_cell_counts_step7.json
    _pattern = re.compile(r'(?:.*?_)?([A-Z]\d+)(?:_s(\d+|\{[^\}]*\}))?(?:_w(\d+|\{[^\}]*\}))?(?:_z(\d+|\{[^\}]*\}))?(?:_t(\d+|\{[^\}]*\}))?(?:_.*?)?(\.\w+)?$')

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
    def can_parse(cls, filename: Union[str, Any]) -> bool:
        """
        Check if this parser can parse the given filename.

        Args:
            filename: Filename to check (str or VirtualPath)

        Returns:
            bool: True if this parser can parse the filename, False otherwise
        """
        # For strings and other objects, convert to string and get basename
        # Use Path.name instead of os.path.basename for string operations
        basename = Path(str(filename)).name

        # Check if the filename matches the ImageXpress pattern
        return bool(cls._pattern.match(basename))

    # This is a string operation that doesn't perform actual file I/O
    # but is needed for filename parsing during runtime.
    def parse_filename(self, filename: Union[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parse an ImageXpress filename to extract all components, including extension.

        Args:
            filename: Filename to parse (str or VirtualPath)

        Returns:
            dict or None: Dictionary with extracted components or None if parsing fails
        """

        basename = Path(str(filename)).name

        match = self._pattern.match(basename)

        if match:
            well, site_str, channel_str, z_str, t_str, ext = match.groups()

            #handle {} place holders
            parse_comp = lambda s: None if not s or '{' in s else int(s)
            site = parse_comp(site_str)
            channel = parse_comp(channel_str)
            z_index = parse_comp(z_str)
            timepoint = parse_comp(t_str)

            # Use the parsed components in the result
            result = {
                'well': well,
                'site': site,
                'channel': channel,
                'z_index': z_index,
                'timepoint': timepoint,
                'extension': ext if ext else '.tif'  # Default if somehow empty
            }

            return result
        else:
            logger.debug("Could not parse ImageXpress filename: %s", filename)
            return None

    def extract_component_coordinates(self, component_value: str) -> Tuple[str, str]:
        """
        Extract coordinates from component identifier (typically well).

        Args:
            component_value (str): Component identifier (e.g., 'A01', 'C04')

        Returns:
            Tuple[str, str]: (row, column) where row is like 'A', 'C' and column is like '01', '04'

        Raises:
            ValueError: If component format is invalid
        """
        if not component_value or len(component_value) < 2:
            raise ValueError(f"Invalid component format: {component_value}")

        # ImageXpress format: A01, B02, C04, etc.
        row = component_value[0]
        col = component_value[1:]

        if not row.isalpha() or not col.isdigit():
            raise ValueError(f"Invalid ImageXpress component format: {component_value}. Expected format like 'A01', 'C04'")

        return row, col

    def construct_filename(self, extension: str = '.tif', site_padding: int = 3, z_padding: int = 3, timepoint_padding: int = 3, **component_values) -> str:
        """
        Construct an ImageXpress filename from components.

        This method now uses **kwargs to accept any component values dynamically,
        making it compatible with the generic parser interface.

        Args:
            extension (str, optional): File extension (default: '.tif')
            site_padding (int, optional): Width to pad site numbers to (default: 3)
            z_padding (int, optional): Width to pad Z-index numbers to (default: 3)
            timepoint_padding (int, optional): Width to pad timepoint numbers to (default: 3)
            **component_values: Component values as keyword arguments.
                               Expected keys: well, site, channel, z_index, timepoint

        Returns:
            str: Constructed filename
        """
        # Extract components from kwargs
        well = component_values.get('well')
        site = component_values.get('site')
        channel = component_values.get('channel')
        z_index = component_values.get('z_index')
        timepoint = component_values.get('timepoint')

        if not well:
            raise ValueError("Well ID cannot be empty or None.")

        # Default all components to 1 if not provided - ensures consistent filename structure
        if site is None:
            site = 1
        if channel is None:
            channel = 1
        if z_index is None:
            z_index = 1
        if timepoint is None:
            timepoint = 1

        parts = [well]

        # Always add site
        if isinstance(site, str):
            parts.append(f"_s{site}")
        else:
            parts.append(f"_s{site:0{site_padding}d}")

        # Always add channel
        parts.append(f"_w{channel}")

        # Always add z_index
        if isinstance(z_index, str):
            parts.append(f"_z{z_index}")
        else:
            parts.append(f"_z{z_index:0{z_padding}d}")

        # Always add timepoint
        if isinstance(timepoint, str):
            parts.append(f"_t{timepoint}")
        else:
            parts.append(f"_t{timepoint:0{timepoint_padding}d}")

        base_name = "".join(parts)
        return f"{base_name}{extension}"


class ImageXpressMetadataHandler(MetadataHandler):
    """
    Metadata handler for ImageXpress microscopes.

    Handles finding and parsing HTD files for ImageXpress microscopes.
    Inherits fallback values from MetadataHandler ABC.
    """

    def __init__(self, filemanager: FileManager):
        """
        Initialize the metadata handler.

        Args:
            filemanager: FileManager instance for file operations.
        """
        super().__init__()  # Call parent's __init__ without parameters
        self.filemanager = filemanager  # Store filemanager as an instance attribute

    def find_metadata_file(self, plate_path: Union[str, Path],
                           context: Optional['ProcessingContext'] = None) -> Path:
        """
        Find the HTD file for an ImageXpress plate.

        Args:
            plate_path: Path to the plate folder
            context: Optional ProcessingContext (not used)

        Returns:
            Path to the HTD file

        Raises:
            MetadataNotFoundError: If no HTD file is found
            TypeError: If plate_path is not a valid path type
        """
        # Ensure plate_path is a Path object
        if isinstance(plate_path, str):
            plate_path = Path(plate_path)
        elif not isinstance(plate_path, Path):
            raise TypeError(f"Expected str or Path, got {type(plate_path).__name__}")

        # Ensure the path exists
        if not plate_path.exists():
            raise FileNotFoundError(f"Plate path does not exist: {plate_path}")

        # Use filemanager to list files
        # Pass the backend parameter as required by Clause 306 (Backend Positional Parameters)
        htd_files = self.filemanager.list_files(plate_path, Backend.DISK.value, pattern="*.HTD")
        if htd_files:
            for htd_file in htd_files:
                # Convert to Path if it's a string
                if isinstance(htd_file, str):
                    htd_file = Path(htd_file)

                if 'plate' in htd_file.name.lower():
                    return htd_file

            # Return the first file
            first_file = htd_files[0]
            if isinstance(first_file, str):
                return Path(first_file)
            return first_file

        # Fail loudly if no HTD file is found
        raise MetadataNotFoundError("No HTD or metadata file found. ImageXpressHandler requires declared metadata.")

    def get_grid_dimensions(self, plate_path: Union[str, Path],
                           context: Optional['ProcessingContext'] = None) -> Tuple[int, int]:
        """
        Get grid dimensions for stitching from HTD file.

        Args:
            plate_path: Path to the plate folder
            context: Optional ProcessingContext (not used)

        Returns:
            (grid_rows, grid_cols) - UPDATED: Now returns (rows, cols) for MIST compatibility

        Raises:
            MetadataNotFoundError: If no HTD file is found
            ValueError: If grid dimensions cannot be determined from metadata
        """
        htd_file = self.find_metadata_file(plate_path, context)

        # Parse HTD file
        try:
            # HTD files are plain text, but may use different encodings
            # Try multiple encodings in order of likelihood
            encodings_to_try = ['utf-8', 'windows-1252', 'latin-1', 'cp1252', 'iso-8859-1']
            htd_content = None

            for encoding in encodings_to_try:
                try:
                    with open(htd_file, 'r', encoding=encoding) as f:
                        htd_content = f.read()
                    logger.debug("Successfully read HTD file with encoding: %s", encoding)
                    break
                except UnicodeDecodeError:
                    logger.debug("Failed to read HTD file with encoding: %s", encoding)
                    continue

            if htd_content is None:
                raise ValueError(f"Could not read HTD file with any supported encoding: {encodings_to_try}")

            # Extract grid dimensions - try multiple formats
            # First try the new format with "XSites" and "YSites"
            cols_match = re.search(r'"XSites", (\d+)', htd_content)
            rows_match = re.search(r'"YSites", (\d+)', htd_content)

            # If not found, try the old format with SiteColumns and SiteRows
            if not (cols_match and rows_match):
                cols_match = re.search(r'SiteColumns=(\d+)', htd_content)
                rows_match = re.search(r'SiteRows=(\d+)', htd_content)

            if cols_match and rows_match:
                grid_size_x = int(cols_match.group(1))  # cols from metadata
                grid_size_y = int(rows_match.group(1))  # rows from metadata
                logger.info("Using grid dimensions from HTD file: %dx%d (cols x rows)", grid_size_x, grid_size_y)
                # FIXED: Return (rows, cols) for MIST compatibility instead of (cols, rows)
                return grid_size_y, grid_size_x

            # Fail loudly if grid dimensions cannot be determined
            raise ValueError(f"Could not find grid dimensions in HTD file {htd_file}")
        except Exception as e:
            # Fail loudly on any error
            raise ValueError(f"Error parsing HTD file {htd_file}: {e}")

    def get_pixel_size(self, plate_path: Union[str, Path],
                       context: Optional['ProcessingContext'] = None) -> float:
        """
        Gets pixel size by reading TIFF tags from an image file via FileManager.

        Args:
            plate_path: Path to the plate folder
            context: Optional ProcessingContext (not used)

        Returns:
            Pixel size in micrometers

        Raises:
            ValueError: If pixel size cannot be determined from metadata
        """
        # This implementation requires:
        # 1. The backend used by filemanager supports listing image files.
        # 2. The backend allows direct reading of TIFF file tags.
        # 3. Images are in TIFF format.
        try:
            # Use filemanager to list potential image files
            # Pass the backend parameter as required by Clause 306 (Backend Positional Parameters)
            image_files = self.filemanager.list_image_files(plate_path, Backend.DISK.value, extensions={'.tif', '.tiff'}, recursive=True)
            if not image_files:
                # Fail loudly if no image files are found
                raise ValueError(f"No TIFF images found in {plate_path} to read pixel size")

            # Attempt to read tags from the first found image
            first_image_path = image_files[0]

            # Convert to Path if it's a string
            if isinstance(first_image_path, str):
                first_image_path = Path(first_image_path)
            elif not isinstance(first_image_path, Path):
                raise TypeError(f"Expected str or Path, got {type(first_image_path).__name__}")

            # Use the path with tifffile
            with tifffile.TiffFile(first_image_path) as tif:
                 # Try to get ImageDescription tag
                 if tif.pages[0].tags.get('ImageDescription'):
                     desc = tif.pages[0].tags['ImageDescription'].value
                     # Look for spatial calibration using regex
                     match = re.search(r'id="spatial-calibration-x"[^>]*value="([0-9.]+)"', desc)
                     if match:
                         logger.info("Found pixel size metadata %.3f in %s",
                                    float(match.group(1)), first_image_path)
                         return float(match.group(1))

                     # Alternative pattern for some formats
                     match = re.search(r'Spatial Calibration: ([0-9.]+) [uµ]m', desc)
                     if match:
                         logger.info("Found pixel size metadata %.3f in %s",
                                    float(match.group(1)), first_image_path)
                         return float(match.group(1))

            # Fail loudly if pixel size cannot be determined
            raise ValueError(f"Could not find pixel size in image metadata for {plate_path}")

        except Exception as e:
            # Fail loudly on any error
            raise ValueError(f"Error getting pixel size from {plate_path}: {e}")

    def get_channel_values(self, plate_path: Union[str, Path]) -> Optional[Dict[str, Optional[str]]]:
        """
        Get channel key->name mapping from ImageXpress HTD file.

        Args:
            plate_path: Path to the plate folder (str or Path)

        Returns:
            Dict mapping channel IDs to channel names from metadata
            Example: {"1": "TL-20", "2": "DAPI", "3": "FITC", "4": "CY5"}
        """
        try:
            # Find and parse HTD file
            htd_file = self.find_metadata_file(plate_path)

            # Read HTD file content
            encodings_to_try = ['utf-8', 'windows-1252', 'latin-1', 'cp1252', 'iso-8859-1']
            htd_content = None

            for encoding in encodings_to_try:
                try:
                    with open(htd_file, 'r', encoding=encoding) as f:
                        htd_content = f.read()
                    break
                except UnicodeDecodeError:
                    continue

            if htd_content is None:
                logger.debug("Could not read HTD file with any supported encoding")
                return None

            # Extract channel information from WaveName entries
            channel_mapping = {}

            # ImageXpress stores channel names as WaveName1, WaveName2, etc.
            wave_pattern = re.compile(r'"WaveName(\d+)", "([^"]*)"')
            matches = wave_pattern.findall(htd_content)

            for wave_num, wave_name in matches:
                if wave_name:  # Only add non-empty wave names
                    channel_mapping[wave_num] = wave_name

            return channel_mapping if channel_mapping else None

        except Exception as e:
            logger.debug(f"Could not extract channel names from ImageXpress metadata: {e}")
            return None

    def get_well_values(self, plate_path: Union[str, Path]) -> Optional[Dict[str, Optional[str]]]:
        """
        Get well key→name mapping from ImageXpress metadata.

        Args:
            plate_path: Path to the plate folder (str or Path)

        Returns:
            None - ImageXpress doesn't provide rich well names in metadata
        """
        return None

    def get_site_values(self, plate_path: Union[str, Path]) -> Optional[Dict[str, Optional[str]]]:
        """
        Get site key→name mapping from ImageXpress metadata.

        Args:
            plate_path: Path to the plate folder (str or Path)

        Returns:
            None - ImageXpress doesn't provide rich site names in metadata
        """
        return None

    def get_z_index_values(self, plate_path: Union[str, Path]) -> Optional[Dict[str, Optional[str]]]:
        """
        Get z_index key→name mapping from ImageXpress metadata.

        Args:
            plate_path: Path to the plate folder (str or Path)

        Returns:
            None - ImageXpress doesn't provide rich z_index names in metadata
        """
        return None

    def get_timepoint_values(self, plate_path: Union[str, Path]) -> Optional[Dict[str, Optional[str]]]:
        """
        Get timepoint key→name mapping from ImageXpress metadata.

        Args:
            plate_path: Path to the plate folder (str or Path)

        Returns:
            None - ImageXpress doesn't provide rich timepoint names in metadata
        """
        return None

    # Uses default get_image_files() implementation from MetadataHandler ABC




# Set metadata handler class after class definition for automatic registration
from openhcs.microscopes.microscope_base import register_metadata_handler
ImageXpressHandler._metadata_handler_class = ImageXpressMetadataHandler
register_metadata_handler(ImageXpressHandler, ImageXpressMetadataHandler)
