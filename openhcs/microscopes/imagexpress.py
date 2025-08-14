"""
ImageXpress microscope implementations for openhcs.

This module provides concrete implementations of FilenameParser and MetadataHandler
for ImageXpress microscopes.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Type

import tifffile

from openhcs.constants.constants import Backend
from openhcs.io.exceptions import MetadataNotFoundError
from openhcs.io.filemanager import FileManager
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
    def common_dirs(self) -> List[str]:
        """Subdirectory names commonly used by ImageXpress"""
        return ['TimePoint_1']

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

    def _prepare_workspace(self, workspace_path: Path, filemanager: FileManager) -> Path:
        """
        Flattens the Z-step folder structure and renames image files for
        consistent padding and Z-plane resolution.

        This method performs preparation but does not determine the final image directory.

        Args:
            workspace_path: Path to the symlinked workspace
            filemanager: FileManager instance for file operations

        Returns:
            Path to the flattened image directory.
        """
        # Find all subdirectories in workspace using the filemanager
        # Pass the backend parameter as required by Clause 306 (Backend Positional Parameters)
        entries = filemanager.list_dir(workspace_path, Backend.DISK.value)

        # Filter entries to get only directories
        subdirs = []
        for entry in entries:
            entry_path = Path(workspace_path) / entry
            if entry_path.is_dir():
                subdirs.append(entry_path)

        # Check if any subdirectory contains common_dirs string
        common_dir_found = False

        for subdir in subdirs:
            if any(common_dir in subdir.name for common_dir in self.common_dirs):
                self._flatten_zsteps(subdir, filemanager)
                common_dir_found = True

        # If no common directory found, process the workspace directly
        if not common_dir_found:
            self._flatten_zsteps(workspace_path, filemanager)

        # Remove thumbnail symlinks after processing
        # Find all files in workspace recursively
        _, all_files = filemanager.collect_dirs_and_files(workspace_path, Backend.DISK.value, recursive=True)

        for file_path in all_files:
            # Check if filename contains "thumb" and if it's a symlink
            if "thumb" in Path(file_path).name.lower() and filemanager.is_symlink(file_path, Backend.DISK.value):
                try:
                    filemanager.delete(file_path, Backend.DISK.value)
                    logger.debug("Removed thumbnail symlink: %s", file_path)
                except Exception as e:
                    logger.warning("Failed to remove thumbnail symlink %s: %s", file_path, e)

        # Return the image directory
        return workspace_path

    def _flatten_zsteps(self, directory: Path, fm: FileManager):
        """
        Process Z-step folders in the given directory.

        Args:
            directory: Path to directory that might contain Z-step folders
            fm: FileManager instance for file operations
        """
        # Check for Z step folders
        zstep_pattern = re.compile(r"ZStep[_-]?(\d+)", re.IGNORECASE)

        # List all subdirectories using the filemanager
        # Pass the backend parameter as required by Clause 306 (Backend Positional Parameters)
        entries = fm.list_dir(directory, Backend.DISK.value)

        # Filter entries to get only directories
        subdirs = []
        for entry in entries:
            entry_path = Path(directory) / entry
            if entry_path.is_dir():
                subdirs.append(entry_path)

        # Find potential Z-step folders
        potential_z_folders = []
        for d in subdirs:
            dir_name = d.name if isinstance(d, Path) else os.path.basename(str(d))
            if zstep_pattern.search(dir_name):
                potential_z_folders.append(d)

        if not potential_z_folders:
            logger.info("No Z step folders found in %s. Processing files directly in directory.", directory)
            # Process files directly in the directory to ensure complete metadata
            self._process_files_in_directory(directory, fm)
            return

        # Sort Z folders by index
        z_folders = []
        for d in potential_z_folders:
            dir_name = d.name if isinstance(d, Path) else os.path.basename(str(d))
            match = zstep_pattern.search(dir_name)
            if match:
                z_index = int(match.group(1))
                z_folders.append((z_index, d))

        # Sort by Z index
        z_folders.sort(key=lambda x: x[0])

        # Process each Z folder
        for z_index, z_dir in z_folders:
            # List all files in the Z folder
            # Pass the backend parameter as required by Clause 306 (Backend Positional Parameters)
            img_files = fm.list_files(z_dir, Backend.DISK.value)

            for img_file in img_files:
                # Skip if not a file
                # Pass the backend parameter as required by Clause 306 (Backend Positional Parameters)
                if not fm.is_file(img_file, Backend.DISK.value):
                    continue

                # Get the filename
                img_file_name = img_file.name if isinstance(img_file, Path) else os.path.basename(str(img_file))

                # Parse the original filename to extract components
                components = self.parser.parse_filename(img_file_name)

                if not components:
                    continue

                # Update the z_index in the components
                components['z_index'] = z_index

                # Use the parser to construct a new filename with the updated z_index
                new_name = self.parser.construct_filename(**components)

                # Create the new path in the parent directory
                new_path = directory / new_name if isinstance(directory, Path) else Path(os.path.join(str(directory), new_name))

                try:
                    # Pass the backend parameter as required by Clause 306 (Backend Positional Parameters)
                    # Use replace_symlinks=True to allow overwriting existing symlinks
                    fm.move(img_file, new_path, Backend.DISK.value, replace_symlinks=True)
                    logger.debug("Moved %s to %s", img_file, new_path)
                except FileExistsError as e:
                    # Propagate FileExistsError with clear message
                    logger.error("Cannot move %s to %s: %s", img_file, new_path, e)
                    raise
                except Exception as e:
                    logger.error("Error moving %s to %s: %s", img_file, new_path, e)
                    raise

        # Remove Z folders after all files have been moved
        for _, z_dir in z_folders:
            try:
                # Pass the backend parameter as required by Clause 306 (Backend Positional Parameters)
                fm.delete_all(z_dir, Backend.DISK.value)
                logger.debug("Removed Z-step folder: %s", z_dir)
            except Exception as e:
                logger.warning("Failed to remove Z-step folder %s: %s", z_dir, e)

    def _process_files_in_directory(self, directory: Path, fm: FileManager):
        """
        Process files directly in a directory to ensure complete metadata.

        This handles files that are not in Z-step folders but may be missing
        channel or z-index information. Similar to how Z-step processing adds
        z_index, this adds default values for missing components.

        Args:
            directory: Path to directory containing image files
            fm: FileManager instance for file operations
        """
        # List all image files in the directory
        img_files = fm.list_files(directory, Backend.DISK.value)

        for img_file in img_files:
            # Skip if not a file
            if not fm.is_file(img_file, Backend.DISK.value):
                continue

            # Get the filename
            img_file_name = img_file.name if isinstance(img_file, Path) else os.path.basename(str(img_file))

            # Parse the original filename to extract components
            components = self.parser.parse_filename(img_file_name)

            if not components:
                continue

            # Check if we need to add missing metadata
            needs_rebuild = False

            # Add default channel if missing (like we do for z_index in Z-step processing)
            if components['channel'] is None:
                components['channel'] = 1
                needs_rebuild = True
                logger.debug("Added default channel=1 to file without channel info: %s", img_file_name)

            # Add default z_index if missing (for 2D images)
            if components['z_index'] is None:
                components['z_index'] = 1
                needs_rebuild = True
                logger.debug("Added default z_index=1 to file without z_index info: %s", img_file_name)

            # Only rebuild filename if we added missing components
            if needs_rebuild:
                # Construct new filename with complete metadata
                new_name = self.parser.construct_filename(**components)

                # Only rename if the filename actually changed
                if new_name != img_file_name:
                    new_path = directory / new_name

                    try:
                        # Pass the backend parameter as required by Clause 306
                        # Use replace_symlinks=True to allow overwriting existing symlinks
                        fm.move(img_file, new_path, Backend.DISK.value, replace_symlinks=True)
                        logger.debug("Rebuilt filename with complete metadata: %s -> %s", img_file_name, new_name)
                    except FileExistsError as e:
                        logger.error("Cannot rename %s to %s: %s", img_file, new_path, e)
                        raise
                    except Exception as e:
                        logger.error("Error renaming %s to %s: %s", img_file, new_path, e)
                        raise


class ImageXpressFilenameParser(FilenameParser):
    """
    Parser for ImageXpress microscope filenames.

    Handles standard ImageXpress format filenames like:
    - A01_s001_w1.tif
    - A01_s1_w1_z1.tif
    """

    # Regular expression pattern for ImageXpress filenames
    _pattern = re.compile(r'(?:.*?_)?([A-Z]\d+)(?:_s(\d+|\{[^\}]*\}))?(?:_w(\d+|\{[^\}]*\}))?(?:_z(\d+|\{[^\}]*\}))?(\.\w+)?$')

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
        # 🔒 Clause 17 — VFS Boundary Method
        # Use Path.name instead of os.path.basename for string operations
        basename = Path(str(filename)).name

        # Check if the filename matches the ImageXpress pattern
        return bool(cls._pattern.match(basename))

    # 🔒 Clause 17 — VFS Boundary Method
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
            well, site_str, channel_str, z_str, ext = match.groups()

            #handle {} place holders
            parse_comp = lambda s: None if not s or '{' in s else int(s)
            site = parse_comp(site_str)
            channel = parse_comp(channel_str)
            z_index = parse_comp(z_str)

            # Use the parsed components in the result
            result = {
                'well': well,
                'site': site,
                'channel': channel,
                'z_index': z_index,
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

    def construct_filename(self, extension: str = '.tif', site_padding: int = 3, z_padding: int = 3, **component_values) -> str:
        """
        Construct an ImageXpress filename from components, only including parts if provided.

        This method now uses **kwargs to accept any component values dynamically,
        making it compatible with the generic parser interface.

        Args:
            extension (str, optional): File extension (default: '.tif')
            site_padding (int, optional): Width to pad site numbers to (default: 3)
            z_padding (int, optional): Width to pad Z-index numbers to (default: 3)
            **component_values: Component values as keyword arguments.
                               Expected keys: well, site, channel, z_index

        Returns:
            str: Constructed filename
        """
        # Extract components from kwargs
        well = component_values.get('well')
        site = component_values.get('site')
        channel = component_values.get('channel')
        z_index = component_values.get('z_index')
        if not well:
            raise ValueError("Well ID cannot be empty or None.")

        parts = [well]
        if site is not None:
            if isinstance(site, str):
                # If site is a string (e.g., '{iii}'), use it directly
                parts.append(f"_s{site}")
            else:
                # Otherwise, format it as a padded integer
                parts.append(f"_s{site:0{site_padding}d}")

        if channel is not None:
            parts.append(f"_w{channel}")

        if z_index is not None:
            if isinstance(z_index, str):
                # If z_index is a string (e.g., '{zzz}'), use it directly
                parts.append(f"_z{z_index}")
            else:
                # Otherwise, format it as a padded integer
                parts.append(f"_z{z_index:0{z_padding}d}")

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

        # 🔒 Clause 65 — No Fallback Logic
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

            # 🔒 Clause 65 — No Fallback Logic
            # Fail loudly if grid dimensions cannot be determined
            raise ValueError(f"Could not find grid dimensions in HTD file {htd_file}")
        except Exception as e:
            # 🔒 Clause 65 — No Fallback Logic
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
                # 🔒 Clause 65 — No Fallback Logic
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

            # 🔒 Clause 65 — No Fallback Logic
            # Fail loudly if pixel size cannot be determined
            raise ValueError(f"Could not find pixel size in image metadata for {plate_path}")

        except Exception as e:
            # 🔒 Clause 65 — No Fallback Logic
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




# Set metadata handler class after class definition for automatic registration
from openhcs.microscopes.microscope_base import register_metadata_handler
ImageXpressHandler._metadata_handler_class = ImageXpressMetadataHandler
register_metadata_handler(ImageXpressHandler, ImageXpressMetadataHandler)
