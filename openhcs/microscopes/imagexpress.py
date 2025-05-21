"""
ImageXpress microscope implementations for openhcs.

This module provides concrete implementations of FilenameParser and MetadataHandler
for ImageXpress microscopes.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import tifffile

from openhcs.constants.constants import Backend
from openhcs.io.exceptions import MetadataNotFoundError
from openhcs.io.filemanager import FileManager
from openhcs.microscopes.microscope_base import MicroscopeHandler
from openhcs.microscopes.microscope_interfaces_base import (FilenameParser,
                                                               MetadataHandler)

logger = logging.getLogger(__name__)

class ImageXpressHandler(MicroscopeHandler):
    """
    MicroscopeHandler implementation for Molecular Devices ImageXpress systems.

    This handler binds the ImageXpress filename parser and metadata handler,
    enforcing semantic alignment between file layout parsing and metadata resolution.
    """

    def __init__(self, filemanager: FileManager, pattern_format: Optional[str] = None):
        # Initialize parser with filemanager, respecting its interface
        self.parser = ImageXpressFilenameParser(filemanager, pattern_format)
        self.metadata_handler = ImageXpressMetadataHandler(filemanager)
        super().__init__(parser=self.parser, metadata_handler=self.metadata_handler)

    @property
    def common_dirs(self) -> List[str]:
        """Subdirectory names commonly used by ImageXpress"""
        return 'TimePoint_1'

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
            if self.common_dirs in subdir.name:
                self._flatten_zsteps(subdir, filemanager)
                common_dir_found = True

        # If no common directory found, process the workspace directly
        if not common_dir_found:
            self._flatten_zsteps(workspace_path, filemanager)

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
            logger.info("No Z step folders found in %s. Skipping flattening.", directory)
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
                new_name = self.parser.construct_filename(
                    well=components['well'],
                    site=components['site'],
                    channel=components['channel'],
                    z_index=z_index,
                    extension=components['extension']
                )

                # Create the new path in the parent directory
                new_path = directory / new_name if isinstance(directory, Path) else Path(os.path.join(str(directory), new_name))

                try:
                    # Pass the backend parameter as required by Clause 306 (Backend Positional Parameters)
                    fm.move(img_file, new_path, Backend.DISK.value)
                    logger.debug("Moved %s to %s", img_file, new_path)
                except Exception as e:
                    raise e

        # Remove Z folders after all files have been moved
        for _, z_dir in z_folders:
            try:
                # Pass the backend parameter as required by Clause 306 (Backend Positional Parameters)
                fm.delete_all(z_dir, Backend.DISK.value)
                logger.debug("Removed Z-step folder: %s", z_dir)
            except Exception as e:
                logger.warning("Failed to remove Z-step folder %s: %s", z_dir, e)


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

    def construct_filename(self, well: str, site: Optional[Union[int, str]] = None,
                          channel: Optional[int] = None,
                          z_index: Optional[Union[int, str]] = None,
                          extension: str = '.tif',
                          site_padding: int = 3, z_padding: int = 3) -> str:
        """
        Construct an ImageXpress filename from components, only including parts if provided.

        Args:
            well (str): Well ID (e.g., 'A01')
            site (int or str, optional): Site number or placeholder string (e.g., '{iii}')
            channel (int, optional): Channel number
            z_index (int or str, optional): Z-index or placeholder string (e.g., '{zzz}')
            extension (str, optional): File extension
            site_padding (int, optional): Width to pad site numbers to (default: 3)
            z_padding (int, optional): Width to pad Z-index numbers to (default: 3)

        Returns:
            str: Constructed filename
        """
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
    Metadata for ImageXpressHandler must be present. Legacy fallback is not supported.
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
            (grid_size_x, grid_size_y)

        Raises:
            MetadataNotFoundError: If no HTD file is found
            ValueError: If grid dimensions cannot be determined from metadata
        """
        htd_file = self.find_metadata_file(plate_path, context)

        # Parse HTD file
        try:
            # Use filemanager instead of direct open()
            with self.filemanager.open_file(htd_file, 'r') as f:
                htd_content = f.read()

            # Extract grid dimensions - try multiple formats
            # First try the new format with "XSites" and "YSites"
            cols_match = re.search(r'"XSites", (\d+)', htd_content)
            rows_match = re.search(r'"YSites", (\d+)', htd_content)

            # If not found, try the old format with SiteColumns and SiteRows
            if not (cols_match and rows_match):
                cols_match = re.search(r'SiteColumns=(\d+)', htd_content)
                rows_match = re.search(r'SiteRows=(\d+)', htd_content)

            if cols_match and rows_match:
                grid_size_x = int(cols_match.group(1))
                grid_size_y = int(rows_match.group(1))
                logger.info("Using grid dimensions from HTD file: %dx%d", grid_size_x, grid_size_y)
                return grid_size_x, grid_size_y

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
