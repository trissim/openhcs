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

from openhcs.constants.constants import Backend
from openhcs.microscopes.opera_phenix_xml_parser import OperaPhenixXmlParser
from openhcs.io.filemanager import FileManager
from openhcs.microscopes.microscope_base import MicroscopeHandler
from openhcs.microscopes.microscope_interfaces import (FilenameParser,
                                                            MetadataHandler)

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
    _microscope_type = 'opera_phenix'

    # Class attribute for automatic metadata handler registration (set after class definition)
    _metadata_handler_class = None

    def __init__(self, filemanager: FileManager, pattern_format: Optional[str] = None):
        self.parser = OperaPhenixFilenameParser(filemanager, pattern_format=pattern_format)
        self.metadata_handler = OperaPhenixMetadataHandler(filemanager)
        super().__init__(parser=self.parser, metadata_handler=self.metadata_handler)

    @property
    def common_dirs(self) -> List[str]:
        """Subdirectory names commonly used by Opera Phenix."""
        return ['Images']

    @property
    def microscope_type(self) -> str:
        """Microscope type identifier (for interface enforcement only)."""
        return 'opera_phenix'

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

    def _prepare_workspace(self, workspace_path: Path, filemanager: FileManager):
        """
        Renames Opera Phenix images to follow a consistent field order
        based on spatial layout extracted from Index.xml. Uses remapped
        filenames and replaces the directory in-place.

        This method performs preparation but does not determine the final image directory.

        Args:
            workspace_path: Path to the symlinked workspace
            filemanager: FileManager instance for file operations

        Returns:
            Path to the normalized image directory.
        """

        # Check if workspace has already been processed by looking for temp directory
        # If temp directory exists, workspace was already processed - skip processing
        temp_dir_name = "__opera_phenix_temp"
        for entry in filemanager.list_dir(workspace_path, Backend.DISK.value):
            entry_path = Path(workspace_path) / entry
            if entry_path.is_dir() and entry_path.name == temp_dir_name:
                logger.info(f"📁 WORKSPACE ALREADY PROCESSED: Found {temp_dir_name} - skipping Opera Phenix preparation")
                return workspace_path

        logger.info(f"🔄 PROCESSING WORKSPACE: Applying Opera Phenix name remapping to {workspace_path}")
        # Find the image directory using the common_dirs property
        # Clause 245: Workspace operations are disk-only by design
        # This call is structurally hardcoded to use the "disk" backend

        # Get all entries in the directory
        entries = filemanager.list_dir(workspace_path, Backend.DISK.value)

        # Look for a directory matching any of the common_dirs patterns
        image_dir = workspace_path
        for entry in entries:
            entry_lower = entry.lower()
            if any(common_dir.lower() in entry_lower for common_dir in self.common_dirs):
                # Found a matching directory
                image_dir = Path(workspace_path) / entry if isinstance(workspace_path, (str, Path)) else workspace_path / entry
                logger.info("Found directory matching common_dirs pattern: %s", image_dir)
                break

        # Default to empty field mapping (no remapping)
        field_mapping = {}

        # Try to load field mapping from Index.xml if available
        try:
            # Clause 245: Workspace operations are disk-only by design
            # This call is structurally hardcoded to use the "disk" backend
            index_xml = filemanager.find_file_recursive(workspace_path, "Index.xml", Backend.DISK.value)
            if index_xml:
                xml_parser = OperaPhenixXmlParser(index_xml)
                field_mapping = xml_parser.get_field_id_mapping()
                logger.debug("Loaded field mapping from Index.xml: %s", field_mapping)
            else:
                logger.debug("Index.xml not found. Using default field mapping.")
        except Exception as e:
            logger.error("Error loading Index.xml: %s", e)
            logger.debug("Using default field mapping due to error.")

        # Get all image files in the directory BEFORE creating temp directory
        # This prevents recursive mirroring of the temp directory
        # Clause 245: Workspace operations are disk-only by design
        # This call is structurally hardcoded to use the "disk" backend
        image_files = filemanager.list_image_files(image_dir, Backend.DISK.value)

        # Create a uniquely named temporary directory for renamed files
        # Use "__opera_phenix_temp" to make it clearly identifiable
        if isinstance(image_dir, str):
            temp_dir = os.path.join(image_dir, "__opera_phenix_temp")
        else:  # Path object
            temp_dir = image_dir / "__opera_phenix_temp"

        # SAFETY CHECK: Ensure temp directory is within workspace
        if not str(temp_dir).startswith(str(workspace_path)):
            logger.error("SAFETY VIOLATION: Temp directory would be created outside workspace: %s", temp_dir)
            raise RuntimeError(f"Temp directory would be created outside workspace: {temp_dir}")

        # Clause 245: Workspace operations are disk-only by design
        # This call is structurally hardcoded to use the "disk" backend
        filemanager.ensure_directory(temp_dir, Backend.DISK.value)

        logger.debug("Created temporary directory for Opera Phenix workspace preparation: %s", temp_dir)

        # Process each file
        for file_path in image_files:
            # FileManager should return strings, but handle Path objects too
            if isinstance(file_path, str):
                file_name = os.path.basename(file_path)
                file_path_obj = Path(file_path)
            elif isinstance(file_path, Path):
                file_name = file_path.name
                file_path_obj = file_path
            else:
                # Skip any unexpected types
                logger.warning("Unexpected file path type: %s", type(file_path).__name__)
                continue

            # Check if this is a symlink
            if file_path_obj.is_symlink():
                try:
                    # Get the target of the symlink (what it points to)
                    real_file_path = file_path_obj.resolve()
                    if not real_file_path.exists():
                        logger.warning("Broken symlink detected: %s -> %s", file_path, real_file_path)
                        continue
                    # Store both the symlink path and the real file path
                    source_path = str(real_file_path)
                    symlink_target = str(real_file_path)
                except Exception as e:
                    logger.warning("Failed to resolve symlink %s: %s", file_path, e)
                    continue
            else:
                # This should never happen in a properly mirrored workspace
                logger.error("SAFETY VIOLATION: Found real file in workspace (should be symlink): %s", file_path)
                raise RuntimeError(f"Workspace contains real file instead of symlink: {file_path}")
                
            # Store the original symlink path for reference
            original_symlink_path = str(file_path_obj)

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

            # Create the new path in the temporary directory
            if isinstance(temp_dir, str):
                new_path = os.path.join(temp_dir, new_name)
            else:  # Path object
                new_path = temp_dir / new_name

            # Check if destination already exists in temp directory
            try:
                # Clause 245: Workspace operations are disk-only by design
                # This call is structurally hardcoded to use the "disk" backend
                if filemanager.exists(new_path, Backend.DISK.value):
                    # For temp directory, we can be more aggressive and delete any existing file
                    logger.debug("File exists in temp directory, removing before copy: %s", new_path)
                    filemanager.delete(new_path, Backend.DISK.value)
                
                # Create a symlink in the temp directory pointing to the original file
                # Clause 245: Workspace operations are disk-only by design
                # This call is structurally hardcoded to use the "disk" backend
                filemanager.create_symlink(source_path, new_path, Backend.DISK.value)
                logger.debug("Created symlink in temp directory: %s -> %s", new_path, source_path)
                
            except Exception as e:
                logger.error("Failed to copy file to temp directory: %s -> %s: %s", 
                             source_path, new_path, e)
                raise RuntimeError(f"Failed to copy file to temp directory: {e}") from e

        # Clean up and replace old files - ONLY delete symlinks in workspace, NEVER original files
        for file_path in image_files:
            # Convert to Path object for symlink checking
            file_path_obj = Path(file_path) if isinstance(file_path, str) else file_path

            # SAFETY CHECK: Only delete if it's within the workspace directory
            if not str(file_path_obj).startswith(str(workspace_path)):
                logger.error("SAFETY VIOLATION: Attempted to delete file outside workspace: %s", file_path)
                raise RuntimeError(f"Workspace preparation tried to delete file outside workspace: {file_path}")

            # SAFETY CHECK: In workspace, only delete symlinks, never real files
            if file_path_obj.is_symlink():
                # Safe to delete - it's a symlink in the workspace
                logger.debug("Deleting symlink in workspace: %s", file_path)
                filemanager.delete(file_path, Backend.DISK.value)
            elif file_path_obj.is_file():
                # This should never happen in a properly mirrored workspace
                logger.error("SAFETY VIOLATION: Found real file in workspace (should be symlink): %s", file_path)
                raise RuntimeError(f"Workspace contains real file instead of symlink: {file_path}")
            else:
                logger.warning("File not found or not accessible: %s", file_path)

        # Get all files in the temporary directory
        # Clause 245: Workspace operations are disk-only by design
        # This call is structurally hardcoded to use the "disk" backend
        temp_files = filemanager.list_files(temp_dir, Backend.DISK.value)

        # Move files from temporary directory to image directory
        for temp_file in temp_files:
            # FileManager should return strings, but handle Path objects too
            if isinstance(temp_file, str):
                temp_file_name = os.path.basename(temp_file)
            elif isinstance(temp_file, Path):
                temp_file_name = temp_file.name
            else:
                # Skip any unexpected types
                logger.warning("Unexpected file path type: %s", type(temp_file).__name__)
                continue
            if isinstance(image_dir, str):
                dest_path = os.path.join(image_dir, temp_file_name)
            else:  # Path object
                dest_path = image_dir / temp_file_name

            try:
                # Check if destination already exists in image directory
                # Clause 245: Workspace operations are disk-only by design
                # This call is structurally hardcoded to use the "disk" backend
                if filemanager.exists(dest_path, Backend.DISK.value):
                    # If destination is a symlink, ok to remove and replace
                    if filemanager.is_symlink(dest_path, Backend.DISK.value):
                        logger.debug("Destination is a symlink, removing before copy: %s", dest_path)
                        filemanager.delete(dest_path, Backend.DISK.value)
                    else:
                        # Not a symlink - could be a real file
                        logger.error("SAFETY VIOLATION: Destination exists and is not a symlink: %s", dest_path)
                        raise FileExistsError(f"Destination exists and is not a symlink: {dest_path}")
                
                # First, if the temp file is a symlink, get its target
                temp_file_obj = Path(temp_file) if isinstance(temp_file, str) else temp_file
                if temp_file_obj.is_symlink():
                    try:
                        # Get the target that the temp symlink points to
                        real_target = temp_file_obj.resolve()
                        real_target_path = str(real_target)
                        
                        # Create a new symlink in the image directory pointing to the original file
                        # Clause 245: Workspace operations are disk-only by design
                        # This call is structurally hardcoded to use the "disk" backend
                        filemanager.create_symlink(real_target_path, dest_path, Backend.DISK.value)
                        logger.debug("Created symlink in image directory: %s -> %s", dest_path, real_target_path)
                    except Exception as e:
                        logger.error("Failed to resolve symlink in temp directory: %s: %s", temp_file, e)
                        raise RuntimeError(f"Failed to resolve symlink: {e}") from e
                else:
                    # This should never happen if we're using symlinks consistently
                    logger.warning("Temp file is not a symlink: %s", temp_file)
                    # Fall back to copying the file
                    filemanager.copy(temp_file, dest_path, Backend.DISK.value)
                    logger.debug("Copied file (not symlink) to image directory: %s -> %s", temp_file, dest_path)
                
                # Remove the file from the temporary directory
                # Clause 245: Workspace operations are disk-only by design
                # This call is structurally hardcoded to use the "disk" backend
                filemanager.delete(temp_file, Backend.DISK.value)
                
            except FileExistsError as e:
                # Re-raise with clear message
                logger.error("Cannot copy to destination: %s", e)
                raise
            except Exception as e:
                logger.error("Error copying from temp to destination: %s -> %s: %s", 
                             temp_file, dest_path, e)
                raise RuntimeError(f"Failed to process file from temp directory: {e}") from e

        # SAFETY CHECK: Validate temp directory before deletion 
        if not str(temp_dir).startswith(str(workspace_path)):
            logger.error("SAFETY VIOLATION: Attempted to delete temp directory outside workspace: %s", temp_dir)
            raise RuntimeError(f"Attempted to delete temp directory outside workspace: {temp_dir}")
            
        if not "__opera_phenix_temp" in str(temp_dir):
            logger.error("SAFETY VIOLATION: Attempted to delete non-temp directory: %s", temp_dir)
            raise RuntimeError(f"Attempted to delete non-temp directory: {temp_dir}")
            
        # Remove the temporary directory
        # Clause 245: Workspace operations are disk-only by design
        # This call is structurally hardcoded to use the "disk" backend
        try:
            filemanager.delete(temp_dir, Backend.DISK.value)
            logger.debug("Successfully removed temporary directory: %s", temp_dir)
        except Exception as e:
            # Non-fatal error, just log it
            logger.warning("Failed to remove temporary directory %s: %s", temp_dir, e)

        return image_dir


class OperaPhenixFilenameParser(FilenameParser):
    """Parser for Opera Phenix microscope filenames.

    Handles Opera Phenix format filenames like:
    - r01c01f001p01-ch1sk1fk1fl1.tiff
    - r01c01f001p01-ch1.tiff
    """

    # Regular expression pattern for Opera Phenix filenames
    _pattern = re.compile(r"r(\d{1,2})c(\d{1,2})f(\d+|\{[^\}]*\})p(\d+|\{[^\}]*\})-ch(\d+|\{[^\}]*\})(?:sk\d+)?(?:fk\d+)?(?:fl\d+)?(\.\w+)$", re.I)

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
        # 🔒 Clause 17 — VFS Boundary Method
        # This is a string operation that doesn't perform actual file I/O
        # Extract just the basename
        basename = os.path.basename(filename)
        # Check if the filename matches the Opera Phenix pattern
        return bool(cls._pattern.match(basename))

    def parse_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Parse an Opera Phenix filename to extract all components.
        Supports placeholders like {iii} which will return None for that field.

        Args:
            filename (str): Filename to parse

        Returns:
            dict or None: Dictionary with extracted components or None if parsing fails.
        """
        # 🔒 Clause 17 — VFS Boundary Method
        # This is a string operation that doesn't perform actual file I/O
        basename = os.path.basename(filename)
        logger.debug("OperaPhenixFilenameParser attempting to parse basename: '%s'", basename)

        # Try parsing using the Opera Phenix pattern
        match = self._pattern.match(basename)
        if match:
            logger.debug("Regex match successful for '%s'", basename)
            row, col, site_str, z_str, channel_str, ext = match.groups()

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

            result = {
                'well': well,
                'site': site,
                'channel': channel,
                'wavelength': channel,  # For backward compatibility
                'z_index': z_index,
                'extension': ext if ext else '.tif'
            }
            return result

        logger.warning("Regex match failed for basename: '%s'", basename)
        return None

    def construct_filename(self, extension: str = '.tiff', site_padding: int = 3, z_padding: int = 3, **component_values) -> str:
        """
        Construct an Opera Phenix filename from components.

        This method now uses **kwargs to accept any component values dynamically,
        making it compatible with the generic parser interface.

        Args:
            extension (str, optional): File extension (default: '.tiff')
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
            raise ValueError("Well component is required for filename construction")

        # Extract row and column from well name
        # Check if well is in Opera Phenix format (e.g., 'R01C03')
        match = self._well_pattern.match(well)
        if match:
            # Extract row and column from Opera Phenix format
            row = int(match.group(1))
            col = int(match.group(2))
        else:
            raise ValueError(f"Invalid well format: {well}. Expected format: 'R01C03'")

        # Default Z-index to 1 if not provided
        z_index = 1 if z_index is None else z_index
        channel = 1 if channel is None else channel

        # Construct filename in Opera Phenix format
        if isinstance(site, str):
            # If site is a string (e.g., '{iii}'), use it directly
            site_part = f"f{site}"
        else:
            # Otherwise, format it as a padded integer
            site_part = f"f{site:0{site_padding}d}"

        if isinstance(z_index, str):
            # If z_index is a string (e.g., '{zzz}'), use it directly
            z_part = f"p{z_index}"
        else:
            # Otherwise, format it as a padded integer
            z_part = f"p{z_index:0{z_padding}d}"

        return f"r{row:02d}c{col:02d}{site_part}{z_part}-ch{channel}sk1fk1fl1{extension}"

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


class OperaPhenixMetadataHandler(MetadataHandler):
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
