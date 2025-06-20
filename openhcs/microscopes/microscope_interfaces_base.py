"""
Microscope interfaces base for openhcs.

This module provides the base interfaces for microscope-specific functionality,
including filename parsing and metadata handling. It is separate from the
implementation to avoid circular imports.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union


class FilenameParser(ABC):
    """
    Abstract base class for parsing microscopy image filenames.
    """

    # Constants
    FILENAME_COMPONENTS = ['well', 'site', 'channel', 'z_index', 'extension']
    PLACEHOLDER_PATTERN = '{iii}'

    @classmethod
    @abstractmethod
    def can_parse(cls, filename: str) -> bool:
        """
        Check if this parser can parse the given filename.

        Args:
            filename (str): Filename to check

        Returns:
            bool: True if this parser can parse the filename, False otherwise
        """
        pass

    @abstractmethod
    def parse_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Parse a microscopy image filename to extract all components.

        Args:
            filename (str): Filename to parse

        Returns:
            dict or None: Dictionary with extracted components or None if parsing fails
        """
        pass

    @abstractmethod
    def construct_filename(self, well: str, site: Optional[Union[int, str]] = None,
                          channel: Optional[int] = None,
                          z_index: Optional[Union[int, str]] = None,
                          extension: str = '.tif',
                          site_padding: int = 3, z_padding: int = 3) -> str:
        """
        Construct a filename from components.

        Args:
            well (str): Well ID (e.g., 'A01')
            site (int or str, optional): Site number or placeholder string (e.g., '{iii}')
            channel (int, optional): Channel/wavelength number
            z_index (int or str, optional): Z-index or placeholder string (e.g., '{zzz}')
            extension (str, optional): File extension
            site_padding (int, optional): Width to pad site numbers to (default: 3)
            z_padding (int, optional): Width to pad Z-index numbers to (default: 3)

        Returns:
            str: Constructed filename
        """
        pass


class MetadataHandler(ABC):
    """
    Abstract base class for handling microscope metadata.

    All metadata methods require str or Path objects for file paths.
    """

    @abstractmethod
    def find_metadata_file(self, plate_path: Union[str, Path]) -> Path:
        """
        Find the metadata file for a plate.

        Args:
            plate_path: Path to the plate folder (str or Path)

        Returns:
            Path to the metadata file

        Raises:
            TypeError: If plate_path is not a valid path type
            FileNotFoundError: If no metadata file is found
        """
        pass

    @abstractmethod
    def get_grid_dimensions(self, plate_path: Union[str, Path]) -> Tuple[int, int]:
        """
        Get grid dimensions for stitching from metadata.

        Args:
            plate_path: Path to the plate folder (str or Path)

        Returns:
            Tuple of (grid_size_x, grid_size_y)

        Raises:
            TypeError: If plate_path is not a valid path type
            FileNotFoundError: If no metadata file is found
            ValueError: If grid dimensions cannot be determined
        """
        pass

    @abstractmethod
    def get_pixel_size(self, plate_path: Union[str, Path]) -> float:
        """
        Get the pixel size from metadata.

        Args:
            plate_path: Path to the plate folder (str or Path)

        Returns:
            Pixel size in micrometers

        Raises:
            TypeError: If plate_path is not a valid path type
            FileNotFoundError: If no metadata file is found
            ValueError: If pixel size cannot be determined
        """
        pass

    @abstractmethod
    def get_channel_values(self, plate_path: Union[str, Path]) -> Optional[Dict[str, Optional[str]]]:
        """
        Get channel key→name mapping from metadata.

        Args:
            plate_path: Path to the plate folder (str or Path)

        Returns:
            Dict mapping channel keys to display names, or None if not available
            Example: {"1": "HOECHST 33342", "2": "Calcein", "3": "Alexa 647"}
        """
        pass

    @abstractmethod
    def get_well_values(self, plate_path: Union[str, Path]) -> Optional[Dict[str, Optional[str]]]:
        """
        Get well key→name mapping from metadata.

        Args:
            plate_path: Path to the plate folder (str or Path)

        Returns:
            Dict mapping well keys to display names, or None if not available
            Example: {"A01": "Control", "A02": "Treatment"} or None
        """
        pass

    @abstractmethod
    def get_site_values(self, plate_path: Union[str, Path]) -> Optional[Dict[str, Optional[str]]]:
        """
        Get site key→name mapping from metadata.

        Args:
            plate_path: Path to the plate folder (str or Path)

        Returns:
            Dict mapping site keys to display names, or None if not available
            Example: {"1": "Center", "2": "Edge"} or None
        """
        pass

    @abstractmethod
    def get_z_index_values(self, plate_path: Union[str, Path]) -> Optional[Dict[str, Optional[str]]]:
        """
        Get z_index key→name mapping from metadata.

        Args:
            plate_path: Path to the plate folder (str or Path)

        Returns:
            Dict mapping z_index keys to display names, or None if not available
            Example: {"1": "Bottom", "2": "Middle", "3": "Top"} or None
        """
        pass

    def parse_metadata(self, plate_path: Union[str, Path]) -> Dict[str, Dict[str, Optional[str]]]:
        """
        Parse all metadata using enum→method mapping.

        This method iterates through GroupBy components and calls the corresponding
        abstract methods to collect all available metadata.

        Args:
            plate_path: Path to the plate folder (str or Path)

        Returns:
            Dict mapping component names to their key→name mappings
            Example: {"channel": {"1": "HOECHST 33342", "2": "Calcein"}}
        """
        # Import here to avoid circular imports
        from openhcs.constants.constants import GroupBy

        method_map = {
            GroupBy.CHANNEL: self.get_channel_values,
            GroupBy.WELL: self.get_well_values,
            GroupBy.SITE: self.get_site_values,
            GroupBy.Z_INDEX: self.get_z_index_values
        }

        result = {}
        for group_by, method in method_map.items():
            values = method(plate_path)
            if values:
                result[group_by.value] = values
        return result