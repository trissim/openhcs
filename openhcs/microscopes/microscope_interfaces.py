"""
Microscope interfaces for openhcs.

This module provides abstract base classes for microscope-specific functionality,
including filename parsing and metadata handling.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from openhcs.constants.constants import VariableComponents
from openhcs.core.components.parser_metaprogramming import GenericFilenameParser

from openhcs.constants.constants import DEFAULT_PIXEL_SIZE


class FilenameParser(GenericFilenameParser):
    """
    Abstract base class for parsing microscopy image filenames.

    This class now uses the metaprogramming system to generate component-specific
    methods dynamically based on the VariableComponents enum, eliminating hardcoded
    component assumptions.
    """

    def __init__(self):
        """Initialize the parser with VariableComponents enum."""
        super().__init__(VariableComponents)

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
            dict or None: Dictionary with extracted components or None if parsing fails.
            The dictionary should contain keys matching VariableComponents enum values plus 'extension'.
        """
        pass

    @abstractmethod
    def extract_row_column(self, well: str) -> Tuple[str, str]:
        """
        Extract row and column from a well identifier.

        Args:
            well (str): Well identifier (e.g., 'A01', 'R03C04', 'C04')

        Returns:
            Tuple[str, str]: (row, column) where row is like 'A', 'B' and column is like '01', '04'

        Raises:
            ValueError: If well format is invalid for this parser
        """
        pass

    @abstractmethod
    def construct_filename(self, extension: str = '.tif', **component_values) -> str:
        """
        Construct a filename from component values.

        This method now uses **kwargs to accept any component values dynamically,
        making it truly generic and adaptable to any component configuration.

        Args:
            extension (str, optional): File extension (default: '.tif')
            **component_values: Component values as keyword arguments.
                               Keys should match VariableComponents enum values.
                               Example: well='A01', site=1, channel=2, z_index=1

        Returns:
            str: Constructed filename

        Example:
            construct_filename(well='A01', site=1, channel=2, z_index=1, extension='.tif')
        """
        pass


class MetadataHandler(ABC):
    """
    Abstract base class for handling microscope metadata.

    All metadata methods require str or Path objects for file paths.

    Subclasses can define FALLBACK_VALUES for explicit fallbacks:
    FALLBACK_VALUES = {'pixel_size': 1.0, 'grid_dimensions': (3, 3)}
    """

    FALLBACK_VALUES = {
        'pixel_size': DEFAULT_PIXEL_SIZE,  # Default pixel size in micrometers
        'grid_dimensions': None,  # No grid dimensions by default
    }

    def _get_with_fallback(self, method_name: str, *args, **kwargs):
        try:
            return getattr(self, method_name)(*args, **kwargs)
        except Exception:
            key = method_name.replace('get_', '')
            return self.FALLBACK_VALUES[key]

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