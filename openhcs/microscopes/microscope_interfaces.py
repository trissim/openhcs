"""
Microscope interfaces for openhcs.

This module provides abstract base classes for microscope-specific functionality,
including filename parsing and metadata handling.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, Optional, Tuple, Union
from openhcs.constants.constants import VariableComponents, AllComponents
from openhcs.core.components.parser_metaprogramming import GenericFilenameParser
from metaclass_registry import AutoRegisterMeta, LazyDiscoveryDict

from openhcs.constants.constants import DEFAULT_PIXEL_SIZE


class FilenameParser(GenericFilenameParser, metaclass=AutoRegisterMeta):
    """
    Abstract base class for parsing microscopy image filenames.

    This class now uses the metaprogramming system to generate component-specific
    methods dynamically based on the VariableComponents enum, eliminating hardcoded
    component assumptions.
    """

    # Registry configuration for AutoRegisterMeta
    __registry_key__ = '__name__'  # Use class name as registration key
    __registry_name__ = 'filename parser'  # Human-readable name for logging

    def __init__(self):
        """Initialize the parser with AllComponents enum."""
        super().__init__(AllComponents)

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
    def extract_component_coordinates(self, component_value: str) -> Tuple[str, str]:
        """
        Extract coordinates from component identifier (typically well).

        Args:
            component_value (str): Component identifier (e.g., 'A01', 'R03C04', 'C04')

        Returns:
            Tuple[str, str]: (row, column) where row is like 'A', 'B' and column is like '01', '04'

        Raises:
            ValueError: If component format is invalid for this parser
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
        'grid_dimensions': (1, 1),  # Default grid dimensions (1x1) when not available
    }
    COMPONENT_VALUE_GETTERS: ClassVar[
        Dict[AllComponents, Callable[["MetadataHandler", Union[str, Path]], Optional[Dict[str, Optional[str]]]]]
    ] = {
        AllComponents.CHANNEL: lambda handler, plate_path: handler.get_channel_values(plate_path),
        AllComponents.WELL: lambda handler, plate_path: handler.get_well_values(plate_path),
        AllComponents.SITE: lambda handler, plate_path: handler.get_site_values(plate_path),
        AllComponents.Z_INDEX: lambda handler, plate_path: handler.get_z_index_values(plate_path),
    }

    def __init__(self):
        """Initialize metadata handler with VariableComponents enum."""
        self.component_enum = VariableComponents

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

    def get_component_values(
        self,
        plate_path: Union[str, Path],
        component_name: str,
    ) -> Optional[Dict[str, Optional[str]]]:
        """Get display values for a named microscope component."""
        component = AllComponents(component_name)
        if component not in self.COMPONENT_VALUE_GETTERS:
            raise ValueError(f"Unsupported metadata component {component_name!r}.")
        return self.COMPONENT_VALUE_GETTERS[component](self, plate_path)

    def get_image_files(self, plate_path: Union[str, Path], all_subdirs: bool = False) -> list[str]:
        """
        Get list of image files from OpenHCS metadata.

        Default implementation reads from openhcs_metadata.json after virtual workspace preparation.
        Derives image list from workspace_mapping keys if available, otherwise from image_files list.

        Subclasses can override if they need different behavior (e.g., OpenHCS reads directly from metadata).

        Args:
            plate_path: Path to the plate folder (str or Path)
            all_subdirs: If True, return files from all subdirectories. If False (default), return only main subdirectory files.

        Returns:
            List of image filenames with subdirectory prefix (e.g., "Images/file.tif" or "file.tif")

        Raises:
            TypeError: If plate_path is not a valid path type
            FileNotFoundError: If plate path does not exist or no metadata found
        """
        from pathlib import Path

        # Ensure plate_path is a Path object
        if isinstance(plate_path, str):
            plate_path = Path(plate_path)
        elif not isinstance(plate_path, Path):
            raise TypeError(f"Expected str or Path, got {type(plate_path).__name__}")

        # Ensure the path exists
        if not plate_path.exists():
            raise FileNotFoundError(f"Plate path does not exist: {plate_path}")

        # Read from OpenHCS metadata (unified approach for all microscopes)
        from openhcs.microscopes.openhcs import OpenHCSMetadataHandler
        import logging
        logger = logging.getLogger(__name__)

        openhcs_handler = OpenHCSMetadataHandler(self.filemanager)

        try:
            metadata = openhcs_handler._load_metadata_dict(plate_path)
            subdirs = metadata.get("subdirectories", {})
            logger.info(f"get_image_files: Found {len(subdirs)} subdirectories")

            if all_subdirs:
                # Collect files from ALL subdirectories
                all_files = []
                for subdir_key, subdir_data in subdirs.items():
                    # Prefer workspace_mapping keys (virtual paths) if available
                    if workspace_mapping := subdir_data.get("workspace_mapping"):
                        all_files.extend(workspace_mapping.keys())
                    else:
                        # Otherwise use image_files list
                        image_files = subdir_data.get("image_files", [])
                        all_files.extend(image_files)

                logger.info(f"get_image_files: Returning {len(all_files)} files from {len(subdirs)} subdirectories")
                return all_files
            else:
                # Return only main subdirectory files (default behavior)
                # Find main subdirectory
                main_subdir_key = next((key for key, data in subdirs.items() if data.get("main")), None)
                if not main_subdir_key:
                    main_subdir_key = next(iter(subdirs.keys()))

                logger.info(f"get_image_files: Using main subdirectory '{main_subdir_key}'")
                subdir_data = subdirs[main_subdir_key]

                # Prefer workspace_mapping keys (virtual paths) if available
                if workspace_mapping := subdir_data.get("workspace_mapping"):
                    logger.info(f"get_image_files: Returning {len(workspace_mapping)} files from workspace_mapping")
                    return list(workspace_mapping.keys())

                # Otherwise use image_files list
                image_files = subdir_data.get("image_files", [])
                logger.info(f"get_image_files: Returning {len(image_files)} files from image_files list")
                return image_files

        except Exception:
            # Fallback: no metadata yet, return empty list
            return []

    def parse_metadata(self, plate_path: Union[str, Path]) -> Dict[str, Dict[str, Optional[str]]]:
        """
        Parse all metadata using dynamic method resolution.

        This method iterates through VariableComponents and calls the corresponding
        abstract methods to collect all available metadata.

        Args:
            plate_path: Path to the plate folder (str or Path)

        Returns:
            Dict mapping component names to their key→name mappings
            Example: {"channel": {"1": "HOECHST 33342", "2": "Calcein"}}
        """
        result = {}
        for component in self.component_enum:
            component_name = component.value
            method_name = f"get_{component_name}_values"
            method = getattr(self, method_name)  # Let AttributeError bubble up
            values = method(plate_path)
            if values:
                result[component_name] = values
        return result


# ============================================================================
# Parser Registry Export
# ============================================================================
# Reference to the auto-created registry from FilenameParser metaclass
# The AutoRegisterMeta metaclass creates FilenameParser.__registry__ automatically
FILENAME_PARSERS = FilenameParser.__registry__
