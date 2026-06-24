"""
Microscope interfaces for openhcs.

This module provides abstract base classes for microscope-specific functionality,
including filename parsing and metadata handling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ClassVar, Dict, Mapping, Optional, TYPE_CHECKING, Tuple, Union
from openhcs.constants.constants import Backend, VariableComponents, AllComponents
from openhcs.core.components.parser_metaprogramming import (
    FilenameParseResult,
    GenericFilenameParser,
)
from metaclass_registry import AutoRegisterMeta, LazyDiscoveryDict
from polystore.streaming.viewer_transport import (
    ViewerFilenameParserABC,
    ViewerMetadataHandlerABC,
    ViewerMicroscopeHandlerABC,
)
from polystore.filemanager import FileManager

if TYPE_CHECKING:
    from openhcs.microscopes.openhcs import OpenHCSMetadata


@dataclass(frozen=True)
class MetadataComponentValueSet:
    """Named component metadata projection for display and export surfaces."""

    channels: Optional[Dict[str, Optional[str]]]
    wells: Optional[Dict[str, Optional[str]]]
    sites: Optional[Dict[str, Optional[str]]]
    z_indexes: Optional[Dict[str, Optional[str]]]
    timepoints: Optional[Dict[str, Optional[str]]]


@dataclass(frozen=True)
class AnalysisResultDirectory:
    """Named analysis-results directory declared by microscope metadata."""

    subdirectory_name: str
    path: Path

@dataclass(frozen=True)
class MetadataViewEntry:
    """One metadata object projected for UI/document consumers."""

    name: str
    object_instance: "OpenHCSMetadata"
    summary: str | None = None


@dataclass(frozen=True)
class MetadataViewDocument:
    """UI-neutral metadata document projection."""

    title: str
    entries: tuple[MetadataViewEntry, ...]
    selector_label: str = "Entry:"

    def __post_init__(self) -> None:
        if not self.entries:
            raise ValueError("MetadataViewDocument requires at least one entry.")


def list_relative_disk_image_files(
    filemanager: FileManager,
    plate_path: Union[str, Path],
    *,
    recursive: bool,
) -> list[str]:
    """List disk-backed image files relative to a declared plate root."""
    root_path = Path(plate_path)
    image_paths = filemanager.list_image_files(
        root_path,
        Backend.DISK.value,
        recursive=recursive,
    )
    relative_paths: list[str] = []
    for image_path in image_paths:
        path = Path(image_path)
        if path.is_absolute():
            relative_paths.append(path.relative_to(root_path).as_posix())
        else:
            relative_paths.append(path.as_posix())
    return sorted(relative_paths)

class FilenameParser(
    ViewerFilenameParserABC,
    GenericFilenameParser,
    metaclass=AutoRegisterMeta,
):
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
        self.pattern_format: str | None = None
        super().__init__(AllComponents)

    def semantic_identity(self) -> tuple[Hashable, ...]:
        """Return the parser semantics that affect filename parsing."""
        parser_type = type(self)
        return (
            parser_type.__module__,
            parser_type.__qualname__,
            self.pattern_format,
        )

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
    def parse_filename(self, filename: str) -> Optional[FilenameParseResult]:
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


class MetadataHandler(ViewerMetadataHandlerABC, ABC):
    """
    Abstract base class for handling microscope metadata.

    All metadata methods require str or Path objects for file paths.

    Subclasses must return required metadata from their declared metadata source
    or raise an exception explaining why the contract cannot be satisfied.
    """
    COMPONENT_VALUE_GETTERS: ClassVar[
        Dict[AllComponents, Callable[["MetadataHandler", Union[str, Path]], Optional[Dict[str, Optional[str]]]]]
    ] = {
        AllComponents.CHANNEL: lambda handler, plate_path: handler.get_channel_values(plate_path),
        AllComponents.WELL: lambda handler, plate_path: handler.get_well_values(plate_path),
        AllComponents.SITE: lambda handler, plate_path: handler.get_site_values(plate_path),
        AllComponents.Z_INDEX: lambda handler, plate_path: handler.get_z_index_values(plate_path),
        AllComponents.TIMEPOINT: lambda handler, plate_path: handler.get_timepoint_values(plate_path),
    }

    def __init__(self):
        """Initialize metadata handler with VariableComponents enum."""
        self.component_enum = VariableComponents

    def source_workspace_metadata_document(
        self,
        plate_path: Union[str, Path],
    ) -> Mapping[str, object] | None:
        """Return source-workspace metadata for handlers that own virtual mappings."""

        return None

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

    @abstractmethod
    def get_image_files(self, plate_path: Union[str, Path], all_subdirs: bool = False) -> list[str]:
        """
        Get image files exposed by this metadata handler.

        Subclasses own their format's file-listing authority; the base class
        must not infer another microscope's workspace metadata layout.
        """
        pass

    def analysis_result_directories(
        self,
        plate_path: Union[str, Path],
    ) -> tuple[AnalysisResultDirectory, ...]:
        """Return analysis-result directories declared by this metadata source."""
        return ()

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
            values = self.get_component_values(plate_path, component_name)
            if values:
                result[component_name] = values
        return result

    def component_value_set(self, plate_path: Union[str, Path]) -> MetadataComponentValueSet:
        """Return metadata components as a named projection instead of a dict bag."""
        return MetadataComponentValueSet(
            channels=self.get_channel_values(plate_path),
            wells=self.get_well_values(plate_path),
            sites=self.get_site_values(plate_path),
            z_indexes=self.get_z_index_values(plate_path),
            timepoints=self.get_timepoint_values(plate_path),
        )

    def build_metadata_view_document(
        self,
        plate_path: Union[str, Path],
        microscope_handler: ViewerMicroscopeHandlerABC,
    ) -> MetadataViewDocument:
        """Project this handler's metadata into the standard read-only UI document."""
        from openhcs.microscopes.openhcs import OpenHCSMetadata

        component_values = self.component_value_set(plate_path)
        grid_dims = self.get_grid_dimensions(plate_path)
        pixel_size = self.get_pixel_size(plate_path)
        image_files = self.get_image_files(plate_path)

        parser = microscope_handler.parser
        if parser is None:
            raise ValueError(
                f"{microscope_handler.__class__.__name__} cannot build metadata view without a parser."
            )

        metadata = OpenHCSMetadata(
            microscope_handler_name=microscope_handler.microscope_type,
            source_filename_parser_name=parser.__class__.__name__,
            grid_dimensions=list(grid_dims),
            pixel_size=pixel_size,
            image_files=image_files,
            channels=component_values.channels,
            wells=component_values.wells,
            sites=component_values.sites,
            z_indexes=component_values.z_indexes,
            timepoints=component_values.timepoints,
            available_backends={"disk": True},
            main=None,
        )
        title = f"Metadata - {microscope_handler.microscope_type}"
        return MetadataViewDocument(
            title=title,
            entries=(
                MetadataViewEntry(
                    name=microscope_handler.microscope_type,
                    object_instance=metadata,
                    summary=f"Image files: {len(metadata.image_files)} (hidden)",
                ),
            ),
        )


class DiskImageFileListingMetadataHandler(MetadataHandler):
    """Metadata handler mixin for formats whose images are disk files under the plate root."""

    filemanager: FileManager

    def get_image_files(self, plate_path: Union[str, Path], all_subdirs: bool = False) -> list[str]:
        return list_relative_disk_image_files(
            self.filemanager,
            plate_path,
            recursive=True,
        )


# ============================================================================
# Parser Registry Export
# ============================================================================
# Reference to the auto-created registry from FilenameParser metaclass
# The AutoRegisterMeta metaclass creates FilenameParser.__registry__ automatically
FILENAME_PARSERS = FilenameParser.__registry__
