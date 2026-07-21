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
from typing import ClassVar, Dict, Mapping, Optional, TYPE_CHECKING, Tuple, Union
from openhcs.constants.constants import Backend, AllComponents
from openhcs.core.components.parser_metaprogramming import (
    FilenameParseResult,
    GenericFilenameParser,
)
from metaclass_registry import AutoRegisterMeta
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

    def component_values(
        self,
    ) -> tuple[tuple[AllComponents, Optional[Dict[str, Optional[str]]]], ...]:
        """Return component metadata in the order declared by OpenHCS axes."""
        return (
            (AllComponents.CHANNEL, self.channels),
            (AllComponents.WELL, self.wells),
            (AllComponents.SITE, self.sites),
            (AllComponents.Z_INDEX, self.z_indexes),
            (AllComponents.TIMEPOINT, self.timepoints),
        )

    def values_for(
        self,
        component: AllComponents,
    ) -> Optional[Dict[str, Optional[str]]]:
        """Return metadata values for one OpenHCS component declaration."""
        for declared_component, values in self.component_values():
            if declared_component == component:
                return values
        raise ValueError(f"Unsupported metadata component {component.value!r}.")


@dataclass(frozen=True)
class AnalysisResultDirectory:
    """Named analysis-results directory declared by microscope metadata."""

    subdirectory_name: str
    path: Path


class MetadataArtifactProvider(ABC, metaclass=AutoRegisterMeta):
    """Nominal metadata artifact resolver owned by microscope metadata."""

    __registry_key__ = "artifact_name"
    __skip_if_no_key__ = True

    artifact_name: ClassVar[str | None] = None
    description: ClassVar[str] = ""

    @classmethod
    def require_artifact_name(cls) -> str:
        artifact_name = cls.artifact_name
        if artifact_name is None:
            raise ValueError(f"{cls.__name__} must declare artifact_name.")
        return artifact_name

    @classmethod
    def registered_provider_types(cls) -> tuple[type["MetadataArtifactProvider"], ...]:
        registered = set(cls.__registry__.values())
        ordered: list[type[MetadataArtifactProvider]] = []
        seen: set[type[MetadataArtifactProvider]] = set()

        def visit(owner: type[MetadataArtifactProvider]) -> None:
            for child in owner.__subclasses__():
                visit(child)
            if owner in registered and owner not in seen:
                ordered.append(owner)
                seen.add(owner)

        visit(cls)
        return tuple(ordered)

    @classmethod
    def provider_type_for_handler(
        cls,
        artifact_name: str,
        handler: "MetadataHandler",
    ) -> type["MetadataArtifactProvider"] | None:
        for provider_type in cls.registered_provider_types():
            if (
                provider_type.require_artifact_name() == artifact_name
                and provider_type.supports_handler(handler)
            ):
                return provider_type
        return None

    @classmethod
    @abstractmethod
    def supports_handler(cls, handler: "MetadataHandler") -> bool:
        """Return whether this provider can resolve artifacts for a handler."""

    @abstractmethod
    def resolve(
        self,
        handler: "MetadataHandler",
        plate_path: Union[str, Path],
    ) -> object:
        """Return this metadata artifact for a plate."""


class GridDimensionsMetadataArtifactProvider(MetadataArtifactProvider):
    """Provide stitching grid dimensions from microscope metadata."""

    artifact_name = "grid_dimensions"
    description = "Grid dimensions (num_rows, num_cols) for position generation."

    @classmethod
    def supports_handler(cls, handler: "MetadataHandler") -> bool:
        return isinstance(handler, MetadataHandler)

    def resolve(
        self,
        handler: "MetadataHandler",
        plate_path: Union[str, Path],
    ) -> tuple[int, int]:
        return handler.get_grid_dimensions(plate_path)


class PixelSizeMetadataArtifactProvider(MetadataArtifactProvider):
    """Provide plate pixel size for physical-unit analysis backends."""

    artifact_name = "pixel_size"
    description = "Pixel size in micrometers per pixel for physical-unit analysis."

    @classmethod
    def supports_handler(cls, handler: "MetadataHandler") -> bool:
        return isinstance(handler, MetadataHandler)

    def resolve(
        self,
        handler: "MetadataHandler",
        plate_path: Union[str, Path],
    ) -> float:
        return handler.get_pixel_size(plate_path)


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

    def source_workspace_metadata_document(
        self,
        plate_path: Union[str, Path],
    ) -> Mapping[str, object] | None:
        """Return source-workspace metadata for handlers that own virtual mappings."""

        return None

    def physical_source_paths(
        self,
        plate_path: Union[str, Path],
    ) -> tuple[Path, ...]:
        """Return physical source files for format-recognition diagnostics."""

        root = Path(plate_path)
        return tuple(
            path if path.is_absolute() else root / path
            for value in self.get_image_files(plate_path, all_subdirs=True)
            for path in (Path(value),)
        )

    def source_diagnostics(
        self,
        plate_path: Union[str, Path],
    ) -> tuple[Mapping[str, object], ...]:
        """Return structured source-level diagnostics owned by this handler."""

        del plate_path
        return ()

    def can_resolve_metadata_artifact(self, artifact_name: str) -> bool:
        """Return whether this handler can provide a declared metadata artifact."""
        return (
            MetadataArtifactProvider.provider_type_for_handler(
                artifact_name,
                self,
            )
            is not None
        )

    def resolve_metadata_artifact(
        self,
        artifact_name: str,
        plate_path: Union[str, Path],
    ) -> object:
        """Resolve a declared metadata artifact for this plate."""
        provider_type = MetadataArtifactProvider.provider_type_for_handler(
            artifact_name,
            self,
        )
        if provider_type is None:
            raise ValueError(f"No metadata artifact provider for {artifact_name!r}.")
        return provider_type().resolve(self, plate_path)

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
        return self.component_value_set(plate_path).values_for(
            AllComponents(component_name)
        )

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
        component_values = self.component_value_set(plate_path)
        for component, values in component_values.component_values():
            if values:
                result[component.value] = values
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
            source_diagnostics=[
                dict(diagnostic)
                for diagnostic in self.source_diagnostics(plate_path)
            ] or None,
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
