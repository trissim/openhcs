"""
BBBC (Broad Bioimage Benchmark Collection) microscope implementations.

This module provides handlers for BBBC datasets in different formats:
- BBBC021: ImageXpress-like format with UUID, files in Week*/Week*_##### subdirectories
- BBBC038: Simple hex ID filenames in stage1_train/{ImageId}/images/ subdirectories

Each dataset gets its own handler following the established MicroscopeHandler pattern.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Type

from openhcs.constants.constants import Backend
from openhcs.microscopes.microscope_base import MicroscopeHandler
from openhcs.microscopes.microscope_interfaces import FilenameParser, MetadataHandler
from openhcs.microscopes.tiff_metadata_mixin import TiffPixelSizeMixin
from openhcs.microscopes.detect_mixins import MetadataDetectMixin
from polystore.exceptions import MetadataNotFoundError
from polystore.filemanager import FileManager

logger = logging.getLogger(__name__)


# ============================================================================
# BBBC021 Handler (ImageXpress-like with UUID, in Week subfolders)
# ============================================================================

class BBBC021FilenameParser(FilenameParser):
    """
    Parser for BBBC021 dataset filenames.

    Format: {Well}_s{Site}_w{Channel}{UUID}.tif
    Example: G10_s1_w1BEDC2073-A983-4B98-95E9-84466707A25D.tif

    Components:
    - Well: Alphanumeric plate coordinate (e.g., A01, G10, P24)
    - Site: Numeric site/field ID (e.g., 1, 2, 3)
    - Channel: Single digit channel ID (1=DAPI, 2=Tubulin, 4=Actin)
    - UUID: Hex identifier with dashes (ignored for parsing, but part of filename)
    - z_index: Not in filename, defaults to 1
    - timepoint: Not in filename, defaults to 1

    Note: Channel 3 is not used in BBBC021 (only 1, 2, 4).
    """

    # Pattern matches both original and virtual workspace filenames:
    # Original: G10_s1_w1{UUID}.tif
    # Virtual:  G10_s1_w1_z001_t001.tif
    _pattern = re.compile(
        r'^.*?'                  # Optional prefix (non-greedy)
        r'([A-P][0-9]{2})'       # Well: letter A-P + two digits
        r'_s(\d+|\{[^\}]*\})'    # Site: _s + digits or placeholder
        r'_w(\d|\{[^\}]*\})'     # Channel: _w + single digit or placeholder
        r'(?:_z(\d+|\{[^\}]*\}))?'  # Optional z
        r'(?:_t(\d+|\{[^\}]*\}))?'  # Optional timepoint
        r'([A-F0-9-]*)'          # Optional UUID
        r'(\.\w+)$',             # Extension
        re.IGNORECASE
    )

    def __init__(self, filemanager=None, pattern_format=None):
        super().__init__()
        self.filemanager = filemanager
        self.pattern_format = pattern_format

    @classmethod
    def can_parse(cls, filename: Union[str, Any]) -> bool:
        """Check if filename matches BBBC021 pattern."""
        basename = Path(str(filename)).name
        return cls._pattern.match(basename) is not None

    def parse_filename(self, filename: Union[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parse BBBC021 filename into components.

        Args:
            filename: Filename to parse

        Returns:
            Dict with keys: well, site, channel, z_index, timepoint, extension
            Or None if parsing fails
        """
        basename = Path(str(filename)).name
        match = self._pattern.match(basename)

        if not match:
            logger.debug("Could not parse BBBC021 filename: %s", filename)
            return None

        well, site_str, channel_str, z_str, t_str, uuid, ext = match.groups()

        def parse_component(value: str | None) -> int | None:
            if not value or "{" in value:
                return None
            return int(value)

        return {
            'well': well,
            'site': parse_component(site_str),
            'channel': parse_component(channel_str),
            'z_index': parse_component(z_str),
            'timepoint': parse_component(t_str),
            'extension': ext,
        }

    def extract_component_coordinates(self, component_value: str) -> Tuple[str, str]:
        """
        Extract row/column from well identifier.

        Args:
            component_value: Well like 'A01', 'G10', etc.

        Returns:
            (row, column) tuple like ('A', '01'), ('G', '10')
        """
        if not component_value or len(component_value) < 2:
            raise ValueError(f"Invalid well format: {component_value}")

        row = component_value[0]  # First character (letter)
        col = component_value[1:]  # Remaining digits

        if not row.isalpha() or not col.isdigit():
            raise ValueError(f"Invalid BBBC021 well format: {component_value}. Expected format like 'A01', 'G10'")

        return (row, col)

    def construct_filename(
        self,
        extension: str = '.tif',
        site_padding: int = 1,  # BBBC021 uses single digits for sites
        z_padding: int = 3,
        timepoint_padding: int = 3,
        **component_values
    ) -> str:
        """
        Construct BBBC021 filename from components for virtual workspace.

        Note: UUID is NOT reconstructed. Virtual workspace filenames include
        ALL components (z_index, timepoint) even if not in original filenames.
        This ensures consistent pattern discovery.

        Args:
            well: Well ID (e.g., 'A01', 'G10')
            site: Site number
            channel: Channel number
            z_index: Z-index (defaults to 1)
            timepoint: Timepoint (defaults to 1)
            extension: File extension
            **component_values: Other component values

        Returns:
            Filename: {Well}_s{Site}_w{Channel}_z{Z}_t{T}.tif
        """
        well = component_values.get('well')
        site = component_values.get('site')
        channel = component_values.get('channel')
        z_index = component_values.get('z_index')
        timepoint = component_values.get('timepoint')

        if not well:
            raise ValueError("Well ID cannot be empty or None.")

        # Default ALL components to 1 (required for virtual workspace)
        site = 1 if site is None else site
        channel = 1 if channel is None else channel
        z_index = 1 if z_index is None else z_index
        timepoint = 1 if timepoint is None else timepoint

        # Build filename parts
        parts = [well]

        # Site
        if isinstance(site, str):
            parts.append(f"_s{site}")
        else:
            parts.append(f"_s{site:0{site_padding}d}")

        # Channel (no padding)
        parts.append(f"_w{channel}")

        # Z-index (ALWAYS include for virtual workspace)
        if isinstance(z_index, str):
            parts.append(f"_z{z_index}")
        else:
            parts.append(f"_z{z_index:0{z_padding}d}")

        # Timepoint (ALWAYS include for virtual workspace)
        if isinstance(timepoint, str):
            parts.append(f"_t{timepoint}")
        else:
            parts.append(f"_t{timepoint:0{timepoint_padding}d}")

        return "".join(parts) + extension


class BBBC021MetadataHandler(TiffPixelSizeMixin, MetadataHandler):
    """
    Metadata handler for BBBC021 dataset.

    BBBC021 public mirror ships only TIFFs; we extract metadata from TIFF tags.
    """

    def __init__(self, filemanager: FileManager):
        super().__init__()
        self.filemanager = filemanager

    def find_metadata_file(self, plate_path: Union[str, Path]) -> Optional[Path]:
        """
        BBBC021 ship we have contains no separate metadata files; rely solely on TIFFs.
        Ensure caller pointed at the expected plate directory.
        """
        plate_path = Path(plate_path)
        if plate_path.name != "Week1_22123":
            raise MetadataNotFoundError(
                f"BBBC021 plate must be the Week1_22123 directory, got '{plate_path.name}'"
            )
        return None

    def get_grid_dimensions(self, plate_path: Union[str, Path]) -> Tuple[int, int]:
        """No stitching grid needed."""
        return (1, 1)

    def get_pixel_size(self, plate_path: Union[str, Path]) -> float:
        return self._pixel_size_from_tiff(plate_path, self.filemanager)

    def get_channel_values(self, plate_path: Union[str, Path]) -> Optional[Dict[str, Optional[str]]]:
        # Derive channel names from TIFF tag (if present). May return {'1': 'DAPI'} etc.
        return self._channel_from_tiff(plate_path, self.filemanager)

    def get_well_values(self, plate_path: Union[str, Path]) -> Optional[Dict[str, Optional[str]]]:
        """Get well metadata - would require parsing CSV."""
        return None

    def get_site_values(self, plate_path: Union[str, Path]) -> Optional[Dict[str, Optional[str]]]:
        """Get site metadata - none available."""
        return None

    def get_z_index_values(self, plate_path: Union[str, Path]) -> Optional[Dict[str, Optional[str]]]:
        """Get z-index metadata - BBBC021 has no Z-stacks."""
        return None

    def get_timepoint_values(self, plate_path: Union[str, Path]) -> Optional[Dict[str, Optional[str]]]:
        """Single timepoint dataset."""
        return None


class BBBC021Handler(MicroscopeHandler):
    """
    Microscope handler for BBBC021 dataset.

    BBBC021: Human MCF7 cells from compound profiling experiment.
    Format: ImageXpress-like with {Well}_s{Site}_w{Channel}{UUID}.tif pattern.
    Files are in Week#/Week#_#####/ subdirectories.
    """

    _microscope_type = 'bbbc021'
    _metadata_handler_class = BBBC021MetadataHandler

    @classmethod
    def detect(cls, plate_folder: Path, filemanager: FileManager) -> bool:
        """
        Detect via metadata CSV first, else via filename parser match.
        """
        plate_folder = Path(plate_folder)
        # Filename signal only (no external metadata shipped)
        try:
            files = filemanager.list_files(plate_folder, Backend.DISK.value, recursive=True)
            parser = BBBC021FilenameParser()
            for f in files:
                name = Path(f).name
                if name.lower().endswith((".tif", ".tiff")) and parser.can_parse(name):
                    return True
        except Exception:
            return False
        return False

    def __init__(self, filemanager: FileManager, pattern_format: Optional[str] = None):
        self.parser = BBBC021FilenameParser(filemanager, pattern_format)
        self.metadata_handler = BBBC021MetadataHandler(filemanager)
        super().__init__(parser=self.parser, metadata_handler=self.metadata_handler)

    @property
    def root_dir(self) -> str:
        """
        BBBC021 virtual workspace is at plate root.

        Files are physically in Week#/Week#_##### subdirectories,
        but virtually flattened to plate root.
        """
        return "."

    @property
    def microscope_type(self) -> str:
        return 'bbbc021'

    @property
    def metadata_handler_class(self) -> Type[MetadataHandler]:
        return BBBC021MetadataHandler

    @property
    def compatible_backends(self) -> List[Backend]:
        """BBBC021 uses standard DISK backend."""
        return [Backend.DISK]

    def _build_virtual_mapping(self, plate_path: Path, filemanager: FileManager) -> Path:
        """
        Build virtual workspace mapping for BBBC021.

        Flattens Week#/Week#_##### subdirectory structure to plate root,
        and adds missing z_index and timepoint components to filenames.

        Args:
            plate_path: Path to plate directory
            filemanager: FileManager instance

        Returns:
            Path to plate root
        """
        plate_path = Path(plate_path)

        logger.info(f"🔄 BUILDING VIRTUAL MAPPING: BBBC021 folder flattening for {plate_path}")

        # Initialize mapping dict (PLATE-RELATIVE paths)
        workspace_mapping = {}

        # Recursively find all .tif files
        image_files = filemanager.list_image_files(plate_path, Backend.DISK.value, recursive=True)

        for file_path in image_files:
            # Get filename
            if isinstance(file_path, str):
                filename = os.path.basename(file_path)
            elif isinstance(file_path, Path):
                filename = file_path.name
            else:
                continue

            # Parse original filename
            metadata = self.parser.parse_filename(filename)
            if not metadata:
                logger.warning(f"Could not parse BBBC021 filename: {filename}")
                continue

            # Add default z_index and timepoint (missing from original filenames)
            if metadata['z_index'] is None:
                metadata['z_index'] = 1
            if metadata['timepoint'] is None:
                metadata['timepoint'] = 1

            # Reconstruct filename with all components (standardized)
            new_filename = self.parser.construct_filename(**metadata)

            # Build PLATE-RELATIVE virtual path (at plate root)
            virtual_relative = new_filename

            # Build PLATE-RELATIVE real path (in subfolder)
            real_relative = Path(file_path).relative_to(plate_path).as_posix()

            # Add to mapping
            workspace_mapping[virtual_relative] = real_relative
            logger.debug(f"  Mapped: {virtual_relative} → {real_relative}")

        logger.info(f"Built {len(workspace_mapping)} virtual path mappings for BBBC021")

        # Save virtual workspace mapping
        self._save_virtual_workspace_metadata(plate_path, workspace_mapping)

        return plate_path


# ============================================================================
# BBBC038 Handler (Kaggle Nuclei - Hex ID Format)
# ============================================================================

class BBBC038FilenameParser(FilenameParser):
    """
    Parser for BBBC038 dataset (Kaggle 2018 Data Science Bowl).

    Format: {HexID}.png
    Example: 0a7e06cd488667b8fe53a1521d88ab3f4e8d8a05b5663e89dc5df7b02ca93f38.png

    BBBC038 uses simple hex string identifiers as filenames.
    Each ImageId represents a unique image (treated as a unique "well").

    Organization: stage1_train/{ImageId}/images/{ImageId}.png
    Parser only sees the filename, not the full path structure.
    """

    # Pattern: hex string + .png extension
    _pattern = re.compile(r'^([a-f0-9]+)\.png$', re.IGNORECASE)

    def __init__(self, filemanager=None, pattern_format=None):
        super().__init__()
        self.filemanager = filemanager
        self.pattern_format = pattern_format

    @classmethod
    def can_parse(cls, filename: Union[str, Any]) -> bool:
        """Check if filename matches BBBC038 pattern (hex ID + .png)."""
        basename = Path(str(filename)).name
        return cls._pattern.match(basename) is not None

    def parse_filename(self, filename: Union[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parse BBBC038 filename into components.

        Args:
            filename: Filename to parse

        Returns:
            Dict with well=ImageId, site/channel/z all fixed at 1
            Or None if parsing fails
        """
        basename = Path(str(filename)).name
        match = self._pattern.match(basename)

        if not match:
            logger.debug("Could not parse BBBC038 filename: %s", filename)
            return None

        image_id = match.group(1)

        return {
            'well': image_id,  # ImageId is the well identifier
            'site': 1,          # Single image per ID
            'channel': 1,       # Single channel (nuclei stain)
            'z_index': None,    # No Z-stacks, will default to 1
            'timepoint': None,  # No timepoints, will default to 1
            'extension': '.png',
        }

    def extract_component_coordinates(self, component_value: str) -> Tuple[str, str]:
        """
        Extract coordinates from ImageId.

        BBBC038 has no spatial grid layout - ImageIds are arbitrary identifiers.
        Split the hex string for display purposes only.

        Args:
            component_value: ImageId (hex string)

        Returns:
            (first_half, second_half) of the hex ID
        """
        if not component_value:
            raise ValueError("Invalid ImageId: empty")

        mid = len(component_value) // 2
        return (component_value[:mid], component_value[mid:])

    def construct_filename(
        self,
        extension: str = '.png',
        **component_values
    ) -> str:
        """
        Construct BBBC038 filename from components.

        Args:
            well: ImageId (hex string)
            extension: File extension
            **component_values: Other components (ignored)

        Returns:
            Filename string: {ImageId}.png
        """
        image_id = component_values.get('well')

        if not image_id:
            raise ValueError("ImageId (well) cannot be empty or None.")

        return f"{image_id}{extension}"


class BBBC038MetadataHandler(MetadataHandler):
    """
    Metadata handler for BBBC038 (Kaggle nuclei dataset).

    Metadata comes from:
    - metadata.xlsx
    - stage1_train_labels.csv (run-length encoded masks)
    - stage1_solution.csv (evaluation metrics)
    """

    def __init__(self, filemanager: FileManager):
        super().__init__()
        self.filemanager = filemanager

    def find_metadata_file(self, plate_path: Union[str, Path]) -> Path:
        """Find metadata.xlsx or stage1_train_labels.csv."""
        plate_path = Path(plate_path)

        candidates = [
            plate_path / "metadata.xlsx",
            plate_path / "stage1_train_labels.csv",
            plate_path.parent / "metadata.xlsx",
            plate_path.parent / "stage1_train_labels.csv",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise MetadataNotFoundError(
            f"BBBC038 metadata not found in {plate_path}. "
            "Download from https://data.broadinstitute.org/bbbc/BBBC038/"
        )

    def get_grid_dimensions(self, plate_path: Union[str, Path]) -> Tuple[int, int]:
        """BBBC038 has no grid layout - each image is independent."""
        return (1, 1)

    def get_pixel_size(self, plate_path: Union[str, Path]) -> float:
        """BBBC038 pixel size varies across different imaging conditions."""
        return 1.0  # No standard pixel size (diverse sources)

    def get_channel_values(self, plate_path: Union[str, Path]) -> Optional[Dict[str, Optional[str]]]:
        """BBBC038 is single-channel (nuclei stain)."""
        return {"1": "Nuclei"}

    def get_well_values(self, plate_path: Union[str, Path]) -> Optional[Dict[str, Optional[str]]]:
        return None

    def get_site_values(self, plate_path: Union[str, Path]) -> Optional[Dict[str, Optional[str]]]:
        return None

    def get_z_index_values(self, plate_path: Union[str, Path]) -> Optional[Dict[str, Optional[str]]]:
        return None


class BBBC038Handler(MetadataDetectMixin, MicroscopeHandler):
    """
    Microscope handler for BBBC038 dataset (Kaggle nuclei, PNG format).

    BBBC038: Nuclei from diverse organisms and imaging conditions.
    Format: {HexID}.png in stage1_train/{ImageId}/images/ subdirectories.
    """

    _microscope_type = 'bbbc038'
    _metadata_handler_class = BBBC038MetadataHandler

    @classmethod
    def detect(cls, plate_folder: Path, filemanager: FileManager) -> bool:
        """
        Detect BBBC038 by presence of stage1_train folder with PNGs.
        """
        stage1 = Path(plate_folder) / "stage1_train"
        if not stage1.exists():
            return False
        try:
            files = filemanager.list_files(stage1, Backend.DISK.value, pattern="*.png", recursive=True)
            return len(files) > 0
        except Exception:
            return False

    def __init__(self, filemanager: FileManager, pattern_format: Optional[str] = None):
        self.parser = BBBC038FilenameParser(filemanager, pattern_format)
        self.metadata_handler = BBBC038MetadataHandler(filemanager)
        super().__init__(parser=self.parser, metadata_handler=self.metadata_handler)

    @property
    def root_dir(self) -> str:
        """
        BBBC038 virtual workspace is at stage1_train directory.

        Images are in stage1_train/{ImageId}/images/ subdirectories.
        """
        return "stage1_train"

    @property
    def microscope_type(self) -> str:
        return 'bbbc038'

    @property
    def metadata_handler_class(self) -> Type[MetadataHandler]:
        return BBBC038MetadataHandler

    @property
    def compatible_backends(self) -> List[Backend]:
        return [Backend.DISK]

    def _build_virtual_mapping(self, plate_path: Path, filemanager: FileManager) -> Path:
        """
        Build virtual workspace mapping for BBBC038.

        Flattens stage1_train/{ImageId}/images/ structure.
        Since filenames are already unique (ImageId), just flatten to stage1_train/.

        Args:
            plate_path: Path to plate directory (contains stage1_train/)
            filemanager: FileManager instance

        Returns:
            Path to stage1_train directory
        """
        plate_path = Path(plate_path)
        stage1_path = plate_path / "stage1_train"

        if not stage1_path.exists():
            logger.warning(f"stage1_train directory not found in {plate_path}")
            return plate_path

        logger.info(f"🔄 BUILDING VIRTUAL MAPPING: BBBC038 folder flattening for {plate_path}")

        # Initialize mapping dict (PLATE-RELATIVE paths)
        workspace_mapping = {}

        # Find all .png files in images/ subdirectories
        image_files = filemanager.list_image_files(stage1_path, Backend.DISK.value, recursive=True)

        for file_path in image_files:
            # Only process files in images/ subdirectories (skip masks/)
            if '/images/' not in str(file_path):
                continue

            # Get filename
            if isinstance(file_path, str):
                filename = os.path.basename(file_path)
            elif isinstance(file_path, Path):
                filename = file_path.name
            else:
                continue

            # Parse filename
            metadata = self.parser.parse_filename(filename)
            if not metadata:
                logger.warning(f"Could not parse BBBC038 filename: {filename}")
                continue

            # Filename is already correct (ImageId.png)
            # Just flatten to stage1_train/ directory

            # Build PLATE-RELATIVE virtual path (in stage1_train/)
            virtual_relative = (Path("stage1_train") / filename).as_posix()

            # Build PLATE-RELATIVE real path (in stage1_train/{ImageId}/images/)
            real_relative = Path(file_path).relative_to(plate_path).as_posix()

            # Add to mapping
            workspace_mapping[virtual_relative] = real_relative
            logger.debug(f"  Mapped: {virtual_relative} → {real_relative}")

        logger.info(f"Built {len(workspace_mapping)} virtual path mappings for BBBC038")

        # Save virtual workspace mapping
        self._save_virtual_workspace_metadata(plate_path, workspace_mapping)

        return stage1_path
