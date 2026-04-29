"""
Well and channel discovery via OpenHCS MicroscopeHandler API.

This module provides functions to:
- Discover all well/channel combinations in a plate
- Group image files by well/channel for Z-stack loading

Invariants:
- Discovery returns data structures, never triggers processing
- All return types are immutable (frozen dataclasses, tuples)
- Single source of truth: uses MicroscopeHandler API + FilenameParser
- Parsed components preserved for output filename construction
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional, Tuple


@dataclass(frozen=True)
class WellChannelKey:
    """
    Immutable identifier for a well/channel combination.

    Populated from OpenHCS metadata and FilenameParser.
    """

    well_id: str
    channel_id: str
    channel_name: str
    site_id: Optional[str] = None
    parsed_components: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash((self.well_id, self.channel_id))


def get_microscope_handler(plate_path: Path):
    """
    Get the appropriate microscope handler for a plate.

    Args:
        plate_path: Path to the plate directory

    Returns:
        MicroscopeHandler instance
    """
    from openhcs.microscopes.openhcs import OpenHCSMicroscopeHandler
    from polystore.filemanager import FileManager
    from polystore.base import storage_registry, ensure_storage_registry

    ensure_storage_registry()
    filemanager = FileManager(storage_registry)

    handler = OpenHCSMicroscopeHandler(filemanager=filemanager)
    handler.plate_folder = Path(plate_path)

    return handler


def discover_well_channels(
    plate_path: Path,
    microscope_handler=None,
    channel_filter: FrozenSet[str] = None,
    include_mode: bool = False,
) -> Tuple[WellChannelKey, ...]:
    """
    Discover all well/channel combinations in a plate.

    Uses MicroscopeHandler.metadata_handler for channel names.
    Uses MicroscopeHandler.parser for component extraction.

    Args:
        plate_path: Path to the plate directory
        microscope_handler: Optional pre-configured handler
        channel_filter: Set of channel IDs to filter
        include_mode: If True, include ONLY these channels. If False, exclude these.

    Returns:
        Immutable tuple of WellChannelKey instances
    """
    plate_path = Path(plate_path)

    if microscope_handler is None:
        microscope_handler = get_microscope_handler(plate_path)

    metadata = microscope_handler.metadata_handler._load_metadata(plate_path)

    image_files = metadata.get("image_files", [])
    if not image_files:
        subdirs = metadata.get("subdirectories", {})
        for subdir_data in subdirs.values():
            if "image_files" in subdir_data:
                image_files = subdir_data["image_files"]
                break

    channel_map = microscope_handler.metadata_handler.get_channel_values(plate_path)
    if channel_map is None:
        channel_map = {}

    seen_keys = set()
    well_channels = []

    for filename in image_files:
        parsed = microscope_handler.parser.parse_filename(filename)
        if parsed is None:
            continue

        well_id = _extract_well_id(parsed)
        channel_id = str(parsed.get("channel", ""))
        site_id = parsed.get("site")

        if not well_id or not channel_id:
            continue

        if channel_filter:
            if include_mode:
                if channel_id not in channel_filter:
                    continue
            else:
                if channel_id in channel_filter:
                    continue

        key = (well_id, channel_id)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        channel_name = channel_map.get(channel_id, f"Channel {channel_id}")

        well_channel = WellChannelKey(
            well_id=well_id,
            channel_id=channel_id,
            channel_name=channel_name if channel_name else f"Channel {channel_id}",
            site_id=str(site_id) if site_id else None,
            parsed_components=parsed,
        )

        well_channels.append(well_channel)

    well_channels.sort(key=lambda x: (x.well_id, x.channel_id))

    return tuple(well_channels)


def group_z_slices_by_well_channel(
    plate_path: Path, well_keys: Tuple[WellChannelKey, ...], microscope_handler=None
) -> Dict[WellChannelKey, Tuple[Path, ...]]:
    """
    Group image files by well/channel for Z-stack loading.

    Uses FilenameParser.parse_filename() to extract components from each file.

    Args:
        plate_path: Path to the plate directory
        well_keys: Tuple of WellChannelKey instances
        microscope_handler: Optional pre-configured handler

    Returns:
        Dict mapping WellChannelKey to tuple of file paths (ordered by Z-index)
    """
    plate_path = Path(plate_path)

    if microscope_handler is None:
        microscope_handler = get_microscope_handler(plate_path)

    metadata = microscope_handler.metadata_handler._load_metadata(plate_path)
    image_files = metadata.get("image_files", [])
    if not image_files:
        subdirs = metadata.get("subdirectories", {})
        for subdir_data in subdirs.values():
            if "image_files" in subdir_data:
                image_files = subdir_data["image_files"]
                break

    key_lookup = {(wk.well_id, wk.channel_id): wk for wk in well_keys}

    grouped = {}

    for filename in image_files:
        parsed = microscope_handler.parser.parse_filename(filename)
        if parsed is None:
            continue

        well_id = _extract_well_id(parsed)
        channel_id = str(parsed.get("channel", ""))

        key = (well_id, channel_id)
        if key not in key_lookup:
            continue

        well_key = key_lookup[key]
        file_path = plate_path / filename

        if well_key not in grouped:
            grouped[well_key] = []

        z_index = parsed.get("z_index", 0)
        grouped[well_key].append((z_index, file_path))

    result = {}
    for well_key, files in grouped.items():
        files.sort(key=lambda x: x[0])
        result[well_key] = tuple(f for _, f in files)

    return result


def _extract_well_id(parsed: Dict[str, Any]) -> Optional[str]:
    """
    Extract well ID from parsed filename components.

    Handles various formats:
    - row + col: "01" + "06" -> "r01c06"
    - well: "A01" -> "A01"
    """
    if "well" in parsed:
        return str(parsed["well"])

    row = parsed.get("row")
    col = parsed.get("col")

    if row is not None and col is not None:
        return f"r{int(row):02d}c{int(col):02d}"

    return None
