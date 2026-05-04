"""Typed writer options.

Greenfield design:
- the abstraction boundary is the *output format* (CSV/JSON/ROI_ZIP/TIFF/etc)
- options types are used for dispatch (metaprogramming-friendly)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class FileOutputOptions:
    """Common filename + strip behavior."""

    filename_suffix: str = ""
    strip_roi_suffix: bool = False
    strip_pkl: bool = True


@dataclass(frozen=True)
class SourceOptions:
    """Select a sub-value from the input before writing.

    The selector is a dot-path. For dicts, keys are used. For objects, attributes.
    Example: source="branch_data".
    """

    source: Optional[str] = None


@dataclass(frozen=True)
class TabularExtractionOptions:
    """Generic extraction options for tabular writers."""

    fields: Optional[List[str]] = None
    row_field: Optional[str] = None
    row_columns: Dict[str, str] = field(default_factory=dict)
    row_unpacker: Optional[Callable[[Any], List[Dict[str, Any]]]] = None


@dataclass(frozen=True)
class CsvOptions(FileOutputOptions, SourceOptions, TabularExtractionOptions):
    """CSV writer options."""

    filename_suffix: str = "_details.csv"


@dataclass(frozen=True)
class JsonOptions(FileOutputOptions, SourceOptions, TabularExtractionOptions):
    """JSON writer options."""

    filename_suffix: str = ".json"
    indent: int = 2
    wrap_list: bool = False


@dataclass(frozen=True)
class ROIOptions(FileOutputOptions, SourceOptions):
    """ROI ZIP writer options."""

    min_area: int = 10
    extract_contours: bool = True
    roi_suffix: str = "_rois.roi.zip"
    summary_suffix: str = "_segmentation_summary.txt"


@dataclass(frozen=True)
class TiffStackOptions(FileOutputOptions, SourceOptions):
    """TIFF stack writer options (per-slice TIFF + summary)."""

    normalize_uint8: bool = False
    preserve_channels_last_color: bool = True
    slice_pattern: str = "_slice_{index:03d}.tif"
    summary_suffix: str = "_summary.txt"
    empty_summary: str = "No images generated (empty data)\n"


@dataclass(frozen=True)
class TextOptions(FileOutputOptions, SourceOptions):
    """Text writer options."""

    filename_suffix: str = ".txt"
