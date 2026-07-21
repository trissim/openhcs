"""Constants for the materialization system (greenfield)."""

from __future__ import annotations

from enum import Enum


class MaterializationFormat(str, Enum):
    """Built-in output formats (writer keys)."""

    CSV = "csv"
    JSON = "json"
    ROI_ZIP = "roi_zip"
    TIFF_STACK = "tiff_stack"
    TEXT = "text"
    IMAGE_FILE = "image_file"
    FILE_BUNDLE = "file_bundle"


class WriteMode(str, Enum):
    """Overwrite/delete semantics for materialization writes."""

    OVERWRITE = "overwrite"
    ERROR = "error"
