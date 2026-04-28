"""Materialization public API (writer-based)."""

from openhcs.processing.materialization.constants import MaterializationFormat, WriteMode
from openhcs.processing.materialization.core import (
    BackendSaver,
    MaterializationContext,
    MaterializationSpec,
    Output,
    PathHelper,
    materialize,
)
from openhcs.processing.materialization.options import (
    CsvOptions,
    FileOutputOptions,
    JsonOptions,
    ROIOptions,
    TextOptions,
    TiffStackOptions,
)
from openhcs.processing.materialization.presets import (
    csv_materializer,
    csv_only,
    json_and_csv,
    json_only,
    roi_zip,
    segmentation_mask_rois,
    text_only,
    tiff_stack,
)

__all__ = [
    "MaterializationFormat",
    "WriteMode",
    "MaterializationSpec",
    "MaterializationContext",
    "Output",
    "PathHelper",
    "BackendSaver",
    "materialize",
    "FileOutputOptions",
    "CsvOptions",
    "JsonOptions",
    "ROIOptions",
    "TiffStackOptions",
    "TextOptions",
    "csv_materializer",
    "json_only",
    "csv_only",
    "json_and_csv",
    "roi_zip",
    "segmentation_mask_rois",
    "tiff_stack",
    "text_only",
]
