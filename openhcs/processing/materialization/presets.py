"""Convenience presets for common materialization patterns.

These are intentionally small: they make analysis modules read declaratively
without repeating JsonOptions/CsvOptions boilerplate.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from openhcs.processing.materialization.core import MaterializationSpec
from openhcs.processing.materialization.options import (
    CsvOptions,
    JsonOptions,
    ROIOptions,
    TextOptions,
    TiffStackOptions,
)


def json_only(
    *,
    source: Optional[str] = None,
    suffix: str = ".json",
    indent: int = 2,
    wrap_list: bool = False,
    allowed_backends: Optional[List[str]] = None,
) -> MaterializationSpec:
    return MaterializationSpec(
        JsonOptions(source=source, filename_suffix=suffix, indent=indent, wrap_list=wrap_list),
        allowed_backends=allowed_backends,
    )


def csv_only(
    *,
    source: Optional[str] = None,
    suffix: str = "_details.csv",
    fields: Optional[List[str]] = None,
    row_field: Optional[str] = None,
    row_columns: Optional[Dict[str, str]] = None,
    row_unpacker: Optional[Callable[[Any], List[Dict[str, Any]]]] = None,
    allowed_backends: Optional[List[str]] = None,
) -> MaterializationSpec:
    return MaterializationSpec(
        CsvOptions(
            source=source,
            filename_suffix=suffix,
            fields=fields,
            row_field=row_field,
            row_columns=row_columns or {},
            row_unpacker=row_unpacker,
        ),
        allowed_backends=allowed_backends,
    )


def csv_materializer(
    *,
    fields: Optional[List[str]] = None,
    analysis_type: Optional[str] = None,
    source: Optional[str] = None,
    row_field: Optional[str] = None,
    row_columns: Optional[Dict[str, str]] = None,
    row_unpacker: Optional[Callable[[Any], List[Dict[str, Any]]]] = None,
    suffix: Optional[str] = None,
    allowed_backends: Optional[List[str]] = None,
) -> MaterializationSpec:
    """Compatibility helper for CSV analysis outputs in absorbed functions.

    ``analysis_type`` maps to the historical per-analysis filename convention
    ``_<analysis_type>.csv``. Callers can still override the suffix directly.
    """

    resolved_suffix = suffix or (
        f"_{analysis_type}.csv" if analysis_type else "_details.csv"
    )
    return csv_only(
        source=source,
        suffix=resolved_suffix,
        fields=fields,
        row_field=row_field,
        row_columns=row_columns,
        row_unpacker=row_unpacker,
        allowed_backends=allowed_backends,
    )


def json_and_csv(
    *,
    json_source: Optional[str] = None,
    csv_source: Optional[str] = None,
    json_suffix: str = ".json",
    csv_suffix: str = "_details.csv",
    json_indent: int = 2,
    wrap_list: bool = False,
    fields: Optional[List[str]] = None,
    row_field: Optional[str] = None,
    row_columns: Optional[Dict[str, str]] = None,
    row_unpacker: Optional[Callable[[Any], List[Dict[str, Any]]]] = None,
    primary: str = "json",
    allowed_backends: Optional[List[str]] = None,
) -> MaterializationSpec:
    primary_idx = 0 if primary == "json" else 1
    return MaterializationSpec(
        JsonOptions(
            source=json_source,
            filename_suffix=json_suffix,
            indent=json_indent,
            wrap_list=wrap_list,
        ),
        CsvOptions(
            source=csv_source,
            filename_suffix=csv_suffix,
            fields=fields,
            row_field=row_field,
            row_columns=row_columns or {},
            row_unpacker=row_unpacker,
        ),
        primary=primary_idx,
        allowed_backends=allowed_backends,
    )


def roi_zip(
    *,
    source: Optional[str] = None,
    min_area: int = 10,
    extract_contours: bool = True,
    roi_suffix: str = "_rois.roi.zip",
    summary_suffix: str = "_segmentation_summary.txt",
    allowed_backends: Optional[List[str]] = None,
) -> MaterializationSpec:
    return MaterializationSpec(
        ROIOptions(
            source=source,
            min_area=min_area,
            extract_contours=extract_contours,
            roi_suffix=roi_suffix,
            summary_suffix=summary_suffix,
        ),
        allowed_backends=allowed_backends,
    )


def tiff_stack(
    *,
    source: Optional[str] = None,
    normalize_uint8: bool = False,
    slice_pattern: str = "_slice_{index:03d}.tif",
    summary_suffix: str = "_summary.txt",
    empty_summary: str = "No images generated (empty data)\n",
    allowed_backends: Optional[List[str]] = None,
) -> MaterializationSpec:
    return MaterializationSpec(
        TiffStackOptions(
            source=source,
            normalize_uint8=normalize_uint8,
            slice_pattern=slice_pattern,
            summary_suffix=summary_suffix,
            empty_summary=empty_summary,
        ),
        allowed_backends=allowed_backends,
    )


def text_only(
    *,
    source: Optional[str] = None,
    suffix: str = ".txt",
    allowed_backends: Optional[List[str]] = None,
) -> MaterializationSpec:
    return MaterializationSpec(TextOptions(source=source, filename_suffix=suffix), allowed_backends=allowed_backends)
