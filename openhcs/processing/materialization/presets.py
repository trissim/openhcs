"""Convenience presets for common materialization patterns.

These are intentionally small: they make analysis modules read declaratively
without repeating JsonOptions/CsvOptions boilerplate.
"""

from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields, is_dataclass
from typing import Any, Callable, Dict, List, Optional

from openhcs.processing.materialization.core import MaterializationSpec
from openhcs.processing.materialization.options import (
    CsvOptions,
    JsonOptions,
    MaterializedFilenameIdentity,
    ROIOptions,
    TextOptions,
    TiffStackOptions,
)


@dataclass(frozen=True, slots=True)
class _TabularPreset:
    """Shared tabular writer settings used by CSV and JSON presets."""

    fields: Optional[List[str]] = None
    row_field: Optional[str] = None
    row_columns: Optional[Dict[str, str]] = None
    row_unpacker: Optional[Callable[[Any], List[Dict[str, Any]]]] = None

    def csv_options(
        self,
        source: Optional[str],
        suffix: str,
    ) -> CsvOptions:
        return CsvOptions(
            source=source,
            filename_suffix=suffix,
            fields=self.fields,
            row_field=self.row_field,
            row_columns=self.row_columns or {},
            row_unpacker=self.row_unpacker,
        )

    def json_options(
        self,
        source: Optional[str],
        suffix: str,
        indent: int,
        wrap_list: bool,
    ) -> JsonOptions:
        return JsonOptions(
            source=source,
            filename_suffix=suffix,
            indent=indent,
            wrap_list=wrap_list,
            fields=self.fields,
            row_field=self.row_field,
            row_columns=self.row_columns or {},
            row_unpacker=self.row_unpacker,
        )


def json_only(
    *,
    source: Optional[str] = None,
    suffix: str = ".json",
    indent: int = 2,
    wrap_list: bool = False,
    fields: Optional[List[str]] = None,
    row_field: Optional[str] = None,
    row_columns: Optional[Dict[str, str]] = None,
    row_unpacker: Optional[Callable[[Any], List[Dict[str, Any]]]] = None,
    allowed_backends: Optional[List[str]] = None,
) -> MaterializationSpec:
    tabular = _TabularPreset(fields, row_field, row_columns, row_unpacker)
    return MaterializationSpec(
        tabular.json_options(source, suffix, indent, wrap_list),
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
    tabular = _TabularPreset(fields, row_field, row_columns, row_unpacker)
    return MaterializationSpec(
        tabular.csv_options(source, suffix),
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


def csv_dataclass_materializer(
    row_type: type[Any],
    *,
    analysis_type: Optional[str] = None,
    source: Optional[str] = None,
    suffix: Optional[str] = None,
    allowed_backends: Optional[List[str]] = None,
) -> MaterializationSpec:
    """Build a CSV materializer from a nominal dataclass row schema."""
    if not is_dataclass(row_type):
        raise TypeError(
            "csv_dataclass_materializer requires a dataclass row type, got "
            f"{getattr(row_type, '__name__', type(row_type).__name__)}."
        )
    return csv_materializer(
        fields=[field.name for field in dataclass_fields(row_type)],
        analysis_type=analysis_type,
        source=source,
        suffix=suffix,
        allowed_backends=allowed_backends,
    )


def json_materializer(
    *,
    fields: Optional[List[str]] = None,
    analysis_type: Optional[str] = None,
    source: Optional[str] = None,
    row_field: Optional[str] = None,
    row_columns: Optional[Dict[str, str]] = None,
    row_unpacker: Optional[Callable[[Any], List[Dict[str, Any]]]] = None,
    suffix: Optional[str] = None,
    indent: int = 2,
    wrap_list: bool = False,
    allowed_backends: Optional[List[str]] = None,
) -> MaterializationSpec:
    """Compatibility helper for JSON analysis outputs in absorbed functions."""

    resolved_suffix = suffix or (
        f"_{analysis_type}.json" if analysis_type else ".json"
    )
    return json_only(
        source=source,
        suffix=resolved_suffix,
        indent=indent,
        wrap_list=wrap_list,
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
    tabular = _TabularPreset(fields, row_field, row_columns, row_unpacker)
    return MaterializationSpec(
        tabular.json_options(json_source, json_suffix, json_indent, wrap_list),
        tabular.csv_options(csv_source, csv_suffix),
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


def segmentation_mask_rois(
    *,
    source: Optional[str] = None,
    min_area: int = 10,
    extract_contours: bool = True,
    roi_suffix: str = "_rois.roi.zip",
    summary_suffix: str = "_segmentation_summary.txt",
    allowed_backends: Optional[List[str]] = None,
) -> MaterializationSpec:
    """Materialize labeled object masks as ImageJ ROI archives."""
    return roi_zip(
        source=source,
        min_area=min_area,
        extract_contours=extract_contours,
        roi_suffix=roi_suffix,
        summary_suffix=summary_suffix,
        allowed_backends=allowed_backends,
    )


def tiff_stack(
    options: TiffStackOptions = TiffStackOptions(),
    *,
    allowed_backends: Optional[List[str]] = None,
) -> MaterializationSpec:
    return MaterializationSpec(
        options,
        allowed_backends=allowed_backends,
    )


def text_only(
    *,
    source: Optional[str] = None,
    suffix: str = ".txt",
    allowed_backends: Optional[List[str]] = None,
) -> MaterializationSpec:
    return MaterializationSpec(TextOptions(source=source, filename_suffix=suffix), allowed_backends=allowed_backends)
