"""Typed lowering for CellProfiler Resize settings."""

from __future__ import annotations

from typing import Any

from benchmark.cellprofiler_library.functions.resize import (
    InterpolationMethod,
    ResizeMethod,
)
from benchmark.cellprofiler_library.functions._enum import _coerce_function_enum
from openhcs.interop.cellprofiler.setting_names import optional_setting_value

from .parser import ModuleBlock
from .settings_binder import (
    SettingsBinder,
    parse_cellprofiler_float,
    parse_cellprofiler_int,
)


def resize_bound_kwargs(
    module: ModuleBlock,
    binder: SettingsBinder,
) -> dict[str, Any]:
    """Return absorbed-function kwargs for CellProfiler Resize."""
    del binder
    kwargs: dict[str, Any] = {}
    resizing_method = optional_setting_value(module, "Resizing method")
    if resizing_method is not None:
        kwargs["resize_method"] = _resize_method(resizing_method).value

    resizing_factor = optional_setting_value(module, "Resizing factor")
    if resizing_factor is not None:
        factor = parse_cellprofiler_float(resizing_factor)
        kwargs["resizing_factor_x"] = factor
        kwargs["resizing_factor_y"] = factor

    width = optional_setting_value(module, "Width of the final image")
    if width is not None:
        kwargs["specific_width"] = parse_cellprofiler_int(width)

    height = optional_setting_value(module, "Height of the final image")
    if height is not None:
        kwargs["specific_height"] = parse_cellprofiler_int(height)

    interpolation = optional_setting_value(module, "Interpolation method")
    if interpolation is not None:
        kwargs["interpolation"] = _coerce_function_enum(
            InterpolationMethod,
            interpolation,
        ).value

    return kwargs


def _resize_method(value: str) -> ResizeMethod:
    normalized = value.strip().lower()
    if "fraction" in normalized or "multiple" in normalized:
        return ResizeMethod.BY_FACTOR
    if "specific" in normalized or "dimension" in normalized or "manual" in normalized:
        return ResizeMethod.TO_SIZE
    return _coerce_function_enum(ResizeMethod, value)
