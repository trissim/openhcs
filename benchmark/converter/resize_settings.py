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
from .setting_names import SettingNameFamily


RESIZE_METHOD_SETTING = SettingNameFamily("Resizing method")
RESIZE_FACTOR_SETTING = SettingNameFamily("Resizing factor")
RESIZE_FACTOR_X_SETTING = SettingNameFamily("X Resizing factor")
RESIZE_FACTOR_Y_SETTING = SettingNameFamily("Y Resizing factor")
RESIZE_FACTOR_Z_SETTING = SettingNameFamily("Z Resizing factor")
RESIZE_WIDTH_SETTING = SettingNameFamily("Width of the final image", aliases=("Width (x) of the final image",))
RESIZE_HEIGHT_SETTING = SettingNameFamily("Height of the final image", aliases=("Height (y) of the final image",))
RESIZE_PLANES_SETTING = SettingNameFamily("# of planes (z) in the final image")
RESIZE_INTERPOLATION_SETTING = SettingNameFamily("Interpolation method")


def resize_bound_kwargs(
    module: ModuleBlock,
    binder: SettingsBinder,
) -> dict[str, Any]:
    """Return absorbed-function kwargs for CellProfiler Resize."""
    del binder
    kwargs: dict[str, Any] = {}
    resizing_method = optional_setting_value(module, RESIZE_METHOD_SETTING)
    if resizing_method is not None:
        kwargs["resize_method"] = _resize_method(resizing_method).value

    resizing_factor = optional_setting_value(module, RESIZE_FACTOR_SETTING)
    if resizing_factor is not None:
        factor = parse_cellprofiler_float(resizing_factor)
        kwargs["resizing_factor_x"] = factor
        kwargs["resizing_factor_y"] = factor

    factor_x = optional_setting_value(module, RESIZE_FACTOR_X_SETTING)
    if factor_x is not None:
        kwargs["resizing_factor_x"] = parse_cellprofiler_float(factor_x)

    factor_y = optional_setting_value(module, RESIZE_FACTOR_Y_SETTING)
    if factor_y is not None:
        kwargs["resizing_factor_y"] = parse_cellprofiler_float(factor_y)

    factor_z = optional_setting_value(module, RESIZE_FACTOR_Z_SETTING)
    if factor_z is not None:
        kwargs["resizing_factor_z"] = parse_cellprofiler_float(factor_z)

    width = optional_setting_value(module, RESIZE_WIDTH_SETTING)
    if width is not None:
        kwargs["specific_width"] = parse_cellprofiler_int(width)

    height = optional_setting_value(module, RESIZE_HEIGHT_SETTING)
    if height is not None:
        kwargs["specific_height"] = parse_cellprofiler_int(height)

    planes = optional_setting_value(module, RESIZE_PLANES_SETTING)
    if planes is not None:
        kwargs["specific_planes"] = parse_cellprofiler_int(planes)

    interpolation = optional_setting_value(module, RESIZE_INTERPOLATION_SETTING)
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
