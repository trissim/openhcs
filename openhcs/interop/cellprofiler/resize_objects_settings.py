"""Typed lowering for CellProfiler ResizeObjects settings."""

from __future__ import annotations

from typing import Any

from .parser import ModuleBlock
from .setting_names import SettingNameFamily
from .settings_binder import (
    SettingToKeywordBinding,
    SettingsBinder,
    parse_cellprofiler_float,
    parse_cellprofiler_int,
)


RESIZE_OBJECTS_FACTOR_Z_SETTING = SettingNameFamily("Z Factor")
RESIZE_OBJECTS_PLANES_SETTING = SettingNameFamily("Planes (Z)")


RESIZE_OBJECTS_SETTINGS: tuple[SettingToKeywordBinding, ...] = (
    SettingToKeywordBinding("Method", "method", lambda value: value.strip().lower()),
    SettingToKeywordBinding("X Factor", "factor_x", parse_cellprofiler_float),
    SettingToKeywordBinding("Y Factor", "factor_y", parse_cellprofiler_float),
    SettingToKeywordBinding(
        RESIZE_OBJECTS_FACTOR_Z_SETTING,
        "factor_z",
        parse_cellprofiler_float,
    ),
    SettingToKeywordBinding("Width (X)", "width", parse_cellprofiler_int),
    SettingToKeywordBinding("Height (Y)", "height", parse_cellprofiler_int),
    SettingToKeywordBinding(
        RESIZE_OBJECTS_PLANES_SETTING,
        "planes",
        parse_cellprofiler_int,
    ),
)


def resize_objects_bound_kwargs(
    module: ModuleBlock,
    binder: SettingsBinder,
) -> dict[str, Any]:
    """Return absorbed-function kwargs for CellProfiler ResizeObjects."""
    return dict(binder.bind_declared(module, RESIZE_OBJECTS_SETTINGS))
