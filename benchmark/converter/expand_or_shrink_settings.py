"""CellProfiler ExpandOrShrinkObjects setting lowering."""

from __future__ import annotations

from enum import Enum
from typing import Any

from benchmark.cellprofiler_library.functions.expandorshrinkobjects import (
    ExpandShrinkMode,
)

from .parser import ModuleBlock
from .settings_binder import (
    SettingToKeywordBinding,
    SettingsBinder,
    normalize_cellprofiler_setting_name,
    parse_cellprofiler_bool,
    parse_cellprofiler_int,
)


class CellProfilerExpandShrinkOperation(str, Enum):
    """Closed CellProfiler UI operation dialect for ExpandOrShrinkObjects."""

    SHRINK_TO_POINT = "Shrink objects to a point"
    EXPAND_UNTIL_TOUCHING = "Expand objects until touching"
    ADD_DIVIDING_LINES = "Add partial dividing lines between objects"
    SHRINK_DEFINED_PIXELS = "Shrink objects by a specified number of pixels"
    SHRINK_BY_MEASUREMENT = "Shrink objects by a previous measurement"
    EXPAND_DEFINED_PIXELS = "Expand objects by a specified number of pixels"
    EXPAND_BY_MEASUREMENT = "Expand objects by a previous measurement"
    SKELETONIZE = "Skeletonize each object"
    DESPUR = "Remove spurs"


_EXPAND_SHRINK_MODE_BY_OPERATION: dict[
    CellProfilerExpandShrinkOperation,
    ExpandShrinkMode,
] = {
    CellProfilerExpandShrinkOperation.SHRINK_TO_POINT: (
        ExpandShrinkMode.SHRINK_TO_POINT
    ),
    CellProfilerExpandShrinkOperation.EXPAND_UNTIL_TOUCHING: (
        ExpandShrinkMode.EXPAND_INFINITE
    ),
    CellProfilerExpandShrinkOperation.ADD_DIVIDING_LINES: (
        ExpandShrinkMode.ADD_DIVIDING_LINES
    ),
    CellProfilerExpandShrinkOperation.SHRINK_DEFINED_PIXELS: (
        ExpandShrinkMode.SHRINK_DEFINED_PIXELS
    ),
    CellProfilerExpandShrinkOperation.SHRINK_BY_MEASUREMENT: (
        ExpandShrinkMode.SHRINK_DEFINED_PIXELS
    ),
    CellProfilerExpandShrinkOperation.EXPAND_DEFINED_PIXELS: (
        ExpandShrinkMode.EXPAND_DEFINED_PIXELS
    ),
    CellProfilerExpandShrinkOperation.EXPAND_BY_MEASUREMENT: (
        ExpandShrinkMode.EXPAND_DEFINED_PIXELS
    ),
    CellProfilerExpandShrinkOperation.SKELETONIZE: ExpandShrinkMode.SKELETONIZE,
    CellProfilerExpandShrinkOperation.DESPUR: ExpandShrinkMode.DESPUR,
}


EXPAND_OR_SHRINK_OBJECTS_SETTINGS: tuple[SettingToKeywordBinding, ...] = (
    SettingToKeywordBinding(
        "Select the operation",
        "mode",
        lambda value: expand_shrink_mode_for_operation(value).value,
    ),
    SettingToKeywordBinding(
        "Number of pixels by which to expand or shrink",
        "iterations",
        parse_cellprofiler_int,
    ),
    SettingToKeywordBinding(
        "Fill holes in objects so that all objects shrink to a single point?",
        "fill_holes",
        parse_cellprofiler_bool,
    ),
)


def expand_or_shrink_bound_kwargs(
    module: ModuleBlock,
    binder: SettingsBinder,
) -> dict[str, Any]:
    """Return absorbed-function kwargs for CellProfiler ExpandOrShrinkObjects."""
    return binder.bind_declared(module, EXPAND_OR_SHRINK_OBJECTS_SETTINGS)


def expand_shrink_mode_for_operation(value: str) -> ExpandShrinkMode:
    """Map one CellProfiler operation literal to the local runtime mode."""
    operation = _coerce_cellprofiler_expand_shrink_operation(value)
    return _EXPAND_SHRINK_MODE_BY_OPERATION[operation]


def _coerce_cellprofiler_expand_shrink_operation(
    value: str,
) -> CellProfilerExpandShrinkOperation:
    normalized_value = normalize_cellprofiler_setting_name(value)
    for operation in CellProfilerExpandShrinkOperation:
        if normalize_cellprofiler_setting_name(operation.value) == normalized_value:
            return operation
    raise ValueError(f"Unsupported ExpandOrShrinkObjects operation: {value!r}.")
