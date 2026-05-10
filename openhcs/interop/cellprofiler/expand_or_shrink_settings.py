"""CellProfiler ExpandOrShrinkObjects setting lowering."""

from __future__ import annotations

from abc import ABC
from enum import Enum
from typing import Any, ClassVar

from metaclass_registry import AutoRegisterMeta

from .parser import ModuleBlock
from .settings_binder import (
    SettingToKeywordBinding,
    SettingsBinder,
    coerce_cellprofiler_enum,
    parse_cellprofiler_bool,
    parse_cellprofiler_int,
)


class CellProfilerExpandShrinkOperation(Enum):
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


class ExpandShrinkMode(Enum):
    """Runtime mode literals consumed by ExpandOrShrinkObjects execution."""

    EXPAND_DEFINED_PIXELS = "expand_defined_pixels"
    EXPAND_INFINITE = "expand_infinite"
    SHRINK_DEFINED_PIXELS = "shrink_defined_pixels"
    SHRINK_TO_POINT = "shrink_to_point"
    ADD_DIVIDING_LINES = "add_dividing_lines"
    DESPUR = "despur"
    SKELETONIZE = "skeletonize"


class ExpandShrinkOperationModeBinding(ABC, metaclass=AutoRegisterMeta):
    """Registered lowering from CP UI operation literals to runtime modes."""

    __registry_key__ = "mode_label"
    __skip_if_no_key__ = True

    mode: ClassVar[ExpandShrinkMode | None] = None
    mode_label: ClassVar[str | None] = None
    operations: ClassVar[tuple[CellProfilerExpandShrinkOperation, ...]] = ()

    @classmethod
    def mode_for(
        cls,
        operation: CellProfilerExpandShrinkOperation | str,
    ) -> ExpandShrinkMode:
        resolved = coerce_cellprofiler_enum(CellProfilerExpandShrinkOperation, operation)
        matches = tuple(
            binding_type.mode
            for binding_type in cls.__registry__.values()
            if resolved in binding_type.operations
        )
        if len(matches) != 1 or matches[0] is None:
            raise ValueError(
                "Expected exactly one ExpandOrShrinkObjects mode for operation "
                f"{resolved.value!r}; found {len(matches)}."
            )
        return matches[0]


class ExpandDefinedPixelsModeBinding(ExpandShrinkOperationModeBinding):
    mode = ExpandShrinkMode.EXPAND_DEFINED_PIXELS
    mode_label = mode.value
    operations = (
        CellProfilerExpandShrinkOperation.EXPAND_DEFINED_PIXELS,
        CellProfilerExpandShrinkOperation.EXPAND_BY_MEASUREMENT,
    )


class ExpandInfiniteModeBinding(ExpandShrinkOperationModeBinding):
    mode = ExpandShrinkMode.EXPAND_INFINITE
    mode_label = mode.value
    operations = (CellProfilerExpandShrinkOperation.EXPAND_UNTIL_TOUCHING,)


class ShrinkDefinedPixelsModeBinding(ExpandShrinkOperationModeBinding):
    mode = ExpandShrinkMode.SHRINK_DEFINED_PIXELS
    mode_label = mode.value
    operations = (
        CellProfilerExpandShrinkOperation.SHRINK_DEFINED_PIXELS,
        CellProfilerExpandShrinkOperation.SHRINK_BY_MEASUREMENT,
    )


class ShrinkToPointModeBinding(ExpandShrinkOperationModeBinding):
    mode = ExpandShrinkMode.SHRINK_TO_POINT
    mode_label = mode.value
    operations = (CellProfilerExpandShrinkOperation.SHRINK_TO_POINT,)


class AddDividingLinesModeBinding(ExpandShrinkOperationModeBinding):
    mode = ExpandShrinkMode.ADD_DIVIDING_LINES
    mode_label = mode.value
    operations = (CellProfilerExpandShrinkOperation.ADD_DIVIDING_LINES,)


class DespurModeBinding(ExpandShrinkOperationModeBinding):
    mode = ExpandShrinkMode.DESPUR
    mode_label = mode.value
    operations = (CellProfilerExpandShrinkOperation.DESPUR,)


class SkeletonizeModeBinding(ExpandShrinkOperationModeBinding):
    mode = ExpandShrinkMode.SKELETONIZE
    mode_label = mode.value
    operations = (CellProfilerExpandShrinkOperation.SKELETONIZE,)


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
    """Map one CellProfiler operation literal to the runtime mode."""
    return ExpandShrinkOperationModeBinding.mode_for(value)
