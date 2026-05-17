"""CellProfiler ExpandOrShrinkObjects setting lowering."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

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


@dataclass(frozen=True)
class ExpandShrinkOperationModeDeclaration:
    """Declared lowering from CP UI operation literals to one runtime mode."""

    mode: ExpandShrinkMode
    operations: tuple[CellProfilerExpandShrinkOperation, ...]

    @property
    def mode_label(self) -> str:
        return self.mode.value

    def includes_operation(self, operation: CellProfilerExpandShrinkOperation) -> bool:
        return operation in self.operations


class ExpandShrinkOperationModeBinding:
    """Authoritative registry projection for ExpandOrShrinkObjects modes."""

    __registry__: dict[str, ExpandShrinkOperationModeDeclaration] = {}

    @classmethod
    def declare(
        cls,
        declaration: ExpandShrinkOperationModeDeclaration,
    ) -> ExpandShrinkOperationModeDeclaration:
        cls.__registry__[declaration.mode_label] = declaration
        return declaration

    @classmethod
    def mode_for(
        cls,
        operation: CellProfilerExpandShrinkOperation | str,
    ) -> ExpandShrinkMode:
        resolved = coerce_cellprofiler_enum(CellProfilerExpandShrinkOperation, operation)
        matches = tuple(
            declaration.mode
            for declaration in cls.__registry__.values()
            if declaration.includes_operation(resolved)
        )
        if len(matches) != 1:
            raise ValueError(
                "Expected exactly one ExpandOrShrinkObjects mode for operation "
                f"{resolved.value!r}; found {len(matches)}."
            )
        return matches[0]


EXPAND_SHRINK_OPERATION_MODE_DECLARATIONS: tuple[
    ExpandShrinkOperationModeDeclaration,
    ...,
] = (
    ExpandShrinkOperationModeDeclaration(
        ExpandShrinkMode.EXPAND_DEFINED_PIXELS,
        (
            CellProfilerExpandShrinkOperation.EXPAND_DEFINED_PIXELS,
            CellProfilerExpandShrinkOperation.EXPAND_BY_MEASUREMENT,
        ),
    ),
    ExpandShrinkOperationModeDeclaration(
        ExpandShrinkMode.EXPAND_INFINITE,
        (CellProfilerExpandShrinkOperation.EXPAND_UNTIL_TOUCHING,),
    ),
    ExpandShrinkOperationModeDeclaration(
        ExpandShrinkMode.SHRINK_DEFINED_PIXELS,
        (
            CellProfilerExpandShrinkOperation.SHRINK_DEFINED_PIXELS,
            CellProfilerExpandShrinkOperation.SHRINK_BY_MEASUREMENT,
        ),
    ),
    ExpandShrinkOperationModeDeclaration(
        ExpandShrinkMode.SHRINK_TO_POINT,
        (CellProfilerExpandShrinkOperation.SHRINK_TO_POINT,),
    ),
    ExpandShrinkOperationModeDeclaration(
        ExpandShrinkMode.ADD_DIVIDING_LINES,
        (CellProfilerExpandShrinkOperation.ADD_DIVIDING_LINES,),
    ),
    ExpandShrinkOperationModeDeclaration(
        ExpandShrinkMode.DESPUR,
        (CellProfilerExpandShrinkOperation.DESPUR,),
    ),
    ExpandShrinkOperationModeDeclaration(
        ExpandShrinkMode.SKELETONIZE,
        (CellProfilerExpandShrinkOperation.SKELETONIZE,),
    ),
)

for declaration in EXPAND_SHRINK_OPERATION_MODE_DECLARATIONS:
    ExpandShrinkOperationModeBinding.declare(declaration)


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
