"""CellProfiler ExpandOrShrinkObjects setting lowering."""

from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar

from metaclass_registry import AutoRegisterMeta

from benchmark.cellprofiler_library.functions.expandorshrinkobjects import (
    CellProfilerExpandShrinkOperation,
    ExpandShrinkMode,
    ExpandShrinkOperationStrategy,
)

from .parser import ModuleBlock
from .settings_binder import (
    SettingToKeywordBinding,
    SettingsBinder,
    normalize_cellprofiler_setting_name,
    parse_cellprofiler_bool,
    parse_cellprofiler_int,
)


class ExpandShrinkOperationModeBinding(ABC, metaclass=AutoRegisterMeta):
    """Registered lowering from CP UI operation literals to runtime modes."""

    __registry_key__ = "operation_key"
    __skip_if_no_key__ = True

    operation_key: ClassVar[str | None] = None
    mode: ClassVar[ExpandShrinkMode | None] = None

    @classmethod
    def mode_for(
        cls,
        operation: CellProfilerExpandShrinkOperation,
    ) -> ExpandShrinkMode:
        binding_type = cls.__registry__.get(operation.value)
        if binding_type is None or binding_type.mode is None:
            raise ValueError(
                "Unsupported ExpandOrShrinkObjects operation: "
                f"{operation.value!r}."
            )
        return binding_type.mode


def _register_expand_shrink_operation_mode(
    operation: CellProfilerExpandShrinkOperation,
) -> None:
    mode = _runtime_mode_for_cellprofiler_operation(operation)
    class_name = f"{operation.name.title().replace('_', '')}ModeBinding"
    globals()[class_name] = type(
        class_name,
        (ExpandShrinkOperationModeBinding,),
        {
            "__module__": __name__,
            "operation_key": operation.value,
            "mode": mode,
        },
    )


def _runtime_mode_for_cellprofiler_operation(
    operation: CellProfilerExpandShrinkOperation,
) -> ExpandShrinkMode:
    matches = tuple(
        strategy_type.mode
        for strategy_type in ExpandShrinkOperationStrategy.__registry__.values()
        if operation in strategy_type.cellprofiler_operations
    )
    if len(matches) != 1 or matches[0] is None:
        raise ValueError(
            "Expected exactly one runtime ExpandOrShrinkObjects strategy for "
            f"CellProfiler operation {operation.value!r}; found {len(matches)}."
        )
    return matches[0]


for _operation in CellProfilerExpandShrinkOperation:
    _register_expand_shrink_operation_mode(_operation)

del _operation


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
    return ExpandShrinkOperationModeBinding.mode_for(operation)


def _coerce_cellprofiler_expand_shrink_operation(
    value: str,
) -> CellProfilerExpandShrinkOperation:
    normalized_value = normalize_cellprofiler_setting_name(value)
    for operation in CellProfilerExpandShrinkOperation:
        if normalize_cellprofiler_setting_name(operation.value) == normalized_value:
            return operation
    raise ValueError(f"Unsupported ExpandOrShrinkObjects operation: {value!r}.")
