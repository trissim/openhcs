"""Typed lowering for CellProfiler morphology structuring-element settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchmark.cellprofiler_library.functions.structuring_elements import (
    StructuringElement,
    coerce_structuring_element,
)

from .parser import ModuleBlock
from .settings_binder import SettingsBinder


STRUCTURING_ELEMENT_SETTING_NAME = "Structuring element"
DEFAULT_STRUCTURING_ELEMENT_SETTING = "disk,3"


@dataclass(frozen=True, slots=True)
class StructuringElementSetting:
    """Typed CellProfiler morphology footprint setting."""

    structuring_element: StructuringElement
    size: int

    @classmethod
    def from_cellprofiler_value(
        cls,
        value: Any,
    ) -> "StructuringElementSetting":
        shape, size = _structuring_element_parts(value)
        return cls(
            structuring_element=coerce_structuring_element(shape),
            size=_positive_size(size),
        )

    def bound_kwargs(self) -> dict[str, str | int]:
        """Return generated-code-safe absorbed-function kwargs."""
        return {
            "structuring_element": self.structuring_element.value,
            "size": self.size,
        }


def structuring_element_bound_kwargs(
    module: ModuleBlock,
    binder: SettingsBinder,
) -> dict[str, str | int]:
    """Lower the common CellProfiler morphology setting into function kwargs."""
    raw_value = module.get_setting(
        STRUCTURING_ELEMENT_SETTING_NAME,
        DEFAULT_STRUCTURING_ELEMENT_SETTING,
    )
    parsed_value = binder.parse_value(STRUCTURING_ELEMENT_SETTING_NAME, raw_value)
    return StructuringElementSetting.from_cellprofiler_value(
        parsed_value
    ).bound_kwargs()


def _structuring_element_parts(value: Any) -> tuple[Any, Any]:
    if isinstance(value, str):
        parts = tuple(part.strip() for part in value.split(","))
    elif isinstance(value, (list, tuple)):
        parts = tuple(value)
    else:
        raise TypeError(
            "Structuring element setting must be a comma-separated string or "
            f"sequence, got {type(value).__name__}."
        )
    if len(parts) != 2:
        raise ValueError(
            "Structuring element setting must contain shape and size, got "
            f"{value!r}."
        )
    return parts[0], parts[1]


def _positive_size(value: Any) -> int:
    size = int(value)
    if size <= 0:
        raise ValueError(f"Structuring element size must be positive: {size!r}")
    return size
