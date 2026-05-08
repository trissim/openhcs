"""Typed lowering for CellProfiler morphology structuring-element settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchmark.cellprofiler_library.functions.structuring_elements import (
    StructuringElement,
    coerce_structuring_element,
)
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    optional_setting_value,
    setting_names,
)

from .parser import ModuleBlock
from .settings_binder import SettingsBinder, normalize_cellprofiler_setting_name


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

    def bound_kwargs(
        self,
        *,
        shape_keyword: str = "structuring_element",
        size_keyword: str = "size",
    ) -> dict[str, str | int]:
        """Return generated-code-safe absorbed-function kwargs."""
        return {
            shape_keyword: self.structuring_element.value,
            size_keyword: self.size,
        }


@dataclass(frozen=True, slots=True)
class StructuringElementSettingBinding:
    """Bind one named CellProfiler structuring-element setting to kwargs."""

    setting_name: str | SettingNameFamily = STRUCTURING_ELEMENT_SETTING_NAME
    default_value: str = DEFAULT_STRUCTURING_ELEMENT_SETTING
    shape_keyword: str = "structuring_element"
    size_keyword: str = "size"

    @property
    def normalized_setting_names(self) -> frozenset[str]:
        return frozenset(
            normalize_cellprofiler_setting_name(setting_name)
            for setting_name in setting_names(self.setting_name)
        )

    def bound_kwargs(
        self,
        module: ModuleBlock,
        binder: SettingsBinder,
    ) -> dict[str, str | int]:
        raw_value = optional_setting_value(module, self.setting_name)
        if raw_value is None:
            raw_value = self.default_value
        parsed_value = binder.parse_value(setting_names(self.setting_name)[0], raw_value)
        return StructuringElementSetting.from_cellprofiler_value(
            parsed_value
        ).bound_kwargs(
            shape_keyword=self.shape_keyword,
            size_keyword=self.size_keyword,
        )


def structuring_element_bound_kwargs(
    module: ModuleBlock,
    binder: SettingsBinder,
    binding: StructuringElementSettingBinding = StructuringElementSettingBinding(),
) -> dict[str, str | int]:
    """Lower the common CellProfiler morphology setting into function kwargs."""
    return binding.bound_kwargs(module, binder)


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
