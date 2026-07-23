"""Structuring-element setting ownership for CellProfiler modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
from openhcs.interop.cellprofiler.module_settings import BoundModuleSettings
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    optional_setting_value,
    setting_names,
)
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.interop.cellprofiler_setting_normalization import (
    normalize_cellprofiler_setting_name,
)
from openhcs.processing.backends.cellprofiler.structuring_elements import (
    StructuringElement,
)

if TYPE_CHECKING:
    from openhcs.interop.cellprofiler.parser import ModuleBlock
    from openhcs.interop.cellprofiler.settings_binder import SettingsBinder


STRUCTURING_ELEMENT_SETTING_NAME = "Structuring element"
DEFAULT_STRUCTURING_ELEMENT_SETTING = "disk,3"


@dataclass(frozen=True, slots=True)
class StructuringElementSetting:
    """Typed CellProfiler morphology footprint setting."""

    structuring_element: StructuringElement
    size: int

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError(
                f"Structuring element size must be positive: {self.size!r}"
            )

    @classmethod
    def from_cellprofiler_value(cls, value: Any) -> "StructuringElementSetting":
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
        shape, size = parts
        return cls(
            structuring_element=coerce_cellprofiler_enum(StructuringElement, shape),
            size=int(size),
        )

    def bound_kwargs(
        self,
        *,
        shape_keyword: str = "structuring_element",
        size_keyword: str = "size",
    ) -> dict[str, str | int]:
        """Return generated-code-safe absorbed-function kwargs."""
        return {shape_keyword: self.structuring_element.value, size_keyword: self.size}


@dataclass(frozen=True, slots=True)
class StructuringElementSettingBinding:
    """Bind one named CellProfiler structuring-element setting to kwargs."""

    setting_name: str | SettingNameFamily = STRUCTURING_ELEMENT_SETTING_NAME
    legacy_size_setting_name: str | SettingNameFamily | None = "Size"
    default_value: str = DEFAULT_STRUCTURING_ELEMENT_SETTING
    shape_keyword: str = "structuring_element"
    size_keyword: str = "size"

    @property
    def normalized_setting_names(self) -> frozenset[str]:
        names = {
            normalize_cellprofiler_setting_name(setting_name)
            for setting_name in setting_names(self.setting_name)
        }
        if self.legacy_size_setting_name is not None:
            names.update(
                normalize_cellprofiler_setting_name(setting_name)
                for setting_name in setting_names(self.legacy_size_setting_name)
            )
        return frozenset(names)

    def bound_kwargs(self, module: ModuleBlock) -> dict[str, str | int]:
        return self.declared_setting(module).bound_kwargs(
            shape_keyword=self.shape_keyword,
            size_keyword=self.size_keyword,
        )

    def declared_setting(self, module: ModuleBlock) -> StructuringElementSetting:
        """Return the typed footprint declared by one parsed module."""
        raw_value = optional_setting_value(module, self.setting_name)
        if raw_value is not None:
            return StructuringElementSetting.from_cellprofiler_value(raw_value)

        default = StructuringElementSetting.from_cellprofiler_value(self.default_value)
        if self.legacy_size_setting_name is None:
            return default
        legacy_size = optional_setting_value(module, self.legacy_size_setting_name)
        if legacy_size is None:
            return default
        return StructuringElementSetting(
            structuring_element=default.structuring_element,
            size=int(legacy_size),
        )


class StructuringElementSettingsModule(CellProfilerModule):
    """Parent for modules sharing CellProfiler structuring-element lowering."""

    structuring_element_binding: ClassVar[StructuringElementSettingBinding] = (
        StructuringElementSettingBinding()
    )

    @classmethod
    def declared_structuring_element_setting(
        cls,
        module: ModuleBlock,
    ) -> StructuringElementSetting:
        """Return the typed footprint setting owned by one parsed module."""
        return cls.structuring_element_binding.declared_setting(module)

    @classmethod
    def bind_settings(
        cls,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
    ) -> BoundModuleSettings:
        bound = cls._bind_declared_settings(module, binder=binder)
        kwargs = {
            **bound.kwargs,
            **cls.structuring_element_binding.bound_kwargs(module),
        }
        unmapped_kwargs = dict(bound.unmapped_kwargs)
        for setting_name in cls.structuring_element_binding.normalized_setting_names:
            unmapped_kwargs.pop(setting_name, None)
        return cls._finalize_bound_settings(
            module,
            binder=binder,
            bound=cls.postprocess_bound_settings(
                module,
                BoundModuleSettings(
                    kwargs,
                    unmapped_kwargs,
                    bound.setting_coverage,
                ),
            ),
        )
