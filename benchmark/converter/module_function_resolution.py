"""Module-level raw-function resolution for generated CellProfiler steps."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta

from benchmark.cellprofiler_library import canonical_module_name

from .parser import ModuleBlock
from .setting_names import (
    OBJECT_MEASUREMENT_SETTING,
    SettingNameFamily,
    required_setting_value,
    setting_values,
)


@dataclass(frozen=True, slots=True)
class ResolvedModuleFunction:
    """Typed raw-function selection for one generated module."""

    function_name: str


class MeasurementTargetScope(str, Enum):
    """Generic CellProfiler measurement target scope."""

    IMAGE = "image"
    OBJECT = "object"
    BOTH = "both"


class ModuleFunctionResolutionStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal family for resolving raw absorbed-function variants."""

    __registry_key__ = "module_name"
    __skip_if_no_key__ = True
    module_name: ClassVar[str | None] = None

    @classmethod
    def for_module(cls, module_name: str) -> "ModuleFunctionResolutionStrategy":
        strategy_type = cls.__registry__.get(
            canonical_module_name(module_name),
            DefaultModuleFunctionResolutionStrategy,
        )
        return strategy_type()

    @abstractmethod
    def resolve(
        self,
        module: ModuleBlock,
        *,
        default_function_name: str,
    ) -> ResolvedModuleFunction:
        """Resolve the raw absorbed function for one parsed module."""


class DefaultModuleFunctionResolutionStrategy(ModuleFunctionResolutionStrategy):
    """Use the registry-declared function unchanged."""

    def resolve(
        self,
        module: ModuleBlock,
        *,
        default_function_name: str,
    ) -> ResolvedModuleFunction:
        del module
        return ResolvedModuleFunction(function_name=default_function_name)


class ScopedMeasurementFunctionResolutionStrategy(ModuleFunctionResolutionStrategy):
    """Resolve image-vs-object absorbed variants from a CellProfiler scope setting."""

    scope_setting_name: ClassVar[SettingNameFamily | None] = None
    default_scope_value: ClassVar[str | None] = None
    object_setting_name: ClassVar[SettingNameFamily] = OBJECT_MEASUREMENT_SETTING
    object_function_name: ClassVar[str | None] = None

    def resolve(
        self,
        module: ModuleBlock,
        *,
        default_function_name: str,
    ) -> ResolvedModuleFunction:
        scope = _measurement_target_scope(
            _scope_setting_value(
                module,
                _required_class_attr(
                    type(self).scope_setting_name,
                    "scope_setting_name",
                ),
                _required_class_attr(
                    type(self).default_scope_value,
                    "default_scope_value",
                ),
            )
        )
        if scope is MeasurementTargetScope.IMAGE or not _setting_has_symbolic_values(
            module,
            type(self).object_setting_name,
        ):
            return ResolvedModuleFunction(function_name=default_function_name)
        return ResolvedModuleFunction(
            function_name=_required_class_attr(
                type(self).object_function_name,
                "object_function_name",
            )
        )


class MeasureTextureFunctionResolutionStrategy(
    ScopedMeasurementFunctionResolutionStrategy
):
    """Resolve MeasureTexture image-vs-object absorbed variants."""

    module_name = "MeasureTexture"
    scope_setting_name = SettingNameFamily(
        "Measure images or objects?",
        aliases=("Measure whole images or objects?",),
    )
    default_scope_value = "Images"
    object_function_name = "measure_texture_objects"


class MeasureColocalizationFunctionResolutionStrategy(
    ScopedMeasurementFunctionResolutionStrategy
):
    """Resolve MeasureColocalization image-vs-object absorbed variants."""

    module_name = "MeasureColocalization"
    scope_setting_name = SettingNameFamily("Select where to measure correlation")
    default_scope_value = "Across entire image"
    object_function_name = "measure_colocalization_objects"


class ObjectInputMeasurementFunctionResolutionStrategy(ModuleFunctionResolutionStrategy):
    """Resolve object-measurement variants when object inputs are declared."""

    object_setting_name: ClassVar[SettingNameFamily] = OBJECT_MEASUREMENT_SETTING
    object_function_name: ClassVar[str | None] = None

    def resolve(
        self,
        module: ModuleBlock,
        *,
        default_function_name: str,
    ) -> ResolvedModuleFunction:
        if not _setting_has_symbolic_values(
            module,
            type(self).object_setting_name,
        ):
            return ResolvedModuleFunction(function_name=default_function_name)
        return ResolvedModuleFunction(
            function_name=_required_class_attr(
                type(self).object_function_name,
                "object_function_name",
            )
        )


class MeasureGranularityFunctionResolutionStrategy(
    ObjectInputMeasurementFunctionResolutionStrategy
):
    """Resolve MeasureGranularity image-vs-object absorbed variants."""

    module_name = "MeasureGranularity"
    object_function_name = "measure_granularity_objects"


class MeasureImageAreaOccupiedFunctionResolutionStrategy(
    ObjectInputMeasurementFunctionResolutionStrategy
):
    """Resolve MeasureImageAreaOccupied binary-vs-object absorbed variants."""

    module_name = "MeasureImageAreaOccupiedBinary"
    object_setting_name = SettingNameFamily("Select objects to measure")
    object_function_name = "measure_image_area_occupied_objects"


def _scope_setting_value(
    module: ModuleBlock,
    setting: SettingNameFamily,
    default: str,
) -> str:
    try:
        return required_setting_value(module, setting)
    except ValueError:
        return default


def _required_class_attr[T](value: T | None, name: str) -> T:
    if value is None:
        raise TypeError(f"Measurement resolution strategy must define {name}.")
    return value


def _setting_has_symbolic_values(
    module: ModuleBlock,
    setting: SettingNameFamily,
) -> bool:
    return any(
        _is_meaningful_symbolic_value(part)
        for value in setting_values(module, setting)
        for part in value.split(",")
    )


def _is_meaningful_symbolic_value(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized not in {"", "none", "do not use", "leave this black"}


def _measurement_target_scope(value: str) -> MeasurementTargetScope:
    normalized = value.strip().lower()
    if "both" in normalized:
        return MeasurementTargetScope.BOTH
    if "object" in normalized:
        return MeasurementTargetScope.OBJECT
    if "image" in normalized or "entire" in normalized:
        return MeasurementTargetScope.IMAGE
    raise ValueError(f"Unsupported measurement target scope: {value!r}")
