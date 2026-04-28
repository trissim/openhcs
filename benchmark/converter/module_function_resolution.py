"""Module-level raw-function resolution for generated CellProfiler steps."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta

from .parser import ModuleBlock


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
            module_name,
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

    scope_setting_name: ClassVar[str | None] = None
    default_scope_value: ClassVar[str | None] = None
    object_function_name: ClassVar[str | None] = None

    def resolve(
        self,
        module: ModuleBlock,
        *,
        default_function_name: str,
    ) -> ResolvedModuleFunction:
        scope = _measurement_target_scope(
            module.get_setting(
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
        if scope is MeasurementTargetScope.IMAGE:
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
    scope_setting_name = "Measure whole images or objects?"
    default_scope_value = "Images"
    object_function_name = "measure_texture_objects"


class MeasureColocalizationFunctionResolutionStrategy(
    ScopedMeasurementFunctionResolutionStrategy
):
    """Resolve MeasureColocalization image-vs-object absorbed variants."""

    module_name = "MeasureColocalization"
    scope_setting_name = "Select where to measure correlation"
    default_scope_value = "Across entire image"
    object_function_name = "measure_colocalization_objects"


def _required_class_attr(value: str | None, name: str) -> str:
    if value is None:
        raise TypeError(f"Measurement resolution strategy must define {name}.")
    return value


def _measurement_target_scope(value: str) -> MeasurementTargetScope:
    normalized = value.strip().lower()
    if "both" in normalized:
        return MeasurementTargetScope.BOTH
    if "object" in normalized:
        return MeasurementTargetScope.OBJECT
    if "image" in normalized or "entire" in normalized:
        return MeasurementTargetScope.IMAGE
    raise ValueError(f"Unsupported measurement target scope: {value!r}")
