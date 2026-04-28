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


class MeasureTextureFunctionResolutionStrategy(ModuleFunctionResolutionStrategy):
    """Resolve MeasureTexture image-vs-object absorbed variants."""

    module_name = "MeasureTexture"

    def resolve(
        self,
        module: ModuleBlock,
        *,
        default_function_name: str,
    ) -> ResolvedModuleFunction:
        scope = _measurement_target_scope(
            module.get_setting("Measure whole images or objects?", "Images")
        )
        if scope is MeasurementTargetScope.IMAGE:
            return ResolvedModuleFunction(function_name=default_function_name)
        return ResolvedModuleFunction(function_name="measure_texture_objects")


class MeasureColocalizationFunctionResolutionStrategy(
    ModuleFunctionResolutionStrategy
):
    """Resolve MeasureColocalization image-vs-object absorbed variants."""

    module_name = "MeasureColocalization"

    def resolve(
        self,
        module: ModuleBlock,
        *,
        default_function_name: str,
    ) -> ResolvedModuleFunction:
        scope = _measurement_target_scope(
            module.get_setting(
                "Select where to measure correlation",
                "Across entire image",
            )
        )
        if scope is MeasurementTargetScope.IMAGE:
            return ResolvedModuleFunction(function_name=default_function_name)
        return ResolvedModuleFunction(
            function_name="measure_colocalization_objects"
        )


def _measurement_target_scope(value: str) -> MeasurementTargetScope:
    normalized = value.strip().lower()
    if "both" in normalized:
        return MeasurementTargetScope.BOTH
    if "object" in normalized:
        return MeasurementTargetScope.OBJECT
    if "image" in normalized or "entire" in normalized:
        return MeasurementTargetScope.IMAGE
    raise ValueError(f"Unsupported measurement target scope: {value!r}")
