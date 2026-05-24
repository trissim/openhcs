"""Module-level raw-function resolution for generated CellProfiler steps."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, TypeVar

from metaclass_registry import AutoRegisterMeta

from openhcs.interop.cellprofiler.classify_objects_settings import (
    ClassifyObjectsVariant,
)
from openhcs.interop.cellprofiler.grid_settings import (
    FunctionNameVariantResolver,
)
from openhcs.interop.cellprofiler.measurement_scope import (
    CellProfilerMeasurementTargetScope as MeasurementTargetScope,
)
from openhcs.interop.cellprofiler.resize_objects_settings import (
    RESIZE_OBJECTS_FACTOR_Z_SETTING,
    RESIZE_OBJECTS_PLANES_SETTING,
)
from openhcs.interop.cellprofiler.resize_settings import (
    RESIZE_FACTOR_Z_SETTING,
    RESIZE_PLANES_SETTING,
)
from openhcs.interop.cellprofiler.parser import ModuleBlock
from openhcs.interop.cellprofiler.setting_names import (
    OBJECT_MEASUREMENT_SETTING,
    SettingNameFamily,
    setting_values,
    split_symbol_names,
)

RequiredAttrT = TypeVar("RequiredAttrT")
from openhcs.processing.backends.cellprofiler.library import canonical_module_name


@dataclass(frozen=True, slots=True)
class ResolvedModuleFunction:
    """Typed raw-function selection for one generated module."""

    function_name: str


class _ModuleFunctionResolutionStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal family for resolving raw absorbed-function variants."""

    __registry_key__ = "module_name"
    __skip_if_no_key__ = True
    module_name: ClassVar[str | None] = None

    @classmethod
    def for_module(cls, module_name: str) -> "_ModuleFunctionResolutionStrategy":
        canonical_name = canonical_module_name(module_name)
        strategy_type = cls.__registry__.get(canonical_name)
        if strategy_type is not None:
            return strategy_type()
        rule = MODULE_FUNCTION_RESOLUTION_RULES.get(canonical_name)
        if rule is not None:
            return RuleBackedModuleFunctionResolutionStrategy(rule)
        return DefaultModuleFunctionResolutionStrategy()

    @abstractmethod
    def resolve(
        self,
        module: ModuleBlock,
        *,
        default_function_name: str,
    ) -> ResolvedModuleFunction:
        """Resolve the raw absorbed function for one parsed module."""


class DefaultModuleFunctionResolutionStrategy(_ModuleFunctionResolutionStrategy):
    """Use the registry-declared function unchanged."""

    def resolve(
        self,
        module: ModuleBlock,
        *,
        default_function_name: str,
    ) -> ResolvedModuleFunction:
        del module
        return ResolvedModuleFunction(function_name=default_function_name)


class ScopedMeasurementFunctionResolutionStrategy(_ModuleFunctionResolutionStrategy):
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
        scope = measurement_target_scope(
            module,
            setting=_required_class_attr(
                type(self).scope_setting_name,
                "scope_setting_name",
            ),
            default=_required_class_attr(
                type(self).default_scope_value,
                "default_scope_value",
            ),
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


class ObjectInputMeasurementFunctionResolutionStrategy(_ModuleFunctionResolutionStrategy):
    """Resolve object-measurement variants when object inputs are declared."""

    object_setting_name: ClassVar[SettingNameFamily] = OBJECT_MEASUREMENT_SETTING
    object_function_name: ClassVar[str | None] = None

    def resolve(
        self,
        module: ModuleBlock,
        *,
        default_function_name: str,
    ) -> ResolvedModuleFunction:
        if not _setting_has_symbolic_values(module, type(self).object_setting_name):
            return ResolvedModuleFunction(function_name=default_function_name)
        return ResolvedModuleFunction(
            function_name=_required_class_attr(
                type(self).object_function_name,
                "object_function_name",
            )
        )


class ClassifyObjectsFunctionResolutionStrategy(_ModuleFunctionResolutionStrategy):
    """Resolve absorbed ClassifyObjects variants from typed module settings."""

    module_name = "ClassifyObjectsSingleMeasurement"

    def resolve(
        self,
        module: ModuleBlock,
        *,
        default_function_name: str,
    ) -> ResolvedModuleFunction:
        del default_function_name
        return ResolvedModuleFunction(
            function_name=ClassifyObjectsVariant.from_module(module).function_name
        )


class DefineGridFunctionResolutionStrategy(_ModuleFunctionResolutionStrategy):
    """Resolve absorbed DefineGrid variants from typed module settings."""

    module_name = "DefineGridManual"

    def resolve(
        self,
        module: ModuleBlock,
        *,
        default_function_name: str,
    ) -> ResolvedModuleFunction:
        del default_function_name
        return ResolvedModuleFunction(
            function_name=FunctionNameVariantResolver.for_module_name(
                "DefineGrid"
            ).function_name(module)
        )


class IdentifyObjectsInGridFunctionResolutionStrategy(
    _ModuleFunctionResolutionStrategy
):
    """Resolve grid-object identification with or without guiding labels."""

    module_name = "IdentifyObjectsInGrid"

    def resolve(
        self,
        module: ModuleBlock,
        *,
        default_function_name: str,
    ) -> ResolvedModuleFunction:
        del default_function_name
        return ResolvedModuleFunction(
            function_name=FunctionNameVariantResolver.for_module_name(
                "IdentifyObjectsInGrid"
            ).function_name(module)
        )


class VolumetricSettingFunctionResolutionStrategy(_ModuleFunctionResolutionStrategy):
    """Resolve modules with a distinct absorbed function for volumetric settings."""

    volumetric_function_name: ClassVar[str | None] = None
    volumetric_settings: ClassVar[tuple[SettingNameFamily, ...]] = ()

    def resolve(
        self,
        module: ModuleBlock,
        *,
        default_function_name: str,
    ) -> ResolvedModuleFunction:
        if self._has_volumetric_settings(module):
            return ResolvedModuleFunction(
                function_name=_required_class_attr(
                    type(self).volumetric_function_name,
                    "volumetric_function_name",
                )
            )
        return ResolvedModuleFunction(function_name=default_function_name)

    def _has_volumetric_settings(self, module: ModuleBlock) -> bool:
        return any(
            setting_values(module, setting)
            for setting in _required_class_attr(
                type(self).volumetric_settings,
                "volumetric_settings",
            )
        )


@dataclass(frozen=True, slots=True)
class ModuleFunctionResolutionRule:
    """Data-owned function-resolution rule for metadata-only variants."""

    module_name: str
    object_function_name: str | None = None
    scope_setting_name: SettingNameFamily | None = None
    default_scope_value: str | None = None
    volumetric_function_name: str | None = None
    volumetric_settings: tuple[SettingNameFamily, ...] = ()

    @property
    def canonical_module_name(self) -> str:
        return canonical_module_name(self.module_name)

    def resolve(
        self,
        module: ModuleBlock,
        *,
        default_function_name: str,
    ) -> ResolvedModuleFunction:
        if self.volumetric_function_name is not None:
            return self._resolve_volumetric(module, default_function_name)
        if self.scope_setting_name is not None:
            return self._resolve_scoped_measurement(module, default_function_name)
        if self.object_function_name is not None:
            return self._resolve_object_input(module, default_function_name)
        return ResolvedModuleFunction(function_name=default_function_name)

    def _resolve_scoped_measurement(
        self,
        module: ModuleBlock,
        default_function_name: str,
    ) -> ResolvedModuleFunction:
        scope = measurement_target_scope(
            module,
            setting=_required_class_attr(
                self.scope_setting_name,
                "scope_setting_name",
            ),
            default=_required_class_attr(
                self.default_scope_value,
                "default_scope_value",
            ),
        )
        if scope is MeasurementTargetScope.IMAGE or not _setting_has_symbolic_values(
            module,
            OBJECT_MEASUREMENT_SETTING,
        ):
            return ResolvedModuleFunction(function_name=default_function_name)
        return ResolvedModuleFunction(
            function_name=_required_class_attr(
                self.object_function_name,
                "object_function_name",
            )
        )

    def _resolve_object_input(
        self,
        module: ModuleBlock,
        default_function_name: str,
    ) -> ResolvedModuleFunction:
        if not _setting_has_symbolic_values(module, OBJECT_MEASUREMENT_SETTING):
            return ResolvedModuleFunction(function_name=default_function_name)
        return ResolvedModuleFunction(
            function_name=_required_class_attr(
                self.object_function_name,
                "object_function_name",
            )
        )

    def _resolve_volumetric(
        self,
        module: ModuleBlock,
        default_function_name: str,
    ) -> ResolvedModuleFunction:
        if any(setting_values(module, setting) for setting in self.volumetric_settings):
            return ResolvedModuleFunction(
                function_name=_required_class_attr(
                    self.volumetric_function_name,
                    "volumetric_function_name",
                )
            )
        return ResolvedModuleFunction(function_name=default_function_name)


@dataclass(frozen=True, slots=True)
class RuleBackedModuleFunctionResolutionStrategy(_ModuleFunctionResolutionStrategy):
    """Resolve a module function using a declarative metadata-only rule."""

    rule: ModuleFunctionResolutionRule

    def resolve(
        self,
        module: ModuleBlock,
        *,
        default_function_name: str,
    ) -> ResolvedModuleFunction:
        return self.rule.resolve(
            module,
            default_function_name=default_function_name,
        )


MODULE_FUNCTION_RESOLUTION_RULES: dict[str, ModuleFunctionResolutionRule] = {
    rule.canonical_module_name: rule
    for rule in (
        ModuleFunctionResolutionRule(
            module_name="MeasureTexture",
            scope_setting_name=SettingNameFamily(
                "Measure images or objects?",
                aliases=("Measure whole images or objects?",),
            ),
            default_scope_value="Images",
            object_function_name="measure_texture_objects",
        ),
        ModuleFunctionResolutionRule(
            module_name="MeasureColocalization",
            scope_setting_name=SettingNameFamily("Select where to measure correlation"),
            default_scope_value="Across entire image",
            object_function_name="measure_colocalization_objects",
        ),
        ModuleFunctionResolutionRule(
            module_name="MeasureGranularity",
            object_function_name="measure_granularity_objects",
        ),
        ModuleFunctionResolutionRule(
            module_name="Resize",
            volumetric_function_name="resize_volumetric",
            volumetric_settings=(RESIZE_FACTOR_Z_SETTING, RESIZE_PLANES_SETTING),
        ),
        ModuleFunctionResolutionRule(
            module_name="ResizeObjects",
            volumetric_function_name="resize_objects_3d",
            volumetric_settings=(
                RESIZE_OBJECTS_FACTOR_Z_SETTING,
                RESIZE_OBJECTS_PLANES_SETTING,
            ),
        ),
    )
}


def measurement_target_scope(
    module: ModuleBlock,
    *,
    setting: SettingNameFamily,
    default: str,
) -> MeasurementTargetScope:
    """Return a typed measurement target scope from a parsed module setting."""
    return _measurement_target_scope(
        MeasurementTargetScopeSettingResolution(
            module=module,
            setting=setting,
            default=default,
        ).value
    )


@dataclass(frozen=True, slots=True)
class MeasurementTargetScopeSettingResolution:
    """Resolve an optional CP target-scope setting without exception fallback."""

    module: ModuleBlock
    setting: SettingNameFamily
    default: str

    @property
    def value(self) -> str:
        values = setting_values(self.module, self.setting)
        if not values:
            return self.default
        return values[0]


def _required_class_attr(value: RequiredAttrT | None, name: str) -> RequiredAttrT:
    if value is None:
        raise TypeError(f"Measurement resolution strategy must define {name}.")
    return value


def _setting_has_symbolic_values(
    module: ModuleBlock,
    setting: SettingNameFamily,
) -> bool:
    return any(split_symbol_names(value) for value in setting_values(module, setting))


def _measurement_target_scope(value: str) -> MeasurementTargetScope:
    normalized = value.strip().lower()
    if "both" in normalized:
        return MeasurementTargetScope.BOTH
    if "object" in normalized:
        return MeasurementTargetScope.OBJECT
    if "image" in normalized or "entire" in normalized:
        return MeasurementTargetScope.IMAGE
    raise ValueError(f"Unsupported measurement target scope: {value!r}")
