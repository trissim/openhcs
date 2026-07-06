"""CellProfiler setting-to-artifact semantic projection."""

from __future__ import annotations
from dataclasses import dataclass, replace
from openhcs.core.artifacts import ArtifactInputPlan, ArtifactSpec, ArtifactType
from openhcs.core.pipeline.function_contracts import special_output_specs_from_callable
from openhcs.core.source_bindings import EMPTY_SOURCE_BINDINGS
from openhcs.core.special_outputs import (
    SpecialOutputKindClassifier,
    special_output_name,
)
from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting
from openhcs.interop.cellprofiler.settings_binder import (
    normalize_cellprofiler_setting_name,
)


@dataclass(frozen=True, slots=True)
class ArtifactSettingSymbol:
    """One CellProfiler setting value classified as an artifact symbol."""

    artifact_spec: ArtifactSpec
    setting_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_spec, ArtifactSpec):
            raise TypeError(
                f"ArtifactSettingSymbol.artifact_spec must be an ArtifactSpec, got {type(self.artifact_spec).__name__}."
            )
        object.__setattr__(
            self,
            "artifact_spec",
            replace(
                self.artifact_spec,
                name=_normalized_nonempty_name(
                    self.artifact_spec.name, "ArtifactSettingSymbol.name"
                ),
            ),
        )

    @property
    def name(self) -> str:
        return self.artifact_spec.name

    @property
    def artifact_type(self) -> type[ArtifactType]:
        return self.artifact_spec.artifact_type

    @property
    def is_input(self) -> bool:
        return self.artifact_spec.plan_type is ArtifactInputPlan


@dataclass(frozen=True, slots=True)
class FunctionSpecialOutput:
    """One function-declared auxiliary output projected onto artifact type."""

    name: str
    artifact_type: type[ArtifactType]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _normalized_nonempty_name(self.name, "FunctionSpecialOutput.name"),
        )


@dataclass(frozen=True, slots=True)
class DeclaredArtifactSetting:
    """One artifact setting declared by a CellProfilerModule."""

    setting_name: str
    capability_type: type

    def __post_init__(self) -> None:
        from openhcs.interop.cellprofiler.module_declarations import (
            CellProfilerArtifactCapability,
        )

        if not isinstance(self.capability_type, type) or not issubclass(
            self.capability_type, CellProfilerArtifactCapability
        ):
            raise TypeError(
                "DeclaredArtifactSetting.capability_type must inherit CellProfilerArtifactCapability."
            )

    def symbols(self, module: ModuleBlock) -> tuple[ArtifactSettingSymbol, ...]:
        return tuple(
            (
                ArtifactSettingSymbol(self.capability_type.spec(name), setting.name)
                for setting in _iter_module_settings(module)
                if _normalized_setting(setting.name)
                == _normalized_setting(self.setting_name)
                for name in _symbol_names_from_setting(setting)
            )
        )


@dataclass(frozen=True, slots=True)
class DeclaredArtifactSymbolCollector:
    """Lightweight builder that records contract-declared artifact symbols."""

    @property
    def source_schema(self) -> "PipelineImageSchema":
        from openhcs.core.pipeline_image_schema import PipelineImageSchema

        return PipelineImageSchema.empty()

    def require_artifact(self, spec, module: ModuleBlock):
        from openhcs.interop.cellprofiler.symbol_table import CellProfilerSymbol

        del module
        return CellProfilerSymbol(spec)

    def declare_artifact(self, spec, module: ModuleBlock):
        from openhcs.interop.cellprofiler.symbol_table import CellProfilerSymbol

        return CellProfilerSymbol(spec, producer_module_num=module.module_num)

    def optional_artifact(self, spec):
        del spec
        return None

    def measurement_output_for_module_num(self, module_num: int | None):
        del module_num
        return None

    def measurement_outputs(self) -> tuple[object, ...]:
        return ()

    def source_bindings_for(self, symbols, **kwargs):
        del symbols, kwargs
        return EMPTY_SOURCE_BINDINGS


@dataclass(frozen=True, slots=True)
class DeclaredArtifactSettingSymbols:
    """Artifact settings derived from a module declaration and its contract."""

    module_type: type
    module: ModuleBlock

    def symbols(self) -> tuple[ArtifactSettingSymbol, ...]:
        return self._unique(
            (
                *self.explicit_symbols(
                    self.module_type.declared_artifact_input_settings()
                ),
                *self.explicit_symbols(
                    self.module_type.declared_artifact_output_settings()
                ),
                *self.contract_symbols(),
            )
        )

    def explicit_symbols(self, setting_roles) -> tuple[ArtifactSettingSymbol, ...]:
        symbols: list[ArtifactSettingSymbol] = []
        for setting_name, capability_type in setting_roles:
            for concrete_setting_name in self._setting_names(setting_name):
                symbols.extend(
                    DeclaredArtifactSetting(
                        concrete_setting_name, capability_type
                    ).symbols(self.module)
                )
        return tuple(symbols)

    def contract_symbols(self) -> tuple[ArtifactSettingSymbol, ...]:
        from openhcs.interop.cellprofiler.symbol_table import (
            CellProfilerContractAssemblyMixin,
        )

        try:
            contract = self.module_type.artifact_contract(
                CellProfilerContractAssemblyMixin(),
                DeclaredArtifactSymbolCollector(),
                self.module,
            )
        except ValueError:
            return ()
        if contract is None:
            return ()
        symbols_by_name: dict[str, set[ArtifactSpec]] = {}
        for spec in (*contract.inputs, *contract.outputs, *contract.declared_outputs):
            symbols_by_name.setdefault(spec.name, set()).add(spec)
        symbols: list[ArtifactSettingSymbol] = []
        for setting in _iter_module_settings(self.module):
            for name in _symbol_names_from_setting(setting):
                if name not in symbols_by_name:
                    continue
                symbols.extend(
                    (
                        ArtifactSettingSymbol(spec, setting.name)
                        for spec in symbols_by_name[name]
                    )
                )
        return tuple(symbols)

    @staticmethod
    def _setting_names(setting_name) -> tuple[str, ...]:
        from openhcs.interop.cellprofiler.setting_names import setting_names

        return setting_names(setting_name)

    @staticmethod
    def _unique(
        symbols: tuple[ArtifactSettingSymbol, ...],
    ) -> tuple[ArtifactSettingSymbol, ...]:
        unique: list[ArtifactSettingSymbol] = []
        seen: set[tuple[object, str]] = set()
        for symbol in symbols:
            key = (symbol.artifact_spec.ref(), symbol.setting_name)
            if key in seen:
                continue
            unique.append(symbol)
            seen.add(key)
        return tuple(unique)


def artifact_setting_symbols(module: ModuleBlock) -> tuple[ArtifactSettingSymbol, ...]:
    """Return declaration-owned artifact-name settings in .cppipe order."""
    from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule

    module_type = CellProfilerModule.for_module(module.name)
    if module_type is not None:
        return DeclaredArtifactSettingSymbols(module_type, module).symbols()
    return ()


def function_special_outputs(module_name: str) -> tuple[FunctionSpecialOutput, ...]:
    """Return function-declared auxiliary outputs with semantic artifact kinds."""
    from openhcs.processing.backends.cellprofiler import CellProfilerFunctionCatalog

    raw_outputs = special_output_specs_from_callable(
        CellProfilerFunctionCatalog.require_function(module_name)
    )
    return tuple(
        (
            FunctionSpecialOutput(
                name=special_output_name(spec),
                artifact_type=SpecialOutputKindClassifier.kind_for(spec),
            )
            for spec in raw_outputs
        )
    )


def _iter_module_settings(module: ModuleBlock) -> tuple[ModuleSetting, ...]:
    records = module.iter_settings()
    if records:
        return records
    return tuple(
        (
            ModuleSetting(name=name, value=value)
            for name, value in module.settings.items()
        )
    )


def _symbol_names_from_setting(setting: ModuleSetting) -> tuple[str, ...]:
    return _symbol_names_from_value(setting.value)


def _symbol_names_from_value(value: str) -> tuple[str, ...]:
    return tuple(
        (
            part
            for part in (part.strip() for part in value.split(","))
            if part and (not _is_blank_symbol(part))
        )
    )


def _is_blank_symbol(value: str) -> bool:
    return _normalized_setting(value) in {
        "leave_this_black",
        "none",
        "do_not_use",
        "no",
        "not_using",
    }


def _normalized_nonempty_name(value: str, field_name: str) -> str:
    normalized_name = value.strip()
    if not normalized_name:
        raise ValueError(f"{field_name} cannot be empty.")
    return normalized_name


def _normalized_setting(value: str) -> str:
    return normalize_cellprofiler_setting_name(value)
