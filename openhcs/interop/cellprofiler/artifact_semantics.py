"""CellProfiler setting-to-artifact semantic projection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from openhcs.core.alias_property import AliasProperty

from openhcs.core.artifacts import ArtifactKind
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


class ArtifactSettingDirection(str, Enum):
    """Whether one setting names a consumed or produced artifact."""

    INPUT = "input"
    OUTPUT = "output"


class ArtifactSettingRole(Enum):
    """Closed semantic roles for CellProfiler artifact-name settings."""

    INPUT_IMAGE = (ArtifactSettingDirection.INPUT, ArtifactKind.IMAGE)
    INPUT_OBJECTS = (ArtifactSettingDirection.INPUT, ArtifactKind.OBJECT_LABELS)
    OUTPUT_IMAGE = (ArtifactSettingDirection.OUTPUT, ArtifactKind.IMAGE)
    OUTPUT_OBJECTS = (
        ArtifactSettingDirection.OUTPUT,
        ArtifactKind.OBJECT_LABELS,
    )
    INPUT_SPATIAL_GRID = (
        ArtifactSettingDirection.INPUT,
        ArtifactKind.SPATIAL_GRID,
    )
    OUTPUT_SPATIAL_GRID = (
        ArtifactSettingDirection.OUTPUT,
        ArtifactKind.SPATIAL_GRID,
    )

    def __init__(
        self,
        direction: ArtifactSettingDirection,
        artifact_kind: ArtifactKind,
    ) -> None:
        self._direction = direction
        self._artifact_kind = artifact_kind

    direction = AliasProperty[ArtifactSettingDirection]("_direction")
    artifact_kind = AliasProperty[ArtifactKind]("_artifact_kind")

    @property
    def is_input(self) -> bool:
        return self.direction is ArtifactSettingDirection.INPUT


@dataclass(frozen=True, slots=True)
class ArtifactSettingSymbol:
    """One CellProfiler setting value classified as an artifact symbol."""

    role: ArtifactSettingRole
    name: str
    setting_name: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _normalized_nonempty_name(
                self.name,
                "ArtifactSettingSymbol.name",
            ),
        )


@dataclass(frozen=True, slots=True)
class FunctionSpecialOutput:
    """One function-declared auxiliary output projected onto artifact kind."""

    name: str
    kind: ArtifactKind

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _normalized_nonempty_name(
                self.name,
                "FunctionSpecialOutput.name",
            ),
        )


class ArtifactSettingRoleAuthority:
    """Resolve a direction/kind pair to the closed artifact-setting role."""

    @classmethod
    def role_for(
        cls,
        direction: ArtifactSettingDirection,
        artifact_kind: ArtifactKind,
    ) -> ArtifactSettingRole | None:
        del cls
        for role in ArtifactSettingRole:
            if role.direction is direction and role.artifact_kind is artifact_kind:
                return role
        return None


@dataclass(frozen=True, slots=True)
class DeclaredArtifactSetting:
    """One artifact setting declared by a CellProfilerModule."""

    setting_name: str
    role: ArtifactSettingRole

    def symbols(self, module: ModuleBlock) -> tuple[ArtifactSettingSymbol, ...]:
        return tuple(
            ArtifactSettingSymbol(self.role, name, setting.name)
            for setting in _iter_module_settings(module)
            if _normalized_setting(setting.name)
            == _normalized_setting(self.setting_name)
            for name in _symbol_names_from_setting(setting)
        )


@dataclass(frozen=True, slots=True)
class DeclaredArtifactSymbolCollector:
    """Lightweight builder that records contract-declared artifact symbols."""

    @property
    def source_schema(self) -> "PipelineImageSchema":
        from openhcs.core.pipeline_image_schema import PipelineImageSchema

        return PipelineImageSchema.empty()

    def require_artifact(self, spec, module: ModuleBlock):
        from openhcs.interop.cellprofiler.symbol_table import (
            CellProfilerSymbol,
            CellProfilerSymbolKind,
        )

        del module
        return CellProfilerSymbol(
            spec.name,
            CellProfilerSymbolKind.from_artifact_kind(spec.kind),
        )

    def declare_artifact(self, spec, module: ModuleBlock):
        from openhcs.interop.cellprofiler.symbol_table import (
            CellProfilerSymbol,
            CellProfilerSymbolKind,
        )

        return CellProfilerSymbol(
            spec.name,
            CellProfilerSymbolKind.from_artifact_kind(spec.kind),
            producer_module_num=module.module_num,
            sidecar_role=spec.sidecar_role,
        )

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
                    ArtifactSettingDirection.INPUT,
                    self.module_type.declared_artifact_input_settings(),
                ),
                *self.explicit_symbols(
                    ArtifactSettingDirection.OUTPUT,
                    self.module_type.declared_artifact_output_settings(),
                ),
                *self.contract_symbols(),
            )
        )

    def explicit_symbols(
        self,
        direction: ArtifactSettingDirection,
        setting_roles,
    ) -> tuple[ArtifactSettingSymbol, ...]:
        symbols: list[ArtifactSettingSymbol] = []
        for setting_name, artifact_kind in setting_roles:
            role = ArtifactSettingRoleAuthority.role_for(direction, artifact_kind)
            if role is None:
                continue
            for concrete_setting_name in self._setting_names(setting_name):
                symbols.extend(
                    DeclaredArtifactSetting(concrete_setting_name, role).symbols(
                        self.module
                    )
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
        symbols_by_name: dict[str, set[ArtifactSettingRole]] = {}
        for direction, contract_symbols in (
            (ArtifactSettingDirection.INPUT, contract.input_symbols),
            (ArtifactSettingDirection.OUTPUT, contract.output_symbols),
            (ArtifactSettingDirection.OUTPUT, contract.declared_output_symbols),
        ):
            for symbol in contract_symbols:
                role = ArtifactSettingRoleAuthority.role_for(
                    direction,
                    symbol.kind.artifact_kind,
                )
                if role is None:
                    continue
                symbols_by_name.setdefault(symbol.name, set()).add(role)
        return tuple(
            ArtifactSettingSymbol(role, name, setting.name)
            for setting in _iter_module_settings(self.module)
            for name in _symbol_names_from_setting(setting)
            for role in symbols_by_name.get(name, ())
        )

    @staticmethod
    def _setting_names(setting_name) -> tuple[str, ...]:
        from openhcs.interop.cellprofiler.setting_names import setting_names

        return setting_names(setting_name)

    @staticmethod
    def _unique(
        symbols: tuple[ArtifactSettingSymbol, ...],
    ) -> tuple[ArtifactSettingSymbol, ...]:
        unique: list[ArtifactSettingSymbol] = []
        seen: set[tuple[ArtifactSettingRole, str, str]] = set()
        for symbol in symbols:
            key = (symbol.role, symbol.name, symbol.setting_name)
            if key in seen:
                continue
            unique.append(symbol)
            seen.add(key)
        return tuple(unique)


def artifact_setting_symbols(module: ModuleBlock) -> tuple[ArtifactSettingSymbol, ...]:
    """Return declaration-owned artifact-name settings in .cppipe order."""
    from openhcs.processing.backends.cellprofiler.module_classes import (
        CellProfilerModule,
    )

    module_type = CellProfilerModule.for_module(module.name)
    if module_type is not None:
        return DeclaredArtifactSettingSymbols(module_type, module).symbols()
    return ()


def function_special_outputs(module_name: str) -> tuple[FunctionSpecialOutput, ...]:
    """Return function-declared auxiliary outputs with semantic artifact kinds."""
    from openhcs.processing.backends.cellprofiler import require_cellprofiler_function

    raw_outputs = special_output_specs_from_callable(
        require_cellprofiler_function(module_name)
    )
    return tuple(
        FunctionSpecialOutput(
            name=special_output_name(spec),
            kind=SpecialOutputKindClassifier.kind_for(spec),
        )
        for spec in raw_outputs
    )


def _iter_module_settings(module: ModuleBlock) -> tuple[ModuleSetting, ...]:
    records = module.iter_settings()
    if records:
        return records
    return tuple(
        ModuleSetting(name=name, value=value)
        for name, value in module.settings.items()
    )


def _symbol_names_from_setting(setting: ModuleSetting) -> tuple[str, ...]:
    return _symbol_names_from_value(setting.value)


def _symbol_names_from_value(value: str) -> tuple[str, ...]:
    return tuple(
        part
        for part in (part.strip() for part in value.split(","))
        if part and not _is_blank_symbol(part)
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
