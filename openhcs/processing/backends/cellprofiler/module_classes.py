"""CellProfiler module class declarations.

This file is the source-of-truth catalog for absorbed CellProfiler modules.
Compatibility registry payloads are derived from these classes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING, Any, ClassVar

from metaclass_registry import AutoRegisterMeta, LazyDiscoveryDict

from openhcs.constants.constants import GroupBy, VariableComponents
from openhcs.core.artifacts import ArtifactKind
from openhcs.core.runtime_invocation import RuntimeInvocationOptions
from openhcs.interop.cellprofiler_setting_normalization import (
    normalize_cellprofiler_setting_name,
)
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    setting_names,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract

if TYPE_CHECKING:
    from openhcs.core.module_artifact_contract import ModuleArtifactContract
    from openhcs.core.pipeline_image_schema import PipelineImageSchema
    from openhcs.interop.cellprofiler.module_artifact_inputs import (
        ModuleArtifactInput,
    )
    from openhcs.interop.cellprofiler.measurement_scope import (
        CellProfilerMeasurementTargetScope,
    )
    from openhcs.interop.cellprofiler.module_function_resolution import (
        ResolvedModuleFunction,
    )
    from openhcs.interop.cellprofiler.module_processing_components import (
        ModuleProcessingComponentRequest,
        ModuleProcessingComponents,
    )
    from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting
    from openhcs.interop.cellprofiler.runtime.payload_types import (
        CellProfilerKwargs,
    )
    from openhcs.interop.cellprofiler.module_roles import ArtifactSpecKey
    from openhcs.core.runtime_exports import RuntimeImageExportSpec
    from openhcs.interop.cellprofiler.semantic_defaults import (
        CellProfilerSemanticDefaultContract,
    )
    from openhcs.interop.cellprofiler.settings_binder import SettingToKeywordBinding
    from openhcs.interop.cellprofiler.settings_binder import SettingsBinder
    from openhcs.interop.cellprofiler.symbol_table import ModuleArtifactContracts
    from openhcs.interop.cellprofiler.symbol_table import (
        CellProfilerContractAssemblyMixin,
        _SymbolTableBuilder,
    )


@dataclass(frozen=True, slots=True)
class ModuleSettingRowRecord:
    """Concrete CellProfiler setting row identity and value."""

    module_name: str
    module_num: int
    setting_name: str
    normalized_setting_name: str
    value: Any


@dataclass(frozen=True, slots=True)
class ModuleSettingCoverageRecord(ModuleSettingRowRecord):
    """Coverage status for one concrete CellProfiler setting row."""

    status: "ModuleSettingCoverageStatus"


class ModuleSettingCoverageStatus(str, Enum):
    """How one CellProfiler setting row was accounted for by import binding."""

    BOUND = "bound"
    ARTIFACT_CONTRACT = "artifact_contract"
    TYPED_IGNORE = "typed_ignore"
    CALLER_IGNORE = "caller_ignore"
    INFRASTRUCTURE = "infrastructure"
    UNMAPPED = "unmapped"

    @classmethod
    def for_setting(
        cls,
        normalized_name: str,
        *,
        binder: "SettingsBinder",
        unmapped_kwargs: Mapping[str, Any],
        ignored_unmapped_settings: frozenset[str],
        artifact_setting_names: frozenset[str],
        typed_ignore_setting_names: frozenset[str],
    ) -> "ModuleSettingCoverageStatus":
        """Return the coverage status owned by this status enum."""
        if normalized_name in binder.SKIP_SETTINGS:
            return cls.INFRASTRUCTURE
        if normalized_name not in unmapped_kwargs:
            return cls.BOUND
        if normalized_name in ignored_unmapped_settings:
            return cls.CALLER_IGNORE
        if normalized_name in artifact_setting_names:
            return cls.ARTIFACT_CONTRACT
        if normalized_name in typed_ignore_setting_names:
            return cls.TYPED_IGNORE
        return cls.UNMAPPED


@dataclass(frozen=True, slots=True)
class BoundModuleSettings:
    """Typed module-setting translation result."""

    kwargs: Mapping[str, Any]
    unmapped_kwargs: Mapping[str, Any] = field(default_factory=dict)
    invocation_options: RuntimeInvocationOptions | None = None
    setting_coverage: tuple[ModuleSettingCoverageRecord, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "kwargs", dict(self.kwargs))
        object.__setattr__(self, "unmapped_kwargs", dict(self.unmapped_kwargs))
        object.__setattr__(self, "setting_coverage", tuple(self.setting_coverage))
        if (
            self.invocation_options is not None
            and not isinstance(self.invocation_options, RuntimeInvocationOptions)
        ):
            raise TypeError(
                "BoundModuleSettings.invocation_options must inherit "
                "RuntimeInvocationOptions."
            )

    def with_kwargs(self, kwargs: Mapping[str, Any]) -> "BoundModuleSettings":
        """Return this binding with additional generated function kwargs."""
        return BoundModuleSettings(
            {**self.kwargs, **kwargs},
            self.unmapped_kwargs,
            self.invocation_options,
            self.setting_coverage,
        )


GeneratedImportCollector = set[tuple[str, str]]


@dataclass(frozen=True, slots=True)
class UnmappedModuleSetting:
    """A CellProfiler setting that no registered binding strategy consumed."""

    module_name: str
    module_num: int
    setting_name: str
    value: Any


class UnmappedModuleSettingsError(ValueError):
    """Raised when enabled module settings are not mapped or explicitly ignored."""

    def __init__(self, settings: tuple[UnmappedModuleSetting, ...]) -> None:
        self.settings = settings
        rendered = "; ".join(
            f"{setting.module_name}({setting.module_num})."
            f"{setting.setting_name}={setting.value!r}"
            for setting in settings
        )
        super().__init__(
            "Enabled CellProfiler modules have unmapped settings. "
            "Add a module settings binding hook or an explicit typed ignore: "
            f"{rendered}"
        )


def _required_string(value: object, name: str, owner: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{owner}.{name} must be a non-empty string.")
    return value


def _string_tuple(value: object, name: str, owner: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise TypeError(f"{owner}.{name} must be a tuple of strings, not str.")
    try:
        values = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{owner}.{name} must be an iterable of strings.") from exc
    if not all(isinstance(item, str) and item.strip() for item in values):
        raise ValueError(f"{owner}.{name} must contain only non-empty strings.")
    return values


def _module_lookup_key(name: str) -> str:
    return _normalize_setting_name(name)


def _declared_lookup_keys(module_type: type["CellProfilerModule"]) -> frozenset[str]:
    return frozenset(
        _module_lookup_key(name)
        for name in (str(module_type.module_name), *module_type.aliases)
    )


def _validate_unique_module_names(module_type: type["CellProfilerModule"]) -> None:
    declared_keys = _declared_lookup_keys(module_type)
    for existing_type in dict.values(CellProfilerModule.__registry__):
        if existing_type is module_type or (
            existing_type.__module__ == module_type.__module__
            and existing_type.__qualname__ == module_type.__qualname__
        ):
            continue
        overlap = declared_keys & _declared_lookup_keys(existing_type)
        if not overlap:
            continue
        names = tuple(sorted(overlap))
        raise ValueError(
            f"{module_type.__name__} duplicates CellProfiler module names or "
            f"aliases declared by {existing_type.__name__}: {names!r}."
        )


class ArtifactContractModule(ABC, metaclass=AutoRegisterMeta):
    """Nominal marker for module declarations that own artifact flow."""

    __registry__ = LazyDiscoveryDict(enable_cache=False)
    __registry_key__ = "module_name"
    __skip_if_no_key__ = True

    module_name: ClassVar[str | None] = None


class CellProfilerModule(ABC, metaclass=AutoRegisterMeta):
    """Auto-registered base class for absorbed CellProfiler modules."""

    __registry__ = LazyDiscoveryDict(enable_cache=False)
    __registry_key__ = "module_name"
    __skip_if_no_key__ = True

    module_name: ClassVar[str | None] = None
    function_name: ClassVar[str | None] = None
    aliases: ClassVar[tuple[str, ...]] = ()
    function_variants: ClassVar[tuple[str, ...]] = ()
    setting_bindings: ClassVar[tuple["SettingToKeywordBinding", ...]] = ()
    setting_parameter_aliases: ClassVar[
        Mapping[str | "SettingNameFamily", str | list[str] | None]
    ] = {}
    ignored_settings: ClassVar[tuple[str | "SettingNameFamily", ...]] = ()
    contract: ClassVar[str] = ProcessingContract.PURE_2D.declared_name
    category: ClassVar[str] = "image_operation"
    confidence: ClassVar[float] = 0.5
    validated: ClassVar[bool] = False
    required_variable_components: ClassVar[tuple[VariableComponents, ...]] = ()
    allowed_group_by: ClassVar[tuple[GroupBy, ...]] = (GroupBy.NONE,)
    semantic_default_contract_types: ClassVar[
        tuple[type["CellProfilerSemanticDefaultContract"], ...]
    ] = ()
    semantic_default_contract_module_name: ClassVar[str | None] = None
    infrastructure_import_note: ClassVar[str | None] = None
    infrastructure_exports_tables: ClassVar[bool] = False
    infrastructure_exports_images: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        module_name = cls.__dict__.get("module_name")
        if module_name is None:
            return
        cls.module_name = _required_string(
            module_name,
            "module_name",
            cls.__name__,
        )
        cls.function_name = _required_string(
            cls.function_name,
            "function_name",
            cls.__name__,
        )
        cls.aliases = _string_tuple(
            cls.aliases,
            "aliases",
            cls.__name__,
        )
        cls.function_variants = _string_tuple(
            cls.function_variants,
            "function_variants",
            cls.__name__,
        )
        if cls.function_name in cls.function_variants:
            raise ValueError(
                f"{cls.__name__} declares primary function "
                f"{cls.function_name!r} as a variant."
            )
        cls.contract = _required_string(
            cls.contract,
            "contract",
            cls.__name__,
        )
        cls.category = _required_string(
            cls.category,
            "category",
            cls.__name__,
        )
        cls.confidence = float(cls.confidence)
        cls.validated = bool(cls.validated)
        cls.required_variable_components = tuple(
            component
            if isinstance(component, VariableComponents)
            else VariableComponents(component)
            for component in cls.required_variable_components
        )
        cls.allowed_group_by = tuple(
            group_by
            if isinstance(group_by, GroupBy)
            else GroupBy(group_by)
            for group_by in cls.allowed_group_by
        )
        _validate_unique_module_names(cls)
        CellProfilerModule.for_module.__func__.cache_clear()

    @classmethod
    def declared_function_names(cls) -> tuple[str, ...]:
        """Return the primary and variant function names declared by this module."""
        return (str(cls.function_name), *cls.function_variants)

    @classmethod
    def normalize_setting_name(cls, setting_name: str) -> str:
        """Return the canonical lookup key for CellProfiler setting labels."""
        del cls
        return normalize_cellprofiler_setting_name(setting_name)

    @classmethod
    @lru_cache(maxsize=512)
    def for_module(
        cls,
        module_name: str,
    ) -> type["CellProfilerModule"] | None:
        """Return the registered module class for a canonical name or alias."""
        lookup_key = _module_lookup_key(module_name)
        for module_type in cls.__registry__.values():
            if _module_lookup_key(str(module_type.module_name)) == lookup_key:
                return module_type
            if lookup_key in {
                _module_lookup_key(alias) for alias in module_type.aliases
            }:
                return module_type
        return None

    @classmethod
    def canonical_module_name(cls, module_name: str) -> str:
        """Return the canonical module name declared by the module class root."""
        module_type = cls.for_module(module_name)
        if module_type is None:
            return _required_string(module_name, "module_name", cls.__name__)
        return str(module_type.module_name)

    @classmethod
    def semantic_default_contracts(
        cls,
    ) -> tuple["CellProfilerSemanticDefaultContract", ...]:
        """Return source-validation contracts owned by this module declaration."""
        contracts = []
        for contract_type in cls.semantic_default_contract_types:
            contract = contract_type()
            if contract.module_name is None:
                contract.module_name = (
                    cls.semantic_default_contract_module_name
                    or str(cls.module_name)
                )
            contracts.append(contract)
        return tuple(contracts)

    @classmethod
    def _bind_declared_settings(
        cls,
        module: "ModuleBlock",
        *,
        binder: "SettingsBinder",
        param_mapping: Mapping[str, Any],
    ) -> "BoundModuleSettings":
        """Bind rows using setting declarations inherited by this module class."""
        del param_mapping
        setting_bindings = tuple(cls.setting_bindings)
        ignored_settings = tuple(cls.ignored_settings_for(module))
        bound_details = binder.bind_with_details(module.settings)
        kwargs = binder.bind_declared(module, setting_bindings)
        mapped_settings = {
            _normalize_setting_name(setting_name)
            for binding in setting_bindings
            for setting_name in setting_names(binding.setting_name)
        }
        mapped_settings.update(
            _normalize_setting_name(concrete_setting_name)
            for setting_name in ignored_settings
            for concrete_setting_name in setting_names(setting_name)
        )
        unmapped_kwargs = {
            detail.name: detail.original_value
            for detail in bound_details
            if detail.name not in mapped_settings
        }
        return BoundModuleSettings(kwargs, unmapped_kwargs)

    @classmethod
    def _bind_generic_settings(
        cls,
        module: "ModuleBlock",
        *,
        binder: "SettingsBinder",
        param_mapping: Mapping[str, Any],
        use_declaration: bool = True,
    ) -> "BoundModuleSettings":
        """Bind docstring/signature-mapped settings through this declaration."""
        from enum import Enum as EnumType
        from inspect import signature
        from typing import Literal, get_args, get_origin, get_type_hints

        from openhcs.processing.backends.cellprofiler.function_documentation import (
            cellprofiler_source_setting_parameter_mapping,
        )
        from openhcs.processing.backends.cellprofiler.library import require_function

        setting_parameter_mapping = dict(param_mapping)
        if use_declaration:
            absorbed_function = require_function(
                str(cls.module_name),
                function_name=str(cls.function_name),
            )
            parameter_names = tuple(signature(absorbed_function).parameters)
            setting_parameter_mapping.update({name: name for name in parameter_names})
            setting_parameter_mapping.update(
                cellprofiler_source_setting_parameter_mapping(
                    str(cls.module_name),
                    parameter_names,
                )
            )
            for setting_name, parameter_name in cls.setting_parameter_aliases.items():
                for concrete_setting_name in setting_names(setting_name):
                    setting_parameter_mapping[
                        _normalize_setting_name(concrete_setting_name)
                    ] = parameter_name
            for setting_name in cls.ignored_settings_for(module):
                for concrete_setting_name in setting_names(setting_name):
                    setting_parameter_mapping[
                        _normalize_setting_name(concrete_setting_name)
                    ] = None
            annotations = get_type_hints(absorbed_function)
        else:
            annotations = {}

        bound_kwargs = binder.bind(module.settings)
        coerced_kwargs: dict[str, Any] = {}
        for setting_name, value in bound_kwargs.items():
            parameter_name = setting_parameter_mapping.get(setting_name)
            annotation = (
                annotations.get(parameter_name)
                if isinstance(parameter_name, str)
                else None
            )
            if annotation is None:
                coerced_kwargs[setting_name] = value
                continue
            origin = get_origin(annotation)
            if origin is Literal and isinstance(value, str):
                normalized_value = _normalize_setting_name(value)
                for literal in get_args(annotation):
                    if (
                        isinstance(literal, str)
                        and normalized_value
                        == _normalize_setting_name(literal)
                    ):
                        value = literal
                        break
            elif isinstance(annotation, type) and issubclass(annotation, EnumType):
                value = _coerce_enum_literal(annotation, value)
            coerced_kwargs[setting_name] = value

        translated_kwargs: dict[str, Any] = {}
        unmapped_kwargs: dict[str, Any] = {}
        for cp_setting, value in coerced_kwargs.items():
            if cp_setting not in setting_parameter_mapping:
                unmapped_kwargs[cp_setting] = value
                continue
            py_param = setting_parameter_mapping[cp_setting]
            if py_param is None:
                continue
            if isinstance(py_param, list):
                if isinstance(value, tuple) and len(value) == len(py_param):
                    for index, param_name in enumerate(py_param):
                        translated_kwargs[param_name] = value[index]
                else:
                    translated_kwargs[py_param[0]] = value
            else:
                translated_kwargs[py_param] = value

        return BoundModuleSettings(translated_kwargs, unmapped_kwargs)

    @classmethod
    def _finalize_bound_settings(
        cls,
        module: "ModuleBlock",
        *,
        binder: "SettingsBinder",
        bound: "BoundModuleSettings",
        ignored_unmapped_settings: frozenset[str] = frozenset(),
    ) -> "BoundModuleSettings":
        """Validate and annotate a binding result produced by this declaration."""
        from openhcs.interop.cellprofiler.artifact_semantics import (
            artifact_setting_symbols,
        )
        runtime_kwargs = cls.runtime_kwargs(module)
        if runtime_kwargs:
            bound = bound.with_kwargs(runtime_kwargs)

        artifact_setting_names = frozenset(
            _normalize_setting_name(symbol.setting_name)
            for symbol in artifact_setting_symbols(module)
        )
        typed_ignore_setting_names = frozenset(
            _normalize_setting_name(concrete_name)
            for setting_name in cls.ignored_settings_for(module)
            for concrete_name in setting_names(setting_name)
        )
        unmapped_kwargs = {
            setting_name: value
            for setting_name, value in bound.unmapped_kwargs.items()
            if setting_name not in ignored_unmapped_settings
            and setting_name not in artifact_setting_names
            and setting_name not in typed_ignore_setting_names
        }
        setting_coverage: list[ModuleSettingCoverageRecord] = []
        for setting in module.iter_settings():
            normalized_name = _normalize_setting_name(setting.name)
            setting_coverage.append(
                ModuleSettingCoverageRecord(
                    module_name=module.name,
                    module_num=module.module_num,
                    setting_name=setting.name,
                    normalized_setting_name=normalized_name,
                    value=setting.value,
                    status=ModuleSettingCoverageStatus.for_setting(
                        normalized_name,
                        binder=binder,
                        unmapped_kwargs=bound.unmapped_kwargs,
                        ignored_unmapped_settings=ignored_unmapped_settings,
                        artifact_setting_names=artifact_setting_names,
                        typed_ignore_setting_names=typed_ignore_setting_names,
                    ),
                )
            )
        if unmapped_kwargs:
            raise UnmappedModuleSettingsError(
                tuple(
                    UnmappedModuleSetting(
                        module_name=module.name,
                        module_num=module.module_num,
                        setting_name=setting_name,
                        value=value,
                    )
                    for setting_name, value in sorted(unmapped_kwargs.items())
                )
            )
        return BoundModuleSettings(
            bound.kwargs,
            unmapped_kwargs,
            bound.invocation_options,
            tuple(setting_coverage),
        )

    @classmethod
    def bind_settings(
        cls,
        module: "ModuleBlock",
        *,
        binder: "SettingsBinder",
        param_mapping: Mapping[str, Any],
        ignored_unmapped_settings: frozenset[str] = frozenset(),
    ) -> "BoundModuleSettings":
        """Bind parsed module settings through this module declaration."""
        if cls.setting_parameter_aliases:
            bound = cls._bind_generic_settings(
                module,
                binder=binder,
                param_mapping=param_mapping,
            )
            return cls._finalize_bound_settings(
                module,
                binder=binder,
                bound=cls.postprocess_bound_settings(module, bound),
                ignored_unmapped_settings=ignored_unmapped_settings,
            )

        if cls.setting_bindings:
            bound = cls._bind_declared_settings(
                module,
                binder=binder,
                param_mapping=param_mapping,
            )
            return cls._finalize_bound_settings(
                module,
                binder=binder,
                bound=cls.postprocess_bound_settings(module, bound),
                ignored_unmapped_settings=ignored_unmapped_settings,
            )

        bound = cls._bind_generic_settings(
            module,
            binder=binder,
            param_mapping=param_mapping,
            use_declaration=bool(cls.ignored_settings),
        )
        return cls._finalize_bound_settings(
            module,
            binder=binder,
            bound=cls.postprocess_bound_settings(module, bound),
            ignored_unmapped_settings=ignored_unmapped_settings,
        )

    @classmethod
    def postprocess_bound_settings(
        cls,
        module: "ModuleBlock",
        bound: "BoundModuleSettings",
    ) -> "BoundModuleSettings":
        """Apply module-local binding semantics after declared settings bind."""
        del module
        return bound

    @classmethod
    def runtime_kwargs(cls, module: "ModuleBlock") -> Mapping[str, Any]:
        """Return declaration-owned runtime-selection kwargs for this module."""
        del module
        return {}

    @classmethod
    def generated_invocation_options_literal(
        cls,
        options: RuntimeInvocationOptions | None,
        *,
        import_collector: GeneratedImportCollector,
    ) -> str | None:
        """Return generated-source literal for declaration-owned invocation options."""
        del import_collector
        if options is None:
            return None
        raise TypeError(
            f"{cls.__name__} does not declare generated lowering for "
            f"{type(options).__name__}."
        )

    @classmethod
    def ignored_settings_for(
        cls,
        module: "ModuleBlock",
    ) -> tuple[str | "SettingNameFamily", ...]:
        """Return settings consumed outside direct runtime kwargs."""
        del module
        return cls.ignored_settings

    @classmethod
    def setting_value(
        cls,
        module: "ModuleBlock",
        setting_name: str | "SettingNameFamily",
        *,
        include_blank: bool = False,
    ) -> str | None:
        """Return a module setting value through the module declaration boundary."""
        if not include_blank:
            return _optional_setting_value(module, setting_name)
        for setting in module.iter_settings():
            if _setting_name_matches(setting.name, setting_name):
                return setting.value.strip()
        for candidate_name, value in module.settings.items():
            if _setting_name_matches(candidate_name, setting_name):
                return value.strip()
        return None

    @classmethod
    def artifact_inputs(
        cls,
        module: "ModuleBlock",
        source_schema: "PipelineImageSchema",
    ) -> tuple["ModuleArtifactInput", ...]:
        """Return artifact inputs declared directly by this module class."""
        del source_schema
        from openhcs.interop.cellprofiler.module_artifact_inputs import (
            ModuleArtifactInput,
        )

        inputs: list[ModuleArtifactInput] = []
        for setting_name, artifact_kind in cls.declared_artifact_input_settings():
            setting_value = _optional_setting_value(
                module,
                cls.declared_setting_name(setting_name),
            )
            if setting_value is None:
                continue
            artifact_name = _normalized_symbol_name(setting_value)
            if artifact_name is not None:
                inputs.append(ModuleArtifactInput(artifact_name, artifact_kind))
        return tuple(inputs)

    @classmethod
    def declared_artifact_input_settings(
        cls,
    ) -> tuple[
        tuple[
            str | "SettingNameFamily",
            ArtifactKind,
        ],
        ...,
    ]:
        """Return declared CellProfiler setting families and their artifact kind."""
        return ()

    @classmethod
    def declared_artifact_output_settings(
        cls,
    ) -> tuple[
        tuple[
            str | "SettingNameFamily",
            ArtifactKind,
        ],
        ...,
    ]:
        """Return declared CellProfiler output setting families and their artifact kind."""
        return ()

    @classmethod
    def source_image_types_by_alias(
        cls,
        module: "ModuleBlock",
    ) -> Mapping[str, str]:
        """Return source-image role refinements implied by this module's inputs."""
        del module
        return {}

    @classmethod
    def resolve_function(
        cls,
        module: "ModuleBlock",
        *,
        default_function_name: str | None = None,
    ) -> "ResolvedModuleFunction":
        """Return the function selected by this module declaration."""
        del module
        from openhcs.interop.cellprofiler.module_function_resolution import (
            ResolvedModuleFunction,
        )

        return ResolvedModuleFunction(default_function_name or str(cls.function_name))

    @classmethod
    def processing_components(
        cls,
        request: "ModuleProcessingComponentRequest",
    ) -> "ModuleProcessingComponents":
        """Return generated FunctionStep component semantics for this module."""
        from openhcs.interop.cellprofiler.module_processing_components import (
            default_module_processing_components,
        )

        components = default_module_processing_components(request)
        if not cls.required_variable_components:
            return components
        return components.with_required_variable_components(
            cls.required_variable_components,
            module_name=str(cls.module_name),
        )

    @classmethod
    def runtime_object_measurement_row_policy(cls):
        """Return the object-measurement row policy declared by this module."""
        from openhcs.interop.cellprofiler.runtime.object_measurement_row_policies import (
            CellProfilerObjectMeasurementRowPolicy,
            DefaultObjectMeasurementRowPolicy,
        )

        if issubclass(cls, CellProfilerObjectMeasurementRowPolicy):
            return cls()
        return DefaultObjectMeasurementRowPolicy()

    @classmethod
    def infrastructure_retained_artifacts(
        cls,
        module: "ModuleBlock",
        *,
        contracts_by_module_num: Mapping[int, "ModuleArtifactContracts"],
    ) -> frozenset["ArtifactSpecKey"]:
        """Return artifacts this module keeps alive when handled as infrastructure."""
        del module, contracts_by_module_num
        return frozenset()

    @classmethod
    def image_export_specs(
        cls,
        module: "ModuleBlock",
    ) -> tuple["RuntimeImageExportSpec", ...]:
        """Return runtime image-export expectations declared by this module."""
        del module
        return ()

    @classmethod
    def measurement_artifact_name(cls, module: "ModuleBlock") -> str:
        """Return the standard CellProfiler measurement artifact name."""
        return f"{module.name}_{module.module_num}_measurements"

    @classmethod
    def declared_setting_name(
        cls,
        setting: str | "SettingNameFamily",
    ) -> str | "SettingNameFamily":
        """Return a concrete setting declaration."""
        return setting

    @staticmethod
    def declared_setting_value(
        setting: str | "SettingNameFamily" | Callable[[], str | "SettingNameFamily"],
    ) -> str | "SettingNameFamily":
        """Resolve a declaration supplied as a value or lazy class hook."""
        return setting() if callable(setting) else setting

    @classmethod
    def artifact_inputs_from_setting(
        cls,
        builder: "_SymbolTableBuilder",
        module: "ModuleBlock",
        setting: str | "SettingNameFamily" | None,
        kind: ArtifactKind,
    ) -> tuple[object, ...]:
        """Require artifacts named by one declared CellProfiler setting family."""
        from openhcs.core.artifacts import ArtifactSpec
        from openhcs.interop.cellprofiler.setting_names import (
            setting_values,
            split_symbol_names,
        )

        if setting is None:
            return ()
        declared_setting = cls.declared_setting_name(cls.declared_setting_value(setting))
        return tuple(
            builder.require_artifact(ArtifactSpec(name, kind), module)
            for value in setting_values(module, declared_setting)
            for name in split_symbol_names(value)
        )

    @classmethod
    def measurement_artifact_contract_from_declared_settings(
        cls,
        assembler: "CellProfilerContractAssemblyMixin",
        builder: "_SymbolTableBuilder",
        module: "ModuleBlock",
    ) -> "ModuleArtifactContracts":
        """Assemble the standard measurement contract from declared settings."""
        from openhcs.core.artifacts import ArtifactSpec

        inputs = cls.measurement_artifact_inputs(builder, module)
        outputs = [
            builder.declare_artifact(
                ArtifactSpec(
                    cls.measurement_artifact_name(module),
                    ArtifactKind.MEASUREMENTS,
                ),
                module,
            )
        ]
        return assembler.assemble_contract(module, builder, inputs=inputs, outputs=outputs)

    @classmethod
    def measurement_artifact_inputs(
        cls,
        builder: "_SymbolTableBuilder",
        module: "ModuleBlock",
    ) -> tuple[object, ...]:
        """Return measurement-input artifacts from declared module settings."""
        return tuple(
            artifact
            for setting, kind in cls.declared_artifact_input_settings()
            for artifact in cls.artifact_inputs_from_setting(
                builder,
                module,
                setting,
                kind,
            )
        )

    @classmethod
    def artifact_contract_inputs(
        cls,
        builder: "_SymbolTableBuilder",
        module: "ModuleBlock",
    ) -> tuple[object, ...]:
        """Return artifacts consumed by this module's declared contract."""
        return cls.measurement_artifact_inputs(builder, module)

    @classmethod
    def artifact_contract_outputs(
        cls,
        builder: "_SymbolTableBuilder",
        module: "ModuleBlock",
    ) -> tuple[object, ...]:
        """Return artifacts produced by this module's declared contract."""
        return cls.declared_output_artifacts_from_settings(builder, module)

    @classmethod
    def declared_output_artifacts_from_settings(
        cls,
        builder: "_SymbolTableBuilder",
        module: "ModuleBlock",
    ) -> tuple[object, ...]:
        """Declare output artifacts named by CellProfiler setting families."""
        from openhcs.core.artifacts import ArtifactSpec
        from openhcs.interop.cellprofiler.setting_names import (
            setting_values,
            split_symbol_names,
        )

        return tuple(
            builder.declare_artifact(ArtifactSpec(name, kind), module)
            for setting, kind in cls.declared_artifact_output_settings()
            for value in setting_values(
                module,
                cls.declared_setting_name(cls.declared_setting_value(setting)),
            )
            for name in split_symbol_names(value)
        )

    @classmethod
    def measurement_output_artifact(
        cls,
        builder: "_SymbolTableBuilder",
        module: "ModuleBlock",
    ) -> object:
        """Declare the standard CellProfiler measurement output artifact."""
        from openhcs.core.artifacts import ArtifactSpec

        return builder.declare_artifact(
            ArtifactSpec(
                cls.measurement_artifact_name(module),
                ArtifactKind.MEASUREMENTS,
            ),
            module,
        )

    @classmethod
    def parent_child_relationship_output_artifact(
        cls,
        builder: "_SymbolTableBuilder",
        module: "ModuleBlock",
        *,
        parent_name: str,
        child_name: str,
    ) -> object:
        """Declare a relationship artifact from parent/child object artifacts."""
        from openhcs.core.artifacts import ArtifactSpec
        from openhcs.core.runtime_semantics import parent_child_relationship_artifact_name

        return builder.declare_artifact(
            ArtifactSpec(
                parent_child_relationship_artifact_name(parent_name, child_name),
                ArtifactKind.RELATIONSHIPS,
            ),
            module,
        )

    @classmethod
    def artifact_contract(
        cls,
        assembler: "CellProfilerContractAssemblyMixin",
        builder: "_SymbolTableBuilder",
        module: "ModuleBlock",
    ) -> "ModuleArtifactContracts | None":
        """Return typed declaration-owned artifact flow, when this module owns it."""
        inputs = cls.artifact_contract_inputs(builder, module)
        outputs = cls.artifact_contract_outputs(builder, module)
        if not inputs and not outputs:
            return None
        return assembler.assemble_contract(
            module,
            builder,
            inputs=inputs,
            outputs=outputs,
        )

    @classmethod
    def preserve_duplicate_artifact_inputs(cls, module: "ModuleBlock") -> bool:
        """Return whether repeated same-name inputs are distinct module roles."""
        del module
        return False

    @classmethod
    def source_binding_participates_in_image_stack(
        cls,
        module: "ModuleBlock",
        symbol: "CellProfilerSymbol",
        input_symbols: tuple["CellProfilerSymbol", ...],
    ) -> bool:
        """Return whether a source-bound symbol anchors image-stack execution."""
        del module, symbol, input_symbols
        return True

    @classmethod
    def relationship_measurement_rows(
        cls,
        request: object,
    ) -> object:
        """Return the relationship-row projector owned by this module declaration."""
        from openhcs.interop.cellprofiler.runtime.relationship_measurement_rows import (
            GenericRelationshipMeasurementRows,
        )

        return GenericRelationshipMeasurementRows(request)

    @classmethod
    def relationship_endpoint_contract(
        cls,
        resolver: object,
        relationship_spec: object,
    ) -> object | None:
        """Return a declaration-owned endpoint contract for one relationship output."""
        del resolver, relationship_spec
        return None

    @classmethod
    def relationship_distance_measurements_apply(
        cls,
        resolver: object,
        relationship_spec: object,
    ) -> bool:
        """Return whether this relationship output owns distance measurement rows."""
        del resolver, relationship_spec
        return False

    @classmethod
    def measurement_record(cls, request: object) -> object:
        """Return the measurement record declared by this module."""
        from openhcs.interop.cellprofiler.runtime.measurement_recording import (
            DefaultMeasurementRecordModule,
        )

        return DefaultMeasurementRecordModule.measurement_record(request)


class PlaneRuntimeArtifactModule(ABC):
    """Parent for modules that consume source-aligned runtime artifacts by plane."""

    allowed_group_by: ClassVar[tuple[GroupBy, ...]] = tuple(GroupBy)


class PerObjectMeasurementExecutionModule(PlaneRuntimeArtifactModule):
    """Parent for modules invoked once per measured object set."""


class ComposedImageObjectMeasurementExecutionModule(
    PerObjectMeasurementExecutionModule,
):
    """Parent for object measurements that consume composed image payloads."""


class ObjectMeasurementRowsModule(CellProfilerModule):
    """Parent for modules whose declaration is also the object-row policy."""

    @classmethod
    def object_measurement_setting(cls) -> "SettingNameFamily":
        from openhcs.interop.cellprofiler.setting_names import SettingNameFamily

        return SettingNameFamily(
            "Select object sets to measure",
            aliases=("Select objects to measure", "Select an object to measure"),
        )


class InfrastructureCellProfilerModule(CellProfilerModule):
    """Parent for modules handled as OpenHCS import/runtime infrastructure."""

    @classmethod
    def declared_function_names(cls) -> tuple[str, ...]:
        """Infrastructure declarations are not executable backend functions."""
        return ()


class ModuleSettingsSourceModule(CellProfilerModule):
    """Parent for declarations that lower settings without binder context."""

    invocation_options_source: ClassVar[
        Callable[["ModuleBlock"], RuntimeInvocationOptions] | None
    ] = None

    @classmethod
    @abstractmethod
    def settings_source(cls, module: "ModuleBlock") -> "CellProfilerKwargs":
        """Return absorbed-function kwargs owned by this module declaration."""
        raise NotImplementedError

    @classmethod
    def bind_settings(
        cls,
        module: "ModuleBlock",
        *,
        binder: "SettingsBinder",
        param_mapping: Mapping[str, Any],
        ignored_unmapped_settings: frozenset[str] = frozenset(),
    ) -> "BoundModuleSettings":
        del param_mapping
        return cls._finalize_bound_settings(
            module,
            binder=binder,
            bound=BoundModuleSettings(
                cls.settings_source(module),
                {},
                cls.invocation_options_source(module)
                if cls.invocation_options_source is not None
                else None,
            ),
            ignored_unmapped_settings=ignored_unmapped_settings,
        )


class BinderSettingsSourceModule(CellProfilerModule):
    """Parent for declarations that lower settings with binder parsing."""

    invocation_options_source: ClassVar[
        Callable[["ModuleBlock"], RuntimeInvocationOptions] | None
    ] = None

    @classmethod
    @abstractmethod
    def settings_source(
        cls,
        module: "ModuleBlock",
        binder: "SettingsBinder",
    ) -> "CellProfilerKwargs":
        """Return absorbed-function kwargs owned by this module declaration."""
        raise NotImplementedError

    @classmethod
    def bind_settings(
        cls,
        module: "ModuleBlock",
        *,
        binder: "SettingsBinder",
        param_mapping: Mapping[str, Any],
        ignored_unmapped_settings: frozenset[str] = frozenset(),
    ) -> "BoundModuleSettings":
        del param_mapping
        return cls._finalize_bound_settings(
            module,
            binder=binder,
            bound=BoundModuleSettings(
                cls.settings_source(module, binder),
                {},
                cls.invocation_options_source(module)
                if cls.invocation_options_source is not None
                else None,
            ),
            ignored_unmapped_settings=ignored_unmapped_settings,
        )


class CellProfilerStructuringElement(Enum):
    """CellProfiler morphology structuring-element shape literal."""

    DISK = "disk"
    SQUARE = "square"
    DIAMOND = "diamond"
    OCTAGON = "octagon"
    STAR = "star"
    BALL = "ball"
    CUBE = "cube"
    OCTAHEDRON = "octahedron"


STRUCTURING_ELEMENT_SETTING_NAME = "Structuring element"
DEFAULT_STRUCTURING_ELEMENT_SETTING = "disk,3"


@dataclass(frozen=True, slots=True)
class StructuringElementSetting:
    """Typed CellProfiler morphology footprint setting."""

    structuring_element: CellProfilerStructuringElement
    size: int

    @classmethod
    def from_cellprofiler_value(
        cls,
        value: Any,
    ) -> "StructuringElementSetting":
        from openhcs.interop.cellprofiler.settings_binder import (
            coerce_cellprofiler_enum,
        )

        shape, size = _structuring_element_parts(value)
        return cls(
            structuring_element=coerce_cellprofiler_enum(
                CellProfilerStructuringElement,
                shape,
            ),
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

    setting_name: str | "SettingNameFamily" = STRUCTURING_ELEMENT_SETTING_NAME
    legacy_size_setting_name: str | "SettingNameFamily" | None = "Size"
    default_value: str = DEFAULT_STRUCTURING_ELEMENT_SETTING
    shape_keyword: str = "structuring_element"
    size_keyword: str = "size"

    @property
    def normalized_setting_names(self) -> frozenset[str]:
        from openhcs.interop.cellprofiler.setting_names import setting_names

        names = set(
            normalize_cellprofiler_setting_name(setting_name)
            for setting_name in setting_names(self.setting_name)
        )
        if self.legacy_size_setting_name is not None:
            names.update(
                normalize_cellprofiler_setting_name(setting_name)
                for setting_name in setting_names(self.legacy_size_setting_name)
            )
        return frozenset(names)

    def bound_kwargs(
        self,
        module: "ModuleBlock",
        binder: "SettingsBinder",
    ) -> dict[str, str | int]:
        parsed_value = self.parsed_setting(module, binder)
        return StructuringElementSetting.from_cellprofiler_value(
            parsed_value
        ).bound_kwargs(
            shape_keyword=self.shape_keyword,
            size_keyword=self.size_keyword,
        )

    def parsed_setting(
        self,
        module: "ModuleBlock",
        binder: "SettingsBinder",
    ) -> tuple[Any, Any]:
        from openhcs.interop.cellprofiler.setting_names import (
            optional_setting_value,
            setting_names,
        )

        raw_value = optional_setting_value(module, self.setting_name)
        if raw_value is not None:
            return _structuring_element_parts(
                binder.parse_value(setting_names(self.setting_name)[0], raw_value)
            )

        legacy_size = self.legacy_size(module, binder)
        if legacy_size is None:
            return _structuring_element_parts(
                binder.parse_value(setting_names(self.setting_name)[0], self.default_value)
            )
        default_shape, _default_size = _structuring_element_parts(self.default_value)
        return default_shape, legacy_size

    def legacy_size(
        self,
        module: "ModuleBlock",
        binder: "SettingsBinder",
    ) -> Any | None:
        from openhcs.interop.cellprofiler.setting_names import (
            optional_setting_value,
            setting_names,
        )

        if self.legacy_size_setting_name is None:
            return None
        raw_value = optional_setting_value(module, self.legacy_size_setting_name)
        if raw_value is None:
            return None
        return binder.parse_value(setting_names(self.legacy_size_setting_name)[0], raw_value)


def structuring_element_bound_kwargs(
    module: "ModuleBlock",
    binder: "SettingsBinder",
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


class StructuringElementSettingsModule(BinderSettingsSourceModule):
    """Parent for modules sharing CellProfiler structuring-element lowering."""

    structuring_element_binding: ClassVar[
        "StructuringElementSettingBinding | None"
    ] = None

    @classmethod
    def _resolved_structuring_element_binding(
        cls,
    ) -> "StructuringElementSettingBinding":
        if cls.structuring_element_binding is not None:
            return cls.structuring_element_binding

        return StructuringElementSettingBinding()

    @classmethod
    def settings_source(
        cls,
        module: "ModuleBlock",
        binder: "SettingsBinder",
    ) -> "CellProfilerKwargs":
        binding = cls._resolved_structuring_element_binding()
        return binding.bound_kwargs(module, binder)

    @classmethod
    def bind_settings(
        cls,
        module: "ModuleBlock",
        *,
        binder: "SettingsBinder",
        param_mapping: Mapping[str, Any],
        ignored_unmapped_settings: frozenset[str] = frozenset(),
    ) -> "BoundModuleSettings":
        binding = cls._resolved_structuring_element_binding()
        if cls.setting_bindings:
            bound = cls._bind_declared_settings(
                module,
                binder=binder,
                param_mapping=param_mapping,
            )
        else:
            bound = cls._bind_generic_settings(
                module,
                binder=binder,
                param_mapping=param_mapping,
                use_declaration=bool(cls.ignored_settings),
            )
        kwargs = dict(bound.kwargs)
        kwargs.update(binding.bound_kwargs(module, binder))
        unmapped_kwargs = dict(bound.unmapped_kwargs)
        for setting_name in binding.normalized_setting_names:
            unmapped_kwargs.pop(setting_name, None)
        return cls._finalize_bound_settings(
            module,
            binder=binder,
            bound=cls.postprocess_bound_settings(
                module,
                BoundModuleSettings(
                    kwargs,
                    unmapped_kwargs,
                    bound.invocation_options,
                ),
            ),
            ignored_unmapped_settings=ignored_unmapped_settings,
        )


def _normalize_setting_name(value: str) -> str:
    return CellProfilerModule.normalize_setting_name(str(value))


def _setting_name_matches(
    candidate_name: str,
    declared_name: str | SettingNameFamily,
) -> bool:
    normalized_candidate = _normalize_setting_name(candidate_name)
    return any(
        normalized_candidate == _normalize_setting_name(name)
        for name in setting_names(declared_name)
    )


def _setting_values(
    module: "ModuleBlock",
    setting_name: str | SettingNameFamily,
) -> tuple[str, ...]:
    values: list[str] = []
    for setting in module.iter_settings():
        if _setting_name_matches(setting.name, setting_name):
            values.append(setting.value.strip())
    if values:
        return tuple(values)
    for candidate_name, value in module.settings.items():
        if _setting_name_matches(candidate_name, setting_name):
            values.append(value.strip())
    return tuple(values)


def _optional_setting_value(module: "ModuleBlock", setting_name: object) -> str | None:
    values = _setting_values(module, setting_name)
    return values[-1] if values else None


def _normalized_symbol_name(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _coerce_enum_literal(enum_type: type[Enum], value: object) -> Enum:
    if isinstance(value, enum_type):
        return value
    normalized = str(value).strip().lower()
    for member in enum_type:
        if str(member.value).strip().lower() == normalized or member.name.lower() == normalized:
            return member
    raise ValueError(f"Unsupported {enum_type.__name__} literal: {value!r}.")


def _parse_cellprofiler_float(value: str) -> float:
    return float(str(value).strip())


def _parse_cellprofiler_int(value: str) -> int:
    return int(float(str(value).strip()))


def _parse_cellprofiler_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"yes", "true", "1"}:
        return True
    if normalized in {"no", "false", "0"}:
        return False
    raise ValueError(f"Unsupported CellProfiler boolean literal: {value!r}.")


def _cellprofiler_setting_token(value: Any) -> str:
    """Return a stable comparison token for parsed CellProfiler settings."""
    if isinstance(value, Enum) and isinstance(value.value, str):
        value = value.value
    return " ".join(str(value).strip().lower().replace("-", " ").split())


class RepeatedSettingValuePolicy(ABC, metaclass=AutoRegisterMeta):
    """Nominal resolver for CellProfiler settings that reuse the same label."""

    __registry_key__ = "policy_key"
    __skip_if_no_key__ = True
    setting_name: ClassVar[str | None] = None
    policy_key: ClassVar[str | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.__dict__.get("policy_key") is not None:
            return
        setting_name = cls.__dict__.get("setting_name")
        if isinstance(setting_name, str):
            cls.policy_key = setting_name

    @classmethod
    def for_setting(
        cls,
        setting_name: str,
    ) -> "RepeatedSettingValuePolicy":
        lookup_key = CellProfilerModule.normalize_setting_name(setting_name)
        strategy_type = next(
            (
                policy_type
                for policy_type in cls.__registry__.values()
                if CellProfilerModule.normalize_setting_name(str(policy_type.policy_key))
                == lookup_key
            ),
            LastRepeatedSettingValuePolicy,
        )
        return strategy_type()

    def value(
        self,
        module: "ModuleBlock",
        setting_name: str | "SettingNameFamily",
    ) -> str | None:
        values = _setting_values(module, setting_name)
        if not values:
            return None
        if len(values) == 1:
            return values[0]
        return self._resolve_repeated_value(module, setting_name, tuple(values))

    @abstractmethod
    def _resolve_repeated_value(
        self,
        module: "ModuleBlock",
        setting_name: str | "SettingNameFamily",
        values: tuple[str, ...],
    ) -> str:
        """Return the semantically active value for a repeated setting label."""


class LastRepeatedSettingValuePolicy(RepeatedSettingValuePolicy):
    """Default CellProfiler scalar behavior: the later row is authoritative."""

    def _resolve_repeated_value(
        self,
        module: "ModuleBlock",
        setting_name: str | "SettingNameFamily",
        values: tuple[str, ...],
    ) -> str:
        del module, setting_name
        return values[-1]


class ImageArtifactInputModule(CellProfilerModule, ArtifactContractModule):
    """Parent for modules that consume image artifacts through declared settings."""

    image_input_settings: ClassVar[tuple[str | "SettingNameFamily", ...]] = ()
    artifact_kind = ArtifactKind.IMAGE

    @classmethod
    def image_input_setting_names(cls) -> tuple[str | "SettingNameFamily", ...]:
        """Return image-input setting families declared by this module."""
        return cls.image_input_settings

    @classmethod
    def declared_artifact_input_settings(
        cls,
    ) -> tuple[tuple[str | "SettingNameFamily", ArtifactKind], ...]:
        return (
            *super().declared_artifact_input_settings(),
            *(
                (setting, ImageArtifactInputModule.artifact_kind)
                for setting in cls.image_input_setting_names()
            ),
        )


class ObjectArtifactInputModule(CellProfilerModule, ArtifactContractModule):
    """Parent for modules that consume object-label artifacts through declared settings."""

    object_input_settings: ClassVar[tuple[str | "SettingNameFamily", ...]] = ()
    artifact_kind = ArtifactKind.OBJECT_LABELS

    @classmethod
    def object_input_setting_names(cls) -> tuple[str | "SettingNameFamily", ...]:
        """Return object-label input setting families declared by this module."""
        return cls.object_input_settings

    @classmethod
    def declared_artifact_input_settings(
        cls,
    ) -> tuple[tuple[str | "SettingNameFamily", ArtifactKind], ...]:
        return (
            *super().declared_artifact_input_settings(),
            *(
                (setting, ObjectArtifactInputModule.artifact_kind)
                for setting in cls.object_input_setting_names()
            ),
        )


class ImageArtifactOutputModule(CellProfilerModule, ArtifactContractModule):
    """Parent for modules that emit image artifacts through declared settings."""

    image_output_settings: ClassVar[tuple[str | "SettingNameFamily", ...]] = ()
    artifact_kind = ArtifactKind.IMAGE

    @classmethod
    def image_output_setting_names(cls) -> tuple[str | "SettingNameFamily", ...]:
        """Return image-output setting families declared by this module."""
        return cls.image_output_settings

    @classmethod
    def declared_artifact_output_settings(
        cls,
    ) -> tuple[tuple[str | "SettingNameFamily", ArtifactKind], ...]:
        return (
            *super().declared_artifact_output_settings(),
            *(
                (setting, ImageArtifactOutputModule.artifact_kind)
                for setting in cls.image_output_setting_names()
            ),
        )


class ObjectArtifactOutputModule(CellProfilerModule, ArtifactContractModule):
    """Parent for modules that emit object-label artifacts through declared settings."""

    object_output_settings: ClassVar[tuple[str | "SettingNameFamily", ...]] = ()
    artifact_kind = ArtifactKind.OBJECT_LABELS

    @classmethod
    def object_output_setting_names(cls) -> tuple[str | "SettingNameFamily", ...]:
        """Return object-label output setting families declared by this module."""
        return cls.object_output_settings

    @classmethod
    def declared_artifact_output_settings(
        cls,
    ) -> tuple[tuple[str | "SettingNameFamily", ArtifactKind], ...]:
        return (
            *super().declared_artifact_output_settings(),
            *(
                (setting, ObjectArtifactOutputModule.artifact_kind)
                for setting in cls.object_output_setting_names()
            ),
        )


class MeasurementArtifactOutputModule(CellProfilerModule, ArtifactContractModule):
    """Parent for modules that emit the standard measurement artifact."""

    artifact_kind = ArtifactKind.MEASUREMENTS

    @classmethod
    def artifact_contract_outputs(
        cls,
        builder: "_SymbolTableBuilder",
        module: "ModuleBlock",
    ) -> tuple[object, ...]:
        return (
            cls.measurement_output_artifact(builder, module),
            *super().artifact_contract_outputs(builder, module),
        )


class ObjectLineageTransformContractModule(
    PlaneRuntimeArtifactModule,
    MeasurementArtifactOutputModule,
    ObjectArtifactInputModule,
    ObjectArtifactOutputModule,
):
    """Parent for one-object-input modules that emit object lineage."""

    artifact_kind = ArtifactKind.RELATIONSHIPS

    input_objects_setting: ClassVar[str | "SettingNameFamily"]
    output_objects_setting: ClassVar[str | "SettingNameFamily"]

    @classmethod
    def object_input_setting_names(cls) -> tuple[str | "SettingNameFamily", ...]:
        return (cls.input_objects_setting,)

    @classmethod
    def object_output_setting_names(cls) -> tuple[str | "SettingNameFamily", ...]:
        return (cls.output_objects_setting,)

    @classmethod
    def artifact_contract_outputs(
        cls,
        builder: "_SymbolTableBuilder",
        module: "ModuleBlock",
    ) -> tuple[object, ...]:
        from openhcs.interop.cellprofiler.setting_names import required_setting_value

        parent_name = required_setting_value(module, cls.input_objects_setting)
        child_name = required_setting_value(module, cls.output_objects_setting)
        return (
            cls.measurement_output_artifact(builder, module),
            cls.parent_child_relationship_output_artifact(
                builder,
                module,
                parent_name=parent_name,
                child_name=child_name,
            ),
            *cls.declared_output_artifacts_from_settings(builder, module),
        )


class ImageMeasurementInputModule(CellProfilerModule):
    """Parent for measurement modules that consume image measurement inputs."""

    @classmethod
    def image_measurement_setting(cls) -> "SettingNameFamily":
        from openhcs.interop.cellprofiler.setting_names import SettingNameFamily

        return SettingNameFamily(
            "Select images to measure",
            aliases=("Select an image to measure", "Select the image to measure"),
        )

    @classmethod
    def measurement_artifact_inputs(
        cls,
        builder: "_SymbolTableBuilder",
        module: "ModuleBlock",
    ) -> tuple[object, ...]:
        return (
            *cls.artifact_inputs_from_setting(
                builder,
                module,
                cls.declared_setting_value(cls.image_measurement_setting),
                ArtifactKind.IMAGE,
            ),
            *super().measurement_artifact_inputs(builder, module),
        )


class ObjectMeasurementInputModule(CellProfilerModule):
    """Parent for measurement modules that consume object-label measurement inputs."""

    @classmethod
    def object_measurement_setting(cls) -> "SettingNameFamily":
        from openhcs.interop.cellprofiler.setting_names import SettingNameFamily

        return SettingNameFamily(
            "Select object sets to measure",
            aliases=("Select objects to measure", "Select an object to measure"),
        )

    @classmethod
    def measurement_artifact_inputs(
        cls,
        builder: "_SymbolTableBuilder",
        module: "ModuleBlock",
    ) -> tuple[object, ...]:
        return (
            *super().measurement_artifact_inputs(builder, module),
            *cls.artifact_inputs_from_setting(
                builder,
                module,
                cls.declared_setting_value(cls.object_measurement_setting),
                ArtifactKind.OBJECT_LABELS,
            ),
        )




class ScopedMeasurementModule(ImageMeasurementInputModule, ObjectMeasurementInputModule):
    """Module declaration parent for CellProfiler modules with target-scope settings."""

    measurement_scope_setting: ClassVar["SettingNameFamily"]

    @classmethod
    @abstractmethod
    def measurement_target_scope(cls, module: "ModuleBlock") -> Enum:
        """Return the declaration-owned typed measurement target scope."""
        raise NotImplementedError

    @classmethod
    def runtime_measurement_target_scope(
        cls,
        module: "ModuleBlock",
    ) -> "CellProfilerMeasurementTargetScope":
        """Lower the module-local target-scope enum to the runtime scope enum."""
        from openhcs.interop.cellprofiler.measurement_scope import (
            CellProfilerMeasurementTargetScope,
        )

        target_scope = cls.measurement_target_scope(module)
        member_name = "OBJECT" if target_scope.name == "objects" else target_scope.name
        return CellProfilerMeasurementTargetScope[member_name.upper()]

    @classmethod
    def postprocess_bound_settings(
        cls,
        module: "ModuleBlock",
        bound: "BoundModuleSettings",
    ) -> "BoundModuleSettings":
        """Attach the declaration-owned measurement target scope to runtime kwargs."""
        from openhcs.interop.cellprofiler.runtime.binding_authorities import (
            CellProfilerInvocationOverrideKwarg,
        )

        return super().postprocess_bound_settings(module, bound).with_kwargs(
            {
                CellProfilerInvocationOverrideKwarg.measurement_target_scope: (
                    cls.runtime_measurement_target_scope(module)
                ),
            }
        )


























































































































































































__all__ = (
    "CellProfilerModule",
    "ModuleSettingsSourceModule",
    "BinderSettingsSourceModule",
    "CellProfilerStructuringElement",
    "DEFAULT_STRUCTURING_ELEMENT_SETTING",
    "STRUCTURING_ELEMENT_SETTING_NAME",
    "StructuringElementSetting",
    "StructuringElementSettingBinding",
    "StructuringElementSettingsModule",
    "structuring_element_bound_kwargs",
    "ScopedMeasurementModule",
    "PerObjectMeasurementExecutionModule",
    "ComposedImageObjectMeasurementExecutionModule",
    "PlaneRuntimeArtifactModule",
    "ObjectMeasurementRowsModule",
)
