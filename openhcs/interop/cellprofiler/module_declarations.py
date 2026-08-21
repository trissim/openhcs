"""CellProfiler module class declarations.

This file owns the registered CellProfiler module catalog and discovery boundary.
"""

from __future__ import annotations
from abc import ABC
from collections.abc import Callable, Iterable
from dataclasses import replace
import importlib
import inspect
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
)
from metaclass_registry import AutoRegisterMeta, LazyDiscoveryDict, RegistryConfig
from openhcs.constants.constants import AllComponents, GroupBy, VariableComponents
from openhcs.constants.input_source import InputSource
from openhcs.core.artifacts import ArtifactSpec, ArtifactSpecRef
from openhcs.core.callable_contract import (
    CallableContract,
    CallableImportIdentity,
    FunctionStepExecutionScope,
)
from openhcs.core.config import ProcessingConfig, StepSourceBindingsConfig
from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
from openhcs.core.source_bindings import (
    SourceBindingsConfig,
)
from openhcs.interop.cellprofiler_setting_normalization import (
    normalize_cellprofiler_setting_name,
)
from openhcs.interop.cellprofiler.setting_names import (
    setting_name_matches,
    setting_names,
)
from openhcs.interop.cellprofiler.runtime.measurement_recording import (
    TableMeasurementRecordRowsMixin,
)
from openhcs.interop.cellprofiler.module_artifact_contracts import (
    CellProfilerModuleArtifactContracts,
)
from openhcs.interop.cellprofiler.module_callable_abi import (
    CellProfilerModuleCallableABI,
)
from openhcs.interop.cellprofiler.module_measurement_features import (
    CellProfilerMeasurementFeatureOwner,
)
from openhcs.interop.cellprofiler.module_settings import (
    CellProfilerModuleSettings,
)
from openhcs.processing.backends.lib_registry.openhcs_registry import (
    OpenHCSFunctionCatalogDeclaration,
)

_CELLPROFILER_BACKEND_PACKAGE = "openhcs.processing.backends.cellprofiler"
_CELLPROFILER_MODULE_REGISTRY = LazyDiscoveryDict(enable_cache=False)
if TYPE_CHECKING:
    from openhcs.core.function_patterns import (
        NormalizedFunctionItem,
    )
    from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting


def _required_string(value: object, name: str, owner: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{owner}.{name} must be a non-empty string.")
    return value


def _string_tuple(value: object, name: str, owner: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise TypeError(f"{owner}.{name} must be a tuple of strings, not str.")
    try:
        values = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{owner}.{name} must be an iterable of strings.") from exc
    if not all((isinstance(item, str) and item.strip() for item in values)):
        raise ValueError(f"{owner}.{name} must contain only non-empty strings.")
    return values


def _module_lookup_key(name: str) -> str:
    return normalize_cellprofiler_setting_name(name)


def _declared_lookup_keys(module_type: type["CellProfilerModule"]) -> frozenset[str]:
    return frozenset(
        (
            _module_lookup_key(name)
            for name in (module_type.require_module_name(), *module_type.aliases)
        )
    )


def _validate_unique_module_names(module_type: type["CellProfilerModule"]) -> None:
    declared_keys = _declared_lookup_keys(module_type)
    declared_functions = module_type.declared_function_names()
    duplicate_functions = tuple(
        dict.fromkeys(
            function_name
            for function_name in declared_functions
            if declared_functions.count(function_name) > 1
        )
    )
    if duplicate_functions:
        raise ValueError(
            f"{module_type.__name__} declares duplicate CellProfiler function "
            f"names: {duplicate_functions!r}."
        )
    for existing_type in dict.values(CellProfilerModule.__registry__):
        if existing_type is module_type or (
            existing_type.__module__ == module_type.__module__
            and existing_type.__qualname__ == module_type.__qualname__
        ):
            continue
        module_overlap = declared_keys & _declared_lookup_keys(existing_type)
        if module_overlap:
            names = tuple(sorted(module_overlap))
            raise ValueError(
                f"{module_type.__name__} duplicates CellProfiler module names or "
                f"aliases declared by {existing_type.__name__}: {names!r}."
            )
        function_overlap = frozenset(declared_functions) & frozenset(
            existing_type.declared_function_names()
        )
        if function_overlap:
            names = tuple(sorted(function_overlap))
            raise ValueError(
                f"{module_type.__name__} duplicates CellProfiler function names "
                f"declared by {existing_type.__name__}: {names!r}."
            )


class CellProfilerModule(
    OpenHCSFunctionCatalogDeclaration,
    CellProfilerModuleCallableABI,
    CellProfilerModuleArtifactContracts,
    CellProfilerModuleSettings,
    TableMeasurementRecordRowsMixin,
    CellProfilerMeasurementFeatureOwner,
    ABC,
    metaclass=AutoRegisterMeta,
    registry_config=RegistryConfig(
        registry_dict=_CELLPROFILER_MODULE_REGISTRY,
        key_attribute="module_name",
        skip_if_no_key=True,
        registry_name="CellProfiler module",
        discovery_package=_CELLPROFILER_BACKEND_PACKAGE,
    ),
):
    """Auto-registered base class for absorbed CellProfiler modules."""

    __registry__ = _CELLPROFILER_MODULE_REGISTRY
    __registry_key__ = "module_name"
    __skip_if_no_key__ = True
    module_name: ClassVar[str | None] = None
    function_name: ClassVar[str | None] = None
    aliases: ClassVar[tuple[str, ...]] = ()
    function_variants: ClassVar[tuple[str, ...]] = ()
    registry_catalog_module: ClassVar[str] = _CELLPROFILER_BACKEND_PACKAGE
    confidence: ClassVar[float] = 0.5
    validated: ClassVar[bool] = False
    respects_masks: ClassVar[bool] = False
    group_by: ClassVar[GroupBy | None] = None
    """Whether coalesced generated emissions must stay explicit per group.

    Most grouped emissions with identical public callable settings can be exposed
    as one normal OpenHCS callable. Modules that need per-group invocation
    contracts can opt into explicit dict-pattern emission at the declaration
    boundary.
    """

    @staticmethod
    def number_step_invocation_blocks(
        invocation_blocks: tuple[tuple["ModuleBlock", ...], ...],
        *,
        first_module_num: int,
    ) -> tuple[tuple[tuple["ModuleBlock", ...], ...], int]:
        """Number known logical module-block tuples within one FunctionStep.

        Each outer tuple is already one declared logical module occurrence. Public
        FunctionStep reconstruction uses ``number_public_step_invocation_blocks``
        to derive those occurrences from nominal repeated-setting ownership.
        """

        from openhcs.interop.cellprofiler.parser import ModuleBlock

        if type(first_module_num) is not int or first_module_num < 1:
            raise TypeError(
                "CellProfiler module numbering requires a positive integer "
                f"first_module_num, got {first_module_num!r}."
            )
        if not isinstance(invocation_blocks, tuple):
            raise TypeError(
                "CellProfiler module numbering requires a tuple of invocation "
                "block tuples."
            )

        numbered_invocations: list[tuple[ModuleBlock, ...]] = []
        next_module_num = first_module_num
        for blocks in invocation_blocks:
            if not isinstance(blocks, tuple) or not blocks:
                raise TypeError(
                    "Each CellProfiler invocation must provide a non-empty "
                    "ModuleBlock tuple."
                )
            invalid = tuple(
                block for block in blocks if not isinstance(block, ModuleBlock)
            )
            if invalid:
                raise TypeError(
                    "CellProfiler invocation numbering requires ModuleBlock "
                    f"values, got {type(invalid[0]).__name__}."
                )
            module_numbers = tuple(
                range(next_module_num, next_module_num + len(blocks))
            )
            next_module_num += len(blocks)
            numbered_invocations.append(
                tuple(
                    replace(block, module_num=module_num)
                    for block, module_num in zip(
                        blocks,
                        module_numbers,
                        strict=True,
                    )
                )
            )

        return tuple(numbered_invocations), next_module_num

    @classmethod
    def public_invocation_module_identity(
        cls,
        invocation: "NormalizedFunctionItem",
        blocks: tuple["ModuleBlock", ...],
    ) -> object:
        """Return one public invocation's declaration-owned logical module identity."""

        repeated_setting_names = tuple(
            binding.setting_name
            for binding in cls.declared_setting_bindings()
            if binding.repeated
        )
        if not repeated_setting_names:
            return (cls, invocation.key)
        projected_blocks = tuple(
            replace(
                block,
                module_num=0,
                setting_records=[
                    record
                    for record in block.setting_records
                    if not any(
                        setting_name_matches(record.name, setting_name)
                        for setting_name in repeated_setting_names
                    )
                ],
            )
            for block in blocks
        )
        return (cls, projected_blocks)

    @classmethod
    def number_public_step_invocation_blocks(
        cls,
        invocations: tuple[
            tuple["NormalizedFunctionItem", tuple["ModuleBlock", ...]], ...
        ],
        *,
        first_module_num: int,
    ) -> tuple[tuple[tuple["ModuleBlock", ...], ...], int]:
        """Number public invocations from nominal module and repeated-row identity."""

        logical_identities: list[object] = []
        numbered_invocations: list[tuple[ModuleBlock, ...]] = []
        next_module_num = first_module_num
        for invocation, blocks in invocations:
            module_type = cls.require_callable_contract_owner(invocation.contract)
            logical_identity = module_type.public_invocation_module_identity(
                invocation,
                blocks,
            )
            try:
                identity_index = logical_identities.index(logical_identity)
            except ValueError:
                (numbered_blocks,), next_module_num = cls.number_step_invocation_blocks(
                    (blocks,),
                    first_module_num=next_module_num,
                )
            else:
                module_numbers = tuple(
                    block.module_num for block in numbered_invocations[identity_index]
                )
                numbered_blocks = tuple(
                    replace(block, module_num=module_num)
                    for block, module_num in zip(
                        blocks,
                        module_numbers,
                        strict=True,
                    )
                )
            logical_identities.append(logical_identity)
            numbered_invocations.append(numbered_blocks)

        return tuple(numbered_invocations), next_module_num

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if "source_qualified_measurement_feature_enum_types" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must inherit SourceQualifiedMeasurementFeatureModule "
                "instead of declaring source_qualified_measurement_feature_enum_types."
            )
        if "measurement_feature_enum_types" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must declare RuntimeMeasurementFeature enum classes "
                "on the module class instead of maintaining measurement_feature_enum_types."
            )
        module_name = cls.__dict__.get("module_name")
        if module_name is None:
            return
        cls.module_name = _required_string(module_name, "module_name", cls.__name__)
        cls.aliases = _string_tuple(cls.aliases, "aliases", cls.__name__)
        cls.function_variants = _string_tuple(
            cls.function_variants, "function_variants", cls.__name__
        )
        if cls.function_name is None:
            if cls.function_variants:
                raise ValueError(
                    f"{cls.__name__} cannot declare function variants without a "
                    "primary function."
                )
        else:
            cls.function_name = _required_string(
                cls.function_name, "function_name", cls.__name__
            )
        if cls.function_name is not None and cls.function_name in cls.function_variants:
            raise ValueError(
                f"{cls.__name__} declares primary function {cls.function_name!r} as a variant."
            )
        resolved_setting_parameters = tuple(
            binding.require_parameter_name()
            for binding in cls.declared_setting_bindings()
            if not binding.declares_artifact
        )
        duplicate_setting_parameters = tuple(
            dict.fromkeys(
                parameter_name
                for parameter_name in resolved_setting_parameters
                if resolved_setting_parameters.count(parameter_name) > 1
            )
        )
        if duplicate_setting_parameters:
            raise ValueError(
                f"{cls.__name__} declares duplicate setting keyword bindings: "
                f"{duplicate_setting_parameters!r}."
            )
        cls.confidence = float(cls.confidence)
        cls.validated = bool(cls.validated)
        if cls.group_by is not None:
            cls.group_by = (
                cls.group_by
                if isinstance(cls.group_by, GroupBy)
                else GroupBy(cls.group_by)
            )
        _validate_unique_module_names(cls)
        CellProfilerModule._measurement_feature_marker_types_for_key_payload.__func__.cache_clear()
        CellProfilerModule.measurement_feature_types.__func__.cache_clear()
        CellProfilerModule.alternative_measurement_feature_part_aliases.__func__.cache_clear()
        CellProfilerModule.measurement_feature_part_rewrite_declarations.__func__.cache_clear()
        CellProfilerModule.measurement_category_prefix_declarations.__func__.cache_clear()
        CellProfilerModule.measurement_source_feature_prefix_declarations.__func__.cache_clear()
        CellProfilerModule.calculated_measurement_feature_prefix_declarations.__func__.cache_clear()
        CellProfilerModule.numbered_measurement_feature_prefix_alias_declarations.__func__.cache_clear()
        CellProfilerModule.scale_qualified_measurement_feature_prefix_declarations.__func__.cache_clear()

    @classmethod
    def declared_function_names(cls) -> tuple[str, ...]:
        """Return the primary and variant function names declared by this module."""
        if cls.function_name is None:
            return ()
        return (str(cls.function_name), *cls.function_variants)

    @classmethod
    def contribute_source_bindings(
        cls,
        module: "ModuleBlock",
        config: "SourceBindingsConfig",
    ) -> "SourceBindingsConfig":
        """Return source declarations contributed by one parsed setup module."""
        del cls, module
        return config

    @classmethod
    def source_bindings_for_modules(
        cls,
        modules: Iterable["ModuleBlock"],
        config: "SourceBindingsConfig",
    ) -> "SourceBindingsConfig":
        """Fold enabled modules into one public source-binding configuration."""
        from openhcs.core.source_bindings import SourceBindingsConfig

        if not isinstance(config, SourceBindingsConfig):
            raise TypeError(
                "CellProfiler source binding import requires SourceBindingsConfig, "
                f"got {type(config).__name__}."
            )
        for module in modules:
            if not module.enabled:
                continue
            config = cls.require_module(module.name).contribute_source_bindings(
                module,
                config,
            )
        return config

    @classmethod
    def emits_function_step(cls) -> bool:
        """Return whether an enabled parsed module emits an executable step."""
        return True

    @classmethod
    def uses_cellprofiler_runtime_adapter(cls) -> bool:
        """Return whether axis execution requires the CellProfiler workspace adapter."""

        return True

    @classmethod
    def for_backend_function_name(
        cls,
        function_name: str,
    ) -> type["CellProfilerModule"] | None:
        """Resolve a function exposed inside the CellProfiler backend namespace."""
        normalized_name = _required_string(
            function_name,
            "function_name",
            cls.__name__,
        )
        matches = tuple(
            module_type
            for module_type in cls.__registry__.values()
            if normalized_name in module_type.declared_function_names()
        )
        if len(matches) > 1:
            raise ValueError(
                f"CellProfiler function {normalized_name!r} is owned by multiple "
                "module declarations: "
                f"{tuple(item.require_module_name() for item in matches)!r}."
            )
        return matches[0] if matches else None

    @classmethod
    def for_callable_import_identity(
        cls,
        identity: CallableImportIdentity,
    ) -> type["CellProfilerModule"] | None:
        """Return the module owning one complete callable import identity."""

        if not isinstance(identity, CallableImportIdentity):
            raise TypeError(
                "CellProfiler callable ownership requires CallableImportIdentity, "
                f"got {type(identity).__name__}."
            )
        matches = tuple(
            module_type
            for module_type in cls.__registry__.values()
            if module_type.__module__ == identity.module_name
            and identity.function_name in module_type.declared_function_names()
        )
        if len(matches) > 1:
            raise ValueError(
                f"CellProfiler callable identity {identity.import_path!r} is owned "
                "by multiple module declarations: "
                f"{tuple(item.require_module_name() for item in matches)!r}."
            )
        return matches[0] if matches else None

    @classmethod
    def for_callable_contract(
        cls,
        contract: CallableContract,
    ) -> type["CellProfilerModule"] | None:
        """Resolve and validate the CellProfiler owner of a callable contract."""

        if not isinstance(contract, CallableContract):
            raise TypeError(
                "CellProfiler callable ownership requires CallableContract, got "
                f"{type(contract).__name__}."
            )
        raw_callable = contract.resolve_canonical_raw_callable()
        identity = CallableImportIdentity.from_callable(raw_callable)
        module_type = cls.for_callable_import_identity(identity)
        if module_type is None:
            return None
        canonical_callable = module_type.require_callable(identity.function_name)
        if raw_callable is not canonical_callable:
            raise ValueError(
                f"Callable import identity {identity.import_path!r} claims "
                f"CellProfiler module {module_type.__name__}, but its object is "
                "not the declaration-owned canonical callable."
            )
        return module_type

    @classmethod
    def require_callable_contract_owner(
        cls,
        contract: CallableContract,
    ) -> type["CellProfilerModule"]:
        """Return the exact CellProfiler owner of ``contract`` or fail."""

        module_type = cls.for_callable_contract(contract)
        if module_type is None:
            identity = contract.canonical_raw_import_identity()
            raise KeyError(
                "No CellProfiler module declaration owns callable "
                f"{identity.import_path!r}."
            )
        return module_type

    @classmethod
    def require_module(cls, module_name: str) -> type["CellProfilerModule"]:
        """Return the registered declaration for ``module_name`` or fail."""
        module_type = cls.for_module(module_name)
        if module_type is None:
            raise KeyError(
                f"No CellProfiler module declaration is registered for {module_name!r}."
            )
        return module_type

    @classmethod
    def require_module_name(cls) -> str:
        """Return this registered declaration's nominal module name."""

        return _required_string(cls.module_name, "module_name", cls.__name__)

    @classmethod
    def require_callable(
        cls,
        function_name: str | None = None,
    ) -> Callable[..., Any]:
        """Load one raw backend callable declared by this module class."""
        selected_name = cls.function_name if function_name is None else function_name
        selected_name = _required_string(
            selected_name,
            "function_name",
            cls.__name__,
        )
        if selected_name not in cls.declared_function_names():
            raise KeyError(
                f"CellProfiler module {cls.module_name!r} does not declare "
                f"function {selected_name!r}."
            )
        implementation_module = importlib.import_module(cls.__module__)
        implementation = vars(implementation_module).get(selected_name)
        if not callable(implementation):
            raise KeyError(
                f"CellProfiler module {cls.module_name!r} declares missing "
                f"callable {selected_name!r} in {cls.__module__!r}."
            )
        cls._install_callable_parameter_help(implementation)
        from python_introspect import parameter_exclusions

        contract = CallableContract.from_callable(implementation)
        excluded_names = parameter_exclusions(implementation)
        unowned_exclusions = tuple(
            sorted(excluded_names - contract.runtime_owned_parameter_names)
        )
        if unowned_exclusions:
            raise ValueError(
                f"CellProfiler module {cls.module_name!r} callable "
                f"{selected_name!r} excludes parameters without a runtime-owned "
                "callable contract declaration: "
                f"{unowned_exclusions!r}."
            )
        return implementation

    @classmethod
    def _install_callable_parameter_help(
        cls,
        implementation: Callable[..., Any],
    ) -> None:
        """Project missing setting help onto the final public callable."""

        from python_introspect import DocstringExtractor, signature_analysis_target

        signature = inspect.signature(implementation)
        help_target = signature_analysis_target(implementation)
        documented = DocstringExtractor.extract(help_target).parameters or {}
        missing = tuple(
            binding
            for binding in cls.declared_setting_bindings()
            if binding.require_parameter_name() in signature.parameters
            and binding.require_parameter_name() not in documented
        )
        if not missing:
            return

        summary = inspect.getdoc(help_target) or (
            f"Execute the {cls.require_module_name()} CellProfiler operation."
        )
        lines = [summary, "", "Additional Parameters:"]
        lines.extend(
            f"    {binding.require_parameter_name()}: "
            f"{binding.parameter_help_description()}"
            for binding in missing
        )
        help_target.__doc__ = "\n".join(lines)

    @classmethod
    def for_module(cls, module_name: str) -> type["CellProfilerModule"] | None:
        """Return the registered module class for a canonical name or alias."""
        lookup_key = _module_lookup_key(module_name)
        for module_type in cls.__registry__.values():
            if _module_lookup_key(module_type.require_module_name()) == lookup_key:
                return module_type
            if lookup_key in {
                _module_lookup_key(alias) for alias in module_type.aliases
            }:
                return module_type
        return None

    @classmethod
    def invocation_module_blocks(
        cls, module: "ModuleBlock"
    ) -> tuple["ModuleBlock", ...]:
        """Return module-owned public invocation blocks for one parsed module."""
        return (module,)

    @classmethod
    def module_blocks_for_invocation(
        cls,
        *,
        invocation: "NormalizedFunctionItem",
        step_context: "ArtifactDeclarationStepContext",
    ) -> tuple[tuple["ModuleBlock", ...], tuple[str, ...]]:
        """Enumerate exact transient setting blocks for one public invocation.

        An empty block tuple means the scoped artifact context cannot satisfy the
        invocation with the authored identities. Import lowering uses that exact
        result to retain an identity; final compilation rejects it.
        """

        from openhcs.interop.cellprofiler.parser import ModuleBlock

        reconstruction_context = cls._artifact_context_for_group(
            step_context,
            group_key=invocation.key.group_key,
        )

        raw_callable = cls.require_callable(invocation.contract.function_name)
        invocation_callable = invocation.contract.resolve_canonical_raw_callable()
        if invocation_callable is not raw_callable:
            raise ValueError(
                f"CellProfiler invocation {invocation.key!r} does not reference "
                f"the canonical {cls.__name__} callable object."
            )
        signature = inspect.signature(raw_callable)
        explicit_kwargs = invocation.kwargs_dict
        declared_bindings = cls.declared_setting_bindings()
        records_by_position: list[tuple["ModuleSetting", ...]] = [
            () for _binding in declared_bindings
        ]
        consumed_identity_kwargs: list[str] = []

        for position, binding in enumerate(declared_bindings):
            parameter_name = binding.require_parameter_name()
            if binding.declares_artifact:
                if parameter_name not in explicit_kwargs:
                    continue
                records_by_position[position] = binding.records_from_kwargs(
                    explicit_kwargs
                )
                consumed_identity_kwargs.append(parameter_name)
                continue
            parameter = signature.parameters.get(parameter_name)
            if parameter is None:
                if parameter_name not in explicit_kwargs:
                    continue
                records_by_position[position] = binding.records_from_kwargs(
                    explicit_kwargs
                )
                consumed_identity_kwargs.append(parameter_name)
                continue
            if parameter_name in explicit_kwargs:
                value = explicit_kwargs[parameter_name]
            elif parameter.default is not inspect.Parameter.empty:
                value = parameter.default
            else:
                raise ValueError(
                    f"CellProfiler module {cls.module_name!r} requires public "
                    f"kwarg {parameter_name!r} for setting "
                    f"{setting_names(binding.setting_name)[0]!r}."
                )
            records_by_position[position] = binding.records_from_kwargs(
                {parameter_name: value}
            )

        records = [
            record
            for position_records in records_by_position
            for record in position_records
        ]

        blocks: list[ModuleBlock] = []
        provisional_module = ModuleBlock(
            name=cls.require_module_name(),
            module_num=0,
            enabled=True,
            setting_records=list(records),
        )
        input_record_groups = cls._artifact_input_record_groups(
            module=provisional_module,
            invocation_key=invocation.key,
            step_context=reconstruction_context,
        )
        for block_position, input_records in enumerate(input_record_groups):
            block_records = (*records, *input_records)
            block_records = (
                *block_records,
                *cls._derived_identity_setting_records(
                    invocation=invocation,
                    block_position=block_position,
                    existing_records=block_records,
                    step_context=reconstruction_context,
                ),
            )
            blocks.append(
                ModuleBlock(
                    name=cls.require_module_name(),
                    module_num=0,
                    enabled=True,
                    setting_records=list(block_records),
                )
            )
        consumed_identity_names = frozenset(consumed_identity_kwargs)
        finalized_blocks = cls.finalize_module_blocks_for_invocation(
            tuple(blocks),
            invocation=invocation,
            step_context=step_context,
        )
        return finalized_blocks, tuple(
            name for name in explicit_kwargs if name in consumed_identity_names
        )

    @classmethod
    def combine_callable_contracts(
        cls,
        contracts: Iterable[CallableContract],
    ) -> CallableContract:
        """Combine dynamic declarations for one canonical callable by occurrence."""

        contract_values = tuple(contracts)
        if not contract_values:
            raise ValueError(
                f"{cls.__name__}.combine_callable_contracts requires contracts."
            )
        first = contract_values[0]
        canonical_callable = cls.require_callable(first.function_name)
        expected_metadata = replace(
            first.metadata,
            artifact_inputs=(),
            artifact_outputs=(),
        )
        for contract in contract_values:
            if contract.resolve_canonical_raw_callable() is not canonical_callable:
                raise ValueError(
                    f"{cls.__name__} can only combine contracts for its canonical "
                    f"callable {canonical_callable.__name__!r}."
                )
            if (
                replace(
                    contract.metadata,
                    artifact_inputs=(),
                    artifact_outputs=(),
                )
                != expected_metadata
            ):
                raise ValueError(
                    f"{cls.__name__} dynamic callable contracts disagree outside "
                    "artifact declarations."
                )

        def combined_occurrences(
            collections: Iterable[tuple[ArtifactSpec, ...]],
        ) -> tuple[ArtifactSpec, ...]:
            combined: list[ArtifactSpec] = []
            for collection in collections:
                occurrence_counts: dict[ArtifactSpecRef, int] = {}
                for spec in collection:
                    ref = spec.ref()
                    occurrence_index = occurrence_counts.get(ref, 0)
                    occurrence_counts[ref] = occurrence_index + 1
                    matching_indices = tuple(
                        index
                        for index, declared in enumerate(combined)
                        if declared.ref() == ref
                    )
                    if occurrence_index == len(matching_indices):
                        combined.append(spec)
                        continue
                    declared = combined[matching_indices[occurrence_index]]
                    if (
                        declared != spec
                        or declared.parameter_name != spec.parameter_name
                    ):
                        raise ValueError(
                            f"{cls.__name__} has conflicting dynamic artifact "
                            f"declarations for occurrence {occurrence_index + 1} of "
                            f"{ref!r}: {declared!r} and {spec!r}."
                        )
            return tuple(combined)

        return replace(
            first,
            metadata=replace(
                first.metadata,
                artifact_inputs=combined_occurrences(
                    contract.metadata.artifact_inputs for contract in contract_values
                ),
                artifact_outputs=combined_occurrences(
                    contract.metadata.artifact_outputs for contract in contract_values
                ),
            ),
        )

    @classmethod
    def invocation_callable_contract(
        cls,
        *,
        invocation: "NormalizedFunctionItem",
        numbered_module_blocks: tuple["ModuleBlock", ...],
        consumed_kwarg_names: tuple[str, ...],
        step_context: "ArtifactDeclarationStepContext",
    ) -> tuple[CallableContract, tuple[str, ...]]:
        """Resolve one invocation's exact canonically numbered blocks."""

        contracts: list[CallableContract] = []
        blocks = tuple(numbered_module_blocks)
        if not blocks or any(block.module_num < 1 for block in blocks):
            raise ValueError(
                "CellProfiler invocation contract construction requires a "
                "non-empty tuple of canonically numbered ModuleBlock values."
            )
        consumed_names = frozenset(consumed_kwarg_names)
        if len(consumed_names) != len(consumed_kwarg_names):
            raise ValueError(
                "CellProfiler invocation consumed kwarg names cannot contain "
                "duplicates."
            )
        raw_func = cls.require_callable(invocation.key.function_name)
        for block in blocks:
            contract = cls.callable_contract(
                module=block,
                invocation_key=invocation.key,
                step_context=step_context,
            )
            cls.validate_callable_artifact_abi(raw_func, contract)
            contracts.append(contract)
        if not contracts:
            raise ValueError(
                f"CellProfiler contract compilation produced no contracts for step "
                f"{step_context.step_index!r} ({step_context.step_name!r}), "
                f"invocation {invocation.key!r}, module {cls.__name__}."
            )
        combined = cls.combine_callable_contracts(contracts)
        resolved = replace(
            invocation.contract,
            metadata=replace(
                invocation.contract.metadata,
                artifact_inputs=combined.metadata.artifact_inputs,
                artifact_outputs=combined.metadata.artifact_outputs,
            ),
        )
        return resolved, tuple(
            name for name in invocation.kwargs_dict if name in consumed_names
        )

    @classmethod
    def resolve_function(
        cls,
        module: "ModuleBlock",
        *,
        contract: CallableContract,
        source_bindings: "StepSourceBindingsConfig",
    ) -> Callable[..., Any]:
        """Return the canonical raw callable selected for one parsed module."""
        del module, contract, source_bindings
        return cls.require_callable()

    @classmethod
    def processing_config(
        cls,
        *,
        callable_contract: CallableContract,
        inherited: ProcessingConfig,
        step_context: ArtifactDeclarationStepContext | None = None,
    ) -> ProcessingConfig:
        """Translate one module invocation into ordinary processing semantics."""
        if not isinstance(callable_contract, CallableContract):
            raise TypeError(
                "CellProfilerModule.processing_config requires CallableContract, got "
                f"{type(callable_contract).__name__}."
            )
        if not isinstance(inherited, ProcessingConfig):
            raise TypeError(
                "CellProfilerModule.processing_config requires ProcessingConfig, got "
                f"{type(inherited).__name__}."
            )
        if callable_contract.execution_scope is FunctionStepExecutionScope.PLATE:
            variable_components: tuple[VariableComponents, ...] = ()
            group_by = GroupBy.NONE
            input_source = InputSource.PREVIOUS_STEP
        else:
            variable_components = tuple(
                callable_contract.required_variable_components
                or inherited.variable_components
            )
            group_by = cls.group_by if cls.group_by is not None else inherited.group_by
            input_source = inherited.input_source
            if (
                group_by is not None
                and group_by.value is not None
                and step_context is not None
            ):
                grouping_component = AllComponents.from_value(group_by.value)
                source_anchor_group_keys = step_context.source_bindings.component_group_keys_for_artifact_specs(
                    grouping_component,
                    tuple(
                        spec
                        for spec in callable_contract.artifact_inputs
                        if spec.parameter_name is None
                    ),
                    step_context.available_artifacts,
                )
                if len(source_anchor_group_keys) > 1:
                    group_by = GroupBy.NONE
        return replace(
            inherited,
            variable_components=list(variable_components),
            group_by=group_by,
            input_source=input_source,
        )
