"""CellProfiler name-to-artifact symbol table compiler.

The converter needs one place where CellProfiler's string workspace names become
typed OpenHCS artifact contracts.  This module owns that conversion boundary:
same-type declarations update the current workspace binding, type conflicts fail
loudly, and image names with no producer are treated as external source images
supplied by the plate/input metadata.
"""

from __future__ import annotations
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Iterable, Mapping
from openhcs.core.artifact_materialization_policy import NO_ARTIFACT_MATERIALIZATION
from openhcs.core.artifacts import (
    CROP_MASK_ARTIFACT_SIDECAR,
    ArtifactSpec,
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSidecarRole,
    ArtifactType,
    ImageArtifactType,
    ObjectLabelsArtifactType,
    MeasurementsArtifactType,
)
from openhcs.core.module_artifact_contract import (
    DeclaredArtifactOutputPartition,
    ModuleArtifactContract,
    ModuleArtifactContractItem,
    ModuleArtifactContractPartition,
    RecordedArtifactOutputPartition,
    RuntimeArtifactInputPartition,
    SourceArtifactInputPartition,
)
from openhcs.core.pipeline_image_schema import PipelineImageSchema
from openhcs.core.source_bindings import (
    ComponentSelector,
    EMPTY_SOURCE_BINDINGS,
    MetadataExtractionRule,
    MetadataSource,
    MetadataSelector,
    NamedSourceBinding,
    SourceBindingMatchDimension,
    SourceBindingMatchField,
    SourceBindingMatchMethod,
    SourceBindingMatchPlan,
    SourceBindingOrigin,
    SourceFilterClause,
    SourceFilterMatchType,
    SourceFilterSubject,
    SourceSelector,
    SourceBindingsConfig,
    StepSourceBindingsConfig,
)
from openhcs.interop.cellprofiler.parser import ModuleBlock
from openhcs.processing.backends.cellprofiler.library import canonical_module_name
from pycodify import FormatContext, to_source
import openhcs.serialization.pycodify_formatters as _openhcs_pycodify_formatters
from openhcs.interop.cellprofiler.module_roles import cellprofiler_module_role
from openhcs.interop.cellprofiler.module_artifact_inputs import (
    module_declared_artifact_inputs,
)
from openhcs.interop.cellprofiler.source_schema import compile_image_schema


@dataclass(frozen=True, slots=True)
class CellProfilerSymbolKey:
    """Typed CellProfiler workspace identity."""

    name: str
    artifact_type: type[ArtifactType]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _normalize_symbol_name(self.name))
        object.__setattr__(
            self, "artifact_type", ArtifactType.coerce(self.artifact_type)
        )
        if not self.name:
            raise ValueError("CellProfilerSymbolKey.name cannot be empty.")


@dataclass(frozen=True, slots=True)
class CellProfilerSymbol:
    """One named CellProfiler workspace value known at conversion time."""

    artifact_spec: ArtifactSpec
    producer_module_num: int | None = None
    source_bound: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_spec, ArtifactSpec):
            raise TypeError(
                f"CellProfilerSymbol.artifact_spec must be an ArtifactSpec, got {type(self.artifact_spec).__name__}."
            )
        normalized_name = _normalize_symbol_name(self.artifact_spec.name)
        if not normalized_name:
            raise ValueError("CellProfilerSymbol.name cannot be empty.")
        if normalized_name != self.artifact_spec.name:
            object.__setattr__(
                self, "artifact_spec", replace(self.artifact_spec, name=normalized_name)
            )

    @property
    def name(self) -> str:
        return self.artifact_spec.name

    @property
    def artifact_type(self) -> type[ArtifactType]:
        return self.artifact_spec.artifact_type

    @property
    def key(self) -> CellProfilerSymbolKey:
        return CellProfilerSymbolKey(self.name, self.artifact_type)

    @property
    def is_external_source(self) -> bool:
        """Whether this symbol is supplied by source bindings rather than a module."""
        return self.source_bound and self.producer_module_num is None

    @staticmethod
    def unique_by_key(
        symbols: Iterable["CellProfilerSymbol"],
    ) -> tuple["CellProfilerSymbol", ...]:
        """Return symbols deduplicated by typed workspace identity."""
        unique: list[CellProfilerSymbol] = []
        seen: set[CellProfilerSymbolKey] = set()
        for symbol in symbols:
            if symbol.key not in seen:
                unique.append(symbol)
                seen.add(symbol.key)
        return tuple(unique)


@dataclass(frozen=True, slots=True)
class ModuleArtifactContracts:
    """Artifact inputs/outputs compiled for one CellProfiler module."""

    module_name: str
    module_num: int
    input_symbols: tuple[CellProfilerSymbol, ...] = ()
    output_symbols: tuple[CellProfilerSymbol, ...] = ()
    declared_output_symbols: tuple[CellProfilerSymbol, ...] = ()
    source_bindings: StepSourceBindingsConfig = EMPTY_SOURCE_BINDINGS

    def __post_init__(self) -> None:
        object.__setattr__(self, "module_name", canonical_module_name(self.module_name))
        if not isinstance(self.source_bindings, StepSourceBindingsConfig):
            raise TypeError(
                f"ModuleArtifactContracts.source_bindings must be StepSourceBindingsConfig, got {type(self.source_bindings).__name__}."
            )

    @property
    def inputs(self) -> tuple[ArtifactSpec, ...]:
        """All named values consumed by the module as artifact specs."""
        return tuple(
            (
                symbol.artifact_spec.for_plan_type(ArtifactInputPlan)
                for symbol in self.input_symbols
            )
        )

    @property
    def outputs(self) -> tuple[ArtifactSpec, ...]:
        """All named values produced by the module as artifact specs."""
        return tuple(
            (
                symbol.artifact_spec.for_plan_type(ArtifactOutputPlan)
                for symbol in self.output_symbols
            )
        )

    @property
    def declared_outputs(self) -> tuple[ArtifactSpec, ...]:
        """All CellProfiler outputs before runtime dead-artifact pruning."""
        symbols = self.declared_output_symbols or self.output_symbols
        return tuple(
            (
                symbol.artifact_spec.for_plan_type(ArtifactOutputPlan)
                for symbol in symbols
            )
        )

    @property
    def runtime_artifact_inputs(self) -> tuple[ArtifactSpec, ...]:
        """Inputs that should be routed through OpenHCS artifact storage.

        Source-bound artifacts are intentionally excluded: they are normal inputs
        from the source-binding layer, not side-channel artifact reads. Values
        produced by prior modules remain artifact inputs. Repeated function
        roles can preserve duplicate ``input_symbols`` for positional binding,
        but the runtime artifact store has one value per semantic name/type.
        """
        return tuple(
            (
                symbol.artifact_spec.for_plan_type(ArtifactInputPlan)
                for symbol in CellProfilerSymbol.unique_by_key(self.input_symbols)
                if not symbol.is_external_source
            )
        )

    @property
    def external_source_symbols(self) -> tuple[CellProfilerSymbol, ...]:
        """Source-bound names this module expects from input metadata/channels."""
        return tuple(
            (symbol for symbol in self.input_symbols if symbol.is_external_source)
        )

    @property
    def module_contract(self) -> ModuleArtifactContract:
        return ModuleArtifactContract(
            module_name=self.module_name,
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition, self.inputs
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition, self.runtime_artifact_inputs
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition, self.outputs
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition, self.declared_outputs
                ),
            ),
        )


@dataclass(frozen=True)
class CellProfilerSymbolTable:
    """Compiled CellProfiler symbol table and per-module artifact contracts."""

    symbols: Mapping[CellProfilerSymbolKey, CellProfilerSymbol]
    module_contracts: tuple[ModuleArtifactContracts, ...] = ()
    source_schema: PipelineImageSchema = PipelineImageSchema.empty()

    @property
    def contracts_by_module_num(self) -> dict[int, ModuleArtifactContracts]:
        return {contract.module_num: contract for contract in self.module_contracts}

    def contract_for(self, module: ModuleBlock) -> ModuleArtifactContracts:
        """Return compiled contracts for a parsed module."""
        try:
            return self.contracts_by_module_num[module.module_num]
        except KeyError as exc:
            raise KeyError(
                f"No CellProfiler artifact contract compiled for {module.name}({module.module_num})."
            ) from exc

    def symbol_for(
        self, name: str, artifact_type: type[ArtifactType]
    ) -> CellProfilerSymbol:
        """Return the symbol for one typed CellProfiler workspace identity."""
        key = CellProfilerSymbolKey(name, artifact_type)
        try:
            return self.symbols[key]
        except KeyError as exc:
            raise KeyError(
                f"No CellProfiler {key.artifact_type.value} symbol named {key.name!r}."
            ) from exc

    @classmethod
    def compile(
        cls,
        modules: Iterable[ModuleBlock],
        source_schema: PipelineImageSchema | None = None,
    ) -> "CellProfilerSymbolTable":
        ordered_modules = tuple(modules)
        builder = _SymbolTableBuilder(
            source_schema if source_schema is not None else compile_image_schema(ordered_modules)
        )
        for module in ordered_modules:
            if module.enabled:
                builder.visit(module)
        return builder.build()


class _SymbolTableBuilder:

    def __init__(self, source_schema: PipelineImageSchema) -> None:
        self._symbols: dict[CellProfilerSymbolKey, CellProfilerSymbol] = {}
        self._contracts: list[ModuleArtifactContracts] = []
        self._source_schema = source_schema

    def visit(self, module: ModuleBlock) -> None:
        self._contracts.append(_module_artifact_contract(self, module))

    def build(self) -> CellProfilerSymbolTable:
        return CellProfilerSymbolTable(
            symbols=MappingProxyType(dict(self._symbols)),
            module_contracts=tuple(self._contracts),
            source_schema=self._source_schema,
        )

    @property
    def source_schema(self) -> PipelineImageSchema:
        return self._source_schema

    def source_bindings_for(
        self,
        symbols: Iterable[CellProfilerSymbol],
        *,
        module: ModuleBlock | None = None,
        input_symbols: tuple[CellProfilerSymbol, ...] = (),
    ) -> StepSourceBindingsConfig:
        external_symbols = CellProfilerSymbol.unique_by_key(symbols)
        if not external_symbols:
            return EMPTY_SOURCE_BINDINGS
        module_type = None
        if module is not None:
            from openhcs.interop.cellprofiler.module_declarations import (
                CellProfilerModule,
            )

            module_type = CellProfilerModule.for_module(module.name)
        bindings = tuple(
            (
                self._source_binding_for_symbol(
                    symbol,
                    participates_in_image_stack=(
                        True
                        if module is None or module_type is None
                        else module_type.source_binding_participates_in_image_stack(
                            module, symbol, input_symbols
                        )
                    ),
                )
                for symbol in external_symbols
            )
        )
        return StepSourceBindingsConfig(bindings=bindings, enabled=True)

    def external_image(self, name: str) -> CellProfilerSymbol:
        return self._declare(
            ArtifactSpec.input(name, ImageArtifactType), None, source_bound=True
        )

    def external_source_artifact(
        self, name: str, artifact_type: type[ArtifactType]
    ) -> CellProfilerSymbol:
        return self._declare(
            ArtifactSpec.input(name, artifact_type), None, source_bound=True
        )

    def require(
        self, name: str, artifact_type: type[ArtifactType], module: ModuleBlock
    ) -> CellProfilerSymbol:
        normalized_name = _normalize_symbol_name(name)
        resolved_artifact_type = ArtifactType.coerce(artifact_type)
        symbol = self._symbols.get(
            CellProfilerSymbolKey(normalized_name, resolved_artifact_type)
        )
        if symbol is None:
            try:
                source_artifact = (
                    self._source_schema.resolved_source_artifact_for_alias(
                        normalized_name, resolved_artifact_type
                    )
                )
            except ValueError as exc:
                raise ValueError(
                    f"Module {module.name}({module.module_num}) expects '{normalized_name}' as {resolved_artifact_type.value}, but setup declares a different source artifact type."
                ) from exc
            if source_artifact is not None:
                return self.external_source_artifact(
                    normalized_name, resolved_artifact_type
                )
            if resolved_artifact_type is ObjectLabelsArtifactType and (
                not self._source_schema.is_empty
            ):
                return self.external_source_artifact(
                    normalized_name, resolved_artifact_type
                )
            if resolved_artifact_type is ImageArtifactType:
                self._raise_if_name_is_known_as_other_kind(
                    normalized_name, resolved_artifact_type, module
                )
                return self.external_image(normalized_name)
            raise ValueError(
                f"Module {module.name}({module.module_num}) references unknown {resolved_artifact_type.value} symbol '{normalized_name}'. No prior module produces it."
            )
        return symbol

    def require_artifact(
        self, spec: ArtifactSpec, module: ModuleBlock
    ) -> CellProfilerSymbol:
        """Require a typed artifact through the generic compiler boundary."""
        return self.require(spec.name, spec.artifact_type, module)

    def optional(
        self, name: str, artifact_type: type[ArtifactType]
    ) -> CellProfilerSymbol | None:
        """Return a previously declared symbol if it exists."""
        normalized_name = _normalize_symbol_name(name)
        return self._symbols.get(CellProfilerSymbolKey(normalized_name, artifact_type))

    def optional_artifact(self, spec: ArtifactSpec) -> CellProfilerSymbol | None:
        """Return a previously declared typed artifact if it exists."""
        return self.optional(spec.name, spec.artifact_type)

    def measurement_output_for_module_num(
        self, module_num: int | None
    ) -> CellProfilerSymbol | None:
        """Return the unique measurement output produced by a prior module."""
        if module_num is None:
            return None
        measurement_outputs = tuple(
            (
                symbol
                for contract in self._contracts
                if contract.module_num == module_num
                for symbol in contract.output_symbols
                if symbol.artifact_type is MeasurementsArtifactType
            )
        )
        if len(measurement_outputs) > 1:
            raise ValueError(
                f"Module {module_num} produced multiple measurement outputs: {[symbol.name for symbol in measurement_outputs]!r}."
            )
        return measurement_outputs[0] if measurement_outputs else None

    def measurement_outputs(self) -> tuple[CellProfilerSymbol, ...]:
        """Return measurement outputs produced by previously visited modules."""
        return CellProfilerSymbol.unique_by_key(
            (
                symbol
                for contract in self._contracts
                for symbol in contract.output_symbols
                if symbol.artifact_type is MeasurementsArtifactType
            )
        )

    def declare(
        self,
        name: str,
        artifact_type: type[ArtifactType],
        module: ModuleBlock,
        *,
        sidecar_role: ArtifactSidecarRole | None = None,
    ) -> CellProfilerSymbol:
        return self._declare(
            ArtifactSpec.output(name, artifact_type, sidecar_role=sidecar_role),
            module.module_num,
        )

    def declare_artifact(
        self, spec: ArtifactSpec, module: ModuleBlock
    ) -> CellProfilerSymbol:
        """Declare a typed artifact through the generic compiler boundary."""
        return self._declare(spec, module.module_num)

    def _declare(
        self,
        spec: ArtifactSpec,
        producer_module_num: int | None,
        *,
        source_bound: bool = False,
    ) -> CellProfilerSymbol:
        symbol = CellProfilerSymbol(
            artifact_spec=spec,
            producer_module_num=producer_module_num,
            source_bound=source_bound,
        )
        existing = self._symbols.get(symbol.key)
        if existing is not None:
            if existing == symbol:
                return existing
        self._symbols[symbol.key] = symbol
        return symbol

    def _raise_if_name_is_known_as_other_kind(
        self, name: str, expected_artifact_type: type[ArtifactType], module: ModuleBlock
    ) -> None:
        conflicting_artifact_types = tuple(
            (
                key.artifact_type
                for key in self._symbols
                if key.name == name and key.artifact_type is not expected_artifact_type
            )
        )
        if not conflicting_artifact_types:
            return
        existing = conflicting_artifact_types[0]
        raise ValueError(
            f"Module {module.name}({module.module_num}) expects '{name}' as {expected_artifact_type.value}, but it is already registered as {existing.value} and no source schema declares a {expected_artifact_type.value} binding for that name."
        )

    def _source_binding_for_symbol(
        self, symbol: CellProfilerSymbol, *, participates_in_image_stack: bool = True
    ) -> NamedSourceBinding:
        assignment = self._source_schema.source_assignment_for_alias(
            symbol.name, symbol.artifact_type
        )
        if assignment is None:
            return NamedSourceBinding(
                alias=symbol.name,
                artifact_kind=symbol.artifact_type,
                participates_in_image_stack=participates_in_image_stack,
            )
        binding = assignment.to_binding()
        return replace(
            binding,
            participates_in_image_stack=binding.participates_in_image_stack
            and participates_in_image_stack,
        )


def _module_artifact_contract(
    builder: _SymbolTableBuilder, module: ModuleBlock
) -> ModuleArtifactContracts:
    """Compile one module contract from declaration SSOT, then generic inference."""
    from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule

    module_type = CellProfilerModule.for_module(module.name)
    if module_type is not None:
        contract = module_type.artifact_contract(
            CellProfilerContractAssemblyMixin(), builder, module
        )
        if contract is not None:
            return contract
    if cellprofiler_module_role(module.name).is_infrastructure:
        return ModuleArtifactContracts(module.name, module.module_num)
    raise ValueError(
        f"Module {module.name}({module.module_num}) has no declared or inferable CellProfiler artifact contract. Add artifact_contract() to the module declaration before converting this module."
    )


def module_contract_literal(
    contract: ModuleArtifactContracts,
    *,
    externally_materialized_outputs: frozenset[
        tuple[type[ArtifactType], str]
    ] = frozenset(),
    artifact_name_materialized_outputs: frozenset[
        tuple[type[ArtifactType], str]
    ] = frozenset(),
    import_collector: set[tuple[str, str]] | None = None,
) -> str:
    """Render a deterministic Python literal for generated pipeline files."""
    artifact_literals = ArtifactSpecLiteralAuthority()
    item_literals = [
        artifact_literals.item_literal(SourceArtifactInputPartition, spec)
        for spec in contract.inputs
    ]
    item_literals.extend(
        (
            artifact_literals.item_literal(RuntimeArtifactInputPartition, spec)
            for spec in contract.runtime_artifact_inputs
        )
    )
    item_literals.extend(
        (
            artifact_literals.item_literal(
                RecordedArtifactOutputPartition,
                spec,
                preserve_default_materialization=(spec.artifact_type, spec.name)
                not in externally_materialized_outputs,
                materialization_literal=(
                    f"tiff_stack(normalize_uint8=True, filename_identity=MaterializedFilenameIdentity.{('ARTIFACT_NAME' if (spec.artifact_type, spec.name) in artifact_name_materialized_outputs else 'SOURCE_IDENTITY')})"
                    if (spec.artifact_type, spec.name)
                    in externally_materialized_outputs
                    else None
                ),
            )
            for spec in contract.outputs
        )
    )
    item_literals.extend(
        (
            artifact_literals.item_literal(
                DeclaredArtifactOutputPartition,
                spec,
                preserve_default_materialization=(spec.artifact_type, spec.name)
                not in externally_materialized_outputs,
                materialization_literal=(
                    f"tiff_stack(normalize_uint8=True, filename_identity=MaterializedFilenameIdentity.{('ARTIFACT_NAME' if (spec.artifact_type, spec.name) in artifact_name_materialized_outputs else 'SOURCE_IDENTITY')})"
                    if (spec.artifact_type, spec.name)
                    in externally_materialized_outputs
                    else None
                ),
            )
            for spec in contract.declared_outputs
        )
    )
    items_literal = ", ".join(item_literals)
    if len(item_literals) == 1:
        items_literal += ","
    from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule

    required_variable_components = tuple(
        (
            component
            for module_type in (CellProfilerModule.for_module(contract.module_name),)
            if module_type is not None
            for component in module_type.required_variable_components
        )
    )
    required_components_literal = ""
    if required_variable_components:
        required_components_literal = (
            ", required_variable_components=("
            + ", ".join(
                (
                    f"VariableComponents.{component.name}"
                    for component in required_variable_components
                )
            )
            + ("," if len(required_variable_components) == 1 else "")
            + ")"
        )
    if import_collector is not None:
        import_collector.update(artifact_literals.imports)
        import_collector.update(artifact_literals.partition_imports)
        if required_variable_components:
            import_collector.add(("openhcs.constants.constants", "VariableComponents"))
    return f"ModuleArtifactContract(module_name={contract.module_name!r}, items=({items_literal}){required_components_literal})"


class ArtifactSpecLiteralAuthority:
    """Nominal authority for generated ArtifactSpec Python literals."""

    def __init__(self) -> None:
        self.imports: set[tuple[str, str]] = set()
        self.partition_imports: set[tuple[str, str]] = set()
        self.artifact_types = ArtifactTypeLiteralAuthority(self.imports)

    def item_literal(
        self,
        partition_type: type[ModuleArtifactContractPartition],
        spec: ArtifactSpec,
        *,
        preserve_default_materialization: bool = False,
        materialization_literal: str | None = None,
    ) -> str:
        """Render one partitioned ModuleArtifactContract item."""
        resolved_partition = ModuleArtifactContractPartition.require_registered(
            partition_type
        )
        self.partition_imports.update(
            {
                ("openhcs.core.module_artifact_contract", "ModuleArtifactContractItem"),
                (resolved_partition.__module__, resolved_partition.__name__),
            }
        )
        return f"ModuleArtifactContractItem({resolved_partition.__name__}, {self.literal_for(spec, preserve_default_materialization=preserve_default_materialization, materialization_literal=materialization_literal)})"

    def literal_for(
        self,
        spec: ArtifactSpec,
        *,
        preserve_default_materialization: bool = False,
        materialization_literal: str | None = None,
    ) -> str:
        keyword_args: list[str] = []
        if materialization_literal is not None:
            keyword_args.append(f"materialization={materialization_literal}")
        elif spec.materialization is NO_ARTIFACT_MATERIALIZATION:
            keyword_args.append("materialization=NO_ARTIFACT_MATERIALIZATION")
        elif spec.materialization is not None:
            materialization_source = to_source(spec.materialization, FormatContext())
            self.imports.update(materialization_source.imports)
            keyword_args.append(f"materialization={materialization_source.code}")
        elif preserve_default_materialization and (
            not spec.artifact_type.has_default_materialization()
        ):
            keyword_args.append("materialization=NO_ARTIFACT_MATERIALIZATION")
        if spec.sidecar_role is not None:
            keyword_args.append(
                f"sidecar_role=ArtifactSidecarRole.{spec.sidecar_role.name}"
            )
        if spec.relations:
            relation_sources = [
                to_source(relation, FormatContext()) for relation in spec.relations
            ]
            for relation_source in relation_sources:
                self.imports.update(relation_source.imports)
            relation_literals = ", ".join(
                (relation_source.code for relation_source in relation_sources)
            )
            if len(spec.relations) == 1:
                relation_literals += ","
            keyword_args.append(f"relations=({relation_literals})")
        args = [
            repr(spec.name),
            self.artifact_types.literal_for(spec.artifact_type),
            *keyword_args,
        ]
        return f"ArtifactSpec.{spec.plan_type.plan_role}({', '.join(args)})"


class ArtifactTypeLiteralAuthority:
    """Render registered ArtifactType classes and collect their imports."""

    def __init__(self, imports: set[tuple[str, str]]) -> None:
        self.imports = imports

    def literal_for(self, artifact_type: type[ArtifactType]) -> str:
        resolved_type = ArtifactType.coerce(artifact_type)
        self.imports.add((resolved_type.__module__, resolved_type.__name__))
        return resolved_type.__name__


def step_source_bindings_literal(
    config: StepSourceBindingsConfig,
    *,
    import_collector: set[tuple[str, str]] | None = None,
) -> str:
    """Render a deterministic Python literal for generated step source bindings."""
    if config.is_empty:
        return "EMPTY_SOURCE_BINDINGS"
    imports = import_collector if import_collector is not None else set()
    field_literals = _source_bindings_field_literals(
        config, artifact_types=ArtifactTypeLiteralAuthority(imports)
    )
    if config.enabled is True:
        field_literals.append("enabled=True")
    return f"LazyStepSourceBindingsConfig({', '.join(field_literals)})"


def source_bindings_config_literal(
    config: SourceBindingsConfig,
    *,
    import_collector: set[tuple[str, str]] | None = None,
) -> str:
    """Render a deterministic Python literal for generated pipeline source bindings."""
    imports = import_collector if import_collector is not None else set()
    return (
        "LazySourceBindingsConfig("
        + ", ".join(
            _source_bindings_field_literals(
                config, artifact_types=ArtifactTypeLiteralAuthority(imports)
            )
        )
        + ")"
    )


def _source_bindings_field_literals(
    config: SourceBindingsConfig, *, artifact_types: ArtifactTypeLiteralAuthority
) -> list[str]:
    """Render source-binding payload fields for one config declaration."""
    field_literals: list[str] = []
    if config.source_filters:
        filter_literals_authority = SourceFilterClauseLiteralAuthority()
        filter_literals = ", ".join(
            (
                filter_literals_authority.literal_for(clause)
                for clause in config.source_filters
            )
        )
        if len(config.source_filters) == 1:
            filter_literals += ","
        field_literals.append(f"source_filters=({filter_literals})")
    if config.bindings:
        binding_literals = ", ".join(
            (
                _named_source_binding_literal(binding, artifact_types=artifact_types)
                for binding in config.bindings
            )
        )
        if len(config.bindings) == 1:
            binding_literals += ","
        field_literals.append(f"bindings=({binding_literals})")
    if config.metadata_rules:
        metadata_rule_literals = ", ".join(
            (_metadata_extraction_rule_literal(rule) for rule in config.metadata_rules)
        )
        if len(config.metadata_rules) == 1:
            metadata_rule_literals += ","
        field_literals.append(f"metadata_rules=({metadata_rule_literals})")
    if config.match_plan is not None:
        field_literals.append(
            f"match_plan={_source_binding_match_plan_literal(config.match_plan)}"
        )
    return field_literals


def _named_source_binding_literal(
    binding: NamedSourceBinding, *, artifact_types: ArtifactTypeLiteralAuthority
) -> str:
    field_literals = [f"alias={binding.alias!r}"]
    if binding.artifact_kind is not ImageArtifactType:
        field_literals.append(
            f"artifact_kind={artifact_types.literal_for(binding.artifact_kind)}"
        )
    if binding.selector != SourceSelector():
        field_literals.append(f"selector={_source_selector_literal(binding.selector)}")
    if binding.origin is not SourceBindingOrigin.STEP_INPUT:
        field_literals.append(f"origin=SourceBindingOrigin.{binding.origin.name}")
    if binding.component_identity:
        component_literals = ", ".join(
            _component_selector_literal(component)
            for component in binding.component_identity
        )
        if len(binding.component_identity) == 1:
            component_literals += ","
        field_literals.append(f"component_identity=({component_literals})")
    if not binding.participates_in_image_stack:
        field_literals.append("participates_in_image_stack=False")
    return f"NamedSourceBinding({', '.join(field_literals)})"


def _source_selector_literal(selector: SourceSelector) -> str:
    field_literals: list[str] = []
    if selector.components:
        component_literals = ", ".join(
            (
                _component_selector_literal(component)
                for component in selector.components
            )
        )
        if len(selector.components) == 1:
            component_literals += ","
        field_literals.append(f"components=({component_literals})")
    if selector.metadata:
        metadata_literals = ", ".join(
            (_metadata_selector_literal(metadata) for metadata in selector.metadata)
        )
        if len(selector.metadata) == 1:
            metadata_literals += ","
        field_literals.append(f"metadata=({metadata_literals})")
    if selector.filters:
        filter_literals_authority = SourceFilterClauseLiteralAuthority()
        filter_literals = ", ".join(
            (
                filter_literals_authority.literal_for(clause)
                for clause in selector.filters
            )
        )
        if len(selector.filters) == 1:
            filter_literals += ","
        field_literals.append(f"filters=({filter_literals})")
    if not selector.inherit_current_scope:
        field_literals.append("inherit_current_scope=False")
    return f"SourceSelector({', '.join(field_literals)})"


def _component_selector_literal(selector: ComponentSelector) -> str:
    return f"ComponentSelector(AllComponents.{selector.component.name}, {selector.value!r})"


def _metadata_selector_literal(selector: MetadataSelector) -> str:
    return f"MetadataSelector({selector.field!r}, {selector.value!r})"


def _metadata_extraction_rule_literal(rule: MetadataExtractionRule) -> str:
    field_literals = [
        f"source=MetadataSource.{rule.source.name}",
        f"pattern={rule.pattern!r}",
    ]
    if rule.filters:
        filter_literals_authority = SourceFilterClauseLiteralAuthority()
        filter_literals = ", ".join(
            (filter_literals_authority.literal_for(clause) for clause in rule.filters)
        )
        if len(rule.filters) == 1:
            filter_literals += ","
        field_literals.append(f"filters=({filter_literals})")
    return f"MetadataExtractionRule({', '.join(field_literals)})"


class SourceFilterClauseLiteralAuthority:
    """Nominal authority for generated SourceFilterClause Python literals."""

    def literal_for(self, clause: SourceFilterClause) -> str:
        field_literals = [
            f"subject=SourceFilterSubject.{clause.subject.name}",
            f"match_type=SourceFilterMatchType.{clause.match_type.name}",
        ]
        if clause.value is not None:
            field_literals.append(f"value={clause.value!r}")
        return f"SourceFilterClause({', '.join(field_literals)})"


def _source_binding_match_plan_literal(plan: SourceBindingMatchPlan) -> str:
    field_literals = [f"method=SourceBindingMatchMethod.{plan.method.name}"]
    if plan.dimensions:
        dimension_literals = ", ".join(
            (
                _source_binding_match_dimension_literal(dimension)
                for dimension in plan.dimensions
            )
        )
        if len(plan.dimensions) == 1:
            dimension_literals += ","
        field_literals.append(f"dimensions=({dimension_literals})")
    return f"SourceBindingMatchPlan({', '.join(field_literals)})"


def _source_binding_match_dimension_literal(
    dimension: SourceBindingMatchDimension,
) -> str:
    field_literals = ", ".join(
        (_source_binding_match_field_literal(field) for field in dimension.fields)
    )
    if len(dimension.fields) == 1:
        field_literals += ","
    return f"SourceBindingMatchDimension(fields=({field_literals}))"


def _source_binding_match_field_literal(field: SourceBindingMatchField) -> str:
    return f"SourceBindingMatchField(alias={field.alias!r}, metadata_field={field.metadata_field!r})"


class CellProfilerContractAssemblyMixin:
    """Shared artifact-contract assembly semantics for CellProfiler builders."""

    def assemble_contract(
        self,
        module: ModuleBlock,
        builder: _SymbolTableBuilder,
        *,
        inputs: Iterable[CellProfilerSymbol] = (),
        outputs: Iterable[CellProfilerSymbol] = (),
        preserve_duplicate_inputs: bool = False,
    ) -> ModuleArtifactContracts:
        from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule

        module_type = CellProfilerModule.for_module(module.name)
        preserve_role_inputs = preserve_duplicate_inputs or (
            module_type is not None
            and module_type.preserve_duplicate_artifact_inputs(module)
        )
        input_symbols = (
            tuple(inputs)
            if preserve_role_inputs
            else CellProfilerSymbol.unique_by_key(inputs)
        )
        output_symbols = CellProfilerSymbol.unique_by_key(outputs)
        return ModuleArtifactContracts(
            module_name=module.name,
            module_num=module.module_num,
            input_symbols=input_symbols,
            output_symbols=output_symbols,
            declared_output_symbols=output_symbols,
            source_bindings=builder.source_bindings_for(
                (symbol for symbol in input_symbols if symbol.is_external_source),
                module=module,
                input_symbols=input_symbols,
            ),
        )


def _any_truthy_setting_value(
    module: ModuleBlock, setting: str | SettingNameFamily
) -> bool:
    return any(
        (
            value.strip().lower() in {"yes", "true", "1", "on"}
            for value in setting_values(module, setting)
        )
    )


def _split_names(value: str) -> tuple[str, ...]:
    return tuple((_normalize_symbol_name(part) for part in split_symbol_names(value)))


def _normalize_symbol_name(name: str) -> str:
    normalized = name.strip()
    if not normalized:
        raise ValueError("CellProfiler symbol names cannot be empty.")
    return normalized
