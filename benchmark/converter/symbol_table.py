"""CellProfiler name-to-artifact symbol table compiler.

The converter needs one place where CellProfiler's string workspace names become
typed OpenHCS artifact contracts.  This module owns that conversion boundary:
names are unique, kind conflicts fail loudly, and image names with no producer
are treated as external source images supplied by the plate/input metadata.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import ClassVar, Iterable, Mapping

from metaclass_registry import AutoRegisterMeta

from benchmark.cellprofiler_compat.module_contract import CellProfilerModuleContract
from openhcs.core.artifact_materialization_policy import (
    DEFAULT_ARTIFACT_MATERIALIZATION_RULES,
)
from openhcs.core.artifacts import ArtifactKind, ArtifactSpec
from openhcs.core.pipeline_image_schema import CellProfilerImageSchema
from openhcs.core.source_bindings import (
    ComponentSelector,
    EMPTY_SOURCE_BINDINGS,
    GroupedSourceBindings,
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
    StepSourceBindingsConfig,
)

from .gray_to_color_settings import GrayToColorInputNameResolver
from .parser import ModuleBlock
from .source_schema import compile_image_schema


class CellProfilerSymbolKind(str, Enum):
    """CellProfiler workspace symbol categories mapped to OpenHCS artifacts."""

    IMAGE = "image"
    OBJECTS = "objects"
    MEASUREMENTS = "measurements"
    RELATIONSHIPS = "relationships"

    @property
    def artifact_kind(self) -> ArtifactKind:
        return {
            CellProfilerSymbolKind.IMAGE: ArtifactKind.IMAGE,
            CellProfilerSymbolKind.OBJECTS: ArtifactKind.OBJECT_LABELS,
            CellProfilerSymbolKind.MEASUREMENTS: ArtifactKind.MEASUREMENTS,
            CellProfilerSymbolKind.RELATIONSHIPS: ArtifactKind.RELATIONSHIPS,
        }[self]


@dataclass(frozen=True, slots=True)
class CellProfilerSymbol:
    """One named CellProfiler workspace value known at conversion time."""

    name: str
    kind: CellProfilerSymbolKind
    producer_module_num: int | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("CellProfilerSymbol.name cannot be empty.")

    def artifact_spec(self) -> ArtifactSpec:
        return ArtifactSpec(self.name, self.kind.artifact_kind)

    @property
    def is_external_source(self) -> bool:
        """Whether this symbol is supplied by input metadata rather than a module."""
        return (
            self.kind is CellProfilerSymbolKind.IMAGE
            and self.producer_module_num is None
        )


@dataclass(frozen=True, slots=True)
class ModuleArtifactContracts:
    """Artifact inputs/outputs compiled for one CellProfiler module."""

    module_name: str
    module_num: int
    input_symbols: tuple[CellProfilerSymbol, ...] = ()
    output_symbols: tuple[CellProfilerSymbol, ...] = ()
    source_bindings: StepSourceBindingsConfig = EMPTY_SOURCE_BINDINGS

    def __post_init__(self) -> None:
        if not isinstance(self.source_bindings, StepSourceBindingsConfig):
            raise TypeError(
                "ModuleArtifactContracts.source_bindings must be "
                f"StepSourceBindingsConfig, got {type(self.source_bindings).__name__}."
            )

    @property
    def inputs(self) -> tuple[ArtifactSpec, ...]:
        """All named values consumed by the module as artifact specs."""
        return tuple(symbol.artifact_spec() for symbol in self.input_symbols)

    @property
    def outputs(self) -> tuple[ArtifactSpec, ...]:
        """All named values produced by the module as artifact specs."""
        return tuple(symbol.artifact_spec() for symbol in self.output_symbols)

    @property
    def runtime_artifact_inputs(self) -> tuple[ArtifactSpec, ...]:
        """Inputs that should be routed through OpenHCS artifact storage.

        External source images are intentionally excluded: they are normal image
        inputs from the plate/channel layer, not side-channel artifact reads.
        Images produced by prior modules remain artifact inputs.
        """
        return tuple(
            symbol.artifact_spec()
            for symbol in self.input_symbols
            if not symbol.is_external_source
        )

    @property
    def external_source_symbols(self) -> tuple[CellProfilerSymbol, ...]:
        """Source image names this module expects from input metadata/channels."""
        return tuple(
            symbol
            for symbol in self.input_symbols
            if symbol.is_external_source
        )

    @property
    def module_contract(self) -> CellProfilerModuleContract:
        return CellProfilerModuleContract(
            module_name=self.module_name,
            inputs=self.inputs,
            runtime_artifact_inputs=self.runtime_artifact_inputs,
            outputs=self.outputs,
        )


@dataclass(frozen=True, slots=True)
class SettingNameFamily:
    """Canonical CellProfiler setting plus accepted schema aliases."""

    canonical: str
    aliases: tuple[str, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        return (self.canonical, *self.aliases)


@dataclass(frozen=True, slots=True)
class CellProfilerSymbolTable:
    """Compiled CellProfiler symbol table and per-module artifact contracts."""

    symbols: Mapping[str, CellProfilerSymbol]
    module_contracts: tuple[ModuleArtifactContracts, ...] = ()
    source_schema: CellProfilerImageSchema = CellProfilerImageSchema.empty()

    @property
    def contracts_by_module_num(self) -> dict[int, ModuleArtifactContracts]:
        return {contract.module_num: contract for contract in self.module_contracts}

    def contract_for(self, module: ModuleBlock) -> ModuleArtifactContracts:
        """Return compiled contracts for a parsed module."""
        try:
            return self.contracts_by_module_num[module.module_num]
        except KeyError as exc:
            raise KeyError(
                f"No CellProfiler artifact contract compiled for "
                f"{module.name}({module.module_num})."
            ) from exc

    @classmethod
    def compile(
        cls,
        modules: Iterable[ModuleBlock],
    ) -> "CellProfilerSymbolTable":
        ordered_modules = tuple(modules)
        builder = _SymbolTableBuilder(compile_image_schema(ordered_modules))
        for module in ordered_modules:
            if module.enabled:
                builder.visit(module)
        return builder.build()


IMAGE_MEASUREMENT_SETTING = SettingNameFamily(
    "Select images to measure",
    aliases=("Select an image to measure",),
)
OBJECT_MEASUREMENT_SETTING = SettingNameFamily(
    "Select object sets to measure",
    aliases=("Select objects to measure",),
)
OUTPUT_IMAGE_SETTING = SettingNameFamily(
    "Name the output image",
    aliases=("Name the output image file",),
)
DISPLAY_OBJECTS_SETTING = SettingNameFamily(
    "Select objects to display",
    aliases=("Select object to display",),
)


class _SymbolTableBuilder:
    def __init__(self, source_schema: CellProfilerImageSchema) -> None:
        self._symbols: dict[str, CellProfilerSymbol] = {}
        self._contracts: list[ModuleArtifactContracts] = []
        self._source_schema = source_schema

    def visit(self, module: ModuleBlock) -> None:
        self._contracts.append(
            ModuleContractBuilder.for_module(module.name).build(self, module)
        )

    def build(self) -> CellProfilerSymbolTable:
        return CellProfilerSymbolTable(
            symbols=MappingProxyType(dict(self._symbols)),
            module_contracts=tuple(self._contracts),
            source_schema=self._source_schema,
        )

    def source_bindings_for(
        self,
        symbols: Iterable[CellProfilerSymbol],
    ) -> StepSourceBindingsConfig:
        external_symbols = tuple(symbols)
        if not external_symbols:
            return EMPTY_SOURCE_BINDINGS
        bindings = tuple(
            self._source_binding_for_symbol(symbol)
            for symbol in external_symbols
        )
        return StepSourceBindingsConfig(
            groups=(GroupedSourceBindings(bindings=bindings),),
            metadata_rules=self._source_schema.metadata_rules,
            match_plan=self._source_schema.match_plan,
        )

    def external_image(self, name: str) -> CellProfilerSymbol:
        return self._declare(name, CellProfilerSymbolKind.IMAGE, None)

    def require(
        self,
        name: str,
        kind: CellProfilerSymbolKind,
        module: ModuleBlock,
    ) -> CellProfilerSymbol:
        normalized_name = _normalize_symbol_name(name)
        symbol = self._symbols.get(normalized_name)
        if symbol is None:
            if kind is CellProfilerSymbolKind.IMAGE:
                return self.external_image(normalized_name)
            raise ValueError(
                f"Module {module.name}({module.module_num}) references unknown "
                f"{kind.value} symbol '{normalized_name}'. No prior module "
                "produces it."
            )
        if symbol.kind is not kind:
            raise ValueError(
                f"Module {module.name}({module.module_num}) expects "
                f"'{normalized_name}' as {kind.value}, but it is already "
                f"registered as {symbol.kind.value}."
            )
        return symbol

    def declare(
        self,
        name: str,
        kind: CellProfilerSymbolKind,
        module: ModuleBlock,
    ) -> CellProfilerSymbol:
        return self._declare(name, kind, module.module_num)

    def _declare(
        self,
        name: str,
        kind: CellProfilerSymbolKind,
        producer_module_num: int | None,
    ) -> CellProfilerSymbol:
        normalized_name = _normalize_symbol_name(name)
        symbol = CellProfilerSymbol(
            name=normalized_name,
            kind=kind,
            producer_module_num=producer_module_num,
        )
        existing = self._symbols.get(normalized_name)
        if existing is not None:
            if existing.kind is not kind:
                raise ValueError(
                    f"CellProfiler symbol '{normalized_name}' is already "
                    f"registered as {existing.kind.value}, cannot also register "
                    f"as {kind.value}."
                )
            if existing != symbol:
                raise ValueError(
                    f"CellProfiler symbol '{normalized_name}' ({kind.value}) "
                    f"already produced by module {existing.producer_module_num}, "
                    f"cannot also produce from module {producer_module_num}."
                )
            return existing
        self._symbols[normalized_name] = symbol
        return symbol

    def _source_binding_for_symbol(
        self,
        symbol: CellProfilerSymbol,
    ) -> NamedSourceBinding:
        assignment = self._source_schema.resolved_assignment_for_alias(symbol.name)
        if assignment is None:
            return NamedSourceBinding(alias=symbol.name)
        return assignment.to_binding()


def module_contract_literal(contract: ModuleArtifactContracts) -> str:
    """Render a deterministic Python literal for generated pipeline files."""
    input_specs = ", ".join(_artifact_spec_literal(spec) for spec in contract.inputs)
    output_specs = ", ".join(
        _artifact_spec_literal(spec, preserve_default_materialization=True)
        for spec in contract.outputs
    )
    runtime_input_specs = ", ".join(
        _artifact_spec_literal(spec)
        for spec in contract.runtime_artifact_inputs
    )
    if len(contract.inputs) == 1:
        input_specs += ","
    if len(contract.outputs) == 1:
        output_specs += ","
    if len(contract.runtime_artifact_inputs) == 1:
        runtime_input_specs += ","
    return (
        "CellProfilerModuleContract("
        f"module_name={contract.module_name!r}, "
        f"inputs=({input_specs}), "
        f"runtime_artifact_inputs=({runtime_input_specs}), "
        f"outputs=({output_specs})"
        ")"
    )


def source_bindings_literal(config: StepSourceBindingsConfig) -> str:
    """Render a deterministic Python literal for generated step source bindings."""
    if config.is_empty:
        return "EMPTY_SOURCE_BINDINGS"
    field_literals: list[str] = []
    if config.groups:
        group_literals = ", ".join(
            _grouped_source_bindings_literal(group)
            for group in config.groups
        )
        if len(config.groups) == 1:
            group_literals += ","
        field_literals.append(f"groups=({group_literals})")
    if config.metadata_rules:
        metadata_rule_literals = ", ".join(
            _metadata_extraction_rule_literal(rule)
            for rule in config.metadata_rules
        )
        if len(config.metadata_rules) == 1:
            metadata_rule_literals += ","
        field_literals.append(f"metadata_rules=({metadata_rule_literals})")
    if config.match_plan is not None:
        field_literals.append(
            f"match_plan={_source_binding_match_plan_literal(config.match_plan)}"
        )
    return f"StepSourceBindingsConfig({', '.join(field_literals)})"


def _grouped_source_bindings_literal(group: GroupedSourceBindings) -> str:
    binding_literals = ", ".join(
        _named_source_binding_literal(binding)
        for binding in group.bindings
    )
    if len(group.bindings) == 1:
        binding_literals += ","
    group_key = "None" if group.group_key is None else repr(group.group_key)
    return (
        "GroupedSourceBindings("
        f"group_key={group_key}, "
        f"bindings=({binding_literals})"
        ")"
    )


def _named_source_binding_literal(binding: NamedSourceBinding) -> str:
    field_literals = [f"alias={binding.alias!r}"]
    if binding.selector != SourceSelector():
        field_literals.append(
            f"selector={_source_selector_literal(binding.selector)}"
        )
    if binding.origin is not SourceBindingOrigin.STEP_INPUT:
        field_literals.append(
            f"origin=SourceBindingOrigin.{binding.origin.name}"
        )
    return f"NamedSourceBinding({', '.join(field_literals)})"


def _source_selector_literal(selector: SourceSelector) -> str:
    field_literals: list[str] = []
    if selector.components:
        component_literals = ", ".join(
            _component_selector_literal(component)
            for component in selector.components
        )
        if len(selector.components) == 1:
            component_literals += ","
        field_literals.append(f"components=({component_literals})")
    if selector.metadata:
        metadata_literals = ", ".join(
            _metadata_selector_literal(metadata)
            for metadata in selector.metadata
        )
        if len(selector.metadata) == 1:
            metadata_literals += ","
        field_literals.append(f"metadata=({metadata_literals})")
    if not selector.inherit_current_scope:
        field_literals.append("inherit_current_scope=False")
    return f"SourceSelector({', '.join(field_literals)})"


def _component_selector_literal(selector: ComponentSelector) -> str:
    return (
        "ComponentSelector("
        f"AllComponents.{selector.component.name}, {selector.value!r}"
        ")"
    )


def _metadata_selector_literal(selector: MetadataSelector) -> str:
    return f"MetadataSelector({selector.field!r}, {selector.value!r})"


def _metadata_extraction_rule_literal(rule: MetadataExtractionRule) -> str:
    field_literals = [
        f"source=MetadataSource.{rule.source.name}",
        f"pattern={rule.pattern!r}",
    ]
    if rule.filters:
        filter_literals = ", ".join(
            _source_filter_clause_literal(clause)
            for clause in rule.filters
        )
        if len(rule.filters) == 1:
            filter_literals += ","
        field_literals.append(f"filters=({filter_literals})")
    return f"MetadataExtractionRule({', '.join(field_literals)})"


def _source_filter_clause_literal(clause: SourceFilterClause) -> str:
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
            _source_binding_match_dimension_literal(dimension)
            for dimension in plan.dimensions
        )
        if len(plan.dimensions) == 1:
            dimension_literals += ","
        field_literals.append(f"dimensions=({dimension_literals})")
    return f"SourceBindingMatchPlan({', '.join(field_literals)})"


def _source_binding_match_dimension_literal(
    dimension: SourceBindingMatchDimension,
) -> str:
    field_literals = ", ".join(
        _source_binding_match_field_literal(field)
        for field in dimension.fields
    )
    if len(dimension.fields) == 1:
        field_literals += ","
    return f"SourceBindingMatchDimension(fields=({field_literals}))"


def _source_binding_match_field_literal(field: SourceBindingMatchField) -> str:
    return (
        "SourceBindingMatchField("
        f"alias={field.alias!r}, metadata_field={field.metadata_field!r}"
        ")"
    )


def _artifact_spec_literal(
    spec: ArtifactSpec,
    *,
    preserve_default_materialization: bool = False,
) -> str:
    if (
        preserve_default_materialization
        and spec.kind not in DEFAULT_ARTIFACT_MATERIALIZATION_RULES
    ):
        return (
            f"ArtifactSpec({spec.name!r}, ArtifactKind.{spec.kind.name}, "
            "materialization=NO_ARTIFACT_MATERIALIZATION)"
        )
    return f"ArtifactSpec({spec.name!r}, ArtifactKind.{spec.kind.name})"


def _identify_primary_objects(
    builder: _SymbolTableBuilder,
    module: ModuleBlock,
) -> ModuleArtifactContracts:
    image = builder.require(
        _setting(module, "Select the input image"),
        CellProfilerSymbolKind.IMAGE,
        module,
    )
    objects = builder.declare(
        _setting(module, "Name the primary objects to be identified"),
        CellProfilerSymbolKind.OBJECTS,
        module,
    )
    return _contracts(module, builder, inputs=[image], outputs=[objects])


def _identify_secondary_objects(
    builder: _SymbolTableBuilder,
    module: ModuleBlock,
) -> ModuleArtifactContracts:
    input_objects = builder.require(
        _setting(module, "Select the input objects"),
        CellProfilerSymbolKind.OBJECTS,
        module,
    )
    image = builder.require(
        _setting(module, "Select the input image"),
        CellProfilerSymbolKind.IMAGE,
        module,
    )
    output_objects = builder.declare(
        _setting(module, "Name the objects to be identified"),
        CellProfilerSymbolKind.OBJECTS,
        module,
    )
    return _contracts(
        module,
        builder,
        inputs=[input_objects, image],
        outputs=[output_objects],
    )


def _identify_tertiary_objects(
    builder: _SymbolTableBuilder,
    module: ModuleBlock,
) -> ModuleArtifactContracts:
    larger = builder.require(
        _setting(module, "Select the larger identified objects"),
        CellProfilerSymbolKind.OBJECTS,
        module,
    )
    smaller = builder.require(
        _setting(module, "Select the smaller identified objects"),
        CellProfilerSymbolKind.OBJECTS,
        module,
    )
    output = builder.declare(
        _setting(module, "Name the tertiary objects to be identified"),
        CellProfilerSymbolKind.OBJECTS,
        module,
    )
    return _contracts(module, builder, inputs=[larger, smaller], outputs=[output])


def _measure_object_size_shape(
    builder: _SymbolTableBuilder,
    module: ModuleBlock,
) -> ModuleArtifactContracts:
    objects = [
        builder.require(name, CellProfilerSymbolKind.OBJECTS, module)
        for name in _split_names(_setting(module, OBJECT_MEASUREMENT_SETTING))
    ]
    measurements = builder.declare(
        _measurement_name(module),
        CellProfilerSymbolKind.MEASUREMENTS,
        module,
    )
    return _contracts(module, builder, inputs=objects, outputs=[measurements])


def _measure_object_intensity(
    builder: _SymbolTableBuilder,
    module: ModuleBlock,
) -> ModuleArtifactContracts:
    images = [
        builder.require(name, CellProfilerSymbolKind.IMAGE, module)
        for name in _split_names(_setting(module, IMAGE_MEASUREMENT_SETTING))
    ]
    objects = [
        builder.require(name, CellProfilerSymbolKind.OBJECTS, module)
        for name in _split_names(_setting(module, OBJECT_MEASUREMENT_SETTING))
    ]
    measurements = builder.declare(
        _measurement_name(module),
        CellProfilerSymbolKind.MEASUREMENTS,
        module,
    )
    return _contracts(
        module,
        builder,
        inputs=[*images, *objects],
        outputs=[measurements],
    )


def _measure_image_intensity(
    builder: _SymbolTableBuilder,
    module: ModuleBlock,
) -> ModuleArtifactContracts:
    images = [
        builder.require(name, CellProfilerSymbolKind.IMAGE, module)
        for name in _split_names(_setting(module, IMAGE_MEASUREMENT_SETTING))
    ]
    object_setting = _optional_setting(module, "Select input object sets")
    objects = (
        [
            builder.require(name, CellProfilerSymbolKind.OBJECTS, module)
            for name in _split_names(object_setting)
        ]
        if object_setting
        else []
    )
    measurements = builder.declare(
        _measurement_name(module),
        CellProfilerSymbolKind.MEASUREMENTS,
        module,
    )
    return _contracts(
        module,
        builder,
        inputs=[*images, *objects],
        outputs=[measurements],
    )


def _measure_object_neighbors(
    builder: _SymbolTableBuilder,
    module: ModuleBlock,
) -> ModuleArtifactContracts:
    measured = builder.require(
        _setting(module, OBJECT_MEASUREMENT_SETTING),
        CellProfilerSymbolKind.OBJECTS,
        module,
    )
    neighbors = builder.require(
        _setting(module, "Select neighboring objects to measure"),
        CellProfilerSymbolKind.OBJECTS,
        module,
    )
    measurements = builder.declare(
        _measurement_name(module),
        CellProfilerSymbolKind.MEASUREMENTS,
        module,
    )
    return _contracts(
        module,
        builder,
        inputs=[measured, neighbors],
        outputs=[measurements],
    )


def _measure_granularity(
    builder: _SymbolTableBuilder,
    module: ModuleBlock,
) -> ModuleArtifactContracts:
    images = [
        builder.require(name, CellProfilerSymbolKind.IMAGE, module)
        for name in _split_names(_setting(module, IMAGE_MEASUREMENT_SETTING))
    ]
    object_setting = _optional_setting(module, OBJECT_MEASUREMENT_SETTING)
    objects = (
        [
            builder.require(name, CellProfilerSymbolKind.OBJECTS, module)
            for name in _split_names(object_setting)
        ]
        if object_setting is not None
        else []
    )
    measurements = builder.declare(
        _measurement_name(module),
        CellProfilerSymbolKind.MEASUREMENTS,
        module,
    )
    return _contracts(
        module,
        builder,
        inputs=[*images, *objects],
        outputs=[measurements],
    )


def _correct_illumination_apply(
    builder: _SymbolTableBuilder,
    module: ModuleBlock,
) -> ModuleArtifactContracts:
    image = builder.require(
        _setting(module, "Select the input image"),
        CellProfilerSymbolKind.IMAGE,
        module,
    )
    illumination = builder.require(
        _setting(module, "Select the illumination function"),
        CellProfilerSymbolKind.IMAGE,
        module,
    )
    output = builder.declare(
        _setting(module, OUTPUT_IMAGE_SETTING),
        CellProfilerSymbolKind.IMAGE,
        module,
    )
    return _contracts(
        module,
        builder,
        inputs=[image, illumination],
        outputs=[output],
    )


def _opening(
    builder: _SymbolTableBuilder,
    module: ModuleBlock,
) -> ModuleArtifactContracts:
    image = builder.require(
        _setting(module, "Select the input image"),
        CellProfilerSymbolKind.IMAGE,
        module,
    )
    output = builder.declare(
        _setting(module, OUTPUT_IMAGE_SETTING),
        CellProfilerSymbolKind.IMAGE,
        module,
    )
    return _contracts(module, builder, inputs=[image], outputs=[output])


def _convert_objects_to_image(
    builder: _SymbolTableBuilder,
    module: ModuleBlock,
) -> ModuleArtifactContracts:
    objects = builder.require(
        _setting(module, "Select the input objects"),
        CellProfilerSymbolKind.OBJECTS,
        module,
    )
    output = builder.declare(
        _setting(module, OUTPUT_IMAGE_SETTING),
        CellProfilerSymbolKind.IMAGE,
        module,
    )
    return _contracts(module, builder, inputs=[objects], outputs=[output])


def _gray_to_color(
    builder: _SymbolTableBuilder,
    module: ModuleBlock,
) -> ModuleArtifactContracts:
    images = [
        builder.require(name, CellProfilerSymbolKind.IMAGE, module)
        for name in GrayToColorInputNameResolver.for_module(module).input_names(module)
    ]
    output = builder.declare(
        _setting(module, OUTPUT_IMAGE_SETTING),
        CellProfilerSymbolKind.IMAGE,
        module,
    )
    return _contracts(module, builder, inputs=images, outputs=[output])


def _overlay_outlines(
    builder: _SymbolTableBuilder,
    module: ModuleBlock,
) -> ModuleArtifactContracts:
    image = builder.require(
        _setting(module, "Select image on which to display outlines"),
        CellProfilerSymbolKind.IMAGE,
        module,
    )
    objects = [
        builder.require(name, CellProfilerSymbolKind.OBJECTS, module)
        for name in _split_names(_setting(module, DISPLAY_OBJECTS_SETTING))
    ]
    output = builder.declare(
        _setting(module, OUTPUT_IMAGE_SETTING),
        CellProfilerSymbolKind.IMAGE,
        module,
    )
    return _contracts(
        module,
        builder,
        inputs=[image, *objects],
        outputs=[output],
    )


def _relate_objects(
    builder: _SymbolTableBuilder,
    module: ModuleBlock,
) -> ModuleArtifactContracts:
    parent = builder.require(
        _setting(module, "Select the parent objects"),
        CellProfilerSymbolKind.OBJECTS,
        module,
    )
    child = builder.require(
        _setting(module, "Select the child objects"),
        CellProfilerSymbolKind.OBJECTS,
        module,
    )
    relationship = builder.declare(
        _relationship_name(parent.name, child.name),
        CellProfilerSymbolKind.RELATIONSHIPS,
        module,
    )
    measurements = builder.declare(
        _measurement_name(module),
        CellProfilerSymbolKind.MEASUREMENTS,
        module,
    )
    return _contracts(
        module,
        builder,
        inputs=[parent, child],
        outputs=[relationship, measurements],
    )


class ModuleContractBuilder(ABC, metaclass=AutoRegisterMeta):
    """Nominal family for per-module CellProfiler artifact contract compilation."""

    __registry_key__ = "module_name"
    __skip_if_no_key__ = True
    module_name: ClassVar[str | None] = None

    @classmethod
    def for_module(cls, module_name: str) -> "ModuleContractBuilder":
        builder_type = cls.__registry__.get(module_name, UnsupportedModuleContractBuilder)
        return builder_type()

    @abstractmethod
    def build(
        self,
        builder: _SymbolTableBuilder,
        module: ModuleBlock,
    ) -> ModuleArtifactContracts:
        """Compile artifact contracts for one parsed CellProfiler module."""


class FunctionBackedModuleContractBuilder(ModuleContractBuilder):
    """Module builder backed by one shared helper function."""

    builder_function: ClassVar[
        Callable[[_SymbolTableBuilder, ModuleBlock], ModuleArtifactContracts] | None
    ] = None

    def build(
        self,
        builder: _SymbolTableBuilder,
        module: ModuleBlock,
    ) -> ModuleArtifactContracts:
        builder_function = type(self).builder_function
        if builder_function is None:
            raise TypeError(
                f"{type(self).__name__} must define builder_function."
            )
        return builder_function(builder, module)


class UnsupportedModuleContractBuilder(ModuleContractBuilder):
    """Default builder for modules whose runtime artifact semantics are not declared."""

    def build(
        self,
        builder: _SymbolTableBuilder,
        module: ModuleBlock,
    ) -> ModuleArtifactContracts:
        return ModuleArtifactContracts(module.name, module.module_num)


_FUNCTION_BACKED_MODULE_BUILDER_SPECS: tuple[
    tuple[tuple[str, ...], Callable[[_SymbolTableBuilder, ModuleBlock], ModuleArtifactContracts]],
    ...,
] = (
    (("CorrectIlluminationApply",), _correct_illumination_apply),
    (("Opening",), _opening),
    (("IdentifyPrimaryObjects",), _identify_primary_objects),
    (("IdentifySecondaryObjects",), _identify_secondary_objects),
    (("IdentifyTertiaryObjects",), _identify_tertiary_objects),
    (("ConvertObjectsToImage",), _convert_objects_to_image),
    (("GrayToColor",), _gray_to_color),
    (("OverlayOutlines",), _overlay_outlines),
    (("MeasureObjectSizeShape",), _measure_object_size_shape),
    (
        (
            "MeasureObjectIntensity",
            "MeasureObjectIntensityDistribution",
            "MeasureTexture",
            "MeasureColocalization",
        ),
        _measure_object_intensity,
    ),
    (("MeasureGranularity",), _measure_granularity),
    (("MeasureImageIntensity",), _measure_image_intensity),
    (("MeasureObjectNeighbors",), _measure_object_neighbors),
    (("RelateObjects",), _relate_objects),
)


def _declare_function_backed_module_builder(
    module_name: str,
    builder_function: Callable[
        [_SymbolTableBuilder, ModuleBlock],
        ModuleArtifactContracts,
    ],
) -> None:
    type(
        f"{module_name}ContractBuilder",
        (FunctionBackedModuleContractBuilder,),
        {
            "__module__": __name__,
            "module_name": module_name,
            "builder_function": builder_function,
        },
    )


for _module_names, _builder_function in _FUNCTION_BACKED_MODULE_BUILDER_SPECS:
    for _module_name in _module_names:
        _declare_function_backed_module_builder(_module_name, _builder_function)


def _contracts(
    module: ModuleBlock,
    builder: _SymbolTableBuilder,
    *,
    inputs: Iterable[CellProfilerSymbol] = (),
    outputs: Iterable[CellProfilerSymbol] = (),
) -> ModuleArtifactContracts:
    input_symbols = _unique_symbols(inputs)
    return ModuleArtifactContracts(
        module_name=module.name,
        module_num=module.module_num,
        input_symbols=input_symbols,
        output_symbols=_unique_symbols(outputs),
        source_bindings=builder.source_bindings_for(
            symbol for symbol in input_symbols if symbol.is_external_source
        ),
    )


def _setting(module: ModuleBlock, name: str | SettingNameFamily) -> str:
    value = _optional_setting(module, name)
    if value is None:
        raise ValueError(
            f"Module {module.name}({module.module_num}) missing setting "
            f"{_setting_names(name)}."
        )
    return value


def _optional_setting(
    module: ModuleBlock,
    name: str | SettingNameFamily,
) -> str | None:
    for setting_name in _setting_name_candidates(name):
        value = module.settings.get(setting_name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _split_names(value: str) -> tuple[str, ...]:
    return tuple(
        _normalize_symbol_name(part)
        for part in value.split(",")
        if part.strip()
    )
def _normalized_setting_symbol(
    module: ModuleBlock,
    setting: str | SettingNameFamily,
) -> str | None:
    value = _optional_setting(module, setting)
    if value is None:
        return None
    normalized = _normalize_symbol_name(value)
    return _normalized_optional_symbol_value(normalized)


def _normalized_optional_symbol_value(value: str) -> str | None:
    normalized = _normalize_symbol_name(value)
    if normalized.lower() in {"leave this black", "none", "do not use"}:
        return None
    return normalized


def _setting_name_candidates(
    name: str | SettingNameFamily,
) -> tuple[str, ...]:
    if isinstance(name, SettingNameFamily):
        return name.names
    return (name,)


def _setting_names(name: str | SettingNameFamily) -> tuple[str, ...]:
    return _setting_name_candidates(name)


def _normalize_symbol_name(name: str) -> str:
    normalized = name.strip()
    if not normalized:
        raise ValueError("CellProfiler symbol names cannot be empty.")
    return normalized


def _unique_symbols(
    symbols: Iterable[CellProfilerSymbol],
) -> tuple[CellProfilerSymbol, ...]:
    unique: list[CellProfilerSymbol] = []
    seen: set[tuple[str, CellProfilerSymbolKind]] = set()
    for symbol in symbols:
        key = (symbol.name, symbol.kind)
        if key not in seen:
            unique.append(symbol)
            seen.add(key)
    return tuple(unique)


def _measurement_name(module: ModuleBlock) -> str:
    return f"{module.name}_{module.module_num}_measurements"


def _relationship_name(parent: str, child: str) -> str:
    return f"{parent}_{child}_relationships"
