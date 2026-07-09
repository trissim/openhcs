"""
PipelineGenerator - Generate complete runnable OpenHCS pipelines.

DETERMINISTIC ONLY:
Uses pre-absorbed cellprofiler_library. No LLM fallback.
Fails loudly if modules are missing from the absorbed library.

Takes parsed .cppipe modules and generates a complete pipeline file with:
- All imports
- Function references from absorbed library
- FunctionStep wrappers with correct variable_components
- Pipeline configuration
"""

from __future__ import annotations
import json
import logging
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field as dataclass_field, replace
from enum import Enum
from pathlib import Path
from typing import ClassVar, List, Optional, TypeAlias
from metaclass_registry import AutoRegisterMeta
from openhcs.constants import Backend
from openhcs.constants.constants import (
    GroupBy,
    Microscope,
    VariableComponents,
    get_default_group_by,
)
from openhcs.core.artifact_materialization_policy import NO_ARTIFACT_MATERIALIZATION
from openhcs.core.artifact_observability import externally_required_artifact_outputs
from openhcs.core.artifacts import (
    ArtifactSpec,
    ArtifactType,
    SpecialArtifactType,
    ImageArtifactType,
)
from openhcs.core.config import (
    LazyProcessingConfig,
    LazySourceBindingsConfig,
    PipelineConfig,
)
from openhcs.core.module_artifact_contract import (
    DeclaredArtifactOutputPartition,
    ModuleArtifactContract,
    RecordedArtifactOutputPartition,
    RuntimeArtifactInputPartition,
    SourceArtifactInputPartition,
)
from openhcs.core.pipeline_image_schema import (
    PipelineImageSchema,
    PipelineImageSchemaSourceBindingsRepresentability,
)
from openhcs.core.python_source_literal import PythonSourceLiteral
from openhcs.core.source_bindings import SourceBindingsConfig
from openhcs.core.vfs_protocol import FileManagerLike
from openhcs.interop.cellprofiler.module_roles import (
    ArtifactSpecKey,
    cellprofiler_infrastructure_import_note,
    cellprofiler_infrastructure_retained_artifacts,
)
from openhcs.interop.cellprofiler.parser import ModuleBlock
from openhcs.interop.cellprofiler.artifact_semantics import artifact_setting_symbols
from openhcs.interop.cellprofiler.settings_binder import (
    SettingsBinder,
    normalize_cellprofiler_setting_name,
)
from openhcs.interop.cellprofiler.setting_names import SettingNameFamily
from openhcs.processing.backends.cellprofiler.library import canonical_module_name
from openhcs.interop.cellprofiler.module_declarations import (
    BoundModuleSettings,
    CellProfilerCompileTimeArtifactFlow,
    CellProfilerModule,
    ModuleSettingCoverageRecord,
)
from openhcs.processing.materialization import MaterializedFilenameIdentity, tiff_stack
from openhcs.interop.cellprofiler.symbol_table import (
    CellProfilerSymbolTable,
    ModuleArtifactContracts,
    step_source_bindings_literal,
)
from openhcs.interop.cellprofiler.module_processing_components import (
    GeneratedLiteralValue,
    GeneratedParameterName,
    GeneratedStepSettingKey,
    GeneratedStepSettings,
    ModuleProcessingComponentRequest,
    ModuleProcessingComponents,
    RuntimeArtifactLineageScope,
    RuntimeArtifactSourceLineage,
    group_by_component_axis,
)

logger = logging.getLogger(__name__)
ModuleGenerationRecord: TypeAlias = type[CellProfilerModule]


ExternallyMaterializedOutputs = frozenset[ArtifactSpecKey]
ArtifactNameMaterializedOutputs = frozenset[ArtifactSpecKey]


def _has_materialized_output(contract: ModuleArtifactContracts) -> bool:
    """Return whether any output is externally observable by artifact policy."""
    return any(
        (
            spec.materialization is not None
            or spec.artifact_type.has_default_materialization()
            for spec in contract.outputs
        )
    )


@dataclass
class GeneratedPipeline:
    """Complete generated OpenHCS pipeline."""

    name: str
    code: str
    source_cppipe: str
    converted_modules: List[str]
    failed_modules: List[str]
    artifact_contracts: tuple[ModuleArtifactContracts, ...] = ()
    runtime_module_contracts: tuple[tuple[int, ModuleArtifactContract], ...] = ()
    source_schema: PipelineImageSchema = PipelineImageSchema.empty()
    pipeline_config: PipelineConfig = dataclass_field(default_factory=PipelineConfig)
    setting_coverage: tuple[ModuleSettingCoverageRecord, ...] = ()

    @property
    def runtime_module_contracts_by_module_num(
        self,
    ) -> dict[int, ModuleArtifactContract]:
        """Runtime artifact contracts keyed by original CellProfiler module number."""
        return dict(self.runtime_module_contracts)

    def save(
        self,
        output_path: Path,
        *,
        filemanager: FileManagerLike | None = None,
        backend: Backend = Backend.DISK,
    ) -> None:
        """Save pipeline to file."""
        if not isinstance(backend, Backend):
            raise TypeError(
                f"GeneratedPipeline.save backend must be Backend, got {type(backend).__name__}."
            )
        if filemanager is None:
            output_path.write_text(self.code)
        else:
            filemanager.ensure_directory(str(output_path.parent), backend.value)
            filemanager.save(self.code, str(output_path), backend.value)
        logger.info(f"Saved pipeline to {output_path}")


@dataclass(frozen=True, slots=True)
class SkippedModuleSelection:
    """Public optional skipped-module argument normalized for generation."""

    modules: tuple[ModuleBlock, ...] = ()

    @classmethod
    def from_optional(
        cls, modules: Optional[List[ModuleBlock]]
    ) -> "SkippedModuleSelection":
        if modules is None:
            return cls(())
        return cls(tuple(modules))


@dataclass(frozen=True)
class GeneratedPipelineRequest:
    """Nominal request for one registry-backed CellProfiler pipeline generation."""

    pipeline_name: str
    source_cppipe: Path
    skipped_modules: tuple[ModuleBlock, ...] = ()
    prune_dead_unmaterialized_artifact_steps: bool = False
    materialize_skipped_save_images: bool = True
    materialize_terminal_images: bool = True

    @classmethod
    def from_public_args(
        cls,
        *,
        pipeline_name: str,
        source_cppipe: Path,
        skipped_modules: Optional[List[ModuleBlock]],
        prune_dead_unmaterialized_artifact_steps: bool,
        materialize_skipped_save_images: bool,
        materialize_terminal_images: bool,
    ) -> "GeneratedPipelineRequest":
        """Build a generation request from the stable public API arguments."""
        return cls(
            pipeline_name=pipeline_name,
            source_cppipe=source_cppipe,
            skipped_modules=SkippedModuleSelection.from_optional(
                skipped_modules
            ).modules,
            prune_dead_unmaterialized_artifact_steps=prune_dead_unmaterialized_artifact_steps,
            materialize_skipped_save_images=materialize_skipped_save_images,
            materialize_terminal_images=materialize_terminal_images,
        )


@dataclass(frozen=True)
class PipelineGeneratorRegistryStage:
    """CellProfiler module-class loading and module lookup."""

    generator: "PipelineGenerator"

    def load_registry(self) -> dict[str, ModuleGenerationRecord]:
        """Load module generation records from class declarations."""
        if self.generator._explicit_library_root:
            return self.load_legacy_registry(self.generator.library_root)
        try:
            registry = {
                str(module_type.module_name): module_type
                for module_type in CellProfilerModule.__registry__.values()
                if module_type.validated
            }
            logger.info(f"Loaded {len(registry)} absorbed CellProfiler module classes")
            return registry
        except Exception as e:
            raise RuntimeError(f"Failed to load registry: {e}")

    def load_legacy_registry(
        self, library_root: Path
    ) -> dict[str, ModuleGenerationRecord]:
        """Load an explicit absorbed-library root through module declarations."""
        contracts_file = library_root / "contracts.json"
        if not contracts_file.exists():
            raise FileNotFoundError(
                f"No absorbed library found at {contracts_file}. Run 'python -m benchmark.converter.absorb' first."
            )
        try:
            data = json.loads(contracts_file.read_text())
            registry: dict[str, ModuleGenerationRecord] = {}
            for module_name, info in data.items():
                if not isinstance(info, Mapping):
                    raise TypeError(
                        f"Absorbed library metadata for {module_name!r} must be a mapping."
                    )
                if "validated" not in info or not isinstance(info["validated"], bool):
                    raise ValueError(
                        f"Absorbed library metadata for {module_name!r} must declare boolean validated."
                    )
                if not info["validated"]:
                    continue
                module_type = CellProfilerModule.for_module(module_name)
                if module_type is None:
                    raise KeyError(
                        f"Absorbed library metadata names undeclared CellProfiler module {module_name!r}."
                    )
                registry[str(module_type.module_name)] = module_type
            return registry
        except Exception as e:
            raise RuntimeError(f"Failed to load registry: {e}")

    def has_module(self, module_name: str) -> bool:
        """Check if module exists in absorbed library."""
        return canonical_module_name(module_name) in self.generator._registry

    def module_record(self, module_name: str) -> ModuleGenerationRecord:
        """Return the module generation record after canonical name resolution."""
        return self.generator._registry[canonical_module_name(module_name)]

    def module_class(self, module_name: str) -> type[CellProfilerModule] | None:
        """Return the selected module class for class-backed generation."""
        return self.module_record(module_name)

    def required_module_class(self, module_name: str) -> type[CellProfilerModule]:
        """Return the module declaration class required for semantic queries."""
        module_type = self.module_class(module_name)
        if module_type is None:
            raise KeyError(
                f"CellProfiler module {module_name!r} is not declared by CellProfilerModule."
            )
        return module_type

    def bind_settings(
        self,
        module: ModuleBlock,
        *,
        param_mapping: Mapping[str, GeneratedParameterName],
        ignored_unmapped_settings: frozenset[str] = frozenset(),
    ) -> BoundModuleSettings:
        """Bind module settings through the selected module declaration."""
        return self.required_module_class(module.name).bind_settings(
            module,
            binder=self.generator.settings_binder,
            param_mapping=param_mapping,
            ignored_unmapped_settings=ignored_unmapped_settings,
        )

    def resolve_function(
        self,
        module: ModuleBlock,
        *,
        request: ModuleProcessingComponentRequest | None = None,
    ) -> str:
        """Resolve the generated function name through the module declaration."""
        module_record = self.module_record(module.name)
        module_type = self.required_module_class(module.name)
        resolved = (
            module_type.resolve_semantic_function(
                module,
                default_function_name=module_record.function_name,
                request=request,
            )
            if request is not None
            else module_type.resolve_function(
                module,
                default_function_name=module_record.function_name,
            )
        )
        return resolved.function_name

    def processing_components(
        self, module: ModuleBlock, request: ModuleProcessingComponentRequest
    ):
        """Return generated FunctionStep component semantics from the module class."""
        return self.required_module_class(module.name).processing_components(request)


@dataclass(slots=True)
class OutputSymbolsBySetting:
    """Output artifact symbols grouped by normalized CellProfiler setting name."""

    values: dict[str, set[ArtifactSpecKey]]

    @classmethod
    def empty(cls) -> "OutputSymbolsBySetting":
        return cls({})

    def add(self, setting_name: str, artifact: ArtifactSpecKey) -> None:
        if setting_name not in self.values:
            self.values[setting_name] = set()
        self.values[setting_name].add(artifact)

    def dead_setting_names(
        self, retained_outputs: frozenset[ArtifactSpecKey]
    ) -> frozenset[str]:
        return frozenset(
            (
                setting_name
                for setting_name, output_symbols in self.values.items()
                if output_symbols and (not output_symbols & retained_outputs)
            )
        )


@dataclass(frozen=True)
class PipelineGeneratorArtifactPruner:
    """Dead-artifact pruning and setting-pruning authority."""

    generator: "PipelineGenerator"

    def prune_dead_unmaterialized_artifact_steps(
        self,
        modules: list[ModuleBlock],
        artifact_contracts: dict[int, ModuleArtifactContracts],
        *,
        externally_required_artifacts: set[ArtifactSpecKey] | None = None,
    ) -> list[ModuleBlock]:
        """Remove artifact-producing steps whose outputs are neither consumed nor materialized."""
        live_artifacts = {
            ArtifactSpecKey.from_spec(output)
            for contract in artifact_contracts.values()
            for output in contract.outputs
            if output.materialization is not None
            or output.artifact_type.has_default_materialization()
        } | {
            ArtifactSpecKey.from_spec(output)
            for contract in artifact_contracts.values()
            for output in externally_required_artifact_outputs(
                contract.declared_outputs
            )
        }
        if externally_required_artifacts:
            live_artifacts.update(externally_required_artifacts)
        live_module_nums: set[int] = set()
        for module in reversed(modules):
            contract = artifact_contracts[module.module_num]
            output_keys = {
                ArtifactSpecKey.from_spec(output) for output in contract.outputs
            }
            keep = (
                not contract.outputs
                or _has_materialized_output(contract)
                or bool(output_keys & live_artifacts)
            )
            if not keep:
                continue
            live_module_nums.add(module.module_num)
            retained_outputs = tuple(
                (
                    output
                    for output in contract.outputs
                    if output.materialization is not None
                    or output.artifact_type.has_default_materialization()
                    or ArtifactSpecKey.from_spec(output) in live_artifacts
                )
            )
            artifact_contracts[module.module_num] = replace(
                contract,
                output_symbols=tuple(
                    (
                        symbol
                        for symbol in contract.output_symbols
                        if symbol.artifact_spec in retained_outputs
                    )
                ),
            )
            live_artifacts.update(
                (
                    ArtifactSpecKey.from_spec(input_spec)
                    for input_spec in contract.runtime_artifact_inputs
                )
            )
        pruned = [module for module in modules if module.module_num in live_module_nums]
        skipped = [
            module for module in modules if module.module_num not in live_module_nums
        ]
        if skipped:
            logger.info(
                "Pruned %d dead unmaterialized artifact step(s): %s",
                len(skipped),
                [module.name for module in skipped],
            )
        return pruned

    def prune_dead_output_setting_kwargs(
        self,
        *,
        module: ModuleBlock,
        translated_kwargs: GeneratedStepSettings,
        param_mapping: Mapping[str, GeneratedParameterName],
        artifact_contract: ModuleArtifactContracts,
    ) -> GeneratedStepSettings:
        """Drop function kwargs for output-name settings pruned from artifacts."""
        dead_settings = self.dead_output_setting_names(
            module=module, artifact_contract=artifact_contract
        )
        return translated_kwargs.without_dead_output_settings(
            dead_settings=dead_settings, param_mapping=param_mapping
        )

    @staticmethod
    def dead_output_setting_names(
        *, module: ModuleBlock, artifact_contract: ModuleArtifactContracts
    ) -> frozenset[str]:
        """Return CellProfiler output-name settings whose artifacts were pruned."""
        retained_outputs = frozenset(
            (
                ArtifactSpecKey(symbol.artifact_type, symbol.name)
                for symbol in artifact_contract.output_symbols
            )
        )
        output_symbols_by_setting = OutputSymbolsBySetting.empty()
        for symbol in artifact_setting_symbols(module):
            if symbol.is_input:
                continue
            normalized_setting = normalize_cellprofiler_setting_name(
                symbol.setting_name
            )
            output_symbols_by_setting.add(
                normalized_setting, ArtifactSpecKey(symbol.artifact_type, symbol.name)
            )
        return output_symbols_by_setting.dead_setting_names(retained_outputs)

    @staticmethod
    def terminal_image_artifacts(
        modules: list[ModuleBlock],
        artifact_contracts: Mapping[int, ModuleArtifactContracts],
        *,
        external_consumers: Iterable[ArtifactSpec] = (),
    ) -> set[ArtifactSpecKey]:
        """Return final image outputs that remain observable pipeline products."""
        consumed: set[ArtifactSpecKey] = {
            ArtifactSpecKey.from_spec(spec) for spec in external_consumers
        }
        terminal: set[ArtifactSpecKey] = set()
        for module in reversed(modules):
            contract = artifact_contracts[module.module_num]
            module_outputs = {
                ArtifactSpecKey.from_spec(spec)
                for spec in contract.outputs
                if spec.artifact_type is ImageArtifactType
            }
            terminal.update(
                (output for output in module_outputs if output not in consumed)
            )
            consumed.update(
                (ArtifactSpecKey.from_spec(spec) for spec in contract.inputs)
            )
            consumed.update(
                (
                    ArtifactSpecKey.from_spec(spec)
                    for spec in contract.runtime_artifact_inputs
                )
            )
        return terminal


@dataclass(frozen=True)
class PipelineGeneratorRuntimeContractProjector:
    """Projection from symbol-table contracts to runtime artifact contracts."""

    generator: "PipelineGenerator"

    def by_module_num(
        self,
        modules: List[ModuleBlock],
        artifact_contracts: dict[int, ModuleArtifactContracts],
        *,
        externally_materialized_outputs: ExternallyMaterializedOutputs = frozenset(),
        artifact_name_materialized_outputs: ArtifactNameMaterializedOutputs = frozenset(),
    ) -> dict[int, ModuleArtifactContract]:
        """Build product-runtime contracts without serializing them into generated code."""
        return {
            module.module_num: self.runtime_module_contract(
                artifact_contracts[module.module_num],
                externally_materialized_outputs=externally_materialized_outputs,
                artifact_name_materialized_outputs=artifact_name_materialized_outputs,
            )
            for module in modules
            if artifact_contracts[module.module_num].inputs
            or artifact_contracts[module.module_num].outputs
        }

    def runtime_module_contract(
        self,
        contract: ModuleArtifactContracts,
        *,
        externally_materialized_outputs: ExternallyMaterializedOutputs,
        artifact_name_materialized_outputs: ArtifactNameMaterializedOutputs,
    ) -> ModuleArtifactContract:
        """Project symbol-table contracts into runtime module contracts."""
        outputs = tuple(
            (
                self.runtime_output_spec(
                    spec,
                    externally_materialized_outputs=externally_materialized_outputs,
                    artifact_name_materialized_outputs=artifact_name_materialized_outputs,
                )
                for spec in contract.outputs
            )
        )
        declared_outputs = tuple(
            (
                self.runtime_output_spec(
                    spec,
                    externally_materialized_outputs=externally_materialized_outputs,
                    artifact_name_materialized_outputs=artifact_name_materialized_outputs,
                )
                for spec in contract.declared_outputs
            )
        )
        return ModuleArtifactContract(
            module_name=contract.module_name,
            items=(
                *ModuleArtifactContract.items_for_partition(
                    SourceArtifactInputPartition, contract.inputs
                ),
                *ModuleArtifactContract.items_for_partition(
                    RuntimeArtifactInputPartition, contract.runtime_artifact_inputs
                ),
                *ModuleArtifactContract.items_for_partition(
                    RecordedArtifactOutputPartition, outputs
                ),
                *ModuleArtifactContract.items_for_partition(
                    DeclaredArtifactOutputPartition, declared_outputs
                ),
            ),
            required_variable_components=self.required_variable_components(
                contract.module_name
            ),
        )

    @staticmethod
    def required_variable_components(module_name: str):
        """Return FunctionStep component requirements declared by the module."""
        module_type = CellProfilerModule.for_module(module_name)
        if module_type is None:
            return ()
        return module_type.required_variable_components

    @staticmethod
    def runtime_output_spec(
        spec: ArtifactSpec,
        *,
        externally_materialized_outputs: ExternallyMaterializedOutputs,
        artifact_name_materialized_outputs: ArtifactNameMaterializedOutputs,
    ) -> ArtifactSpec:
        """Apply runtime-only materialization required by skipped SaveImages modules."""
        if ArtifactSpecKey.from_spec(spec) not in externally_materialized_outputs:
            if (
                spec.materialization is None
                and spec.artifact_type is not SpecialArtifactType
                and (not spec.artifact_type.has_default_materialization())
            ):
                return replace(spec, materialization=NO_ARTIFACT_MATERIALIZATION)
            return spec
        filename_identity = (
            MaterializedFilenameIdentity.ARTIFACT_NAME
            if ArtifactSpecKey.from_spec(spec) in artifact_name_materialized_outputs
            else MaterializedFilenameIdentity.SOURCE_IDENTITY
        )
        return replace(
            spec,
            materialization=tiff_stack(
                normalize_uint8=True, filename_identity=filename_identity
            ),
        )


class ArtifactContractCommentSection(ABC, metaclass=AutoRegisterMeta):
    """Auto-registered generated-comment section for artifact contracts."""

    __registry_key__ = "section_name"
    __skip_if_no_key__ = True
    section_name: ClassVar[str]

    @classmethod
    def lines_for(cls, contract: ModuleArtifactContracts) -> list[str]:
        lines: list[str] = []
        for section_type in cls.__registry__.values():
            section = section_type()
            if section.matches(contract):
                lines.append(section.line(contract))
        return lines

    @staticmethod
    def format_artifact_specs(specs: tuple[ArtifactSpec, ...]) -> str:
        """Format artifact specs for deterministic generated-code comments."""
        return ", ".join((f"{spec.artifact_type.value}:{spec.name}" for spec in specs))

    @abstractmethod
    def matches(self, contract: ModuleArtifactContracts) -> bool:
        """Return whether this comment section has content."""

    @abstractmethod
    def line(self, contract: ModuleArtifactContracts) -> str:
        """Return the generated comment line for this section."""


class InputArtifactCommentSection(ArtifactContractCommentSection):
    """Comment section for declared CellProfiler artifact inputs."""

    section_name = "inputs"

    def matches(self, contract: ModuleArtifactContracts) -> bool:
        return bool(contract.inputs)

    def line(self, contract: ModuleArtifactContracts) -> str:
        return "        # CellProfiler artifact inputs: " + self.format_artifact_specs(
            contract.inputs
        )


class SourceBindingCommentSection(ArtifactContractCommentSection):
    """Comment section for external source-image bindings."""

    section_name = "source_bindings"

    def matches(self, contract: ModuleArtifactContracts) -> bool:
        return bool(contract.external_source_symbols)

    def line(self, contract: ModuleArtifactContracts) -> str:
        return "        # Source bindings: " + ", ".join(
            (symbol.name for symbol in contract.external_source_symbols)
        )


class RuntimeArtifactCommentSection(ArtifactContractCommentSection):
    """Comment section for runtime artifact dependencies."""

    section_name = "runtime_artifact_inputs"

    def matches(self, contract: ModuleArtifactContracts) -> bool:
        return bool(contract.runtime_artifact_inputs)

    def line(self, contract: ModuleArtifactContracts) -> str:
        return "        # Runtime artifact inputs: " + self.format_artifact_specs(
            contract.runtime_artifact_inputs
        )


class OutputArtifactCommentSection(ArtifactContractCommentSection):
    """Comment section for declared CellProfiler artifact outputs."""

    section_name = "outputs"

    def matches(self, contract: ModuleArtifactContracts) -> bool:
        return bool(contract.outputs)

    def line(self, contract: ModuleArtifactContracts) -> str:
        return "        # CellProfiler artifact outputs: " + self.format_artifact_specs(
            contract.outputs
        )


@dataclass(frozen=True, slots=True)
class StepInputSourceLiteral:
    """Generated LazyProcessingConfig input_source fragment for one step."""

    value: str | None = None

    @classmethod
    def from_contract(
        cls, contract: ModuleArtifactContracts
    ) -> "StepInputSourceLiteral":
        if contract.external_source_symbols:
            return cls("InputSource.PIPELINE_START")
        return cls()

    def append_to(self, lines: list[str]) -> None:
        if self.value is None:
            return
        lines.append(f"            input_source={self.value},")


@dataclass(frozen=True, slots=True)
class GeneratedProcessingConfigShape:
    """Concrete processing-config semantics emitted or inherited by a step."""

    variable_components: tuple[str, ...]
    group_by: str | None
    input_source: str | None = None

    @classmethod
    def from_emission(
        cls,
        emission: "GeneratedStepEmission",
    ) -> "GeneratedProcessingConfigShape":
        return cls(
            variable_components=(
                emission.processing_components.variable_component_literals
            ),
            group_by=effective_group_by_literal(emission.processing_components),
            input_source=emission.input_source_literal.value,
        )

    def without_input_source(self) -> "GeneratedProcessingConfigShape":
        return GeneratedProcessingConfigShape(
            variable_components=self.variable_components,
            group_by=self.group_by,
        )

    def to_lazy_processing_config(self) -> LazyProcessingConfig:
        variable_components = [
            VariableComponents[literal.rsplit(".", 1)[1]]
            for literal in self.variable_components
        ]
        return LazyProcessingConfig(
            variable_components=variable_components,
            group_by=(
                None if self.group_by is None else GroupBy[self.group_by.rsplit(".", 1)[1]]
            ),
        )


def effective_group_by_literal(
    processing_components: ModuleProcessingComponents,
) -> str | None:
    """Return concrete group_by literal, resolving inherited generator default."""

    explicit_literal = processing_components.group_by_literal
    if explicit_literal is not None:
        return explicit_literal
    default_group_by = get_default_group_by()
    if default_group_by is None:
        return None
    return f"GroupBy.{default_group_by.name}"


@dataclass(frozen=True, slots=True)
class GeneratedPipelineConfigDefaults:
    """Pipeline-scope defaults that generated FunctionSteps may inherit."""

    processing: GeneratedProcessingConfigShape | None = None
    source_bindings: SourceBindingsConfig | None = None

    @classmethod
    def from_emission_groups(
        cls,
        emission_groups: tuple["GeneratedStepEmissionGroup", ...],
        source_schema: PipelineImageSchema,
    ) -> "GeneratedPipelineConfigDefaults":
        processing_default = cls._processing_default(emission_groups)
        source_bindings_default = cls._source_bindings_default(source_schema)
        return cls(
            processing=processing_default,
            source_bindings=source_bindings_default,
        )

    @staticmethod
    def _processing_default(
        emission_groups: tuple["GeneratedStepEmissionGroup", ...],
    ) -> GeneratedProcessingConfigShape | None:
        shapes = tuple(
            GeneratedProcessingConfigShape.from_emission(group.first)
            .without_input_source()
            for group in emission_groups
        )
        if not shapes:
            return None
        shape, count = Counter(shapes).most_common(1)[0]
        if count < 2:
            return None
        return shape

    @staticmethod
    def _source_bindings_default(
        source_schema: PipelineImageSchema,
    ) -> SourceBindingsConfig | None:
        if source_schema.is_empty:
            return None
        source_bindings = source_schema.to_runtime_source_bindings_config()
        if source_bindings.is_empty:
            return None
        return source_bindings

    @staticmethod
    def _lazy_source_bindings_config(
        source_bindings: SourceBindingsConfig,
    ) -> LazySourceBindingsConfig:
        return LazySourceBindingsConfig(
            metadata_rules=source_bindings.metadata_rule_declarations,
            match_plan=source_bindings.match_plan,
            source_filters=source_bindings.source_filter_declarations,
            bindings=source_bindings.binding_declarations,
        )

    def pipeline_config_kwargs(self) -> dict:
        kwargs = {}
        if self.processing is not None:
            kwargs["processing_config"] = self.processing.to_lazy_processing_config()
        return kwargs


@dataclass(frozen=True)
class GeneratedStepEmission:
    """Compiler-derived source for one generated FunctionStep item."""

    module: ModuleBlock
    module_type: type[CellProfilerModule]
    step_name: str
    artifact_contract: ModuleArtifactContracts
    func_name: str
    translated_kwargs: GeneratedStepSettings
    invocation_options_literal: str | None
    processing_components: ModuleProcessingComponents
    input_source_literal: StepInputSourceLiteral


@dataclass(frozen=True)
class GeneratedStepEmissionGroup:
    """One generated FunctionStep, possibly containing dict-pattern items."""

    emissions: tuple[GeneratedStepEmission, ...]

    @property
    def first(self) -> GeneratedStepEmission:
        return self.emissions[0]

    @property
    def is_grouped(self) -> bool:
        return len(self.emissions) > 1

    @property
    def public_function_spec_is_grouped(self) -> bool:
        """Return whether public callable behavior differs across grouped emissions."""
        if not self.is_grouped:
            return False
        if self.first.module_type.force_grouped_public_function_spec:
            return True
        signatures = {
            self.public_function_spec_signature(emission)
            for emission in self.emissions
        }
        return len(signatures) > 1

    @staticmethod
    def public_function_spec_signature(emission: GeneratedStepEmission) -> tuple:
        """Return the public callable signature that generated source emits."""
        return (
            emission.func_name,
            tuple(
                (
                    key,
                    python_literal(value, import_collector=None),
                )
                for key, value in GeneratedStepEmissionGroup.runtime_function_settings(
                    emission
                ).items()
            ),
            tuple(
                (
                    key,
                    python_literal(value, import_collector=None),
                )
                for key, value in GeneratedStepEmissionGroup.grouped_public_settings(
                    emission
                ).items()
            ),
            emission.invocation_options_literal,
        )

    @staticmethod
    def runtime_function_settings(
        emission: GeneratedStepEmission,
    ) -> GeneratedStepSettings:
        """Return kwargs that affect runtime callable behavior."""
        consumed_names = set(emission.module_type.compile_time_consumed_kwarg_names())
        if not consumed_names:
            return emission.translated_kwargs
        return GeneratedStepSettings.from_mapping(
            {
                key: value
                for key, value in emission.translated_kwargs.items()
                if key not in consumed_names
            }
        )

    @staticmethod
    def grouped_public_settings(
        emission: GeneratedStepEmission,
    ) -> GeneratedStepSettings:
        """Return compile-time public kwargs that distinguish grouped calls."""
        grouped_names = set(emission.module_type.compile_time_grouped_public_kwarg_names())
        if not grouped_names:
            return GeneratedStepSettings()
        return GeneratedStepSettings.from_mapping(
            {
                key: value
                for key, value in emission.translated_kwargs.items()
                if key in grouped_names
            }
        )

    @staticmethod
    def coalesced_public_settings(
        emissions: tuple[GeneratedStepEmission, ...],
    ) -> GeneratedStepSettings:
        """Return compile-time public kwargs merged across one coalesced call."""
        if not emissions:
            return GeneratedStepSettings()
        module_type = emissions[0].module_type
        coalesced_names = module_type.compile_time_coalesced_public_kwarg_names()
        if not coalesced_names:
            return GeneratedStepSettings()

        values_by_name: dict[GeneratedStepSettingKey, list[GeneratedLiteralValue]] = {
            name: [] for name in coalesced_names
        }
        for emission in emissions:
            settings = dict(emission.translated_kwargs.items())
            for name in coalesced_names:
                if name in settings:
                    values_by_name[name].append(settings[name])

        return GeneratedStepSettings.from_mapping(
            {
                name: module_type.coalesce_compile_time_public_kwarg_values(
                    name,
                    tuple(values),
                )
                for name, values in values_by_name.items()
                if values
            }
        )


@dataclass(frozen=True)
class PipelineGeneratorCodeEmitter:
    """Generated-code emission for imports, FunctionStep declarations, and comments."""

    generator: "PipelineGenerator"

    def generate_steps_from_registry(
        self,
        modules: List[ModuleBlock],
        function_names_by_module: Mapping[int, str],
        artifact_contracts: dict[int, ModuleArtifactContracts],
        runtime_artifact_contracts: Mapping[int, ModuleArtifactContract],
        source_schema: PipelineImageSchema,
    ) -> tuple[
        str,
        tuple[ModuleSettingCoverageRecord, ...],
        GeneratedPipelineConfigDefaults,
    ]:
        """Generate pipeline_steps using registry functions with bound settings."""
        lines = [
            "# Pipeline Steps",
            "# Settings from .cppipe are bound as default parameters",
            "# variable_components derived from module declarations and source semantics",
            "pipeline_steps = [",
        ]
        literal_imports: set[tuple[str, str]] = set()
        setting_coverage: list[ModuleSettingCoverageRecord] = []
        source_lineage = RuntimeArtifactSourceLineage(artifact_contracts, source_schema)
        emissions: list[GeneratedStepEmission] = []
        artifact_flow = CellProfilerCompileTimeArtifactFlow.empty()
        for module in modules:
            module_type = self.generator.registry.required_module_class(module.name)
            step_name = module.name
            artifact_contract = artifact_contracts[module.module_num]
            func_name = function_names_by_module[module.module_num]
            input_source_literal = StepInputSourceLiteral.from_contract(
                artifact_contract
            )
            param_mapping: dict[str, GeneratedParameterName] = {}
            dead_output_settings = self.generator.pruner.dead_output_setting_names(
                module=module, artifact_contract=artifact_contract
            )
            bound_settings = self.generator.registry.bind_settings(
                module,
                param_mapping=param_mapping,
                ignored_unmapped_settings=dead_output_settings,
            )
            setting_coverage.extend(bound_settings.setting_coverage)
            translated_kwargs = GeneratedStepSettings.from_mapping(
                bound_settings.kwargs
            )
            translated_kwargs = self.generator.pruner.prune_dead_output_setting_kwargs(
                module=module,
                translated_kwargs=translated_kwargs,
                param_mapping=param_mapping,
                artifact_contract=artifact_contract,
            )
            translated_kwargs = translated_kwargs.with_defaults(
                self._compile_time_public_kwargs(
                    module_type,
                    module,
                    source_schema,
                    artifact_flow=artifact_flow,
                    group_key="default",
                    runtime_artifact_contract=(
                        runtime_artifact_contracts.get(module.module_num)
                    ),
                )
            )
            artifact_flow = module_type.compile_time_artifact_flow_after_invocation(
                artifact_flow,
                group_key="default",
                module=module,
            )
            invocation_options_literal = (
                module_type.generated_invocation_options_literal(
                    bound_settings.invocation_options, import_collector=literal_imports
                )
            )
            component_request = ModuleProcessingComponentRequest(
                module_type=module_type,
                function_name=func_name,
                runtime_lineage=RuntimeArtifactLineageScope(
                    artifact_contract,
                    source_lineage.variable_components_for(artifact_contract),
                    source_lineage.requires_pairwise_object_domain_scope_for(
                        artifact_contract
                    ),
                    source_lineage.source_stack_runtime_image_names_for(
                        artifact_contract
                    ),
                ),
                bound_settings=translated_kwargs,
                source_schema=source_schema,
            )
            processing_components = self.generator.registry.processing_components(
                module,
                component_request,
            )
            emissions.append(
                GeneratedStepEmission(
                    module=module,
                    module_type=module_type,
                    step_name=step_name,
                    artifact_contract=artifact_contract,
                    func_name=func_name,
                    translated_kwargs=translated_kwargs,
                    invocation_options_literal=invocation_options_literal,
                    processing_components=processing_components,
                    input_source_literal=input_source_literal,
                )
            )
        emission_groups = self.coalesced_emission_groups(tuple(emissions))
        pipeline_defaults = GeneratedPipelineConfigDefaults.from_emission_groups(
            emission_groups,
            source_schema,
        )
        for emission_group in emission_groups:
            self.emit_function_step(
                lines,
                emission_group,
                import_collector=literal_imports,
                pipeline_defaults=pipeline_defaults,
            )
        lines.append("]")
        if literal_imports:
            import_lines = [
                f"from {module_name} import {symbol_name}"
                for module_name, symbol_name in sorted(literal_imports)
            ]
            return (
                "\n".join((*import_lines, "", *lines)),
                tuple(setting_coverage),
                pipeline_defaults,
            )
        return ("\n".join(lines), tuple(setting_coverage), pipeline_defaults)

    def _compile_time_public_kwargs(
        self,
        module_type: type[CellProfilerModule],
        module: ModuleBlock,
        source_schema: PipelineImageSchema,
        *,
        artifact_flow: CellProfilerCompileTimeArtifactFlow,
        group_key: str,
        runtime_artifact_contract: ModuleArtifactContract | None,
    ) -> dict[GeneratedStepSettingKey, GeneratedLiteralValue]:
        """Return declaration-owned compile-time kwargs for generated source."""
        grouped_values: dict[GeneratedStepSettingKey, list[GeneratedLiteralValue]] = {}
        for record in module_type.compile_time_public_setting_records_for_generation(
            module,
            source_schema,
            artifact_flow=artifact_flow,
            group_key=group_key,
        ):
            key = normalize_cellprofiler_setting_name(record.name)
            value = self.generator.settings_binder.parse_value(record.name, record.value)
            grouped_values.setdefault(key, []).append(value)
        public_kwargs = {
            key: values[0] if len(values) == 1 else tuple(values)
            for key, values in grouped_values.items()
        }
        public_kwargs.update(
            module_type.compile_time_public_kwargs(module, source_schema)
        )
        if runtime_artifact_contract is not None:
            public_kwargs.update(
                module_type.compile_time_public_artifact_materialization_kwargs(
                    module,
                    runtime_artifact_contract,
                )
            )
        return public_kwargs

    def coalesced_emission_groups(
        self,
        emissions: tuple[GeneratedStepEmission, ...],
    ) -> tuple[GeneratedStepEmissionGroup, ...]:
        """Coalesce adjacent emissions into dict-pattern steps when semantics match."""
        groups: list[GeneratedStepEmissionGroup] = []
        current: list[GeneratedStepEmission] = []
        for emission in emissions:
            if current and self.can_coalesce_with_group(tuple(current), emission):
                current.append(emission)
                continue
            if current:
                groups.append(GeneratedStepEmissionGroup(tuple(current)))
            current = [emission]
        if current:
            groups.append(GeneratedStepEmissionGroup(tuple(current)))
        return tuple(groups)

    def can_coalesce_with_group(
        self,
        emissions: tuple[GeneratedStepEmission, ...],
        candidate: GeneratedStepEmission,
    ) -> bool:
        """Return whether ``candidate`` can join an existing generated step group."""
        first = emissions[0]
        candidate_key = self.group_key_for_emission(candidate)
        existing_keys = tuple(self.group_key_for_emission(emission) for emission in emissions)
        return (
            candidate_key is not None
            and all(key is not None for key in existing_keys)
            and candidate_key not in existing_keys
            and candidate.module.name == first.module.name
            and candidate.module_type is first.module_type
            and candidate.step_name == first.step_name
            and candidate.func_name == first.func_name
            and candidate.processing_components == first.processing_components
            and candidate.input_source_literal == first.input_source_literal
            and self.source_binding_config_shape(candidate)
            == self.source_binding_config_shape(first)
        )

    @staticmethod
    def source_binding_config_shape(emission: GeneratedStepEmission):
        """Return source-binding config with per-group bindings removed."""
        return replace(emission.artifact_contract.source_bindings, bindings=())

    @staticmethod
    def group_key_for_emission(emission: GeneratedStepEmission) -> str | None:
        """Return the dict-pattern key declared by source-binding component identity."""
        group_axis = group_by_component_axis(
            emission.processing_components.group_by_component
        )
        if group_axis is None:
            return None
        bindings = emission.artifact_contract.source_bindings.binding_declarations
        matches = tuple(
            selector.value
            for binding in bindings
            for selector in binding.component_identity
            if selector.component is group_axis
        )
        unique_matches = tuple(dict.fromkeys(matches))
        if len(unique_matches) != 1:
            return None
        return str(unique_matches[0])

    def emit_function_step(
        self,
        lines: list[str],
        emission_group: GeneratedStepEmissionGroup,
        *,
        import_collector: set[tuple[str, str]],
        pipeline_defaults: GeneratedPipelineConfigDefaults,
    ) -> None:
        """Append generated source for one FunctionStep emission group."""
        first = emission_group.first
        lines.append("    FunctionStep(")
        for emission in emission_group.emissions:
            lines.extend(self.artifact_contract_comments(emission.artifact_contract))
        self.emit_function_spec(
            lines,
            emission_group,
            import_collector=import_collector,
        )
        lines.append(f'        name="{first.step_name}",')
        if emission_group.is_grouped:
            source_bindings = self.merged_group_source_bindings(emission_group)
            self.emit_source_bindings(
                lines,
                source_bindings,
                import_collector=import_collector,
                pipeline_defaults=pipeline_defaults,
            )
        else:
            self.emit_source_bindings(
                lines,
                first.artifact_contract.source_bindings,
                import_collector=import_collector,
                pipeline_defaults=pipeline_defaults,
            )
        self.emit_processing_config(
            lines,
            first,
            pipeline_defaults=pipeline_defaults,
        )
        lines.append("    ),")

    def emit_source_bindings(
        self,
        lines: list[str],
        source_bindings,
        *,
        import_collector: set[tuple[str, str]],
        pipeline_defaults: GeneratedPipelineConfigDefaults,
    ) -> None:
        """Append sparse step source bindings, inheriting pipeline defaults when exact."""
        if source_bindings.is_empty:
            return
        if self.source_bindings_can_inherit_pipeline_defaults(
            source_bindings,
            pipeline_defaults.source_bindings,
        ):
            return
        lines.append(
            "        source_bindings="
            f"{step_source_bindings_literal(source_bindings, import_collector=import_collector)},"
        )

    @staticmethod
    def source_bindings_can_inherit_pipeline_defaults(
        source_bindings,
        defaults: SourceBindingsConfig | None,
    ) -> bool:
        """Return whether step bindings can be represented by inherited defaults."""
        return defaults is not None and source_bindings.can_inherit_from(defaults)

    def emit_processing_config(
        self,
        lines: list[str],
        emission: GeneratedStepEmission,
        *,
        pipeline_defaults: GeneratedPipelineConfigDefaults,
    ) -> None:
        """Append a sparse LazyProcessingConfig override for one generated step."""

        step_shape = GeneratedProcessingConfigShape.from_emission(emission)
        default_shape = pipeline_defaults.processing
        emit_component_shape = (
            default_shape is None
            or step_shape.without_input_source() != default_shape
        )
        emit_input_source = step_shape.input_source is not None
        if not emit_component_shape and not emit_input_source:
            return

        lines.append("        processing_config=LazyProcessingConfig(")
        if emit_component_shape:
            lines.append(
                "            variable_components=["
                + ", ".join(step_shape.variable_components)
                + "],"
            )
            if step_shape.group_by is not None:
                lines.append(f"            group_by={step_shape.group_by},")
        if emit_input_source:
            lines.append(f"            input_source={step_shape.input_source},")
        lines.append("        ),")

    @staticmethod
    def merged_group_source_bindings(emission_group: GeneratedStepEmissionGroup):
        """Return one source-binding config containing every grouped binding."""
        first_config = emission_group.first.artifact_contract.source_bindings
        bindings_by_alias = {}
        for emission in emission_group.emissions:
            for binding in emission.artifact_contract.source_bindings.binding_declarations:
                bindings_by_alias[binding.alias] = binding
        return replace(
            first_config,
            bindings=tuple(bindings_by_alias.values()),
        )

    def emit_function_spec(
        self,
        lines: list[str],
        emission_group: GeneratedStepEmissionGroup,
        *,
        import_collector: set[tuple[str, str]],
    ) -> None:
        """Append the ``func=`` source for one generated FunctionStep."""
        if emission_group.public_function_spec_is_grouped:
            lines.append("        func={")
            for emission in emission_group.emissions:
                group_key = self.group_key_for_emission(emission)
                if group_key is None:
                    raise ValueError(
                        f"Generated step {emission.step_name!r} cannot emit a "
                        "dict-pattern item without a source-binding group key."
                    )
                lines.append(
                    f"            {group_key!r}: "
                    f"{self.compact_function_item_literal(emission, import_collector=import_collector)},"
                )
            lines.append("        },")
            return
        self.emit_single_function_spec(
            lines,
            emission_group.first,
            import_collector=import_collector,
            settings=(
                GeneratedStepEmissionGroup.runtime_function_settings(
                    emission_group.first
                ).with_defaults(
                    dict(
                        GeneratedStepEmissionGroup.coalesced_public_settings(
                            emission_group.emissions
                        ).items()
                    )
                )
                if emission_group.is_grouped
                else None
            ),
        )

    def emit_single_function_spec(
        self,
        lines: list[str],
        emission: GeneratedStepEmission,
        *,
        import_collector: set[tuple[str, str]],
        settings: GeneratedStepSettings | None = None,
    ) -> None:
        """Append the legacy single-item ``func=`` source."""
        emitted_settings = settings if settings is not None else emission.translated_kwargs
        if emitted_settings:
            kwargs_str = self.multiline_kwargs_literal(
                emitted_settings,
                import_collector=import_collector,
            )
            if emission.invocation_options_literal is None:
                lines.append(f"        func=({emission.func_name}, {kwargs_str}),")
            else:
                lines.append(
                    f"        func=({emission.func_name}, {kwargs_str}, {emission.invocation_options_literal}),"
                )
        elif emission.invocation_options_literal is None:
            lines.append(f"        func={emission.func_name},")
        else:
            lines.append(
                f"        func=({emission.func_name}, {{}}, {emission.invocation_options_literal}),"
            )

    def compact_function_item_literal(
        self,
        emission: GeneratedStepEmission,
        *,
        import_collector: set[tuple[str, str]],
    ) -> str:
        """Return a compact callable-item literal for dict-pattern values."""
        if not emission.translated_kwargs and emission.invocation_options_literal is None:
            return emission.func_name
        kwargs = self.compact_kwargs_literal(
            emission.translated_kwargs,
            import_collector=import_collector,
        )
        if emission.invocation_options_literal is None:
            return f"({emission.func_name}, {kwargs})"
        return f"({emission.func_name}, {kwargs}, {emission.invocation_options_literal})"

    def compact_kwargs_literal(
        self,
        settings: GeneratedStepSettings,
        *,
        import_collector: set[tuple[str, str]],
    ) -> str:
        """Return a compact kwargs mapping literal."""
        if not settings:
            return "{}"
        items = tuple(
            f"{self.kwarg_key_literal(key)}: {python_literal(value, import_collector=import_collector)}"
            for key, value in settings.items()
        )
        return "{" + ", ".join(items) + "}"

    def multiline_kwargs_literal(
        self,
        settings: GeneratedStepSettings,
        *,
        import_collector: set[tuple[str, str]],
    ) -> str:
        """Return the existing multi-line kwargs mapping literal."""
        kwargs_lines = ["{"]
        for key, value in settings.items():
            kwargs_lines.append(
                "            "
                f"{self.kwarg_key_literal(key)}: "
                f"{python_literal(value, import_collector=import_collector)},"
            )
        kwargs_lines.append("        }")
        return "\n".join(kwargs_lines)

    @staticmethod
    def kwarg_key_literal(key: GeneratedStepSettingKey) -> str:
        """Render generated source for a CellProfiler step kwarg key."""
        return repr(key)

    @staticmethod
    def backend_function_import_block(function_names: Iterable[str]) -> str:
        """Return imports for the absorbed backend functions used by the pipeline."""
        unique_function_names = tuple(dict.fromkeys(sorted(function_names)))
        if not unique_function_names:
            return ""
        lines = [
            "from openhcs.processing.backends.cellprofiler import CellProfilerFunctionCatalog"
        ]
        lines.extend(
            (
                f"{function_name} = CellProfilerFunctionCatalog.get_function({function_name!r})"
                for function_name in unique_function_names
            )
        )
        lines.append("")
        return "\n".join(lines)

    def artifact_contract_comments(
        self, contract: ModuleArtifactContracts
    ) -> list[str]:
        """Return generated comments summarizing artifact contract semantics."""
        return ArtifactContractCommentSection.lines_for(contract)


@dataclass(frozen=True)
class PipelineGeneratorBuildStage:
    """Top-level CellProfiler module partitioning and generated-pipeline assembly."""

    generator: "PipelineGenerator"

    def generate(
        self, request: GeneratedPipelineRequest, modules: List[ModuleBlock]
    ) -> GeneratedPipeline:
        """Generate pipeline using absorbed library (instant, no LLM)."""
        skipped_modules = list(request.skipped_modules)
        registry_modules = []
        missing_modules = []
        for module in modules:
            if self.generator.registry.has_module(module.name):
                module_type = self.generator.registry.required_module_class(module.name)
                registry_modules.extend(module_type.generated_module_blocks(module))
            else:
                missing_modules.append(module)
                logger.warning(f"Module {module.name} not in absorbed library")
        imports = self.generator.IMPORTS_BASE.format(
            source_file=request.source_cppipe.name
        )
        if skipped_modules:
            skip_note = "\n# Skipped infrastructure modules (handled by OpenHCS):\n"
            for module in skipped_modules:
                skip_note += (
                    f"#   - {cellprofiler_infrastructure_import_note(module.name)}\n"
                )
            imports += skip_note + "\n"
        if missing_modules:
            raise ValueError(
                f"Missing {len(missing_modules)} modules from absorbed library: {[m.name for m in missing_modules]}. Re-run absorption with --force to regenerate."
            )
        ordered_modules = [*skipped_modules, *registry_modules]
        symbol_table = CellProfilerSymbolTable.compile(ordered_modules)
        contracts_by_module = {
            module.module_num: symbol_table.contract_for(module)
            for module in registry_modules
        }
        infrastructure_contracts = tuple(
            (symbol_table.contract_for(module) for module in skipped_modules)
        )
        infrastructure_retained_artifacts = (
            frozenset(
                (
                    artifact
                    for module in skipped_modules
                    for artifact in cellprofiler_infrastructure_retained_artifacts(
                        module,
                        contracts_by_module_num=symbol_table.contracts_by_module_num,
                    )
                )
            )
            if request.materialize_skipped_save_images
            else frozenset()
        )
        infrastructure_input_artifacts = {
            ArtifactSpecKey.from_spec(input_spec)
            for contract in infrastructure_contracts
            for input_spec in (*contract.inputs, *contract.runtime_artifact_inputs)
        }
        terminal_image_artifacts = (
            self.generator.pruner.terminal_image_artifacts(
                registry_modules,
                contracts_by_module,
                external_consumers=(
                    input_spec
                    for contract in infrastructure_contracts
                    for input_spec in (
                        *contract.inputs,
                        *contract.runtime_artifact_inputs,
                    )
                ),
            )
            if request.materialize_terminal_images
            else frozenset()
        )
        externally_materialized_outputs = (
            infrastructure_retained_artifacts | terminal_image_artifacts
        )
        artifact_name_materialized_outputs = infrastructure_retained_artifacts
        executable_modules = (
            self.generator.pruner.prune_dead_unmaterialized_artifact_steps(
                registry_modules,
                contracts_by_module,
                externally_required_artifacts=infrastructure_input_artifacts
                | externally_materialized_outputs,
            )
            if request.prune_dead_unmaterialized_artifact_steps
            else registry_modules
        )
        runtime_module_contracts_by_module = (
            self.generator.runtime_contracts.by_module_num(
                executable_modules,
                contracts_by_module,
                externally_materialized_outputs=externally_materialized_outputs,
                artifact_name_materialized_outputs=artifact_name_materialized_outputs,
            )
        )
        source_lineage = RuntimeArtifactSourceLineage(
            contracts_by_module,
            symbol_table.source_schema,
        )
        function_names_by_module: dict[int, str] = {}
        for module in executable_modules:
            module_record = self.generator.registry.module_record(module.name)
            module_type = self.generator.registry.required_module_class(module.name)
            artifact_contract = contracts_by_module[module.module_num]
            function_names_by_module[module.module_num] = (
                self.generator.registry.resolve_function(
                    module,
                    request=ModuleProcessingComponentRequest(
                        module_type=module_type,
                        function_name=module_record.function_name,
                        runtime_lineage=RuntimeArtifactLineageScope(
                            artifact_contract,
                            source_lineage.variable_components_for(artifact_contract),
                            source_lineage.requires_pairwise_object_domain_scope_for(
                                artifact_contract
                            ),
                            source_lineage.source_stack_runtime_image_names_for(
                                artifact_contract
                            ),
                        ),
                        bound_settings=GeneratedStepSettings(),
                        source_schema=symbol_table.source_schema,
                    ),
                )
            )
        if executable_modules:
            imports += "# Absorbed CellProfiler functions\n"
            imports += self.generator.emitter.backend_function_import_block(
                function_names_by_module.values()
            )
        (
            steps,
            setting_coverage,
            pipeline_defaults,
        ) = self.generator.emitter.generate_steps_from_registry(
            executable_modules,
            function_names_by_module,
            contracts_by_module,
            runtime_module_contracts_by_module,
            symbol_table.source_schema,
        )
        pipeline_config = self._pipeline_config(
            symbol_table.source_schema,
            pipeline_defaults=pipeline_defaults,
        )
        code = imports + steps
        return GeneratedPipeline(
            name=request.pipeline_name,
            code=code,
            source_cppipe=str(request.source_cppipe),
            converted_modules=[m.name for m in executable_modules],
            failed_modules=[m.name for m in missing_modules],
            artifact_contracts=tuple(
                (
                    contracts_by_module[module.module_num]
                    for module in executable_modules
                )
            ),
            runtime_module_contracts=tuple(
                (
                    (
                        module.module_num,
                        runtime_module_contracts_by_module[module.module_num],
                    )
                    for module in executable_modules
                    if module.module_num in runtime_module_contracts_by_module
                )
            ),
            source_schema=symbol_table.source_schema,
            pipeline_config=pipeline_config,
            setting_coverage=setting_coverage,
        )

    @staticmethod
    def _pipeline_config(
        source_schema: PipelineImageSchema,
        *,
        pipeline_defaults: GeneratedPipelineConfigDefaults,
    ) -> PipelineConfig:
        """Return ObjectState-owned pipeline config derived from source schema."""
        default_kwargs = pipeline_defaults.pipeline_config_kwargs()
        if source_schema.is_empty:
            return PipelineConfig(**default_kwargs)
        source_bindings_config = source_schema.to_runtime_source_bindings_config()
        if source_bindings_config.is_empty:
            return PipelineConfig(**default_kwargs)
        lazy_source_bindings_config = (
            GeneratedPipelineConfigDefaults._lazy_source_bindings_config(
                source_bindings_config
            )
        )
        if PipelineImageSchemaSourceBindingsRepresentability(
            source_schema
        ).unsupported_fields():
            return PipelineConfig(
                source_bindings_config=lazy_source_bindings_config,
                **default_kwargs,
            )
        return PipelineConfig(
            microscope=Microscope.SOURCE_BINDINGS,
            source_bindings_config=lazy_source_bindings_config,
            **default_kwargs,
        )

class PipelineGenerator:
    """
    Generate complete OpenHCS pipeline from converted functions.

    TWO MODES:
    1. Registry-based: Uses pre-absorbed cellprofiler_library (instant, no LLM)
    2. LLM-based: Inline function definitions (fallback for unabsorbed modules)

    Creates a runnable pipeline file with:
    1. Standard imports (+ registry imports if using absorbed library)
    2. Converted function definitions (only for non-registry functions)
    3. FunctionStep wrappers for each function
    4. pipeline_steps list
    """

    IMPORTS_BASE = '"""\nOpenHCS Pipeline - Converted from CellProfiler\nSource: {source_file}\n\nAuto-generated by CellProfiler to OpenHCS converter.\n"""\n\nimport numpy as np\nfrom typing import Tuple, List, Optional, Dict, Any\nfrom dataclasses import dataclass\nfrom enum import Enum\n\n# OpenHCS imports\nfrom openhcs.core.artifact_materialization_policy import NO_ARTIFACT_MATERIALIZATION\nfrom openhcs.core.artifacts import ArtifactSidecarRole, ArtifactSpec\nfrom openhcs.core.steps.function_step import FunctionStep\nfrom openhcs.core.source_bindings import (\n    ComponentSelector,\n    EMPTY_SOURCE_BINDINGS,\n    MetadataExtractionRule,\n    MetadataSource,\n    MetadataSelector,\n    NamedSourceBinding,\n    SourceBindingMatchDimension,\n    SourceBindingMatchField,\n    SourceBindingMatchMethod,\n    SourceBindingMatchPlan,\n    SourceBindingOrigin,\n    SourceFilterClause,\n    SourceFilterMatchType,\n    SourceFilterSubject,\n    SourceSelector,\n)\nfrom openhcs.core.config import LazyProcessingConfig, LazySourceBindingsConfig, LazyStepSourceBindingsConfig\nfrom openhcs.constants.constants import VariableComponents, GroupBy\nfrom openhcs.constants.constants import AllComponents\nfrom openhcs.constants.input_source import InputSource\nfrom openhcs.interop.cellprofiler.measurement_scope import CellProfilerMeasurementTargetScope\nfrom openhcs.processing.backends.cellprofiler._backend import CellProfilerBackendProvider\nfrom openhcs.processing.materialization import MaterializedFilenameIdentity, tiff_stack\n\n'

    def __init__(self, library_root: Optional[Path] = None):
        """
        Initialize generator.

        Args:
            library_root: Path to absorbed cellprofiler_library
        """
        self._explicit_library_root = library_root is not None
        self.library_root = (
            library_root or Path(__file__).parent.parent / "cellprofiler_library"
        )
        self.settings_binder = SettingsBinder()
        self.registry = PipelineGeneratorRegistryStage(self)
        self.pruner = PipelineGeneratorArtifactPruner(self)
        self.runtime_contracts = PipelineGeneratorRuntimeContractProjector(self)
        self.emitter = PipelineGeneratorCodeEmitter(self)
        self.builder = PipelineGeneratorBuildStage(self)
        self._registry = self.registry.load_registry()

    def has_module(self, module_name: str) -> bool:
        """Check if module exists in absorbed library."""
        return self.registry.has_module(module_name)

    def generate_from_registry(
        self,
        pipeline_name: str,
        source_cppipe: Path,
        modules: List[ModuleBlock],
        skipped_modules: Optional[List[ModuleBlock]] = None,
        prune_dead_unmaterialized_artifact_steps: bool = False,
        materialize_skipped_save_images: bool = True,
        materialize_terminal_images: bool = True,
    ) -> GeneratedPipeline:
        """
        Generate pipeline using absorbed library (instant, no LLM).

        Args:
            pipeline_name: Name for the generated pipeline
            source_cppipe: Path to source .cppipe file
            modules: ModuleBlocks from .cppipe parser (processing modules only)
            skipped_modules: Infrastructure modules that were skipped

        Returns:
            GeneratedPipeline using registry functions
        """
        return self.builder.generate(
            GeneratedPipelineRequest.from_public_args(
                pipeline_name=pipeline_name,
                source_cppipe=source_cppipe,
                skipped_modules=skipped_modules,
                prune_dead_unmaterialized_artifact_steps=prune_dead_unmaterialized_artifact_steps,
                materialize_skipped_save_images=materialize_skipped_save_images,
                materialize_terminal_images=materialize_terminal_images,
            ),
            modules=modules,
        )


def python_literal(
    value: GeneratedLiteralValue,
    *,
    import_collector: GeneratedImportCollector | None = None,
) -> str:
    """Render a deterministic generated-code literal for bound setting values."""
    if isinstance(value, PythonSourceLiteral):
        if import_collector is not None:
            import_collector.update(value.source_literal_imports())
        return value.source_literal()
    if isinstance(value, Enum):
        enum_type = type(value)
        if "<locals>" in enum_type.__qualname__:
            return repr(value)
        root_name, _, nested_path = enum_type.__qualname__.partition(".")
        if import_collector is not None:
            import_collector.add((enum_type.__module__, root_name))
        enum_reference = f"{root_name}.{nested_path}" if nested_path else root_name
        return f"{enum_reference}.{value.name}"
    if isinstance(value, tuple):
        trailing_comma = ""
        if len(value) == 1:
            trailing_comma = ","
        return (
            "("
            + ", ".join(
                (
                    python_literal(item, import_collector=import_collector)
                    for item in value
                )
            )
            + trailing_comma
            + ")"
        )
    if isinstance(value, list):
        return (
            "["
            + ", ".join(
                (
                    python_literal(item, import_collector=import_collector)
                    for item in value
                )
            )
            + "]"
        )
    if isinstance(value, dict):
        return (
            "{"
            + ", ".join(
                (
                    f"{python_literal(key, import_collector=import_collector)}: "
                    f"{python_literal(item, import_collector=import_collector)}"
                    for key, item in value.items()
                )
            )
            + "}"
        )
    return repr(value)
