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
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import ClassVar, List, Optional, TypeAlias
from metaclass_registry import AutoRegisterMeta
from openhcs.constants import Backend
from openhcs.constants.constants import Microscope
from openhcs.core.artifact_materialization_policy import NO_ARTIFACT_MATERIALIZATION
from openhcs.core.artifact_observability import externally_required_artifact_outputs
from openhcs.core.artifacts import (
    ArtifactSpec,
    ArtifactType,
    SpecialArtifactType,
    ImageArtifactType,
)
from openhcs.core.config import PipelineConfig
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
from openhcs.core.vfs_protocol import FileManagerLike
from openhcs.interop.cellprofiler.module_roles import (
    ArtifactSpecKey,
    cellprofiler_infrastructure_import_note,
    cellprofiler_infrastructure_retained_artifacts,
)
from openhcs.interop.cellprofiler.parser import ModuleBlock
from openhcs.interop.cellprofiler.measurement_scope import (
    CellProfilerMeasurementTargetScope,
)
from openhcs.processing.backends.cellprofiler._backend import (
    CellProfilerBackendProvider,
)
from openhcs.interop.cellprofiler.artifact_semantics import artifact_setting_symbols
from openhcs.interop.cellprofiler.settings_binder import (
    SettingsBinder,
    normalize_cellprofiler_setting_name,
)
from openhcs.interop.cellprofiler.setting_names import SettingNameFamily
from openhcs.processing.backends.cellprofiler.library import canonical_module_name
from openhcs.interop.cellprofiler.module_declarations import (
    BoundModuleSettings,
    CellProfilerModule,
    ModuleSettingCoverageRecord,
)
from openhcs.processing.materialization import MaterializedFilenameIdentity, tiff_stack
from openhcs.interop.cellprofiler.symbol_table import (
    CellProfilerSymbolTable,
    ModuleArtifactContracts,
)
from openhcs.interop.cellprofiler.module_processing_components import (
    GeneratedLiteralValue,
    GeneratedParameterName,
    GeneratedStepSettings,
    ModuleProcessingComponentRequest,
    RuntimeArtifactLineageScope,
    RuntimeArtifactSourceLineage,
    generated_function_step_semantic_argument_lines,
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
    pipeline_config: PipelineConfig | None = None
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


@dataclass(frozen=True)
class PipelineGeneratorCodeEmitter:
    """Generated-code emission for imports, FunctionStep declarations, and comments."""

    generator: "PipelineGenerator"

    def generate_steps_from_registry(
        self,
        modules: List[ModuleBlock],
        function_names_by_module: Mapping[int, str],
        artifact_contracts: dict[int, ModuleArtifactContracts],
        source_schema: PipelineImageSchema,
    ) -> tuple[str, tuple[ModuleSettingCoverageRecord, ...]]:
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
                ),
                bound_settings=translated_kwargs,
                source_schema=source_schema,
            )
            translated_kwargs = translated_kwargs.with_defaults(
                component_request.generated_semantic_control_defaults()
            )
            component_request = replace(
                component_request,
                bound_settings=translated_kwargs,
            )
            processing_components = self.generator.registry.processing_components(
                module,
                component_request,
            )
            lines.append("    FunctionStep(")
            lines.extend(self.artifact_contract_comments(artifact_contract))
            if translated_kwargs:
                kwargs_lines = ["{"]
                for k, v in translated_kwargs.items():
                    kwargs_lines.append(f"            {repr(k)}: {python_literal(v)},")
                kwargs_lines.append("        }")
                kwargs_str = "\n".join(kwargs_lines)
                if invocation_options_literal is None:
                    lines.append(f"        func=({func_name}, {kwargs_str}),")
                else:
                    lines.append(
                        f"        func=({func_name}, {kwargs_str}, {invocation_options_literal}),"
                    )
            elif invocation_options_literal is None:
                lines.append(f"        func={func_name},")
            else:
                lines.append(
                    f"        func=({func_name}, {{}}, {invocation_options_literal}),"
                )
            lines.append(f'        name="{step_name}",')
            lines.extend(
                generated_function_step_semantic_argument_lines(
                    processing_components=processing_components,
                    artifact_contract=artifact_contract,
                    import_collector=literal_imports,
                )
            )
            lines.append("        processing_config=LazyProcessingConfig(")
            lines.append(
                "            variable_components=["
                + ", ".join(processing_components.variable_component_literals)
                + "],"
            )
            group_by_literal = processing_components.group_by_literal
            if group_by_literal is not None:
                lines.append(f"            group_by={group_by_literal},")
            input_source_literal.append_to(lines)
            lines.append("        ),")
            lines.append("    ),")
        lines.append("]")
        if literal_imports:
            import_lines = [
                f"from {module_name} import {symbol_name}"
                for module_name, symbol_name in sorted(literal_imports)
            ]
            return ("\n".join((*import_lines, "", *lines)), tuple(setting_coverage))
        return ("\n".join(lines), tuple(setting_coverage))

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
                registry_modules.append(module)
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
        pipeline_config = self._pipeline_config(symbol_table.source_schema)
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
        steps, setting_coverage = self.generator.emitter.generate_steps_from_registry(
            executable_modules,
            function_names_by_module,
            contracts_by_module,
            symbol_table.source_schema,
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
    def _pipeline_config(source_schema: PipelineImageSchema) -> PipelineConfig | None:
        """Return ObjectState-owned pipeline config derived from source schema."""
        if source_schema.is_empty:
            return None
        source_bindings_config = source_schema.to_runtime_source_bindings_config()
        if source_bindings_config.is_empty:
            return None
        if PipelineImageSchemaSourceBindingsRepresentability(
            source_schema
        ).unsupported_fields():
            return PipelineConfig(source_bindings_config=source_bindings_config)
        return PipelineConfig(
            microscope=Microscope.SOURCE_BINDINGS,
            source_bindings_config=source_bindings_config,
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

    IMPORTS_BASE = '"""\nOpenHCS Pipeline - Converted from CellProfiler\nSource: {source_file}\n\nAuto-generated by CellProfiler to OpenHCS converter.\n"""\n\nimport numpy as np\nfrom typing import Tuple, List, Optional, Dict, Any\nfrom dataclasses import dataclass\nfrom enum import Enum\n\n# OpenHCS imports\nfrom openhcs.core.artifact_materialization_policy import NO_ARTIFACT_MATERIALIZATION\nfrom openhcs.core.artifacts import ArtifactSidecarRole, ArtifactSpec\nfrom openhcs.core.steps.function_step import FunctionStep\nfrom openhcs.core.source_bindings import (\n    ComponentSelector,\n    EMPTY_SOURCE_BINDINGS,\n    MetadataExtractionRule,\n    MetadataSource,\n    MetadataSelector,\n    NamedSourceBinding,\n    SourceBindingMatchDimension,\n    SourceBindingMatchField,\n    SourceBindingMatchMethod,\n    SourceBindingMatchPlan,\n    SourceBindingOrigin,\n    SourceFilterClause,\n    SourceFilterMatchType,\n    SourceFilterSubject,\n    SourceSelector,\n)\nfrom openhcs.core.config import LazyProcessingConfig, LazyStepSourceBindingsConfig\nfrom openhcs.constants.constants import VariableComponents, GroupBy\nfrom openhcs.constants.constants import AllComponents\nfrom openhcs.constants.input_source import InputSource\nfrom openhcs.interop.cellprofiler.measurement_scope import CellProfilerMeasurementTargetScope\nfrom openhcs.processing.backends.cellprofiler._backend import CellProfilerBackendProvider\nfrom openhcs.processing.materialization import MaterializedFilenameIdentity, tiff_stack\n\n'

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


def python_literal(value: GeneratedLiteralValue) -> str:
    """Render a deterministic generated-code literal for bound setting values."""
    if isinstance(value, CellProfilerMeasurementTargetScope):
        return f"CellProfilerMeasurementTargetScope.{value.name}"
    if isinstance(value, CellProfilerBackendProvider):
        return f"CellProfilerBackendProvider.{value.name}"
    if isinstance(value, Enum):
        return repr(value.value)
    if isinstance(value, tuple):
        trailing_comma = ""
        if len(value) == 1:
            trailing_comma = ","
        return (
            "("
            + ", ".join((python_literal(item) for item in value))
            + trailing_comma
            + ")"
        )
    if isinstance(value, list):
        return "[" + ", ".join((python_literal(item) for item in value)) + "]"
    if isinstance(value, dict):
        return (
            "{"
            + ", ".join(
                (
                    f"{python_literal(key)}: {python_literal(item)}"
                    for key, item in value.items()
                )
            )
            + "}"
        )
    return repr(value)
