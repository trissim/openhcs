"""
PipelineGenerator - Generate complete runnable OpenHCS pipelines.

DETERMINISTIC ONLY:
Uses pre-absorbed cellprofiler_library. No LLM fallback.
Fails loudly if modules are missing from the absorbed library.

Takes parsed .cppipe modules and generates a complete pipeline file with:
- All imports
- Function references from absorbed library
- FunctionStep wrappers with correct variable_components (from LLM-inferred category)
- Pipeline configuration
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Iterable, List, Mapping, Optional

from metaclass_registry import AutoRegisterMeta
from openhcs.constants import Backend
from openhcs.core.artifact_materialization_policy import (
    DEFAULT_ARTIFACT_MATERIALIZATION_RULES,
    NO_ARTIFACT_MATERIALIZATION,
)
from openhcs.core.artifact_observability import externally_required_artifact_outputs
from openhcs.core.artifacts import ArtifactKind, ArtifactSpec
from openhcs.core.module_artifact_contract import ModuleArtifactContract
from openhcs.core.pipeline_image_schema import PipelineImageSchema
from openhcs.core.runtime_invocation import RuntimeInvocationOptions
from openhcs.core.vfs_protocol import FileManagerLike
from openhcs.interop.cellprofiler.runtime import (
    CellProfilerGridCycleScope,
    CellProfilerInvocationOptions,
)
from openhcs.interop.cellprofiler.module_roles import (
    cellprofiler_infrastructure_import_note,
)
from openhcs.interop.cellprofiler.parser import ModuleBlock

from openhcs.interop.cellprofiler.illumination_settings import (
    IlluminationCalculationScope,
)
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum

from openhcs.interop.cellprofiler.artifact_semantics import artifact_setting_symbols
from openhcs.interop.cellprofiler.module_function_resolution import (
    _ModuleFunctionResolutionStrategy,
)
from openhcs.interop.cellprofiler.module_settings_binding import (
    ModuleSettingCoverageRecord,
    _ModuleSettingsBindingStrategy,
)
from openhcs.interop.cellprofiler.settings_binder import (
    SettingsBinder,
    normalize_cellprofiler_setting_name,
)
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    setting_values,
    split_symbol_names,
)
from openhcs.processing.backends.cellprofiler.library import (
    canonical_module_name,
    validated_contracts,
)
from openhcs.processing.materialization import tiff_stack
from openhcs.interop.cellprofiler.symbol_table import (
    CellProfilerSymbolTable,
    ModuleArtifactContracts,
    source_bindings_literal,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArtifactSpecKey:
    """Scope-free artifact identity used while pruning generated CP steps."""

    kind: ArtifactKind
    name: str

    @classmethod
    def from_spec(cls, spec: ArtifactSpec) -> ArtifactSpecKey:
        return cls(kind=spec.kind, name=spec.name)


ExternallyMaterializedOutputs = frozenset[ArtifactSpecKey]


@dataclass(frozen=True)
class AbsorbedModuleMetadata:
    """Validated absorbed-library metadata needed by generated pipelines."""

    function_name: str
    contract: str = "pure_2d"
    category: str = "image_operation"
    confidence: float = 0.5

    @classmethod
    def from_registry_record(
        cls,
        info: Mapping[str, Any],
    ) -> AbsorbedModuleMetadata:
        return cls(
            function_name=str(info["function_name"]),
            contract=str(info.get("contract", "pure_2d")),
            category=str(info.get("category", "image_operation")),
            confidence=float(info.get("confidence", 0.5)),
        )


_INPUTLESS_ARTIFACT_ONLY_KINDS = frozenset(
    {
        ArtifactKind.MEASUREMENTS,
        ArtifactKind.RELATIONSHIPS,
    }
)
_SAVE_IMAGES_SOURCE_IMAGE_SETTING = SettingNameFamily("Select the image to save")


@dataclass(frozen=True, slots=True)
class ModuleProcessingComponents:
    """Generated OpenHCS processing-component literals for one module."""

    variable_components: tuple[str, ...]
    group_by_literal: str | None = None


class ModuleProcessingComponentStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal family for lowering module runtime-scope semantics."""

    __registry_key__ = "module_name"
    module_name: ClassVar[str]

    @classmethod
    def for_module(cls, module_name: str) -> "ModuleProcessingComponentStrategy":
        canonical_name = canonical_module_name(module_name)
        strategy_type = cls.__registry__.get(canonical_name)
        if strategy_type is None:
            return DefaultModuleProcessingComponentStrategy()
        return strategy_type()

    @abstractmethod
    def components(
        self,
        *,
        category: str,
        contract: ModuleArtifactContracts,
        bound_kwargs: Mapping[str, Any],
        category_defaults: Mapping[str, tuple[str, ...]],
    ) -> ModuleProcessingComponents:
        """Return generated processing-component literals for this module."""


class DefaultModuleProcessingComponentStrategy(ModuleProcessingComponentStrategy):
    """Default conversion from source bindings/contracts to OpenHCS runtime scope."""

    module_name = "__default__"

    def components(
        self,
        *,
        category: str,
        contract: ModuleArtifactContracts,
        bound_kwargs: Mapping[str, Any],
        category_defaults: Mapping[str, tuple[str, ...]],
    ) -> ModuleProcessingComponents:
        del bound_kwargs
        if canonical_module_name(contract.module_name) == "TrackObjects":
            return ModuleProcessingComponents(
                (
                    "VariableComponents.SITE",
                    "VariableComponents.CHANNEL",
                ),
                "GroupBy.NONE",
            )
        source_bindings = contract.source_bindings
        if not source_bindings.is_empty:
            if source_bindings.requires_step_input_channel_stack:
                return ModuleProcessingComponents(
                    ("VariableComponents.CHANNEL",),
                    "GroupBy.SITE",
                )
            if source_bindings.requires_pipeline_start_resolution:
                return ModuleProcessingComponents(
                    ("VariableComponents.CHANNEL",),
                    "GroupBy.SITE",
                )
            return ModuleProcessingComponents(("VariableComponents.SITE",))
        if contract.runtime_artifact_inputs:
            return ModuleProcessingComponents(("VariableComponents.SITE",), "GroupBy.NONE")
        if _is_inputless_artifact_only_contract(contract):
            return ModuleProcessingComponents(("VariableComponents.SITE",), "GroupBy.NONE")
        return ModuleProcessingComponents(
            tuple(
                category_defaults.get(
                    category,
                    ("VariableComponents.SITE",),
                )
            )
        )


class CorrectIlluminationCalculateProcessingComponentStrategy(
    DefaultModuleProcessingComponentStrategy
):
    """Lower CellProfiler all-image illumination scope to a site stack per channel."""

    module_name = "CorrectIlluminationCalculate"

    def components(
        self,
        *,
        category: str,
        contract: ModuleArtifactContracts,
        bound_kwargs: Mapping[str, Any],
        category_defaults: Mapping[str, tuple[str, ...]],
    ) -> ModuleProcessingComponents:
        raw_scope = bound_kwargs.get(
            "calculation_scope",
            IlluminationCalculationScope.EACH,
        )
        scope = coerce_cellprofiler_enum(IlluminationCalculationScope, raw_scope)
        if scope.requires_channel_grouping:
            return ModuleProcessingComponents(
                ("VariableComponents.SITE",),
                "GroupBy.CHANNEL",
            )
        return super().components(
            category=category,
            contract=contract,
            bound_kwargs=bound_kwargs,
            category_defaults=category_defaults,
        )


def _is_inputless_artifact_only_contract(contract: ModuleArtifactContracts) -> bool:
    """Return whether a step should execute once per axis, not per image channel."""
    return (
        not contract.inputs
        and not contract.runtime_artifact_inputs
        and bool(contract.outputs)
        and all(spec.kind in _INPUTLESS_ARTIFACT_ONLY_KINDS for spec in contract.outputs)
    )


def _has_materialized_output(contract: ModuleArtifactContracts) -> bool:
    """Return whether any output is externally observable by artifact policy."""
    return any(
        spec.materialization is not None
        or spec.kind in DEFAULT_ARTIFACT_MATERIALIZATION_RULES
        for spec in contract.outputs
    )


def _save_images_required_artifacts(
    skipped_modules: list[ModuleBlock],
) -> set[ArtifactSpecKey]:
    """Return image artifacts required by skipped CellProfiler SaveImages modules."""
    return {
        ArtifactSpecKey(ArtifactKind.IMAGE, image_name)
        for module in skipped_modules
        if module.name == "SaveImages"
        for value in setting_values(module, _SAVE_IMAGES_SOURCE_IMAGE_SETTING)
        for image_name in split_symbol_names(value)
    }


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
                "GeneratedPipeline.save backend must be Backend, "
                f"got {type(backend).__name__}."
            )
        if filemanager is None:
            output_path.write_text(self.code)
        else:
            filemanager.ensure_directory(str(output_path.parent), backend.value)
            filemanager.save(self.code, str(output_path), backend.value)
        logger.info(f"Saved pipeline to {output_path}")


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
            skipped_modules=tuple(skipped_modules or ()),
            prune_dead_unmaterialized_artifact_steps=(
                prune_dead_unmaterialized_artifact_steps
            ),
            materialize_skipped_save_images=materialize_skipped_save_images,
            materialize_terminal_images=materialize_terminal_images,
        )


@dataclass(frozen=True)
class PipelineGeneratorRegistryStage:
    """Absorbed-library registry loading and module metadata lookup."""

    generator: Any

    def load_registry(self) -> dict[str, AbsorbedModuleMetadata]:
        """Load full module metadata from the OpenHCS-owned absorbed catalog."""
        if self.generator._explicit_library_root:
            return self.load_legacy_registry(self.generator.library_root)

        try:
            registry = {
                module_name: AbsorbedModuleMetadata.from_registry_record(info)
                for module_name, info in validated_contracts().items()
            }
            logger.info(f"Loaded {len(registry)} absorbed functions from registry")
            return registry
        except Exception as e:
            raise RuntimeError(f"Failed to load registry: {e}")

    def load_legacy_registry(
        self,
        library_root: Path,
    ) -> dict[str, AbsorbedModuleMetadata]:
        """Load metadata from an explicit maintenance-time absorbed-library root."""
        contracts_file = library_root / "contracts.json"
        if not contracts_file.exists():
            raise FileNotFoundError(
                f"No absorbed library found at {contracts_file}. "
                "Run 'python -m benchmark.converter.absorb' first."
            )

        try:
            data = json.loads(contracts_file.read_text())
            return {
                module_name: AbsorbedModuleMetadata.from_registry_record(info)
                for module_name, info in data.items()
                if info.get("validated", False)
            }
        except Exception as e:
            raise RuntimeError(f"Failed to load registry: {e}")

    def has_module(self, module_name: str) -> bool:
        """Check if module exists in absorbed library."""
        return canonical_module_name(module_name) in self.generator._registry

    def module_metadata(self, module_name: str) -> AbsorbedModuleMetadata:
        """Return absorbed metadata for a module after canonical name resolution."""
        return self.generator._registry[canonical_module_name(module_name)]


@dataclass(frozen=True)
class PipelineGeneratorArtifactPruner:
    """Dead-artifact pruning and setting-pruning authority."""

    generator: Any

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
            if (
                output.materialization is not None
                or output.kind in DEFAULT_ARTIFACT_MATERIALIZATION_RULES
            )
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
                output
                for output in contract.outputs
                if (
                    output.materialization is not None
                    or output.kind in DEFAULT_ARTIFACT_MATERIALIZATION_RULES
                    or ArtifactSpecKey.from_spec(output) in live_artifacts
                )
            )
            artifact_contracts[module.module_num] = replace(
                contract,
                output_symbols=tuple(
                    symbol
                    for symbol in contract.output_symbols
                    if symbol.artifact_spec() in retained_outputs
                ),
            )
            live_artifacts.update(
                ArtifactSpecKey.from_spec(input_spec)
                for input_spec in contract.runtime_artifact_inputs
            )

        pruned = [module for module in modules if module.module_num in live_module_nums]
        skipped = [module for module in modules if module.module_num not in live_module_nums]
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
        translated_kwargs: dict[str, Any],
        param_mapping: Mapping[str, Any],
        artifact_contract: ModuleArtifactContracts,
    ) -> dict[str, Any]:
        """Drop function kwargs for output-name settings pruned from artifacts."""
        dead_settings = self.dead_output_setting_names(
            module=module,
            artifact_contract=artifact_contract,
        )
        pruned_kwargs = dict(translated_kwargs)
        for setting_name in dead_settings:
            mapped_parameter = param_mapping.get(setting_name)
            if mapped_parameter is None:
                mapped_parameter = setting_name
            if isinstance(mapped_parameter, list):
                for parameter_name in mapped_parameter:
                    pruned_kwargs.pop(parameter_name, None)
            else:
                pruned_kwargs.pop(mapped_parameter, None)
        return pruned_kwargs

    @staticmethod
    def dead_output_setting_names(
        *,
        module: ModuleBlock,
        artifact_contract: ModuleArtifactContracts,
    ) -> frozenset[str]:
        """Return CellProfiler output-name settings whose artifacts were pruned."""
        retained_outputs = frozenset(
            ArtifactSpecKey(symbol.kind.artifact_kind, symbol.name)
            for symbol in artifact_contract.output_symbols
        )
        output_symbols_by_setting: dict[str, set[ArtifactSpecKey]] = {}
        for symbol in artifact_setting_symbols(module):
            if symbol.role.is_input:
                continue
            normalized_setting = normalize_cellprofiler_setting_name(
                symbol.setting_name
            )
            output_symbols_by_setting.setdefault(normalized_setting, set()).add(
                ArtifactSpecKey(symbol.role.artifact_kind, symbol.name)
            )
        return frozenset(
            setting_name
            for setting_name, output_symbols in output_symbols_by_setting.items()
            if output_symbols and not output_symbols & retained_outputs
        )

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
                if spec.kind is ArtifactKind.IMAGE
            }
            terminal.update(output for output in module_outputs if output not in consumed)
            consumed.update(ArtifactSpecKey.from_spec(spec) for spec in contract.inputs)
            consumed.update(
                ArtifactSpecKey.from_spec(spec)
                for spec in contract.runtime_artifact_inputs
            )
        return terminal


@dataclass(frozen=True)
class PipelineGeneratorRuntimeContractProjector:
    """Projection from symbol-table contracts to runtime artifact contracts."""

    generator: Any

    def by_module_num(
        self,
        modules: List[ModuleBlock],
        artifact_contracts: dict[int, ModuleArtifactContracts],
        *,
        externally_materialized_outputs: ExternallyMaterializedOutputs = frozenset(),
    ) -> dict[int, ModuleArtifactContract]:
        """Build product-runtime contracts without serializing them into generated code."""
        return {
            module.module_num: self.runtime_module_contract(
                artifact_contracts[module.module_num],
                externally_materialized_outputs=externally_materialized_outputs,
            )
            for module in modules
            if (
                artifact_contracts[module.module_num].inputs
                or artifact_contracts[module.module_num].outputs
            )
        }

    def runtime_module_contract(
        self,
        contract: ModuleArtifactContracts,
        *,
        externally_materialized_outputs: ExternallyMaterializedOutputs,
    ) -> ModuleArtifactContract:
        """Project symbol-table contracts into runtime module contracts."""
        outputs = tuple(
            self.runtime_output_spec(
                spec,
                externally_materialized_outputs=externally_materialized_outputs,
            )
            for spec in contract.outputs
        )
        declared_outputs = tuple(
            self.runtime_output_spec(
                spec,
                externally_materialized_outputs=externally_materialized_outputs,
            )
            for spec in contract.declared_outputs
        )
        return ModuleArtifactContract(
            module_name=contract.module_name,
            inputs=contract.inputs,
            runtime_artifact_inputs=contract.runtime_artifact_inputs,
            outputs=outputs,
            declared_outputs=declared_outputs,
        )

    @staticmethod
    def runtime_output_spec(
        spec: ArtifactSpec,
        *,
        externally_materialized_outputs: ExternallyMaterializedOutputs,
    ) -> ArtifactSpec:
        """Apply runtime-only materialization required by skipped SaveImages modules."""
        if ArtifactSpecKey.from_spec(spec) not in externally_materialized_outputs:
            if (
                spec.materialization is None
                and spec.kind is not ArtifactKind.SPECIAL
                and spec.kind not in DEFAULT_ARTIFACT_MATERIALIZATION_RULES
            ):
                return replace(spec, materialization=NO_ARTIFACT_MATERIALIZATION)
            return spec
        return replace(spec, materialization=tiff_stack(normalize_uint8=True))


@dataclass(frozen=True)
class PipelineGeneratorCodeEmitter:
    """Generated-code emission for imports, FunctionStep declarations, and comments."""

    generator: Any

    def generate_steps_from_registry(
        self,
        modules: List[ModuleBlock],
        function_names_by_module: Mapping[int, str],
        artifact_contracts: dict[int, ModuleArtifactContracts],
    ) -> tuple[str, tuple[ModuleSettingCoverageRecord, ...]]:
        """Generate pipeline_steps using registry functions with bound settings."""
        lines = [
            "# Pipeline Steps",
            "# Settings from .cppipe are bound as default parameters",
            "# variable_components derived from LLM-inferred category",
            "pipeline_steps = [",
        ]
        setting_coverage: list[ModuleSettingCoverageRecord] = []

        for module in modules:
            meta = self.generator.registry.module_metadata(module.name)
            category = meta.category
            step_name = module.name
            artifact_contract = artifact_contracts[module.module_num]
            func_name = function_names_by_module[module.module_num]

            input_source_literal = (
                "InputSource.PIPELINE_START"
                if artifact_contract.external_source_symbols
                else None
            )

            param_mapping = {}
            dead_output_settings = self.generator.pruner.dead_output_setting_names(
                module=module,
                artifact_contract=artifact_contract,
            )
            bound_settings = _ModuleSettingsBindingStrategy.for_module(
                module.name
            ).bind(
                module,
                binder=self.generator.settings_binder,
                param_mapping=param_mapping,
                ignored_unmapped_settings=dead_output_settings,
            )
            setting_coverage.extend(bound_settings.setting_coverage)
            translated_kwargs = dict(bound_settings.kwargs)
            translated_kwargs = self.generator.pruner.prune_dead_output_setting_kwargs(
                module=module,
                translated_kwargs=translated_kwargs,
                param_mapping=param_mapping,
                artifact_contract=artifact_contract,
            )
            invocation_options_literal = self.invocation_options_literal(
                bound_settings.invocation_options
            )
            processing_components = ModuleProcessingComponentStrategy.for_module(
                module.name
            ).components(
                category=category,
                contract=artifact_contract,
                bound_kwargs=translated_kwargs,
                category_defaults=self.generator.CATEGORY_TO_VARIABLE_COMPONENTS,
            )

            lines.append("    FunctionStep(")
            lines.extend(self.artifact_contract_comments(artifact_contract))
            if translated_kwargs:
                kwargs_lines = ["{"]
                for k, v in translated_kwargs.items():
                    kwargs_lines.append(
                        f"            {repr(k)}: {python_literal(v)},"
                    )
                kwargs_lines.append("        }")
                kwargs_str = "\n".join(kwargs_lines)

                if invocation_options_literal is None:
                    lines.append(f"        func=({func_name}, {kwargs_str}),")
                else:
                    lines.append(
                        f"        func=({func_name}, {kwargs_str}, "
                        f"{invocation_options_literal}),"
                    )
            else:
                if invocation_options_literal is None:
                    lines.append(f"        func={func_name},")
                else:
                    lines.append(
                        f"        func=({func_name}, {{}}, "
                        f"{invocation_options_literal}),"
                    )

            lines.append(f'        name="{step_name}",')
            if not artifact_contract.source_bindings.is_empty:
                lines.append(
                    "        source_bindings="
                    f"{source_bindings_literal(artifact_contract.source_bindings)},"
                )
            lines.append("        processing_config=LazyProcessingConfig(")
            lines.append(
                "            variable_components=["
                + ", ".join(processing_components.variable_components)
                + "],"
            )
            if processing_components.group_by_literal is not None:
                lines.append(
                    f"            group_by={processing_components.group_by_literal},"
                )
            if input_source_literal is not None:
                lines.append(f"            input_source={input_source_literal},")
            lines.append("        ),")

            lines.append("    ),")

        lines.append("]")
        return "\n".join(lines), tuple(setting_coverage)

    @staticmethod
    def invocation_options_literal(
        options: RuntimeInvocationOptions | None,
    ) -> str | None:
        """Return generated-code literal for typed invocation options."""
        if options is None:
            return None
        if isinstance(options, CellProfilerInvocationOptions):
            scope = options.grid_cycle_scope
            if not isinstance(scope, CellProfilerGridCycleScope):
                raise TypeError(
                    "CellProfilerInvocationOptions.grid_cycle_scope must be "
                    "CellProfilerGridCycleScope."
                )
            return (
                "CellProfilerInvocationOptions("
                f"grid_cycle_scope=CellProfilerGridCycleScope.{scope.name})"
            )
        raise TypeError(
            "Unsupported RuntimeInvocationOptions for generated pipeline: "
            f"{type(options).__name__}."
        )

    @staticmethod
    def backend_function_import_block(function_names: Iterable[str]) -> str:
        """Return imports for the absorbed backend functions used by the pipeline."""
        unique_function_names = tuple(dict.fromkeys(sorted(function_names)))
        if not unique_function_names:
            return ""
        lines = [
            "from openhcs.processing.backends.cellprofiler import "
            "get_cellprofiler_function as _get_cellprofiler_function",
        ]
        lines.extend(
            f"{function_name} = _get_cellprofiler_function({function_name!r})"
            for function_name in unique_function_names
        )
        lines.append("")
        return "\n".join(lines)

    def artifact_contract_comments(
        self,
        contract: ModuleArtifactContracts,
    ) -> list[str]:
        """Return generated comments summarizing artifact contract semantics."""
        lines: list[str] = []
        if contract.inputs:
            lines.append(
                "        # CellProfiler artifact inputs: "
                + self.format_artifact_specs(contract.inputs)
            )
        if contract.external_source_symbols:
            lines.append(
                "        # Source bindings: "
                + ", ".join(
                    symbol.name for symbol in contract.external_source_symbols
                )
            )
        if contract.runtime_artifact_inputs:
            lines.append(
                "        # Runtime artifact inputs: "
                + self.format_artifact_specs(contract.runtime_artifact_inputs)
            )
        if contract.outputs:
            lines.append(
                "        # CellProfiler artifact outputs: "
                + self.format_artifact_specs(contract.outputs)
            )
        return lines

    @staticmethod
    def format_artifact_specs(specs: tuple[ArtifactSpec, ...]) -> str:
        """Format artifact specs for deterministic generated-code comments."""
        return ", ".join(f"{spec.kind.value}:{spec.name}" for spec in specs)


@dataclass(frozen=True)
class PipelineGeneratorBuildStage:
    """Top-level CellProfiler module partitioning and generated-pipeline assembly."""

    generator: Any

    def generate(
        self,
        request: GeneratedPipelineRequest,
        modules: List[ModuleBlock],
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
                f"Missing {len(missing_modules)} modules from absorbed library: "
                f"{[m.name for m in missing_modules]}. "
                "Re-run absorption with --force to regenerate."
            )

        ordered_modules = [*skipped_modules, *registry_modules]
        symbol_table = CellProfilerSymbolTable.compile(ordered_modules)
        contracts_by_module = {
            module.module_num: symbol_table.contract_for(module)
            for module in registry_modules
        }
        infrastructure_contracts = tuple(
            symbol_table.contract_for(module)
            for module in skipped_modules
        )
        save_images_required_artifacts = (
            frozenset(_save_images_required_artifacts(skipped_modules))
            if request.materialize_skipped_save_images
            else frozenset()
        )
        infrastructure_input_artifacts = {
            ArtifactSpecKey.from_spec(input_spec)
            for contract in infrastructure_contracts
            for input_spec in (
                *contract.inputs,
                *contract.runtime_artifact_inputs,
            )
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
            save_images_required_artifacts | terminal_image_artifacts
        )
        executable_modules = (
            self.generator.pruner.prune_dead_unmaterialized_artifact_steps(
                registry_modules,
                contracts_by_module,
                externally_required_artifacts=(
                    infrastructure_input_artifacts
                    | externally_materialized_outputs
                ),
            )
            if request.prune_dead_unmaterialized_artifact_steps
            else registry_modules
        )
        runtime_module_contracts_by_module = self.generator.runtime_contracts.by_module_num(
            executable_modules,
            contracts_by_module,
            externally_materialized_outputs=externally_materialized_outputs,
        )

        function_names_by_module: dict[int, str] = {}
        for module in executable_modules:
            meta = self.generator.registry.module_metadata(module.name)
            resolved_function = _ModuleFunctionResolutionStrategy.for_module(
                module.name
            ).resolve(
                module,
                default_function_name=meta.function_name,
            )
            function_names_by_module[module.module_num] = (
                resolved_function.function_name
            )

        if executable_modules:
            imports += "# Absorbed CellProfiler functions\n"
            imports += self.generator.emitter.backend_function_import_block(
                function_names_by_module.values()
            )
            imports += (
                "from openhcs.interop.cellprofiler.runtime import (\n"
                "    CellProfilerGridCycleScope,\n"
                "    CellProfilerInvocationOptions,\n"
                ")\n"
                "\n"
            )

        steps, setting_coverage = self.generator.emitter.generate_steps_from_registry(
            executable_modules,
            function_names_by_module,
            contracts_by_module,
        )
        code = imports + steps

        return GeneratedPipeline(
            name=request.pipeline_name,
            code=code,
            source_cppipe=str(request.source_cppipe),
            converted_modules=[m.name for m in executable_modules],
            failed_modules=[m.name for m in missing_modules],
            artifact_contracts=tuple(
                contracts_by_module[module.module_num]
                for module in executable_modules
            ),
            runtime_module_contracts=tuple(
                (
                    module.module_num,
                    runtime_module_contracts_by_module[module.module_num],
                )
                for module in executable_modules
                if module.module_num in runtime_module_contracts_by_module
            ),
            source_schema=symbol_table.source_schema,
            setting_coverage=setting_coverage,
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

    # Standard imports for generated pipelines
    IMPORTS_BASE = '''"""
OpenHCS Pipeline - Converted from CellProfiler
Source: {source_file}

Auto-generated by CellProfiler to OpenHCS converter.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

# OpenHCS imports
from openhcs.core.artifacts import ArtifactKind
from openhcs.core.steps.function_step import FunctionStep
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
from openhcs.core.config import LazyProcessingConfig
from openhcs.constants.constants import VariableComponents, GroupBy
from openhcs.constants.constants import AllComponents
from openhcs.constants.input_source import InputSource

'''

    def __init__(self, library_root: Optional[Path] = None):
        """
        Initialize generator.

        Args:
            library_root: Path to absorbed cellprofiler_library
        """
        self._explicit_library_root = library_root is not None
        self.library_root = library_root or Path(__file__).parent.parent / "cellprofiler_library"
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
                prune_dead_unmaterialized_artifact_steps=(
                    prune_dead_unmaterialized_artifact_steps
                ),
                materialize_skipped_save_images=materialize_skipped_save_images,
                materialize_terminal_images=materialize_terminal_images,
            ),
            modules=modules,
        )

    # Category → variable_components mapping
    CATEGORY_TO_VARIABLE_COMPONENTS = {
        "image_operation": ("VariableComponents.SITE",),
        "z_projection": ("VariableComponents.Z_INDEX",),
        "channel_operation": ("VariableComponents.CHANNEL",),
    }


def python_literal(value: Any) -> str:
    """Render a deterministic generated-code literal for bound setting values."""
    if isinstance(value, Enum):
        return repr(value.value)
    if isinstance(value, tuple):
        return "(" + ", ".join(python_literal(item) for item in value) + (
            "," if len(value) == 1 else ""
        ) + ")"
    if isinstance(value, list):
        return "[" + ", ".join(python_literal(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(
            f"{python_literal(key)}: {python_literal(item)}"
            for key, item in value.items()
        ) + "}"
    return repr(value)
