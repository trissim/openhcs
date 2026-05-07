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
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Mapping, Optional

from metaclass_registry import AutoRegisterMeta
from openhcs.constants import Backend
from openhcs.core.artifact_materialization_policy import (
    DEFAULT_ARTIFACT_MATERIALIZATION_RULES,
)
from openhcs.core.artifact_observability import externally_required_artifact_outputs
from openhcs.core.artifacts import ArtifactKind, ArtifactSpec
from openhcs.core.pipeline_image_schema import PipelineImageSchema
from openhcs.core.runtime_invocation import RuntimeInvocationOptions
from openhcs.core.vfs_protocol import FileManagerLike
from openhcs.interop.cellprofiler.runtime import (
    CellProfilerGridCycleScope,
    CellProfilerInvocationOptions,
)
from openhcs.interop.cellprofiler.parser import ModuleBlock

from benchmark.cellprofiler_library import canonical_module_name
from benchmark.cellprofiler_library.functions._enum import _coerce_function_enum
from benchmark.cellprofiler_library.functions.correctilluminationcalculate import (
    CalculationScope,
)

from .artifact_semantics import artifact_setting_symbols
from .module_function_resolution import ModuleFunctionResolutionStrategy
from .module_settings_binding import ModuleSettingsBindingStrategy
from .processing_contract_resolution import resolve_processing_contract
from .settings_binder import SettingsBinder, normalize_cellprofiler_setting_name
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    setting_values,
    split_symbol_names,
)
from .symbol_table import (
    CellProfilerSymbolTable,
    ModuleArtifactContracts,
    module_contract_literal,
    source_bindings_literal,
)

logger = logging.getLogger(__name__)


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
        raw_scope = bound_kwargs.get("calculation_scope", CalculationScope.EACH)
        scope = _coerce_function_enum(CalculationScope, raw_scope)
        if scope in {
            CalculationScope.ALL_FIRST_CYCLE,
            CalculationScope.ALL_ACROSS_CYCLES,
        }:
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


def _artifact_key(spec: ArtifactSpec) -> tuple[ArtifactKind, str]:
    return (spec.kind, spec.name)


def _has_materialized_output(contract: ModuleArtifactContracts) -> bool:
    """Return whether any output is externally observable by artifact policy."""
    return any(
        spec.materialization is not None
        or spec.kind in DEFAULT_ARTIFACT_MATERIALIZATION_RULES
        for spec in contract.outputs
    )


def _save_images_required_artifacts(
    skipped_modules: list[ModuleBlock],
) -> set[tuple[ArtifactKind, str]]:
    """Return image artifacts required by skipped CellProfiler SaveImages modules."""
    return {
        (ArtifactKind.IMAGE, image_name)
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
    source_schema: PipelineImageSchema = PipelineImageSchema.empty()
    
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

Auto-generated by CellProfiler → OpenHCS converter.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

# OpenHCS imports
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
        self.library_root = library_root or Path(__file__).parent.parent / "cellprofiler_library"
        self.settings_binder = SettingsBinder()
        self._registry = self._load_registry()

    def _load_registry(self) -> Dict[str, dict]:
        """Load full module metadata from absorbed library."""
        contracts_file = self.library_root / "contracts.json"
        if not contracts_file.exists():
            raise FileNotFoundError(
                f"No absorbed library found at {contracts_file}. "
                "Run 'python -m benchmark.converter.absorb' first."
            )

        try:
            data = json.loads(contracts_file.read_text())
            # Store full metadata, not just function name
            registry = {
                module_name: {
                    "function_name": info["function_name"],
                    "contract": info.get("contract", "pure_2d"),
                    "category": info.get("category", "image_operation"),
                    "confidence": info.get("confidence", 0.5),
                }
                for module_name, info in data.items()
                if info.get("validated", False)
            }
            logger.info(f"Loaded {len(registry)} absorbed functions from registry")
            return registry
        except Exception as e:
            raise RuntimeError(f"Failed to load registry: {e}")

    def has_module(self, module_name: str) -> bool:
        """Check if module exists in absorbed library."""
        return canonical_module_name(module_name) in self._registry

    def _module_metadata(self, module_name: str) -> dict[str, Any]:
        """Return absorbed metadata for a module after canonical name resolution."""
        return self._registry[canonical_module_name(module_name)]
    
    def generate_from_registry(
        self,
        pipeline_name: str,
        source_cppipe: Path,
        modules: List[ModuleBlock],
        skipped_modules: Optional[List[ModuleBlock]] = None,
        prune_dead_unmaterialized_artifact_steps: bool = False,
        materialize_skipped_save_images: bool = True,
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
        skipped_modules = skipped_modules or []

        # Partition modules into registry-available and missing
        registry_modules = []
        missing_modules = []

        for module in modules:
            if self.has_module(module.name):
                registry_modules.append(module)
            else:
                missing_modules.append(module)
                logger.warning(f"Module {module.name} not in absorbed library")

        # Build imports
        imports = self.IMPORTS_BASE.format(source_file=source_cppipe.name)

        # Add note about skipped infrastructure modules
        if skipped_modules:
            skip_note = "\n# Skipped infrastructure modules (handled by OpenHCS):\n"
            for module in skipped_modules:
                if module.name == "LoadData":
                    skip_note += "#   - LoadData -> handled by plate_path + openhcs_metadata.json\n"
                elif module.name == "ExportToSpreadsheet":
                    skip_note += "#   - ExportToSpreadsheet -> handled by @special_outputs(csv_materializer(...))\n"
                else:
                    skip_note += f"#   - {module.name}\n"
            imports += skip_note + "\n"

        # Fail-loud if any modules are missing (no LLM fallback)
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
            if materialize_skipped_save_images
            else frozenset()
        )
        executable_modules = (
            self._prune_dead_unmaterialized_artifact_steps(
                registry_modules,
                contracts_by_module,
                externally_required_artifacts={
                    _artifact_key(input_spec)
                    for contract in infrastructure_contracts
                    for input_spec in (
                        *contract.inputs,
                        *contract.runtime_artifact_inputs,
                    )
                }
                | save_images_required_artifacts,
            )
            if prune_dead_unmaterialized_artifact_steps
            else registry_modules
        )

        # Add registry imports for available modules
        raw_function_bindings: dict[int, str] = {}
        runtime_function_bindings: dict[int, str] = {}
        if executable_modules:
            imports += "# Absorbed CellProfiler functions (dynamically loaded)\n"
            imports += (
                "from openhcs.processing.backends.cellprofiler import "
                "require_cellprofiler_function\n\n"
            )
            imports += (
                "from benchmark.cellprofiler_compat import (\n"
                "    CellProfilerModuleExecutor,\n"
                "    cellprofiler_runtime_adapter_factory,\n"
                ")\n"
                "from openhcs.interop.cellprofiler.runtime import (\n"
                "    CellProfilerGridCycleScope,\n"
                "    CellProfilerInvocationOptions,\n"
                ")\n"
                "from openhcs.core.module_artifact_contract import ModuleArtifactContract\n"
                "from openhcs.core.callable_contract import (\n"
                "    RUNTIME_IMAGE_EXECUTION_MODE_ATTR,\n"
                "    attach_callable_contract_metadata,\n"
                "    prepare_processing_callable,\n"
                ")\n"
                "from openhcs.core.pipeline.function_contracts import artifact_inputs, artifact_outputs\n"
                "from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract\n"
                "from openhcs.core.runtime_adapters import runtime_adapter\n\n"
            )

            # Generate function assignments
            func_assignments = []
            for module in executable_modules:
                resolved_function = ModuleFunctionResolutionStrategy.for_module(
                    module.name
                ).resolve(
                    module,
                    default_function_name=self._module_metadata(module.name)[
                        "function_name"
                    ],
                )
                func_name = resolved_function.function_name
                binding_name = self._function_binding_name(module, func_name)
                raw_function_bindings[module.module_num] = binding_name
                contract = contracts_by_module[module.module_num]
                if contract.inputs or contract.outputs:
                    runtime_function_bindings[module.module_num] = (
                        self._runtime_binding_name(module, func_name)
                    )
                else:
                    runtime_function_bindings[module.module_num] = binding_name
                func_assignments.append(
                    f'{binding_name} = require_cellprofiler_function("{module.name}", '
                    f'function_name="{func_name}")'
                )
                func_assignments.append(
                    "attach_callable_contract_metadata("
                    f"{binding_name}, "
                    "declared_processing_contract="
                    f"{self._resolved_processing_contract_literal(module.name, func_name)})"
                )
            imports += "\n".join(func_assignments) + "\n\n"

        imports += self._generate_artifact_contracts(
            executable_modules,
            contracts_by_module,
            externally_materialized_outputs=save_images_required_artifacts,
        )
        imports += self._generate_runtime_wrappers(
            executable_modules,
            raw_function_bindings,
            runtime_function_bindings,
            contracts_by_module,
        )

        # Generate steps with bound settings
        steps = self._generate_steps_from_registry(
            executable_modules,
            runtime_function_bindings,
            contracts_by_module,
        )

        # Combine
        code = imports + steps

        return GeneratedPipeline(
            name=pipeline_name,
            code=code,
            source_cppipe=str(source_cppipe),
            converted_modules=[m.name for m in executable_modules],
            failed_modules=[m.name for m in missing_modules],
            artifact_contracts=tuple(
                contracts_by_module[module.module_num]
                for module in executable_modules
            ),
            source_schema=symbol_table.source_schema,
        )

    # Category → variable_components mapping
    CATEGORY_TO_VARIABLE_COMPONENTS = {
        "image_operation": ("VariableComponents.SITE",),
        "z_projection": ("VariableComponents.Z_INDEX",),
        "channel_operation": ("VariableComponents.CHANNEL",),
    }

    def _prune_dead_unmaterialized_artifact_steps(
        self,
        modules: list[ModuleBlock],
        artifact_contracts: dict[int, ModuleArtifactContracts],
        *,
        externally_required_artifacts: set[tuple[ArtifactKind, str]] | None = None,
    ) -> list[ModuleBlock]:
        """Remove artifact-producing steps whose outputs are neither consumed nor materialized."""
        live_artifacts = {
            _artifact_key(output)
            for contract in artifact_contracts.values()
            for output in contract.outputs
            if (
                output.materialization is not None
                or output.kind in DEFAULT_ARTIFACT_MATERIALIZATION_RULES
            )
        } | {
            _artifact_key(output)
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
            output_keys = {_artifact_key(output) for output in contract.outputs}
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
                    or _artifact_key(output) in live_artifacts
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
                _artifact_key(input_spec)
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

    def _generate_steps_from_registry(
        self,
        modules: List[ModuleBlock],
        function_bindings: dict[int, str],
        artifact_contracts: dict[int, ModuleArtifactContracts],
    ) -> str:
        """Generate pipeline_steps using registry functions with bound settings."""
        lines = [
            "# Pipeline Steps",
            "# Settings from .cppipe are bound as default parameters",
            "# variable_components derived from LLM-inferred category",
            "pipeline_steps = [",
        ]

        for module in modules:
            meta = self._module_metadata(module.name)
            resolved_function = ModuleFunctionResolutionStrategy.for_module(
                module.name
            ).resolve(
                module,
                default_function_name=meta["function_name"],
            )
            func_name = resolved_function.function_name
            binding_name = function_bindings[module.module_num]
            category = meta.get("category", "image_operation")
            step_name = module.name
            artifact_contract = artifact_contracts[module.module_num]

            input_source_literal = (
                "InputSource.PIPELINE_START"
                if artifact_contract.external_source_symbols
                else None
            )

            # Parse parameter mapping from function docstring
            param_mapping = self._parse_parameter_mapping(func_name)
            bound_settings = ModuleSettingsBindingStrategy.for_module(
                module.name
            ).bind(
                module,
                binder=self.settings_binder,
                param_mapping=param_mapping,
            )
            translated_kwargs = dict(bound_settings.kwargs)
            unmapped_kwargs = dict(bound_settings.unmapped_kwargs)
            translated_kwargs = self._prune_dead_output_setting_kwargs(
                module=module,
                translated_kwargs=translated_kwargs,
                param_mapping=param_mapping,
                artifact_contract=artifact_contract,
            )
            unmapped_kwargs = self._prune_dead_output_setting_comments(
                module=module,
                unmapped_kwargs=unmapped_kwargs,
                artifact_contract=artifact_contract,
            )
            invocation_options_literal = self._invocation_options_literal(
                bound_settings.invocation_options
            )
            processing_components = ModuleProcessingComponentStrategy.for_module(
                module.name
            ).components(
                category=category,
                contract=artifact_contract,
                bound_kwargs=translated_kwargs,
                category_defaults=self.CATEGORY_TO_VARIABLE_COMPONENTS,
            )

            # Build func parameter - either just the function or (function, kwargs_dict)
            lines.append("    FunctionStep(")
            lines.extend(self._artifact_contract_comments(artifact_contract))
            if translated_kwargs:
                # Format kwargs dict
                kwargs_lines = ["{"]
                for k, v in translated_kwargs.items():
                    kwargs_lines.append(f"            {repr(k)}: {repr(v)},")
                kwargs_lines.append("        }")
                kwargs_str = "\n".join(kwargs_lines)

                if invocation_options_literal is None:
                    lines.append(f"        func=({binding_name}, {kwargs_str}),")
                else:
                    lines.append(
                        f"        func=({binding_name}, {kwargs_str}, "
                        f"{invocation_options_literal}),"
                    )
            else:
                if invocation_options_literal is None:
                    lines.append(f"        func={binding_name},")
                else:
                    lines.append(
                        f"        func=({binding_name}, {{}}, "
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

            # Add unmapped settings as comments (for debugging)
            if unmapped_kwargs:
                lines.append("        # Unmapped settings:")
                for k, v in list(unmapped_kwargs.items())[:3]:
                    lines.append(f"        # {k}={repr(v)}")

            lines.append("    ),")

        lines.append("]")
        return "\n".join(lines)

    def _prune_dead_output_setting_kwargs(
        self,
        *,
        module: ModuleBlock,
        translated_kwargs: dict[str, Any],
        param_mapping: Mapping[str, Any],
        artifact_contract: ModuleArtifactContracts,
    ) -> dict[str, Any]:
        """Drop function kwargs for output-name settings pruned from artifacts."""
        dead_settings = self._dead_output_setting_names(
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

    def _prune_dead_output_setting_comments(
        self,
        *,
        module: ModuleBlock,
        unmapped_kwargs: dict[str, Any],
        artifact_contract: ModuleArtifactContracts,
    ) -> dict[str, Any]:
        """Drop comments for output-name settings pruned from artifacts."""
        dead_settings = self._dead_output_setting_names(
            module=module,
            artifact_contract=artifact_contract,
        )
        return {
            setting_name: value
            for setting_name, value in unmapped_kwargs.items()
            if setting_name not in dead_settings
        }

    def _dead_output_setting_names(
        self,
        *,
        module: ModuleBlock,
        artifact_contract: ModuleArtifactContracts,
    ) -> frozenset[str]:
        retained_outputs = frozenset(
            (symbol.kind.artifact_kind, symbol.name)
            for symbol in artifact_contract.output_symbols
        )
        output_symbols_by_setting: dict[str, set[tuple[ArtifactKind, str]]] = {}
        for symbol in artifact_setting_symbols(module):
            if symbol.role.is_input:
                continue
            normalized_setting = normalize_cellprofiler_setting_name(
                symbol.setting_name
            )
            output_symbols_by_setting.setdefault(normalized_setting, set()).add(
                (symbol.role.artifact_kind, symbol.name)
            )
        return frozenset(
            setting_name
            for setting_name, output_symbols in output_symbols_by_setting.items()
            if output_symbols and not output_symbols & retained_outputs
        )

    def _invocation_options_literal(
        self,
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

    def _generate_artifact_contracts(
        self,
        modules: List[ModuleBlock],
        artifact_contracts: dict[int, ModuleArtifactContracts],
        *,
        externally_materialized_outputs: frozenset[tuple[ArtifactKind, str]] = (
            frozenset()
        ),
    ) -> str:
        """Emit converter-owned artifact contracts into generated pipeline code."""
        contracts = []
        for module in modules:
            contract = artifact_contracts[module.module_num]
            if contract.inputs or contract.outputs:
                contracts.append(contract)
        if not contracts:
            return ""

        requires_no_materialization_import = any(
            spec.kind not in DEFAULT_ARTIFACT_MATERIALIZATION_RULES
            for contract in contracts
            for spec in contract.outputs
        )
        uses_sidecar_roles = any(
            spec.sidecar_role is not None
            for contract in contracts
            for spec in (*contract.outputs, *contract.declared_outputs)
        )
        artifact_imports = "ArtifactKind, ArtifactSpec"
        if uses_sidecar_roles:
            artifact_imports += ", ArtifactSidecarRole"
        lines = [
            "# CellProfiler name-to-artifact contracts compiled from .cppipe",
            f"from openhcs.core.artifacts import {artifact_imports}",
        ]
        if requires_no_materialization_import:
            lines.append(
                "from openhcs.core.artifact_materialization_policy import NO_ARTIFACT_MATERIALIZATION"
            )
        if externally_materialized_outputs:
            lines.append("from openhcs.processing.materialization import tiff_stack")
        lines.extend(("", "CELLPROFILER_MODULE_CONTRACTS = {"))
        for contract in contracts:
            lines.append(
                f"    {contract.module_num}: "
                f"{module_contract_literal(contract, externally_materialized_outputs=externally_materialized_outputs)},"
            )
        lines.append("}")
        lines.append("")
        return "\n".join(lines) + "\n"

    def _generate_runtime_wrappers(
        self,
        modules: List[ModuleBlock],
        raw_function_bindings: dict[int, str],
        runtime_function_bindings: dict[int, str],
        artifact_contracts: dict[int, ModuleArtifactContracts],
    ) -> str:
        """Emit adapter-aware wrappers around absorbed CellProfiler functions."""
        if not modules:
            return ""

        lines = [
            "# Adapter-aware CellProfiler execution wrappers",
        ]
        for module in modules:
            contract = artifact_contracts[module.module_num]
            if not contract.inputs and not contract.outputs:
                continue
            raw_binding = raw_function_bindings[module.module_num]
            runtime_binding = runtime_function_bindings[module.module_num]
            executor_name = self._executor_binding_name(module)
            resolved_function = ModuleFunctionResolutionStrategy.for_module(
                module.name
            ).resolve(
                module,
                default_function_name=self._module_metadata(module.name)[
                    "function_name"
                ],
            )
            processing_contract = self._runtime_processing_contract_expression(
                module.name,
                resolved_function.function_name,
                contract,
            )

            lines.append(
                f"{executor_name} = "
                f"CellProfilerModuleExecutor(CELLPROFILER_MODULE_CONTRACTS[{module.module_num}])"
            )
            lines.append(
                f"@artifact_inputs(*CELLPROFILER_MODULE_CONTRACTS[{module.module_num}].runtime_artifact_inputs)"
            )
            lines.append(
                f"@artifact_outputs(*CELLPROFILER_MODULE_CONTRACTS[{module.module_num}].outputs)"
            )
            lines.append(
                "@runtime_adapter(\"cellprofiler_runtime\", "
                "cellprofiler_runtime_adapter_factory, "
                "manages_artifact_inputs=True)"
            )
            lines.append(
                f"def {runtime_binding}(image, *, cellprofiler_runtime, "
                "runtime_invocation_options=None, enabled=True, **kwargs):"
            )
            lines.append(
                "    if not enabled:"
            )
            lines.append(
                "        return image"
            )
            lines.append(
                '    kwargs.pop("slice_by_slice", None)'
            )
            lines.append(
                f"    return {executor_name}.run("
                f"{raw_binding}, image, "
                "cellprofiler_runtime=cellprofiler_runtime, "
                "invocation_options=runtime_invocation_options, **kwargs)"
            )
            prepare_binding = f"_prepare_{runtime_binding}"
            lines.append(
                f"def {prepare_binding}():"
            )
            lines.append(
                f"    prepare_processing_callable({raw_binding})"
            )
            lines.append(
                f"    {executor_name}.prepare({raw_binding})"
            )
            lines.append(
                f"{runtime_binding}.input_memory_type = {raw_binding}.input_memory_type"
            )
            lines.append(
                f"{runtime_binding}.output_memory_type = {raw_binding}.output_memory_type"
            )
            lines.append(
                f"{runtime_binding}.__processing_contract__ = "
                f"{processing_contract}"
            )
            lines.append(
                "attach_callable_contract_metadata("
                f"{runtime_binding}, "
                "declared_processing_contract="
                f"{self._resolved_processing_contract_literal(module.name, resolved_function.function_name)}, "
                f"raw_processing_function={raw_binding}, "
                f"prepare={prepare_binding}, "
                "runtime_image_execution_mode="
                f"getattr({raw_binding}, RUNTIME_IMAGE_EXECUTION_MODE_ATTR, None))"
            )
            lines.append("")

        return "\n".join(lines) + "\n"

    def _artifact_contract_comments(
        self,
        contract: ModuleArtifactContracts,
    ) -> list[str]:
        lines: list[str] = []
        if contract.inputs:
            lines.append(
                "        # CellProfiler artifact inputs: "
                + self._format_artifact_specs(contract.inputs)
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
                + self._format_artifact_specs(contract.runtime_artifact_inputs)
            )
        if contract.outputs:
            lines.append(
                "        # CellProfiler artifact outputs: "
                + self._format_artifact_specs(contract.outputs)
            )
        return lines

    def _format_artifact_specs(self, specs: tuple[ArtifactSpec, ...]) -> str:
        return ", ".join(f"{spec.kind.value}:{spec.name}" for spec in specs)

    def _function_binding_name(self, module: ModuleBlock, func_name: str) -> str:
        """Return a per-module binding name so repeated modules do not alias."""
        return f"{func_name}_{module.module_num}"

    def _runtime_binding_name(self, module: ModuleBlock, func_name: str) -> str:
        """Return the generated adapter-aware function binding name."""
        return f"{self._function_binding_name(module, func_name)}_runtime"

    def _executor_binding_name(self, module: ModuleBlock) -> str:
        """Return the generated executor binding name for one module."""
        return f"_CELLPROFILER_EXECUTOR_{module.module_num}"

    def _module_to_function_name(self, module_name: str) -> str:
        """Convert module name to function name (snake_case)."""
        # IdentifyPrimaryObjects -> identify_primary_objects
        name = re.sub(r'([A-Z])', r'_\1', module_name).lower().lstrip('_')
        return name

    def _runtime_processing_contract_expression(
        self,
        module_name: str,
        function_name: str,
        contract: ModuleArtifactContracts,
    ) -> str:
        """Return the effective runtime contract for one generated wrapper.

        Adapter-managed CellProfiler wrappers resolve named source/runtime inputs
        before calling the absorbed function. They must therefore execute once per
        pattern group, and the typed CellProfiler executor applies the absorbed
        function contract after the image payload is resolved.
        """
        if contract.inputs or contract.outputs:
            return "ProcessingContract.FLEXIBLE"
        return self._processing_contract_expression(module_name, function_name)

    def _resolved_processing_contract_literal(
        self,
        module_name: str,
        function_name: str,
    ) -> str:
        resolved_contract = resolve_processing_contract(
            module_name,
            function_name,
            str(self._module_metadata(module_name)["contract"]),
        )
        return repr(resolved_contract.contract.name)

    def _processing_contract_expression(
        self,
        module_name: str,
        function_name: str,
    ) -> str:
        """Return generated-code expression for the raw absorbed module contract."""
        resolved_contract = resolve_processing_contract(
            module_name,
            function_name,
            str(self._module_metadata(module_name)["contract"]),
        )
        return f"ProcessingContract.{resolved_contract.contract.name}"

    def _parse_parameter_mapping(self, func_name: str) -> Dict[str, Any]:
        """
        Parse parameter mapping from function docstring.

        Returns dict mapping CellProfiler setting names to Python parameter names.
        Example: {'Typical diameter...' -> ['min_diameter', 'max_diameter']}
        """
        try:
            # Read the file directly (no imports needed - mappings are in the .py files)
            module_name = func_name.replace('_', '')
            func_file = Path(__file__).parent.parent / "cellprofiler_library" / "functions" / f"{module_name}.py"

            if not func_file.exists():
                return {}

            # Read file content
            content = func_file.read_text()

            # Find the parameter mapping section (anywhere in the file)
            mapping = {}
            in_mapping_section = False

            for line in content.split('\n'):
                stripped = line.strip()

                if 'CellProfiler Parameter Mapping:' in stripped:
                    in_mapping_section = True
                    continue

                if in_mapping_section:
                    # Stop at empty line, next section, or another mapping block
                    if not stripped:
                        # Empty line - might be end of section
                        continue
                    if (stripped.startswith('Args:') or
                        stripped.startswith('Returns:') or
                        stripped.startswith('Identify') or
                        stripped.startswith('Measure') or
                        stripped.startswith('"""') or
                        stripped.startswith('from ') or
                        stripped.startswith('import ')):
                        # Reached end of mapping section
                        if mapping:  # Only break if we've collected some mappings
                            break
                        continue

                    # Skip header line
                    if 'CellProfiler setting' in stripped and 'Python parameter' in stripped:
                        continue

                    # Parse mapping line: 'Setting Name' -> param_name
                    # or 'Setting Name' -> [param1, param2]
                    # or 'Setting Name' -> (pipeline-handled)
                    if '->' in stripped:
                        parts = stripped.split('->', 1)
                        if len(parts) == 2:
                            cp_setting = parts[0].strip().strip("'\"")
                            py_param = parts[1].strip()

                            normalized_key = normalize_cellprofiler_setting_name(
                                cp_setting
                            )

                            # Handle (pipeline-handled) or null
                            if 'pipeline-handled' in py_param or py_param == 'null':
                                mapping[normalized_key] = None
                            # Handle list [param1, param2]
                            elif py_param.startswith('[') and py_param.endswith(']'):
                                params = py_param[1:-1].split(',')
                                mapping[normalized_key] = [p.strip() for p in params]
                            # Handle single parameter
                            else:
                                mapping[normalized_key] = py_param

            return mapping

        except Exception as e:
            logger.warning(f"Could not parse parameter mapping for {func_name}: {e}")
            return {}
