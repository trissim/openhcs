"""Generated CellProfiler pipeline import and registry authorities."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import sys
from collections.abc import Sequence
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from openhcs.core.callable_contract import CallableContract
from openhcs.constants.input_source import InputSource
from openhcs.core.function_contract_metadata import FunctionContractAttribute
from openhcs.core.function_patterns import FunctionInvocationKey, normalize_function_pattern
from openhcs.core.function_step_invocation_contracts import (
    FunctionStepInvocationContractPayload,
)
from openhcs.core.function_reference import FunctionReference
from openhcs.core.invocation_artifacts import (
    ArtifactDeclarationStepContext,
    InvocationContractPlan,
)
from openhcs.core.artifact_contract_preview import SourceBindingRuntimeContractGuard
from openhcs.core.pipeline import Pipeline
from openhcs.core.pipeline.step_snapshot import StepSnapshot
from openhcs.core.source_bindings import (
    SourceBindingsConfig,
    StepSourceBindingsConfig,
    resolve_effective_step_source_bindings,
)
from openhcs.core.steps.function_step import FunctionStep
from openhcs.processing.backends.lib_registry.openhcs_registry import OpenHCSRegistry
from openhcs.processing.backends.lib_registry.registry_service import RegistryService
from openhcs.processing.backends.lib_registry.unified_registry import (
    FunctionMetadata,
    ProcessingContract,
)
from openhcs.processing.func_registry import register_function
from openhcs.interop.cellprofiler.runtime.module_execution import (
    CellProfilerGroupedModuleContracts,
    CellProfilerGroupedRuntimeCallable,
    CellProfilerGroupedRuntimeStepBinding,
    CellProfilerRuntimeCallable,
    CellProfilerRuntimeStepBinding,
)
from openhcs.interop.cellprofiler.symbol_table import ModuleArtifactContracts
from openhcs.core.module_artifact_contract import ModuleArtifactContract
from openhcs.processing.backends.cellprofiler import (
    CellProfilerFunctionCatalog,
)


@dataclass(frozen=True, slots=True)
class GeneratedPipelineModuleIdentity:
    """Stable import identity for one generated pipeline module."""

    module_path: Path
    code: str
    explicit_module_name: str | None = None

    @property
    def module_name(self) -> str:
        if self.explicit_module_name is not None:
            return self.explicit_module_name
        digest = hashlib.sha1(
            f"{self.module_path.resolve()}::{self.code}".encode("utf-8")
        ).hexdigest()[:12]
        stem = "".join(
            character if character.isalnum() else "_"
            for character in self.module_path.stem
        ).strip("_")
        return f"benchmark_generated_{stem or 'pipeline'}_{digest}"


class GeneratedPipelineRuntimeModule:
    """Nominal authority for generated pipeline module runtime integration."""

    @classmethod
    def for_generated_source(
        cls,
        *,
        module_path: Path,
        code: str,
    ) -> "GeneratedPipelineRuntimeModule":
        return cls(GeneratedPipelineModuleIdentity(module_path=module_path, code=code))

    def __init__(self, identity: GeneratedPipelineModuleIdentity) -> None:
        self.identity = identity

    @property
    def module_name(self) -> str:
        return self.identity.module_name

    def load_from_path(self, module_path: Path) -> ModuleType:
        """Import generated pipeline code from disk under the stable module name."""
        spec = importlib.util.spec_from_file_location(self.module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to create module spec for {module_path}.")

        module = importlib.util.module_from_spec(spec)
        sys.modules[self.module_name] = module
        spec.loader.exec_module(module)
        return module

    def load_from_source(
        self,
        *,
        filename: str,
        artifact_contracts: dict[int, Any] | None = None,
        source_bindings_config: SourceBindingsConfig | None = None,
        step_source_bindings_config: StepSourceBindingsConfig | None = None,
        semantic_contracts: tuple[ModuleArtifactContracts, ...] = (),
        semantic_contract_fingerprint: str | None = None,
    ) -> ModuleType:
        """Import generated pipeline code from source with the stable module name."""
        del (
            artifact_contracts,
            source_bindings_config,
            step_source_bindings_config,
            semantic_contracts,
            semantic_contract_fingerprint,
        )
        module = ModuleType(self.module_name)
        module.__file__ = filename
        sys.modules[self.module_name] = module
        exec(compile(self.identity.code, filename, "exec"), module.__dict__)
        return module

    def materialize_import_module(
        self,
        *,
        importable_path: Path,
        artifact_contracts: dict[int, Any] | None = None,
        source_bindings_config: SourceBindingsConfig | None = None,
        step_source_bindings_config: StepSourceBindingsConfig | None = None,
        semantic_contracts: tuple[ModuleArtifactContracts, ...] = (),
        semantic_contract_fingerprint: str | None = None,
    ) -> Path:
        """Write an importable module that restores generated declarations on import."""
        del (
            artifact_contracts,
            source_bindings_config,
            step_source_bindings_config,
            semantic_contracts,
            semantic_contract_fingerprint,
        )
        output_dir = importable_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        importable_source = (
            self.identity.code
            + "\n\n"
            + "if __name__ != '__main__':\n"
            + "    import sys as _openhcs_generated_sys\n"
            + "    from openhcs.interop.cellprofiler.runtime.generated_pipeline import "
            + "GeneratedPipelineFunctionRegistration as _openhcs_registration\n"
            + "    _openhcs_registration("
            + "_openhcs_generated_sys.modules[__name__]).register()\n"
        )
        if (
            not importable_path.exists()
            or importable_path.read_text(encoding="utf-8") != importable_source
        ):
            importable_path.write_text(importable_source, encoding="utf-8")
        output_dir_text = str(output_dir)
        if output_dir_text not in sys.path:
            sys.path.insert(0, output_dir_text)
        return importable_path

    def pipeline_from_module(
        self, module: ModuleType, *, pipeline_name: str
    ) -> Pipeline:
        """Build a Pipeline object from generated module exports."""
        pipeline_steps = GeneratedPipelineModuleExports(module).pipeline_steps
        if isinstance(pipeline_steps, Pipeline):
            pipeline = pipeline_steps
            pipeline.name = pipeline.name or pipeline_name
        elif isinstance(pipeline_steps, list):
            pipeline = Pipeline(steps=pipeline_steps, name=pipeline_name)
        else:
            raise TypeError(
                f"Generated module {module.__name__}.pipeline_steps must be list or "
                f"Pipeline, got {type(pipeline_steps).__name__}."
            )
        return pipeline


@dataclass(frozen=True, slots=True)
class CellProfilerGeneratedStepContract:
    """One executable generated step matched to its original CP module contract."""

    module_num: int
    contract: ModuleArtifactContract

    @property
    def planning_contract(self) -> ModuleArtifactContract:
        """Return the compile-time artifact declaration for this invocation."""
        return self.contract

    def validate_callable_metadata(self, metadata: Any) -> None:
        """Ensure generated step callable and contract describe the same CP module."""
        if metadata.module_name == self.contract.module_name:
            return
        raise ValueError(
            "Generated CellProfiler step callable does not match runtime "
            f"artifact contract for module {self.module_num}: callable "
            f"{metadata.module_name!r}, contract {self.contract.module_name!r}."
        )


@dataclass(frozen=True, slots=True)
class CellProfilerGeneratedGroupedStepContract(FunctionStepInvocationContractPayload):
    """Several source-grouped CP module contracts represented by one invocation."""

    contracts_by_group_key: Mapping[str, CellProfilerGeneratedStepContract]

    def __post_init__(self) -> None:
        normalized = {
            str(group_key): contract
            for group_key, contract in self.contracts_by_group_key.items()
        }
        if not normalized:
            raise ValueError("Grouped CellProfiler generated contract cannot be empty.")
        object.__setattr__(self, "contracts_by_group_key", normalized)

    @property
    def planning_contract(self) -> ModuleArtifactContract:
        """Return the aggregate contract visible to native artifact planning."""
        return CellProfilerGroupedModuleContracts(
            {
                group_key: contract.contract
                for group_key, contract in self.contracts_by_group_key.items()
            }
        ).planning_contract()

    def validate_callable_metadata(self, metadata: Any) -> None:
        """Ensure every grouped contract matches the public callable."""
        for contract in self.contracts_by_group_key.values():
            contract.validate_callable_metadata(metadata)


@dataclass(frozen=True, slots=True)
class CellProfilerGeneratedStepContracts:
    """Ordered executable-step contract stream for generated CP pipelines."""

    contracts_by_module_num: Mapping[int, ModuleArtifactContract]

    def ordered(self) -> "Iterator[CellProfilerGeneratedStepContract]":
        """Yield contracts in original CellProfiler module execution order."""
        return iter(
            CellProfilerGeneratedStepContract(module_num, contract)
            for module_num, contract in sorted(self.contracts_by_module_num.items())
        )


@dataclass(frozen=True, slots=True)
class CellProfilerGeneratedStepFunctionSpec:
    """CellProfiler runtime metadata projection for one FunctionStep func spec."""

    func_spec: Any

    def metadata(self) -> Any | None:
        metadata = tuple(self._metadata_items(self.func_spec))
        if not metadata:
            return None
        module_names = {item.module_name for item in metadata}
        if len(module_names) != 1:
            raise ValueError(
                "Generated CellProfiler FunctionStep mixes multiple module "
                f"callables: {sorted(module_names)!r}."
            )
        return metadata[0]

    def _metadata_items(self, func_spec: Any) -> Iterator[Any]:
        if isinstance(func_spec, CellProfilerRuntimeCallable):
            metadata = CellProfilerFunctionCatalog.runtime_metadata(func_spec.raw_func)
            if metadata is not None:
                yield metadata
            return
        if isinstance(func_spec, CellProfilerGroupedRuntimeCallable):
            metadata = CellProfilerFunctionCatalog.runtime_metadata(func_spec.raw_func)
            if metadata is not None:
                yield metadata
            return
        if isinstance(func_spec, FunctionReference):
            metadata = CellProfilerFunctionCatalog.runtime_metadata(func_spec.resolve())
            if metadata is not None:
                yield metadata
            return
        if callable(func_spec):
            metadata = CellProfilerFunctionCatalog.runtime_metadata(func_spec)
            if metadata is not None:
                yield metadata
            return
        if isinstance(func_spec, tuple) and len(func_spec) in {2, 3}:
            func = func_spec[0]
            yield from self._metadata_items(func)
            return
        if isinstance(func_spec, list):
            for item in func_spec:
                yield from self._metadata_items(item)
            return
        if isinstance(func_spec, dict):
            for item in func_spec.values():
                yield from self._metadata_items(item)


def _normalized_cellprofiler_invocation_items(func_spec: Any) -> tuple[Any, ...]:
    """Return normalized function items for generated runtime contract matching."""
    try:
        pattern = normalize_function_pattern(func_spec)
    except TypeError:
        return ()
    return tuple(pattern.iter_items())


def _cellprofiler_metadata_for_normalized_item(item: Any) -> Any | None:
    """Return CellProfiler callable metadata for one normalized function item."""
    raw_callable = _cellprofiler_public_callable(
        item.contract.resolve_runtime_callable()
    )
    return CellProfilerFunctionCatalog.runtime_metadata(raw_callable)


def _cellprofiler_public_callable(raw_callable: Callable[..., Any]) -> Callable[..., Any]:
    """Return the public CP callable represented by a runtime-bound wrapper."""
    if isinstance(raw_callable, CellProfilerRuntimeCallable):
        return raw_callable.raw_func
    if isinstance(raw_callable, CellProfilerGroupedRuntimeCallable):
        return raw_callable.raw_func
    return raw_callable


def _compile_time_consumed_kwarg_names(metadata: Any) -> tuple[str, ...]:
    """Return declaration-owned kwargs consumed by compile-time CP planning."""
    from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule

    module_type = CellProfilerModule.for_module(metadata.module_name)
    if module_type is None:
        return ()
    return module_type.compile_time_consumed_kwarg_names()


def _source_binding_group_keys(
    source_bindings: StepSourceBindingsConfig,
) -> tuple[str, ...]:
    """Return declared source-binding component values usable as group keys."""
    keys: list[str] = []
    for binding in source_bindings.binding_declarations:
        for selector in binding.component_identity:
            key = str(selector.value)
            if key not in keys:
                keys.append(key)
    return tuple(keys)


def _step_activates_pipeline_source_bindings(step: FunctionStep) -> bool:
    """Return whether generic OpenHCS input semantics should activate source aliases."""
    return step.processing_config.input_source == InputSource.PIPELINE_START


@dataclass(slots=True)
class CellProfilerGeneratedStepContractMatcher:
    """Match edited generated steps to their original CP module contracts."""

    contracts_by_module_num: Mapping[int, ModuleArtifactContract]
    _contracts: tuple[CellProfilerGeneratedStepContract, ...] = field(init=False)
    _matched_module_nums: set[int] = field(init=False)

    def __post_init__(self) -> None:
        self._contracts = tuple(
            CellProfilerGeneratedStepContracts(self.contracts_by_module_num).ordered()
        )
        self._matched_module_nums: set[int] = set()

    def match_public_invocation(
        self,
        metadata: Any,
        source_bindings: StepSourceBindingsConfig,
    ) -> CellProfilerGeneratedStepContract | CellProfilerGeneratedGroupedStepContract:
        """Match one public invocation to one or more grouped CP module contracts."""
        grouped = self.match_grouped(metadata, source_bindings)
        if grouped is not None:
            return grouped
        return self.match(metadata, source_bindings)

    def match_grouped(
        self,
        metadata: Any,
        source_bindings: StepSourceBindingsConfig,
    ) -> CellProfilerGeneratedGroupedStepContract | None:
        """Match repeated CP modules represented by one native default invocation."""
        candidates = self._unmatched_module_candidates(metadata.module_name)
        if len(candidates) <= 1:
            return None
        group_keys = _source_binding_group_keys(source_bindings)
        if not group_keys:
            return None
        if len(group_keys) < len(candidates):
            return None
        if any(
            not SourceBindingRuntimeContractGuard(
                candidate.contract,
                source_bindings,
            ).source_bound_input_keys
            for candidate in candidates
        ):
            return None

        grouped: dict[str, CellProfilerGeneratedStepContract] = {}
        used_module_nums: set[int] = set()
        for group_key in group_keys:
            scoped_source_bindings = source_bindings.for_group_key(group_key)
            aligned = tuple(
                candidate
                for candidate in candidates
                if candidate.module_num not in used_module_nums
                and SourceBindingRuntimeContractGuard(
                    candidate.contract,
                    scoped_source_bindings,
                )
                .alignment()
                .ok
            )
            if not aligned:
                continue
            if len(aligned) > 1:
                raise ValueError(
                    "Generated CellProfiler grouped invocation has ambiguous "
                    f"runtime artifact contracts for group {group_key!r}: "
                    f"{[candidate.module_num for candidate in aligned]!r}."
                )
            selected = aligned[0]
            selected.validate_callable_metadata(metadata)
            SourceBindingRuntimeContractGuard(
                selected.contract,
                scoped_source_bindings,
            ).validate()
            grouped[str(group_key)] = selected
            used_module_nums.add(selected.module_num)

        if len(used_module_nums) != len(candidates):
            return None

        self._matched_module_nums.update(used_module_nums)
        return CellProfilerGeneratedGroupedStepContract(grouped)

    def match(
        self,
        metadata: Any,
        source_bindings: StepSourceBindingsConfig,
    ) -> CellProfilerGeneratedStepContract:
        candidates = self._unmatched_module_candidates(metadata.module_name)
        if len(candidates) == 1:
            selected = candidates[0]
            selected.validate_callable_metadata(metadata)
            SourceBindingRuntimeContractGuard(
                selected.contract,
                source_bindings,
            ).validate_required()
            self._matched_module_nums.add(selected.module_num)
            return selected

        aligned = tuple(
            candidate
            for candidate in candidates
            if SourceBindingRuntimeContractGuard(
                candidate.contract,
                source_bindings,
            )
            .alignment()
            .ok
        )
        if not aligned:
            self._raise_no_matching_contract(metadata, source_bindings, candidates)
        selected = aligned[0]
        selected.validate_callable_metadata(metadata)
        SourceBindingRuntimeContractGuard(
            selected.contract,
            source_bindings,
        ).validate()
        self._matched_module_nums.add(selected.module_num)
        return selected

    def validate_complete(self) -> None:
        unmatched = tuple(
            contract
            for contract in self._contracts
            if contract.module_num not in self._matched_module_nums
        )
        if not unmatched:
            return
        summary = ", ".join(
            f"{contract.module_num} ({contract.contract.module_name!r})"
            for contract in unmatched
        )
        raise ValueError(
            "Generated CellProfiler runtime artifact contract has no matching "
            f"step for module(s): {summary}."
        )

    def _unmatched_module_candidates(
        self,
        module_name: str,
    ) -> tuple[CellProfilerGeneratedStepContract, ...]:
        return tuple(
            contract
            for contract in self._contracts
            if contract.module_num not in self._matched_module_nums
            and contract.contract.module_name == module_name
        )

    def _raise_no_matching_contract(
        self,
        metadata: Any,
        source_bindings: StepSourceBindingsConfig,
        candidates: tuple[CellProfilerGeneratedStepContract, ...],
    ) -> None:
        if not candidates:
            raise ValueError(
                "Generated CellProfiler step callable does not match any remaining "
                f"runtime artifact contract: callable {metadata.module_name!r}."
            )
        alignments = tuple(
            SourceBindingRuntimeContractGuard(
                candidate.contract,
                source_bindings,
            )
            .alignment()
            .message
            for candidate in candidates
        )
        raise ValueError(
            "Generated CellProfiler step source bindings drifted from its "
            "source-binding-compatible "
            f"runtime artifact contract: callable {metadata.module_name!r}; "
            f"candidate modules {[candidate.module_num for candidate in candidates]!r}; "
            f"alignment failures {alignments!r}."
        )


@dataclass(frozen=True, slots=True)
class CellProfilerGeneratedInvocationContractProvider:
    """Compile-time runtime contract provider for generated CellProfiler steps."""

    contracts_by_module_num: Mapping[int, ModuleArtifactContract]
    contracts_by_invocation_key: Mapping[
        tuple[int, FunctionInvocationKey],
        CellProfilerGeneratedStepContract | CellProfilerGeneratedGroupedStepContract,
    ] = field(default_factory=dict)
    contracts_by_step_index: Mapping[int, CellProfilerGeneratedStepContract] = field(
        default_factory=dict
    )

    @classmethod
    def for_steps(
        cls,
        contracts_by_module_num: Mapping[int, ModuleArtifactContract],
        steps: Sequence[Any],
        *,
        source_bindings_config: SourceBindingsConfig | None = None,
        step_source_bindings_config: StepSourceBindingsConfig | None = None,
    ) -> "CellProfilerGeneratedInvocationContractProvider":
        """Build a provider aligned to the actual generated FunctionStep stream."""
        return cls._for_step_specs(
            contracts_by_module_num,
            (
                (
                    step_index,
                    step.func,
                    resolve_effective_step_source_bindings(
                        step.source_bindings,
                        source_bindings_defaults=source_bindings_config,
                        step_source_bindings_defaults=step_source_bindings_config,
                        activate_source_bindings=(
                            _step_activates_pipeline_source_bindings(step)
                        ),
                    ),
                )
                for step_index, step in enumerate(steps)
                if isinstance(step, FunctionStep)
            ),
        )

    @classmethod
    def for_snapshots(
        cls,
        contracts_by_module_num: Mapping[int, ModuleArtifactContract],
        snapshots: Sequence[StepSnapshot],
        *,
        source_bindings_config: SourceBindingsConfig | None = None,
        step_source_bindings_config: StepSourceBindingsConfig | None = None,
    ) -> "CellProfilerGeneratedInvocationContractProvider":
        """Build a provider aligned to compiler StepSnapshot indices."""
        return cls._for_step_specs(
            contracts_by_module_num,
            (
                (
                    snapshot.index,
                    snapshot.func,
                    resolve_effective_step_source_bindings(
                        snapshot.source_bindings,
                        source_bindings_defaults=source_bindings_config,
                        step_source_bindings_defaults=step_source_bindings_config,
                        activate_source_bindings=(
                            snapshot.input_source == InputSource.PIPELINE_START
                        ),
                    ),
                )
                for snapshot in snapshots
                if snapshot.is_function_step
            ),
        )

    @classmethod
    def _for_step_specs(
        cls,
        contracts_by_module_num: Mapping[int, ModuleArtifactContract],
        step_specs: Iterator[tuple[int, Any, StepSourceBindingsConfig]],
    ) -> "CellProfilerGeneratedInvocationContractProvider":
        matcher = CellProfilerGeneratedStepContractMatcher(contracts_by_module_num)
        contracts_by_invocation_key: dict[
            tuple[int, FunctionInvocationKey],
            CellProfilerGeneratedStepContract | CellProfilerGeneratedGroupedStepContract,
        ] = {}
        contracts_by_step_index: dict[int, CellProfilerGeneratedStepContract] = {}
        for step_index, func_spec, source_bindings in step_specs:
            step_matches: list[CellProfilerGeneratedStepContract] = []
            for item in _normalized_cellprofiler_invocation_items(func_spec):
                metadata = _cellprofiler_metadata_for_normalized_item(item)
                if metadata is None:
                    continue
                scoped_source_bindings = source_bindings.for_group_key(
                    item.key.group_key,
                )
                matched_contract = matcher.match_public_invocation(
                    metadata,
                    scoped_source_bindings,
                )
                contracts_by_invocation_key[(step_index, item.key)] = matched_contract
                if isinstance(matched_contract, CellProfilerGeneratedStepContract):
                    step_matches.append(matched_contract)
            if len(step_matches) == 1:
                contracts_by_step_index[step_index] = step_matches[0]
        matcher.validate_complete()
        return cls(
            contracts_by_module_num=contracts_by_module_num,
            contracts_by_invocation_key=contracts_by_invocation_key,
            contracts_by_step_index=contracts_by_step_index,
        )

    def __call__(
        self,
        invocation: Any,
        step_context: ArtifactDeclarationStepContext,
    ) -> InvocationContractPlan | None:
        raw_callable = _cellprofiler_public_callable(
            invocation.contract.resolve_runtime_callable()
        )

        metadata = CellProfilerFunctionCatalog.runtime_metadata(raw_callable)
        if metadata is None:
            return None

        step_contract = self.step_contract_for(
            metadata,
            step_context,
            invocation.key,
        )
        if isinstance(step_contract, CellProfilerGeneratedGroupedStepContract):
            runtime_callable = CellProfilerGroupedRuntimeStepBinding(
                raw_callable=raw_callable,
                contracts_by_group_key={
                    group_key: contract.contract
                    for group_key, contract in step_contract.contracts_by_group_key.items()
                },
                processing_contract=metadata.processing_contract,
                declared_processing_contract=metadata.declared_processing_contract,
            ).load()
        else:
            runtime_callable = CellProfilerRuntimeStepBinding(
                raw_callable=raw_callable,
                contract=step_contract.contract,
                processing_contract=metadata.processing_contract,
                declared_processing_contract=metadata.declared_processing_contract,
            ).load()
        return InvocationContractPlan(
            CallableContract.from_callable(runtime_callable),
            consumed_kwarg_names=_compile_time_consumed_kwarg_names(metadata),
        )

    def step_contract_for(
        self,
        metadata: Any,
        step_context: ArtifactDeclarationStepContext,
        invocation_key: FunctionInvocationKey | None = None,
    ) -> CellProfilerGeneratedStepContract | CellProfilerGeneratedGroupedStepContract:
        mapped = self._mapped_step_contract(metadata, step_context, invocation_key)
        if mapped is not None:
            return mapped

        indexed = self._indexed_contract(metadata, step_context)
        if indexed is not None:
            return indexed

        aligned = tuple(
            candidate
            for candidate in self._module_candidates(metadata.module_name)
            if SourceBindingRuntimeContractGuard(
                candidate.contract,
                step_context.source_bindings,
            )
            .alignment()
            .ok
        )
        if len(aligned) == 1:
            return aligned[0]
        if not aligned:
            return CellProfilerGeneratedStepContractMatcher(
                self.contracts_by_module_num
            ).match_public_invocation(metadata, step_context.source_bindings)
        raise ValueError(
            "Generated CellProfiler step has ambiguous runtime artifact "
            f"contracts for module {metadata.module_name!r}. "
            f"Matched module numbers {[candidate.module_num for candidate in aligned]!r}; "
            "keep generated steps in module order or make source bindings "
            "distinguish the repeated module instances."
        )

    def _mapped_step_contract(
        self,
        metadata: Any,
        step_context: ArtifactDeclarationStepContext,
        invocation_key: FunctionInvocationKey | None,
    ) -> CellProfilerGeneratedStepContract | CellProfilerGeneratedGroupedStepContract | None:
        step_index = step_context.step_index
        if step_index is None:
            return None
        if invocation_key is not None:
            candidate = self.contracts_by_invocation_key.get(
                (step_index, invocation_key)
            )
            if candidate is not None:
                if isinstance(candidate, CellProfilerGeneratedGroupedStepContract):
                    candidate.validate_callable_metadata(metadata)
                    return candidate
                if candidate.contract.module_name != metadata.module_name:
                    return None
                scoped_source_bindings = step_context.source_bindings.for_group_key(
                    invocation_key.group_key,
                )
                alignment = SourceBindingRuntimeContractGuard(
                    candidate.contract,
                    scoped_source_bindings,
                ).alignment()
                if not alignment.ok:
                    return None
                candidate.validate_callable_metadata(metadata)
                return candidate
        candidate = self.contracts_by_step_index.get(step_index)
        if candidate is None:
            return None
        if candidate.contract.module_name != metadata.module_name:
            return None
        alignment = SourceBindingRuntimeContractGuard(
            candidate.contract,
            step_context.source_bindings,
        ).alignment()
        if not alignment.ok:
            return None
        candidate.validate_callable_metadata(metadata)
        return candidate

    def _indexed_contract(
        self,
        metadata: Any,
        step_context: ArtifactDeclarationStepContext,
    ) -> CellProfilerGeneratedStepContract | None:
        step_index = step_context.step_index
        if step_index is None:
            return None
        ordered = tuple(
            CellProfilerGeneratedStepContracts(self.contracts_by_module_num).ordered()
        )
        if not 0 <= step_index < len(ordered):
            return None
        candidate = ordered[step_index]
        if candidate.contract.module_name != metadata.module_name:
            return None
        alignment = SourceBindingRuntimeContractGuard(
            candidate.contract,
            step_context.source_bindings,
        ).alignment()
        if not alignment.ok:
            return None
        candidate.validate_callable_metadata(metadata)
        return candidate

    def _module_candidates(
        self,
        module_name: str,
    ) -> tuple[CellProfilerGeneratedStepContract, ...]:
        return tuple(
            contract
            for contract in CellProfilerGeneratedStepContracts(
                self.contracts_by_module_num
            ).ordered()
            if contract.contract.module_name == module_name
        )


@dataclass(frozen=True, slots=True)
class GeneratedPipelineModuleExports:
    """Typed access to generated module exports."""

    module: ModuleType

    @property
    def pipeline_steps(self) -> Any:
        return self.module.pipeline_steps

    @property
    def step_callables(self) -> tuple[Callable[..., Any], ...]:
        callables: list[Callable[..., Any]] = []
        seen: set[int] = set()
        for step in self.pipeline_steps:
            for func in GeneratedFunctionSpec(step.func).callables:
                func_id = id(func)
                if func_id in seen:
                    continue
                seen.add(func_id)
                callables.append(func)
        return tuple(callables)


@dataclass(frozen=True, slots=True)
class GeneratedFunctionSpec:
    """Callable extraction for generated FunctionStep function specifications."""

    func_spec: Any

    @property
    def callables(self) -> tuple[Callable[..., Any], ...]:
        callables = tuple(_function_spec_callables(self.func_spec))
        if callables:
            return callables
        raise TypeError(
            "Unsupported generated FunctionStep func spec "
            f"{type(self.func_spec).__name__}."
        )


def _function_spec_callables(func_spec: Any) -> Iterator[Callable[..., Any]]:
    """Yield public callables from generated FunctionStep syntax."""
    if callable(func_spec):
        yield func_spec
        return
    if isinstance(func_spec, tuple) and len(func_spec) in {2, 3}:
        func = func_spec[0]
        if callable(func):
            yield func
        return
    if isinstance(func_spec, list):
        for item in func_spec:
            yield from _function_spec_callables(item)
        return
    if isinstance(func_spec, dict):
        for item in func_spec.values():
            yield from _function_spec_callables(item)


@dataclass(frozen=True, slots=True)
class GeneratedPipelineFunctionRegistration:
    """OpenHCS registry registration authority for generated public callables."""

    module: ModuleType

    def register(self) -> tuple[str, ...]:
        registry = OpenHCSRegistry()
        existing_references = {
            (
                inspect.unwrap(metadata.func).__module__,
                inspect.unwrap(metadata.func).__name__,
            )
            for metadata in RegistryService.get_all_functions_with_metadata().values()
        }
        registered_names: list[str] = []
        registered_new_function = False

        for func in GeneratedPipelineModuleExports(self.module).step_callables:
            reference = (
                inspect.unwrap(func).__module__,
                inspect.unwrap(func).__name__,
            )
            metadata_name = GeneratedPipelineFunction(func).metadata_name
            if reference in existing_references:
                registered_names.append(metadata_name)
                continue

            contract = GeneratedPipelineFunction(func).processing_contract
            vars(func)[FunctionContractAttribute.processing_contract] = contract
            wrapped_func = registry.apply_contract_wrapper(func, contract)
            vars(wrapped_func)[FunctionContractAttribute.processing_contract] = contract
            callable_contract = CallableContract.from_callable(wrapped_func)
            wrapped_func.__function_metadata__ = FunctionMetadata(
                name=metadata_name,
                func=wrapped_func,
                contract=contract,
                registry=registry,
                module=GeneratedPipelineFunction(wrapped_func).module_name,
                doc=GeneratedPipelineFunction(wrapped_func).documentation,
                tags=["openhcs", "generated", "cellprofiler"],
                original_name=wrapped_func.__name__,
                memory_type=callable_contract.input_memory_type,
            )
            register_function(wrapped_func, backend="openhcs")
            existing_references.add(reference)
            registered_names.append(metadata_name)
            registered_new_function = True

        if registered_new_function:
            RegistryService.clear_metadata_cache()
        return tuple(registered_names)


@dataclass(frozen=True, slots=True)
class GeneratedPipelineFunction:
    """Typed metadata projection for one generated runtime function."""

    func: Callable[..., Any]

    @property
    def processing_contract(self) -> ProcessingContract:
        contract = CallableContract.from_callable(self.func)
        if isinstance(contract.processing_contract, ProcessingContract):
            return contract.processing_contract
        raise TypeError(
            f"Generated function {contract.function_name!r} has no nominal "
            "__processing_contract__ metadata. Coerce declared contracts during "
            "callable metadata attachment before registry registration."
        )

    @property
    def metadata_name(self) -> str:
        return f"{self.func.__module__}:{self.func.__name__}"

    @property
    def module_name(self) -> str:
        module_name = self.func.__module__
        if not isinstance(module_name, str) or not module_name:
            raise ValueError(
                f"Generated function {self.func.__name__!r} has no module name."
            )
        return module_name

    @property
    def documentation(self) -> str:
        doc = self.func.__doc__
        if doc is None:
            return ""
        return doc


def generated_pipeline_module_name(module_path: Path, code: str) -> str:
    """Compatibility facade for generated module identity projection."""
    return GeneratedPipelineModuleIdentity(
        module_path=module_path, code=code
    ).module_name


def load_generated_pipeline_module(
    module_path: Path,
    *,
    module_name: str,
) -> ModuleType:
    """Compatibility facade for importing generated code from disk."""
    return GeneratedPipelineRuntimeModule(
        GeneratedPipelineModuleIdentity(
            module_path=module_path,
            code="",
            explicit_module_name=module_name,
        )
    ).load_from_path(module_path)


def load_generated_pipeline_module_from_source(
    source: str,
    *,
    module_name: str,
    filename: str,
    artifact_contracts: dict[int, Any] | None = None,
    source_bindings_config: SourceBindingsConfig | None = None,
    step_source_bindings_config: StepSourceBindingsConfig | None = None,
    semantic_contracts: tuple[ModuleArtifactContracts, ...] = (),
    semantic_contract_fingerprint: str | None = None,
) -> ModuleType:
    """Compatibility facade for importing generated code from source."""
    module_path = Path(filename)
    return GeneratedPipelineRuntimeModule(
        GeneratedPipelineModuleIdentity(
            module_path=module_path,
            code=source,
            explicit_module_name=module_name,
        )
    ).load_from_source(
        filename=filename,
        artifact_contracts=artifact_contracts,
        source_bindings_config=source_bindings_config,
        step_source_bindings_config=step_source_bindings_config,
        semantic_contracts=semantic_contracts,
        semantic_contract_fingerprint=semantic_contract_fingerprint,
    )


def materialize_generated_pipeline_import_module(
    source: str,
    *,
    module_name: str,
    output_dir: Path,
    artifact_contracts: dict[int, Any] | None = None,
    source_bindings_config: SourceBindingsConfig | None = None,
    step_source_bindings_config: StepSourceBindingsConfig | None = None,
    semantic_contracts: tuple[ModuleArtifactContracts, ...] = (),
    semantic_contract_fingerprint: str | None = None,
) -> Path:
    """Compatibility facade for generated module materialization."""
    module_path = output_dir / f"{module_name}.py"
    return GeneratedPipelineRuntimeModule(
        GeneratedPipelineModuleIdentity(
            module_path=module_path,
            code=source,
            explicit_module_name=module_name,
        )
    ).materialize_import_module(
        importable_path=module_path,
        artifact_contracts=artifact_contracts,
        source_bindings_config=source_bindings_config,
        step_source_bindings_config=step_source_bindings_config,
        semantic_contracts=semantic_contracts,
        semantic_contract_fingerprint=semantic_contract_fingerprint,
    )


def pipeline_from_generated_module(
    module: ModuleType,
    *,
    pipeline_name: str,
) -> Pipeline:
    """Compatibility facade for building a Pipeline from generated exports."""
    module_file = module.__file__
    if module_file is None:
        raise ValueError(
            f"Generated module {module.__name__!r} has no __file__ for identity."
        )
    return GeneratedPipelineRuntimeModule(
        GeneratedPipelineModuleIdentity(module_path=Path(module_file), code="")
    ).pipeline_from_module(module, pipeline_name=pipeline_name)
