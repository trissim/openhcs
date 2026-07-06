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
from typing import Any, ClassVar

from openhcs.core.callable_contract import CallableContract
from openhcs.core.function_contract_metadata import FunctionContractAttribute
from openhcs.core.function_reference import FunctionReference
from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
from openhcs.core.artifact_contract_preview import SourceBindingRuntimeContractGuard
from openhcs.core.pipeline import Pipeline
from openhcs.core.pipeline.step_snapshot import StepSnapshot
from openhcs.core.source_bindings import StepSourceBindingsConfig
from openhcs.core.steps.function_step import FunctionStep
from openhcs.processing.backends.lib_registry.openhcs_registry import OpenHCSRegistry
from openhcs.processing.backends.lib_registry.registry_service import RegistryService
from openhcs.processing.backends.lib_registry.unified_registry import (
    FunctionMetadata,
    ProcessingContract,
)
from openhcs.processing.func_registry import register_function
from openhcs.interop.cellprofiler.runtime.module_execution import (
    CellProfilerRuntimeCallable,
    CellProfilerRuntimeStepBinding,
)
from openhcs.interop.cellprofiler.symbol_table import ModuleArtifactContracts
from openhcs.core.module_artifact_contract import ModuleArtifactContract
from openhcs.processing.backends.cellprofiler import (
    CellProfilerFunctionCatalog,
)


@dataclass(frozen=True, slots=True)
class GeneratedPipelineContractSidecar:
    """Generated Python persistence for CellProfiler runtime artifact contracts."""

    contracts_by_module_num: Mapping[int, ModuleArtifactContract]
    export_name: ClassVar[str] = "CELLPROFILER_ARTIFACT_CONTRACTS"

    def write(self, path: Path) -> None:
        """Write runtime artifact contracts as importable Python source."""
        import openhcs.serialization.pycodify_formatters  # noqa: F401
        from pycodify import Assignment, generate_python_source

        source = generate_python_source(
            Assignment(
                self.export_name,
                dict(sorted(self.contracts_by_module_num.items())),
            ),
            header="# Generated CellProfiler runtime artifact contracts.",
            clean_mode=False,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_text(encoding="utf-8") != source:
            path.write_text(source, encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> dict[int, ModuleArtifactContract]:
        """Load runtime artifact contracts from a generated Python sidecar."""
        sidecar_path = Path(path)
        module_name = f"_openhcs_{sidecar_path.stem}_artifact_contracts"
        spec = importlib.util.spec_from_file_location(module_name, sidecar_path)
        if spec is None or spec.loader is None:
            raise ImportError(
                f"Unable to create artifact contract module spec for {sidecar_path}."
            )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        contracts = vars(module)[cls.export_name]
        return {
            int(module_num): contract
            for module_num, contract in dict(contracts).items()
        }


@dataclass(frozen=True, slots=True)
class GeneratedPipelineSemanticContractsModule:
    """Generated Python persistence for CellProfiler semantic contracts."""

    contracts: tuple[ModuleArtifactContracts, ...]
    fingerprint: str | None = None
    export_name: ClassVar[str] = "CELLPROFILER_SEMANTIC_CONTRACTS"
    fingerprint_export_name: ClassVar[str] = (
        "CELLPROFILER_SEMANTIC_CONTRACT_FINGERPRINT"
    )

    def __post_init__(self) -> None:
        for contract in self.contracts:
            if not isinstance(contract, ModuleArtifactContracts):
                raise TypeError(
                    "GeneratedPipelineSemanticContractsModule requires "
                    f"ModuleArtifactContracts values, got {type(contract).__name__}."
                )

    def write(self, path: Path) -> None:
        """Write semantic contracts as importable Python source."""
        import openhcs.serialization.pycodify_formatters  # noqa: F401
        from pycodify import Assignment, CodeBlock, generate_python_source

        assignments = [
            Assignment(self.export_name, self.contracts),
            Assignment(self.fingerprint_export_name, self.fingerprint),
        ]

        source = generate_python_source(
            CodeBlock(tuple(assignments)),
            header="# Generated CellProfiler semantic artifact contracts.",
            clean_mode=False,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_text(encoding="utf-8") != source:
            path.write_text(source, encoding="utf-8")

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_fingerprint: str | None = None,
    ) -> tuple[ModuleArtifactContracts, ...]:
        """Load semantic contracts from a generated Python sidecar."""
        sidecar_path = Path(path)
        module_name = f"_openhcs_{sidecar_path.stem}_semantic_contracts"
        spec = importlib.util.spec_from_file_location(module_name, sidecar_path)
        if spec is None or spec.loader is None:
            raise ImportError(
                f"Unable to create semantic contract module spec for {sidecar_path}."
            )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        contracts = module.CELLPROFILER_SEMANTIC_CONTRACTS
        actual_fingerprint = module.CELLPROFILER_SEMANTIC_CONTRACT_FINGERPRINT
        if (
            expected_fingerprint is not None
            and actual_fingerprint != expected_fingerprint
        ):
            raise ValueError(
                "Generated CellProfiler semantic contract sidecar fingerprint "
                f"mismatch for {sidecar_path}."
            )
        normalized = tuple(contracts)
        cls(normalized, fingerprint=actual_fingerprint)
        return normalized


@dataclass(frozen=True, slots=True)
class GeneratedPipelineSemanticContractsFingerprint:
    """Stable fingerprint for generated semantic contract sidecars."""

    value: str

    @classmethod
    def from_generation(
        cls,
        *,
        source_cppipe: Path | None,
        generated_code: str,
        semantic_contracts: Sequence[ModuleArtifactContracts],
    ) -> "GeneratedPipelineSemanticContractsFingerprint":
        import openhcs.serialization.pycodify_formatters  # noqa: F401
        from pycodify import Assignment, generate_python_source

        semantic_source = generate_python_source(
            Assignment("contracts", tuple(semantic_contracts)),
            clean_mode=False,
        )
        payload_parts = (
            cls._source_cppipe_digest(source_cppipe),
            hashlib.sha256(generated_code.encode("utf-8")).hexdigest(),
            hashlib.sha256(semantic_source.encode("utf-8")).hexdigest(),
        )
        return cls(hashlib.sha256("::".join(payload_parts).encode("utf-8")).hexdigest())

    @staticmethod
    def _source_cppipe_digest(source_cppipe: Path | None) -> str:
        if source_cppipe is None:
            return ""
        path = Path(source_cppipe)
        if not path.exists():
            return hashlib.sha256(str(path).encode("utf-8")).hexdigest()
        return hashlib.sha256(path.read_bytes()).hexdigest()


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
        semantic_contracts: tuple[ModuleArtifactContracts, ...] = (),
        semantic_contract_fingerprint: str | None = None,
    ) -> ModuleType:
        """Import generated pipeline code from source with the stable module name."""
        module = ModuleType(self.module_name)
        module.__file__ = filename
        sys.modules[self.module_name] = module
        exec(compile(self.identity.code, filename, "exec"), module.__dict__)
        module.CELLPROFILER_SEMANTIC_CONTRACTS = tuple(semantic_contracts)
        module.CELLPROFILER_SEMANTIC_CONTRACT_FINGERPRINT = (
            semantic_contract_fingerprint
        )
        if artifact_contracts:
            setattr(
                module,
                CellProfilerGeneratedPipelineInvocationContracts.module_attribute,
                CellProfilerGeneratedPipelineInvocationContracts.from_mapping(
                    artifact_contracts
                ).contracts_by_module_num,
            )
        return module

    def materialize_import_module(
        self,
        *,
        importable_path: Path,
        artifact_contracts: dict[int, Any] | None = None,
        semantic_contracts: tuple[ModuleArtifactContracts, ...] = (),
        semantic_contract_fingerprint: str | None = None,
    ) -> Path:
        """Write an importable module that restores generated declarations on import."""
        output_dir = importable_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        contract_sidecar = output_dir / (
            f"{self.module_name}.cellprofiler_contracts.py"
        )
        contract_prelude = ""
        if artifact_contracts:
            GeneratedPipelineContractSidecar(artifact_contracts).write(contract_sidecar)
            contract_prelude = (
                "    from openhcs.interop.cellprofiler.runtime.generated_pipeline import "
                "GeneratedPipelineContractSidecar as _openhcs_cp_contract_sidecar\n"
                "    _openhcs_cp_contract_values = _openhcs_cp_contract_sidecar.read("
                f"{str(contract_sidecar)!r})\n"
            )
        semantic_sidecar = output_dir / (
            f"{self.module_name}.cellprofiler_semantic_contracts.py"
        )
        semantic_prelude = "    CELLPROFILER_SEMANTIC_CONTRACTS = ()\n"
        if semantic_contracts:
            if semantic_contract_fingerprint is None:
                raise ValueError(
                    "Generated semantic contract sidecars require a fingerprint."
                )
            GeneratedPipelineSemanticContractsModule(
                semantic_contracts,
                fingerprint=semantic_contract_fingerprint,
            ).write(semantic_sidecar)
            semantic_prelude = (
                "    from openhcs.interop.cellprofiler.runtime.generated_pipeline import "
                "GeneratedPipelineSemanticContractsModule as _openhcs_cp_semantic_contracts\n"
                "    CELLPROFILER_SEMANTIC_CONTRACTS = "
                "_openhcs_cp_semantic_contracts.load("
                f"{str(semantic_sidecar)!r}, "
                f"expected_fingerprint={semantic_contract_fingerprint!r})\n"
                "    CELLPROFILER_SEMANTIC_CONTRACT_FINGERPRINT = "
                f"{semantic_contract_fingerprint!r}\n"
            )
        importable_source = (
            self.identity.code
            + "\n\n"
            + "if __name__ != '__main__':\n"
            + contract_prelude
            + semantic_prelude
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
        invocation_contracts = (
            CellProfilerGeneratedPipelineInvocationContracts.from_module(module)
        )
        if invocation_contracts is not None:
            pipeline.metadata[
                CellProfilerGeneratedPipelineInvocationContracts.module_attribute
            ] = invocation_contracts.contracts_by_module_num
        semantic_contracts = getattr(module, "CELLPROFILER_SEMANTIC_CONTRACTS", None)
        if semantic_contracts is not None:
            pipeline.metadata["CELLPROFILER_SEMANTIC_CONTRACTS"] = semantic_contracts
        return pipeline


def bind_generated_pipeline_runtime(
    module: ModuleType,
    artifact_contracts: Mapping[int, Any],
) -> None:
    """Apply product-owned runtime wrappers to imported generated FunctionSteps."""
    normalized: dict[int, ModuleArtifactContract] = {}
    for module_num, contract in artifact_contracts.items():
        if not isinstance(contract, ModuleArtifactContract):
            raise TypeError(
                "Generated CellProfiler artifact contracts must be "
                f"ModuleArtifactContract values, got {type(contract).__name__}."
            )
        normalized[int(module_num)] = contract
    GeneratedPipelineRuntimeBindings(module, normalized).apply()


@dataclass(frozen=True, slots=True)
class GeneratedPipelineRuntimeBindings:
    """Artifact-aware binding authority for imported generated pipeline modules."""

    module: ModuleType
    artifact_contracts: Mapping[int, ModuleArtifactContract]

    def apply(self) -> None:
        """Replace direct backend callables with artifact-managed runtime callables."""
        if not self.artifact_contracts:
            return
        if CellProfilerGeneratedRuntimeBindingState(
            GeneratedPipelineModuleExports(self.module).pipeline_steps,
            self.artifact_contracts,
        ).matches_expected_contracts():
            return

        contract_matcher = CellProfilerGeneratedStepContractMatcher(
            self.artifact_contracts
        )
        for step in GeneratedPipelineModuleExports(self.module).pipeline_steps:
            if not isinstance(step, FunctionStep):
                raise TypeError(
                    "Generated CellProfiler pipeline steps must be FunctionStep "
                    f"instances, got {type(step).__name__}."
                )
            metadata = CellProfilerGeneratedStepFunctionSpec(step.func).metadata()
            if metadata is None:
                continue
            step_contract = contract_matcher.match(
                metadata,
                step.source_bindings,
            )
            step.func = self._bind_func_spec(
                step.func,
                step_contract,
                step.source_bindings,
            )
        contract_matcher.validate_complete()

    def _bind_func_spec(
        self,
        func_spec: Any,
        step_contract: "CellProfilerGeneratedStepContract",
        source_bindings: StepSourceBindingsConfig,
    ) -> Any:
        if callable(func_spec):
            return self._bind_callable(
                func_spec,
                step_contract,
                source_bindings,
            )
        if isinstance(func_spec, tuple) and len(func_spec) in {2, 3}:
            bound_callable = self._bind_callable(
                func_spec[0],
                step_contract,
                source_bindings,
            )
            kwargs = self._runtime_kwargs(func_spec[1])
            tail = func_spec[2:]
            if kwargs or tail:
                return (bound_callable, kwargs, *tail)
            return bound_callable
        if isinstance(func_spec, list):
            return [
                self._bind_func_spec(
                    item,
                    step_contract,
                    source_bindings,
                )
                for item in func_spec
            ]
        return func_spec

    @staticmethod
    def _runtime_kwargs(kwargs: Any) -> dict:
        if not isinstance(kwargs, dict):
            raise TypeError(
                "Generated CellProfiler tuple function specs must carry a dict "
                f"of kwargs, got {type(kwargs).__name__}."
            )
        return dict(kwargs)

    def _bind_callable(
        self,
        func: Callable[..., Any],
        step_contract: "CellProfilerGeneratedStepContract",
        source_bindings: StepSourceBindingsConfig,
    ) -> Callable[..., Any]:
        metadata = CellProfilerFunctionCatalog.runtime_metadata(func)
        if metadata is None:
            return func
        step_contract.validate_callable_metadata(metadata)
        contract = step_contract.contract
        SourceBindingRuntimeContractGuard(
            contract,
            source_bindings,
        ).validate()
        return CellProfilerRuntimeStepBinding(
            raw_callable=func,
            contract=contract,
            processing_contract=metadata.processing_contract,
            declared_processing_contract=metadata.declared_processing_contract,
        ).load()


@dataclass(frozen=True, slots=True)
class CellProfilerGeneratedRuntimeBindingState:
    """Detect whether generated CP steps already use runtime-bound callables."""

    pipeline_steps: Sequence[Any]
    contracts_by_module_num: Mapping[int, ModuleArtifactContract]

    def matches_expected_contracts(self) -> bool:
        unmatched_contracts = [
            contract.contract
            for contract in CellProfilerGeneratedStepContracts(
                self.contracts_by_module_num
            ).ordered()
        ]
        actual_contracts: list[ModuleArtifactContract] = []
        for step in self.pipeline_steps:
            if not isinstance(step, FunctionStep):
                continue
            for func in self.function_spec_callables(step.func):
                if isinstance(func, self.runtime_callable_type()):
                    actual_contracts.append(func.contract)
        if len(actual_contracts) != len(unmatched_contracts):
            return False
        for actual_contract in actual_contracts:
            for index, expected_contract in enumerate(unmatched_contracts):
                if actual_contract == expected_contract:
                    del unmatched_contracts[index]
                    break
            else:
                return False
        return not unmatched_contracts

    @classmethod
    def pipeline_requires_rebinding(cls, pipeline_steps: Sequence[Any]) -> bool:
        """Return whether any generated CellProfiler step still uses raw callables."""
        for step in pipeline_steps:
            if not isinstance(step, FunctionStep):
                continue
            if CellProfilerGeneratedStepFunctionSpec(step.func).metadata() is None:
                continue
            if not cls.step_has_runtime_bound_callable(step):
                return True
        return False

    @classmethod
    def step_has_runtime_bound_callable(cls, step: FunctionStep) -> bool:
        """Return whether one generated step already carries runtime contracts."""
        return any(
            isinstance(func, cls.runtime_callable_type())
            for func in cls.function_spec_callables(step.func)
        )

    @staticmethod
    def runtime_callable_type() -> type:
        from openhcs.interop.cellprofiler.runtime.module_execution import (
            CellProfilerRuntimeCallable,
        )

        return CellProfilerRuntimeCallable

    @classmethod
    def function_spec_callables(cls, func_spec: Any) -> Iterator[Callable[..., Any]]:
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
                yield from cls.function_spec_callables(item)


@dataclass(frozen=True, slots=True)
class CellProfilerPipelineRuntimeRebinder:
    """Re-derive runtime CellProfiler callables from generated-pipeline contracts."""

    generated_module_name: str
    contracts_by_module_num: Mapping[int, ModuleArtifactContract]

    @classmethod
    def from_import_result(
        cls,
        import_result: Any,
    ) -> "CellProfilerPipelineRuntimeRebinder":
        return cls(
            generated_module_name=import_result.generated_module_name,
            contracts_by_module_num={
                semantic_contract.module_num: artifact_contract
                for semantic_contract, artifact_contract in zip(
                    import_result.semantic_contracts,
                    import_result.artifact_contracts,
                    strict=True,
                )
            },
        )

    def rebind(self, pipeline_steps: Sequence[Any]) -> list[Any]:
        """Return steps with raw CellProfiler functions rebound to runtime callables."""
        pipeline_steps = list(pipeline_steps)
        if CellProfilerGeneratedRuntimeBindingState(
            pipeline_steps,
            self.contracts_by_module_num,
        ).matches_expected_contracts():
            return pipeline_steps
        module = ModuleType(self.generated_module_name)
        module.pipeline_steps = pipeline_steps
        GeneratedPipelineRuntimeBindings(
            module,
            self.contracts_by_module_num,
        ).apply()
        return list(module.pipeline_steps)


@dataclass(frozen=True, slots=True)
class CellProfilerGeneratedStepContract:
    """One executable generated step matched to its original CP module contract."""

    module_num: int
    contract: ModuleArtifactContract

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

    def match(
        self,
        metadata: Any,
        source_bindings: StepSourceBindingsConfig,
    ) -> CellProfilerGeneratedStepContract:
        candidates = self._unmatched_module_candidates(metadata.module_name)
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
    contracts_by_step_index: Mapping[int, CellProfilerGeneratedStepContract] = field(
        default_factory=dict
    )

    @classmethod
    def for_steps(
        cls,
        contracts_by_module_num: Mapping[int, ModuleArtifactContract],
        steps: Sequence[Any],
    ) -> "CellProfilerGeneratedInvocationContractProvider":
        """Build a provider aligned to the actual generated FunctionStep stream."""
        return cls._for_step_specs(
            contracts_by_module_num,
            (
                (step_index, step.func, step.source_bindings)
                for step_index, step in enumerate(steps)
                if isinstance(step, FunctionStep)
            ),
        )

    @classmethod
    def for_snapshots(
        cls,
        contracts_by_module_num: Mapping[int, ModuleArtifactContract],
        snapshots: Sequence[StepSnapshot],
    ) -> "CellProfilerGeneratedInvocationContractProvider":
        """Build a provider aligned to compiler StepSnapshot indices."""
        return cls._for_step_specs(
            contracts_by_module_num,
            (
                (snapshot.index, snapshot.func, snapshot.source_bindings)
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
        contracts_by_step_index: dict[int, CellProfilerGeneratedStepContract] = {}
        for step_index, func_spec, source_bindings in step_specs:
            metadata = CellProfilerGeneratedStepFunctionSpec(func_spec).metadata()
            if metadata is None:
                continue
            contracts_by_step_index[step_index] = matcher.match(
                metadata,
                source_bindings,
            )
        matcher.validate_complete()
        return cls(
            contracts_by_module_num=contracts_by_module_num,
            contracts_by_step_index=contracts_by_step_index,
        )

    def __call__(
        self,
        invocation: Any,
        step_context: ArtifactDeclarationStepContext,
    ) -> CallableContract | None:
        raw_callable = invocation.contract.resolve_runtime_callable()
        if isinstance(raw_callable, CellProfilerRuntimeCallable):
            return invocation.contract

        metadata = CellProfilerFunctionCatalog.runtime_metadata(raw_callable)
        if metadata is None:
            return None

        step_contract = self.step_contract_for(
            metadata,
            step_context,
        )
        runtime_callable = CellProfilerRuntimeStepBinding(
            raw_callable=raw_callable,
            contract=step_contract.contract,
            processing_contract=metadata.processing_contract,
            declared_processing_contract=metadata.declared_processing_contract,
        ).load()
        return CallableContract.from_callable(runtime_callable)

    def step_contract_for(
        self,
        metadata: Any,
        step_context: ArtifactDeclarationStepContext,
    ) -> CellProfilerGeneratedStepContract:
        mapped = self._mapped_step_contract(metadata, step_context)
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
            ).match(metadata, step_context.source_bindings)
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
    ) -> CellProfilerGeneratedStepContract | None:
        step_index = step_context.step_index
        if step_index is None:
            return None
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
class CellProfilerGeneratedPipelineInvocationContracts:
    """Pipeline metadata projection for generated CP runtime contracts."""

    module_attribute: ClassVar[str] = "_openhcs_cp_contract_values"
    contracts_by_module_num: Mapping[int, ModuleArtifactContract]

    @classmethod
    def from_mapping(
        cls,
        contracts_by_module_num: Mapping[int, ModuleArtifactContract],
    ) -> "CellProfilerGeneratedPipelineInvocationContracts":
        normalized: dict[int, ModuleArtifactContract] = {}
        for module_num, contract in contracts_by_module_num.items():
            if not isinstance(contract, ModuleArtifactContract):
                raise TypeError(
                    "Generated CellProfiler pipeline metadata requires "
                    "ModuleArtifactContract values, got "
                    f"{type(contract).__name__}."
                )
            normalized[int(module_num)] = contract
        return cls(normalized)

    @classmethod
    def from_module(
        cls,
        module: ModuleType,
    ) -> "CellProfilerGeneratedPipelineInvocationContracts | None":
        value = vars(module).get(cls.module_attribute)
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise TypeError(
                "Generated CellProfiler contract module attribute must be a "
                f"mapping, got {type(value).__name__}."
            )
        return cls.from_mapping(value)

    @property
    def invocation_contract_provider(
        self,
    ) -> CellProfilerGeneratedInvocationContractProvider:
        return CellProfilerGeneratedInvocationContractProvider(
            self.contracts_by_module_num
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
        if callable(self.func_spec):
            return (self.func_spec,)
        if (
            isinstance(self.func_spec, tuple)
            and len(self.func_spec) in {2, 3}
            and callable(self.func_spec[0])
        ):
            return (self.func_spec[0],)
        if isinstance(self.func_spec, list):
            callables: list[Callable[..., Any]] = []
            for item in self.func_spec:
                callables.extend(GeneratedFunctionSpec(item).callables)
            return tuple(callables)
        raise TypeError(
            "Unsupported generated FunctionStep func spec "
            f"{type(self.func_spec).__name__}."
        )


@dataclass(frozen=True, slots=True)
class GeneratedPipelineFunctionRegistration:
    """OpenHCS registry registration authority for generated runtime wrappers."""

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
        semantic_contracts=semantic_contracts,
        semantic_contract_fingerprint=semantic_contract_fingerprint,
    )


def materialize_generated_pipeline_import_module(
    source: str,
    *,
    module_name: str,
    output_dir: Path,
    artifact_contracts: dict[int, Any] | None = None,
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
