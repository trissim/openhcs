"""Generated CellProfiler pipeline import and registry authorities."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar

from openhcs.core.callable_contract import CallableContract
from openhcs.core.artifact_materialization_policy import NO_ARTIFACT_MATERIALIZATION
from openhcs.core.artifacts import ArtifactKind, ArtifactSidecarRole, ArtifactSpec
from openhcs.core.pipeline import Pipeline
from openhcs.processing.materialization import (
    CsvOptions,
    JsonOptions,
    MaterializationSpec,
    ROIOptions,
    TextOptions,
    TiffStackOptions,
)
from openhcs.processing.backends.lib_registry.openhcs_registry import OpenHCSRegistry
from openhcs.processing.backends.lib_registry.registry_service import RegistryService
from openhcs.processing.backends.lib_registry.unified_registry import (
    FunctionMetadata,
    ProcessingContract,
)
from openhcs.processing.func_registry import register_function
from openhcs.interop.cellprofiler.runtime.module_execution import (
    CellProfilerModuleContractRegistry,
    CellProfilerRuntimeStepBinding,
)
from openhcs.core.module_artifact_contract import ModuleArtifactContract
from openhcs.processing.backends.cellprofiler import (
    cellprofiler_function_runtime_metadata,
)

_CONTRACT_SIDECAR_SCHEMA = "openhcs.cellprofiler.generated_contracts"
_CONTRACT_SIDECAR_VERSION = 1


@dataclass(frozen=True, slots=True)
class GeneratedPipelineContractSidecar:
    """Versioned JSON persistence facade for generated CP runtime contracts."""

    contracts_by_module_num: Mapping[int, ModuleArtifactContract]

    def write(self, path: Path) -> None:
        """Write contracts to a deterministic JSON sidecar."""
        codec = GeneratedPipelineContractSidecarCodec()
        payload = {
            "schema": _CONTRACT_SIDECAR_SCHEMA,
            "version": _CONTRACT_SIDECAR_VERSION,
            "contracts": [
                codec.payload(module_num, contract)
                for module_num, contract in sorted(
                    self.contracts_by_module_num.items()
                )
            ],
        }
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def read(cls, path: Path) -> dict[int, ModuleArtifactContract]:
        """Read contracts from a versioned JSON sidecar."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != _CONTRACT_SIDECAR_SCHEMA:
            raise ValueError(
                f"Unsupported CellProfiler contract sidecar schema: "
                f"{payload.get('schema')!r}."
            )
        if payload.get("version") != _CONTRACT_SIDECAR_VERSION:
            raise ValueError(
                f"Unsupported CellProfiler contract sidecar version: "
                f"{payload.get('version')!r}."
            )
        codec = GeneratedPipelineContractSidecarCodec()
        contracts: dict[int, ModuleArtifactContract] = {}
        for contract_payload in payload.get("contracts", ()):
            module_num = int(contract_payload["module_num"])
            contracts[module_num] = codec.from_payload(contract_payload)
        return contracts

    @classmethod
    def register(
        cls,
        *,
        generated_module_name: str,
        path: str | Path,
    ) -> dict[int, ModuleArtifactContract]:
        """Read a sidecar and register its contracts for a generated module."""
        contracts = cls.read(Path(path))
        CellProfilerModuleContractRegistry.register(generated_module_name, contracts)
        return contracts


@dataclass(frozen=True, slots=True)
class GeneratedPipelineContractSidecarCodec:
    """Codec for one ModuleArtifactContract sidecar record."""

    spec_codec: "GeneratedPipelineArtifactSpecSidecarCodec" = field(
        default_factory=lambda: GeneratedPipelineArtifactSpecSidecarCodec()
    )

    def payload(
        self,
        module_num: int,
        contract: ModuleArtifactContract,
    ) -> dict[str, Any]:
        if not isinstance(contract, ModuleArtifactContract):
            raise TypeError(
                "GeneratedPipelineContractSidecar requires ModuleArtifactContract "
                f"values, got {type(contract).__name__}."
            )
        return {
            "module_num": module_num,
            "module_name": contract.module_name,
            "inputs": self.spec_codec.sequence_payload(contract.inputs),
            "runtime_artifact_inputs": self.spec_codec.sequence_payload(
                contract.runtime_artifact_inputs
            ),
            "outputs": self.spec_codec.sequence_payload(contract.outputs),
            "declared_outputs": self.spec_codec.sequence_payload(
                contract.declared_outputs
            ),
        }

    def from_payload(
        self,
        payload: Mapping[str, Any],
    ) -> ModuleArtifactContract:
        return ModuleArtifactContract(
            module_name=str(payload["module_name"]),
            inputs=self.spec_codec.sequence_from_payload(payload.get("inputs", ())),
            runtime_artifact_inputs=self.spec_codec.sequence_from_payload(
                payload.get("runtime_artifact_inputs", ())
            ),
            outputs=self.spec_codec.sequence_from_payload(payload.get("outputs", ())),
            declared_outputs=self.spec_codec.sequence_from_payload(
                payload.get("declared_outputs", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class GeneratedPipelineArtifactSpecSidecarCodec:
    """Codec for ArtifactSpec sequences inside generated contract sidecars."""

    materialization_codec: "GeneratedPipelineMaterializationSidecarCodec" = field(
        default_factory=lambda: GeneratedPipelineMaterializationSidecarCodec()
    )

    def sequence_payload(
        self,
        specs: tuple[ArtifactSpec, ...],
    ) -> list[dict[str, Any]]:
        return [self.payload(spec) for spec in specs]

    def sequence_from_payload(self, payload: Any) -> tuple[ArtifactSpec, ...]:
        return tuple(self.from_payload(spec_payload) for spec_payload in payload)

    def payload(self, spec: ArtifactSpec) -> dict[str, Any]:
        return {
            "name": spec.name,
            "kind": spec.kind.value,
            "required": spec.required,
            "sidecar_role": (
                None if spec.sidecar_role is None else spec.sidecar_role.value
            ),
            "materialization": self.materialization_codec.payload(
                spec.materialization
            ),
        }

    def from_payload(self, payload: Mapping[str, Any]) -> ArtifactSpec:
        sidecar_role = payload.get("sidecar_role")
        return ArtifactSpec(
            name=str(payload["name"]),
            kind=ArtifactKind(str(payload["kind"])),
            required=bool(payload.get("required", True)),
            sidecar_role=(
                None
                if sidecar_role is None
                else ArtifactSidecarRole(str(sidecar_role))
            ),
            materialization=self.materialization_codec.from_payload(
                payload.get("materialization")
            ),
        )


@dataclass(frozen=True, slots=True)
class GeneratedPipelineMaterializationSidecarCodec:
    """Codec for materialization policy values inside generated sidecars."""

    option_types: ClassVar[Mapping[str, type[Any]]] = {
        option_cls.__name__: option_cls
        for option_cls in (
            CsvOptions,
            JsonOptions,
            ROIOptions,
            TextOptions,
            TiffStackOptions,
        )
    }

    def payload(self, materialization: Any) -> Any:
        if materialization is None:
            return None
        if materialization is NO_ARTIFACT_MATERIALIZATION:
            return {"type": "none"}
        if not isinstance(materialization, MaterializationSpec):
            raise TypeError(
                "Generated CellProfiler contract sidecars only support "
                "MaterializationSpec or NO_ARTIFACT_MATERIALIZATION values, "
                f"got {type(materialization).__name__}."
            )
        return {
            "type": "materialization_spec",
            "allowed_backends": materialization.allowed_backends,
            "primary": materialization.primary,
            "outputs": [
                self.option_payload(option)
                for option in materialization.outputs
            ],
        }

    def from_payload(self, payload: Any) -> Any:
        if payload is None:
            return None
        payload_type = payload.get("type")
        if payload_type == "none":
            return NO_ARTIFACT_MATERIALIZATION
        if payload_type != "materialization_spec":
            raise ValueError(f"Unsupported materialization payload: {payload!r}.")
        return MaterializationSpec(
            tuple(
                self.option_from_payload(option_payload)
                for option_payload in payload["outputs"]
            ),
            allowed_backends=payload.get("allowed_backends"),
            primary=int(payload.get("primary", 0)),
        )

    def option_payload(self, option: Any) -> dict[str, Any]:
        if not is_dataclass(option):
            raise TypeError(
                "Materialization options in generated sidecars must be dataclass "
                f"instances, got {type(option).__name__}."
            )
        payload = asdict(option)
        if any(callable(value) for value in payload.values()):
            raise TypeError(
                "Generated CellProfiler contract sidecars cannot serialize "
                f"callable materialization options on {type(option).__name__}."
            )
        return {"type": type(option).__name__, "fields": payload}

    def option_from_payload(self, payload: Mapping[str, Any]) -> Any:
        option_type = str(payload["type"])
        option_fields = dict(payload.get("fields", {}))
        try:
            option_cls = self.option_types[option_type]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported generated materialization option type {option_type!r}."
            ) from exc
        return option_cls(**option_fields)


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
    ) -> ModuleType:
        """Import generated pipeline code from source with the stable module name."""
        if artifact_contracts:
            CellProfilerModuleContractRegistry.register(
                self.module_name,
                artifact_contracts,
            )
        module = ModuleType(self.module_name)
        module.__file__ = filename
        sys.modules[self.module_name] = module
        exec(compile(self.identity.code, filename, "exec"), module.__dict__)
        if artifact_contracts:
            bind_generated_pipeline_runtime(module, artifact_contracts)
        return module

    def materialize_import_module(
        self,
        *,
        output_dir: Path,
        artifact_contracts: dict[int, Any] | None = None,
    ) -> Path:
        """Write an importable module that restores registry visibility on import."""
        importable_path = output_dir / f"{self.module_name}.py"
        contract_sidecar = output_dir / f"{self.module_name}.cellprofiler_contracts.json"
        contract_prelude = ""
        if artifact_contracts:
            GeneratedPipelineContractSidecar(artifact_contracts).write(
                contract_sidecar
            )
            contract_prelude = (
                "    from openhcs.interop.cellprofiler.runtime.generated_pipeline import "
                "GeneratedPipelineContractSidecar as _openhcs_cp_contract_sidecar\n"
                "    _openhcs_cp_contract_values = _openhcs_cp_contract_sidecar.register("
                f"generated_module_name=__name__, path={str(contract_sidecar)!r})\n"
            )
        importable_source = (
            self.identity.code
            + "\n\n"
            + "if __name__ != '__main__':\n"
            + contract_prelude
            + "    import sys as _openhcs_generated_sys\n"
            + "    from openhcs.interop.cellprofiler.runtime.generated_pipeline import "
            + "GeneratedPipelineFunctionRegistration as _openhcs_registration\n"
            + "    from openhcs.interop.cellprofiler.runtime.generated_pipeline import "
            + "bind_generated_pipeline_runtime as _openhcs_bind_runtime\n"
            + "    _openhcs_bind_runtime("
            + "_openhcs_generated_sys.modules[__name__], "
            + "globals().get('_openhcs_cp_contract_values', {}))\n"
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

    def pipeline_from_module(self, module: ModuleType, *, pipeline_name: str) -> Pipeline:
        """Build a Pipeline object from generated module exports."""
        pipeline_steps = GeneratedPipelineModuleExports(module).pipeline_steps
        if isinstance(pipeline_steps, Pipeline):
            return pipeline_steps
        if not isinstance(pipeline_steps, list):
            raise TypeError(
                f"Generated module {module.__name__}.pipeline_steps must be list or "
                f"Pipeline, got {type(pipeline_steps).__name__}."
            )
        return Pipeline(steps=pipeline_steps, name=pipeline_name)


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

        contracts_by_module_name: dict[str, list[int]] = {}
        for module_num, contract in self.artifact_contracts.items():
            contracts_by_module_name.setdefault(contract.module_name, []).append(
                module_num
            )

        for step in GeneratedPipelineModuleExports(self.module).pipeline_steps:
            step.func = self._bind_func_spec(step.func, contracts_by_module_name)

    def _bind_func_spec(
        self,
        func_spec: Any,
        contracts_by_module_name: dict[str, list[int]],
    ) -> Any:
        if callable(func_spec):
            return self._bind_callable(func_spec, contracts_by_module_name)
        if isinstance(func_spec, tuple) and len(func_spec) in {2, 3}:
            return (
                self._bind_callable(func_spec[0], contracts_by_module_name),
                *func_spec[1:],
            )
        if isinstance(func_spec, list):
            return [
                self._bind_func_spec(item, contracts_by_module_name)
                for item in func_spec
            ]
        return func_spec

    def _bind_callable(
        self,
        func: Callable[..., Any],
        contracts_by_module_name: dict[str, list[int]],
    ) -> Callable[..., Any]:
        metadata = cellprofiler_function_runtime_metadata(func)
        if metadata is None:
            return func
        module_nums = contracts_by_module_name.get(metadata.module_name)
        if not module_nums:
            return func
        module_num = module_nums.pop(0)
        return CellProfilerRuntimeStepBinding(
            raw_callable=func,
            generated_module_name=self.module.__name__,
            module_num=module_num,
            declared_processing_contract=metadata.declared_processing_contract,
            runtime_name=(
                f"{self.module.__name__}_{metadata.function_name}_"
                f"{module_num}_runtime"
            ),
        ).load()


@dataclass(frozen=True, slots=True)
class GeneratedPipelineModuleExports:
    """Typed access to generated module exports."""

    module: ModuleType

    @property
    def pipeline_steps(self) -> Any:
        try:
            return self.module.pipeline_steps
        except AttributeError as exc:
            raise AttributeError(
                f"Generated module {self.module.__name__} does not define "
                "pipeline_steps."
            ) from exc

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
            func.__processing_contract__ = contract
            wrapped_func = registry.apply_contract_wrapper(func, contract)
            wrapped_func.__processing_contract__ = contract
            wrapped_func.__function_metadata__ = FunctionMetadata(
                name=metadata_name,
                func=wrapped_func,
                contract=contract,
                registry=registry,
                module=wrapped_func.__module__ or "",
                doc=wrapped_func.__doc__ or "",
                tags=["openhcs", "generated", "cellprofiler"],
                original_name=wrapped_func.__name__,
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


def generated_pipeline_module_name(module_path: Path, code: str) -> str:
    """Compatibility facade for generated module identity projection."""
    return GeneratedPipelineModuleIdentity(module_path=module_path, code=code).module_name


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
    )


def materialize_generated_pipeline_import_module(
    source: str,
    *,
    module_name: str,
    output_dir: Path,
    artifact_contracts: dict[int, Any] | None = None,
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
        output_dir=output_dir,
        artifact_contracts=artifact_contracts,
    )


def pipeline_from_generated_module(
    module: ModuleType,
    *,
    pipeline_name: str,
) -> Pipeline:
    """Compatibility facade for building a Pipeline from generated exports."""
    return GeneratedPipelineRuntimeModule(
        GeneratedPipelineModuleIdentity(module_path=Path(module.__file__ or ""), code="")
    ).pipeline_from_module(module, pipeline_name=pipeline_name)
