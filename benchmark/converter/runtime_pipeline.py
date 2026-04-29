"""Shared runtime wiring for generated CellProfiler -> OpenHCS pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import importlib.util
import inspect
import multiprocessing
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar

from metaclass_registry import AutoRegisterMeta

from openhcs.constants import MULTIPROCESSING_AXIS
from openhcs.core.config import DtypeConfig
from openhcs.core.pipeline import Pipeline
from openhcs.core.pipeline_image_schema import PipelineImageSchema
from openhcs.core.progress import set_progress_queue
from openhcs.processing.backends.lib_registry.openhcs_registry import (
    OpenHCSRegistry,
)
from openhcs.processing.backends.lib_registry.registry_service import (
    RegistryService,
)
from openhcs.processing.backends.lib_registry.unified_registry import (
    FunctionMetadata,
    ProcessingContract,
)
from openhcs.processing.func_registry import register_function

from .contract_inference import InferredContract, infer_contract
from .cppipe_module_roles import INFRASTRUCTURE_MODULE_NAMES
from .parser import CPPipeParser, ModuleBlock
from .pipeline_generator import GeneratedPipeline, PipelineGenerator


@dataclass(frozen=True, slots=True)
class CPPipeModulePartition:
    """CellProfiler module partition between runtime steps and infrastructure."""

    modules: tuple[ModuleBlock, ...]
    processing_modules: tuple[ModuleBlock, ...]
    infrastructure_modules: tuple[ModuleBlock, ...]


@dataclass(frozen=True, slots=True)
class CPPipePipelineArtifact(ABC):
    """Shared generated-pipeline context projected from a parsed .cppipe."""

    cppipe_path: Path
    processing_modules: tuple[ModuleBlock, ...]
    infrastructure_modules: tuple[ModuleBlock, ...]
    source_schema: PipelineImageSchema
    generated_pipeline: GeneratedPipeline


@dataclass(frozen=True, slots=True)
class GeneratedCPPipePipeline(CPPipePipelineArtifact):
    """Generated OpenHCS pipeline plus its parsed CellProfiler context."""

    modules: tuple[ModuleBlock, ...]


@dataclass(frozen=True, slots=True)
class PreparedGeneratedPipeline(CPPipePipelineArtifact):
    """Imported and registry-visible generated pipeline ready for execution."""

    module_name: str
    module_path: Path
    module: ModuleType
    pipeline: Pipeline
    registered_functions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DirectPipelineExecution:
    """Compilation and execution results for one direct orchestrator run."""

    compiled_contexts: dict[str, Any]
    execution_results: dict[str, Any]


class InferredContractMapper(ABC, metaclass=AutoRegisterMeta):
    """Map one inferred absorbed-function contract onto OpenHCS runtime semantics."""

    __registry_key__ = "contract"
    __skip_if_no_key__ = True
    contract: ClassVar[InferredContract | None] = None

    @classmethod
    def for_contract(
        cls,
        contract: InferredContract,
    ) -> InferredContractMapper | None:
        mapper_cls = cls.__registry__.get(contract)
        if mapper_cls is None:
            return None
        return mapper_cls()

    @abstractmethod
    def processing_contract(self) -> ProcessingContract | None:
        """Return the OpenHCS processing contract for one inferred contract."""


class Pure2DInferredContractMapper(InferredContractMapper):
    contract = InferredContract.PURE_2D

    def processing_contract(self) -> ProcessingContract:
        return ProcessingContract.PURE_2D


class Pure3DInferredContractMapper(InferredContractMapper):
    contract = InferredContract.PURE_3D

    def processing_contract(self) -> ProcessingContract:
        return ProcessingContract.PURE_3D


class FlexibleInferredContractMapper(InferredContractMapper):
    contract = InferredContract.FLEXIBLE

    def processing_contract(self) -> ProcessingContract:
        return ProcessingContract.FLEXIBLE


class VolumetricToSliceInferredContractMapper(InferredContractMapper):
    contract = InferredContract.VOLUMETRIC_TO_SLICE

    def processing_contract(self) -> ProcessingContract:
        return ProcessingContract.VOLUMETRIC_TO_SLICE


def partition_cppipe_modules(
    modules: Sequence[ModuleBlock],
    *,
    infrastructure_module_names: frozenset[str] = INFRASTRUCTURE_MODULE_NAMES,
) -> CPPipeModulePartition:
    """Split CellProfiler modules into OpenHCS steps vs infrastructure modules."""
    processing_modules = tuple(
        module
        for module in modules
        if module.name not in infrastructure_module_names
    )
    infrastructure_modules = tuple(
        module
        for module in modules
        if module.name in infrastructure_module_names
    )
    return CPPipeModulePartition(
        modules=tuple(modules),
        processing_modules=processing_modules,
        infrastructure_modules=infrastructure_modules,
    )


def generate_pipeline_from_cppipe(
    cppipe_path: Path,
    *,
    parser: CPPipeParser | None = None,
    generator: PipelineGenerator | None = None,
    infrastructure_module_names: frozenset[str] = INFRASTRUCTURE_MODULE_NAMES,
) -> GeneratedCPPipePipeline:
    """Parse and convert a .cppipe file into generated OpenHCS pipeline code."""
    cppipe_parser = parser or CPPipeParser()
    modules = tuple(cppipe_parser.parse(cppipe_path))
    partition = partition_cppipe_modules(
        modules,
        infrastructure_module_names=infrastructure_module_names,
    )
    pipeline_generator = generator or PipelineGenerator()

    missing_modules = tuple(
        module.name
        for module in partition.processing_modules
        if not pipeline_generator.has_module(module.name)
    )
    if missing_modules:
        raise ValueError(
            "Missing modules from absorbed library: "
            f"{sorted(missing_modules)}. Run `python -m benchmark.converter.absorb`."
        )

    generated_pipeline = pipeline_generator.generate_from_registry(
        pipeline_name=cppipe_path.stem,
        source_cppipe=cppipe_path,
        modules=list(partition.processing_modules),
        skipped_modules=list(partition.infrastructure_modules),
    )
    return GeneratedCPPipePipeline(
        cppipe_path=cppipe_path,
        modules=partition.modules,
        processing_modules=partition.processing_modules,
        infrastructure_modules=partition.infrastructure_modules,
        source_schema=generated_pipeline.source_schema,
        generated_pipeline=generated_pipeline,
    )


def prepare_generated_pipeline(
    cppipe_path: Path,
    *,
    output_path: Path,
    parser: CPPipeParser | None = None,
    generator: PipelineGenerator | None = None,
    infrastructure_module_names: frozenset[str] = INFRASTRUCTURE_MODULE_NAMES,
) -> PreparedGeneratedPipeline:
    """Generate, import, and register a .cppipe-derived OpenHCS pipeline."""
    converted = generate_pipeline_from_cppipe(
        cppipe_path,
        parser=parser,
        generator=generator,
        infrastructure_module_names=infrastructure_module_names,
    )
    converted.generated_pipeline.save(output_path)

    module_name = _generated_module_name(
        output_path,
        converted.generated_pipeline.code,
    )
    module = load_generated_pipeline_module(output_path, module_name=module_name)
    pipeline = _pipeline_from_generated_module(
        module,
        pipeline_name=converted.generated_pipeline.name,
    )
    registered_functions = register_generated_pipeline_functions(module)
    return PreparedGeneratedPipeline(
        cppipe_path=converted.cppipe_path,
        module_name=module_name,
        module_path=output_path,
        module=module,
        pipeline=pipeline,
        processing_modules=converted.processing_modules,
        infrastructure_modules=converted.infrastructure_modules,
        source_schema=converted.source_schema,
        generated_pipeline=converted.generated_pipeline,
        registered_functions=registered_functions,
    )


def load_generated_pipeline_module(
    module_path: Path,
    *,
    module_name: str,
) -> ModuleType:
    """Import generated pipeline code from disk under a deterministic module name."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create module spec for {module_path}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def register_generated_pipeline_functions(module: ModuleType) -> tuple[str, ...]:
    """Register generated pipeline callables so the OpenHCS compiler can resolve them."""
    registry = OpenHCSRegistry()
    existing_references = {
        (inspect.unwrap(metadata.func).__module__, inspect.unwrap(metadata.func).__name__)
        for metadata in RegistryService.get_all_functions_with_metadata().values()
    }
    registered_names: list[str] = []
    registered_new_function = False

    for func in _generated_step_callables(module):
        reference = (inspect.unwrap(func).__module__, inspect.unwrap(func).__name__)
        metadata_name = _generated_metadata_name(func)
        if reference in existing_references:
            registered_names.append(metadata_name)
            continue

        contract = _processing_contract_for(func)
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


def execute_pipeline_direct(
    orchestrator: Any,
    pipeline: Pipeline,
    *,
    well_filter: Sequence[str] | None = None,
) -> DirectPipelineExecution:
    """Compile and execute a pipeline through the direct orchestrator path."""
    wells = list(well_filter or orchestrator.get_component_keys(MULTIPROCESSING_AXIS))
    if not wells:
        raise RuntimeError("No wells found for pipeline execution.")

    mp_context = multiprocessing.get_context("spawn")
    progress_queue = mp_context.Queue()
    consumer = threading.Thread(
        target=_drain_progress_queue,
        args=(progress_queue,),
        daemon=True,
    )
    consumer.start()

    try:
        set_progress_queue(progress_queue)
        compilation_result = orchestrator.compile_pipelines(
            pipeline_definition=pipeline.steps,
            well_filter=wells,
        )
        compiled_contexts = compilation_result["compiled_contexts"]
        progress_context = {
            "execution_id": f"direct::{int(time.time() * 1_000_000)}",
            "plate_id": str(orchestrator.plate_path),
            "axis_id": "",
        }
        execution_results = orchestrator.execute_compiled_plate(
            pipeline_definition=pipeline.steps,
            compiled_contexts=compiled_contexts,
            progress_queue=progress_queue,
            progress_context=progress_context,
        )
        return DirectPipelineExecution(
            compiled_contexts=compiled_contexts,
            execution_results=execution_results,
        )
    finally:
        set_progress_queue(None)
        progress_queue.put(None)
        consumer.join(timeout=10)
        progress_queue.close()
        progress_queue.join_thread()


def _pipeline_from_generated_module(
    module: ModuleType,
    *,
    pipeline_name: str,
) -> Pipeline:
    """Build a Pipeline object from generated module exports."""
    pipeline_steps = _module_pipeline_steps(module)
    if isinstance(pipeline_steps, Pipeline):
        return pipeline_steps
    if not isinstance(pipeline_steps, list):
        raise TypeError(
            f"Generated module {module.__name__}.pipeline_steps must be list or "
            f"Pipeline, got {type(pipeline_steps).__name__}."
        )
    return Pipeline(steps=pipeline_steps, name=pipeline_name)


def _generated_step_callables(module: ModuleType) -> tuple[Callable[..., Any], ...]:
    """Extract unique callable objects referenced by generated pipeline steps."""
    callables: list[Callable[..., Any]] = []
    seen: set[int] = set()
    for step in _module_pipeline_steps(module):
        for func in _function_spec_callables(step.func):
            func_id = id(func)
            if func_id in seen:
                continue
            seen.add(func_id)
            callables.append(func)
    return tuple(callables)


def _module_pipeline_steps(module: ModuleType) -> Any:
    """Return validated generated pipeline steps exported by a module."""
    try:
        return module.pipeline_steps
    except AttributeError as exc:
        raise AttributeError(
            f"Generated module {module.__name__} does not define pipeline_steps."
        ) from exc


def _function_spec_callables(func_spec: Any) -> tuple[Callable[..., Any], ...]:
    """Extract callables from FunctionStep func specifications."""
    if callable(func_spec):
        return (func_spec,)
    if isinstance(func_spec, tuple) and len(func_spec) == 2 and callable(func_spec[0]):
        return (func_spec[0],)
    if isinstance(func_spec, list):
        callables: list[Callable[..., Any]] = []
        for item in func_spec:
            callables.extend(_function_spec_callables(item))
        return tuple(callables)
    raise TypeError(
        f"Unsupported generated FunctionStep func spec {type(func_spec).__name__}."
    )


def _processing_contract_for(func: Callable[..., Any]) -> ProcessingContract:
    """Resolve the generated function processing contract from typed function metadata."""
    metadata = func.__dict__
    contract = metadata.get("__processing_contract__")
    declared_contract = metadata.get("__cellprofiler_declared_contract__")
    if isinstance(contract, ProcessingContract):
        return contract
    if declared_contract == "unknown":
        inferred = _infer_unknown_processing_contract(func)
        if inferred is not None:
            return inferred
    if isinstance(declared_contract, str):
        mapped = _declared_processing_contract(declared_contract)
        if mapped is not None:
            return mapped
    return ProcessingContract.FLEXIBLE


def _infer_unknown_processing_contract(
    func: Callable[..., Any],
) -> ProcessingContract | None:
    """Infer contract for absorbed functions whose stored registry contract is unknown."""
    raw_func = func.__dict__.get("__cellprofiler_raw_function__", func)
    inference = infer_contract(raw_func, dtype_config=DtypeConfig())
    mapper = InferredContractMapper.for_contract(inference.contract)
    if mapper is None:
        return None
    return mapper.processing_contract()


def _declared_processing_contract(
    contract_name: str,
) -> ProcessingContract | None:
    """Map stored absorbed-library contract strings onto OpenHCS ProcessingContract."""
    normalized = contract_name.upper()
    if normalized not in {"PURE_2D", "PURE_3D", "FLEXIBLE", "VOLUMETRIC_TO_SLICE"}:
        return None
    return ProcessingContract[normalized]


def _generated_metadata_name(func: Callable[..., Any]) -> str:
    """Build stable registry metadata name for a generated runtime wrapper."""
    return f"{func.__module__}:{func.__name__}"


def _generated_module_name(module_path: Path, code: str) -> str:
    """Derive deterministic import name from module path and generated code."""
    digest = hashlib.sha1(
        f"{module_path.resolve()}::{code}".encode("utf-8")
    ).hexdigest()[:12]
    stem = "".join(
        character if character.isalnum() else "_"
        for character in module_path.stem
    ).strip("_")
    normalized_stem = stem or "pipeline"
    return f"benchmark_generated_{normalized_stem}_{digest}"


def _drain_progress_queue(queue: Any) -> None:
    """Drain progress events so worker feeder threads never deadlock on a full pipe."""
    while True:
        item = queue.get()
        if item is None:
            break
