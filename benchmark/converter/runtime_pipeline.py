"""Shared runtime wiring for generated CellProfiler -> OpenHCS pipelines."""

from __future__ import annotations

from abc import ABC
import multiprocessing
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from openhcs.constants import Backend, MULTIPROCESSING_AXIS
from openhcs.core.pipeline import Pipeline
from openhcs.core.pipeline_image_schema import PipelineImageSchema
from openhcs.core.progress import set_progress_queue
from openhcs.core.steps.function_runtime import prepare_compiled_context_callables
from openhcs.core.vfs_protocol import FileManagerLike
from openhcs.interop.cellprofiler.import_records import (
    CellProfilerModuleReference,
    CellProfilerPipelineImportResult,
    CellProfilerPipelineProvenance,
)
from openhcs.interop.cellprofiler.import_service import (
    CellProfilerPipelineImporter,
    CellProfilerPipelineImportRequest,
)
from openhcs.interop.cellprofiler.compiler_registry import (
    register_cellprofiler_dialect_compiler,
)
from openhcs.interop.cellprofiler.module_roles import (
    CellProfilerModuleRole,
    INFRASTRUCTURE_MODULE_NAMES,
)
from openhcs.interop.cellprofiler.parser import CPPipeParser, ModuleBlock
from openhcs.interop.cellprofiler.runtime.generated_pipeline import (
    generated_pipeline_module_name,
    load_generated_pipeline_module,
    load_generated_pipeline_module_from_source,
    materialize_generated_pipeline_import_module,
    pipeline_from_generated_module,
    register_generated_pipeline_functions,
)

from benchmark.timing import BenchmarkPhase, PhaseTimingTrace

from openhcs.interop.cellprofiler.pipeline_generator import (
    GeneratedPipeline,
    PipelineGenerator,
)


@dataclass(frozen=True, slots=True)
class CPPipeModulePartition:
    """CellProfiler module partition between runtime steps and infrastructure."""

    modules: tuple[ModuleBlock, ...]
    processing_modules: tuple[ModuleBlock, ...]
    infrastructure_modules: tuple[ModuleBlock, ...]
    disabled_modules: tuple[ModuleBlock, ...]

    def provenance(self, cppipe_path: Path) -> CellProfilerPipelineProvenance:
        """Build product provenance from the partitioned module roles."""
        role_by_module_num = {
            module.module_num: CellProfilerModuleRole.PROCESSING
            for module in self.processing_modules
        }
        role_by_module_num.update(
            {
                module.module_num: CellProfilerModuleRole.INFRASTRUCTURE
                for module in self.infrastructure_modules
            }
        )
        role_by_module_num.update(
            {
                module.module_num: CellProfilerModuleRole.DISABLED
                for module in self.disabled_modules
            }
        )
        return CellProfilerPipelineProvenance(
            cppipe_path=cppipe_path,
            modules=tuple(
                CellProfilerModuleReference(
                    name=module.name,
                    module_num=module.module_num,
                    role=role_by_module_num[module.module_num],
                )
                for module in self.modules
            ),
        )


@dataclass(frozen=True, slots=True)
class CPPipePipelineArtifact(ABC):
    """Shared generated-pipeline context projected from a parsed .cppipe."""

    cppipe_path: Path
    processing_modules: tuple[ModuleBlock, ...]
    infrastructure_modules: tuple[ModuleBlock, ...]
    disabled_modules: tuple[ModuleBlock, ...]
    source_schema: PipelineImageSchema
    generated_pipeline: GeneratedPipeline
    provenance: CellProfilerPipelineProvenance


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

    @property
    def import_result(self) -> CellProfilerPipelineImportResult:
        """Return the product-facing import record for this prepared pipeline."""
        return CellProfilerPipelineImportResult(
            provenance=self.provenance,
            pipeline=self.pipeline,
            source_schema=self.source_schema,
            generated_source=self.generated_pipeline.code,
            generated_module_name=self.module_name,
            generated_module_path=self.module_path,
            artifact_contracts=tuple(
                contract.module_contract
                for contract in self.generated_pipeline.artifact_contracts
            ),
            registered_functions=self.registered_functions,
        )


@dataclass(frozen=True, slots=True)
class DirectPipelineExecution:
    """Compilation and execution results for one direct orchestrator run."""

    compiled_contexts: dict[str, Any]
    execution_results: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CPPipePipelineGenerationRequest:
    """Nominal request for parsing and generating one CellProfiler pipeline."""

    cppipe_path: Path
    parser: CPPipeParser | None = None
    generator: PipelineGenerator | None = None
    infrastructure_module_names: frozenset[str] = INFRASTRUCTURE_MODULE_NAMES
    prune_dead_unmaterialized_artifact_steps: bool = False
    materialize_skipped_save_images: bool = True
    filemanager: FileManagerLike | None = None
    cppipe_backend: Backend = Backend.DISK

    @property
    def cppipe_parser(self) -> CPPipeParser:
        return self.parser or CPPipeParser()

    @property
    def pipeline_generator(self) -> PipelineGenerator:
        return self.generator or PipelineGenerator()

    def parse_modules(self) -> tuple[ModuleBlock, ...]:
        return tuple(
            self.cppipe_parser.parse(
                self.cppipe_path,
                filemanager=self.filemanager,
                backend=self.cppipe_backend,
            )
        )

    def partition_modules(self) -> CPPipeModulePartition:
        return partition_cppipe_modules(
            self.parse_modules(),
            infrastructure_module_names=self.infrastructure_module_names,
        )

    def generate(self) -> GeneratedCPPipePipeline:
        """Parse and convert this request into generated OpenHCS pipeline code."""
        partition = self.partition_modules()
        pipeline_generator = self.pipeline_generator

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
            pipeline_name=self.cppipe_path.stem,
            source_cppipe=self.cppipe_path,
            modules=list(partition.processing_modules),
            skipped_modules=list(partition.infrastructure_modules),
            prune_dead_unmaterialized_artifact_steps=(
                self.prune_dead_unmaterialized_artifact_steps
            ),
            materialize_skipped_save_images=self.materialize_skipped_save_images,
        )
        return GeneratedCPPipePipeline(
            cppipe_path=self.cppipe_path,
            modules=partition.modules,
            processing_modules=partition.processing_modules,
            infrastructure_modules=partition.infrastructure_modules,
            disabled_modules=partition.disabled_modules,
            source_schema=generated_pipeline.source_schema,
            generated_pipeline=generated_pipeline,
            provenance=partition.provenance(self.cppipe_path),
        )


@dataclass(frozen=True, slots=True)
class CPPipePipelinePreparationRequest:
    """Nominal request for materializing and importing a generated pipeline."""

    generation: CPPipePipelineGenerationRequest
    output_path: Path
    generated_pipeline_backend: Backend = Backend.DISK

    def prepare(self) -> PreparedGeneratedPipeline:
        """Generate, persist, import, and register the requested pipeline."""
        converted = self.generation.generate()
        converted.generated_pipeline.save(
            self.output_path,
            filemanager=self.generation.filemanager,
            backend=self.generated_pipeline_backend,
        )

        module_name = generated_pipeline_module_name(
            self.output_path,
            converted.generated_pipeline.code,
        )
        module = load_generated_pipeline_module_from_source(
            converted.generated_pipeline.code,
            module_name=module_name,
            filename=str(self.output_path),
        )
        materialize_generated_pipeline_import_module(
            converted.generated_pipeline.code,
            module_name=module_name,
            output_dir=self.output_path.parent,
        )
        pipeline = pipeline_from_generated_module(
            module,
            pipeline_name=converted.generated_pipeline.name,
        )
        registered_functions = register_generated_pipeline_functions(module)
        return PreparedGeneratedPipeline(
            cppipe_path=converted.cppipe_path,
            module_name=module_name,
            module_path=self.output_path,
            module=module,
            pipeline=pipeline,
            processing_modules=converted.processing_modules,
            infrastructure_modules=converted.infrastructure_modules,
            disabled_modules=converted.disabled_modules,
            source_schema=converted.source_schema,
            generated_pipeline=converted.generated_pipeline,
            provenance=converted.provenance,
            registered_functions=registered_functions,
        )


class BenchmarkCellProfilerDialectCompiler(CellProfilerPipelineImporter):
    """Current dialect compiler implementation backed by the benchmark converter."""

    def import_pipeline(
        self,
        request: CellProfilerPipelineImportRequest,
    ) -> CellProfilerPipelineImportResult:
        prepared = prepare_generated_pipeline(
            request.cppipe_path,
            output_path=request.generated_pipeline_path,
            filemanager=request.filemanager,
            cppipe_backend=request.cppipe_backend,
            generated_pipeline_backend=request.generated_pipeline_backend,
            prune_dead_unmaterialized_artifact_steps=(
                request.prune_dead_unmaterialized_artifact_steps
            ),
        )
        return prepared.import_result


BenchmarkCellProfilerPipelineImporter = BenchmarkCellProfilerDialectCompiler


def register_benchmark_cellprofiler_dialect_compiler() -> (
    BenchmarkCellProfilerDialectCompiler
):
    """Register the current benchmark-backed compiler as the product provider."""
    compiler = BenchmarkCellProfilerDialectCompiler()
    register_cellprofiler_dialect_compiler(compiler)
    return compiler


def partition_cppipe_modules(
    modules: Sequence[ModuleBlock],
    *,
    infrastructure_module_names: frozenset[str] = INFRASTRUCTURE_MODULE_NAMES,
) -> CPPipeModulePartition:
    """Split CellProfiler modules into OpenHCS steps vs infrastructure modules."""
    processing_modules = tuple(
        module
        for module in modules
        if module.enabled and module.name not in infrastructure_module_names
    )
    infrastructure_modules = tuple(
        module
        for module in modules
        if module.enabled and module.name in infrastructure_module_names
    )
    disabled_modules = tuple(
        module
        for module in modules
        if not module.enabled
    )
    return CPPipeModulePartition(
        modules=tuple(modules),
        processing_modules=processing_modules,
        infrastructure_modules=infrastructure_modules,
        disabled_modules=disabled_modules,
    )


def generate_pipeline_from_cppipe(
    cppipe_path: Path,
    *,
    parser: CPPipeParser | None = None,
    generator: PipelineGenerator | None = None,
    infrastructure_module_names: frozenset[str] = INFRASTRUCTURE_MODULE_NAMES,
    prune_dead_unmaterialized_artifact_steps: bool = False,
    materialize_skipped_save_images: bool = True,
    filemanager: FileManagerLike | None = None,
    cppipe_backend: Backend = Backend.DISK,
) -> GeneratedCPPipePipeline:
    """Parse and convert a .cppipe file into generated OpenHCS pipeline code."""
    return CPPipePipelineGenerationRequest(
        cppipe_path=cppipe_path,
        parser=parser,
        generator=generator,
        infrastructure_module_names=infrastructure_module_names,
        prune_dead_unmaterialized_artifact_steps=(
            prune_dead_unmaterialized_artifact_steps
        ),
        materialize_skipped_save_images=materialize_skipped_save_images,
        filemanager=filemanager,
        cppipe_backend=cppipe_backend,
    ).generate()


def prepare_generated_pipeline(
    cppipe_path: Path,
    *,
    output_path: Path,
    parser: CPPipeParser | None = None,
    generator: PipelineGenerator | None = None,
    infrastructure_module_names: frozenset[str] = INFRASTRUCTURE_MODULE_NAMES,
    prune_dead_unmaterialized_artifact_steps: bool = False,
    materialize_skipped_save_images: bool = True,
    filemanager: FileManagerLike | None = None,
    cppipe_backend: Backend = Backend.DISK,
    generated_pipeline_backend: Backend = Backend.DISK,
) -> PreparedGeneratedPipeline:
    """Generate, import, and register a .cppipe-derived OpenHCS pipeline."""
    return CPPipePipelinePreparationRequest(
        generation=CPPipePipelineGenerationRequest(
            cppipe_path=cppipe_path,
            parser=parser,
            generator=generator,
            infrastructure_module_names=infrastructure_module_names,
            prune_dead_unmaterialized_artifact_steps=(
                prune_dead_unmaterialized_artifact_steps
            ),
            materialize_skipped_save_images=materialize_skipped_save_images,
            filemanager=filemanager,
            cppipe_backend=cppipe_backend,
        ),
        output_path=output_path,
        generated_pipeline_backend=generated_pipeline_backend,
    ).prepare()


def execute_pipeline_direct(
    orchestrator: Any,
    pipeline: Pipeline,
    *,
    well_filter: Sequence[str] | None = None,
    phase_timing: PhaseTimingTrace | None = None,
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
        if phase_timing is None:
            compilation_result = orchestrator.compile_pipelines(
                pipeline_definition=pipeline.steps,
                well_filter=wells,
            )
        else:
            with phase_timing.phase(BenchmarkPhase.COMPILE_OPENHCS):
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
        if phase_timing is None:
            execution_results = orchestrator.execute_compiled_plate(
                pipeline_definition=pipeline.steps,
                compiled_contexts=compiled_contexts,
                progress_queue=progress_queue,
                progress_context=progress_context,
            )
        else:
            with phase_timing.phase(BenchmarkPhase.COMPILE_OPENHCS):
                prepare_compiled_context_callables(compiled_contexts)
            with phase_timing.phase(BenchmarkPhase.EXECUTE_OPENHCS):
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


def _drain_progress_queue(queue: Any) -> None:
    """Drain progress events so worker feeder threads never deadlock on a full pipe."""
    while True:
        item = queue.get()
        if item is None:
            break
