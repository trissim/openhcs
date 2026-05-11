"""Benchmark execution helpers for generated CellProfiler -> OpenHCS pipelines."""

from __future__ import annotations

import multiprocessing
import threading
import time
from collections.abc import Sequence
from typing import Any

from openhcs.constants import MULTIPROCESSING_AXIS
from openhcs.core.pipeline import Pipeline
from openhcs.core.progress import set_progress_queue
from openhcs.core.steps.function_runtime import prepare_compiled_context_callables
from openhcs.interop.cellprofiler.runtime_pipeline import (
    BenchmarkCellProfilerDialectCompiler,
    BenchmarkCellProfilerPipelineImporter,
    CPPipeModulePartition,
    CPPipePipelineGenerationRequest,
    CPPipePipelinePreparationRequest,
    CellProfilerGeneratedPipelineDialectCompiler,
    CellProfilerGeneratedPipelineImporter,
    DirectPipelineExecution,
    GeneratedCPPipePipeline,
    PreparedGeneratedPipeline,
    generate_pipeline_from_cppipe,
    partition_cppipe_modules,
    prepare_generated_pipeline,
    register_benchmark_cellprofiler_dialect_compiler,
    register_generated_cellprofiler_dialect_compiler,
)

from benchmark.timing import BenchmarkPhase, PhaseTimingTrace


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
