"""Benchmark execution helpers for generated CellProfiler -> OpenHCS pipelines."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from collections.abc import Sequence
from typing import Any

from openhcs.constants import MULTIPROCESSING_AXIS
from openhcs.config_framework.global_config import get_current_global_config
from openhcs.config_framework.lazy_factory import ensure_global_config_context
from openhcs.core.config import GlobalPipelineConfig
from openhcs.core.pipeline import Pipeline
from openhcs.core.progress import set_progress_queue
from openhcs.core.steps.function_runtime import prepare_compiled_context_callables
from openhcs.core.worker_start_policy import WorkerStartDecision
from openhcs.core.worker_start_policy import WorkerStartExecutionFacts
from openhcs.core.worker_start_policy import resolve_worker_start_context
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


@dataclass
class DirectExecutionProgressBridge:
    """Progress queue lifecycle tied to the resolved worker-start context."""

    decision: WorkerStartDecision
    queue: Any
    consumer: threading.Thread

    @classmethod
    def from_decision(
        cls,
        decision: WorkerStartDecision,
    ) -> "DirectExecutionProgressBridge":
        queue = decision.context.Queue()
        consumer = threading.Thread(
            target=_drain_progress_queue,
            args=(queue,),
            daemon=True,
        )
        consumer.start()
        return cls(decision=decision, queue=queue, consumer=consumer)

    def close(self) -> None:
        self.queue.put(None)
        self.consumer.join(timeout=10)
        self.queue.close()
        self.queue.join_thread()


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

    global_config = get_current_global_config(GlobalPipelineConfig)
    if global_config is None:
        global_config = orchestrator.get_effective_config()
    progress_bridge = DirectExecutionProgressBridge.from_decision(
        resolve_worker_start_context(
            global_config,
            server_mode=False,
            gpu_enabled=False,
        )
    )

    try:
        set_progress_queue(progress_bridge.queue)
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
        execution_facts = WorkerStartExecutionFacts.from_compiled_contexts(
            compiled_contexts
        )
        execution_decision = resolve_worker_start_context(
            global_config,
            server_mode=False,
            gpu_enabled=execution_facts.gpu_enabled,
        )
        if execution_decision.resolved is not progress_bridge.decision.resolved:
            global_config = replace(
                global_config,
                multiprocessing_start_method=execution_decision.resolved,
            )
            ensure_global_config_context(GlobalPipelineConfig, global_config)
            set_progress_queue(None)
            progress_bridge.close()
            progress_bridge = DirectExecutionProgressBridge.from_decision(
                execution_decision
            )
            set_progress_queue(progress_bridge.queue)
        progress_context = {
            "execution_id": f"direct::{int(time.time() * 1_000_000)}",
            "plate_id": str(orchestrator.plate_path),
            "axis_id": "",
        }
        if phase_timing is None:
            prepare_compiled_context_callables(compiled_contexts)
            execution_results = orchestrator.execute_compiled_plate(
                pipeline_definition=pipeline.steps,
                compiled_contexts=compiled_contexts,
                progress_queue=progress_bridge.queue,
                progress_context=progress_context,
            )
        else:
            with phase_timing.phase(BenchmarkPhase.COMPILE_OPENHCS):
                prepare_compiled_context_callables(compiled_contexts)
            with phase_timing.phase(BenchmarkPhase.EXECUTE_OPENHCS):
                execution_results = orchestrator.execute_compiled_plate(
                    pipeline_definition=pipeline.steps,
                    compiled_contexts=compiled_contexts,
                    progress_queue=progress_bridge.queue,
                    progress_context=progress_context,
                )
        return DirectPipelineExecution(
            compiled_contexts=compiled_contexts,
            execution_results=execution_results,
        )
    finally:
        set_progress_queue(None)
        progress_bridge.close()


def _drain_progress_queue(queue: Any) -> None:
    """Drain progress events so worker feeder threads never deadlock on a full pipe."""
    while True:
        item = queue.get()
        if item is None:
            break
