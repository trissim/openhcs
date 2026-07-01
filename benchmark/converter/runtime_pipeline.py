"""Benchmark execution helpers for generated CellProfiler -> OpenHCS pipelines."""

from __future__ import annotations

from typing import Any

from openhcs.core.pipeline import Pipeline
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
    execute_pipeline_direct as execute_pipeline_direct_runtime,
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
    phase_timing: PhaseTimingTrace | None = None,
) -> DirectPipelineExecution:
    """Benchmark facade over product-owned direct CellProfiler execution."""
    return execute_pipeline_direct_runtime(
        orchestrator,
        pipeline,
        phase_timing=phase_timing,
        compile_phase=BenchmarkPhase.COMPILE_OPENHCS,
        execute_phase=BenchmarkPhase.EXECUTE_OPENHCS,
    )
