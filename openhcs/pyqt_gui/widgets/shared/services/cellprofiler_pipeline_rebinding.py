"""Shared UI service for preserving public CellProfiler pipeline steps."""

from __future__ import annotations

from collections.abc import Sequence

from openhcs.core.steps.function_step import FunctionStep


class CellProfilerImportResultProvider:
    """Compatibility marker for callers that still pass an import surface."""


class CellProfilerPipelineRuntimeBindingService:
    """Return public CellProfiler steps without UI/import contract mutation."""

    @classmethod
    def runtime_bound_pipeline_for_plate(
        cls,
        *,
        import_result_provider: CellProfilerImportResultProvider | None,
        plate_path: str,
        pipeline_steps: Sequence[FunctionStep],
    ) -> list[FunctionStep]:
        del import_result_provider, plate_path
        return list(pipeline_steps)
