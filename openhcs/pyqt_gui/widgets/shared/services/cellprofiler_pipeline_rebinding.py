"""Shared UI service for preserving CellProfiler runtime bindings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from openhcs.core.steps.function_step import FunctionStep
from openhcs.interop.cellprofiler.import_records import (
    CellProfilerPipelineImportResult,
)
from openhcs.interop.cellprofiler.runtime.generated_pipeline import (
    CellProfilerGeneratedRuntimeBindingState,
    CellProfilerPipelineRuntimeRebinder,
)
from openhcs.pyqt_gui.services.plate_scope_identity import PlateScopeIdentity


class CellProfilerImportResultProvider(Protocol):
    """Pipeline-editor surface required for CellProfiler runtime rebinding."""

    current_plate: str | None
    cellprofiler_import_result: CellProfilerPipelineImportResult | None
    cellprofiler_import_results_by_plate: Mapping[str, CellProfilerPipelineImportResult]


class CellProfilerPipelineRuntimeBindingService:
    """Rebind generated CellProfiler steps to artifact-aware runtime callables."""

    @classmethod
    def runtime_bound_pipeline_for_plate(
        cls,
        *,
        plate_pipeline_editor: CellProfilerImportResultProvider | None,
        plate_path: str,
        pipeline_steps: Sequence[FunctionStep],
    ) -> list[FunctionStep]:
        steps = list(pipeline_steps)
        if not CellProfilerGeneratedRuntimeBindingState.pipeline_requires_rebinding(
            steps
        ):
            return steps

        if plate_pipeline_editor is None:
            raise RuntimeError(
                "Cannot compile or run CellProfiler pipeline code without a "
                "pipeline editor import context."
            )

        import_result = cls.import_result_for_plate(
            plate_pipeline_editor,
            plate_path,
        )
        if import_result is None:
            identity = PlateScopeIdentity.from_scope_id(plate_path)
            if identity.cppipe_path is None:
                return steps
            raise RuntimeError(
                "Cannot compile or run CellProfiler pipeline code for "
                f"{plate_path!r} because the .cppipe import context is not loaded. "
                "Initialize the plate before editing or running generated pipeline code."
            )

        return CellProfilerPipelineRuntimeRebinder.from_import_result(
            import_result,
        ).rebind(steps)

    @staticmethod
    def import_result_for_plate(
        plate_pipeline_editor: CellProfilerImportResultProvider,
        plate_path: str,
    ) -> CellProfilerPipelineImportResult | None:
        """Return the import result associated with one logical plate scope."""
        plate_key = str(plate_path)
        if plate_key in plate_pipeline_editor.cellprofiler_import_results_by_plate:
            return plate_pipeline_editor.cellprofiler_import_results_by_plate[plate_key]
        if plate_pipeline_editor.current_plate == plate_key:
            return plate_pipeline_editor.cellprofiler_import_result
        return None
