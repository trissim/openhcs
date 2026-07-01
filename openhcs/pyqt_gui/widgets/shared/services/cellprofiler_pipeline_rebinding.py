"""Shared UI service for preserving CellProfiler runtime bindings."""

from __future__ import annotations

from collections.abc import Sequence
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
    """Plate/workspace surface required for CellProfiler runtime rebinding."""

    def cellprofiler_import_result_for_plate(
        self,
        plate_path: str,
    ) -> CellProfilerPipelineImportResult | None:
        """Return the import result for one logical plate scope."""
        ...


class CellProfilerPipelineRuntimeBindingService:
    """Rebind generated CellProfiler steps to artifact-aware runtime callables."""

    @classmethod
    def runtime_bound_pipeline_for_plate(
        cls,
        *,
        import_result_provider: CellProfilerImportResultProvider | None,
        plate_path: str,
        pipeline_steps: Sequence[FunctionStep],
    ) -> list[FunctionStep]:
        steps = list(pipeline_steps)
        if not CellProfilerGeneratedRuntimeBindingState.pipeline_requires_rebinding(
            steps
        ):
            return steps

        if import_result_provider is None:
            raise RuntimeError(
                "Cannot compile or run CellProfiler pipeline code without a "
                "CellProfiler import context."
            )

        import_result = cls.import_result_for_plate(
            import_result_provider,
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
        import_result_provider: CellProfilerImportResultProvider,
        plate_path: str,
    ) -> CellProfilerPipelineImportResult | None:
        """Return the import result associated with one logical plate scope."""
        return import_result_provider.cellprofiler_import_result_for_plate(
            str(plate_path)
        )
