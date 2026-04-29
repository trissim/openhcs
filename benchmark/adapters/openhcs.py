"""OpenHCS tool adapter."""

from __future__ import annotations

import logging
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark.adapters.cppipe_source import (
    CPPipeSourceRequest,
    CPPipeSourceResolution,
    materialize_cppipe_reference,
    resolve_cppipe_source,
)
from benchmark.converter.runtime_pipeline import (
    execute_pipeline_direct,
    prepare_generated_pipeline,
)
from benchmark.converter.execution_validation import (
    CPPipeExecutionValidationError,
    validate_cppipe_execution,
)
from benchmark.contracts.tool_adapter import (
    BenchmarkResult,
    ToolAdapter,
    ToolExecutionError,
    ToolNotInstalledError,
)
from benchmark.contracts.metric import MetricCollector
from openhcs.constants.constants import Microscope
from openhcs.core.runtime_equivalence import (
    RuntimeOutputSnapshot,
    runtime_output_equivalence,
)
from openhcs.core.source_schema_workspace import materialize_source_schema_workspace

logger = logging.getLogger(__name__)


_MICROSCOPES_BY_NORMALIZED_LITERAL = {
    member.value.lower(): member for member in Microscope
}


@dataclass(frozen=True, slots=True)
class OpenHCSRunRequest:
    """Authoritative benchmark run request for one OpenHCS execution."""

    dataset_path: Path
    pipeline_name: str
    pipeline_params: dict[str, Any]
    metrics: tuple[MetricCollector, ...]
    output_dir: Path

    @property
    def dataset_id(self) -> str:
        return str(self.pipeline_params.get("dataset_id", self.dataset_path.name))

    @property
    def microscope_type(self) -> str | None:
        value = self.pipeline_params.get("microscope_type")
        if value is None:
            return None
        return str(value)

    @property
    def cppipe_source(self) -> CPPipeSourceRequest:
        return CPPipeSourceRequest.from_pipeline_params(
            dataset_id=self.dataset_id,
            output_dir=self.output_dir,
            pipeline_params=self.pipeline_params,
        )

    @property
    def equivalence_reference_output_dir(self) -> Path | None:
        value = self.pipeline_params.get("equivalence_reference_output_dir")
        if value is None:
            return None
        return Path(value)


class OpenHCSAdapter(ToolAdapter):
    """OpenHCS tool adapter."""

    name = "OpenHCS"

    def __init__(self):
        import openhcs

        self.version = openhcs.__version__

    def validate_installation(self) -> None:
        """Check OpenHCS is importable."""
        try:
            import openhcs  # noqa: F401
        except ImportError as exc:
            raise ToolNotInstalledError(f"OpenHCS not installed: {exc}") from exc

    def _run_converted_cppipe_pipeline(
        self,
        request: OpenHCSRunRequest,
    ) -> BenchmarkResult:
        """Execute a converted CellProfiler pipeline through the OpenHCS orchestrator."""
        from openhcs.config_framework.lazy_factory import ensure_global_config_context
        from openhcs.core.config import (
            GlobalPipelineConfig,
            LazyPathPlanningConfig,
            MaterializationBackend,
            PipelineConfig,
            VFSConfig,
        )
        from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator

        cppipe_source = self._resolve_cppipe_source(request)
        cppipe_path = cppipe_source.path
        reference_url = cppipe_source.reference_url

        output_suffix = f"_{request.pipeline_name}_converted_cppipe"
        output_plate_root = request.output_dir / f"{request.dataset_path.name}{output_suffix}"
        generated_module_path = request.output_dir / f"{cppipe_path.stem}_openhcs.py"
        try:
            prepared = prepare_generated_pipeline(
                cppipe_path,
                output_path=generated_module_path,
            )
        except ValueError as exc:
            raise ToolExecutionError(
                f"Failed to prepare converted .cppipe pipeline {cppipe_path.name}: "
                f"{exc}"
            ) from exc
        source_workspace = None
        execution_plate_path = request.dataset_path
        execution_microscope = self._configured_microscope(request.microscope_type)
        if not prepared.source_schema.is_empty:
            source_workspace_path = (
                request.output_dir
                / f"{request.dataset_path.name}_{cppipe_path.stem}_source_workspace"
            )
            try:
                source_workspace = materialize_source_schema_workspace(
                    request.dataset_path,
                    source_workspace_path,
                    prepared.source_schema,
                )
            except Exception as exc:
                raise ToolExecutionError(
                    f"Failed to materialize CellProfiler source schema for "
                    f"{cppipe_path.name}: {exc}"
                ) from exc
            execution_plate_path = source_workspace.workspace_root
            execution_microscope = Microscope.AUTO

        global_config = GlobalPipelineConfig(
            num_workers=1,
            use_threading=True,
            materialization_results_path=output_plate_root / "results",
            microscope=execution_microscope,
        )
        ensure_global_config_context(GlobalPipelineConfig, global_config)
        pipeline_config = PipelineConfig(
            path_planning_config=LazyPathPlanningConfig(
                global_output_folder=request.output_dir,
                output_dir_suffix=output_suffix,
            ),
            vfs_config=VFSConfig(
                materialization_backend=MaterializationBackend.DISK,
            ),
        )
        orchestrator = PipelineOrchestrator(
            execution_plate_path,
            pipeline_config=pipeline_config,
        )
        orchestrator.initialize()

        with ExitStack() as stack:
            for metric in request.metrics:
                stack.enter_context(metric)
            execution = execute_pipeline_direct(orchestrator, prepared.pipeline)
        try:
            validation = validate_cppipe_execution(
                prepared,
                execution,
                output_plate_root,
            )
        except CPPipeExecutionValidationError as exc:
            raise ToolExecutionError(str(exc)) from exc
        equivalence_reference = request.equivalence_reference_output_dir
        equivalence_report = None
        if equivalence_reference is not None:
            if not equivalence_reference.exists():
                raise ToolExecutionError(
                    f"Equivalence reference output directory does not exist: "
                    f"{equivalence_reference}"
                )
            equivalence_report = runtime_output_equivalence(
                RuntimeOutputSnapshot.from_output_root(equivalence_reference),
                RuntimeOutputSnapshot.from_export_observation(
                    validation.observation.exports
                ),
            )
            if not equivalence_report.is_equivalent:
                raise ToolExecutionError(
                    "Converted CellProfiler output did not match semantic "
                    f"reference output {equivalence_reference}:\n"
                    + "\n".join(
                        f"- {message}"
                        for message in equivalence_report.failure_messages()
                    )
                )

        metric_results: dict[str, Any] = {
            metric.name: metric.get_result() for metric in request.metrics
        }
        output_plate_root.mkdir(parents=True, exist_ok=True)

        provenance = {
            "openhcs_version": self.version,
            "microscope_type": request.microscope_type,
            "pipeline_source": "converted_cppipe",
            "cppipe_path": str(cppipe_path),
            "generated_pipeline_module": prepared.module_name,
            "axis_count": len(execution.execution_results),
            "csv_output_count": len(validation.observation.exports.table_outputs),
            "image_output_count": len(validation.observation.exports.image_outputs),
        }
        if equivalence_reference is not None:
            provenance["equivalence_reference_output_dir"] = str(equivalence_reference)
            provenance["equivalence_difference_count"] = len(
                equivalence_report.differences if equivalence_report else ()
            )
        if source_workspace is not None:
            provenance["source_workspace"] = str(source_workspace.workspace_root)
        if reference_url is not None:
            provenance["cppipe_reference_url"] = reference_url

        return BenchmarkResult(
            tool_name=self.name,
            dataset_id=request.dataset_id,
            pipeline_name=request.pipeline_name,
            metrics=metric_results,
            output_path=output_plate_root,
            success=True,
            error_message=None,
            provenance=provenance,
        )

    def _configured_microscope(
        self,
        microscope_type: str | None,
    ) -> Microscope:
        """Normalize benchmark microscope literals onto the OpenHCS enum SSOT."""
        if microscope_type is None:
            return Microscope.AUTO
        normalized = microscope_type.strip().lower()
        try:
            return _MICROSCOPES_BY_NORMALIZED_LITERAL[normalized]
        except KeyError as exc:
            raise ToolExecutionError(
                f"Unsupported OpenHCS microscope_type {microscope_type!r}."
            ) from exc

    def _resolve_cppipe_source(
        self,
        request: OpenHCSRunRequest,
    ) -> CPPipeSourceResolution:
        """Resolve .cppipe source metadata through the shared adapter helper."""
        return resolve_cppipe_source(
            request.cppipe_source,
            materialize_reference=self._materialize_cppipe_reference,
        )

    def _materialize_cppipe_reference(
        self,
        reference_url: str,
        target_dir: Path,
    ) -> Path:
        """Download one canonical .cppipe file into a stable local path."""
        return materialize_cppipe_reference(reference_url, target_dir)

    def run(
        self,
        dataset_path: Path,
        pipeline_name: str,
        pipeline_params: dict[str, Any],
        metrics: list[Any],
        output_dir: Path,
    ) -> BenchmarkResult:
        """Execute OpenHCS pipeline with metrics."""
        output_dir.mkdir(parents=True, exist_ok=True)

        request = OpenHCSRunRequest(
            dataset_path=dataset_path,
            pipeline_name=pipeline_name,
            pipeline_params=pipeline_params,
            metrics=self._validated_metric_collectors(metrics),
            output_dir=output_dir,
        )
        return self._run_converted_cppipe_pipeline(request)

    def _validated_metric_collectors(
        self,
        metrics: list[Any],
    ) -> tuple[MetricCollector, ...]:
        """Validate metric collectors once and return a typed immutable bundle."""
        validated_metrics: list[MetricCollector] = []
        for metric in metrics:
            if not isinstance(metric, MetricCollector):
                raise ToolExecutionError(
                    f"Metric {metric} does not extend MetricCollector"
                )
            validated_metrics.append(metric)
        return tuple(validated_metrics)
