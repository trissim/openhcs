"""OpenHCS tool adapter."""

from __future__ import annotations

import json
import logging
import pickle
import importlib.util
import os
import signal
import threading
import time
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from benchmark.adapters.cppipe_source import (
    CPPipeSourceRequest,
    CPPipeSourceResolution,
    materialize_cppipe_reference,
    resolve_cppipe_source,
)
from benchmark.converter.runtime_pipeline import execute_pipeline_direct
from benchmark.converter.execution_validation import (
    CPPipeExecutionValidation,
    CPPipeExecutionValidationError,
    validate_cppipe_execution,
)
from benchmark.timing import BenchmarkPhase, PhaseTimingTrace
from openhcs.interop.cellprofiler.measurement_dialect import (
    cellprofiler_runtime_equivalence_policy,
)
from benchmark.contracts.tool_adapter import (
    BenchmarkResult,
    ToolAdapter,
    ToolExecutionError,
    ToolNotInstalledError,
)
from benchmark.contracts.metric import MetricCollector
from openhcs.core.config import (
    CompilationDebugConfig,
    GlobalPipelineConfig,
    LazyCompilationDebugConfig,
    LazyWellFilterConfig,
)
from openhcs.core.equivalence import RuntimeEquivalencePolicy
from openhcs.core.equivalence.outputs import RuntimeOutputSnapshot
from openhcs.core.runtime_equivalence import (
    runtime_reference_artifact_equivalence,
)
from openhcs.core.runtime_execution_validation import runtime_output_roots
from openhcs.core.runtime_exports import RuntimeExportObservation
from openhcs.core.source_schema_workspace import SourceSchemaImageSetSelection
from openhcs.interop.cellprofiler.source_schema_ingestion import (
    CellProfilerPipelinePreparationError,
    CellProfilerSourceSchemaWorkspaceRequest,
    CellProfilerSourceWorkspaceMaterializationError,
    prepare_cellprofiler_source_schema_workspace,
)

logger = logging.getLogger(__name__)


_RUNTIME_EXECUTION_CACHE_SCHEMA_VERSION = 1
_RUNTIME_EXECUTION_OBSERVATION_PICKLE_NAME = "runtime_execution_observation.pkl"
_DUMP_COMPILED_PLANS_ENV = "OPENHCS_BENCHMARK_DUMP_COMPILED_PLANS"


def _strict_cellprofiler_runtime_equivalence_policy() -> RuntimeEquivalencePolicy:
    """Return the benchmark parity policy with broad dialect relaxations disabled."""
    return cellprofiler_runtime_equivalence_policy(
        numeric_abs_tolerance=1e-6,
        numeric_rel_tolerance=1e-6,
        feature_numeric_tolerances=(),
        allow_extra_candidate_measurements=False,
        allow_tie_sensitive_location_mismatches=True,
        allow_unstable_shape_descriptors=False,
        allow_sparse_object_boundary_jitter=False,
        allow_unstable_zernike_descriptors=False,
        threshold_entropy_abs_tolerance=1e-6,
        threshold_sensitive_pair_abs_tolerance=1e-6,
        threshold_sensitive_pair_rel_tolerance=1e-6,
        image_abs_tolerance=1e-6,
        image_rel_tolerance=1e-6,
        image_max_different_fraction=0.0,
    )


@dataclass(frozen=True, slots=True)
class OpenHCSRunRequest:
    """Authoritative benchmark run request for one OpenHCS execution."""

    dataset_path: Path
    pipeline_name: str
    pipeline_params: dict[str, Any]
    metrics: tuple[MetricCollector, ...]
    output_dir: Path
    source_schema_image_set_selection: SourceSchemaImageSetSelection | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir).resolve())

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

    @property
    def runtime_execution_cache_manifest(self) -> Path | None:
        value = self.pipeline_params.get("runtime_execution_cache_manifest")
        if value is None:
            return None
        return Path(value)

    @property
    def runtime_execution_cache_key(self) -> object | None:
        return self.pipeline_params.get("runtime_execution_cache_key")

    @property
    def reuse_runtime_execution_cache(self) -> bool:
        return bool(self.pipeline_params.get("reuse_runtime_execution_cache", True))

    @property
    def compare_image_outputs(self) -> bool:
        return bool(self.pipeline_params.get("compare_image_outputs", True))

    @property
    def materialize_runtime_artifacts(self) -> bool:
        return bool(self.pipeline_params.get("materialize_runtime_artifacts", True))

    @property
    def raise_on_equivalence_failure(self) -> bool:
        return bool(self.pipeline_params.get("raise_on_equivalence_failure", True))

    @property
    def openhcs_timeout_seconds(self) -> float:
        value = self.pipeline_params.get("openhcs_timeout_seconds")
        if value is None:
            value = os.environ.get("OPENHCS_BENCHMARK_OPENHCS_TIMEOUT_SECONDS", "120")
        seconds = float(value)
        if seconds <= 0:
            raise ValueError("openhcs_timeout_seconds must be positive.")
        return seconds

    @property
    def dump_compiled_plans(self) -> bool:
        value = self.pipeline_params.get("dump_compiled_plans")
        if value is None:
            value = os.environ.get(_DUMP_COMPILED_PLANS_ENV)
        return _truthy_debug_flag(value)


@dataclass(frozen=True, slots=True)
class RuntimeExecutionCacheWritePolicy:
    """Typed cache-write contract for OpenHCS runtime execution observations."""

    write_manifest: bool

    @classmethod
    def for_request(
        cls, request: OpenHCSRunRequest
    ) -> "RuntimeExecutionCacheWritePolicy":
        if (
            request.runtime_execution_cache_manifest is None
            or request.runtime_execution_cache_key is None
        ):
            return cls.disabled()
        return cls(write_manifest=True)

    @classmethod
    def disabled(cls) -> "RuntimeExecutionCacheWritePolicy":
        return cls(write_manifest=False)


@dataclass(frozen=True, slots=True)
class _RuntimeExecutionCacheHit:
    """Cached OpenHCS execution state, before external equivalence comparison."""

    validation: CPPipeExecutionValidation
    output_roots: tuple[Path, ...]
    execution_output_root: Path
    axis_count: int


def _runtime_execution_cache_key_matches(
    cached_key: object,
    expected_key: object,
) -> bool:
    """Return whether a runtime execution cache key is valid for this request."""
    return cached_key == expected_key


class OpenHCSAdapter(ToolAdapter):
    """OpenHCS tool adapter."""

    name = "OpenHCS"

    def __init__(
        self,
        *,
        global_config: GlobalPipelineConfig | None = None,
        source_schema_image_set_selection: SourceSchemaImageSetSelection | None = None,
    ) -> None:
        import openhcs
        from polystore.base import ensure_storage_registry, storage_registry
        from polystore.filemanager import FileManager

        self.version = openhcs.__version__
        ensure_storage_registry()
        self._filemanager = FileManager(storage_registry)
        self.global_config = global_config or GlobalPipelineConfig()
        self.source_schema_image_set_selection = source_schema_image_set_selection

    def validate_installation(self) -> None:
        """Check OpenHCS is importable."""
        if importlib.util.find_spec("openhcs") is None:
            raise ToolNotInstalledError("OpenHCS not installed")
        import openhcs  # noqa: F401

    def _run_converted_cppipe_pipeline(
        self,
        request: OpenHCSRunRequest,
    ) -> BenchmarkResult:
        """Execute a converted CellProfiler pipeline through the OpenHCS orchestrator."""
        from openhcs.config_framework.lazy_factory import (
            ensure_global_config_context,
            rebuild_lazy_config_with_new_global_reference,
        )
        from openhcs.core.config import (
            AnalysisConsolidationConfig,
            MaterializationBackend,
            PathPlanningConfig,
            PipelineConfig,
            VFSConfig,
        )
        from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator

        phase_timing = PhaseTimingTrace(
            run_id=f"{request.dataset_id}:{request.pipeline_name}:openhcs",
            pipeline_name=request.pipeline_name,
            tool=self.name,
        )
        with phase_timing.phase(BenchmarkPhase.RESOLVE_SOURCE):
            cppipe_source = self._resolve_cppipe_source(request)
        cppipe_path = cppipe_source.path
        reference_url = cppipe_source.reference_url

        output_suffix = f"_{request.pipeline_name}_converted_cppipe"
        output_plate_root = (
            request.output_dir / f"{request.dataset_path.name}{output_suffix}"
        )
        generated_module_path = request.output_dir / f"{cppipe_path.stem}_openhcs.py"
        generated_pipeline_module_name = generated_module_path.stem
        source_workspace_path = (
            request.output_dir
            / f"{request.dataset_path.name}_{cppipe_path.stem}_source_workspace"
        )
        compilation_debug_config = _benchmark_compilation_debug_config(
            self.global_config.compilation_debug_config,
            request=request,
            cppipe_path=cppipe_path,
        )
        compiled_bundle_dump_path = (
            compilation_debug_config.compiled_execution_bundle_path
        )

        with phase_timing.phase(BenchmarkPhase.READ_CACHE):
            cache_hit = (
                None
                if compilation_debug_config.enabled
                else self._load_runtime_execution_cache(request)
            )
        equivalence_report = None
        equivalence_failure_message = None
        if cache_hit is not None:
            phase_timing.record(
                BenchmarkPhase.COMPILE_OPENHCS, seconds=0.0, cached=True
            )
            phase_timing.record(
                BenchmarkPhase.EXECUTE_OPENHCS, seconds=0.0, cached=True
            )
            phase_timing.record(
                BenchmarkPhase.VALIDATE_RUNTIME,
                seconds=0.0,
                cached=True,
            )
            validation = cache_hit.validation
            output_roots = cache_hit.output_roots
            execution_output_root = cache_hit.execution_output_root
            axis_count = cache_hit.axis_count
            executed_axes = tuple(validation.observation.records_by_axis)
            csv_output_count = len(validation.observation.exports.table_outputs)
            image_output_count = len(validation.observation.exports.image_outputs)
            reused_runtime_execution_cache = True
        else:
            try:
                with phase_timing.phase(BenchmarkPhase.COMPILE_DIALECT):
                    ingestion = prepare_cellprofiler_source_schema_workspace(
                        CellProfilerSourceSchemaWorkspaceRequest(
                            source_root=request.dataset_path,
                            cppipe_path=cppipe_path,
                            workspace_root=source_workspace_path,
                            generated_pipeline_path=generated_module_path,
                            filemanager=self._filemanager,
                            image_set_selection=(
                                request.source_schema_image_set_selection
                            ),
                        )
                    )
            except CellProfilerPipelinePreparationError as exc:
                raise ToolExecutionError(str(exc)) from exc
            except CellProfilerSourceWorkspaceMaterializationError as exc:
                raise ToolExecutionError(str(exc)) from exc

            prepared = ingestion.prepared_pipeline
            generated_pipeline_module_name = prepared.module_name
            execution_plate_path = ingestion.execution_plate_path
            source_workspace_path = ingestion.source_workspace_path
            pipeline_config = (
                prepared.generated_pipeline.pipeline_config or PipelineConfig()
            )
            selection = request.source_schema_image_set_selection
            if selection is not None and selection.well_filter:
                pipeline_config = replace(
                    pipeline_config,
                    well_filter_config=LazyWellFilterConfig(
                        well_filter=list(selection.well_filter),
                    ),
                )
            if compilation_debug_config.enabled:
                pipeline_config = replace(
                    pipeline_config,
                    compilation_debug_config=LazyCompilationDebugConfig(
                        enabled=compilation_debug_config.enabled,
                        compiled_execution_bundle_path=(
                            compilation_debug_config.compiled_execution_bundle_path
                        ),
                    ),
                )

            global_config = replace(
                self.global_config,
                analysis_consolidation_config=AnalysisConsolidationConfig(
                    enabled=False,
                ),
                path_planning_config=PathPlanningConfig(
                    global_output_folder=request.output_dir,
                    output_dir_suffix=output_suffix,
                ),
                vfs_config=VFSConfig(
                    materialization_backend=MaterializationBackend.DISK,
                ),
                compilation_debug_config=compilation_debug_config,
                materialize_runtime_artifacts=request.materialize_runtime_artifacts,
                materialization_results_path=output_plate_root / "results",
            )
            ensure_global_config_context(GlobalPipelineConfig, global_config)
            pipeline_config = rebuild_lazy_config_with_new_global_reference(
                pipeline_config,
                global_config,
                GlobalPipelineConfig,
            )
            orchestrator = PipelineOrchestrator(
                execution_plate_path,
                pipeline_config=pipeline_config,
            )
            with phase_timing.phase(BenchmarkPhase.INITIALIZE_RUNTIME):
                orchestrator.initialize()
            with ExitStack() as stack:
                for metric in request.metrics:
                    stack.enter_context(metric)
                with _openhcs_execution_watchdog(request.openhcs_timeout_seconds):
                    execution = execute_pipeline_direct(
                        orchestrator,
                        prepared.pipeline,
                        phase_timing=phase_timing,
                    )
            output_roots = runtime_output_roots(
                execution.compiled_contexts,
                output_plate_root,
            )
            execution_output_root = (
                output_roots[0] if len(output_roots) == 1 else request.output_dir
            )
            try:
                with phase_timing.phase(BenchmarkPhase.VALIDATE_RUNTIME):
                    validation = validate_cppipe_execution(
                        prepared,
                        execution,
                        execution_output_root,
                        validate_table_exports=request.materialize_runtime_artifacts,
                        validate_image_exports=request.compare_image_outputs,
                    )
            except CPPipeExecutionValidationError as exc:
                raise ToolExecutionError(str(exc)) from exc
            axis_count = len(execution.execution_results)
            executed_axes = tuple(validation.observation.records_by_axis)
            csv_output_count = len(validation.observation.exports.table_outputs)
            image_output_count = len(validation.observation.exports.image_outputs)
            reused_runtime_execution_cache = False
            with phase_timing.phase(BenchmarkPhase.WRITE_CACHE):
                self._write_runtime_execution_cache(
                    request,
                    validation=validation,
                    output_roots=output_roots,
                    execution_output_root=execution_output_root,
                    axis_count=axis_count,
                )
            reused_runtime_execution_cache = False
        equivalence_reference = request.equivalence_reference_output_dir
        if equivalence_reference is not None:
            if not equivalence_reference.exists():
                raise ToolExecutionError(
                    f"Equivalence reference output directory does not exist: "
                    f"{equivalence_reference}"
                )
            equivalence_policy = _strict_cellprofiler_runtime_equivalence_policy()
            with phase_timing.phase(BenchmarkPhase.COMPARE_EQUIVALENCE):
                reference_snapshot = RuntimeOutputSnapshot.from_output_root(
                    equivalence_reference
                )
                if not request.compare_image_outputs:
                    reference_snapshot = RuntimeOutputSnapshot(
                        tables=reference_snapshot.tables,
                    )
                equivalence_report = runtime_reference_artifact_equivalence(
                    reference_snapshot,
                    validation.observation,
                    policy=equivalence_policy,
                    candidate_image_artifact_names=(
                        validation.expectation.exports.image_artifact_names
                    ),
                    candidate_image_export_specs=(
                        validation.expectation.exports.image_export_specs
                    ),
                    candidate_image_snapshots=(
                        _candidate_image_snapshots_for_equivalence(validation)
                    ),
                )
            if not equivalence_report.is_equivalent:
                equivalence_failure_message = (
                    "Converted CellProfiler output did not match semantic "
                    f"reference output {equivalence_reference}:\n"
                    + "\n".join(
                        f"- {message}"
                        for message in equivalence_report.failure_messages()
                    )
                )
                if request.raise_on_equivalence_failure:
                    raise ToolExecutionError(equivalence_failure_message)

        metric_results = self._metric_results(request.metrics)
        output_plate_root.mkdir(parents=True, exist_ok=True)
        execution_output_root.mkdir(parents=True, exist_ok=True)

        provenance = {
            "openhcs_version": self.version,
            "microscope_type": request.microscope_type,
            "pipeline_source": "converted_cppipe",
            "cppipe_path": str(cppipe_path),
            "generated_pipeline_module": generated_pipeline_module_name,
            "axis_count": axis_count,
            "csv_output_count": csv_output_count,
            "image_output_count": image_output_count,
            "compiled_output_roots": tuple(str(root) for root in output_roots),
            "reused_runtime_execution_cache": reused_runtime_execution_cache,
            "phase_timing_records": phase_timing.payloads(),
            "executed_axes": executed_axes,
        }
        if request.runtime_execution_cache_manifest is not None:
            provenance["runtime_execution_cache_manifest"] = str(
                request.runtime_execution_cache_manifest
            )
        if compiled_bundle_dump_path is not None:
            provenance["compiled_execution_bundle_path"] = str(
                compiled_bundle_dump_path
            )
        if equivalence_reference is not None:
            provenance["equivalence_reference_output_dir"] = str(equivalence_reference)
            provenance["equivalence_difference_count"] = len(
                equivalence_report.differences if equivalence_report else ()
            )
        if reference_url is not None:
            provenance["cppipe_reference_url"] = reference_url

        return BenchmarkResult(
            tool_name=self.name,
            dataset_id=request.dataset_id,
            pipeline_name=request.pipeline_name,
            metrics=metric_results,
            output_path=execution_output_root,
            success=equivalence_failure_message is None,
            error_message=equivalence_failure_message,
            provenance=provenance,
        )

    def _load_runtime_execution_cache(
        self,
        request: OpenHCSRunRequest,
    ) -> _RuntimeExecutionCacheHit | None:
        """Load a validated OpenHCS execution snapshot when cache identity matches."""
        manifest_path = request.runtime_execution_cache_manifest
        cache_key = request.runtime_execution_cache_key
        if (
            manifest_path is None
            or cache_key is None
            or not request.reuse_runtime_execution_cache
            or not manifest_path.exists()
        ):
            return None
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if manifest.get("schema_version") != _RUNTIME_EXECUTION_CACHE_SCHEMA_VERSION:
            return None
        if not _runtime_execution_cache_key_matches(
            manifest.get("cache_key"),
            cache_key,
        ):
            return None
        validation_path = _cache_payload_path(
            manifest_path,
            manifest.get("validation_pickle_path"),
        )
        if validation_path is None or not validation_path.exists():
            return None
        output_roots = tuple(Path(path) for path in manifest.get("output_roots", ()))
        execution_output_root_value = manifest.get("execution_output_root")
        if not execution_output_root_value:
            return None
        execution_output_root = Path(str(execution_output_root_value))
        if not execution_output_root.exists():
            return None
        if any(not root.exists() for root in output_roots):
            return None
        try:
            with validation_path.open("rb") as handle:
                validation_payload = pickle.load(handle)
            validation = _validation_from_cache_payload(validation_payload)
        except Exception:
            logger.exception(
                "Failed to load OpenHCS runtime execution cache %s",
                validation_path,
            )
            return None
        return _RuntimeExecutionCacheHit(
            validation=validation,
            output_roots=output_roots,
            execution_output_root=execution_output_root,
            axis_count=int(manifest.get("axis_count", 0)),
        )

    def _write_runtime_execution_cache(
        self,
        request: OpenHCSRunRequest,
        *,
        validation: CPPipeExecutionValidation,
        output_roots: tuple[Path, ...],
        execution_output_root: Path,
        axis_count: int,
    ) -> None:
        """Persist completed OpenHCS execution state before equivalence comparison."""
        manifest_path = request.runtime_execution_cache_manifest
        cache_key = request.runtime_execution_cache_key
        policy = RuntimeExecutionCacheWritePolicy.for_request(request)
        if not policy.write_manifest or manifest_path is None or cache_key is None:
            return
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        validation_path = (
            manifest_path.parent / _RUNTIME_EXECUTION_OBSERVATION_PICKLE_NAME
        )
        with validation_path.open("wb") as handle:
            pickle.dump(
                _validation_cache_payload(validation),
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": _RUNTIME_EXECUTION_CACHE_SCHEMA_VERSION,
                    "cache_key": cache_key,
                    "validation_pickle_path": validation_path.name,
                    "output_roots": tuple(str(root) for root in output_roots),
                    "execution_output_root": str(execution_output_root),
                    "axis_count": axis_count,
                },
                indent=2,
                sort_keys=True,
            )
        )

    def _metric_results(
        self,
        metrics: tuple[MetricCollector, ...],
    ) -> dict[str, Any]:
        """Return metric results, skipping metrics unused by cached execution."""
        results: dict[str, Any] = {}
        for metric in metrics:
            try:
                results[metric.name] = metric.get_result()
            except RuntimeError:
                continue
        return results

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
            source_schema_image_set_selection=self.source_schema_image_set_selection,
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


@contextmanager
def _openhcs_execution_watchdog(timeout_seconds: float):
    """Interrupt benchmark OpenHCS execution that exceeds the run budget."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)

    def _raise_timeout(_signum: int, _frame: object) -> None:
        raise TimeoutError(
            f"OpenHCS execution exceeded {timeout_seconds:.1f}s watchdog."
        )

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        yield
    except TimeoutError as exc:
        raise ToolExecutionError(str(exc)) from exc
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _validation_cache_payload(
    validation: CPPipeExecutionValidation,
) -> dict[str, Any]:
    """Return a pickle-safe payload for runtime execution validation."""
    exports = validation.observation.exports
    return {
        "expectation": validation.expectation,
        "records_by_axis": {
            axis: tuple(records)
            for axis, records in validation.observation.records_by_axis.items()
        },
        "exports": {
            "table_outputs": tuple(str(path) for path in exports.table_outputs),
            "image_outputs": tuple(str(path) for path in exports.image_outputs),
            "table_headers_by_path": {
                str(path): tuple(headers)
                for path, headers in exports.table_headers_by_path.items()
            },
            "table_row_counts_by_path": {
                str(path): int(row_count)
                for path, row_count in exports.table_row_counts_by_path.items()
            },
        },
    }


def _candidate_image_snapshots_for_equivalence(
    validation: CPPipeExecutionValidation,
) -> tuple[Any, ...] | None:
    """Return candidate export snapshots when exports are the authoritative images.

    SaveImages declares exact runtime image artifacts and export encodings.  In
    that case equivalence must use the typed artifact records, not incidental
    final-step image files that OpenHCS may also materialize.
    """
    if validation.expectation.exports.image_export_specs:
        return None
    if not validation.observation.exports.image_outputs:
        return None
    return RuntimeOutputSnapshot.from_export_observation(
        validation.observation.exports
    ).images


def _validation_from_cache_payload(
    payload: object,
) -> CPPipeExecutionValidation:
    """Rebuild runtime execution validation from a pickle-safe payload."""
    if not isinstance(payload, Mapping):
        raise TypeError(
            "OpenHCS runtime execution cache payload must be a mapping, "
            f"got {type(payload).__name__}."
        )
    exports_payload = payload.get("exports")
    if not isinstance(exports_payload, Mapping):
        raise TypeError("OpenHCS runtime execution cache exports are missing.")
    exports = RuntimeExportObservation(
        table_outputs=tuple(
            Path(path) for path in exports_payload.get("table_outputs", ())
        ),
        image_outputs=tuple(
            Path(path) for path in exports_payload.get("image_outputs", ())
        ),
        table_headers_by_path={
            Path(path): tuple(headers)
            for path, headers in (
                exports_payload.get("table_headers_by_path", {}) or {}
            ).items()
        },
        table_row_counts_by_path={
            Path(path): int(row_count)
            for path, row_count in (
                exports_payload.get("table_row_counts_by_path", {}) or {}
            ).items()
        },
    )
    from openhcs.core.runtime_execution_validation import (
        RuntimeArtifactExecutionObservation,
    )

    return CPPipeExecutionValidation(
        expectation=payload["expectation"],
        observation=RuntimeArtifactExecutionObservation(
            records_by_axis={
                str(axis): tuple(records)
                for axis, records in (payload.get("records_by_axis", {}) or {}).items()
            },
            exports=exports,
        ),
    )


def _cache_payload_path(
    manifest_path: Path,
    value: object,
) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def _truthy_debug_flag(value: object) -> bool:
    """Return whether a benchmark debug flag is enabled."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _benchmark_compilation_debug_config(
    base_config: CompilationDebugConfig,
    *,
    request: OpenHCSRunRequest,
    cppipe_path: Path,
) -> CompilationDebugConfig:
    if not request.dump_compiled_plans:
        return base_config
    return replace(
        base_config,
        enabled=True,
        compiled_execution_bundle_path=(
            request.output_dir / f"{cppipe_path.stem}_compiled_execution_bundle.pkl"
        ),
    )
