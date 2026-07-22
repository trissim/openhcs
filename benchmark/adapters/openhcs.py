"""OpenHCS tool adapter."""

from __future__ import annotations
from openhcs.core.pipeline_document import PipelineDocumentAuthority

import hashlib
import importlib.util
import json
import logging
import os
import signal
import threading
import time
from collections.abc import Mapping, Sequence
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
from benchmark.contracts.tool_adapter import (
    BenchmarkResult,
    ToolAdapter,
    ToolExecutionError,
    ToolNotInstalledError,
)
from benchmark.contracts.metric import MetricCollector
from benchmark.cellprofiler_export_equivalence import (
    cellprofiler_database_export_equivalence,
)
from benchmark.timing import BenchmarkPhase, PhaseTimingTrace
from openhcs.core.config import (
    CompilationDebugConfig,
    GlobalPipelineConfig,
    LazyCompilationDebugConfig,
)
from openhcs.core.equivalence import RuntimeEquivalencePolicy, RuntimeEquivalenceReport
from openhcs.core.equivalence.outputs import RuntimeOutputSnapshot
from openhcs.core.function_step_transport import FunctionStepTransportAuthority
from openhcs.core.input_workspace import InputWorkspacePreparationRequest
from openhcs.core.runtime_equivalence import (
    runtime_reference_artifact_equivalence,
)
from openhcs.core.steps.abstract import AbstractStep
from openhcs.interop.cellprofiler.measurement_dialect import (
    cellprofiler_runtime_equivalence_policy,
)
from openhcs.interop.cellprofiler.plate_workspace import (
    prepare_cellprofiler_input_workspace,
)
from openhcs.runtime.zmq_execution_client import (
    OpenHCSExecutionSubmission,
    ZMQExecutionClient,
)
from openhcs.runtime.zmq_execution_observation import (
    ZMQRuntimeExecutionObservationExport,
)
from zmqruntime.execution import ExecutionSubmissionResponse, ExecutionWaitResult

logger = logging.getLogger(__name__)


_DUMP_COMPILED_PLANS_ENV = "OPENHCS_BENCHMARK_DUMP_COMPILED_PLANS"
ZMQ_RESULTS_SUMMARY_FILENAME = "zmq_results_summary.json"


@dataclass(frozen=True, slots=True)
class _ZMQOpenHCSExecution:
    """Server-side execution observation returned to the benchmark adapter."""

    execution_id: str
    observation_export: ZMQRuntimeExecutionObservationExport
    output_roots: tuple[Path, ...]
    results_summary: Mapping[str, Any]

    @property
    def execution_output_root(self) -> Path:
        if len(self.output_roots) == 1:
            return self.output_roots[0]
        summary_root = self.results_summary.get("output_plate_root")
        if summary_root is not None:
            return Path(str(summary_root))
        return self.output_roots[0] if self.output_roots else Path(".")

    @property
    def axis_count(self) -> int:
        return self.observation_export.axis_count


@dataclass(slots=True)
class _ZMQProgressTimingObserver:
    """Capture server progress timestamps for benchmark phase accounting."""

    compile_started_at: float | None = None
    compile_completed_at: float | None = None
    execution_started_at: float | None = None
    execution_completed_at: float | None = None

    def __call__(self, event: Mapping[str, Any]) -> None:
        phase = str(event.get("phase", ""))
        status = str(event.get("status", ""))
        timestamp = self._timestamp(event)
        if phase == "compile" and status == "started":
            self.compile_started_at = self.compile_started_at or timestamp
            return
        if phase == "compile" and status == "success":
            self.compile_completed_at = timestamp
            return
        if phase == "axis_started":
            self.execution_started_at = self.execution_started_at or timestamp
            return
        if phase == "axis_completed":
            self.execution_completed_at = timestamp

    @staticmethod
    def _timestamp(event: Mapping[str, Any]) -> float:
        value = event.get("timestamp")
        if isinstance(value, (int, float)):
            return float(value)
        return time.time()

    def record_phase_timings(
        self,
        phase_timing: PhaseTimingTrace,
        *,
        completion_observed_at: float | None = None,
    ) -> None:
        compile_seconds = self._duration(
            self.compile_started_at,
            self.compile_completed_at,
        )
        if compile_seconds is not None:
            phase_timing.record(
                BenchmarkPhase.COMPILE_OPENHCS,
                seconds=compile_seconds,
            )
        execute_seconds = self._duration(
            self.execution_started_at,
            self.execution_completed_at,
        )
        if execute_seconds is None:
            execute_seconds = self._completion_bounded_execution_seconds(
                completion_observed_at,
                compile_seconds=compile_seconds,
                wait_seconds=_phase_seconds_total(
                    phase_timing,
                    BenchmarkPhase.WAIT_OPENHCS,
                ),
            )
        if execute_seconds is not None:
            phase_timing.record(
                BenchmarkPhase.EXECUTE_OPENHCS,
                seconds=execute_seconds,
            )

    @staticmethod
    def _duration(started_at: float | None, ended_at: float | None) -> float | None:
        if started_at is None or ended_at is None:
            return None
        return max(0.0, ended_at - started_at)

    def _completion_bounded_execution_seconds(
        self,
        completion_observed_at: float | None,
        *,
        compile_seconds: float | None,
        wait_seconds: float | None,
    ) -> float | None:
        start_at = self.execution_started_at or self.compile_completed_at
        if start_at is not None and completion_observed_at is not None:
            return max(0.0, completion_observed_at - start_at)
        if wait_seconds is None:
            return None
        return max(0.0, wait_seconds - (compile_seconds or 0.0))


def _phase_seconds_total(
    phase_timing: PhaseTimingTrace,
    phase: BenchmarkPhase,
) -> float | None:
    records = [
        record.seconds for record in phase_timing.records if record.phase is phase
    ]
    if not records:
        return None
    return sum(records)


def _strict_cellprofiler_runtime_equivalence_policy() -> RuntimeEquivalencePolicy:
    """Return the benchmark parity policy with broad dialect relaxations disabled."""
    return cellprofiler_runtime_equivalence_policy(
        numeric_abs_tolerance=1e-6,
        numeric_rel_tolerance=1e-6,
        feature_numeric_tolerances=(),
        allow_extra_candidate_measurements=False,
        allow_tie_sensitive_location_mismatches=False,
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


def _execute_pipeline_via_zmq_server(
    *,
    plate_id: str | Path,
    execution_plate_id: str | Path,
    selected_pipeline_path: str | Path,
    pipeline_steps: Sequence[AbstractStep],
    global_config: GlobalPipelineConfig,
    pipeline_config: Any,
    observation_export_path: Path,
    phase_timing: PhaseTimingTrace,
) -> tuple[_ZMQOpenHCSExecution, str]:
    """Submit pipeline through the ZMQ compiler/executor and load observation."""

    transport_pipeline = FunctionStepTransportAuthority.normalize_pipeline(
        pipeline_steps
    )
    submission = OpenHCSExecutionSubmission(
        plate_id=plate_id,
        execution_plate_id=execution_plate_id,
        selected_pipeline_path=selected_pipeline_path,
        pipeline_document=PipelineDocumentAuthority.from_values(
            pipeline_config=pipeline_config, pipeline_steps=transport_pipeline
        ),
        global_config=global_config,
        config_params={
            "runtime_observation_export_path": str(observation_export_path),
        },
    )
    pipeline_source = submission.pipeline_code()
    timing_observer = _ZMQProgressTimingObserver()
    client = ZMQExecutionClient(persistent=False, progress_callback=timing_observer)
    try:
        with client:
            with phase_timing.phase(BenchmarkPhase.SUBMIT_OPENHCS):
                compile_submission_response = ExecutionSubmissionResponse.from_wire(
                    client.submit_compile(submission)
                )
            if not compile_submission_response.accepted:
                raise ToolExecutionError(
                    compile_submission_response.require_failure_text(
                        "OpenHCS ZMQ compile submission"
                    )
                )
            compile_artifact_id = compile_submission_response.require_execution_id(
                "OpenHCS ZMQ compile submission"
            )
            with phase_timing.phase(BenchmarkPhase.WAIT_OPENHCS):
                compile_wait_response = client.wait_for_completion(compile_artifact_id)
            compile_wait_result = ExecutionWaitResult.from_wire(compile_wait_response)
            compile_wait_result.require_complete("OpenHCS ZMQ compilation failed")

            execution_submission = OpenHCSExecutionSubmission(
                plate_id=plate_id,
                execution_plate_id=execution_plate_id,
                selected_pipeline_path=selected_pipeline_path,
                pipeline_document=PipelineDocumentAuthority.from_values(
                    pipeline_config=pipeline_config, pipeline_steps=transport_pipeline
                ),
                global_config=global_config,
                config_params={
                    "runtime_observation_export_path": str(observation_export_path),
                },
                compile_artifact_id=compile_artifact_id,
            )
            with phase_timing.phase(BenchmarkPhase.SUBMIT_OPENHCS):
                execution_submission_response = ExecutionSubmissionResponse.from_wire(
                    client.submit_pipeline(execution_submission)
                )
            if not execution_submission_response.accepted:
                raise ToolExecutionError(
                    execution_submission_response.require_failure_text(
                        "OpenHCS ZMQ execution submission"
                    )
                )
            execution_id = execution_submission_response.require_execution_id(
                "OpenHCS ZMQ execution submission"
            )
            with phase_timing.phase(BenchmarkPhase.WAIT_OPENHCS):
                wait_response = client.wait_for_completion(execution_id)
            completion_observed_at = time.time()
            wait_result = ExecutionWaitResult.from_wire(wait_response)
            wait_result.require_complete("OpenHCS ZMQ execution failed")
    finally:
        client.disconnect()

    timing_observer.record_phase_timings(
        phase_timing,
        completion_observed_at=completion_observed_at,
    )
    if not observation_export_path.exists():
        raise ToolExecutionError(
            "OpenHCS ZMQ execution completed without writing runtime observation "
            f"export: {observation_export_path}"
        )
    observation_export = ZMQRuntimeExecutionObservationExport.read(
        observation_export_path
    )
    output_roots = tuple(Path(root) for root in observation_export.output_roots)
    results_summary_payload = wait_response.get("results", {}) or wait_response.get(
        "results_summary",
        {},
    )
    if not isinstance(results_summary_payload, Mapping):
        results_summary_payload = {}
    results_summary = dict(results_summary_payload)
    results_summary_path = observation_export_path.with_name(
        ZMQ_RESULTS_SUMMARY_FILENAME
    )
    results_summary_path.write_text(
        json.dumps(results_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return (
        _ZMQOpenHCSExecution(
            execution_id=execution_id,
            observation_export=observation_export,
            output_roots=output_roots,
            results_summary=results_summary,
        ),
        pipeline_source,
    )


@dataclass(frozen=True, slots=True)
class OpenHCSRunRequest:
    """Authoritative benchmark run request for one OpenHCS execution."""

    dataset_path: Path
    pipeline_name: str
    pipeline_params: dict[str, Any]
    metrics: tuple[MetricCollector, ...]
    output_dir: Path

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


class OpenHCSAdapter(ToolAdapter):
    """OpenHCS tool adapter."""

    name = "OpenHCS"

    def __init__(
        self,
        *,
        global_config: GlobalPipelineConfig | None = None,
    ) -> None:
        import openhcs

        self.version = openhcs.__version__
        self.global_config = global_config or GlobalPipelineConfig()

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
            VFSConfig,
        )

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
        generated_source_path = request.output_dir / f"{cppipe_path.stem}_openhcs.py"
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

        equivalence_report = None
        equivalence_failure_message = None
        try:
            with phase_timing.phase(BenchmarkPhase.COMPILE_DIALECT):
                ingestion = prepare_cellprofiler_input_workspace(
                    InputWorkspacePreparationRequest(
                        selected_path=request.dataset_path,
                        selected_pipeline_path=cppipe_path,
                        workspace_root=source_workspace_path,
                        generated_source_path=generated_source_path,
                    )
                )
        except ValueError as exc:
            raise ToolExecutionError(
                "Failed to prepare CellProfiler source workspace for "
                f"{cppipe_path.name}: {exc}"
            ) from exc
        if ingestion.pipeline_import_error is not None:
            raise ToolExecutionError(ingestion.pipeline_import_error.message)
        pipeline_steps = ingestion.pipeline_steps
        pipeline_config = ingestion.pipeline_config
        if pipeline_steps is None or pipeline_config is None:
            raise ToolExecutionError(
                f"CellProfiler pipeline preparation produced no pipeline: {cppipe_path}"
            )
        execution_plate_path = ingestion.execution_plate_path
        generated_pipeline_source = generated_source_path.read_text(encoding="utf-8")
        canonical_pipeline_source = FunctionStepTransportAuthority.source_from_pipeline(
            pipeline_steps
        )
        if generated_pipeline_source != canonical_pipeline_source:
            raise ToolExecutionError(
                "CellProfiler benchmark pipeline source differs from the canonical "
                "UI/ZMQ FunctionStep source before execution."
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
        observation_export_path = (
            request.output_dir / "runtime_execution_server_observation.pkl"
        )
        with ExitStack() as stack:
            for metric in request.metrics:
                stack.enter_context(metric)
            with _openhcs_execution_watchdog(request.openhcs_timeout_seconds):
                server_execution, pipeline_source = _execute_pipeline_via_zmq_server(
                    plate_id=request.dataset_path,
                    execution_plate_id=execution_plate_path,
                    selected_pipeline_path=cppipe_path,
                    pipeline_steps=pipeline_steps,
                    global_config=global_config,
                    pipeline_config=pipeline_config,
                    observation_export_path=observation_export_path,
                    phase_timing=phase_timing,
                )
        submitted_pipeline_source_sha = hashlib.sha256(
            pipeline_source.encode("utf-8")
        ).hexdigest()[:12]
        output_roots = server_execution.output_roots
        execution_output_root = (
            server_execution.execution_output_root
            if server_execution.output_roots
            else request.output_dir
        )
        try:
            with phase_timing.phase(BenchmarkPhase.VALIDATE_RUNTIME):
                observation = (
                    server_execution.observation_export.require_valid_observation()
                )
        except RuntimeError as exc:
            raise ToolExecutionError(str(exc)) from exc
        axis_count = server_execution.axis_count
        executed_axes = tuple(observation.records_by_axis)
        csv_output_count = len(observation.exports.table_outputs)
        image_output_count = len(observation.exports.image_outputs)
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
                    observation,
                    policy=equivalence_policy,
                )
                database_export_report = cellprofiler_database_export_equivalence(
                    equivalence_reference,
                    observation.exports,
                    policy=equivalence_policy,
                )
                equivalence_report = RuntimeEquivalenceReport(
                    (
                        *equivalence_report.differences,
                        *database_export_report.differences,
                    )
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
            "generated_source_path": str(generated_source_path),
            "submitted_pipeline_source_sha": submitted_pipeline_source_sha,
            "axis_count": axis_count,
            "csv_output_count": csv_output_count,
            "image_output_count": image_output_count,
            "compiled_output_roots": tuple(str(root) for root in output_roots),
            "phase_timing_records": phase_timing.payloads(),
            "executed_axes": executed_axes,
            "zmq_results_summary_path": str(
                observation_export_path.with_name(ZMQ_RESULTS_SUMMARY_FILENAME)
            ),
        }
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
