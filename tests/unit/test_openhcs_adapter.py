from __future__ import annotations

import ast
import json
import signal
from pathlib import Path

import pytest

from benchmark.adapters.openhcs import (
    ZMQ_RESULTS_SUMMARY_FILENAME,
    _ZMQProgressTimingObserver,
    _execute_pipeline_via_zmq_server,
    _openhcs_execution_watchdog,
    _strict_cellprofiler_runtime_equivalence_policy,
)
from benchmark.contracts.tool_adapter import ToolExecutionError
from benchmark.timing import BenchmarkPhase, PhaseTimingTrace
from openhcs.core.config import GlobalPipelineConfig, PipelineConfig, WellFilterConfig
from openhcs.core.pipeline_document import PipelineDocumentAuthority
from openhcs.core.pipeline_document_fields import PipelineDocumentField
from openhcs.core.runtime_execution_validation import (
    RuntimeArtifactExecutionExpectation,
)
from openhcs.core.runtime_exports import (
    RuntimeExportExpectation,
    RuntimeExportObservation,
)
from openhcs.core.steps.function_step import FunctionStep
from openhcs.processing.backends import cellprofiler as cellprofiler_backend
from openhcs.processing.backends.cellprofiler.thresholding import (
    CellProfilerThresholdMethod,
)
from openhcs.ui.shared.plate_scope_identity import PlateScopeIdentity
from openhcs.pyqt_gui.widgets.shared.services.plate_pipeline_request_builder import (
    PlatePipelineRequest,
)
from openhcs.runtime.zmq_execution_client import (
    OpenHCSExecutionSubmission,
    ZMQExecutionRequestBuilder,
)
from openhcs.runtime.zmq_execution_observation import (
    ZMQ_RUNTIME_OBSERVATION_EXPORT_SCHEMA_VERSION,
    ZMQRuntimeExecutionObservationExport,
)


def _public_steps() -> list[FunctionStep]:
    return [
        FunctionStep(
            func=(
                cellprofiler_backend.identify_primary_objects,
                {"threshold_method": CellProfilerThresholdMethod.OTSU},
            ),
            name="IdentifyPrimaryObjects",
        )
    ]


def test_strict_cellprofiler_policy_has_no_tolerance_coarser_than_one_e_minus_six() -> (
    None
):
    policy = _strict_cellprofiler_runtime_equivalence_policy()

    assert (
        max(
            policy.numeric_abs_tolerance,
            policy.numeric_rel_tolerance,
            policy.threshold_entropy_abs_tolerance,
            policy.threshold_sensitive_pair_abs_tolerance,
            policy.threshold_sensitive_pair_rel_tolerance,
            policy.image_abs_tolerance,
            policy.image_rel_tolerance,
        )
        == 1e-6
    )
    assert policy.feature_numeric_tolerances == ()
    assert not policy.allow_tie_sensitive_location_mismatches
    assert not policy.allow_sparse_object_boundary_jitter
    assert not policy.allow_unstable_shape_descriptors
    assert not policy.allow_unstable_zernike_descriptors


def test_benchmark_executes_pipeline_via_zmq_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeZMQExecutionClient:
        submitted: list[OpenHCSExecutionSubmission] = []
        waits: list[str] = []

        def __init__(self, *, port, persistent, progress_callback):
            assert port is None
            self.progress_callback = progress_callback

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def disconnect(self):
            return None

        def submit_compile(self, submission):
            compile_submission = submission.compile_request()
            self.submitted.append(compile_submission)
            self.progress_callback(
                {"phase": "compile", "status": "started", "timestamp": 10.0}
            )
            self.progress_callback(
                {"phase": "compile", "status": "success", "timestamp": 12.0}
            )
            return {"status": "accepted", "execution_id": "compile-1"}

        def submit_pipeline(self, submission):
            self.submitted.append(submission)
            assert submission.compile_artifact_id == "compile-1"
            return {"status": "accepted", "execution_id": "exec-1"}

        def wait_for_completion(self, execution_id):
            self.waits.append(execution_id)
            if execution_id == "compile-1":
                return {
                    "status": "complete",
                    "execution_id": execution_id,
                    "results": {"well_count": 1, "wells": ["A01"]},
                }
            assert execution_id == "exec-1"
            self.progress_callback(
                {"phase": "axis_started", "status": "started", "timestamp": 13.0}
            )
            self.progress_callback(
                {"phase": "axis_completed", "status": "success", "timestamp": 17.0}
            )
            observation_path = Path(
                self.submitted[-1].config_params["runtime_observation_export_path"]
            )
            ZMQRuntimeExecutionObservationExport(
                schema_version=ZMQ_RUNTIME_OBSERVATION_EXPORT_SCHEMA_VERSION,
                expectation=RuntimeArtifactExecutionExpectation(
                    artifact_kinds=frozenset(),
                    exports=RuntimeExportExpectation(),
                ),
                records_by_axis={},
                exports=RuntimeExportObservation.from_output_roots((tmp_path,)),
                output_roots=(tmp_path,),
                execution_success_by_axis={"A01": True},
            ).write(observation_path)
            return {
                "status": "complete",
                "execution_id": execution_id,
                "results": {"output_plate_root": str(tmp_path)},
            }

    monkeypatch.setattr(
        "benchmark.adapters.openhcs.ZMQExecutionClient",
        FakeZMQExecutionClient,
    )
    timing = PhaseTimingTrace(run_id="run", pipeline_name="pipe", tool="OpenHCS")
    global_config = GlobalPipelineConfig(
        well_filter_config=WellFilterConfig(well_filter=1),
    )

    execution, source = _execute_pipeline_via_zmq_server(
        plate_id="/tmp/plate",
        execution_plate_id="/tmp/execution_plate",
        selected_pipeline_path="/tmp/pipeline.cppipe",
        pipeline_steps=_public_steps(),
        global_config=global_config,
        pipeline_config=PipelineConfig(),
        observation_export_path=tmp_path / "observation.pkl",
        phase_timing=timing,
        timing_observer=_ZMQProgressTimingObserver(),
    )

    assert execution.execution_id == "exec-1"
    assert execution.output_roots == (tmp_path,)
    assert execution.results_summary == {"output_plate_root": str(tmp_path)}
    assert (
        json.loads(
            (tmp_path / ZMQ_RESULTS_SUMMARY_FILENAME).read_text(encoding="utf-8")
        )
        == execution.results_summary
    )
    assert source == FakeZMQExecutionClient.submitted[0].pipeline_code()
    assert all(
        submission.global_pipeline_config is global_config
        for submission in FakeZMQExecutionClient.submitted
    )
    assert [
        submission.compile_only for submission in FakeZMQExecutionClient.submitted
    ] == [
        True,
        False,
    ]
    assert [
        submission.compile_artifact_id
        for submission in FakeZMQExecutionClient.submitted
    ] == [None, "compile-1"]
    assert FakeZMQExecutionClient.waits == ["compile-1", "exec-1"]
    assert {record["phase"] for record in timing.payloads()} >= {
        BenchmarkPhase.SUBMIT_OPENHCS.name,
        BenchmarkPhase.WAIT_OPENHCS.name,
        BenchmarkPhase.COMPILE_OPENHCS.name,
        BenchmarkPhase.EXECUTE_OPENHCS.name,
    }


def test_openhcs_progress_timing_uses_completion_bound_without_axis_events() -> None:
    observer = _ZMQProgressTimingObserver()
    observer({"phase": "compile", "status": "started", "timestamp": 100.0})
    observer({"phase": "compile", "status": "success", "timestamp": 102.0})
    timing = PhaseTimingTrace(run_id="run", pipeline_name="pipe", tool="OpenHCS")
    timing.record(BenchmarkPhase.WAIT_OPENHCS, seconds=7.0)

    observer.record_phase_timings(timing, completion_observed_at=106.5)

    phase_seconds = {record["phase"]: record["seconds"] for record in timing.payloads()}
    assert phase_seconds[BenchmarkPhase.COMPILE_OPENHCS.name] == 2.0
    assert phase_seconds[BenchmarkPhase.EXECUTE_OPENHCS.name] == 4.5


def test_openhcs_progress_observer_tracks_every_server_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("benchmark.adapters.openhcs.time.monotonic", lambda: 14.5)
    observer = _ZMQProgressTimingObserver(last_progress_monotonic=10.0)

    observer({"phase": "artifact_transfer", "status": "running"})

    assert observer.last_progress_monotonic == 14.5
    assert observer.inactivity_seconds(observed_at=17.0) == 2.5
    assert observer.progress_description() == "artifact_transfer/running"


def test_openhcs_watchdog_renews_after_recent_server_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observer = _ZMQProgressTimingObserver(last_progress_monotonic=95.0)
    handlers: list[object] = []
    timers: list[float] = []
    monkeypatch.setattr("benchmark.adapters.openhcs.time.monotonic", lambda: 100.0)
    monkeypatch.setattr("benchmark.adapters.openhcs.signal.getsignal", lambda _sig: None)
    monkeypatch.setattr(
        "benchmark.adapters.openhcs.signal.signal",
        lambda _sig, handler: handlers.append(handler),
    )
    monkeypatch.setattr(
        "benchmark.adapters.openhcs.signal.setitimer",
        lambda _which, seconds: timers.append(seconds),
    )

    with _openhcs_execution_watchdog(20.0, observer):
        handler = handlers[-1]
        assert callable(handler)
        handler(signal.SIGALRM, None)

    assert timers[:2] == [20.0, 15.0]
    assert timers[-1] == 0.0


def test_openhcs_watchdog_reports_progress_inactivity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observer = _ZMQProgressTimingObserver(
        last_progress_monotonic=70.0,
        last_progress_phase="axis_started",
        last_progress_status="running",
    )
    handlers: list[object] = []
    monkeypatch.setattr("benchmark.adapters.openhcs.time.monotonic", lambda: 100.0)
    monkeypatch.setattr("benchmark.adapters.openhcs.signal.getsignal", lambda _sig: None)
    monkeypatch.setattr(
        "benchmark.adapters.openhcs.signal.signal",
        lambda _sig, handler: handlers.append(handler),
    )
    monkeypatch.setattr(
        "benchmark.adapters.openhcs.signal.setitimer",
        lambda _which, _seconds: None,
    )

    with pytest.raises(
        ToolExecutionError,
        match=(
            "made no server progress for 30.0s.*"
            "last progress: axis_started/running"
        ),
    ):
        with _openhcs_execution_watchdog(20.0, observer):
            handler = handlers[-1]
            assert callable(handler)
            handler(signal.SIGALRM, None)


def test_benchmark_submission_matches_pyqt_submission_payload() -> None:
    steps = _public_steps()
    global_config = GlobalPipelineConfig(
        well_filter_config=WellFilterConfig(well_filter=1),
    )
    pipeline_config = PipelineConfig()
    plate_id = "/tmp/plate#openhcs-cppipe=pipeline.cppipe"
    execution_plate_id = "/tmp/plate/.openhcs/source_bindings/pipeline"
    selected_pipeline_path = "/tmp/plate/pipeline.cppipe"

    benchmark_submission = OpenHCSExecutionSubmission(
        plate_id=plate_id,
        pipeline_document=PipelineDocumentAuthority.from_values(
            pipeline_config=pipeline_config, pipeline_steps=steps
        ),
        global_config=global_config,
        execution_plate_id=execution_plate_id,
        selected_pipeline_path=selected_pipeline_path,
        config_params={
            "runtime_observation_export_path": "/tmp/runtime_observation.pkl",
        },
    )
    ui_request = PlatePipelineRequest(
        plate_scope=PlateScopeIdentity.from_scope_id(plate_id),
        execution_plate_path=execution_plate_id,
        selected_pipeline_path=selected_pipeline_path,
        definition_pipeline=steps,
        pipeline_config=pipeline_config,
    )
    ui_submission = ui_request.submission(global_config=global_config)

    assert (
        benchmark_submission.global_pipeline_config
        == ui_submission.global_pipeline_config
    )
    assert benchmark_submission.pipeline_config == ui_submission.pipeline_config
    assert benchmark_submission.pipeline_code() == ui_submission.pipeline_code()
    benchmark_request = ZMQExecutionRequestBuilder.from_task(benchmark_submission)
    ui_request_builder = ZMQExecutionRequestBuilder.from_task(ui_submission)
    assert (
        benchmark_request.config_projection.source_fields
        == ui_request_builder.config_projection.source_fields
    )

    source = benchmark_submission.pipeline_code()
    module = ast.parse(source)
    assigned_names = {
        target.id
        for node in module.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert PipelineDocumentField.PIPELINE_STEPS.value in assigned_names
    assert "__openhcs_step_invocation_contracts" not in assigned_names
