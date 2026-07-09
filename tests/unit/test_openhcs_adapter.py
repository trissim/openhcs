from __future__ import annotations

import ast
from pathlib import Path

from benchmark.adapters.openhcs import (
    OpenHCSAdapter,
    OpenHCSRunRequest,
    RuntimeExecutionCacheWritePolicy,
    _ZMQProgressTimingObserver,
    _execute_pipeline_via_zmq_server,
    _strict_cellprofiler_runtime_equivalence_policy,
)
from benchmark.timing import BenchmarkPhase, PhaseTimingTrace
import openhcs.processing.backends.cellprofiler as cellprofiler_backend
from openhcs.core.artifacts import (
    ArtifactSpec,
    ImageArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.core.config import GlobalPipelineConfig, PipelineConfig
from openhcs.core.function_step_invocation_contracts import (
    FunctionStepInvocationContractBinding,
    FunctionStepInvocationContracts,
)
from openhcs.core.function_patterns import FunctionInvocationKey
from openhcs.core.module_artifact_contract import (
    DeclaredArtifactOutputPartition,
    ModuleArtifactContract,
    RecordedArtifactOutputPartition,
    SourceArtifactInputPartition,
)
from openhcs.core.pipeline import Pipeline
from openhcs.core.runtime_exports import RuntimeExportObservation
from openhcs.core.source_schema_workspace import SourceSchemaImageSetSelection
from openhcs.core.steps.function_step import FunctionStep
from openhcs.pyqt_gui.services.plate_scope_identity import PlateScopeIdentity
from openhcs.pyqt_gui.widgets.shared.services.plate_pipeline_request_builder import (
    PlatePipelineRequest,
)
from openhcs.runtime.zmq_execution_client import (
    OpenHCSExecutionSubmission,
    PycodifiedPipelineCode,
)
from openhcs.runtime.zmq_execution_observation import (
    ZMQRuntimeExecutionObservationExport,
)


def test_openhcs_adapter_stores_injected_product_config_and_source_selection() -> None:
    global_config = GlobalPipelineConfig(num_workers=2, use_threading=True)
    selection = SourceSchemaImageSetSelection(
        well_filter=("A01",),
        max_image_set_count=1,
    )
    adapter = OpenHCSAdapter(
        global_config=global_config,
        source_schema_image_set_selection=selection,
    )

    assert adapter.global_config is global_config
    assert adapter.source_schema_image_set_selection is selection


def test_openhcs_run_request_carries_source_schema_selection() -> None:
    selection = SourceSchemaImageSetSelection(
        well_filter=("A01",),
        max_image_set_count=1,
    )
    request = OpenHCSRunRequest(
        dataset_path=Path("/tmp/dataset"),
        pipeline_name="pipeline",
        pipeline_params={},
        metrics=(),
        output_dir=Path("/tmp/out"),
        source_schema_image_set_selection=selection,
    )

    assert request.source_schema_image_set_selection is selection


def test_strict_cellprofiler_policy_keeps_threshold_entropy_tolerance() -> None:
    policy = _strict_cellprofiler_runtime_equivalence_policy()

    assert policy.numeric_abs_tolerance == 1e-6
    assert policy.threshold_entropy_abs_tolerance == 0.04


def test_runtime_execution_cache_policy_disables_without_manifest() -> None:
    request = OpenHCSRunRequest(
        dataset_path=Path("/tmp/dataset"),
        pipeline_name="pipeline",
        pipeline_params={"runtime_execution_cache_key": {"case": "x"}},
        metrics=(),
        output_dir=Path("/tmp/out"),
    )

    policy = RuntimeExecutionCacheWritePolicy.for_request(request)

    assert not policy.write_manifest


def test_runtime_execution_cache_policy_writes_single_validation_payload() -> None:
    request = OpenHCSRunRequest(
        dataset_path=Path("/tmp/dataset"),
        pipeline_name="pipeline",
        pipeline_params={
            "runtime_execution_cache_manifest": "/tmp/out/cache.json",
            "runtime_execution_cache_key": {"case": "x"},
            "compare_image_outputs": False,
        },
        metrics=(),
        output_dir=Path("/tmp/out"),
    )

    policy = RuntimeExecutionCacheWritePolicy.for_request(request)

    assert policy.write_manifest


def test_benchmark_executes_pipeline_via_zmq_client(monkeypatch, tmp_path) -> None:
    class FakeZMQExecutionClient:
        submitted = []
        waits = []

        def __init__(self, *, persistent, progress_callback):
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
                schema_version=1,
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
    pipeline = Pipeline(
        steps=[
            FunctionStep(
                func=cellprofiler_backend.identify_primary_objects,
                name="IdentifyPrimaryObjects",
            )
        ],
        name="server",
    )
    timing = PhaseTimingTrace(run_id="run", pipeline_name="pipe", tool="OpenHCS")

    execution, source = _execute_pipeline_via_zmq_server(
        plate_id="/tmp/plate",
        execution_plate_id="/tmp/execution_plate",
        selected_pipeline_path="/tmp/pipeline.cppipe",
        pipeline_steps=pipeline.steps,
        global_config=GlobalPipelineConfig(),
        pipeline_config=PipelineConfig(),
        observation_export_path=tmp_path / "observation.pkl",
        phase_timing=timing,
    )

    assert execution.execution_id == "exec-1"
    assert execution.output_roots == (tmp_path,)
    assert "pipeline_steps" in source
    assert [submission.compile_only for submission in FakeZMQExecutionClient.submitted] == [
        True,
        False,
    ]
    assert [
        submission.compile_artifact_id
        for submission in FakeZMQExecutionClient.submitted
    ] == [None, "compile-1"]
    assert FakeZMQExecutionClient.waits == ["compile-1", "exec-1"]
    assert {
        record["phase"] for record in timing.payloads()
    } >= {
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

    phase_seconds = {
        record["phase"]: record["seconds"] for record in timing.payloads()
    }
    assert phase_seconds[BenchmarkPhase.COMPILE_OPENHCS.name] == 2.0
    assert phase_seconds[BenchmarkPhase.EXECUTE_OPENHCS.name] == 4.5


def test_benchmark_transport_matches_pyqt_submission_source() -> None:
    contract = ModuleArtifactContract(
        module_name="IdentifyPrimaryObjects",
        items=(
            *ModuleArtifactContract.items_for_partition(
                SourceArtifactInputPartition,
                (ArtifactSpec.input("OrigBlue", ImageArtifactType),),
            ),
            *ModuleArtifactContract.items_for_partition(
                RecordedArtifactOutputPartition,
                (ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType),),
            ),
            *ModuleArtifactContract.items_for_partition(
                DeclaredArtifactOutputPartition,
                (ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType),),
            ),
        ),
    )
    key = FunctionInvocationKey.from_callable(
        cellprofiler_backend.identify_primary_objects,
        "default",
        0,
    )
    step = FunctionStep(
        func=cellprofiler_backend.identify_primary_objects,
        name="IdentifyPrimaryObjects",
        invocation_contracts=FunctionStepInvocationContracts(
            (FunctionStepInvocationContractBinding(key, contract),)
        ),
    )
    pipeline = Pipeline(steps=[step], name="ui-equivalent")
    global_config = GlobalPipelineConfig()
    pipeline_config = PipelineConfig()
    plate_id = "/tmp/plate#openhcs-cppipe=pipeline.cppipe"
    execution_plate_id = "/tmp/plate/.openhcs/source_schema/pipeline"
    selected_pipeline_path = "/tmp/plate/pipeline.cppipe"

    benchmark_submission = OpenHCSExecutionSubmission(
        plate_id=plate_id,
        pipeline_steps=list(pipeline.steps),
        global_config=global_config,
        execution_plate_id=execution_plate_id,
        selected_pipeline_path=selected_pipeline_path,
        pipeline_config=pipeline_config,
        config_params={
            "runtime_observation_export_path": "/tmp/runtime_observation.pkl",
        },
    )
    benchmark_source = PycodifiedPipelineCode.from_task(benchmark_submission).source
    ui_request = PlatePipelineRequest(
        plate_scope=PlateScopeIdentity.from_scope_id(plate_id),
        execution_plate_path=execution_plate_id,
        selected_pipeline_path=selected_pipeline_path,
        definition_pipeline=list(pipeline.steps),
        pipeline_config=pipeline_config,
    )
    ui_source = PycodifiedPipelineCode.from_task(
        ui_request.submission(global_config=global_config)
    ).source

    assert benchmark_source == ui_source
    module = ast.parse(benchmark_source)
    assigned_names = {
        target.id
        for node in module.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    imported_names = {
        alias.name
        for node in module.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "pipeline_steps" in assigned_names
    assert "__openhcs_step_invocation_contracts" not in assigned_names
    assert "ModuleArtifactContract" not in imported_names
    assert "FunctionStepInvocationContracts" not in imported_names
