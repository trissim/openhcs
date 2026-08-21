from __future__ import annotations

import time

from openhcs.core.compiled_execution import (
    CompiledExecutionBundle,
    CompiledRuntimeEnvironmentPlan,
    CompiledWorkerStartPlan,
)
from openhcs.core.compiled_step_plan import CompiledStepPlan
from openhcs.core.config import MultiprocessingStartMethod
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.runtime.zmq_compilation import (
    ZMQCompilationRequest,
    ZMQCompilationResult,
    ZMQCompileArtifactRecord,
)


class _StrippedStepShell:
    @property
    def name(self) -> str:
        raise AssertionError("compile artifact reuse must use compiled step plans")


class _ProgressEmitter:
    def __init__(self) -> None:
        self.artifact_init_events: list[dict] = []
        self.compiled_init_events: list[dict] = []
        self.compile_success_events: list[dict] = []
        self.axis_compile_success_events: list[str] = []
        self.compile_heartbeat_events: list[int] = []

    def artifact_init_started(
        self,
        *,
        compiled_axis_ids: list[str],
        worker_assignments: dict[str, list[str]],
        step_names: list[str],
    ) -> None:
        self.artifact_init_events.append(
            {
                "compiled_axis_ids": compiled_axis_ids,
                "worker_assignments": worker_assignments,
                "step_names": step_names,
            }
        )

    def compiled_init_started(
        self,
        *,
        compiled_axis_ids: list[str],
        worker_assignments: dict[str, list[str]],
    ) -> None:
        self.compiled_init_events.append(
            {
                "compiled_axis_ids": compiled_axis_ids,
                "worker_assignments": worker_assignments,
            }
        )

    def compile_succeeded(
        self,
        *,
        step_count: int,
        compiled_axis_ids: list[str],
        worker_assignments: dict[str, list[str]],
    ) -> None:
        self.compile_success_events.append(
            {
                "step_count": step_count,
                "compiled_axis_ids": compiled_axis_ids,
                "worker_assignments": worker_assignments,
            }
        )

    def axis_compile_succeeded(self, axis_id: str) -> None:
        self.axis_compile_success_events.append(axis_id)

    def compile_heartbeat(self, step_count: int) -> None:
        self.compile_heartbeat_events.append(step_count)


class _ProgressQueue:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def put(self, event: dict) -> None:
        self.events.append(event)


class _FreshCompileOrchestrator:
    def __init__(self, bundle: CompiledExecutionBundle) -> None:
        self.bundle = bundle
        self.calls: list[dict] = []

    def compile_pipelines(
        self,
        *,
        pipeline_definition,
        well_filter,
        is_zmq_execution,
        debug_execution_policy,
    ) -> dict:
        self.calls.append(
            {
                "pipeline_definition": pipeline_definition,
                "well_filter": well_filter,
                "is_zmq_execution": is_zmq_execution,
                "debug_execution_policy": debug_execution_policy,
            }
        )
        time.sleep(0.03)
        return {"execution_bundle": self.bundle}


def _runtime_environment() -> CompiledRuntimeEnvironmentPlan:
    return CompiledRuntimeEnvironmentPlan(
        worker_start=CompiledWorkerStartPlan(
            requested=MultiprocessingStartMethod.SPAWN,
            resolved=MultiprocessingStartMethod.SPAWN,
            reason="test",
            gpu_enabled=False,
            server_mode=False,
        ),
        use_threading=True,
        configured_num_workers=1,
    )


def _compiled_context(axis_id: str) -> ProcessingContext:
    return ProcessingContext(
        step_plans={
            1: CompiledStepPlan(
                step_index=1,
                step_name="measure",
                step_type="FunctionStep",
                axis_id=axis_id,
            ),
            0: CompiledStepPlan(
                step_index=0,
                step_name="segment",
                step_type="FunctionStep",
                axis_id=axis_id,
            ),
        }
    )


def test_reused_compile_artifact_reads_step_names_from_compiled_plans() -> None:
    progress_emitter = _ProgressEmitter()
    bundle = CompiledExecutionBundle(
        pipeline_definition=(_StrippedStepShell(),),
        runtime_contexts={"A01": _compiled_context("A01")},
        transport_contexts={},
        worker_assignments={"worker_0": ["A01"]},
        runtime_environment=_runtime_environment(),
    )
    compilation = ZMQCompilationResult(
        execution_bundle=bundle,
        compiled_axis_ids=["A01"],
    )
    artifacts = {
        "artifact-1": ZMQCompileArtifactRecord(
            execution_id="compile-1",
            plate_id="/tmp/plate",
            request_signature="signature",
            debug_replay_signature="debug-signature",
            compilation=compilation,
        )
    }

    request = ZMQCompilationRequest(
        execution_id="exec-1",
        plate_id="/tmp/plate",
        pipeline_steps=(),
        orchestrator=None,
        wells=["A01"],
        compile_artifact_id="artifact-1",
        request_signature="signature",
        debug_replay_signature="debug-signature",
        retain_compile_artifact=False,
        compiled_artifacts=artifacts,
        progress_emitter=progress_emitter,
        compiler_progress_queue=None,
        debug_execution_policy=None,
    )

    result = request.reuse_artifact()

    assert result.execution_bundle is bundle
    assert progress_emitter.artifact_init_events == [
        {
            "compiled_axis_ids": ["A01"],
            "worker_assignments": {"worker_0": ["A01"]},
            "step_names": ["segment", "measure"],
        }
    ]


def test_compile_fresh_emits_heartbeat_during_long_compilation() -> None:
    progress_emitter = _ProgressEmitter()
    bundle = CompiledExecutionBundle(
        pipeline_definition=(),
        runtime_contexts={"A01": _compiled_context("A01")},
        transport_contexts={},
        worker_assignments={"worker_0": ["A01"]},
        runtime_environment=_runtime_environment(),
    )
    orchestrator = _FreshCompileOrchestrator(bundle)

    request = ZMQCompilationRequest(
        execution_id="exec-1",
        plate_id="/tmp/plate",
        pipeline_steps=(_StrippedStepShell(), _StrippedStepShell()),
        orchestrator=orchestrator,
        wells=["A01"],
        compile_artifact_id=None,
        request_signature="signature",
        debug_replay_signature="debug-signature",
        retain_compile_artifact=False,
        compiled_artifacts={},
        progress_emitter=progress_emitter,
        compiler_progress_queue=_ProgressQueue(),
        debug_execution_policy="debug-policy",
        compile_heartbeat_interval_seconds=0.001,
    )

    result = request.compile_fresh()

    assert result.execution_bundle is bundle
    assert progress_emitter.compile_heartbeat_events
    assert set(progress_emitter.compile_heartbeat_events) == {2}
    assert orchestrator.calls == [
        {
            "pipeline_definition": request.pipeline_steps,
            "well_filter": ["A01"],
            "is_zmq_execution": True,
            "debug_execution_policy": "debug-policy",
        }
    ]
    assert progress_emitter.compiled_init_events == [
        {
            "compiled_axis_ids": ["A01"],
            "worker_assignments": {"worker_0": ["A01"]},
        }
    ]
    assert progress_emitter.compile_success_events == [
        {
            "step_count": 2,
            "compiled_axis_ids": ["A01"],
            "worker_assignments": {"worker_0": ["A01"]},
        }
    ]
    assert progress_emitter.axis_compile_success_events == ["A01"]
