from __future__ import annotations

from openhcs.core.compiled_execution import (
    CompiledExecutionBundle,
    CompiledGpuRegistryPlan,
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
        gpu_registry=CompiledGpuRegistryPlan(configured_num_workers=1),
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
        flush_progress=lambda: None,
        immediate_progress_queue=None,
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
