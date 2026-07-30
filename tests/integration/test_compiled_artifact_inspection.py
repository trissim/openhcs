"""Compiler-to-control-to-Artifact-view integration."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from contextlib import contextmanager
import multiprocessing
import threading
import time
from types import SimpleNamespace

from openhcs.core.artifact_inspection import (
    CompiledArtifactInspectionControlPayload,
    CompiledArtifactInspectionRequest,
    CompiledArtifactInspectionResponse,
)
from openhcs.core.artifacts import ArtifactOutputPlan, ArtifactSpec, ImageArtifactType
from openhcs.core.callable_contract import CallableContract
from openhcs.core.compiled_execution import CompiledExecutionBundle
from openhcs.core.compiled_step_plan import CompiledStepPlan
from openhcs.core.component_group_scope import RuntimeExecutionAxisScope
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.function_patterns import (
    DEFAULT_GROUP_KEY,
    CompiledFunctionGroup,
    CompiledFunctionInvocation,
    CompiledFunctionPattern,
    FunctionInvocationKey,
)
from openhcs.core.pipeline.function_contracts import artifact_outputs
from openhcs.core.progress import (
    ProgressEvent,
    ProgressIdentity,
    ProgressPhase,
    ProgressStatus,
)
from openhcs.core.progress.runtime_artifacts import RuntimeArtifactProgressPayload
from openhcs.core.runtime_artifact_values import ArtifactKey
from openhcs.core.runtime_artifact_values import RuntimeValue
from openhcs.core.runtime_stores import (
    RuntimeArtifactAddress,
    RuntimeArtifactLocation,
    RuntimeValueStore,
)
from openhcs.pyqt_gui.widgets.artifact_plan_view import ArtifactPlanViewModel
from openhcs.pyqt_gui.config import ProgressUIConfig
from openhcs.pyqt_gui.widgets.shared.services.debug_progress_service import (
    DebugProgressNotificationService,
)
from openhcs.pyqt_gui.widgets.shared.services.execution_server_status_presenter import (
    ExecutionServerStatusPresenter,
)
from openhcs.pyqt_gui.widgets.shared.services.compile_workflow_service import (
    CompileWorkflowService,
    PlateCompiledState,
)
from openhcs.pyqt_gui.widgets.shared.services.progress_workflow_service import (
    ProgressWorkflowService,
)
from openhcs.pyqt_gui.widgets.shared.services.runtime_artifact_progress_service import (
    RuntimeArtifactAvailableNotification,
    RuntimeArtifactProgressNotificationService,
)
from openhcs.runtime.zmq_compilation import (
    ZMQCompilationResult,
    ZMQCompileArtifactRecord,
)
from openhcs.runtime.zmq_control import (
    ZMQControlMessageRouter,
    ZMQControlRequestContext,
)
from openhcs.runtime.zmq_execution_client import ZMQExecutionClient
from openhcs.runtime.zmq_execution_server import ZMQExecutionServer
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG
from zmqruntime import TcpDataControlPortPairAuthority
from zmqruntime.config import TransportMode


class ImmediateBlockingContext:
    async def run_blocking(self, loop, operation):
        del loop
        return operation()


class RecordingProgressTracker:
    def __init__(self) -> None:
        self.events: list[tuple[str, ProgressEvent]] = []

    def register_event(self, execution_id: str, event: ProgressEvent) -> None:
        self.events.append((execution_id, event))


@contextmanager
def _live_artifact_server(record: ZMQCompileArtifactRecord):
    port = TcpDataControlPortPairAuthority.acquire(
        OPENHCS_ZMQ_CONFIG,
    ).data_port
    server = ZMQExecutionServer(
        port=port,
        host="127.0.0.1",
        transport_mode=TransportMode.TCP,
    )
    server._compiled_artifacts[record.execution_id] = record

    def pump_server() -> None:
        while server.is_running():
            server.process_messages()
            time.sleep(0.01)

    server.start()
    server_thread = threading.Thread(target=pump_server, daemon=True)
    server_thread.start()
    client = ZMQExecutionClient(
        port=port,
        host="127.0.0.1",
        persistent=True,
        transport_mode=TransportMode.TCP,
    )
    try:
        assert client.connect(timeout=5.0)
        yield server, client
    finally:
        client.disconnect()
        server.stop()
        server_thread.join(timeout=2.0)


def _emit_spawned_runtime_observation(worker_queue) -> None:
    from openhcs.core.orchestrator.worker_execution import (
        _runtime_observation_progress_context,
    )

    store = RuntimeValueStore()
    cursor = store.observation_cursor()
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="ResultImage",
                artifact_type=ImageArtifactType,
                scope=RuntimeExecutionAxisScope.from_raw(
                    "A01",
                    component=None,
                    value=None,
                ),
            ),
            data=b"pixels",
        ),
        path="/memory/ResultImage.pkl",
        backend="memory",
    )
    context = _runtime_observation_progress_context(store.observed_values_after(cursor))
    worker_queue.put(
        ProgressEvent(
            identity=ProgressIdentity(
                execution_id="run-1",
                plate_id="/plates/one",
                axis_id="A01",
                step_name="Produce",
            ),
            phase=ProgressPhase.STEP_COMPLETED,
            status=ProgressStatus.SUCCESS,
            percent=100.0,
            completed=1,
            total=1,
            timestamp=time.time(),
            pid=0,
            worker_slot="worker_0",
            owned_wells=["A01"],
            context=context,
        ).to_dict()
    )


def _compiled_record() -> tuple[ZMQCompileArtifactRecord, ArtifactOutputPlan]:
    output_spec = ArtifactSpec.output("ResultImage", ImageArtifactType)

    @artifact_outputs(output_spec)
    def produce(image):
        return image

    output_plan = ArtifactOutputPlan(
        name=output_spec.name,
        path="/memory/ResultImage.pkl",
        artifact_type=output_spec.artifact_type,
        producer_step_index=0,
        producer_step_name="Produce",
    )
    invocation = CompiledFunctionInvocation(
        key=FunctionInvocationKey("produce", DEFAULT_GROUP_KEY, 0),
        contract=CallableContract.from_callable(produce),
        artifact_output_plans=(output_plan,),
    )
    pattern = CompiledFunctionPattern(
        groups=(
            CompiledFunctionGroup(
                group_key=DEFAULT_GROUP_KEY,
                invocations=(invocation,),
            ),
        ),
        is_grouped=False,
    )
    step_plan = CompiledStepPlan(
        step_index=0,
        step_name="Produce",
        step_type="FunctionStep",
        axis_id="A01",
        artifact_inputs=OrderedDict(),
        artifact_outputs=OrderedDict(((output_plan.ref(), output_plan),)),
        compiled_function_pattern=pattern,
    )
    context = ProcessingContext(step_plans={0: step_plan}, axis_id="A01")
    bundle = CompiledExecutionBundle(
        pipeline_definition=(),
        runtime_contexts={"A01": context},
        transport_contexts={"A01": context},
        worker_assignments={},
        runtime_environment=object(),
    )
    return (
        ZMQCompileArtifactRecord(
            execution_id="compile-1",
            plate_id="/plates/one",
            request_signature="request",
            debug_replay_signature="debug",
            compilation=ZMQCompilationResult(
                execution_bundle=bundle,
                compiled_axis_ids=["A01"],
            ),
        ),
        output_plan,
    )


def test_compiled_bundle_populates_static_view_then_runtime_enriches_same_row() -> None:
    record, output_plan = _compiled_record()
    context = ZMQControlRequestContext(compiled_artifacts={record.execution_id: record})

    class RoutedClient:
        def wait_for_completion(self, execution_id):
            return {"status": "complete", "execution_id": execution_id}

        def get_compiled_artifact_inspection(self, execution_id):
            request = CompiledArtifactInspectionRequest(execution_id)
            response = ZMQControlMessageRouter.handle(
                CompiledArtifactInspectionControlPayload.from_request(
                    request
                ).to_dict(),
                context,
            )
            return CompiledArtifactInspectionResponse.from_control_response(
                response
            ).inspection

    inspection = asyncio.run(
        CompileWorkflowService(
            context=ImmediateBlockingContext()
        ).wait_for_compile_completion(
            zmq_client=RoutedClient(),
            loop=object(),
            execution_id="compile-1",
            plate_path="/plates/one",
        )
    )
    state = PlateCompiledState(
        compile_artifact_id="compile-1",
        definition_pipeline=(),
        inspection=inspection,
    )
    static_model = ArtifactPlanViewModel.from_inspection(
        state.inspection,
        step_index=0,
    )

    address = RuntimeArtifactAddress(
        key=ArtifactKey(
            name=output_plan.name,
            artifact_type=output_plan.artifact_type,
            scope=RuntimeExecutionAxisScope.from_raw(
                "A01",
                component=None,
                value=None,
            ),
        ),
        location=RuntimeArtifactLocation(
            path=output_plan.path,
            backend="memory",
        ),
        value_type="ndarray",
    )
    event = ProgressEvent(
        identity=ProgressIdentity(
            execution_id="run-1",
            plate_id="/plates/one",
            axis_id="A01",
            step_name="Produce",
        ),
        phase=ProgressPhase.STEP_COMPLETED,
        status=ProgressStatus.SUCCESS,
        percent=100.0,
        completed=1,
        total=1,
        timestamp=1.0,
        pid=1,
        context=RuntimeArtifactProgressPayload((address,)).to_context(),
    )
    enriched = static_model.with_runtime_notification(
        RuntimeArtifactAvailableNotification(
            event=event,
            payload=RuntimeArtifactProgressPayload.from_context(event.context),
        )
    )

    assert len(static_model.rows) == 1
    assert static_model.rows[0].plan == output_plan
    assert static_model.rows[0].runtime_location == ""
    assert enriched.rows[0].runtime_location == "memory:/memory/ResultImage.pkl"
    assert enriched.rows[0].value_type == "ndarray"


def test_artifact_ui_inspection_crosses_live_zmq_from_server_compiled_record() -> None:
    record, output_plan = _compiled_record()

    with _live_artifact_server(record) as (_server, client):
        inspection = asyncio.run(
            CompileWorkflowService(
                context=ImmediateBlockingContext()
            ).inspect_compile_artifact(
                zmq_client=client,
                loop=object(),
                compile_artifact_id=record.execution_id,
            )
        )

    model = ArtifactPlanViewModel.from_inspection(inspection, step_index=0)

    assert inspection.compile_artifact_id == record.execution_id
    assert inspection.plate_id == record.plate_id
    assert len(model.rows) == 1
    assert model.rows[0].plan == output_plan
    assert model.rows[0].runtime_location == ""


def test_spawned_worker_runtime_observation_crosses_server_zmq_to_artifact_ui() -> None:
    record, _output_plan = _compiled_record()

    with _live_artifact_server(record) as (server, client):
        inspection = asyncio.run(
            CompileWorkflowService(
                context=ImmediateBlockingContext()
            ).inspect_compile_artifact(
                zmq_client=client,
                loop=object(),
                compile_artifact_id=record.execution_id,
            )
        )
        model = {
            "value": ArtifactPlanViewModel.from_inspection(inspection, step_index=0)
        }
        notifications: list[RuntimeArtifactAvailableNotification] = []
        available = threading.Event()
        runtime_artifacts = RuntimeArtifactProgressNotificationService()

        def apply_runtime_notification(
            notification: RuntimeArtifactAvailableNotification,
        ) -> None:
            notifications.append(notification)
            model["value"] = model["value"].with_runtime_notification(notification)
            available.set()

        runtime_artifacts.add_listener(apply_runtime_notification)
        tracker = RecordingProgressTracker()
        progress_service = ProgressWorkflowService(
            host=SimpleNamespace(_progress_tracker=tracker),
            context=SimpleNamespace(
                zmq=SimpleNamespace(current_client=client),
            ),
            debug_notifications=DebugProgressNotificationService(),
            status_presenter=ExecutionServerStatusPresenter(),
            config=ProgressUIConfig(),
            runtime_artifacts=runtime_artifacts,
            start_timer=False,
        )
        client.progress_callback = progress_service.on_progress
        client.enable_progress_stream()
        time.sleep(0.15)

        server._worker_assignments_by_execution["run-1"] = {"worker_0": ["A01"]}
        multiprocessing_context = multiprocessing.get_context("spawn")
        worker_queue = multiprocessing_context.Queue()
        progress_forwarder = threading.Thread(
            target=server._forward_worker_progress,
            args=(worker_queue,),
            daemon=True,
        )
        worker_process = multiprocessing_context.Process(
            target=_emit_spawned_runtime_observation,
            args=(worker_queue,),
        )
        progress_forwarder.start()
        worker_process.start()
        try:
            worker_process.join(timeout=20.0)
            assert not worker_process.is_alive()
            assert worker_process.exitcode == 0
        finally:
            if worker_process.is_alive():
                worker_process.terminate()
                worker_process.join(timeout=5.0)
            worker_queue.put(None)
            progress_forwarder.join(timeout=5.0)
            worker_queue.close()
            worker_queue.join_thread()
            server._worker_assignments_by_execution.pop("run-1", None)

        assert available.wait(timeout=5.0)

    assert len(notifications) == 1
    assert tracker.events == [("run-1", notifications[0].event)]
    assert notifications[0].event.worker_assignments == {"worker_0": ["A01"]}
    assert model["value"].rows[0].runtime_location == ("memory:/memory/ResultImage.pkl")
    assert model["value"].rows[0].value_type == "bytes"
