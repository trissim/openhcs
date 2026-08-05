from queue import SimpleQueue

from objectstate import get_current_global_config

from openhcs.constants.constants import GroupBy
from openhcs.core.config import (
    GlobalPipelineConfig,
    PipelineConfig,
    ProcessingConfig,
)
from openhcs.core.progress import (
    ProgressEvent,
    ProgressEventPayload,
    ProgressExecutionContext,
    ProgressPhase,
    ProgressStatus,
    create_event,
)
from openhcs.runtime.zmq_execution_server import (
    ZMQExecutionContext,
    ZMQExecutionServer,
)
from openhcs.runtime.zmq_execution_signature import (
    OpenHCSExecutionConfigBundle,
    ZMQExecutionCompileControl,
    ZMQExecutionConfigTransport,
    ZMQExecutionIdentity,
    ZMQExecutionRequestPayload,
)


def test_zmq_execution_context_seeds_saved_global_config_for_compilation() -> None:
    global_config = GlobalPipelineConfig(
        processing_config=ProcessingConfig(group_by=GroupBy.CHANNEL)
    )
    configs = OpenHCSExecutionConfigBundle(
        global_pipeline=global_config,
        plate_pipeline=PipelineConfig(),
    )
    context = ZMQExecutionContext(
        execution_id="exec-1",
        request_payload=ZMQExecutionRequestPayload(
            identity=ZMQExecutionIdentity(plate_id="/tmp/plate"),
            pipeline_code=(
                "from openhcs.core.config import PipelineConfig\n"
                "pipeline_config = PipelineConfig()\n"
                "pipeline_steps = []\n"
            ),
            config_transport=ZMQExecutionConfigTransport(),
            compile_control=ZMQExecutionCompileControl(),
        ),
        pipeline_steps=[],
        configs=configs,
    )

    assert type(context.pipeline_steps) is list
    assert context.configs is configs
    assert "execution_pipeline" not in dir(context)
    assert "pipeline_steps_boundary" not in dir(context)
    assert "config_carrier" not in dir(context)

    ZMQExecutionServer._ensure_request_global_config_context(context)

    saved_global_config = get_current_global_config(
        GlobalPipelineConfig,
        use_live=False,
    )
    assert saved_global_config is global_config
    assert saved_global_config.processing_config.group_by is GroupBy.CHANNEL


def test_zmq_server_reconstructs_pipeline_and_configs_for_artifact_execution(
    monkeypatch,
) -> None:
    import openhcs.processing.func_registry as func_registry_module

    monkeypatch.setattr(func_registry_module, "_registry_initialized", True)
    monkeypatch.setattr(
        ZMQExecutionServer,
        "_cleanup_compiled_artifacts",
        lambda self: None,
    )
    monkeypatch.setattr(
        ZMQExecutionServer,
        "_execute_with_orchestrator",
        lambda self, context: context,
    )
    server = ZMQExecutionServer()
    request_payload = ZMQExecutionRequestPayload(
        identity=ZMQExecutionIdentity(plate_id="/tmp/plate"),
        pipeline_code=(
            "from openhcs.core.config import PipelineConfig\n"
            "pipeline_config = PipelineConfig()\n"
            "pipeline_steps = []\n"
        ),
        config_transport=ZMQExecutionConfigTransport(
            config_code=(
                "from openhcs.core.config import GlobalPipelineConfig\n"
                "config = GlobalPipelineConfig()\n"
            ),
        ),
        compile_control=ZMQExecutionCompileControl(compile_artifact_id="compile-1"),
    )

    context = server._execute_pipeline("exec-1", request_payload)

    assert type(context.pipeline_steps) is list
    assert context.pipeline_steps == []
    assert isinstance(context.configs, OpenHCSExecutionConfigBundle)
    assert isinstance(context.configs.global_pipeline, GlobalPipelineConfig)
    assert isinstance(context.pipeline_config, PipelineConfig)
    assert context.compile_artifact_id == "compile-1"


def test_zmq_server_forwards_parent_execution_progress_without_worker_claim() -> None:
    server = object.__new__(ZMQExecutionServer)
    server._worker_assignments_by_execution = {
        "execution-1": {"worker_0": ["A01", "B01"]}
    }
    server.progress_queue = SimpleQueue()
    worker_queue = SimpleQueue()
    progress_context = ProgressExecutionContext(
        execution_id="execution-1",
        plate_id="plate-1",
    )
    worker_queue.put(
        create_event(
            ProgressEventPayload(
                identity=progress_context.identity_for_event(
                    axis_id="",
                    step_name="ExportToDatabase",
                ),
                phase=ProgressPhase.RUNNING,
                status=ProgressStatus.RUNNING,
                completed=32,
                total=33,
                percent=(32 / 33) * 100.0,
            )
        ).to_dict()
    )
    worker_queue.put(None)

    server._forward_worker_progress(worker_queue)

    event = ProgressEvent.from_dict(server.progress_queue.get())
    assert event.axis_id == ""
    assert event.worker_slot is None
    assert event.owned_wells is None
    assert event.worker_assignments == {"worker_0": ["A01", "B01"]}
    assert event.total_wells == ["A01", "B01"]
