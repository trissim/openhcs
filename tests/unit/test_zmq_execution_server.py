from objectstate import get_current_global_config

from openhcs.constants.constants import GroupBy
from openhcs.core.config import (
    GlobalPipelineConfig,
    PipelineConfig,
    ProcessingConfig,
)
from openhcs.runtime.zmq_execution_server import (
    ZMQExecutionContext,
    ZMQExecutionServer,
    ZMQResolvedConfig,
)
from openhcs.runtime.zmq_execution_signature import (
    OpenHCSExecutionConfigBundle,
    OpenHCSExecutionConfigCarrier,
    ZMQExecutionCompileControl,
    ZMQExecutionConfigTransport,
    ZMQExecutionIdentity,
    ZMQExecutionRequestPayload,
)
from openhcs.runtime.zmq_pipeline_transport import PipelineStepsBoundary


def test_zmq_execution_context_seeds_saved_global_config_for_compilation() -> None:
    global_config = GlobalPipelineConfig(
        processing_config=ProcessingConfig(group_by=GroupBy.CHANNEL)
    )
    context = ZMQExecutionContext(
        execution_id="exec-1",
        request_payload=ZMQExecutionRequestPayload(
            identity=ZMQExecutionIdentity(plate_id="/tmp/plate"),
            pipeline_code="pipeline_steps = []",
            config_transport=ZMQExecutionConfigTransport(),
            compile_control=ZMQExecutionCompileControl(),
        ),
        execution_pipeline=PipelineStepsBoundary(()),
        config_carrier=ZMQResolvedConfig(
            configs=OpenHCSExecutionConfigBundle(
                global_pipeline=global_config,
                plate_pipeline=PipelineConfig(),
            )
        ),
    )

    assert isinstance(context, OpenHCSExecutionConfigCarrier)

    ZMQExecutionServer._ensure_request_global_config_context(context)

    saved_global_config = get_current_global_config(
        GlobalPipelineConfig,
        use_live=False,
    )
    assert saved_global_config is global_config
    assert saved_global_config.processing_config.group_by is GroupBy.CHANNEL
