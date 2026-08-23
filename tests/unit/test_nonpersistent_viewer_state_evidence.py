from types import SimpleNamespace

from zmqruntime.messages import (
    ExecutionRecord,
    ExecutionStatus,
    MessageFields,
    ResponseType,
)

from openhcs.core.execution_state import ExecutionOutputPlateSummary
from openhcs.core.orchestrator.compiled_plate_execution import (
    CompiledPlateExecutionExtras,
    CompiledPlateExecutionResults,
)
from openhcs.core.orchestrator.execution_result import ExecutionResult
from openhcs.runtime.viewer_protocol import ViewerControlResponse
from openhcs.runtime.zmq_server_hooks import ZMQResultsSummaryEnricher
from openhcs.runtime.zmq_worker_execution import ZMQWorkerExecutionRequest


def test_settled_viewer_state_survives_zmq_execution_result_transport(
    monkeypatch,
    tmp_path,
):
    response = ViewerControlResponse(
        payload={
            "status": "success",
            "layers": (
                {
                    "route_key": "step-2:objects",
                    "producer_identities": (
                        {
                            "step_name": "Identify objects",
                            "invocation_key": "identify",
                        },
                    ),
                    "data_types": ("roi",),
                    "item_count": 2,
                    "payload_summary_count": 2,
                    "component_values": ({"channel": "DAPI", "site": 0, "z_index": 0},),
                },
            ),
        }
    )
    compiled_results = CompiledPlateExecutionResults(
        {"A01": ExecutionResult.success("A01")},
        extras=CompiledPlateExecutionExtras(viewer_states_by_port={5563: response}),
    )

    class ProgressQueue:
        def __init__(self):
            self.values = []

        def put(self, value):
            self.values.append(value)

    progress_queue = ProgressQueue()
    multiprocessing_context = SimpleNamespace(Queue=lambda: progress_queue)
    worker_start = SimpleNamespace(
        requested=SimpleNamespace(value="spawn"),
        resolved=SimpleNamespace(value="spawn"),
        reason="focused transport test",
        multiprocessing_context=lambda: multiprocessing_context,
    )
    execution_bundle = SimpleNamespace(
        pipeline_definition=("step",),
        runtime_environment=SimpleNamespace(worker_start=worker_start),
    )

    class Orchestrator:
        def execute_compiled_plate(self, **_kwargs):
            return compiled_results

    record = ExecutionRecord(
        execution_id="exec",
        plate_id="plate",
        client_address=None,
        status=ExecutionStatus.RUNNING.value,
    )
    monkeypatch.setattr(
        "openhcs.runtime.zmq_worker_execution.Path.home",
        lambda: tmp_path,
    )

    returned = ZMQWorkerExecutionRequest(
        execution_id="exec",
        orchestrator=Orchestrator(),
        execution_bundle=execution_bundle,
        progress_context={"execution_id": "exec", "plate_id": "plate"},
        debug_execution_policy=object(),
        active_execution_record=record,
        forward_worker_progress=lambda _queue: None,
    ).execute()

    assert returned is compiled_results
    assert progress_queue.values == [None]
    assert record.get_extra(CompiledPlateExecutionExtras.EXECUTION_RECORD_KEY) is (
        compiled_results.extras
    )

    record.status = ExecutionStatus.COMPLETE.value
    record.results_summary = {"well_count": 1, "wells": ["A01"]}
    output_plate = ExecutionOutputPlateSummary(
        output_plate_root="/tmp/output",
        auto_add_output_plate_to_plate_manager=True,
    )
    record.set_extra(output_plate.EXECUTION_RECORD_KEY, output_plate)
    status_response = {
        MessageFields.STATUS: ResponseType.OK.value,
        MessageFields.EXECUTION: record.to_dict(),
    }
    enriched = ZMQResultsSummaryEnricher({"exec": record}).attach_to_status_response(
        execution_id="exec",
        response=status_response,
    )

    summary = enriched[MessageFields.EXECUTION][MessageFields.RESULTS_SUMMARY]
    assert summary["output_plate_root"] == "/tmp/output"
    assert summary["auto_add_output_plate_to_plate_manager"] is True
    states_by_port = summary[CompiledPlateExecutionExtras.RESULTS_SUMMARY_KEY]
    state_payload = states_by_port["5563"]["payload"]
    layer = state_payload["layers"][0]
    assert layer["route_key"] == "step-2:objects"
    assert layer["producer_identities"] == [
        {
            "step_name": "Identify objects",
            "invocation_key": "identify",
        }
    ]
    assert layer["data_types"] == ["roi"]
    assert layer["item_count"] == layer["payload_summary_count"] == 2
    assert layer["component_values"] == [{"channel": "DAPI", "site": 0, "z_index": 0}]
    assert record.to_dict()[MessageFields.RESULTS_SUMMARY] == summary
