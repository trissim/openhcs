from openhcs.core.progress import ProgressPhase, ProgressStatus
from openhcs.runtime.zmq_progress import ImmediateZMQProgressQueue, ZMQProgressEmitter


def test_immediate_progress_queue_uses_request_plate_identity():
    emitted: list[dict] = []
    flushes: list[None] = []
    queue = ImmediateZMQProgressQueue(
        enqueue=emitted.append,
        flush=lambda: flushes.append(None),
        plate_id="/tmp/plate#openhcs-cppipe=Analysis.cppipe",
    )

    compiler_update = {
        "execution_id": "exec-1",
        "plate_id": "/tmp/plate",
        "axis_id": "A01",
        "step_name": "compilation",
        "phase": ProgressPhase.COMPILE.value,
        "status": ProgressStatus.RUNNING.value,
        "percent": 50.0,
        "completed": 1,
        "total": 2,
    }

    queue.put(compiler_update)

    assert emitted[0]["plate_id"] == "/tmp/plate#openhcs-cppipe=Analysis.cppipe"
    assert compiler_update["plate_id"] == "/tmp/plate"
    assert flushes == [None]


def test_compile_failed_emits_failed_compile_events_for_axes():
    emitted: list[dict] = []
    emitter = ZMQProgressEmitter(
        enqueue=emitted.append,
        execution_id="exec-1",
        plate_id="/tmp/plate",
    )

    emitter.compile_failed(axis_ids=["B02", "A01"], error="compile exploded")

    assert [event["axis_id"] for event in emitted] == ["A01", "B02"]
    assert {event["phase"] for event in emitted} == {ProgressPhase.COMPILE.value}
    assert {event["status"] for event in emitted} == {ProgressStatus.FAILED.value}
    assert {event["error"] for event in emitted} == {"compile exploded"}


def test_compile_failed_emits_pipeline_failure_when_axes_are_unknown():
    emitted: list[dict] = []
    emitter = ZMQProgressEmitter(
        enqueue=emitted.append,
        execution_id="exec-1",
        plate_id="/tmp/plate",
    )

    emitter.compile_failed(axis_ids=[], error="early compile exploded")

    assert [event["axis_id"] for event in emitted] == ["pipeline"]
    assert emitted[0]["phase"] == ProgressPhase.COMPILE.value
    assert emitted[0]["status"] == ProgressStatus.FAILED.value
    assert emitted[0]["error"] == "early compile exploded"
