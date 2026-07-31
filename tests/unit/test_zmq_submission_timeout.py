from types import MethodType

from zmqruntime.messages import ControlMessageType, MessageFields

from openhcs.runtime.zmq_execution_client import ZMQExecutionClient


class _Submission:
    def to_task(self):
        return object()


def test_submission_timeout_covers_progress_registration_and_execution_request():
    client = ZMQExecutionClient()
    client._connected = True
    observed: list[tuple[str, int]] = []

    def ensure_progress_subscription(self, *, timeout_ms: int):
        observed.append(("progress", timeout_ms))

    def serialize_task(self, _task, _config=None):
        return {}

    def send_control_request(self, request, *, timeout_ms: int):
        observed.append((request[MessageFields.TYPE], timeout_ms))
        return {MessageFields.STATUS: "accepted"}

    client._ensure_progress_subscription = MethodType(
        ensure_progress_subscription,
        client,
    )
    client.serialize_task = MethodType(serialize_task, client)
    client._send_control_request_bounded = MethodType(
        send_control_request,
        client,
    )

    client._submit_submission(_Submission(), timeout_ms=15000)

    assert observed == [
        ("progress", 15000),
        (ControlMessageType.EXECUTE.value, 15000),
    ]
