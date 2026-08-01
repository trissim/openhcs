from dataclasses import replace
from types import MethodType

from zmqruntime.messages import ControlMessageType, MessageFields

from openhcs.runtime.zmq_execution_client import ZMQExecutionClient
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG


class _Submission:
    def to_task(self):
        return object()


def test_submission_uses_control_timeout_for_progress_registration():
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
        ("progress", OPENHCS_ZMQ_CONFIG.control_timeout_ms),
        (ControlMessageType.EXECUTE.value, 15000),
    ]


def test_submission_uses_declared_client_connection_timeout() -> None:
    config = replace(
        OPENHCS_ZMQ_CONFIG,
        client_connect_timeout_seconds=3.25,
    )
    client = ZMQExecutionClient(config=config)
    observed: list[float] = []

    def connect(self, timeout: float):
        observed.append(timeout)
        return False

    client.connect = MethodType(connect, client)

    try:
        client._submit_submission(_Submission(), timeout_ms=15000)
    except RuntimeError as error:
        assert str(error) == "Failed to connect to execution server"
    else:
        raise AssertionError("Disconnected submission unexpectedly succeeded.")

    assert observed == [3.25]
