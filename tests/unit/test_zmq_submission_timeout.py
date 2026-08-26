import time
from dataclasses import replace
from types import MethodType

import pytest
from zmqruntime.client import AttachedEndpointConnection
from zmqruntime.execution import ExecutionClient
from zmqruntime.messages import (
    ControlMessageType,
    MessageFields,
    PongResponse,
    ServerRole,
)
from zmqruntime.timeouts import OperationDeadline

from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG
from openhcs.runtime.zmq_execution_client import (
    ExecutionSubmissionPreparationTimeoutError,
    ZMQExecutionClient,
)


class _Submission:
    def to_task(self):
        return object()


def test_submission_uses_declared_timeout_for_progress_registration():
    client = ZMQExecutionClient()
    client._connection = AttachedEndpointConnection(
        PongResponse(
            port=client.port,
            control_port=client.control_port,
            ready=True,
            server="ZMQExecutionServer",
            server_role=ServerRole.EXECUTION,
        )
    )
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
    client._send_control_request = MethodType(
        send_control_request,
        client,
    )

    client._submit_submission(_Submission(), timeout_ms=15000)

    assert [phase for phase, _timeout_ms in observed] == [
        "progress",
        ControlMessageType.EXECUTE.value,
    ]
    assert 0 < observed[1][1] <= observed[0][1] <= 15000


def test_submission_default_is_owned_separately_from_short_control_requests():
    config = replace(
        OPENHCS_ZMQ_CONFIG,
        control_timeout_ms=17,
        execution_submission_timeout_ms=23000,
    )
    client = ZMQExecutionClient(config=config)
    observed: list[int] = []

    def submit_submission(self, submission, *, timeout_ms: int):
        del submission
        observed.append(timeout_ms)
        return {MessageFields.STATUS: "accepted"}

    client._submit_submission = MethodType(submit_submission, client)

    client.submit_pipeline(_Submission())

    assert observed == [23000]


def test_submission_uses_declared_client_connection_timeout() -> None:
    config = replace(
        OPENHCS_ZMQ_CONFIG,
        client_connect_timeout_seconds=3.25,
    )
    client = ZMQExecutionClient(config=config)
    observed: list[tuple[float, OperationDeadline]] = []

    def connect(
        self,
        timeout: float,
        *,
        operation_deadline: OperationDeadline,
    ):
        observed.append((timeout, operation_deadline))
        return False

    client.connect = MethodType(connect, client)

    try:
        client._submit_submission(_Submission(), timeout_ms=15000)
    except RuntimeError as error:
        assert str(error) == "Failed to connect to execution server"
    else:
        raise AssertionError("Disconnected submission unexpectedly succeeded.")

    assert len(observed) == 1
    assert observed[0][0] == 3.25
    assert observed[0][1].operation == "execution submission"


def test_submission_deadline_prevents_request_after_slow_connection() -> None:
    client = ZMQExecutionClient()
    observed: list[str] = []

    def connect(
        self,
        timeout: float,
        *,
        operation_deadline: OperationDeadline,
    ) -> bool:
        del timeout, operation_deadline
        observed.append("connect")
        time.sleep(0.03)
        return True

    def ensure_progress_subscription(self, *, timeout_ms: int) -> None:
        del timeout_ms
        observed.append("progress")

    def send_control_request(self, request, *, timeout_ms: int):
        del request, timeout_ms
        observed.append("submit")
        return {MessageFields.STATUS: "accepted"}

    client.connect = MethodType(connect, client)
    client._ensure_progress_subscription = MethodType(
        ensure_progress_subscription,
        client,
    )
    client._send_control_request = MethodType(send_control_request, client)

    with pytest.raises(
        ExecutionSubmissionPreparationTimeoutError,
        match="no execute request was sent",
    ):
        client._submit_submission(_Submission(), timeout_ms=10)

    assert observed == ["connect"]


def test_direct_connection_uses_declared_client_connection_timeout(monkeypatch) -> None:
    config = replace(
        OPENHCS_ZMQ_CONFIG,
        client_connect_timeout_seconds=7.5,
    )
    client = ZMQExecutionClient(config=config)
    observed = []
    monkeypatch.setattr(
        ExecutionClient,
        "connect",
        lambda self, timeout, *, operation_deadline=None: (
            observed.append((timeout, operation_deadline)) or True
        ),
    )

    assert client.connect() is True
    assert observed == [(7.5, None)]
