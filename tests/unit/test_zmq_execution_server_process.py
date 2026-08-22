"""Execution-server process launch contract tests."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from openhcs.runtime import zmq_execution_client
from openhcs.runtime import zmq_execution_server_launcher
from openhcs.runtime.zmq_execution_client import ZMQExecutionClient
from openhcs.runtime.zmq_config import OpenHCSZMQConfig
from zmqruntime import TransportMode
from zmqruntime.messages import PongResponse, ServerRole
from zmqruntime.startup import (
    EndpointStartupPhase,
    EndpointStartupObserver,
    EndpointStartupStatusWriter,
)


def test_execution_server_preserves_worker_interpreter_and_background_flags(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The server keeps worker-compatible Python plus background launch flags."""

    popen_call: dict[str, object] = {}
    process = object()

    def capture_popen(command, **kwargs):
        popen_call["command"] = command
        popen_call["stdout"] = kwargs["stdout"]
        popen_call.update(kwargs)
        return process

    class _LaunchPolicy:
        @classmethod
        def current(cls, *, detached=False):
            assert detached is False
            return SimpleNamespace(
                popen_arguments=lambda: {"creationflags": 73},
            )

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr("subprocess.Popen", capture_popen)
    monkeypatch.setattr(
        zmq_execution_client,
        "BackgroundProcessLaunchPolicy",
        _LaunchPolicy,
    )

    client = ZMQExecutionClient(port=22307, persistent=False)
    try:
        assert client._spawn_server_process() is process
    finally:
        popen_call["stdout"].close()

    command = popen_call["command"]
    assert command[:5] == [
        sys.executable,
        "-X",
        "faulthandler",
        "-m",
        "openhcs.runtime.zmq_execution_server_launcher",
    ]
    assert popen_call["creationflags"] == 73
    assert "start_new_session" not in popen_call


def test_execution_server_launcher_advertises_ready_after_start(monkeypatch) -> None:
    """The launcher does not claim readiness before capabilities and sockets start."""

    events: list[str] = []

    class _Server:
        def __init__(self, **_kwargs) -> None:
            events.append("construct")

        def start(self) -> None:
            events.append("start")

        def prepare_runtime_capabilities(self, status_callback) -> None:
            events.append("prepare_runtime")

    def serve_forever(server, **_kwargs) -> None:
        assert isinstance(server, _Server)
        events.append("serve")

    def log_info(message, *_args) -> None:
        if message == "Server ready - waiting for requests...":
            events.append("ready")

    monkeypatch.setattr(sys, "argv", ["openhcs-zmq-server"])
    monkeypatch.setattr(zmq_execution_server_launcher.logger, "info", log_info)

    zmq_execution_server_launcher.main(
        execution_server_type=_Server,
        server_runner=serve_forever,
    )

    assert events == ["construct", "prepare_runtime", "start", "ready", "serve"]


def test_execution_server_launcher_projects_endpoint_overrides_into_config(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    log_records: list[tuple[str, tuple[object, ...]]] = []

    class _Server:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        def prepare_runtime_capabilities(self, _status_callback) -> None:
            return None

        def start(self) -> None:
            return None

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "openhcs-zmq-server",
            "--port",
            "23000",
            "--transport-mode",
            "tcp",
        ],
    )
    monkeypatch.setattr(
        zmq_execution_server_launcher.logger,
        "info",
        lambda message, *args: log_records.append((message, args)),
    )

    zmq_execution_server_launcher.main(
        execution_server_type=_Server,
        server_runner=lambda *_args, **_kwargs: None,
    )

    config = captured["config"]
    assert isinstance(config, OpenHCSZMQConfig)
    assert config.default_port == 23000
    assert config.transport_mode is TransportMode.TCP
    assert ("Port: %s (control: %s)", (23000, 24000)) in log_records


def test_execution_server_launcher_prepares_capabilities_without_binding(
    monkeypatch,
) -> None:
    """Installer preparation uses the server owner without opening an endpoint."""

    events: list[str] = []

    class _Server:
        def __init__(self, **_kwargs) -> None:
            events.append("construct")

    monkeypatch.setattr(
        sys,
        "argv",
        ["openhcs-zmq-server", "--prepare-capabilities"],
    )
    zmq_execution_server_launcher.main(
        execution_server_type=_Server,
        server_runner=lambda *_args, **_kwargs: events.append("serve"),
        capability_preparer=lambda: events.append("prepare"),
    )

    assert events == ["prepare"]


def test_child_startup_events_are_resequenced_by_client_owner(
    monkeypatch,
    tmp_path: Path,
) -> None:
    startup_path = tmp_path / "startup.jsonl"
    writer = EndpointStartupStatusWriter(startup_path)
    writer.emit(EndpointStartupPhase.LOADING_CONFIG, "Loading config")
    writer.emit(EndpointStartupPhase.IMPORTING_RUNTIME, "Importing runtime")
    statuses = []
    client = ZMQExecutionClient(connection_status_callback=statuses.append)
    client._startup_status_path = startup_path
    process = SimpleNamespace(exit=lambda: None)
    endpoint = PongResponse(
        port=client.port,
        control_port=client.control_port,
        ready=True,
        server="ZMQExecutionServer",
        server_role=ServerRole.EXECUTION,
    )

    def wait_for_ready(*_args, **kwargs):
        observer = kwargs["startup_observer"]
        assert isinstance(observer, EndpointStartupObserver)
        assert observer.poll_activity() is True
        assert observer.poll_activity() is False
        assert observer.should_abort() is False
        return endpoint

    monkeypatch.setattr(
        zmq_execution_client,
        "wait_for_endpoint_ready",
        wait_for_ready,
    )
    assert client._wait_for_endpoint_ready(process) is endpoint
    assert [status.sequence for status in statuses] == [1, 2]
    assert [status.phase for status in statuses] == [
        EndpointStartupPhase.LOADING_CONFIG,
        EndpointStartupPhase.IMPORTING_RUNTIME,
    ]
    assert not startup_path.exists()
