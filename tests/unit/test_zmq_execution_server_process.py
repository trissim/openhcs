"""Execution-server process launch contract tests."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from openhcs.runtime import zmq_execution_client
from openhcs.runtime import zmq_execution_server_launcher
from openhcs.runtime.zmq_execution_client import ZMQExecutionClient


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

    def serve_forever(server, **_kwargs) -> None:
        assert isinstance(server, _Server)
        events.append("serve")

    def log_info(message, *_args) -> None:
        if message == "Server ready - waiting for requests...":
            events.append("ready")

    monkeypatch.setattr(sys, "argv", ["openhcs-zmq-server"])
    monkeypatch.setattr(zmq_execution_server_launcher, "ZMQExecutionServer", _Server)
    monkeypatch.setattr(zmq_execution_server_launcher, "serve_forever", serve_forever)
    monkeypatch.setattr(zmq_execution_server_launcher.logger, "info", log_info)

    zmq_execution_server_launcher.main()

    assert events == ["construct", "start", "ready", "serve"]
