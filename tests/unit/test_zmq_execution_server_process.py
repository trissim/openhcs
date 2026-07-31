"""Execution-server process launch contract tests."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from openhcs.runtime import zmq_execution_client
from openhcs.runtime.zmq_execution_client import ZMQExecutionClient


def test_execution_server_enables_fatal_signal_tracebacks(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Owned Python servers emit a traceback before fatal-signal termination."""

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
                popen_arguments=lambda: {"creationflags": 73}
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
