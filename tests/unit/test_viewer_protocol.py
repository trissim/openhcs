import subprocess
import sys

import pytest

from openhcs.runtime.viewer_protocol import (
    NapariDetachedProcessRequest,
    NapariViewerProcessEntrypoint,
    NapariViewerServerRequest,
    ManagedViewerLifecycleMixin,
    ViewerProcessPlatform,
    ViewerControlPingMode,
    ViewerControlPingRequest,
    ViewerLifecycleState,
    ViewerQtEnvironmentPolicy,
    ViewerProcessHandle,
)


def test_viewer_process_handle_wraps_subprocess_lifecycle():
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        handle = ViewerProcessHandle.from_process(process)

        assert handle.pid == process.pid
        assert handle.pid_label == str(process.pid)
        assert handle.is_alive()
        assert not handle.terminate(timeout=1, kill_timeout=1)
        assert not handle.is_alive()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=1)


def test_viewer_process_handle_rejects_structural_process_lookalikes():
    class ProcessLike:
        def is_alive(self):
            return True

    with pytest.raises(TypeError, match="Unsupported viewer process handle"):
        ViewerProcessHandle.from_process(ProcessLike())


def test_viewer_control_ping_request_owns_quick_and_ready_projection(monkeypatch):
    calls = []

    def fake_ping_control_port(*args, **kwargs):
        calls.append((args, kwargs))
        return True

    monkeypatch.setattr(
        "zmqruntime.transport.ping_control_port",
        fake_ping_control_port,
    )

    assert ViewerControlPingRequest.from_mode(
        mode=ViewerControlPingMode.QUICK,
        port=55,
        transport_mode="ipc",
        config="config",
    ).check()
    assert ViewerControlPingRequest.from_mode(
        mode=ViewerControlPingMode.EXISTING_VIEWER,
        port=56,
        transport_mode="tcp",
        config="config",
    ).check()

    assert calls == [
        (
            (55, "ipc"),
            {
                "host": "localhost",
                "config": "config",
                "timeout_ms": 200,
                "require_ready": False,
            },
        ),
        (
            (56, "tcp"),
            {
                "host": "localhost",
                "config": "config",
                "timeout_ms": 500,
                "require_ready": True,
            },
        ),
    ]


def test_viewer_qt_environment_policy_applies_platform_rows():
    linux_env = ViewerQtEnvironmentPolicy(ViewerProcessPlatform.LINUX).apply_to({})
    assert linux_env == {"QT_QPA_PLATFORM": "xcb", "QT_X11_NO_MITSHM": "1"}

    linux_existing = ViewerQtEnvironmentPolicy(ViewerProcessPlatform.LINUX).apply_to(
        {"QT_QPA_PLATFORM": "offscreen"}
    )
    assert linux_existing == {
        "QT_QPA_PLATFORM": "offscreen",
        "QT_X11_NO_MITSHM": "1",
    }

    darwin_env = ViewerQtEnvironmentPolicy(ViewerProcessPlatform.DARWIN).apply_to({})
    assert darwin_env == {"QT_QPA_PLATFORM": "cocoa"}

    windows_env = ViewerQtEnvironmentPolicy(ViewerProcessPlatform.WINDOWS).apply_to({})
    assert windows_env == {}


def test_napari_viewer_server_request_owns_legacy_signature_projection():
    request = NapariViewerServerRequest.from_legacy_signature(
        1234,
        "Viewer",
        True,
        "/tmp/viewer.log",
        "ipc",
    )

    assert request == NapariViewerServerRequest(
        port=1234,
        viewer_title="Viewer",
        replace_layers=True,
        log_file_path="/tmp/viewer.log",
        transport_mode="ipc",
    )


def test_napari_viewer_process_entrypoint_generates_public_process_call(tmp_path):
    class FakeTransportMode:
        name = "IPC"

    request = NapariViewerServerRequest.from_legacy_signature(
        1234,
        "Viewer",
        True,
        "/tmp/viewer.log",
        FakeTransportMode(),
    )

    python_code = NapariViewerProcessEntrypoint(
        request=request,
        python_path_root=tmp_path,
    ).python_code()

    assert "run_napari_viewer_process_from_legacy_signature" in python_code
    assert "openhcs.runtime.napari_viewer_server" in python_code
    assert " import _napari_viewer_process" not in python_code
    assert str(tmp_path) in python_code
    assert "TransportMode.IPC" in python_code


def test_napari_detached_process_request_owns_log_and_python_command(tmp_path):
    class FakeTransportMode:
        name = "TCP"

    launch = NapariDetachedProcessRequest.from_legacy_signature(
        4321,
        "Detached",
        False,
        FakeTransportMode(),
        cwd=tmp_path,
        log_dir=tmp_path / "logs",
    )

    process_request = launch.to_process_request()

    assert launch.log_file == tmp_path / "logs" / "napari_detached_port_4321.log"
    assert launch.server_request.log_file_path == str(launch.log_file)
    assert process_request.log_file == launch.log_file
    assert process_request.cwd == tmp_path
    assert "TransportMode.TCP" in process_request.python_code


def test_managed_viewer_lifecycle_uses_nominal_state_for_external_viewer():
    class ExternalViewer(ManagedViewerLifecycleMixin):
        viewer_process_label = "External"

        def __init__(self):
            self.lifecycle_state = ViewerLifecycleState.stopped()
            self.port = 42
            self.process = None
            self.connected = True

        def check_connected_viewer(self) -> bool:
            return self.connected

    viewer = ExternalViewer()
    assert not viewer.is_running

    viewer.lifecycle_state.mark_connected_external()
    assert viewer.is_running

    viewer.connected = False
    assert not viewer.is_running
    assert not viewer.lifecycle_state.is_active
