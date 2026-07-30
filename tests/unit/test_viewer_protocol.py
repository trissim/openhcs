import subprocess
import sys

import pytest
from zmqruntime.config import TransportMode
from zmqruntime.messages import ControlMessageType

from openhcs.core.streaming_config_factory import (
    StreamingViewerPresentation,
    StreamingViewerRuntimeConfig,
)
from polystore.streaming.viewer_transport import ViewerTransportEndpoint
from openhcs.core.streaming_config_declarations import ViewerType
from openhcs.runtime.viewer_protocol import (
    DetachedViewerLaunchLog,
    DetachedViewerLaunchRequest,
    DetachedViewerPythonArguments,
    DetachedViewerPythonExpression,
    DetachedViewerServerEntrypointSpec,
    ManagedViewerLifecycleMixin,
    ViewerGraphicalSessionUnavailableError,
    ViewerLaunchContext,
    ViewerProcessPlatform,
    ViewerControlMessageRequest,
    ViewerControlMessageType,
    ViewerControlResponse,
    ViewerControlPingMode,
    ViewerControlPingRequest,
    ViewerQtEnvironmentPolicy,
    ViewerProcessHandle,
    ViewerRuntimeEndpoint,
    ViewerSettlePhase,
    ViewerSettleProgress,
)
import openhcs.runtime.viewer_protocol as viewer_protocol
from openhcs.runtime.viewer_controls import ViewerStateControlOptions
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG


@pytest.fixture(autouse=True)
def preserve_managed_viewer_registry():
    """Keep function-local viewer test classes out of the production registry."""

    registry = ManagedViewerLifecycleMixin.__registry__
    registered_viewers = registry.copy()
    try:
        yield
    finally:
        registry.clear()
        registry.update(registered_viewers)


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

    quick_request = ViewerControlPingRequest.from_mode(
        mode=ViewerControlPingMode.QUICK,
        endpoint=ViewerRuntimeEndpoint(
            transport=ViewerTransportEndpoint(
                port=55,
                host="localhost",
                transport_mode=TransportMode.IPC,
            ),
            config=OPENHCS_ZMQ_CONFIG,
        ),
    )
    assert quick_request.endpoint.ping(
        timeout_ms=quick_request.timeout_ms,
        require_ready=quick_request.require_ready,
    )

    ready_request = ViewerControlPingRequest.from_mode(
        mode=ViewerControlPingMode.EXISTING_VIEWER,
        endpoint=ViewerRuntimeEndpoint(
            transport=ViewerTransportEndpoint(
                port=56,
                host="localhost",
                transport_mode=TransportMode.TCP,
            ),
            config=OPENHCS_ZMQ_CONFIG,
        ),
    )
    assert ready_request.endpoint.ping(
        timeout_ms=ready_request.timeout_ms,
        require_ready=ready_request.require_ready,
    )

    assert calls[0][0][1].value == "ipc"
    assert calls[1][0][1].value == "tcp"
    assert calls == [
        (
            calls[0][0],
            {
                "host": "localhost",
                "config": OPENHCS_ZMQ_CONFIG,
                "timeout_ms": 200,
                "require_ready": False,
            },
        ),
        (
            calls[1][0],
            {
                "host": "localhost",
                "config": OPENHCS_ZMQ_CONFIG,
                "timeout_ms": 500,
                "require_ready": True,
            },
        ),
    ]


def test_viewer_endpoint_removes_only_proven_stale_ipc_data_and_control_paths(
    monkeypatch,
):
    endpoint = ViewerRuntimeEndpoint(
        transport=ViewerTransportEndpoint(
            port=5900,
            host="localhost",
            transport_mode=TransportMode.IPC,
        ),
        config=OPENHCS_ZMQ_CONFIG,
    )
    probed: list[int] = []
    removed: list[int] = []

    def fake_is_stale(port, config):
        assert config is OPENHCS_ZMQ_CONFIG
        probed.append(port)
        return port == endpoint.port

    def fake_remove(port, config):
        assert config is OPENHCS_ZMQ_CONFIG
        removed.append(port)
        return True

    monkeypatch.setattr("zmqruntime.transport.ipc_socket_is_stale", fake_is_stale)
    monkeypatch.setattr("zmqruntime.transport.remove_ipc_socket", fake_remove)

    assert endpoint.remove_stale_ipc_sockets() == (endpoint.port,)
    assert probed == [endpoint.port, endpoint.control_port]
    assert removed == [endpoint.port]

    tcp_endpoint = ViewerRuntimeEndpoint(
        transport=ViewerTransportEndpoint(
            port=5900,
            host="localhost",
            transport_mode=TransportMode.TCP,
        ),
        config=OPENHCS_ZMQ_CONFIG,
    )
    assert tcp_endpoint.remove_stale_ipc_sockets() == ()
    assert probed == [endpoint.port, endpoint.control_port]


def test_viewer_endpoint_liveness_covers_data_and_control_bindings(monkeypatch):
    endpoint = ViewerRuntimeEndpoint(
        transport=ViewerTransportEndpoint(
            port=5900,
            host="localhost",
            transport_mode=TransportMode.IPC,
        ),
        config=OPENHCS_ZMQ_CONFIG,
    )
    bound_ports = {endpoint.control_port}
    probed: list[int] = []

    def fake_in_use(port, mode, *, host, config):
        assert mode.value == "ipc"
        assert host == "localhost"
        assert config is OPENHCS_ZMQ_CONFIG
        probed.append(port)
        return port in bound_ports

    monkeypatch.setattr("zmqruntime.transport.is_port_in_use", fake_in_use)

    assert endpoint.in_use()
    assert probed == [endpoint.port, endpoint.control_port]

    bound_ports.clear()
    probed.clear()
    assert not endpoint.in_use()
    assert probed == [endpoint.port, endpoint.control_port]


def test_managed_viewer_readiness_uses_endpoint_binding_authority(monkeypatch):
    calls = []

    class ProbeViewer(ManagedViewerLifecycleMixin):
        viewer_process_label = "Probe"
        detached_server_entrypoint = DetachedViewerServerEntrypointSpec(
            viewer_type=ViewerType.NAPARI,
            module_name="tests.fake_viewer",
            function_name="run",
        )

        def __init__(self):
            super().__init__(
                runtime_config=StreamingViewerRuntimeConfig(
                    transport_endpoint=ViewerTransportEndpoint(
                        port=42,
                        host="localhost",
                        transport_mode=TransportMode.IPC,
                    ),
                    persistent=False,
                    presentation=StreamingViewerPresentation("Probe"),
                )
            )

        def start_viewer(self, async_mode: bool = False) -> None:
            raise AssertionError("test does not launch a process")

        def detached_server_arguments(
            self,
            *,
            log_file,
        ) -> DetachedViewerPythonArguments:
            return DetachedViewerPythonArguments.from_literals(str(log_file))

    def wait_ready(_endpoint, *, timeout, require_ready):
        calls.append((timeout, require_ready))
        return True

    monkeypatch.setattr(ViewerRuntimeEndpoint, "wait_ready", wait_ready)

    assert ProbeViewer().wait_for_ready(timeout=0.5)
    assert calls == [(0.5, True)]


def test_managed_viewer_lifecycle_reads_state_through_typed_control_request(
    monkeypatch,
):
    requests = []
    response = ViewerControlResponse(
        payload={
            "status": "success",
            "layers": ({"route_key": "step-1:image", "item_count": 1},),
        }
    )

    class StateViewer(ManagedViewerLifecycleMixin):
        viewer_process_label = "State"
        detached_server_entrypoint = DetachedViewerServerEntrypointSpec(
            viewer_type=ViewerType.NAPARI,
            module_name="tests.fake_viewer",
            function_name="run",
        )

        def __init__(self):
            super().__init__(
                runtime_config=StreamingViewerRuntimeConfig(
                    transport_endpoint=ViewerTransportEndpoint(
                        port=42,
                        host="localhost",
                        transport_mode=TransportMode.IPC,
                    ),
                    persistent=False,
                    presentation=StreamingViewerPresentation("State"),
                )
            )

        def check_connected_viewer(self) -> bool:
            return True

        def start_viewer(self, async_mode: bool = False) -> None:
            raise AssertionError("test does not launch a process")

        def detached_server_arguments(
            self,
            *,
            log_file,
        ) -> DetachedViewerPythonArguments:
            return DetachedViewerPythonArguments.from_literals(str(log_file))

    def send(request):
        requests.append(request)
        return response

    monkeypatch.setattr(ViewerControlMessageRequest, "send", send)
    viewer = StateViewer()
    viewer.lifecycle_state.mark_connected_external()

    observed = viewer.read_viewer_state(timeout=7.5)

    assert observed is response
    assert len(requests) == 1
    assert requests[0].message_type == "state"
    assert requests[0].timeout == 7.5
    assert isinstance(requests[0].payload, ViewerStateControlOptions)
    assert requests[0].payload.include_component_values
    assert requests[0].payload.include_payload_summaries

    assert viewer.request_bound_viewer_shutdown()
    assert requests[-1].message_type == ControlMessageType.FORCE_SHUTDOWN.value
    assert viewer.clear_viewer_state()
    assert requests[-1].message_type == ViewerControlMessageType.CLEAR_STATE.value

    failed_response = ViewerControlResponse(
        payload={"status": "error", "message": "state unavailable"}
    )
    monkeypatch.setattr(
        ViewerControlMessageRequest,
        "send",
        lambda _request: failed_response,
    )
    with pytest.raises(RuntimeError, match="state request failed"):
        viewer.read_viewer_state()


def test_managed_viewer_settlement_tracks_progress_without_total_timeout(
    monkeypatch,
):
    class ProgressViewer(ManagedViewerLifecycleMixin):
        viewer_process_label = "Progress"
        detached_server_entrypoint = DetachedViewerServerEntrypointSpec(
            viewer_type=ViewerType.NAPARI,
            module_name="tests.fake_viewer",
            function_name="run",
        )

        def __init__(self):
            super().__init__(
                runtime_config=StreamingViewerRuntimeConfig(
                    transport_endpoint=ViewerTransportEndpoint(
                        port=42,
                        host="localhost",
                        transport_mode=TransportMode.IPC,
                    ),
                    persistent=False,
                    presentation=StreamingViewerPresentation("Progress"),
                )
            )

        def start_viewer(self, async_mode: bool = False) -> None:
            raise AssertionError("test does not launch a process")

        def detached_server_arguments(
            self,
            *,
            log_file,
        ) -> DetachedViewerPythonArguments:
            return DetachedViewerPythonArguments.from_literals(str(log_file))

    progress_rows = (
        ViewerSettleProgress(ViewerSettlePhase.RUNNING, 0, 1, "large", 0, True),
        ViewerSettleProgress(ViewerSettlePhase.RUNNING, 0, 1, "large", 0, True),
        ViewerSettleProgress(ViewerSettlePhase.RUNNING, 0, 1, "large", 1),
        ViewerSettleProgress.complete(1),
    )
    responses = [
        ViewerControlResponse(
            payload={"status": "success", **progress.to_wire_mapping()}
        )
        for progress in progress_rows
    ]
    requests = []

    def send(request):
        requests.append(request)
        return responses.pop(0)

    monotonic_values = iter((0.0, 0.9, 1.8, 2.7))
    monkeypatch.setattr(ViewerControlMessageRequest, "send", send)
    monkeypatch.setattr(viewer_protocol.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(viewer_protocol.time, "sleep", lambda _seconds: None)
    viewer = ProgressViewer()
    viewer.lifecycle_state.mark_connected_external()
    observed_progress = []

    assert viewer.settle_viewer_state(
        timeout=1.0,
        progress_callback=observed_progress.append,
    )
    assert len(requests) == 4
    assert observed_progress == list(progress_rows)
    assert all(
        request.message_type == ViewerControlMessageType.SETTLE.value
        for request in requests
    )


def test_managed_viewer_settlement_rejects_no_progress(monkeypatch):
    progress = ViewerSettleProgress(
        ViewerSettlePhase.RUNNING,
        0,
        1,
        "stalled",
    )
    response = ViewerControlResponse(
        payload={"status": "success", **progress.to_wire_mapping()}
    )
    viewer = type(
        "StalledViewer",
        (ManagedViewerLifecycleMixin,),
        {
            "viewer_process_label": "Stalled",
            "detached_server_entrypoint": DetachedViewerServerEntrypointSpec(
                viewer_type=ViewerType.NAPARI,
                module_name="tests.fake_viewer",
                function_name="run",
            ),
            "start_viewer": lambda self, async_mode=False: None,
            "detached_server_arguments": lambda self, *, log_file: (
                DetachedViewerPythonArguments.from_literals(str(log_file))
            ),
        },
    )(
        runtime_config=StreamingViewerRuntimeConfig(
            transport_endpoint=ViewerTransportEndpoint(
                port=42,
                host="localhost",
                transport_mode=TransportMode.IPC,
            ),
            persistent=False,
            presentation=StreamingViewerPresentation("Stalled"),
        )
    )
    viewer.lifecycle_state.mark_connected_external()
    monotonic_values = iter((0.0, 0.0, 1.1))
    monkeypatch.setattr(ViewerControlMessageRequest, "send", lambda _request: response)
    monkeypatch.setattr(viewer_protocol.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(viewer_protocol.time, "sleep", lambda _seconds: None)

    assert not viewer.settle_viewer_state(timeout=1.0)


def test_viewer_qt_environment_policy_applies_platform_rows():
    plugin_path = ViewerQtEnvironmentPolicy.active_qt_plugin_path()
    linux_env = ViewerQtEnvironmentPolicy(ViewerProcessPlatform.LINUX).apply_to({})
    expected_linux = {
        "QT_QPA_PLATFORM": "xcb",
        "QT_X11_NO_MITSHM": "1",
        "vblank_mode": "0",
    }
    if plugin_path is not None:
        expected_linux["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_path
    assert linux_env == expected_linux

    linux_existing = ViewerQtEnvironmentPolicy(ViewerProcessPlatform.LINUX).apply_to(
        {
            "QT_QPA_PLATFORM": "offscreen",
            "QT_PLUGIN_PATH": "/dependency/private/plugins",
            "QT_QPA_PLATFORM_PLUGIN_PATH": "/dependency/private/platforms",
        }
    )
    expected_existing = {
        "QT_QPA_PLATFORM": "offscreen",
        "QT_X11_NO_MITSHM": "1",
        "vblank_mode": "0",
    }
    if plugin_path is None:
        expected_existing.update(
            {
                "QT_PLUGIN_PATH": "/dependency/private/plugins",
                "QT_QPA_PLATFORM_PLUGIN_PATH": "/dependency/private/platforms",
            }
        )
    else:
        expected_existing["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_path
    assert linux_existing == expected_existing

    darwin_env = ViewerQtEnvironmentPolicy(ViewerProcessPlatform.DARWIN).apply_to({})
    expected_darwin = {"QT_QPA_PLATFORM": "cocoa"}
    if plugin_path is not None:
        expected_darwin["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_path
    assert darwin_env == expected_darwin

    windows_env = ViewerQtEnvironmentPolicy(ViewerProcessPlatform.WINDOWS).apply_to({})
    expected_windows = {}
    if plugin_path is not None:
        expected_windows["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_path
    assert windows_env == expected_windows


def test_detached_viewer_entrypoint_generates_public_process_call(tmp_path):
    python_code = DetachedViewerServerEntrypointSpec(
        viewer_type=ViewerType.NAPARI,
        module_name="openhcs.runtime.napari_viewer_server",
        function_name="run_napari_viewer_process",
    ).python_code(
        tmp_path,
        transport_mode=TransportMode.IPC,
        arguments=DetachedViewerPythonArguments.from_literals(
            1234,
            "Viewer",
            True,
            "/tmp/viewer.log",
        ).append(DetachedViewerPythonExpression.symbol("transport_mode")),
    )

    assert "run_napari_viewer_process" in python_code
    assert "openhcs.runtime.napari_viewer_server" in python_code
    assert " import _napari_viewer_process" not in python_code
    assert str(tmp_path) in python_code
    assert "TransportMode.IPC" in python_code
    assert 'if os.name == "posix"' in python_code
    assert "hasattr" not in python_code


def test_detached_viewer_launch_request_owns_log_and_python_command(tmp_path):
    spec = DetachedViewerServerEntrypointSpec(
        viewer_type=ViewerType.NAPARI,
        module_name="openhcs.runtime.napari_viewer_server",
        function_name="run_napari_viewer_process",
    )
    log_file = DetachedViewerLaunchRequest.log_file_for(
        viewer_type=spec.viewer_type,
        port=4321,
        log_dir=tmp_path / "logs",
    )
    launch = spec.launch_request(
        port=4321,
        transport_mode=TransportMode.TCP,
        arguments=DetachedViewerPythonArguments.from_literals(
            4321,
            "Detached",
            False,
            str(log_file),
        ).append(DetachedViewerPythonExpression.symbol("transport_mode")),
        log_file=log_file,
        cwd=tmp_path,
    )

    assert launch.log_file == tmp_path / "logs" / "napari_detached_port_4321.log"
    assert launch.cwd == tmp_path
    assert "TransportMode.TCP" in launch.python_code
    assert launch.command() == [sys.executable, "-c", launch.python_code]


def test_detached_viewer_launch_log_tail_is_bounded_by_bytes_and_lines(tmp_path):
    log_file = tmp_path / "viewer.log"
    log_file.write_text(
        "\n".join(f"line-{index:03d}-{'x' * 40}" for index in range(100)),
        encoding="utf-8",
    )

    tail = DetachedViewerLaunchLog(
        log_file,
        max_bytes=1024,
        max_lines=5,
    ).tail()

    assert tail.splitlines() == [
        f"line-{index:03d}-{'x' * 40}" for index in range(95, 100)
    ]
    assert len(tail.encode()) <= 1024


def test_managed_viewer_lifecycle_inherits_projected_launch_context():
    class EnvironmentViewer(ManagedViewerLifecycleMixin):
        viewer_process_label = "Environment"
        detached_server_entrypoint = DetachedViewerServerEntrypointSpec(
            viewer_type=ViewerType.NAPARI,
            module_name="tests.fake_viewer",
            function_name="run",
        )

        def __init__(self):
            super().__init__(
                runtime_config=StreamingViewerRuntimeConfig(
                    transport_endpoint=ViewerTransportEndpoint(
                        port=42,
                        host="localhost",
                        transport_mode=TransportMode.IPC,
                    ),
                    persistent=True,
                    presentation=StreamingViewerPresentation("Environment"),
                ),
            )

        def start_viewer(self, async_mode: bool = False) -> None:
            raise AssertionError("test does not launch a process")

        def detached_server_arguments(
            self,
            *,
            log_file,
        ) -> DetachedViewerPythonArguments:
            return DetachedViewerPythonArguments.from_literals(str(log_file))

    viewer = EnvironmentViewer()
    launch_context = ViewerLaunchContext.projected_graphical_session(
        {
            "DISPLAY": ":23",
            "XAUTHORITY": "/run/user/1000/xauth",
            "XDG_RUNTIME_DIR": "/run/user/1000",
        }
    )
    viewer.configure_launch_context(launch_context)

    request = viewer.detached_launch_request()

    assert request.launch_context is launch_context
    assert request.launch_context.environment_overlay == {
        "DISPLAY": ":23",
        "XAUTHORITY": "/run/user/1000/xauth",
        "XDG_RUNTIME_DIR": "/run/user/1000",
    }
    assert request.failure(RuntimeError("Qt/xcb unavailable")).log_file == (
        request.log_file
    )


def test_detached_viewer_launch_rejects_headless_context_before_spawn(
    monkeypatch,
    tmp_path,
):
    log_file = tmp_path / "viewer.log"
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("headless launch reached Popen")
        ),
    )
    request = DetachedViewerLaunchRequest(
        viewer_type=ViewerType.NAPARI,
        port=5555,
        python_code="raise AssertionError",
        log_file=log_file,
        launch_context=ViewerLaunchContext.headless(),
    )

    with pytest.raises(
        ViewerGraphicalSessionUnavailableError,
        match="no authoritative graphical session",
    ):
        request.launch()

    assert not log_file.exists()


def test_managed_viewer_lifecycle_uses_nominal_state_for_external_viewer():
    class ExternalViewer(ManagedViewerLifecycleMixin):
        viewer_process_label = "External"
        detached_server_entrypoint = DetachedViewerServerEntrypointSpec(
            viewer_type=ViewerType.NAPARI,
            module_name="tests.fake_viewer",
            function_name="run",
        )

        def __init__(self):
            super().__init__(
                runtime_config=StreamingViewerRuntimeConfig(
                    transport_endpoint=ViewerTransportEndpoint(
                        port=42,
                        host="localhost",
                        transport_mode=TransportMode.IPC,
                    ),
                    persistent=True,
                    presentation=StreamingViewerPresentation("External"),
                ),
            )
            self.connected = True

        def check_connected_viewer(self) -> bool:
            return self.connected

        def start_viewer(self, async_mode: bool = False) -> None:
            raise AssertionError("test does not launch a process")

        def detached_server_arguments(
            self,
            *,
            log_file,
        ) -> DetachedViewerPythonArguments:
            return DetachedViewerPythonArguments.from_literals(str(log_file))

    viewer = ExternalViewer()
    assert not viewer.is_running

    viewer.lifecycle_state.mark_connected_external()
    assert viewer.is_running

    viewer.connected = False
    assert not viewer.is_running
    assert not viewer.lifecycle_state.is_active


def test_prepare_fresh_viewer_start_releases_endpoint_after_shutdown_ack():
    class BoundEndpoint:
        def __init__(self):
            self.bound = True
            self.release_calls = 0
            self.wait_calls = 0

        def in_use(self):
            return self.bound

        def wait_until_released(self, *, timeout, poll_interval=0.1):
            self.wait_calls += 1
            return not self.bound

        def release_bound_ports(self):
            self.release_calls += 1
            self.bound = False

    class FreshViewer(ManagedViewerLifecycleMixin):
        viewer_process_label = "Fresh"
        detached_server_entrypoint = DetachedViewerServerEntrypointSpec(
            viewer_type=ViewerType.NAPARI,
            module_name="tests.fake_viewer",
            function_name="run",
        )

        def __init__(self, endpoint):
            super().__init__(
                runtime_config=StreamingViewerRuntimeConfig(
                    transport_endpoint=ViewerTransportEndpoint(
                        port=42,
                        host="localhost",
                        transport_mode=TransportMode.IPC,
                    ),
                    persistent=True,
                    presentation=StreamingViewerPresentation("Fresh"),
                ),
            )
            self.runtime_endpoint = endpoint
            self.shutdown_requests = 0

        def check_connected_viewer(self) -> bool:
            return True

        def request_bound_viewer_shutdown(self, timeout: float = 1.0) -> bool:
            self.shutdown_requests += 1
            return True

        def start_viewer(self, async_mode: bool = False) -> None:
            raise AssertionError("test does not launch a process")

        def detached_server_arguments(
            self,
            *,
            log_file,
        ) -> DetachedViewerPythonArguments:
            return DetachedViewerPythonArguments.from_literals(str(log_file))

    endpoint = BoundEndpoint()
    viewer = FreshViewer(endpoint)

    viewer.prepare_fresh_viewer_start()

    assert viewer.shutdown_requests == 1
    assert endpoint.wait_calls == 2
    assert endpoint.release_calls == 1
    assert not endpoint.in_use()


def test_prepare_fresh_viewer_start_reports_still_bound_after_forced_release():
    class StuckEndpoint:
        def __init__(self):
            self.release_calls = 0

        def in_use(self):
            return True

        def wait_until_released(self, *, timeout, poll_interval=0.1):
            return False

        def release_bound_ports(self):
            self.release_calls += 1

    class StuckViewer(ManagedViewerLifecycleMixin):
        viewer_process_label = "Stuck"
        detached_server_entrypoint = DetachedViewerServerEntrypointSpec(
            viewer_type=ViewerType.NAPARI,
            module_name="tests.fake_viewer",
            function_name="run",
        )

        def __init__(self, endpoint):
            super().__init__(
                runtime_config=StreamingViewerRuntimeConfig(
                    transport_endpoint=ViewerTransportEndpoint(
                        port=42,
                        host="localhost",
                        transport_mode=TransportMode.IPC,
                    ),
                    persistent=True,
                    presentation=StreamingViewerPresentation("Stuck"),
                ),
            )
            self.runtime_endpoint = endpoint

        def check_connected_viewer(self) -> bool:
            return True

        def request_bound_viewer_shutdown(self, timeout: float = 1.0) -> bool:
            return True

        def start_viewer(self, async_mode: bool = False) -> None:
            raise AssertionError("test does not launch a process")

        def detached_server_arguments(
            self,
            *,
            log_file,
        ) -> DetachedViewerPythonArguments:
            return DetachedViewerPythonArguments.from_literals(str(log_file))

    endpoint = StuckEndpoint()
    viewer = StuckViewer(endpoint)

    with pytest.raises(RuntimeError, match="forced endpoint release"):
        viewer.prepare_fresh_viewer_start()

    assert endpoint.release_calls == 1
