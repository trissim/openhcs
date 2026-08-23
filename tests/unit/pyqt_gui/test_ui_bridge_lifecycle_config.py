from __future__ import annotations

import pickle
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import zmq
from pyqt_reactive.services.zmq_server_scan_service import ZMQServerScanService
from zmqruntime import (
    ControlMessageType,
    ResponseType,
    ServerRole,
    TcpDataControlPortPairAuthority,
    TransportEndpoint,
    TransportMode,
)
from zmqruntime.transport import get_default_transport_mode

from openhcs.pyqt_gui.config import (
    AgentUiBridgeConfig,
    PyQtGuiRuntimeContext,
    get_default_ui_config,
)
from openhcs.pyqt_gui.main import OpenHCSMainWindow
from openhcs.pyqt_gui.services.main_window_workflows import MainWindowUiBridgeLifecycle
from openhcs.pyqt_gui.services.ui_bridge_composition import (
    OpenHCSUiBridgeCompositionRoot,
)
from openhcs.pyqt_gui.services.ui_bridge_server import (
    UI_BRIDGE_BROWSER_SERVER_NAME,
    UiBridgeControlServer,
)
from openhcs.runtime.zmq_application import OPENHCS_ENDPOINT_APPLICATION
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG, OpenHCSZMQConfig


def test_agent_ui_bridge_config_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv("OPENHCS_ENABLE_UI_BRIDGE", "1")
    monkeypatch.setenv("OPENHCS_UI_BRIDGE_HOST", "127.0.0.2")
    monkeypatch.setenv("OPENHCS_UI_BRIDGE_PORT", "7999")
    monkeypatch.setenv(
        "OPENHCS_UI_BRIDGE_TRANSPORT_MODE",
        TransportMode.TCP.value,
    )
    monkeypatch.setenv("OPENHCS_UI_BRIDGE_DESCRIPTOR_DIR", "/tmp/openhcs-bridge")

    config = get_default_ui_config().agent_bridge

    assert config.enabled is True
    assert config.host == "127.0.0.2"
    assert config.port == 7999
    assert config.transport_mode is TransportMode.TCP
    assert config.descriptor_directory_path == Path("/tmp/openhcs-bridge")


def test_agent_ui_bridge_config_is_enabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("OPENHCS_ENABLE_UI_BRIDGE", raising=False)

    assert AgentUiBridgeConfig.from_environment().enabled is True
    assert (
        AgentUiBridgeConfig.from_environment().transport_mode
        is get_default_transport_mode()
    )


def test_main_window_ui_bridge_lifecycle_stop_is_idempotent() -> None:
    server = _FakeBridgeServer(
        config=AgentUiBridgeConfig(),
        transport_config=OpenHCSZMQConfig(),
    )
    lifecycle = MainWindowUiBridgeLifecycle(server=server)

    lifecycle.close()
    lifecycle.close()

    assert server.stop_count == 1
    assert lifecycle.server is None


def test_ui_bridge_stop_preserves_live_worker_ownership_on_timeout() -> None:
    class _BlockedThread:
        def __init__(self) -> None:
            self.joined_with = None

        def join(self, timeout) -> None:
            self.joined_with = timeout

        def is_alive(self) -> bool:
            return True

    server = UiBridgeControlServer.__new__(UiBridgeControlServer)
    server._config = AgentUiBridgeConfig(shutdown_timeout_seconds=0.01)
    server._stop_event = threading.Event()
    server._thread = _BlockedThread()
    server._binding = object()
    server._bridge = SimpleNamespace(close=lambda: None)

    with pytest.raises(TimeoutError, match="UI bridge server"):
        server.stop()

    assert server._stop_event.is_set()
    assert server._thread is not None
    assert server._binding is not None
    assert server._thread.joined_with == 0.01


def test_main_window_zmq_scan_ports_include_actual_ui_bridge_binding(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "openhcs.core.config.get_all_streaming_ports",
        lambda num_ports_per_type: [5555],
    )
    ui_config = replace(
        get_default_ui_config(),
        agent_bridge=AgentUiBridgeConfig(
            host="127.0.0.1",
            port=7999,
            transport_mode=get_default_transport_mode(),
        ),
    )
    server = _FakeBridgeServer(
        config=ui_config.agent_bridge,
        transport_config=ui_config.zmq,
        binding_port=8124,
    )
    server.start()
    harness = SimpleNamespace(
        runtime_context=PyQtGuiRuntimeContext(ui_config),
        ui_bridge_lifecycle=MainWindowUiBridgeLifecycle(server=server),
    )

    assert OpenHCSMainWindow.zmq_server_manager_ports_to_scan(harness, ui_config) == [
        OPENHCS_ZMQ_CONFIG.default_port,
        5555,
        8124,
    ]


def test_main_window_zmq_scan_ports_exclude_unstarted_configured_bridge(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "openhcs.core.config.get_all_streaming_ports",
        lambda num_ports_per_type: [5555],
    )
    ui_config = replace(
        get_default_ui_config(),
        agent_bridge=replace(get_default_ui_config().agent_bridge, port=7999),
    )
    harness = SimpleNamespace(
        runtime_context=PyQtGuiRuntimeContext(ui_config),
        ui_bridge_lifecycle=MainWindowUiBridgeLifecycle(),
    )

    assert OpenHCSMainWindow.zmq_server_manager_ports_to_scan(harness, ui_config) == [
        OPENHCS_ZMQ_CONFIG.default_port,
        5555,
    ]


def test_ui_bridge_reconcile_starts_candidate_and_replaces_previous() -> None:
    transport_config = OpenHCSZMQConfig()
    original_config = AgentUiBridgeConfig(port=7998)
    replacement_config = replace(original_config, port=7999)
    original = _FakeBridgeServer(
        config=original_config,
        transport_config=transport_config,
    )
    original.start()
    replacement = _FakeBridgeServer(
        config=replacement_config,
        transport_config=transport_config,
    )
    lifecycle = MainWindowUiBridgeLifecycle(server=original)

    binding = lifecycle.reconcile(
        config=replacement_config,
        transport_config=transport_config,
        create_server=lambda config, transport: replacement,
    )

    assert binding is replacement.binding
    assert lifecycle.server is replacement
    assert replacement.start_count == 1
    assert original.stop_count == 1
    assert original.is_running is False


def test_main_window_builds_one_bridge_service_per_server(
    monkeypatch,
) -> None:
    services = []

    class _Composition:
        def build_service(self):
            service = SimpleNamespace(close=lambda: None)
            services.append(service)
            return service

    harness = SimpleNamespace()

    def composition_for_main_window(cls, main_window):
        assert cls is OpenHCSUiBridgeCompositionRoot
        assert main_window is harness
        return _Composition()

    monkeypatch.setattr(
        OpenHCSUiBridgeCompositionRoot,
        "for_main_window",
        classmethod(composition_for_main_window),
    )
    bridge_config = AgentUiBridgeConfig(port=7999)
    transport_config = OpenHCSZMQConfig()

    first = OpenHCSMainWindow._create_ui_bridge_server(
        harness,
        bridge_config,
        transport_config,
    )
    second = OpenHCSMainWindow._create_ui_bridge_server(
        harness,
        bridge_config,
        transport_config,
    )

    assert first._bridge is services[0]
    assert second._bridge is services[1]
    assert first._bridge is not second._bridge
    assert len(services) == 2


def test_ui_bridge_reconcile_is_noop_for_exact_running_config() -> None:
    config = AgentUiBridgeConfig(port=7999)
    transport_config = OpenHCSZMQConfig()
    server = _FakeBridgeServer(
        config=config,
        transport_config=transport_config,
    )
    server.start()
    lifecycle = MainWindowUiBridgeLifecycle(server=server)
    factory_calls = 0

    def create_server(config, transport):
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError("exact reconciliation must not create a server")

    binding = lifecycle.reconcile(
        config=config,
        transport_config=transport_config,
        create_server=create_server,
    )

    assert binding is server.binding
    assert factory_calls == 0
    assert server.stop_count == 0
    assert server.start_count == 1


def test_ui_bridge_reconcile_disables_and_clears_server() -> None:
    config = AgentUiBridgeConfig(port=7999)
    transport_config = OpenHCSZMQConfig()
    server = _FakeBridgeServer(
        config=config,
        transport_config=transport_config,
    )
    server.start()
    lifecycle = MainWindowUiBridgeLifecycle(server=server)

    result = lifecycle.reconcile(
        config=replace(config, enabled=False),
        transport_config=transport_config,
        create_server=lambda config, transport: (_ for _ in ()).throw(
            AssertionError("disabled reconciliation must not create a server")
        ),
    )

    assert result is None
    assert lifecycle.server is None
    assert server.stop_count == 1


def test_ui_bridge_reconcile_restores_previous_after_factory_failure() -> None:
    config = AgentUiBridgeConfig(port=7998)
    replacement_config = replace(config, port=7999)
    transport_config = OpenHCSZMQConfig()
    original = _FakeBridgeServer(
        config=config,
        transport_config=transport_config,
    )
    original.start()
    rollback = _FakeBridgeServer(
        config=config,
        transport_config=transport_config,
    )
    lifecycle = MainWindowUiBridgeLifecycle(server=original)

    with pytest.raises(ValueError, match="factory failed"):
        lifecycle.reconcile(
            config=replacement_config,
            transport_config=transport_config,
            create_server=lambda requested, transport: (
                rollback
                if requested == config
                else (_ for _ in ()).throw(ValueError("factory failed"))
            ),
        )

    assert lifecycle.server is rollback
    assert original.stop_count == 1
    assert original.start_count == 1
    assert rollback.start_count == 1
    assert rollback.is_running is True


def test_ui_bridge_reconcile_cleans_partial_start_and_restores_previous() -> None:
    config = AgentUiBridgeConfig(port=7998)
    replacement_config = replace(config, port=7999)
    transport_config = OpenHCSZMQConfig()
    original = _FakeBridgeServer(
        config=config,
        transport_config=transport_config,
    )
    original.start()
    candidate = _FakeBridgeServer(
        config=replacement_config,
        transport_config=transport_config,
        start_error=ValueError("candidate start failed"),
    )
    rollback = _FakeBridgeServer(
        config=config,
        transport_config=transport_config,
    )
    lifecycle = MainWindowUiBridgeLifecycle(server=original)

    with pytest.raises(ValueError, match="candidate start failed"):
        lifecycle.reconcile(
            config=replacement_config,
            transport_config=transport_config,
            create_server=lambda requested, transport: (
                candidate if requested == replacement_config else rollback
            ),
        )

    assert candidate.start_count == 1
    assert candidate.stop_count == 1
    assert candidate.is_running is False
    assert lifecycle.server is rollback
    assert original.start_count == 1
    assert rollback.start_count == 1
    assert rollback.is_running is True


def test_ui_bridge_reconcile_reports_failed_replacement_and_rollback() -> None:
    config = AgentUiBridgeConfig(port=7998)
    replacement_config = replace(config, port=7999)
    transport_config = OpenHCSZMQConfig()
    original = _FakeBridgeServer(
        config=config,
        transport_config=transport_config,
    )
    original.start()
    candidate = _FakeBridgeServer(
        config=replacement_config,
        transport_config=transport_config,
        start_error=ValueError("candidate start failed"),
    )
    rollback = _FakeBridgeServer(
        config=config,
        transport_config=transport_config,
        start_error=ValueError("old restart failed"),
    )
    lifecycle = MainWindowUiBridgeLifecycle(server=original)

    with pytest.raises(
        RuntimeError,
        match="previous bridge could not be restored",
    ) as caught:
        lifecycle.reconcile(
            config=replacement_config,
            transport_config=transport_config,
            create_server=lambda requested, transport: (
                candidate if requested == replacement_config else rollback
            ),
        )

    assert isinstance(caught.value.__cause__, ExceptionGroup)
    assert lifecycle.server is None
    assert original.start_count == 1
    assert rollback.start_count == 1
    assert candidate.stop_count == 1


def test_ui_bridge_reconcile_retains_candidate_when_cleanup_times_out() -> None:
    config = AgentUiBridgeConfig(port=7998)
    replacement_config = replace(config, port=7999)
    transport_config = OpenHCSZMQConfig()
    original = _FakeBridgeServer(
        config=config,
        transport_config=transport_config,
    )
    original.start()
    candidate = _FakeBridgeServer(
        config=replacement_config,
        transport_config=transport_config,
        start_error=ValueError("candidate start failed"),
        stop_error=TimeoutError("candidate still stopping"),
    )
    lifecycle = MainWindowUiBridgeLifecycle(server=original)

    with pytest.raises(RuntimeError, match="candidate bridge could not be stopped"):
        lifecycle.reconcile(
            config=replacement_config,
            transport_config=transport_config,
            create_server=lambda config, transport: candidate,
        )

    assert lifecycle.server is candidate
    assert candidate.stop_count == 1
    assert original.start_count == 1


def test_ui_bridge_answers_zmq_browser_control_ping(tmp_path) -> None:
    port = TcpDataControlPortPairAuthority.acquire(OPENHCS_ZMQ_CONFIG).data_port
    server = UiBridgeControlServer(
        bridge=SimpleNamespace(close=lambda: None),
        config=AgentUiBridgeConfig(
            host="127.0.0.1",
            port=port,
            transport_mode=TransportMode.TCP,
            descriptor_directory_path=tmp_path,
        ),
    )

    context = None
    socket = None
    try:
        binding = server.start()
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, 1000)
        socket.connect(
            f"tcp://127.0.0.1:{port + OPENHCS_ZMQ_CONFIG.control_port_offset}"
        )
        socket.send(
            pickle.dumps({"type": ControlMessageType.PING.value})
        )
        response = pickle.loads(socket.recv())
    finally:
        server.stop()
        if socket is not None:
            socket.close(linger=0)
        if context is not None:
            context.term()

    assert response["type"] == ResponseType.PONG.value
    assert response["server"] == UI_BRIDGE_BROWSER_SERVER_NAME
    assert response["server_role"] == ServerRole.GENERIC.value
    assert response["port"] == port
    assert response["control_port"] == (port + OPENHCS_ZMQ_CONFIG.control_port_offset)
    assert response["ready"] is True
    assert response["bridge_instance_id"] == binding.bridge_instance_id
    assert response["application"] == OPENHCS_ENDPOINT_APPLICATION.to_dict()


def test_ui_bridge_matches_only_its_exact_live_declarations(tmp_path) -> None:
    transport_config = OpenHCSZMQConfig()
    port = TcpDataControlPortPairAuthority.acquire(transport_config).data_port
    bridge_config = AgentUiBridgeConfig(
        host="127.0.0.1",
        port=port,
        transport_mode=TransportMode.TCP,
        descriptor_directory_path=tmp_path,
    )
    server = UiBridgeControlServer(
        bridge=SimpleNamespace(close=lambda: None),
        config=bridge_config,
        transport_config=transport_config,
    )

    try:
        server.start()

        assert not server.matches_configuration(
            bridge_config,
            replace(
                transport_config,
                server_info_timeout_ms=(transport_config.server_info_timeout_ms + 137),
            ),
        )
        assert not server.matches_configuration(
            bridge_config,
            replace(
                transport_config,
                control_port_offset=transport_config.control_port_offset + 1,
            ),
        )
    finally:
        server.stop()


def test_zmq_browser_scan_service_discovers_ui_bridge_default_transport(
    tmp_path,
) -> None:
    port = TcpDataControlPortPairAuthority.acquire(OPENHCS_ZMQ_CONFIG).data_port
    server = UiBridgeControlServer(
        bridge=SimpleNamespace(close=lambda: None),
        config=AgentUiBridgeConfig(
            host="127.0.0.1",
            port=port,
            transport_mode=get_default_transport_mode(),
            descriptor_directory_path=tmp_path,
        ),
    )
    scan_service = ZMQServerScanService(
        config=OPENHCS_ZMQ_CONFIG,
        host="localhost",
        transport_mode=get_default_transport_mode(),
        timeout_ms=1000,
    )

    try:
        server.start()
        snapshot = scan_service.scan_ports([port])
    finally:
        server.stop()

    endpoint = TransportEndpoint(
        host="127.0.0.1",
        port=port,
        transport_mode=get_default_transport_mode(),
    )

    assert len(snapshot.responses) == 1
    assert snapshot.responses[0].server == UI_BRIDGE_BROWSER_SERVER_NAME
    assert snapshot.responses[0].port == port
    assert snapshot.responses[0].application == OPENHCS_ENDPOINT_APPLICATION
    assert endpoint.occupied_ports(OPENHCS_ZMQ_CONFIG) == frozenset()


class _FakeBridgeServer:
    def __init__(
        self,
        *,
        config: AgentUiBridgeConfig,
        transport_config: OpenHCSZMQConfig,
        binding_port: int | None = None,
        start_error: Exception | None = None,
        start_errors: tuple[Exception | None, ...] = (),
        stop_error: Exception | None = None,
    ) -> None:
        self.config = config
        self.transport_config = transport_config
        self.binding = SimpleNamespace(
            connection=SimpleNamespace(
                port=binding_port if binding_port is not None else config.port
            )
        )
        self.start_error = start_error
        self.start_errors = start_errors
        self.stop_error = stop_error
        self.start_count = 0
        self.stop_count = 0
        self.is_running = False

    def start(self):
        self.start_count += 1
        error = (
            self.start_errors[self.start_count - 1]
            if self.start_count <= len(self.start_errors)
            else self.start_error
        )
        if error is not None:
            raise error
        self.is_running = True
        return self.binding

    def stop(self) -> None:
        self.stop_count += 1
        if self.stop_error is not None:
            raise self.stop_error
        self.is_running = False

    def matches_configuration(
        self,
        config: AgentUiBridgeConfig,
        transport_config: OpenHCSZMQConfig,
    ) -> bool:
        return (
            self.is_running
            and self.config == config
            and self.transport_config == transport_config
        )
