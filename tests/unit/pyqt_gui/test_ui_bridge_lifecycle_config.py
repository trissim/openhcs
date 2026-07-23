from __future__ import annotations

from dataclasses import replace
import pickle
from types import SimpleNamespace

import zmq

from openhcs.pyqt_gui.config import (
    DEFAULT_AGENT_UI_BRIDGE_TRANSPORT,
    AgentUiBridgeConfig,
    PyQtGuiRuntimeContext,
    get_default_ui_config,
)
from openhcs.pyqt_gui.main import OpenHCSMainWindow
from openhcs.pyqt_gui.services.main_window_workflows import MainWindowUiBridgeLifecycle
from openhcs.pyqt_gui.services.ui_bridge_server import (
    DEFAULT_UI_BRIDGE_TRANSPORT,
    UI_BRIDGE_BROWSER_SERVER_NAME,
    UI_BRIDGE_BROWSER_PONG_TYPE,
    UiBridgeControlServer,
    UiBridgeBrowserControlMessageType,
)
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG
from pyqt_reactive.services.zmq_server_scan_service import ZMQServerScanService
from zmqruntime import TcpDataControlPortPairAuthority


def test_agent_ui_bridge_config_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv("OPENHCS_ENABLE_UI_BRIDGE", "1")
    monkeypatch.setenv("OPENHCS_UI_BRIDGE_HOST", "127.0.0.2")
    monkeypatch.setenv("OPENHCS_UI_BRIDGE_PORT", "7999")
    monkeypatch.setenv("OPENHCS_UI_BRIDGE_TRANSPORT_MODE", DEFAULT_UI_BRIDGE_TRANSPORT)
    monkeypatch.setenv("OPENHCS_UI_BRIDGE_TIMEOUT_MS", "1234")
    monkeypatch.setenv("OPENHCS_UI_BRIDGE_DESCRIPTOR_DIR", "/tmp/openhcs-bridge")

    config = get_default_ui_config().agent_bridge

    assert config.enabled is True
    assert config.host == "127.0.0.2"
    assert config.port == 7999
    assert config.transport_mode == DEFAULT_UI_BRIDGE_TRANSPORT
    assert config.timeout_ms == 1234
    assert config.descriptor_directory_path == "/tmp/openhcs-bridge"


def test_agent_ui_bridge_config_is_enabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("OPENHCS_ENABLE_UI_BRIDGE", raising=False)

    assert AgentUiBridgeConfig.from_environment().enabled is True
    assert AgentUiBridgeConfig.from_environment().transport_mode == DEFAULT_AGENT_UI_BRIDGE_TRANSPORT


def test_main_window_ui_bridge_lifecycle_stop_is_idempotent() -> None:
    server = _FakeBridgeServer()
    lifecycle = MainWindowUiBridgeLifecycle()

    lifecycle.set_server(server)
    lifecycle.close()
    lifecycle.close()

    assert server.stop_count == 1
    assert lifecycle.server is None


def test_main_window_zmq_scan_ports_include_ui_bridge(monkeypatch) -> None:
    monkeypatch.setattr(
        "openhcs.core.config.get_all_streaming_ports",
        lambda num_ports_per_type: [5555],
    )
    ui_config = replace(
        get_default_ui_config(),
        agent_bridge=AgentUiBridgeConfig(
            host="127.0.0.1",
            port=7999,
            transport_mode=DEFAULT_AGENT_UI_BRIDGE_TRANSPORT,
        ),
    )
    harness = SimpleNamespace(runtime_context=PyQtGuiRuntimeContext(ui_config))

    assert OpenHCSMainWindow.zmq_server_manager_ports_to_scan(harness) == [
        OPENHCS_ZMQ_CONFIG.default_port,
        5555,
        7999,
    ]


def test_ui_bridge_answers_zmq_browser_control_ping(tmp_path) -> None:
    port = TcpDataControlPortPairAuthority.acquire(OPENHCS_ZMQ_CONFIG).data_port
    server = UiBridgeControlServer(
        bridge=object(),
        config=AgentUiBridgeConfig(
            host="127.0.0.1",
            port=port,
            transport_mode=DEFAULT_UI_BRIDGE_TRANSPORT,
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
            pickle.dumps({"type": UiBridgeBrowserControlMessageType.PING.value})
        )
        response = pickle.loads(socket.recv())
    finally:
        server.stop()
        if socket is not None:
            socket.close(linger=0)
        if context is not None:
            context.term()

    assert response["type"] == UI_BRIDGE_BROWSER_PONG_TYPE
    assert response["server"] == UI_BRIDGE_BROWSER_SERVER_NAME
    assert response["port"] == port
    assert response["control_port"] == (
        port + OPENHCS_ZMQ_CONFIG.control_port_offset
    )
    assert response["ready"] is True
    assert response["bridge_instance_id"] == binding.bridge_instance_id


def test_zmq_browser_scan_service_discovers_ui_bridge_default_transport(tmp_path) -> None:
    port = TcpDataControlPortPairAuthority.acquire(OPENHCS_ZMQ_CONFIG).data_port
    server = UiBridgeControlServer(
        bridge=object(),
        config=AgentUiBridgeConfig(
            host="127.0.0.1",
            port=port,
            transport_mode=DEFAULT_AGENT_UI_BRIDGE_TRANSPORT,
            descriptor_directory_path=tmp_path,
        ),
    )
    scan_service = ZMQServerScanService(
        control_port_offset=OPENHCS_ZMQ_CONFIG.control_port_offset,
        config=OPENHCS_ZMQ_CONFIG,
        host="localhost",
        timeout_ms=1000,
    )

    try:
        server.start()
        servers = scan_service.scan_ports([port])
    finally:
        server.stop()

    assert len(servers) == 1
    assert servers[0]["type"] == UI_BRIDGE_BROWSER_PONG_TYPE
    assert servers[0]["server"] == UI_BRIDGE_BROWSER_SERVER_NAME
    assert servers[0]["port"] == port


class _FakeBridgeServer:
    def __init__(self) -> None:
        self.stop_count = 0

    def stop(self) -> None:
        self.stop_count += 1
