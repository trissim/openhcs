from __future__ import annotations

import ast
import inspect
from pathlib import Path

from openhcs.agent.dto.execution import ExecutionConnectionSpec
from openhcs.agent.services.execution_session_service import ZMQExecutionClientFactory
from openhcs.agent.services.runtime_server_service import (
    RuntimeServerGatewayABC,
    RuntimeServerService,
    ZMQRuntimeServerGateway,
)
from openhcs.runtime.zmq_config import OpenHCSZMQConfig
from openhcs.pyqt_gui.windows.live_measurements_window import LiveMeasurementsWindow


REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_ROOT = REPO_ROOT / "openhcs"


class RecordingRuntimeGateway(RuntimeServerGatewayABC):
    def __init__(self) -> None:
        self.scan_request = None

    def server_info(self, connection, *, timeout_ms):
        raise AssertionError("server_info is not used by this test")

    def execution_status(self, connection, execution_id=None, *, timeout_ms):
        raise AssertionError("execution_status is not used by this test")

    def scan(self, *, host, ports, transport_mode, timeout_ms):
        self.scan_request = (host, ports, transport_mode, timeout_ms)
        return ()


def test_agent_factories_pass_the_exact_zmq_config(monkeypatch) -> None:
    import openhcs.agent.services.execution_session_service as execution_module
    import openhcs.agent.services.runtime_server_service as runtime_module

    captured = []

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            captured.append(kwargs)

    monkeypatch.setattr(execution_module, "ZMQExecutionClient", FakeClient)
    monkeypatch.setattr(runtime_module, "ZMQExecutionClient", FakeClient)
    config = OpenHCSZMQConfig(default_port=8123)
    connection = ExecutionConnectionSpec(port=8124, persistent=False)

    factory_client = ZMQExecutionClientFactory(config).create_client(connection)
    gateway_client = ZMQRuntimeServerGateway(config)._client(connection)

    assert factory_client.client is not None
    assert gateway_client is not None
    assert len(captured) == 2
    assert all(request["config"] is config for request in captured)
    assert all(request["port"] == 8124 for request in captured)


def test_runtime_server_defaults_are_resolved_from_injected_config() -> None:
    gateway = RecordingRuntimeGateway()
    config = OpenHCSZMQConfig(
        default_port=8123,
        server_scan_timeout_ms=321,
    )
    service = RuntimeServerService(gateway=gateway, config=config)

    result = service.scan()

    assert result.ports == (8123,)
    assert result.timeout_ms == 321
    assert gateway.scan_request == ("localhost", (8123,), None, 321)


def test_live_measurements_window_requires_explicit_zmq_config() -> None:
    parameter = inspect.signature(LiveMeasurementsWindow.__init__).parameters[
        "zmq_config"
    ]

    assert parameter.default is inspect.Parameter.empty


def test_deleted_parallel_config_authorities_do_not_reappear() -> None:
    production_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(PRODUCTION_ROOT.rglob("*.py"))
    )

    for deleted_name in (
        "PyQtGUIConfig",
        "get_default_pyqt_gui_config",
        "get_shortcut_config",
        "UI_CONFIG_SCOPE_ID",
        "DEFAULT_EXECUTION_SERVER_PORT",
        "CONTROL_PORT_OFFSET",
        "IPC_SOCKET_DIR_NAME",
        "IPC_SOCKET_PREFIX",
        "IPC_SOCKET_EXTENSION",
        "DEFAULT_EXECUTION_SUBMIT_TIMEOUT_MS",
        "DEFAULT_EXECUTION_STATUS_TIMEOUT_MS",
        "DEFAULT_EXECUTION_WAIT_TIMEOUT_MS",
        "DEFAULT_RUNTIME_SERVER_INFO_TIMEOUT_MS",
        "DEFAULT_RUNTIME_SERVER_SCAN_TIMEOUT_MS",
        "POLYSTORE_ZMQ_",
        "main_window.global_config",
    ):
        assert deleted_name not in production_source


def test_production_zmq_consumers_receive_config_objects() -> None:
    required_config_calls = {
        "ImageBrowserWidget",
        "LiveMeasurementsWindow",
        "ZMQClientService",
        "ZMQExecutionClient",
        "ZMQExecutionServer",
        "ZMQServerManagerWidget",
    }
    missing: list[str] = []

    for path in sorted(PRODUCTION_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = None
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name not in required_config_calls:
                continue
            if any(
                keyword.arg in {"config", "zmq_config"}
                for keyword in node.keywords
            ):
                continue
            missing.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}:{name}")

    assert missing == []
