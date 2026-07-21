from __future__ import annotations

import importlib.util
from types import SimpleNamespace

from openhcs.agent.capabilities import (
    CapabilityCliConnectionProfile,
    get_agent_capability,
    get_agent_capability_declaration,
)
from openhcs.agent.dto.execution import (
    ExecutionConnectionSpec,
    RuntimeDebugInspectionRequest,
    RuntimeDebugInspectionResult,
)
from openhcs.agent.services.runtime_server_service import (
    RuntimeServerService,
    ZMQRuntimeServerGateway,
)
from openhcs.agent.services import runtime_server_service as runtime_server_module
from openhcs.core.debug_views import (
    DebugViewModel,
    DebugViewSection,
    DebugViewSectionKind,
    DebugViewTable,
    DebugViewTableProjection,
)
from openhcs.serialization.json import to_jsonable


DEBUG_SESSION_ID = "debug-session-1"


def _runtime_view() -> DebugViewModel:
    return DebugViewModel(
        title="Runtime Values",
        sections=(
            DebugViewSection(
                kind=DebugViewSectionKind.RUNTIME_VALUES,
                title="Runtime Values",
                table=DebugViewTable(
                    columns=("key", "location", "value_type"),
                    rows=(("image-key", "/memory/image.npy", "ImageArray"),),
                    projection=DebugViewTableProjection.RUNTIME_VALUE_RECORDS,
                ),
            ),
        ),
    )


class _RuntimeDebugGateway:
    def __init__(self, view_model: DebugViewModel) -> None:
        self.view_model = view_model
        self.requests = []

    def runtime_debug_inspection(
        self,
        connection,
        debug_session_id,
        *,
        timeout_ms,
    ):
        self.requests.append((connection, debug_session_id, timeout_ms))
        return self.view_model


class _FailingRuntimeDebugGateway(_RuntimeDebugGateway):
    def runtime_debug_inspection(
        self,
        connection,
        debug_session_id,
        *,
        timeout_ms,
    ):
        self.requests.append((connection, debug_session_id, timeout_ms))
        raise RuntimeError("debug worker is not paused")


def test_runtime_debug_request_projects_connection_and_session_fields():
    request = RuntimeDebugInspectionRequest.from_fields(
        debug_session_id=DEBUG_SESSION_ID,
        host="127.0.0.1",
        port=7787,
        transport_mode="tcp",
        persistent=False,
        timeout_ms=321,
    )

    assert request.as_tool_arguments() == {
        "host": "127.0.0.1",
        "port": 7787,
        "transport_mode": "tcp",
        "persistent": False,
        "timeout_ms": 321,
        "debug_session_id": DEBUG_SESSION_ID,
    }


def test_runtime_debug_service_preserves_exact_debug_view_model():
    view_model = _runtime_view()
    gateway = _RuntimeDebugGateway(view_model)
    service = RuntimeServerService(gateway=gateway)

    result = service.runtime_debug_inspection(
        debug_session_id=DEBUG_SESSION_ID,
        port=7787,
        timeout_ms=321,
    )

    assert result == RuntimeDebugInspectionResult(
        schema_version="openhcs.agent.v1",
        connection=ExecutionConnectionSpec(port=7787),
        debug_session_id=DEBUG_SESSION_ID,
        view_model=view_model,
    )
    assert result.view_model is view_model
    assert gateway.requests == [
        (ExecutionConnectionSpec(port=7787), DEBUG_SESSION_ID, 321)
    ]
    assert to_jsonable(result)["view_model"]["sections"][0]["kind"] == (
        "runtime_values"
    )


def test_runtime_debug_service_returns_structured_transport_error():
    gateway = _FailingRuntimeDebugGateway(_runtime_view())
    service = RuntimeServerService(gateway=gateway)

    result = service.runtime_debug_inspection(
        debug_session_id=DEBUG_SESSION_ID,
        port=7787,
    )

    assert result.view_model is None
    assert result.errors[0].code == "runtime_debug_inspection_error"
    assert result.errors[0].message == "debug worker is not paused"
    assert "paused debug_session_id" in result.errors[0].hint


def test_zmq_runtime_gateway_reuses_typed_client_inspection(monkeypatch):
    view_model = _runtime_view()
    client_requests = []
    client_configs = []

    class _Client:
        def __init__(self, **kwargs):
            client_configs.append(kwargs)

        def get_debug_runtime_inspection(self, *, debug_session_id):
            client_requests.append(debug_session_id)
            return view_model

    monkeypatch.setattr(runtime_server_module, "ZMQExecutionClient", _Client)
    gateway = ZMQRuntimeServerGateway()

    result = gateway.runtime_debug_inspection(
        ExecutionConnectionSpec(port=7787),
        DEBUG_SESSION_ID,
        timeout_ms=4321,
    )

    assert result is view_model
    assert client_requests == [DEBUG_SESSION_ID]
    assert client_configs[0]["port"] == 7787
    assert client_configs[0]["config"].control_timeout_ms == 4321


def test_runtime_debug_capability_uses_declared_service_boundary():
    view_model = _runtime_view()
    service = RuntimeServerService(gateway=_RuntimeDebugGateway(view_model))
    declaration = get_agent_capability_declaration(
        "openhcs_inspect_debug_runtime_values"
    )
    request = RuntimeDebugInspectionRequest.from_fields(
        debug_session_id=DEBUG_SESSION_ID,
        port=7787,
    )

    result = declaration.execute_request(
        SimpleNamespace(runtime_server_service=service),
        request,
    )
    capability = get_agent_capability("openhcs_inspect_debug_runtime_values")

    assert result.view_model is view_model
    assert capability.input_contract is RuntimeDebugInspectionRequest
    assert capability.output_contract is RuntimeDebugInspectionResult
    assert capability.cli_command == "runtime-debug-values"
    assert capability.cli_connection_profile is (
        CapabilityCliConnectionProfile.RUNTIME_SERVER
    )
    assert capability.side_effects == ()
    assert capability.runtime_requirements == (
        "running_openhcs_execution_server",
        "paused_debug_session",
    )


def test_runtime_debug_generated_cli_projects_exact_request_fields():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    args = dev_client._build_parser().parse_args(
        (
            "runtime-debug-values",
            "7787",
            DEBUG_SESSION_ID,
            "--host",
            "127.0.0.1",
            "--non-persistent",
            "--timeout-ms",
            "4321",
        )
    )

    call = dev_client._calls_from_args(args)[0]

    assert call.name == "openhcs_inspect_debug_runtime_values"
    assert call.arguments == {
        "host": "127.0.0.1",
        "port": 7787,
        "transport_mode": None,
        "persistent": False,
        "timeout_ms": 4321,
        "debug_session_id": DEBUG_SESSION_ID,
    }
