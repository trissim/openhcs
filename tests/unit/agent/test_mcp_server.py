import asyncio
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import tomllib

import pytest

from openhcs.agent.capabilities import get_capability_registry

import openhcs.mcp.bootstrap as bootstrap
import openhcs.mcp.server as server


def test_mcp_server_module_import_does_not_require_mcp_dependency():
    assert callable(server.build_server)
    assert "main" not in vars(server)


def test_installed_mcp_script_uses_fail_soft_bootstrap_entrypoint():
    with (Path(__file__).resolve().parents[3] / "pyproject.toml").open("rb") as fh:
        pyproject = tomllib.load(fh)

    assert pyproject["project"]["scripts"]["openhcs-mcp"] == (
        "openhcs.mcp.bootstrap:main"
    )
    assert pyproject["project"]["scripts"]["openhcs-mcp-dev"] == (
        "openhcs.mcp.dev_client:main"
    )


def test_development_extra_installs_mcp_dependency():
    with (Path(__file__).resolve().parents[3] / "pyproject.toml").open("rb") as fh:
        pyproject = tomllib.load(fh)

    assert "mcp>=1.0" in pyproject["project"]["optional-dependencies"]["dev"]


def test_agent_dto_package_exports_mcp_debugging_contracts():
    import openhcs.agent.dto as dto

    exported_names = (
        "McpServerHealthResult",
        "UiSelectedPlateWorkflowKind",
        "UiSelectedPlateWorkflowRequest",
        "UiSelectedPlateWorkflowResult",
        "UiWidgetRect",
        "UiWidgetTreeNode",
        "UiWidgetTreeRequest",
        "UiWidgetTreeResult",
        "ViewerWindowLayerPayloads",
        "ViewerWindowPayloadRecord",
        "ViewerWindowPayloadRequest",
        "ViewerWindowPayloadResult",
    )

    for exported_name in exported_names:
        assert exported_name in vars(dto)


def test_mcp_server_builds_when_optional_dependency_is_installed():
    if importlib.util.find_spec("mcp") is None:
        return

    built = server.build_server()

    assert built is not None


def test_mcp_tools_have_blind_agent_descriptions():
    if importlib.util.find_spec("mcp") is None:
        return

    built = server.build_server()

    listed_tools = built.list_tools()
    if inspect.isawaitable(listed_tools):
        tools = asyncio.run(listed_tools)
    else:
        tools = listed_tools

    assert tools
    assert all(tool.description for tool in tools)


def test_mcp_tool_descriptions_expose_debugging_result_contracts():
    if importlib.util.find_spec("mcp") is None:
        return

    built = server.build_server()

    listed_tools = built.list_tools()
    if inspect.isawaitable(listed_tools):
        tools = asyncio.run(listed_tools)
    else:
        tools = listed_tools

    descriptions = {tool.name: tool.description for tool in tools}

    assert "per-axis" in descriptions["openhcs_get_viewer_window_payloads"]
    assert "arrays and shapes" in descriptions["openhcs_get_viewer_window_payloads"]
    assert "revision" in descriptions["openhcs_ui_apply_code_document"]
    assert "snapshot" in descriptions["openhcs_ui_apply_code_document"]
    assert "undo" in descriptions["openhcs_ui_apply_code_document"]
    assert "clickable geometry" in descriptions["openhcs_ui_get_widget_tree"]
    assert "action kinds" in descriptions["openhcs_ui_get_widget_tree"]


def test_selected_plate_workflow_tool_schema_exposes_workflow_enum():
    if importlib.util.find_spec("mcp") is None:
        return

    built = server.build_server()

    listed_tools = built.list_tools()
    if inspect.isawaitable(listed_tools):
        tools = asyncio.run(listed_tools)
    else:
        tools = listed_tools

    schemas = {tool.name: tool.inputSchema for tool in tools}
    schema = schemas["openhcs_ui_selected_plate_workflow"]

    assert schema["$defs"]["UiSelectedPlateWorkflowKind"]["enum"] == [
        "init_plate",
        "compile_plate",
        "run_plate",
    ]
    assert schema["properties"]["workflow"] == {
        "$ref": "#/$defs/UiSelectedPlateWorkflowKind"
    }


def test_selected_plate_workflow_rejects_unknown_workflow_before_dispatch():
    if importlib.util.find_spec("mcp") is None:
        return

    from mcp.server.fastmcp.exceptions import ToolError

    async def call_unknown_workflow():
        built = server.build_server()
        return await asyncio.wait_for(
            built.call_tool(
                "openhcs_ui_selected_plate_workflow",
                {"workflow": "not_a_workflow"},
            ),
            timeout=2,
        )

    with pytest.raises(ToolError, match="not_a_workflow"):
        asyncio.run(call_unknown_workflow())


def test_mcp_health_check_tool_returns_promptly():
    if importlib.util.find_spec("mcp") is None:
        return

    async def call_health_check():
        built = server.build_server()
        return await asyncio.wait_for(
            built.call_tool("openhcs_health_check", {}),
            timeout=2,
        )

    result = asyncio.run(call_health_check())
    payload = json.loads(result[0].text)

    assert payload["status"] == "ok"
    assert payload["service"] == "openhcs.mcp"
    assert isinstance(payload["server_process_id"], int)
    assert isinstance(payload["started_at_unix"], float)
    assert payload["server_source_path"].endswith("openhcs/mcp/server.py")
    assert isinstance(payload["server_import_mtime_ns"], int)
    assert isinstance(payload["server_current_mtime_ns"], int)
    assert payload["server_source_changed_since_import"] is False
    assert payload["stale_source_paths"] == []


def test_mcp_stale_watchlist_includes_agent_contract_sources():
    watched_paths = {
        source_path.as_posix()
        for source_path in server.MCP_SERVER_SOURCE_PATHS
    }

    assert any(path.endswith("openhcs/mcp/server.py") for path in watched_paths)
    assert any(path.endswith("openhcs/mcp/context.py") for path in watched_paths)
    assert any(path.endswith("openhcs/agent/capabilities.py") for path in watched_paths)
    assert any(path.endswith("openhcs/agent/path_policy.py") for path in watched_paths)
    assert any(path.endswith("openhcs/agent/dto/mcp.py") for path in watched_paths)
    assert any(path.endswith("openhcs/agent/dto/ui_bridge.py") for path in watched_paths)
    assert any(path.endswith("openhcs/agent/dto/viewer.py") for path in watched_paths)
    assert any(
        path.endswith("openhcs/agent/services/execution_session_service.py")
        for path in watched_paths
    )
    assert any(
        path.endswith("openhcs/agent/services/ui_bridge_service.py")
        for path in watched_paths
    )
    assert any(
        path.endswith("openhcs/agent/services/ui_bridge_transport.py")
        for path in watched_paths
    )
    assert any(
        path.endswith("openhcs/agent/services/viewer_window_service.py")
        for path in watched_paths
    )
    assert any(
        path.endswith("openhcs/agent/services/runtime_server_service.py")
        for path in watched_paths
    )
    assert any(path.endswith("openhcs/runtime/viewer_protocol.py") for path in watched_paths)
    assert any(path.endswith("openhcs/runtime/window_snapshot.py") for path in watched_paths)


def test_mcp_tools_fail_fast_when_server_source_is_stale(monkeypatch):
    if importlib.util.find_spec("mcp") is None:
        return

    monkeypatch.setattr(
        server,
        "_mcp_server_stale_source_paths",
        lambda: (server.MCP_SERVER_SOURCE_PATH,),
    )

    async def call_stale_server():
        built = server.build_server()
        blocked_tool = await asyncio.wait_for(
            built.call_tool("openhcs_list_capabilities", {}),
            timeout=2,
        )
        health_tool = await asyncio.wait_for(
            built.call_tool("openhcs_health_check", {}),
            timeout=2,
        )
        return blocked_tool, health_tool

    blocked_result, health_result = asyncio.run(call_stale_server())
    blocked_payload = json.loads(blocked_result[0].text)
    health_payload = json.loads(health_result[0].text)

    assert blocked_payload["schema_version"] == "openhcs.agent.v1"
    assert blocked_payload["ok"] is False
    assert blocked_payload["tool"] == "openhcs_list_capabilities"
    assert blocked_payload["errors"][0]["code"] == "mcp_server_stale"
    assert blocked_payload["errors"][0]["path"].endswith("openhcs/mcp/server.py")

    assert health_payload["status"] == "ok"
    assert health_payload["server_source_changed_since_import"] is True
    assert health_payload["stale_source_paths"][0].endswith("openhcs/mcp/server.py")


def test_mcp_tools_fail_fast_when_agent_source_is_stale(monkeypatch):
    if importlib.util.find_spec("mcp") is None:
        return

    stale_agent_path = next(
        source_path
        for source_path in server.MCP_SERVER_SOURCE_PATHS
        if source_path.name == "capabilities.py"
    )
    monkeypatch.setattr(
        server,
        "_mcp_server_stale_source_paths",
        lambda: (stale_agent_path,),
    )

    async def call_stale_server():
        built = server.build_server()
        blocked_tool = await asyncio.wait_for(
            built.call_tool("openhcs_search_functions", {}),
            timeout=2,
        )
        health_tool = await asyncio.wait_for(
            built.call_tool("openhcs_health_check", {}),
            timeout=2,
        )
        return blocked_tool, health_tool

    blocked_result, health_result = asyncio.run(call_stale_server())
    blocked_payload = json.loads(blocked_result[0].text)
    health_payload = json.loads(health_result[0].text)

    assert blocked_payload["schema_version"] == "openhcs.agent.v1"
    assert blocked_payload["ok"] is False
    assert blocked_payload["tool"] == "openhcs_search_functions"
    assert blocked_payload["errors"][0]["code"] == "mcp_server_stale"
    assert blocked_payload["errors"][0]["path"].endswith(
        "openhcs/agent/capabilities.py"
    )
    assert health_payload["server_source_changed_since_import"] is True
    assert health_payload["stale_source_paths"][0].endswith(
        "openhcs/agent/capabilities.py"
    )


def test_mcp_stdio_server_roundtrip_returns_errors_as_payloads():
    if importlib.util.find_spec("mcp") is None:
        return

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def call_stdio_server():
        parameters = StdioServerParameters(
            command=sys.executable,
            args=("-m", "openhcs.mcp"),
        )
        async with stdio_client(parameters) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await asyncio.wait_for(session.initialize(), timeout=5)
                health = await asyncio.wait_for(
                    session.call_tool("openhcs_health_check", {}),
                    timeout=5,
                )
                bad_viewer_call = await asyncio.wait_for(
                    session.call_tool(
                        "openhcs_probe_viewer_window",
                        {"port": 1, "timeout_ms": 120_000},
                    ),
                    timeout=5,
                )
                return health, bad_viewer_call

    health, bad_viewer_call = asyncio.run(call_stdio_server())
    health_payload = json.loads(health.content[0].text)
    bad_payload = json.loads(bad_viewer_call.content[0].text)

    assert health_payload["status"] == "ok"
    assert health_payload["server_source_path"].endswith("openhcs/mcp/server.py")
    assert isinstance(health_payload["server_process_id"], int)
    assert isinstance(health_payload["started_at_unix"], float)
    assert isinstance(health_payload["server_import_mtime_ns"], int)
    assert isinstance(health_payload["server_current_mtime_ns"], int)
    assert health_payload["server_source_changed_since_import"] is False
    assert bad_payload["ok"] is False
    assert bad_payload["tool"] == "openhcs_probe_viewer_window"
    assert bad_payload["errors"][0]["code"] == "mcp_tool_failed"


def test_mcp_stdio_validation_error_keeps_session_alive():
    if importlib.util.find_spec("mcp") is None:
        return

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def call_stdio_server():
        parameters = StdioServerParameters(
            command=sys.executable,
            args=("-m", "openhcs.mcp"),
        )
        async with stdio_client(parameters) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await asyncio.wait_for(session.initialize(), timeout=5)
                invalid_workflow = await asyncio.wait_for(
                    session.call_tool(
                        "openhcs_ui_selected_plate_workflow",
                        {"workflow": "not_a_workflow"},
                    ),
                    timeout=5,
                )
                health = await asyncio.wait_for(
                    session.call_tool("openhcs_health_check", {}),
                    timeout=5,
                )
                return invalid_workflow, health

    invalid_workflow, health = asyncio.run(call_stdio_server())
    invalid_text = invalid_workflow.content[0].text
    health_payload = json.loads(health.content[0].text)

    assert invalid_workflow.isError is True
    assert "not_a_workflow" in invalid_text
    assert "init_plate" in invalid_text
    assert health_payload["status"] == "ok"


def test_mcp_dev_client_rejects_non_object_tool_arguments():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    assert dev_client.parse_json_object('{"port": 5565}') == {"port": 5565}

    with pytest.raises(ValueError, match="JSON object"):
        dev_client.parse_json_object("[5565]")


def test_mcp_dev_client_accepts_common_flags_after_subcommands():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    parser = dev_client._build_parser()

    after_command = parser.parse_args(("ui-smoke", "--allow-error-payloads"))
    before_command = parser.parse_args(("--allow-error-payloads", "ui-smoke"))

    assert after_command.allow_error_payloads is True
    assert before_command.allow_error_payloads is True


def test_mcp_dev_client_launches_fresh_current_source_server():
    if importlib.util.find_spec("mcp") is None:
        return

    import openhcs.mcp.dev_client as dev_client

    async def call_health_through_dev_client():
        return await dev_client.call_fresh_mcp_server(
            dev_client.McpDevServerSpec(sys.executable),
            (dev_client.McpDevToolCall("openhcs_health_check", {}),),
            timeout_seconds=5,
        )

    payload = asyncio.run(call_health_through_dev_client())
    result = payload["results"][0]
    health_payload = result["payloads"][0]

    assert result["tool"] == "openhcs_health_check"
    assert result["mcp_error"] is False
    assert health_payload["status"] == "ok"
    assert health_payload["server_source_changed_since_import"] is False


def test_mcp_stdio_bootstrap_failure_keeps_transport_open(tmp_path):
    if importlib.util.find_spec("mcp") is None:
        return

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    (tmp_path / "sitecustomize.py").write_text(
        "\n".join(
            (
                "import openhcs.mcp.server as openhcs_mcp_server",
                "",
                "def fail_build_server():",
                "    raise RuntimeError('stdio construction failed')",
                "",
                "openhcs_mcp_server.build_server = fail_build_server",
            )
        )
    )
    pythonpath_parts = [str(tmp_path)]
    current_pythonpath = os.environ.get("PYTHONPATH")
    if current_pythonpath is not None:
        pythonpath_parts.append(current_pythonpath)

    async def call_stdio_server():
        parameters = StdioServerParameters(
            command=sys.executable,
            args=("-m", "openhcs.mcp"),
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join(pythonpath_parts),
            },
        )
        async with stdio_client(parameters) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await asyncio.wait_for(session.initialize(), timeout=5)
                health = await asyncio.wait_for(
                    session.call_tool("openhcs_health_check", {}),
                    timeout=5,
                )
                failure = await asyncio.wait_for(
                    session.call_tool("openhcs_bootstrap_failure", {}),
                    timeout=5,
                )
                return health, failure

    health, failure = asyncio.run(call_stdio_server())
    health_payload = json.loads(health.content[0].text)
    failure_payload = json.loads(failure.content[0].text)

    assert health_payload["schema_version"] == "openhcs.mcp.bootstrap.v1"
    assert health_payload["ok"] is False
    assert health_payload["status"] == "unavailable"
    assert health_payload["phase"] == "build_server"
    assert health_payload["message"] == "stdio construction failed"
    assert failure_payload == health_payload


def test_mcp_bootstrap_failure_server_reports_startup_exception():
    if importlib.util.find_spec("mcp") is None:
        return

    async def call_bootstrap_failure_tool():
        built = bootstrap.build_bootstrap_failure_server(RuntimeError("startup failed"))
        return await asyncio.wait_for(
            built.call_tool("openhcs_health_check", {}),
            timeout=2,
        )

    result = asyncio.run(call_bootstrap_failure_tool())
    payload = json.loads(result[0][0].text)

    assert payload["schema_version"] == "openhcs.mcp.bootstrap.v1"
    assert payload["ok"] is False
    assert payload["status"] == "unavailable"
    assert payload["service"] == "openhcs.mcp"
    assert payload["phase"] == "build_server"
    assert payload["exception_type"] == "RuntimeError"
    assert payload["message"] == "startup failed"


def test_mcp_bootstrap_wraps_server_construction_failure(monkeypatch):
    if importlib.util.find_spec("mcp") is None:
        return

    def fail_build_server():
        raise RuntimeError("construction failed")

    monkeypatch.setattr(server, "build_server", fail_build_server)

    async def call_bootstrap_failure_tool():
        built = bootstrap.build_bootstrapped_server()
        return await asyncio.wait_for(
            built.call_tool("openhcs_bootstrap_failure", {}),
            timeout=2,
        )

    result = asyncio.run(call_bootstrap_failure_tool())
    payload = json.loads(result[0][0].text)

    assert payload["schema_version"] == "openhcs.mcp.bootstrap.v1"
    assert payload["ok"] is False
    assert payload["status"] == "unavailable"
    assert payload["service"] == "openhcs.mcp"
    assert payload["phase"] == "build_server"
    assert payload["exception_type"] == "RuntimeError"
    assert payload["message"] == "construction failed"


def test_mcp_bootstrap_wraps_server_run_failure(monkeypatch):
    phases: list[bootstrap.McpBootstrapFailurePhase] = []
    messages: list[str] = []
    failure_server_runs: list[bool] = []

    class FailingRunServer:
        def run(self) -> None:
            raise RuntimeError("run failed")

    class FailureReportServer:
        def run(self) -> None:
            failure_server_runs.append(True)

    def build_failure_server(
        exception: BaseException,
        phase: bootstrap.McpBootstrapFailurePhase = (
            bootstrap.McpBootstrapFailurePhase.BUILD_SERVER
        ),
    ) -> FailureReportServer:
        phases.append(phase)
        messages.append(str(exception))
        return FailureReportServer()

    monkeypatch.setattr(
        bootstrap,
        "build_bootstrapped_server",
        FailingRunServer,
    )
    monkeypatch.setattr(
        bootstrap,
        "build_bootstrap_failure_server",
        build_failure_server,
    )

    bootstrap.run_bootstrapped_server()

    assert phases == [bootstrap.McpBootstrapFailurePhase.RUN_SERVER]
    assert messages == ["run failed"]
    assert failure_server_runs == [True]


def test_mcp_tool_adapter_returns_error_payload_instead_of_raising():
    if importlib.util.find_spec("mcp") is None:
        return

    async def call_bad_tools():
        built = server.build_server()
        calls = (
            (
                "openhcs_ui_bridge_status",
                {"connection": {"timeout_ms": 120_000}},
            ),
            (
                "openhcs_probe_viewer_window",
                {"port": 1, "timeout_ms": 120_000},
            ),
        )
        results = []
        for tool_name, arguments in calls:
            result = await asyncio.wait_for(
                built.call_tool(tool_name, arguments),
                timeout=2,
            )
            results.append((tool_name, result))
        return tuple(results)

    results = asyncio.run(call_bad_tools())

    for tool_name, result in results:
        payload = json.loads(result[0].text)
        assert payload["schema_version"] == "openhcs.agent.v1"
        assert payload["ok"] is False
        assert payload["tool"] == tool_name
        assert payload["errors"][0]["code"] == "mcp_tool_failed"
        assert payload["errors"][0]["exception_type"] == "ValueError"


def test_mcp_server_exposes_execution_session_tools():
    if importlib.util.find_spec("mcp") is None:
        return

    built = server.build_server()

    listed_tools = built.list_tools()
    if inspect.isawaitable(listed_tools):
        tools = asyncio.run(listed_tools)
    else:
        tools = listed_tools
    tool_names = {tool.name for tool in tools}

    assert "openhcs_create_orchestrator_session" in tool_names
    assert "openhcs_create_orchestrator_session_from_pipeline_source" in tool_names
    assert "openhcs_inspect_pipeline_source_artifact_plan" in tool_names
    assert "openhcs_submit_compile" in tool_names
    assert "openhcs_submit_pipeline_execution" in tool_names
    assert "openhcs_get_execution_status" in tool_names
    assert "openhcs_viewer_snapshot_window" in tool_names
    assert "openhcs_get_viewer_window_state" in tool_names
    assert "openhcs_get_viewer_window_payloads" in tool_names
    assert "openhcs_probe_viewer_window" in tool_names
    assert "openhcs_validate_viewer_window_state" in tool_names


def test_viewer_capabilities_advertise_payload_coordinate_validation():
    capabilities = {
        capability.name: capability
        for capability in get_capability_registry().capabilities
    }

    state_capability = capabilities["openhcs_get_viewer_window_state"]
    payload_capability = capabilities["openhcs_get_viewer_window_payloads"]
    validation_capability = capabilities["openhcs_validate_viewer_window_state"]

    assert "viewer_payload_summaries" in state_capability.data_exposure
    assert "viewer_shape_bounds" in state_capability.data_exposure
    assert "per-axis image and shape payload records" in payload_capability.description
    assert "viewer_payload_records" in payload_capability.data_exposure
    assert payload_capability.output_type == "ViewerWindowPayloadResult"
    assert "viewer_coordinate_coverage" in validation_capability.data_exposure
    assert (
        "viewer_payload_spatial_compatibility"
        in validation_capability.data_exposure
    )
    assert "routed coordinate coverage" in validation_capability.description


def test_mcp_server_exposes_ui_bridge_tools():
    if importlib.util.find_spec("mcp") is None:
        return

    built = server.build_server()

    listed_tools = built.list_tools()
    if inspect.isawaitable(listed_tools):
        tools = asyncio.run(listed_tools)
    else:
        tools = listed_tools
    tool_names = {tool.name for tool in tools}

    assert "openhcs_ui_bridge_status" in tool_names
    assert "openhcs_ui_list_state_surfaces" in tool_names
    assert "openhcs_ui_get_state_surface" in tool_names
    assert "openhcs_ui_get_code_document" in tool_names
    assert "openhcs_ui_apply_code_document" in tool_names
    assert "openhcs_ui_close_window" in tool_names
    assert "openhcs_ui_snapshot_window" in tool_names
    assert "openhcs_ui_get_widget_tree" in tool_names
    assert "openhcs_ui_selected_plate_workflow" in tool_names
    assert "openhcs_ui_restore_snapshot" in tool_names
    assert "openhcs_ui_get_operation_status" in tool_names


def test_mcp_ui_bridge_timeout_policy_is_fail_fast():
    assert server.McpUiBridgeTimeoutPolicy.resolve(None) == 750
    assert server.McpUiBridgeTimeoutPolicy.resolve(2000) == 2000

    try:
        server.McpUiBridgeTimeoutPolicy.resolve(120_000)
    except ValueError as exc:
        assert "must not exceed" in str(exc)
    else:
        raise AssertionError("large UI bridge MCP timeout was accepted")


def test_mcp_viewer_timeout_policy_is_fail_fast():
    assert server.McpViewerTimeoutPolicy.resolve(None) == 750
    assert server.McpViewerTimeoutPolicy.resolve(2000) == 2000

    try:
        server.McpViewerTimeoutPolicy.resolve(120_000)
    except ValueError as exc:
        assert "must not exceed" in str(exc)
    else:
        raise AssertionError("large viewer MCP timeout was accepted")
