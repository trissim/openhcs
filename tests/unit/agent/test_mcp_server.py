import asyncio
import importlib.util
import inspect
import json

from openhcs.agent.capabilities import get_capability_registry

import openhcs.mcp.server as server


def test_mcp_server_module_import_does_not_require_mcp_dependency():
    assert callable(server.build_server)


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
    assert "openhcs_probe_viewer_window" in tool_names
    assert "openhcs_validate_viewer_window_state" in tool_names


def test_viewer_capabilities_advertise_payload_coordinate_validation():
    capabilities = {
        capability.name: capability
        for capability in get_capability_registry().capabilities
    }

    state_capability = capabilities["openhcs_get_viewer_window_state"]
    validation_capability = capabilities["openhcs_validate_viewer_window_state"]

    assert "viewer_payload_summaries" in state_capability.data_exposure
    assert "viewer_shape_bounds" in state_capability.data_exposure
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
