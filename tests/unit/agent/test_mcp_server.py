import asyncio
import importlib.util
import inspect

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
    assert "openhcs_submit_compile" in tool_names
    assert "openhcs_submit_pipeline_execution" in tool_names
    assert "openhcs_get_execution_status" in tool_names


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
    assert "openhcs_ui_restore_snapshot" in tool_names
    assert "openhcs_ui_get_operation_status" in tool_names
