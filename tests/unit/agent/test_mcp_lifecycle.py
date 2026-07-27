from __future__ import annotations

import asyncio
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

from openhcs.mcp.bootstrap import (
    MCP_INSTALLATION_POINTER_ENVIRONMENT_VARIABLE,
    MCP_STABLE_LAUNCH_COMMAND_ENVIRONMENT_VARIABLE,
)
from openhcs.mcp.lifecycle import (
    McpInstallationPointerSnapshot,
    McpLifecycleConfigurationError,
    McpProcessLifecycle,
    McpProcessRecoveryReason,
)
import openhcs.mcp.server as server


def _lifecycle(
    *,
    environment: dict[str, str] | None = None,
) -> McpProcessLifecycle:
    return McpProcessLifecycle.from_environment(
        fallback_restart_command=(sys.executable, "-m", "openhcs.mcp"),
        environment={} if environment is None else environment,
    )


def _direct_tool_payload(result) -> dict:
    content = result.content if hasattr(result, "content") else result[0]
    return json.loads(content[0].text)


def test_source_change_requires_client_owned_reconnect():
    lifecycle = _lifecycle()

    current = lifecycle.recovery_status(source_changed=False)
    changed = lifecycle.recovery_status(source_changed=True)

    assert current.reason is McpProcessRecoveryReason.CURRENT
    assert current.restart_required is False
    assert current.restart_command == ()
    assert current.reconnect_owner is None
    assert current.retry_after_reconnect is False

    assert changed.reason is McpProcessRecoveryReason.SOURCE_CHANGED
    assert changed.restart_command == (
        sys.executable,
        "-m",
        "openhcs.mcp",
    )
    assert changed.restart_command_is_stable is False
    assert changed.reconnect_required is True
    assert changed.reconnect_owner.value == "mcp_client"
    assert changed.retry_after_reconnect is True
    assert changed.automatic_recovery_on_reconnect is False
    assert "new MCP initialize handshake" in changed.hint


def test_installer_generation_change_uses_stable_launch_command(tmp_path):
    pointer = tmp_path / "Launch-OpenHCS.ps1"
    pointer.write_text("launch environment-a\n", encoding="utf-8")
    stable_launcher = pointer
    environment = {
        MCP_STABLE_LAUNCH_COMMAND_ENVIRONMENT_VARIABLE: json.dumps(
            [str(stable_launcher), "mcp"]
        ),
        MCP_INSTALLATION_POINTER_ENVIRONMENT_VARIABLE: str(pointer),
    }
    lifecycle = _lifecycle(environment=environment)

    pointer.write_text("launch environment-b\n", encoding="utf-8")
    changed = lifecycle.recovery_status(source_changed=False)

    assert changed.reason is McpProcessRecoveryReason.INSTALLATION_CHANGED
    assert changed.installation_pointer_path == str(pointer)
    assert changed.installation_pointer_changed_since_import is True
    assert changed.installation_pointer_available is True
    assert changed.restart_required is True
    assert changed.restart_command == (str(stable_launcher), "mcp")
    assert changed.restart_command_is_stable is True
    assert changed.automatic_recovery_on_reconnect is True
    assert "stable launcher" in changed.hint


def test_deleted_installation_pointer_fails_closed_without_false_auto_recovery(
    tmp_path,
):
    pointer = tmp_path / "current"
    pointer.symlink_to("environment-a")
    stable_launcher = tmp_path / "launch-openhcs"
    lifecycle = _lifecycle(
        environment={
            MCP_STABLE_LAUNCH_COMMAND_ENVIRONMENT_VARIABLE: json.dumps(
                [str(stable_launcher), "mcp"]
            ),
            MCP_INSTALLATION_POINTER_ENVIRONMENT_VARIABLE: str(pointer),
        }
    )

    pointer.unlink()
    changed = lifecycle.recovery_status(source_changed=False)

    assert changed.reason is McpProcessRecoveryReason.INSTALLATION_CHANGED
    assert changed.installation_pointer_changed_since_import is True
    assert changed.installation_pointer_available is False
    assert changed.restart_command == (str(stable_launcher), "mcp")
    assert changed.restart_command_is_stable is True
    assert changed.automatic_recovery_on_reconnect is False


def test_pointer_snapshot_retries_atomic_file_replacement(monkeypatch, tmp_path):
    pointer = tmp_path / "Launch-OpenHCS.ps1"
    pointer.write_text("launch environment-a\n", encoding="utf-8")
    original_lstat = Path.lstat
    call_count = 0

    def replacing_lstat(path):
        nonlocal call_count
        call_count += 1
        result = original_lstat(path)
        if path == pointer and call_count == 1:
            candidate = pointer.with_suffix(".candidate")
            candidate.write_text("launch environment-b\n", encoding="utf-8")
            os.replace(candidate, pointer)
        return result

    monkeypatch.setattr(Path, "lstat", replacing_lstat)

    snapshot = McpInstallationPointerSnapshot.from_path(pointer)

    assert snapshot.exists is True
    assert snapshot.content_sha256 == sha256(b"launch environment-b\n").hexdigest()
    assert call_count >= 4


def test_pointer_snapshot_survives_deletion_after_lstat(monkeypatch, tmp_path):
    pointer = tmp_path / "Launch-OpenHCS.ps1"
    pointer.write_text("launch environment-a\n", encoding="utf-8")
    original_lstat = Path.lstat
    removed = False

    def deleting_lstat(path):
        nonlocal removed
        result = original_lstat(path)
        if path == pointer and not removed:
            pointer.unlink()
            removed = True
        return result

    monkeypatch.setattr(Path, "lstat", deleting_lstat)

    snapshot = McpInstallationPointerSnapshot.from_path(pointer)

    assert snapshot.exists is False
    assert snapshot.content_sha256 is None


def test_pointer_snapshot_ignores_access_time_only_changes(monkeypatch, tmp_path):
    pointer = tmp_path / "Launch-OpenHCS.ps1"
    pointer.write_text("launch environment-a\n", encoding="utf-8")
    original_lstat = Path.lstat
    touched = False

    def touching_lstat(path):
        nonlocal touched
        result = original_lstat(path)
        if path == pointer and not touched:
            os.utime(
                pointer,
                ns=(result.st_atime_ns + 1_000_000_000, result.st_mtime_ns),
            )
            touched = True
        return result

    monkeypatch.setattr(Path, "lstat", touching_lstat)

    snapshot = McpInstallationPointerSnapshot.from_path(pointer)

    assert snapshot.exists is True
    assert snapshot.content_sha256 == sha256(b"launch environment-a\n").hexdigest()


def test_deleted_environment_source_remains_health_reportable(monkeypatch, tmp_path):
    old_environment_source = tmp_path / "environment-a" / "openhcs" / "server.py"
    old_environment_source.parent.mkdir(parents=True)
    old_environment_source.write_text("source = 'old'\n", encoding="utf-8")
    import_snapshot = server.McpSourceSnapshot.from_path(old_environment_source)
    old_environment_source.unlink()
    monkeypatch.setattr(
        server,
        "MCP_SERVER_IMPORT_SOURCE_SNAPSHOTS",
        {old_environment_source: import_snapshot},
    )
    monkeypatch.setattr(server, "MCP_SERVER_SOURCE_PATH", old_environment_source)

    stale_paths = server._mcp_server_stale_source_paths()
    recovery = server._mcp_server_recovery_status(stale_paths)

    assert stale_paths == (old_environment_source,)
    assert server._mcp_server_current_source_mtime_ns() is None
    assert recovery.reason is McpProcessRecoveryReason.SOURCE_CHANGED
    assert recovery.restart_required is True
    assert recovery.retry_after_reconnect is True


@pytest.mark.parametrize(
    "environment",
    (
        {MCP_STABLE_LAUNCH_COMMAND_ENVIRONMENT_VARIABLE: "not-json"},
        {MCP_STABLE_LAUNCH_COMMAND_ENVIRONMENT_VARIABLE: "[]"},
        {MCP_STABLE_LAUNCH_COMMAND_ENVIRONMENT_VARIABLE: '["relative"]'},
        {MCP_INSTALLATION_POINTER_ENVIRONMENT_VARIABLE: "relative"},
    ),
)
def test_invalid_launcher_lifecycle_contract_fails_before_server_construction(
    environment,
):
    with pytest.raises(McpLifecycleConfigurationError):
        _lifecycle(environment=environment)


def test_real_stdio_session_blocks_changed_installation_generation(tmp_path):
    if importlib.util.find_spec("mcp") is None:
        return

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    pointer = tmp_path / "Launch-OpenHCS.ps1"
    pointer.write_text("launch environment-a\n", encoding="utf-8")
    stable_launcher = pointer

    async def call_stdio_server():
        parameters = StdioServerParameters(
            command=sys.executable,
            args=("-m", "openhcs.mcp", "--surface", "core"),
            env={
                **os.environ,
                MCP_STABLE_LAUNCH_COMMAND_ENVIRONMENT_VARIABLE: json.dumps(
                    [str(stable_launcher), "mcp"]
                ),
                MCP_INSTALLATION_POINTER_ENVIRONMENT_VARIABLE: str(pointer),
            },
        )
        async with stdio_client(parameters) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await asyncio.wait_for(session.initialize(), timeout=10)
                initial_health = await asyncio.wait_for(
                    session.call_tool("openhcs_health_check", {}),
                    timeout=10,
                )
                pointer.write_text("launch environment-b\n", encoding="utf-8")
                blocked = await asyncio.wait_for(
                    session.call_tool("openhcs_list_capabilities", {}),
                    timeout=10,
                )
                changed_health = await asyncio.wait_for(
                    session.call_tool("openhcs_health_check", {}),
                    timeout=10,
                )
                return initial_health, blocked, changed_health

    initial_result, blocked_result, health_result = asyncio.run(call_stdio_server())
    initial = _direct_tool_payload(initial_result)
    blocked = _direct_tool_payload(blocked_result)
    health = _direct_tool_payload(health_result)

    assert initial["recovery_reason"] == "current"
    assert initial["restart_required"] is False
    assert blocked["errors"][0]["code"] == "mcp_server_stale"
    assert blocked["errors"][0]["path"] == str(pointer)
    assert blocked["stale_source_paths"] == []
    assert blocked["recovery_reason"] == "installation_changed"
    assert blocked["restart_command"] == [str(stable_launcher), "mcp"]
    assert blocked["restart_command_is_stable"] is True
    assert blocked["reconnect_required"] is True
    assert blocked["reconnect_owner"] == "mcp_client"
    assert blocked["retry_after_reconnect"] is True
    assert blocked["automatic_recovery_on_reconnect"] is True
    assert health["server_source_changed_since_import"] is False
    assert health["installation_pointer_changed_since_import"] is True
    assert health["restart_required"] is True


def test_real_stdio_session_reports_deleted_environment_source(tmp_path):
    if importlib.util.find_spec("mcp") is None:
        return

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    watched_source = tmp_path / "environment-a" / "openhcs" / "server.py"
    watched_source.parent.mkdir(parents=True)
    watched_source.write_text("source = 'old'\n", encoding="utf-8")
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(
        "\n".join(
            (
                "import os",
                "from pathlib import Path",
                "import openhcs.mcp.server as server",
                "watched = Path(os.environ['OPENHCS_TEST_WATCHED_SOURCE'])",
                "snapshot = server.McpSourceSnapshot.from_path(watched)",
                "server.MCP_SERVER_IMPORT_SOURCE_SNAPSHOTS = {watched: snapshot}",
                "server.MCP_SERVER_SOURCE_PATH = watched",
                "server.MCP_SERVER_IMPORT_MTIME_NS = snapshot.mtime_ns",
            )
        ),
        encoding="utf-8",
    )

    async def call_stdio_server():
        current_pythonpath = os.environ.get("PYTHONPATH")
        pythonpath = (
            str(tmp_path)
            if current_pythonpath is None
            else os.pathsep.join((str(tmp_path), current_pythonpath))
        )
        parameters = StdioServerParameters(
            command=sys.executable,
            args=("-m", "openhcs.mcp", "--surface", "core"),
            env={
                **os.environ,
                "PYTHONPATH": pythonpath,
                "OPENHCS_TEST_WATCHED_SOURCE": str(watched_source),
            },
        )
        async with stdio_client(parameters) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await asyncio.wait_for(session.initialize(), timeout=10)
                watched_source.unlink()
                blocked = await asyncio.wait_for(
                    session.call_tool("openhcs_list_capabilities", {}),
                    timeout=10,
                )
                health = await asyncio.wait_for(
                    session.call_tool("openhcs_health_check", {}),
                    timeout=10,
                )
                return blocked, health

    blocked_result, health_result = asyncio.run(call_stdio_server())
    blocked = _direct_tool_payload(blocked_result)
    health = _direct_tool_payload(health_result)

    assert blocked["errors"][0]["code"] == "mcp_server_stale"
    assert blocked["recovery_reason"] == "source_changed"
    assert blocked["stale_source_paths"] == [str(watched_source)]
    assert blocked["restart_command"][-2:] == ["-m", "openhcs.mcp"]
    assert blocked["restart_command_is_stable"] is False
    assert blocked["reconnect_owner"] == "mcp_client"
    assert blocked["retry_after_reconnect"] is True
    assert health["server_source_changed_since_import"] is True
    assert health["server_current_mtime_ns"] is None
    assert health["restart_required"] is True
