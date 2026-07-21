"""Focused coverage for persistent MCP dev-client command execution."""

from __future__ import annotations

import io
import json
import subprocess
import sys

import openhcs.mcp.dev_client as dev_client
import pytest


def test_persistent_client_initializes_once_for_distinct_command_specs(
    monkeypatch,
) -> None:
    class FakeMcpDevStdioSession:
        initialize_count = 0
        tool_calls: list[str] = []

        def __init__(self, server_spec, server_stderr) -> None:
            del server_stderr
            self.server_spec = server_spec

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_value, traceback) -> None:
            del exc_type, exc_value, traceback

        async def initialize(self, *, timeout_seconds: float) -> None:
            assert timeout_seconds == 30.0
            type(self).initialize_count += 1

        async def list_tools(self, *, timeout_seconds: float):
            assert timeout_seconds == 5.0
            return (
                {
                    "name": "openhcs_health_check",
                    "description": "Health",
                    "inputSchema": {"type": "object"},
                },
            )

        async def call_tool(
            self,
            name: str,
            arguments,
            *,
            timeout_seconds: float,
        ):
            assert arguments == {}
            assert timeout_seconds == 5.0
            type(self).tool_calls.append(name)
            return {
                "isError": False,
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({"status": "ok"}),
                    }
                ],
            }

    monkeypatch.setattr(
        dev_client,
        "McpDevStdioSession",
        FakeMcpDevStdioSession,
    )

    with dev_client.McpDevClient(sys.executable) as client:
        tools = client.execute(("tools", "--json"))
        health = client.execute(("health", "--json"))

    assert FakeMcpDevStdioSession.initialize_count == 1
    assert FakeMcpDevStdioSession.tool_calls == ["openhcs_health_check"]
    assert tools.returncode == 0
    assert tools.payload["tools"][0]["name"] == "openhcs_health_check"
    assert health.returncode == 0
    assert health.payload["results"][0]["payloads"] == [{"status": "ok"}]
    assert type(dev_client.McpDevCommandSpec.for_name("tools")) is not type(
        dev_client.McpDevCommandSpec.for_name("health")
    )


def test_persistent_client_preserves_local_usage_errors(monkeypatch) -> None:
    class FakeMcpDevStdioSession:
        def __init__(self, server_spec, server_stderr) -> None:
            del server_stderr
            self.server_spec = server_spec

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_value, traceback) -> None:
            del exc_type, exc_value, traceback

        async def initialize(self, *, timeout_seconds: float) -> None:
            del timeout_seconds

    monkeypatch.setattr(
        dev_client,
        "McpDevStdioSession",
        FakeMcpDevStdioSession,
    )

    with dev_client.McpDevClient(sys.executable) as client:
        with pytest.raises(dev_client.McpDevCliUsageError, match="requires a port"):
            client.execute(("viewer-state", "--json"))


def test_persistent_shell_reuses_client_and_preserves_quoted_arguments(
    monkeypatch,
) -> None:
    executions: list[tuple[str, ...]] = []
    construction_count = 0

    class FakePersistentClient:
        def __init__(self, *args, **kwargs) -> None:
            nonlocal construction_count
            del args, kwargs
            construction_count += 1

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            del exc_type, exc_value, traceback

        def execute(self, argv):
            normalized = tuple(argv)
            executions.append(normalized)
            return dev_client.McpDevCommandExecution(
                argv=normalized,
                payload={"ok": True},
                rendered_output=f"rendered:{normalized[-1]}",
                returncode=0,
                server_stderr_tail=None,
            )

    monkeypatch.setattr(dev_client, "McpDevClient", FakePersistentClient)
    parser = dev_client._build_parser()
    args = parser.parse_args(("shell", "--no-prompt"))
    stdin = io.StringIO(
        '# ignored\n\nknowledge-search "source bindings"\nhealth\nquit\n'
    )
    stdout = io.StringIO()
    stderr = io.StringIO()

    returncode = dev_client._run_persistent_shell(
        args,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
    )

    assert returncode == 0
    assert construction_count == 1
    assert executions == [
        ("--timeout-seconds", "5.0", "knowledge-search", "source bindings"),
        ("--timeout-seconds", "5.0", "health"),
    ]
    assert stdout.getvalue().splitlines() == [
        "rendered:source bindings",
        "rendered:health",
    ]
    assert stderr.getvalue() == ""


def test_persistent_shell_aggregates_command_failures(monkeypatch) -> None:
    class FakePersistentClient:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            del exc_type, exc_value, traceback

        def execute(self, argv):
            normalized = tuple(argv)
            return dev_client.McpDevCommandExecution(
                argv=normalized,
                payload={"errors": [{"code": "test"}]},
                rendered_output="failed",
                returncode=1,
                server_stderr_tail=None,
            )

    monkeypatch.setattr(dev_client, "McpDevClient", FakePersistentClient)
    args = dev_client._build_parser().parse_args(
        ("shell", "--command", "health", "--stop-on-error")
    )

    assert (
        dev_client._run_persistent_shell(
            args,
            stdin=io.StringIO(),
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )
        == 1
    )


def test_dev_client_help_does_not_emit_registry_success_logs() -> None:
    completed = subprocess.run(
        (sys.executable, "-m", "openhcs.mcp.dev_client", "--help"),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0
    assert completed.stdout.startswith("usage:")
    assert "shell" in completed.stdout
    assert "metaclass_registry" not in completed.stderr
