"""Focused coverage for persistent MCP dev-client command execution."""

from __future__ import annotations

import asyncio
import io
import json
import subprocess
import sys

import pytest

import openhcs.mcp.dev_client as dev_client


def test_multi_call_command_honors_its_declared_timeout_floor() -> None:
    parser = dev_client._build_parser()
    default_args = parser.parse_args(
        (
            "execute-source",
            "/tmp/plate",
            "--source-text",
            "pipeline_config = None\npipeline_steps = []",
        )
    )
    extended_args = parser.parse_args(
        (
            "--timeout-seconds",
            "180",
            "execute-source",
            "/tmp/plate",
            "--source-text",
            "pipeline_config = None\npipeline_steps = []",
        )
    )
    command = dev_client.McpDevCommandSpec.for_name("execute-source")

    assert default_args.timeout_seconds == dev_client.DEFAULT_CALL_TIMEOUT_SECONDS
    assert command.default_timeout_seconds == 120.0
    assert command.timeout_seconds(default_args) == 120.0
    assert command.timeout_seconds(extended_args) == 180.0


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


def test_persistent_client_does_not_close_caller_owned_server_stderr() -> None:
    server_stderr = io.StringIO()
    client = dev_client.McpDevClient(
        sys.executable,
        server_stderr=server_stderr,
    )

    client.close()

    assert server_stderr.closed is False


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


def test_persistent_client_timeout_is_transport_inactivity_not_total_duration(
    monkeypatch,
) -> None:
    class ProgressAwareFakeMcpDevStdioSession:
        def __init__(self, server_spec, server_stderr) -> None:
            del server_stderr
            self.server_spec = server_spec

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_value, traceback) -> None:
            del exc_type, exc_value, traceback

        async def initialize(self, *, timeout_seconds: float) -> None:
            del timeout_seconds

        async def call_tool(
            self,
            name: str,
            arguments,
            *,
            timeout_seconds: float,
        ):
            assert name == "openhcs_health_check"
            assert arguments == {}
            assert timeout_seconds == 0.02
            for _ in range(3):
                await asyncio.sleep(0.01)
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
        ProgressAwareFakeMcpDevStdioSession,
    )
    monkeypatch.setattr(
        type(dev_client.McpDevCommandSpec.for_name("health")),
        "default_timeout_seconds",
        0.0,
    )

    with dev_client.McpDevClient(sys.executable) as client:
        execution = client.execute(
            ("health", "--json"),
            timeout_seconds=0.02,
        )

    assert execution.returncode == 0
    assert execution.payload["results"][0]["payloads"] == [{"status": "ok"}]


def test_stdio_tool_call_requests_and_consumes_progress_notifications(monkeypatch):
    session = dev_client.McpDevStdioSession(
        dev_client.McpDevServerSpec(sys.executable),
        io.StringIO(),
    )
    written_messages = []
    read_timeouts = []
    responses = iter(
        (
            {
                "jsonrpc": "2.0",
                "method": "notifications/progress",
                "params": {
                    "progressToken": 1,
                    "progress": 10.0,
                    "message": "still running",
                },
            },
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"content": []},
            },
        )
    )

    async def fake_write_message(message):
        written_messages.append(message)

    async def fake_read_message(*, timeout_seconds):
        read_timeouts.append(timeout_seconds)
        return next(responses)

    monkeypatch.setattr(session, "write_message", fake_write_message)
    monkeypatch.setattr(session, "read_message", fake_read_message)

    result = asyncio.run(
        session.call_tool("openhcs_slow_tool", {}, timeout_seconds=7.0)
    )

    assert result == {"content": []}
    assert written_messages[0]["params"]["_meta"] == {"progressToken": 1}
    assert read_timeouts == [7.0, 7.0]
    assert "MCP progress: progress=10.0 message='still running'" in (
        session.server_stderr.getvalue()
    )


def test_stdio_tool_call_progress_renews_inactivity_timeout(monkeypatch) -> None:
    session = dev_client.McpDevStdioSession(
        dev_client.McpDevServerSpec(sys.executable),
        io.StringIO(),
    )
    responses = iter(
        {
            "jsonrpc": "2.0",
            "method": "notifications/progress",
            "params": {"progressToken": 1, "progress": progress},
        }
        for progress in (10.0, 20.0, 30.0)
    )
    final_response = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"content": []},
    }

    async def fake_write_message(message):
        del message

    async def fake_read_message(*, timeout_seconds):
        await asyncio.wait_for(asyncio.sleep(0.01), timeout=timeout_seconds)
        return next(responses, final_response)

    monkeypatch.setattr(session, "write_message", fake_write_message)
    monkeypatch.setattr(session, "read_message", fake_read_message)

    result = asyncio.run(
        session.call_tool("openhcs_slow_tool", {}, timeout_seconds=0.02)
    )

    assert result == {"content": []}


def test_stdio_session_times_out_when_no_message_activity_arrives() -> None:
    session = dev_client.McpDevStdioSession(
        dev_client.McpDevServerSpec(sys.executable),
        io.StringIO(),
    )

    async def read_without_activity() -> None:
        stdout = asyncio.StreamReader()

        class FakeProcess:
            pass

        process = FakeProcess()
        process.stdout = stdout
        session.process = process
        await session.read_message(timeout_seconds=0.01)

    with pytest.raises(TimeoutError):
        asyncio.run(read_without_activity())


def test_stdio_session_reads_json_message_larger_than_stream_separator_limit() -> None:
    session = dev_client.McpDevStdioSession(
        dev_client.McpDevServerSpec(sys.executable),
        io.StringIO(),
    )
    large_text = "x" * (8 * 1024 * 1024)
    first_message = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"content": [{"type": "text", "text": large_text}]},
    }
    second_message = {
        "jsonrpc": "2.0",
        "id": 2,
        "result": {"content": []},
    }

    async def read_messages():
        stdout = asyncio.StreamReader()
        stdout.feed_data(
            json.dumps(first_message).encode("utf-8")
            + b"\n"
            + json.dumps(second_message).encode("utf-8")
            + b"\n"
        )
        stdout.feed_eof()

        class FakeProcess:
            pass

        process = FakeProcess()
        process.stdout = stdout
        session.process = process
        return (
            await session.read_message(timeout_seconds=1.0),
            await session.read_message(timeout_seconds=1.0),
        )

    first, second = asyncio.run(read_messages())

    assert first["result"]["content"][0]["text"] == large_text
    assert second == second_message


def test_stdio_session_bounds_stdin_pipe_close(monkeypatch) -> None:
    session = dev_client.McpDevStdioSession(
        dev_client.McpDevServerSpec(sys.executable),
        io.StringIO(),
    )
    observed: list[str] = []

    class BlockingStdin:
        def close(self) -> None:
            observed.append("stdin.close")

        async def wait_closed(self) -> None:
            await asyncio.Event().wait()

    class FakeProcess:
        stdin = BlockingStdin()
        returncode = None

        def terminate(self) -> None:
            observed.append("process.terminate")
            self.returncode = 0

        async def wait(self) -> int:
            observed.append("process.wait")
            return 0

    monkeypatch.setattr(session, "teardown_timeout_seconds", 0.001)
    session.process = FakeProcess()

    asyncio.run(session.__aexit__(None, None, None))

    assert observed == ["stdin.close", "process.terminate", "process.wait"]


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
