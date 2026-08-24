import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def test_stdio_transport_reserves_protocol_stdout_for_persistent_session(
    tmp_path: Path,
) -> None:
    server_script = tmp_path / "noisy_mcp_server.py"
    server_script.write_text(
        """\
import os

from mcp.server.fastmcp import FastMCP

from openhcs.mcp.stdio import McpStdioTransport


server = FastMCP("noisy-test")


@server.tool()
def emit_noise() -> dict[str, bool]:
    print("python stdout noise", flush=True)
    os.write(1, b"native stdout noise\\n")
    return {"ok": True}


@server.tool()
def health() -> dict[str, bool]:
    return {"ok": True}


with McpStdioTransport.reserve_process_stdout() as transport:
    transport.run(server)
""",
        encoding="utf-8",
    )
    stderr_path = tmp_path / "server.stderr.log"

    async def call_persistent_server() -> tuple[object, object]:
        parameters = StdioServerParameters(
            command=sys.executable,
            args=(str(server_script),),
        )
        with stderr_path.open("w", encoding="utf-8") as stderr:
            async with stdio_client(parameters, errlog=stderr) as (
                read_stream,
                write_stream,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await asyncio.wait_for(session.initialize(), timeout=5)
                    noisy = await asyncio.wait_for(
                        session.call_tool("emit_noise", {}),
                        timeout=5,
                    )
                    healthy = await asyncio.wait_for(
                        session.call_tool("health", {}),
                        timeout=5,
                    )
                    return noisy, healthy

    noisy, healthy = asyncio.run(call_persistent_server())

    assert noisy.structuredContent == {"ok": True}
    assert healthy.structuredContent == {"ok": True}
    stderr_text = stderr_path.read_text(encoding="utf-8")
    assert "python stdout noise" in stderr_text
    assert "native stdout noise" in stderr_text
    assert not any(
        line.startswith("{") and json.loads(line).get("jsonrpc") == "2.0"
        for line in stderr_text.splitlines()
    )
