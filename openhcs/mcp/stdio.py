"""Process-level stdout ownership for the OpenHCS MCP stdio transport."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from io import TextIOWrapper
from typing import TYPE_CHECKING

import anyio
from mcp.server.stdio import stdio_server

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


class McpStdioTransport:
    """Reserve protocol stdout and route all application output to stderr.

    MCP stdio assigns stdout exclusively to JSON-RPC. Libraries imported by an
    OpenHCS capability may write through either ``sys.stdout`` or file
    descriptor 1, including from background work that outlives the initiating
    call. The transport therefore owns the process channel for its entire
    lifetime rather than asking individual services to redirect output.
    """

    def __init__(self, protocol_stdout: TextIOWrapper) -> None:
        self._protocol_stdout = protocol_stdout

    @classmethod
    @contextmanager
    def reserve_process_stdout(cls) -> Iterator[McpStdioTransport]:
        """Reserve the current stdout descriptor before server construction."""

        process_stdout = sys.stdout
        stdout_fd = process_stdout.fileno()
        stderr_fd = sys.stderr.fileno()
        process_stdout.flush()
        sys.stderr.flush()

        saved_stdout_fd = os.dup(stdout_fd)
        protocol_binary = os.fdopen(os.dup(stdout_fd), "wb", buffering=0)
        protocol_stdout = TextIOWrapper(
            protocol_binary,
            encoding="utf-8",
            errors="strict",
            write_through=True,
        )
        try:
            os.dup2(stderr_fd, stdout_fd)
            sys.stdout = sys.stderr
            yield cls(protocol_stdout)
        finally:
            sys.stdout.flush()
            protocol_stdout.flush()
            sys.stdout = process_stdout
            os.dup2(saved_stdout_fd, stdout_fd)
            os.close(saved_stdout_fd)
            protocol_stdout.close()

    def run(self, server: FastMCP) -> None:
        """Run one FastMCP server against the reserved protocol channel."""

        anyio.run(self._run, server)

    async def _run(self, server: FastMCP) -> None:
        async_stdout = anyio.wrap_file(self._protocol_stdout)
        async with stdio_server(stdout=async_stdout) as (read_stream, write_stream):
            await server._mcp_server.run(
                read_stream,
                write_stream,
                server._mcp_server.create_initialization_options(),
            )
