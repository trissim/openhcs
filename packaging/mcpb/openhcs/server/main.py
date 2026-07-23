"""MCPB entry point for the PyPI-distributed OpenHCS MCP server."""

from openhcs.mcp.bootstrap import run_bootstrapped_server


if __name__ == "__main__":
    run_bootstrapped_server()
