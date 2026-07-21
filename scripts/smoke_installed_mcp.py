"""Smoke-test an installed OpenHCS MCP wheel from outside its source checkout."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path


def _tool_payload(result) -> dict:
    if not result.content or not hasattr(result.content[0], "text"):
        raise AssertionError("MCP tool result did not contain text content.")
    payload = json.loads(result.content[0].text)
    if not isinstance(payload, dict):
        raise AssertionError("MCP tool result payload was not an object.")
    if not isinstance(result.structuredContent, dict):
        raise AssertionError("MCP tool result did not contain structured content.")
    if payload != result.structuredContent:
        raise AssertionError("MCP text and structured tool payloads diverged.")
    return payload


async def _run_protocol_smoke() -> dict:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    parameters = StdioServerParameters(
        command=sys.executable,
        args=("-m", "openhcs.mcp"),
    )
    async with stdio_client(parameters) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await asyncio.wait_for(session.initialize(), timeout=30)
            tools_result = await asyncio.wait_for(session.list_tools(), timeout=30)
            health_result = await asyncio.wait_for(
                session.call_tool("openhcs_health_check", {}),
                timeout=30,
            )
            catalog_result = await asyncio.wait_for(
                session.call_tool("openhcs_list_knowledge_documents", {}),
                timeout=30,
            )
            capabilities_result = await asyncio.wait_for(
                session.call_tool("openhcs_list_capabilities", {}),
                timeout=30,
            )
            document_result = await asyncio.wait_for(
                session.call_tool(
                    "openhcs_get_knowledge_document",
                    {
                        "document_id": "openhcs_core_model",
                        "max_chars": 1_000,
                    },
                ),
                timeout=30,
            )

    health = _tool_payload(health_result)
    capabilities = _tool_payload(capabilities_result)
    catalog = _tool_payload(catalog_result)
    document = _tool_payload(document_result)
    if health.get("status") != "ok":
        raise AssertionError(f"Installed MCP health failed: {health}")
    if capabilities.get("surface_profile") != "desktop":
        raise AssertionError(
            f"Installed MCP did not select the desktop surface: {capabilities}"
        )
    declared_tool_names = {
        item.get("name")
        for item in capabilities.get("capabilities", ())
        if isinstance(item, dict) and item.get("kind") == "tool"
    }
    listed_tool_names = {tool.name for tool in tools_result.tools}
    if listed_tool_names != declared_tool_names:
        raise AssertionError(
            "Installed MCP tools diverged from capability discovery: "
            f"listed={listed_tool_names} declared={declared_tool_names}"
        )
    if not all(tool.outputSchema for tool in tools_result.tools):
        raise AssertionError("Installed MCP tools are missing output schemas.")
    document_ids = {
        item.get("document_id")
        for item in catalog.get("documents", ())
        if isinstance(item, dict)
    }
    if "openhcs_core_model" not in document_ids:
        raise AssertionError(f"Installed knowledge catalog is incomplete: {catalog}")
    if "OpenHCS" not in str(document.get("content", "")):
        raise AssertionError(f"Installed knowledge document is empty: {document}")
    if document.get("errors"):
        raise AssertionError(
            f"Installed knowledge document returned errors: {document}"
        )
    return {
        "health_status": health["status"],
        "mcp_surface_profile": capabilities["surface_profile"],
        "mcp_tool_count": len(listed_tool_names),
        "knowledge_document_count": len(document_ids),
        "knowledge_document": "openhcs_core_model",
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forbid-import-root",
        type=Path,
        required=True,
        help="Source checkout that must not own the imported openhcs package.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    forbidden_root = args.forbid_import_root.resolve()
    original_working_directory = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="openhcs-installed-mcp-") as directory:
        working_directory = Path(directory).resolve()
        os.chdir(working_directory)
        try:
            import openhcs
            from openhcs.agent.knowledge_manifest import (
                default_knowledge_base_manifest_path,
                default_repo_root,
            )

            package_path = Path(openhcs.__file__).resolve()
            if package_path.is_relative_to(forbidden_root):
                raise AssertionError(
                    "Smoke test imported the source checkout instead of the wheel: "
                    f"{package_path}"
                )
            knowledge_root = default_repo_root().resolve()
            if knowledge_root.is_relative_to(forbidden_root):
                raise AssertionError(
                    f"Knowledge root resolved into the source checkout: {knowledge_root}"
                )
            manifest_path = default_knowledge_base_manifest_path()
            if not manifest_path.is_file():
                raise AssertionError(
                    f"Packaged knowledge manifest is missing: {manifest_path}"
                )
            for command in (
                "openhcs",
                "openhcs-gui",
                "openhcs-mcp",
                "openhcs-mcp-http",
                "openhcs-mcp-dev",
                "openhcs-recache",
            ):
                if shutil.which(command) is None:
                    raise AssertionError(
                        f"Installed console script is missing: {command}"
                    )
            if importlib.util.find_spec("PyQt6") is None:
                raise AssertionError(
                    "The combined local client installation is missing the PyQt6 UI."
                )

            result = asyncio.run(_run_protocol_smoke())
            result.update(
                {
                    "package_path": str(package_path),
                    "knowledge_root": str(knowledge_root),
                    "working_directory": str(working_directory),
                }
            )
        finally:
            os.chdir(original_working_directory)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
