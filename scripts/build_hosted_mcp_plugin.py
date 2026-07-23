#!/usr/bin/env python3
"""Build the remote OpenHCS plugin artifact for browser-hosted Codex clients."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PLUGIN_ROOT = REPO_ROOT / "packaging" / "codex" / "openhcs"
LOCAL_PLUGIN_MANIFEST = LOCAL_PLUGIN_ROOT / ".codex-plugin" / "plugin.json"


def validate_remote_mcp_url(url: str) -> str:
    """Return a production-safe remote MCP URL."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"Hosted MCP plugin URL must use HTTPS: {url}")
    return url


def build_hosted_plugin(
    *,
    remote_mcp_url: str,
    output_dir: Path,
    source_manifest: Path = LOCAL_PLUGIN_MANIFEST,
) -> tuple[Path, Path]:
    """Project local release metadata into a remote-only plugin directory."""
    url = validate_remote_mcp_url(remote_mcp_url)
    destination = output_dir.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"Hosted plugin output directory is not empty: {destination}")

    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    manifest.pop("skills", None)
    manifest["description"] = (
        "Discover OpenHCS functions, architecture, configuration schemas, and "
        "packaged guidance through the authenticated hosted MCP service."
    )
    interface = manifest["interface"]
    interface["shortDescription"] = "Explore OpenHCS through hosted MCP."
    interface["longDescription"] = (
        "Use the authenticated, read-only OpenHCS hosted service for function "
        "discovery, architecture guidance, and configuration schema reflection."
    )
    interface["capabilities"] = ["Read"]
    interface["defaultPrompt"] = [
        "Explain the OpenHCS pipeline architecture.",
        "Find OpenHCS processing functions for this microscopy workflow.",
        "Describe the OpenHCS pipeline configuration schema.",
    ]

    manifest_path = destination / ".codex-plugin" / "plugin.json"
    mcp_path = destination / ".mcp.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    mcp_path.write_text(
        json.dumps(
            {"mcpServers": {"openhcs": {"url": url}}},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path, mcp_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Public HTTPS MCP endpoint.")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    for path in build_hosted_plugin(
        remote_mcp_url=args.url,
        output_dir=args.output_dir,
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
