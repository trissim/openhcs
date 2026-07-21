"""Tests for the generated browser-hosted Codex plugin artifact."""

import json

import pytest

from scripts.build_hosted_mcp_plugin import (
    build_hosted_plugin,
    validate_remote_mcp_url,
)


def _source_manifest(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "name": "openhcs",
                "version": "1.2.3",
                "description": "local",
                "skills": "./skills/",
                "mcpServers": "./.mcp.json",
                "interface": {
                    "shortDescription": "local",
                    "longDescription": "local",
                    "capabilities": ["Read", "Write", "Interactive"],
                    "defaultPrompt": ["local"],
                },
            }
        ),
        encoding="utf-8",
    )
    return source


def test_hosted_plugin_is_remote_read_only_projection(tmp_path):
    destination = tmp_path / "plugin"

    manifest_path, mcp_path = build_hosted_plugin(
        remote_mcp_url="https://mcp.openhcs.example/mcp",
        output_dir=destination,
        source_manifest=_source_manifest(tmp_path),
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mcp = json.loads(mcp_path.read_text(encoding="utf-8"))
    assert manifest["version"] == "1.2.3"
    assert "skills" not in manifest
    assert manifest["interface"]["capabilities"] == ["Read"]
    assert mcp == {
        "mcpServers": {"openhcs": {"url": "https://mcp.openhcs.example/mcp"}}
    }


def test_hosted_plugin_rejects_insecure_url():
    with pytest.raises(ValueError, match="HTTPS"):
        validate_remote_mcp_url("http://mcp.openhcs.example/mcp")


def test_hosted_plugin_does_not_overwrite_nonempty_output(tmp_path):
    destination = tmp_path / "plugin"
    destination.mkdir()
    (destination / "keep.txt").write_text("user data", encoding="utf-8")

    with pytest.raises(ValueError, match="not empty"):
        build_hosted_plugin(
            remote_mcp_url="https://mcp.openhcs.example/mcp",
            output_dir=destination,
            source_manifest=_source_manifest(tmp_path),
        )

    assert (destination / "keep.txt").read_text(encoding="utf-8") == "user data"
