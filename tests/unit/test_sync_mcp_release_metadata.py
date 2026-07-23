"""Tests for generated MCP client-distribution version metadata."""

import argparse

from packaging.version import Version

from scripts import sync_mcp_release_metadata as metadata


def test_plugin_semver_projects_pep440_prereleases():
    assert metadata.plugin_semver(Version("0.5.22.dev0")) == "0.5.22-dev.0"
    assert metadata.plugin_semver(Version("1.0.0rc2")) == "1.0.0-rc.2"
    assert metadata.plugin_semver(Version("1.2.3")) == "1.2.3"


def test_checked_in_mcp_release_metadata_matches_package_version():
    assert metadata.synchronize(check=True) == ()


def test_main_rejects_release_version_mismatch(monkeypatch, capsys):
    monkeypatch.setattr(
        metadata,
        "read_package_version",
        lambda: Version("1.2.3"),
    )
    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: argparse.Namespace(
            check=True,
            expected_version="1.2.4",
        ),
    )

    assert metadata.main() == 1
    assert "1.2.3 != 1.2.4" in capsys.readouterr().out
