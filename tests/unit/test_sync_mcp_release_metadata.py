"""Tests for generated MCP client-distribution version metadata."""

import argparse
import json
import tomllib

from packaging.requirements import Requirement
from packaging.version import Version

from scripts import sync_mcp_release_metadata as metadata


def test_plugin_semver_projects_pep440_prereleases():
    assert metadata.plugin_semver(Version("0.5.22.dev0")) == "0.5.22-dev.0"
    assert metadata.plugin_semver(Version("1.0.0rc2")) == "1.0.0-rc.2"
    assert metadata.plugin_semver(Version("1.2.3")) == "1.2.3"


def test_desktop_distribution_requirement_derives_installed_capability_extras():
    requirement = Requirement(
        metadata.desktop_package_requirement(
            Version("1.2.3"),
            capability_requirements=True,
        )
    )

    assert requirement.name == metadata.read_package_name()
    assert requirement.extras == {"gui", "mcp", "viz"}
    assert requirement.specifier.contains("1.2.3")


def test_every_mcp_distribution_projection_uses_the_desktop_requirement():
    package_version = Version("1.2.3")
    expected = metadata.desktop_package_requirement(
        package_version,
        capability_requirements=True,
    )
    projected = metadata.projected_files(
        package_version,
        capability_requirements=True,
    )

    plugin = json.loads(projected[metadata.PLUGIN_MCP_PATH])
    mcpb = tomllib.loads(projected[metadata.MCPB_PYPROJECT_PATH])
    registry = json.loads(projected[metadata.REGISTRY_PATH])

    assert expected in plugin["mcpServers"]["openhcs"]["args"]
    assert mcpb["project"]["dependencies"] == [expected]
    assert registry["packages"][0]["runtimeArguments"][0]["value"] == expected


def test_checked_in_mcp_release_metadata_matches_package_version():
    assert metadata.synchronize(check=True) == ()
    assert metadata.synchronize(check=True, capability_requirements=True) == ()


def test_dependency_free_metadata_check_only_reads_synchronized_declared_extras():
    assert metadata.metadata_package_extras() == ("gui", "mcp", "viz")


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
            capability_requirements=False,
            print_desktop_extras=False,
        ),
    )

    assert metadata.main() == 1
    assert "1.2.3 != 1.2.4" in capsys.readouterr().out
