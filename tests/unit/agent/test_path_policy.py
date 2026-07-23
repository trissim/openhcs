from pathlib import Path

import pytest

import openhcs.agent.path_policy as path_policy_module
from openhcs.agent.path_policy import (
    AgentPathLocationAuthority,
    AgentPathPolicy,
    AgentPathPolicyError,
    AgentPathRootSet,
)


def test_path_policy_allows_paths_under_roots(tmp_path: Path):
    readable = tmp_path / "readable.txt"
    readable.write_text("ok")
    writable = tmp_path / "nested" / "output.py"
    policy = AgentPathPolicy.with_roots(
        readable_roots=(tmp_path.resolve(),),
        writable_roots=(tmp_path.resolve(),),
    )

    assert policy.assert_readable(readable) == readable.resolve()
    assert policy.assert_writable(writable) == writable.resolve(strict=False)


def test_path_policy_rejects_paths_outside_roots(tmp_path: Path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("no")
    policy = AgentPathPolicy.with_roots(
        readable_roots=(allowed.resolve(),),
        writable_roots=(allowed.resolve(),),
    )

    with pytest.raises(AgentPathPolicyError):
        policy.assert_readable(outside)

    with pytest.raises(AgentPathPolicyError):
        policy.assert_writable(outside)


def test_environment_roots_use_platform_path_separator(monkeypatch, tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    monkeypatch.setattr(path_policy_module.os, "pathsep", ";")
    monkeypatch.setenv("OPENHCS_TEST_ROOTS", f"{first};{second}")

    roots = AgentPathRootSet.from_environment(
        "OPENHCS_TEST_ROOTS",
        AgentPathRootSet(()),
    )

    assert roots.roots == (first.resolve(), second.resolve())


def test_writable_root_can_be_created_lazily(tmp_path: Path):
    output_root = tmp_path / "not-created-yet"
    output_path = output_root / "nested" / "result.json"
    policy = AgentPathPolicy.with_roots(
        readable_roots=(tmp_path,),
        writable_roots=(output_root,),
    )

    assert policy.assert_writable(output_path) == output_path.resolve(strict=False)


def test_installed_wheel_defaults_do_not_write_to_site_packages(
    monkeypatch,
    tmp_path: Path,
):
    site_packages = tmp_path / "site-packages"
    package_root = site_packages / "openhcs"
    temporary_root = tmp_path / "runtime"
    user_data_root = tmp_path / "user-data" / "openhcs"
    package_root.mkdir(parents=True)
    temporary_root.mkdir()
    monkeypatch.setattr(
        AgentPathLocationAuthority,
        "package_root",
        staticmethod(lambda: package_root),
    )
    monkeypatch.setattr(
        AgentPathLocationAuthority,
        "source_checkout_root",
        classmethod(lambda cls: None),
    )
    monkeypatch.setattr(
        AgentPathLocationAuthority,
        "temporary_root",
        staticmethod(lambda: temporary_root),
    )
    monkeypatch.setattr(
        AgentPathLocationAuthority,
        "user_data_root",
        staticmethod(lambda: user_data_root),
    )

    policy = AgentPathPolicy.default()

    assert site_packages.resolve() not in policy.readable_roots.roots
    assert package_root.resolve() in policy.readable_roots.roots
    assert temporary_root.resolve() in policy.writable_roots.roots
    assert (user_data_root / "mcp_outputs").resolve() in policy.writable_roots.roots
    assert policy.assert_writable(user_data_root / "mcp_outputs" / "result.json") == (
        user_data_root / "mcp_outputs" / "result.json"
    ).resolve(strict=False)
