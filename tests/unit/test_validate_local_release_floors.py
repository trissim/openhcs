"""Tests for local extracted-package release-floor validation."""

from pathlib import Path

from scripts import validate_local_release_floors as floors


def _write_project(
    path: Path,
    *,
    name: str,
    version: str,
    dependencies: tuple[str, ...] = (),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dependency_lines = ",\n".join(f'    "{dependency}"' for dependency in dependencies)
    path.write_text(
        "\n".join(
            (
                "[project]",
                f'name = "{name}"',
                f'version = "{version}"',
                "dependencies = [",
                dependency_lines,
                "]",
                "",
            )
        ),
        encoding="utf-8",
    )


def test_checked_in_local_candidate_versions_satisfy_declared_floors():
    assert floors.validate() == ()


def test_rejects_openhcs_floor_above_available_candidate(tmp_path):
    _write_project(
        tmp_path / "pyproject.toml",
        name="openhcs",
        version="1.0.0",
        dependencies=("example-package>=2.0.0",),
    )
    _write_project(
        tmp_path / "external" / "example" / "pyproject.toml",
        name="example-package",
        version="1.9.0",
    )

    assert floors.validate(tmp_path) == (
        "OpenHCS requirement example-package>=2.0.0 excludes available local "
        "candidate example-package==1.9.0",
    )


def test_rejects_openhcs_floor_below_available_candidate(tmp_path):
    _write_project(
        tmp_path / "pyproject.toml",
        name="openhcs",
        version="1.0.0",
        dependencies=("example-package>=1.0.0",),
    )
    _write_project(
        tmp_path / "external" / "example" / "pyproject.toml",
        name="example-package",
        version="1.1.0",
    )

    assert floors.validate(tmp_path) == (
        "OpenHCS requirement example-package>=1.0.0 does not require local "
        "candidate floor example-package>=1.1.0",
    )


def test_rejects_local_dependency_floor_above_available_candidate(tmp_path):
    _write_project(
        tmp_path / "pyproject.toml",
        name="openhcs",
        version="1.0.0",
        dependencies=("consumer>=1.0.0", "authority>=2.0.0"),
    )
    _write_project(
        tmp_path / "external" / "authority" / "pyproject.toml",
        name="authority",
        version="2.0.0",
    )
    _write_project(
        tmp_path / "external" / "consumer" / "pyproject.toml",
        name="consumer",
        version="1.0.0",
        dependencies=("authority>=2.1.0",),
    )

    assert floors.validate(tmp_path) == (
        "consumer==1.0.0 requirement authority>=2.1.0 excludes available local "
        "candidate authority==2.0.0",
    )
