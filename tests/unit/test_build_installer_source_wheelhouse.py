"""Tests for the installer source wheelhouse projection."""

from pathlib import Path
from types import SimpleNamespace

from scripts import build_installer_source_wheelhouse as wheelhouse


def test_pypi_mode_builds_only_the_openhcs_candidate(tmp_path: Path) -> None:
    output_directory = tmp_path / "wheelhouse"
    commands: list[tuple[str, ...]] = []

    projects = wheelhouse.build_wheelhouse(
        output_directory,
        "pypi",
        repo_root=tmp_path,
        runner=lambda command: commands.append(tuple(command)),
    )

    assert projects == (tmp_path,)
    assert output_directory.is_dir()
    assert len(commands) == 1
    assert commands[0][-1] == str(tmp_path)


def test_submodule_mode_derives_dependency_candidates_from_project_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "wheelhouse"
    dependency_paths = (
        tmp_path / "external" / "PolyStore" / "pyproject.toml",
        tmp_path / "external" / "zmqruntime" / "pyproject.toml",
    )
    monkeypatch.setattr(
        wheelhouse,
        "discover_local_projects",
        lambda repo_root: tuple(
            SimpleNamespace(path=path) for path in dependency_paths
        ),
    )
    commands: list[tuple[str, ...]] = []

    projects = wheelhouse.build_wheelhouse(
        output_directory,
        "submodules",
        repo_root=tmp_path,
        runner=lambda command: commands.append(tuple(command)),
    )

    assert projects == (
        tmp_path,
        dependency_paths[0].parent,
        dependency_paths[1].parent,
    )
    assert [command[-1] for command in commands] == [
        str(project) for project in projects
    ]
    assert all(
        command[-3:-1] == ("--outdir", str(output_directory)) for command in commands
    )
