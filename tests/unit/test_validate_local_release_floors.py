"""Tests for local extracted-package release-floor validation."""

import ast
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


def test_setup_py_only_projects_build_command_hooks():
    setup_path = floors.REPO_ROOT / "setup.py"
    module = ast.parse(setup_path.read_text(encoding="utf-8"), filename=str(setup_path))
    setup_calls = tuple(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "setup"
    )

    assert len(setup_calls) == 1
    assert {keyword.arg for keyword in setup_calls[0].keywords} == {"cmdclass"}
    obsolete_dependency_selectors = {
        "PYPI_DEPENDENCIES",
        "get_local_external_dependencies",
        "get_external_dependencies",
        "is_development_mode",
    }
    declared_names = {
        node.name
        for node in module.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    declared_names.update(
        target.id
        for node in module.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets if isinstance(node, ast.Assign) else (node.target,)
        )
        if isinstance(target, ast.Name)
    )
    assert declared_names.isdisjoint(obsolete_dependency_selectors)


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
