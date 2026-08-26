"""Tests for local extracted-package release-floor validation."""

import ast
import json
import subprocess
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

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


def _write_dynamic_hatch_project(
    path: Path,
    *,
    name: str,
    version: str,
    dependencies: tuple[str, ...] = (),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    source_path = path.parent / "src" / name.replace("-", "_") / "__init__.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(f'__version__ = "{version}"\n', encoding="utf-8")
    dependency_lines = ",\n".join(f'    "{dependency}"' for dependency in dependencies)
    path.write_text(
        "\n".join(
            (
                "[project]",
                f'name = "{name}"',
                'dynamic = ["version"]',
                "dependencies = [",
                dependency_lines,
                "]",
                "",
                "[tool.hatch.version]",
                f'path = "{source_path.relative_to(path.parent)}"',
                "",
            )
        ),
        encoding="utf-8",
    )


def _git(path: Path, *args: str) -> None:
    subprocess.run(("git", "-C", str(path), *args), check=True, capture_output=True)


def test_checked_in_local_candidate_snapshots_are_mutually_compatible():
    assert floors.validate_local_candidate_compatibility() == ()


def test_dynamic_hatch_version_path_is_the_candidate_authority(tmp_path):
    _write_project(
        tmp_path / "pyproject.toml",
        name="openhcs",
        version="1.0.0",
        dependencies=("example-package>=1.2.3,<2",),
    )
    candidate_path = tmp_path / "external" / "example" / "pyproject.toml"
    _write_dynamic_hatch_project(
        candidate_path,
        name="example-package",
        version="1.2.3",
    )

    candidate = floors.read_release_candidate(candidate_path)

    assert str(candidate.version) == "1.2.3"
    assert floors.validate(tmp_path) == ()


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
        for target in (node.targets if isinstance(node, ast.Assign) else (node.target,))
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


def test_rejects_prerelease_candidates_from_installer_facing_floors(tmp_path):
    _write_project(
        tmp_path / "pyproject.toml",
        name="openhcs",
        version="1.0.0",
        dependencies=("example-package>=2.0.0rc1",),
    )
    _write_project(
        tmp_path / "external" / "example" / "pyproject.toml",
        name="example-package",
        version="2.0.0rc1",
    )

    assert floors.validate(tmp_path) == (
        "Local release candidate example-package==2.0.0rc1 is a prerelease; "
        "installer-facing dependency floors must use stable published versions",
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

    assert floors.validate_local_candidate_compatibility(tmp_path) == ()
    assert floors.validate(tmp_path) == (
        "OpenHCS requirement example-package>=1.0.0 does not require local "
        "candidate floor example-package>=1.1.0",
    )


def test_rejects_unbounded_openhcs_first_party_requirement(tmp_path):
    _write_project(
        tmp_path / "pyproject.toml",
        name="openhcs",
        version="1.0.0",
        dependencies=("example-package>=1.2.3",),
    )
    _write_project(
        tmp_path / "external" / "example" / "pyproject.toml",
        name="example-package",
        version="1.2.3",
    )

    assert floors.validate(tmp_path) == (
        "OpenHCS requirement example-package>=1.2.3 does not exclude next "
        "breaking series example-package>=2.0.0",
    )


def test_zero_major_compatibility_ceiling_uses_the_next_minor(tmp_path):
    _write_project(
        tmp_path / "pyproject.toml",
        name="openhcs",
        version="1.0.0",
        dependencies=("example-package>=0.3.4,<0.4",),
    )
    _write_project(
        tmp_path / "external" / "example" / "pyproject.toml",
        name="example-package",
        version="0.3.4",
    )

    assert floors.validate(tmp_path) == ()


def test_candidate_requirement_compatibility_owns_release_range_proof(tmp_path):
    candidate = floors.ReleaseCandidate(
        name="example-package",
        version=Version("0.3.4"),
        dependencies=(),
        path=tmp_path / "pyproject.toml",
    )

    compatible = floors.CandidateRequirementCompatibility(
        Requirement("example-package>=0.3.4,<0.4"),
        candidate,
    )
    assert compatible.accepts_candidate
    assert compatible.requires_candidate_floor
    assert compatible.breaking_release_boundary == Version("0.4.0")
    assert compatible.excludes_next_breaking_series

    boundary_inclusive = floors.CandidateRequirementCompatibility(
        Requirement("example-package>=0.3.4,<=0.4"),
        candidate,
    )
    assert not boundary_inclusive.excludes_next_breaking_series


def test_rejects_local_dependency_floor_above_available_candidate(tmp_path):
    _write_project(
        tmp_path / "pyproject.toml",
        name="openhcs",
        version="1.0.0",
        dependencies=("consumer>=1.0.0,<2", "authority>=2.0.0,<3"),
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


def test_rejects_published_source_drift_without_a_version_change(tmp_path):
    _write_project(
        tmp_path / "pyproject.toml",
        name="openhcs",
        version="1.0.0",
        dependencies=("example-package>=1.2.3,<2",),
    )
    project_root = tmp_path / "external" / "example"
    _write_project(
        project_root / "pyproject.toml",
        name="example-package",
        version="1.2.3",
    )
    source_path = project_root / "src" / "example_package" / "__init__.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("VALUE = 1\n", encoding="utf-8")
    _git(project_root, "init", "--initial-branch=main")
    _git(project_root, "config", "user.email", "ci@example.invalid")
    _git(project_root, "config", "user.name", "CI")
    _git(project_root, "add", "pyproject.toml", "src")
    _git(project_root, "commit", "-m", "release")
    _git(project_root, "tag", "v1.2.3")
    source_path.write_text("VALUE = 2\n", encoding="utf-8")

    assert floors.validate_local_candidate_compatibility(tmp_path) == ()
    assert floors.validate(tmp_path) == (
        "Local release candidate example-package==1.2.3 differs from v1.2.3 "
        "in published inputs: src/example_package/__init__.py",
    )


def test_waits_for_metadata_discovered_candidates_without_a_package_manifest(
    tmp_path,
):
    _write_project(
        tmp_path / "external" / "second" / "pyproject.toml",
        name="Second_Package",
        version="2.0.0",
    )
    _write_project(
        tmp_path / "external" / "first" / "pyproject.toml",
        name="first-package",
        version="1.2.3",
    )
    calls = []
    times = iter((10.0, 11.0, 12.0))

    def waiter(project, version, **kwargs):
        calls.append((project, version, kwargs))
        return floors.PyPIReleaseProbe(True, f"published {project}")

    publications = floors.wait_for_published_candidates(
        tmp_path,
        timeout_seconds=30,
        poll_interval_seconds=2,
        waiter=waiter,
        monotonic=lambda: next(times),
    )

    assert tuple(
        (publication.project.name, str(publication.project.version))
        for publication in publications
    ) == (
        ("first-package", "1.2.3"),
        ("Second_Package", "2.0.0"),
    )
    assert calls == [
        (
            "first-package",
            "1.2.3",
            {"timeout_seconds": 29.0, "poll_interval_seconds": 2},
        ),
        (
            "Second_Package",
            "2.0.0",
            {"timeout_seconds": 28.0, "poll_interval_seconds": 2},
        ),
    ]


def test_candidate_publication_wait_stops_at_the_first_unavailable_release(
    tmp_path,
):
    for directory, name in (("first", "first-package"), ("second", "second-package")):
        _write_project(
            tmp_path / "external" / directory / "pyproject.toml",
            name=name,
            version="1.0.0",
        )
    calls = []

    def waiter(project, version, **kwargs):
        calls.append((project, version))
        return floors.PyPIReleaseProbe(False, "simple index is still stale")

    publications = floors.wait_for_published_candidates(
        tmp_path,
        timeout_seconds=30,
        poll_interval_seconds=2,
        waiter=waiter,
        monotonic=lambda: 10.0,
    )

    assert calls == [("first-package", "1.0.0")]
    assert len(publications) == 1
    assert publications[0].probe.detail == "simple index is still stale"


def test_candidate_publication_owns_its_verified_wheel_requirement(tmp_path):
    project_path = tmp_path / "external" / "example" / "pyproject.toml"
    _write_project(
        project_path,
        name="example-package",
        version="1.2.3",
    )
    project = floors.read_release_candidate(project_path)
    wheel_url = "https://files.pythonhosted.org/example.whl#sha256=" + "a" * 64

    publication = floors.CandidatePublication(
        project,
        floors.PyPIReleaseProbe(True, "published", wheel_url),
    )

    assert publication.verified_wheel_requirement() == wheel_url


def test_main_writes_metadata_derived_wheel_requirements(monkeypatch, tmp_path):
    project_path = tmp_path / "external" / "example" / "pyproject.toml"
    _write_project(
        project_path,
        name="example-package",
        version="1.2.3",
    )
    project = floors.read_release_candidate(project_path)
    wheel_url = "https://files.pythonhosted.org/example.whl#sha256=" + "a" * 64
    monkeypatch.setattr(floors, "validate", lambda: ())
    monkeypatch.setattr(
        floors,
        "wait_for_published_candidates",
        lambda **_kwargs: (
            floors.CandidatePublication(
                project,
                floors.PyPIReleaseProbe(True, "published", wheel_url),
            ),
        ),
    )
    output_path = tmp_path / "requirements.json"

    assert (
        floors.main(
            (
                "--wait-for-pypi",
                "--wheel-requirements-output",
                str(output_path),
            )
        )
        == 0
    )
    output = output_path.read_text(encoding="utf-8")
    assert output.endswith("\n")
    assert json.loads(output) == [wheel_url]
