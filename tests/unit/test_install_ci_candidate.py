"""Tests for wheel-boundary CI candidate installation."""

from pathlib import Path

import pytest
from packaging.version import Version

from scripts import install_ci_candidate as installer
from scripts.validate_local_release_floors import (
    CandidatePublication,
    ReleaseCandidate,
)
from scripts.wait_for_pypi_release import PyPIReleaseProbe


def test_pypi_install_uses_metadata_discovered_hash_pinned_wheels(
    monkeypatch,
    tmp_path: Path,
) -> None:
    candidate = ReleaseCandidate(
        name="example-package",
        version=Version("1.2.3"),
        dependencies=(),
        path=tmp_path / "external" / "example" / "pyproject.toml",
    )
    wheel_requirement = (
        "https://files.pythonhosted.org/example_package-1.2.3-py3-none-any.whl"
        "#sha256=" + "a" * 64
    )
    publication = CandidatePublication(
        candidate,
        PyPIReleaseProbe(True, "published", wheel_requirement),
    )
    commands = []

    monkeypatch.setattr(installer, "discover_local_projects", lambda: (candidate,))

    def build_wheel(_project_root: Path, wheel_directory: Path) -> None:
        (wheel_directory / "openhcs-0.7.21-py3-none-any.whl").touch()

    monkeypatch.setattr(installer, "_build_wheel", build_wheel)
    monkeypatch.setattr(installer, "validate_wheel_deployment", lambda _wheel: ())
    monkeypatch.setattr(
        installer.subprocess,
        "run",
        lambda command, **kwargs: commands.append((command, kwargs)),
    )

    installer.build_and_install_candidate(
        extras=("dev",),
        dependency_source=installer.CandidateDependencySource.PYPI,
        wheel_directory=tmp_path / "wheels",
        additional_requirements=("pytest-split==0.11.0",),
        local_project_extras=(),
        published_wheel_requirements=(publication.verified_wheel_requirement(),),
    )

    install_command = commands[0][0]
    assert wheel_requirement in install_command
    assert install_command.index(wheel_requirement) < install_command.index(
        str(tmp_path / "wheels" / "openhcs-0.7.21-py3-none-any.whl") + "[dev]"
    )


def test_existing_candidate_wheel_reuses_hash_pinned_dependency_projection(
    monkeypatch,
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "dist" / "openhcs-0.7.26-py3-none-any.whl"
    wheel.parent.mkdir()
    wheel.touch()
    wheel_requirement = (
        "https://files.pythonhosted.org/zmqruntime-0.2.18-py3-none-any.whl"
        "#sha256=" + "b" * 64
    )
    commands = []

    def unexpected_build(_project_root: Path, _wheel_directory: Path) -> None:
        raise AssertionError("existing candidate must not be rebuilt")

    monkeypatch.setattr(installer, "_build_wheel", unexpected_build)
    monkeypatch.setattr(installer, "validate_wheel_deployment", lambda _wheel: ())
    monkeypatch.setattr(
        installer.subprocess,
        "run",
        lambda command, **kwargs: commands.append((command, kwargs)),
    )

    installer.build_and_install_candidate(
        extras=("gui",),
        dependency_source=installer.CandidateDependencySource.PYPI,
        wheel_directory=tmp_path / "wheel-links",
        additional_requirements=(),
        local_project_extras=(),
        published_wheel_requirements=(wheel_requirement,),
        candidate_wheel=wheel,
    )

    install_command = commands[0][0]
    assert wheel_requirement in install_command
    assert f"{wheel.resolve()}[gui]" in install_command


def test_existing_candidate_rejects_a_non_openhcs_wheel(tmp_path: Path) -> None:
    wheel = tmp_path / "example_package-1.0-py3-none-any.whl"
    wheel.touch()

    with pytest.raises(RuntimeError, match="Candidate wheel is not OpenHCS"):
        installer._existing_root_wheel(wheel)


def test_source_candidate_wheelhouse_builds_metadata_discovered_projects(
    monkeypatch,
    tmp_path: Path,
) -> None:
    candidate = ReleaseCandidate(
        name="example-package",
        version=Version("1.2.3"),
        dependencies=(),
        path=tmp_path / "external" / "example" / "pyproject.toml",
    )
    wheel_directory = tmp_path / "wheelhouse"
    built_projects: list[Path] = []

    monkeypatch.setattr(installer, "discover_local_projects", lambda: (candidate,))
    monkeypatch.setattr(installer, "validate_local_candidate_compatibility", lambda: ())
    monkeypatch.setattr(installer, "validate_wheel_deployment", lambda _wheel: ())

    def build_wheel(project_root: Path, destination: Path) -> None:
        built_projects.append(project_root)
        destination.mkdir(parents=True, exist_ok=True)
        wheel_name = (
            "openhcs-0.7.27-py3-none-any.whl"
            if project_root == installer.REPO_ROOT
            else "example_package-1.2.3-py3-none-any.whl"
        )
        destination.joinpath(wheel_name).touch()

    monkeypatch.setattr(installer, "_build_wheel", build_wheel)

    wheelhouse = installer.build_source_candidate_wheelhouse(
        wheel_directory=wheel_directory
    )

    assert wheelhouse.local_projects == (candidate,)
    assert wheelhouse.root_wheel == (
        wheel_directory / "openhcs-0.7.27-py3-none-any.whl"
    )
    assert built_projects == [candidate.path.parent, installer.REPO_ROOT]


def test_build_only_cli_rejects_installation_requirements(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        installer.sys,
        "argv",
        [
            "install_ci_candidate",
            "--dependency-source",
            "submodules",
            "--wheel-directory",
            str(tmp_path / "wheelhouse"),
            "--build-only",
            "--extras",
            "gui",
        ],
    )

    with pytest.raises(SystemExit):
        installer.main()
