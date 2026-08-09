"""Tests for wheel-boundary CI candidate installation."""

from pathlib import Path

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
    monkeypatch.setattr(
        installer.subprocess,
        "run",
        lambda command, **kwargs: commands.append((command, kwargs)),
    )

    installer.build_and_install_candidate(
        extras=("dev",),
        dependency_source="pypi",
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
