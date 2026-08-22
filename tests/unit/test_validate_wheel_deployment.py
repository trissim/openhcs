from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

from scripts.validate_wheel_deployment import validate_wheel_deployment

COMPOSE = """\
services:
  web:
    build:
      context: .
      dockerfile: Dockerfile.web
"""

DOCKERFILE = """\
FROM example.invalid/base:1
COPY plugin /tmp/plugin
"""


def _write_wheel(path: Path, members: dict[str, str]) -> None:
    with ZipFile(path, "w") as wheel:
        for member, contents in members.items():
            wheel.writestr(member, contents)


def test_deployment_validator_derives_complete_build_inputs(tmp_path: Path) -> None:
    wheel_path = tmp_path / "candidate.whl"
    _write_wheel(
        wheel_path,
        {
            "package/deployment/docker-compose.yml": COMPOSE,
            "package/deployment/Dockerfile.web": DOCKERFILE,
            "package/deployment/plugin/setup.py": "from setuptools import setup\n",
        },
    )

    assert validate_wheel_deployment(wheel_path) == ()


def test_deployment_validator_rejects_missing_declared_dockerfile(
    tmp_path: Path,
) -> None:
    wheel_path = tmp_path / "candidate.whl"
    _write_wheel(
        wheel_path,
        {"package/deployment/docker-compose.yml": COMPOSE},
    )

    assert validate_wheel_deployment(wheel_path) == (
        "package/deployment/docker-compose.yml: missing declared Dockerfile "
        "package/deployment/Dockerfile.web",
    )


def test_deployment_validator_rejects_missing_dockerfile_copy_source(
    tmp_path: Path,
) -> None:
    wheel_path = tmp_path / "candidate.whl"
    _write_wheel(
        wheel_path,
        {
            "package/deployment/docker-compose.yml": COMPOSE,
            "package/deployment/Dockerfile.web": DOCKERFILE,
        },
    )

    assert validate_wheel_deployment(wheel_path) == (
        "package/deployment/Dockerfile.web: missing declared build source "
        "package/deployment/plugin",
    )


def test_deployment_validator_rejects_nested_package_build_output(
    tmp_path: Path,
) -> None:
    wheel_path = tmp_path / "candidate.whl"
    nested_build_member = "openhcs/omero/plugin/build/lib/omero_openhcs/apps.py"
    _write_wheel(wheel_path, {nested_build_member: "INSTALLED_APPS = []\n"})

    assert validate_wheel_deployment(wheel_path) == (
        f"{nested_build_member}: wheel contains nested build output",
    )


def test_deployment_validator_rejects_nested_package_metadata(
    tmp_path: Path,
) -> None:
    wheel_path = tmp_path / "candidate.whl"
    nested_metadata_member = "openhcs/omero/plugin/example.egg-info/PKG-INFO"
    _write_wheel(wheel_path, {nested_metadata_member: "Name: example\n"})

    assert validate_wheel_deployment(wheel_path) == (
        f"{nested_metadata_member}: wheel contains nested package metadata",
    )


def test_deployment_validator_rejects_developer_home_paths(tmp_path: Path) -> None:
    wheel_path = tmp_path / "candidate.whl"
    macos_source_member = "openhcs/macos_example.py"
    windows_source_member = "openhcs/windows_example.py"
    _write_wheel(
        wheel_path,
        {
            macos_source_member: 'template = "/Users/developer/private/template.tif"\n',
            windows_source_member: (
                r'template = "C:\Users\developer\private\template.tif"' + "\n"
            ),
        },
    )

    assert validate_wheel_deployment(wheel_path) == (
        f"{macos_source_member}: wheel contains developer-home paths: "
        "/Users/developer",
        f"{windows_source_member}: wheel contains developer-home paths: "
        r"C:\Users\developer",
    )
