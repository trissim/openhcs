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
