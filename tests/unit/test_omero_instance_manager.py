from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from openhcs.runtime.omero_instance_manager import OMEROInstanceManager


def test_default_compose_declaration_is_a_packaged_resource() -> None:
    manager = OMEROInstanceManager()

    assert manager.docker_compose_path is not None
    assert manager.docker_compose_path.name == "docker-compose.yml"
    assert manager.docker_compose_path.parent.name == "omero"


def test_failed_required_web_build_stops_compose_startup(
    monkeypatch,
    tmp_path: Path,
) -> None:
    compose_path = tmp_path / "docker-compose.yml"
    compose_path.write_text("services: {}\n", encoding="utf-8")
    commands: list[list[str]] = []

    def run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr("openhcs.runtime.omero_instance_manager.subprocess.run", run)
    manager = OMEROInstanceManager(docker_compose_path=compose_path)

    assert manager._start_omero_docker() is False
    assert commands == [["sudo", "docker", "compose", "build"]]
