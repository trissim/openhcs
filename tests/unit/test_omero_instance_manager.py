from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml
from polystore.omero_tables import OMEROTableServiceUnavailableError

from openhcs.runtime.omero_instance_manager import OMEROInstanceManager


def test_default_compose_declaration_is_a_packaged_resource() -> None:
    manager = OMEROInstanceManager()

    assert manager.docker_compose_path is not None
    assert manager.docker_compose_path.name == "docker-compose.yml"
    assert manager.docker_compose_path.parent.name == "omero"


def test_packaged_compose_pins_validated_server_releases() -> None:
    manager = OMEROInstanceManager()
    assert manager.docker_compose_path is not None

    compose = yaml.safe_load(manager.docker_compose_path.read_text(encoding="utf-8"))

    assert compose["services"]["database"]["image"] == "postgres:16.15"
    assert (
        compose["services"]["omeroserver"]["image"]
        == "openmicroscopy/omero-server:5.6.18"
    )


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


class _ConnectedGateway:
    def __init__(self) -> None:
        self.closed = False

    def connect(self) -> bool:
        return True

    def close(self) -> None:
        self.closed = True

    def getEventContext(self) -> object:
        return object()


def test_connection_requires_declared_table_service(monkeypatch) -> None:
    gateway = _ConnectedGateway()
    observed_connections = []
    table_service = SimpleNamespace(
        wait_until_available=observed_connections.append,
    )
    monkeypatch.setattr("omero.gateway.BlitzGateway", lambda *args, **kwargs: gateway)
    monkeypatch.setattr(
        "openhcs.runtime.omero_instance_manager.OMERO_TABLE_SERVICE",
        table_service,
    )
    manager = OMEROInstanceManager()

    assert manager._connect_to_omero() is True
    assert manager.conn is gateway
    assert observed_connections == [gateway]


def test_connection_rejects_unavailable_table_service(monkeypatch) -> None:
    gateway = _ConnectedGateway()

    def reject_table_service(_connection) -> None:
        raise OMEROTableServiceUnavailableError("not ready")

    monkeypatch.setattr("omero.gateway.BlitzGateway", lambda *args, **kwargs: gateway)
    monkeypatch.setattr(
        "openhcs.runtime.omero_instance_manager.OMERO_TABLE_SERVICE",
        SimpleNamespace(wait_until_available=reject_table_service),
    )
    manager = OMEROInstanceManager()

    assert manager._connect_to_omero() is False
    assert manager.conn is None
    assert gateway.closed is True


def test_existing_connection_revalidates_declared_table_service(monkeypatch) -> None:
    gateway = _ConnectedGateway()
    observed_connections = []
    monkeypatch.setattr(
        "openhcs.runtime.omero_instance_manager.OMERO_TABLE_SERVICE",
        SimpleNamespace(wait_until_available=observed_connections.append),
    )
    manager = OMEROInstanceManager()
    manager.conn = gateway

    assert manager.connect() is True
    assert observed_connections == [gateway]
