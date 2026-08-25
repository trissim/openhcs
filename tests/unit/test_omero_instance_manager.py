from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml
from polystore.omero_tables import OMEROTableServiceUnavailableError

from openhcs.runtime.omero_instance_manager import OMEROInstanceManager


def _write_compose(path: Path) -> None:
    path.write_text(
        """\
x-openhcs-connection:
  host: localhost
  port: 4064
  web_port: 4080
  user: root
  password: openhcs
services: {}
""",
        encoding="utf-8",
    )


def _install_gateway_module(monkeypatch, gateway: object) -> None:
    gateway_module = ModuleType("omero.gateway")
    gateway_module.BlitzGateway = lambda *args, **kwargs: gateway
    omero_module = ModuleType("omero")
    omero_module.gateway = gateway_module
    monkeypatch.setitem(sys.modules, "omero", omero_module)
    monkeypatch.setitem(sys.modules, "omero.gateway", gateway_module)


def test_default_compose_declaration_is_a_packaged_resource() -> None:
    manager = OMEROInstanceManager()

    assert manager.docker_compose_path is not None
    assert manager.docker_compose_path.name == "docker-compose.yml"
    assert manager.docker_compose_path.parent.name == "omero"
    assert tuple(manager.docker_compose_path.parent.glob("docker-compose*.yml")) == (
        manager.docker_compose_path,
    )


def test_packaged_compose_pins_validated_server_releases() -> None:
    manager = OMEROInstanceManager()
    assert manager.docker_compose_path is not None

    compose = yaml.safe_load(manager.docker_compose_path.read_text(encoding="utf-8"))

    assert compose["services"]["database"]["image"] == "postgres:16.15"
    assert (
        compose["services"]["omeroserver"]["image"]
        == "openmicroscopy/omero-server:5.6.18"
    )
    assert (
        compose["services"]["omeroweb"]["image"]
        == "openmicroscopy/omero-web-standalone:5.33.0"
    )
    assert "build" not in compose["services"]["omeroweb"]
    connection = compose["x-openhcs-connection"]
    assert manager.host == connection["host"]
    assert manager.port == connection["port"]
    assert manager.web_port == connection["web_port"]
    assert manager.user == connection["user"]
    assert manager.password == connection["password"]
    assert compose["services"]["omeroserver"]["environment"]["ROOTPASS"] == (
        connection["password"]
    )
    assert compose["services"]["omeroserver"]["ports"][1]["published"] == (
        connection["port"]
    )
    assert compose["services"]["omeroweb"]["ports"][0]["published"] == (
        connection["web_port"]
    )


def test_explicit_connection_settings_override_local_deployment() -> None:
    manager = OMEROInstanceManager(
        host="omero.example.org",
        port=14064,
        web_port=14080,
        user="analyst",
        password="secret",
    )

    assert manager.host == "omero.example.org"
    assert manager.port == 14064
    assert manager.web_port == 14080
    assert manager.user == "analyst"
    assert manager.password == "secret"


def test_docker_command_prefers_the_callers_direct_access(monkeypatch) -> None:
    commands: list[list[str]] = []

    def run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("openhcs.runtime.omero_instance_manager.subprocess.run", run)

    assert OMEROInstanceManager()._docker_command() == ("docker",)
    assert commands == [["docker", "info"]]


def test_docker_command_uses_noninteractive_sudo_when_required(monkeypatch) -> None:
    commands: list[list[str]] = []

    def run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=len(commands) - 2)

    monkeypatch.setattr("openhcs.runtime.omero_instance_manager.subprocess.run", run)

    assert OMEROInstanceManager()._docker_command() == ("sudo", "-n", "docker")
    assert commands == [["docker", "info"], ["sudo", "-n", "docker", "info"]]


def test_failed_compose_up_reports_startup_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    compose_path = tmp_path / "docker-compose.yml"
    _write_compose(compose_path)
    commands: list[list[str]] = []

    def run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr("openhcs.runtime.omero_instance_manager.subprocess.run", run)
    manager = OMEROInstanceManager(docker_compose_path=compose_path)
    monkeypatch.setattr(manager, "_docker_command", lambda: ("docker",))

    assert manager._start_omero_docker() is False
    assert commands == [
        [
            "docker",
            "compose",
            "--file",
            str(compose_path),
            "up",
            "-d",
        ]
    ]


def test_context_manager_rejects_an_unavailable_stack(monkeypatch) -> None:
    manager = OMEROInstanceManager()
    monkeypatch.setattr(manager, "connect", lambda: False)

    with pytest.raises(ConnectionError, match="OMERO stack is unavailable"):
        manager.__enter__()


def test_missing_docker_fails_without_platform_process_side_effects(
    monkeypatch,
) -> None:
    manager = OMEROInstanceManager()
    started: list[bool] = []
    monkeypatch.setattr(manager, "is_omero_stack_running", lambda: False)
    monkeypatch.setattr(manager, "_is_docker_running", lambda: False)
    monkeypatch.setattr(
        manager,
        "_start_omero_docker",
        lambda: started.append(True) or True,
    )

    assert manager.connect() is False
    assert started == []


def test_stop_targets_the_same_compose_declaration(monkeypatch, tmp_path: Path) -> None:
    compose_path = tmp_path / "docker-compose.yml"
    _write_compose(compose_path)
    commands: list[list[str]] = []

    def run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("openhcs.runtime.omero_instance_manager.subprocess.run", run)
    manager = OMEROInstanceManager(docker_compose_path=compose_path)
    monkeypatch.setattr(manager, "_docker_command", lambda: ("docker",))

    manager.stop_omero_docker()

    assert commands == [["docker", "compose", "--file", str(compose_path), "down"]]


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
    _install_gateway_module(monkeypatch, gateway)
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

    _install_gateway_module(monkeypatch, gateway)
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
