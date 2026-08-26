"""Ownership checks for the installed GUI startup smoke test."""

from pathlib import Path

import pytest
from zmqruntime import DataControlPortPair, DataControlPortPairAuthority, TransportMode

from openhcs.pyqt_gui.config import UIConfig
from openhcs.runtime.zmq_config import OpenHCSZMQConfig
from scripts.smoke_installed_gui import (
    assert_not_source_checkout_import,
    with_isolated_execution_endpoint,
)


def test_gui_smoke_rejects_source_package(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"

    with pytest.raises(AssertionError, match="source checkout instead of the wheel"):
        assert_not_source_checkout_import(
            package_path=checkout / "openhcs" / "__init__.py",
            forbidden_root=checkout,
        )


def test_gui_smoke_allows_wheel_venv_inside_checkout(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    site_packages = checkout / "test_gui" / "lib" / "site-packages"

    assert_not_source_checkout_import(
        package_path=site_packages / "openhcs" / "__init__.py",
        forbidden_root=checkout,
    )


def test_gui_smoke_allocates_endpoint_through_transport_authority(monkeypatch) -> None:
    original_transport = OpenHCSZMQConfig(
        default_port=8123,
        transport_mode=TransportMode.TCP,
    )
    ui_config = UIConfig(zmq=original_transport)
    observed: dict[str, object] = {}

    def acquire(config, *, transport_mode, host):
        observed.update(
            config=config,
            transport_mode=transport_mode,
            host=host,
        )
        return DataControlPortPair(data_port=8124, control_port=9124)

    monkeypatch.setattr(
        DataControlPortPairAuthority,
        "acquire",
        staticmethod(acquire),
    )

    isolated = with_isolated_execution_endpoint(ui_config)

    assert isolated.zmq == OpenHCSZMQConfig(
        default_port=8124,
        transport_mode=TransportMode.TCP,
    )
    assert ui_config.zmq is original_transport
    assert observed == {
        "config": original_transport,
        "transport_mode": TransportMode.TCP,
        "host": original_transport.client_host,
    }
