from __future__ import annotations

import socket

from zmqruntime import transport_modes
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG
from zmqruntime import TcpDataControlPortPairAuthority


def test_tcp_port_pair_authority_returns_free_configured_pair() -> None:
    assert TcpDataControlPortPairAuthority.__module__ == "zmqruntime.transport"

    pair = TcpDataControlPortPairAuthority.acquire(OPENHCS_ZMQ_CONFIG)

    assert pair.control_port == (
        pair.data_port + OPENHCS_ZMQ_CONFIG.control_port_offset
    )
    assert pair.ports == frozenset((pair.data_port, pair.control_port))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as data_socket:
        data_socket.bind(("127.0.0.1", pair.data_port))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as control_socket:
        control_socket.bind(("127.0.0.1", pair.control_port))


def test_tcp_port_pair_authority_scans_both_ports_together(monkeypatch) -> None:
    first_port = OPENHCS_ZMQ_CONFIG.default_port
    control_offset = OPENHCS_ZMQ_CONFIG.control_port_offset
    attempted_ports: list[int] = []

    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def bind(self, address) -> None:
            port = int(address[1])
            attempted_ports.append(port)
            if port == first_port + control_offset:
                raise OSError("simulated reserved Windows control port")

    monkeypatch.setattr(transport_modes.socket, "socket", lambda *_args: FakeSocket())

    pair = TcpDataControlPortPairAuthority.acquire(OPENHCS_ZMQ_CONFIG)

    assert pair.data_port == first_port + 1
    assert pair.control_port == first_port + 1 + control_offset
    assert attempted_ports == (
        [first_port, first_port + control_offset]
        + [first_port + 1, first_port + 1 + control_offset]
    )
