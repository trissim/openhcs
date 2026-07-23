"""OpenHCS execution transport configuration."""

from __future__ import annotations

import socket
from collections.abc import Collection
from dataclasses import dataclass

from zmqruntime import ZMQConfig
from zmqruntime.config import TransportMode
from zmqruntime.transport import get_default_transport_mode


@dataclass(frozen=True, slots=True)
class OpenHCSZMQConfig(ZMQConfig):
    """Process-level execution transport topology and lifecycle defaults."""

    control_port_offset: int = 1000
    """Offset added to a data port to derive its control port."""
    default_port: int = 7777
    """Default execution-server data port."""
    ipc_socket_dir: str = "ipc"
    """Directory for IPC socket files, relative to the runtime socket root unless absolute."""
    ipc_socket_prefix: str = "openhcs-zmq"
    """Filename prefix for generated IPC sockets."""
    ipc_socket_extension: str = ".sock"
    """Filename extension for generated IPC sockets."""
    shared_ack_port: int = 7555
    """Shared acknowledgement port used by streaming clients and servers."""
    app_name: str = "openhcs"
    """Application namespace used in generated transport identities and paths."""
    client_host: str = "localhost"
    """Host clients connect to when TCP transport is selected."""
    server_host: str = "*"
    """Interface servers bind to when TCP transport is selected; * binds all interfaces."""
    transport_mode: TransportMode = get_default_transport_mode()
    """Execution transport mode. IPC is local-only; TCP supports network hosts."""
    persistent: bool = True
    """Keep execution connections open for reuse across operations."""
    control_timeout_ms: int = 5000
    """Default timeout in milliseconds for execution control requests."""
    server_info_timeout_ms: int = 500
    """Timeout in milliseconds for querying one server descriptor."""
    server_scan_timeout_ms: int = 200
    """Per-port timeout in milliseconds while discovering execution servers."""
    progress_connect_timeout_seconds: float = 1.0
    """Maximum seconds to wait when connecting progress subscribers."""
    client_connect_timeout_seconds: float = 15.0
    """Maximum seconds to wait for an execution client connection to become ready."""
    server_poll_interval_seconds: float = 0.01
    """Polling interval in seconds while waiting for a launched execution server."""
    ports_per_server_type: int = 10
    """Number of consecutive ports scanned for each server type. Must be at least one."""
    compiled_artifact_ttl_seconds: float = 30.0 * 60.0
    """Seconds compiled artifact inspection records remain available before expiry."""


@dataclass(frozen=True, slots=True)
class TcpDataControlPortPair:
    """One loopback TCP endpoint pair derived from a ZMQ configuration."""

    data_port: int
    control_port: int

    @property
    def ports(self) -> frozenset[int]:
        """Return both owned ports for subsequent allocation exclusion."""
        return frozenset((self.data_port, self.control_port))


class TcpDataControlPortPairAuthority:
    """Acquire free loopback TCP pairs without assuming ephemeral adjacency."""

    @staticmethod
    def acquire(
        config: ZMQConfig,
        *,
        excluded: Collection[int] = (),
        host: str = "127.0.0.1",
    ) -> TcpDataControlPortPair:
        first_port = config.default_port
        last_port = 65535 - config.control_port_offset
        for data_port in range(first_port, last_port + 1):
            control_port = data_port + config.control_port_offset
            if data_port in excluded or control_port in excluded:
                continue
            try:
                with (
                    socket.socket(socket.AF_INET, socket.SOCK_STREAM) as data_socket,
                    socket.socket(
                        socket.AF_INET,
                        socket.SOCK_STREAM,
                    ) as control_socket,
                ):
                    data_socket.bind((host, data_port))
                    control_socket.bind((host, control_port))
            except OSError:
                continue
            return TcpDataControlPortPair(
                data_port=data_port,
                control_port=control_port,
            )
        raise RuntimeError("Could not allocate a free TCP data/control port pair.")


OPENHCS_ZMQ_CONFIG = OpenHCSZMQConfig()
