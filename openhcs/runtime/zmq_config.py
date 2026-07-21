"""OpenHCS execution transport configuration."""
from __future__ import annotations

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


OPENHCS_ZMQ_CONFIG = OpenHCSZMQConfig()
