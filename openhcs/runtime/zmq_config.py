"""OpenHCS execution transport configuration."""

from __future__ import annotations

from dataclasses import dataclass

from zmqruntime import ZMQConfig
from zmqruntime.config import (
    NonBlankString,
    PositiveFloat,
    PositiveInteger,
    TransportMode,
)
from zmqruntime.transport import get_default_transport_mode


@dataclass(frozen=True, slots=True)
class OpenHCSZMQConfig(ZMQConfig):
    """Process-level execution transport topology and lifecycle defaults."""

    ipc_socket_prefix: NonBlankString = "openhcs-zmq"
    """OpenHCS namespace prefix for generated IPC data and control sockets."""

    app_name: NonBlankString = "openhcs"
    """OpenHCS application namespace used in transport identities and runtime paths."""

    client_host: NonBlankString = "localhost"
    """Execution-server hostname used by clients in TCP mode.

    IPC mode is local and ignores this network hostname.
    """

    server_host: NonBlankString = "*"
    """Network interface bound by execution servers in TCP mode.

    ``*`` accepts connections on every interface; use a loopback address to
    restrict the server to this machine. IPC mode ignores this field.
    """

    transport_mode: TransportMode = get_default_transport_mode()
    """Transport used for execution data and control sockets.

    IPC is local-only and owns filesystem socket cleanup. TCP uses
    ``client_host`` and ``server_host`` and can cross machine boundaries.
    """

    persistent: bool = True
    """Reuse an execution client's open sockets across successive operations.

    Disable this when each request should create and close its own connection;
    it does not control execution-server lifetime.
    """

    control_timeout_ms: PositiveInteger = 5000
    """Default deadline for execution control requests such as ping, submit, and stop.

    Individual agent or client requests may supply a narrower or wider timeout.
    """

    server_info_timeout_ms: PositiveInteger = 500
    """Deadline for retrieving the typed descriptor of one discovered server endpoint."""

    server_scan_timeout_ms: PositiveInteger = 200
    """Per-endpoint heartbeat deadline used while scanning for execution servers.

    Larger values tolerate slow hosts but multiply the duration of a full port
    scan.
    """

    client_connect_timeout_seconds: PositiveFloat = 15.0
    """Inactivity deadline while a client waits for an execution endpoint to become ready.

    Reported startup activity resets this deadline, allowing slow registry
    preparation to continue while genuinely stalled launches still fail.
    """

    server_poll_interval_seconds: PositiveFloat = 0.01
    """Delay between endpoint readiness probes during execution-server startup."""

    ports_per_server_type: PositiveInteger = 10
    """Consecutive candidate data ports reserved and scanned for each server role.

    Increasing this permits more concurrent role instances but makes discovery
    inspect more endpoints.
    """

    compiled_artifact_ttl_seconds: PositiveFloat = 30.0 * 60.0
    """Retention time for server-side compiled-artifact inspection records.

    Expired records are discarded from the execution server's inspection cache;
    this does not delete pipeline outputs or compiled bundles saved to disk.
    """


OPENHCS_ZMQ_CONFIG = OpenHCSZMQConfig()
