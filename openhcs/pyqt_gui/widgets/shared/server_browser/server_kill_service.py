"""Server kill execution service for ZMQ server browser."""

from __future__ import annotations

from collections.abc import Callable
from typing import Self

from pyqt_reactive.widgets.shared import KillOperationKind
from zmqruntime import EndpointShutdownMode, EndpointShutdownResult
from zmqruntime.queue_tracker import GlobalQueueTrackerRegistry

from openhcs.runtime.zmq_config import OpenHCSZMQConfig


class ServerKillService:
    """Performs server kill operations with explicit policy."""

    @classmethod
    def openhcs_default(cls, config: OpenHCSZMQConfig) -> Self:
        """Build the OpenHCS ZMQ kill service with the runtime dependencies."""

        from zmqruntime import ZMQClient

        return cls(
            shutdown_endpoint_fn=(
                lambda port, mode, cfg: ZMQClient.shutdown_endpoint_on_port(
                    port,
                    mode=mode,
                    config=cfg,
                )
            ),
            queue_tracker_registry_factory=GlobalQueueTrackerRegistry,
            config=config,
        )

    def __init__(
        self,
        shutdown_endpoint_fn: Callable[
            [int, EndpointShutdownMode, OpenHCSZMQConfig],
            EndpointShutdownResult,
        ],
        queue_tracker_registry_factory: Callable[[], GlobalQueueTrackerRegistry],
        config: OpenHCSZMQConfig,
    ) -> None:
        self._shutdown_endpoint = shutdown_endpoint_fn
        self._queue_tracker_registry_factory = queue_tracker_registry_factory
        self._config = config

    def kill_ports(
        self,
        *,
        ports: list[int],
        kind: KillOperationKind,
        log_info: Callable[..., None],
        log_warning: Callable[..., None],
        log_error: Callable[..., None],
        on_operation_succeeded: Callable[[int], None] | None = None,
        on_endpoint_terminated: Callable[[int], None] | None = None,
    ) -> tuple[bool, str]:
        failed_ports: list[int] = []
        registry = self._queue_tracker_registry_factory()

        for port in ports:
            try:
                log_info(
                    "Shutting down endpoint on port %s (mode=%s)",
                    port,
                    kind.shutdown_mode.value,
                )
                result = self._shutdown_endpoint(
                    port,
                    kind.shutdown_mode,
                    self._config,
                )
                if result.succeeded:
                    registry.remove_tracker(port)
                    if on_operation_succeeded is not None:
                        on_operation_succeeded(port)
                    if (
                        result.endpoint_terminated
                        and on_endpoint_terminated is not None
                    ):
                        on_endpoint_terminated(port)
                    continue

                log_warning(
                    "Endpoint shutdown failed on port %s (mode=%s)",
                    port,
                    kind.shutdown_mode.value,
                )
                failed_ports.append(port)
            except Exception as error:
                log_error(
                    "Error killing server on port %s (mode=%s): %s",
                    port,
                    kind.shutdown_mode.value,
                    error,
                )
                failed_ports.append(port)

        if failed_ports:
            return False, f"Failed to quit servers on ports: {failed_ports}"
        return True, kind.success_message
