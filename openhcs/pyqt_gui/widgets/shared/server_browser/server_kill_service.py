"""Server kill execution service for ZMQ server browser."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple, Self


@dataclass(frozen=True)
class ServerKillPlan:
    """Execution policy for killing selected server ports."""

    graceful: bool
    strict_failures: bool
    emit_signal_on_failure: bool
    success_message: str


class ServerKillService:
    """Performs server kill operations with explicit policy."""

    @classmethod
    def openhcs_default(cls) -> Self:
        """Build the OpenHCS ZMQ kill service with the runtime dependencies."""

        from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG
        from zmqruntime.client import ZMQClient
        from zmqruntime.queue_tracker import GlobalQueueTrackerRegistry

        return cls(
            kill_server_fn=lambda port, graceful, cfg: ZMQClient.kill_server_on_port(
                port, graceful=graceful, config=cfg
            ),
            queue_tracker_registry_factory=GlobalQueueTrackerRegistry,
            config=OPENHCS_ZMQ_CONFIG,
        )

    def __init__(
        self,
        kill_server_fn: Callable[[int, bool, object], bool],
        queue_tracker_registry_factory: Callable[[], object],
        config: object,
    ) -> None:
        self._kill_server_fn = kill_server_fn
        self._queue_tracker_registry_factory = queue_tracker_registry_factory
        self._config = config

    def kill_ports(
        self,
        *,
        ports: List[int],
        plan: ServerKillPlan,
        on_server_killed: Callable[[int], None],
        log_info: Callable[..., None],
        log_warning: Callable[..., None],
        log_error: Callable[..., None],
    ) -> Tuple[bool, str]:
        failed_ports: List[int] = []
        registry = self._queue_tracker_registry_factory()

        for port in ports:
            try:
                log_info("Killing server on port %s (graceful=%s)", port, plan.graceful)
                success = self._kill_server_fn(port, plan.graceful, self._config)
                if success:
                    registry.remove_tracker(port)
                    on_server_killed(port)
                    continue

                log_warning(
                    "kill_server_on_port returned False for port %s (graceful=%s)",
                    port,
                    plan.graceful,
                )
                if plan.strict_failures:
                    failed_ports.append(port)
                if plan.emit_signal_on_failure:
                    registry.remove_tracker(port)
                    on_server_killed(port)
            except Exception as error:
                log_error(
                    "Error killing server on port %s (graceful=%s): %s",
                    port,
                    plan.graceful,
                    error,
                )
                if plan.strict_failures:
                    failed_ports.append(port)
                if plan.emit_signal_on_failure:
                    registry.remove_tracker(port)
                    on_server_killed(port)

        if plan.strict_failures and failed_ports:
            return False, f"Failed to quit servers on ports: {failed_ports}"
        return True, plan.success_message
