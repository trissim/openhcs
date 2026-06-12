"""Execution stop, disconnect, and failure handling for batch workflows."""

from __future__ import annotations

import logging
import threading

from openhcs.constants.constants import OrchestratorState
from openhcs.pyqt_gui.widgets.shared.server_browser import (
    ServerKillPlan,
    ServerKillService,
)
from openhcs.pyqt_gui.widgets.shared.services.execution_state import (
    ManagerExecutionState,
    STOP_PENDING_MANAGER_STATES,
    TerminalExecutionStatus,
)
from openhcs.pyqt_gui.widgets.shared.services.zmq_client_service import ZMQClientService

logger = logging.getLogger(__name__)


class ExecutionControlService:
    """Owns execution cancellation, failure terminalization, and client teardown."""

    def __init__(
        self,
        *,
        host,
        client_service: ZMQClientService,
        port: int,
        server_kill_service: ServerKillService,
    ) -> None:
        self._host = host
        self._client_service = client_service
        self._port = port
        self._server_kill_service = server_kill_service

    @classmethod
    def openhcs_default(
        cls,
        *,
        host,
        client_service: ZMQClientService,
        port: int,
    ) -> "ExecutionControlService":
        return cls(
            host=host,
            client_service=client_service,
            port=port,
            server_kill_service=ServerKillService.openhcs_default(),
        )

    def check_all_completed(self) -> None:
        if self._host.execution_state not in (
            ManagerExecutionState.RUNNING,
            *STOP_PENDING_MANAGER_STATES,
        ):
            return
        if not self._host.execution_runtime.all_batch_terminal():
            return
        completed, failed = self._host.execution_runtime.terminal_counts()
        self._host.notify_all_plates_completed(completed, failed)

    async def handle_execution_failure(self, loop) -> None:
        from objectstate import ObjectStateRegistry

        for plate_path in tuple(self._host.execution_runtime.active_plates):
            self._host.execution_runtime.mark_terminal(
                plate_path, TerminalExecutionStatus.FAILED
            )
            orchestrator = ObjectStateRegistry.get_object(plate_path)
            if orchestrator is not None:
                orchestrator._state = OrchestratorState.EXEC_FAILED
                self._host.emit_orchestrator_state(
                    plate_path, OrchestratorState.EXEC_FAILED.value
                )

        self._host.execution_state = ManagerExecutionState.IDLE
        await self.disconnect_client(loop)
        self._host.current_execution_id = None
        self.refresh_host_execution_ui()

    async def disconnect_client(self, loop) -> None:
        if self._client_service.zmq_client is None:
            return
        try:
            await self._client_service.disconnect()
        except Exception as error:
            logger.warning("Error disconnecting old client: %s", error)

    def stop_execution(self, force: bool = False) -> None:
        port = self._port

        def kill_server() -> None:
            try:
                plan = ServerKillPlan(
                    graceful=not force,
                    strict_failures=not force,
                    emit_signal_on_failure=force,
                    success_message=f"Stopped execution server on port {port}",
                )
                success, message = self._server_kill_service.kill_ports(
                    ports=[port],
                    plan=plan,
                    on_server_killed=lambda _port: self.emit_cancelled_for_all_plates(),
                    log_info=logger.info,
                    log_warning=logger.warning,
                    log_error=logger.error,
                )
                if not success:
                    if self._host.execution_state.suppresses_stop_failure:
                        logger.info(
                            "Suppressing stale stop failure while stop is already terminalizing: %s",
                            message,
                        )
                        self.emit_cancelled_for_all_plates()
                        return
                    self._host.emit_error(message)
                    return
            except Exception as error:
                logger.error("Error stopping server: %s", error)
                self._host.emit_error(f"Error stopping execution: {error}")

        threading.Thread(target=kill_server, daemon=True).start()

        if force:
            self.emit_cancelled_for_all_plates()
            self.disconnect_async()

    def emit_cancelled_for_all_plates(self) -> None:
        for plate_path in self._host.execution_runtime.cancellable_plates():
            self._host.emit_execution_complete(
                {"status": TerminalExecutionStatus.CANCELLED.value}, plate_path
            )

    def disconnect(self) -> None:
        if self._client_service.zmq_client is None:
            return
        try:
            self._client_service.disconnect_sync()
        except Exception as error:
            logger.warning("Error disconnecting ZMQ client: %s", error)

    def disconnect_async(self) -> None:
        def _disconnect() -> None:
            self.disconnect()

        threading.Thread(target=_disconnect, daemon=True).start()

    def refresh_host_execution_ui(self) -> None:
        refresh_fn = getattr(self._host, "refresh_execution_ui", None)
        if callable(refresh_fn):
            refresh_fn()
            return
        self._host.update_item_list()
        self._host.update_button_states()


def is_execution_control_service_export(name: str, value: object) -> bool:
    return (
        isinstance(value, type)
        and value.__module__ == __name__
        and not name.startswith("_")
    )


__all__ = tuple(
    name
    for name, value in globals().items()
    if is_execution_control_service_export(name, value)
)
