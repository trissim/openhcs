"""Execution stop, disconnect, and failure handling for batch workflows."""

from __future__ import annotations

import logging
import threading

from openhcs.constants.constants import OrchestratorState
from openhcs.pyqt_gui.widgets.shared.services.batch_context import (
    BatchWorkflowContext,
)
from openhcs.pyqt_gui.widgets.shared.server_browser import (
    ServerKillPlan,
    ServerKillService,
)
from openhcs.pyqt_gui.widgets.shared.services.execution_state import (
    ManagerExecutionState,
    STOP_PENDING_MANAGER_STATES,
    TerminalExecutionStatus,
)

logger = logging.getLogger(__name__)


class ExecutionControlService:
    """Owns execution cancellation, failure terminalization, and client teardown."""

    def __init__(
        self,
        *,
        host,
        context: BatchWorkflowContext,
        port: int,
        server_kill_service: ServerKillService,
    ) -> None:
        self._host = host
        self._context = context
        self._port = port
        self._server_kill_service = server_kill_service

    @classmethod
    def openhcs_default(
        cls,
        *,
        host,
        context: BatchWorkflowContext,
        port: int,
        config,
    ) -> "ExecutionControlService":
        return cls(
            host=host,
            context=context,
            port=port,
            server_kill_service=ServerKillService.openhcs_default(config),
        )

    def check_all_completed(self) -> None:
        if self._host.execution_state not in (
            ManagerExecutionState.RUNNING,
            *STOP_PENDING_MANAGER_STATES,
        ):
            return
        if not self._host.plate_terminal_activity_status.all_batch_terminal():
            return
        completed, failed = self._host.plate_terminal_activity_status.terminal_counts()
        self._host.notify_all_plates_completed(completed, failed)

    async def handle_execution_failure(self, loop) -> None:
        from objectstate import ObjectStateRegistry

        for plate_path in tuple(self._host.plate_terminal_activity_status.active_plates):
            self._host.plate_terminal_activity_status.mark_terminal(
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
        if not self._context.zmq.has_client():
            return
        try:
            await self._context.zmq.disconnect()
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
        for plate_path in self._host.plate_terminal_activity_status.cancellable_plates():
            self._host.emit_execution_complete(
                {"status": TerminalExecutionStatus.CANCELLED.value}, plate_path
            )

    def disconnect(self) -> None:
        if not self._context.zmq.has_client():
            return
        try:
            self._context.zmq.disconnect_sync()
        except Exception as error:
            logger.warning("Error disconnecting ZMQ client: %s", error)

    def disconnect_async(self) -> None:
        def _disconnect() -> None:
            self.disconnect()

        threading.Thread(target=_disconnect, daemon=True).start()

    def refresh_host_execution_ui(self) -> None:
        self._host.refresh_execution_ui()


def is_execution_control_service_export(name: str, value) -> bool:
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
