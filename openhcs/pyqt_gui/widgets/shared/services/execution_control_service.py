"""Execution stop, disconnect, and failure handling for batch workflows."""

from __future__ import annotations

import logging
import threading

from zmqruntime import EndpointShutdownMode
from zmqruntime.shutdown import EndpointShutdownService

from openhcs.constants.constants import OrchestratorState
from openhcs.core.execution_state import (
    ManagerExecutionState,
    TerminalExecutionStatus,
)
from openhcs.pyqt_gui.widgets.shared.services.batch_context import (
    BatchWorkflowContext,
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
        endpoint_shutdown_service: EndpointShutdownService,
    ) -> None:
        self._host = host
        self._context = context
        self._port = port
        self._endpoint_shutdown_service = endpoint_shutdown_service

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
            endpoint_shutdown_service=EndpointShutdownService.for_config(config),
        )

    def check_all_completed(self) -> None:
        if not self._host.execution_state.busy:
            return
        if not self._host.plate_terminal_activity_status.all_batch_terminal():
            return
        completed, failed = self._host.plate_terminal_activity_status.terminal_counts()
        self._host.notify_all_plates_completed(completed, failed)

    async def handle_execution_failure(self, loop) -> None:
        from objectstate import ObjectStateRegistry

        for plate_path in tuple(
            self._host.plate_terminal_activity_status.active_plates
        ):
            self._host.plate_terminal_activity_status.mark_terminal(
                plate_path, TerminalExecutionStatus.FAILED
            )
            orchestrator = ObjectStateRegistry.get_object(plate_path)
            if orchestrator is not None:
                orchestrator._state = OrchestratorState.EXEC_FAILED
                self._host.emit_orchestrator_state(
                    plate_path,
                    OrchestratorState.EXEC_FAILED,
                )

        self._host.execution_state = ManagerExecutionState.IDLE
        await self.disconnect_client()
        self.refresh_host_execution_ui()

    async def disconnect_client(self) -> None:
        """Retire established and in-progress client ownership."""

        try:
            await self._context.zmq.disconnect()
        except Exception as error:
            logger.warning("Error disconnecting old client: %s", error)

    def stop_execution(self, force: bool = False) -> None:
        port = self._port

        def kill_server() -> None:
            try:
                result = self._endpoint_shutdown_service.shutdown_ports(
                    ports=[port],
                    mode=EndpointShutdownMode.from_force(force),
                )
                if not result.succeeded:
                    if self._host.execution_state.suppresses_stop_failure:
                        logger.info(
                            "Suppressing stale stop failure while stop is already terminalizing: %s",
                            result.failure_message,
                        )
                        self.emit_cancelled_for_all_plates()
                        return
                    self._host.emit_error(result.failure_message)
                    return
                self.emit_cancelled_for_all_plates()
            except Exception as error:
                logger.error("Error stopping server: %s", error)
                self._host.emit_error(f"Error stopping execution: {error}")

        threading.Thread(target=kill_server, daemon=True).start()

        if force:
            self.emit_cancelled_for_all_plates()
            self.disconnect_async()

    def emit_cancelled_for_all_plates(self) -> None:
        for (
            plate_path
        ) in self._host.plate_terminal_activity_status.cancellable_plates():
            self._host.emit_execution_complete(
                TerminalExecutionStatus.CANCELLED.completion_payload(
                    execution_id=(
                        self._host.plate_terminal_activity_status.execution_id(
                            plate_path
                        )
                    ),
                    execution_payload={},
                ),
                plate_path,
            )

    def disconnect(self) -> None:
        """Retire established and in-progress client ownership."""

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
