"""Normal execution submission and completion polling for PyQt batch workflows."""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, TypeVar

from openhcs.core.orchestrator.orchestrator import OrchestratorState
from openhcs.pyqt_gui.widgets.shared.services.compile_workflow_service import (
    CompileWorkflowService,
)
from openhcs.pyqt_gui.widgets.shared.services.execution_state import (
    TerminalExecutionStatus,
    parse_terminal_status,
)
from openhcs.pyqt_gui.widgets.shared.services.plate_pipeline_request_builder import (
    RunSpec,
)
from openhcs.pyqt_gui.widgets.shared.services.terminal_result_builder import (
    TerminalExecutionResultBuilder,
)
from openhcs.pyqt_gui.widgets.shared.services.zmq_client_service import ZMQClientService
from openhcs.runtime.zmq_execution_client import OpenHCSExecutionSubmission
from zmqruntime.execution import (
    CallbackExecutionStatusPollPolicy,
    ExecutionStatusPoller,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")


RunBlockingCallable = Callable[[object, Callable[[], T]], Any]
CompletionCallback = Callable[[], None]


class ExecutionSubmissionService:
    """Owns normal pipeline submission, execution id tracking, and polling."""

    def __init__(
        self,
        *,
        host,
        client_service: ZMQClientService,
        run_blocking: RunBlockingCallable,
        completion_poller: ExecutionStatusPoller,
        terminal_result_builder: TerminalExecutionResultBuilder,
        on_completion_update: CompletionCallback,
    ) -> None:
        self._host = host
        self._client_service = client_service
        self._run_blocking = run_blocking
        self._completion_poller = completion_poller
        self._terminal_result_builder = terminal_result_builder
        self._on_completion_update = on_completion_update

    async def submit_plate(
        self,
        *,
        run_spec: RunSpec,
        compile_artifact_id: str,
        loop,
    ) -> None:
        if self._client_service.zmq_client is None:
            raise RuntimeError("ZMQ client is not connected")
        plate_path = run_spec.plate_path
        execution_plate_path = run_spec.execution_plate_path
        definition_pipeline = run_spec.definition_pipeline
        logger.info("Executing plate: %s", plate_path)
        logger.info(
            "Submit run: plate=%s execution_plate=%s artifact_id=%s steps=%d fingerprint=%s",
            plate_path,
            execution_plate_path,
            compile_artifact_id,
            len(definition_pipeline),
            CompileWorkflowService.pipeline_fingerprint(definition_pipeline),
        )

        def submit() -> Dict[str, Any]:
            return self._client_service.zmq_client.submit_pipeline(
                OpenHCSExecutionSubmission(
                    plate_id=plate_path,
                    execution_plate_id=execution_plate_path,
                    selected_pipeline_path=run_spec.selected_pipeline_path,
                    pipeline_steps=definition_pipeline,
                    global_config=run_spec.global_config,
                    pipeline_config=run_spec.pipeline_config,
                    compile_artifact_id=compile_artifact_id,
                )
            )

        response = await self._run_blocking(loop, submit)
        execution_id = response.get("execution_id")
        if execution_id:
            self._host.plate_execution_ids[plate_path] = execution_id
            self._host.current_execution_id = execution_id

        if response.get("status") == "accepted":
            self._host.emit_status(f"Submitted {plate_path} (queued on server)")
            if execution_id:
                self.start_completion_poller(str(execution_id), plate_path)
            return

        error_msg = response.get("message", "Unknown error")
        logger.error("Plate %s submission failed: %s", plate_path, error_msg)
        self._host.emit_error(f"Submission failed for {plate_path}: {error_msg}")
        self._host.execution_runtime.mark_terminal(
            plate_path,
            TerminalExecutionStatus.FAILED,
        )
        self._set_orchestrator_exec_failed(plate_path)

    def start_completion_poller(self, execution_id: str, plate_path: str) -> None:
        """Start background status polling for one submitted execution."""

        class _ClientDisconnected(RuntimeError):
            pass

        def poll_status(polled_execution_id: str) -> Dict[str, Any]:
            if self._client_service.zmq_client is None:
                raise _ClientDisconnected("ZMQ client disconnected")
            return self._client_service.zmq_client.get_status(polled_execution_id)

        def on_running(_execution_id: str, _execution_payload: Dict[str, Any]) -> None:
            self._host.update_item_list()
            self._host.emit_status(f"▶️ Running {plate_path}")

        def on_terminal(
            terminal_execution_id: str,
            terminal_status: str,
            execution_payload: Dict[str, Any],
        ) -> None:
            current_execution_id = self._host.plate_execution_ids.get(plate_path)
            if current_execution_id != terminal_execution_id:
                logger.info(
                    "Ignoring stale terminal status for %s: execution_id=%s current=%s",
                    plate_path,
                    terminal_execution_id,
                    current_execution_id,
                )
                return
            parsed_terminal_status = parse_terminal_status(terminal_status)

            self._host.execution_runtime.mark_terminal(
                plate_path,
                parsed_terminal_status,
            )
            result = self._terminal_result_builder.build(
                terminal_status=parsed_terminal_status.value,
                execution_id=terminal_execution_id,
                execution_payload=execution_payload,
            )
            self._host.notify_plate_completed(
                plate_path,
                parsed_terminal_status.value,
                result,
            )
            self._on_completion_update()

        def on_status_error(execution_id_with_error: str, message: str) -> None:
            current_execution_id = self._host.plate_execution_ids.get(plate_path)
            if current_execution_id != execution_id_with_error:
                logger.info(
                    "Ignoring stale status error for %s: execution_id=%s current=%s",
                    plate_path,
                    execution_id_with_error,
                    current_execution_id,
                )
                return
            self._host.execution_runtime.mark_terminal(
                plate_path,
                TerminalExecutionStatus.FAILED,
            )
            self._host.notify_plate_completed(
                plate_path,
                TerminalExecutionStatus.FAILED.value,
                {
                    "status": TerminalExecutionStatus.FAILED.value,
                    "execution_id": execution_id_with_error,
                    "message": message,
                },
            )
            self._on_completion_update()

        def on_poll_exception(_execution_id: str, error: Exception) -> bool:
            if isinstance(error, _ClientDisconnected):
                return False
            logger.warning("Error polling status for %s: %s", plate_path, error)
            return True

        policy = CallbackExecutionStatusPollPolicy(
            poll_status_fn=poll_status,
            poll_interval_seconds_value=0.5,
            on_running_fn=on_running,
            on_terminal_fn=on_terminal,
            on_status_error_fn=on_status_error,
            on_poll_exception_fn=on_poll_exception,
        )

        def poll_completion() -> None:
            try:
                self._completion_poller.run(execution_id, policy)
            except Exception as error:
                logger.error(
                    "Error in completion poller for %s: %s",
                    plate_path,
                    error,
                    exc_info=True,
                )
                self._host.emit_error(f"{plate_path}: {error}")

        threading.Thread(target=poll_completion, daemon=True).start()

    def _set_orchestrator_exec_failed(self, plate_path: str) -> None:
        from objectstate import ObjectStateRegistry

        orchestrator = ObjectStateRegistry.get_object(plate_path)
        if orchestrator is None:
            return
        orchestrator._state = OrchestratorState.EXEC_FAILED
        self._host.emit_orchestrator_state(
            plate_path,
            OrchestratorState.EXEC_FAILED.value,
        )


def is_execution_submission_service_export(name: str, value: object) -> bool:
    return (
        isinstance(value, type)
        and value.__module__ == __name__
        and not name.startswith("_")
    )


__all__ = tuple(
    name
    for name, value in globals().items()
    if is_execution_submission_service_export(name, value)
)
