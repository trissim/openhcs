"""Normal execution submission and completion polling for PyQt batch workflows."""

from __future__ import annotations

import logging
import threading

from zmqruntime.execution import (
    CallbackExecutionStatusPollPolicy,
    ExecutionStatusPoller,
    ExecutionSubmissionResponse,
)
from zmqruntime.messages import MessageFields

from openhcs.core.execution_state import (
    TerminalExecutionStatus,
    parse_terminal_status,
)
from openhcs.core.orchestrator.orchestrator import OrchestratorState
from openhcs.pyqt_gui.widgets.shared.services.batch_context import (
    BatchWorkflowContext,
)
from openhcs.pyqt_gui.widgets.shared.services.compile_workflow_service import (
    CompileWorkflowService,
)
from openhcs.pyqt_gui.widgets.shared.services.plate_pipeline_request_builder import (
    RunSpec,
)

logger = logging.getLogger(__name__)


class ExecutionSubmissionService:
    """Owns normal pipeline submission, execution id tracking, and polling."""

    def __init__(
        self,
        *,
        host,
        context: BatchWorkflowContext,
        completion_poller: ExecutionStatusPoller,
    ) -> None:
        self._host = host
        self._context = context
        self._completion_poller = completion_poller

    async def submit_plate(
        self,
        *,
        run_spec: RunSpec,
        compile_artifact_id: str,
        loop,
    ) -> None:
        zmq_client = self._context.zmq.require_client()
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

        def submit() -> dict:
            return zmq_client.submit_pipeline(
                run_spec.submission(
                    global_config=run_spec.global_config,
                    compile_artifact_id=compile_artifact_id,
                )
            )

        response = ExecutionSubmissionResponse.from_wire(
            await self._context.run_blocking(loop, submit)
        )

        if response.accepted:
            execution_id = response.require_execution_id("Execution submission")
            self._host.plate_terminal_activity_status.record_execution(
                plate_path,
                execution_id,
            )
            self._host.emit_status(f"Submitted {plate_path} (queued on server)")
            self.start_completion_poller(execution_id, plate_path)
            return

        error_msg = response.require_failure_text("Execution submission")
        logger.error("Plate %s submission failed: %s", plate_path, error_msg)
        self._host.emit_error(f"Submission failed for {plate_path}: {error_msg}")
        self._host.plate_terminal_activity_status.mark_terminal(
            plate_path,
            TerminalExecutionStatus.FAILED,
        )
        self._set_orchestrator_exec_failed(plate_path)

    def start_completion_poller(self, execution_id: str, plate_path: str) -> None:
        """Start background status polling for one submitted execution."""

        class _ClientDisconnected(RuntimeError):
            pass

        def poll_status(polled_execution_id: str) -> dict:
            zmq_client = self._context.zmq.zmq_client
            if zmq_client is None:
                raise _ClientDisconnected("ZMQ client disconnected")
            return zmq_client.get_status(polled_execution_id)

        def on_running(_execution_id: str, _execution_payload: dict) -> None:
            self._host.notify_plate_running(plate_path)

        def on_terminal(
            terminal_execution_id: str,
            terminal_status: str,
            execution_payload: dict,
        ) -> None:
            current_execution_id = (
                self._host.plate_terminal_activity_status.execution_id(plate_path)
            )
            if current_execution_id != terminal_execution_id:
                logger.info(
                    "Ignoring stale terminal status for %s: execution_id=%s current=%s",
                    plate_path,
                    terminal_execution_id,
                    current_execution_id,
                )
                return
            parsed_terminal_status = parse_terminal_status(terminal_status)

            completion = parsed_terminal_status.completion_payload(
                execution_id=terminal_execution_id,
                execution_payload=execution_payload,
            )
            self._host.emit_execution_complete(completion, plate_path)

        def on_status_error(execution_id_with_error: str, message: str) -> None:
            current_execution_id = (
                self._host.plate_terminal_activity_status.execution_id(plate_path)
            )
            if current_execution_id != execution_id_with_error:
                logger.info(
                    "Ignoring stale status error for %s: execution_id=%s current=%s",
                    plate_path,
                    execution_id_with_error,
                    current_execution_id,
                )
                return
            self._host.emit_execution_complete(
                TerminalExecutionStatus.FAILED.completion_payload(
                    execution_id=execution_id_with_error,
                    execution_payload={MessageFields.ERROR: message},
                ),
                plate_path,
            )

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
            OrchestratorState.EXEC_FAILED,
        )
