"""Worker execution phase for ZMQ orchestrator runs."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import threading
from typing import Any, Callable

from zmqruntime.messages import ExecutionStatus

from openhcs.core.compiled_execution import CompiledExecutionBundle
from openhcs.core.orchestrator.compiled_plate_execution import (
    CompiledPlateExecutionExtras,
    CompiledPlateExecutionResults,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ZMQWorkerExecutionRequest:
    """Inputs needed to run compiled OpenHCS work under the ZMQ server."""

    execution_id: str
    orchestrator: Any
    execution_bundle: CompiledExecutionBundle
    progress_context: dict[str, Any]
    debug_execution_policy: Any
    active_execution_record: Any
    forward_worker_progress: Callable[[Any], None]

    def execute(self) -> Any:
        log_dir = Path.home() / ".local" / "share" / "openhcs" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        execution_bundle = self.execution_bundle
        worker_start_plan = execution_bundle.runtime_environment.worker_start
        logger.info(
            "[%s] Worker start method requested=%s resolved=%s reason=%s",
            self.execution_id,
            worker_start_plan.requested.value,
            worker_start_plan.resolved.value,
            worker_start_plan.reason,
        )

        worker_progress_queue = worker_start_plan.multiprocessing_context().Queue()
        progress_forwarder = threading.Thread(
            target=self.forward_worker_progress,
            args=(worker_progress_queue,),
            daemon=True,
        )
        progress_forwarder.start()
        try:
            self.raise_if_cancelled("before starting workers")
            logger.info(
                "[%s] Passing %d compiled step(s) to worker execution",
                self.execution_id,
                len(execution_bundle.pipeline_definition),
            )
            execution_results: CompiledPlateExecutionResults = (
                self.orchestrator.execute_compiled_plate(
                    execution_bundle=execution_bundle,
                    log_file_base=str(
                        log_dir / f"zmq_worker_exec_{self.execution_id}"
                    ),
                    progress_queue=worker_progress_queue,
                    progress_context=self.progress_context,
                    debug_execution_policy=self.debug_execution_policy,
                )
            )
            self.active_execution_record.set_extra(
                CompiledPlateExecutionExtras.EXECUTION_RECORD_KEY,
                execution_results.extras,
            )
            return execution_results
        finally:
            worker_progress_queue.put(None)
            progress_forwarder.join()

    def raise_if_cancelled(self, phase: str) -> None:
        if self.active_execution_record.status == ExecutionStatus.CANCELLED.value:
            logger.info(
                "[%s] Execution cancelled %s, aborting",
                self.execution_id,
                phase,
            )
            raise RuntimeError("Execution cancelled by user")
