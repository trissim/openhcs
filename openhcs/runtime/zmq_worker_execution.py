"""Worker execution phase for ZMQ orchestrator runs."""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from pathlib import Path
import threading
from typing import Any, Callable

from zmqruntime.messages import ExecutionStatus


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ZMQWorkerExecutionRequest:
    """Inputs needed to run compiled OpenHCS work under the ZMQ server."""

    execution_id: str
    global_config: Any
    orchestrator: Any
    pipeline_steps: list[Any]
    compiled_pipeline_definition: Any
    compiled_contexts: dict[str, Any]
    execution_bundle: Any
    worker_assignments: dict[str, list[str]]
    progress_context: dict[str, Any]
    debug_execution_policy: Any
    active_execution_record: Any
    forward_worker_progress: Callable[[Any], None]

    def execute(self) -> Any:
        from openhcs.config_framework.lazy_factory import ensure_global_config_context
        from openhcs.core.config import GlobalPipelineConfig
        from openhcs.core.worker_start_policy import WorkerStartExecutionFacts
        from openhcs.core.worker_start_policy import resolve_worker_start_context

        log_dir = Path.home() / ".local" / "share" / "openhcs" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        worker_start_decision = resolve_worker_start_context(
            self.global_config,
            server_mode=True,
            gpu_enabled=WorkerStartExecutionFacts.from_compiled_contexts(
                self.compiled_contexts
            ).gpu_enabled,
        )
        if worker_start_decision.changed:
            ensure_global_config_context(
                GlobalPipelineConfig,
                replace(
                    self.global_config,
                    multiprocessing_start_method=worker_start_decision.resolved,
                ),
            )
        logger.info(
            "[%s] Worker start method requested=%s resolved=%s reason=%s",
            self.execution_id,
            worker_start_decision.requested.value,
            worker_start_decision.resolved.value,
            worker_start_decision.reason,
        )

        worker_progress_queue = worker_start_decision.context.Queue()
        progress_forwarder = threading.Thread(
            target=self.forward_worker_progress,
            args=(worker_progress_queue,),
            daemon=True,
        )
        progress_forwarder.start()
        try:
            self.raise_if_cancelled("before starting workers")
            steps_to_execute = self.steps_to_execute()
            logger.info(
                "[%s] Passing %d compiled step(s) to worker execution",
                self.execution_id,
                len(steps_to_execute),
            )
            return self.orchestrator.execute_compiled_plate(
                pipeline_definition=steps_to_execute,
                compiled_contexts=self.compiled_contexts,
                execution_bundle=self.execution_bundle,
                log_file_base=str(log_dir / f"zmq_worker_exec_{self.execution_id}"),
                progress_queue=worker_progress_queue,
                progress_context=self.progress_context,
                worker_assignments=self.worker_assignments,
                debug_execution_policy=self.debug_execution_policy,
            )
        finally:
            worker_progress_queue.put(None)
            progress_forwarder.join()

    def steps_to_execute(self) -> list[Any]:
        if self.compiled_pipeline_definition is not None:
            return self.compiled_pipeline_definition
        return self.pipeline_steps

    def raise_if_cancelled(self, phase: str) -> None:
        if self.active_execution_record.status == ExecutionStatus.CANCELLED.value:
            logger.info(
                "[%s] Execution cancelled %s, aborting",
                self.execution_id,
                phase,
            )
            raise RuntimeError("Execution cancelled by user")
