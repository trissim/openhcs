"""Batch compile orchestration for PyQt workflows."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, TypeVar

from openhcs.core.orchestrator.orchestrator import OrchestratorState
from openhcs.pyqt_gui.widgets.shared.services.compile_workflow_service import (
    CompileJob,
    CompileJobCallback,
    CompileJobErrorCallback,
    CompileJobStatusCallback,
    CompileWorkflowService,
)
from openhcs.pyqt_gui.widgets.shared.services.execution_state import (
    ManagerExecutionState,
    TerminalExecutionStatus,
)
from openhcs.pyqt_gui.widgets.shared.services.plate_pipeline_request_builder import (
    PlatePipelineRequestBuilder,
    RunSpec,
)
from openhcs.pyqt_gui.widgets.shared.services.zmq_client_service import ZMQClientService
from zmqruntime.execution import (
    BatchSubmitWaitEngine,
    CallbackBatchSubmitWaitPolicy,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")

RunBlockingCallable = Callable[[object, Callable[[], T]], Any]
ProgressClientConnector = Callable[[], Any]


class CompileBatchWorkflowService:
    """Owns compile-only and compile-before-execution batch policy."""

    def __init__(
        self,
        *,
        host,
        client_service: ZMQClientService,
        global_config_provider: Callable[[], Any],
        run_blocking: RunBlockingCallable,
        connect_progress_client: ProgressClientConnector,
        compile_workflow: CompileWorkflowService | None = None,
        plate_request_builder: PlatePipelineRequestBuilder | None = None,
        compile_batch_engine: BatchSubmitWaitEngine[CompileJob] | None = None,
    ) -> None:
        self.host = host
        self.client_service = client_service
        self._connect_progress_client = connect_progress_client
        self._compile_batch_engine = (
            compile_batch_engine or BatchSubmitWaitEngine[CompileJob]()
        )
        self._compile_workflow = compile_workflow or CompileWorkflowService(
            global_config_provider=global_config_provider,
            run_blocking=run_blocking,
        )
        self._plate_request_builder = plate_request_builder or PlatePipelineRequestBuilder(
            host
        )

    async def compile_plates(self, selected_items: List[Dict]) -> None:
        """Compile pipelines for selected plates."""
        self.host.emit_progress_started(len(selected_items))
        import asyncio

        loop = asyncio.get_event_loop()

        try:
            zmq_client = await self._connect_progress_client()
            plate_paths = [str(item["path"]) for item in selected_items]
            for plate_path in plate_paths:
                self.host.clear_plate_execution_tracking(plate_path)
            self.host.plate_compile_pending.update(plate_paths)
            self.host.update_item_list()
            self.host.emit_status(
                f"Queueing compilation for {len(selected_items)} plate(s)..."
            )

            completed_count = 0
            compile_jobs: List[CompileJob] = []
            for plate_data in selected_items:
                plate_path = str(plate_data["path"])
                try:
                    compile_jobs.append(
                        self._plate_request_builder.build_compile_job_from_plate_data(
                            plate_data
                        )
                    )
                except Exception as error:
                    self._handle_compile_failure(plate_data, plate_path, error)
                    completed_count += 1
                    self.host.emit_progress_updated(completed_count)

            waiting_announced = False

            def _on_wait_success(
                job: CompileJob, _execution_id: str, _idx: int, _total: int
            ) -> None:
                self.host.plate_compiled_data[job.plate_path] = {
                    "definition_pipeline": job.definition_pipeline,
                }
                self.host.clear_plate_execution_tracking(job.plate_path)
                self._set_orchestrator_state(job.plate_path, OrchestratorState.COMPILED)
                self.host.emit_orchestrator_state(job.plate_path, "COMPILED")
                logger.info("Successfully compiled %s", job.plate_path)

            def _on_wait_error(
                job: CompileJob, error: Exception, _idx: int, _total: int
            ) -> None:
                self._handle_compile_failure(
                    {"name": job.plate_name}, job.plate_path, error
                )

            def _on_wait_start(_job: CompileJob, _idx: int, total: int) -> None:
                nonlocal waiting_announced
                if waiting_announced:
                    return
                waiting_announced = True
                self.host.emit_status(
                    f"Queued {total} compilation job(s). Waiting for completion..."
                )

            def _on_wait_finally(job: CompileJob, _idx: int, _total: int) -> None:
                nonlocal completed_count
                self.host.plate_compile_pending.discard(job.plate_path)
                self.host.update_item_list()
                completed_count += 1
                self.host.emit_progress_updated(completed_count)

            compile_policy = self._make_compile_policy(
                zmq_client=zmq_client,
                loop=loop,
                fail_fast_submit=False,
                fail_fast_wait=False,
                on_submit_error=lambda job,
                error,
                _idx,
                _total: self._handle_compile_failure(
                    {"name": job.plate_name}, job.plate_path, error
                ),
                on_wait_start=_on_wait_start,
                on_wait_success=_on_wait_success,
                on_wait_error=_on_wait_error,
                on_wait_finally=_on_wait_finally,
            )
            await self._compile_batch_engine.run(compile_jobs, compile_policy)
        finally:
            if self.host.execution_state != ManagerExecutionState.RUNNING:
                await self.client_service.disconnect()

        self.host.emit_progress_finished()
        self.host.emit_status(
            f"Compilation completed for {len(selected_items)} plate(s)"
        )
        self.host.update_button_states()

    async def compile_before_execution(
        self,
        run_specs: List[RunSpec],
        loop,
        config_params_by_plate: dict[str, dict[str, Any]] | None = None,
    ) -> Dict[str, str]:
        """Compile all selected plates before submitting execution jobs."""
        if self.client_service.zmq_client is None:
            raise RuntimeError("ZMQ client is not connected")

        zmq_client = self.client_service.zmq_client
        compile_config_params = config_params_by_plate or {}
        compile_jobs = [
            PlatePipelineRequestBuilder.compile_job_from_run_spec(
                run_spec,
                config_params=compile_config_params.get(run_spec.plate_path),
            )
            for run_spec in run_specs
        ]
        waiting_announced = False

        def _on_wait_start(job: CompileJob, _idx: int, _total: int) -> None:
            nonlocal waiting_announced
            if not waiting_announced:
                waiting_announced = True
                self.host.emit_status(
                    f"Queued {len(compile_jobs)} compile job(s) before execution. Waiting for completion..."
                )
            self.host.update_item_list()

        def _on_wait_success(
            job: CompileJob, _execution_id: str, index: int, total: int
        ) -> None:
            self.host.emit_status(f"Compiled {index}/{total}: {job.plate_path}")
            self.host.update_item_list()

        def _on_wait_error(
            job: CompileJob, error: Exception, _idx: int, _total: int
        ) -> None:
            self._mark_execution_compile_failed(job.plate_path, error)

        compile_policy = self._make_compile_policy(
            zmq_client=zmq_client,
            loop=loop,
            fail_fast_submit=True,
            fail_fast_wait=True,
            on_submit_error=lambda job,
            error,
            _idx,
            _total: self._mark_execution_compile_failed(job.plate_path, error),
            on_wait_start=_on_wait_start,
            on_wait_success=_on_wait_success,
            on_wait_error=_on_wait_error,
        )
        return await self._compile_batch_engine.run(compile_jobs, compile_policy)

    def _make_compile_policy(
        self,
        *,
        zmq_client,
        loop,
        fail_fast_submit: bool,
        fail_fast_wait: bool,
        on_submit_error: CompileJobErrorCallback = None,
        on_wait_start: CompileJobCallback = None,
        on_wait_success: CompileJobStatusCallback = None,
        on_wait_error: CompileJobErrorCallback = None,
        on_wait_finally: CompileJobCallback = None,
    ) -> CallbackBatchSubmitWaitPolicy[CompileJob]:
        return CallbackBatchSubmitWaitPolicy(
            submit_fn=lambda job: self._submit_compile_job(
                job=job,
                zmq_client=zmq_client,
                loop=loop,
            ),
            wait_fn=lambda submission_id, job: self._wait_compile_job(
                submission_id=submission_id,
                job=job,
                zmq_client=zmq_client,
                loop=loop,
            ),
            job_key_fn=lambda job: job.plate_path,
            fail_fast_submit_value=fail_fast_submit,
            fail_fast_wait_value=fail_fast_wait,
            on_submit_error_fn=on_submit_error,
            on_wait_start_fn=on_wait_start,
            on_wait_success_fn=on_wait_success,
            on_wait_error_fn=on_wait_error,
            on_wait_finally_fn=on_wait_finally,
        )

    async def _submit_compile_job(self, *, job: CompileJob, zmq_client, loop) -> str:
        return await self._compile_workflow.submit_compile_job(
            job=job,
            zmq_client=zmq_client,
            loop=loop,
        )

    async def _wait_compile_job(
        self, *, submission_id: str, job: CompileJob, zmq_client, loop
    ) -> None:
        await self._compile_workflow.wait_compile_job(
            submission_id=submission_id,
            job=job,
            zmq_client=zmq_client,
            loop=loop,
        )

    def _mark_execution_compile_failed(self, plate_path: str, error: Exception) -> None:
        logger.error(
            "Compile-before-execution failed for %s: %s",
            plate_path,
            error,
            exc_info=True,
        )
        self.host.execution_runtime.mark_terminal(
            plate_path, TerminalExecutionStatus.FAILED
        )
        self.host.emit_error(f"Compile failed for {plate_path}: {error}")
        self.host.update_item_list()

    @staticmethod
    def _set_orchestrator_state(plate_path: str, state: OrchestratorState) -> None:
        from objectstate import ObjectStateRegistry

        orchestrator = ObjectStateRegistry.get_object(plate_path)
        if orchestrator is not None:
            orchestrator._state = state

    def _handle_compile_failure(
        self, plate_data: Dict[str, Any], plate_path: str, error: Exception
    ) -> None:
        logger.error("COMPILATION ERROR: %s: %s", plate_path, error, exc_info=True)
        plate_data["error"] = str(error)
        self.host.clear_plate_execution_tracking(plate_path)
        self._set_orchestrator_state(plate_path, OrchestratorState.COMPILE_FAILED)
        self.host.plate_compile_pending.discard(plate_path)
        self.host.update_item_list()
        self.host.emit_orchestrator_state(plate_path, "COMPILE_FAILED")
        self.host.emit_compilation_error(plate_data["name"], str(error))


__all__ = ("CompileBatchWorkflowService",)
