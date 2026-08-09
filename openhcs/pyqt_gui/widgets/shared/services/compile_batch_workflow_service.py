"""Batch compile orchestration for PyQt workflows."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta
from openhcs.core.orchestrator.orchestrator import OrchestratorState
from openhcs.pyqt_gui.services.plate_manager_row import PlateManagerRow
from openhcs.pyqt_gui.widgets.shared.services.batch_context import (
    BatchWorkflowContext,
)
from openhcs.pyqt_gui.widgets.shared.services.compile_workflow_service import (
    CompileJob,
    CompileJobCallback,
    CompileJobErrorCallback,
    CompileJobStatusCallback,
    CompileWorkflowService,
    PlateCompiledState,
)
from openhcs.core.execution_state import (
    STOP_PENDING_MANAGER_STATES,
    TerminalExecutionStatus,
)
from openhcs.pyqt_gui.widgets.shared.services.plate_pipeline_request_builder import (
    PlatePipelineRequestBuilder,
    RunSpec,
)
from zmqruntime.execution import (
    BatchSubmitWaitEngine,
    CallbackBatchSubmitWaitPolicy,
)

logger = logging.getLogger(__name__)


class CompileConfigParamsByPlate(ABC, metaclass=AutoRegisterMeta):
    """Variant family for compile-time config params keyed by plate scope."""

    __registry_key__ = "registry_key"
    __skip_if_no_key__ = True

    registry_key: ClassVar[str | None] = None

    @abstractmethod
    def params_for_plate(self, plate_path: str) -> dict | None:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class EmptyCompileConfigParamsByPlate(CompileConfigParamsByPlate):
    """Compile policy with no per-plate config params."""

    registry_key: ClassVar[str] = "empty"

    def params_for_plate(self, plate_path: str) -> dict | None:
        del plate_path
        return None


@dataclass(frozen=True, slots=True)
class ExplicitCompileConfigParamsByPlate(CompileConfigParamsByPlate):
    """Explicit per-plate compile params supplied by a caller."""

    registry_key: ClassVar[str] = "explicit"
    params_by_plate: dict[str, dict]

    def params_for_plate(self, plate_path: str) -> dict | None:
        return self.params_by_plate.get(plate_path)


EMPTY_COMPILE_CONFIG_PARAMS_BY_PLATE = EmptyCompileConfigParamsByPlate()


class CompileBatchWorkflowService:
    """Owns compile-only and compile-before-execution batch policy."""

    def __init__(
        self,
        *,
        host,
        context: BatchWorkflowContext,
        compile_workflow: CompileWorkflowService | None = None,
        plate_request_builder: PlatePipelineRequestBuilder | None = None,
        compile_batch_engine: BatchSubmitWaitEngine[CompileJob] | None = None,
    ) -> None:
        self.host = host
        self._context = context
        self._compile_batch_engine = (
            compile_batch_engine or BatchSubmitWaitEngine[CompileJob]()
        )
        self._compile_workflow = compile_workflow or CompileWorkflowService(
            context=context,
        )
        self._plate_request_builder = (
            plate_request_builder or PlatePipelineRequestBuilder(host)
        )

    async def compile_plates(self, selected_items: list[PlateManagerRow]) -> None:
        """Compile pipelines for selected plates."""
        import asyncio

        loop = asyncio.get_event_loop()

        try:
            zmq_client = await self._context.connect_progress_client()
            plate_paths = [row.scope_id for row in selected_items]
            for plate_path in plate_paths:
                self.host.clear_plate_execution_tracking(plate_path)
            self.host.plate_compile_pending.update(plate_paths)
            self.host.update_item_list()
            self.host.emit_status(
                f"Queueing compilation for {len(selected_items)} plate(s)..."
            )

            compile_jobs: list[CompileJob] = []
            for row in selected_items:
                plate_path = row.scope_id
                try:
                    compile_jobs.append(
                        self._plate_request_builder.build_compile_job_from_plate_row(
                            row
                        )
                    )
                except Exception as error:
                    self._handle_compile_failure(row.name, plate_path, error)

            waiting_announced = False

            def _on_wait_success(
                job: CompileJob, _execution_id: str, _idx: int, _total: int
            ) -> None:
                self.host.clear_plate_execution_tracking(job.plate_path)
                self._set_orchestrator_state(job.plate_path, OrchestratorState.COMPILED)
                self.host.emit_orchestrator_state(
                    job.plate_path,
                    OrchestratorState.COMPILED,
                )
                logger.info("Successfully compiled %s", job.plate_path)

            def _on_wait_error(
                job: CompileJob, error: Exception, _idx: int, _total: int
            ) -> None:
                self._handle_compile_failure(job.plate_name, job.plate_path, error)

            def _on_submit_error(
                job: CompileJob, error: Exception, _idx: int, _total: int
            ) -> None:
                self._handle_compile_failure(job.plate_name, job.plate_path, error)

            def _on_wait_start(_job: CompileJob, _idx: int, total: int) -> None:
                nonlocal waiting_announced
                if waiting_announced:
                    return
                waiting_announced = True
                self.host.emit_status(
                    f"Queued {total} compilation job(s). Waiting for completion..."
                )

            def _on_wait_finally(job: CompileJob, _idx: int, _total: int) -> None:
                self.host.plate_compile_pending.discard(job.plate_path)
                self.host.update_item_list()

            compile_policy = self._make_compile_policy(
                zmq_client=zmq_client,
                loop=loop,
                fail_fast_submit=False,
                fail_fast_wait=False,
                on_submit_error=_on_submit_error,
                on_wait_start=_on_wait_start,
                on_wait_success=_on_wait_success,
                on_wait_error=_on_wait_error,
                on_wait_finally=_on_wait_finally,
            )
            await self._compile_batch_engine.run(compile_jobs, compile_policy)
        finally:
            if self.host.execution_state in STOP_PENDING_MANAGER_STATES:
                await self._context.zmq.disconnect()

        self.host.emit_status(
            f"Compilation completed for {len(selected_items)} plate(s)"
        )
        self.host.update_button_states()

    async def compile_before_execution(
        self,
        run_specs: list[RunSpec],
        loop,
        config_params_by_plate: CompileConfigParamsByPlate = (
            EMPTY_COMPILE_CONFIG_PARAMS_BY_PLATE
        ),
    ) -> dict[str, str]:
        """Compile all selected plates before submitting execution jobs."""
        zmq_client = self._context.zmq.require_client()
        compile_jobs = [
            PlatePipelineRequestBuilder.compile_job_from_run_spec(
                run_spec,
                config_params=config_params_by_plate.params_for_plate(
                    run_spec.plate_path
                ),
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
        self.host.emit_compiled_state(job.plate_path, None)
        execution_id = await self._compile_workflow.submit_compile_job(
            job=job,
            zmq_client=zmq_client,
            loop=loop,
        )
        self.host.plate_execution_ids[job.plate_path] = execution_id
        return execution_id

    async def _wait_compile_job(
        self, *, submission_id: str, job: CompileJob, zmq_client, loop
    ) -> None:
        inspection = await self._compile_workflow.wait_compile_job(
            submission_id=submission_id,
            job=job,
            zmq_client=zmq_client,
            loop=loop,
        )
        compiled_state = PlateCompiledState(
            compile_artifact_id=submission_id,
            definition_pipeline=tuple(job.definition_pipeline),
            inspection=inspection,
        )
        self.host.emit_compiled_state(job.plate_path, compiled_state)

    def _mark_execution_compile_failed(self, plate_path: str, error: Exception) -> None:
        logger.error(
            "Compile-before-execution failed for %s: %s",
            plate_path,
            error,
            exc_info=True,
        )
        self.host.plate_terminal_activity_status.mark_terminal(
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
        self, plate_name: str, plate_path: str, error: Exception
    ) -> None:
        logger.error("COMPILATION ERROR: %s: %s", plate_path, error, exc_info=True)
        self.host.clear_plate_execution_tracking(plate_path)
        self._set_orchestrator_state(plate_path, OrchestratorState.COMPILE_FAILED)
        self.host.plate_compile_pending.discard(plate_path)
        self.host.update_item_list()
        self.host.emit_orchestrator_state(
            plate_path,
            OrchestratorState.COMPILE_FAILED,
        )
        self.host.emit_compilation_error(plate_name, str(error))


__all__ = ("CompileBatchWorkflowService",)
