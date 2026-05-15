"""Unified batch workflow service for compile + execute flows."""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Callable, TypeVar

from PyQt6.QtCore import QEventLoop
from PyQt6.QtWidgets import QApplication

from openhcs.core.orchestrator.orchestrator import OrchestratorState
from openhcs.core.progress import ProgressEvent
from openhcs.core.debug import (
    DebugArtifactRef,
    DebugArtifactExportResponse,
    DebugCommandType,
    DebugPausedWorkerStatus,
    DebugReplayMode,
)
from openhcs.pyqt_gui.widgets.shared.services.compile_workflow_service import (
    CompileJob,
    CompileJobCallback,
    CompileJobErrorCallback,
    CompileJobStatusCallback,
    CompileWorkflowService,
)
from openhcs.pyqt_gui.widgets.shared.services.debug_progress_service import (
    DebugProgressNotificationService,
    DebugSnapshotAvailableNotification,
)
from openhcs.pyqt_gui.widgets.shared.services.plate_pipeline_request_builder import (
    PlatePipelineRequestBuilder,
    RunSpec,
)
from openhcs.pyqt_gui.widgets.shared.services.terminal_result_builder import (
    TerminalExecutionResultBuilder,
)
from openhcs.pyqt_gui.widgets.shared.services.execution_submission_service import (
    ExecutionSubmissionService,
)
from openhcs.pyqt_gui.widgets.shared.services.debug_workflow_service import (
    DebugPlateRunRequest,
    DebugWorkflowService,
)
from openhcs.pyqt_gui.widgets.shared.services.progress_workflow_service import (
    ProgressWorkflowService,
)
from openhcs.pyqt_gui.widgets.shared.services.execution_server_status_presenter import (
    ExecutionServerStatusPresenter,
)
from openhcs.pyqt_gui.widgets.shared.services.execution_state import (
    STOP_PENDING_MANAGER_STATES,
    ManagerExecutionState,
    TerminalExecutionStatus,
)
from openhcs.pyqt_gui.widgets.shared.server_browser import (
    ServerKillPlan,
    ServerKillService,
)
from openhcs.pyqt_gui.widgets.shared.services.zmq_client_service import ZMQClientService
from pyqt_reactive.services import (
    DefaultServerInfoParser,
    ServerInfoParserABC,
)
from zmqruntime.execution import (
    BatchSubmitWaitEngine,
    CallbackBatchSubmitWaitPolicy,
    ExecutionStatusPoller,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass(frozen=True)
class ZMQClientConnectionSpec:
    """Connection request for execution-server progress clients."""

    progress_callback: Callable[[ProgressEvent], None]
    persistent: bool = True
    timeout: int = 15

    async def connect(self, client_service: ZMQClientService):
        return await client_service.connect(
            progress_callback=self.progress_callback,
            persistent=self.persistent,
            timeout=self.timeout,
        )


class BatchWorkflowService:
    """Single owner of batch compilation and execution workflow."""

    _compile_workflow: CompileWorkflowService | None = None
    _debug_progress_notifications: DebugProgressNotificationService | None = None
    _plate_request_builder: PlatePipelineRequestBuilder | None = None
    _terminal_result_builder: TerminalExecutionResultBuilder | None = None
    _execution_submission: ExecutionSubmissionService | None = None
    _debug_workflow: DebugWorkflowService | None = None
    _progress_workflow: ProgressWorkflowService | None = None

    def __init__(
        self,
        host,
        port: int = 7777,
        client_service: ZMQClientService | None = None,
        server_info_parser: ServerInfoParserABC | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.client_service = client_service or ZMQClientService(port=port)
        server_info_parser_impl = (
            server_info_parser
            if server_info_parser is not None
            else DefaultServerInfoParser()
        )

        self._server_info_parser = server_info_parser_impl
        self._compile_batch_engine = BatchSubmitWaitEngine[CompileJob]()
        self._compile_workflow = CompileWorkflowService(
            global_config_provider=lambda: self.host.global_config,
            run_blocking=self._run_blocking,
        )
        self._plate_request_builder = PlatePipelineRequestBuilder(self.host)
        self._terminal_result_builder = TerminalExecutionResultBuilder()
        self._execution_status_poller = ExecutionStatusPoller()
        self._execution_submission = ExecutionSubmissionService(
            host=self.host,
            client_service=self.client_service,
            run_blocking=self._run_blocking,
            completion_poller=self._execution_status_poller,
            terminal_result_builder=self._terminal_result_builder,
            on_completion_update=self._check_all_completed,
        )
        self._debug_workflow = self._build_debug_workflow_service()
        self._debug_progress_notifications = DebugProgressNotificationService()
        self._server_kill_service = ServerKillService()
        self._server_status_presenter = ExecutionServerStatusPresenter()
        self._progress_workflow = self._build_progress_workflow_service()
        self._registry_listener = self._progress_workflow.mark_dirty
        self.host._progress_tracker.add_listener(self._registry_listener)
        self._registry_listener_registered = True
        self._cleaned_up = False

    def cleanup(self) -> None:
        """Release timers/listeners owned by this service."""
        if self._cleaned_up:
            return
        self._cleaned_up = True

        if self._registry_listener_registered:
            removed = self.host._progress_tracker.remove_listener(
                self._registry_listener
            )
            if not removed:
                raise RuntimeError(
                    "BatchWorkflowService listener removal failed: listener not registered"
                )
            self._registry_listener_registered = False

        self._progress_workflow_service().cleanup()

    async def compile_plates(self, selected_items: List[Dict]) -> None:
        """Compile pipelines for selected plates."""
        self._flush_pending_ui_edits()
        self._progress_workflow_service().reset_for_new_batch()
        self.host.emit_progress_started(len(selected_items))
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
                        self._plate_pipeline_request_builder().build_compile_job_from_plate_data(
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

    def add_debug_snapshot_listener(
        self,
        listener: Callable[[DebugSnapshotAvailableNotification], None],
    ) -> None:
        """Subscribe to debug snapshot availability announced through progress."""

        self._debug_progress_notification_service().add_listener(listener)

    async def run_plates(self, ready_items: List[Dict]) -> None:
        """Run selected plates using compile-all then execute-all workflow."""
        self._flush_pending_ui_edits()
        loop = asyncio.get_event_loop()
        try:
            plate_paths = [str(item["path"]) for item in ready_items]
            logger.info("Starting ZMQ execution for %d plates", len(plate_paths))

            self._progress_workflow_service().reset_for_new_batch()
            self.host.emit_clear_logs()

            await self._connect_progress_client()

            self.host.plate_execution_ids.clear()
            self.host.execution_runtime.begin_batch(plate_paths)
            self.host.plate_progress.clear()

            from objectstate import ObjectStateRegistry

            for item in ready_items:
                plate_path = str(item["path"])
                orchestrator = ObjectStateRegistry.get_object(plate_path)
                if orchestrator is not None:
                    orchestrator._state = OrchestratorState.EXECUTING
                    self.host.emit_orchestrator_state(
                        plate_path, OrchestratorState.EXECUTING.value
                    )

            self.host.execution_state = ManagerExecutionState.RUNNING
            self.host.emit_status(
                f"Compiling {len(ready_items)} plate(s) before execution..."
            )
            self.host.update_button_states()
            self.host.update_item_list()

            run_specs = [
                self._plate_pipeline_request_builder().build_run_spec(plate_path)
                for plate_path in plate_paths
            ]
            compile_artifacts = await self._compile_plates_before_execution(
                run_specs=run_specs,
                loop=loop,
            )

            self.host.emit_status(
                f"Compilation complete. Submitting {len(run_specs)} plate(s) for execution..."
            )
            for run_spec in run_specs:
                await self._execution_submission_service().submit_plate(
                    run_spec=run_spec,
                    compile_artifact_id=compile_artifacts[run_spec.plate_path],
                    loop=loop,
                )
        except Exception as error:
            logger.error("Failed to execute plates via ZMQ: %s", error, exc_info=True)
            self.host.emit_error(f"Failed to execute: {error}")
            await self._handle_execution_failure(loop)

    async def run_debug_plate(
        self,
        *,
        plate_path: str,
        debug_session_id: str,
        snapshot_store_ref: str,
        command_type,
        snapshot_store_backend: str | None = None,
        selected_source_group: str | None = None,
        pause_step_indices: tuple[int, ...] = (),
        start_step_index: int = 0,
        start_after_invocation_key: str | None = None,
        replay_mode: DebugReplayMode = DebugReplayMode.WARM_ARTIFACT,
    ) -> None:
        """Compile one plate and submit a bounded debug execution."""

        self._flush_pending_ui_edits()
        loop = asyncio.get_event_loop()
        try:
            await self._connect_progress_client()
            self._progress_workflow_service().reset_for_new_batch()
            self.host.execution_runtime.begin_batch([plate_path])
            self.host.execution_state = ManagerExecutionState.RUNNING
            self.host.emit_status(f"Compiling debug run for {plate_path}...")
            self.host.update_button_states()
            self.host.update_item_list()

            debug_request = DebugPlateRunRequest(
                debug_session_id=debug_session_id,
                snapshot_store_ref=snapshot_store_ref,
                snapshot_store_backend=snapshot_store_backend,
                command_type=DebugCommandType(command_type),
                selected_source_group=selected_source_group,
                pause_step_indices=tuple(pause_step_indices),
                start_step_index=start_step_index,
                start_after_invocation_key=start_after_invocation_key,
                replay_mode=replay_mode,
            )
            run_spec = self._plate_pipeline_request_builder().build_run_spec(plate_path)
            compile_artifact_id = await self._debug_workflow_service().compile_artifact_id(
                run_spec=run_spec,
                debug_request=debug_request,
                loop=loop,
            )
            self.host.emit_status(f"Submitting debug run for {plate_path}...")
            await self._debug_workflow_service().submit_debug_plate(
                run_spec=run_spec,
                compile_artifact_id=compile_artifact_id,
                debug_request=debug_request,
                loop=loop,
            )
        except Exception as error:
            logger.error("Failed to execute debug run via ZMQ: %s", error, exc_info=True)
            self.host.emit_error(f"Failed to execute debug run: {error}")
            await self._handle_execution_failure(loop)

    async def send_debug_worker_command(
        self,
        *,
        debug_session_id: str,
        command_type: DebugCommandType,
    ) -> DebugPausedWorkerStatus:
        """Send a control command to an already-running persistent debug worker."""

        loop = asyncio.get_event_loop()
        await self._connect_progress_client()
        return await self._debug_workflow_service().send_worker_command(
            debug_session_id=debug_session_id,
            command_type=command_type,
            loop=loop,
        )

    async def export_debug_artifact(
        self,
        *,
        debug_session_id: str,
        artifact_ref: DebugArtifactRef,
        export_root: str,
        snapshot_store_ref: str | None = None,
        snapshot_store_backend: str | None = None,
    ) -> DebugArtifactExportResponse:
        """Materialize one debug artifact through the execution server namespace."""

        loop = asyncio.get_event_loop()
        await self._connect_progress_client()
        return await self._debug_workflow_service().export_artifact(
            debug_session_id=debug_session_id,
            artifact_ref=artifact_ref,
            export_root=export_root,
            snapshot_store_ref=snapshot_store_ref,
            snapshot_store_backend=snapshot_store_backend,
            loop=loop,
        )

    async def _compile_plates_before_execution(
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
        compile_artifacts = await self._compile_batch_engine.run(
            compile_jobs, compile_policy
        )
        return compile_artifacts

    async def _connect_progress_client(self):
        """Connect the shared ZMQ client with the standard progress callback."""

        return await ZMQClientConnectionSpec(
            progress_callback=self._progress_workflow_service().on_progress,
        ).connect(self.client_service)

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
        return await self._compile_workflow_service().submit_compile_job(
            job=job,
            zmq_client=zmq_client,
            loop=loop,
        )

    async def _wait_compile_job(
        self, *, submission_id: str, job: CompileJob, zmq_client, loop
    ) -> None:
        await self._compile_workflow_service().wait_compile_job(
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

    async def _handle_execution_failure(self, loop) -> None:
        from objectstate import ObjectStateRegistry

        for plate_path in tuple(self.host.execution_runtime.active_plates):
            self.host.execution_runtime.mark_terminal(
                plate_path, TerminalExecutionStatus.FAILED
            )
            orchestrator = ObjectStateRegistry.get_object(plate_path)
            if orchestrator is not None:
                orchestrator._state = OrchestratorState.EXEC_FAILED
                self.host.emit_orchestrator_state(
                    plate_path, OrchestratorState.EXEC_FAILED.value
                )

        self.host.execution_state = ManagerExecutionState.IDLE
        await self._disconnect_client(loop)
        self.host.current_execution_id = None
        self._refresh_host_execution_ui()

    async def _disconnect_client(self, loop) -> None:
        if self.client_service.zmq_client is None:
            return
        try:
            await self.client_service.disconnect()
        except Exception as error:
            logger.warning("Error disconnecting old client: %s", error)

    @staticmethod
    async def _run_blocking(loop, func: Callable[[], T]) -> T:
        return await loop.run_in_executor(None, func)

    def _compile_workflow_service(self) -> CompileWorkflowService:
        workflow = self._compile_workflow
        if workflow is None:
            workflow = CompileWorkflowService(
                global_config_provider=lambda: self.host.global_config,
                run_blocking=self._run_blocking,
            )
            self._compile_workflow = workflow
        return workflow

    def _debug_progress_notification_service(self) -> DebugProgressNotificationService:
        service = self._debug_progress_notifications
        if service is None:
            service = DebugProgressNotificationService()
            self._debug_progress_notifications = service
        return service

    def _plate_pipeline_request_builder(self) -> PlatePipelineRequestBuilder:
        builder = self._plate_request_builder
        if builder is None:
            builder = PlatePipelineRequestBuilder(self.host)
            self._plate_request_builder = builder
        return builder

    def _terminal_result_builder_service(self) -> TerminalExecutionResultBuilder:
        builder = self._terminal_result_builder
        if builder is None:
            builder = TerminalExecutionResultBuilder()
            self._terminal_result_builder = builder
        return builder

    def _execution_submission_service(self) -> ExecutionSubmissionService:
        service = self._execution_submission
        if service is None:
            service = ExecutionSubmissionService(
                host=self.host,
                client_service=self.client_service,
                run_blocking=self._run_blocking,
                completion_poller=self._execution_status_poller,
                terminal_result_builder=self._terminal_result_builder_service(),
                on_completion_update=self._check_all_completed,
            )
            self._execution_submission = service
        return service

    def _debug_workflow_service(self) -> DebugWorkflowService:
        service = self._debug_workflow
        if service is None:
            service = self._build_debug_workflow_service()
            self._debug_workflow = service
        return service

    def _build_debug_workflow_service(self) -> DebugWorkflowService:
        return DebugWorkflowService(
            host=self.host,
            client_service=self.client_service,
            run_blocking=self._run_blocking,
            compile_before_execution=self._compile_plates_before_execution,
            execution_submission=self._execution_submission_service(),
        )

    def _progress_workflow_service(self) -> ProgressWorkflowService:
        service = self._progress_workflow
        if service is None:
            service = self._build_progress_workflow_service()
            self._progress_workflow = service
        return service

    def _build_progress_workflow_service(self) -> ProgressWorkflowService:
        return ProgressWorkflowService(
            host=self.host,
            client_service=self.client_service,
            server_info_parser=self._server_info_parser,
            debug_notifications=self._debug_progress_notification_service(),
            status_presenter=self._server_status_presenter,
        )

    @staticmethod
    def _flush_pending_ui_edits() -> None:
        """Commit pending editor widget state before reading pipeline definitions."""
        app = QApplication.instance()
        if app is None:
            return
        focus_widget = app.focusWidget()
        if focus_widget is not None:
            focus_widget.clearFocus()
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents)

    def _check_all_completed(self) -> None:
        if self.host.execution_state not in (
            ManagerExecutionState.RUNNING,
            *STOP_PENDING_MANAGER_STATES,
        ):
            return
        if not self.host.execution_runtime.all_batch_terminal():
            return
        completed, failed = self.host.execution_runtime.terminal_counts()
        self.host.notify_all_plates_completed(completed, failed)

    def stop_execution(self, force: bool = False) -> None:
        port = self.port

        def kill_server() -> None:
            try:
                # Force-kill is best-effort: the server may already be gone if a graceful
                # stop just completed, so treat "not found" style outcomes as success.
                plan = ServerKillPlan(
                    graceful=not force,
                    strict_failures=not force,
                    emit_signal_on_failure=force,
                    success_message=f"Stopped execution server on port {port}",
                )
                success, message = self._server_kill_service.kill_ports(
                    ports=[port],
                    plan=plan,
                    on_server_killed=lambda _port: self._emit_cancelled_for_all_plates(),
                    log_info=logger.info,
                    log_warning=logger.warning,
                    log_error=logger.error,
                )
                if not success:
                    if self.host.execution_state.suppresses_stop_failure:
                        logger.info(
                            "Suppressing stale stop failure while stop is already terminalizing: %s",
                            message,
                        )
                        self._emit_cancelled_for_all_plates()
                        return
                    self.host.emit_error(message)
                    return
            except Exception as error:
                logger.error("Error stopping server: %s", error)
                self.host.emit_error(f"Error stopping execution: {error}")

        threading.Thread(target=kill_server, daemon=True).start()

        if force:
            # Keep UI responsive on force-kill: mark plates cancelled immediately on the
            # caller thread while kill work continues in the background.
            self._emit_cancelled_for_all_plates()
            self.disconnect_async()

    def _emit_cancelled_for_all_plates(self) -> None:
        for plate_path in self.host.execution_runtime.cancellable_plates():
            self.host.emit_execution_complete(
                {"status": TerminalExecutionStatus.CANCELLED.value}, plate_path
            )

    def disconnect(self) -> None:
        if self.client_service.zmq_client is None:
            return
        try:
            self.client_service.disconnect_sync()
        except Exception as error:
            logger.warning("Error disconnecting ZMQ client: %s", error)

    def disconnect_async(self) -> None:
        """Disconnect client on a background thread to avoid UI stalls."""

        def _disconnect() -> None:
            self.disconnect()

        threading.Thread(target=_disconnect, daemon=True).start()

    def _refresh_host_execution_ui(self) -> None:
        refresh_fn = getattr(self.host, "refresh_execution_ui", None)
        if callable(refresh_fn):
            refresh_fn()
            return
        self.host.update_item_list()
        self.host.update_button_states()

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
