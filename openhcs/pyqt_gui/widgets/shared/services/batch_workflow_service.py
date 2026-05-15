"""Unified batch workflow service for compile + execute flows."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Callable, Optional, TypeVar

from PyQt6.QtCore import QEventLoop
from PyQt6.QtWidgets import QApplication

from openhcs.core.orchestrator.orchestrator import OrchestratorState
from openhcs.core.progress import ProgressEvent
from openhcs.core.debug import (
    DebugArtifactRef,
    DebugArtifactExportResponse,
    DebugCommandType,
    DebugExecutionConfig,
    DebugPausedWorkerStatus,
    DebugProgressContext,
    DebugReplayMode,
    DebugSnapshot,
)
from openhcs.core.progress.projection import (
    ExecutionRuntimeProjection,
    build_execution_runtime_projection_from_registry,
)
from openhcs.pyqt_gui.widgets.shared.services.plate_config_resolver import (
    resolve_pipeline_config_for_plate,
)
from openhcs.pyqt_gui.widgets.shared.services.progress_batch_reset import (
    reset_progress_views_for_new_batch,
)
from openhcs.pyqt_gui.widgets.shared.services.execution_server_status_presenter import (
    ExecutionServerStatusPresenter,
)
from openhcs.pyqt_gui.widgets.shared.services.execution_state import (
    STOP_PENDING_MANAGER_STATES,
    ManagerExecutionState,
    TerminalExecutionStatus,
    parse_terminal_status,
)
from openhcs.pyqt_gui.widgets.shared.server_browser import (
    ServerKillPlan,
    ServerKillService,
)
from openhcs.pyqt_gui.widgets.shared.services.zmq_client_service import ZMQClientService
from pyqt_reactive.services import (
    CallbackIntervalSnapshotPollerPolicy,
    DefaultServerInfoParser,
    ExecutionServerInfo,
    IntervalSnapshotPoller,
    ServerInfoParserABC,
)
from zmqruntime.execution import (
    BatchSubmitWaitEngine,
    CallbackBatchSubmitWaitPolicy,
    ExecutionStatusPoller,
    CallbackExecutionStatusPollPolicy,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass(frozen=True)
class PlatePipelineRequest:
    """Shared plate/pipeline/config identity for compile and run requests."""

    plate_path: str
    definition_pipeline: List
    pipeline_config: Any


@dataclass(frozen=True)
class CompileJob(PlatePipelineRequest):
    """Single compile unit for a plate."""

    plate_name: str
    config_params: dict[str, Any] | None = None


@dataclass(frozen=True)
class CompileRequestResult:
    """Accepted compile submission returned by the execution server."""

    execution_id: str
    response: dict[str, Any]


@dataclass(frozen=True)
class RunSpec(PlatePipelineRequest):
    """Execution request assembled from one plate and its resolved config."""

    global_config: Any


@dataclass(frozen=True)
class DebugPlateRunRequest:
    """Debug execution identity threaded through compile and run submission."""

    debug_session_id: str
    snapshot_store_ref: str
    snapshot_store_backend: str | None
    command_type: DebugCommandType
    selected_source_group: str | None
    pause_step_indices: tuple[int, ...]
    start_step_index: int = 0
    start_after_invocation_key: str | None = None
    replay_mode: DebugReplayMode = DebugReplayMode.WARM_ARTIFACT

    @property
    def execution_config(self) -> DebugExecutionConfig:
        return DebugExecutionConfig(
            debug_session_id=self.debug_session_id,
            snapshot_store_ref=self.snapshot_store_ref,
            snapshot_store_backend=self.snapshot_store_backend,
            command_type=self.command_type,
            selected_source_group=self.selected_source_group,
            pause_step_indices=self.pause_step_indices,
            start_step_index=self.start_step_index,
            start_after_invocation_key=self.start_after_invocation_key,
            replay_mode=self.replay_mode,
        )

    @property
    def config_params(self) -> dict[str, Any]:
        return self.execution_config.to_config_params()

    @property
    def compile_config_params(self) -> dict[str, Any]:
        return self.execution_config.compile_cache_config_params()


@dataclass(frozen=True)
class DebugCompileArtifactCacheKey:
    """Stable identity for reusable debug compile artifacts."""

    plate_path: str
    pipeline_fingerprint: str
    selected_source_group: str | None
    replay_mode: DebugReplayMode


CompileJobCallback = Callable[[CompileJob, int, int], None] | None
CompileJobStatusCallback = Callable[[CompileJob, str, int, int], None] | None
CompileJobErrorCallback = Callable[[CompileJob, Exception, int, int], None] | None


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


@dataclass(frozen=True)
class DebugSnapshotAvailableNotification:
    """GUI-side notification that one debug snapshot can be read."""

    progress_event: ProgressEvent
    debug_context: DebugProgressContext
    snapshot: DebugSnapshot | None = None


class BatchWorkflowService:
    """Single owner of batch compilation and execution workflow."""

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

        self._progress_dirty = False
        from PyQt6.QtCore import QTimer
        from openhcs.pyqt_gui.config import ProgressUIConfig

        self._progress_coalesce_timer = QTimer()
        self._progress_coalesce_timer.timeout.connect(self._coalesced_progress_update)
        self._progress_coalesce_timer.start(ProgressUIConfig().update_interval_ms)
        self._runtime_projection = ExecutionRuntimeProjection()
        self._server_info_parser = server_info_parser_impl
        self._server_info_poller = IntervalSnapshotPoller[ExecutionServerInfo](
            CallbackIntervalSnapshotPollerPolicy(
                fetch_snapshot_fn=self._fetch_server_info_snapshot,
                clone_snapshot_fn=lambda snapshot: snapshot,
                poll_interval_seconds_value=1.0,
                on_snapshot_changed_fn=lambda _snapshot: self._mark_progress_dirty(),
                on_poll_error_fn=lambda error: logger.debug(
                    "Server info poll failed: %s", error
                ),
            )
        )
        self._compile_batch_engine = BatchSubmitWaitEngine[CompileJob]()
        self._debug_compile_artifacts: dict[DebugCompileArtifactCacheKey, str] = {}
        self._execution_status_poller = ExecutionStatusPoller()
        self._debug_snapshot_listeners: list[
            Callable[[DebugSnapshotAvailableNotification], None]
        ] = []
        self._server_kill_service = ServerKillService()
        self._server_status_presenter = ExecutionServerStatusPresenter()
        self._registry_listener = self._on_registry_event
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

        if self._progress_coalesce_timer is not None:
            self._progress_coalesce_timer.stop()
            self._progress_coalesce_timer.deleteLater()
            self._progress_coalesce_timer = None

    async def compile_plates(self, selected_items: List[Dict]) -> None:
        """Compile pipelines for selected plates."""
        self._flush_pending_ui_edits()
        reset_progress_views_for_new_batch(self.host)
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
                        self._build_compile_job_from_plate_data(plate_data)
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

        self._debug_snapshot_listeners.append(listener)

    def remove_debug_snapshot_listener(
        self,
        listener: Callable[[DebugSnapshotAvailableNotification], None],
    ) -> bool:
        """Remove one debug snapshot listener if it is registered."""

        try:
            self._debug_snapshot_listeners.remove(listener)
        except ValueError:
            return False
        return True

    async def run_plates(self, ready_items: List[Dict]) -> None:
        """Run selected plates using compile-all then execute-all workflow."""
        self._flush_pending_ui_edits()
        loop = asyncio.get_event_loop()
        try:
            plate_paths = [str(item["path"]) for item in ready_items]
            logger.info("Starting ZMQ execution for %d plates", len(plate_paths))

            self._reset_progress_for_new_batch()
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

            run_specs = [self._build_run_spec(plate_path) for plate_path in plate_paths]
            compile_artifacts = await self._compile_plates_before_execution(
                run_specs=run_specs,
                loop=loop,
            )

            self.host.emit_status(
                f"Compilation complete. Submitting {len(run_specs)} plate(s) for execution..."
            )
            for run_spec in run_specs:
                await self._submit_plate(
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
            self._reset_progress_for_new_batch()
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
            run_spec = self._build_run_spec(plate_path)
            compile_artifact_id = await self._debug_compile_artifact_id(
                run_spec=run_spec,
                debug_request=debug_request,
                loop=loop,
            )
            self.host.emit_status(f"Submitting debug run for {plate_path}...")
            await self._submit_debug_plate(
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

        def send_command() -> DebugPausedWorkerStatus:
            if self.client_service.zmq_client is None:
                raise RuntimeError("ZMQ client is not connected")
            return self.client_service.zmq_client.send_debug_worker_command(
                debug_session_id=debug_session_id,
                command_type=command_type,
            ).status

        status = await self._run_blocking(loop, send_command)
        self.host.emit_status(
            f"Debug worker {status.state.value} for session {debug_session_id[:8]}"
        )
        return status

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

        def export_artifact() -> DebugArtifactExportResponse:
            if self.client_service.zmq_client is None:
                raise RuntimeError("ZMQ client is not connected")
            return self.client_service.zmq_client.export_debug_artifact(
                debug_session_id=debug_session_id,
                artifact_ref=artifact_ref,
                export_root=export_root,
                snapshot_store_ref=snapshot_store_ref,
                snapshot_store_backend=snapshot_store_backend,
            )

        response = await self._run_blocking(loop, export_artifact)
        self.host.emit_status(f"Exported debug artifact to {response.exported_ref}")
        return response

    async def _debug_compile_artifact_id(
        self,
        *,
        run_spec: RunSpec,
        debug_request: DebugPlateRunRequest,
        loop,
    ) -> str:
        cache_key = DebugCompileArtifactCacheKey(
            plate_path=run_spec.plate_path,
            pipeline_fingerprint=self._pipeline_fingerprint(
                run_spec.definition_pipeline
            ),
            selected_source_group=debug_request.selected_source_group,
            replay_mode=debug_request.replay_mode,
        )
        if debug_request.replay_mode.retains_compile_artifact:
            cached_artifact_id = self._debug_compile_artifacts.get(cache_key)
            if cached_artifact_id is not None:
                logger.info(
                    "Reusing debug compile artifact: plate=%s artifact_id=%s",
                    run_spec.plate_path,
                    cached_artifact_id,
                )
                return cached_artifact_id

        compile_artifacts = await self._compile_plates_before_execution(
            run_specs=[run_spec],
            loop=loop,
            config_params_by_plate={
                run_spec.plate_path: debug_request.compile_config_params
            },
        )
        compile_artifact_id = compile_artifacts[run_spec.plate_path]
        if debug_request.replay_mode.retains_compile_artifact:
            self._debug_compile_artifacts[cache_key] = compile_artifact_id
        return compile_artifact_id

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
            self._build_compile_job_from_run_spec(
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
            progress_callback=self._on_progress,
        ).connect(self.client_service)

    def _build_compile_job_from_plate_data(
        self, plate_data: Dict[str, Any]
    ) -> CompileJob:
        plate_path = str(plate_data["path"])
        definition_pipeline = self.host.get_pipeline_definition(plate_path)
        if not definition_pipeline:
            logger.warning(
                "No pipeline defined for %s, using empty pipeline",
                plate_data["name"],
            )
            definition_pipeline = []

        self._validate_pipeline_steps(definition_pipeline)
        pipeline_config = resolve_pipeline_config_for_plate(self.host, plate_path)
        logger.info(
            "Compile snapshot: plate=%s steps=%d fingerprint=%s step_names=%s",
            plate_path,
            len(definition_pipeline),
            self._pipeline_fingerprint(definition_pipeline),
            self._pipeline_step_names(definition_pipeline),
        )
        return CompileJob(
            plate_path=plate_path,
            plate_name=str(plate_data["name"]),
            definition_pipeline=definition_pipeline,
            pipeline_config=pipeline_config,
        )

    @staticmethod
    def _build_compile_job_from_run_spec(
        run_spec: RunSpec,
        *,
        config_params: dict[str, Any] | None = None,
    ) -> CompileJob:
        plate_path = run_spec.plate_path
        definition_pipeline = run_spec.definition_pipeline
        logger.info(
            "Compile-before-run snapshot: plate=%s steps=%d fingerprint=%s step_names=%s",
            plate_path,
            len(definition_pipeline),
            BatchWorkflowService._pipeline_fingerprint(definition_pipeline),
            BatchWorkflowService._pipeline_step_names(definition_pipeline),
        )
        return CompileJob(
            plate_path=plate_path,
            plate_name=plate_path,
            definition_pipeline=definition_pipeline,
            pipeline_config=run_spec.pipeline_config,
            config_params=config_params,
        )

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
        response = await self._submit_compile_request(
            zmq_client=zmq_client,
            loop=loop,
            plate_path=job.plate_path,
            definition_pipeline=job.definition_pipeline,
            pipeline_config=job.pipeline_config,
            config_params=job.config_params,
        )
        return response.execution_id

    async def _wait_compile_job(
        self, *, submission_id: str, job: CompileJob, zmq_client, loop
    ) -> None:
        await self._wait_for_compile_completion(
            zmq_client=zmq_client,
            loop=loop,
            execution_id=submission_id,
            plate_path=job.plate_path,
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

    async def _submit_compile_request(
        self,
        *,
        zmq_client,
        loop,
        plate_path: str,
        definition_pipeline: List,
        pipeline_config,
        config_params: dict[str, Any] | None = None,
    ) -> CompileRequestResult:
        def _submit_compile() -> Dict[str, Any]:
            logger.info(
                "Submit compile: plate=%s steps=%d fingerprint=%s",
                plate_path,
                len(definition_pipeline),
                self._pipeline_fingerprint(definition_pipeline),
            )
            return zmq_client.submit_compile(
                plate_id=plate_path,
                pipeline_steps=definition_pipeline,
                global_config=self.host.global_config,
                pipeline_config=pipeline_config,
                config_params=config_params,
            )

        response = await self._run_blocking(loop, _submit_compile)
        if response.get("status") != "accepted":
            raise RuntimeError(
                f"Compile submission failed for {plate_path}: "
                f"{response.get('message', 'Unknown error')}"
            )
        execution_id = response.get("execution_id")
        if not execution_id:
            raise RuntimeError(
                f"Compile submission missing execution_id for {plate_path}"
            )
        return CompileRequestResult(
            execution_id=str(execution_id),
            response=response,
        )

    async def _wait_for_compile_completion(
        self,
        *,
        zmq_client,
        loop,
        execution_id: str,
        plate_path: str,
    ) -> None:
        wait_result = await self._run_blocking(
            loop,
            lambda: zmq_client.wait_for_completion(execution_id),
        )
        if wait_result.get("status") != "complete":
            raise RuntimeError(
                f"Compilation failed for {plate_path}: "
                f"{wait_result.get('message', 'Unknown error')}"
            )

    def _build_run_spec(self, plate_path: str) -> RunSpec:
        plate_path = str(plate_path)
        definition_pipeline = self.host.get_pipeline_definition(plate_path)
        if not definition_pipeline:
            logger.warning(
                "No pipeline defined for %s, using empty pipeline",
                plate_path,
            )
            definition_pipeline = []
        self._validate_pipeline_steps(definition_pipeline)
        pipeline_config = resolve_pipeline_config_for_plate(self.host, plate_path)
        return RunSpec(
            plate_path=plate_path,
            definition_pipeline=definition_pipeline,
            global_config=self.host.global_config,
            pipeline_config=pipeline_config,
        )

    async def _submit_plate(
        self, run_spec: RunSpec, compile_artifact_id: str, loop
    ) -> None:
        if self.client_service.zmq_client is None:
            raise RuntimeError("ZMQ client is not connected")
        plate_path = run_spec.plate_path
        definition_pipeline = run_spec.definition_pipeline
        logger.info("Executing plate: %s", plate_path)
        logger.info(
            "Submit run: plate=%s artifact_id=%s steps=%d fingerprint=%s",
            plate_path,
            compile_artifact_id,
            len(definition_pipeline),
            self._pipeline_fingerprint(definition_pipeline),
        )

        def _submit() -> Dict[str, Any]:
            return self.client_service.zmq_client.submit_pipeline(
                plate_id=plate_path,
                pipeline_steps=definition_pipeline,
                global_config=run_spec.global_config,
                pipeline_config=run_spec.pipeline_config,
                compile_artifact_id=compile_artifact_id,
            )

        response = await self._run_blocking(loop, _submit)

        execution_id = response.get("execution_id")
        if execution_id:
            self.host.plate_execution_ids[plate_path] = execution_id
            self.host.current_execution_id = execution_id

        status = response.get("status")
        if status == "accepted":
            self.host.emit_status(f"Submitted {plate_path} (queued on server)")
            if execution_id:
                self._start_completion_poller(execution_id, plate_path)
            return

        error_msg = response.get("message", "Unknown error")
        logger.error("Plate %s submission failed: %s", plate_path, error_msg)
        self.host.emit_error(f"Submission failed for {plate_path}: {error_msg}")
        self.host.execution_runtime.mark_terminal(
            plate_path, TerminalExecutionStatus.FAILED
        )
        from objectstate import ObjectStateRegistry

        orchestrator = ObjectStateRegistry.get_object(plate_path)
        if orchestrator is not None:
            orchestrator._state = OrchestratorState.EXEC_FAILED
            self.host.emit_orchestrator_state(
                plate_path, OrchestratorState.EXEC_FAILED.value
            )

    async def _submit_debug_plate(
        self,
        *,
        run_spec: RunSpec,
        compile_artifact_id: str,
        debug_request: DebugPlateRunRequest,
        loop,
    ) -> None:
        if self.client_service.zmq_client is None:
            raise RuntimeError("ZMQ client is not connected")
        plate_path = run_spec.plate_path
        definition_pipeline = run_spec.definition_pipeline
        logger.info(
            "Submit debug run: plate=%s artifact_id=%s steps=%d fingerprint=%s",
            plate_path,
            compile_artifact_id,
            len(definition_pipeline),
            self._pipeline_fingerprint(definition_pipeline),
        )

        def submit_debug() -> Dict[str, Any]:
            return self.client_service.zmq_client.submit_debug_pipeline(
                plate_id=plate_path,
                pipeline_steps=definition_pipeline,
                global_config=run_spec.global_config,
                pipeline_config=run_spec.pipeline_config,
                compile_artifact_id=compile_artifact_id,
                debug_session_id=debug_request.debug_session_id,
                snapshot_store_ref=debug_request.snapshot_store_ref,
                snapshot_store_backend=debug_request.snapshot_store_backend,
                command_type=debug_request.command_type,
                selected_source_group=debug_request.selected_source_group,
                pause_step_indices=debug_request.pause_step_indices,
                start_step_index=debug_request.start_step_index,
                start_after_invocation_key=debug_request.start_after_invocation_key,
                replay_mode=debug_request.replay_mode,
                config_params=debug_request.config_params,
            )

        response = await self._run_blocking(loop, submit_debug)
        execution_id = response.get("execution_id")
        if execution_id:
            self.host.plate_execution_ids[plate_path] = execution_id
            self.host.current_execution_id = execution_id

        if response.get("status") == "accepted":
            self.host.emit_status(f"Submitted debug run for {plate_path}")
            if execution_id:
                self._start_completion_poller(execution_id, plate_path)
            return

        error_msg = response.get("message", "Unknown error")
        logger.error("Debug run %s submission failed: %s", plate_path, error_msg)
        self.host.emit_error(f"Debug submission failed for {plate_path}: {error_msg}")
        self.host.execution_runtime.mark_terminal(
            plate_path,
            TerminalExecutionStatus.FAILED,
        )

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

    @staticmethod
    def _pipeline_fingerprint(definition_pipeline: List) -> str:
        import openhcs.serialization.pycodify_formatters  # noqa: F401
        from pycodify import Assignment, generate_python_source

        pipeline_code = generate_python_source(
            Assignment("pipeline_steps", definition_pipeline),
            header="# Edit this pipeline and save to apply changes",
            clean_mode=True,
        )
        return hashlib.sha256(pipeline_code.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _pipeline_step_names(definition_pipeline: List) -> List[str]:
        return [str(getattr(step, "name", "<unnamed>")) for step in definition_pipeline]

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

    def _start_completion_poller(self, execution_id: str, plate_path: str) -> None:
        class _ClientDisconnected(RuntimeError):
            pass

        def _poll_status(polled_execution_id: str) -> Dict[str, Any]:
            if self.client_service.zmq_client is None:
                raise _ClientDisconnected("ZMQ client disconnected")
            return self.client_service.zmq_client.get_status(polled_execution_id)

        def _on_running(_execution_id: str, _execution_payload: Dict[str, Any]) -> None:
            self.host.update_item_list()
            self.host.emit_status(f"▶️ Running {plate_path}")

        def _on_terminal(
            _execution_id: str, terminal_status: str, execution_payload: Dict[str, Any]
        ) -> None:
            current_execution_id = self.host.plate_execution_ids.get(plate_path)
            if current_execution_id != _execution_id:
                logger.info(
                    "Ignoring stale terminal status for %s: execution_id=%s current=%s",
                    plate_path,
                    _execution_id,
                    current_execution_id,
                )
                return
            parsed_terminal_status = parse_terminal_status(terminal_status)

            self.host.execution_runtime.mark_terminal(
                plate_path, parsed_terminal_status
            )
            result = self._build_terminal_result(
                terminal_status=parsed_terminal_status.value,
                execution_id=_execution_id,
                execution_payload=execution_payload,
            )
            self.host.notify_plate_completed(
                plate_path, parsed_terminal_status.value, result
            )
            self._check_all_completed()

        def _on_status_error(_execution_id: str, message: str) -> None:
            current_execution_id = self.host.plate_execution_ids.get(plate_path)
            if current_execution_id != _execution_id:
                logger.info(
                    "Ignoring stale status error for %s: execution_id=%s current=%s",
                    plate_path,
                    _execution_id,
                    current_execution_id,
                )
                return
            self.host.execution_runtime.mark_terminal(
                plate_path, TerminalExecutionStatus.FAILED
            )
            self.host.notify_plate_completed(
                plate_path,
                TerminalExecutionStatus.FAILED.value,
                {
                    "status": TerminalExecutionStatus.FAILED.value,
                    "execution_id": _execution_id,
                    "message": message,
                },
            )
            self._check_all_completed()

        def _on_poll_exception(_execution_id: str, error: Exception) -> bool:
            if isinstance(error, _ClientDisconnected):
                return False
            logger.warning("Error polling status for %s: %s", plate_path, error)
            return True

        policy = CallbackExecutionStatusPollPolicy(
            poll_status_fn=_poll_status,
            poll_interval_seconds_value=0.5,
            on_running_fn=_on_running,
            on_terminal_fn=_on_terminal,
            on_status_error_fn=_on_status_error,
            on_poll_exception_fn=_on_poll_exception,
        )

        def poll_completion() -> None:
            try:
                self._execution_status_poller.run(execution_id, policy)
            except Exception as error:
                logger.error(
                    "Error in completion poller for %s: %s",
                    plate_path,
                    error,
                    exc_info=True,
                )
                self.host.emit_error(f"{plate_path}: {error}")

        threading.Thread(target=poll_completion, daemon=True).start()

    @staticmethod
    def _build_terminal_result(
        *,
        terminal_status: str,
        execution_id: str,
        execution_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        status = parse_terminal_status(terminal_status)
        builder = TERMINAL_RESULT_BUILDERS[status]
        return builder(execution_id, execution_payload)

    @staticmethod
    def _build_complete_terminal_result(
        execution_id: str, execution_payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        results_summary = execution_payload.get("results_summary", {}) or {}
        output_plate_root = None
        auto_add_output_plate = None
        if isinstance(results_summary, dict):
            output_plate_root = results_summary.get("output_plate_root")
            auto_add_output_plate = results_summary.get(
                "auto_add_output_plate_to_plate_manager"
            )
        return {
            "status": TerminalExecutionStatus.COMPLETE.value,
            "execution_id": execution_id,
            "results": results_summary,
            "output_plate_root": output_plate_root,
            "auto_add_output_plate_to_plate_manager": auto_add_output_plate,
        }

    @staticmethod
    def _build_failed_terminal_result(
        execution_id: str, execution_payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "status": TerminalExecutionStatus.FAILED.value,
            "execution_id": execution_id,
            "message": execution_payload.get("error", "Unknown error"),
            "traceback": execution_payload.get("traceback", ""),
        }

    @staticmethod
    def _build_cancelled_terminal_result(
        execution_id: str, _execution_payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "status": TerminalExecutionStatus.CANCELLED.value,
            "execution_id": execution_id,
            "message": "Execution was cancelled",
        }

    def _coalesced_progress_update(self) -> None:
        if self.client_service.zmq_client is not None:
            self._server_info_poller.tick()
        if not self._progress_dirty:
            return
        self._progress_dirty = False
        self._runtime_projection = build_execution_runtime_projection_from_registry(
            self.host._progress_tracker
        )
        self.host.set_runtime_progress_projection(self._runtime_projection)
        self.host.set_execution_server_info(self._get_server_info_snapshot())
        self._emit_execution_server_status()
        self.host.update_item_list()

    def _on_progress(self, message: dict) -> None:
        try:
            event = ProgressEvent.from_dict(message)
            self.host._progress_tracker.register_event(event.execution_id, event)
            self._notify_debug_snapshot_available(event)
        except Exception as error:
            logger.warning("Failed to parse/register progress event: %s", error)
        finally:
            self._mark_progress_dirty()

    def _notify_debug_snapshot_available(self, event: ProgressEvent) -> None:
        if not event.context:
            return
        try:
            debug_context = DebugProgressContext.from_progress_context(event.context)
        except (KeyError, TypeError, ValueError):
            return
        if debug_context.snapshot_id is None:
            return
        notification = DebugSnapshotAvailableNotification(
            progress_event=event,
            debug_context=debug_context,
            snapshot=self._read_debug_snapshot_from_server(debug_context),
        )
        for listener in tuple(self._debug_snapshot_listeners):
            listener(notification)

    def _read_debug_snapshot_from_server(
        self,
        debug_context: DebugProgressContext,
    ) -> DebugSnapshot | None:
        if (
            debug_context.snapshot_store_ref is None
            or debug_context.snapshot_id is None
            or self.client_service.zmq_client is None
        ):
            return None
        try:
            return self.client_service.zmq_client.get_debug_snapshot(
                debug_session_id=debug_context.debug_session_id,
                snapshot_id=debug_context.snapshot_id,
                snapshot_store_ref=debug_context.snapshot_store_ref,
                snapshot_store_backend=debug_context.snapshot_store_backend,
            )
        except Exception as error:
            logger.debug("Server debug snapshot readback failed: %s", error)
            return None

    def _on_registry_event(self, _execution_id: str, _event: ProgressEvent) -> None:
        """Mark projection dirty when shared registry changes from any producer."""
        self._mark_progress_dirty()

    def _emit_execution_server_status(self) -> None:
        status_view = self._server_status_presenter.build_status_text(
            projection=self._runtime_projection,
            server_info=self._get_server_info_snapshot(),
        )
        self.host.emit_status(status_view.text)

    def _get_server_info_snapshot(self) -> ExecutionServerInfo | None:
        return self._server_info_poller.get_snapshot_copy()

    def _fetch_server_info_snapshot(self) -> ExecutionServerInfo:
        if self.client_service.zmq_client is None:
            raise RuntimeError("ZMQ client is not connected")
        pong = self.client_service.zmq_client.get_server_info_snapshot()
        parsed = self._server_info_parser.parse(pong.to_dict())
        if not isinstance(parsed, ExecutionServerInfo):
            raise ValueError(
                f"Expected ExecutionServerInfo, got {type(parsed).__name__}"
            )
        return parsed

    def _mark_progress_dirty(self) -> None:
        self._progress_dirty = True

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

    def _validate_pipeline_steps(self, pipeline: List) -> None:
        for step in pipeline:
            if step.func is None:
                raise AttributeError(
                    f"Step '{step.name}' has func=None. "
                    "This usually means the pipeline was loaded from a compiled state."
                )

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

    def _reset_progress_for_new_batch(self) -> None:
        self._runtime_projection = reset_progress_views_for_new_batch(
            self.host,
            projection=ExecutionRuntimeProjection(),
        )
        self._server_info_poller.reset()
        self._mark_progress_dirty()


TERMINAL_RESULT_BUILDERS = {
    TerminalExecutionStatus.COMPLETE: BatchWorkflowService._build_complete_terminal_result,
    TerminalExecutionStatus.FAILED: BatchWorkflowService._build_failed_terminal_result,
    TerminalExecutionStatus.CANCELLED: BatchWorkflowService._build_cancelled_terminal_result,
}
