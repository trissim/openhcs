"""Debug execution orchestration for PyQt batch workflows."""

from __future__ import annotations

import logging
from asyncio import AbstractEventLoop
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Callable

from zmqruntime.execution import ExecutionSubmissionResponse

from openhcs.core.debug import (
    DebugArtifactExportResponse,
    DebugArtifactRef,
    DebugCommandType,
    DebugExecutionConfig,
    DebugPausedWorkerStatus,
    DebugReplayMode,
)
from openhcs.core.execution_state import (
    TerminalExecutionStatus,
)
from openhcs.pyqt_gui.widgets.shared.services.batch_context import (
    BatchWorkflowContext,
)
from openhcs.pyqt_gui.widgets.shared.services.compile_batch_workflow_service import (
    CompileConfigParamsByPlate,
    ExplicitCompileConfigParamsByPlate,
)
from openhcs.pyqt_gui.widgets.shared.services.compile_workflow_service import (
    CompileWorkflowService,
)
from openhcs.pyqt_gui.widgets.shared.services.execution_submission_service import (
    ExecutionSubmissionService,
)
from openhcs.pyqt_gui.widgets.shared.services.plate_pipeline_request_builder import (
    RunSpec,
)
from openhcs.runtime.zmq_execution_client import ZMQExecutionRequestBuilder

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class DebugReplayScope:
    """Debug settings carried into the runtime replay contract."""

    selected_source_group: str | None
    replay_mode: DebugReplayMode = DebugReplayMode.WARM_ARTIFACT


@dataclass(frozen=True)
class DebugPlateRunRequest(DebugReplayScope):
    """Debug execution identity threaded through compile and run submission."""

    debug_session_id: str
    snapshot_store_ref: str
    snapshot_store_backend: str | None
    command_type: DebugCommandType
    pause_step_indices: tuple[int, ...]
    start_step_index: int = 0
    start_after_invocation_key: str | None = None

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
    def compile_config_params(self) -> dict:
        return self.execution_config.compile_cache_config_params()


@dataclass(frozen=True)
class DebugCompileArtifactCacheKey:
    """Runtime-compatible identity for reusable debug compile artifacts."""

    debug_replay_signature: str

    @classmethod
    def from_run_spec(
        cls,
        *,
        run_spec: RunSpec,
        debug_request: DebugPlateRunRequest,
    ) -> "DebugCompileArtifactCacheKey":
        signature_payload = ZMQExecutionRequestBuilder.from_task(
            run_spec.submission(
                global_config=run_spec.global_config,
                config_params=debug_request.compile_config_params,
            )
        ).request_payload
        return cls(
            debug_replay_signature=signature_payload.debug_replay_signature,
        )


CompileBeforeExecutionCallable = Callable[
    [list[RunSpec], AbstractEventLoop, CompileConfigParamsByPlate],
    Awaitable[dict[str, str]],
]


class DebugWorkflowService:
    """Owns debug compile reuse, debug submission, worker controls, and export."""

    def __init__(
        self,
        *,
        host,
        context: BatchWorkflowContext,
        compile_before_execution: CompileBeforeExecutionCallable,
        execution_submission: ExecutionSubmissionService,
    ) -> None:
        self._host = host
        self._context = context
        self._compile_before_execution = compile_before_execution
        self._execution_submission = execution_submission
        self._debug_compile_artifacts: dict[DebugCompileArtifactCacheKey, str] = {}

    async def compile_artifact_id(
        self,
        *,
        run_spec: RunSpec,
        debug_request: DebugPlateRunRequest,
        loop,
    ) -> str:
        cache_key = DebugCompileArtifactCacheKey.from_run_spec(
            run_spec=run_spec,
            debug_request=debug_request,
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

        compile_artifacts = await self._compile_before_execution(
            [run_spec],
            loop,
            ExplicitCompileConfigParamsByPlate(
                {run_spec.plate_path: debug_request.compile_config_params}
            ),
        )
        compile_artifact_id = compile_artifacts[run_spec.plate_path]
        if debug_request.replay_mode.retains_compile_artifact:
            self._debug_compile_artifacts[cache_key] = compile_artifact_id
        return compile_artifact_id

    async def submit_debug_plate(
        self,
        *,
        run_spec: RunSpec,
        compile_artifact_id: str,
        debug_request: DebugPlateRunRequest,
        loop,
    ) -> None:
        zmq_client = self._context.zmq.require_client()
        plate_path = run_spec.plate_path
        execution_plate_path = run_spec.execution_plate_path
        definition_pipeline = run_spec.definition_pipeline
        logger.info(
            "Submit debug run: plate=%s execution_plate=%s artifact_id=%s steps=%d fingerprint=%s",
            plate_path,
            execution_plate_path,
            compile_artifact_id,
            len(definition_pipeline),
            CompileWorkflowService.pipeline_fingerprint(definition_pipeline),
        )

        def submit_debug() -> dict:
            return zmq_client.submit_debug_pipeline(
                run_spec.submission(
                    global_config=run_spec.global_config,
                    compile_artifact_id=compile_artifact_id,
                ),
                debug_config=debug_request.execution_config,
            )

        response = ExecutionSubmissionResponse.from_wire(
            await self._context.run_blocking(loop, submit_debug)
        )

        if response.accepted:
            execution_id = response.require_execution_id("Debug submission")
            self._host.plate_terminal_activity_status.record_execution(
                plate_path,
                execution_id,
            )
            self._host.emit_status(f"Submitted debug run for {plate_path}")
            self._execution_submission.start_completion_poller(
                execution_id,
                plate_path,
            )
            return

        error_msg = response.require_failure_text("Debug submission")
        logger.error("Debug run %s submission failed: %s", plate_path, error_msg)
        self._host.emit_error(f"Debug submission failed for {plate_path}: {error_msg}")
        self._host.plate_terminal_activity_status.mark_terminal(
            plate_path,
            TerminalExecutionStatus.FAILED,
        )

    async def send_worker_command(
        self,
        *,
        debug_session_id: str,
        command_type: DebugCommandType,
        loop,
    ) -> DebugPausedWorkerStatus:
        def send_command() -> DebugPausedWorkerStatus:
            return (
                self._context.zmq.require_client()
                .send_debug_worker_command(
                    debug_session_id=debug_session_id,
                    command_type=command_type,
                )
                .status
            )

        status = await self._context.run_blocking(loop, send_command)
        self._host.emit_status(
            f"Debug worker {status.state.value} for session {debug_session_id[:8]}"
        )
        return status

    async def inspect_runtime(
        self,
        *,
        debug_session_id: str,
        loop,
    ):
        def inspect_runtime():
            return self._context.zmq.require_client().get_debug_runtime_inspection(
                debug_session_id=debug_session_id,
            )

        view_model = await self._context.run_blocking(loop, inspect_runtime)
        self._host.emit_status(
            f"Loaded runtime inspection for session {debug_session_id[:8]}"
        )
        return view_model

    async def export_artifact(
        self,
        *,
        debug_session_id: str,
        artifact_ref: DebugArtifactRef,
        export_root: str,
        snapshot_store_ref: str | None,
        snapshot_store_backend: str | None,
        loop,
    ) -> DebugArtifactExportResponse:
        def export_artifact() -> DebugArtifactExportResponse:
            return self._context.zmq.require_client().export_debug_artifact(
                debug_session_id=debug_session_id,
                artifact_ref=artifact_ref,
                export_root=export_root,
                snapshot_store_ref=snapshot_store_ref,
                snapshot_store_backend=snapshot_store_backend,
            )

        response = await self._context.run_blocking(loop, export_artifact)
        self._host.emit_status(f"Exported debug artifact to {response.exported_ref}")
        return response
