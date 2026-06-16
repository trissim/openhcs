"""Debug execution orchestration for PyQt batch workflows."""

from __future__ import annotations

from asyncio import AbstractEventLoop
from collections.abc import Awaitable
import logging
from dataclasses import dataclass
from typing import Callable, TypeVar

from openhcs.core.debug import (
    DebugArtifactExportResponse,
    DebugArtifactRef,
    DebugCommandType,
    DebugExecutionConfig,
    DebugPausedWorkerStatus,
    DebugReplayMode,
)
from openhcs.pyqt_gui.widgets.shared.services.compile_workflow_service import (
    CompileWorkflowService,
)
from openhcs.pyqt_gui.widgets.shared.services.compile_batch_workflow_service import (
    CompileConfigParamsByPlate,
    ExplicitCompileConfigParamsByPlate,
)
from openhcs.pyqt_gui.widgets.shared.services.execution_state import (
    TerminalExecutionStatus,
)
from openhcs.pyqt_gui.widgets.shared.services.execution_submission_service import (
    ExecutionSubmissionService,
)
from openhcs.pyqt_gui.widgets.shared.services.plate_pipeline_request_builder import (
    RunSpec,
)
from openhcs.pyqt_gui.widgets.shared.services.zmq_client_service import ZMQClientService
from openhcs.runtime.zmq_execution_client import OpenHCSExecutionSubmission
from zmqruntime.messages import MessageFields, ResponseType

logger = logging.getLogger(__name__)
T = TypeVar("T")


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
    def config_params(self) -> dict:
        return self.execution_config.to_config_params()

    @property
    def compile_config_params(self) -> dict:
        return self.execution_config.compile_cache_config_params()


@dataclass(frozen=True)
class DebugCompileArtifactCacheKey:
    """Stable identity for reusable debug compile artifacts."""

    plate_path: str
    pipeline_fingerprint: str
    selected_source_group: str | None
    replay_mode: DebugReplayMode


@dataclass(frozen=True, slots=True)
class DebugSubmissionResponse:
    """Typed response from a debug execution submission."""

    status: ResponseType
    execution_id: str | None
    message: str | None

    @classmethod
    def from_wire(cls, response: dict) -> "DebugSubmissionResponse":
        return cls(
            status=ResponseType(response[MessageFields.STATUS]),
            execution_id=response.get(MessageFields.EXECUTION_ID),
            message=response.get(MessageFields.MESSAGE),
        )

    @property
    def accepted(self) -> bool:
        return self.status is ResponseType.ACCEPTED

    @property
    def failure_message(self) -> str:
        if self.message is not None:
            return self.message
        return f"Debug submission returned status {self.status.value!r}."


RunBlockingCallable = Callable[[AbstractEventLoop, Callable[[], T]], Awaitable[T]]
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
        client_service: ZMQClientService,
        run_blocking: RunBlockingCallable,
        compile_before_execution: CompileBeforeExecutionCallable,
        execution_submission: ExecutionSubmissionService,
    ) -> None:
        self._host = host
        self._client_service = client_service
        self._run_blocking = run_blocking
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
        cache_key = DebugCompileArtifactCacheKey(
            plate_path=run_spec.plate_path,
            pipeline_fingerprint=CompileWorkflowService.pipeline_fingerprint(
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
        if self._client_service.zmq_client is None:
            raise RuntimeError("ZMQ client is not connected")
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
            return self._client_service.zmq_client.submit_debug_pipeline(
                OpenHCSExecutionSubmission(
                    plate_id=plate_path,
                    execution_plate_id=execution_plate_path,
                    selected_pipeline_path=run_spec.selected_pipeline_path,
                    pipeline_steps=definition_pipeline,
                    global_config=run_spec.global_config,
                    pipeline_config=run_spec.pipeline_config,
                    compile_artifact_id=compile_artifact_id,
                    config_params=debug_request.config_params,
                ),
                debug_session_id=debug_request.debug_session_id,
                snapshot_store_ref=debug_request.snapshot_store_ref,
                snapshot_store_backend=debug_request.snapshot_store_backend,
                command_type=debug_request.command_type,
                selected_source_group=debug_request.selected_source_group,
                pause_step_indices=debug_request.pause_step_indices,
                start_step_index=debug_request.start_step_index,
                start_after_invocation_key=debug_request.start_after_invocation_key,
                replay_mode=debug_request.replay_mode,
            )

        response = DebugSubmissionResponse.from_wire(
            await self._run_blocking(loop, submit_debug)
        )
        execution_id = response.execution_id
        if execution_id:
            self._host.plate_execution_ids[plate_path] = execution_id
            self._host.current_execution_id = execution_id

        if response.accepted:
            self._host.emit_status(f"Submitted debug run for {plate_path}")
            if execution_id:
                self._execution_submission.start_completion_poller(
                    str(execution_id),
                    plate_path,
            )
            return

        error_msg = response.failure_message
        logger.error("Debug run %s submission failed: %s", plate_path, error_msg)
        self._host.emit_error(f"Debug submission failed for {plate_path}: {error_msg}")
        self._host.execution_runtime.mark_terminal(
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
            if self._client_service.zmq_client is None:
                raise RuntimeError("ZMQ client is not connected")
            return self._client_service.zmq_client.send_debug_worker_command(
                debug_session_id=debug_session_id,
                command_type=command_type,
            ).status

        status = await self._run_blocking(loop, send_command)
        self._host.emit_status(
            f"Debug worker {status.state.value} for session {debug_session_id[:8]}"
        )
        return status

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
            if self._client_service.zmq_client is None:
                raise RuntimeError("ZMQ client is not connected")
            return self._client_service.zmq_client.export_debug_artifact(
                debug_session_id=debug_session_id,
                artifact_ref=artifact_ref,
                export_root=export_root,
                snapshot_store_ref=snapshot_store_ref,
                snapshot_store_backend=snapshot_store_backend,
            )

        response = await self._run_blocking(loop, export_artifact)
        self._host.emit_status(f"Exported debug artifact to {response.exported_ref}")
        return response


def is_debug_workflow_service_export(name: str, value) -> bool:
    return (
        isinstance(value, type)
        and value.__module__ == __name__
        and not name.startswith("_")
    )


__all__ = tuple(
    name
    for name, value in globals().items()
    if is_debug_workflow_service_export(name, value)
)
