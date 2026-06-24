"""Compile workflow transport and request models for the PyQt batch UI."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List

from openhcs.core.config import GlobalPipelineConfig, PipelineConfig
from openhcs.core.function_step_transport import FunctionStepTransportAuthority
from openhcs.pyqt_gui.services.plate_scope_identity import PlateScopeIdentity
from openhcs.pyqt_gui.widgets.shared.services.batch_context import (
    BatchWorkflowContext,
)
from openhcs.runtime.zmq_execution_client import (
    OpenHCSExecutionSubmission,
    PycodifiedPipelineStepSource,
    PycodifiedSource,
)
from openhcs.runtime.zmq_execution_signature import TransportValue
from openhcs.runtime.zmq_pipeline_transport import PipelineStepsBoundary
from zmqruntime.execution import (
    CallbackBatchSubmitWaitPolicy,
    ExecutionSubmissionResponse,
    ExecutionWaitResult,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlateExecutionIdentity:
    """Stable plate/workspace identity shared by compile, run, and debug flows."""

    plate_scope: PlateScopeIdentity
    execution_plate_path: str | None
    selected_pipeline_path: str | None

    @property
    def scope_id(self) -> str:
        return self.plate_scope.scope_id

    @property
    def plate_path(self) -> str:
        return self.scope_id

    @classmethod
    def from_request(cls, request: "PlatePipelineRequest") -> "PlateExecutionIdentity":
        return cls(
            plate_scope=request.plate_scope,
            execution_plate_path=request.execution_plate_path,
            selected_pipeline_path=request.selected_pipeline_path,
        )


@dataclass(frozen=True)
class PlatePipelineRequest(PlateExecutionIdentity):
    """Shared plate/pipeline/config identity for compile and run requests."""

    definition_pipeline: List
    pipeline_config: PipelineConfig

    def submission(
        self,
        *,
        global_config: GlobalPipelineConfig,
        compile_artifact_id: str | None = None,
        config_params: dict[str, TransportValue] | None = None,
    ) -> OpenHCSExecutionSubmission:
        return OpenHCSExecutionSubmission(
            plate_id=self.scope_id,
            execution_plate_id=self.execution_plate_path,
            selected_pipeline_path=self.selected_pipeline_path,
            pipeline_steps=self.definition_pipeline,
            global_config=global_config,
            pipeline_config=self.pipeline_config,
            compile_artifact_id=compile_artifact_id,
            config_params=config_params,
        )


@dataclass(frozen=True)
class CompileJob(PlatePipelineRequest):
    """Single compile unit for a plate."""

    plate_name: str
    config_params: dict[str, TransportValue] | None = None


@dataclass(frozen=True)
class CompileRequestResult:
    """Accepted compile submission returned by the execution server."""

    execution_id: str


CompileJobCallback = Callable[[CompileJob, int, int], None] | None
CompileJobStatusCallback = Callable[[CompileJob, str, int, int], None] | None
CompileJobErrorCallback = Callable[[CompileJob, Exception, int, int], None] | None


class CompileWorkflowService:
    """Owns compile transport behavior independent of batch UI orchestration."""

    def __init__(
        self,
        *,
        context: BatchWorkflowContext,
    ) -> None:
        self._context = context

    def make_policy(
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
        """Build the generic submit/wait policy for compile batches."""

        return CallbackBatchSubmitWaitPolicy(
            submit_fn=lambda job: self.submit_compile_job(
                job=job,
                zmq_client=zmq_client,
                loop=loop,
            ),
            wait_fn=lambda submission_id, job: self.wait_compile_job(
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

    async def submit_compile_job(self, *, job: CompileJob, zmq_client, loop) -> str:
        response = await self.submit_compile_request(
            zmq_client=zmq_client,
            loop=loop,
            request=job,
            config_params=job.config_params,
        )
        return response.execution_id

    async def wait_compile_job(
        self,
        *,
        submission_id: str,
        job: CompileJob,
        zmq_client,
        loop,
    ) -> None:
        await self.wait_for_compile_completion(
            zmq_client=zmq_client,
            loop=loop,
            execution_id=submission_id,
            plate_path=job.plate_path,
        )

    async def submit_compile_request(
        self,
        *,
        zmq_client,
        loop,
        request: PlatePipelineRequest,
        config_params: dict[str, TransportValue] | None = None,
        display_plate_path: str | None = None,
    ) -> CompileRequestResult:
        if zmq_client is None:
            raise RuntimeError("ZMQ client is not connected")
        normalized_pipeline = self.normalize_pipeline_for_transport(
            request.definition_pipeline
        )
        transport_request = PlatePipelineRequest(
            plate_scope=request.plate_scope,
            execution_plate_path=request.execution_plate_path,
            selected_pipeline_path=request.selected_pipeline_path,
            definition_pipeline=normalized_pipeline,
            pipeline_config=request.pipeline_config,
        )
        if display_plate_path is None:
            display_plate_path = request.plate_path

        def submit_compile() -> dict:
            logger.info(
                "Submit compile: plate=%s execution_plate=%s steps=%d fingerprint=%s",
                display_plate_path,
                self._display_execution_plate_path(
                    plate_path=transport_request.plate_path,
                    execution_plate_path=transport_request.execution_plate_path,
                ),
                len(transport_request.definition_pipeline),
                self.pipeline_fingerprint(transport_request.definition_pipeline),
            )
            return zmq_client.submit_compile(
                transport_request.submission(
                    global_config=self._context.global_config(),
                    config_params=config_params,
                )
            )

        response = ExecutionSubmissionResponse.from_wire(
            await self._context.run_blocking(loop, submit_compile)
        )
        if not response.accepted:
            raise RuntimeError(
                f"Compile submission failed for {display_plate_path}: "
                f"{response.require_failure_text('Compile submission')}"
            )
        execution_id = response.require_execution_id("Compile submission")
        return CompileRequestResult(
            execution_id=execution_id,
        )

    async def wait_for_compile_completion(
        self,
        *,
        zmq_client,
        loop,
        execution_id: str,
        plate_path: str,
    ) -> None:
        if zmq_client is None:
            raise RuntimeError("ZMQ client is not connected")
        wait_result = ExecutionWaitResult.from_wire(
            await self._context.run_blocking(
                loop,
                lambda: zmq_client.wait_for_completion(execution_id),
            )
        )
        wait_result.require_complete(f"Compilation failed for {plate_path}")

    @staticmethod
    def pipeline_fingerprint(definition_pipeline: List) -> str:
        definition_pipeline = CompileWorkflowService.normalize_pipeline_for_transport(
            definition_pipeline
        )
        pipeline_source = PycodifiedPipelineStepSource(
            PipelineStepsBoundary(definition_pipeline)
        ).source()
        return PycodifiedSource(pipeline_source).sha_label()

    @staticmethod
    def normalize_pipeline_for_transport(definition_pipeline: List) -> List:
        return FunctionStepTransportAuthority.normalize_pipeline(definition_pipeline)

    @staticmethod
    def pipeline_step_names(definition_pipeline: List) -> List[str]:
        return [str(step.name) for step in definition_pipeline]

    @staticmethod
    def _display_execution_plate_path(
        *,
        plate_path: str,
        execution_plate_path: str | None,
    ) -> str:
        if execution_plate_path is None:
            return plate_path
        return execution_plate_path


def is_compile_workflow_export(name: str, value) -> bool:
    return (
        isinstance(value, type)
        and value.__module__ == __name__
        and not name.startswith("_")
    )


__all__ = tuple(
    name
    for name, value in globals().items()
    if is_compile_workflow_export(name, value)
)
