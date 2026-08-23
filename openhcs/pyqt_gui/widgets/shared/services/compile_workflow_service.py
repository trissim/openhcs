"""Compile workflow transport and request models for the PyQt batch UI."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Callable, List

from zmqruntime.execution import (
    CallbackBatchSubmitWaitPolicy,
    ExecutionSubmissionResponse,
    ExecutionWaitResult,
)

from openhcs.core.artifact_inspection import CompiledArtifactInspection
from openhcs.core.config import GlobalPipelineConfig, PipelineConfig
from openhcs.core.function_step_transport import FunctionStepTransportAuthority
from openhcs.core.pipeline_document import PipelineDocumentAuthority
from openhcs.pyqt_gui.widgets.shared.services.batch_context import (
    BatchWorkflowContext,
)
from openhcs.runtime.zmq_execution_client import (
    OpenHCSExecutionSubmission,
)
from openhcs.runtime.zmq_execution_signature import TransportValue
from openhcs.ui.shared.plate_scope_identity import PlateScopeIdentity

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
        transport_pipeline = FunctionStepTransportAuthority.normalize_pipeline(
            self.definition_pipeline
        )
        return OpenHCSExecutionSubmission(
            plate_id=self.scope_id,
            execution_plate_id=self.execution_plate_path,
            selected_pipeline_path=self.selected_pipeline_path,
            pipeline_document=PipelineDocumentAuthority.from_values(
                pipeline_config=self.pipeline_config, pipeline_steps=transport_pipeline
            ),
            global_config=global_config,
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


@dataclass(frozen=True, slots=True)
class PlateCompiledState:
    """Typed GUI state retained after one plate compilation."""

    compile_artifact_id: str
    definition_pipeline: tuple
    inspection: CompiledArtifactInspection

    def __post_init__(self) -> None:
        if self.compile_artifact_id != self.inspection.compile_artifact_id:
            raise ValueError(
                "PlateCompiledState compile artifact identity does not match its "
                "inspection projection."
            )


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
    ) -> CompiledArtifactInspection:
        return await self.wait_for_compile_completion(
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
        if display_plate_path is None:
            display_plate_path = request.plate_path

        def submit_compile() -> dict:
            logger.info(
                "Submit compile: plate=%s execution_plate=%s steps=%d fingerprint=%s",
                display_plate_path,
                self._display_execution_plate_path(
                    plate_path=request.plate_path,
                    execution_plate_path=request.execution_plate_path,
                ),
                len(request.definition_pipeline),
                self.pipeline_fingerprint(request.definition_pipeline),
            )
            return zmq_client.submit_compile(
                request.submission(
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
    ) -> CompiledArtifactInspection:
        if zmq_client is None:
            raise RuntimeError("ZMQ client is not connected")
        wait_result = ExecutionWaitResult.from_wire(
            await self._context.run_blocking(
                loop,
                lambda: zmq_client.wait_for_completion(execution_id),
            )
        )
        wait_result.require_complete(f"Compilation failed for {plate_path}")
        return await self.inspect_compile_artifact(
            zmq_client=zmq_client,
            loop=loop,
            compile_artifact_id=execution_id,
        )

    async def inspect_compile_artifact(
        self,
        *,
        zmq_client,
        loop,
        compile_artifact_id: str,
    ) -> CompiledArtifactInspection:
        """Fetch one compiler-owned artifact projection after compilation."""

        return await self._context.run_blocking(
            loop,
            lambda: zmq_client.get_compiled_artifact_inspection(compile_artifact_id),
        )

    @staticmethod
    def pipeline_fingerprint(definition_pipeline: List) -> str:
        definition_pipeline = CompileWorkflowService.normalize_pipeline_for_transport(
            definition_pipeline
        )
        pipeline_source = FunctionStepTransportAuthority.source_from_pipeline(
            definition_pipeline
        )
        return hashlib.sha256(pipeline_source.encode("utf-8")).hexdigest()[:12]

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
