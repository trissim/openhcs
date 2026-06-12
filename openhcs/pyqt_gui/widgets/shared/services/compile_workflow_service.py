"""Compile workflow transport and request models for the PyQt batch UI."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, TypeVar

from openhcs.core.function_step_transport import FunctionStepTransportAuthority
from openhcs.runtime.zmq_execution_client import OpenHCSExecutionSubmission
from zmqruntime.execution import CallbackBatchSubmitWaitPolicy

logger = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass(frozen=True)
class PlatePipelineRequest:
    """Shared plate/pipeline/config identity for compile and run requests."""

    plate_path: str
    execution_plate_path: str
    selected_pipeline_path: str | None
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


CompileJobCallback = Callable[[CompileJob, int, int], None] | None
CompileJobStatusCallback = Callable[[CompileJob, str, int, int], None] | None
CompileJobErrorCallback = Callable[[CompileJob, Exception, int, int], None] | None
RunBlockingCallable = Callable[[object, Callable[[], T]], Any]
GlobalConfigProvider = Callable[[], Any]


class CompileWorkflowService:
    """Owns compile transport behavior independent of batch UI orchestration."""

    def __init__(
        self,
        *,
        global_config_provider: GlobalConfigProvider,
        run_blocking: RunBlockingCallable,
    ) -> None:
        self._global_config_provider = global_config_provider
        self._run_blocking = run_blocking

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
            plate_path=job.plate_path,
            execution_plate_path=job.execution_plate_path,
            selected_pipeline_path=job.selected_pipeline_path,
            display_plate_path=job.plate_path,
            definition_pipeline=job.definition_pipeline,
            pipeline_config=job.pipeline_config,
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
        plate_path: str,
        execution_plate_path: str | None,
        selected_pipeline_path: str | None,
        definition_pipeline: List,
        pipeline_config,
        config_params: dict[str, Any] | None = None,
        display_plate_path: str | None = None,
    ) -> CompileRequestResult:
        if zmq_client is None:
            raise RuntimeError("ZMQ client is not connected")
        definition_pipeline = self.normalize_pipeline_for_transport(
            definition_pipeline
        )
        display_plate_path = display_plate_path or plate_path

        def submit_compile() -> Dict[str, Any]:
            logger.info(
                "Submit compile: plate=%s execution_plate=%s steps=%d fingerprint=%s",
                display_plate_path,
                execution_plate_path or plate_path,
                len(definition_pipeline),
                self.pipeline_fingerprint(definition_pipeline),
            )
            return zmq_client.submit_compile(
                OpenHCSExecutionSubmission(
                    plate_id=plate_path,
                    execution_plate_id=execution_plate_path,
                    selected_pipeline_path=selected_pipeline_path,
                    pipeline_steps=definition_pipeline,
                    global_config=self._global_config_provider(),
                    pipeline_config=pipeline_config,
                    config_params=config_params,
                )
            )

        response = await self._run_blocking(loop, submit_compile)
        if response.get("status") != "accepted":
            raise RuntimeError(
                f"Compile submission failed for {display_plate_path}: "
                f"{response.get('message', 'Unknown error')}"
            )
        execution_id = response.get("execution_id")
        if not execution_id:
            raise RuntimeError(
                f"Compile submission missing execution_id for {display_plate_path}"
            )
        return CompileRequestResult(
            execution_id=str(execution_id),
            response=response,
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
        wait_result = await self._run_blocking(
            loop,
            lambda: zmq_client.wait_for_completion(execution_id),
        )
        if wait_result.get("status") != "complete":
            raise RuntimeError(
                f"Compilation failed for {plate_path}: "
                f"{wait_result.get('message', 'Unknown error')}"
            )

    @staticmethod
    def pipeline_fingerprint(definition_pipeline: List) -> str:
        import openhcs.serialization.pycodify_formatters  # noqa: F401
        from pycodify import Assignment, generate_python_source

        definition_pipeline = CompileWorkflowService.normalize_pipeline_for_transport(
            definition_pipeline
        )
        pipeline_code = generate_python_source(
            Assignment("pipeline_steps", definition_pipeline),
            header="# Edit this pipeline and save to apply changes",
            clean_mode=True,
        )
        return hashlib.sha256(pipeline_code.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def normalize_pipeline_for_transport(definition_pipeline: List) -> List:
        return FunctionStepTransportAuthority.normalize_pipeline(definition_pipeline)

    @staticmethod
    def pipeline_step_names(definition_pipeline: List) -> List[str]:
        return [str(getattr(step, "name", "<unnamed>")) for step in definition_pipeline]


def is_compile_workflow_export(name: str, value: object) -> bool:
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
