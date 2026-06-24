"""OpenHCS execution server built on zmqruntime ExecutionServer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
import logging
import sys
import time
import json
from types import ModuleType
from typing import Any

from metaclass_registry import AutoRegisterMeta
from zmqruntime.execution import ExecutionServer
from zmqruntime.messages import (
    ExecuteRequest,
    ExecutionStatus,
    MessageFields,
    ResponseType,
    StatusRequest,
)

from zmqruntime.transport import coerce_transport_mode
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG
from openhcs.runtime.zmq_debug_control import DebugControlMessageRouter
from openhcs.runtime.zmq_compilation import (
    ZMQCompilationRequest,
    ZMQCompileArtifactRecord,
)
from openhcs.runtime.zmq_execution_signature import (
    OpenHCSExecutionConfigBundle,
    OpenHCSExecutionConfigCarrier,
    ZMQExecutionRequestPayload,
)
from openhcs.runtime.zmq_orchestrator_environment import (
    ZMQOrchestratorEnvironmentRequest,
)
from openhcs.runtime.zmq_progress import ImmediateZMQProgressQueue, ZMQProgressEmitter
from openhcs.runtime.zmq_server_hooks import (
    ZMQPongResponseEnricher,
    ZMQResultsSummaryEnricher,
    ZMQWorkerCleanup,
)
from openhcs.runtime.zmq_worker_execution import ZMQWorkerExecutionRequest
from openhcs.core.config import GlobalPipelineConfig, PipelineConfig
from openhcs.core.progress import ProgressEvent, ProgressPhase
from openhcs.core.steps.abstract import AbstractStep
from openhcs.runtime.zmq_pipeline_transport import (
    PipelineSourceExport,
    PipelineStepsNamespaceProjection,
    PipelineStepsBoundary,
    PipelineStepsCarrier,
)


logger = logging.getLogger(__name__)


CONFIG_CODE_OBJECT_NAME = "config"


@dataclass(frozen=True, slots=True)
class ConfigCodeNamespace:
    """Validated namespace produced by config-code execution."""

    values: dict[str, Any]

    def require_config(self, missing_message: str):
        if CONFIG_CODE_OBJECT_NAME not in self.values:
            raise ValueError(missing_message)
        return self.values[CONFIG_CODE_OBJECT_NAME]


@dataclass(frozen=True, slots=True)
class ZMQConfigParamsPayload:
    """Formal defaults for the legacy config_params transport payload."""

    num_workers: int
    output_dir_suffix: str
    materialization_backend: str
    axis_filter: list[str] | tuple[str, ...] | str | int | None
    use_threading: bool

    @classmethod
    def from_mapping(cls, params) -> "ZMQConfigParamsPayload":
        return cls(
            num_workers=int(cls._value(params, "num_workers", 4)),
            output_dir_suffix=str(cls._value(params, "output_dir_suffix", "_output")),
            materialization_backend=str(
                cls._value(params, "materialization_backend", "disk")
            ),
            axis_filter=cls._value(params, "well_filter", None),
            use_threading=bool(cls._value(params, "use_threading", False)),
        )

    @staticmethod
    def _value(params, field_name: str, formal_default):
        if field_name in params:
            return params[field_name]
        return formal_default

    def concrete_axis_filter(self) -> list[str] | None:
        if self.axis_filter is None:
            return None
        if isinstance(self.axis_filter, list):
            return [str(axis_id) for axis_id in self.axis_filter]
        if isinstance(self.axis_filter, tuple):
            return [str(axis_id) for axis_id in self.axis_filter]
        raise TypeError(
            "ZMQ config_params well_filter must be a concrete axis-id sequence, "
            f"got {type(self.axis_filter).__name__}."
        )


@dataclass(frozen=True, slots=True)
class ZMQResolvedConfig(OpenHCSExecutionConfigCarrier):
    """Resolved execution configs from one request payload."""

    registry_key = "zmq_resolved_config"

    configs: OpenHCSExecutionConfigBundle

    @property
    def execution_config_bundle(self) -> OpenHCSExecutionConfigBundle:
        return self.configs

    def with_global_config(
        self,
        global_config: GlobalPipelineConfig,
    ) -> "ZMQResolvedConfig":
        return ZMQResolvedConfig(
            configs=self.configs.with_global_pipeline(global_config),
        )


class ZMQPipelineConfigCodePolicy(ABC, metaclass=AutoRegisterMeta):
    """Policy for resolving the optional pipeline config code payload."""

    __registry_key__ = "registry_key"
    __skip_if_no_key__ = True
    registry_key = None

    @abstractmethod
    def resolve(self) -> PipelineConfig:
        raise NotImplementedError

    @classmethod
    def from_transport_code(cls, pipeline_config_code) -> "ZMQPipelineConfigCodePolicy":
        if pipeline_config_code is None:
            return cls.__registry__["default"]()
        return cls.__registry__["provided"](pipeline_config_code)


@dataclass(frozen=True, slots=True)
class DefaultZMQPipelineConfigCodePolicy(ZMQPipelineConfigCodePolicy):
    """Default pipeline config policy for requests without config code."""

    registry_key = "default"

    def resolve(self) -> PipelineConfig:
        return PipelineConfig()


@dataclass(frozen=True, slots=True)
class ProvidedZMQPipelineConfigCodePolicy(ZMQPipelineConfigCodePolicy):
    """Pipeline config policy backed by provided config code."""

    registry_key = "provided"
    pipeline_config_code: str

    def resolve(self) -> PipelineConfig:
        return ZMQExecutionServer._config_from_code(
            self.pipeline_config_code,
            "pipeline_config_code must define 'config'",
        )


@dataclass(frozen=True, slots=True)
class ZMQExecutionContext(PipelineStepsCarrier, OpenHCSExecutionConfigCarrier):
    """Request context shared by compile and execution completion phases."""

    registry_key = "zmq_execution_context"

    execution_id: str
    request_payload: ZMQExecutionRequestPayload
    execution_pipeline: PipelineStepsBoundary
    config_carrier: ZMQResolvedConfig

    @property
    def pipeline_steps_boundary(self) -> PipelineStepsBoundary:
        return self.execution_pipeline

    @property
    def execution_config_bundle(self) -> OpenHCSExecutionConfigBundle:
        return self.config_carrier.execution_config_bundle

    @property
    def plate_id(self) -> str:
        return self.request_payload.plate_id

    @property
    def execution_plate_id(self) -> str | None:
        return self.request_payload.execution_plate_id

    @property
    def config_params(self) -> dict | None:
        return self.request_payload.config_params

    @property
    def compile_only(self) -> bool:
        return self.request_payload.compile_only

    @property
    def compile_artifact_id(self) -> str | None:
        return self.request_payload.compile_artifact_id

    @property
    def request_signature(self) -> str:
        return self.request_payload.request_signature

    @property
    def debug_replay_signature(self) -> str:
        return self.request_payload.debug_replay_signature

    @property
    def pipeline_sha(self) -> str:
        return self.request_payload.pipeline_sha

    def validate_compile_request(self) -> None:
        if self.compile_only and self.compile_artifact_id:
            raise ValueError("compile_only and compile_artifact_id cannot both be set")

    def with_global_config(
        self,
        global_config: GlobalPipelineConfig,
    ) -> "ZMQExecutionContext":
        return ZMQExecutionContext(
            execution_id=self.execution_id,
            request_payload=self.request_payload,
            execution_pipeline=self.execution_pipeline,
            config_carrier=self.config_carrier.with_global_config(global_config),
        )

    def progress_context(self) -> dict:
        return {
            MessageFields.EXECUTION_ID: self.execution_id,
            MessageFields.PLATE_ID: self.plate_id,
            MessageFields.AXIS_ID: "",
        }


class ZMQExecutionServer(ExecutionServer):
    """OpenHCS-specific execution server."""

    _server_type = "execution"

    def __init__(
        self,
        port: int | None = None,
        host: str = "*",
        log_file_path: str | None = None,
        transport_mode=None,
    ):
        super().__init__(
            port=port or OPENHCS_ZMQ_CONFIG.default_port,
            host=host,
            log_file_path=log_file_path,
            transport_mode=coerce_transport_mode(transport_mode),
            config=OPENHCS_ZMQ_CONFIG,
        )
        self._compile_status: str | None = None
        self._compile_message: str | None = None
        self._compile_status_expires_at: float | None = None
        self._worker_assignments_by_execution: dict[str, dict[str, list[str]]] = {}
        self._compiled_artifacts: dict[str, dict[str, Any]] = {}
        self._compiled_artifact_ttl_seconds: float = 30.0 * 60.0

    def handle_control_message(self, message):
        if DebugControlMessageRouter.handles(message):
            return DebugControlMessageRouter.handle(message)
        return super().handle_control_message(message)

    @staticmethod
    def _validate_worker_claim(
        worker_slot: str,
        owned_wells: list[str],
        assignments: dict[str, list[str]],
    ) -> None:
        if worker_slot not in assignments:
            raise ValueError(
                f"Unknown worker_slot '{worker_slot}'. Expected one of: {list(assignments.keys())}"
            )
        expected = assignments[worker_slot]
        if sorted(owned_wells) != sorted(expected):
            raise ValueError(
                f"Invalid worker claim for {worker_slot}: expected={expected}, got={owned_wells}"
            )

    def _flush_progress_only(self) -> None:
        """Flush only progress messages to ZMQ, without processing control messages.

        This is called during synchronous execution when we can't receive from
        the control socket (would block or fail with NOBLOCK).
        """
        if not self.data_socket:
            return
        import queue

        logger = logging.getLogger(__name__)
        count = 0
        while True:
            try:
                progress_update = self.progress_queue.get_nowait()
            except queue.Empty:
                if count > 0:
                    logger.info(f"Flushed {count} progress update(s) to ZMQ")
                break
            event = ProgressEvent.from_dict(progress_update)
            progress_payload = event.to_dict()
            logger.info(
                "Flushing to ZMQ: step_name=%r, axis=%r, plate_id=%r, "
                "percent=%r, total_wells=%r",
                event.step_name,
                event.axis_id,
                event.plate_id,
                event.percent,
                event.total_wells,
            )
            json_str = json.dumps(progress_payload)
            logger.info(f"Full JSON being sent: {json_str[:300]}")
            self.data_socket.send_string(json_str)
            count += 1

    def _set_compile_status(
        self, status: str, message: str | None = None, ttl_seconds: float = 4.0
    ) -> None:
        self._compile_status = status
        self._compile_message = message
        self._compile_status_expires_at = time.time() + ttl_seconds

    def _get_compile_status(self) -> tuple[str | None, str | None]:
        if self._compile_status_expires_at is None:
            return None, None
        if time.time() > self._compile_status_expires_at:
            self._compile_status = None
            self._compile_message = None
            self._compile_status_expires_at = None
            return None, None
        return self._compile_status, self._compile_message

    def _cleanup_compiled_artifacts(self) -> None:
        now = time.time()
        expired_ids = [
            artifact_id
            for artifact_id, artifact in self._compiled_artifacts.items()
            if now - artifact["created_at"] > self._compiled_artifact_ttl_seconds
        ]
        for artifact_id in expired_ids:
            del self._compiled_artifacts[artifact_id]
        if expired_ids:
            logger.info("Cleaned up %d expired compile artifact(s)", len(expired_ids))

    def _create_pong_response(self):
        self._cleanup_compiled_artifacts()
        return ZMQPongResponseEnricher(
            active_executions=self.active_executions,
            compile_status=self._get_compile_status,
        ).enrich(super()._create_pong_response())

    def _enqueue_progress(self, progress_update: dict) -> None:
        event = ProgressEvent.from_dict(progress_update)
        if event.total_wells is not None:
            logger.info(
                "_enqueue_progress: total_wells=%s, keys=%s, step_name=%s",
                event.total_wells,
                list(progress_update.keys()),
                event.step_name,
            )
        self.progress_queue.put(event.to_dict())

    def _forward_worker_progress(self, worker_queue) -> None:
        import logging

        logger = logging.getLogger(__name__)
        while True:
            progress_update = worker_queue.get()
            if progress_update is None:
                logger.info("Progress forwarder received None, exiting")
                break
            event = ProgressEvent.from_dict(progress_update)
            assignments = self._worker_assignments_for_execution(event.execution_id)

            # Pipeline-level INIT events (e.g. viewer launch) bypass worker
            # claim validation — they carry no worker_slot / owned_wells.
            if event.phase == ProgressPhase.INIT and not event.axis_id:
                self.progress_queue.put(event.to_dict())
                continue

            if event.worker_slot is None:
                raise ValueError(
                    "Worker progress missing required field 'worker_slot': "
                    f"{progress_update}"
                )
            if event.owned_wells is None:
                raise ValueError(
                    "Worker progress missing required field 'owned_wells': "
                    f"{progress_update}"
                )
            owned_wells = [str(axis_id) for axis_id in event.owned_wells]
            self._validate_worker_claim(event.worker_slot, owned_wells, assignments)
            # Attach topology metadata to every worker progress event so the UI
            # cannot lose worker/well ownership due first-message ordering.
            enriched_event = event.with_worker_topology(
                worker_assignments=assignments,
                total_wells=sorted(
                    {
                        axis_id
                        for assigned_axes in assignments.values()
                        for axis_id in assigned_axes
                    }
                ),
            )
            logger.info(
                "Forwarding progress: pid=%s, axis=%s, step_name=%s, worker_slot=%s",
                enriched_event.pid,
                enriched_event.axis_id,
                enriched_event.step_name,
                enriched_event.worker_slot,
            )
            self.progress_queue.put(enriched_event.to_dict())

    def _worker_assignments_for_execution(
        self,
        execution_id: str,
    ) -> dict[str, list[str]]:
        if execution_id not in self._worker_assignments_by_execution:
            raise ValueError(
                f"Missing worker assignments for execution_id={execution_id}"
            )
        return self._worker_assignments_by_execution[execution_id]

    def _get_worker_info(self):
        """Return raw worker info (no enrichment).

        Worker axis_id comes from progress tracker, not from ping responses.
        Ping is for process tracking (CPU, memory), not application state.
        """
        return super()._get_worker_info()

    def _attach_results_summary_extras(
        self, execution_id: str, record, execution_payload: dict | None = None
    ) -> None:
        ZMQResultsSummaryEnricher(self.active_executions).attach(
            execution_id=execution_id,
            record=record,
            execution_payload=execution_payload,
        )

    def run_execution(self, execution_id, request, record):
        """Run an execution and enrich results_summary with output plate path.

        The base zmqruntime ExecutionServer only populates well_count/wells in
        results_summary. OpenHCS needs the final output plate root (computed by
        path planning during compilation) so the UI can optionally auto-add it
        as a new orchestrator in Plate Manager.
        """
        super().run_execution(execution_id, request, record)

        try:
            self._attach_results_summary_extras(
                execution_id=execution_id, record=record
            )
        except Exception as e:
            logger.warning(
                "[%s] Failed to attach output_plate_root to results_summary: %s",
                execution_id,
                e,
            )

    def handle_status(self, msg):
        response = super().handle_status(msg)
        return ZMQResultsSummaryEnricher(
            self.active_executions
        ).attach_to_status_response(
            execution_id=StatusRequest.from_dict(msg).execution_id,
            response=response,
        )

    def execute_task(self, execution_id: str, request: ExecuteRequest):
        return self._execute_pipeline(
            execution_id,
            ZMQExecutionRequestPayload.from_execute_request(request),
        )

    def _execute_pipeline(
        self,
        execution_id: str,
        request_payload: ZMQExecutionRequestPayload,
    ):
        logger.info("[%s] Starting plate %s", execution_id, request_payload.plate_id)

        import openhcs.processing.func_registry as func_registry_module

        logger.info(
            "[%s] Registry initialized status BEFORE check: %s",
            execution_id,
            func_registry_module._registry_initialized,
        )
        with func_registry_module._registry_lock:
            if not func_registry_module._registry_initialized:
                logger.info("[%s] Initializing registry...", execution_id)
                func_registry_module._auto_initialize_registry()
                logger.info(
                    "[%s] Registry initialized status AFTER init: %s",
                    execution_id,
                    func_registry_module._registry_initialized,
                )
            else:
                logger.info("[%s] Registry already initialized, skipping", execution_id)

        self._cleanup_compiled_artifacts()

        resolved_config = self._resolve_request_config(request_payload)

        module_name = f"openhcs_zmq_pipeline_{request_payload.pipeline_sha}"
        module = ModuleType(module_name)
        module.__file__ = f"<{module_name}>"
        sys.modules[module_name] = module
        try:
            exec(
                compile(
                    request_payload.pipeline_code,
                    module.__file__,
                    "exec",
                ),
                module.__dict__,
            )
        finally:
            sys.modules.pop(module_name, None)
        execution_pipeline = PipelineStepsNamespaceProjection(
            module.__dict__
        ).boundary_or_none()
        if not execution_pipeline:
            raise ValueError(
                f"Code must define {PipelineSourceExport.PIPELINE_STEPS.value!r}"
            )
        request_context = ZMQExecutionContext(
            execution_id=execution_id,
            request_payload=request_payload,
            execution_pipeline=execution_pipeline,
            config_carrier=resolved_config,
        )
        request_context.validate_compile_request()
        logger.info(
            "[%s] Request received: plate=%s compile_only=%s artifact_id=%s step_count=%d pipeline_sha=%s request_sig=%s",
            execution_id,
            request_context.plate_id,
            request_context.compile_only,
            request_context.compile_artifact_id,
            len(request_context.pipeline_steps),
            request_context.pipeline_sha,
            request_context.request_signature[:12],
        )

        try:
            return self._execute_with_orchestrator(request_context)
        except Exception as e:
            if request_payload.compile_only:
                self._set_compile_status("compile failed", str(e))
            raise

    def _resolve_request_config(
        self,
        request_payload: ZMQExecutionRequestPayload,
    ) -> ZMQResolvedConfig:
        if request_payload.config_code is not None:
            global_config = self._global_config_from_code(request_payload.config_code)
            pipeline_config = ZMQPipelineConfigCodePolicy.from_transport_code(
                request_payload.pipeline_config_code
            ).resolve()
            return ZMQResolvedConfig(
                configs=OpenHCSExecutionConfigBundle(
                    global_pipeline=global_config,
                    plate_pipeline=pipeline_config,
                ),
            )
        if request_payload.config_params is not None:
            global_config, pipeline_config = self._build_config_from_params(
                ZMQConfigParamsPayload.from_mapping(request_payload.config_params)
            )
            return ZMQResolvedConfig(
                configs=OpenHCSExecutionConfigBundle(
                    global_pipeline=global_config,
                    plate_pipeline=pipeline_config,
                ),
            )
        raise ValueError("Either config_params or config_code required")

    @staticmethod
    def _global_config_from_code(config_code: str):
        from openhcs.core.config import GlobalPipelineConfig

        if (
            "GlobalPipelineConfig(\n\n)" in config_code
            or "GlobalPipelineConfig()" in config_code
        ):
            return GlobalPipelineConfig()
        return ZMQExecutionServer._config_from_code(
            config_code,
            "config_code must define 'config'",
        )

    @staticmethod
    def _config_from_code(config_code: str, missing_message: str):
        namespace = {}
        exec(config_code, namespace)
        return ConfigCodeNamespace(namespace).require_config(missing_message)

    def _build_config_from_params(self, p: ZMQConfigParamsPayload):
        from openhcs.core.config import (
            GlobalPipelineConfig,
            MaterializationBackend,
            PathPlanningConfig,
            StepWellFilterConfig,
            VFSConfig,
            PipelineConfig,
        )

        return (
            GlobalPipelineConfig(
                num_workers=p.num_workers,
                path_planning_config=PathPlanningConfig(
                    output_dir_suffix=p.output_dir_suffix
                ),
                vfs_config=VFSConfig(
                    materialization_backend=MaterializationBackend(p.materialization_backend)
                ),
                step_well_filter_config=StepWellFilterConfig(
                    well_filter=p.axis_filter
                ),
                use_threading=p.use_threading,
            ),
            PipelineConfig(),
        )

    def _execute_with_orchestrator(
        self,
        request_context: ZMQExecutionContext,
    ):
        from openhcs.core.debug import (
            DebugPausedWorkerRegistry,
            DebugReplayMode,
        )

        environment = ZMQOrchestratorEnvironmentRequest(
            execution_id=request_context.execution_id,
            plate_id=request_context.plate_id,
            execution_plate_id=request_context.execution_plate_id,
            selected_pipeline_path=request_context.request_payload.selected_pipeline_path,
            global_config=request_context.global_config,
            config_params=request_context.config_params,
        ).prepare()
        request_context = request_context.with_global_config(environment.global_config)
        debug_execution_policy = environment.debug_execution_policy
        debug_execution_config = environment.debug_execution_config
        plate_path_str = environment.plate_path_str

        progress_context = request_context.progress_context()
        progress_emitter = ZMQProgressEmitter(
            enqueue=self._enqueue_progress,
            execution_id=request_context.execution_id,
            plate_id=request_context.plate_id,
        )

        wells: list[str] | None = None
        compilation_resolved = False
        try:
            self._emit_compile_started(
                progress_emitter,
                request_context.pipeline_steps,
                request_context.compile_artifact_id,
            )
            orchestrator = self._initialize_orchestrator(
                request_context.execution_id,
                plate_path_str,
                request_context.pipeline_config,
            )
            self._raise_if_cancelled(request_context.execution_id, "initialization")
            wells = self._wells_for_execution(
                request_context.config_params,
                orchestrator,
                debug_execution_policy,
            )
            self._emit_planned_init_started(
                progress_emitter,
                request_context.pipeline_steps,
                wells,
                request_context.compile_artifact_id,
            )
            compilation = self._resolve_compilation(
                request_context=request_context,
                orchestrator=orchestrator,
                wells=wells,
                debug_execution_config=debug_execution_config,
                progress_emitter=progress_emitter,
            )
            compilation_resolved = True
            self._record_compilation_outputs(request_context.execution_id, compilation)
            self._raise_if_cancelled(request_context.execution_id, "compilation")
            return self._finish_compilation_or_execute(
                request_context=request_context,
                orchestrator=orchestrator,
                compilation=compilation,
                progress_context=progress_context,
                debug_execution_policy=debug_execution_policy,
            )
        except Exception as error:
            self._emit_compile_failure(
                progress_emitter=progress_emitter,
                compile_artifact_id=request_context.compile_artifact_id,
                compilation_resolved=compilation_resolved,
                wells=wells,
                error=error,
            )
            raise
        finally:
            if (
                debug_execution_config is not None
                and debug_execution_config.replay_mode
                is DebugReplayMode.PERSISTENT_PAUSED_WORKER
            ):
                DebugPausedWorkerRegistry.remove(
                    debug_execution_config.debug_session_id
                )
            self._worker_assignments_by_execution.pop(
                request_context.execution_id,
                None,
            )

    @staticmethod
    def _emit_compile_started(
        progress_emitter: ZMQProgressEmitter,
        pipeline_steps: Sequence[AbstractStep],
        compile_artifact_id: str | None,
    ) -> None:
        if compile_artifact_id is None:
            progress_emitter.compile_started(len(pipeline_steps))

    def _initialize_orchestrator(
        self,
        execution_id: str,
        plate_path_str: str,
        pipeline_config,
    ):
        from pathlib import Path

        from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator(
            plate_path=Path(plate_path_str),
            pipeline_config=pipeline_config,
            progress_callback=None,
        )
        orchestrator.execution_id = execution_id
        orchestrator.initialize()
        self.active_executions[execution_id].set_extra("orchestrator", orchestrator)
        return orchestrator

    def _raise_if_cancelled(self, execution_id: str, phase_name: str) -> None:
        if (
            self.active_executions[execution_id].status
            == ExecutionStatus.CANCELLED.value
        ):
            logger.info(
                "[%s] Execution cancelled after %s, aborting",
                execution_id,
                phase_name,
            )
            raise RuntimeError("Execution cancelled by user")

    @staticmethod
    def _wells_for_execution(
        config_params,
        orchestrator,
        debug_execution_policy,
    ) -> list[str]:
        from openhcs.constants import MULTIPROCESSING_AXIS

        if config_params is not None:
            configured_wells = ZMQConfigParamsPayload.from_mapping(
                config_params
            ).concrete_axis_filter()
            if configured_wells is not None:
                return configured_wells
        available_axis_ids = tuple(orchestrator.get_component_keys(MULTIPROCESSING_AXIS))
        return debug_execution_policy.axis_filter_for_available(available_axis_ids)

    @staticmethod
    def _emit_planned_init_started(
        progress_emitter: ZMQProgressEmitter,
        pipeline_steps: Sequence[AbstractStep],
        wells: list[str],
        compile_artifact_id: str | None,
    ) -> None:
        if compile_artifact_id is None:
            progress_emitter.planned_init_started(
                wells=wells,
                step_names=[step.name for step in pipeline_steps],
            )

    def _resolve_compilation(
        self,
        *,
        request_context: ZMQExecutionContext,
        orchestrator,
        wells: list[str],
        debug_execution_config,
        progress_emitter: ZMQProgressEmitter,
    ):
        if request_context.compile_artifact_id is not None:
            self._cleanup_compiled_artifacts()
        return ZMQCompilationRequest(
            execution_id=request_context.execution_id,
            plate_id=request_context.plate_id,
            pipeline_steps=request_context.pipeline_steps,
            orchestrator=orchestrator,
            wells=wells,
            compile_artifact_id=request_context.compile_artifact_id,
            request_signature=request_context.request_signature,
            debug_replay_signature=request_context.debug_replay_signature,
            retain_compile_artifact=self._retain_compile_artifact(
                debug_execution_config
            ),
            compiled_artifacts=self._compiled_artifacts,
            progress_emitter=progress_emitter,
            flush_progress=self._flush_progress_only,
            immediate_progress_queue=ImmediateZMQProgressQueue(
                enqueue=self._enqueue_progress,
                flush=self._flush_progress_only,
                plate_id=request_context.plate_id,
            ),
        ).resolve()

    @staticmethod
    def _retain_compile_artifact(debug_execution_config) -> bool:
        if debug_execution_config is None:
            return False
        return debug_execution_config.replay_mode.retains_compile_artifact

    def _record_compilation_outputs(self, execution_id: str, compilation) -> None:
        self._worker_assignments_by_execution[execution_id] = (
            compilation.worker_assignments
        )
        if compilation.output_plate_root:
            self.active_executions[execution_id].set_extra(
                "output_plate_root",
                compilation.output_plate_root,
            )
        if compilation.auto_add_output_plate is not None:
            self.active_executions[execution_id].set_extra(
                "auto_add_output_plate",
                compilation.auto_add_output_plate,
            )

    def _finish_compilation_or_execute(
        self,
        *,
        request_context: ZMQExecutionContext,
        orchestrator,
        compilation,
        progress_context: dict,
        debug_execution_policy,
    ):
        if request_context.compile_only:
            return self._store_compile_artifact(
                request_context=request_context,
                compilation=compilation,
            )
        return ZMQWorkerExecutionRequest(
            execution_id=request_context.execution_id,
            global_config=request_context.global_config,
            orchestrator=orchestrator,
            pipeline_steps=request_context.pipeline_steps,
            compiled_pipeline_definition=compilation.compiled_pipeline_definition,
            compiled_contexts=compilation.compiled_contexts,
            execution_bundle=compilation.execution_bundle,
            progress_context=progress_context,
            worker_assignments=compilation.worker_assignments,
            debug_execution_policy=debug_execution_policy,
            active_execution_record=self.active_executions[
                request_context.execution_id
            ],
            forward_worker_progress=self._forward_worker_progress,
        ).execute()

    def _store_compile_artifact(
        self,
        *,
        request_context: ZMQExecutionContext,
        compilation,
    ):
        self._compiled_artifacts[request_context.execution_id] = (
            ZMQCompileArtifactRecord(
                execution_id=request_context.execution_id,
                plate_id=request_context.plate_id,
                request_signature=request_context.request_signature,
                debug_replay_signature=request_context.debug_replay_signature,
                compilation=compilation,
            ).as_dict()
        )
        logger.info(
            "[%s] Compilation-only request completed and artifact stored (artifact_id=%s sig=%s)",
            request_context.execution_id,
            request_context.execution_id,
            request_context.request_signature[:12],
        )
        self._set_compile_status("compiled success")
        return compilation.compiled_contexts

    def _emit_compile_failure(
        self,
        *,
        progress_emitter: ZMQProgressEmitter,
        compile_artifact_id: str | None,
        compilation_resolved: bool,
        wells: list[str] | None,
        error: Exception,
    ) -> None:
        if compile_artifact_id is not None:
            return
        if compilation_resolved:
            return
        progress_emitter.compile_failed(
            axis_ids=self._compile_failure_axis_ids(wells),
            error=str(error),
        )
        self._flush_progress_only()

    @staticmethod
    def _compile_failure_axis_ids(wells: list[str] | None) -> list[str]:
        if wells is None:
            return []
        return wells

    def _kill_worker_processes(self) -> int:
        """OpenHCS-specific worker cleanup (graceful cancellation + kill)."""
        ZMQWorkerCleanup(self.active_executions).cancel_orchestrators()
        return super()._kill_worker_processes()
