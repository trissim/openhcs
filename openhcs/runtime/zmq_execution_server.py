"""OpenHCS execution server built on zmqruntime ExecutionServer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import sys
import time
from types import ModuleType
from typing import Any, TYPE_CHECKING

from zmqruntime.execution import ExecutionServer
from zmqruntime.messages import (
    ExecuteRequest,
    ExecutionStatus,
    MessageFields,
    StatusRequest,
)
from zmqruntime.startup import EndpointStartupStatusCallback

from zmqruntime.config import TransportMode
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG, OpenHCSZMQConfig
from openhcs.runtime.zmq_control import (
    ZMQControlMessageRouter,
    ZMQControlRequestContext,
)
from openhcs.runtime.zmq_compilation import (
    ZMQCompilationRequest,
    ZMQCompileArtifactRecord,
)
from openhcs.runtime.zmq_execution_signature import (
    OpenHCSExecutionConfigBundle,
    ZMQExecutionRequestPayload,
)
from openhcs.runtime.zmq_orchestrator_environment import (
    ZMQOrchestratorEnvironmentRequest,
)
from openhcs.runtime.zmq_progress import ZMQCompilerProgressQueue, ZMQProgressEmitter
from openhcs.runtime.zmq_server_hooks import (
    ZMQPongResponseEnricher,
    ZMQResultsSummaryEnricher,
    ZMQWorkerCleanup,
)
from openhcs.runtime.zmq_worker_execution import ZMQWorkerExecutionRequest
from openhcs.core.config import GlobalPipelineConfig, PipelineConfig
from openhcs.core.config_document import ConfigDocumentAuthority
from openhcs.core.pipeline_document import PipelineDocumentAuthority
from openhcs.core.progress import ProgressEvent
from openhcs.core.steps.function_step import FunctionStep


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from openhcs.core.debug import DebugExecutionConfig


@dataclass(frozen=True, slots=True)
class ZMQAuxiliaryExecutionParams:
    """Typed OpenHCS auxiliary fields carried by zmqruntime config_params."""

    axis_filter: tuple[str, ...] | None = None
    debug_execution_config: "DebugExecutionConfig | None" = None
    runtime_observation_export_path: Path | None = None

    @classmethod
    def from_transport(
        cls,
        config_params: Mapping[str, Any] | None,
    ) -> "ZMQAuxiliaryExecutionParams":
        if not config_params:
            return cls()
        return cls(
            axis_filter=cls._axis_filter_from_transport(
                config_params.get(ZMQAuxiliaryParamField.WELL_FILTER.value)
            ),
            debug_execution_config=cls._debug_config_from_transport(config_params),
            runtime_observation_export_path=cls._path_from_transport(
                config_params.get(
                    ZMQAuxiliaryParamField.RUNTIME_OBSERVATION_EXPORT_PATH.value
                )
            ),
        )

    @staticmethod
    def _axis_filter_from_transport(
        axis_filter: list[str] | tuple[str, ...] | str | int | None,
    ) -> tuple[str, ...] | None:
        if axis_filter is None:
            return None
        if isinstance(axis_filter, list):
            return tuple(str(axis_id) for axis_id in axis_filter)
        if isinstance(axis_filter, tuple):
            return tuple(str(axis_id) for axis_id in axis_filter)
        raise TypeError(
            "ZMQ config_params well_filter must be a concrete axis-id sequence, "
            f"got {type(axis_filter).__name__}."
        )

    @staticmethod
    def _debug_config_from_transport(
        config_params: Mapping[str, Any],
    ) -> "DebugExecutionConfig | None":
        from openhcs.core.debug import DebugExecutionConfig

        payload = config_params.get(DebugExecutionConfig.CONFIG_PARAMS_KEY)
        if payload is None:
            return None
        return DebugExecutionConfig.from_payload(payload)

    @staticmethod
    def _path_from_transport(value: str | Path | None) -> Path | None:
        if value is None:
            return None
        if isinstance(value, Path):
            return value
        if isinstance(value, str):
            return Path(value)
        raise TypeError(
            "ZMQ config_params runtime observation export path must be a path "
            f"string, got {type(value).__name__}."
        )


class ZMQAuxiliaryParamField(Enum):
    """Transport keys consumed as typed auxiliary execution inputs."""

    WELL_FILTER = "well_filter"
    RUNTIME_OBSERVATION_EXPORT_PATH = "runtime_observation_export_path"


@dataclass(frozen=True, slots=True)
class ZMQExecutionContext:
    """Request context shared by compile and execution completion phases."""

    execution_id: str
    request_payload: ZMQExecutionRequestPayload
    pipeline_steps: list[FunctionStep]
    configs: OpenHCSExecutionConfigBundle

    @property
    def pipeline_config(self) -> PipelineConfig:
        return self.configs.plate_pipeline

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
    def auxiliary_params(self) -> ZMQAuxiliaryExecutionParams:
        return ZMQAuxiliaryExecutionParams.from_transport(
            self.request_payload.config_params
        )

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
        host: str | None = None,
        log_file_path: str | None = None,
        transport_mode: TransportMode | None = None,
        config: OpenHCSZMQConfig = OPENHCS_ZMQ_CONFIG,
    ):
        super().__init__(
            port=config.default_port if port is None else port,
            host=config.server_host if host is None else host,
            log_file_path=log_file_path,
            transport_mode=(
                config.transport_mode if transport_mode is None else transport_mode
            ),
            config=config,
        )
        self._compile_status: str | None = None
        self._compile_message: str | None = None
        self._compile_status_expires_at: float | None = None
        self._worker_assignments_by_execution: dict[str, dict[str, list[str]]] = {}
        self._compiled_artifacts: dict[str, ZMQCompileArtifactRecord] = {}
        self._compiled_artifact_ttl_seconds = config.compiled_artifact_ttl_seconds
        from openhcs.agent.services.function_catalog_service import (
            FunctionCatalogService,
        )
        from openhcs.runtime.function_catalog_preparation import (
            FunctionCatalogPreparation,
        )

        self._function_catalog = FunctionCatalogService()
        self._function_catalog_preparation = FunctionCatalogPreparation(
            self._function_catalog
        )

    def prepare_capabilities(self) -> None:
        """Materialize endpoint-owned capabilities and their persistent caches."""

        from openhcs.processing.backends.lib_registry.registry_service import (
            RegistryService,
        )

        RegistryService.prepare_in_current_process()
        self._function_catalog.catalog(compact_signatures=True)

    def prepare_runtime_capabilities(
        self,
        status_callback: EndpointStartupStatusCallback | None = None,
    ) -> None:
        """Materialize cached capabilities before exposing the live endpoint."""

        self._function_catalog_preparation.wait_until_ready(status_callback)

    def handle_control_message(self, message):
        if ZMQControlMessageRouter.handles(message):
            self._cleanup_compiled_artifacts()
            return ZMQControlMessageRouter.handle(
                message,
                ZMQControlRequestContext(
                    compiled_artifacts=self._compiled_artifacts,
                    function_catalog=self._function_catalog,
                    function_catalog_preparation=self._function_catalog_preparation,
                ),
            )
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
            if now - artifact.created_at > self._compiled_artifact_ttl_seconds
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

            # Axisless events are owned by the parent execution lifecycle, not
            # by a worker lane. They carry topology but never a worker claim.
            if not event.axis_id:
                if event.worker_slot is not None or event.owned_wells is not None:
                    raise ValueError(
                        "Execution-level progress cannot carry a worker claim: "
                        f"{progress_update}"
                    )
                self.progress_queue.put(
                    event.with_worker_topology(
                        worker_assignments=assignments,
                        total_wells=sorted(
                            {
                                axis_id
                                for assigned_axes in assignments.values()
                                for axis_id in assigned_axes
                            }
                        ),
                    ).to_dict()
                )
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

        self._function_catalog_preparation.wait_until_ready()

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
        pipeline_document = PipelineDocumentAuthority.from_namespace(module.__dict__)
        resolved_config = self._resolve_request_config(
            request_payload,
            pipeline_document.pipeline_config,
        )
        request_context = ZMQExecutionContext(
            execution_id=execution_id,
            request_payload=request_payload,
            pipeline_steps=pipeline_document.pipeline_steps,
            configs=resolved_config,
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
        pipeline_config: PipelineConfig,
    ) -> OpenHCSExecutionConfigBundle:
        if request_payload.config_code is not None:
            global_config = ConfigDocumentAuthority.from_source(
                request_payload.config_code,
                expected_config_type=GlobalPipelineConfig,
            )
            return OpenHCSExecutionConfigBundle(
                global_pipeline=global_config,
                plate_pipeline=pipeline_config,
            )
        raise ValueError("config_code is required for execution config resolution")

    def _execute_with_orchestrator(
        self,
        request_context: ZMQExecutionContext,
    ):
        from openhcs.core.debug import (
            DebugPausedWorkerRegistry,
            DebugReplayMode,
        )

        auxiliary_params = request_context.auxiliary_params
        environment = ZMQOrchestratorEnvironmentRequest(
            execution_id=request_context.execution_id,
            plate_id=request_context.plate_id,
            execution_plate_id=request_context.execution_plate_id,
            selected_pipeline_path=request_context.request_payload.selected_pipeline_path,
            debug_execution_config=auxiliary_params.debug_execution_config,
        ).prepare()
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
            self._ensure_request_global_config_context(request_context)
            orchestrator = self._initialize_orchestrator(
                request_context.execution_id,
                plate_path_str,
                request_context.pipeline_config,
                selected_pipeline_path=request_context.request_payload.selected_pipeline_path,
            )
            self._raise_if_cancelled(request_context.execution_id, "initialization")
            wells = self._wells_for_execution(
                auxiliary_params.axis_filter,
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
                debug_execution_policy=debug_execution_policy,
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
        pipeline_steps: Sequence[FunctionStep],
        compile_artifact_id: str | None,
    ) -> None:
        if compile_artifact_id is None:
            progress_emitter.compile_started(len(pipeline_steps))

    def _initialize_orchestrator(
        self,
        execution_id: str,
        plate_path_str: str,
        pipeline_config,
        selected_pipeline_path: str | None = None,
    ):
        from pathlib import Path

        from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator(
            plate_path=Path(plate_path_str),
            pipeline_config=pipeline_config,
            selected_pipeline_path=selected_pipeline_path,
            progress_callback=None,
            transport_config=self.config,
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
        axis_filter: tuple[str, ...] | None,
        orchestrator,
        debug_execution_policy,
    ) -> list[str]:
        from openhcs.constants import MULTIPROCESSING_AXIS

        if axis_filter is not None:
            return list(axis_filter)
        available_axis_ids = tuple(
            orchestrator.get_component_keys(MULTIPROCESSING_AXIS)
        )
        return debug_execution_policy.axis_filter_for_available(available_axis_ids)

    @staticmethod
    def _emit_planned_init_started(
        progress_emitter: ZMQProgressEmitter,
        pipeline_steps: Sequence[FunctionStep],
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
        debug_execution_policy,
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
            compiler_progress_queue=ZMQCompilerProgressQueue(
                enqueue=self._enqueue_progress,
                plate_id=request_context.plate_id,
            ),
            debug_execution_policy=debug_execution_policy,
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
        execution_results = ZMQWorkerExecutionRequest(
            execution_id=request_context.execution_id,
            orchestrator=orchestrator,
            execution_bundle=compilation.execution_bundle,
            progress_context=progress_context,
            debug_execution_policy=debug_execution_policy,
            active_execution_record=self.active_executions[
                request_context.execution_id
            ],
            forward_worker_progress=self._forward_worker_progress,
        ).execute()
        self._export_runtime_observation(
            request_context=request_context,
            compilation=compilation,
            execution_results=execution_results,
        )
        return execution_results

    def _export_runtime_observation(
        self,
        *,
        request_context: ZMQExecutionContext,
        compilation,
        execution_results,
    ) -> None:
        export_path = request_context.auxiliary_params.runtime_observation_export_path
        if export_path is None:
            return

        from openhcs.core.runtime_execution_validation import runtime_output_roots
        from openhcs.runtime.zmq_execution_observation import (
            ZMQRuntimeExecutionObservationExport,
        )

        execution_bundle = compilation.execution_bundle
        output_roots = runtime_output_roots(
            execution_bundle.runtime_contexts,
            compilation.output_plate_root,
        )
        ZMQRuntimeExecutionObservationExport.from_execution(
            compiled_contexts=execution_bundle.runtime_contexts,
            execution_results=execution_results,
            output_roots=output_roots,
        ).write(export_path)
        self.active_executions[request_context.execution_id].set_extra(
            "runtime_observation_export_path",
            str(export_path),
        )
        logger.info(
            "[%s] Exported runtime observation to %s",
            request_context.execution_id,
            export_path,
        )

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
            )
        )
        logger.info(
            "[%s] Compilation-only request completed and artifact stored (artifact_id=%s sig=%s)",
            request_context.execution_id,
            request_context.execution_id,
            request_context.request_signature[:12],
        )
        self._set_compile_status("compiled success")
        return compilation.execution_bundle.runtime_contexts

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

    @staticmethod
    def _ensure_request_global_config_context(
        request_context: ZMQExecutionContext,
    ) -> None:
        from objectstate.lazy_factory import (
            ensure_global_config_context,
        )

        ensure_global_config_context(
            GlobalPipelineConfig,
            request_context.configs.global_pipeline,
        )

    @staticmethod
    def _compile_failure_axis_ids(wells: list[str] | None) -> list[str]:
        if wells is None:
            return []
        return wells

    def _kill_worker_processes(self) -> int:
        """OpenHCS-specific worker cleanup (graceful cancellation + kill)."""
        ZMQWorkerCleanup(self.active_executions).cancel_orchestrators()
        return super()._kill_worker_processes()
