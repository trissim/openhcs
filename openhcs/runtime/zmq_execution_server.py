"""OpenHCS execution server built on zmqruntime ExecutionServer."""

from __future__ import annotations

import logging
import time
import json
from typing import Any

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
from openhcs.runtime.zmq_debug_control import DebugControlMessageStrategy
from openhcs.runtime.zmq_compilation import (
    ZMQCompilationRequest,
    ZMQCompileArtifactRecord,
)
from openhcs.runtime.zmq_execution_signature import ZMQExecutionRequestPayload
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


logger = logging.getLogger(__name__)


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
        from openhcs.core.debug import DebugControlMessageType

        if message.get(MessageFields.TYPE) in {
            DebugControlMessageType.READ_SNAPSHOT.value,
            DebugControlMessageType.WORKER_COMMAND.value,
            DebugControlMessageType.EXPORT_ARTIFACT.value,
        }:
            return DebugControlMessageStrategy.for_message(message).handle(message)
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
        import json
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
            logger.info(
                f"Flushing to ZMQ: step_name={progress_update.get('step_name')!r}, "
                f"axis={progress_update.get('axis_id')!r}, plate_id={progress_update.get('plate_id')!r}, "
                f"percent={progress_update.get('percent')!r}, total_wells={progress_update.get('total_wells')!r}"
            )
            json_str = json.dumps(progress_update)
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
        # DEBUG: Log what's being enqueued
        if "total_wells" in progress_update:
            logger.info(
                f"_enqueue_progress: total_wells={progress_update.get('total_wells')}, keys={list(progress_update.keys())}, step_name={progress_update.get('step_name')}"
            )
        self.progress_queue.put(progress_update)

    def _forward_worker_progress(self, worker_queue) -> None:
        import logging

        logger = logging.getLogger(__name__)
        while True:
            progress_update = worker_queue.get()
            if progress_update is None:
                logger.info("Progress forwarder received None, exiting")
                break
            execution_id = progress_update.get("execution_id")
            if not execution_id:
                raise ValueError(
                    f"Worker progress missing execution_id: {progress_update}"
                )
            assignments = self._worker_assignments_by_execution.get(execution_id)
            if assignments is None:
                raise ValueError(
                    f"Missing worker assignments for execution_id={execution_id}"
                )

            # Pipeline-level INIT events (e.g. viewer launch) bypass worker
            # claim validation — they carry no worker_slot / owned_wells.
            phase = progress_update.get("phase")
            axis_id = progress_update.get("axis_id", "")
            if phase == "init" and not axis_id:
                self.progress_queue.put(progress_update)
                continue

            worker_slot = progress_update.get("worker_slot")
            owned_wells = progress_update.get("owned_wells")
            if not worker_slot or owned_wells is None:
                raise ValueError(
                    f"Worker progress missing claim fields: worker_slot={worker_slot}, owned_wells={owned_wells}"
                )
            self._validate_worker_claim(worker_slot, owned_wells, assignments)
            # Attach topology metadata to every worker progress event so the UI
            # cannot lose worker/well ownership due first-message ordering.
            progress_update = dict(progress_update)
            progress_update["worker_assignments"] = assignments
            progress_update["total_wells"] = sorted(
                {
                    axis_id
                    for assigned_axes in assignments.values()
                    for axis_id in assigned_axes
                }
            )
            logger.info(
                f"Forwarding progress: pid={progress_update.get('pid')}, axis={progress_update.get('axis_id')}, step_name={progress_update.get('step_name')}, worker_slot={worker_slot}"
            )
            self.progress_queue.put(progress_update)

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

    def execute_task(self, execution_id: str, request: ExecuteRequest) -> Any:
        return self._execute_pipeline(
            execution_id,
            ZMQExecutionRequestPayload.from_execute_request(request),
        )

    def _execute_pipeline(
        self,
        execution_id: str,
        request_payload: ZMQExecutionRequestPayload,
    ):
        from openhcs.core.config import GlobalPipelineConfig, PipelineConfig

        plate_id = request_payload.plate_id
        pipeline_code = request_payload.pipeline_code
        config_params = request_payload.config_params
        config_code = request_payload.config_code
        pipeline_config_code = request_payload.pipeline_config_code
        compile_only = request_payload.compile_only
        compile_artifact_id = request_payload.compile_artifact_id
        request_signature = request_payload.request_signature
        debug_replay_signature = request_payload.debug_replay_signature
        pipeline_sha = request_payload.pipeline_sha

        logger.info("[%s] Starting plate %s", execution_id, plate_id)

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

        if compile_only and compile_artifact_id:
            raise ValueError("compile_only and compile_artifact_id cannot both be set")

        namespace = {}
        exec(pipeline_code, namespace)
        if not (pipeline_steps := namespace.get("pipeline_steps")):
            raise ValueError("Code must define 'pipeline_steps'")
        logger.info(
            "[%s] Request received: plate=%s compile_only=%s artifact_id=%s step_count=%d pipeline_sha=%s request_sig=%s",
            execution_id,
            plate_id,
            bool(compile_only),
            compile_artifact_id,
            len(pipeline_steps),
            pipeline_sha,
            request_signature[:12],
        )

        if config_code:
            is_empty = (
                "GlobalPipelineConfig(\n\n)" in config_code
                or "GlobalPipelineConfig()" in config_code
            )
            global_config = (
                GlobalPipelineConfig()
                if is_empty
                else (exec(config_code, ns := {}) or ns.get("config"))
            )
            if not global_config:
                raise ValueError("config_code must define 'config'")
            pipeline_config = (
                exec(pipeline_config_code, ns := {}) or ns.get("config")
                if pipeline_config_code
                else PipelineConfig()
            )
            if pipeline_config_code and not pipeline_config:
                raise ValueError("pipeline_config_code must define 'config'")
        elif config_params:
            global_config, pipeline_config = self._build_config_from_params(
                config_params
            )
        else:
            raise ValueError("Either config_params or config_code required")

        try:
            return self._execute_with_orchestrator(
                execution_id,
                plate_id,
                pipeline_steps,
                global_config,
                pipeline_config,
                config_params,
                compile_only=compile_only,
                compile_artifact_id=compile_artifact_id,
                request_signature=request_signature,
                debug_replay_signature=debug_replay_signature,
            )
        except Exception as e:
            if compile_only:
                self._set_compile_status("compiled failed", str(e))
            raise

    def _build_config_from_params(self, p):
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
                num_workers=p.get("num_workers", 4),
                path_planning_config=PathPlanningConfig(
                    output_dir_suffix=p.get("output_dir_suffix", "_output")
                ),
                vfs_config=VFSConfig(
                    materialization_backend=MaterializationBackend(
                        p.get("materialization_backend", "disk")
                    )
                ),
                step_well_filter_config=StepWellFilterConfig(
                    well_filter=p.get("well_filter")
                ),
                use_threading=p.get("use_threading", False),
            ),
            PipelineConfig(),
        )

    def _execute_with_orchestrator(
        self,
        execution_id,
        plate_id,
        pipeline_steps,
        global_config,
        pipeline_config,
        config_params,
        compile_only: bool = False,
        compile_artifact_id: str | None = None,
        request_signature: str | None = None,
        debug_replay_signature: str | None = None,
    ):
        from pathlib import Path
        from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
        from openhcs.core.debug import (
            DebugPausedWorkerRegistry,
            DebugReplayMode,
        )
        from openhcs.constants import MULTIPROCESSING_AXIS

        environment = ZMQOrchestratorEnvironmentRequest(
            execution_id=execution_id,
            plate_id=plate_id,
            global_config=global_config,
            config_params=config_params,
        ).prepare()
        global_config = environment.global_config
        debug_execution_policy = environment.debug_execution_policy
        debug_execution_config = environment.debug_execution_config
        plate_path_str = environment.plate_path_str

        progress_context = {
            MessageFields.EXECUTION_ID: execution_id,
            MessageFields.PLATE_ID: plate_id,
            MessageFields.AXIS_ID: "",
        }
        compiled_contexts: dict[str, Any] | None = None
        progress_emitter = ZMQProgressEmitter(
            self._enqueue_progress,
            execution_id,
            plate_id,
        )

        try:
            if compile_artifact_id is None:
                progress_emitter.compile_started(len(pipeline_steps))
            orchestrator = PipelineOrchestrator(
                plate_path=Path(plate_path_str),
                pipeline_config=pipeline_config,
                progress_callback=None,
            )
            orchestrator.execution_id = execution_id
            orchestrator.initialize()
            self.active_executions[execution_id].set_extra("orchestrator", orchestrator)

            if (
                self.active_executions[execution_id].status
                == ExecutionStatus.CANCELLED.value
            ):
                logger.info(
                    "[%s] Execution cancelled after initialization, aborting",
                    execution_id,
                )
                raise RuntimeError("Execution cancelled by user")

            if config_params and config_params.get("well_filter"):
                wells = list(config_params["well_filter"])
            else:
                available_axis_ids = tuple(
                    orchestrator.get_component_keys(MULTIPROCESSING_AXIS)
                )
                wells = debug_execution_policy.axis_filter_for_available(
                    available_axis_ids
                )

            if compile_artifact_id is None:
                step_names = [step.name for step in pipeline_steps]
                progress_emitter.planned_init_started(
                    wells=wells,
                    step_names=step_names,
                )

            if compile_artifact_id is not None:
                self._cleanup_compiled_artifacts()
            compilation = ZMQCompilationRequest(
                execution_id=execution_id,
                plate_id=plate_id,
                pipeline_steps=pipeline_steps,
                orchestrator=orchestrator,
                wells=wells,
                compile_artifact_id=compile_artifact_id,
                request_signature=request_signature,
                debug_replay_signature=debug_replay_signature,
                retain_compile_artifact=(
                    debug_execution_config is not None
                    and debug_execution_config.replay_mode.retains_compile_artifact
                ),
                compiled_artifacts=self._compiled_artifacts,
                progress_emitter=progress_emitter,
                flush_progress=self._flush_progress_only,
                immediate_progress_queue=ImmediateZMQProgressQueue(
                    enqueue=self._enqueue_progress,
                    flush=self._flush_progress_only,
                ),
            ).resolve()
            compiled_contexts = compilation.compiled_contexts
            execution_bundle = compilation.execution_bundle
            worker_assignments = compilation.worker_assignments
            compiled_pipeline_definition = compilation.compiled_pipeline_definition
            self._worker_assignments_by_execution[execution_id] = worker_assignments
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

            if (
                self.active_executions[execution_id].status
                == ExecutionStatus.CANCELLED.value
            ):
                logger.info(
                    "[%s] Execution cancelled after compilation, aborting",
                    execution_id,
                )
                raise RuntimeError("Execution cancelled by user")

            if compile_only:
                if request_signature is None or debug_replay_signature is None:
                    raise ValueError(
                        "Missing request signature for compile artifact storage"
                    )
                self._compiled_artifacts[execution_id] = ZMQCompileArtifactRecord(
                    execution_id=execution_id,
                    plate_id=plate_id,
                    request_signature=request_signature,
                    debug_replay_signature=debug_replay_signature,
                    compilation=compilation,
                ).as_dict()
                logger.info(
                    "[%s] Compilation-only request completed and artifact stored (artifact_id=%s sig=%s)",
                    execution_id,
                    execution_id,
                    request_signature[:12],
                )
                self._set_compile_status("compiled success")
                return compiled_contexts

            return ZMQWorkerExecutionRequest(
                execution_id=execution_id,
                global_config=global_config,
                orchestrator=orchestrator,
                pipeline_steps=pipeline_steps,
                compiled_pipeline_definition=compiled_pipeline_definition,
                compiled_contexts=compiled_contexts,
                execution_bundle=execution_bundle,
                progress_context=progress_context,
                worker_assignments=worker_assignments,
                debug_execution_policy=debug_execution_policy,
                active_execution_record=self.active_executions[execution_id],
                forward_worker_progress=self._forward_worker_progress,
            ).execute()
        finally:
            if (
                debug_execution_config is not None
                and debug_execution_config.replay_mode
                is DebugReplayMode.PERSISTENT_PAUSED_WORKER
            ):
                DebugPausedWorkerRegistry.remove(
                    debug_execution_config.debug_session_id
                )
            self._worker_assignments_by_execution.pop(execution_id, None)

    def _kill_worker_processes(self) -> int:
        """OpenHCS-specific worker cleanup (graceful cancellation + kill)."""
        ZMQWorkerCleanup(self.active_executions).cancel_orchestrators()
        return super()._kill_worker_processes()
