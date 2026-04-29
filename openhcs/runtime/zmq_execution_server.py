"""OpenHCS execution server built on zmqruntime ExecutionServer."""

from __future__ import annotations

import logging
import time
import threading
import hashlib
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
from openhcs.core.progress import ProgressPhase, ProgressStatus, ProgressEvent


logger = logging.getLogger(__name__)


def _emit_zmq_progress(enqueue_fn, **kwargs) -> None:
    """Helper to emit progress to ZMQ queue using new schema.

    All fields are explicit on ProgressEvent - just pass kwargs directly.
    """
    import time
    import os

    # Create event - all fields are explicit, just pass kwargs directly
    event = ProgressEvent(timestamp=time.time(), pid=os.getpid(), **kwargs)
    enqueue_fn(event.to_dict())


class _ImmediateProgressQueue:
    """Queue adapter that forwards progress updates immediately to ZMQ."""

    def __init__(self, server: "ZMQExecutionServer"):
        self._server = server

    def put(self, progress_update: dict) -> None:
        self._server._enqueue_progress(progress_update)
        self._server._flush_progress_only()


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

    @staticmethod
    def _extract_compiled_axis_ids(compiled_contexts: dict[str, Any]) -> list[str]:
        """Extract unique axis ids from compiled context keys.

        Context keys may be either plain axis ids (e.g. "A01") or sequential keys
        (e.g. "A01__combo_0"). Worker ownership must always be at axis granularity.
        """
        axis_ids: list[str] = []
        seen: set[str] = set()
        for context_key in compiled_contexts.keys():
            axis_id = (
                context_key.split("__combo_", 1)[0]
                if "__combo_" in context_key
                else context_key
            )
            if axis_id not in seen:
                seen.add(axis_id)
                axis_ids.append(axis_id)
        return sorted(axis_ids)

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

    @staticmethod
    def _build_request_signature(
        plate_id: str,
        pipeline_code: str,
        config_params: dict | None,
        config_code: str | None,
        pipeline_config_code: str | None,
    ) -> str:
        payload = {
            MessageFields.PLATE_ID: plate_id,
            MessageFields.PIPELINE_CODE: pipeline_code,
            MessageFields.CONFIG_PARAMS: config_params,
            MessageFields.CONFIG_CODE: config_code,
            MessageFields.PIPELINE_CONFIG_CODE: pipeline_config_code,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

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
        response = super()._create_pong_response()

        # Add OpenHCS-specific compile status
        compile_status, compile_message = self._get_compile_status()
        if compile_status is not None:
            response[MessageFields.COMPILE_STATUS] = compile_status
        if compile_message is not None:
            response[MessageFields.COMPILE_MESSAGE] = compile_message

        queued = [
            (execution_id, record)
            for execution_id, record in self.active_executions.items()
            if record.status == ExecutionStatus.QUEUED.value
        ]
        response["queued_executions"] = [
            {
                MessageFields.EXECUTION_ID: execution_id,
                MessageFields.PLATE_ID: str(record.plate_id),
                "queue_position": index + 1,
            }
            for index, (execution_id, record) in enumerate(queued)
        ]

        return response

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
        if record.status != ExecutionStatus.COMPLETE.value:
            return

        summary = record.results_summary
        if not isinstance(summary, dict):
            summary = {}
            record.results_summary = summary

        output_plate_root = record.get_extra("output_plate_root")
        auto_add_output_plate = record.get_extra("auto_add_output_plate")
        if output_plate_root:
            summary["output_plate_root"] = str(output_plate_root)
        if auto_add_output_plate is not None:
            summary["auto_add_output_plate_to_plate_manager"] = bool(
                auto_add_output_plate
            )

        if isinstance(execution_payload, dict):
            execution_payload[MessageFields.RESULTS_SUMMARY] = summary

        logger.info(
            "[%s] Attached results_summary extras: output_plate_root=%s auto_add=%s",
            execution_id,
            summary.get("output_plate_root"),
            summary.get("auto_add_output_plate_to_plate_manager"),
        )

    def _run_execution(self, execution_id, request, record):
        """Run an execution and enrich results_summary with output plate path.

        The base zmqruntime ExecutionServer only populates well_count/wells in
        results_summary. OpenHCS needs the final output plate root (computed by
        path planning during compilation) so the UI can optionally auto-add it
        as a new orchestrator in Plate Manager.
        """
        super()._run_execution(execution_id, request, record)

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

    def _handle_status(self, msg):
        response = super()._handle_status(msg)
        if response.get(MessageFields.STATUS) != ResponseType.OK.value:
            return response

        execution_id = StatusRequest.from_dict(msg).execution_id
        if not execution_id:
            return response

        record = self.active_executions[execution_id]
        if record.status != ExecutionStatus.COMPLETE.value:
            return response

        execution_payload = response.get(MessageFields.EXECUTION)
        self._attach_results_summary_extras(
            execution_id=execution_id,
            record=record,
            execution_payload=execution_payload
            if isinstance(execution_payload, dict)
            else None,
        )
        return response

    def execute_task(self, execution_id: str, request: ExecuteRequest) -> Any:
        return self._execute_pipeline(
            execution_id,
            request.plate_id,
            request.pipeline_code,
            request.config_params,
            request.config_code,
            request.pipeline_config_code,
            request.client_address,
            request.compile_only,
            request.compile_artifact_id,
        )

    def _execute_pipeline(
        self,
        execution_id,
        plate_id,
        pipeline_code,
        config_params,
        config_code,
        pipeline_config_code,
        client_address=None,
        compile_only: bool = False,
        compile_artifact_id: str | None = None,
    ):
        from openhcs.constants import AllComponents, VariableComponents, GroupBy
        from openhcs.core.config import GlobalPipelineConfig, PipelineConfig

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

        request_signature = self._build_request_signature(
            plate_id=plate_id,
            pipeline_code=pipeline_code,
            config_params=config_params,
            config_code=config_code,
            pipeline_config_code=pipeline_config_code,
        )
        pipeline_sha = hashlib.sha256(pipeline_code.encode("utf-8")).hexdigest()[:12]

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
    ):
        from pathlib import Path
        import multiprocessing
        from openhcs.config_framework.lazy_factory import ensure_global_config_context
        from openhcs.core.config import GlobalPipelineConfig
        from openhcs.core.orchestrator.gpu_scheduler import setup_global_gpu_registry
        from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
        from openhcs.constants import (
            AllComponents,
            VariableComponents,
            GroupBy,
            MULTIPROCESSING_AXIS,
        )
        from polystore.base import reset_memory_backend, storage_registry

        try:
            if multiprocessing.get_start_method(allow_none=True) != "spawn":
                multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        reset_memory_backend()

        try:
            from openhcs.core.memory import cleanup_all_gpu_frameworks

            cleanup_all_gpu_frameworks()
        except Exception as cleanup_error:
            logger.warning(
                "[%s] Failed to trigger GPU cleanup: %s", execution_id, cleanup_error
            )

        if not isinstance(global_config, GlobalPipelineConfig):
            raise TypeError(
                f"Expected GlobalPipelineConfig, got {type(global_config).__name__}"
            )

        setup_global_gpu_registry(global_config=global_config)
        ensure_global_config_context(GlobalPipelineConfig, global_config)

        plate_path_str = str(plate_id)
        is_omero_plate_id = False
        try:
            int(plate_path_str)
            is_omero_plate_id = True
        except ValueError:
            is_omero_plate_id = plate_path_str.startswith("/omero/")

        if is_omero_plate_id:
            # Lazy-load OMERO dependencies only when OMERO is actually used
            # Import OMERO parsers BEFORE creating backend to ensure registration
            # This is required because OMEROLocalBackend accesses FilenameParser.__registry__
            # which is a LazyDiscoveryDict that only populates when first accessed
            from openhcs.runtime.omero_instance_manager import OMEROInstanceManager
            from openhcs.microscopes import omero  # noqa: F401 - Import OMERO parsers to register them
            from polystore.omero_local import OMEROLocalBackend

            omero_manager = OMEROInstanceManager()
            if not omero_manager.connect(timeout=60):
                raise RuntimeError("OMERO server not available")
            storage_registry["omero_local"] = OMEROLocalBackend(
                omero_conn=omero_manager.conn,
                namespace_prefix="openhcs",
                lock_dir_name=".openhcs",
            )

            if not plate_path_str.startswith("/omero/"):
                plate_path_str = f"/omero/plate_{plate_path_str}"

        progress_context = {
            MessageFields.EXECUTION_ID: execution_id,
            MessageFields.PLATE_ID: plate_id,
            MessageFields.AXIS_ID: "",
        }
        worker_progress_queue = None
        progress_forwarder = None
        compiled_contexts: dict[str, Any] | None = None

        try:
            if compile_artifact_id is None:
                _emit_zmq_progress(
                    self._enqueue_progress,
                    execution_id=execution_id,
                    plate_id=plate_id,
                    axis_id="",
                    step_name="pipeline",
                    total=len(pipeline_steps),
                    phase=ProgressPhase.COMPILE,
                    status=ProgressStatus.STARTED,
                    completed=0,
                    percent=0.0,
                )
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
                wells = orchestrator.get_component_keys(MULTIPROCESSING_AXIS)

            # Initialize compiled_pipeline_definition - will be set by either artifact reuse or fresh compilation
            compiled_pipeline_definition = None

            if compile_artifact_id is None:
                planned_metadata = {"total_wells": sorted(wells)}
                step_names = [step.name for step in pipeline_steps]
                planned_metadata["step_names"] = step_names
                _emit_zmq_progress(
                    self._enqueue_progress,
                    execution_id=execution_id,
                    plate_id=plate_id,
                    axis_id="",
                    step_name="",
                    phase=ProgressPhase.INIT,
                    status=ProgressStatus.STARTED,
                    percent=0.0,
                    completed=0,
                    total=1,
                    **planned_metadata,
                )

            if compile_artifact_id is not None:
                self._cleanup_compiled_artifacts()
                artifact = self._compiled_artifacts.pop(compile_artifact_id, None)
                if artifact is None:
                    raise ValueError(
                        f"Missing compile artifact '{compile_artifact_id}'. "
                        "Re-run compilation before execution."
                    )
                if request_signature is None:
                    raise ValueError(
                        "Missing request signature for artifact validation"
                    )
                if artifact["request_signature"] != request_signature:
                    logger.error(
                        "[%s] Compile artifact signature mismatch: artifact_id=%s artifact_sig=%s request_sig=%s",
                        execution_id,
                        compile_artifact_id,
                        str(artifact["request_signature"])[:12],
                        request_signature[:12],
                    )
                    raise ValueError(
                        f"Compile artifact '{compile_artifact_id}' does not match execution request"
                    )
                if artifact[MessageFields.PLATE_ID] != str(plate_id):
                    raise ValueError(
                        f"Compile artifact '{compile_artifact_id}' is for plate "
                        f"{artifact[MessageFields.PLATE_ID]}, not {plate_id}"
                    )

                compiled_contexts = artifact["compiled_contexts"]
                if compiled_contexts is None:
                    raise ValueError("Compile artifact missing compiled_contexts")
                compiled_pipeline_definition = artifact.get(
                    "compiled_pipeline_definition"
                )  # Get the stripped pipeline_definition from artifact
                worker_assignments = artifact["worker_assignments"]
                self._worker_assignments_by_execution[execution_id] = worker_assignments
                compiled_axis_ids = self._extract_compiled_axis_ids(compiled_contexts)

                # Emit filtered metadata for this execution id.
                step_names = [step.name for step in pipeline_steps]
                _emit_zmq_progress(
                    self._enqueue_progress,
                    execution_id=execution_id,
                    plate_id=plate_id,
                    axis_id="",
                    step_name="",
                    phase=ProgressPhase.INIT,
                    status=ProgressStatus.STARTED,
                    percent=0.0,
                    completed=0,
                    total=1,
                    total_wells=compiled_axis_ids,
                    worker_assignments=worker_assignments,
                    step_names=step_names,
                )

                logger.info(
                    "[%s] Reused compile artifact %s for plate %s (sig=%s)",
                    execution_id,
                    compile_artifact_id,
                    plate_id,
                    request_signature[:12] if request_signature else "missing",
                )

                output_plate_root = artifact.get("output_plate_root")
                if output_plate_root:
                    self.active_executions[execution_id].set_extra(
                        "output_plate_root",
                        str(output_plate_root),
                    )
                if artifact.get("auto_add_output_plate") is not None:
                    self.active_executions[execution_id].set_extra(
                        "auto_add_output_plate", bool(artifact["auto_add_output_plate"])
                    )
            else:
                # Compilation runs in THIS process (queue worker thread), not a
                # separate worker process. Use an immediate adapter so compile
                # events are forwarded to ZMQ as soon as they are emitted.
                from openhcs.core.progress import set_progress_queue

                set_progress_queue(_ImmediateProgressQueue(self))
                try:
                    compilation = orchestrator.compile_pipelines(
                        pipeline_definition=pipeline_steps,
                        well_filter=wells,
                        is_zmq_execution=True,
                    )
                finally:
                    set_progress_queue(None)

                if (
                    not isinstance(compilation, dict)
                    or "compiled_contexts" not in compilation
                ):
                    raise ValueError("Compilation did not return compiled_contexts")
                compiled_contexts = compilation["compiled_contexts"]
                # CRITICAL: Use the returned pipeline_definition, not the original pipeline_steps
                # The compiler modifies pipeline_definition in-place (converts functions to FunctionReference)
                # and returns the modified version
                compiled_pipeline_definition = compilation.get(
                    "pipeline_definition", pipeline_steps
                )

                # DEBUG: Check if they're the same object
                logger.info(
                    f"🔍 ZMQ: pipeline_steps is compiled_pipeline_definition? {pipeline_steps is compiled_pipeline_definition}"
                )
                logger.info(
                    f"🔍 ZMQ: pipeline_steps[0].func = {type(getattr(pipeline_steps[0], 'func', None)).__name__ if getattr(pipeline_steps[0], 'func', None) else 'None'}"
                )
                logger.info(
                    f"🔍 ZMQ: compiled_pipeline_definition[0].func = {type(getattr(compiled_pipeline_definition[0], 'func', None)).__name__ if getattr(compiled_pipeline_definition[0], 'func', None) else 'None'}"
                )
                if not compiled_contexts:
                    raise ValueError("Compilation produced no compiled contexts")

                # Get worker_assignments from compiler result (uses PipelineConfig's num_workers)
                worker_assignments = compilation["worker_assignments"]
                self._worker_assignments_by_execution[execution_id] = worker_assignments
                compiled_axis_ids = self._extract_compiled_axis_ids(compiled_contexts)

                # Emit filtered metadata for this execution id.
                _emit_zmq_progress(
                    self._enqueue_progress,
                    execution_id=execution_id,
                    plate_id=plate_id,
                    axis_id="",
                    step_name="",
                    phase=ProgressPhase.INIT,
                    status=ProgressStatus.STARTED,
                    percent=0.0,
                    completed=0,
                    total=1,
                    total_wells=compiled_axis_ids,
                    worker_assignments=worker_assignments,
                )

                _emit_zmq_progress(
                    self._enqueue_progress,
                    execution_id=execution_id,
                    plate_id=plate_id,
                    axis_id="",
                    step_name="pipeline",
                    total=len(pipeline_steps),
                    phase=ProgressPhase.COMPILE,
                    status=ProgressStatus.SUCCESS,
                    completed=1,
                    percent=100.0,
                    total_wells=compiled_axis_ids,
                    worker_assignments=worker_assignments,
                )
                self._flush_progress_only()

                for axis_id in compiled_axis_ids:
                    _emit_zmq_progress(
                        self._enqueue_progress,
                        execution_id=execution_id,
                        plate_id=plate_id,
                        axis_id=axis_id,
                        step_name="compilation",
                        phase=ProgressPhase.COMPILE,
                        status=ProgressStatus.SUCCESS,
                        completed=1,
                        total=1,
                        percent=100.0,
                    )
                self._flush_progress_only()

                first_context = next(iter(compiled_contexts.values()))
                output_plate_root = first_context.output_plate_root
                if output_plate_root:
                    self.active_executions[execution_id].set_extra(
                        "output_plate_root",
                        str(output_plate_root),
                    )
                self.active_executions[execution_id].set_extra(
                    "auto_add_output_plate",
                    bool(first_context.auto_add_output_plate_to_plate_manager),
                )
                logger.info(
                    "[%s] Captured auto_add_output_plate=%s output_plate_root=%s",
                    execution_id,
                    bool(first_context.auto_add_output_plate_to_plate_manager),
                    output_plate_root,
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
                if request_signature is None:
                    raise ValueError(
                        "Missing request signature for compile artifact storage"
                    )
                self._compiled_artifacts[execution_id] = {
                    "created_at": time.time(),
                    "request_signature": request_signature,
                    MessageFields.PLATE_ID: str(plate_id),
                    "compiled_contexts": compiled_contexts,
                    "compiled_pipeline_definition": compiled_pipeline_definition,  # Store the stripped pipeline_definition
                    "worker_assignments": worker_assignments,
                    "output_plate_root": self.active_executions[execution_id].get_extra(
                        "output_plate_root"
                    ),
                    "auto_add_output_plate": self.active_executions[
                        execution_id
                    ].get_extra("auto_add_output_plate"),
                }
                logger.info(
                    "[%s] Compilation-only request completed and artifact stored (artifact_id=%s sig=%s)",
                    execution_id,
                    execution_id,
                    request_signature[:12],
                )
                self._set_compile_status("compiled success")
                return compiled_contexts

            log_dir = Path.home() / ".local" / "share" / "openhcs" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)

            worker_progress_queue = multiprocessing.get_context("spawn").Queue()
            progress_forwarder = threading.Thread(
                target=self._forward_worker_progress,
                args=(worker_progress_queue,),
                daemon=True,
            )
            progress_forwarder.start()

            if (
                self.active_executions[execution_id].status
                == ExecutionStatus.CANCELLED.value
            ):
                logger.info(
                    "[%s] Execution cancelled before starting workers, aborting",
                    execution_id,
                )
                raise RuntimeError("Execution cancelled by user")

            # DEBUG: Check steps right before execution
            logger.info(
                f"🔍 PRE-EXEC: compiled_pipeline_definition is None? {compiled_pipeline_definition is None}"
            )
            if compiled_pipeline_definition is not None:
                logger.info(
                    f"🔍 PRE-EXEC: compiled_pipeline_definition[0].func = {type(getattr(compiled_pipeline_definition[0], 'func', None)).__name__ if getattr(compiled_pipeline_definition[0], 'func', None) else 'None'}"
                )
            logger.info(
                f"🔍 PRE-EXEC: pipeline_steps[0].func = {type(getattr(pipeline_steps[0], 'func', None)).__name__ if getattr(pipeline_steps[0], 'func', None) else 'None'}"
            )

            # Use compiled pipeline_definition if available (from fresh compilation),
            # otherwise use original pipeline_steps (from artifact reuse)
            # NOTE: For artifact reuse, we don't have the compiled pipeline_definition,
            # so we use the original pipeline_steps (functions will be resolved from context)
            steps_to_execute = (
                compiled_pipeline_definition
                if compiled_pipeline_definition is not None
                else pipeline_steps
            )

            # DEBUG: Log what we're passing to execution
            logger.info(
                f"🚀 ZMQ SERVER: Passing {len(steps_to_execute)} steps to execution"
            )
            for i, step in enumerate(steps_to_execute):
                func_attr = getattr(step, "func", None)
                func_type = type(func_attr).__name__ if func_attr else "None"
                logger.info(f"🚀 ZMQ SERVER: step[{i}].func = {func_type}")

            return orchestrator.execute_compiled_plate(
                pipeline_definition=steps_to_execute,
                compiled_contexts=compiled_contexts,
                log_file_base=str(log_dir / f"zmq_worker_exec_{execution_id}"),
                progress_queue=worker_progress_queue,
                progress_context=progress_context,
                worker_assignments=worker_assignments,
            )
        finally:
            if worker_progress_queue is not None:
                worker_progress_queue.put(None)
            if progress_forwarder is not None:
                progress_forwarder.join()
            self._worker_assignments_by_execution.pop(execution_id, None)

    def _kill_worker_processes(self) -> int:
        """OpenHCS-specific worker cleanup (graceful cancellation + kill)."""
        for eid, record in self.active_executions.items():
            orchestrator = record.get_extra("orchestrator")
            if orchestrator is not None:
                try:
                    logger.info("[%s] Requesting graceful cancellation...", eid)
                    orchestrator.cancel_execution()
                except Exception as e:
                    logger.warning("[%s] Graceful cancellation failed: %s", eid, e)
        return super()._kill_worker_processes()
