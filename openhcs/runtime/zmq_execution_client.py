"""OpenHCS execution client built on zmqruntime ExecutionClient."""

from __future__ import annotations

import hashlib
import logging
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from typing_extensions import override
from zmqruntime.execution import ExecutionClient

from zmqruntime.transport import coerce_transport_mode
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OpenHCSExecutionSubmission:
    """Nominal client-side submission payload for OpenHCS ZMQ execution."""

    plate_id: Any
    pipeline_steps: Any
    global_config: Any
    execution_plate_id: Any = None
    selected_pipeline_path: Any = None
    pipeline_config: Any = None
    config_params: Any = None
    compile_artifact_id: Any = None
    compile_only: bool = False

    def to_task(self) -> dict[str, Any]:
        task = {
            "plate_id": self.plate_id,
            "execution_plate_id": self.execution_plate_id,
            "selected_pipeline_path": self.selected_pipeline_path,
            "pipeline_steps": self.pipeline_steps,
            "global_config": self.global_config,
            "pipeline_config": self.pipeline_config,
            "config_params": self.config_params,
        }
        if self.compile_artifact_id is not None:
            task["compile_artifact_id"] = self.compile_artifact_id
        if self.compile_only:
            task["compile_only"] = True
        return task

    def with_config_params(self, config_params: Any) -> "OpenHCSExecutionSubmission":
        return OpenHCSExecutionSubmission(
            plate_id=self.plate_id,
            pipeline_steps=self.pipeline_steps,
            global_config=self.global_config,
            execution_plate_id=self.execution_plate_id,
            selected_pipeline_path=self.selected_pipeline_path,
            pipeline_config=self.pipeline_config,
            config_params=config_params,
            compile_artifact_id=self.compile_artifact_id,
            compile_only=self.compile_only,
        )

    def compile_request(self) -> "OpenHCSExecutionSubmission":
        return OpenHCSExecutionSubmission(
            plate_id=self.plate_id,
            pipeline_steps=self.pipeline_steps,
            global_config=self.global_config,
            execution_plate_id=self.execution_plate_id,
            selected_pipeline_path=self.selected_pipeline_path,
            pipeline_config=self.pipeline_config,
            config_params=self.config_params,
            compile_artifact_id=self.compile_artifact_id,
            compile_only=True,
        )


class ZMQExecutionClient(ExecutionClient):
    """ZMQ client for OpenHCS pipeline execution with progress streaming."""

    def __init__(
        self,
        port: int = OPENHCS_ZMQ_CONFIG.default_port,
        host: str = "localhost",
        persistent: bool = True,
        progress_callback=None,
        transport_mode=None,
    ):
        super().__init__(
            port,
            host,
            persistent,
            progress_callback=progress_callback,
            transport_mode=coerce_transport_mode(transport_mode),
            config=OPENHCS_ZMQ_CONFIG,
        )

    def serialize_task(self, task: Any, config: Any = None) -> dict:
        import openhcs.serialization.pycodify_formatters  # noqa: F401
        from pycodify import Assignment, generate_python_source
        from openhcs.core.config import GlobalPipelineConfig, PipelineConfig

        plate_id = task.get("plate_id")
        pipeline_steps = task.get("pipeline_steps")
        execution_plate_id = task.get("execution_plate_id")
        selected_pipeline_path = task.get("selected_pipeline_path")
        global_config = task.get("global_config")
        pipeline_config = task.get("pipeline_config")
        config_params = task.get("config_params")
        compile_only = task.get("compile_only", False)
        compile_artifact_id = task.get("compile_artifact_id")

        pipeline_code = generate_python_source(
            Assignment("pipeline_steps", pipeline_steps),
            header="# Edit this pipeline and save to apply changes",
            clean_mode=True,
        )
        from openhcs.runtime.zmq_pipeline_transport import ZMQPipelineCodeTransport

        pipeline_code = ZMQPipelineCodeTransport.from_pipeline_source(
            source=pipeline_code,
            pipeline_steps=pipeline_steps,
        ).source
        request = {
            "type": "execute",
            "plate_id": str(plate_id),
            "pipeline_code": pipeline_code,
        }
        if execution_plate_id is not None:
            request["execution_plate_id"] = str(execution_plate_id)
        if selected_pipeline_path is not None:
            request["selected_pipeline_path"] = str(selected_pipeline_path)
        pipeline_sha = hashlib.sha256(pipeline_code.encode("utf-8")).hexdigest()[:12]
        if compile_only:
            request["compile_only"] = True
        if compile_artifact_id:
            request["compile_artifact_id"] = str(compile_artifact_id)
        if config_params:
            request["config_params"] = config_params
        else:
            config_code = generate_python_source(
                Assignment("config", global_config),
                header="# Configuration Code",
                clean_mode=True,
            )
            request["config_code"] = config_code
            config_sha = hashlib.sha256(config_code.encode("utf-8")).hexdigest()[:12]
            if pipeline_config:
                pipeline_config_code = generate_python_source(
                    Assignment("config", pipeline_config),
                    header="# Configuration Code",
                    clean_mode=True,
                )
                request["pipeline_config_code"] = pipeline_config_code
                pipeline_config_sha = hashlib.sha256(
                    pipeline_config_code.encode("utf-8")
                ).hexdigest()[:12]
            else:
                pipeline_config_sha = "-"
        if config_params:
            config_sha = "params"
            pipeline_config_sha = "params"
        logger.info(
            "Serialize task: plate=%s compile_only=%s artifact_id=%s step_count=%s pipeline_sha=%s config_sha=%s pipeline_config_sha=%s",
            plate_id,
            bool(compile_only),
            compile_artifact_id,
            len(pipeline_steps) if isinstance(pipeline_steps, list) else "?",
            pipeline_sha,
            config_sha,
            pipeline_config_sha,
        )
        return request

    def submit_pipeline(
        self,
        submission: OpenHCSExecutionSubmission,
    ):
        return self.submit_execution(submission.to_task())

    def submit_debug_pipeline(
        self,
        submission: OpenHCSExecutionSubmission,
        *,
        debug_session_id: str,
        snapshot_store_ref: str | None = None,
        snapshot_store_backend: str | None = None,
        command_type=None,
        selected_source_group: str | None = None,
        pause_step_indices=(),
        start_step_index: int = 0,
        start_after_invocation_key: str | None = None,
        replay_mode=None,
    ):
        from openhcs.core.debug import DebugCommandType, DebugExecutionConfig, DebugReplayMode

        merged_config_params = dict(submission.config_params or {})
        merged_config_params.update(
            DebugExecutionConfig(
                debug_session_id=debug_session_id,
                snapshot_store_ref=snapshot_store_ref,
                snapshot_store_backend=snapshot_store_backend,
                command_type=(
                    DebugCommandType.RUN
                    if command_type is None
                    else DebugCommandType(command_type)
                ),
                selected_source_group=selected_source_group,
                pause_step_indices=tuple(pause_step_indices),
                start_step_index=start_step_index,
                start_after_invocation_key=start_after_invocation_key,
                replay_mode=(
                    DebugReplayMode.WARM_ARTIFACT
                    if replay_mode is None
                    else DebugReplayMode(replay_mode)
                ),
            ).to_config_params()
        )
        return self.submit_execution(submission.with_config_params(merged_config_params).to_task())

    def submit_compile(
        self,
        submission: OpenHCSExecutionSubmission,
    ):
        return self.submit_execution(submission.compile_request().to_task())

    def execute_pipeline(
        self,
        submission: OpenHCSExecutionSubmission,
    ):
        response = self.submit_pipeline(submission)
        if response.get("status") == "accepted":
            execution_id = response.get("execution_id")
            return self.wait_for_completion(execution_id)
        return response

    def get_status(self, execution_id=None):
        return self.poll_status(execution_id)

    def get_debug_snapshot(
        self,
        *,
        debug_session_id: str,
        snapshot_id: str,
        snapshot_store_ref: str,
        snapshot_store_backend: str | None = None,
    ):
        from openhcs.core.debug import (
            DebugSnapshotReadControlPayload,
            DebugSnapshotReadRequest,
            DebugSnapshotReadResponse,
        )

        if not self._connected and not self.connect():
            raise RuntimeError("Failed to connect to execution server")
        request = DebugSnapshotReadRequest(
            debug_session_id=debug_session_id,
            snapshot_id=snapshot_id,
            snapshot_store_ref=snapshot_store_ref,
            snapshot_store_backend=snapshot_store_backend,
        )
        response = self._send_control_request(
            DebugSnapshotReadControlPayload.from_request(request).to_dict()
        )
        return DebugSnapshotReadResponse.from_control_response(response).snapshot

    def send_debug_worker_command(
        self,
        *,
        debug_session_id: str,
        command_type,
    ):
        from openhcs.core.debug import (
            DebugCommandType,
            DebugWorkerCommandControlPayload,
            DebugWorkerCommandRequest,
            DebugWorkerCommandResponse,
        )

        if not self._connected and not self.connect():
            raise RuntimeError("Failed to connect to execution server")
        request = DebugWorkerCommandRequest(
            debug_session_id=debug_session_id,
            command_type=DebugCommandType(command_type),
        )
        response = self._send_control_request(
            DebugWorkerCommandControlPayload.from_request(request).to_dict()
        )
        return DebugWorkerCommandResponse.from_control_response(response)

    def export_debug_artifact(
        self,
        *,
        debug_session_id: str,
        artifact_ref,
        export_root: str,
        snapshot_store_ref: str | None = None,
        snapshot_store_backend: str | None = None,
    ):
        from openhcs.core.debug import (
            DebugArtifactExportControlPayload,
            DebugArtifactExportRequest,
            DebugArtifactExportResponse,
        )

        if not self._connected and not self.connect():
            raise RuntimeError("Failed to connect to execution server")
        request = DebugArtifactExportRequest(
            debug_session_id=debug_session_id,
            artifact_ref=artifact_ref,
            export_root=export_root,
            snapshot_store_ref=snapshot_store_ref,
            snapshot_store_backend=snapshot_store_backend,
        )
        response = self._send_control_request(
            DebugArtifactExportControlPayload.from_request(request).to_dict()
        )
        return DebugArtifactExportResponse.from_control_response(response)

    @override
    def _spawn_server_process(self):
        import os
        import glob
        import logging

        log_dir = Path.home() / ".local" / "share" / "openhcs" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = (
            log_dir
            / f"openhcs_zmq_server_port_{self.port}_{int(time.time() * 1000000)}.log"
        )
        cmd = [
            sys.executable,
            "-m",
            "openhcs.runtime.zmq_execution_server_launcher",
            "--port",
            str(self.port),
        ]
        if self.persistent:
            cmd.append("--persistent")
        cmd.extend(["--log-file-path", str(log_file_path)])
        cmd.extend(["--transport-mode", self.transport_mode.value])

        # Pass the current process's logging level to the server
        # Get the root logger's effective level
        root_logger = logging.getLogger()
        current_log_level = root_logger.getEffectiveLevel()
        log_level_name = logging.getLevelName(current_log_level)

        # Log what we're passing to help debug
        logger = logging.getLogger(__name__)
        logger.debug(f"Spawning ZMQ server with log level: {log_level_name} (numeric: {current_log_level})")

        cmd.extend(["--log-level", log_level_name])

        env = os.environ.copy()
        site_packages = (
            Path(sys.executable).parent.parent
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        )
        nvidia_lib_pattern = str(site_packages / "nvidia" / "*" / "lib")
        venv_nvidia_libs = [
            p for p in glob.glob(nvidia_lib_pattern) if os.path.isdir(p)
        ]

        if venv_nvidia_libs:
            existing_ld_path = env.get("LD_LIBRARY_PATH", "")
            nvidia_paths = ":".join(venv_nvidia_libs)
            env["LD_LIBRARY_PATH"] = (
                f"{nvidia_paths}:{existing_ld_path}"
                if existing_ld_path
                else nvidia_paths
            )

        return subprocess.Popen(
            cmd,
            stdout=open(log_file_path, "w"),
            stderr=subprocess.STDOUT,
            start_new_session=self.persistent,
            env=env,
        )

    @override
    def send_data(self, data):
        pass

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
