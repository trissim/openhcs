"""OpenHCS execution client built on zmqruntime ExecutionClient."""

from __future__ import annotations

import hashlib
import logging
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from typing_extensions import override
from zmqruntime.execution import ExecutionClient

from zmqruntime.transport import coerce_transport_mode
from openhcs.core.config import GlobalPipelineConfig, PipelineConfig
from openhcs.core.steps.abstract import AbstractStep
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG
from openhcs.runtime.zmq_execution_signature import (
    OpenHCSExecutionConfigBundle,
    OpenHCSExecutionConfigCarrier,
    ZMQExecutionCompileControl,
    ZMQExecutionIdentity,
)
from openhcs.runtime.zmq_pipeline_transport import (
    PipelineStepsBoundary,
    PipelineStepsCarrier,
)

logger = logging.getLogger(__name__)


ZMQScalar: TypeAlias = str | int | float | bool | None
ZMQValue: TypeAlias = (
    ZMQScalar
    | Mapping[str, "ZMQValue"]
    | Sequence["ZMQValue"]
)
ZMQParams: TypeAlias = Mapping[str, ZMQValue]
ZMQRequest: TypeAlias = dict[str, ZMQValue]
PlateIdentifier: TypeAlias = str | Path


def _optional_plate_identifier(value: PlateIdentifier | None) -> str | None:
    if value is None:
        return None
    return str(value)


@dataclass(slots=True, init=False)
class OpenHCSExecutionSubmission(PipelineStepsCarrier, OpenHCSExecutionConfigCarrier):
    """Nominal client-side submission payload for OpenHCS ZMQ execution."""

    registry_key = "openhcs_execution_submission"

    identity: ZMQExecutionIdentity
    submission_pipeline: PipelineStepsBoundary
    configs: OpenHCSExecutionConfigBundle
    config_boundary: ZMQConfigParamsBoundary
    compile_control: ZMQExecutionCompileControl = ZMQExecutionCompileControl()
    pipeline_source: str | None = None

    def __init__(
        self,
        *,
        plate_id: PlateIdentifier,
        pipeline_steps: Sequence[AbstractStep],
        global_config: GlobalPipelineConfig,
        execution_plate_id: PlateIdentifier | None = None,
        selected_pipeline_path: PlateIdentifier | None = None,
        pipeline_config: PipelineConfig | None = None,
        config_params: ZMQParams | None = None,
        compile_artifact_id: str | None = None,
        pipeline_source: str | None = None,
        compile_only: bool = False,
    ) -> None:
        self._set_parts(
            identity=ZMQExecutionIdentity(
                plate_id=str(plate_id),
                execution_plate_id=_optional_plate_identifier(execution_plate_id),
                selected_pipeline_path=_optional_plate_identifier(
                    selected_pipeline_path
                ),
            ),
            submission_pipeline=PipelineStepsBoundary(pipeline_steps),
            configs=OpenHCSExecutionConfigBundle(
                global_pipeline=global_config,
                plate_pipeline=pipeline_config,
            ),
            config_boundary=ZMQConfigParamsBoundary(config_params),
            compile_control=ZMQExecutionCompileControl(
                compile_artifact_id=compile_artifact_id,
                compile_only=compile_only,
            ),
            pipeline_source=pipeline_source,
        )

    @classmethod
    def _from_parts(
        cls,
        *,
        identity: ZMQExecutionIdentity,
        submission_pipeline: PipelineStepsBoundary,
        configs: OpenHCSExecutionConfigBundle,
        config_boundary: ZMQConfigParamsBoundary,
        compile_control: ZMQExecutionCompileControl,
        pipeline_source: str | None,
    ) -> "OpenHCSExecutionSubmission":
        submission = cls.__new__(cls)
        submission._set_parts(
            identity=identity,
            submission_pipeline=submission_pipeline,
            configs=configs,
            config_boundary=config_boundary,
            compile_control=compile_control,
            pipeline_source=pipeline_source,
        )
        return submission

    def _set_parts(
        self,
        *,
        identity: ZMQExecutionIdentity,
        submission_pipeline: PipelineStepsBoundary,
        configs: OpenHCSExecutionConfigBundle,
        config_boundary: ZMQConfigParamsBoundary,
        compile_control: ZMQExecutionCompileControl,
        pipeline_source: str | None,
    ) -> None:
        self.identity = identity
        self.submission_pipeline = submission_pipeline
        self.configs = configs
        self.config_boundary = config_boundary
        self.compile_control = compile_control
        self.pipeline_source = pipeline_source

    @property
    def plate_id(self) -> str:
        return self.identity.plate_id

    @property
    def execution_plate_id(self) -> str | None:
        return self.identity.execution_plate_id

    @property
    def selected_pipeline_path(self) -> str | None:
        return self.identity.selected_pipeline_path

    @property
    def pipeline_steps_boundary(self) -> PipelineStepsBoundary:
        return self.submission_pipeline

    @property
    def config_params(self) -> ZMQParams | None:
        return self.config_boundary.params

    @property
    def execution_config_bundle(self) -> OpenHCSExecutionConfigBundle:
        return self.configs

    @property
    def compile_artifact_id(self) -> str | None:
        return self.compile_control.compile_artifact_id

    @property
    def compile_only(self) -> bool:
        return self.compile_control.compile_only

    def to_task(self) -> "OpenHCSExecutionSubmission":
        return self

    def with_config_params(self, config_params: ZMQParams) -> "OpenHCSExecutionSubmission":
        return self._from_parts(
            identity=self.identity,
            submission_pipeline=self.submission_pipeline,
            configs=self.configs,
            config_boundary=ZMQConfigParamsBoundary(config_params),
            compile_control=self.compile_control,
            pipeline_source=self.pipeline_source,
        )

    def compile_request(self) -> "OpenHCSExecutionSubmission":
        return self._from_parts(
            identity=self.identity,
            submission_pipeline=self.submission_pipeline,
            configs=self.configs,
            config_boundary=self.config_boundary,
            compile_control=self.compile_control.as_compile_request(),
            pipeline_source=self.pipeline_source,
        )

    def step_count_label(self) -> str:
        return str(len(self.submission_pipeline))


@dataclass(frozen=True, slots=True)
class PycodifiedSource:
    source: str

    def sha_label(self) -> str:
        return hashlib.sha256(self.source.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True, slots=True)
class PycodifiedPipelineCode(PycodifiedSource):
    @classmethod
    def from_task(cls, task: OpenHCSExecutionSubmission) -> "PycodifiedPipelineCode":
        if task.pipeline_source is not None:
            return cls(source=task.pipeline_source)
        step_source = PycodifiedPipelineStepSource(task.submission_pipeline)
        return cls(source=step_source.source())


@dataclass(frozen=True, slots=True)
class PycodifiedPipelineStepSource(PipelineStepsCarrier):
    registry_key = "pycodified_pipeline_step_source"

    source_pipeline: PipelineStepsBoundary

    @property
    def pipeline_steps_boundary(self) -> PipelineStepsBoundary:
        return self.source_pipeline

    def source(self) -> str:
        from openhcs.runtime.zmq_pipeline_transport import (
            ZMQPipelineCodeTransport,
            ZMQPipelineSourcePayload,
        )

        generated_source = PycodifyAssignmentSourceRequest(
            variable_name="pipeline_steps",
            value=self.source_pipeline.steps,
            header="# Edit this pipeline and save to apply changes",
        ).source()
        return ZMQPipelineCodeTransport.from_pipeline_source(
            ZMQPipelineSourcePayload(
                source=generated_source,
                source_pipeline=self.source_pipeline,
            )
        ).source


@dataclass(frozen=True, slots=True)
class PycodifiedConfigSource(PycodifiedSource):
    @classmethod
    def from_config(
        cls,
        config: GlobalPipelineConfig | PipelineConfig,
    ) -> "PycodifiedConfigSource":
        source_request = PycodifyAssignmentSourceRequest(
            variable_name="config",
            value=config,
            header="# Configuration Code",
        )
        return cls(source=source_request.source())


@dataclass(frozen=True, slots=True)
class PycodifyAssignmentSourceRequest:
    variable_name: str
    value: Sequence[AbstractStep] | GlobalPipelineConfig | PipelineConfig
    header: str

    def source(self) -> str:
        import openhcs.serialization.pycodify_formatters  # noqa: F401
        from pycodify import Assignment, generate_python_source

        return generate_python_source(
            Assignment(self.variable_name, self.value),
            self.header,
            True,
        )


@dataclass(frozen=True, slots=True)
class ZMQExecutionRequestBuilder:
    task: OpenHCSExecutionSubmission
    pipeline_transport: PycodifiedPipelineCode

    def request(self) -> ZMQRequest:
        request: ZMQRequest = {
            "type": "execute",
            "pipeline_code": self.pipeline_transport.source,
        }
        request.update(self.task.identity.request_items())
        request.update(self.task.compile_control.request_items())
        return request


@dataclass(frozen=True, slots=True)
class ZMQConfigParamsBoundary:
    params: ZMQParams | None

    @classmethod
    def from_optional(
        cls,
        params: ZMQParams | None,
    ) -> "ZMQConfigParamsBoundary":
        if params is None:
            return cls(params={})
        return cls(params=params)

    def with_updates(
        self,
        updates: ZMQParams,
    ) -> "ZMQConfigParamsBoundary":
        if self.params is None:
            merged_params: dict[str, ZMQValue] = {}
        else:
            merged_params = dict(self.params)
        merged_params.update(updates)
        return ZMQConfigParamsBoundary(params=merged_params)

    def request_items(self) -> tuple[tuple[str, ZMQValue], ...]:
        return (("config_params", self.params),)


@dataclass(frozen=True, slots=True)
class ZMQConfigSourceFields:
    config_code: str | None = None
    pipeline_config_code: str | None = None

    def request_items(self) -> tuple[tuple[str, ZMQValue], ...]:
        items: list[tuple[str, ZMQValue]] = []
        if self.config_code is not None:
            items.append(("config_code", self.config_code))
        if self.pipeline_config_code is not None:
            items.append(("pipeline_config_code", self.pipeline_config_code))
        return tuple(items)


@dataclass(frozen=True, slots=True)
class ZMQConfigProjection:
    params_boundary: ZMQConfigParamsBoundary | None
    source_fields: ZMQConfigSourceFields | None
    config_sha: str
    pipeline_config_sha: str

    @classmethod
    def from_task(
        cls,
        task: OpenHCSExecutionSubmission,
    ) -> "ZMQConfigProjection":
        if task.config_boundary.params is not None:
            return cls(
                params_boundary=task.config_boundary,
                source_fields=None,
                config_sha="params",
                pipeline_config_sha="params",
            )

        config_source = PycodifiedConfigSource.from_config(task.global_config)
        if task.pipeline_config is None:
            pipeline_config_code = None
            pipeline_config_sha = "-"
        else:
            pipeline_config_source = PycodifiedConfigSource.from_config(
                task.pipeline_config
            )
            pipeline_config_code = pipeline_config_source.source
            pipeline_config_sha = pipeline_config_source.sha_label()
        return cls(
            params_boundary=None,
            source_fields=ZMQConfigSourceFields(
                config_source.source,
                pipeline_config_code,
            ),
            config_sha=config_source.sha_label(),
            pipeline_config_sha=pipeline_config_sha,
        )

    def request_items(self) -> tuple[tuple[str, ZMQValue], ...]:
        items: list[tuple[str, ZMQValue]] = []
        if self.params_boundary is not None:
            items.extend(self.params_boundary.request_items())
        if self.source_fields is not None:
            items.extend(self.source_fields.request_items())
        return tuple(items)


@dataclass(frozen=True, slots=True)
class ZMQClientResponseView:
    response: Mapping[str, ZMQValue]

    def accepted(self) -> bool:
        if "status" not in self.response:
            return False
        return self.response["status"] == "accepted"

    def execution_id(self) -> str:
        if "execution_id" not in self.response:
            raise RuntimeError("Accepted execution response missing execution_id")
        value = self.response["execution_id"]
        if value is None:
            raise RuntimeError("Accepted execution response has null execution_id")
        return str(value)

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

    def serialize_task(self, task: OpenHCSExecutionSubmission, config=None) -> ZMQRequest:
        pipeline_code = PycodifiedPipelineCode.from_task(task)
        request = ZMQExecutionRequestBuilder(
            task=task,
            pipeline_transport=pipeline_code,
        ).request()
        config_projection = ZMQConfigProjection.from_task(task)
        request.update(config_projection.request_items())
        logger.info(
            "Serialize task: plate=%s compile_only=%s artifact_id=%s step_count=%s pipeline_sha=%s config_sha=%s pipeline_config_sha=%s",
            task.plate_id,
            task.compile_only,
            task.compile_artifact_id,
            task.step_count_label(),
            pipeline_code.sha_label(),
            config_projection.config_sha,
            config_projection.pipeline_config_sha,
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

        config_params_boundary = ZMQConfigParamsBoundary.from_optional(
            submission.config_params
        ).with_updates(
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
        return self.submit_execution(
            submission.with_config_params(config_params_boundary.params).to_task()
        )

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
        response_view = ZMQClientResponseView(response)
        if response_view.accepted():
            return self.wait_for_completion(response_view.execution_id())
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
            nvidia_paths = ":".join(venv_nvidia_libs)
            if "LD_LIBRARY_PATH" in env and env["LD_LIBRARY_PATH"]:
                env["LD_LIBRARY_PATH"] = f"{nvidia_paths}:{env['LD_LIBRARY_PATH']}"
            else:
                env["LD_LIBRARY_PATH"] = nvidia_paths

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
