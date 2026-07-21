"""OpenHCS execution client built on zmqruntime ExecutionClient."""

from __future__ import annotations

import hashlib
import logging
import pickle
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TypeAlias

from typing_extensions import override
import zmq
from zmqruntime.execution import ExecutionClient
from zmqruntime.messages import ControlMessageType, MessageFields

from zmqruntime.transport import coerce_transport_mode, get_zmq_transport_url
from openhcs.core.config import GlobalPipelineConfig, PipelineConfig
from openhcs.core.config_document import ConfigDocumentAuthority
from openhcs.core.artifact_inspection import CompiledArtifactInspection
from openhcs.core.debug import DebugExecutionConfig
from openhcs.core.pipeline_document import (
    PipelineDocument,
    PipelineDocumentAuthority,
)
from openhcs.core.steps.function_step import FunctionStep
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG, OpenHCSZMQConfig
from openhcs.runtime.zmq_execution_signature import (
    ZMQExecutionConfigTransport,
    ZMQExecutionCompileControl,
    ZMQExecutionIdentity,
    ZMQExecutionRequestPayload,
)

logger = logging.getLogger(__name__)


ZMQScalar: TypeAlias = str | int | float | bool | None
ZMQValue: TypeAlias = ZMQScalar | Mapping[str, "ZMQValue"] | Sequence["ZMQValue"]
ZMQParams: TypeAlias = Mapping[str, ZMQValue]
PlateIdentifier: TypeAlias = str | Path


def _optional_plate_identifier(value: PlateIdentifier | None) -> str | None:
    if value is None:
        return None
    return str(value)


@dataclass(slots=True, init=False)
class OpenHCSExecutionSubmission:
    """Nominal client-side submission payload for OpenHCS ZMQ execution."""

    identity: ZMQExecutionIdentity
    pipeline_document: PipelineDocument
    global_pipeline_config: GlobalPipelineConfig
    config_boundary: ZMQConfigParamsBoundary
    compile_control: ZMQExecutionCompileControl = ZMQExecutionCompileControl()

    def __init__(
        self,
        *,
        plate_id: PlateIdentifier,
        pipeline_document: PipelineDocument,
        global_config: GlobalPipelineConfig,
        execution_plate_id: PlateIdentifier | None = None,
        selected_pipeline_path: PlateIdentifier | None = None,
        config_params: ZMQParams | None = None,
        compile_artifact_id: str | None = None,
        compile_only: bool = False,
    ) -> None:
        normalized_document = PipelineDocumentAuthority.from_values(
            pipeline_config=pipeline_document.pipeline_config,
            pipeline_steps=pipeline_document.pipeline_steps,
        )
        normalized_document = replace(
            normalized_document,
            original_source=pipeline_document.original_source,
        )
        self._set_parts(
            identity=ZMQExecutionIdentity(
                plate_id=str(plate_id),
                execution_plate_id=_optional_plate_identifier(execution_plate_id),
                selected_pipeline_path=_optional_plate_identifier(
                    selected_pipeline_path
                ),
            ),
            pipeline_document=normalized_document,
            global_pipeline_config=global_config,
            config_boundary=ZMQConfigParamsBoundary(config_params),
            compile_control=ZMQExecutionCompileControl(
                compile_artifact_id=compile_artifact_id,
                compile_only=compile_only,
            ),
        )

    @classmethod
    def _from_parts(
        cls,
        *,
        identity: ZMQExecutionIdentity,
        pipeline_document: PipelineDocument,
        global_pipeline_config: GlobalPipelineConfig,
        config_boundary: ZMQConfigParamsBoundary,
        compile_control: ZMQExecutionCompileControl,
    ) -> "OpenHCSExecutionSubmission":
        submission = cls.__new__(cls)
        submission._set_parts(
            identity=identity,
            pipeline_document=pipeline_document,
            global_pipeline_config=global_pipeline_config,
            config_boundary=config_boundary,
            compile_control=compile_control,
        )
        return submission

    def _set_parts(
        self,
        *,
        identity: ZMQExecutionIdentity,
        pipeline_document: PipelineDocument,
        global_pipeline_config: GlobalPipelineConfig,
        config_boundary: ZMQConfigParamsBoundary,
        compile_control: ZMQExecutionCompileControl,
    ) -> None:
        self.identity = identity
        self.pipeline_document = pipeline_document
        self.global_pipeline_config = global_pipeline_config
        self.config_boundary = config_boundary
        self.compile_control = compile_control

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
    def pipeline_steps(self) -> list[FunctionStep]:
        return self.pipeline_document.pipeline_steps

    @property
    def pipeline_config(self) -> PipelineConfig:
        return self.pipeline_document.pipeline_config

    @property
    def config_params(self) -> ZMQParams | None:
        return self.config_boundary.params

    @property
    def compile_artifact_id(self) -> str | None:
        return self.compile_control.compile_artifact_id

    @property
    def compile_only(self) -> bool:
        return self.compile_control.compile_only

    def to_task(self) -> "OpenHCSExecutionSubmission":
        return self

    def with_config_params(
        self, config_params: ZMQParams
    ) -> "OpenHCSExecutionSubmission":
        return self._from_parts(
            identity=self.identity,
            pipeline_document=self.pipeline_document,
            global_pipeline_config=self.global_pipeline_config,
            config_boundary=ZMQConfigParamsBoundary(config_params),
            compile_control=self.compile_control,
        )

    def compile_request(self) -> "OpenHCSExecutionSubmission":
        return self._from_parts(
            identity=self.identity,
            pipeline_document=self.pipeline_document,
            global_pipeline_config=self.global_pipeline_config,
            config_boundary=self.config_boundary,
            compile_control=self.compile_control.as_compile_request(),
        )

    def pipeline_code(self) -> str:
        return PipelineDocumentAuthority.execution_source(self.pipeline_document)

    def step_count_label(self) -> str:
        return str(len(self.pipeline_steps))


def _pycodify_config_source(
    config: GlobalPipelineConfig | OpenHCSZMQConfig,
) -> str:
    return ConfigDocumentAuthority.render(
        config,
        expected_config_type=type(config),
    )


@dataclass(frozen=True, slots=True)
class ZMQExecutionRequestBuilder:
    task: OpenHCSExecutionSubmission
    pipeline_code: str
    config_projection: "ZMQConfigProjection"

    @classmethod
    def from_task(
        cls,
        task: OpenHCSExecutionSubmission,
    ) -> "ZMQExecutionRequestBuilder":
        return cls(
            task=task,
            pipeline_code=task.pipeline_code(),
            config_projection=ZMQConfigProjection.from_task(task),
        )

    @property
    def request_payload(self) -> ZMQExecutionRequestPayload:
        return ZMQExecutionRequestPayload(
            identity=self.task.identity,
            pipeline_code=self.pipeline_code,
            config_transport=self.config_projection.signature_transport(),
            compile_control=self.task.compile_control,
        )

    def request(self) -> "ZMQRequest":
        return ZMQRequest.from_items(
            (
                (MessageFields.TYPE, ControlMessageType.EXECUTE.value),
                (MessageFields.PIPELINE_CODE, self.pipeline_code),
                *self.task.identity.request_items(),
                *self.task.compile_control.request_items(),
                *self.config_projection.request_items(),
            )
        )


@dataclass(frozen=True, slots=True)
class ZMQRequest:
    """Nominal OpenHCS execution request before lowering to zmqruntime."""

    values: Mapping[str, ZMQValue]

    @classmethod
    def from_items(
        cls,
        items: Sequence[tuple[str, ZMQValue]],
    ) -> "ZMQRequest":
        return cls(values=dict(items))

    def with_items(
        self,
        items: Sequence[tuple[str, ZMQValue]],
    ) -> "ZMQRequest":
        values = dict(self.values)
        values.update(items)
        return ZMQRequest(values=values)

    def as_wire_payload(self) -> dict[str, ZMQValue]:
        return dict(self.values)


@dataclass(frozen=True, slots=True)
class ZMQConfigParamsBoundary:
    """Boundary for zmqruntime's auxiliary params field, not config authority."""

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
        return ((MessageFields.CONFIG_PARAMS, self.params),)


@dataclass(frozen=True, slots=True)
class ZMQConfigSourceFields:
    config_code: str | None = None

    def request_items(self) -> tuple[tuple[str, ZMQValue], ...]:
        items: list[tuple[str, ZMQValue]] = []
        if self.config_code is not None:
            items.append((MessageFields.CONFIG_CODE, self.config_code))
        return tuple(items)


@dataclass(frozen=True, slots=True)
class ZMQConfigProjection:
    params_boundary: ZMQConfigParamsBoundary | None
    source_fields: ZMQConfigSourceFields | None
    config_sha: str

    @classmethod
    def from_task(
        cls,
        task: OpenHCSExecutionSubmission,
    ) -> "ZMQConfigProjection":
        config_source = _pycodify_config_source(task.global_pipeline_config)
        return cls(
            params_boundary=(
                task.config_boundary
                if task.config_boundary.params is not None
                else None
            ),
            source_fields=ZMQConfigSourceFields(config_source),
            config_sha=hashlib.sha256(config_source.encode("utf-8")).hexdigest()[:12],
        )

    def request_items(self) -> tuple[tuple[str, ZMQValue], ...]:
        items: list[tuple[str, ZMQValue]] = []
        if self.params_boundary is not None:
            items.extend(self.params_boundary.request_items())
        if self.source_fields is not None:
            items.extend(self.source_fields.request_items())
        return tuple(items)

    def signature_transport(self) -> ZMQExecutionConfigTransport:
        source_fields = self.source_fields
        return ZMQExecutionConfigTransport(
            config_params=(
                None
                if self.params_boundary is None
                else (
                    None
                    if self.params_boundary.params is None
                    else dict(self.params_boundary.params)
                )
            ),
            config_code=None if source_fields is None else source_fields.config_code,
        )


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


class ZMQExecutionClient(ExecutionClient[OpenHCSExecutionSubmission, None]):
    """ZMQ client for OpenHCS pipeline execution with progress streaming."""

    def __init__(
        self,
        port: int | None = None,
        host: str | None = None,
        persistent: bool | None = None,
        progress_callback=None,
        transport_mode=None,
        config: OpenHCSZMQConfig = OPENHCS_ZMQ_CONFIG,
    ):
        super().__init__(
            config.default_port if port is None else port,
            config.client_host if host is None else host,
            config.persistent if persistent is None else persistent,
            progress_callback=progress_callback,
            transport_mode=(
                config.transport_mode
                if transport_mode is None
                else coerce_transport_mode(transport_mode)
            ),
            config=config,
        )

    def serialize_task(
        self,
        task: OpenHCSExecutionSubmission,
        config=None,
    ) -> dict[str, ZMQValue]:
        request_builder = ZMQExecutionRequestBuilder.from_task(task)
        request = request_builder.request()
        request_payload = request_builder.request_payload
        logger.info(
            "Serialize task: plate=%s compile_only=%s artifact_id=%s step_count=%s pipeline_sha=%s config_sha=%s",
            task.plate_id,
            task.compile_only,
            task.compile_artifact_id,
            task.step_count_label(),
            request_payload.pipeline_sha,
            request_builder.config_projection.config_sha,
        )
        return request.as_wire_payload()

    def submit_pipeline(
        self,
        submission: OpenHCSExecutionSubmission,
        *,
        timeout_ms: int | None = None,
    ):
        return self._submit_submission(
            submission.to_task(),
            timeout_ms=self._control_timeout_ms(timeout_ms),
        )

    def submit_debug_pipeline(
        self,
        submission: OpenHCSExecutionSubmission,
        *,
        debug_config: DebugExecutionConfig,
        timeout_ms: int | None = None,
    ):
        config_params_boundary = ZMQConfigParamsBoundary.from_optional(
            submission.config_params
        ).with_updates(debug_config.to_config_params())
        return self._submit_submission(
            submission.with_config_params(config_params_boundary.params).to_task(),
            timeout_ms=self._control_timeout_ms(timeout_ms),
        )

    def submit_compile(
        self,
        submission: OpenHCSExecutionSubmission,
        *,
        timeout_ms: int | None = None,
    ):
        return self._submit_submission(
            submission.compile_request().to_task(),
            timeout_ms=self._control_timeout_ms(timeout_ms),
        )

    def execute_pipeline(
        self,
        submission: OpenHCSExecutionSubmission,
    ):
        response = self.submit_pipeline(submission)
        response_view = ZMQClientResponseView(response)
        if response_view.accepted():
            return self.wait_for_completion(response_view.execution_id())
        return response

    def get_status(
        self,
        execution_id=None,
        *,
        timeout_ms: int | None = None,
    ):
        request = {MessageFields.TYPE: ControlMessageType.STATUS.value}
        if execution_id:
            request[MessageFields.EXECUTION_ID] = execution_id
        return self._send_control_request_bounded(
            request,
            timeout_ms=self._control_timeout_ms(timeout_ms),
        )

    def _control_timeout_ms(self, timeout_ms: int | None) -> int:
        return self.config.control_timeout_ms if timeout_ms is None else timeout_ms

    def _submit_submission(
        self,
        submission: OpenHCSExecutionSubmission,
        *,
        timeout_ms: int,
    ):
        connect_timeout_seconds = max(timeout_ms / 1000, 0.001)
        if not self._connected and not self.connect(timeout=connect_timeout_seconds):
            raise RuntimeError("Failed to connect to execution server")
        self._ensure_progress_subscription()
        request = self.serialize_task(submission, None)
        if MessageFields.TYPE not in request:
            request[MessageFields.TYPE] = ControlMessageType.EXECUTE.value
        return self._send_control_request_bounded(request, timeout_ms=timeout_ms)

    def _send_control_request_bounded(self, request, *, timeout_ms: int):
        owns_context = self.zmq_context is None
        ctx = zmq.Context() if owns_context else self.zmq_context
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        control_url = get_zmq_transport_url(
            self.control_port,
            host=self.host,
            mode=self.transport_mode,
            config=self.config,
        )
        request_type = request.get(MessageFields.TYPE, "control")
        sock.connect(control_url)
        poller = zmq.Poller()
        try:
            poller.register(sock, zmq.POLLOUT)
            writable = dict(poller.poll(timeout_ms))
            if not writable.get(sock):
                raise TimeoutError(
                    f"Server was not writable for {request_type} request within "
                    f"{timeout_ms}ms"
                )
            sock.send(pickle.dumps(request), flags=zmq.NOBLOCK)
            poller.unregister(sock)
            poller.register(sock, zmq.POLLIN)
            readable = dict(poller.poll(timeout_ms))
            if not readable.get(sock):
                raise TimeoutError(
                    f"Server did not respond to {request_type} request within "
                    f"{timeout_ms}ms"
                )
            return pickle.loads(sock.recv(flags=zmq.NOBLOCK))
        except zmq.Again as exc:
            raise TimeoutError(
                f"Server did not complete {request_type} request within {timeout_ms}ms"
            ) from exc
        finally:
            try:
                poller.unregister(sock)
            except Exception:
                pass
            sock.close(linger=0)
            if owns_context:
                ctx.term()

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

    def get_compiled_artifact_inspection(
        self,
        compile_artifact_id: str,
    ) -> CompiledArtifactInspection:
        """Read the compiler-owned artifact inspection projection."""

        from openhcs.core.artifact_inspection import (
            CompiledArtifactInspectionControlPayload,
            CompiledArtifactInspectionRequest,
            CompiledArtifactInspectionResponse,
        )

        if not self._connected and not self.connect():
            raise RuntimeError("Failed to connect to execution server")
        request = CompiledArtifactInspectionRequest(
            compile_artifact_id=compile_artifact_id
        )
        response = self._send_control_request(
            CompiledArtifactInspectionControlPayload.from_request(request).to_dict()
        )
        return CompiledArtifactInspectionResponse.from_control_response(
            response
        ).inspection

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

    def get_debug_runtime_inspection(
        self,
        *,
        debug_session_id: str,
    ):
        from openhcs.core.debug import (
            DebugRuntimeInspectionControlPayload,
            DebugRuntimeInspectionRequest,
            DebugRuntimeInspectionResponse,
        )

        if not self._connected and not self.connect():
            raise RuntimeError("Failed to connect to execution server")
        request = DebugRuntimeInspectionRequest(debug_session_id=debug_session_id)
        response = self._send_control_request(
            DebugRuntimeInspectionControlPayload.from_request(request).to_dict()
        )
        return DebugRuntimeInspectionResponse.from_control_response(response).view_model

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
        server_config = replace(
            self.config,
            default_port=self.port,
            persistent=self.persistent,
            transport_mode=self.transport_mode,
        )
        cmd = [
            sys.executable,
            "-m",
            "openhcs.runtime.zmq_execution_server_launcher",
        ]
        cmd.extend(["--log-file-path", str(log_file_path)])
        cmd.extend(["--config-source", _pycodify_config_source(server_config)])

        # Pass the current process's logging level to the server
        # Get the root logger's effective level
        root_logger = logging.getLogger()
        current_log_level = root_logger.getEffectiveLevel()
        log_level_name = logging.getLevelName(current_log_level)

        # Log what we're passing to help debug
        logger = logging.getLogger(__name__)
        logger.debug(
            f"Spawning ZMQ server with log level: {log_level_name} (numeric: {current_log_level})"
        )

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
