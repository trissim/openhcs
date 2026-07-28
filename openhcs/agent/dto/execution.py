"""Execution session DTOs for the OpenHCS agent API."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Self

from openhcs.agent.dto.common import (
    AgentCliArgumentSpec,
    AgentCliRequest,
    AgentError,
    AgentResultEnvelope,
    AgentWarning,
    JsonObject,
    JsonValue,
    SCHEMA_VERSION,
)
from openhcs.agent.ui_bridge_identities import (
    PlateManagerOrchestratorCodeDocumentIdentity,
)
from openhcs.core.debug_views import DebugViewModel
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG


MAX_EXECUTION_STATUS_TRACEBACK_CHARS = 3000


@dataclass(frozen=True, slots=True)
class ExecutionConnectionSpec:
    host: str = "localhost"
    port: int | None = None
    transport_mode: str | None = None
    persistent: bool = True

    @classmethod
    def from_fields(
        cls,
        *,
        host: str = "localhost",
        port: int | None = None,
        transport_mode: str | None = None,
        persistent: bool = True,
    ) -> Self:
        return cls(
            host=host,
            port=port,
            transport_mode=transport_mode,
            persistent=persistent,
        )

    def require_port(self, purpose: str) -> int:
        if self.port is None:
            raise ValueError(f"{purpose} requires an explicit port.")
        return self.port

    def resolved_transport_mode(self):
        from zmqruntime.transport import coerce_transport_mode

        return coerce_transport_mode(self.transport_mode)

    def zmq_data_url(self, config) -> str:
        from zmqruntime.transport import get_zmq_transport_url

        return get_zmq_transport_url(
            self.require_port("ZMQ data URL"),
            host=self.host,
            mode=self.resolved_transport_mode(),
            config=config,
        )

    def zmq_control_port(self, config) -> int:
        from zmqruntime.transport import get_control_port

        return get_control_port(self.require_port("ZMQ control port"), config)

    def zmq_control_url(self, config) -> str:
        from zmqruntime.transport import get_control_url

        return get_control_url(
            self.require_port("ZMQ control URL"),
            self.transport_mode,
            host=self.host,
            config=config,
        )

    def zmq_client_kwargs(self) -> JsonObject:
        kwargs: JsonObject = {
            "host": self.host,
            "persistent": self.persistent,
            "transport_mode": self.transport_mode,
        }
        if self.port is not None:
            kwargs["port"] = self.port
        return kwargs


@dataclass(frozen=True, kw_only=True)
class ExecutionConnectionProjection:
    connection: ExecutionConnectionSpec = field(default_factory=ExecutionConnectionSpec)

    @property
    def host(self) -> str:
        return self.connection.host

    @property
    def port(self) -> int | None:
        return self.connection.port

    @property
    def transport_mode(self) -> str | None:
        return self.connection.transport_mode


class RuntimeServerToolRequest(AgentCliRequest):
    """Nominal request type for generated runtime-server dev-client commands."""


class RuntimeServerConnectionToolRequest(RuntimeServerToolRequest):
    """Nominal request type for runtime-server tools using connection fields."""

    connection: ExecutionConnectionSpec
    timeout_ms: int

    @classmethod
    @abstractmethod
    def from_fields(
        cls,
        *,
        host: str = "localhost",
        port: int | None = None,
        transport_mode: str | None = None,
        persistent: bool = True,
        timeout_ms: int | None = None,
    ) -> Self:
        raise NotImplementedError

    def as_tool_arguments(self) -> JsonObject:
        return {
            "host": self.connection.host,
            "port": self.connection.port,
            "transport_mode": self.connection.transport_mode,
            "persistent": self.connection.persistent,
            "timeout_ms": self.timeout_ms,
        }


@dataclass(frozen=True, kw_only=True)
class OrchestratorSessionIdentity:
    session_id: str


@dataclass(frozen=True, slots=True)
class OrchestratorSessionRequest(OrchestratorSessionIdentity):
    """Request one stored execution session by opaque session id."""


@dataclass(frozen=True, slots=True)
class OrchestratorSessionRef(OrchestratorSessionIdentity):
    schema_version: str
    uri: str


@dataclass(frozen=True, slots=True)
class OrchestratorSessionCreationRequest(ExecutionConnectionProjection):
    """Create an execution session from an in-memory pipeline draft."""

    plate_path: str
    pipeline_id: str
    execution_plate_path: str | None = None
    selected_pipeline_path: str | None = None
    global_config_id: str | None = None

    @classmethod
    def from_fields(
        cls,
        *,
        plate_path: str,
        pipeline_id: str,
        execution_plate_path: str | None = None,
        selected_pipeline_path: str | None = None,
        global_config_id: str | None = None,
        host: str = "localhost",
        port: int | None = None,
        transport_mode: str | None = None,
        persistent: bool = True,
    ) -> "OrchestratorSessionCreationRequest":
        return cls(
            plate_path=plate_path,
            pipeline_id=pipeline_id,
            execution_plate_path=execution_plate_path,
            selected_pipeline_path=selected_pipeline_path,
            global_config_id=global_config_id,
            connection=ExecutionConnectionSpec.from_fields(
                host=host,
                port=port,
                transport_mode=transport_mode,
                persistent=persistent,
            ),
        )


@dataclass(frozen=True, slots=True)
class PipelineSourceOrchestratorSessionRequest(ExecutionConnectionProjection):
    """Create an execution session from pycodified pipeline source."""

    plate_path: str
    pipeline_source: str
    global_config_id: str | None = None

    @classmethod
    def from_fields(
        cls,
        *,
        plate_path: str,
        pipeline_source: str,
        global_config_id: str | None = None,
        host: str = "localhost",
        port: int | None = None,
        transport_mode: str | None = None,
        persistent: bool = True,
    ) -> "PipelineSourceOrchestratorSessionRequest":
        return cls(
            plate_path=plate_path,
            pipeline_source=pipeline_source,
            global_config_id=global_config_id,
            connection=ExecutionConnectionSpec.from_fields(
                host=host,
                port=port,
                transport_mode=transport_mode,
                persistent=persistent,
            ),
        )


@dataclass(frozen=True, slots=True)
class PipelineSourceArtifactPlanInspectionRequest:
    """Compile pycodified pipeline source and inspect bounded artifact planning."""

    plate_path: str
    pipeline_source: str
    axis_filter: tuple[str, ...] = ()
    global_config_id: str | None = None

    @classmethod
    def from_fields(
        cls,
        *,
        plate_path: str,
        pipeline_source: str,
        axis_filter: list[str] | None = None,
        well_filter: list[str] | None = None,
        global_config_id: str | None = None,
    ) -> "PipelineSourceArtifactPlanInspectionRequest":
        selected_axis_filter = axis_filter if axis_filter is not None else well_filter
        return cls(
            plate_path=plate_path,
            pipeline_source=pipeline_source,
            axis_filter=tuple(selected_axis_filter or ()),
            global_config_id=global_config_id,
        )


@dataclass(frozen=True, slots=True)
class OrchestratorSession(OrchestratorSessionIdentity):
    schema_version: str
    uri: str
    plate_path: str
    execution_plate_path: str
    pipeline_id: str
    selected_pipeline_path: str | None = None
    global_config_id: str | None = None
    pipeline_config_id: str | None = None
    connection: ExecutionConnectionSpec = field(default_factory=ExecutionConnectionSpec)
    status: str = "ready"


@dataclass(frozen=True, slots=True)
class ExecutionJobIdentity(OrchestratorSessionIdentity):
    job_id: str
    kind: str
    uri: str
    server_execution_id: str | None


@dataclass(frozen=True, slots=True)
class ExecutionJobRef(ExecutionJobIdentity):
    schema_version: str
    status: str


@dataclass(frozen=True, slots=True)
class CompileSubmissionRequest(OrchestratorSessionIdentity):
    wait: bool = False
    submit_timeout_ms: int = OPENHCS_ZMQ_CONFIG.control_timeout_ms
    wait_timeout_ms: int = OPENHCS_ZMQ_CONFIG.control_timeout_ms


@dataclass(frozen=True, slots=True)
class PipelineExecutionSubmissionRequest(OrchestratorSessionIdentity):
    compile_artifact_id: str | None = None
    wait: bool = False
    submit_timeout_ms: int = OPENHCS_ZMQ_CONFIG.control_timeout_ms
    wait_timeout_ms: int = OPENHCS_ZMQ_CONFIG.control_timeout_ms


@dataclass(frozen=True, slots=True)
class ExecutionStatusRequest:
    job_id: str
    timeout_ms: int = OPENHCS_ZMQ_CONFIG.control_timeout_ms


@dataclass(frozen=True, slots=True)
class ExecutionJobStatus(ExecutionJobIdentity, AgentResultEnvelope):
    status: str
    response: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ArtifactMaterializationPathSummary:
    group_key: str | None
    shared_output_stem: str
    candidate_paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ArtifactMaterializationPlanSummary:
    persistent_enabled: bool
    persistent_backend: str | None = None
    analysis_output_dir: str | None = None
    paths: tuple[ArtifactMaterializationPathSummary, ...] = ()
    runtime_resolved: bool = False
    disabled: bool = False
    filename_uses_source_identity: bool = False
    runtime_metadata_can_refine_paths: bool = False
    note: str | None = None


@dataclass(frozen=True, slots=True)
class ArtifactStoragePlanSummary:
    name: str
    kind: str
    path: str
    group_keys: tuple[str | None, ...] = ()
    paths_by_group: tuple[JsonObject, ...] = ()


@dataclass(frozen=True, slots=True)
class ArtifactInputPlanSummary(ArtifactStoragePlanSummary):
    source_step_id: int | str | None = None
    source_step_scope_id: str | None = None


@dataclass(frozen=True, slots=True)
class ArtifactPlanSummary(ArtifactStoragePlanSummary):
    materialization: ArtifactMaterializationPlanSummary | None = None


@dataclass(frozen=True, slots=True)
class MainFlowMaterializationPlanSummary:
    """Compiled persistent checkpoint for a step's ordinary main-flow result."""

    output_dir: str
    backend: str
    plate_root: str
    sub_dir: str
    analysis_results_dir: str | None = None


@dataclass(frozen=True, slots=True)
class ViewerStreamingPlanSummary:
    """One enabled, compile-resolved viewer config attached to a step plan."""

    config_key: str
    viewer_type: str
    backend: str
    effective_config: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CompiledStepPlanSummary:
    step_index: int
    step_name: str
    axis_id: str
    output_dir: str | None
    main_flow_axis_persistence_enabled: bool | None = None
    execution_groups: tuple[str | None, ...] = ()
    main_flow_materialization: MainFlowMaterializationPlanSummary | None = None
    viewer_streaming: tuple[ViewerStreamingPlanSummary, ...] = ()
    artifact_inputs: tuple[ArtifactInputPlanSummary, ...] = ()
    artifact_outputs: tuple[ArtifactPlanSummary, ...] = ()
    truncated_artifact_input_count: int = 0
    truncated_artifact_output_count: int = 0


@dataclass(frozen=True, slots=True)
class SourceWorkspaceFileRecord:
    """One source image exposed through the OpenHCS virtual workspace."""

    virtual_path: str
    full_virtual_path: str
    source_path: str | None = None
    source_metadata: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SourceWorkspaceSummary:
    """Bounded source-workspace inventory visible to compiled execution."""

    file_count: int = 0
    files: tuple[SourceWorkspaceFileRecord, ...] = ()
    truncated_file_count: int = 0
    axis_file_counts: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactPlanInspection(AgentResultEnvelope):
    schema_version: str
    plate_path: str
    axis_filter: tuple[str, ...] = ()
    axis_count: int = 0
    axes: tuple[str, ...] = ()
    truncated_axis_count: int = 0
    step_count: int = 0
    steps: tuple[CompiledStepPlanSummary, ...] = ()
    truncated_step_count: int = 0
    worker_assignments: JsonObject = field(default_factory=dict)
    source_workspace: SourceWorkspaceSummary = field(
        default_factory=SourceWorkspaceSummary
    )
    progress_event_count: int = 0
    errors: tuple[AgentError, ...] = ()


@dataclass(frozen=True, slots=True)
class RuntimeServerInfo(ExecutionConnectionProjection):
    schema_version: str
    reachable: bool
    ready: bool | None = None
    server: str | None = None
    control_port: int | None = None
    active_executions: int | None = None
    running_executions: tuple[JsonObject, ...] = ()
    queued_executions: tuple[JsonObject, ...] = ()
    workers: tuple[JsonObject, ...] = ()
    uptime: float | None = None
    log_file_path: str | None = None
    response: JsonObject = field(default_factory=dict)
    errors: tuple[AgentError, ...] = ()

    @classmethod
    def from_response(
        cls,
        *,
        connection: ExecutionConnectionSpec,
        response: JsonObject,
    ) -> "RuntimeServerInfo":
        fields = RuntimeServerPayload(response)
        resolved_connection = replace(
            connection,
            port=fields.optional_int(
                "port",
                protocol_default=connection.port,
            ),
        )
        return cls(
            schema_version=SCHEMA_VERSION,
            connection=resolved_connection,
            reachable=True,
            ready=fields.optional_bool("ready"),
            server=fields.optional_str("server"),
            control_port=fields.optional_int("control_port"),
            active_executions=fields.optional_int("active_executions"),
            running_executions=fields.json_object_tuple("running_executions"),
            queued_executions=fields.json_object_tuple("queued_executions"),
            workers=fields.json_object_tuple("workers"),
            uptime=fields.optional_float("uptime"),
            log_file_path=fields.optional_str("log_file_path"),
            response=response,
        )


@dataclass(frozen=True, slots=True)
class RuntimeServerScanResult(ExecutionConnectionProjection):
    schema_version: str
    ports: tuple[int, ...]
    timeout_ms: int
    servers: tuple[RuntimeServerInfo, ...]


@dataclass(frozen=True, slots=True)
class RuntimeServerScanRequest(RuntimeServerToolRequest):
    """Scan candidate ports for running OpenHCS execution servers."""

    ports: tuple[int, ...] | None = None
    host: str = "localhost"
    transport_mode: str | None = None
    timeout_ms: int = OPENHCS_ZMQ_CONFIG.server_scan_timeout_ms

    @classmethod
    def agent_cli_factory(cls):
        return cls.from_cli_fields

    @classmethod
    def agent_cli_argument_specs(cls) -> tuple[AgentCliArgumentSpec, ...]:
        return (
            AgentCliArgumentSpec(
                field_name="ports",
                positional=True,
                nargs="*",
                help=(
                    "Ports to scan; pass space-separated values or "
                    "comma-separated groups."
                ),
            ),
            AgentCliArgumentSpec(
                field_name="port_groups",
                flags=("--ports",),
                action="append",
                help=(
                    "Alias for positional ports; may be repeated or comma-separated."
                ),
            ),
        )

    @classmethod
    def from_cli_fields(
        cls,
        *,
        ports: Sequence[str] = (),
        port_groups: Sequence[str] | None = None,
        host: str = "localhost",
        transport_mode: str | None = None,
        timeout_ms: int = OPENHCS_ZMQ_CONFIG.server_scan_timeout_ms,
    ) -> "RuntimeServerScanRequest":
        parsed_ports = cls.parse_port_values((*ports, *(port_groups or ())))
        return cls.from_fields(
            ports=list(parsed_ports) or None,
            host=host,
            transport_mode=transport_mode,
            timeout_ms=timeout_ms,
        )

    @classmethod
    def from_fields(
        cls,
        *,
        ports: list[int] | None = None,
        host: str = "localhost",
        transport_mode: str | None = None,
        timeout_ms: int = OPENHCS_ZMQ_CONFIG.server_scan_timeout_ms,
    ) -> "RuntimeServerScanRequest":
        return cls(
            ports=tuple(ports) if ports is not None else None,
            host=host,
            transport_mode=transport_mode,
            timeout_ms=timeout_ms,
        )

    @staticmethod
    def parse_port_values(values: Sequence[str]) -> tuple[int, ...]:
        ports: list[int] = []
        for value in values:
            for part in value.split(","):
                stripped = part.strip()
                if not stripped:
                    continue
                try:
                    ports.append(int(stripped))
                except ValueError as exc:
                    raise ValueError(
                        "Runtime scan ports must be integers separated by "
                        "spaces or commas."
                    ) from exc
        return tuple(ports)

    def as_tool_arguments(self) -> JsonObject:
        return {
            "ports": list(self.ports) if self.ports is not None else None,
            "host": self.host,
            "transport_mode": self.transport_mode,
            "timeout_ms": self.timeout_ms,
        }


@dataclass(frozen=True, slots=True)
class RuntimeServerInfoRequest(
    RuntimeServerConnectionToolRequest,
    ExecutionConnectionProjection,
):
    """Request a read-only runtime-server snapshot."""

    timeout_ms: int = OPENHCS_ZMQ_CONFIG.server_info_timeout_ms

    @classmethod
    def from_fields(
        cls,
        *,
        host: str = "localhost",
        port: int | None = None,
        transport_mode: str | None = None,
        persistent: bool = True,
        timeout_ms: int | None = OPENHCS_ZMQ_CONFIG.server_info_timeout_ms,
    ) -> "RuntimeServerInfoRequest":
        return cls(
            connection=ExecutionConnectionSpec.from_fields(
                host=host,
                port=port,
                transport_mode=transport_mode,
                persistent=persistent,
            ),
            timeout_ms=(
                OPENHCS_ZMQ_CONFIG.server_info_timeout_ms
                if timeout_ms is None
                else timeout_ms
            ),
        )


@dataclass(frozen=True, slots=True)
class RuntimeExecutionStatus(ExecutionConnectionProjection):
    schema_version: str
    execution_id: str | None
    status: str
    response: JsonObject = field(default_factory=dict)
    errors: tuple[AgentError, ...] = ()


@dataclass(frozen=True, slots=True)
class RuntimeServerExecutionStatusRequest(
    RuntimeServerConnectionToolRequest,
    ExecutionConnectionProjection,
):
    """Request bounded execution status from a runtime server."""

    execution_id: str | None = None
    timeout_ms: int = OPENHCS_ZMQ_CONFIG.control_timeout_ms

    @classmethod
    def agent_cli_factory(cls):
        return cls.from_cli_fields

    @classmethod
    def from_cli_fields(
        cls,
        *,
        execution_id: str | None = None,
        host: str = "localhost",
        port: int | None = None,
        transport_mode: str | None = None,
        persistent: bool = True,
        timeout_ms: int | None = OPENHCS_ZMQ_CONFIG.server_info_timeout_ms,
    ) -> "RuntimeServerExecutionStatusRequest":
        return cls.from_fields(
            execution_id=execution_id,
            host=host,
            port=port,
            transport_mode=transport_mode,
            persistent=persistent,
            timeout_ms=(
                OPENHCS_ZMQ_CONFIG.server_info_timeout_ms
                if timeout_ms is None
                else timeout_ms
            ),
        )

    @classmethod
    def from_fields(
        cls,
        *,
        execution_id: str | None = None,
        host: str = "localhost",
        port: int | None = None,
        transport_mode: str | None = None,
        persistent: bool = True,
        timeout_ms: int | None = OPENHCS_ZMQ_CONFIG.control_timeout_ms,
    ) -> "RuntimeServerExecutionStatusRequest":
        return cls(
            connection=ExecutionConnectionSpec.from_fields(
                host=host,
                port=port,
                transport_mode=transport_mode,
                persistent=persistent,
            ),
            execution_id=execution_id,
            timeout_ms=(
                OPENHCS_ZMQ_CONFIG.control_timeout_ms
                if timeout_ms is None
                else timeout_ms
            ),
        )

    def as_tool_arguments(self) -> JsonObject:
        payload = dict(RuntimeServerConnectionToolRequest.as_tool_arguments(self))
        payload["execution_id"] = self.execution_id
        return payload


@dataclass(frozen=True, slots=True)
class RuntimeDebugInspectionResult(ExecutionConnectionProjection):
    """Renderer-independent values visible in one paused debug worker."""

    schema_version: str
    debug_session_id: str
    view_model: DebugViewModel | None = None
    errors: tuple[AgentError, ...] = ()


@dataclass(frozen=True, slots=True)
class RuntimeDebugInspectionRequest(
    RuntimeServerConnectionToolRequest,
    ExecutionConnectionProjection,
):
    """Request the typed runtime-value view for one paused debug session."""

    debug_session_id: str
    timeout_ms: int = OPENHCS_ZMQ_CONFIG.control_timeout_ms

    @classmethod
    def agent_cli_argument_specs(cls) -> tuple[AgentCliArgumentSpec, ...]:
        return (
            AgentCliArgumentSpec(
                field_name="debug_session_id",
                positional=True,
                help="Exact debug session id reported by the execution server.",
            ),
        )

    @classmethod
    def from_fields(
        cls,
        *,
        debug_session_id: str,
        host: str = "localhost",
        port: int | None = None,
        transport_mode: str | None = None,
        persistent: bool = True,
        timeout_ms: int | None = OPENHCS_ZMQ_CONFIG.control_timeout_ms,
    ) -> "RuntimeDebugInspectionRequest":
        return cls(
            connection=ExecutionConnectionSpec.from_fields(
                host=host,
                port=port,
                transport_mode=transport_mode,
                persistent=persistent,
            ),
            debug_session_id=debug_session_id,
            timeout_ms=(
                OPENHCS_ZMQ_CONFIG.control_timeout_ms
                if timeout_ms is None
                else timeout_ms
            ),
        )

    def as_tool_arguments(self) -> JsonObject:
        payload = dict(RuntimeServerConnectionToolRequest.as_tool_arguments(self))
        payload["debug_session_id"] = self.debug_session_id
        return payload


@dataclass(frozen=True, slots=True)
class RuntimeServerPayload:
    response: JsonObject

    def optional_bool(self, name: str) -> bool | None:
        value = self.response.get(name)
        if isinstance(value, bool):
            return value
        return None

    def optional_float(self, name: str) -> float | None:
        value = self.response.get(name)
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def optional_int(
        self, name: str, *, protocol_default: int | None = None
    ) -> int | None:
        value = self.response.get(name)
        if isinstance(value, bool):
            return protocol_default
        if isinstance(value, int):
            return value
        return protocol_default

    def optional_str(self, name: str) -> str | None:
        value = self.response.get(name)
        if value is None:
            return None
        return str(value)

    def json_object_tuple(self, name: str) -> tuple[JsonObject, ...]:
        value = self.response.get(name)
        if not isinstance(value, list):
            return ()
        return tuple(dict(item) for item in value if isinstance(item, dict))


def execution_status_from_response(
    response: JsonObject,
    *,
    fallback: str,
) -> str:
    execution = response.get("execution")
    if isinstance(execution, dict) and "status" in execution:
        return str(execution["status"])
    if "status" in response:
        return str(response["status"])
    return fallback


def bounded_execution_status_response(response: JsonObject) -> JsonObject:
    bounded = _bounded_execution_status_value(response)
    if isinstance(bounded, Mapping):
        return dict(bounded)
    return {"response": bounded}


def _bounded_execution_status_value(value: JsonValue) -> JsonValue:
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            name = str(key)
            if (
                name == "traceback"
                and isinstance(item, str)
                and len(item) > MAX_EXECUTION_STATUS_TRACEBACK_CHARS
            ):
                result[name] = item[:MAX_EXECUTION_STATUS_TRACEBACK_CHARS]
                result[f"{name}_truncated"] = True
                result[f"{name}_original_chars"] = len(item)
                continue
            result[name] = _bounded_execution_status_value(item)
        return result
    if isinstance(value, tuple):
        return tuple(_bounded_execution_status_value(item) for item in value)
    if isinstance(value, list):
        return [_bounded_execution_status_value(item) for item in value]
    return value


def execution_status_errors(
    response: JsonObject,
    status: str,
) -> tuple[AgentError, ...]:
    if status != "failed":
        return ()
    return (
        AgentError(
            code="execution_failed",
            message=_execution_failure_message(response),
            hint=_execution_failure_hint(response),
        ),
    )


def _execution_failure_hint(response: JsonObject) -> str:
    failure_text = _execution_failure_text(response)
    if (
        "cannot import name" in failure_text
        and "openhcs.processing.custom_functions" in failure_text
    ):
        return (
            "The runtime could not import a custom function. Ensure the MCP/UI "
            "process and ZMQ runtime server share the same XDG_DATA_HOME and "
            "OpenHCS custom_functions directory; if the function was just "
            "persisted, restart or reload the runtime server before retrying. "
            "Long tracebacks are truncated in the MCP status payload."
        )
    return (
        "Inspect response.execution.error and the runtime server log. "
        "Long tracebacks are truncated in the MCP status payload."
    )


def _execution_failure_text(response: JsonObject) -> str:
    execution = response.get("execution")
    if not isinstance(execution, Mapping):
        return ""
    values = (
        execution.get("error"),
        execution.get("traceback"),
    )
    return "\n".join(str(value) for value in values if value is not None)


def execution_status_warnings(response: JsonObject) -> tuple[AgentWarning, ...]:
    orchestrator_code_document_id = (
        PlateManagerOrchestratorCodeDocumentIdentity.require_value()
    )
    warnings: list[AgentWarning] = []
    if response.get("wait_timed_out") is True:
        timeout_ms = response.get("wait_timeout_ms")
        timeout_label = f" within {timeout_ms}ms" if isinstance(timeout_ms, int) else ""
        warnings.append(
            AgentWarning(
                code="execution_wait_timeout",
                message=(
                    "Execution wait timed out before a terminal status was reached"
                    f"{timeout_label}."
                ),
                hint=(
                    "The job is still tracked. Poll openhcs_get_execution_status "
                    "with the returned job_id instead of blocking the submit tool."
                ),
            )
        )

    execution = response.get("execution")
    if isinstance(execution, Mapping):
        results_summary = execution.get("results_summary")
        if (
            isinstance(results_summary, Mapping)
            and results_summary.get("auto_add_output_plate_to_plate_manager") is False
        ):
            warnings.append(
                AgentWarning(
                    code="headless_execution_did_not_update_plate_manager",
                    message=(
                        "This direct execution session completed without adding "
                        "the output plate to the running UI PlateManager."
                    ),
                    hint=(
                        "For user-visible work in an open UI, load plate_paths "
                        f"and pipeline_data through the {orchestrator_code_document_id} "
                        "code document, then run init/compile/run with "
                        "openhcs_ui_selected_plate_workflow."
                    ),
                )
            )
    return tuple(warnings)


def _execution_failure_message(response: JsonObject) -> str:
    execution = response.get("execution")
    if isinstance(execution, Mapping):
        error = execution.get("error")
        if isinstance(error, str) and error:
            return error
    error = response.get("error")
    if isinstance(error, str) and error:
        return error
    return "OpenHCS execution failed."


def unreachable_runtime_server_info(
    *,
    connection: ExecutionConnectionSpec,
    error: AgentError,
) -> RuntimeServerInfo:
    return RuntimeServerInfo(
        schema_version=SCHEMA_VERSION,
        connection=connection,
        reachable=False,
        errors=(error,),
    )


def runtime_execution_status_from_response(
    *,
    connection: ExecutionConnectionSpec,
    execution_id: str | None,
    response: JsonObject,
) -> RuntimeExecutionStatus:
    bounded_response = bounded_execution_status_response(response)
    status = execution_status_from_response(bounded_response, fallback="unknown")
    return RuntimeExecutionStatus(
        schema_version=SCHEMA_VERSION,
        connection=connection,
        execution_id=execution_id,
        status=status,
        response=bounded_response,
        errors=execution_status_errors(bounded_response, status),
    )


def runtime_execution_status_error(
    *,
    connection: ExecutionConnectionSpec,
    execution_id: str | None,
    error: AgentError,
) -> RuntimeExecutionStatus:
    return RuntimeExecutionStatus(
        schema_version=SCHEMA_VERSION,
        connection=connection,
        execution_id=execution_id,
        status="status_error",
        errors=(error,),
    )
