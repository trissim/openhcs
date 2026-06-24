"""Execution session DTOs for the OpenHCS agent API."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from openhcs.agent.dto.common import (
    AgentError,
    AgentResultEnvelope,
    JsonObject,
    SCHEMA_VERSION,
)


@dataclass(frozen=True, slots=True)
class ExecutionConnectionSpec:
    host: str = "localhost"
    port: int | None = None
    transport_mode: str | None = None
    persistent: bool = True

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


@dataclass(frozen=True, kw_only=True)
class OrchestratorSessionIdentity:
    session_id: str


@dataclass(frozen=True, slots=True)
class OrchestratorSessionRef(OrchestratorSessionIdentity):
    schema_version: str
    uri: str


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
class ExecutionJobStatus(ExecutionJobIdentity, AgentResultEnvelope):
    status: str
    response: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ArtifactPlanSummary:
    name: str
    kind: str
    path: str
    group_keys: tuple[str | None, ...] = ()
    paths_by_group: tuple[JsonObject, ...] = ()


@dataclass(frozen=True, slots=True)
class CompiledStepPlanSummary:
    step_index: int
    step_name: str
    axis_id: str
    output_dir: str | None
    execution_groups: tuple[str | None, ...] = ()
    artifact_outputs: tuple[ArtifactPlanSummary, ...] = ()
    truncated_artifact_output_count: int = 0


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
class RuntimeExecutionStatus(ExecutionConnectionProjection):
    schema_version: str
    execution_id: str | None
    status: str
    response: JsonObject = field(default_factory=dict)
    errors: tuple[AgentError, ...] = ()


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

    def optional_int(self, name: str, *, protocol_default: int | None = None) -> int | None:
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
    return RuntimeExecutionStatus(
        schema_version=SCHEMA_VERSION,
        connection=connection,
        execution_id=execution_id,
        status=execution_status_from_response(response, fallback="unknown"),
        response=response,
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
