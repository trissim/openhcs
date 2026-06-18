"""Execution session DTOs for the OpenHCS agent API."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import ClassVar

from openhcs.agent.dto.common import (
    AgentError,
    AgentResultEnvelope,
    JsonObject,
    JsonValue,
    SCHEMA_VERSION,
)


@dataclass(frozen=True, slots=True)
class ExecutionConnectionSpec:
    host: str = "localhost"
    port: int | None = None
    transport_mode: str | None = None
    persistent: bool = True

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
            port=fields.field("port").as_optional_int(
                protocol_default=connection.port,
            ),
        )
        return cls(
            schema_version=SCHEMA_VERSION,
            connection=resolved_connection,
            reachable=True,
            ready=fields.field("ready").as_optional_bool(),
            server=fields.field("server").as_optional_str(),
            control_port=fields.field("control_port").as_optional_int(),
            active_executions=fields.field("active_executions").as_optional_int(),
            running_executions=fields.field("running_executions").as_json_object_tuple(),
            queued_executions=fields.field("queued_executions").as_json_object_tuple(),
            workers=fields.field("workers").as_json_object_tuple(),
            uptime=fields.field("uptime").as_optional_float(),
            log_file_path=fields.field("log_file_path").as_optional_str(),
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
class RuntimeServerPayloadField:
    value: JsonValue

    def as_optional_bool(self) -> bool | None:
        if isinstance(self.value, bool):
            return self.value
        return None

    def as_optional_float(self) -> float | None:
        if isinstance(self.value, (int, float)):
            return float(self.value)
        return None

    def as_optional_int(self, *, protocol_default: int | None = None) -> int | None:
        if isinstance(self.value, bool):
            return protocol_default
        if isinstance(self.value, int):
            return self.value
        return protocol_default

    def as_optional_str(self) -> str | None:
        if self.value is None:
            return None
        return str(self.value)

    def as_json_object_tuple(self) -> tuple[JsonObject, ...]:
        if not isinstance(self.value, list):
            return ()
        return tuple(dict(item) for item in self.value if isinstance(item, dict))


@dataclass(frozen=True, slots=True)
class RuntimeServerPayload:
    response: JsonObject

    def field(self, name: str) -> RuntimeServerPayloadField:
        if name in self.response:
            return RuntimeServerPayloadField(self.response[name])
        return RuntimeServerPayloadField(None)


@dataclass(frozen=True, slots=True)
class RuntimeExecutionStatusResponsePayload:
    MISSING_STATUS: ClassVar[str] = "unknown"

    status: str
    raw: JsonObject

    @classmethod
    def from_response(
        cls,
        response: JsonObject,
    ) -> "RuntimeExecutionStatusResponsePayload":
        if "status" in response:
            return cls(status=str(response["status"]), raw=response)
        return cls(status=cls.MISSING_STATUS, raw=response)


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
    payload = RuntimeExecutionStatusResponsePayload.from_response(response)
    return RuntimeExecutionStatus(
        schema_version=SCHEMA_VERSION,
        connection=connection,
        execution_id=execution_id,
        status=payload.status,
        response=payload.raw,
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
