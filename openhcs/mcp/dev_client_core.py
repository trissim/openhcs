"""Shared primitives for the MCP dev client."""

from __future__ import annotations

import argparse
import asyncio
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import inspect
import json
import os
from pathlib import Path
import sys
from typing import ClassVar, Self, TextIO, TypeVar, cast

from metaclass_registry import AutoRegisterMeta

from openhcs.agent.capabilities import (
    AgentCapabilitySpec,
    FullLocalCapabilitySurfaceProfile,
    LocalCapabilitySurfaceProfile,
)
from openhcs.agent.dto.common import JsonObject, JsonValue
from openhcs.agent.dto.execution import PipelineExecutionSubmissionRequest
from openhcs.agent.dto.ui_bridge import (
    UiObjectStateFieldFilter,
    UiSelectedPlateWorkflowKind,
)
from openhcs.agent.ui_bridge_identities import (
    PlateManagerStateSurfaceIdentityDeclaration,
)
from openhcs.serialization.json import to_jsonable
from openhcs.constants.constants import AllComponents, OrchestratorState
from openhcs.core.execution_state import TerminalExecutionStatus
from openhcs.core.plate_file_inventory import PlateFileInventoryQuery
from openhcs.mcp.control_timeout import (
    McpControlTimeoutPolicy,
    McpUiBridgeTimeoutPolicy,
    McpViewerTimeoutPolicy,
)

DEFAULT_CALL_TIMEOUT_SECONDS = 5.0
DEFAULT_REGISTRY_DISCOVERY_TIMEOUT_SECONDS = 30.0
DEFAULT_WORKFLOW_POLL_INTERVAL_SECONDS = 0.5
DEFAULT_WORKFLOW_POLL_TIMEOUT_SECONDS = 30.0
AliasValueT = TypeVar("AliasValueT")
MCP_DEV_TRANSPORT_FAILURE_HINT = (
    "The fresh OpenHCS MCP subprocess did not complete the requested stdio "
    "exchange. The dev client captures a bounded server stderr tail on "
    "transport failures so startup tracebacks remain visible without noisy "
    "successful calls."
)


class McpDevCliUsageError(ValueError):
    """Local command-line validation failure before an MCP call is made."""


def resolve_positional_option_alias(
    positional_value: AliasValueT | None,
    option_value: AliasValueT | None,
    *,
    default: AliasValueT | None,
    value_name: str,
    option_name: str,
) -> AliasValueT | None:
    """Resolve one CLI convenience pair without duplicating request semantics."""
    if (
        positional_value is not None
        and option_value is not None
        and positional_value != option_value
    ):
        raise McpDevCliUsageError(
            f"Cannot pass both positional {value_name} and {option_name} "
            "with different values."
        )
    if option_value is not None:
        return option_value
    if positional_value is not None:
        return positional_value
    return default


class McpDevClientPhase(str, Enum):
    """Named phases for fresh-process MCP development diagnostics."""

    START_SERVER = "start_server"
    INITIALIZE = "initialize"
    LIST_TOOLS = "list_tools"
    CALL_TOOL = "call_tool"
    TEARDOWN = "teardown"


class McpWireMethod(str, Enum):
    """MCP JSON-RPC method names used by the dev-client transport."""

    INITIALIZE = "initialize"
    INITIALIZED = "notifications/initialized"
    LIST_TOOLS = "tools/list"
    CALL_TOOL = "tools/call"


class McpWireContentType(str, Enum):
    """MCP content block types consumed by the dev client."""

    TEXT = "text"


class McpWireProtocolVersion(str, Enum):
    """MCP protocol versions accepted by the local stdio dev client."""

    V_2024_11_05 = "2024-11-05"
    V_2025_03_26 = "2025-03-26"
    V_2025_06_18 = "2025-06-18"
    LATEST = "2025-11-25"

    @classmethod
    def supported_values(cls) -> frozenset[str]:
        return frozenset(version.value for version in cls)


class McpDevJsonRpcError(RuntimeError):
    """JSON-RPC error returned by the fresh MCP subprocess."""

    def __init__(self, method: McpWireMethod, error: Mapping[str, JsonValue]) -> None:
        self.method = method
        self.error = error
        message = error.get("message")
        code = error.get("code")
        super().__init__(f"{method.value} failed with code {code}: {message}")


class McpDevProtocolError(RuntimeError):
    """Malformed or unsupported MCP wire response."""


class WorkflowPollSummaryStatus(str, Enum):
    """Machine-readable selected-workflow polling outcome."""

    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class WorkflowPollSkipReason(str, Enum):
    """Machine-readable reason selected-workflow polling did not start."""

    WORKFLOW_NOT_ACCEPTED = "workflow_not_accepted"
    WORKFLOW_TOOL_ERROR = "workflow_tool_error"


@dataclass(frozen=True, slots=True)
class McpDevToolCall:
    """One MCP tool invocation issued against a fresh stdio server."""

    name: str
    arguments: dict[str, JsonValue]


class McpToolArgumentRecord(ABC):
    """Nominal record that can project to MCP tool arguments."""

    @abstractmethod
    def as_tool_arguments(self) -> dict[str, JsonValue]:
        """Return the sparse JSON object passed to an MCP tool."""


class McpToolArgumentAuthority:
    """Project typed command payloads to MCP tool argument mappings."""

    @staticmethod
    def from_payload(payload: JsonValue) -> dict[str, JsonValue]:
        if not isinstance(payload, dict):
            raise TypeError("MCP tool arguments must project to a JSON object.")
        return payload

    @classmethod
    def from_record(
        cls,
        payload: McpToolArgumentRecord,
    ) -> dict[str, JsonValue]:
        return cls.from_payload(payload.as_tool_arguments())


class McpTimeoutValidatedArguments:
    """Shared CLI-side timeout validation for MCP control argument records."""

    timeout_policy: ClassVar[type[McpControlTimeoutPolicy]]
    timeout_option_name: ClassVar[str] = "--timeout-ms"

    @classmethod
    def validated_timeout_ms(cls, timeout_ms: int | None) -> int | None:
        if timeout_ms is None:
            return None
        try:
            return cls.timeout_policy.resolve(timeout_ms)
        except ValueError as exc:
            raise McpDevCliUsageError(f"{cls.timeout_option_name}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class McpDevServerSpec:
    """Command used to launch the active checkout MCP server."""

    python_executable: str
    module_name: str = "openhcs.mcp"
    surface_profile: LocalCapabilitySurfaceProfile = field(
        default_factory=FullLocalCapabilitySurfaceProfile
    )
    default_environment_keys: ClassVar[tuple[str, ...]] = (
        "HOME",
        "LOGNAME",
        "PATH",
        "SHELL",
        "TERM",
        "USER",
    )
    gui_environment_keys: ClassVar[tuple[str, ...]] = (
        "DISPLAY",
        "XAUTHORITY",
        "WAYLAND_DISPLAY",
        "XDG_RUNTIME_DIR",
        "DBUS_SESSION_BUS_ADDRESS",
        "SESSION_MANAGER",
        "XDG_SESSION_TYPE",
        "XDG_SESSION_DESKTOP",
        "XDG_DATA_HOME",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "DESKTOP_SESSION",
        "QT_QPA_PLATFORM",
        "QT_PLUGIN_PATH",
        "QT_QPA_PLATFORM_PLUGIN_PATH",
    )

    def environment(self) -> dict[str, str]:
        """Environment entries inherited by the fresh MCP subprocess."""
        return {
            key: value
            for key in (
                *self.default_environment_keys,
                *self.gui_environment_keys,
            )
            if (value := os.environ.get(key)) is not None
        }

    def process_args(self) -> tuple[str, ...]:
        return (
            "-m",
            self.module_name,
            "--surface",
            self.surface_profile.name,
        )


@dataclass(frozen=True, slots=True)
class McpDevServerIdentity:
    """JSON-facing identity for the fresh MCP subprocess."""

    command: str
    module: str

    @classmethod
    def from_spec(cls, server_spec: McpDevServerSpec) -> "McpDevServerIdentity":
        return cls(
            command=server_spec.python_executable,
            module=server_spec.module_name,
        )


@dataclass(frozen=True, slots=True)
class McpDevTransportCause:
    """One exception node from an MCP dev-client transport failure."""

    exception_type: str
    message: str

    @classmethod
    def from_exception(cls, exception: BaseException) -> "McpDevTransportCause":
        return cls(
            exception_type=type(exception).__name__,
            message=str(exception),
        )


@dataclass(frozen=True, slots=True)
class McpDevTransportFailure:
    """Structured dev-client transport failure reported as JSON."""

    code: str
    phase: McpDevClientPhase
    exception_type: str
    message: str
    causes: tuple[McpDevTransportCause, ...]
    hint: str
    server_stderr_tail: str | None = None

    @classmethod
    def from_exception(
        cls,
        phase: McpDevClientPhase,
        exception: BaseException,
        *,
        server_stderr_tail: str | None = None,
    ) -> "McpDevTransportFailure":
        causes = tuple(
            McpDevTransportCause.from_exception(cause)
            for cause in _transport_failure_leaf_causes(exception)
        )
        return cls(
            code="mcp_transport_failed",
            phase=phase,
            exception_type=type(exception).__name__,
            message=str(exception),
            causes=causes,
            hint=MCP_DEV_TRANSPORT_FAILURE_HINT,
            server_stderr_tail=server_stderr_tail,
        )


@dataclass(frozen=True, slots=True)
class McpDevToolResult:
    """JSON-facing result for one MCP tool call."""

    tool: str
    mcp_error: bool
    payloads: tuple[JsonValue, ...]

    @classmethod
    def from_payload(
        cls,
        tool_name: str,
        result: Mapping[str, JsonValue],
    ) -> "McpDevToolResult":
        return cls(
            tool=tool_name,
            mcp_error=result.get("isError") is True,
            payloads=_content_payloads(result),
        )

    def has_errors(self) -> bool:
        """Return whether the tool or any structured agent payload failed."""
        return self.mcp_error or any(
            _contains_agent_error(payload) for payload in self.payloads
        )


@dataclass(frozen=True, slots=True)
class WorkflowPollRowState:
    """Typed PlateManager state row subset used by workflow polling."""

    plate_scope_id: str | None
    orchestrator_state: str | None
    initialized: bool | None
    compiled: bool | None
    init_pending: bool | None
    compile_pending: bool | None
    execution_active: bool | None
    terminal_status: str | None
    queue_position: int | None

    @classmethod
    def from_mapping(cls, row: Mapping[str, JsonValue]) -> "WorkflowPollRowState":
        return cls(
            plate_scope_id=optional_str(row.get("plate_scope_id")),
            orchestrator_state=optional_str(row.get("orchestrator_state")),
            initialized=optional_bool(row.get("initialized")),
            compiled=optional_bool(row.get("compiled")),
            init_pending=optional_bool(row.get("init_pending")),
            compile_pending=optional_bool(row.get("compile_pending")),
            execution_active=optional_bool(row.get("execution_active")),
            terminal_status=optional_str(row.get("terminal_status")),
            queue_position=optional_int(row.get("queue_position")),
        )


class WorkflowTerminalStateCriterion(ABC, metaclass=AutoRegisterMeta):
    """Registered terminal-state criterion for one selected workflow kind."""

    __registry__: ClassVar[
        dict[UiSelectedPlateWorkflowKind, type["WorkflowTerminalStateCriterion"]]
    ] = {}
    __registry_key__ = "workflow"
    __skip_if_no_key__ = True

    workflow: ClassVar[UiSelectedPlateWorkflowKind]
    failed_orchestrator_states: ClassVar[tuple[OrchestratorState, ...]] = ()

    @classmethod
    def for_workflow(
        cls,
        workflow: UiSelectedPlateWorkflowKind,
    ) -> "WorkflowTerminalStateCriterion":
        return cls.__registry__[workflow]()

    @abstractmethod
    def terminal_for_row(self, row: WorkflowPollRowState) -> bool:
        """Return whether this workflow has reached its terminal row state."""

    def failed_for_row(self, row: WorkflowPollRowState) -> bool:
        """Return whether this workflow reached a failed terminal row state."""
        terminal_status = terminal_execution_status(row.terminal_status)
        if terminal_status is not None and terminal_status.counts_as_failed:
            return True

        row_orchestrator_state = parse_orchestrator_state(row.orchestrator_state)
        return (
            row_orchestrator_state is not None
            and row_orchestrator_state in self.failed_orchestrator_states
        )


class InitWorkflowTerminalStateCriterion(WorkflowTerminalStateCriterion):
    workflow = UiSelectedPlateWorkflowKind.INIT
    failed_orchestrator_states = (OrchestratorState.INIT_FAILED,)

    def terminal_for_row(self, row: WorkflowPollRowState) -> bool:
        return row.init_pending is False and row.initialized is True


class CompileWorkflowTerminalStateCriterion(WorkflowTerminalStateCriterion):
    workflow = UiSelectedPlateWorkflowKind.COMPILE
    failed_orchestrator_states = (OrchestratorState.COMPILE_FAILED,)

    def terminal_for_row(self, row: WorkflowPollRowState) -> bool:
        return row.compile_pending is False and row.compiled is True


class RunWorkflowTerminalStateCriterion(WorkflowTerminalStateCriterion):
    workflow = UiSelectedPlateWorkflowKind.RUN
    failed_orchestrator_states = (OrchestratorState.EXEC_FAILED,)

    def terminal_for_row(self, row: WorkflowPollRowState) -> bool:
        return (
            row.execution_active is False
            and row.queue_position is None
            and row.terminal_status is not None
        )


@dataclass(frozen=True, slots=True)
class WorkflowStatePollPolicy:
    """Terminal-state policy for one selected PlateManager workflow."""

    criterion: WorkflowTerminalStateCriterion

    @classmethod
    def from_workflow_text(cls, workflow: str) -> "WorkflowStatePollPolicy":
        return cls(
            criterion=WorkflowTerminalStateCriterion.for_workflow(
                UiSelectedPlateWorkflowKind(workflow)
            )
        )

    def terminal_for_row(self, row: WorkflowPollRowState) -> bool:
        return self.criterion.terminal_for_row(row)

    def failed_for_row(self, row: WorkflowPollRowState) -> bool:
        return self.criterion.failed_for_row(row)


@dataclass(frozen=True, slots=True)
class WorkflowPollSummary:
    """Structured selected-workflow polling summary for agent recovery logic."""

    workflow: str | None
    status: WorkflowPollSummaryStatus
    poll_requested: bool
    poll_completed: bool
    poll_count: int
    target_scope_ids: tuple[str, ...]
    skip_reason: WorkflowPollSkipReason | None
    action_status: str | None

    @property
    def mcp_error(self) -> bool:
        return self.status is not WorkflowPollSummaryStatus.COMPLETED

    def as_payload(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "poll_status": self.status.value,
            "poll_requested": self.poll_requested,
            "poll_completed": self.poll_completed,
            "poll_count": self.poll_count,
            "target_scope_ids": list(self.target_scope_ids),
        }
        if self.workflow is not None:
            payload["workflow"] = self.workflow
        if self.skip_reason is not None:
            payload["skip_reason"] = self.skip_reason.value
        if self.action_status is not None:
            payload["action_status"] = self.action_status
        return payload


@dataclass(frozen=True, slots=True)
class WorkflowPollBaseline:
    """State-surface identity captured before dispatching a UI workflow."""

    revision_token: str | None
    object_state_token: int | None

    @classmethod
    def from_result(
        cls,
        result: McpDevToolResult,
    ) -> "WorkflowPollBaseline | None":
        state_payload = state_surface_payload(result)
        if not state_payload:
            return None
        return cls(
            revision_token=optional_str(
                first_payload_mapping(result).get("current_revision_token")
            )
            or optional_str(state_payload.get("current_revision_token")),
            object_state_token=optional_int(state_payload.get("object_state_token")),
        )

    def changed_by(self, result: McpDevToolResult) -> bool:
        state_payload = state_surface_payload(result)
        if not state_payload:
            return False
        revision_token = optional_str(
            first_payload_mapping(result).get("current_revision_token")
        ) or optional_str(state_payload.get("current_revision_token"))
        object_state_token = optional_int(state_payload.get("object_state_token"))
        return (
            revision_token is not None and revision_token != self.revision_token
        ) or (
            object_state_token is not None
            and object_state_token != self.object_state_token
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class McpDevResponse(ABC):
    """Shared JSON-facing envelope for fresh MCP dev-client responses."""

    server: McpDevServerIdentity
    errors: tuple[McpDevTransportFailure, ...] = ()

    @classmethod
    def from_transport_failure(
        cls,
        server_spec: McpDevServerSpec,
        phase: McpDevClientPhase,
        exception: Exception,
        *,
        server_stderr_tail: str | None = None,
    ) -> Self:
        return cls(
            server=McpDevServerIdentity.from_spec(server_spec),
            errors=(
                McpDevTransportFailure.from_exception(
                    phase,
                    exception,
                    server_stderr_tail=server_stderr_tail,
                ),
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class McpDevToolBatchResponse(McpDevResponse):
    """JSON-facing payload for one or more MCP tool calls."""

    results: tuple[McpDevToolResult, ...] = ()

    @classmethod
    def from_results(
        cls,
        server_spec: McpDevServerSpec,
        results: tuple[McpDevToolResult, ...],
    ) -> "McpDevToolBatchResponse":
        return cls(
            server=McpDevServerIdentity.from_spec(server_spec),
            results=results,
        )


@dataclass(frozen=True, slots=True)
class McpDevToolMetadata:
    """JSON-facing metadata for one advertised MCP tool."""

    name: str
    description: str | None
    input_schema: JsonValue
    title: str | None = None
    annotations: JsonValue | None = None
    output_schema: JsonValue | None = None
    meta: JsonValue | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class McpDevToolListResponse(McpDevResponse):
    """JSON-facing payload for current-source MCP tool metadata."""

    tool_count: int = 0
    tools: tuple[McpDevToolMetadata, ...] = ()

    @classmethod
    def from_tools(
        cls,
        server_spec: McpDevServerSpec,
        tools: tuple[McpDevToolMetadata, ...],
    ) -> "McpDevToolListResponse":
        return cls(
            server=McpDevServerIdentity.from_spec(server_spec),
            tool_count=len(tools),
            tools=tools,
        )


@dataclass(frozen=True, slots=True)
class ViewerConnectionArguments(McpTimeoutValidatedArguments, McpToolArgumentRecord):
    """Typed connection arguments for viewer control tools."""

    timeout_policy = McpViewerTimeoutPolicy

    port: int
    host: str
    transport_mode: str | None
    timeout_ms: int | None

    @classmethod
    def from_args(
        cls,
        args: argparse.Namespace,
        *,
        allow_positional_value_after_port_option: bool = False,
    ) -> "ViewerConnectionArguments":
        return cls(
            port=viewer_port_argument(
                args,
                allow_positional_value_after_port_option=(
                    allow_positional_value_after_port_option
                ),
            ),
            host=args.host,
            transport_mode=args.transport_mode,
            timeout_ms=cls.validated_timeout_ms(args.timeout_ms),
        )

    def as_tool_arguments(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "port": self.port,
            "host": self.host,
        }
        if self.transport_mode is not None:
            payload["transport_mode"] = self.transport_mode
        if self.timeout_ms is not None:
            payload["timeout_ms"] = self.timeout_ms
        return payload


@dataclass(frozen=True, slots=True)
class UiConnectionArguments(McpTimeoutValidatedArguments, McpToolArgumentRecord):
    """Typed connection arguments for UI bridge tools."""

    timeout_policy = McpUiBridgeTimeoutPolicy

    host: str | None
    port: int | None
    transport_mode: str | None
    auth_token: str | None
    descriptor_file_path: str | None
    bridge_instance_id: str | None
    timeout_ms: int | None

    @classmethod
    def from_args(
        cls,
        args: argparse.Namespace,
        *,
        timeout_ms: int | None,
    ) -> "UiConnectionArguments":
        return cls(
            host=args.host,
            port=args.port,
            transport_mode=args.transport_mode,
            auth_token=args.auth_token,
            descriptor_file_path=args.descriptor_file_path,
            bridge_instance_id=args.bridge_instance_id,
            timeout_ms=cls.validated_timeout_ms(timeout_ms),
        )

    def as_tool_arguments(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {}
        if self.host is not None:
            payload["host"] = self.host
        if self.port is not None:
            payload["port"] = self.port
        if self.transport_mode is not None:
            payload["transport_mode"] = self.transport_mode
        if self.auth_token is not None:
            payload["auth_token"] = self.auth_token
        if self.descriptor_file_path is not None:
            payload["descriptor_file_path"] = self.descriptor_file_path
        if self.bridge_instance_id is not None:
            payload["bridge_instance_id"] = self.bridge_instance_id
        if self.timeout_ms is not None:
            payload["timeout_ms"] = self.timeout_ms
        return payload


@dataclass(frozen=True, slots=True)
class UiToolArguments(McpToolArgumentRecord):
    """Typed top-level arguments for UI bridge tools that only need connection."""

    connection: UiConnectionArguments

    def as_tool_arguments(self) -> dict[str, JsonValue]:
        return {"connection": self.connection.as_tool_arguments()}


def parse_json_object(argument_text: str) -> dict[str, JsonValue]:
    """Parse a JSON object for MCP tool arguments."""
    value = cast(JsonValue, json.loads(argument_text))
    if not isinstance(value, dict):
        raise ValueError("MCP tool arguments must be a JSON object.")
    return value


def request_field_parameter(
    request_type: type,
    field_name: str,
) -> inspect.Parameter:
    """Return one DTO from_fields parameter used by CLI projection."""
    return request_factory_parameter(request_type.from_fields, field_name)


def request_factory_parameter(
    request_factory,
    field_name: str,
) -> inspect.Parameter:
    """Return one DTO factory parameter used by CLI projection."""
    return inspect.signature(request_factory).parameters[field_name]


def request_factory_argument_type(
    request_factory,
    field_name: str,
) -> type | None:
    """Return a primitive argparse type from a DTO factory annotation."""
    annotation = request_factory_parameter(request_factory, field_name).annotation
    if annotation in (str, int, float):
        return annotation
    if isinstance(annotation, str):
        if annotation in {"str", "str | None"}:
            return None
        if annotation in {"int", "int | None"}:
            return int
        if annotation in {"float", "float | None"}:
            return float
    return None


def request_field_argument_type(
    request_type: type,
    field_name: str,
) -> type | None:
    """Return a primitive argparse type from a DTO from_fields annotation."""
    return request_factory_argument_type(request_type.from_fields, field_name)


def request_field_bool_default(request_type: type, field_name: str) -> bool:
    """Return a bool argparse default from a DTO from_fields parameter."""
    default = request_field_parameter(request_type, field_name).default
    if type(default) is not bool:
        raise TypeError(f"{request_type.__qualname__}.{field_name} is not a bool")
    return default


def request_field_int_default(request_type: type, field_name: str) -> int:
    """Return an int argparse default from a DTO from_fields parameter."""
    default = request_field_parameter(request_type, field_name).default
    if type(default) is not int:
        raise TypeError(f"{request_type.__qualname__}.{field_name} is not an int")
    return default


def request_field_string_default(request_type: type, field_name: str) -> str:
    """Return a string argparse default from a DTO from_fields parameter."""
    default = request_field_parameter(request_type, field_name).default
    if not isinstance(default, str):
        raise TypeError(f"{request_type.__qualname__}.{field_name} is not a string")
    return default


def plate_file_stream_kind_argument(
    request_type: type,
    requested_kind: str | None,
    file_paths: Sequence[str],
) -> str:
    """Resolve the CLI stream-kind default through the request DTO."""
    if requested_kind is not None:
        return requested_kind
    if file_paths:
        return PlateFileInventoryQuery.ALL_KIND_VALUE
    return request_field_string_default(request_type, "kind")


def add_request_field_option(
    parser: argparse.ArgumentParser,
    request_type: type,
    field_name: str,
    *flags: str,
    **kwargs,
) -> None:
    """Add a CLI option whose default/type comes from a request DTO field."""
    parameter = request_field_parameter(request_type, field_name)
    if "default" not in kwargs and parameter.default is not inspect.Parameter.empty:
        kwargs["default"] = parameter.default
    if "type" not in kwargs and "action" not in kwargs:
        argument_type = request_field_argument_type(request_type, field_name)
        if argument_type is not None:
            kwargs["type"] = argument_type
    parser.add_argument(*flags, **kwargs)


def add_request_factory_option(
    parser: argparse.ArgumentParser,
    request_factory,
    field_name: str,
    *flags: str,
    **kwargs,
) -> None:
    """Add a CLI option whose default/type comes from a request factory field."""
    parameter = request_factory_parameter(request_factory, field_name)
    if "dest" not in kwargs and flags and all(flag.startswith("-") for flag in flags):
        kwargs["dest"] = field_name
    if "default" not in kwargs and parameter.default is not inspect.Parameter.empty:
        kwargs["default"] = parameter.default
    if "type" not in kwargs and "action" not in kwargs:
        argument_type = request_factory_argument_type(request_factory, field_name)
        if argument_type is not None:
            kwargs["type"] = argument_type
    parser.add_argument(*flags, **kwargs)


def parse_cli_json_value(argument_text: str) -> JsonValue:
    """Parse a CLI scalar/container value, falling back to a string literal."""
    try:
        return cast(JsonValue, json.loads(argument_text))
    except json.JSONDecodeError:
        return argument_text


def parse_optional_json_object(
    argument_text: str | None,
) -> dict[str, JsonValue] | None:
    if argument_text is None:
        return None
    return parse_json_object(argument_text)


def optional_bool(value: JsonValue) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def optional_int(value: JsonValue) -> int | None:
    if type(value) is int:
        return value
    return None


def optional_str(value: JsonValue) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _payload_from_text(text: str) -> JsonValue:
    try:
        return cast(JsonValue, json.loads(text))
    except json.JSONDecodeError:
        return {"text": text}


def _content_payloads(result: Mapping[str, JsonValue]) -> tuple[JsonValue, ...]:
    structured_content = result.get("structuredContent")
    if isinstance(structured_content, Mapping):
        return (cast(JsonValue, structured_content),)
    payloads: list[JsonValue] = []
    content_blocks = result.get("content")
    if not isinstance(content_blocks, list):
        raise McpDevProtocolError("MCP tool result did not contain a content list.")
    for content in content_blocks:
        if not isinstance(content, Mapping):
            raise McpDevProtocolError(
                "OpenHCS MCP dev client only supports object content blocks; "
                f"received {type(content).__name__}."
            )
        if content.get("type") != McpWireContentType.TEXT.value:
            raise RuntimeError(
                "OpenHCS MCP dev client only supports text tool responses; "
                f"received {content.get('type')!r}."
            )
        text = content.get("text")
        if not isinstance(text, str):
            raise McpDevProtocolError("MCP text content block did not contain text.")
        payloads.append(_payload_from_text(text))
    return tuple(payloads)


async def call_mcp_tool(
    session: "McpDevStdioSession",
    call: McpDevToolCall,
    timeout_seconds: float,
) -> McpDevToolResult:
    result = await session.call_tool(
        call.name,
        call.arguments,
        timeout_seconds=timeout_seconds,
    )
    return McpDevToolResult.from_payload(call.name, result)


def require_json_object_payload(value: JsonValue) -> JsonObject:
    """Require a top-level JSON object after shared agent serialization."""
    if not isinstance(value, Mapping):
        raise TypeError(
            "MCP dev-client responses must serialize to a JSON object; "
            f"received {type(value).__name__}."
        )
    return cast(JsonObject, value)


def _contains_agent_error(value: JsonValue) -> bool:
    if isinstance(value, Mapping):
        errors = value.get("errors")
        if isinstance(errors, list) and len(errors) > 0:
            return True
        return any(_contains_agent_error(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_agent_error(child) for child in value)
    return False


def _command_failed(payload: JsonObject) -> bool:
    if _contains_agent_error(payload):
        return True
    results = payload.get("results")
    if isinstance(results, list):
        for result in results:
            if isinstance(result, dict):
                if result.get("mcp_error") is True:
                    return True
                if _contains_agent_error(result):
                    return True
    return False


def _transport_failure_leaf_causes(
    exception: BaseException,
) -> tuple[BaseException, ...]:
    if isinstance(exception, BaseExceptionGroup):
        return tuple(
            leaf
            for child in exception.exceptions
            for leaf in _transport_failure_leaf_causes(child)
        )
    return (exception,)


class McpDevStdioSession:
    """Minimal MCP JSON-RPC stdio transport reusable across dev-client commands."""

    stdout_buffer_limit_bytes: ClassVar[int] = 8 * 1024 * 1024

    def __init__(self, server_spec: McpDevServerSpec, server_stderr: TextIO) -> None:
        self.server_spec = server_spec
        self.server_stderr = server_stderr
        self.process: asyncio.subprocess.Process | None = None
        self.request_id = 0

    async def __aenter__(self) -> "McpDevStdioSession":
        self.process = await asyncio.create_subprocess_exec(
            self.server_spec.python_executable,
            *self.server_spec.process_args(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=self.server_stderr,
            env=self.server_spec.environment(),
            limit=self.stdout_buffer_limit_bytes,
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback,
    ) -> None:
        del exc_type, exc_value, traceback
        process = self.require_process()
        if process.stdin is not None:
            process.stdin.close()
            try:
                await process.stdin.wait_closed()
            except BrokenPipeError:
                pass
        if process.returncode is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()

    def require_process(self) -> asyncio.subprocess.Process:
        if self.process is None:
            raise McpDevProtocolError("MCP stdio process was not started.")
        return self.process

    def next_request_id(self) -> int:
        self.request_id += 1
        return self.request_id

    async def initialize(self, *, timeout_seconds: float) -> None:
        result = await self.request(
            McpWireMethod.INITIALIZE,
            {
                "protocolVersion": McpWireProtocolVersion.LATEST.value,
                "capabilities": {},
                "clientInfo": {
                    "name": "openhcs-mcp-dev-client",
                    "version": "0.1.0",
                },
            },
            timeout_seconds=timeout_seconds,
        )
        protocol_version = result.get("protocolVersion")
        if (
            not isinstance(protocol_version, str)
            or protocol_version not in McpWireProtocolVersion.supported_values()
        ):
            raise McpDevProtocolError(
                f"Unsupported MCP protocol version: {protocol_version!r}"
            )
        await self.notification(McpWireMethod.INITIALIZED)

    async def list_tools(
        self,
        *,
        timeout_seconds: float,
    ) -> tuple[Mapping[str, JsonValue], ...]:
        result = await self.request(
            McpWireMethod.LIST_TOOLS,
            None,
            timeout_seconds=timeout_seconds,
        )
        tools = result.get("tools")
        if not isinstance(tools, list):
            raise McpDevProtocolError("MCP tools/list response did not contain tools.")
        tool_records: list[Mapping[str, JsonValue]] = []
        for tool in tools:
            if not isinstance(tool, Mapping):
                raise McpDevProtocolError(
                    "MCP tools/list response contained a non-object tool."
                )
            tool_records.append(tool)
        return tuple(tool_records)

    async def call_tool(
        self,
        name: str,
        arguments: Mapping[str, JsonValue],
        *,
        timeout_seconds: float,
    ) -> Mapping[str, JsonValue]:
        result = await self.request(
            McpWireMethod.CALL_TOOL,
            {
                "name": name,
                "arguments": arguments,
            },
            timeout_seconds=timeout_seconds,
        )
        return result

    async def request(
        self,
        method: McpWireMethod,
        params: Mapping[str, JsonValue] | None,
        *,
        timeout_seconds: float,
    ) -> Mapping[str, JsonValue]:
        request_id = self.next_request_id()
        message: dict[str, JsonValue] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method.value,
        }
        if params is not None:
            message["params"] = dict(params)
        await self.write_message(message)
        while True:
            response = await self.read_message(timeout_seconds=timeout_seconds)
            if response.get("id") != request_id:
                continue
            error = response.get("error")
            if isinstance(error, Mapping):
                raise McpDevJsonRpcError(method, error)
            result = response.get("result")
            if not isinstance(result, Mapping):
                raise McpDevProtocolError(
                    f"MCP {method.value} response did not contain an object result."
                )
            return result

    async def notification(self, method: McpWireMethod) -> None:
        await self.write_message(
            {
                "jsonrpc": "2.0",
                "method": method.value,
            }
        )

    async def write_message(self, message: Mapping[str, JsonValue]) -> None:
        process = self.require_process()
        if process.stdin is None:
            raise McpDevProtocolError("MCP subprocess stdin is unavailable.")
        payload = json.dumps(message, separators=(",", ":")).encode("utf-8") + b"\n"
        process.stdin.write(payload)
        await process.stdin.drain()

    async def read_message(self, *, timeout_seconds: float) -> Mapping[str, JsonValue]:
        process = self.require_process()
        if process.stdout is None:
            raise McpDevProtocolError("MCP subprocess stdout is unavailable.")
        line = await asyncio.wait_for(
            process.stdout.readline(),
            timeout=timeout_seconds,
        )
        if not line:
            raise McpDevProtocolError("MCP subprocess closed stdout.")
        try:
            message = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise McpDevProtocolError("MCP subprocess emitted invalid JSON.") from exc
        if not isinstance(message, Mapping):
            raise McpDevProtocolError("MCP subprocess emitted a non-object message.")
        return cast(Mapping[str, JsonValue], message)


def captured_server_stderr_tail(
    server_stderr: TextIO,
    max_chars: int = 12_000,
) -> str | None:
    """Return a bounded stderr tail from a captured MCP subprocess stream."""
    try:
        server_stderr.flush()
        server_stderr.seek(0)
        stderr_text = server_stderr.read()
    except Exception:
        return None
    if not stderr_text:
        return None
    return stderr_text[-max_chars:]


def mcp_tool_metadata_from_wire(
    tool: Mapping[str, JsonValue],
) -> McpDevToolMetadata:
    """Project one MCP tools/list record into the dev-client response DTO."""
    name = tool.get("name")
    if not isinstance(name, str):
        raise McpDevProtocolError("MCP tool metadata did not contain a string name.")
    description = tool.get("description")
    if description is not None and not isinstance(description, str):
        raise McpDevProtocolError(
            "MCP tool metadata description was neither a string nor null."
        )
    input_schema = tool.get("inputSchema")
    if not isinstance(input_schema, Mapping):
        raise McpDevProtocolError(
            "MCP tool metadata did not contain an object inputSchema."
        )
    title = tool.get("title")
    if title is not None and not isinstance(title, str):
        raise McpDevProtocolError("MCP tool metadata title was not a string or null.")
    annotations = tool.get("annotations")
    if annotations is not None and not isinstance(annotations, Mapping):
        raise McpDevProtocolError("MCP tool annotations were not an object or null.")
    output_schema = tool.get("outputSchema")
    if output_schema is not None and not isinstance(output_schema, Mapping):
        raise McpDevProtocolError("MCP tool outputSchema was not an object or null.")
    meta = tool.get("_meta")
    if meta is not None and not isinstance(meta, Mapping):
        raise McpDevProtocolError("MCP tool _meta was not an object or null.")
    return McpDevToolMetadata(
        name=name,
        description=description,
        input_schema=cast(JsonValue, input_schema),
        title=title,
        annotations=cast(JsonValue, annotations),
        output_schema=cast(JsonValue, output_schema),
        meta=cast(JsonValue, meta),
    )


async def call_mcp_session(
    session: McpDevStdioSession,
    calls: Sequence[McpDevToolCall],
    *,
    timeout_seconds: float,
) -> McpDevToolBatchResponse:
    """Issue tool calls through an initialized MCP session."""
    results: list[McpDevToolResult] = []
    for call in calls:
        results.append(await call_mcp_tool(session, call, timeout_seconds))
    return McpDevToolBatchResponse.from_results(
        session.server_spec,
        tuple(results),
    )


def first_mapping_payload(result: McpDevToolResult) -> Mapping[str, JsonValue] | None:
    if not result.payloads:
        return None
    first_payload = result.payloads[0]
    if isinstance(first_payload, Mapping):
        return first_payload
    return None


def execute_source_session_tool_arguments(
    args: argparse.Namespace,
) -> dict[str, JsonValue]:
    return McpToolArgumentAuthority.from_payload(
        {
            "plate_path": args.plate_path,
            "pipeline_source": pipeline_source_from_args(args),
            "global_config_id": args.global_config_id,
            "host": args.host,
            "port": args.port,
            "transport_mode": args.transport_mode,
            "persistent": args.persistent,
        }
    )


def execute_source_submit_tool_arguments(
    args: argparse.Namespace,
    *,
    session_id: str,
) -> dict[str, JsonValue]:
    return McpToolArgumentAuthority.from_payload(
        to_jsonable(
            PipelineExecutionSubmissionRequest(
                session_id=session_id,
                wait=args.wait,
                submit_timeout_ms=args.submit_timeout_ms,
                wait_timeout_ms=args.wait_timeout_ms,
            )
        )
    )


def execute_source_submit_timeout_seconds(
    args: argparse.Namespace,
    *,
    timeout_seconds: float,
) -> float:
    if not args.wait:
        return timeout_seconds
    tool_timeout = (args.submit_timeout_ms + args.wait_timeout_ms) / 1000.0
    return max(timeout_seconds, tool_timeout + 5.0)


async def list_mcp_session_tools(
    session: McpDevStdioSession,
    *,
    timeout_seconds: float,
) -> McpDevToolListResponse:
    """Return registered tools from an initialized MCP session."""
    result = await session.list_tools(timeout_seconds=timeout_seconds)
    return McpDevToolListResponse.from_tools(
        session.server_spec,
        tuple(mcp_tool_metadata_from_wire(tool) for tool in result),
    )


def viewer_connection_arguments(
    args: argparse.Namespace,
    *,
    allow_positional_value_after_port_option: bool = False,
) -> dict[str, JsonValue]:
    return McpToolArgumentAuthority.from_record(
        ViewerConnectionArguments.from_args(
            args,
            allow_positional_value_after_port_option=(
                allow_positional_value_after_port_option
            ),
        )
    )


def parse_axis_indices(value: str | None) -> tuple[int, ...] | dict[str, int] | None:
    if value is None:
        return None
    if not value.strip():
        return ()
    parts = tuple(
        part.strip() for part in value.replace("/", ",").split(",") if part.strip()
    )
    if any("=" in part for part in parts):
        if not all("=" in part for part in parts):
            raise McpDevCliUsageError(
                "Viewer axis indices must be all positional integers or all NAME=INDEX assignments."
            )
        return parse_navigation_axis_indices(parts)
    try:
        return tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise McpDevCliUsageError(
            "Viewer axis indices must be comma-separated integers."
        ) from exc


def axis_indices_tool_argument(
    value: str | None,
    axis_index_values: Sequence[str] | None = None,
) -> list[int] | dict[str, int] | None:
    parsed = parse_axis_indices(value)
    semantic_indices = parse_navigation_axis_indices(axis_index_values)
    if isinstance(parsed, tuple) and semantic_indices:
        raise McpDevCliUsageError(
            "Cannot combine positional --axis-indices with semantic --axis-index assignments."
        )
    if isinstance(parsed, dict):
        parsed = {**parsed, **semantic_indices}
    elif parsed is None and semantic_indices:
        parsed = semantic_indices
    if parsed is None:
        return None
    if isinstance(parsed, dict):
        return parsed
    return list(parsed)


def axis_indices_wire_argument(
    value: str | None,
    axis_index_values: Sequence[str] | None = None,
) -> tuple[int, ...] | dict[str, int] | None:
    parsed = parse_axis_indices(value)
    semantic_indices = parse_navigation_axis_indices(axis_index_values)
    if isinstance(parsed, tuple) and semantic_indices:
        raise McpDevCliUsageError(
            "Cannot combine positional --axis-indices with semantic --axis-index assignments."
        )
    if isinstance(parsed, dict):
        return {**parsed, **semantic_indices}
    if semantic_indices:
        return semantic_indices
    return parsed


def optional_route_key_argument(
    positional_route_key: str | None,
    option_route_key: str | None,
) -> str | None:
    if (
        positional_route_key
        and option_route_key
        and positional_route_key != option_route_key
    ):
        raise McpDevCliUsageError(
            "Cannot pass both positional route_key and --route-key with different values."
        )
    return option_route_key or positional_route_key


def parse_required_axis_labels(values: Sequence[str] | None) -> list[str]:
    if not values:
        return []
    labels: list[str] = []
    seen: set[str] = set()
    for value in values:
        for label in value.replace("/", ",").split(","):
            normalized = label.strip()
            if normalized and normalized not in seen:
                labels.append(normalized)
                seen.add(normalized)
    return labels


def extend_required_component_labels(
    values: Sequence[str] | None,
    *,
    require_all_components: bool,
) -> list[str]:
    labels = parse_required_axis_labels(values)
    if not require_all_components:
        return labels
    seen = set(labels)
    for component in AllComponents:
        label = component.value
        if label not in seen:
            labels.append(label)
            seen.add(label)
    return labels


def parse_navigation_axis_indices(values: Sequence[str] | None) -> dict[str, int]:
    if not values:
        return {}
    axis_indices: dict[str, int] = {}
    for value in values:
        assignments = tuple(
            assignment.strip()
            for assignment in value.replace("/", ",").split(",")
            if assignment.strip()
        )
        for assignment in assignments:
            axis_name, separator, index_text = assignment.partition("=")
            if not separator or not axis_name.strip():
                raise McpDevCliUsageError(
                    "Viewer navigation axis indices must use NAME=INDEX, "
                    f"received {value!r}."
                )
            try:
                axis_indices[axis_name.strip()] = int(index_text.strip())
            except ValueError as exc:
                raise McpDevCliUsageError(
                    "Viewer navigation axis indices must use integer INDEX values, "
                    f"received {value!r}."
                ) from exc
    return axis_indices


def add_viewer_connection_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--transport-mode")
    parser.add_argument("--timeout-ms", type=int)


def add_viewer_port_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("port", nargs="?")
    parser.add_argument(
        "--port",
        dest="port_option",
        type=int,
        help="Viewer control port; alias for the positional port argument.",
    )


def _optional_int_text(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def viewer_positional_value_after_port_option(args: argparse.Namespace) -> str | None:
    positional_value = args.port
    if args.port_option is None or positional_value is None:
        return None
    if _optional_int_text(positional_value) is not None:
        return None
    return positional_value


def viewer_port_argument(
    args: argparse.Namespace,
    *,
    allow_positional_value_after_port_option: bool = False,
) -> int:
    positional_port = args.port
    option_port = args.port_option
    if positional_port is None and option_port is None:
        raise McpDevCliUsageError("Viewer command requires a port argument or --port.")
    positional_port_value = _optional_int_text(positional_port)
    if positional_port is not None and positional_port_value is None:
        if allow_positional_value_after_port_option and option_port is not None:
            return option_port
        raise McpDevCliUsageError("Viewer positional port must be an integer.")
    if (
        positional_port_value is not None
        and option_port is not None
        and positional_port_value != option_port
    ):
        raise McpDevCliUsageError(
            "Viewer positional port and --port must match when both are set."
        )
    return option_port if option_port is not None else positional_port_value


def viewer_route_key_argument(
    args: argparse.Namespace,
    positional_route_key: str | None,
    option_route_key: str | None,
) -> str | None:
    port_alias_route_key = viewer_positional_value_after_port_option(args)
    return optional_route_key_argument(
        port_alias_route_key or positional_route_key,
        option_route_key,
    )


def required_viewer_route_key_argument(
    args: argparse.Namespace,
    positional_route_key: str | None,
    option_route_key: str | None,
) -> str:
    route_key = viewer_route_key_argument(args, positional_route_key, option_route_key)
    if route_key is None:
        raise McpDevCliUsageError(
            "Viewer command requires a route key argument or --route-key."
        )
    return route_key


def viewer_visible_route_keys_argument(args: argparse.Namespace) -> list[str]:
    route_keys = list(args.visible_route_keys)
    port_alias_route_key = viewer_positional_value_after_port_option(args)
    if port_alias_route_key is not None:
        route_keys.insert(0, port_alias_route_key)
    if not route_keys:
        raise McpDevCliUsageError(
            "Viewer isolation requires at least one visible route key."
        )
    return route_keys


def add_runtime_connection_options(
    parser: argparse.ArgumentParser,
    *,
    include_port: bool,
) -> None:
    if include_port:
        parser.add_argument("port", type=int)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--transport-mode")
    persistence = parser.add_mutually_exclusive_group()
    persistence.add_argument(
        "--persistent",
        dest="persistent",
        action="store_true",
        default=True,
    )
    persistence.add_argument(
        "--non-persistent",
        dest="persistent",
        action="store_false",
    )
    parser.add_argument("--timeout-ms", type=int)


def add_ui_connection_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--transport-mode")
    parser.add_argument("--auth-token")
    parser.add_argument("--descriptor-file-path")
    parser.add_argument("--bridge-instance-id")
    parser.add_argument("--timeout-ms", type=int)


def ui_connection_arguments(
    args: argparse.Namespace,
    *,
    timeout_ms: int | None,
) -> dict[str, JsonValue]:
    return McpToolArgumentAuthority.from_record(
        UiConnectionArguments.from_args(args, timeout_ms=timeout_ms)
    )


def ui_tool_arguments(
    args: argparse.Namespace,
    *,
    timeout_ms: int | None,
) -> dict[str, JsonValue]:
    return McpToolArgumentAuthority.from_record(
        UiToolArguments(
            connection=UiConnectionArguments.from_args(
                args,
                timeout_ms=timeout_ms,
            )
        )
    )


def add_code_document_source_options(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--source-file",
        help="Read code-document source from this path, or '-' for stdin.",
    )
    source.add_argument(
        "--source-text",
        help="Inline code-document source text.",
    )


def code_document_source_from_args(args: argparse.Namespace) -> str:
    if args.source_text is not None:
        return args.source_text
    if args.source_file == "-":
        return sys.stdin.read()
    return Path(args.source_file).read_text(encoding="utf-8")


def add_object_state_field_filter_options(parser: argparse.ArgumentParser) -> None:
    filter_group = parser.add_mutually_exclusive_group()
    parser.set_defaults(field_filter=UiObjectStateFieldFilter.ALL.value)
    filter_group.add_argument(
        "--field-filter",
        choices=tuple(field_filter.value for field_filter in UiObjectStateFieldFilter),
        help="Return fields matching one ObjectState semantic filter.",
    )
    filter_group.add_argument(
        "--dirty-only",
        dest="field_filter",
        action="store_const",
        const=UiObjectStateFieldFilter.DIRTY.value,
        help="Only return unsaved ObjectState fields marked with * semantics.",
    )
    filter_group.add_argument(
        "--changed-only",
        "--semantic-only",
        dest="field_filter",
        action="store_const",
        const=UiObjectStateFieldFilter.SEMANTIC.value,
        help=(
            "Only return dirty, default-diff, inherited, or raw/resolved-none "
            "semantic fields."
        ),
    )
    filter_group.add_argument(
        "--default-diff-only",
        dest="field_filter",
        action="store_const",
        const=UiObjectStateFieldFilter.DEFAULT_DIFF.value,
        help="Only return fields that differ from constructor/default code mode.",
    )
    filter_group.add_argument(
        "--inherited-only",
        dest="field_filter",
        action="store_const",
        const=UiObjectStateFieldFilter.INHERITED.value,
        help="Only return inherited/resolved ObjectState fields.",
    )
    filter_group.add_argument(
        "--raw-resolved-only",
        dest="field_filter",
        action="store_const",
        const=UiObjectStateFieldFilter.RAW_RESOLVED.value,
        help="Only return fields whose raw None/resolved None state differs.",
    )


def add_pipeline_source_options(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--source-file",
        help="Read pycodified pipeline source from this path, or '-' for stdin.",
    )
    source.add_argument(
        "--source-text",
        help="Inline pycodified pipeline source text.",
    )


def pipeline_source_from_args(args: argparse.Namespace) -> str:
    if args.source_text is not None:
        return args.source_text
    if args.source_file == "-":
        return sys.stdin.read()
    return Path(args.source_file).read_text(encoding="utf-8")


def selected_workflow_tool_arguments(
    args: argparse.Namespace,
) -> dict[str, JsonValue]:
    return {
        "workflow": args.workflow,
        "require_confirmation": args.require_confirmation,
        "connection": ui_connection_arguments(
            args,
            timeout_ms=args.timeout_ms,
        ),
    }


def plate_manager_state_surface_tool_arguments(
    args: argparse.Namespace,
    *,
    selection_mode: str,
) -> dict[str, JsonValue]:
    return {
        "surface_id": PlateManagerStateSurfaceIdentityDeclaration.value,
        "selection_mode": selection_mode,
        "connection": ui_connection_arguments(
            args,
            timeout_ms=args.timeout_ms,
        ),
    }


def workflow_result_was_accepted(result: McpDevToolResult) -> bool:
    return workflow_result_action_status(result) == "accepted"


def workflow_result_action_status(result: McpDevToolResult) -> str | None:
    payload = first_payload_mapping(result)
    action_result = nested_mapping(payload, "action_result")
    return optional_str(action_result.get("status"))


def workflow_poll_skip_reason(result: McpDevToolResult) -> WorkflowPollSkipReason:
    if result.mcp_error:
        return WorkflowPollSkipReason.WORKFLOW_TOOL_ERROR
    return WorkflowPollSkipReason.WORKFLOW_NOT_ACCEPTED


def workflow_result_target_scope_ids(result: McpDevToolResult) -> tuple[str, ...]:
    payload = first_payload_mapping(result)
    action_result = nested_mapping(payload, "action_result")
    target_scope_ids = action_result.get("target_scope_ids")
    if not isinstance(target_scope_ids, list):
        return ()
    return tuple(scope_id for scope_id in target_scope_ids if isinstance(scope_id, str))


def first_payload_mapping(result: McpDevToolResult) -> Mapping[str, JsonValue]:
    if not result.payloads:
        return {}
    payload = result.payloads[0]
    if not isinstance(payload, Mapping):
        return {}
    return payload


def nested_mapping(
    payload: Mapping[str, JsonValue],
    key: str,
) -> Mapping[str, JsonValue]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        return {}
    return value


def state_surface_payload(result: McpDevToolResult) -> Mapping[str, JsonValue]:
    return nested_mapping(first_payload_mapping(result), "payload")


def state_surface_rows(result: McpDevToolResult) -> tuple[WorkflowPollRowState, ...]:
    state_payload = state_surface_payload(result)
    rows = state_payload.get("rows")
    if not isinstance(rows, list):
        return ()
    return tuple(
        WorkflowPollRowState.from_mapping(row)
        for row in rows
        if isinstance(row, Mapping)
    )


def workflow_poll_has_reached_terminal_state(
    result: McpDevToolResult,
    *,
    target_scope_ids: tuple[str, ...],
    policy: WorkflowStatePollPolicy,
) -> bool:
    rows = workflow_poll_target_rows(
        result,
        target_scope_ids=target_scope_ids,
    )
    if not rows:
        return False

    return all(policy.terminal_for_row(row) for row in rows)


def workflow_poll_terminal_status(
    result: McpDevToolResult,
    *,
    target_scope_ids: tuple[str, ...],
    policy: WorkflowStatePollPolicy,
) -> WorkflowPollSummaryStatus | None:
    rows = workflow_poll_target_rows(
        result,
        target_scope_ids=target_scope_ids,
    )
    if not rows:
        return None
    if any(policy.failed_for_row(row) for row in rows):
        return WorkflowPollSummaryStatus.FAILED
    if all(policy.terminal_for_row(row) for row in rows):
        return WorkflowPollSummaryStatus.COMPLETED
    return None


def workflow_poll_target_rows(
    result: McpDevToolResult,
    *,
    target_scope_ids: tuple[str, ...],
) -> tuple[WorkflowPollRowState, ...]:
    rows = state_surface_rows(result)
    if not rows:
        return ()
    if not target_scope_ids:
        return rows
    return tuple(row for row in rows if row.plate_scope_id in target_scope_ids)


def terminal_execution_status(
    value: str | None,
) -> TerminalExecutionStatus | None:
    if value is None:
        return None
    try:
        return TerminalExecutionStatus(value)
    except ValueError:
        return None


def parse_orchestrator_state(
    value: str | None,
) -> OrchestratorState | None:
    if value is None:
        return None
    try:
        return OrchestratorState(value)
    except ValueError:
        return None


def workflow_poll_summary_result(
    *,
    workflow: str | None = None,
    status: WorkflowPollSummaryStatus | None = None,
    poll_requested: bool,
    poll_completed: bool,
    poll_count: int,
    target_scope_ids: tuple[str, ...] = (),
    skip_reason: WorkflowPollSkipReason | None = None,
    action_status: str | None = None,
) -> McpDevToolResult:
    if status is None:
        status = (
            WorkflowPollSummaryStatus.COMPLETED
            if poll_completed
            else WorkflowPollSummaryStatus.TIMEOUT
        )
    summary = WorkflowPollSummary(
        workflow=workflow,
        status=status,
        poll_requested=poll_requested,
        poll_completed=poll_completed,
        poll_count=poll_count,
        target_scope_ids=target_scope_ids,
        skip_reason=skip_reason,
        action_status=action_status,
    )
    return McpDevToolResult(
        tool="mcp_dev_selected_workflow_poll",
        mcp_error=summary.mcp_error,
        payloads=(summary.as_payload(),),
    )


def mcp_dev_command_key(
    name: str,
    command_spec_type: type,
) -> str | None:
    """Return the declared CLI command key for a dev-client command spec."""
    del name
    declared_command = vars(command_spec_type).get("command")
    if isinstance(declared_command, str):
        return declared_command
    declared_capability = vars(command_spec_type).get("capability")
    if isinstance(declared_capability, AgentCapabilitySpec):
        return declared_capability.cli_command
    return None
