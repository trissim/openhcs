"""MCP server adapter for the OpenHCS agent API."""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Mapping
from abc import ABC, abstractmethod
from dataclasses import MISSING, dataclass, fields as dataclass_fields, is_dataclass, replace
from functools import wraps
from inspect import Parameter, Signature, getsourcefile, signature as inspect_signature
from pathlib import Path
from typing import ClassVar, Generic, Self, TypeVar, get_type_hints

import openhcs as openhcs_package
from openhcs.agent.knowledge_manifest import knowledge_base_source_paths_from_manifest
from metaclass_registry import AutoRegisterMeta
from openhcs.agent.capabilities import (
    AgentCapabilityDeclaration,
    AgentCapabilitySpec,
    AgentConfigPatchServiceInvocation,
    AgentConnectionScalarServiceInvocation,
    AgentConnectionServiceInvocation,
    AgentConnectionRequestServiceInvocation,
    AgentDataclassRequestServiceInvocation,
    AgentFromFieldsServiceInvocation,
    CapabilityKind,
    AgentScalarInputContract,
    AgentScalarServiceInvocation,
    AgentViewerWindowRequestServiceInvocation,
    CapabilityUiBridgeTimeoutProfile,
    CapabilityViewerControlTimeoutProfile,
    agent_capabilities,
    agent_capability_declarations,
    get_agent_capability_declaration,
    get_capability_registry,
    require_agent_type_contract,
)
from openhcs.agent.dto.common import (
    AgentError,
    JsonValue,
    SCHEMA_VERSION,
)
from openhcs.agent.dto.config import ConfigPatch
from openhcs.agent.dto.execution import (
    ExecutionConnectionSpec,
)
from openhcs.agent.dto.mcp import McpServerHealthResult
from openhcs.agent.exceptions import AgentFacingErrorMixin
from openhcs.agent.dto.ui_bridge import (
    UiBridgeConnectionRequest,
    UiBridgeConnectionSpec,
    UiWidgetTreeRequest,
)
from openhcs.agent.dto.viewer import (
    ViewerWindowControlRequest,
    ViewerWindowNavigationRequest,
    ViewerWindowPayloadRequest,
    ViewerWindowStateRequest,
    ViewerWindowValidationPolicy,
    ViewerWindowValidationRequest,
)
from openhcs.agent.serialization import to_jsonable
from openhcs.mcp.context import (
    OpenHCSAgentContext,
    create_agent_context,
)
from openhcs.mcp.control_timeout import (
    McpControlTimeoutPolicy,
    McpUiBridgeCommandTimeoutPolicy,
    McpUiBridgeTimeoutPolicy,
    McpViewerCommandTimeoutPolicy,
    McpViewerTimeoutPolicy,
)
from openhcs.runtime.viewer_protocol import (
    ViewerNavigationControlOptions,
    ViewerPayloadControlOptions,
    ViewerStateControlOptions,
)

RequestT = TypeVar("RequestT")


@dataclass(frozen=True, slots=True)
class McpSourceSnapshot:
    exists: bool
    mtime_ns: int | None

    @classmethod
    def from_path(cls, source_path: Path) -> "McpSourceSnapshot":
        try:
            stat_result = source_path.stat()
        except FileNotFoundError:
            return cls(exists=False, mtime_ns=None)
        return cls(
            exists=True,
            mtime_ns=stat_result.st_mtime_ns,
        )


def _source_path_for_type(source_type: type) -> Path:
    source_file = getsourcefile(source_type)
    if source_file is None:
        raise RuntimeError(f"No source file available for {source_type.__qualname__}")
    return Path(source_file).resolve()


def _deduplicate_source_paths(source_paths: tuple[Path, ...]) -> tuple[Path, ...]:
    return tuple(dict.fromkeys(source_paths))


def _package_python_source_paths(package) -> tuple[Path, ...]:
    return tuple(
        sorted(
            path.resolve()
            for location in package.__path__
            for path in Path(location).rglob("*.py")
        )
    )


MCP_SERVER_SOURCE_PATHS = _deduplicate_source_paths(
    (
        Path(__file__).resolve(),
        Path(create_agent_context.__code__.co_filename).resolve(),
        Path(get_capability_registry.__code__.co_filename).resolve(),
        Path(to_jsonable.__code__.co_filename).resolve(),
        *_package_python_source_paths(openhcs_package),
        *knowledge_base_source_paths_from_manifest(),
    )
)
MCP_SERVER_IMPORT_SOURCE_SNAPSHOTS = {
    source_path: McpSourceSnapshot.from_path(source_path)
    for source_path in MCP_SERVER_SOURCE_PATHS
}
MCP_SERVER_IMPORT_SOURCE_MTIMES_NS = {
    source_path: snapshot.mtime_ns
    for source_path, snapshot in MCP_SERVER_IMPORT_SOURCE_SNAPSHOTS.items()
    if snapshot.mtime_ns is not None
}
MCP_SERVER_SOURCE_PATH = MCP_SERVER_SOURCE_PATHS[0]


def _required_mcp_source_mtime_ns(source_path: Path) -> int:
    snapshot = MCP_SERVER_IMPORT_SOURCE_SNAPSHOTS[source_path]
    if snapshot.mtime_ns is None:
        raise RuntimeError(f"MCP source path is missing at import: {source_path}")
    return snapshot.mtime_ns


MCP_SERVER_IMPORT_MTIME_NS = _required_mcp_source_mtime_ns(MCP_SERVER_SOURCE_PATH)
MCP_SERVER_PROCESS_ID = os.getpid()
MCP_SERVER_IMPORTED_AT_UNIX = time.time()
MCP_SERVER_RESTART_HINT = (
    "Restart the MCP client/server process so it imports the current OpenHCS "
    "source. For the local stdio server, relaunch with restart_command."
)


@dataclass(frozen=True, slots=True)
class McpWidgetTreePayloadProjection:
    """MCP-facing projection for noisy widget-tree action rows."""

    compact_actions: bool = True

    core_action_fields = frozenset(
        (
            "path",
            "path_id",
            "child_index",
            "class_name",
            "label",
            "visible",
            "enabled",
            "geometry",
            "global_geometry",
            "action_kinds",
            "clickable",
        )
    )
    false_boolean_fields = frozenset(("visible", "enabled", "clickable"))
    value_fields = frozenset(
        (
            "raw_value",
            "resolved_value",
            "raw_value_preview",
            "resolved_value_preview",
        )
    )

    def project(self, result) -> dict:
        payload = to_jsonable(result)
        if not isinstance(payload, Mapping):
            raise TypeError("widget tree serialization did not produce a mapping")
        if not self.compact_actions:
            return dict(payload)
        return self.compact_payload(payload)

    @classmethod
    def compact_payload(cls, payload: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        actions = payload.get("actionable_widgets")
        if not isinstance(actions, list):
            return dict(payload)
        compact = dict(payload)
        compact["actionable_widgets"] = [
            cls.compact_action(action) if isinstance(action, Mapping) else action
            for action in actions
        ]
        return compact

    @classmethod
    def compact_action(cls, action: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        return {
            field: value
            for field, value in action.items()
            if cls.action_field_carries_information(field, value, action)
        }

    @classmethod
    def action_field_carries_information(
        cls,
        field: str,
        value: JsonValue,
        action: Mapping[str, JsonValue],
    ) -> bool:
        if field in cls.core_action_fields:
            return True
        if (
            field in ("raw_value", "resolved_value")
            and action.get(f"{field}_preview") is not None
        ):
            return False
        if field in cls.value_fields:
            return value is not None or action.get(f"{field}_is_none") is True
        if value is None:
            return False
        if value == "" or value == [] or value == {}:
            return False
        if isinstance(value, bool):
            if field in cls.false_boolean_fields:
                return True
            if field == "checked":
                return action.get("checkable") is True
            return value
        return True


@dataclass(frozen=True, slots=True)
class McpUiCatalogPayloadProjection:
    """MCP-facing projection that exposes nested identity ids as flat fields."""

    item_key: str

    def project(self, result) -> dict:
        payload = to_jsonable(result)
        if not isinstance(payload, Mapping):
            raise TypeError("UI catalog serialization did not produce a mapping")
        return self.compact_payload(payload)

    def compact_payload(self, payload: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        items = payload.get(self.item_key)
        if not isinstance(items, list):
            return dict(payload)
        compact = dict(payload)
        compact[self.item_key] = [
            self.compact_item(item) if isinstance(item, Mapping) else item
            for item in items
        ]
        return compact

    @staticmethod
    def compact_item(item: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        compact = dict(item)
        identity = compact.pop("identity", None)
        if not isinstance(identity, Mapping):
            if identity is not None:
                compact["identity"] = identity
            return compact
        for key, value in identity.items():
            if isinstance(key, str) and value is not None:
                compact.setdefault(key, value)
        return compact


def _mcp_server_current_source_mtime_ns() -> int:
    return MCP_SERVER_SOURCE_PATH.stat().st_mtime_ns


def _mcp_server_stale_source_paths() -> tuple[Path, ...]:
    return tuple(
        source_path
        for source_path, import_snapshot in MCP_SERVER_IMPORT_SOURCE_SNAPSHOTS.items()
        if McpSourceSnapshot.from_path(source_path) != import_snapshot
    )


def _mcp_server_source_changed_since_import() -> bool:
    return bool(_mcp_server_stale_source_paths())


def _mcp_server_restart_command() -> tuple[str, ...]:
    return (sys.executable, "-m", "openhcs.mcp")


class McpNoArgumentToolBindingABC(ABC, metaclass=AutoRegisterMeta):
    """Declaration-owned FastMCP binding for no-argument agent tools."""

    __registry__: ClassVar[
        dict[AgentCapabilitySpec, type["McpNoArgumentToolBindingABC"]]
    ] = {}
    __registry_key__ = "capability"
    __skip_if_no_key__ = True

    capability: ClassVar[AgentCapabilitySpec]
    allow_stale_server: ClassVar[bool] = False

    @classmethod
    def execute(cls, ctx: OpenHCSAgentContext) -> dict:
        """Execute the bound capability against the current agent context."""
        return get_agent_capability_declaration(cls.capability.name).execute_no_argument(
            ctx
        )

    @classmethod
    def bind_no_argument_tool(
        cls,
        *,
        capability: AgentCapabilitySpec,
        execute,
        ctx: OpenHCSAgentContext,
        openhcs_tool,
        allow_stale_server: bool = False,
    ) -> None:
        def tool() -> dict:
            return to_jsonable(execute(ctx))

        tool.__name__ = capability.name
        tool.__qualname__ = capability.name
        tool.__doc__ = capability.description
        tool.__annotations__ = {"return": dict}
        tool.__signature__ = Signature(return_annotation=dict)
        openhcs_tool(allow_stale_server=allow_stale_server)(tool)

    @classmethod
    def bind_to_server(
        cls,
        ctx: OpenHCSAgentContext,
        openhcs_tool,
    ) -> None:
        cls.bind_no_argument_tool(
            capability=cls.capability,
            execute=cls.execute,
            ctx=ctx,
            openhcs_tool=openhcs_tool,
            allow_stale_server=cls.allow_stale_server,
        )


class GeneratedMcpNoArgumentToolBinding:
    """Generated FastMCP binding for declaration-owned no-argument tools."""

    @classmethod
    def bind_to_server(
        cls,
        declaration: type[AgentCapabilityDeclaration],
        ctx: OpenHCSAgentContext,
        openhcs_tool,
    ) -> None:
        McpNoArgumentToolBindingABC.bind_no_argument_tool(
            capability=declaration.to_spec(),
            execute=declaration.execute_no_argument,
            ctx=ctx,
            openhcs_tool=openhcs_tool,
        )


def generated_no_argument_capability_declarations() -> tuple[
    type[AgentCapabilityDeclaration],
    ...
]:
    """Return declaration-owned no-argument MCP tools without custom bindings."""
    explicit_capability_names = frozenset(
        capability.name
        for capability in McpNoArgumentToolBindingABC.__registry__
    )
    return tuple(
        declaration
        for declaration in agent_capability_declarations()
        if declaration.name not in explicit_capability_names
        and declaration.kind is CapabilityKind.TOOL
        and declaration.no_argument_invocation is not None
    )


def _mcp_resource_function_name(resource_name: str) -> str:
    return (
        resource_name.replace("://", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace(":", "_")
    )


class GeneratedMcpResourceBinding:
    """Generated FastMCP binding for declaration-owned resources."""

    @classmethod
    def bind_to_server(
        cls,
        declaration: type[AgentCapabilityDeclaration],
        ctx: OpenHCSAgentContext,
        server,
    ) -> None:
        def resource() -> dict:
            if _mcp_server_source_changed_since_import():
                return _mcp_server_stale_error(declaration.name)
            return to_jsonable(declaration.execute_no_argument(ctx))

        resource.__name__ = _mcp_resource_function_name(declaration.name)
        resource.__qualname__ = resource.__name__
        resource.__doc__ = declaration.description
        resource.__annotations__ = {"return": dict}
        resource.__signature__ = Signature(return_annotation=dict)
        server.resource(declaration.name)(resource)


def generated_resource_capability_declarations() -> tuple[
    type[AgentCapabilityDeclaration],
    ...
]:
    """Return declaration-owned MCP resources."""
    return tuple(
        declaration
        for declaration in agent_capability_declarations()
        if declaration.kind is CapabilityKind.RESOURCE
        and declaration.no_argument_invocation is not None
    )


class HealthCheckMcpToolBinding(McpNoArgumentToolBindingABC):
    capability = agent_capabilities.health_check
    allow_stale_server = True

    @classmethod
    def execute(cls, ctx: OpenHCSAgentContext) -> dict:
        del ctx
        current_source_mtime_ns = _mcp_server_current_source_mtime_ns()
        stale_source_paths = _mcp_server_stale_source_paths()
        restart_required = bool(stale_source_paths)
        restart_command: tuple[str, ...] = ()
        restart_hint: str | None = None
        if restart_required:
            restart_command = _mcp_server_restart_command()
            restart_hint = MCP_SERVER_RESTART_HINT
        return McpServerHealthResult(
            schema_version=SCHEMA_VERSION,
            status="ok",
            started_at_unix=MCP_SERVER_IMPORTED_AT_UNIX,
            service="openhcs.mcp",
            server_process_id=MCP_SERVER_PROCESS_ID,
            server_source_path=str(MCP_SERVER_SOURCE_PATH),
            server_import_mtime_ns=MCP_SERVER_IMPORT_MTIME_NS,
            server_current_mtime_ns=current_source_mtime_ns,
            server_source_changed_since_import=restart_required,
            stale_source_paths=tuple(
                str(source_path)
                for source_path in stale_source_paths
            ),
            restart_required=restart_required,
            restart_command=restart_command,
            restart_hint=restart_hint,
        )


class McpUiConnectionToolBindingABC(ABC, metaclass=AutoRegisterMeta):
    """Declaration-owned FastMCP binding for UI tools with only connection input."""

    __registry__: ClassVar[
        dict[AgentCapabilitySpec, type["McpUiConnectionToolBindingABC"]]
    ] = {}
    __registry_key__ = "capability"
    __skip_if_no_key__ = True

    capability: ClassVar[AgentCapabilitySpec]

    @classmethod
    @abstractmethod
    def execute(
        cls,
        ctx: OpenHCSAgentContext,
        connection: UiBridgeConnectionSpec,
    ) -> dict:
        """Execute the bound UI capability against the current agent context."""

    @classmethod
    def bind_to_server(
        cls,
        ctx: OpenHCSAgentContext,
        openhcs_tool,
    ) -> None:
        cls.bind_connection_tool(
            capability=cls.capability,
            execute_connection=lambda context, connection_spec: cls.execute(
                context,
                connection_spec,
            ),
            ctx=ctx,
            openhcs_tool=openhcs_tool,
        )

    @classmethod
    def bind_connection_tool(
        cls,
        *,
        capability: AgentCapabilitySpec,
        execute_connection,
        ctx: OpenHCSAgentContext,
        openhcs_tool,
    ) -> None:
        def tool(connection: McpUiBridgeConnectionRequest | None = None) -> dict:
            connection_spec = UiBridgeConnectionToolArgs.from_mapping(
                connection
            ).resolve(ctx)
            return to_jsonable(execute_connection(ctx, connection_spec))

        tool.__name__ = capability.name
        tool.__qualname__ = capability.name
        tool.__doc__ = capability.description
        tool.__annotations__ = {
            "connection": McpUiBridgeConnectionRequest | None,
            "return": dict,
        }
        tool.__signature__ = Signature(
            parameters=[
                Parameter(
                    "connection",
                    Parameter.POSITIONAL_OR_KEYWORD,
                    default=None,
                    annotation=McpUiBridgeConnectionRequest | None,
                )
            ],
            return_annotation=dict,
        )
        openhcs_tool()(tool)

class GeneratedMcpUiConnectionToolBinding:
    """Generated FastMCP binding for declaration-owned UI connection tools."""

    @classmethod
    def bind_to_server(
        cls,
        declaration: type[AgentCapabilityDeclaration],
        ctx: OpenHCSAgentContext,
        openhcs_tool,
    ) -> None:
        McpUiConnectionToolBindingABC.bind_connection_tool(
            capability=declaration.to_spec(),
            execute_connection=declaration.execute_connection,
            ctx=ctx,
            openhcs_tool=openhcs_tool,
        )


def generated_ui_connection_capability_declarations() -> tuple[
    type[AgentCapabilityDeclaration],
    ...
]:
    """Return declaration-owned UI connection tools without request DTOs."""
    explicit_capability_names = frozenset(
        capability.name
        for capability in McpUiConnectionToolBindingABC.__registry__
    )
    return tuple(
        declaration
        for declaration in agent_capability_declarations()
        if declaration.name not in explicit_capability_names
        and isinstance(
            declaration.connection_invocation,
            AgentConnectionServiceInvocation,
        )
    )


class UiListCodeDocumentsMcpToolBinding(McpUiConnectionToolBindingABC):
    capability = agent_capabilities.ui_list_code_documents

    @classmethod
    def execute(
        cls,
        ctx: OpenHCSAgentContext,
        connection: UiBridgeConnectionSpec,
    ) -> dict:
        return McpUiCatalogPayloadProjection("documents").project(
            ctx.ui_bridge_service.list_documents(connection)
        )


class UiListStateSurfacesMcpToolBinding(McpUiConnectionToolBindingABC):
    capability = agent_capabilities.ui_list_state_surfaces

    @classmethod
    def execute(
        cls,
        ctx: OpenHCSAgentContext,
        connection: UiBridgeConnectionSpec,
    ) -> dict:
        return McpUiCatalogPayloadProjection("surfaces").project(
            ctx.ui_bridge_service.list_state_surfaces(connection)
        )


class UiListActionsMcpToolBinding(McpUiConnectionToolBindingABC):
    capability = agent_capabilities.ui_list_actions

    @classmethod
    def execute(
        cls,
        ctx: OpenHCSAgentContext,
        connection: UiBridgeConnectionSpec,
    ) -> dict:
        return McpUiCatalogPayloadProjection("actions").project(
            ctx.ui_bridge_service.list_actions(connection)
        )


class UiListWindowsMcpToolBinding(McpUiConnectionToolBindingABC):
    capability = agent_capabilities.ui_list_windows

    @classmethod
    def execute(
        cls,
        ctx: OpenHCSAgentContext,
        connection: UiBridgeConnectionSpec,
    ) -> dict:
        return McpUiCatalogPayloadProjection("windows").project(
            ctx.ui_bridge_service.list_windows(connection)
        )


class McpUiRequestToolBindingABC(
    Generic[RequestT],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Declaration-owned FastMCP binding for UI request DTOs."""

    __registry__: ClassVar[
        dict[AgentCapabilitySpec, type["McpUiRequestToolBindingABC"]]
    ] = {}
    __registry_key__ = "capability"
    __skip_if_no_key__ = True

    capability: ClassVar[AgentCapabilitySpec]
    timeout_policy: ClassVar[type[McpControlTimeoutPolicy]] = McpUiBridgeTimeoutPolicy

    @classmethod
    def execute_request(
        cls,
        ctx: OpenHCSAgentContext,
        request: RequestT,
        connection: UiBridgeConnectionSpec,
    ) -> dict:
        """Execute the bound UI operation with its typed request DTO."""
        return get_agent_capability_declaration(
            cls.capability.name
        ).execute_connection_request(
            ctx,
            request,
            connection,
        )

    @classmethod
    def extra_parameters(cls) -> tuple[Parameter, ...]:
        """Return MCP-only parameters that are not part of the UI bridge request."""
        return ()

    @classmethod
    def project_result(
        cls,
        result,
        extra_arguments: Mapping[str, JsonValue],
    ):
        """Project a UI bridge result through MCP-only response options."""
        del extra_arguments
        return result

    @classmethod
    def request_type(cls) -> type[RequestT]:
        return require_agent_type_contract(cls.capability.input_contract)

    @classmethod
    def bind_request_tool(
        cls,
        *,
        capability: AgentCapabilitySpec,
        request_type: type,
        execute_request,
        timeout_policy: type[McpControlTimeoutPolicy],
        ctx: OpenHCSAgentContext,
        openhcs_tool,
        extra_parameters: tuple[Parameter, ...] = (),
        project_result=None,
    ) -> None:
        from_fields = request_type.from_fields
        from_fields_signature = inspect_signature(from_fields)
        parameter_type_hints = get_type_hints(from_fields)
        extra_parameter_names = tuple(parameter.name for parameter in extra_parameters)
        extra_signature = Signature(parameters=extra_parameters)
        connection_parameter = Parameter(
            "connection",
            Parameter.KEYWORD_ONLY,
            default=None,
            annotation=McpUiBridgeConnectionRequest | None,
        )
        request_parameters = tuple(
            parameter.replace(annotation=parameter_type_hints[parameter.name])
            for parameter in from_fields_signature.parameters.values()
        )
        tool_signature = Signature(
            parameters=(
                *request_parameters,
                *extra_parameters,
                connection_parameter,
            ),
            return_annotation=dict,
        )

        def tool(**kwargs: JsonValue) -> dict:
            connection = kwargs.pop("connection", None)
            extra_bound_arguments = extra_signature.bind_partial(
                **{
                    parameter_name: kwargs.pop(parameter_name)
                    for parameter_name in extra_parameter_names
                    if parameter_name in kwargs
                }
            )
            extra_bound_arguments.apply_defaults()
            bound_arguments = from_fields_signature.bind_partial(**kwargs)
            bound_arguments.apply_defaults()
            request = from_fields(**bound_arguments.arguments)
            connection_spec = UiBridgeConnectionToolArgs.from_mapping(
                connection
            ).resolve(ctx, timeout_policy=timeout_policy)
            result = execute_request(ctx, request, connection_spec)
            result_projector = cls.project_result if project_result is None else project_result
            return to_jsonable(
                result_projector(result, extra_bound_arguments.arguments)
            )

        tool.__name__ = capability.name
        tool.__qualname__ = capability.name
        tool.__doc__ = capability.description
        tool.__annotations__ = {
            parameter.name: parameter.annotation
            for parameter in tool_signature.parameters.values()
        } | {"return": dict}
        tool.__signature__ = tool_signature
        openhcs_tool()(tool)

    @classmethod
    def bind_to_server(
        cls,
        ctx: OpenHCSAgentContext,
        openhcs_tool,
    ) -> None:
        cls.bind_request_tool(
            capability=cls.capability,
            request_type=cls.request_type(),
            execute_request=cls.execute_request,
            timeout_policy=cls.timeout_policy,
            ctx=ctx,
            openhcs_tool=openhcs_tool,
            extra_parameters=cls.extra_parameters(),
            project_result=cls.project_result,
        )


class GeneratedMcpUiRequestToolBinding:
    """Generated FastMCP binding for declaration-owned UI request tools."""

    @classmethod
    def bind_to_server(
        cls,
        declaration: type[AgentCapabilityDeclaration],
        ctx: OpenHCSAgentContext,
        openhcs_tool,
    ) -> None:
        request_type = cls.request_type(declaration)
        McpUiRequestToolBindingABC.bind_request_tool(
            capability=declaration.to_spec(),
            request_type=request_type,
            execute_request=declaration.execute_connection_request,
            timeout_policy=cls.timeout_policy(declaration),
            ctx=ctx,
            openhcs_tool=openhcs_tool,
        )

    @staticmethod
    def request_type(
        declaration: type[AgentCapabilityDeclaration],
    ) -> type:
        return require_agent_type_contract(declaration.input_contract)

    @staticmethod
    def timeout_policy(
        declaration: type[AgentCapabilityDeclaration],
    ) -> type[McpControlTimeoutPolicy]:
        invocation = declaration.connection_request_invocation
        if not isinstance(invocation, AgentConnectionRequestServiceInvocation):
            raise TypeError(
                f"{declaration.__name__} requires AgentConnectionRequestServiceInvocation."
            )
        if invocation.timeout_profile is CapabilityUiBridgeTimeoutProfile.COMMAND:
            return McpUiBridgeCommandTimeoutPolicy
        if invocation.timeout_profile is CapabilityUiBridgeTimeoutProfile.DEFAULT:
            return McpUiBridgeTimeoutPolicy
        raise TypeError(
            f"Unsupported UI bridge timeout profile: {invocation.timeout_profile!r}"
        )


def generated_ui_request_capability_declarations() -> tuple[
    type[AgentCapabilityDeclaration],
    ...
]:
    """Return declaration-owned UI request tools."""
    explicit_capability_names = frozenset(
        capability.name
        for capability in McpUiRequestToolBindingABC.__registry__
    )
    return tuple(
        declaration
        for declaration in agent_capability_declarations()
        if declaration.name not in explicit_capability_names
        and isinstance(
            declaration.connection_request_invocation,
            AgentConnectionRequestServiceInvocation,
        )
    )


class UiGetWidgetTreeMcpToolBinding(
    McpUiRequestToolBindingABC[UiWidgetTreeRequest]
):
    capability = agent_capabilities.ui_get_widget_tree

    @classmethod
    def extra_parameters(cls) -> tuple[Parameter, ...]:
        return (
            Parameter(
                "compact_actions",
                Parameter.KEYWORD_ONLY,
                default=True,
                annotation=bool,
            ),
        )

    @classmethod
    def execute_request(
        cls,
        ctx: OpenHCSAgentContext,
        request: UiWidgetTreeRequest,
        connection: UiBridgeConnectionSpec,
    ) -> dict:
        return ctx.ui_bridge_service.widget_tree(request, connection)

    @classmethod
    def project_result(
        cls,
        result,
        extra_arguments: Mapping[str, JsonValue],
    ) -> dict:
        return McpWidgetTreePayloadProjection(
            compact_actions=extra_arguments["compact_actions"],
        ).project(result)


class McpScalarInputToolBindingABC(ABC, metaclass=AutoRegisterMeta):
    """Declaration-owned FastMCP binding for one-string scalar input tools."""

    __registry__: ClassVar[
        dict[AgentCapabilitySpec, type["McpScalarInputToolBindingABC"]]
    ] = {}
    __registry_key__ = "capability"
    __skip_if_no_key__ = True

    capability: ClassVar[AgentCapabilitySpec]

    @classmethod
    @abstractmethod
    def execute(cls, ctx: OpenHCSAgentContext, value: str) -> dict:
        """Execute the bound scalar capability against the current agent context."""

    @classmethod
    def input_contract(cls) -> AgentScalarInputContract:
        contract = cls.capability.input_contract
        if type(contract) is not AgentScalarInputContract:
            raise TypeError(
                f"{cls.__name__} requires AgentScalarInputContract, got {contract!r}."
            )
        return contract

    @classmethod
    def bind_to_server(
        cls,
        ctx: OpenHCSAgentContext,
        openhcs_tool,
    ) -> None:
        cls.bind_scalar_tool(
            capability=cls.capability,
            input_contract=cls.input_contract(),
            execute_scalar=cls.execute,
            ctx=ctx,
            openhcs_tool=openhcs_tool,
        )

    @classmethod
    def bind_scalar_tool(
        cls,
        *,
        capability: AgentCapabilitySpec,
        input_contract: AgentScalarInputContract,
        execute_scalar,
        ctx: OpenHCSAgentContext,
        openhcs_tool,
    ) -> None:
        parameter_name = input_contract.field_name
        parameter_default = (
            Signature.empty
            if input_contract.default_value is None
            else input_contract.default_value
        )

        def tool(**kwargs) -> dict:
            return to_jsonable(execute_scalar(ctx, kwargs[parameter_name]))

        tool.__name__ = capability.name
        tool.__qualname__ = capability.name
        tool.__doc__ = capability.description
        tool.__annotations__ = {parameter_name: str, "return": dict}
        tool.__signature__ = Signature(
            parameters=[
                Parameter(
                    parameter_name,
                    Parameter.KEYWORD_ONLY,
                    default=parameter_default,
                    annotation=str,
                )
            ],
            return_annotation=dict,
        )
        openhcs_tool()(tool)


class GeneratedMcpScalarInputToolBinding:
    """Generated FastMCP binding for declaration-owned scalar tools."""

    @classmethod
    def bind_to_server(
        cls,
        declaration: type[AgentCapabilityDeclaration],
        ctx: OpenHCSAgentContext,
        openhcs_tool,
    ) -> None:
        input_contract = cls.input_contract(declaration)
        McpScalarInputToolBindingABC.bind_scalar_tool(
            capability=declaration.to_spec(),
            input_contract=input_contract,
            execute_scalar=declaration.execute_scalar,
            ctx=ctx,
            openhcs_tool=openhcs_tool,
        )

    @staticmethod
    def input_contract(
        declaration: type[AgentCapabilityDeclaration],
    ) -> AgentScalarInputContract:
        contract = declaration.input_contract
        if type(contract) is not AgentScalarInputContract:
            raise TypeError(
                f"{declaration.__name__} requires AgentScalarInputContract, "
                f"got {contract!r}."
            )
        return contract


def generated_scalar_input_capability_declarations() -> tuple[
    type[AgentCapabilityDeclaration],
    ...
]:
    """Return declaration-owned one-scalar tools."""
    explicit_capability_names = frozenset(
        capability.name
        for capability in McpScalarInputToolBindingABC.__registry__
    )
    return tuple(
        declaration
        for declaration in agent_capability_declarations()
        if declaration.name not in explicit_capability_names
        and isinstance(declaration.scalar_invocation, AgentScalarServiceInvocation)
    )


class McpUiScalarInputToolBindingABC(ABC, metaclass=AutoRegisterMeta):
    """Declaration-owned FastMCP binding for one-scalar UI tools."""

    __registry__: ClassVar[
        dict[AgentCapabilitySpec, type["McpUiScalarInputToolBindingABC"]]
    ] = {}
    __registry_key__ = "capability"
    __skip_if_no_key__ = True

    capability: ClassVar[AgentCapabilitySpec]
    timeout_policy: ClassVar[type[McpControlTimeoutPolicy]] = McpUiBridgeTimeoutPolicy

    @classmethod
    @abstractmethod
    def execute(
        cls,
        ctx: OpenHCSAgentContext,
        value: str,
        connection: UiBridgeConnectionSpec,
    ) -> dict:
        """Execute the bound UI scalar capability."""

    @classmethod
    def input_contract(cls) -> AgentScalarInputContract:
        contract = cls.capability.input_contract
        if type(contract) is not AgentScalarInputContract:
            raise TypeError(
                f"{cls.__name__} requires AgentScalarInputContract, got {contract!r}."
            )
        return contract

    @classmethod
    def bind_to_server(
        cls,
        ctx: OpenHCSAgentContext,
        openhcs_tool,
    ) -> None:
        cls.bind_ui_scalar_tool(
            capability=cls.capability,
            input_contract=cls.input_contract(),
            execute_connection_scalar=cls.execute,
            timeout_policy=cls.timeout_policy,
            ctx=ctx,
            openhcs_tool=openhcs_tool,
        )

    @classmethod
    def bind_ui_scalar_tool(
        cls,
        *,
        capability: AgentCapabilitySpec,
        input_contract: AgentScalarInputContract,
        execute_connection_scalar,
        timeout_policy: type[McpControlTimeoutPolicy],
        ctx: OpenHCSAgentContext,
        openhcs_tool,
    ) -> None:
        parameter_name = input_contract.field_name
        parameter_default = (
            Signature.empty
            if input_contract.default_value is None
            else input_contract.default_value
        )
        connection_parameter = Parameter(
            "connection",
            Parameter.KEYWORD_ONLY,
            default=None,
            annotation=McpUiBridgeConnectionRequest | None,
        )

        def tool(**kwargs) -> dict:
            connection = UiBridgeConnectionToolArgs.from_mapping(
                kwargs.pop("connection", None)
            ).resolve(ctx, timeout_policy=timeout_policy)
            return to_jsonable(
                execute_connection_scalar(ctx, kwargs[parameter_name], connection)
            )

        tool.__name__ = capability.name
        tool.__qualname__ = capability.name
        tool.__doc__ = capability.description
        tool.__annotations__ = {
            parameter_name: str,
            "connection": McpUiBridgeConnectionRequest | None,
            "return": dict,
        }
        tool.__signature__ = Signature(
            parameters=[
                Parameter(
                    parameter_name,
                    Parameter.KEYWORD_ONLY,
                    default=parameter_default,
                    annotation=str,
                ),
                connection_parameter,
            ],
            return_annotation=dict,
        )
        openhcs_tool()(tool)


class GeneratedMcpUiScalarInputToolBinding:
    """Generated FastMCP binding for declaration-owned UI scalar tools."""

    @classmethod
    def bind_to_server(
        cls,
        declaration: type[AgentCapabilityDeclaration],
        ctx: OpenHCSAgentContext,
        openhcs_tool,
    ) -> None:
        input_contract = cls.input_contract(declaration)
        McpUiScalarInputToolBindingABC.bind_ui_scalar_tool(
            capability=declaration.to_spec(),
            input_contract=input_contract,
            execute_connection_scalar=declaration.execute_connection_scalar,
            timeout_policy=McpUiBridgeTimeoutPolicy,
            ctx=ctx,
            openhcs_tool=openhcs_tool,
        )

    @staticmethod
    def input_contract(
        declaration: type[AgentCapabilityDeclaration],
    ) -> AgentScalarInputContract:
        contract = declaration.input_contract
        if type(contract) is not AgentScalarInputContract:
            raise TypeError(
                f"{declaration.__name__} requires AgentScalarInputContract, "
                f"got {contract!r}."
            )
        return contract


def generated_ui_scalar_capability_declarations() -> tuple[
    type[AgentCapabilityDeclaration],
    ...
]:
    """Return declaration-owned UI scalar tools."""
    explicit_capability_names = frozenset(
        capability.name
        for capability in McpUiScalarInputToolBindingABC.__registry__
    )
    return tuple(
        declaration
        for declaration in agent_capability_declarations()
        if declaration.name not in explicit_capability_names
        and isinstance(
            declaration.connection_scalar_invocation,
            AgentConnectionScalarServiceInvocation,
        )
    )


class McpConfigPatchToolBindingABC(ABC, metaclass=AutoRegisterMeta):
    """Declaration-owned FastMCP binding for ConfigPatch-backed tools."""

    __registry__: ClassVar[
        dict[AgentCapabilitySpec, type["McpConfigPatchToolBindingABC"]]
    ] = {}
    __registry_key__ = "capability"
    __skip_if_no_key__ = True

    capability: ClassVar[AgentCapabilitySpec]

    @classmethod
    @abstractmethod
    def execute_patch(cls, ctx: OpenHCSAgentContext, patch: ConfigPatch) -> dict:
        """Execute the bound config operation with a typed patch."""

    @classmethod
    def bind_to_server(
        cls,
        ctx: OpenHCSAgentContext,
        openhcs_tool,
    ) -> None:
        cls.bind_patch_tool(
            capability=cls.capability,
            execute_patch=cls.execute_patch,
            ctx=ctx,
            openhcs_tool=openhcs_tool,
        )

    @classmethod
    def bind_patch_tool(
        cls,
        *,
        capability: AgentCapabilitySpec,
        execute_patch,
        ctx: OpenHCSAgentContext,
        openhcs_tool,
    ) -> None:
        config_type_field, values_field = dataclass_fields(ConfigPatch)
        config_type_name = config_type_field.name
        values_name = values_field.name

        def tool(**kwargs) -> dict:
            patch = ConfigPatch(
                config_type=kwargs[config_type_name],
                values=_json_object_or_empty(kwargs.get(values_name)),
            )
            return to_jsonable(execute_patch(ctx, patch))

        tool.__name__ = capability.name
        tool.__qualname__ = capability.name
        tool.__doc__ = capability.description
        tool.__annotations__ = {
            config_type_name: str,
            values_name: dict | None,
            "return": dict,
        }
        tool.__signature__ = Signature(
            parameters=[
                Parameter(
                    config_type_name,
                    Parameter.KEYWORD_ONLY,
                    annotation=str,
                ),
                Parameter(
                    values_name,
                    Parameter.KEYWORD_ONLY,
                    default=None,
                    annotation=dict | None,
                ),
            ],
            return_annotation=dict,
        )
        openhcs_tool()(tool)


class GeneratedMcpConfigPatchToolBinding:
    """Generated FastMCP binding for declaration-owned ConfigPatch tools."""

    @classmethod
    def bind_to_server(
        cls,
        declaration: type[AgentCapabilityDeclaration],
        ctx: OpenHCSAgentContext,
        openhcs_tool,
    ) -> None:
        McpConfigPatchToolBindingABC.bind_patch_tool(
            capability=declaration.to_spec(),
            execute_patch=declaration.execute_request,
            ctx=ctx,
            openhcs_tool=openhcs_tool,
        )


def generated_config_patch_capability_declarations() -> tuple[
    type[AgentCapabilityDeclaration],
    ...
]:
    """Return declaration-owned ConfigPatch tools."""
    explicit_capability_names = frozenset(
        capability.name
        for capability in McpConfigPatchToolBindingABC.__registry__
    )
    return tuple(
        declaration
        for declaration in agent_capability_declarations()
        if declaration.name not in explicit_capability_names
        and isinstance(
            declaration.request_invocation,
            AgentConfigPatchServiceInvocation,
        )
    )


class McpFromFieldsToolBindingABC(
    Generic[RequestT],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Declaration-owned FastMCP binding for request DTOs with from_fields()."""

    __registry__: ClassVar[
        dict[AgentCapabilitySpec, type["McpFromFieldsToolBindingABC"]]
    ] = {}
    __registry_key__ = "capability"
    __skip_if_no_key__ = True

    capability: ClassVar[AgentCapabilitySpec]

    @classmethod
    def execute_request(cls, ctx: OpenHCSAgentContext, request: RequestT) -> dict:
        """Execute the bound capability with its typed request DTO."""
        return get_agent_capability_declaration(cls.capability.name).execute_request(
            ctx,
            request,
        )

    @classmethod
    def request_type(cls) -> type[RequestT]:
        return require_agent_type_contract(cls.capability.input_contract)

    @classmethod
    def bind_request_tool(
        cls,
        *,
        capability: AgentCapabilitySpec,
        request_type: type,
        execute_request,
        ctx: OpenHCSAgentContext,
        openhcs_tool,
    ) -> None:
        from_fields = request_type.from_fields
        from_fields_signature = inspect_signature(from_fields)
        parameter_type_hints = get_type_hints(from_fields)

        def tool(**kwargs: JsonValue) -> dict:
            bound_arguments = from_fields_signature.bind_partial(**kwargs)
            bound_arguments.apply_defaults()
            request = from_fields(**bound_arguments.arguments)
            return to_jsonable(execute_request(ctx, request))

        tool.__name__ = capability.name
        tool.__qualname__ = capability.name
        tool.__doc__ = capability.description
        tool.__annotations__ = {
            parameter_name: parameter_type
            for parameter_name, parameter_type in parameter_type_hints.items()
            if parameter_name != "return"
        } | {"return": dict}
        tool.__signature__ = Signature(
            parameters=[
                parameter.replace(
                    annotation=parameter_type_hints[parameter.name],
                )
                for parameter in from_fields_signature.parameters.values()
            ],
            return_annotation=dict,
        )
        openhcs_tool()(tool)

    @classmethod
    def bind_to_server(
        cls,
        ctx: OpenHCSAgentContext,
        openhcs_tool,
    ) -> None:
        cls.bind_request_tool(
            capability=cls.capability,
            request_type=cls.request_type(),
            execute_request=cls.execute_request,
            ctx=ctx,
            openhcs_tool=openhcs_tool,
        )


class GeneratedMcpFromFieldsToolBinding:
    """Generated FastMCP binding for declaration-owned from_fields tools."""

    @classmethod
    def bind_to_server(
        cls,
        declaration: type[AgentCapabilityDeclaration],
        ctx: OpenHCSAgentContext,
        openhcs_tool,
    ) -> None:
        request_type = cls.request_type(declaration)
        McpFromFieldsToolBindingABC.bind_request_tool(
            capability=declaration.to_spec(),
            request_type=request_type,
            execute_request=declaration.execute_request,
            ctx=ctx,
            openhcs_tool=openhcs_tool,
        )

    @staticmethod
    def request_type(
        declaration: type[AgentCapabilityDeclaration],
    ) -> type:
        return require_agent_type_contract(declaration.input_contract)


def generated_from_fields_capability_declarations() -> tuple[
    type[AgentCapabilityDeclaration],
    ...
]:
    """Return declaration-owned from_fields MCP tools without custom bindings."""
    explicit_capability_names = frozenset(
        capability.name
        for capability in McpFromFieldsToolBindingABC.__registry__
    )
    return tuple(
        declaration
        for declaration in agent_capability_declarations()
        if declaration.name not in explicit_capability_names
        and isinstance(
            declaration.request_invocation,
            AgentFromFieldsServiceInvocation,
        )
    )


class McpDataclassRequestToolBindingABC(
    Generic[RequestT],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Declaration-owned FastMCP binding for direct scalar dataclass requests."""

    __registry__: ClassVar[
        dict[AgentCapabilitySpec, type["McpDataclassRequestToolBindingABC"]]
    ] = {}
    __registry_key__ = "capability"
    __skip_if_no_key__ = True

    capability: ClassVar[AgentCapabilitySpec]

    @classmethod
    @abstractmethod
    def execute_request(cls, ctx: OpenHCSAgentContext, request: RequestT) -> dict:
        """Execute the bound capability with its typed request DTO."""

    @classmethod
    def request_type(cls) -> type[RequestT]:
        contract = require_agent_type_contract(cls.capability.input_contract)
        if not is_dataclass(contract):
            raise TypeError(
                f"{cls.__name__} requires a dataclass request input contract, got {contract!r}."
            )
        return contract

    @classmethod
    def bind_to_server(
        cls,
        ctx: OpenHCSAgentContext,
        openhcs_tool,
    ) -> None:
        cls.bind_request_tool(
            capability=cls.capability,
            request_type=cls.request_type(),
            execute_request=cls.execute_request,
            ctx=ctx,
            openhcs_tool=openhcs_tool,
        )

    @classmethod
    def bind_request_tool(
        cls,
        *,
        capability: AgentCapabilitySpec,
        request_type: type[RequestT],
        execute_request,
        ctx: OpenHCSAgentContext,
        openhcs_tool,
    ) -> None:
        request_fields = dataclass_fields(request_type)
        request_type_hints = get_type_hints(request_type)
        parameters: list[Parameter] = []
        for request_field in request_fields:
            if request_field.default_factory is not MISSING:
                raise TypeError(
                    f"{cls.__name__} cannot expose default_factory field "
                    f"{request_field.name!r} as a direct MCP parameter."
                )
            default = (
                Signature.empty
                if request_field.default is MISSING
                else request_field.default
            )
            parameters.append(
                Parameter(
                    request_field.name,
                    Parameter.KEYWORD_ONLY,
                    default=default,
                    annotation=request_type_hints[request_field.name],
                )
            )

        request_signature = Signature(
            parameters=parameters,
            return_annotation=dict,
        )

        def tool(**kwargs: JsonValue) -> dict:
            bound_arguments = request_signature.bind_partial(**kwargs)
            bound_arguments.apply_defaults()
            request = request_type(**bound_arguments.arguments)
            return to_jsonable(execute_request(ctx, request))

        tool.__name__ = capability.name
        tool.__qualname__ = capability.name
        tool.__doc__ = capability.description
        tool.__annotations__ = {
            parameter.name: parameter.annotation for parameter in parameters
        } | {"return": dict}
        tool.__signature__ = request_signature
        openhcs_tool()(tool)


class GeneratedMcpDataclassRequestToolBinding:
    """Generated FastMCP binding for declaration-owned dataclass request tools."""

    @classmethod
    def bind_to_server(
        cls,
        declaration: type[AgentCapabilityDeclaration],
        ctx: OpenHCSAgentContext,
        openhcs_tool,
    ) -> None:
        request_type = cls.request_type(declaration)
        McpDataclassRequestToolBindingABC.bind_request_tool(
            capability=declaration.to_spec(),
            request_type=request_type,
            execute_request=declaration.execute_request,
            ctx=ctx,
            openhcs_tool=openhcs_tool,
        )

    @staticmethod
    def request_type(
        declaration: type[AgentCapabilityDeclaration],
    ) -> type:
        contract = require_agent_type_contract(declaration.input_contract)
        if not is_dataclass(contract):
            raise TypeError(
                f"{declaration.__name__} requires a dataclass request input "
                f"contract, got {contract!r}."
            )
        return contract


def generated_dataclass_request_capability_declarations() -> tuple[
    type[AgentCapabilityDeclaration],
    ...
]:
    """Return declaration-owned dataclass request tools."""
    explicit_capability_names = frozenset(
        capability.name
        for capability in McpDataclassRequestToolBindingABC.__registry__
    )
    return tuple(
        declaration
        for declaration in agent_capability_declarations()
        if declaration.name not in explicit_capability_names
        and isinstance(
            declaration.request_invocation,
            AgentDataclassRequestServiceInvocation,
        )
    )


class McpViewerRequestToolBindingABC(ABC, metaclass=AutoRegisterMeta):
    """Generated FastMCP binding for viewer request DTOs."""

    __registry__: ClassVar[
        dict[AgentCapabilitySpec, type["McpViewerRequestToolBindingABC"]]
    ] = {}
    __registry_key__ = "capability"
    __skip_if_no_key__ = True

    capability: ClassVar[AgentCapabilitySpec]

    @classmethod
    @abstractmethod
    def request_signature(cls) -> Signature:
        """Return the public MCP signature for this viewer request."""

    @classmethod
    @abstractmethod
    def request_from_arguments(
        cls,
        arguments: Mapping[str, JsonValue],
    ) -> ViewerWindowControlRequest:
        """Project MCP arguments into one typed viewer request."""

    @classmethod
    @abstractmethod
    def execute_request(
        cls,
        ctx: OpenHCSAgentContext,
        request: ViewerWindowControlRequest,
    ) -> dict:
        """Execute the viewer service operation."""

    @classmethod
    def bind_to_server(
        cls,
        ctx: OpenHCSAgentContext,
        openhcs_tool,
    ) -> None:
        cls.bind_viewer_request_tool(
            capability=cls.capability,
            request_signature=cls.request_signature(),
            request_from_arguments=cls.request_from_arguments,
            execute_request=cls.execute_request,
            ctx=ctx,
            openhcs_tool=openhcs_tool,
        )

    @classmethod
    def bind_viewer_request_tool(
        cls,
        *,
        capability: AgentCapabilitySpec,
        request_signature: Signature,
        request_from_arguments,
        execute_request,
        ctx: OpenHCSAgentContext,
        openhcs_tool,
    ) -> None:
        def tool(**kwargs: JsonValue) -> dict:
            bound_arguments = request_signature.bind_partial(**kwargs)
            bound_arguments.apply_defaults()
            request = request_from_arguments(bound_arguments.arguments)
            return to_jsonable(execute_request(ctx, request))

        tool.__name__ = capability.name
        tool.__qualname__ = capability.name
        tool.__doc__ = capability.description
        tool.__annotations__ = {
            parameter.name: parameter.annotation
            for parameter in request_signature.parameters.values()
        } | {"return": dict}
        tool.__signature__ = request_signature
        openhcs_tool()(tool)

    @classmethod
    def connection_parameters(cls) -> tuple[Parameter, ...]:
        return (
            Parameter(
                "port",
                Parameter.KEYWORD_ONLY,
                annotation=int,
            ),
            Parameter(
                "host",
                Parameter.KEYWORD_ONLY,
                default="localhost",
                annotation=str,
            ),
            Parameter(
                "transport_mode",
                Parameter.KEYWORD_ONLY,
                default=None,
                annotation=str | None,
            ),
            Parameter(
                "timeout_ms",
                Parameter.KEYWORD_ONLY,
                default=None,
                annotation=int | None,
            ),
        )

    @classmethod
    def control_args(
        cls,
        arguments: Mapping[str, JsonValue],
        timeout_policy: type[McpControlTimeoutPolicy] = McpViewerTimeoutPolicy,
    ) -> "McpViewerConnectionToolArgs":
        return McpViewerConnectionToolArgs.from_fields(
            port=arguments["port"],
            host=arguments["host"],
            transport_mode=arguments["transport_mode"],
            timeout_ms=arguments["timeout_ms"],
            timeout_policy=timeout_policy,
        )

    @staticmethod
    def option_parameters(
        factory,
        *,
        default_overrides: Mapping[str, JsonValue] | None = None,
    ) -> tuple[Parameter, ...]:
        factory_signature = inspect_signature(factory)
        factory_type_hints = get_type_hints(factory)
        resolved_default_overrides = default_overrides or {}
        return tuple(
            parameter.replace(
                default=(
                    resolved_default_overrides[parameter.name]
                    if parameter.name in resolved_default_overrides
                    else parameter.default
                ),
                annotation=factory_type_hints[parameter.name],
            )
            for parameter in factory_signature.parameters.values()
        )

    @staticmethod
    def option_arguments(
        arguments: Mapping[str, JsonValue],
        factory,
    ) -> dict[str, JsonValue]:
        factory_signature = inspect_signature(factory)
        return {
            parameter_name: arguments[parameter_name]
            for parameter_name in factory_signature.parameters
        }

    @classmethod
    def request_option_parameters(
        cls,
        request_type: type[ViewerWindowControlRequest],
    ) -> tuple[Parameter, ...]:
        """Return public non-connection parameters from a viewer request DTO."""
        factory = request_type.from_fields
        factory_signature = inspect_signature(factory)
        factory_type_hints = get_type_hints(factory)
        control_fields = cls.viewer_control_field_names()
        return tuple(
            parameter.replace(annotation=factory_type_hints[parameter.name])
            for parameter in factory_signature.parameters.values()
            if parameter.name not in control_fields
        )

    @staticmethod
    def request_option_arguments(
        arguments: Mapping[str, JsonValue],
        request_type: type[ViewerWindowControlRequest],
    ) -> dict[str, JsonValue]:
        factory_signature = inspect_signature(request_type.from_fields)
        control_fields = McpViewerRequestToolBindingABC.viewer_control_field_names()
        return {
            parameter_name: arguments[parameter_name]
            for parameter_name, parameter in factory_signature.parameters.items()
            if parameter_name not in control_fields
        }

    @classmethod
    def request_from_fields_arguments(
        cls,
        request_type: type[ViewerWindowControlRequest],
        arguments: Mapping[str, JsonValue],
        timeout_policy: type[McpControlTimeoutPolicy] = McpViewerTimeoutPolicy,
    ) -> ViewerWindowControlRequest:
        control_args = cls.control_args(arguments, timeout_policy)
        return request_type.from_fields(
            connection=control_args.connection,
            timeout_ms=control_args.timeout_ms,
            **cls.request_option_arguments(arguments, request_type),
        )

    @staticmethod
    def viewer_control_field_names() -> frozenset[str]:
        return ViewerWindowControlRequest.factory_injected_field_names()


class GeneratedMcpViewerRequestToolBinding:
    """Generated FastMCP binding for declaration-owned viewer request tools."""

    @classmethod
    def bind_to_server(
        cls,
        declaration: type[AgentCapabilityDeclaration],
        ctx: OpenHCSAgentContext,
        openhcs_tool,
    ) -> None:
        request_type = cls.request_type(declaration)
        timeout_policy = cls.timeout_policy(declaration)
        request_signature = Signature(
            parameters=(
                *McpViewerRequestToolBindingABC.connection_parameters(),
                *McpViewerRequestToolBindingABC.request_option_parameters(request_type),
            ),
            return_annotation=dict,
        )

        def request_from_arguments(
            arguments: Mapping[str, JsonValue],
        ) -> ViewerWindowControlRequest:
            return McpViewerRequestToolBindingABC.request_from_fields_arguments(
                request_type,
                arguments,
                timeout_policy,
            )

        McpViewerRequestToolBindingABC.bind_viewer_request_tool(
            capability=declaration.to_spec(),
            request_signature=request_signature,
            request_from_arguments=request_from_arguments,
            execute_request=declaration.execute_request,
            ctx=ctx,
            openhcs_tool=openhcs_tool,
        )

    @staticmethod
    def request_type(
        declaration: type[AgentCapabilityDeclaration],
    ) -> type[ViewerWindowControlRequest]:
        contract = require_agent_type_contract(declaration.input_contract)
        if not issubclass(
            contract,
            ViewerWindowControlRequest,
        ):
            raise TypeError(
                f"{declaration.__name__} requires a ViewerWindowControlRequest "
                f"input contract, got {contract!r}."
            )
        return contract

    @staticmethod
    def timeout_policy(
        declaration: type[AgentCapabilityDeclaration],
    ) -> type[McpControlTimeoutPolicy]:
        invocation = declaration.request_invocation
        if not isinstance(invocation, AgentViewerWindowRequestServiceInvocation):
            raise TypeError(
                f"{declaration.__name__} requires AgentViewerWindowRequestServiceInvocation."
            )
        if (
            invocation.timeout_profile
            is CapabilityViewerControlTimeoutProfile.COMMAND
        ):
            return McpViewerCommandTimeoutPolicy
        if (
            invocation.timeout_profile
            is CapabilityViewerControlTimeoutProfile.DEFAULT
        ):
            return McpViewerTimeoutPolicy
        raise TypeError(
            f"Unsupported viewer timeout profile: {invocation.timeout_profile!r}"
        )


def generated_viewer_request_capability_declarations() -> tuple[
    type[AgentCapabilityDeclaration],
    ...
]:
    """Return declaration-owned viewer request tools."""
    explicit_capability_names = frozenset(
        capability.name
        for capability in McpViewerRequestToolBindingABC.__registry__
    )
    return tuple(
        declaration
        for declaration in agent_capability_declarations()
        if declaration.name not in explicit_capability_names
        and isinstance(
            declaration.request_invocation,
            AgentViewerWindowRequestServiceInvocation,
        )
    )


class ViewerProbeMcpToolBinding(McpViewerRequestToolBindingABC):
    capability = agent_capabilities.probe_viewer_window

    @classmethod
    def request_signature(cls) -> Signature:
        return Signature(
            parameters=cls.connection_parameters(),
            return_annotation=dict,
        )

    @classmethod
    def request_from_arguments(
        cls,
        arguments: Mapping[str, JsonValue],
    ) -> ViewerWindowStateRequest:
        return cls.control_args(arguments).state_request()

    @classmethod
    def execute_request(
        cls,
        ctx: OpenHCSAgentContext,
        request: ViewerWindowControlRequest,
    ) -> dict:
        return ctx.viewer_window_service.probe_window(request)


def build_server(context: OpenHCSAgentContext | None = None):
    """Build a FastMCP server without importing PyQt or GUI services."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError(
            "The OpenHCS MCP server requires the optional 'mcp' dependency. "
            "Install with `pip install -e .[mcp]`."
        ) from exc

    ctx = context or create_agent_context()
    server = FastMCP("OpenHCS")

    def openhcs_tool(*, allow_stale_server: bool = False):
        def decorator(fn):
            @wraps(fn)
            def guarded_tool(*args, **kwargs):
                if (
                    not allow_stale_server
                    and _mcp_server_source_changed_since_import()
                ):
                    return _mcp_server_stale_error(fn.__name__)
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    return _mcp_tool_error(fn.__name__, exc)

            server.tool()(guarded_tool)
            return guarded_tool

        return decorator

    for tool_binding_type in McpNoArgumentToolBindingABC.__registry__.values():
        tool_binding_type.bind_to_server(ctx, openhcs_tool)
    for capability_declaration in generated_no_argument_capability_declarations():
        GeneratedMcpNoArgumentToolBinding.bind_to_server(
            capability_declaration,
            ctx,
            openhcs_tool,
        )
    for tool_binding_type in McpUiConnectionToolBindingABC.__registry__.values():
        tool_binding_type.bind_to_server(ctx, openhcs_tool)
    for capability_declaration in generated_ui_connection_capability_declarations():
        GeneratedMcpUiConnectionToolBinding.bind_to_server(
            capability_declaration,
            ctx,
            openhcs_tool,
        )
    for tool_binding_type in McpUiRequestToolBindingABC.__registry__.values():
        tool_binding_type.bind_to_server(ctx, openhcs_tool)
    for capability_declaration in generated_ui_request_capability_declarations():
        GeneratedMcpUiRequestToolBinding.bind_to_server(
            capability_declaration,
            ctx,
            openhcs_tool,
        )
    for tool_binding_type in McpScalarInputToolBindingABC.__registry__.values():
        tool_binding_type.bind_to_server(ctx, openhcs_tool)
    for capability_declaration in generated_scalar_input_capability_declarations():
        GeneratedMcpScalarInputToolBinding.bind_to_server(
            capability_declaration,
            ctx,
            openhcs_tool,
        )
    for tool_binding_type in McpUiScalarInputToolBindingABC.__registry__.values():
        tool_binding_type.bind_to_server(ctx, openhcs_tool)
    for capability_declaration in generated_ui_scalar_capability_declarations():
        GeneratedMcpUiScalarInputToolBinding.bind_to_server(
            capability_declaration,
            ctx,
            openhcs_tool,
        )
    for tool_binding_type in McpConfigPatchToolBindingABC.__registry__.values():
        tool_binding_type.bind_to_server(ctx, openhcs_tool)
    for capability_declaration in generated_config_patch_capability_declarations():
        GeneratedMcpConfigPatchToolBinding.bind_to_server(
            capability_declaration,
            ctx,
            openhcs_tool,
        )
    for tool_binding_type in McpFromFieldsToolBindingABC.__registry__.values():
        tool_binding_type.bind_to_server(ctx, openhcs_tool)
    for capability_declaration in generated_from_fields_capability_declarations():
        GeneratedMcpFromFieldsToolBinding.bind_to_server(
            capability_declaration,
            ctx,
            openhcs_tool,
        )
    for tool_binding_type in McpDataclassRequestToolBindingABC.__registry__.values():
        tool_binding_type.bind_to_server(ctx, openhcs_tool)
    for capability_declaration in generated_dataclass_request_capability_declarations():
        GeneratedMcpDataclassRequestToolBinding.bind_to_server(
            capability_declaration,
            ctx,
            openhcs_tool,
        )
    for tool_binding_type in McpViewerRequestToolBindingABC.__registry__.values():
        tool_binding_type.bind_to_server(ctx, openhcs_tool)
    for capability_declaration in generated_viewer_request_capability_declarations():
        GeneratedMcpViewerRequestToolBinding.bind_to_server(
            capability_declaration,
            ctx,
            openhcs_tool,
        )
    for capability_declaration in generated_resource_capability_declarations():
        GeneratedMcpResourceBinding.bind_to_server(
            capability_declaration,
            ctx,
            server,
        )

    return server


@dataclass(frozen=True, slots=True)
class McpToolErrorResult:
    """Structured MCP boundary error returned instead of raising through transport."""

    schema_version: str
    ok: bool
    tool: str
    errors: tuple[AgentError, ...]


@dataclass(frozen=True, slots=True)
class McpServerStaleErrorResult:
    """Structured stale-process error with agent-actionable restart metadata."""

    schema_version: str
    ok: bool
    tool: str
    errors: tuple[AgentError, ...]
    server_process_id: int
    server_started_at_unix: float
    stale_source_paths: tuple[str, ...]
    restart_required: bool
    restart_command: tuple[str, ...]
    restart_hint: str


def _mcp_tool_error(tool_name: str, exception: Exception) -> JsonValue:
    return to_jsonable(
        McpToolErrorResult(
            schema_version=SCHEMA_VERSION,
            ok=False,
            tool=tool_name,
            errors=(_mcp_tool_agent_error(exception),),
        )
    )


def _mcp_tool_agent_error(exception: Exception) -> AgentError:
    if isinstance(exception, AgentFacingErrorMixin):
        return exception.to_agent_error()
    return AgentError.from_exception(
        "mcp_tool_failed",
        exception,
        hint="The MCP server caught this exception at the tool boundary.",
    )


def _mcp_server_stale_error(tool_name: str) -> JsonValue:
    stale_source_paths = _mcp_server_stale_source_paths()
    if stale_source_paths:
        stale_path = str(stale_source_paths[0])
    else:
        stale_path = str(MCP_SERVER_SOURCE_PATH)
    return to_jsonable(
        McpServerStaleErrorResult(
            schema_version=SCHEMA_VERSION,
            ok=False,
            tool=tool_name,
            errors=(
                AgentError(
                    code="mcp_server_stale",
                    message=(
                        "The OpenHCS MCP server source changed after this process "
                        "started. Restart the MCP server before using agent tools."
                    ),
                    hint=MCP_SERVER_RESTART_HINT,
                    path=stale_path,
                ),
            ),
            server_process_id=MCP_SERVER_PROCESS_ID,
            server_started_at_unix=MCP_SERVER_IMPORTED_AT_UNIX,
            stale_source_paths=tuple(str(source_path) for source_path in stale_source_paths),
            restart_required=True,
            restart_command=_mcp_server_restart_command(),
            restart_hint=MCP_SERVER_RESTART_HINT,
        )
    )


def _json_object_or_empty(value: dict | None) -> dict:
    if value is None:
        return {}
    return dict(value)


@dataclass(frozen=True, slots=True)
class McpViewerConnectionToolFields:
    """Raw MCP viewer connection arguments before policy resolution."""

    port: int
    host: str
    transport_mode: str | None
    timeout_ms: int | None

    def to_control_args(
        self,
        timeout_policy: type[McpControlTimeoutPolicy] = McpViewerTimeoutPolicy,
    ) -> "McpViewerConnectionToolArgs":
        return McpViewerConnectionToolArgs.from_fields(
            port=self.port,
            host=self.host,
            transport_mode=self.transport_mode,
            timeout_ms=self.timeout_ms,
            timeout_policy=timeout_policy,
        )


@dataclass(frozen=True, slots=True)
class McpViewerConnectionToolArgs(ViewerWindowControlRequest):
    """MCP viewer connection fields projected into agent viewer request DTOs."""

    @classmethod
    def from_fields(
        cls,
        *,
        port: int,
        host: str,
        transport_mode: str | None,
        timeout_ms: int | None,
        timeout_policy: type[McpControlTimeoutPolicy] = McpViewerTimeoutPolicy,
    ) -> Self:
        return cls(
            connection=ExecutionConnectionSpec(
                host=host,
                port=port,
                transport_mode=transport_mode,
            ),
            timeout_ms=timeout_policy.resolve(timeout_ms),
        )

    def state_request(
        self,
        state_controls: ViewerStateControlOptions | None = None,
        *,
        include_response: bool = True,
    ) -> ViewerWindowStateRequest:
        return ViewerWindowStateRequest(
            connection=self.connection,
            timeout_ms=self.timeout_ms,
            include_response=include_response,
            state_controls=(
                state_controls
                if state_controls is not None
                else ViewerStateControlOptions()
            ),
        )

    def payload_request(
        self,
        payload_controls: ViewerPayloadControlOptions,
        *,
        include_response: bool = True,
    ) -> ViewerWindowPayloadRequest:
        return ViewerWindowPayloadRequest(
            connection=self.connection,
            timeout_ms=self.timeout_ms,
            payload_controls=payload_controls,
            include_response=include_response,
        )

    def navigation_request(
        self,
        navigation: ViewerNavigationControlOptions,
    ) -> ViewerWindowNavigationRequest:
        return ViewerWindowNavigationRequest(
            connection=self.connection,
            timeout_ms=self.timeout_ms,
            navigation=navigation,
        )

    def validation_request(
        self,
        validation_policy: ViewerWindowValidationPolicy,
        *,
        state_controls: ViewerStateControlOptions | None = None,
        include_state: bool = False,
    ) -> ViewerWindowValidationRequest:
        return ViewerWindowValidationRequest(
            connection=self.connection,
            timeout_ms=self.timeout_ms,
            validation_policy=validation_policy,
            state_controls=(
                state_controls
                if state_controls is not None
                else ViewerStateControlOptions()
            ),
            include_state=include_state,
        )


@dataclass(frozen=True, slots=True)
class McpUiBridgeConnectionRequest:
    """MCP-facing sparse connection request for a running OpenHCS UI bridge."""

    host: str | None = None
    port: int | None = None
    transport_mode: str | None = None
    persistent: bool | None = None
    timeout_ms: int | None = None
    auth_token: str | None = None
    descriptor_file_path: str | None = None
    bridge_instance_id: str | None = None

    def to_agent_request(self) -> UiBridgeConnectionRequest:
        return UiBridgeConnectionRequest.from_values(
            host=self.host,
            port=self.port,
            transport_mode=self.transport_mode,
            persistent=self.persistent,
            timeout_ms=self.timeout_ms,
            auth_token=self.auth_token,
            descriptor_file_path=self.descriptor_file_path,
            bridge_instance_id=self.bridge_instance_id,
        )


class UiBridgeConnectionToolMapping:
    """Typed adapter for external MCP connection mapping values."""

    def __init__(self, values: Mapping[str, JsonValue]) -> None:
        self._values = values

    @classmethod
    def from_optional(cls, value: Mapping[str, JsonValue] | None) -> Self:
        if value is None:
            return cls({})
        return cls(dict(value))

    def optional_str(self, field_name: str) -> str | None:
        value = self._optional_value(field_name)
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError(
                f"UI bridge connection field {field_name!r} must be a string."
            )
        return value

    def optional_int(self, field_name: str) -> int | None:
        value = self._optional_value(field_name)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"UI bridge connection field {field_name!r} must be an int."
            )
        return value

    def optional_bool(self, field_name: str) -> bool | None:
        value = self._optional_value(field_name)
        if value is None:
            return None
        if not isinstance(value, bool):
            raise TypeError(
                f"UI bridge connection field {field_name!r} must be a bool."
            )
        return value

    def _optional_value(self, field_name: str) -> JsonValue | None:
        if field_name not in self._values:
            return None
        return self._values[field_name]


class UiBridgeConnectionToolArgs:
    """MCP tool argument adapter for a UI bridge connection request."""

    def __init__(self, request: UiBridgeConnectionRequest) -> None:
        self._request = request

    @classmethod
    def from_mapping(
        cls,
        value: (
            McpUiBridgeConnectionRequest
            | UiBridgeConnectionRequest
            | Mapping[str, JsonValue]
            | None
        ),
    ) -> Self:
        if isinstance(value, McpUiBridgeConnectionRequest):
            return cls(value.to_agent_request())
        if isinstance(value, UiBridgeConnectionRequest):
            return cls(value)
        mapping = UiBridgeConnectionToolMapping.from_optional(value)
        return cls(
            UiBridgeConnectionRequest.from_values(
                host=mapping.optional_str("host"),
                port=mapping.optional_int("port"),
                transport_mode=mapping.optional_str("transport_mode"),
                persistent=mapping.optional_bool("persistent"),
                timeout_ms=mapping.optional_int("timeout_ms"),
                auth_token=mapping.optional_str("auth_token"),
                descriptor_file_path=mapping.optional_str("descriptor_file_path"),
                bridge_instance_id=mapping.optional_str("bridge_instance_id"),
            )
        )

    def resolve(
        self,
        context: OpenHCSAgentContext,
        *,
        timeout_policy: type[McpControlTimeoutPolicy] = McpUiBridgeTimeoutPolicy,
    ) -> UiBridgeConnectionSpec:
        return context.ui_bridge_service.connection_from_fields(
            replace(
                self._request,
                timeout_ms=timeout_policy.resolve(self._request.timeout_ms),
            )
        )
