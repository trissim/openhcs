"""Command declaration framework for the MCP dev client."""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from collections.abc import Mapping
import inspect
import json
import tempfile
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta

from openhcs.agent.capabilities import (
    AgentCapabilitySpec,
    AgentScalarInputContract,
    CapabilityCliConnectionProfile,
    get_agent_capability,
    get_capability_registry,
    require_agent_type_contract,
)
from openhcs.agent.dto.common import (
    AgentCliArgumentSpec,
    AgentCliRequest,
    JsonObject,
    JsonValue,
)
from openhcs.agent.dto.execution import (
    RuntimeServerConnectionToolRequest,
    RuntimeServerToolRequest,
)
from openhcs.mcp.dev_client_core import (
    DEFAULT_CALL_TIMEOUT_SECONDS,
    McpDevCliUsageError,
    McpDevClientPhase,
    McpDevServerSpec,
    McpDevStdioSession,
    McpDevToolBatchResponse,
    McpDevToolCall,
    McpDevToolListResponse,
    McpToolArgumentAuthority,
    add_request_factory_option,
    add_runtime_connection_options,
    add_ui_connection_options,
    add_viewer_connection_options,
    add_viewer_port_argument,
    call_mcp_session,
    captured_server_stderr_tail,
    list_mcp_session_tools,
    mcp_dev_command_key,
    ui_tool_arguments,
    viewer_connection_arguments,
)
from openhcs.mcp.dev_client_rendering import (
    McpDevOutputRenderOptions,
    McpDevOutputRenderer,
    ToolListRenderer,
)


class McpDevCommandSpec(ABC, metaclass=AutoRegisterMeta):
    """Nominal parser, execution, and rendering owner for one MCP dev command."""

    __registry__: ClassVar[dict[str, type["McpDevCommandSpec"]]] = {}
    __registry_key__ = "command"
    __key_extractor__ = mcp_dev_command_key
    __skip_if_no_key__ = True

    command: ClassVar[str]
    help: ClassVar[str | None] = None
    aliases: ClassVar[tuple[str, ...]] = ()
    execution_phase: ClassVar[McpDevClientPhase] = McpDevClientPhase.CALL_TOOL
    default_timeout_seconds: ClassVar[float] = DEFAULT_CALL_TIMEOUT_SECONDS

    @classmethod
    def all_specs(cls) -> tuple["McpDevCommandSpec", ...]:
        explicit_specs = tuple(
            command_spec_type() for command_spec_type in cls.__registry__.values()
        )
        return (*explicit_specs, *generated_mcp_dev_command_specs())

    @classmethod
    def for_name(cls, command_name: str) -> "McpDevCommandSpec":
        command_spec_type = cls.__registry__.get(command_name)
        if command_spec_type is not None:
            return command_spec_type()
        generated_spec = generated_mcp_dev_command_spec_for_name(command_name)
        if generated_spec is not None:
            return generated_spec
        raise KeyError(command_name)

    def register_parser(
        self,
        subparsers: argparse._SubParsersAction,
        command_options: argparse.ArgumentParser,
    ) -> None:
        parser = subparsers.add_parser(
            self.command,
            aliases=self.parser_aliases(),
            help=self.parser_help(),
            parents=[command_options],
        )
        parser.set_defaults(command=self.command)
        self.configure_parser(parser)
        self.configure_reflected_parser(parser)

    def parser_help(self) -> str:
        return self.help or self.command

    def parser_aliases(self) -> tuple[str, ...]:
        return self.aliases

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        """Add command-specific CLI arguments."""

    def configure_reflected_parser(self, parser: argparse.ArgumentParser) -> None:
        """Add options reflected from declarations owned outside the command."""

    @abstractmethod
    def calls_from_args(
        self,
        args: argparse.Namespace,
    ) -> tuple[McpDevToolCall, ...]:
        """Build MCP tool calls for this command."""

    async def run(
        self,
        server_spec: McpDevServerSpec,
        args: argparse.Namespace,
    ) -> McpDevToolBatchResponse | McpDevToolListResponse:
        """Execute this command through one freshly initialized stdio session."""
        for call in self.calls_from_args(args):
            call.require_surface_profile(server_spec.surface_profile)
        phase = McpDevClientPhase.START_SERVER
        with tempfile.TemporaryFile(
            mode="w+",
            encoding="utf-8",
            errors="replace",
        ) as server_stderr:
            try:
                async with McpDevStdioSession(server_spec, server_stderr) as session:
                    phase = McpDevClientPhase.INITIALIZE
                    await session.initialize(timeout_seconds=self.timeout_seconds(args))
                    phase = self.execution_phase
                    payload = await self.run_session(session, args)
                    phase = McpDevClientPhase.TEARDOWN
                    return payload
            except McpDevCliUsageError:
                raise
            except Exception as exc:
                return self.transport_failure_response(
                    server_spec,
                    phase,
                    exc,
                    server_stderr_tail=captured_server_stderr_tail(server_stderr),
                )

    async def run_session(
        self,
        session: McpDevStdioSession,
        args: argparse.Namespace,
    ) -> McpDevToolBatchResponse | McpDevToolListResponse:
        """Execute this command through an already initialized stdio session."""
        return await call_mcp_session(
            session,
            self.calls_from_args(args),
            timeout_seconds=self.timeout_seconds(args),
        )

    def timeout_seconds(self, args: argparse.Namespace) -> float:
        """Return the timeout shared by initialization and this command's calls."""
        return max(args.timeout_seconds, self.default_timeout_seconds)

    def transport_failure_response(
        self,
        server_spec: McpDevServerSpec,
        phase: McpDevClientPhase,
        exception: BaseException,
        *,
        server_stderr_tail: str | None,
    ) -> McpDevToolBatchResponse | McpDevToolListResponse:
        """Project a transport exception into this command's response shape."""
        return McpDevToolBatchResponse.from_transport_failure(
            server_spec,
            phase,
            exception,
            server_stderr_tail=server_stderr_tail,
        )

    def render_response(
        self,
        payload: JsonObject,
        args: argparse.Namespace,
    ) -> str:
        del args
        return json.dumps(payload, indent=2, sort_keys=True)


class CapabilityBackedCommandSpec(McpDevCommandSpec):
    """Command whose primary MCP tool capability is declared on the command."""

    capability: ClassVar[AgentCapabilitySpec]
    __capability_registry__: ClassVar[
        dict[AgentCapabilitySpec, type["CapabilityBackedCommandSpec"]]
    ] = {}

    def __init_subclass__(cls, **kwargs: JsonValue) -> None:
        super().__init_subclass__(**kwargs)
        try:
            capability = cls.capability
        except AttributeError:
            return
        cls.__capability_registry__[capability] = cls

    @classmethod
    def for_capability_name(
        cls,
        tool_name: str,
    ) -> "CapabilityBackedCommandSpec | None":
        try:
            capability = get_agent_capability(tool_name)
        except KeyError:
            return None
        command_spec_type = cls.__capability_registry__.get(capability)
        if command_spec_type is not None:
            return command_spec_type()
        return generated_mcp_dev_command_spec_for_capability(capability)

    def call_render_args(
        self,
        tool_arguments: Mapping[str, JsonValue],
    ) -> argparse.Namespace:
        del tool_arguments
        argument_values: dict[str, object] = {"json": False}
        renderer_binding = self.output_renderer_binding()
        if renderer_binding is not None:
            argument_values.update(renderer_binding.default_cli_argument_values())
        return argparse.Namespace(**argument_values)

    def parser_help(self) -> str:
        return self.help or self.capability.title

    def parser_aliases(self) -> tuple[str, ...]:
        return self.aliases or self.capability.cli_aliases

    def output_renderer_binding(self):
        output_contract = self.capability.output_contract
        return McpDevOutputRenderer.for_output_contract(
            output_contract if isinstance(output_contract, type) else None
        )

    def configure_reflected_parser(self, parser: argparse.ArgumentParser) -> None:
        renderer_binding = self.output_renderer_binding()
        if renderer_binding is not None:
            renderer_binding.configure_cli_parser(parser)

    def render_call_response(
        self,
        payload: JsonObject,
        tool_arguments: Mapping[str, JsonValue],
    ) -> str:
        return self.render_response(
            payload,
            self.call_render_args(tool_arguments),
        )

    def renderer_options(
        self,
        args: argparse.Namespace,
    ) -> McpDevOutputRenderOptions:
        renderer_binding = self.output_renderer_binding()
        if renderer_binding is None:
            return McpDevOutputRenderOptions()
        return renderer_binding.options_from_cli_args(args)

    def render_response(
        self,
        payload: JsonObject,
        args: argparse.Namespace,
    ) -> str:
        if bool(vars(args).get("json", False)):
            return super().render_response(payload, args)
        renderer_binding = self.output_renderer_binding()
        if renderer_binding is None:
            return super().render_response(payload, args)
        return renderer_binding.render_with_options(
            payload, self.renderer_options(args)
        )


class ToolsCommandSpec(McpDevCommandSpec):
    command = "tools"
    help = "List current-source MCP tools."
    execution_phase = McpDevClientPhase.LIST_TOOLS

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--contains")
        parser.add_argument("--limit", type=int, default=80)
        parser.add_argument(
            "--flat",
            action="store_true",
            help="Render a flat tool list instead of grouping by capability declarations.",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help=(
                "Render structured MCP JSON, preserving full metadata for entries "
                "selected by --contains and --limit."
            ),
        )

    def calls_from_args(
        self,
        args: argparse.Namespace,
    ) -> tuple[McpDevToolCall, ...]:
        return ()

    async def run_session(
        self,
        session: McpDevStdioSession,
        args: argparse.Namespace,
    ) -> McpDevToolListResponse:
        return await list_mcp_session_tools(
            session,
            timeout_seconds=self.timeout_seconds(args),
        )

    def transport_failure_response(
        self,
        server_spec: McpDevServerSpec,
        phase: McpDevClientPhase,
        exception: BaseException,
        *,
        server_stderr_tail: str | None,
    ) -> McpDevToolListResponse:
        return McpDevToolListResponse.from_transport_failure(
            server_spec,
            phase,
            exception,
            server_stderr_tail=server_stderr_tail,
        )

    def render_response(
        self,
        payload: JsonObject,
        args: argparse.Namespace,
    ) -> str:
        if args.json:
            return super().render_response(
                ToolListRenderer.project_response(
                    payload,
                    contains=args.contains,
                    limit=args.limit,
                ),
                args,
            )
        return ToolListRenderer.render(
            payload,
            contains=args.contains,
            limit=args.limit,
            grouped=not args.flat,
        )

    def call_render_args(
        self,
        tool_arguments: Mapping[str, JsonValue],
    ) -> argparse.Namespace:
        del tool_arguments
        return argparse.Namespace(json=False, contains=None, limit=20, flat=False)


class SingleToolCommandSpec(CapabilityBackedCommandSpec):
    """Command that maps to one MCP tool call."""

    @property
    def tool_name(self) -> str:
        return self.capability.name

    def tool_arguments(self, args: argparse.Namespace) -> dict[str, JsonValue]:
        return {}

    def calls_from_args(
        self,
        args: argparse.Namespace,
    ) -> tuple[McpDevToolCall, ...]:
        return (
            McpDevToolCall(
                self.tool_name,
                self.tool_arguments(args),
            ),
        )


class UiBridgeCommandSpec(McpDevCommandSpec):
    """Command specification that accepts live UI bridge connection options."""

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        add_ui_connection_options(parser)


class SingleUiBridgeToolCommandSpec(UiBridgeCommandSpec, CapabilityBackedCommandSpec):
    """UI bridge command whose entire operation is one MCP tool call."""

    capability: ClassVar[AgentCapabilitySpec]

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        super().configure_parser(parser)
        parser.add_argument(
            "--json",
            action="store_true",
            help="Render the complete MCP JSON response instead of a compact summary.",
        )

    @property
    def tool_name(self) -> str:
        return self.capability.name

    def calls_from_args(
        self,
        args: argparse.Namespace,
    ) -> tuple[McpDevToolCall, ...]:
        return (
            McpDevToolCall(
                self.tool_name,
                ui_tool_arguments(args, timeout_ms=args.timeout_ms),
            ),
        )


class GeneratedCapabilityCommandBinding:
    """Instance binding shared by generated capability-backed CLI commands."""

    def __init__(self, capability: AgentCapabilitySpec) -> None:
        command = capability.cli_command
        if command is None:
            raise ValueError(f"{capability.name} does not declare a CLI command.")
        self.capability = capability
        self.command = command


class GeneratedAgentCliRequestCommandMixin:
    """Parser/tool projection for generated commands backed by AgentCliRequest."""

    capability: AgentCapabilitySpec

    @property
    def request_type(self) -> type[AgentCliRequest]:
        input_contract = require_agent_type_contract(self.capability.input_contract)
        if not issubclass(
            input_contract,
            AgentCliRequest,
        ):
            raise TypeError(
                f"{self.capability.name} must declare an agent CLI request "
                "input contract."
            )
        return input_contract

    @property
    def request_factory(self):
        return self.request_type.agent_cli_factory()

    def request_factory_parameters(self) -> tuple[inspect.Parameter, ...]:
        return tuple(inspect.signature(self.request_factory).parameters.values())

    def configure_request_parser(self, parser: argparse.ArgumentParser) -> None:
        for parameter in self.request_factory_parameters():
            argument_spec = self.request_argument_spec(parameter.name)
            add_request_factory_option(
                parser,
                self.request_factory,
                parameter.name,
                *self.request_argument_flags(parameter.name, argument_spec),
                **self.request_argument_kwargs(argument_spec),
            )

    def request_argument_spec(
        self,
        field_name: str,
    ) -> AgentCliArgumentSpec | None:
        for argument_spec in self.request_type.agent_cli_argument_specs():
            if argument_spec.field_name == field_name:
                return argument_spec
        return None

    @staticmethod
    def request_argument_flags(
        field_name: str,
        argument_spec: AgentCliArgumentSpec | None,
    ) -> tuple[str, ...]:
        if argument_spec is None:
            return (f"--{field_name.replace('_', '-')}",)
        if argument_spec.positional:
            return (field_name,)
        if argument_spec.flags:
            return argument_spec.flags
        return (f"--{field_name.replace('_', '-')}",)

    @staticmethod
    def request_argument_kwargs(
        argument_spec: AgentCliArgumentSpec | None,
    ) -> dict[str, object]:
        if argument_spec is None:
            return {}
        kwargs: dict[str, object] = {}
        if argument_spec.nargs is not None:
            kwargs["nargs"] = argument_spec.nargs
        if argument_spec.action is not None:
            kwargs["action"] = argument_spec.action
        if argument_spec.help is not None:
            kwargs["help"] = argument_spec.help
        return kwargs

    def request_fields_from_args(
        self,
        args: argparse.Namespace,
    ) -> dict[str, object]:
        argument_values = vars(args)
        return {
            parameter.name: argument_values[parameter.name]
            for parameter in self.request_factory_parameters()
        }

    def tool_arguments_from_agent_request(
        self,
        args: argparse.Namespace,
    ) -> dict[str, JsonValue]:
        try:
            request = self.request_factory(**self.request_fields_from_args(args))
        except (TypeError, ValueError) as exc:
            raise McpDevCliUsageError(str(exc)) from exc
        return McpToolArgumentAuthority.from_payload(request.as_tool_arguments())


class GeneratedSingleToolCommandSpec(
    GeneratedAgentCliRequestCommandMixin,
    GeneratedCapabilityCommandBinding,
    SingleToolCommandSpec,
):
    """Single-tool command projected directly from a capability declaration."""

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        input_contract = self.capability.input_contract
        if isinstance(input_contract, AgentScalarInputContract):
            self.configure_scalar_input_parser(parser, input_contract)
        elif isinstance(input_contract, type) and issubclass(
            input_contract,
            AgentCliRequest,
        ):
            self.configure_request_parser(parser)
        parser.add_argument(
            "--json",
            action="store_true",
            help="Render the complete MCP JSON response instead of a compact summary.",
        )

    @staticmethod
    def configure_scalar_input_parser(
        parser: argparse.ArgumentParser,
        input_contract: AgentScalarInputContract,
    ) -> None:
        if input_contract.default_value is None:
            parser.add_argument(input_contract.field_name)
            return
        parser.add_argument(
            input_contract.field_name,
            nargs="?",
            default=input_contract.default_value,
        )

    def tool_arguments(self, args: argparse.Namespace) -> dict[str, JsonValue]:
        input_contract = self.capability.input_contract
        if isinstance(input_contract, AgentScalarInputContract):
            return {
                input_contract.field_name: vars(args)[input_contract.field_name],
            }
        if isinstance(input_contract, type) and issubclass(
            input_contract,
            AgentCliRequest,
        ):
            return self.tool_arguments_from_agent_request(args)
        return super().tool_arguments(args)


class GeneratedUiBridgeToolCommandSpec(
    GeneratedCapabilityCommandBinding,
    SingleUiBridgeToolCommandSpec,
):
    """UI bridge command projected directly from a capability declaration."""


class GeneratedViewerWindowToolCommandSpec(
    GeneratedCapabilityCommandBinding,
    SingleToolCommandSpec,
):
    """Viewer-window command projected directly from a capability declaration."""

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        add_viewer_port_argument(parser)
        add_viewer_connection_options(parser)
        parser.add_argument(
            "--json",
            action="store_true",
            help="Render the complete MCP JSON response instead of a compact summary.",
        )

    def tool_arguments(self, args: argparse.Namespace) -> dict[str, JsonValue]:
        return viewer_connection_arguments(args)


class GeneratedRuntimeServerToolCommandSpec(
    GeneratedAgentCliRequestCommandMixin,
    GeneratedCapabilityCommandBinding,
    SingleToolCommandSpec,
):
    """Runtime-server command projected directly from a capability declaration."""

    @property
    def request_type(self) -> type[RuntimeServerToolRequest]:
        input_contract = require_agent_type_contract(self.capability.input_contract)
        if not issubclass(
            input_contract,
            RuntimeServerToolRequest,
        ):
            raise TypeError(
                f"{self.capability.name} must declare a runtime server "
                "request input contract."
            )
        return input_contract

    @property
    def uses_runtime_connection_options(self) -> bool:
        return issubclass(self.request_type, RuntimeServerConnectionToolRequest)

    @staticmethod
    def runtime_connection_parameter_names() -> frozenset[str]:
        return frozenset(
            inspect.signature(RuntimeServerConnectionToolRequest.from_fields).parameters
        )

    def request_factory_parameters(self) -> tuple[inspect.Parameter, ...]:
        runtime_connection_names = (
            self.runtime_connection_parameter_names()
            if self.uses_runtime_connection_options
            else frozenset()
        )
        return tuple(
            parameter
            for parameter in inspect.signature(self.request_factory).parameters.values()
            if parameter.name not in runtime_connection_names
        )

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        if self.uses_runtime_connection_options:
            add_runtime_connection_options(parser, include_port=True)
        self.configure_request_parser(parser)
        parser.add_argument(
            "--json",
            action="store_true",
            help="Render the complete MCP JSON response instead of a compact summary.",
        )

    def request_fields_from_args(
        self,
        args: argparse.Namespace,
    ) -> dict[str, object]:
        argument_values = vars(args)
        request_fields = {}
        if self.uses_runtime_connection_options:
            request_fields.update(
                {
                    parameter_name: argument_values[parameter_name]
                    for parameter_name in self.runtime_connection_parameter_names()
                }
            )
        request_fields.update(
            {
                parameter.name: argument_values[parameter.name]
                for parameter in self.request_factory_parameters()
            }
        )
        return request_fields

    def tool_arguments(self, args: argparse.Namespace) -> dict[str, JsonValue]:
        return self.tool_arguments_from_agent_request(args)


class GeneratedMcpDevCommandProfile(ABC, metaclass=AutoRegisterMeta):
    """Registered generator for one capability CLI connection profile."""

    __registry__: ClassVar[
        dict[CapabilityCliConnectionProfile, type["GeneratedMcpDevCommandProfile"]]
    ] = {}
    __registry_key__ = "profile"
    __skip_if_no_key__ = True

    profile: ClassVar[CapabilityCliConnectionProfile]

    @classmethod
    def for_capability(
        cls,
        capability: AgentCapabilitySpec,
    ) -> type["GeneratedMcpDevCommandProfile"]:
        return cls.__registry__[capability.cli_connection_profile]

    @classmethod
    @abstractmethod
    def command_spec(
        cls,
        capability: AgentCapabilitySpec,
    ) -> CapabilityBackedCommandSpec:
        """Build the generated command spec for this profile."""


class DirectGeneratedMcpDevCommandProfile(GeneratedMcpDevCommandProfile):
    """Generated command profile for direct MCP tool calls."""

    profile = CapabilityCliConnectionProfile.DIRECT

    @classmethod
    def command_spec(
        cls,
        capability: AgentCapabilitySpec,
    ) -> CapabilityBackedCommandSpec:
        return GeneratedSingleToolCommandSpec(capability)


class UiBridgeGeneratedMcpDevCommandProfile(GeneratedMcpDevCommandProfile):
    """Generated command profile for MCP tools requiring a UI bridge connection."""

    profile = CapabilityCliConnectionProfile.UI_BRIDGE

    @classmethod
    def command_spec(
        cls,
        capability: AgentCapabilitySpec,
    ) -> CapabilityBackedCommandSpec:
        return GeneratedUiBridgeToolCommandSpec(capability)


class ViewerWindowGeneratedMcpDevCommandProfile(GeneratedMcpDevCommandProfile):
    """Generated command profile for viewer-window MCP tools."""

    profile = CapabilityCliConnectionProfile.VIEWER_WINDOW

    @classmethod
    def command_spec(
        cls,
        capability: AgentCapabilitySpec,
    ) -> CapabilityBackedCommandSpec:
        return GeneratedViewerWindowToolCommandSpec(capability)


class RuntimeServerGeneratedMcpDevCommandProfile(GeneratedMcpDevCommandProfile):
    """Generated command profile for runtime-server MCP tools."""

    profile = CapabilityCliConnectionProfile.RUNTIME_SERVER

    @classmethod
    def command_spec(
        cls,
        capability: AgentCapabilitySpec,
    ) -> CapabilityBackedCommandSpec:
        return GeneratedRuntimeServerToolCommandSpec(capability)


def generated_mcp_dev_command_spec(
    capability: AgentCapabilitySpec,
) -> CapabilityBackedCommandSpec:
    return GeneratedMcpDevCommandProfile.for_capability(capability).command_spec(
        capability
    )


def generated_mcp_dev_command_specs() -> tuple[CapabilityBackedCommandSpec, ...]:
    explicit_capabilities = frozenset(
        CapabilityBackedCommandSpec.__capability_registry__
    )
    return tuple(
        generated_mcp_dev_command_spec(capability)
        for capability in get_capability_registry().capabilities
        if capability.cli_command is not None
        and capability not in explicit_capabilities
    )


def generated_mcp_dev_command_spec_for_name(
    command_name: str,
) -> CapabilityBackedCommandSpec | None:
    for command_spec in generated_mcp_dev_command_specs():
        if command_spec.command == command_name:
            return command_spec
    return None


def generated_mcp_dev_command_spec_for_capability(
    capability: AgentCapabilitySpec,
) -> CapabilityBackedCommandSpec | None:
    if capability.cli_command is None:
        return None
    explicit_capabilities = frozenset(
        CapabilityBackedCommandSpec.__capability_registry__
    )
    if capability in explicit_capabilities:
        return None
    return generated_mcp_dev_command_spec(capability)
