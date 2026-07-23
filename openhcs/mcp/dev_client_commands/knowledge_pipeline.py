"""Knowledge, architecture, function, and pipeline command declarations."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar

from openhcs.agent.capabilities import agent_capabilities
from openhcs.agent.authoring_contexts import (
    AuthoringContextDeclaration,
)
from openhcs.agent.dto.authoring import AuthoringContextRequest
from openhcs.agent.dto.common import JsonObject, JsonValue
from openhcs.agent.dto.execution import ExecutionConnectionSpec
from openhcs.agent.dto.functions import FunctionDetailRequest, FunctionSearchRequest
from openhcs.agent.dto.pipeline import (
    PipelineSourceRenderRequest,
    PipelineValidationRequest,
)
from openhcs.serialization.json import to_jsonable
from openhcs.mcp.dev_client_commanding import McpDevCommandSpec, SingleToolCommandSpec
from openhcs.mcp.dev_client_core import (
    DEFAULT_REGISTRY_DISCOVERY_TIMEOUT_SECONDS,
    McpDevCliUsageError,
    McpDevStdioSession,
    McpDevToolBatchResponse,
    McpDevToolCall,
    McpToolArgumentAuthority,
    add_pipeline_source_options,
    call_mcp_tool,
    execute_source_session_tool_arguments,
    execute_source_submit_timeout_seconds,
    execute_source_submit_tool_arguments,
    first_mapping_payload,
    optional_int,
    optional_str,
    parse_optional_json_object,
    parse_required_axis_labels,
    pipeline_source_from_args,
    resolve_positional_option_alias,
)
from openhcs.mcp.dev_client_rendering import (
    AuthoringContextRenderOptions,
    CatalogRenderOptions,
)


class KnowledgeCommandSpec(SingleToolCommandSpec):
    capability = agent_capabilities.list_knowledge_documents

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--contains")
        parser.add_argument("--limit", type=int, default=20)
        parser.add_argument(
            "--json",
            action="store_true",
            help="Render the complete MCP JSON response instead of a compact summary.",
        )

    def renderer_options(
        self,
        args: argparse.Namespace,
    ) -> CatalogRenderOptions:
        return CatalogRenderOptions(contains=args.contains, limit=args.limit)

    def call_render_args(
        self,
        tool_arguments: Mapping[str, JsonValue],
    ) -> argparse.Namespace:
        del tool_arguments
        return argparse.Namespace(json=False, contains=None, limit=20)


class ArchitectureCommandSpec(SingleToolCommandSpec):
    capability = agent_capabilities.list_architecture_topics

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--contains")
        parser.add_argument("--limit", type=int, default=20)
        parser.add_argument(
            "--json",
            action="store_true",
            help="Render the complete MCP JSON response instead of a compact summary.",
        )

    def renderer_options(
        self,
        args: argparse.Namespace,
    ) -> CatalogRenderOptions:
        return CatalogRenderOptions(contains=args.contains, limit=args.limit)

    def call_render_args(
        self,
        tool_arguments: Mapping[str, JsonValue],
    ) -> argparse.Namespace:
        del tool_arguments
        return argparse.Namespace(json=False, contains=None, limit=20)


class FunctionsCommandSpec(SingleToolCommandSpec):
    capability = agent_capabilities.search_functions
    default_timeout_seconds = DEFAULT_REGISTRY_DISCOVERY_TIMEOUT_SECONDS

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("query", nargs="?")
        parser.add_argument(
            "--query",
            dest="query_option",
            help="Search text; alias for the positional query argument.",
        )
        parser.add_argument("--library")
        parser.add_argument("--limit", type=int, default=20)
        parser.add_argument(
            "--full-signatures",
            dest="compact_signatures",
            action="store_false",
            default=True,
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Render the complete MCP JSON response instead of a compact summary.",
        )

    def tool_arguments(
        self,
        args: argparse.Namespace,
    ) -> dict[str, JsonValue]:
        query = resolve_positional_option_alias(
            args.query,
            args.query_option,
            default=None,
            value_name="query",
            option_name="--query",
        )
        return McpToolArgumentAuthority.from_payload(
            to_jsonable(
                FunctionSearchRequest(
                    query=query,
                    library=args.library,
                    limit=args.limit,
                    compact_signatures=args.compact_signatures,
                )
            )
        )


class FunctionCommandSpec(SingleToolCommandSpec):
    capability = agent_capabilities.describe_function
    default_timeout_seconds = DEFAULT_REGISTRY_DISCOVERY_TIMEOUT_SECONDS

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("function_id")
        parser.add_argument("--max-doc-chars", type=int, default=2_000)
        parser.add_argument(
            "--full-signature",
            dest="compact_signature",
            action="store_false",
            default=True,
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Render the complete MCP JSON response instead of a compact summary.",
        )

    def tool_arguments(
        self,
        args: argparse.Namespace,
    ) -> dict[str, JsonValue]:
        return McpToolArgumentAuthority.from_payload(
            to_jsonable(
                FunctionDetailRequest(
                    function_id=args.function_id,
                    max_doc_chars=args.max_doc_chars,
                    compact_signature=args.compact_signature,
                )
            )
        )


class RegisterCustomFunctionCommandSpec(SingleToolCommandSpec):
    capability = agent_capabilities.register_custom_function
    default_timeout_seconds = DEFAULT_REGISTRY_DISCOVERY_TIMEOUT_SECONDS

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        source_group = parser.add_mutually_exclusive_group(required=True)
        source_group.add_argument(
            "source_file",
            nargs="?",
            help="Path to a Python file containing one or more decorated custom functions.",
        )
        source_group.add_argument(
            "--source-code",
            help="Inline Python source containing one or more decorated custom functions.",
        )
        parser.add_argument(
            "--no-persist",
            action="store_true",
            help="Register for this MCP process only; do not write to the custom function directory.",
        )
        parser.add_argument(
            "--full-signature",
            dest="compact_signature",
            action="store_false",
            default=True,
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Render the complete MCP JSON response instead of a compact summary.",
        )

    def tool_arguments(
        self,
        args: argparse.Namespace,
    ) -> dict[str, JsonValue]:
        source_code = args.source_code
        if source_code is None:
            source_code = Path(args.source_file).read_text(encoding="utf-8")
        return McpToolArgumentAuthority.from_payload(
            {
                "source_code": source_code,
                "persist": not args.no_persist,
                "compact_signature": args.compact_signature,
            }
        )


class AuthoringContextCommandSpec(SingleToolCommandSpec):
    capability = agent_capabilities.get_authoring_context
    default_timeout_seconds = DEFAULT_REGISTRY_DISCOVERY_TIMEOUT_SECONDS

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "kind",
            nargs="?",
            choices=AuthoringContextDeclaration.allowed_values(),
            help="Authoring-context kind; positional alias for --kind.",
        )
        parser.add_argument(
            "--kind",
            "--topic",
            dest="kind_option",
            choices=AuthoringContextDeclaration.allowed_values(),
        )
        parser.add_argument(
            "--max-chars",
            type=int,
            default=AuthoringContextRequest().max_chars,
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Render the complete MCP JSON response instead of a compact summary.",
        )

    def tool_arguments(
        self,
        args: argparse.Namespace,
    ) -> dict[str, JsonValue]:
        kind = resolve_positional_option_alias(
            args.kind,
            args.kind_option,
            default=AuthoringContextRequest().kind,
            value_name="kind",
            option_name="--kind/--topic",
        )
        return McpToolArgumentAuthority.from_payload(
            to_jsonable(
                AuthoringContextRequest(
                    kind=kind,
                    max_chars=args.max_chars,
                )
            )
        )

    def renderer_options(
        self,
        args: argparse.Namespace,
    ) -> AuthoringContextRenderOptions:
        return AuthoringContextRenderOptions(max_chars=args.max_chars)

    def call_render_args(
        self,
        tool_arguments: Mapping[str, JsonValue],
    ) -> argparse.Namespace:
        return argparse.Namespace(
            json=False,
            max_chars=(
                optional_int(tool_arguments.get("max_chars"))
                or AuthoringContextRequest().max_chars
            ),
        )


class DraftPipelineStepCommandSpec(McpDevCommandSpec):
    command = "draft-pipeline-step"
    help = (
        "Create, add one FunctionStep, validate, and render a draft in one MCP session."
    )

    def calls_from_args(
        self,
        args: argparse.Namespace,
    ) -> tuple[McpDevToolCall, ...]:
        return ()

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("function_id")
        parser.add_argument("--name")
        parser.add_argument("--kwargs")
        parser.add_argument("--step-config-overrides")
        parser.add_argument("--step-id")
        parser.add_argument("--description")
        parser.add_argument("--disabled", action="store_true")
        parser.add_argument("--debug-pause", action="store_true")
        parser.add_argument("--index", type=int)
        cleanliness = parser.add_mutually_exclusive_group()
        cleanliness.add_argument(
            "--clean",
            dest="clean",
            action="store_true",
            default=True,
            help="Render sparse clean source.",
        )
        cleanliness.add_argument(
            "--full",
            dest="clean",
            action="store_false",
            help="Render full resolved source.",
        )
        parser.add_argument("--no-source", action="store_true")
        parser.add_argument("--max-source-chars", type=int, default=2_000)
        parser.add_argument(
            "--json",
            action="store_true",
            help="Render the complete MCP JSON response instead of a compact summary.",
        )

    async def run_session(
        self,
        session: McpDevStdioSession,
        args: argparse.Namespace,
    ) -> McpDevToolBatchResponse:
        timeout_seconds = self.timeout_seconds(args)
        create_result = await call_mcp_tool(
            session,
            McpDevToolCall(agent_capabilities.create_pipeline.name, {}),
            timeout_seconds,
        )
        results = [create_result]
        create_payload = first_mapping_payload(create_result)
        pipeline_id = (
            None if create_payload is None else create_payload.get("pipeline_id")
        )
        if isinstance(pipeline_id, str):
            add_arguments: dict[str, JsonValue] = {
                "pipeline_id": pipeline_id,
                "function_id": args.function_id,
                "name": args.name,
                "kwargs": parse_optional_json_object(args.kwargs),
                "step_config_overrides": parse_optional_json_object(
                    args.step_config_overrides
                ),
                "step_id": args.step_id,
                "description": args.description,
                "enabled": not args.disabled,
                "debug_pause": args.debug_pause,
                "index": args.index,
            }
            results.append(
                await call_mcp_tool(
                    session,
                    McpDevToolCall(
                        agent_capabilities.add_function_step.name,
                        add_arguments,
                    ),
                    timeout_seconds,
                )
            )
            results.append(
                await call_mcp_tool(
                    session,
                    McpDevToolCall(
                        agent_capabilities.validate_pipeline.name,
                        to_jsonable(PipelineValidationRequest(pipeline_id=pipeline_id)),
                    ),
                    timeout_seconds,
                )
            )
            if not args.no_source:
                results.append(
                    await call_mcp_tool(
                        session,
                        McpDevToolCall(
                            agent_capabilities.render_pipeline_source.name,
                            to_jsonable(
                                PipelineSourceRenderRequest(
                                    pipeline_id=pipeline_id,
                                    clean=args.clean,
                                )
                            ),
                        ),
                        timeout_seconds,
                    )
                )
        return McpDevToolBatchResponse.from_results(session.server_spec, tuple(results))

    def render_response(
        self,
        payload: JsonObject,
        args: argparse.Namespace,
    ) -> str:
        if args.json:
            return super().render_response(payload, args)
        from openhcs.mcp.dev_client_renderers.pipeline import PipelineDraftStepRenderer

        return PipelineDraftStepRenderer.render(
            payload,
            max_source_chars=args.max_source_chars,
        )


class ArtifactPlanCommandSpec(SingleToolCommandSpec):
    capability = agent_capabilities.inspect_pipeline_source_artifact_plan
    default_timeout_seconds: ClassVar[float] = 60.0

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("plate_path")
        add_pipeline_source_options(parser)
        parser.add_argument(
            "--axis-filter",
            action="append",
            default=[],
            help="Axis/well id to inspect; repeat or pass comma/slash-separated values.",
        )
        parser.add_argument(
            "--well-filter",
            action="append",
            default=[],
            help="Alias for --axis-filter when axes are wells.",
        )
        parser.add_argument("--global-config-id")
        parser.add_argument(
            "--json",
            action="store_true",
            help="Render the complete MCP JSON response instead of a compact summary.",
        )

    def tool_arguments(
        self,
        args: argparse.Namespace,
    ) -> dict[str, JsonValue]:
        axis_filter = parse_required_axis_labels(args.axis_filter)
        well_filter = parse_required_axis_labels(args.well_filter)
        if axis_filter and well_filter and axis_filter != well_filter:
            raise McpDevCliUsageError(
                "Cannot pass both --axis-filter and --well-filter with different values."
            )
        selected_axis_filter = axis_filter or well_filter
        return McpToolArgumentAuthority.from_payload(
            {
                "plate_path": args.plate_path,
                "pipeline_source": pipeline_source_from_args(args),
                "axis_filter": selected_axis_filter or None,
                "global_config_id": args.global_config_id,
            }
        )


class ExecuteSourceCommandSpec(McpDevCommandSpec):
    command = "execute-source"
    help = "Create and submit a source-backed headless execution session."
    default_timeout_seconds: ClassVar[float] = 120.0

    def calls_from_args(
        self,
        args: argparse.Namespace,
    ) -> tuple[McpDevToolCall, ...]:
        del args
        return ()

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("plate_path")
        add_pipeline_source_options(parser)
        parser.add_argument("--global-config-id")
        parser.add_argument("--host", default=ExecutionConnectionSpec().host)
        parser.add_argument("--port", type=int)
        parser.add_argument("--transport-mode")
        persistence = parser.add_mutually_exclusive_group()
        persistence.add_argument(
            "--persistent",
            dest="persistent",
            action="store_true",
            default=ExecutionConnectionSpec().persistent,
        )
        persistence.add_argument(
            "--non-persistent",
            dest="persistent",
            action="store_false",
        )
        wait_group = parser.add_mutually_exclusive_group()
        wait_group.add_argument(
            "--wait",
            dest="wait",
            action="store_true",
            default=True,
            help="Wait for execution completion before returning.",
        )
        wait_group.add_argument(
            "--no-wait",
            dest="wait",
            action="store_false",
            help="Return after submit and leave polling to runtime-status.",
        )
        parser.add_argument("--submit-timeout-ms", type=int, default=5000)
        parser.add_argument("--wait-timeout-ms", type=int, default=60_000)
        parser.add_argument(
            "--json",
            action="store_true",
            help="Render the complete MCP JSON response instead of a compact summary.",
        )

    async def run_session(
        self,
        session: McpDevStdioSession,
        args: argparse.Namespace,
    ) -> McpDevToolBatchResponse:
        timeout_seconds = self.timeout_seconds(args)
        create_result = await call_mcp_tool(
            session,
            McpDevToolCall(
                agent_capabilities.create_orchestrator_session_from_pipeline_source.name,
                execute_source_session_tool_arguments(args),
            ),
            timeout_seconds,
        )
        results = [create_result]
        create_payload = first_mapping_payload(create_result)
        session_id = (
            optional_str(create_payload.get("session_id"))
            if create_payload is not None
            else None
        )
        if session_id is not None:
            results.append(
                await call_mcp_tool(
                    session,
                    McpDevToolCall(
                        agent_capabilities.submit_pipeline_execution.name,
                        execute_source_submit_tool_arguments(
                            args,
                            session_id=session_id,
                        ),
                    ),
                    execute_source_submit_timeout_seconds(
                        args,
                        timeout_seconds=timeout_seconds,
                    ),
                )
            )
        return McpDevToolBatchResponse.from_results(session.server_spec, tuple(results))

    def render_response(
        self,
        payload: JsonObject,
        args: argparse.Namespace,
    ) -> str:
        if args.json:
            return super().render_response(payload, args)
        from openhcs.mcp.dev_client_renderers.pipeline import ExecuteSourceRenderer

        return ExecuteSourceRenderer.render(payload)
