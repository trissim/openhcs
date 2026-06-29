"""Viewer command declarations."""

from __future__ import annotations

import argparse
from collections.abc import Mapping

from openhcs.agent.capabilities import agent_capabilities
from openhcs.agent.dto.common import JsonValue
from openhcs.agent.dto.execution import ExecutionConnectionSpec
from openhcs.agent.dto.viewer import (
    VIEWER_WINDOW_CONTROL_TIMEOUT_MS_DEFAULT,
    ViewerWindowImageSampleRequest,
    ViewerWindowLayerIsolationRequest,
    ViewerWindowNavigationRequest,
    ViewerWindowPayloadRequest,
    ViewerWindowRoiSummaryRequest,
    ViewerWindowSnapshotRequest,
    ViewerWindowStateRequest,
    ViewerWindowValidationRequest,
)
from openhcs.mcp.dev_client_commanding import CapabilityBackedCommandSpec, SingleToolCommandSpec
from openhcs.mcp.dev_client_core import (
    McpDevToolCall,
    McpToolArgumentAuthority,
    ViewerConnectionArguments,
    add_request_field_option,
    add_viewer_connection_options,
    add_viewer_port_argument,
    axis_indices_tool_argument,
    axis_indices_wire_argument,
    extend_required_component_labels,
    optional_bool,
    parse_navigation_axis_indices,
    parse_required_axis_labels,
    request_factory_parameter,
    request_field_bool_default,
    request_field_int_default,
    request_field_parameter,
    request_field_string_default,
    required_viewer_route_key_argument,
    viewer_route_key_argument,
    viewer_visible_route_keys_argument,
)
from openhcs.mcp.dev_client_rendering import ViewerImageSampleRenderOptions
from openhcs.runtime.window_snapshot import WindowSnapshotCaptureScope

class ViewerPayloadsCommandSpec(CapabilityBackedCommandSpec):
    capability = agent_capabilities.get_viewer_window_payloads

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        add_viewer_port_argument(parser)
        add_request_field_option(
            parser,
            ViewerWindowPayloadRequest,
            "route_key",
            "--route-key",
        )
        parser.add_argument("--axis-indices")
        parser.add_argument(
            "--axis-index",
            action="append",
            default=[],
            metavar="NAME=INDEX",
            help="Route-local semantic payload axis index; repeat for multiple axes.",
        )
        parser.add_argument(
            "--include-array-values",
            dest="include_array_values",
            action="store_true",
            default=None,
        )
        parser.add_argument(
            "--no-array-values",
            dest="include_array_values",
            action="store_false",
            default=None,
        )
        parser.add_argument(
            "--include-shape-payloads",
            dest="include_shape_payloads",
            action="store_true",
            default=None,
        )
        parser.add_argument(
            "--no-shape-payloads",
            dest="include_shape_payloads",
            action="store_false",
            default=None,
        )
        add_request_field_option(
            parser,
            ViewerWindowPayloadRequest,
            "max_array_elements",
            "--max-array-elements",
        )
        add_request_field_option(
            parser,
            ViewerWindowPayloadRequest,
            "max_shape_payloads",
            "--max-shape-payloads",
        )
        parser.add_argument("--host", default=ExecutionConnectionSpec().host)
        parser.add_argument("--transport-mode")
        parser.add_argument(
            "--control-timeout-ms",
            "--timeout-ms",
            dest="timeout_ms",
            type=int,
            help="Viewer control timeout in milliseconds.",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Render the complete MCP JSON response instead of a compact summary.",
        )

    def calls_from_args(
        self,
        args: argparse.Namespace,
    ) -> tuple[McpDevToolCall, ...]:
        connection_args = ViewerConnectionArguments.from_args(args)
        request = ViewerWindowPayloadRequest.from_fields(
            connection=ExecutionConnectionSpec(
                host=connection_args.host,
                port=connection_args.port,
                transport_mode=connection_args.transport_mode,
            ),
            timeout_ms=(
                VIEWER_WINDOW_CONTROL_TIMEOUT_MS_DEFAULT
                if connection_args.timeout_ms is None
                else connection_args.timeout_ms
            ),
            route_key=args.route_key,
            axis_indices=axis_indices_wire_argument(
                args.axis_indices,
                args.axis_index,
            ),
            include_array_values=args.include_array_values,
            max_array_elements=args.max_array_elements,
            include_shape_payloads=args.include_shape_payloads,
            max_shape_payloads=args.max_shape_payloads,
        )
        return (
            McpDevToolCall(
                self.capability.name,
                request.as_tool_arguments(),
            ),
        )

class SnapshotViewerCommandSpec(SingleToolCommandSpec):
    capability = agent_capabilities.viewer_snapshot_window

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        add_viewer_port_argument(parser)
        add_request_field_option(
            parser,
            ViewerWindowSnapshotRequest,
            "output_dir_path",
            "--output-dir-path",
        )
        parser.add_argument(
            "--capture-scope",
            choices=tuple(scope.value for scope in WindowSnapshotCaptureScope),
            default=request_field_string_default(
                ViewerWindowSnapshotRequest,
                "capture_scope",
            ),
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Render the complete MCP JSON response instead of a compact summary.",
        )
        add_viewer_connection_options(parser)

    def tool_arguments(
        self,
        args: argparse.Namespace,
    ) -> dict[str, JsonValue]:
        connection_args = ViewerConnectionArguments.from_args(args)
        return ViewerWindowSnapshotRequest.from_fields(
            connection=ExecutionConnectionSpec(
                host=connection_args.host,
                port=connection_args.port,
                transport_mode=connection_args.transport_mode,
            ),
            timeout_ms=(
                VIEWER_WINDOW_CONTROL_TIMEOUT_MS_DEFAULT
                if connection_args.timeout_ms is None
                else connection_args.timeout_ms
            ),
            output_dir_path=args.output_dir_path,
            capture_scope=args.capture_scope,
        ).as_tool_arguments()

class ViewerStateCommandSpec(SingleToolCommandSpec):
    capability = agent_capabilities.get_viewer_window_state

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        cli_factory = ViewerWindowStateRequest.from_cli_fields
        add_viewer_port_argument(parser)
        parser.add_argument(
            "--route-key",
            default=request_factory_parameter(cli_factory, "route_key").default,
        )
        parser.add_argument(
            "--no-component-values",
            action="store_false",
            dest="include_component_values",
            default=request_factory_parameter(
                cli_factory,
                "include_component_values",
            ).default,
        )
        parser.add_argument(
            "--max-component-values-per-layer",
            type=int,
            default=request_factory_parameter(
                cli_factory,
                "max_component_values_per_layer",
            ).default,
        )
        parser.add_argument(
            "--no-payload-summaries",
            action="store_false",
            dest="include_payload_summaries",
            default=request_factory_parameter(
                cli_factory,
                "include_payload_summaries",
            ).default,
        )
        parser.add_argument(
            "--max-payload-summaries-per-layer",
            type=int,
            default=request_factory_parameter(
                cli_factory,
                "max_payload_summaries_per_layer",
            ).default,
        )
        parser.add_argument(
            "--include-response",
            action="store_true",
            default=request_factory_parameter(cli_factory, "include_response").default,
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Render the complete MCP JSON response instead of a compact summary.",
        )
        add_viewer_connection_options(parser)

    def tool_arguments(
        self,
        args: argparse.Namespace,
    ) -> dict[str, JsonValue]:
        connection_args = ViewerConnectionArguments.from_args(args)
        return ViewerWindowStateRequest.from_cli_fields(
            connection=ExecutionConnectionSpec(
                host=connection_args.host,
                port=connection_args.port,
                transport_mode=connection_args.transport_mode,
            ),
            timeout_ms=(
                VIEWER_WINDOW_CONTROL_TIMEOUT_MS_DEFAULT
                if connection_args.timeout_ms is None
                else connection_args.timeout_ms
            ),
            route_key=args.route_key,
            include_component_values=args.include_component_values,
            max_component_values_per_layer=args.max_component_values_per_layer,
            include_payload_summaries=args.include_payload_summaries,
            max_payload_summaries_per_layer=args.max_payload_summaries_per_layer,
            include_response=args.include_response,
        ).as_tool_arguments()

class ValidateViewerCommandSpec(SingleToolCommandSpec):
    capability = agent_capabilities.validate_viewer_window_state

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        add_viewer_port_argument(parser)
        add_request_field_option(
            parser,
            ViewerWindowValidationRequest,
            "route_key",
            "--route-key",
        )
        add_request_field_option(
            parser,
            ViewerWindowValidationRequest,
            "expected_layer_count",
            "--expected-layer-count",
        )
        parser.add_argument(
            "--required-axis-label",
            "--required-axis",
            "--require-axis",
            action="append",
            default=list(
                request_field_parameter(
                    ViewerWindowValidationRequest,
                    "required_axis_labels",
                ).default
            ),
            help=(
                "Required axis label; repeat or pass comma/slash-separated labels "
                "such as channel,y,x or channel/y/x."
            ),
        )
        parser.add_argument(
            "--required-component-label",
            "--required-component",
            "--require-component",
            action="append",
            default=list(
                request_field_parameter(
                    ViewerWindowValidationRequest,
                    "required_component_labels",
                ).default
            ),
            help=(
                "Required biological/component metadata label; repeat or pass "
                "comma/slash-separated labels such as well,site,timepoint."
            ),
        )
        parser.add_argument(
            "--require-components",
            action="store_true",
            help="Require every OpenHCS component label declared by AllComponents.",
        )
        parser.add_argument(
            "--allow-zero-payloads",
            action="store_true",
            default=not request_field_bool_default(
                ViewerWindowValidationRequest,
                "require_nonzero_payloads",
            ),
            help="Do not require payload records to contain nonzero data.",
        )
        parser.add_argument(
            "--require-nonzero-payloads",
            dest="allow_zero_payloads",
            action="store_false",
            default=not request_field_bool_default(
                ViewerWindowValidationRequest,
                "require_nonzero_payloads",
            ),
            help="Require nonzero payload records. This is the default.",
        )
        add_request_field_option(
            parser,
            ViewerWindowValidationRequest,
            "include_state",
            "--include-state",
            action="store_true",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Render the complete MCP JSON response instead of a compact summary.",
        )
        add_viewer_connection_options(parser)

    def tool_arguments(
        self,
        args: argparse.Namespace,
    ) -> dict[str, JsonValue]:
        connection = ViewerConnectionArguments.from_args(args)
        request = ViewerWindowValidationRequest.from_fields(
            connection=ExecutionConnectionSpec(
                host=connection.host,
                port=connection.port,
                transport_mode=connection.transport_mode,
            ),
            timeout_ms=(
                connection.timeout_ms
                if connection.timeout_ms is not None
                else VIEWER_WINDOW_CONTROL_TIMEOUT_MS_DEFAULT
            ),
            route_key=args.route_key,
            expected_layer_count=args.expected_layer_count,
            required_axis_labels=parse_required_axis_labels(
                args.required_axis_label
            ),
            required_component_labels=extend_required_component_labels(
                args.required_component_label,
                require_all_components=args.require_components,
            ),
            require_nonzero_payloads=not args.allow_zero_payloads,
            include_state=args.include_state,
        )
        return McpToolArgumentAuthority.from_payload(request.as_tool_arguments())

class ViewerRoisCommandSpec(SingleToolCommandSpec):
    capability = agent_capabilities.summarize_viewer_window_rois

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        add_viewer_port_argument(parser)
        parser.add_argument(
            "route_key",
            nargs="?",
            help="Optional ROI/shapes route key; omitted summarizes all ROI layers.",
        )
        parser.add_argument(
            "--route-key",
            dest="route_key_option",
            help="Optional ROI/shapes route key; alias for the positional route_key.",
        )
        parser.add_argument("--axis-indices")
        parser.add_argument(
            "--axis-index",
            action="append",
            default=[],
            metavar="NAME=INDEX",
            help="Route-local semantic payload axis index; repeat for multiple axes.",
        )
        parser.add_argument(
            "--max-rois",
            "--limit",
            dest="max_rois",
            type=int,
            default=request_field_int_default(
                ViewerWindowRoiSummaryRequest,
                "max_rois",
            ),
        )
        add_request_field_option(
            parser,
            ViewerWindowRoiSummaryRequest,
            "max_examples",
            "--max-examples",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Render the complete MCP JSON response instead of a compact summary.",
        )
        add_viewer_connection_options(parser)

    def tool_arguments(
        self,
        args: argparse.Namespace,
    ) -> dict[str, JsonValue]:
        route_key = viewer_route_key_argument(
            args,
            args.route_key,
            args.route_key_option,
        )
        connection = ViewerConnectionArguments.from_args(
            args,
            allow_positional_value_after_port_option=True,
        )
        request = ViewerWindowRoiSummaryRequest.from_fields(
            connection=ExecutionConnectionSpec(
                host=connection.host,
                port=connection.port,
                transport_mode=connection.transport_mode,
            ),
            timeout_ms=(
                connection.timeout_ms
                if connection.timeout_ms is not None
                else VIEWER_WINDOW_CONTROL_TIMEOUT_MS_DEFAULT
            ),
            route_key=route_key,
            axis_indices=axis_indices_tool_argument(
                args.axis_indices,
                args.axis_index,
            ),
            max_rois=args.max_rois,
            max_examples=args.max_examples,
        )
        return McpToolArgumentAuthority.from_payload(request.as_tool_arguments())

class SampleViewerImageCommandSpec(SingleToolCommandSpec):
    capability = agent_capabilities.sample_viewer_window_image

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        add_viewer_port_argument(parser)
        parser.add_argument("route_key", nargs="?")
        parser.add_argument(
            "--route-key",
            dest="route_key_option",
            help="Image layer route key; alias for the positional route_key.",
        )
        parser.add_argument("--axis-indices")
        parser.add_argument(
            "--axis-index",
            action="append",
            default=[],
            metavar="NAME=INDEX",
            help="Route-local semantic payload axis index; repeat for multiple axes.",
        )
        add_request_field_option(parser, ViewerWindowImageSampleRequest, "y", "--y")
        add_request_field_option(parser, ViewerWindowImageSampleRequest, "x", "--x")
        add_request_field_option(
            parser,
            ViewerWindowImageSampleRequest,
            "height",
            "--height",
        )
        add_request_field_option(
            parser,
            ViewerWindowImageSampleRequest,
            "width",
            "--width",
        )
        parser.add_argument(
            "--include-array-values",
            dest="include_array_values",
            action="store_true",
            default=request_field_bool_default(
                ViewerWindowImageSampleRequest,
                "include_array_values",
            ),
            help="Include raw sampled pixel values in the response.",
        )
        parser.add_argument(
            "--no-array-values",
            dest="include_array_values",
            action="store_false",
            help="Return image stats and sample bounds without sampled pixel values.",
        )
        add_request_field_option(
            parser,
            ViewerWindowImageSampleRequest,
            "max_array_elements",
            "--max-array-elements",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Render the complete MCP JSON response instead of a compact summary.",
        )
        add_viewer_connection_options(parser)

    def tool_arguments(
        self,
        args: argparse.Namespace,
    ) -> dict[str, JsonValue]:
        connection = ViewerConnectionArguments.from_args(
            args,
            allow_positional_value_after_port_option=True,
        )
        request = ViewerWindowImageSampleRequest.from_fields(
            connection=ExecutionConnectionSpec(
                host=connection.host,
                port=connection.port,
                transport_mode=connection.transport_mode,
            ),
            timeout_ms=(
                connection.timeout_ms
                if connection.timeout_ms is not None
                else VIEWER_WINDOW_CONTROL_TIMEOUT_MS_DEFAULT
            ),
            route_key=viewer_route_key_argument(
                args,
                args.route_key,
                args.route_key_option,
            ),
            axis_indices=axis_indices_tool_argument(
                args.axis_indices,
                args.axis_index,
            ),
            y=args.y,
            x=args.x,
            height=args.height,
            width=args.width,
            include_array_values=args.include_array_values,
            max_array_elements=args.max_array_elements,
        )
        return McpToolArgumentAuthority.from_payload(request.as_tool_arguments())

    def renderer_options(
        self,
        args: argparse.Namespace,
    ) -> ViewerImageSampleRenderOptions:
        return ViewerImageSampleRenderOptions(
            include_array_values_requested=args.include_array_values,
        )

    def call_render_args(
        self,
        tool_arguments: Mapping[str, JsonValue],
    ) -> argparse.Namespace:
        return argparse.Namespace(
            json=False,
            include_array_values=optional_bool(
                tool_arguments.get("include_array_values")
            )
            or False,
        )

class NavigateViewerCommandSpec(SingleToolCommandSpec):
    capability = agent_capabilities.navigate_viewer_window

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        add_viewer_port_argument(parser)
        parser.add_argument("route_key", nargs="?")
        parser.add_argument(
            "--route-key",
            dest="route_key_option",
            help="Target layer route key; alias for the positional route_key.",
        )
        parser.add_argument(
            "--axis-index",
            action="append",
            metavar="NAME=INDEX",
            help="Route-local semantic axis index; repeat for multiple axes.",
        )
        visibility = parser.add_mutually_exclusive_group()
        visibility.add_argument(
            "--visible",
            dest="visible",
            action="store_const",
            const=True,
            default=request_field_bool_default(
                ViewerWindowNavigationRequest,
                "visible",
            ),
            help="Show the target layer after navigation.",
        )
        visibility.add_argument(
            "--hidden",
            dest="visible",
            action="store_false",
            help="Hide the target layer after navigation.",
        )
        visibility.add_argument(
            "--no-visible-change",
            dest="visible",
            action="store_const",
            const=None,
            help="Leave target layer visibility unchanged.",
        )
        selection = parser.add_mutually_exclusive_group()
        selection.add_argument(
            "--selected",
            dest="selected",
            action="store_const",
            const=True,
            default=request_field_bool_default(
                ViewerWindowNavigationRequest,
                "selected",
            ),
            help="Select the target layer after navigation.",
        )
        selection.add_argument(
            "--deselected",
            dest="selected",
            action="store_false",
            help="Deselect the target layer after navigation.",
        )
        selection.add_argument(
            "--no-selection-change",
            dest="selected",
            action="store_const",
            const=None,
            help="Leave target layer selection unchanged.",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Render the complete MCP JSON response instead of a compact summary.",
        )
        add_viewer_connection_options(parser)

    def tool_arguments(
        self,
        args: argparse.Namespace,
    ) -> dict[str, JsonValue]:
        connection = ViewerConnectionArguments.from_args(
            args,
            allow_positional_value_after_port_option=True,
        )
        request = ViewerWindowNavigationRequest.from_fields(
            connection=ExecutionConnectionSpec(
                host=connection.host,
                port=connection.port,
                transport_mode=connection.transport_mode,
            ),
            timeout_ms=(
                connection.timeout_ms
                if connection.timeout_ms is not None
                else VIEWER_WINDOW_CONTROL_TIMEOUT_MS_DEFAULT
            ),
            route_key=required_viewer_route_key_argument(
                args,
                args.route_key,
                args.route_key_option,
            ),
            axis_indices=parse_navigation_axis_indices(args.axis_index),
            visible=args.visible,
            selected=args.selected,
        )
        return McpToolArgumentAuthority.from_payload(request.as_tool_arguments())

class IsolateViewerCommandSpec(SingleToolCommandSpec):
    capability = agent_capabilities.isolate_viewer_window_layers

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        add_viewer_port_argument(parser)
        parser.add_argument("visible_route_keys", nargs="*")
        parser.add_argument("--selected-route-key")
        parser.add_argument(
            "--axis-index",
            action="append",
            metavar="NAME=INDEX",
            help="Route-local semantic axis index for the selected route.",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Render the complete MCP JSON response instead of a compact summary.",
        )
        add_viewer_connection_options(parser)

    def tool_arguments(
        self,
        args: argparse.Namespace,
    ) -> dict[str, JsonValue]:
        connection = ViewerConnectionArguments.from_args(
            args,
            allow_positional_value_after_port_option=True,
        )
        request = ViewerWindowLayerIsolationRequest.from_fields(
            connection=ExecutionConnectionSpec(
                host=connection.host,
                port=connection.port,
                transport_mode=connection.transport_mode,
            ),
            timeout_ms=(
                connection.timeout_ms
                if connection.timeout_ms is not None
                else VIEWER_WINDOW_CONTROL_TIMEOUT_MS_DEFAULT
            ),
            visible_route_keys=viewer_visible_route_keys_argument(args),
            selected_route_key=args.selected_route_key,
            axis_indices=parse_navigation_axis_indices(args.axis_index),
        )
        return McpToolArgumentAuthority.from_payload(request.as_tool_arguments())
