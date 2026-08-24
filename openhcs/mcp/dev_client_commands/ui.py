"""UI bridge command declarations."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Mapping
from typing import cast

from pyqt_reactive.services.window_snapshot import WindowSnapshotCaptureScope

from openhcs.agent.capabilities import agent_capabilities
from openhcs.agent.dto.common import JsonObject, JsonValue
from openhcs.agent.dto.ui_bridge import (
    UiBridgeOperationStatus,
    UiObjectStateFieldFilter,
    UiSelectedPlateWorkflowKind,
)
from openhcs.agent.services.ui_bridge_service import UiBridgeGatewayTimeoutError
from openhcs.agent.ui_bridge_identities import (
    PlateManagerStateSurfaceIdentityDeclaration,
    PlateManagerWidgetIdentity,
)
from openhcs.core.selection import SelectedAllSelectionMode
from openhcs.mcp.dev_client_commanding import (
    CapabilityBackedCommandSpec,
    McpDevCommandSpec,
    UiBridgeCommandSpec,
)
from openhcs.mcp.dev_client_core import (
    DEFAULT_WORKFLOW_POLL_INTERVAL_SECONDS,
    DEFAULT_WORKFLOW_POLL_TIMEOUT_SECONDS,
    MCP_TOOL_TIMEOUT_MARGIN_SECONDS,
    McpDevCliUsageError,
    McpDevStdioSession,
    McpDevToolBatchResponse,
    McpDevToolCall,
    McpDevToolResult,
    WorkflowPollBaseline,
    WorkflowPollSkipReason,
    WorkflowPollSummaryStatus,
    WorkflowStatePollPolicy,
    add_code_document_source_options,
    add_object_state_field_filter_options,
    add_ui_connection_options,
    call_mcp_tool,
    code_document_source_from_args,
    optional_str,
    parse_cli_json_value,
    parse_json_object,
    plate_manager_state_surface_tool_arguments,
    selected_workflow_tool_arguments,
    ui_bridge_operation_result_status,
    ui_connection_arguments,
    ui_tool_arguments,
    workflow_operation_receipt_tool_arguments,
    workflow_poll_skip_reason,
    workflow_poll_summary_result,
    workflow_poll_terminal_status,
    workflow_result_action_status,
    workflow_result_operation_id,
    workflow_result_target_scope_ids,
    workflow_result_was_accepted,
)
from openhcs.mcp.dev_client_rendering import (
    DEFAULT_CODE_DOCUMENT_MAX_CHARS,
    CodeDocumentRenderOptions,
    McpDiagnosticRenderer,
    UiActionCatalogRenderOptions,
    UiActionInvokeRenderOptions,
    WidgetTreeOutputFormat,
    WidgetTreeRenderOptions,
)


class StateSurfaceCommandSpec(CapabilityBackedCommandSpec):
    capability = agent_capabilities.ui_get_state_surface

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "surface_id",
            nargs="?",
            default=PlateManagerStateSurfaceIdentityDeclaration.require_value(),
        )
        parser.add_argument(
            "--selection-mode",
            choices=tuple(mode.value for mode in SelectedAllSelectionMode),
            default=SelectedAllSelectionMode.ALL.value,
        )
        parser.add_argument("--base-revision-token")
        parser.add_argument(
            "--json",
            action="store_true",
            help="Render the complete MCP JSON response instead of a compact summary.",
        )
        add_ui_connection_options(parser)

    def calls_from_args(
        self,
        args: argparse.Namespace,
    ) -> tuple[McpDevToolCall, ...]:
        tool_arguments: dict[str, JsonValue] = {
            "surface_id": args.surface_id,
            "selection_mode": args.selection_mode,
            "connection": ui_connection_arguments(
                args,
                timeout_ms=args.timeout_ms,
            ),
        }
        if args.base_revision_token is not None:
            tool_arguments["base_revision_token"] = args.base_revision_token
        return (
            McpDevToolCall(
                self.capability.name,
                tool_arguments,
            ),
        )


class CallCommandSpec(McpDevCommandSpec):
    command = "call"
    help = "Call one MCP tool."

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("tool_name")
        parser.add_argument(
            "--arguments",
            default="{}",
            help="JSON object passed as the MCP tool arguments.",
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
        return (
            McpDevToolCall(
                args.tool_name,
                parse_json_object(args.arguments),
            ),
        )

    def render_response(
        self,
        payload: JsonObject,
        args: argparse.Namespace,
    ) -> str:
        if args.json:
            return super().render_response(payload, args)
        command_spec = CapabilityBackedCommandSpec.for_capability_name(args.tool_name)
        if command_spec is None:
            return super().render_response(payload, args)
        return command_spec.render_call_response(
            payload,
            parse_json_object(args.arguments),
        )


class SelectedWorkflowCommandSpec(CapabilityBackedCommandSpec):
    capability = agent_capabilities.ui_selected_plate_workflow

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        state_surface_id = PlateManagerStateSurfaceIdentityDeclaration.require_value()
        parser.add_argument(
            "workflow",
            choices=tuple(workflow.value for workflow in UiSelectedPlateWorkflowKind),
        )
        parser.add_argument(
            "--require-confirmation",
            action="store_true",
            help="Ask the UI bridge to reject the workflow unless confirmation is disabled.",
        )
        parser.add_argument(
            "--wait",
            "--poll-state",
            dest="poll_state",
            action="store_true",
            help=(
                f"After accepted dispatch, wait on {state_surface_id} until the "
                "selected workflow reaches its terminal state. This is workflow "
                "completion, not the wait for a UI operation receipt."
            ),
        )
        parser.add_argument(
            "--wait-selection-mode",
            "--poll-selection-mode",
            dest="poll_selection_mode",
            choices=tuple(mode.value for mode in SelectedAllSelectionMode),
            default=SelectedAllSelectionMode.SELECTED.value,
            help=(f"Selection mode used while waiting on {state_surface_id}."),
        )
        parser.add_argument(
            "--wait-interval-seconds",
            "--poll-interval-seconds",
            dest="poll_interval_seconds",
            type=float,
            default=DEFAULT_WORKFLOW_POLL_INTERVAL_SECONDS,
            help="Advanced delay between workflow state checks.",
        )
        parser.add_argument(
            "--wait-timeout-seconds",
            "--poll-timeout-seconds",
            dest="poll_timeout_seconds",
            type=float,
            default=DEFAULT_WORKFLOW_POLL_TIMEOUT_SECONDS,
            help=f"Maximum elapsed time spent waiting on {state_surface_id}.",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Render the complete MCP JSON response instead of a compact summary.",
        )
        add_ui_connection_options(parser)

    def calls_from_args(
        self,
        args: argparse.Namespace,
    ) -> tuple[McpDevToolCall, ...]:
        return (
            McpDevToolCall(
                self.capability.name,
                selected_workflow_tool_arguments(args),
            ),
        )

    async def run_session(
        self,
        session: McpDevStdioSession,
        args: argparse.Namespace,
    ) -> McpDevToolBatchResponse:
        if not args.poll_state:
            return cast(
                McpDevToolBatchResponse,
                await super().run_session(session, args),
            )

        timeout_seconds = self.timeout_seconds(args)
        state_call = McpDevToolCall(
            agent_capabilities.ui_get_state_surface.name,
            plate_manager_state_surface_tool_arguments(
                args,
                selection_mode=args.poll_selection_mode,
            ),
        )
        baseline_result = await call_mcp_tool(session, state_call, timeout_seconds)
        baseline_timed_out = baseline_result.has_only_agent_error_code(
            UiBridgeGatewayTimeoutError.agent_error_code
        )
        workflow_result = await call_mcp_tool(
            session,
            McpDevToolCall(
                self.capability.name,
                selected_workflow_tool_arguments(args),
            ),
            timeout_seconds,
        )
        results = (
            [workflow_result]
            if baseline_timed_out
            else [baseline_result, workflow_result]
        )
        baseline = WorkflowPollBaseline.from_result(baseline_result)
        poll_completed = False
        poll_count = 0
        target_scope_ids = workflow_result_target_scope_ids(workflow_result)
        poll_status = WorkflowPollSummaryStatus.SKIPPED
        transient_poll_error_count = int(baseline_timed_out)
        skip_reason: WorkflowPollSkipReason | None = None
        action_status = workflow_result_action_status(workflow_result)

        if workflow_result_was_accepted(workflow_result):
            operation_id = workflow_result_operation_id(workflow_result)
            if operation_id is None:
                poll_status = WorkflowPollSummaryStatus.FAILED
                skip_reason = WorkflowPollSkipReason.OPERATION_RECEIPT_MISSING
            else:
                receipt_wait_seconds = min(
                    max(args.poll_timeout_seconds, 0.0),
                    120.0,
                )
                receipt_result = await call_mcp_tool(
                    session,
                    McpDevToolCall(
                        agent_capabilities.ui_wait_for_operation_receipt.name,
                        workflow_operation_receipt_tool_arguments(
                            args,
                            operation_id=operation_id,
                        ),
                    ),
                    max(
                        timeout_seconds,
                        receipt_wait_seconds + MCP_TOOL_TIMEOUT_MARGIN_SECONDS,
                    ),
                )
                results.append(receipt_result)
                receipt_status = ui_bridge_operation_result_status(receipt_result)
                if receipt_status is UiBridgeOperationStatus.COMPLETED:
                    (
                        poll_status,
                        poll_completed,
                        poll_count,
                        transient_poll_error_count,
                    ) = await self._poll_workflow_state(
                        session=session,
                        state_call=state_call,
                        timeout_seconds=timeout_seconds,
                        poll_timeout_seconds=args.poll_timeout_seconds,
                        poll_interval_seconds=args.poll_interval_seconds,
                        workflow=args.workflow,
                        target_scope_ids=target_scope_ids,
                        baseline=baseline,
                        results=results,
                        transient_poll_error_count=transient_poll_error_count,
                    )
                elif receipt_status is UiBridgeOperationStatus.RUNNING:
                    poll_status = WorkflowPollSummaryStatus.TIMEOUT
                    skip_reason = WorkflowPollSkipReason.OPERATION_RECEIPT_TIMEOUT
                else:
                    poll_status = WorkflowPollSummaryStatus.FAILED
                    skip_reason = WorkflowPollSkipReason.OPERATION_RECEIPT_FAILED
        else:
            skip_reason = workflow_poll_skip_reason(workflow_result)

        results.append(
            workflow_poll_summary_result(
                workflow=args.workflow,
                status=poll_status,
                poll_requested=True,
                poll_completed=poll_completed,
                poll_count=poll_count,
                target_scope_ids=target_scope_ids,
                skip_reason=skip_reason,
                action_status=action_status,
                transient_poll_error_count=transient_poll_error_count,
            )
        )
        return McpDevToolBatchResponse.from_results(
            session.server_spec,
            tuple(results),
        )

    @staticmethod
    async def _poll_workflow_state(
        *,
        session: McpDevStdioSession,
        state_call: McpDevToolCall,
        timeout_seconds: float,
        poll_timeout_seconds: float,
        poll_interval_seconds: float,
        workflow: str,
        target_scope_ids: tuple[str, ...],
        baseline: WorkflowPollBaseline | None,
        results: list[McpDevToolResult],
        transient_poll_error_count: int,
    ) -> tuple[WorkflowPollSummaryStatus, bool, int, int]:
        """Poll the domain workflow only after its bridge receipt completes."""

        policy = WorkflowStatePollPolicy.from_workflow_text(workflow)
        poll_deadline = asyncio.get_running_loop().time() + poll_timeout_seconds
        poll_count = 0
        terminal_status: WorkflowPollSummaryStatus | None = None
        while True:
            poll_result = await call_mcp_tool(
                session,
                state_call,
                timeout_seconds,
            )
            poll_count += 1
            if poll_result.has_errors():
                if poll_result.has_only_agent_error_code(
                    UiBridgeGatewayTimeoutError.agent_error_code
                ):
                    transient_poll_error_count += 1
                    if asyncio.get_running_loop().time() >= poll_deadline:
                        results.append(poll_result)
                        break
                    await asyncio.sleep(poll_interval_seconds)
                    continue
                results.append(poll_result)
                terminal_status = WorkflowPollSummaryStatus.FAILED
                break
            results.append(poll_result)
            if policy.can_evaluate(poll_result, baseline):
                terminal_status = workflow_poll_terminal_status(
                    poll_result,
                    target_scope_ids=target_scope_ids,
                    policy=policy,
                )
                if terminal_status is not None:
                    break
            if asyncio.get_running_loop().time() >= poll_deadline:
                break
            await asyncio.sleep(poll_interval_seconds)

        status = terminal_status or WorkflowPollSummaryStatus.TIMEOUT
        return (
            status,
            status is WorkflowPollSummaryStatus.COMPLETED,
            poll_count,
            transient_poll_error_count,
        )

    def render_response(
        self,
        payload: JsonObject,
        args: argparse.Namespace,
    ) -> str:
        if args.json or not args.poll_state:
            return super().render_response(payload, args)
        summary = self._poll_summary_payload(payload)
        if summary is None:
            return super().render_response(payload, args)
        lines = [
            f"Workflow: {self._text(summary.get('workflow'))}",
            (
                "Action: "
                f"{self._text(summary.get('action_status'))} "
                f"poll={self._text(summary.get('poll_status'))} "
                f"count={self._text(summary.get('poll_count'))}"
            ),
        ]
        target_scope_ids = summary.get("target_scope_ids")
        if isinstance(target_scope_ids, list) and target_scope_ids:
            lines.append(
                f"Targets: {', '.join(str(value) for value in target_scope_ids)}"
            )
        skip_reason = summary.get("skip_reason")
        if skip_reason is not None:
            lines.append(f"Skip reason: {self._text(skip_reason)}")
        workflow_errors = self._workflow_errors(payload)
        if workflow_errors:
            lines.append("Errors:")
            lines.extend(McpDiagnosticRenderer.error_lines(workflow_errors))

        final_rows = self._final_state_rows(payload)
        if final_rows:
            lines.append("Rows:")
            lines.extend(self._row_lines(final_rows))
        return "\n".join(lines)

    def render_call_response(
        self,
        payload: JsonObject,
        tool_arguments: Mapping[str, JsonValue],
    ) -> str:
        from openhcs.mcp.dev_client_renderers.ui_bridge import UiActionInvokeRenderer

        workflow = optional_str(tool_arguments.get("workflow"))
        return UiActionInvokeRenderer.render(
            payload,
            widget_id=PlateManagerWidgetIdentity.value,
            action_id=workflow,
        )

    @staticmethod
    def _poll_summary_payload(payload: JsonObject) -> Mapping[str, JsonValue] | None:
        for result in SelectedWorkflowCommandSpec._result_mappings(payload):
            if result.get("tool") != "mcp_dev_selected_workflow_poll":
                continue
            first_payload = SelectedWorkflowCommandSpec._first_payload(result)
            if first_payload is not None:
                return first_payload
        return None

    @staticmethod
    def _final_state_rows(payload: JsonObject) -> tuple[Mapping[str, JsonValue], ...]:
        state_payload: Mapping[str, JsonValue] | None = None
        for result in SelectedWorkflowCommandSpec._result_mappings(payload):
            if result.get("tool") == agent_capabilities.ui_get_state_surface.name:
                state_payload = SelectedWorkflowCommandSpec._first_payload(result)
        if state_payload is None:
            return ()
        nested_payload = state_payload.get("payload")
        if not isinstance(nested_payload, Mapping):
            return ()
        rows = nested_payload.get("rows")
        if not isinstance(rows, list):
            return ()
        return tuple(row for row in rows if isinstance(row, Mapping))

    @staticmethod
    def _result_mappings(payload: JsonObject) -> tuple[Mapping[str, JsonValue], ...]:
        results = payload.get("results")
        if not isinstance(results, list):
            return ()
        return tuple(result for result in results if isinstance(result, Mapping))

    @staticmethod
    def _first_payload(
        result: Mapping[str, JsonValue],
    ) -> Mapping[str, JsonValue] | None:
        payloads = result.get("payloads")
        if not isinstance(payloads, list) or not payloads:
            return None
        first_payload = payloads[0]
        if not isinstance(first_payload, Mapping):
            return None
        return first_payload

    @staticmethod
    def _workflow_errors(payload: JsonObject) -> tuple[Mapping[str, JsonValue], ...]:
        for result in SelectedWorkflowCommandSpec._result_mappings(payload):
            if result.get("tool") != agent_capabilities.ui_selected_plate_workflow.name:
                continue
            first_payload = SelectedWorkflowCommandSpec._first_payload(result)
            if first_payload is None:
                continue
            errors = first_payload.get("errors")
            if isinstance(errors, list):
                return tuple(error for error in errors if isinstance(error, Mapping))
            action_result = first_payload.get("action_result")
            if isinstance(action_result, Mapping):
                action_errors = action_result.get("errors")
                if isinstance(action_errors, list):
                    return tuple(
                        error for error in action_errors if isinstance(error, Mapping)
                    )
        return ()

    @classmethod
    def _row_lines(cls, rows: tuple[Mapping[str, JsonValue], ...]) -> list[str]:
        lines: list[str] = []
        for row in rows:
            state_parts = [
                f"state={cls._text(row.get('orchestrator_state'))}",
                f"status={cls._quoted_text(row.get('status_prefix'))}",
                f"terminal={cls._text(row.get('terminal_status'))}",
            ]
            if row.get("selected") is True:
                state_parts.append("selected=True")
            lines.append(f"- {cls._text(row.get('name'))}: " + ", ".join(state_parts))
        return lines

    @staticmethod
    def _quoted_text(value: JsonValue) -> str:
        if value is None:
            return "<none>"
        return json.dumps(str(value))

    @staticmethod
    def _text(value: JsonValue) -> str:
        if value is None:
            return "<none>"
        return str(value)


class CodeDocumentCommandSpec(CapabilityBackedCommandSpec):
    capability = agent_capabilities.ui_get_code_document

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("document_id")
        parser.add_argument("--selection-mode", default="selected")
        cleanliness = parser.add_mutually_exclusive_group()
        cleanliness.add_argument(
            "--clean",
            dest="clean",
            action="store_true",
            default=True,
            help="Read sparse clean source.",
        )
        cleanliness.add_argument(
            "--full",
            dest="clean",
            action="store_false",
            help="Read full resolved source, including defaults and inherited values.",
        )
        parser.add_argument(
            "--no-source",
            action="store_true",
            help="Only render document metadata, revision, and snapshot information.",
        )
        parser.add_argument(
            "--max-source-chars",
            type=int,
            default=DEFAULT_CODE_DOCUMENT_MAX_CHARS,
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Render the complete MCP JSON response instead of a compact summary.",
        )
        add_ui_connection_options(parser)

    def calls_from_args(
        self,
        args: argparse.Namespace,
    ) -> tuple[McpDevToolCall, ...]:
        return (
            McpDevToolCall(
                self.capability.name,
                {
                    "document_id": args.document_id,
                    "selection_mode": args.selection_mode,
                    "clean": args.clean,
                    "connection": ui_connection_arguments(
                        args,
                        timeout_ms=args.timeout_ms,
                    ),
                },
            ),
        )

    def renderer_options(
        self,
        args: argparse.Namespace,
    ) -> CodeDocumentRenderOptions:
        return CodeDocumentRenderOptions(
            include_source=not args.no_source,
            max_source_chars=args.max_source_chars,
        )

    def call_render_args(
        self,
        tool_arguments: Mapping[str, JsonValue],
    ) -> argparse.Namespace:
        del tool_arguments
        return argparse.Namespace(
            json=False,
            no_source=False,
            max_source_chars=DEFAULT_CODE_DOCUMENT_MAX_CHARS,
        )


class ValidateCodeDocumentCommandSpec(CapabilityBackedCommandSpec):
    capability = agent_capabilities.ui_validate_code_document

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("document_id")
        add_code_document_source_options(parser)
        parser.add_argument("--base-revision-token")
        parser.add_argument(
            "--json",
            action="store_true",
            help="Render the complete MCP JSON response instead of a compact summary.",
        )
        add_ui_connection_options(parser)

    def calls_from_args(
        self,
        args: argparse.Namespace,
    ) -> tuple[McpDevToolCall, ...]:
        return (
            McpDevToolCall(
                self.capability.name,
                {
                    "document_id": args.document_id,
                    "source": code_document_source_from_args(args),
                    "base_revision_token": args.base_revision_token,
                    "connection": ui_connection_arguments(
                        args,
                        timeout_ms=args.timeout_ms,
                    ),
                },
            ),
        )


class ApplyCodeDocumentCommandSpec(CapabilityBackedCommandSpec):
    capability = agent_capabilities.ui_apply_code_document

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("document_id")
        add_code_document_source_options(parser)
        parser.add_argument("--base-revision-token", required=True)
        parser.add_argument(
            "--no-confirmation",
            action="store_true",
            help="Set require_confirmation=False and allow the mutation to proceed.",
        )
        parser.add_argument("--snapshot-label")
        parser.add_argument("--apply-if-time-traveling", action="store_true")
        parser.add_argument("--request-token")
        parser.add_argument(
            "--json",
            action="store_true",
            help="Render the complete MCP JSON response instead of a compact summary.",
        )
        add_ui_connection_options(parser)

    def calls_from_args(
        self,
        args: argparse.Namespace,
    ) -> tuple[McpDevToolCall, ...]:
        return (
            McpDevToolCall(
                self.capability.name,
                {
                    "document_id": args.document_id,
                    "source": code_document_source_from_args(args),
                    "base_revision_token": args.base_revision_token,
                    "require_confirmation": not args.no_confirmation,
                    "snapshot_label": args.snapshot_label,
                    "apply_if_time_traveling": args.apply_if_time_traveling,
                    "request_token": args.request_token,
                    "connection": ui_connection_arguments(
                        args,
                        timeout_ms=args.timeout_ms,
                    ),
                },
            ),
        )


class ActionsCommandSpec(CapabilityBackedCommandSpec):
    capability = agent_capabilities.ui_list_actions

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "widget_id",
            nargs="?",
            help="Optional widget id filter, for example plate_manager.",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Render the complete MCP JSON response instead of a compact summary.",
        )
        add_ui_connection_options(parser)

    def calls_from_args(
        self,
        args: argparse.Namespace,
    ) -> tuple[McpDevToolCall, ...]:
        return (
            McpDevToolCall(
                self.capability.name,
                {
                    "connection": ui_connection_arguments(
                        args,
                        timeout_ms=args.timeout_ms,
                    ),
                },
            ),
        )

    def renderer_options(
        self,
        args: argparse.Namespace,
    ) -> UiActionCatalogRenderOptions:
        return UiActionCatalogRenderOptions(widget_id=args.widget_id)

    def call_render_args(
        self,
        tool_arguments: Mapping[str, JsonValue],
    ) -> argparse.Namespace:
        return argparse.Namespace(
            json=False,
            widget_id=optional_str(tool_arguments.get("widget_id")),
        )


class InvokeActionCommandSpec(CapabilityBackedCommandSpec):
    capability = agent_capabilities.ui_invoke_action

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("widget_id")
        parser.add_argument("action_id")
        parser.add_argument(
            "--target-scope-id",
            action="append",
            default=[],
            help="Target scope id from actions output; repeat for multiple targets.",
        )
        parser.add_argument("--observed-selection-revision-token")
        parser.add_argument("--request-token")
        parser.add_argument(
            "--no-confirmation",
            action="store_true",
            help="Set require_confirmation=False and allow confirmed actions to proceed.",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Render the complete MCP JSON response instead of a compact summary.",
        )
        add_ui_connection_options(parser)

    def calls_from_args(
        self,
        args: argparse.Namespace,
    ) -> tuple[McpDevToolCall, ...]:
        return (
            McpDevToolCall(
                self.capability.name,
                {
                    "widget_id": args.widget_id,
                    "action_id": args.action_id,
                    "target_scope_ids": args.target_scope_id,
                    "observed_selection_revision_token": (
                        args.observed_selection_revision_token
                    ),
                    "request_token": args.request_token,
                    "require_confirmation": not args.no_confirmation,
                    "connection": ui_connection_arguments(
                        args,
                        timeout_ms=args.timeout_ms,
                    ),
                },
            ),
        )

    def renderer_options(
        self,
        args: argparse.Namespace,
    ) -> UiActionInvokeRenderOptions:
        return UiActionInvokeRenderOptions(
            widget_id=args.widget_id,
            action_id=args.action_id,
        )

    def call_render_args(
        self,
        tool_arguments: Mapping[str, JsonValue],
    ) -> argparse.Namespace:
        return argparse.Namespace(
            json=False,
            widget_id=optional_str(tool_arguments.get("widget_id")),
            action_id=optional_str(tool_arguments.get("action_id")),
        )


class InvokeWidgetActionCommandSpec(CapabilityBackedCommandSpec):
    capability = agent_capabilities.ui_invoke_widget_action

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("window_id")
        parser.add_argument("path_id")
        parser.add_argument("--action-kind", default="auto")
        parser.add_argument("--target-index", type=int)
        parser.add_argument("--create-if-missing", action="store_true")
        parser.add_argument("--request-token")
        parser.add_argument(
            "--json",
            action="store_true",
            help="Render the complete MCP JSON response instead of a compact summary.",
        )
        add_ui_connection_options(parser)

    def calls_from_args(
        self,
        args: argparse.Namespace,
    ) -> tuple[McpDevToolCall, ...]:
        return (
            McpDevToolCall(
                self.capability.name,
                {
                    "window_id": args.window_id,
                    "path_id": args.path_id,
                    "action_kind": args.action_kind,
                    "target_index": args.target_index,
                    "create_if_missing": args.create_if_missing,
                    "request_token": args.request_token,
                    "connection": ui_connection_arguments(
                        args,
                        timeout_ms=args.timeout_ms,
                    ),
                },
            ),
        )


class WidgetTreeCommandSpec(CapabilityBackedCommandSpec):
    capability = agent_capabilities.ui_get_widget_tree

    @staticmethod
    def _effective_max_depth(args: argparse.Namespace) -> int | None:
        if args.max_depth is not None:
            return args.max_depth
        return 8

    @staticmethod
    def _effective_max_nodes(args: argparse.Namespace) -> int | None:
        if args.max_nodes is not None:
            return args.max_nodes
        return 800 if args.output == WidgetTreeOutputFormat.OUTLINE.value else 40

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("window_id")
        parser.add_argument("--maximum-text-length", type=int, default=120)
        parser.add_argument("--maximum-item-model-nodes", type=int, default=512)
        parser.add_argument("--max-depth", type=int)
        parser.add_argument("--max-nodes", type=int)
        parser.add_argument("--create-if-missing", action="store_true")
        parser.add_argument(
            "--output",
            choices=WidgetTreeOutputFormat.choices(),
            default=WidgetTreeOutputFormat.OUTLINE.value,
            help="Render JSON or a clean indented widget outline.",
        )
        parser.add_argument(
            "--json",
            dest="output",
            action="store_const",
            const=WidgetTreeOutputFormat.JSON.value,
            help="Alias for --output json.",
        )
        parser.add_argument(
            "--outline-root-class",
            help="When rendering outline output, start at the first node with this Qt class.",
        )
        parser.add_argument(
            "--include-technical-widgets",
            action="store_true",
            help="Include Qt infrastructure nodes such as scrollbars in outline output.",
        )
        parser.add_argument(
            "--include-non-actionable",
            action="store_true",
            help="Include non-actionable widget branches instead of actionable paths only.",
        )
        parser.add_argument(
            "--actionable-only",
            action="store_true",
            help="Return only actionable widget paths.",
        )
        parser.add_argument(
            "--include-tree",
            action="store_true",
            help="Include the bounded nested widget tree in addition to actionable summaries.",
        )
        parser.add_argument(
            "--full-actions",
            action="store_true",
            help="Return full UiWidgetActionSummary rows instead of compact action rows.",
        )
        add_ui_connection_options(parser)

    def calls_from_args(
        self,
        args: argparse.Namespace,
    ) -> tuple[McpDevToolCall, ...]:
        return (
            McpDevToolCall(
                self.capability.name,
                {
                    "window_id": args.window_id,
                    "create_if_missing": args.create_if_missing,
                    "maximum_text_length": args.maximum_text_length,
                    "maximum_item_model_nodes": args.maximum_item_model_nodes,
                    "actionable_only": (
                        True
                        if args.actionable_only
                        else (
                            False
                            if args.output == "outline"
                            else not args.include_non_actionable
                        )
                    ),
                    "include_tree": args.include_tree or args.output == "outline",
                    "max_depth": self._effective_max_depth(args),
                    "max_nodes": self._effective_max_nodes(args),
                    "compact_actions": not args.full_actions,
                    "connection": ui_connection_arguments(
                        args,
                        timeout_ms=args.timeout_ms,
                    ),
                },
            ),
        )

    def renderer_options(
        self,
        args: argparse.Namespace,
    ) -> WidgetTreeRenderOptions:
        return WidgetTreeRenderOptions(
            output=WidgetTreeOutputFormat(args.output),
            outline_root_class=args.outline_root_class,
            include_technical_widgets=args.include_technical_widgets,
        )

    def call_render_args(
        self,
        tool_arguments: Mapping[str, JsonValue],
    ) -> argparse.Namespace:
        del tool_arguments
        return argparse.Namespace(
            output=WidgetTreeOutputFormat.OUTLINE.value,
            outline_root_class=None,
            include_technical_widgets=False,
        )


class WindowSnapshotCommandSpec(CapabilityBackedCommandSpec):
    capability = agent_capabilities.ui_snapshot_window

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("window_id")
        parser.add_argument(
            "--capture-scope",
            choices=tuple(scope.value for scope in WindowSnapshotCaptureScope),
            default=WindowSnapshotCaptureScope.WINDOW.value,
        )
        parser.add_argument("--create-if-missing", action="store_true")
        parser.add_argument(
            "--json",
            action="store_true",
            help="Render the complete MCP JSON response instead of a compact summary.",
        )
        add_ui_connection_options(parser)

    def calls_from_args(
        self,
        args: argparse.Namespace,
    ) -> tuple[McpDevToolCall, ...]:
        return (
            McpDevToolCall(
                self.capability.name,
                {
                    "window_id": args.window_id,
                    "capture_scope": args.capture_scope,
                    "create_if_missing": args.create_if_missing,
                    "connection": ui_connection_arguments(
                        args,
                        timeout_ms=args.timeout_ms,
                    ),
                },
            ),
        )


class ObjectStateScopesCommandSpec(CapabilityBackedCommandSpec):
    capability = agent_capabilities.ui_list_object_state_scopes

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "scope_ids",
            nargs="*",
            help="Optional ObjectState scope id(s); equivalent to repeated --scope-id.",
        )
        parser.add_argument(
            "--scope-id",
            action="append",
            default=[],
            help="ObjectState scope id; repeat to inspect multiple scopes.",
        )
        parser.add_argument("--include-system-scopes", action="store_true")
        parser.add_argument("--include-fields", action="store_true")
        add_object_state_field_filter_options(parser)
        parser.add_argument("--include-field-values", action="store_true")
        parser.add_argument(
            "--field-limit",
            "--max-fields",
            dest="field_limit",
            type=int,
            default=200,
        )
        parser.add_argument("--field-offset", type=int, default=0)
        parser.add_argument(
            "--json",
            action="store_true",
            help="Render the complete MCP JSON response instead of a compact summary.",
        )
        add_ui_connection_options(parser)

    def calls_from_args(
        self,
        args: argparse.Namespace,
    ) -> tuple[McpDevToolCall, ...]:
        scope_ids = list(dict.fromkeys([*args.scope_ids, *args.scope_id]))
        return (
            McpDevToolCall(
                self.capability.name,
                {
                    "scope_ids": scope_ids,
                    "include_system_scopes": args.include_system_scopes,
                    "include_fields": args.include_fields,
                    "field_filter": args.field_filter,
                    "include_field_values": args.include_field_values,
                    "field_limit": args.field_limit,
                    "field_offset": args.field_offset,
                    "connection": ui_connection_arguments(
                        args,
                        timeout_ms=args.timeout_ms,
                    ),
                },
            ),
        )


class ObjectStateFieldsCommandSpec(CapabilityBackedCommandSpec):
    capability = agent_capabilities.ui_get_object_state_fields

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "scope_ids",
            nargs="*",
            help="Optional ObjectState scope id(s); equivalent to repeated --scope-id.",
        )
        parser.add_argument(
            "--scope-id",
            action="append",
            default=[],
            help="ObjectState scope id; repeat to inspect multiple scopes.",
        )
        parser.add_argument(
            "--contains",
            "--query",
            "--field-path-contains",
            "--path-contains",
            action="append",
            default=[],
            help="Case-insensitive field path substring; repeat to OR-match terms.",
        )
        parser.add_argument(
            "--field-path",
            "--path",
            dest="field_path",
            action="append",
            default=[],
            help="Exact ObjectState field path; repeat to inspect multiple fields.",
        )
        parser.add_argument("--include-system-scopes", action="store_true")
        add_object_state_field_filter_options(parser)
        parser.add_argument(
            "--include-field-values",
            "--include-values",
            dest="include_field_values",
            action="store_true",
        )
        parser.add_argument(
            "--include-container-fields",
            action="store_true",
            help=(
                "Include parent dataclass/container fields when searching by "
                "--contains. Exact --field-path matches are always returned."
            ),
        )
        parser.add_argument("--field-limit", type=int, default=200)
        parser.add_argument("--field-offset", type=int, default=0)
        parser.add_argument("--max-fields", type=int, default=100)
        parser.add_argument("--max-value-items", type=int, default=20)
        parser.add_argument("--max-value-chars", type=int, default=1000)
        parser.add_argument(
            "--json",
            action="store_true",
            help="Render the complete MCP JSON response instead of a compact summary.",
        )
        add_ui_connection_options(parser)

    def calls_from_args(
        self,
        args: argparse.Namespace,
    ) -> tuple[McpDevToolCall, ...]:
        scope_ids = list(dict.fromkeys([*args.scope_ids, *args.scope_id]))
        return (
            McpDevToolCall(
                self.capability.name,
                {
                    "scope_ids": scope_ids,
                    "field_paths": args.field_path,
                    "field_path_contains": args.contains,
                    "include_system_scopes": args.include_system_scopes,
                    "include_clean_fields": (
                        args.field_filter == UiObjectStateFieldFilter.ALL.value
                    ),
                    "include_container_fields": args.include_container_fields,
                    "field_filter": args.field_filter,
                    "include_field_values": args.include_field_values,
                    "field_limit": args.field_limit,
                    "field_offset": args.field_offset,
                    "max_fields": args.max_fields,
                    "max_value_items": args.max_value_items,
                    "max_value_chars": args.max_value_chars,
                    "connection": ui_connection_arguments(
                        args,
                        timeout_ms=args.timeout_ms,
                    ),
                },
            ),
        )


class ObjectStateFieldHelpCommandSpec(CapabilityBackedCommandSpec):
    capability = agent_capabilities.ui_describe_object_state_field

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "scope_id",
            nargs="?",
            help=(
                "ObjectState scope id when two positional arguments are "
                "provided; otherwise treated as field_path."
            ),
        )
        parser.add_argument(
            "field_path",
            nargs="?",
            help="Dotted ObjectState field path.",
        )
        parser.add_argument(
            "--scope-id",
            dest="scope_id_option",
            help="Optional scope id; omit when field path is unique.",
        )
        parser.add_argument(
            "--field-path",
            "--path",
            dest="field_path_option",
        )
        parser.add_argument("--window-id")
        parser.add_argument("--max-description-chars", type=int, default=4_000)
        parser.add_argument(
            "--json",
            action="store_true",
            help="Render the complete MCP JSON response instead of a compact summary.",
        )
        add_ui_connection_options(parser)

    def calls_from_args(
        self,
        args: argparse.Namespace,
    ) -> tuple[McpDevToolCall, ...]:
        scope_id = args.scope_id_option or args.scope_id
        field_path = args.field_path_option or args.field_path
        if field_path is None and args.scope_id is not None:
            field_path = args.scope_id
            if args.scope_id_option is None:
                scope_id = None
        if not field_path:
            raise McpDevCliUsageError("object-state-field-help requires a field path.")
        arguments: dict[str, JsonValue] = {
            "field_path": field_path,
            "window_id": args.window_id,
            "max_description_chars": args.max_description_chars,
            "connection": ui_connection_arguments(
                args,
                timeout_ms=args.timeout_ms,
            ),
        }
        if scope_id:
            arguments["object_state_scope_id"] = scope_id
        return (
            McpDevToolCall(
                self.capability.name,
                arguments,
            ),
        )


class ObjectStateSetCommandSpec(CapabilityBackedCommandSpec):
    capability = agent_capabilities.ui_mutate_object_state_field

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("scope_id", nargs="?")
        parser.add_argument("field_path", nargs="?")
        parser.add_argument("--scope-id", dest="scope_id_option")
        parser.add_argument(
            "--field-path",
            "--path",
            dest="field_path_option",
        )
        parser.add_argument(
            "--value",
            help="JSON scalar/container value; non-JSON text is sent as a string.",
        )
        parser.add_argument(
            "--reset",
            action="store_true",
            help="Reset the field through ObjectState.reset_parameter.",
        )
        parser.add_argument("--window-id")
        parser.add_argument("--request-token")
        parser.add_argument(
            "--no-field-values",
            dest="include_field_values",
            action="store_false",
            default=True,
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Render the complete MCP JSON response instead of a compact summary.",
        )
        add_ui_connection_options(parser)

    def calls_from_args(
        self,
        args: argparse.Namespace,
    ) -> tuple[McpDevToolCall, ...]:
        scope_id = args.scope_id_option or args.scope_id
        field_path = args.field_path_option or args.field_path
        if not scope_id:
            raise McpDevCliUsageError("object-state-set requires a scope id.")
        if not field_path:
            raise McpDevCliUsageError("object-state-set requires a field path.")
        if args.reset and args.value is not None:
            raise McpDevCliUsageError("--reset cannot be combined with --value.")
        if not args.reset and args.value is None:
            raise McpDevCliUsageError("object-state-set requires --value or --reset.")
        return (
            McpDevToolCall(
                self.capability.name,
                {
                    "object_state_scope_id": scope_id,
                    "field_path": field_path,
                    "value": (
                        None if args.value is None else parse_cli_json_value(args.value)
                    ),
                    "reset": args.reset,
                    "window_id": args.window_id,
                    "include_field_values": args.include_field_values,
                    "request_token": args.request_token,
                    "connection": ui_connection_arguments(
                        args,
                        timeout_ms=args.timeout_ms,
                    ),
                },
            ),
        )


class UiSmokeCommandSpec(UiBridgeCommandSpec):
    command = "ui-smoke"
    help = "Call health plus UI bridge status, bridge list, and window list."

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        super().configure_parser(parser)
        parser.add_argument(
            "--json",
            action="store_true",
            help="Render the complete MCP JSON response instead of a compact summary.",
        )

    def calls_from_args(
        self,
        args: argparse.Namespace,
    ) -> tuple[McpDevToolCall, ...]:
        connection_arguments = ui_tool_arguments(args, timeout_ms=args.timeout_ms)
        return (
            McpDevToolCall(agent_capabilities.health_check.name, {}),
            McpDevToolCall(
                agent_capabilities.ui_bridge_status.name, connection_arguments
            ),
            McpDevToolCall(agent_capabilities.ui_list_bridges.name, {}),
            McpDevToolCall(
                agent_capabilities.ui_list_windows.name, connection_arguments
            ),
        )

    def render_response(
        self,
        payload: JsonObject,
        args: argparse.Namespace,
    ) -> str:
        if args.json:
            return super().render_response(payload, args)
        from openhcs.mcp.dev_client_renderers.ui_bridge import UiSmokeRenderer

        return UiSmokeRenderer.render(payload)
