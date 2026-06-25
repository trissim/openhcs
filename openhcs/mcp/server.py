"""MCP server adapter for the OpenHCS agent API."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import wraps
from inspect import getsourcefile
from pathlib import Path
from typing import ClassVar, Self

from openhcs.agent.capabilities import get_capability_registry
from openhcs.agent.dto.common import (
    AgentError,
    JsonValue,
    SCHEMA_VERSION,
)
import openhcs.agent.dto.execution as agent_execution_dto
import openhcs.agent.dto.mcp as agent_mcp_dto
import openhcs.agent.dto.ui_bridge as agent_ui_bridge_dto
import openhcs.agent.dto.viewer as agent_viewer_dto
import openhcs.agent.services.ui_bridge_transport as ui_bridge_transport
from openhcs.agent.dto.config import ConfigPatch
from openhcs.agent.dto.execution import ExecutionConnectionSpec
from openhcs.agent.dto.mcp import McpServerHealthResult
from openhcs.agent.dto.ui_bridge import (
    UiActionInvokeRequest,
    UiBranchSwitchRequest,
    UiBridgeConfirmationRequirement,
    UiBridgeConnectionRequest,
    UiBridgeConnectionSpec,
    UiCodeDocumentApplyRequest,
    UiCodeDocumentRequest,
    UiCodeDocumentValidationRequest,
    UiMutationRequestToken,
    UiObjectStateFieldListOptions,
    UiObjectStateScopeListRequest,
    UiObjectStateScopeVisibility,
    UiSelectedPlateWorkflowKind,
    UiSelectedPlateWorkflowRequest,
    UiSnapshotListRequest,
    UiSnapshotRestoreRequest,
    UiStateSurfaceRequest,
    UiTimeTravelHeadRequest,
    UiWindowCloseRequest,
    UiWindowFocusRequest,
    UiWindowNavigateRequest,
    UiWindowOpenPolicy,
    UiWindowSnapshotRequest,
    UiWidgetTreeRequest,
)
from openhcs.agent.dto.viewer import (
    ViewerWindowPayloadRequest,
    ViewerWindowSnapshotRequest,
    ViewerWindowStateRequest,
    ViewerWindowValidationPolicy,
    ViewerWindowValidationRequest,
)
from openhcs.agent.serialization import to_jsonable
from openhcs.agent.services.execution_session_service import (
    PycodifiedPipelineSessionRequest,
)
from openhcs.core.selection import SelectedScopeIdsArgument
from openhcs.mcp.context import (
    OPENHCS_AGENT_CONTEXT_SOURCE_TYPES,
    OpenHCSAgentContext,
    create_agent_context,
)
import openhcs.runtime.viewer_protocol as runtime_viewer_protocol
import openhcs.runtime.window_snapshot as runtime_window_snapshot
from openhcs.runtime.window_snapshot import (
    WindowSnapshotCaptureScope,
)
from openhcs.runtime.viewer_protocol import (
    ViewerPayloadControlOptions,
)
from openhcs.runtime.zmq_execution_signature import ZMQExecutionIdentity


DEFAULT_MCP_WINDOW_SNAPSHOT_DIR = Path("/tmp/openhcs-mcp-window-snapshots")
DEFAULT_MCP_CONTROL_TIMEOUT_MS = 750
MAX_MCP_CONTROL_TIMEOUT_MS = 2_000
MIN_MCP_CONTROL_TIMEOUT_MS = 1


def _source_path_for_type(source_type: type) -> Path:
    source_file = getsourcefile(source_type)
    if source_file is None:
        raise RuntimeError(f"No source file available for {source_type.__qualname__}")
    return Path(source_file).resolve()


def _deduplicate_source_paths(source_paths: tuple[Path, ...]) -> tuple[Path, ...]:
    return tuple(dict.fromkeys(source_paths))


MCP_SERVER_SOURCE_PATHS = _deduplicate_source_paths(
    (
        Path(__file__).resolve(),
        Path(create_agent_context.__code__.co_filename).resolve(),
        Path(get_capability_registry.__code__.co_filename).resolve(),
        Path(to_jsonable.__code__.co_filename).resolve(),
        Path(agent_execution_dto.__file__).resolve(),
        Path(agent_mcp_dto.__file__).resolve(),
        Path(agent_ui_bridge_dto.__file__).resolve(),
        Path(agent_viewer_dto.__file__).resolve(),
        Path(ui_bridge_transport.__file__).resolve(),
        Path(runtime_viewer_protocol.__file__).resolve(),
        Path(runtime_window_snapshot.__file__).resolve(),
        *tuple(
            _source_path_for_type(source_type)
            for source_type in OPENHCS_AGENT_CONTEXT_SOURCE_TYPES
        ),
    )
)
MCP_SERVER_IMPORT_SOURCE_MTIMES_NS = {
    source_path: source_path.stat().st_mtime_ns
    for source_path in MCP_SERVER_SOURCE_PATHS
}
MCP_SERVER_SOURCE_PATH = MCP_SERVER_SOURCE_PATHS[0]
MCP_SERVER_IMPORT_MTIME_NS = MCP_SERVER_IMPORT_SOURCE_MTIMES_NS[MCP_SERVER_SOURCE_PATH]
MCP_SERVER_PROCESS_ID = os.getpid()
MCP_SERVER_IMPORTED_AT_UNIX = time.time()


def _mcp_server_current_source_mtime_ns() -> int:
    return MCP_SERVER_SOURCE_PATH.stat().st_mtime_ns


def _mcp_server_stale_source_paths() -> tuple[Path, ...]:
    return tuple(
        source_path
        for source_path, import_mtime_ns in MCP_SERVER_IMPORT_SOURCE_MTIMES_NS.items()
        if source_path.stat().st_mtime_ns != import_mtime_ns
    )


def _mcp_server_source_changed_since_import() -> bool:
    return bool(_mcp_server_stale_source_paths())


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

    @server.resource("openhcs://capabilities")
    def capabilities_resource() -> dict:
        """Return the canonical OpenHCS agent capability registry."""
        return to_jsonable(get_capability_registry())

    @server.resource("openhcs://architecture/topics")
    def architecture_topics_resource() -> dict:
        """List source-backed architecture topics available to agents."""
        return to_jsonable(ctx.architecture_service.list_topics())

    @openhcs_tool(allow_stale_server=True)
    def openhcs_health_check() -> dict:
        """Report MCP health, process identity, and source freshness."""
        current_source_mtime_ns = _mcp_server_current_source_mtime_ns()
        stale_source_paths = _mcp_server_stale_source_paths()
        return to_jsonable(
            McpServerHealthResult(
                schema_version=SCHEMA_VERSION,
                status="ok",
                started_at_unix=MCP_SERVER_IMPORTED_AT_UNIX,
                service="openhcs.mcp",
                server_process_id=MCP_SERVER_PROCESS_ID,
                server_source_path=str(MCP_SERVER_SOURCE_PATH),
                server_import_mtime_ns=MCP_SERVER_IMPORT_MTIME_NS,
                server_current_mtime_ns=current_source_mtime_ns,
                server_source_changed_since_import=bool(stale_source_paths),
                stale_source_paths=tuple(
                    str(source_path)
                    for source_path in stale_source_paths
                ),
            )
        )

    @openhcs_tool()
    def openhcs_list_capabilities() -> dict:
        """List MCP resources/tools and their OpenHCS agent API contracts."""
        return to_jsonable(get_capability_registry())

    @openhcs_tool()
    def openhcs_search_functions(
        query: str | None = None,
        library: str | None = None,
        limit: int = 50,
        compact_signatures: bool = True,
    ) -> dict:
        """Search registered OpenHCS processing functions by name, library, tag, or docs."""
        return to_jsonable(
            ctx.function_catalog.search(
                query=query,
                library=library,
                limit=limit,
                compact_signatures=compact_signatures,
            )
        )

    @openhcs_tool()
    def openhcs_describe_function(function_id: str) -> dict:
        """Return full signature, parameter, and documentation details for one function."""
        return to_jsonable(ctx.function_catalog.get(function_id))

    @openhcs_tool()
    def openhcs_get_authoring_context(kind: str = "pipeline") -> dict:
        """Return bounded guidance for authoring OpenHCS pipelines or functions."""
        return to_jsonable(ctx.authoring_context_service.get_authoring_context(kind))

    @openhcs_tool()
    def openhcs_list_architecture_topics() -> dict:
        """List architecture topics that explain OpenHCS internals through stable DTOs."""
        return to_jsonable(ctx.architecture_service.list_topics())

    @openhcs_tool()
    def openhcs_explain_architecture(
        topic_id: str = "pipeline_model",
    ) -> dict:
        """Explain one OpenHCS architecture topic with source-backed internal symbols."""
        return to_jsonable(ctx.architecture_service.explain_topic(topic_id))

    @openhcs_tool()
    def openhcs_describe_internal_symbol(symbol_id: str) -> dict:
        """Describe a projected internal OpenHCS symbol without exposing live objects."""
        return to_jsonable(ctx.architecture_service.describe_internal_symbol(symbol_id))

    @openhcs_tool()
    def openhcs_describe_config_schema(config_type: str) -> dict:
        """Reflect GlobalPipelineConfig or PipelineConfig fields for safe config patches."""
        return to_jsonable(ctx.config_service.describe_schema(config_type))

    @openhcs_tool()
    def openhcs_create_config(
        config_type: str,
        values: dict | None = None,
    ) -> dict:
        """Create an in-memory OpenHCS config draft from a config patch."""
        patch = ConfigPatch(
            config_type=config_type, values=_json_object_or_empty(values)
        )
        return to_jsonable(ctx.config_service.create(config_type, patch))

    @openhcs_tool()
    def openhcs_validate_config_patch(
        config_type: str,
        values: dict | None = None,
    ) -> dict:
        """Validate that values can instantiate the requested OpenHCS config type."""
        patch = ConfigPatch(
            config_type=config_type, values=_json_object_or_empty(values)
        )
        return to_jsonable(ctx.config_service.validate_patch(config_type, patch))

    @openhcs_tool()
    def openhcs_render_config_source(
        config_id: str,
        clean: bool = True,
    ) -> dict:
        """Render an in-memory config draft as reviewable Python source."""
        return to_jsonable(ctx.config_service.render_source(config_id, clean=clean))

    @openhcs_tool()
    def openhcs_create_pipeline() -> dict:
        """Create an empty in-memory OpenHCS pipeline draft."""
        return to_jsonable(ctx.pipeline_service.create_pipeline())

    @openhcs_tool()
    def openhcs_add_function_step(
        pipeline_id: str,
        function_id: str,
        name: str | None = None,
        kwargs: dict | None = None,
        step_id: str | None = None,
        description: str | None = None,
        enabled: bool = True,
        debug_pause: bool = False,
        index: int | None = None,
    ) -> dict:
        """Add a registry-backed FunctionStep to an in-memory pipeline draft."""
        step_spec = ctx.pipeline_service.make_step_spec(
            function_id=function_id,
            name=name,
            kwargs=kwargs,
            step_id=step_id,
            description=description,
            enabled=enabled,
            debug_pause=debug_pause,
        )
        return to_jsonable(
            ctx.pipeline_service.add_step(
                pipeline_id,
                step_spec,
                index=index,
            )
        )

    @openhcs_tool()
    def openhcs_validate_pipeline(pipeline_id: str) -> dict:
        """Validate an in-memory pipeline draft against OpenHCS FunctionStep semantics."""
        return to_jsonable(ctx.pipeline_service.validate(pipeline_id))

    @openhcs_tool()
    def openhcs_render_pipeline_source(
        pipeline_id: str,
        clean: bool = True,
    ) -> dict:
        """Render an in-memory pipeline draft as reviewable Python source."""
        return to_jsonable(ctx.pipeline_service.render_source(pipeline_id, clean=clean))

    @openhcs_tool()
    def openhcs_create_orchestrator_session(
        plate_path: str,
        pipeline_id: str,
        execution_plate_path: str | None = None,
        selected_pipeline_path: str | None = None,
        global_config_id: str | None = None,
        pipeline_config_id: str | None = None,
        host: str = "localhost",
        port: int | None = None,
        transport_mode: str | None = None,
        persistent: bool = True,
    ) -> dict:
        """Create an opaque OpenHCS execution session for a plate and pipeline draft."""
        return to_jsonable(
            ctx.execution_service.create_session(
                plate_path=plate_path,
                pipeline_id=pipeline_id,
                execution_plate_path=execution_plate_path,
                selected_pipeline_path=selected_pipeline_path,
                global_config_id=global_config_id,
                pipeline_config_id=pipeline_config_id,
                connection=ExecutionConnectionSpec(
                    host=host,
                    port=port,
                    transport_mode=transport_mode,
                    persistent=persistent,
                ),
            )
        )

    @openhcs_tool()
    def openhcs_create_orchestrator_session_from_pipeline_source(
        plate_path: str,
        pipeline_source: str,
        global_config_id: str | None = None,
        pipeline_config_id: str | None = None,
        host: str = "localhost",
        port: int | None = None,
        transport_mode: str | None = None,
        persistent: bool = True,
    ) -> dict:
        """Create an execution session from pycodified OpenHCS pipeline source."""
        return to_jsonable(
            ctx.execution_service.create_session_from_pipeline_source(
                PycodifiedPipelineSessionRequest(
                    identity=ZMQExecutionIdentity(
                        plate_id=plate_path,
                    ),
                    pipeline_source=pipeline_source,
                    global_config_id=global_config_id,
                    pipeline_config_id=pipeline_config_id,
                    connection=ExecutionConnectionSpec(
                        host=host,
                        port=port,
                        transport_mode=transport_mode,
                        persistent=persistent,
                    ),
                )
            )
        )

    @openhcs_tool()
    def openhcs_get_orchestrator_session(session_id: str) -> dict:
        """Return one opaque execution session's plate, pipeline, and connection identity."""
        return to_jsonable(ctx.execution_service.get_session(session_id))

    @openhcs_tool()
    def openhcs_inspect_pipeline_source_artifact_plan(
        plate_path: str,
        pipeline_source: str,
        axis_filter: list[str] | None = None,
        well_filter: list[str] | None = None,
        global_config_id: str | None = None,
        pipeline_config_id: str | None = None,
    ) -> dict:
        """Compile pycodified pipeline source and return a bounded artifact plan inspection."""
        selected_axis_filter = axis_filter if axis_filter is not None else well_filter
        if selected_axis_filter is None:
            artifact_axis_filter = ()
        else:
            artifact_axis_filter = tuple(selected_axis_filter)
        return to_jsonable(
            ctx.execution_service.inspect_pipeline_source_artifact_plan(
                PycodifiedPipelineSessionRequest(
                    identity=ZMQExecutionIdentity(
                        plate_id=plate_path,
                    ),
                    pipeline_source=pipeline_source,
                    global_config_id=global_config_id,
                    pipeline_config_id=pipeline_config_id,
                    connection=ExecutionConnectionSpec(),
                ),
                axis_filter=artifact_axis_filter,
            )
        )

    @openhcs_tool()
    def openhcs_submit_compile(
        session_id: str,
        wait: bool = False,
    ) -> dict:
        """Submit a compile-only ZMQ job for an OpenHCS execution session."""
        return to_jsonable(ctx.execution_service.submit_compile(session_id, wait=wait))

    @openhcs_tool()
    def openhcs_submit_pipeline_execution(
        session_id: str,
        compile_artifact_id: str | None = None,
        wait: bool = False,
    ) -> dict:
        """Submit a ZMQ pipeline execution job for an OpenHCS execution session."""
        return to_jsonable(
            ctx.execution_service.submit_execution(
                session_id,
                compile_artifact_id=compile_artifact_id,
                wait=wait,
            )
        )

    @openhcs_tool()
    def openhcs_get_execution_status(job_id: str) -> dict:
        """Poll status for one submitted OpenHCS compile or execution job."""
        return to_jsonable(ctx.execution_service.get_job_status(job_id))

    @openhcs_tool()
    def openhcs_scan_runtime_servers(
        ports: list[int] | None = None,
        host: str = "localhost",
        transport_mode: str | None = None,
        timeout_ms: int = 200,
    ) -> dict:
        """Scan candidate ports for running OpenHCS ZMQ execution servers."""
        return to_jsonable(
            ctx.runtime_server_service.scan(
                ports=tuple(ports) if ports is not None else None,
                host=host,
                transport_mode=transport_mode,
                timeout_ms=timeout_ms,
            )
        )

    @openhcs_tool()
    def openhcs_get_runtime_server_info(
        host: str = "localhost",
        port: int | None = None,
        transport_mode: str | None = None,
        persistent: bool = True,
    ) -> dict:
        """Return a read-only snapshot from a running OpenHCS ZMQ execution server."""
        return to_jsonable(
            ctx.runtime_server_service.server_info(
                host=host,
                port=port,
                transport_mode=transport_mode,
                persistent=persistent,
            )
        )

    @openhcs_tool()
    def openhcs_get_runtime_server_execution_status(
        execution_id: str | None = None,
        host: str = "localhost",
        port: int | None = None,
        transport_mode: str | None = None,
        persistent: bool = True,
    ) -> dict:
        """Return bounded execution status from a running OpenHCS runtime server."""
        return to_jsonable(
            ctx.runtime_server_service.execution_status(
                execution_id=execution_id,
                host=host,
                port=port,
                transport_mode=transport_mode,
                persistent=persistent,
            )
        )

    @openhcs_tool()
    def openhcs_viewer_snapshot_window(
        port: int,
        output_dir_path: str | None = None,
        host: str = "localhost",
        transport_mode: str | None = None,
        capture_scope: str = "widget",
        timeout_ms: int | None = None,
    ) -> dict:
        """Capture a running viewer window, such as Napari, to a PNG resource path."""
        resolved_output_dir = _writable_output_dir(ctx, output_dir_path)
        viewer_args = McpViewerConnectionToolArgs(
            port,
            host,
            transport_mode,
            timeout_ms,
        )
        return to_jsonable(
            ctx.viewer_window_service.snapshot_window(
                viewer_args.snapshot_request(
                    output_dir_path=str(resolved_output_dir),
                    capture_scope=WindowSnapshotCaptureScope(capture_scope),
                )
            )
        )

    @openhcs_tool()
    def openhcs_get_viewer_window_state(
        port: int,
        host: str = "localhost",
        transport_mode: str | None = None,
        timeout_ms: int | None = None,
    ) -> dict:
        """Return structured layer, component, and axis state from a running viewer."""
        viewer_args = McpViewerConnectionToolArgs(
            port,
            host,
            transport_mode,
            timeout_ms,
        )
        return to_jsonable(
            ctx.viewer_window_service.window_state(viewer_args.state_request())
        )

    @openhcs_tool()
    def openhcs_get_viewer_window_payloads(
        port: int,
        host: str = "localhost",
        transport_mode: str | None = None,
        timeout_ms: int | None = None,
        route_key: str | None = None,
        include_array_values: bool | None = None,
        max_array_elements: int | None = None,
        include_shape_payloads: bool | None = None,
        max_shape_payloads: int | None = None,
    ) -> dict:
        """Return per-layer, per-axis viewer payload records with optional arrays and shapes."""
        viewer_args = McpViewerConnectionToolArgs(
            port,
            host,
            transport_mode,
            timeout_ms,
        )
        return to_jsonable(
            ctx.viewer_window_service.window_payloads(
                viewer_args.payload_request(
                    ViewerPayloadControlOptions.from_overrides(
                        route_key=route_key,
                        include_array_values=include_array_values,
                        max_array_elements=max_array_elements,
                        include_shape_payloads=include_shape_payloads,
                        max_shape_payloads=max_shape_payloads,
                    )
                )
            )
        )

    @openhcs_tool()
    def openhcs_probe_viewer_window(
        port: int,
        host: str = "localhost",
        transport_mode: str | None = None,
        timeout_ms: int | None = None,
    ) -> dict:
        """Quickly report whether a viewer control endpoint is reachable."""
        viewer_args = McpViewerConnectionToolArgs(
            port,
            host,
            transport_mode,
            timeout_ms,
        )
        return to_jsonable(
            ctx.viewer_window_service.probe_window(viewer_args.state_request())
        )

    @openhcs_tool()
    def openhcs_validate_viewer_window_state(
        port: int,
        host: str = "localhost",
        transport_mode: str | None = None,
        timeout_ms: int | None = None,
        expected_layer_count: int | None = None,
        required_axis_labels: tuple[str, ...] = (),
        require_nonzero_payloads: bool = True,
    ) -> dict:
        """Summarize viewer layers, expected axes, and nonzero payloads."""
        viewer_args = McpViewerConnectionToolArgs(
            port,
            host,
            transport_mode,
            timeout_ms,
        )
        return to_jsonable(
            ctx.viewer_window_service.validation_summary(
                viewer_args.validation_request(
                    ViewerWindowValidationPolicy(
                        expected_layer_count=expected_layer_count,
                        required_axis_labels=required_axis_labels,
                        require_nonzero_payloads=require_nonzero_payloads,
                    )
                )
            )
        )

    @openhcs_tool()
    def openhcs_ui_list_bridges() -> dict:
        """List live local OpenHCS UI bridge descriptors."""
        return to_jsonable(ctx.ui_bridge_service.list_bridges())

    @openhcs_tool()
    def openhcs_ui_bridge_status(
        connection: dict | None = None,
    ) -> dict:
        """Report whether a local running OpenHCS UI bridge is reachable."""
        return to_jsonable(
            ctx.ui_bridge_service.status(
                UiBridgeConnectionToolArgs.from_mapping(connection).resolve(ctx)
            )
        )

    @openhcs_tool()
    def openhcs_ui_list_code_documents(
        connection: dict | None = None,
    ) -> dict:
        """List code documents exposed by a running OpenHCS UI bridge."""
        return to_jsonable(
            ctx.ui_bridge_service.list_documents(
                UiBridgeConnectionToolArgs.from_mapping(connection).resolve(ctx)
            )
        )

    @openhcs_tool()
    def openhcs_ui_list_state_surfaces(
        connection: dict | None = None,
    ) -> dict:
        """List pollable state surfaces exposed by a running OpenHCS UI bridge."""
        return to_jsonable(
            ctx.ui_bridge_service.list_state_surfaces(
                UiBridgeConnectionToolArgs.from_mapping(connection).resolve(ctx)
            )
        )

    @openhcs_tool()
    def openhcs_ui_list_actions(
        connection: dict | None = None,
    ) -> dict:
        """List invokable UI actions exposed by a running OpenHCS UI bridge."""
        return to_jsonable(
            ctx.ui_bridge_service.list_actions(
                UiBridgeConnectionToolArgs.from_mapping(connection).resolve(ctx)
            )
        )

    @openhcs_tool()
    def openhcs_ui_invoke_action(
        widget_id: str,
        action_id: str,
        target_scope_ids: list[str] | None = None,
        observed_selection_revision_token: str | None = None,
        request_token: str | None = None,
        require_confirmation: bool = True,
        connection: dict | None = None,
    ) -> dict:
        """Dispatch one UI action using the selection token from openhcs_ui_list_actions."""
        selected_scope_ids = SelectedScopeIdsArgument.from_optional_iterable(
            target_scope_ids
        )
        return to_jsonable(
            ctx.ui_bridge_service.invoke_action(
                UiActionInvokeRequest(
                    widget_id=widget_id,
                    action_id=action_id,
                    selected_scope_ids=selected_scope_ids.selected_scope_ids,
                    observed_selection_revision_token=observed_selection_revision_token,
                    request_token=UiMutationRequestToken(request_token),
                    confirmation_requirement=UiBridgeConfirmationRequirement.from_flag(
                        require_confirmation
                    ),
                ),
                UiBridgeConnectionToolArgs.from_mapping(connection).resolve(ctx),
            )
        )

    @openhcs_tool()
    def openhcs_ui_selected_plate_workflow(
        workflow: UiSelectedPlateWorkflowKind,
        target_scope_ids: list[str] | None = None,
        observed_selection_revision_token: str | None = None,
        request_token: str | None = None,
        require_confirmation: bool = True,
        connection: dict | None = None,
    ) -> dict:
        """Dispatch init, compile, or run for the current PlateManager selection."""
        selected_scope_ids = SelectedScopeIdsArgument.from_optional_iterable(
            target_scope_ids
        )
        return to_jsonable(
            ctx.ui_bridge_service.selected_plate_workflow(
                UiSelectedPlateWorkflowRequest(
                    workflow=workflow,
                    selected_scope_ids=selected_scope_ids.selected_scope_ids,
                    observed_selection_revision_token=observed_selection_revision_token,
                    request_token=UiMutationRequestToken(request_token),
                    confirmation_requirement=UiBridgeConfirmationRequirement.from_flag(
                        require_confirmation
                    ),
                ),
                UiBridgeConnectionToolArgs.from_mapping(connection).resolve(ctx),
            )
        )

    @openhcs_tool()
    def openhcs_ui_list_windows(
        connection: dict | None = None,
    ) -> dict:
        """List visible and focusable UI windows exposed by a running OpenHCS UI bridge."""
        return to_jsonable(
            ctx.ui_bridge_service.list_windows(
                UiBridgeConnectionToolArgs.from_mapping(connection).resolve(ctx)
            )
        )

    @openhcs_tool()
    def openhcs_ui_focus_window(
        window_id: str,
        create_if_missing: bool = True,
        connection: dict | None = None,
    ) -> dict:
        """Focus one UI window by stable window id or open scope id."""
        return to_jsonable(
            ctx.ui_bridge_service.focus_window(
                UiWindowFocusRequest(
                    window_id=window_id,
                    open_policy=UiWindowOpenPolicy(create_if_missing=create_if_missing),
                ),
                UiBridgeConnectionToolArgs.from_mapping(connection).resolve(ctx),
            )
        )

    @openhcs_tool()
    def openhcs_ui_navigate_window(
        window_id: str,
        field_path: str | None = None,
        item_id: str | None = None,
        create_if_missing: bool = True,
        connection: dict | None = None,
    ) -> dict:
        """Open/focus a UI window scope and reveal an optional field or item."""
        return to_jsonable(
            ctx.ui_bridge_service.navigate_window(
                UiWindowNavigateRequest(
                    window_id=window_id,
                    field_path=field_path,
                    item_id=item_id,
                    open_policy=UiWindowOpenPolicy(create_if_missing=create_if_missing),
                ),
                UiBridgeConnectionToolArgs.from_mapping(connection).resolve(ctx),
            )
        )

    @openhcs_tool()
    def openhcs_ui_close_window(
        window_id: str,
        connection: dict | None = None,
    ) -> dict:
        """Request a normal close for one visible UI bridge window."""
        return to_jsonable(
            ctx.ui_bridge_service.close_window(
                UiWindowCloseRequest(window_id=window_id),
                UiBridgeConnectionToolArgs.from_mapping(connection).resolve(ctx),
            )
        )

    @openhcs_tool()
    def openhcs_ui_snapshot_window(
        window_id: str,
        output_dir_path: str | None = None,
        capture_scope: str = "widget",
        create_if_missing: bool = False,
        connection: dict | None = None,
    ) -> dict:
        """Capture one UI bridge window to a PNG resource path."""
        resolved_output_dir = _writable_output_dir(ctx, output_dir_path)
        return to_jsonable(
            ctx.ui_bridge_service.snapshot_window(
                UiWindowSnapshotRequest(
                    window_id=window_id,
                    output_dir_path=str(resolved_output_dir),
                    capture_scope=WindowSnapshotCaptureScope(capture_scope),
                    open_policy=UiWindowOpenPolicy(create_if_missing=create_if_missing),
                ),
                UiBridgeConnectionToolArgs.from_mapping(connection).resolve(ctx),
            )
        )

    @openhcs_tool()
    def openhcs_ui_get_widget_tree(
        window_id: str,
        create_if_missing: bool = False,
        maximum_text_length: int = UiWidgetTreeRequest.default_maximum_text_length(),
        truncation_suffix: str = UiWidgetTreeRequest.default_truncation_suffix(),
        connection: dict | None = None,
    ) -> dict:
        """Return a generic Qt widget tree with clickable geometry and action kinds."""
        return to_jsonable(
            ctx.ui_bridge_service.widget_tree(
                UiWidgetTreeRequest(
                    window_id=window_id,
                    open_policy=UiWindowOpenPolicy(create_if_missing=create_if_missing),
                    maximum_text_length=maximum_text_length,
                    truncation_suffix=truncation_suffix,
                ),
                UiBridgeConnectionToolArgs.from_mapping(connection).resolve(ctx),
            )
        )

    @openhcs_tool()
    def openhcs_ui_list_object_state_scopes(
        scope_visibility: dict | None = None,
        include_fields: bool = False,
        field_limit: int = 200,
        field_offset: int = 0,
        connection: dict | None = None,
    ) -> dict:
        """List ObjectState scopes, optionally including field-level semantic addresses."""
        visibility = UiObjectStateScopeVisibilityToolArgs.from_mapping(scope_visibility)
        return to_jsonable(
            ctx.ui_bridge_service.list_object_state_scopes(
                visibility.object_state_scope_list_request(
                    field_options=UiObjectStateFieldListOptions(
                        include_fields=include_fields,
                        field_limit=field_limit,
                        field_offset=field_offset,
                    ),
                ),
                UiBridgeConnectionToolArgs.from_mapping(connection).resolve(ctx),
            )
        )

    @openhcs_tool()
    def openhcs_ui_get_state_surface(
        surface_id: str = "plate_manager.state",
        selection_mode: str = "all",
        revision_token: str | None = None,
        connection: dict | None = None,
    ) -> dict:
        """Read or poll one typed UI state surface from the running OpenHCS UI bridge."""
        return to_jsonable(
            ctx.ui_bridge_service.get_state_surface(
                UiStateSurfaceRequest(
                    surface_id=surface_id,
                    selection_mode=selection_mode,
                    base_revision_token=revision_token,
                ),
                UiBridgeConnectionToolArgs.from_mapping(connection).resolve(ctx),
            )
        )

    @openhcs_tool()
    def openhcs_ui_get_code_document(
        document_id: str,
        selection_mode: str = "selected",
        clean: bool = True,
        connection: dict | None = None,
    ) -> dict:
        """Read a bounded code document from the running OpenHCS UI bridge."""
        return to_jsonable(
            ctx.ui_bridge_service.get_document(
                UiCodeDocumentRequest(
                    document_id=document_id,
                    selection_mode=selection_mode,
                    clean=clean,
                ),
                UiBridgeConnectionToolArgs.from_mapping(connection).resolve(ctx),
            )
        )

    @openhcs_tool()
    def openhcs_ui_validate_code_document(
        document_id: str,
        source: str,
        revision_token: str | None = None,
        connection: dict | None = None,
    ) -> dict:
        """Validate a UI code document without mutating running UI state."""
        return to_jsonable(
            ctx.ui_bridge_service.validate_document(
                UiCodeDocumentValidationRequest(
                    document_id=document_id,
                    source=source,
                    base_revision_token=revision_token,
                ),
                UiBridgeConnectionToolArgs.from_mapping(connection).resolve(ctx),
            )
        )

    @openhcs_tool()
    def openhcs_ui_apply_code_document(
        document_id: str,
        source: str,
        revision_token: str,
        require_confirmation: bool = True,
        snapshot_label: str | None = None,
        apply_if_time_traveling: bool = False,
        connection: dict | None = None,
    ) -> dict:
        """Apply a UI code document and return revision, snapshot, and undo targets."""
        return to_jsonable(
            ctx.ui_bridge_service.apply_document(
                UiCodeDocumentApplyRequest(
                    document_id=document_id,
                    source=source,
                    base_revision_token=revision_token,
                    confirmation_requirement=UiBridgeConfirmationRequirement.from_flag(
                        require_confirmation
                    ),
                    snapshot_label=snapshot_label,
                    apply_if_time_traveling=apply_if_time_traveling,
                ),
                UiBridgeConnectionToolArgs.from_mapping(connection).resolve(ctx),
            )
        )

    @openhcs_tool()
    def openhcs_ui_list_snapshots(
        scope_visibility: dict | None = None,
        connection: dict | None = None,
    ) -> dict:
        """List ObjectState snapshots visible to the running UI bridge."""
        visibility = UiObjectStateScopeVisibilityToolArgs.from_mapping(scope_visibility)
        return to_jsonable(
            ctx.ui_bridge_service.list_snapshots(
                visibility.snapshot_list_request(),
                UiBridgeConnectionToolArgs.from_mapping(connection).resolve(ctx),
            )
        )

    @openhcs_tool()
    def openhcs_ui_restore_snapshot(
        snapshot_id: str | None = None,
        index: int | None = None,
        branch: str | None = None,
        scope_visibility: dict | None = None,
        require_confirmation: bool = True,
        allow_auto_branch: bool = False,
        connection: dict | None = None,
    ) -> dict:
        """Restore the running UI to one ObjectState snapshot target."""
        visibility = UiObjectStateScopeVisibilityToolArgs.from_mapping(scope_visibility)
        return to_jsonable(
            ctx.ui_bridge_service.restore_snapshot(
                visibility.snapshot_restore_request(
                    snapshot_id=snapshot_id,
                    index=index,
                    branch=branch,
                    confirmation_requirement=(
                        UiBridgeConfirmationRequirement.from_flag(require_confirmation)
                    ),
                    allow_auto_branch=allow_auto_branch,
                ),
                UiBridgeConnectionToolArgs.from_mapping(connection).resolve(ctx),
            )
        )

    @openhcs_tool()
    def openhcs_ui_time_travel_head(
        require_confirmation: bool = True,
        connection: dict | None = None,
    ) -> dict:
        """Return the running UI to ObjectState branch head."""
        return to_jsonable(
            ctx.ui_bridge_service.time_travel_head(
                UiTimeTravelHeadRequest(
                    confirmation_requirement=UiBridgeConfirmationRequirement.from_flag(
                        require_confirmation
                    )
                ),
                UiBridgeConnectionToolArgs.from_mapping(connection).resolve(ctx),
            )
        )

    @openhcs_tool()
    def openhcs_ui_list_branches(
        connection: dict | None = None,
    ) -> dict:
        """List ObjectState branches visible to the running UI bridge."""
        return to_jsonable(
            ctx.ui_bridge_service.list_branches(
                UiBridgeConnectionToolArgs.from_mapping(connection).resolve(ctx)
            )
        )

    @openhcs_tool()
    def openhcs_ui_switch_branch(
        branch: str,
        require_confirmation: bool = True,
        allow_auto_branch: bool = False,
        connection: dict | None = None,
    ) -> dict:
        """Switch the running UI to an ObjectState branch."""
        return to_jsonable(
            ctx.ui_bridge_service.switch_branch(
                UiBranchSwitchRequest(
                    branch=branch,
                    confirmation_requirement=UiBridgeConfirmationRequirement.from_flag(
                        require_confirmation
                    ),
                    allow_auto_branch=allow_auto_branch,
                ),
                UiBridgeConnectionToolArgs.from_mapping(connection).resolve(ctx),
            )
        )

    @openhcs_tool()
    def openhcs_ui_get_operation_status(
        operation_id: str,
        connection: dict | None = None,
    ) -> dict:
        """Return status for one running or recent UI bridge operation."""
        return to_jsonable(
            ctx.ui_bridge_service.get_operation_status(
                operation_id,
                UiBridgeConnectionToolArgs.from_mapping(connection).resolve(ctx),
            )
        )

    return server


@dataclass(frozen=True, slots=True)
class McpToolErrorResult:
    """Structured MCP boundary error returned instead of raising through transport."""

    schema_version: str
    ok: bool
    tool: str
    errors: tuple[AgentError, ...]


def _mcp_tool_error(tool_name: str, exception: Exception) -> JsonValue:
    return to_jsonable(
        McpToolErrorResult(
            schema_version=SCHEMA_VERSION,
            ok=False,
            tool=tool_name,
            errors=(
                AgentError.from_exception(
                    "mcp_tool_failed",
                    exception,
                    hint="The MCP server caught this exception at the tool boundary.",
                ),
            ),
        )
    )


def _mcp_server_stale_error(tool_name: str) -> JsonValue:
    stale_source_paths = _mcp_server_stale_source_paths()
    if stale_source_paths:
        stale_path = str(stale_source_paths[0])
    else:
        stale_path = str(MCP_SERVER_SOURCE_PATH)
    return to_jsonable(
        McpToolErrorResult(
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
                    hint="Call openhcs_health_check for source freshness details.",
                    path=stale_path,
                ),
            ),
        )
    )


def _json_object_or_empty(value: dict | None) -> dict:
    if value is None:
        return {}
    return dict(value)


@dataclass(frozen=True, slots=True)
class McpViewerConnectionToolArgs:
    """MCP viewer connection fields projected into agent viewer request DTOs."""

    port: int
    host: str
    transport_mode: str | None
    timeout_ms: int | None

    @property
    def connection(self) -> ExecutionConnectionSpec:
        return ExecutionConnectionSpec(
            host=self.host,
            port=self.port,
            transport_mode=self.transport_mode,
        )

    @property
    def resolved_timeout_ms(self) -> int:
        return McpViewerTimeoutPolicy.resolve(self.timeout_ms)

    def snapshot_request(
        self,
        *,
        output_dir_path: str,
        capture_scope: WindowSnapshotCaptureScope,
    ) -> ViewerWindowSnapshotRequest:
        return ViewerWindowSnapshotRequest(
            connection=self.connection,
            timeout_ms=self.resolved_timeout_ms,
            output_dir_path=output_dir_path,
            capture_scope=capture_scope,
        )

    def state_request(self) -> ViewerWindowStateRequest:
        return ViewerWindowStateRequest(
            connection=self.connection,
            timeout_ms=self.resolved_timeout_ms,
        )

    def payload_request(
        self,
        payload_controls: ViewerPayloadControlOptions,
    ) -> ViewerWindowPayloadRequest:
        return ViewerWindowPayloadRequest(
            connection=self.connection,
            timeout_ms=self.resolved_timeout_ms,
            payload_controls=payload_controls,
        )

    def validation_request(
        self,
        validation_policy: ViewerWindowValidationPolicy,
    ) -> ViewerWindowValidationRequest:
        return ViewerWindowValidationRequest(
            connection=self.connection,
            timeout_ms=self.resolved_timeout_ms,
            validation_policy=validation_policy,
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
    def from_mapping(cls, value: Mapping[str, JsonValue] | None) -> Self:
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

    def resolve(self, context: OpenHCSAgentContext) -> UiBridgeConnectionSpec:
        return context.ui_bridge_service.connection_from_fields(
            replace(
                self._request,
                timeout_ms=McpUiBridgeTimeoutPolicy.resolve(self._request.timeout_ms),
            )
        )


@dataclass(frozen=True, slots=True)
class BoundedMcpTimeoutPolicy:
    label: str
    default_ms: int
    min_ms: int
    max_ms: int

    def resolve(self, requested_timeout_ms: int | None) -> int:
        if requested_timeout_ms is None:
            return self.default_ms
        if requested_timeout_ms < self.min_ms:
            raise ValueError(
                f"{self.label} MCP timeout must be at least {self.min_ms}ms."
            )
        if requested_timeout_ms > self.max_ms:
            raise ValueError(
                f"{self.label} MCP timeout must not exceed {self.max_ms}ms."
            )
        return requested_timeout_ms


class McpControlTimeoutPolicy:
    """Shared fail-fast timeout contract for Codex-facing MCP control tools."""

    label: ClassVar[str]

    @classmethod
    def resolve(cls, requested_timeout_ms: int | None) -> int:
        return BoundedMcpTimeoutPolicy(
            label=cls.label,
            default_ms=DEFAULT_MCP_CONTROL_TIMEOUT_MS,
            min_ms=MIN_MCP_CONTROL_TIMEOUT_MS,
            max_ms=MAX_MCP_CONTROL_TIMEOUT_MS,
        ).resolve(requested_timeout_ms)


class McpUiBridgeTimeoutPolicy(McpControlTimeoutPolicy):
    """Fail-fast timeout contract for Codex-facing UI bridge tools."""

    label = "UI bridge"


class McpViewerTimeoutPolicy(McpControlTimeoutPolicy):
    """Fail-fast timeout contract for viewer state and snapshot tools."""

    label = "Viewer"


class UiObjectStateScopeVisibilityToolArgs:
    """MCP argument adapter for ObjectState system-scope visibility."""

    def __init__(self, visibility: UiObjectStateScopeVisibility) -> None:
        self._visibility = visibility

    @classmethod
    def from_mapping(cls, value: Mapping[str, JsonValue] | None) -> Self:
        mapping = UiBridgeConnectionToolMapping.from_optional(value)
        value_from_mapping = mapping.optional_bool("include_system_scopes")
        if value_from_mapping is None:
            return cls(UiObjectStateScopeVisibility())
        return cls(
            UiObjectStateScopeVisibility(include_system_scopes=value_from_mapping)
        )

    def object_state_scope_list_request(
        self,
        *,
        field_options: UiObjectStateFieldListOptions = UiObjectStateFieldListOptions(),
    ) -> UiObjectStateScopeListRequest:
        return UiObjectStateScopeListRequest.from_visibility_options(
            self._visibility,
            field_options,
        )

    def snapshot_list_request(self) -> UiSnapshotListRequest:
        return UiSnapshotListRequest(
            include_system_scopes=self._visibility.include_system_scopes
        )

    def snapshot_restore_request(
        self,
        *,
        snapshot_id: str | None,
        index: int | None,
        branch: str | None,
        confirmation_requirement: UiBridgeConfirmationRequirement,
        allow_auto_branch: bool,
    ) -> UiSnapshotRestoreRequest:
        return UiSnapshotRestoreRequest(
            snapshot_id=snapshot_id,
            index=index,
            branch=branch,
            include_system_scopes=self._visibility.include_system_scopes,
            confirmation_requirement=confirmation_requirement,
            allow_auto_branch=allow_auto_branch,
        )


def _writable_output_dir(
    context: OpenHCSAgentContext,
    output_dir_path: str | None,
) -> Path:
    if output_dir_path is None:
        requested = DEFAULT_MCP_WINDOW_SNAPSHOT_DIR
    else:
        requested = Path(output_dir_path)
    return context.path_policy.assert_writable(requested)
