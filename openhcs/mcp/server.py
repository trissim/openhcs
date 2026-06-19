"""MCP server adapter for the OpenHCS agent API."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Self

from openhcs.agent.capabilities import get_capability_registry
from openhcs.agent.dto.common import SCHEMA_VERSION, JsonValue
from openhcs.agent.dto.config import ConfigPatch
from openhcs.agent.dto.execution import ExecutionConnectionSpec
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
    UiSnapshotListRequest,
    UiSnapshotRestoreRequest,
    UiStateSurfaceRequest,
    UiTimeTravelHeadRequest,
    UiWindowCloseRequest,
    UiWindowFocusRequest,
    UiWindowNavigateRequest,
    UiWindowOpenPolicy,
    UiWindowSnapshotRequest,
)
from openhcs.agent.serialization import to_jsonable
from openhcs.agent.services.execution_session_service import (
    PycodifiedPipelineSessionRequest,
)
from openhcs.core.selection import SelectedScopeIdsArgument
from openhcs.mcp.context import OpenHCSAgentContext, create_agent_context
from openhcs.runtime.window_snapshot import (
    WindowSnapshotCaptureScope,
    WindowSnapshotCaptureSpec,
)
from openhcs.runtime.zmq_execution_signature import ZMQExecutionIdentity


DEFAULT_MCP_WINDOW_SNAPSHOT_DIR = Path("/tmp/openhcs-mcp-window-snapshots")
DEFAULT_MCP_UI_BRIDGE_TIMEOUT_MS = 750
MAX_MCP_UI_BRIDGE_TIMEOUT_MS = 2_000
MIN_MCP_UI_BRIDGE_TIMEOUT_MS = 1


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

    @server.resource("openhcs://capabilities")
    def capabilities_resource() -> dict:
        """Return the canonical OpenHCS agent capability registry."""
        return to_jsonable(get_capability_registry())

    @server.resource("openhcs://architecture/topics")
    def architecture_topics_resource() -> dict:
        """List source-backed architecture topics available to agents."""
        return to_jsonable(ctx.architecture_service.list_topics())

    @server.tool()
    def openhcs_health_check() -> dict:
        """Report OpenHCS MCP server health and agent schema version."""
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "ok",
            "service": "openhcs.mcp",
        }

    @server.tool()
    def openhcs_list_capabilities() -> dict:
        """List MCP resources/tools and their OpenHCS agent API contracts."""
        return to_jsonable(get_capability_registry())

    @server.tool()
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

    @server.tool()
    def openhcs_describe_function(function_id: str) -> dict:
        """Return full signature, parameter, and documentation details for one function."""
        return to_jsonable(ctx.function_catalog.get(function_id))

    @server.tool()
    def openhcs_get_authoring_context(kind: str = "pipeline") -> dict:
        """Return bounded guidance for authoring OpenHCS pipelines or functions."""
        return to_jsonable(ctx.authoring_context_service.get_authoring_context(kind))

    @server.tool()
    def openhcs_list_architecture_topics() -> dict:
        """List architecture topics that explain OpenHCS internals through stable DTOs."""
        return to_jsonable(ctx.architecture_service.list_topics())

    @server.tool()
    def openhcs_explain_architecture(
        topic_id: str = "pipeline_model",
    ) -> dict:
        """Explain one OpenHCS architecture topic with source-backed internal symbols."""
        return to_jsonable(ctx.architecture_service.explain_topic(topic_id))

    @server.tool()
    def openhcs_describe_internal_symbol(symbol_id: str) -> dict:
        """Describe a projected internal OpenHCS symbol without exposing live objects."""
        return to_jsonable(ctx.architecture_service.describe_internal_symbol(symbol_id))

    @server.tool()
    def openhcs_describe_config_schema(config_type: str) -> dict:
        """Reflect GlobalPipelineConfig or PipelineConfig fields for safe config patches."""
        return to_jsonable(ctx.config_service.describe_schema(config_type))

    @server.tool()
    def openhcs_create_config(
        config_type: str,
        values: dict | None = None,
    ) -> dict:
        """Create an in-memory OpenHCS config draft from a config patch."""
        patch = ConfigPatch(config_type=config_type, values=_json_object_or_empty(values))
        return to_jsonable(ctx.config_service.create(config_type, patch))

    @server.tool()
    def openhcs_validate_config_patch(
        config_type: str,
        values: dict | None = None,
    ) -> dict:
        """Validate that values can instantiate the requested OpenHCS config type."""
        patch = ConfigPatch(config_type=config_type, values=_json_object_or_empty(values))
        return to_jsonable(ctx.config_service.validate_patch(config_type, patch))

    @server.tool()
    def openhcs_render_config_source(
        config_id: str,
        clean: bool = True,
    ) -> dict:
        """Render an in-memory config draft as reviewable Python source."""
        return to_jsonable(ctx.config_service.render_source(config_id, clean=clean))

    @server.tool()
    def openhcs_create_pipeline() -> dict:
        """Create an empty in-memory OpenHCS pipeline draft."""
        return to_jsonable(ctx.pipeline_service.create_pipeline())

    @server.tool()
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

    @server.tool()
    def openhcs_validate_pipeline(pipeline_id: str) -> dict:
        """Validate an in-memory pipeline draft against OpenHCS FunctionStep semantics."""
        return to_jsonable(ctx.pipeline_service.validate(pipeline_id))

    @server.tool()
    def openhcs_render_pipeline_source(
        pipeline_id: str,
        clean: bool = True,
    ) -> dict:
        """Render an in-memory pipeline draft as reviewable Python source."""
        return to_jsonable(ctx.pipeline_service.render_source(pipeline_id, clean=clean))

    @server.tool()
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

    @server.tool()
    def openhcs_create_orchestrator_session_from_pipeline_source(
        plate_path: str,
        pipeline_source: str,
        execution_plate_path: str | None = None,
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
                        execution_plate_id=execution_plate_path,
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

    @server.tool()
    def openhcs_get_orchestrator_session(session_id: str) -> dict:
        """Return one opaque execution session's plate, pipeline, and connection identity."""
        return to_jsonable(ctx.execution_service.get_session(session_id))

    @server.tool()
    def openhcs_submit_compile(
        session_id: str,
        wait: bool = False,
    ) -> dict:
        """Submit a compile-only ZMQ job for an OpenHCS execution session."""
        return to_jsonable(ctx.execution_service.submit_compile(session_id, wait=wait))

    @server.tool()
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

    @server.tool()
    def openhcs_get_execution_status(job_id: str) -> dict:
        """Poll status for one submitted OpenHCS compile or execution job."""
        return to_jsonable(ctx.execution_service.get_job_status(job_id))

    @server.tool()
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

    @server.tool()
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

    @server.tool()
    def openhcs_get_runtime_server_execution_status(
        execution_id: str | None = None,
        host: str = "localhost",
        port: int | None = None,
        transport_mode: str | None = None,
        persistent: bool = True,
    ) -> dict:
        """Return raw ZMQ execution status from a running OpenHCS runtime server."""
        return to_jsonable(
            ctx.runtime_server_service.execution_status(
                execution_id=execution_id,
                host=host,
                port=port,
                transport_mode=transport_mode,
                persistent=persistent,
            )
        )

    @server.tool()
    def openhcs_viewer_snapshot_window(
        port: int,
        output_dir_path: str | None = None,
        host: str = "localhost",
        transport_mode: str | None = None,
        capture_scope: str = "widget",
        timeout_ms: int = 5000,
    ) -> dict:
        """Capture a running viewer window, such as Napari, to a PNG resource path."""
        resolved_output_dir = _writable_output_dir(ctx, output_dir_path)
        return to_jsonable(
            ctx.viewer_window_service.snapshot_window(
                port=port,
                snapshot=WindowSnapshotCaptureSpec(
                    output_dir_path=str(resolved_output_dir),
                    capture_scope=WindowSnapshotCaptureScope(capture_scope),
                ),
                host=host,
                transport_mode=transport_mode,
                timeout_ms=timeout_ms,
            )
        )

    @server.tool()
    def openhcs_ui_list_bridges() -> dict:
        """List live local OpenHCS UI bridge descriptors."""
        return to_jsonable(ctx.ui_bridge_service.list_bridges())

    @server.tool()
    def openhcs_ui_bridge_status(
        connection: dict | None = None,
    ) -> dict:
        """Report whether a local running OpenHCS UI bridge is reachable."""
        return to_jsonable(
            ctx.ui_bridge_service.status(
                UiBridgeConnectionToolArgs.from_mapping(connection).resolve(ctx)
            )
        )

    @server.tool()
    def openhcs_ui_list_code_documents(
        connection: dict | None = None,
    ) -> dict:
        """List code documents exposed by a running OpenHCS UI bridge."""
        return to_jsonable(
            ctx.ui_bridge_service.list_documents(
                UiBridgeConnectionToolArgs.from_mapping(connection).resolve(ctx)
            )
        )

    @server.tool()
    def openhcs_ui_list_state_surfaces(
        connection: dict | None = None,
    ) -> dict:
        """List pollable state surfaces exposed by a running OpenHCS UI bridge."""
        return to_jsonable(
            ctx.ui_bridge_service.list_state_surfaces(
                UiBridgeConnectionToolArgs.from_mapping(connection).resolve(ctx)
            )
        )

    @server.tool()
    def openhcs_ui_list_actions(
        connection: dict | None = None,
    ) -> dict:
        """List invokable UI actions exposed by a running OpenHCS UI bridge."""
        return to_jsonable(
            ctx.ui_bridge_service.list_actions(
                UiBridgeConnectionToolArgs.from_mapping(connection).resolve(ctx)
            )
        )

    @server.tool()
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

    @server.tool()
    def openhcs_ui_list_windows(
        connection: dict | None = None,
    ) -> dict:
        """List visible and focusable UI windows exposed by a running OpenHCS UI bridge."""
        return to_jsonable(
            ctx.ui_bridge_service.list_windows(
                UiBridgeConnectionToolArgs.from_mapping(connection).resolve(ctx)
            )
        )

    @server.tool()
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
                    open_policy=UiWindowOpenPolicy(
                        create_if_missing=create_if_missing
                    ),
                ),
                UiBridgeConnectionToolArgs.from_mapping(connection).resolve(ctx),
            )
        )

    @server.tool()
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
                    open_policy=UiWindowOpenPolicy(
                        create_if_missing=create_if_missing
                    ),
                ),
                UiBridgeConnectionToolArgs.from_mapping(connection).resolve(ctx),
            )
        )

    @server.tool()
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

    @server.tool()
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
                    snapshot=WindowSnapshotCaptureSpec(
                        output_dir_path=str(resolved_output_dir),
                        capture_scope=WindowSnapshotCaptureScope(capture_scope),
                    ),
                    open_policy=UiWindowOpenPolicy(
                        create_if_missing=create_if_missing
                    ),
                ),
                UiBridgeConnectionToolArgs.from_mapping(connection).resolve(ctx),
            )
        )

    @server.tool()
    def openhcs_ui_list_object_state_scopes(
        scope_visibility: dict | None = None,
        include_fields: bool = False,
        field_limit: int = 200,
        field_offset: int = 0,
        connection: dict | None = None,
    ) -> dict:
        """List ObjectState scopes, optionally including field-level semantic addresses."""
        visibility = UiObjectStateScopeVisibilityToolArgs.from_mapping(
            scope_visibility
        )
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

    @server.tool()
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

    @server.tool()
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

    @server.tool()
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

    @server.tool()
    def openhcs_ui_apply_code_document(
        document_id: str,
        source: str,
        revision_token: str,
        require_confirmation: bool = True,
        snapshot_label: str | None = None,
        apply_if_time_traveling: bool = False,
        connection: dict | None = None,
    ) -> dict:
        """Apply a UI code document through the running PyQt workflow."""
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

    @server.tool()
    def openhcs_ui_list_snapshots(
        scope_visibility: dict | None = None,
        connection: dict | None = None,
    ) -> dict:
        """List ObjectState snapshots visible to the running UI bridge."""
        visibility = UiObjectStateScopeVisibilityToolArgs.from_mapping(
            scope_visibility
        )
        return to_jsonable(
            ctx.ui_bridge_service.list_snapshots(
                visibility.snapshot_list_request(),
                UiBridgeConnectionToolArgs.from_mapping(connection).resolve(ctx),
            )
        )

    @server.tool()
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
        visibility = UiObjectStateScopeVisibilityToolArgs.from_mapping(
            scope_visibility
        )
        return to_jsonable(
            ctx.ui_bridge_service.restore_snapshot(
                visibility.snapshot_restore_request(
                    snapshot_id=snapshot_id,
                    index=index,
                    branch=branch,
                    confirmation_requirement=(
                        UiBridgeConfirmationRequirement.from_flag(
                            require_confirmation
                        )
                    ),
                    allow_auto_branch=allow_auto_branch,
                ),
                UiBridgeConnectionToolArgs.from_mapping(connection).resolve(ctx),
            )
        )

    @server.tool()
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

    @server.tool()
    def openhcs_ui_list_branches(
        connection: dict | None = None,
    ) -> dict:
        """List ObjectState branches visible to the running UI bridge."""
        return to_jsonable(
            ctx.ui_bridge_service.list_branches(
                UiBridgeConnectionToolArgs.from_mapping(connection).resolve(ctx)
            )
        )

    @server.tool()
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

    @server.tool()
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


def _json_object_or_empty(value: dict | None) -> dict:
    if value is None:
        return {}
    return dict(value)


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
            raise TypeError(f"UI bridge connection field {field_name!r} must be a string.")
        return value

    def optional_int(self, field_name: str) -> int | None:
        value = self._optional_value(field_name)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"UI bridge connection field {field_name!r} must be an int.")
        return value

    def optional_bool(self, field_name: str) -> bool | None:
        value = self._optional_value(field_name)
        if value is None:
            return None
        if not isinstance(value, bool):
            raise TypeError(f"UI bridge connection field {field_name!r} must be a bool.")
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
                timeout_ms=McpUiBridgeTimeoutPolicy.resolve(
                    self._request.timeout_ms
                ),
            )
        )


class McpUiBridgeTimeoutPolicy:
    """Fail-fast timeout contract for Codex-facing UI bridge tools."""

    @staticmethod
    def resolve(requested_timeout_ms: int | None) -> int:
        if requested_timeout_ms is None:
            return DEFAULT_MCP_UI_BRIDGE_TIMEOUT_MS
        if requested_timeout_ms < MIN_MCP_UI_BRIDGE_TIMEOUT_MS:
            raise ValueError(
                "UI bridge MCP timeout must be at least "
                f"{MIN_MCP_UI_BRIDGE_TIMEOUT_MS}ms."
            )
        if requested_timeout_ms > MAX_MCP_UI_BRIDGE_TIMEOUT_MS:
            raise ValueError(
                "UI bridge MCP timeout must not exceed "
                f"{MAX_MCP_UI_BRIDGE_TIMEOUT_MS}ms."
            )
        return requested_timeout_ms


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
        return cls(UiObjectStateScopeVisibility(include_system_scopes=value_from_mapping))

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


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
