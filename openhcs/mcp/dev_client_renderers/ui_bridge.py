"""UI bridge renderers for the MCP dev client."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta

from openhcs.agent.capabilities import agent_capabilities
from openhcs.agent.dto.common import JsonObject, JsonValue
from openhcs.agent.dto.ui_bridge import (
    UiActionCatalog,
    UiActionInvokeResult,
    UiBridgeStatus,
    UiCodeDocument,
    UiCodeDocumentApplyResult,
    UiCodeDocumentCatalog,
    UiCodeDocumentValidationResult,
    UiStateSurfaceCatalog,
    UiStateSurfaceDocument,
    UiWidgetActionInvokeResult,
    UiWidgetTreeResult,
    UiWindowCatalog,
)
from openhcs.agent.ui_bridge_identities import (
    PipelineDebugSessionStateSurfaceIdentityDeclaration,
    PipelineEditorStateSurfaceIdentityDeclaration,
    PlateManagerStateSurfaceIdentityDeclaration,
    UiBridgeIdentityDeclaration,
    UiLiveOverviewStateSurfaceIdentityDeclaration,
    UiStateSurfaceIdentityDeclarationBase,
)
from openhcs.mcp.dev_client_rendering import (
    CodeDocumentRenderOptions,
    McpDevOutputRenderer,
    McpDevPayloadProjection,
    McpDiagnosticRenderer,
    UiActionCatalogRenderOptions,
    UiActionInvokeRenderOptions,
    WidgetTreeOutputFormat,
    WidgetTreeRenderOptions,
)
from openhcs.mcp.dev_client_renderers.object_state import ObjectStateScopeRenderer
from openhcs.mcp.dev_client_renderers.viewer import ViewerValidationRenderer

class UiBridgeStatusRenderer(McpDevOutputRenderer):
    """Compact renderer for live UI bridge status."""

    output_contract = UiBridgeStatus

    @classmethod
    def render(cls, response: JsonObject) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)
        errors = McpDevPayloadProjection.sequence_of_mappings(payload.get("errors"))
        if errors:
            return "\n".join(("UI bridge: unavailable", *cls._error_lines(errors)))

        descriptors = McpDevPayloadProjection.sequence_of_mappings(
            payload.get("descriptors")
        )
        descriptor = descriptors[0] if descriptors else {}
        connection = McpDevPayloadProjection.nested_mapping(payload, "connection")
        supported_operations = payload.get("supported_operations")
        bridge_features = payload.get("bridge_features")
        lines = [
            "UI bridge: "
            f"reachable={McpDevPayloadProjection.text(payload.get('reachable'))} "
            f"descriptor={McpDevPayloadProjection.text(payload.get('descriptor_status'))}",
            (
                "Instance: "
                f"{McpDevPayloadProjection.text(payload.get('bridge_instance_id'))} "
                f"pid={McpDevPayloadProjection.text(descriptor.get('pid'))}"
            ),
            (
                "Connection: "
                f"{McpDevPayloadProjection.text(connection.get('transport_mode'))} "
                f"{McpDevPayloadProjection.text(connection.get('host'))}:"
                f"{McpDevPayloadProjection.text(connection.get('port'))}"
            ),
            f"Descriptor: {McpDevPayloadProjection.text(payload.get('descriptor_file_path'))}",
            (
                "Capabilities: "
                f"{cls._count(supported_operations)} operations, "
                f"{cls._count(bridge_features)} features"
            ),
        ]
        return "\n".join(lines)

    @staticmethod
    def _count(value: JsonValue) -> int:
        return len(value) if isinstance(value, list) else 0

    @staticmethod
    def _error_lines(errors: tuple[Mapping[str, JsonValue], ...]) -> tuple[str, ...]:
        return McpDiagnosticRenderer.error_lines(errors)


class UiWindowCatalogRenderer(McpDevOutputRenderer):
    """Compact renderer for UI window catalogs."""

    output_contract = UiWindowCatalog

    MAIN_WINDOW_TITLE = "OpenHCS"
    TOP_LEVEL_WINDOW_KIND = "qt_top_level"

    @classmethod
    def render(cls, response: JsonObject) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)
        errors = McpDevPayloadProjection.sequence_of_mappings(payload.get("errors"))
        if errors:
            return "\n".join(("Windows: unavailable", *cls._error_lines(errors)))

        windows = McpDevPayloadProjection.sequence_of_mappings(payload.get("windows"))
        lines = [f"Windows: {len(windows)}"]
        attention_windows = cls._attention_windows(windows)
        if attention_windows:
            lines.append(
                "Attention: "
                f"{len(attention_windows)} visible top-level window(s): "
                + ", ".join(cls._attention_label(window) for window in attention_windows)
            )
        for window in windows:
            lines.append(
                "- "
                f"{McpDevPayloadProjection.text(window.get('window_id'))} "
                f"[{McpDevPayloadProjection.text(window.get('window_kind'))}] "
                f"visible={McpDevPayloadProjection.text(window.get('visible'))} "
                f"dirty={McpDevPayloadProjection.text(window.get('dirty'))} "
                f"diff={McpDevPayloadProjection.text(window.get('signature_diff'))} "
                f"title={McpDevPayloadProjection.quoted_text(window.get('title'))}"
            )
        return "\n".join(lines)

    @classmethod
    def _attention_windows(
        cls,
        windows: tuple[Mapping[str, JsonValue], ...],
    ) -> tuple[Mapping[str, JsonValue], ...]:
        return tuple(window for window in windows if cls._needs_attention(window))

    @classmethod
    def _needs_attention(cls, window: Mapping[str, JsonValue]) -> bool:
        return (
            window.get("visible") is True
            and window.get("window_kind") == cls.TOP_LEVEL_WINDOW_KIND
            and window.get("title") != cls.MAIN_WINDOW_TITLE
        )

    @staticmethod
    def _attention_label(window: Mapping[str, JsonValue]) -> str:
        return (
            f"{McpDevPayloadProjection.text(window.get('window_id'))} "
            f"title={McpDevPayloadProjection.quoted_text(window.get('title'))}"
        )

    @staticmethod
    def _error_lines(errors: tuple[Mapping[str, JsonValue], ...]) -> tuple[str, ...]:
        return tuple(
            f"- {McpDevPayloadProjection.text(error.get('code'))}: "
            f"{McpDevPayloadProjection.text(error.get('message'))}"
            for error in errors[:3]
        )


class UiSmokeRenderer:
    """Compact renderer for the multi-tool UI smoke command."""

    HEALTH_TOOL = agent_capabilities.health_check.name
    STATUS_TOOL = agent_capabilities.ui_bridge_status.name
    BRIDGES_TOOL = agent_capabilities.ui_list_bridges.name
    WINDOWS_TOOL = agent_capabilities.ui_list_windows.name

    @classmethod
    def render(cls, response: JsonObject) -> str:
        errors = McpDevPayloadProjection.sequence_of_mappings(response.get("errors"))
        if errors:
            return "\n".join(("UI smoke: unavailable", *McpDiagnosticRenderer.error_lines(errors)))

        results = McpDevPayloadProjection.sequence_of_mappings(response.get("results"))
        mcp_errors = sum(1 for result in results if result.get("mcp_error") is True)
        lines = [
            (
                "UI smoke: "
                f"results={len(results)} mcp_errors={mcp_errors}"
            )
        ]
        lines.append(cls._health_line(response))
        lines.extend(
            UiBridgeStatusRenderer.render(
                McpDevPayloadProjection.tool_response(response, cls.STATUS_TOOL)
            ).splitlines()
        )
        lines.append(cls._bridge_catalog_line(response))
        lines.extend(
            UiWindowCatalogRenderer.render(
                McpDevPayloadProjection.tool_response(response, cls.WINDOWS_TOOL)
            ).splitlines()
        )
        return "\n".join(lines)

    @classmethod
    def _health_line(cls, response: JsonObject) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(
            McpDevPayloadProjection.tool_response(response, cls.HEALTH_TOOL)
        )
        if payload is None:
            return "Health: missing"
        stale_paths = payload.get("stale_source_paths")
        stale_path_count = len(stale_paths) if isinstance(stale_paths, list) else 0
        return (
            "Health: "
            f"status={McpDevPayloadProjection.text(payload.get('status'))} "
            f"restart_required={McpDevPayloadProjection.text(payload.get('restart_required'))} "
            f"stale_paths={stale_path_count}"
        )

    @classmethod
    def _bridge_catalog_line(cls, response: JsonObject) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(
            McpDevPayloadProjection.tool_response(response, cls.BRIDGES_TOOL)
        )
        if payload is None:
            return "Bridges: missing"
        bridges = McpDevPayloadProjection.sequence_of_mappings(payload.get("bridges"))
        errors = McpDevPayloadProjection.sequence_of_mappings(payload.get("errors"))
        return f"Bridges: live={len(bridges)} errors={len(errors)}"


class UiStateSurfaceCatalogRenderer(McpDevOutputRenderer):
    """Compact renderer for UI state-surface catalogs."""

    output_contract = UiStateSurfaceCatalog

    @classmethod
    def render(cls, response: JsonObject) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)

        surfaces = McpDevPayloadProjection.sequence_of_mappings(payload.get("surfaces"))
        lines = [f"State surfaces: count={len(surfaces)}"]
        ObjectStateScopeRenderer._append_messages(lines, payload)
        if surfaces:
            lines.append("Surfaces:")
            lines.extend(cls._surface_lines(surfaces))
        return "\n".join(lines)

    @classmethod
    def _surface_lines(
        cls,
        surfaces: tuple[Mapping[str, JsonValue], ...],
    ) -> list[str]:
        return [cls._surface_line(surface) for surface in surfaces]

    @staticmethod
    def _surface_line(surface: Mapping[str, JsonValue]) -> str:
        return (
            "- "
            f"{UiStateSurfaceCatalogRenderer.surface_id(surface)}: "
            f"widget={McpDevPayloadProjection.text(surface.get('widget_id'))} "
            f"readable={McpDevPayloadProjection.text(surface.get('readable'))} "
            "selection="
            f"{McpDevPayloadProjection.text(surface.get('current_selection_count'))}/"
            f"{McpDevPayloadProjection.text(surface.get('total_scope_count'))} "
            f"modes={ViewerValidationRenderer._sequence_text(surface.get('supported_selection_modes'))} "
            f"title={McpDevPayloadProjection.quoted_text(surface.get('title'))}"
        )

    @staticmethod
    def surface_id(surface: Mapping[str, JsonValue]) -> str:
        surface_id = surface.get("surface_id")
        if isinstance(surface_id, str):
            return surface_id
        identity = McpDevPayloadProjection.nested_mapping(surface, "identity")
        identity_surface_id = identity.get("surface_id")
        if isinstance(identity_surface_id, str):
            return identity_surface_id
        return "<none>"


class UiStateSurfacePayloadRenderer(ABC, metaclass=AutoRegisterMeta):
    """Renderer declaration for one UI state-surface identity."""

    __registry__: ClassVar[
        dict[
            type[UiStateSurfaceIdentityDeclarationBase],
            type["UiStateSurfacePayloadRenderer"],
        ]
    ] = {}
    __registry_key__ = "surface_identity"
    __skip_if_no_key__ = True

    surface_identity: ClassVar[type[UiStateSurfaceIdentityDeclarationBase] | None] = None

    @classmethod
    def for_payload(
        cls,
        payload: Mapping[str, JsonValue],
    ) -> type["UiStateSurfacePayloadRenderer"] | None:
        surface_id = cls.surface_id(payload)
        if surface_id is None:
            return None
        identity_type = UiBridgeIdentityDeclaration.__registry__.get(surface_id)
        if identity_type is None:
            return None
        if not issubclass(identity_type, UiStateSurfaceIdentityDeclarationBase):
            return None
        return cls.__registry__.get(identity_type)

    @staticmethod
    def surface_id(payload: Mapping[str, JsonValue]) -> str | None:
        summary = McpDevPayloadProjection.nested_mapping(payload, "summary")
        identity = McpDevPayloadProjection.nested_mapping(summary, "identity")
        surface_id = identity.get("surface_id")
        if isinstance(surface_id, str):
            return surface_id
        return None

    @classmethod
    @abstractmethod
    def render(cls, response: JsonObject) -> str:
        """Render a state-surface response for this surface identity."""


class UiStateSurfaceRenderer(McpDevOutputRenderer):
    """Compact fallback renderer for non-PlateManager state surfaces."""

    output_contract = UiStateSurfaceDocument

    @classmethod
    def render(cls, response: JsonObject) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)
        renderer_type = UiStateSurfacePayloadRenderer.for_payload(payload)
        if renderer_type is not None:
            return renderer_type.render(response)
        summary = McpDevPayloadProjection.nested_mapping(payload, "summary")
        identity = McpDevPayloadProjection.nested_mapping(summary, "identity")
        lines = [
            f"Surface: {McpDevPayloadProjection.text(identity.get('surface_id'))}",
            f"Title: {McpDevPayloadProjection.text(summary.get('title'))}",
            f"Readable: {McpDevPayloadProjection.text(summary.get('readable'))}",
            f"Selection: {McpDevPayloadProjection.text(payload.get('selection_mode'))}",
            f"Revision: {McpDevPayloadProjection.text(payload.get('current_revision_token'))}",
        ]
        ObjectStateScopeRenderer._append_messages(lines, payload)
        if summary.get("readable") is False:
            lines.append("Next: state-surfaces")
        return "\n".join(lines)


class UiLiveOverviewStateSurfaceRenderer(UiStateSurfacePayloadRenderer):
    """Compact renderer for the live UI overview surface."""

    surface_identity = UiLiveOverviewStateSurfaceIdentityDeclaration

    @classmethod
    def render(cls, response: JsonObject) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)
        errors = McpDevPayloadProjection.sequence_of_mappings(payload.get("errors"))
        if errors:
            return "\n".join(("UI live overview: unavailable", *cls._error_lines(errors)))
        state_payload = McpDevPayloadProjection.nested_mapping(payload, "payload")
        sections = McpDevPayloadProjection.sequence_of_mappings(
            state_payload.get("sections")
        )
        lines = [
            f"UI live overview: sections={len(sections)}",
            f"Revision: {McpDevPayloadProjection.text(state_payload.get('current_revision_token'))}",
        ]
        for section in sections:
            lines.extend(cls._section_lines(section))
        return "\n".join(lines)

    @classmethod
    def _section_lines(cls, section: Mapping[str, JsonValue]) -> list[str]:
        metrics = McpDevPayloadProjection.sequence_of_mappings(section.get("metrics"))
        items = McpDevPayloadProjection.sequence_of_mappings(section.get("items"))
        metric_text = " ".join(
            f"{McpDevPayloadProjection.text(metric.get('label'))}="
            f"{McpDevPayloadProjection.text(metric.get('value'))}"
            for metric in metrics
        )
        title = McpDevPayloadProjection.text(section.get("title"))
        summary = McpDevPayloadProjection.text(section.get("summary"))
        lines = [f"{title}: {summary} {metric_text}".rstrip()]
        lines.extend(cls._item_lines(items))
        return lines

    @staticmethod
    def _item_lines(items: tuple[Mapping[str, JsonValue], ...]) -> list[str]:
        lines: list[str] = []
        for item in items:
            parts = [
                f"severity={McpDevPayloadProjection.text(item.get('severity'))}",
            ]
            if item.get("status") is not None:
                parts.append(f"status={McpDevPayloadProjection.text(item.get('status'))}")
            if item.get("detail") is not None:
                parts.append(f"detail={McpDevPayloadProjection.text(item.get('detail'))}")
            if item.get("source_surface_id") is not None:
                parts.append(
                    "surface="
                    f"{McpDevPayloadProjection.text(item.get('source_surface_id'))}"
                )
            if item.get("source_window_id") is not None:
                parts.append(
                    "window="
                    f"{McpDevPayloadProjection.text(item.get('source_window_id'))}"
                )
            lines.append(
                f"- {McpDevPayloadProjection.text(item.get('label'))}: "
                + " ".join(parts)
            )
        return lines

    @staticmethod
    def _error_lines(errors: tuple[Mapping[str, JsonValue], ...]) -> tuple[str, ...]:
        return tuple(
            f"- {McpDevPayloadProjection.text(error.get('code'))}: "
            f"{McpDevPayloadProjection.text(error.get('message'))}"
            for error in errors[:3]
        )


class PlateManagerStateSurfaceRenderer(UiStateSurfacePayloadRenderer):
    """Compact renderer for PlateManager state-surface rows."""

    surface_identity = PlateManagerStateSurfaceIdentityDeclaration

    @classmethod
    def render(cls, response: JsonObject) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)
        errors = McpDevPayloadProjection.sequence_of_mappings(payload.get("errors"))
        if errors:
            return "\n".join(("Plate manager: unavailable", *cls._error_lines(errors)))
        state_payload = McpDevPayloadProjection.nested_mapping(payload, "payload")
        summary = McpDevPayloadProjection.nested_mapping(state_payload, "summary")
        rows = McpDevPayloadProjection.sequence_of_mappings(state_payload.get("rows"))
        lines = [
            (
                "Plate manager: "
                f"rows={len(rows)} "
                f"selected={McpDevPayloadProjection.text(summary.get('current_selection_count'))} "
                f"manager={McpDevPayloadProjection.text(state_payload.get('manager_execution_state'))}"
            ),
            f"Revision: {McpDevPayloadProjection.text(state_payload.get('current_revision_token'))}",
        ]
        snapshot = McpDevPayloadProjection.nested_mapping(
            state_payload,
            "current_snapshot",
        )
        if snapshot:
            lines.append(
                "Snapshot: "
                f"{McpDevPayloadProjection.text(snapshot.get('index'))} "
                f"{McpDevPayloadProjection.quoted_text(snapshot.get('label'))}"
            )
        if rows:
            lines.append("Rows:")
            lines.extend(cls._row_lines(rows))
        return "\n".join(lines)

    @classmethod
    def _row_lines(cls, rows: tuple[Mapping[str, JsonValue], ...]) -> list[str]:
        lines: list[str] = []
        for row in rows:
            status = McpDevPayloadProjection.text(row.get("status_prefix"))
            if not status:
                status = "<none>"
            row_parts = [
                f"state={McpDevPayloadProjection.text(row.get('orchestrator_state'))}",
                f"status={status}",
                f"init={McpDevPayloadProjection.text(row.get('initialized'))}",
                f"compiled={McpDevPayloadProjection.text(row.get('compiled'))}",
                f"active={McpDevPayloadProjection.text(row.get('execution_active'))}",
                f"terminal={McpDevPayloadProjection.text(row.get('terminal_status'))}",
                f"selected={McpDevPayloadProjection.text(row.get('selected'))}",
            ]
            if row.get("plate_root") is not None:
                row_parts.append(
                    f"root={McpDevPayloadProjection.text(row.get('plate_root'))}"
                )
            if row.get("output_plate_root") is not None:
                row_parts.append(
                    "output="
                    f"{McpDevPayloadProjection.text(row.get('output_plate_root'))}"
                )
            if row.get("source_plate_root") is not None:
                row_parts.append(
                    "source="
                    f"{McpDevPayloadProjection.text(row.get('source_plate_root'))}"
                )
            lines.append(
                f"- {McpDevPayloadProjection.text(row.get('name'))}: "
                + ", ".join(row_parts)
            )
        return lines

    @staticmethod
    def _error_lines(errors: tuple[Mapping[str, JsonValue], ...]) -> tuple[str, ...]:
        return tuple(
            f"- {McpDevPayloadProjection.text(error.get('code'))}: "
            f"{McpDevPayloadProjection.text(error.get('message'))}"
            for error in errors[:3]
        )


class PipelineEditorStateSurfaceRenderer(UiStateSurfacePayloadRenderer):
    """Compact renderer for PipelineEditor state-surface step rows."""

    surface_identity = PipelineEditorStateSurfaceIdentityDeclaration

    @classmethod
    def render(cls, response: JsonObject) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)
        errors = McpDevPayloadProjection.sequence_of_mappings(payload.get("errors"))
        if errors:
            return "\n".join(("Pipeline editor: unavailable", *cls._error_lines(errors)))
        state_payload = McpDevPayloadProjection.nested_mapping(payload, "payload")
        summary = McpDevPayloadProjection.nested_mapping(state_payload, "summary")
        steps = McpDevPayloadProjection.sequence_of_mappings(state_payload.get("steps"))
        lines = [
            (
                "Pipeline editor: "
                f"steps={len(steps)} "
                f"selected={McpDevPayloadProjection.text(summary.get('current_selection_count'))}/"
                f"{McpDevPayloadProjection.text(summary.get('total_scope_count'))} "
                f"plate={McpDevPayloadProjection.text(state_payload.get('current_plate_scope_id'))} "
                f"pipeline={McpDevPayloadProjection.text(state_payload.get('pipeline_scope_id'))}"
            ),
            f"Revision: {McpDevPayloadProjection.text(state_payload.get('current_revision_token'))}",
        ]
        snapshot = McpDevPayloadProjection.nested_mapping(
            state_payload,
            "current_snapshot",
        )
        if snapshot:
            lines.append(
                "Snapshot: "
                f"{McpDevPayloadProjection.text(snapshot.get('index'))} "
                f"{McpDevPayloadProjection.quoted_text(snapshot.get('label'))}"
            )
        selected_scope_ids = state_payload.get("selected_scope_ids")
        if selected_scope_ids:
            lines.append(
                "Selected scopes: "
                f"{ViewerValidationRenderer._sequence_text(selected_scope_ids)}"
            )
        if steps:
            lines.append("Steps:")
            lines.extend(cls._step_lines(steps))
        return "\n".join(lines)

    @classmethod
    def _step_lines(cls, steps: tuple[Mapping[str, JsonValue], ...]) -> list[str]:
        lines: list[str] = []
        for step in steps:
            markers = cls._markers(step)
            function_names = ViewerValidationRenderer._sequence_text(
                step.get("function_names")
            )
            if not function_names:
                function_names = "<none>"
            step_parts = [
                f"enabled={McpDevPayloadProjection.text(step.get('enabled'))}",
                f"selected={McpDevPayloadProjection.text(step.get('selected'))}",
                f"funcs={function_names}",
            ]
            function_ids = cls._function_ids_text(step.get("function_ids"))
            if function_ids:
                step_parts.append(f"ids={function_ids}")
            if step.get("debug_pause"):
                step_parts.append("debug_pause=True")
            if step.get("step_scope_id") is not None:
                step_parts.append(
                    f"scope={McpDevPayloadProjection.text(step.get('step_scope_id'))}"
                )
            lines.append(
                f"- {McpDevPayloadProjection.text(step.get('index'))}. "
                f"{markers}{McpDevPayloadProjection.text(step.get('name'))}: "
                + ", ".join(step_parts)
            )
        return lines

    @staticmethod
    def _function_ids_text(value: JsonValue) -> str:
        if not isinstance(value, list) or not value:
            return ""
        return ",".join(str(item) for item in value if isinstance(item, str))

    @staticmethod
    def _markers(step: Mapping[str, JsonValue]) -> str:
        markers = ""
        if step.get("dirty"):
            markers += "*"
        if step.get("default_diff"):
            markers += "_"
        if markers:
            return f"[{markers}] "
        return ""

    @staticmethod
    def _error_lines(errors: tuple[Mapping[str, JsonValue], ...]) -> tuple[str, ...]:
        return tuple(
            f"- {McpDevPayloadProjection.text(error.get('code'))}: "
            f"{McpDevPayloadProjection.text(error.get('message'))}"
            for error in errors[:3]
        )


class PipelineDebugSessionStateSurfaceRenderer(UiStateSurfacePayloadRenderer):
    """Compact renderer for PipelineEditor debug-session state."""

    surface_identity = PipelineDebugSessionStateSurfaceIdentityDeclaration

    @classmethod
    def render(cls, response: JsonObject) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)
        errors = McpDevPayloadProjection.sequence_of_mappings(payload.get("errors"))
        if errors:
            return "\n".join(("Pipeline debug: unavailable", *cls._error_lines(errors)))
        state_payload = McpDevPayloadProjection.nested_mapping(payload, "payload")
        actions = McpDevPayloadProjection.sequence_of_mappings(state_payload.get("actions"))
        cursor = McpDevPayloadProjection.nested_mapping(state_payload, "cursor")
        lines = [
            (
                "Pipeline debug: "
                f"phase={McpDevPayloadProjection.text(state_payload.get('phase'))} "
                f"plate={McpDevPayloadProjection.text(state_payload.get('current_plate_scope_id'))} "
                f"pipeline={McpDevPayloadProjection.text(state_payload.get('pipeline_scope_id'))} "
                f"manager={McpDevPayloadProjection.text(state_payload.get('manager_execution_state'))}"
            ),
            (
                "Target: "
                f"initialized={McpDevPayloadProjection.text(state_payload.get('initialized'))} "
                f"compiled={McpDevPayloadProjection.text(state_payload.get('compiled'))} "
                f"terminal={McpDevPayloadProjection.text(state_payload.get('terminal_status'))}"
            ),
            (
                "Session: "
                f"id={McpDevPayloadProjection.text(state_payload.get('active_session_id'))} "
                f"execution={McpDevPayloadProjection.text(state_payload.get('execution_id'))} "
                f"axis={McpDevPayloadProjection.text(state_payload.get('axis_id'))} "
                f"source_group={McpDevPayloadProjection.text(state_payload.get('selected_source_group'))}"
            ),
            f"Revision: {McpDevPayloadProjection.text(state_payload.get('current_revision_token'))}",
        ]
        if cursor:
            lines.append(
                "Cursor: "
                f"step={McpDevPayloadProjection.text(cursor.get('step_index'))} "
                f"scope={McpDevPayloadProjection.text(cursor.get('step_scope_id'))} "
                f"group={McpDevPayloadProjection.text(cursor.get('group_key'))} "
                f"invocation={McpDevPayloadProjection.text(cursor.get('invocation_key'))} "
                f"dirty={McpDevPayloadProjection.text(cursor.get('dirty'))}"
            )
        current_frame = McpDevPayloadProjection.nested_mapping(
            state_payload,
            "current_frame",
        )
        last_frame = McpDevPayloadProjection.nested_mapping(
            state_payload,
            "last_frame",
        )
        if current_frame:
            lines.append(cls._frame_line("Current frame", current_frame))
        if last_frame and last_frame != current_frame:
            lines.append(cls._frame_line("Last frame", last_frame))
        if actions:
            lines.append("Actions:")
            lines.extend(cls._action_lines(actions))
        return "\n".join(lines)

    @classmethod
    def _frame_line(
        cls,
        label: str,
        frame: Mapping[str, JsonValue],
    ) -> str:
        progress_identity = McpDevPayloadProjection.nested_mapping(
            frame,
            "progress_identity",
        )
        cursor = McpDevPayloadProjection.nested_mapping(frame, "cursor")
        return (
            f"{label}: "
            f"event={McpDevPayloadProjection.text(frame.get('event_type'))} "
            f"step={McpDevPayloadProjection.text(frame.get('step_name'))} "
            f"callable={McpDevPayloadProjection.text(frame.get('callable_name'))} "
            f"axis={McpDevPayloadProjection.text(progress_identity.get('axis_id'))} "
            f"snapshot={McpDevPayloadProjection.text(frame.get('snapshot_id'))} "
            f"invocation={McpDevPayloadProjection.text(cursor.get('invocation_key'))}"
        )

    @classmethod
    def _action_lines(
        cls,
        actions: tuple[Mapping[str, JsonValue], ...],
    ) -> list[str]:
        lines: list[str] = []
        for action in actions:
            disabled = McpDevPayloadProjection.nested_mapping(action, "disabled_error")
            suffix = ""
            if disabled:
                suffix = (
                    " disabled="
                    f"{McpDevPayloadProjection.text(disabled.get('code'))}"
                )
            lines.append(
                "- "
                f"{McpDevPayloadProjection.text(action.get('action_id'))}: "
                f"enabled={McpDevPayloadProjection.text(action.get('enabled'))} "
                f"placement={McpDevPayloadProjection.text(action.get('placement'))} "
                f"title={McpDevPayloadProjection.quoted_text(action.get('label'))}"
                f"{suffix}"
            )
        return lines

    @staticmethod
    def _error_lines(errors: tuple[Mapping[str, JsonValue], ...]) -> tuple[str, ...]:
        return tuple(
            f"- {McpDevPayloadProjection.text(error.get('code'))}: "
            f"{McpDevPayloadProjection.text(error.get('message'))}"
            for error in errors[:3]
        )


@dataclass(frozen=True, slots=True)
class WidgetTreeOutlineOptions:
    """Controls for human-readable widget-tree outlines."""

    root_class: str | None = None
    include_technical_widgets: bool = False


class WidgetTreeOutlineRenderer(McpDevOutputRenderer):
    """Human-readable outline for a widget-tree MCP payload."""

    output_contract = UiWidgetTreeResult

    MAX_LABEL_CHARS = 96

    @classmethod
    def render_with_options(
        cls,
        response: JsonObject,
        options: WidgetTreeRenderOptions,
    ) -> str:
        if options.output is WidgetTreeOutputFormat.JSON:
            return json.dumps(response, indent=2, sort_keys=True)
        return cls.render(
            response,
            WidgetTreeOutlineOptions(
                root_class=options.outline_root_class,
                include_technical_widgets=options.include_technical_widgets,
            ),
        )

    @classmethod
    def render(
        cls,
        response: Mapping[str, JsonValue],
        options: WidgetTreeOutlineOptions = WidgetTreeOutlineOptions(),
    ) -> str:
        payload = cls._widget_tree_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)

        lines: list[str] = []
        summary = payload.get("summary")
        if isinstance(summary, Mapping):
            lines.extend(cls._summary_lines(summary))

        root = payload.get("root")
        if isinstance(root, Mapping):
            action_summaries = cls._action_summaries_by_path(payload)
            if options.root_class is not None:
                selected_root = cls._first_node_with_class(root, options.root_class)
                if selected_root is None:
                    lines.append(f'Tree: <no node with class "{options.root_class}">')
                    return "\n".join(lines)
                root = selected_root
            if lines:
                lines.append("")
            lines.append("Tree:")
            lines.extend(cls._node_lines(root, "  ", options, action_summaries))
        else:
            lines.append("Tree: <not returned; use --include-tree or outline mode>")

        if payload.get("tree_truncated") is True:
            lines.append("")
            lines.append("Tree truncated by max depth/node limits.")
        if payload.get("actionable_widgets_truncated") is True:
            lines.append("Actionable widget list truncated by max node limits.")
        return "\n".join(lines)

    @classmethod
    def _widget_tree_payload(
        cls,
        response: Mapping[str, JsonValue],
    ) -> Mapping[str, JsonValue] | None:
        results = response.get("results")
        if not isinstance(results, list):
            return None
        for result in results:
            if not isinstance(result, Mapping):
                continue
            if result.get("tool") != agent_capabilities.ui_get_widget_tree.name:
                continue
            payloads = result.get("payloads")
            if not isinstance(payloads, list):
                continue
            for payload in payloads:
                if isinstance(payload, Mapping):
                    return payload
        return None

    @staticmethod
    def _action_summaries_by_path(
        payload: Mapping[str, JsonValue],
    ) -> dict[str, Mapping[str, JsonValue]]:
        actions = payload.get("actionable_widgets")
        if not isinstance(actions, list):
            return {}
        summaries: dict[str, Mapping[str, JsonValue]] = {}
        for action in actions:
            if not isinstance(action, Mapping):
                continue
            path_id = WidgetTreeOutlineRenderer._value_text(action.get("path_id"))
            if path_id:
                summaries[path_id] = action
        return summaries

    @classmethod
    def _first_node_with_class(
        cls,
        node: Mapping[str, JsonValue],
        class_name: str,
    ) -> Mapping[str, JsonValue] | None:
        if node.get("class_name") == class_name:
            return node
        children = node.get("children")
        if not isinstance(children, list):
            return None
        for child in children:
            if not isinstance(child, Mapping):
                continue
            match = cls._first_node_with_class(child, class_name)
            if match is not None:
                return match
        return None

    @classmethod
    def _summary_lines(cls, summary: Mapping[str, JsonValue]) -> list[str]:
        lines = [f"Window: {cls._value_text(summary.get('title'))}"]
        status_parts = [
            f"dirty={cls._value_text(summary.get('dirty'))}",
            f"dirty_fields={cls._value_text(summary.get('dirty_field_count'))}",
            f"default_diff={cls._value_text(summary.get('signature_diff'))}",
            "default_diff_fields="
            f"{cls._value_text(summary.get('signature_diff_field_count'))}",
        ]
        markers = summary.get("semantic_markers")
        if isinstance(markers, list) and markers:
            status_parts.append(
                "markers=" + ",".join(cls._value_text(marker) for marker in markers)
            )
        lines.append("Status: " + " ".join(status_parts))
        return lines

    @classmethod
    def _node_lines(
        cls,
        node: Mapping[str, JsonValue],
        prefix: str,
        options: WidgetTreeOutlineOptions = WidgetTreeOutlineOptions(),
        action_summaries: Mapping[str, Mapping[str, JsonValue]] | None = None,
    ) -> list[str]:
        if action_summaries is None:
            action_summaries = {}
        lines = [f"{prefix}{cls._node_label(node, action_summaries)}"]
        children = node.get("children")
        if isinstance(children, list):
            for child in children:
                if isinstance(child, Mapping):
                    if cls._should_skip_node(child, options):
                        continue
                    lines.extend(
                        cls._node_lines(
                            child,
                            f"{prefix}  ",
                            options,
                            action_summaries,
                        )
                    )
        return lines

    @classmethod
    def _should_skip_node(
        cls,
        node: Mapping[str, JsonValue],
        options: WidgetTreeOutlineOptions,
    ) -> bool:
        if options.include_technical_widgets:
            return False
        class_name = cls._value_text(node.get("class_name"))
        object_name = cls._value_text(node.get("object_name"))
        if class_name in {"QHeaderView", "QScrollBar", "QSplitterHandle"}:
            return True
        if node.get("visible") is False:
            return True
        return object_name.startswith("qt_scrollarea_")

    @classmethod
    def _node_label(
        cls,
        node: Mapping[str, JsonValue],
        action_summaries: Mapping[str, Mapping[str, JsonValue]],
    ) -> str:
        parts: list[str] = []
        class_name = cls._value_text(node.get("class_name"))
        if class_name:
            parts.append(class_name)

        object_name = cls._value_text(node.get("object_name"))
        if object_name:
            parts.append(f"#{object_name}")

        path_id = ""
        action_summary: Mapping[str, JsonValue] | None = None
        if (
            node.get("actionable") is True
            or (node.get("visible") is not False and cls._has_action_kinds(node))
        ):
            path_id = cls._value_text(node.get("path_id"))
            if path_id:
                action_summary = action_summaries.get(path_id)
        semantic_parts = (
            cls._semantic_parts(action_summary)
            if action_summary is not None
            else []
        )

        if not semantic_parts:
            for field in ("label", "text", "title"):
                text = cls._value_text(node.get(field))
                if text:
                    parts.append(f'"{cls._compact_text(text)}"')
                    break

            current_text = cls._value_text(node.get("current_text"))
            if current_text:
                parts.append(f'current="{cls._compact_text(current_text)}"')

        if path_id:
            parts.append(f"path={path_id}")
            parts.extend(semantic_parts)
            parts.extend(cls._interaction_state_parts(node))
        return " ".join(parts)

    @classmethod
    def _has_action_kinds(cls, node: Mapping[str, JsonValue]) -> bool:
        action_kinds = node.get("action_kinds")
        if not isinstance(action_kinds, list):
            return False
        return any(cls._value_text(action_kind) for action_kind in action_kinds)

    @staticmethod
    def _interaction_state_parts(node: Mapping[str, JsonValue]) -> list[str]:
        if node.get("actionable") is True:
            return []
        parts: list[str] = []
        if node.get("visible") is False:
            parts.append("hidden")
        if node.get("enabled") is False:
            parts.append("disabled")
        if node.get("clickable") is False and node.get("enabled") is not False:
            parts.append("not-clickable")
        return parts

    @classmethod
    def _semantic_parts(cls, action_summary: Mapping[str, JsonValue]) -> list[str]:
        scope_id = cls._value_text(action_summary.get("object_state_scope_id"))
        markers = cls._semantic_marker_text(action_summary)
        if not scope_id and not markers:
            return []
        parts = [f"[{markers or '-'}]"]
        if scope_id:
            parts.append(f"scope={cls._compact_text(scope_id)}")
        field_path = cls._value_text(action_summary.get("field_path"))
        if field_path:
            parts.append(f"field={cls._compact_text(field_path)}")
        return parts

    @classmethod
    def _semantic_marker_text(cls, action_summary: Mapping[str, JsonValue]) -> str:
        markers = action_summary.get("semantic_markers")
        if isinstance(markers, list):
            marker_text = "".join(
                cls._value_text(marker)
                for marker in markers
                if cls._value_text(marker)
            )
            if marker_text:
                return marker_text
        marker_text = ""
        if action_summary.get("dirty") is True:
            marker_text += "*"
        if action_summary.get("signature_diff") is True:
            marker_text += "_"
        return marker_text

    @staticmethod
    def _value_text(value: JsonValue) -> str:
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return ""
        return str(value)

    @staticmethod
    def _compact_text(text: str) -> str:
        compact = " ".join(text.split())
        if len(compact) <= WidgetTreeOutlineRenderer.MAX_LABEL_CHARS:
            return compact
        keep = WidgetTreeOutlineRenderer.MAX_LABEL_CHARS - 3
        return f"{compact[:keep]}..."

class CodeDocumentCatalogRenderer(McpDevOutputRenderer):
    """Compact renderer for UI code-document catalogs."""

    output_contract = UiCodeDocumentCatalog

    @classmethod
    def render(cls, response: JsonObject) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)

        documents = McpDevPayloadProjection.sequence_of_mappings(
            payload.get("documents")
        )
        lines = [f"Code documents: count={len(documents)}"]
        ObjectStateScopeRenderer._append_messages(lines, payload)
        if documents:
            lines.append("Documents:")
            for document in documents:
                lines.append(cls._document_line(document))
        return "\n".join(lines)

    @classmethod
    def _document_line(cls, document: Mapping[str, JsonValue]) -> str:
        return (
            "- "
            f"{cls.document_id(document)}: "
            f"widget={McpDevPayloadProjection.text(document.get('widget_id'))} "
            f"readable={McpDevPayloadProjection.text(document.get('readable'))} "
            f"writable={McpDevPayloadProjection.text(document.get('writable'))} "
            "selection="
            f"{McpDevPayloadProjection.text(document.get('current_selection_count'))}/"
            f"{McpDevPayloadProjection.text(document.get('total_scope_count'))} "
            f"modes={ViewerValidationRenderer._sequence_text(document.get('supported_selection_modes'))} "
            f"title={McpDevPayloadProjection.quoted_text(document.get('title'))}"
        )

    @staticmethod
    def document_id(document: Mapping[str, JsonValue]) -> str:
        document_id = document.get("document_id")
        if isinstance(document_id, str):
            return document_id
        identity = McpDevPayloadProjection.nested_mapping(document, "identity")
        identity_document_id = identity.get("document_id")
        if isinstance(identity_document_id, str):
            return identity_document_id
        return "<none>"


class CodeDocumentRenderer(McpDevOutputRenderer):
    """Compact renderer for one UI code document."""

    output_contract = UiCodeDocument

    @classmethod
    def render_with_options(
        cls,
        response: JsonObject,
        options: CodeDocumentRenderOptions,
    ) -> str:
        return cls.render(
            response,
            include_source=options.include_source,
            max_source_chars=options.max_source_chars,
        )

    @classmethod
    def render(
        cls,
        response: JsonObject,
        *,
        include_source: bool = True,
        max_source_chars: int = 12_000,
    ) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)

        summary = McpDevPayloadProjection.nested_mapping(payload, "summary")
        snapshot = McpDevPayloadProjection.nested_mapping(payload, "current_snapshot")
        lines = [
            (
                "Code document: "
                f"id={CodeDocumentCatalogRenderer.document_id(summary)} "
                f"title={McpDevPayloadProjection.quoted_text(summary.get('title'))} "
                f"widget={McpDevPayloadProjection.text(summary.get('widget_id'))} "
                f"writable={McpDevPayloadProjection.text(summary.get('writable'))} "
                f"mode={McpDevPayloadProjection.text(payload.get('selection_mode'))} "
                f"scopes={ViewerValidationRenderer._sequence_text(payload.get('selected_scope_ids'))}"
            ),
            (
                "Revision: "
                f"token={McpDevPayloadProjection.text(payload.get('current_revision_token'))} "
                f"sha256={McpDevPayloadProjection.text(payload.get('sha256'))} "
                f"bytes={McpDevPayloadProjection.text(payload.get('size_bytes'))} "
                f"snapshot={McpDevPayloadProjection.text(snapshot.get('branch'))}@"
                f"{McpDevPayloadProjection.text(snapshot.get('index'))} "
                f"head={McpDevPayloadProjection.text(snapshot.get('is_head'))}"
            ),
        ]
        ObjectStateScopeRenderer._append_messages(lines, payload)
        source = payload.get("source")
        if include_source and isinstance(source, str):
            lines.append("Source:")
            lines.append(cls._source_text(source, max_source_chars=max_source_chars))
        return "\n".join(lines)

    @staticmethod
    def _source_text(source: str, *, max_source_chars: int) -> str:
        if max_source_chars < 0:
            raise ValueError("max_source_chars must be nonnegative.")
        if len(source) <= max_source_chars:
            return source
        return (
            source[:max_source_chars]
            + f"\n...<truncated {len(source) - max_source_chars} chars>"
        )


class CodeDocumentValidationRenderer(McpDevOutputRenderer):
    """Compact renderer for UI code-document validation results."""

    output_contract = UiCodeDocumentValidationResult

    @classmethod
    def render(cls, response: JsonObject) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)

        lines = [
            (
                "Code document validation: "
                f"id={McpDevPayloadProjection.text(payload.get('document_id'))} "
                f"valid={McpDevPayloadProjection.text(payload.get('valid'))} "
                "normalized_scopes="
                f"{ViewerValidationRenderer._sequence_text(payload.get('normalized_scope_ids'))}"
            )
        ]
        ObjectStateScopeRenderer._append_messages(lines, payload)
        return "\n".join(lines)


class CodeDocumentApplyRenderer(McpDevOutputRenderer):
    """Compact renderer for UI code-document apply results."""

    output_contract = UiCodeDocumentApplyResult

    @classmethod
    def render(cls, response: JsonObject) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)

        receipt = McpDevPayloadProjection.nested_mapping(payload, "receipt")
        current_snapshot = McpDevPayloadProjection.nested_mapping(
            payload,
            "current_snapshot",
        )
        undo_snapshot = McpDevPayloadProjection.nested_mapping(
            payload,
            "undo_snapshot",
        )
        lines = [
            (
                "Code document apply: "
                f"id={McpDevPayloadProjection.text(payload.get('document_id'))} "
                f"applied={McpDevPayloadProjection.text(payload.get('applied'))} "
                f"outcome={McpDevPayloadProjection.text(payload.get('outcome'))} "
                f"operation={McpDevPayloadProjection.text(payload.get('operation_id'))}"
            ),
            (
                "Revision: "
                f"base={McpDevPayloadProjection.text(payload.get('base_revision_token'))} "
                f"current={McpDevPayloadProjection.text(payload.get('current_revision_token'))} "
                f"new={McpDevPayloadProjection.text(payload.get('new_revision_token'))}"
            ),
            (
                "Receipt: "
                f"accepted={McpDevPayloadProjection.text(receipt.get('accepted'))} "
                "request_token="
                f"{cls._request_token_text(receipt.get('request_token'))} "
                "bridge_operation="
                f"{McpDevPayloadProjection.text(receipt.get('bridge_operation_id'))}"
            ),
            (
                "Snapshots: "
                f"current={cls._snapshot_text(current_snapshot)} "
                f"undo={cls._snapshot_text(undo_snapshot)}"
            ),
        ]
        ObjectStateScopeRenderer._append_messages(lines, payload)
        return "\n".join(lines)

    @staticmethod
    def _request_token_text(value: JsonValue) -> str:
        if isinstance(value, Mapping):
            return McpDevPayloadProjection.text(value.get("value"))
        return McpDevPayloadProjection.text(value)

    @staticmethod
    def _snapshot_text(snapshot: Mapping[str, JsonValue]) -> str:
        if not snapshot:
            return "<none>"
        branch = McpDevPayloadProjection.text(snapshot.get("branch"))
        index = McpDevPayloadProjection.text(snapshot.get("index"))
        snapshot_id = McpDevPayloadProjection.text(snapshot.get("snapshot_id"))
        return f"{branch}@{index}:{snapshot_id}"


class UiActionCatalogRenderer(McpDevOutputRenderer):
    """Compact renderer for semantic UI action catalogs."""

    output_contract = UiActionCatalog

    @classmethod
    def render_with_options(
        cls,
        response: JsonObject,
        options: UiActionCatalogRenderOptions,
    ) -> str:
        return cls.render(response, widget_id=options.widget_id)

    @classmethod
    def render(cls, response: JsonObject, *, widget_id: str | None = None) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)

        actions = McpDevPayloadProjection.sequence_of_mappings(payload.get("actions"))
        if widget_id is not None:
            actions = tuple(
                action
                for action in actions
                if action.get("widget_id") == widget_id
            )
        header = f"UI actions: count={len(actions)}"
        if widget_id is not None:
            header = f"{header} widget={widget_id}"
        lines = [header]
        cls._append_messages(lines, payload, widget_id=widget_id, actions=actions)
        if actions:
            lines.append("Actions:")
            lines.extend(cls._action_lines(actions))
            disabled_hint_lines = cls._disabled_hint_lines(actions)
            if disabled_hint_lines:
                lines.append("Disabled hints:")
                lines.extend(disabled_hint_lines)
        elif widget_id is not None:
            lines.append(
                "No semantic actions matched this widget. "
                f"Use widget-tree {widget_id} for generic widget action paths."
            )
        return "\n".join(lines)

    @classmethod
    def _append_messages(
        cls,
        lines: list[str],
        payload: Mapping[str, JsonValue],
        *,
        widget_id: str | None,
        actions: tuple[Mapping[str, JsonValue], ...],
    ) -> None:
        errors = McpDevPayloadProjection.sequence_of_mappings(payload.get("errors"))
        if errors:
            lines.append("Errors:")
            lines.extend(ViewerValidationRenderer._error_lines(errors))
        warnings = McpDevPayloadProjection.sequence_of_mappings(payload.get("warnings"))
        warnings = cls._contextual_warnings(
            warnings,
            widget_id=widget_id,
            actions=actions,
        )
        if warnings:
            lines.append("Warnings:")
            lines.extend(ViewerValidationRenderer._error_lines(warnings))

    @staticmethod
    def _contextual_warnings(
        warnings: tuple[Mapping[str, JsonValue], ...],
        *,
        widget_id: str | None,
        actions: tuple[Mapping[str, JsonValue], ...],
    ) -> tuple[Mapping[str, JsonValue], ...]:
        if widget_id is None or not warnings:
            return warnings
        terms = {widget_id}
        terms.update(
            str(value)
            for action in actions
            for value in (action.get("widget_id"), action.get("action_id"))
            if value not in (None, "")
        )
        normalized_terms = tuple(term.lower() for term in terms if term)
        if not normalized_terms:
            return ()
        return tuple(
            warning
            for warning in warnings
            if any(
                term in UiActionCatalogRenderer._warning_text(warning)
                for term in normalized_terms
            )
        )

    @staticmethod
    def _warning_text(warning: Mapping[str, JsonValue]) -> str:
        return " ".join(
            McpDevPayloadProjection.text(warning.get(key)).lower()
            for key in ("code", "message", "hint")
        )

    @classmethod
    def _action_lines(
        cls,
        actions: tuple[Mapping[str, JsonValue], ...],
    ) -> list[str]:
        return [cls._action_line(action) for action in actions]

    @staticmethod
    def _disabled_hint_lines(
        actions: tuple[Mapping[str, JsonValue], ...],
    ) -> list[str]:
        action_keys_by_hint: dict[str, list[str]] = {}
        for action in actions:
            disabled_error = McpDevPayloadProjection.nested_mapping(
                action,
                "disabled_error",
            )
            if not disabled_error:
                continue
            hint_value = disabled_error.get("hint")
            if hint_value in (None, ""):
                continue
            hint = McpDevPayloadProjection.text(hint_value)
            action_keys_by_hint.setdefault(hint, []).append(
                (
                    f"{McpDevPayloadProjection.text(action.get('widget_id'))}/"
                    f"{McpDevPayloadProjection.text(action.get('action_id'))}"
                )
            )
        return [
            f"- {','.join(action_keys)}: {McpDevPayloadProjection.quoted_text(hint)}"
            for hint, action_keys in action_keys_by_hint.items()
        ]

    @staticmethod
    def _action_line(action: Mapping[str, JsonValue]) -> str:
        disabled_error = McpDevPayloadProjection.nested_mapping(
            action,
            "disabled_error",
        )
        disabled_text = ""
        if disabled_error:
            disabled_text = (
                " disabled="
                f"{McpDevPayloadProjection.text(disabled_error.get('code'))}:"
                f"{McpDevPayloadProjection.text(disabled_error.get('message'))}"
            )
        return (
            "- "
            f"{McpDevPayloadProjection.text(action.get('widget_id'))}/"
            f"{McpDevPayloadProjection.text(action.get('action_id'))}: "
            f"title={McpDevPayloadProjection.quoted_text(action.get('title'))} "
            f"enabled={McpDevPayloadProjection.text(action.get('enabled'))} "
            f"confirm={McpDevPayloadProjection.text(action.get('confirmation_required'))} "
            f"mode={McpDevPayloadProjection.text(action.get('invocation_mode'))} "
            "selection="
            f"{McpDevPayloadProjection.text(action.get('current_selection_count'))} "
            "targets="
            f"{ViewerValidationRenderer._sequence_text(action.get('target_scope_ids'))} "
            "selection_rev="
            f"{McpDevPayloadProjection.text(action.get('selection_revision_token'))} "
            f"effects={ViewerValidationRenderer._sequence_text(action.get('side_effects'))}"
            f"{disabled_text}"
        )


class UiActionInvokeRenderer(McpDevOutputRenderer):
    """Compact renderer for semantic UI action invocation results."""

    output_contract = UiActionInvokeResult

    @classmethod
    def render_with_options(
        cls,
        response: JsonObject,
        options: UiActionInvokeRenderOptions,
    ) -> str:
        return cls.render(
            response,
            widget_id=options.widget_id,
            action_id=options.action_id,
        )

    @classmethod
    def render(
        cls,
        response: JsonObject,
        *,
        widget_id: str | None = None,
        action_id: str | None = None,
    ) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)

        receipt = McpDevPayloadProjection.nested_mapping(payload, "receipt")
        resolved_widget_id = payload.get("widget_id")
        resolved_action_id = payload.get("action_id")
        lines = [
            (
                "UI action invoke: "
                f"action={McpDevPayloadProjection.text(resolved_widget_id or widget_id)}/"
                f"{McpDevPayloadProjection.text(resolved_action_id or action_id)} "
                f"status={McpDevPayloadProjection.text(payload.get('status'))}"
            ),
            (
                "Receipt: "
                f"accepted={McpDevPayloadProjection.text(receipt.get('accepted'))} "
                "request_token="
                f"{CodeDocumentApplyRenderer._request_token_text(receipt.get('request_token'))} "
                "bridge_operation="
                f"{McpDevPayloadProjection.text(receipt.get('bridge_operation_id'))}"
            ),
            (
                "Selection: "
                "targets="
                f"{ViewerValidationRenderer._sequence_text(payload.get('target_scope_ids'))} "
                "selection_rev="
                f"{McpDevPayloadProjection.text(payload.get('selection_revision_token'))}"
            ),
            (
                "Polling: "
                "surfaces="
                f"{ViewerValidationRenderer._sequence_text(payload.get('workflow_status_surface_ids'))} "
                "interval_ms="
                f"{McpDevPayloadProjection.text(payload.get('recommended_poll_interval_ms'))}"
            ),
        ]
        ObjectStateScopeRenderer._append_messages(lines, payload)
        return "\n".join(lines)


class UiWidgetActionInvokeRenderer(McpDevOutputRenderer):
    """Compact renderer for generic widget action invocation results."""

    output_contract = UiWidgetActionInvokeResult

    @classmethod
    def render(cls, response: JsonObject) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)

        receipt = McpDevPayloadProjection.nested_mapping(payload, "receipt")
        summary = McpDevPayloadProjection.nested_mapping(payload, "summary")
        lines = [
            (
                "Widget action invoke: "
                f"window={McpDevPayloadProjection.text(payload.get('window_id'))} "
                f"path={McpDevPayloadProjection.text(payload.get('path_id'))} "
                f"kind={McpDevPayloadProjection.text(payload.get('action_kind'))} "
                f"invoked={McpDevPayloadProjection.text(payload.get('invoked'))}"
            ),
            (
                "Receipt: "
                f"accepted={McpDevPayloadProjection.text(receipt.get('accepted'))} "
                "request_token="
                f"{CodeDocumentApplyRenderer._request_token_text(receipt.get('request_token'))} "
                "bridge_operation="
                f"{McpDevPayloadProjection.text(receipt.get('bridge_operation_id'))}"
            ),
        ]
        if summary:
            lines.append(
                "Widget: "
                f"label={McpDevPayloadProjection.quoted_text(summary.get('label'))} "
                f"enabled={McpDevPayloadProjection.text(summary.get('enabled'))} "
                f"clickable={McpDevPayloadProjection.text(summary.get('clickable'))} "
                "actions="
                f"{ViewerValidationRenderer._sequence_text(summary.get('action_kinds'))}"
            )
        ObjectStateScopeRenderer._append_messages(lines, payload)
        return "\n".join(lines)
