"""ObjectState renderers for the MCP dev client."""

from __future__ import annotations

import json
from collections.abc import Mapping

from openhcs.agent.dto.common import JsonObject, JsonValue
from openhcs.agent.dto.ui_bridge import (
    UiObjectStateFieldFilter,
    UiObjectStateFieldHelpResult,
    UiObjectStateFieldListResult,
    UiObjectStateFieldMutationResult,
    UiObjectStateScopeCatalog,
)
from openhcs.mcp.dev_client_core import optional_int
from openhcs.mcp.dev_client_rendering import (
    McpDevOutputRenderer,
    McpDevPayloadProjection,
)
from openhcs.mcp.dev_client_renderers.viewer import ViewerValidationRenderer

class ObjectStateScopeRenderer(McpDevOutputRenderer):
    """Compact renderer for ObjectState scope catalogs."""

    output_contract = UiObjectStateScopeCatalog

    MARKER_LEGEND = "Markers: [*]=unsaved/dirty [_]=differs-from-defaults [-]=clean"

    @classmethod
    def render(cls, response: JsonObject) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)

        scopes = McpDevPayloadProjection.sequence_of_mappings(payload.get("scopes"))
        lines = [
            (
                "ObjectState scopes: "
                f"scopes={McpDevPayloadProjection.text(len(scopes))} "
                f"token={McpDevPayloadProjection.text(payload.get('object_state_token'))} "
                f"branch={McpDevPayloadProjection.text(payload.get('current_branch'))} "
                f"snapshot={McpDevPayloadProjection.text(payload.get('current_snapshot_index'))} "
                f"active={McpDevPayloadProjection.text(payload.get('active'))}"
            )
        ]
        lines.append(cls.MARKER_LEGEND)
        cls._append_messages(lines, payload)
        if scopes:
            lines.append("Scopes:")
            for scope in scopes:
                lines.append(cls._scope_line(scope))
                fields = McpDevPayloadProjection.sequence_of_mappings(scope.get("fields"))
                if fields:
                    lines.extend(
                        f"  {line}"
                        for line in ObjectStateFieldRenderer.field_lines(fields)
                    )
            next_field_page = cls._next_field_page(scopes)
            if next_field_page is not None:
                offset, limit = next_field_page
                lines.append(
                    "Next field page: rerun with "
                    f"--include-fields --field-offset {offset} --field-limit {limit}"
                )
        return "\n".join(lines)

    @classmethod
    def _scope_line(cls, scope: Mapping[str, JsonValue]) -> str:
        scope_id = cls.scope_id(scope)
        mark = cls._scope_mark(scope)
        line = (
            f"- [{mark}] scope={scope_id}: "
            f"type={McpDevPayloadProjection.text(scope.get('object_type'))} "
            f"params={McpDevPayloadProjection.text(scope.get('parameter_count'))} "
            f"dirty={McpDevPayloadProjection.text(scope.get('dirty_field_count'))} "
            "default_diff="
            f"{McpDevPayloadProjection.text(scope.get('signature_diff_field_count'))} "
            f"unsaved={McpDevPayloadProjection.text(scope.get('has_unsaved_changes'))} "
            f"overrides={McpDevPayloadProjection.text(scope.get('has_default_overrides'))} "
            f"changed={McpDevPayloadProjection.text(scope.get('last_changed_field'))}"
        )
        field_page_text = cls._field_page_text(scope)
        if field_page_text is not None:
            line += f" {field_page_text}"
        return line

    @staticmethod
    def _field_page_text(scope: Mapping[str, JsonValue]) -> str | None:
        field_page = McpDevPayloadProjection.nested_mapping(scope, "field_page")
        if not field_page:
            return None
        returned = McpDevPayloadProjection.text(field_page.get("returned_count"))
        total = McpDevPayloadProjection.text(field_page.get("total_count"))
        next_offset = field_page.get("next_offset")
        text = f"fields={returned}/{total}"
        if next_offset is not None:
            text += f" next={McpDevPayloadProjection.text(next_offset)}"
        return text

    @classmethod
    def _next_field_page(
        cls,
        scopes: tuple[Mapping[str, JsonValue], ...],
    ) -> tuple[int, int] | None:
        for scope in scopes:
            field_page = McpDevPayloadProjection.nested_mapping(scope, "field_page")
            next_offset = optional_int(field_page.get("next_offset"))
            limit = optional_int(field_page.get("limit"))
            if next_offset is not None and limit is not None:
                return next_offset, limit
        return None

    @staticmethod
    def scope_id(scope: Mapping[str, JsonValue]) -> str:
        identity = McpDevPayloadProjection.nested_mapping(scope, "identity")
        identity_scope = identity.get("object_state_scope_id")
        if isinstance(identity_scope, str):
            return identity_scope
        scope_id = scope.get("scope_id")
        if isinstance(scope_id, str):
            return scope_id
        return "<none>"

    @staticmethod
    def _scope_mark(scope: Mapping[str, JsonValue]) -> str:
        marks: list[str] = []
        if scope.get("has_unsaved_changes") is True or optional_int(scope.get("dirty_field_count")) not in (None, 0):
            marks.append("*")
        if scope.get("has_default_overrides") is True or optional_int(scope.get("signature_diff_field_count")) not in (None, 0):
            marks.append("_")
        return "".join(marks) or "-"

    @staticmethod
    def _append_messages(lines: list[str], payload: Mapping[str, JsonValue]) -> None:
        errors = McpDevPayloadProjection.sequence_of_mappings(payload.get("errors"))
        if errors:
            lines.append("Errors:")
            lines.extend(ViewerValidationRenderer._error_lines(errors))
        warnings = McpDevPayloadProjection.sequence_of_mappings(payload.get("warnings"))
        if warnings:
            lines.append("Warnings:")
            lines.extend(ViewerValidationRenderer._error_lines(warnings))


class ObjectStateFieldRenderer(McpDevOutputRenderer):
    """Compact renderer for ObjectState field projections."""

    output_contract = UiObjectStateFieldListResult

    @classmethod
    def render(cls, response: JsonObject) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)

        scopes = McpDevPayloadProjection.sequence_of_mappings(payload.get("scopes"))
        page_text = cls._page_text(payload)
        lines = [
            (
                "ObjectState fields: "
                f"scopes={McpDevPayloadProjection.text(payload.get('matched_scope_count'))} "
                f"fields={McpDevPayloadProjection.text(payload.get('matched_field_count'))} "
                f"{page_text}"
                f"truncated={McpDevPayloadProjection.text(payload.get('truncated'))} "
                f"token={McpDevPayloadProjection.text(payload.get('object_state_token'))} "
                f"branch={McpDevPayloadProjection.text(payload.get('current_branch'))} "
                f"snapshot={McpDevPayloadProjection.text(payload.get('current_snapshot_index'))}"
            )
        ]
        lines.append(ObjectStateScopeRenderer.MARKER_LEGEND)
        semantic_summary = cls._returned_semantic_summary(scopes)
        if semantic_summary is not None:
            lines.append(semantic_summary)
        cls._append_filters(lines, payload)
        ObjectStateScopeRenderer._append_messages(lines, payload)
        if scopes:
            lines.append("Scopes:")
            for scope in scopes:
                lines.append(
                    "Scope "
                    f"[{ObjectStateScopeRenderer._scope_mark(scope)}] "
                    f"scope={ObjectStateScopeRenderer.scope_id(scope)}: "
                    f"type={McpDevPayloadProjection.text(scope.get('object_type'))} "
                    f"dirty={McpDevPayloadProjection.text(scope.get('dirty_field_count'))} "
                    "default_diff="
                    f"{McpDevPayloadProjection.text(scope.get('signature_diff_field_count'))} "
                    f"unsaved={McpDevPayloadProjection.text(scope.get('has_unsaved_changes'))} "
                    f"overrides={McpDevPayloadProjection.text(scope.get('has_default_overrides'))}"
                )
                lines.extend(cls.field_lines(McpDevPayloadProjection.sequence_of_mappings(scope.get("fields"))))
        return "\n".join(lines)

    @staticmethod
    def _page_text(payload: Mapping[str, JsonValue]) -> str:
        if "returned_field_count" not in payload:
            return ""
        parts = [
            f"returned={McpDevPayloadProjection.text(payload.get('returned_field_count'))}",
            f"offset={McpDevPayloadProjection.text(payload.get('field_offset'))}",
            f"limit={McpDevPayloadProjection.text(payload.get('field_limit'))}",
        ]
        next_offset = payload.get("next_offset")
        if next_offset is not None:
            parts.append(f"next={McpDevPayloadProjection.text(next_offset)}")
        return " ".join(parts) + " "

    @classmethod
    def _returned_semantic_summary(
        cls,
        scopes: tuple[Mapping[str, JsonValue], ...],
    ) -> str | None:
        fields = tuple(
            field
            for scope in scopes
            for field in McpDevPayloadProjection.sequence_of_mappings(
                scope.get("fields")
            )
        )
        if not fields:
            return None
        dirty_count = sum(1 for field in fields if field.get("dirty") is True)
        default_diff_count = sum(
            1 for field in fields if field.get("signature_diff") is True
        )
        inherited_count = sum(
            1 for field in fields if field.get("inherited_value") is True
        )
        raw_none_resolved_count = sum(
            1 for field in fields if cls._raw_none_resolved_value_present(field)
        )
        resolved_none_raw_count = sum(
            1 for field in fields if cls._resolved_none_raw_value_present(field)
        )
        semantic_count = sum(
            1
            for field in fields
            if (
                field.get("dirty") is True
                or field.get("signature_diff") is True
                or field.get("inherited_value") is True
                or cls._raw_resolved_none_state_differs(field)
            )
        )
        plain_count = len(fields) - semantic_count
        return (
            "Returned semantics: "
            f"dirty={dirty_count} default_diff={default_diff_count} "
            f"inherited={inherited_count} "
            f"raw_none_resolved={raw_none_resolved_count} "
            f"resolved_none_raw={resolved_none_raw_count} "
            f"plain={plain_count}"
        )

    @classmethod
    def _raw_none_resolved_value_present(
        cls,
        field: Mapping[str, JsonValue],
    ) -> bool:
        return (
            cls._value_is_none(field, "raw_value_is_none", "raw_value_preview")
            is True
            and cls._value_is_none(
                field,
                "resolved_value_is_none",
                "resolved_value_preview",
            )
            is False
        )

    @classmethod
    def _resolved_none_raw_value_present(
        cls,
        field: Mapping[str, JsonValue],
    ) -> bool:
        return (
            cls._value_is_none(field, "raw_value_is_none", "raw_value_preview")
            is False
            and cls._value_is_none(
                field,
                "resolved_value_is_none",
                "resolved_value_preview",
            )
            is False
        )

    @classmethod
    def _raw_resolved_none_state_differs(
        cls,
        field: Mapping[str, JsonValue],
    ) -> bool:
        raw_is_none = cls._value_is_none(
            field,
            "raw_value_is_none",
            "raw_value_preview",
        )
        resolved_is_none = cls._value_is_none(
            field,
            "resolved_value_is_none",
            "resolved_value_preview",
        )
        return (
            raw_is_none is not None
            and resolved_is_none is not None
            and raw_is_none is not resolved_is_none
        )

    @staticmethod
    def _value_is_none(
        field: Mapping[str, JsonValue],
        direct_key: str,
        preview_key: str,
    ) -> bool | None:
        direct_value = field.get(direct_key)
        if isinstance(direct_value, bool):
            return direct_value
        preview = McpDevPayloadProjection.nested_mapping(field, preview_key)
        preview_value = preview.get("is_none")
        if isinstance(preview_value, bool):
            return preview_value
        return None

    @classmethod
    def field_lines(
        cls,
        fields: tuple[Mapping[str, JsonValue], ...],
    ) -> list[str]:
        return [cls._field_line(field) for field in fields]

    @classmethod
    def _field_line(cls, field: Mapping[str, JsonValue]) -> str:
        return (
            f"  [{cls._field_mark(field)}] "
            f"{McpDevPayloadProjection.text(cls._field_path(field))}: "
            f"target={cls._short_type(field.get('object_state_path_type'))} "
            f"raw={cls._preview_text(field, 'raw_value_preview', 'raw_value')} "
            "-> "
            f"resolved={cls._preview_text(field, 'resolved_value_preview', 'resolved_value')} "
            f"inherited={McpDevPayloadProjection.text(field.get('inherited_value'))} "
            f"provenance={cls._provenance_text(field.get('provenance'))}"
        )

    @staticmethod
    def _field_path(field: Mapping[str, JsonValue]) -> JsonValue:
        field_path = field.get("field_path")
        if isinstance(field_path, str):
            return field_path
        address = McpDevPayloadProjection.nested_mapping(field, "address")
        return address.get("field_path")

    @staticmethod
    def _field_mark(field: Mapping[str, JsonValue]) -> str:
        marks: list[str] = []
        if field.get("dirty") is True:
            marks.append("*")
        if field.get("signature_diff") is True:
            marks.append("_")
        semantic_markers = field.get("semantic_markers")
        if isinstance(semantic_markers, list):
            for marker in semantic_markers:
                marker_text = McpDevPayloadProjection.text(marker)
                if marker_text and marker_text not in marks:
                    marks.append(marker_text)
        return "".join(marks) or "-"

    @staticmethod
    def _preview_text(
        field: Mapping[str, JsonValue],
        preview_key: str,
        value_key: str,
    ) -> str:
        preview = McpDevPayloadProjection.nested_mapping(field, preview_key)
        text = preview.get("text")
        if isinstance(text, str):
            return text
        return McpDevPayloadProjection.text(field.get(value_key))

    @classmethod
    def _provenance_text(cls, value: JsonValue) -> str:
        if not isinstance(value, Mapping):
            return "<none>"
        scope = McpDevPayloadProjection.text(value.get("source_scope_id"))
        field = McpDevPayloadProjection.text(value.get("source_field_path"))
        source_type = cls._short_type(value.get("source_type"))
        return f"{scope}:{field} ({source_type})"

    @staticmethod
    def _short_type(value: JsonValue) -> str:
        if not isinstance(value, str) or not value:
            return "<none>"
        return value.rsplit(".", 1)[-1]

    @staticmethod
    def _append_filters(lines: list[str], payload: Mapping[str, JsonValue]) -> None:
        requested_scope_ids = payload.get("requested_scope_ids")
        field_paths = payload.get("field_paths")
        field_path_contains = payload.get("field_path_contains")
        filters: list[str] = []
        if isinstance(requested_scope_ids, list) and requested_scope_ids:
            filters.append(
                "scope_ids="
                + ",".join(
                    McpDevPayloadProjection.text(item)
                    for item in requested_scope_ids
                )
            )
        if isinstance(field_paths, list) and field_paths:
            filters.append(
                "field_paths="
                + ",".join(
                    McpDevPayloadProjection.text(item) for item in field_paths
                )
            )
        if isinstance(field_path_contains, list) and field_path_contains:
            filters.append(
                "contains="
                + ",".join(
                    McpDevPayloadProjection.text(item)
                    for item in field_path_contains
                )
            )
        field_filter = payload.get("field_filter")
        if (
            isinstance(field_filter, str)
            and field_filter != UiObjectStateFieldFilter.ALL.value
        ):
            filters.append(f"field_filter={field_filter}")
        if payload.get("include_container_fields") is True:
            filters.append("include_container_fields=True")
        if filters:
            lines.append("Filters: " + " ".join(filters))


class ObjectStateFieldHelpRenderer(McpDevOutputRenderer):
    """Compact renderer for one ObjectState field help result."""

    output_contract = UiObjectStateFieldHelpResult

    MAX_TARGET_SUMMARY_CHARS = 220

    @classmethod
    def render(cls, response: JsonObject) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)

        address = McpDevPayloadProjection.nested_mapping(payload, "address")
        field = McpDevPayloadProjection.nested_mapping(payload, "field")
        lines = [
            (
                "ObjectState field help: "
                f"scope={McpDevPayloadProjection.text(address.get('object_state_scope_id'))} "
                f"field={McpDevPayloadProjection.text(address.get('field_path'))}"
            ),
            (
                "Target: "
                f"object={cls._short_type(payload.get('object_type'))} "
                f"help_target={cls._short_type(payload.get('help_target_type'))} "
                f"parameter={McpDevPayloadProjection.text(payload.get('parameter_name'))}"
            ),
        ]
        ObjectStateScopeRenderer._append_messages(lines, payload)
        if field:
            lines.append("Field:")
            lines.extend(ObjectStateFieldRenderer.field_lines((field,)))
        target_summary = payload.get("target_summary")
        if isinstance(target_summary, str) and target_summary:
            lines.append(f"Target summary: {cls._compact_target_summary(target_summary)}")
        summary = payload.get("summary")
        if isinstance(summary, str) and summary:
            lines.append(f"Summary: {summary}")
        description = payload.get("description")
        if isinstance(description, str) and description:
            lines.append("Description:")
            lines.append(description)
        if payload.get("description_truncated") is True:
            lines.append("Description truncated; rerun with a larger max_description_chars.")
        return "\n".join(lines)

    @staticmethod
    def _short_type(value: JsonValue) -> str:
        if not isinstance(value, str) or not value:
            return "<none>"
        return value.rsplit(".", 1)[-1]

    @classmethod
    def _compact_target_summary(cls, target_summary: str) -> str:
        compact = " ".join(target_summary.split())
        if len(compact) <= cls.MAX_TARGET_SUMMARY_CHARS:
            return compact
        return f"{compact[: cls.MAX_TARGET_SUMMARY_CHARS - 3]}..."


class ObjectStateFieldMutationRenderer(McpDevOutputRenderer):
    """Compact renderer for one ObjectState field update/reset result."""

    output_contract = UiObjectStateFieldMutationResult

    @classmethod
    def render(cls, response: JsonObject) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)

        address = McpDevPayloadProjection.nested_mapping(payload, "address")
        receipt = McpDevPayloadProjection.nested_mapping(payload, "receipt")
        lines = [
            (
                "ObjectState field mutation: "
                f"scope={McpDevPayloadProjection.text(address.get('object_state_scope_id'))} "
                f"field={McpDevPayloadProjection.text(address.get('field_path'))} "
                f"mutated={McpDevPayloadProjection.text(payload.get('mutated'))} "
                f"reset={McpDevPayloadProjection.text(payload.get('reset'))}"
            ),
            (
                "Receipt: "
                f"accepted={McpDevPayloadProjection.text(receipt.get('accepted'))} "
                f"operation={McpDevPayloadProjection.text(receipt.get('bridge_operation_id'))}"
            ),
        ]
        ObjectStateScopeRenderer._append_messages(lines, payload)
        before = McpDevPayloadProjection.nested_mapping(payload, "before")
        after = McpDevPayloadProjection.nested_mapping(payload, "after")
        if before:
            lines.append("Before:")
            lines.extend(ObjectStateFieldRenderer.field_lines((before,)))
        if after:
            lines.append("After:")
            lines.extend(ObjectStateFieldRenderer.field_lines((after,)))
        return "\n".join(lines)
