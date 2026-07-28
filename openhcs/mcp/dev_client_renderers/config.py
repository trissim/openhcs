"""Configuration-schema rendering for the OpenHCS MCP dev client."""

from __future__ import annotations

import json
from collections.abc import Mapping

from openhcs.agent.dto.common import JsonObject, JsonValue
from openhcs.agent.dto.config import (
    ConfigFieldSchema,
    ConfigSchema,
    ConfigTypeSchema,
)
from openhcs.mcp.dev_client_rendering import (
    CatalogRenderOptions,
    McpDiagnosticRenderer,
    McpDevOutputRenderer,
    McpDevPayloadProjection,
)


class ConfigSchemaRenderer(McpDevOutputRenderer):
    """Render reflected config fields as a compact, searchable catalog."""

    output_contract = ConfigSchema
    render_options_type = CatalogRenderOptions

    @classmethod
    def render_with_options(
        cls,
        response: JsonObject,
        options: CatalogRenderOptions,
    ) -> str:
        return cls.render(
            response,
            contains=options.contains,
            limit=options.limit,
        )

    @classmethod
    def render(
        cls,
        response: JsonObject,
        *,
        contains: str | None = None,
        limit: int = 20,
    ) -> str:
        error_lines = McpDiagnosticRenderer.response_error_lines(response)
        if error_lines:
            return "\n".join(("Config schema: unavailable", *error_lines))

        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)

        all_fields = McpDevPayloadProjection.sequence_of_mappings(
            payload.get(ConfigSchema.fields.__name__)
        )
        matched_fields = cls._matching_fields(all_fields, contains)
        visible_fields = matched_fields[: max(limit, 0)]
        registries = McpDevPayloadProjection.sequence_of_mappings(
            payload.get(ConfigSchema.registries.__name__)
        )
        types = McpDevPayloadProjection.sequence_of_mappings(
            payload.get(ConfigSchema.types.__name__)
        )
        path_prefix = payload.get(ConfigSchema.path_prefix.__name__)
        path_text = "<root>" if path_prefix is None else str(path_prefix)
        lines = [
            (
                "Config schema: "
                "type="
                f"{McpDevPayloadProjection.text(payload.get(ConfigSchema.config_type.__name__))} "
                f"path={path_text} "
                "authoring="
                f"{McpDevPayloadProjection.text(payload.get(ConfigSchema.authoring_path.__name__))}"
            ),
            (
                "Fields: "
                f"total={len(all_fields)} matched={len(matched_fields)} "
                f"shown={len(visible_fields)} registries={len(registries)} "
                f"types={len(types)}"
            ),
        ]
        if contains:
            lines.append(f"Filter: contains={contains}")
        if visible_fields:
            lines.append("Field paths:")
            lines.extend(cls._field_line(field) for field in visible_fields)
        if len(visible_fields) < len(matched_fields):
            lines.append(
                f"...<truncated {len(matched_fields) - len(visible_fields)} fields>"
            )
        if types:
            lines.append("Type inheritance (declaration-derived):")
            lines.extend(cls._type_line(type_schema) for type_schema in types)
        return "\n".join(lines)

    @classmethod
    def _matching_fields(
        cls,
        fields: tuple[Mapping[str, JsonValue], ...],
        contains: str | None,
    ) -> tuple[Mapping[str, JsonValue], ...]:
        if not contains:
            return fields
        needle = contains.casefold()
        return tuple(
            field for field in fields if needle in cls._field_line(field).casefold()
        )

    @classmethod
    def _field_line(cls, field: Mapping[str, JsonValue]) -> str:
        parts = [
            f"- {McpDevPayloadProjection.text(field.get(ConfigFieldSchema.path.__name__))}:",
            McpDevPayloadProjection.text(
                field.get(ConfigFieldSchema.type_repr.__name__)
            ),
        ]
        value_type = field.get(ConfigFieldSchema.value_type_repr.__name__)
        if value_type is not None:
            parts.append(f"value={value_type}")
        default = field.get(ConfigFieldSchema.default_repr.__name__)
        if default is not None:
            parts.append(f"default={default}")
        parts.append(f"flags={','.join(cls._field_flags(field))}")
        nested_path = field.get(ConfigFieldSchema.nested_schema_path.__name__)
        if nested_path is not None:
            parts.append(f"nested={nested_path}")
        authoring_value_path = field.get(
            ConfigFieldSchema.authoring_value_path.__name__
        )
        if isinstance(authoring_value_path, list) and authoring_value_path:
            parts.append(
                "authoring="
                + "/".join(
                    str(segment)
                    for segment in authoring_value_path
                    if isinstance(segment, str)
                )
            )
        enum_values = cls._scalar_values_text(
            field.get(ConfigFieldSchema.enum_values.__name__)
        )
        if enum_values:
            parts.append(f"enum={enum_values}")
        registry_values = cls._scalar_values_text(
            field.get(ConfigFieldSchema.registry_values.__name__)
        )
        if registry_values:
            parts.append(f"registry={registry_values}")
        description = field.get(ConfigFieldSchema.description.__name__)
        if isinstance(description, str) and description.strip():
            parts.append(f"help={json.dumps(cls._compact_text(description))}")
        return " ".join(parts)

    @classmethod
    def _type_line(cls, type_schema: Mapping[str, JsonValue]) -> str:
        type_repr = McpDevPayloadProjection.text(
            type_schema.get(ConfigTypeSchema.type_repr.__name__)
        )
        base_types = cls._scalar_values_text(
            type_schema.get(ConfigTypeSchema.base_types.__name__),
            limit=12,
        )
        return f"- {type_repr} extends={base_types or '<none>'}"

    @staticmethod
    def _field_flags(field: Mapping[str, JsonValue]) -> tuple[str, ...]:
        flags = [
            "required"
            if field.get(ConfigFieldSchema.required.__name__) is True
            else "optional"
        ]
        for descriptor in (
            ConfigFieldSchema.lazy,
            ConfigFieldSchema.inheritable,
            ConfigFieldSchema.ui_hidden,
        ):
            if field.get(descriptor.__name__) is True:
                flags.append(descriptor.__name__)
        return tuple(flags)

    @staticmethod
    def _scalar_values_text(value: JsonValue, limit: int = 8) -> str:
        if not isinstance(value, list):
            return ""
        values = tuple(
            str(item) for item in value if not isinstance(item, (dict, list))
        )
        visible_values = values[:limit]
        text = ",".join(visible_values)
        if len(visible_values) < len(values):
            text += f",+{len(values) - len(visible_values)}"
        return text

    @staticmethod
    def _compact_text(value: str, limit: int = 180) -> str:
        compact = " ".join(value.split())
        if len(compact) <= limit:
            return compact
        return f"{compact[: limit - 3]}..."
