"""Shared rendering contracts for the OpenHCS MCP dev client."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, TypeAlias

from metaclass_registry import AutoRegisterMeta

from openhcs.agent.dto.common import JsonObject, JsonValue

DEFAULT_CODE_DOCUMENT_MAX_CHARS = 2_000


class WidgetTreeOutputFormat(str, Enum):
    """CLI presentation formats for widget-tree command output."""

    JSON = "json"
    OUTLINE = "outline"

    @classmethod
    def choices(cls) -> tuple[str, ...]:
        return tuple(output_format.value for output_format in cls)


@dataclass(frozen=True, slots=True)
class McpDevOutputRenderOptions:
    """Typed presentation options for an output-contract renderer."""


@dataclass(frozen=True, slots=True)
class AuthoringContextRenderOptions(McpDevOutputRenderOptions):
    max_chars: int = 2_000


@dataclass(frozen=True, slots=True)
class CodeDocumentRenderOptions(McpDevOutputRenderOptions):
    include_source: bool = True
    max_source_chars: int = DEFAULT_CODE_DOCUMENT_MAX_CHARS


@dataclass(frozen=True, slots=True)
class UiActionCatalogRenderOptions(McpDevOutputRenderOptions):
    widget_id: str | None = None


@dataclass(frozen=True, slots=True)
class UiActionInvokeRenderOptions(McpDevOutputRenderOptions):
    widget_id: str | None = None
    action_id: str | None = None


@dataclass(frozen=True, slots=True)
class ViewerImageSampleRenderOptions(McpDevOutputRenderOptions):
    include_array_values_requested: bool | None = None


@dataclass(frozen=True, slots=True)
class CatalogRenderOptions(McpDevOutputRenderOptions):
    contains: str | None = None
    limit: int = 20


@dataclass(frozen=True, slots=True)
class WidgetTreeRenderOptions(McpDevOutputRenderOptions):
    output: WidgetTreeOutputFormat = WidgetTreeOutputFormat.OUTLINE
    outline_root_class: str | None = None
    include_technical_widgets: bool = False


McpDevOutputRendererKey: TypeAlias = type
McpDevOutputRenderFunction: TypeAlias = Callable[[JsonObject], str]


def mcp_dev_output_renderer_key(
    name: str,
    renderer_type: type,
) -> McpDevOutputRendererKey | None:
    """Return the declared output renderer key for a dev-client renderer."""
    del name
    declared_output_contract = vars(renderer_type).get("output_contract")
    if isinstance(declared_output_contract, type):
        return declared_output_contract
    return None


class McpDevOutputRenderer(metaclass=AutoRegisterMeta):
    """Registered compact renderer keyed by an agent output contract."""

    __registry__: ClassVar[
        dict[McpDevOutputRendererKey, type["McpDevOutputRenderer"]]
    ] = {}
    __registry_key__ = "renderer_key"
    __key_extractor__ = mcp_dev_output_renderer_key
    __skip_if_no_key__ = True

    renderer_key: ClassVar[McpDevOutputRendererKey | None] = None
    output_contract: ClassVar[type | None] = None
    __renderer_types__: ClassVar[tuple[type["McpDevOutputRenderer"], ...]] = ()

    def __init_subclass__(cls, **kwargs: JsonValue) -> None:
        super().__init_subclass__(**kwargs)
        McpDevOutputRenderer.__renderer_types__ = (
            *McpDevOutputRenderer.__renderer_types__,
            cls,
        )

    @classmethod
    def for_output_contract(
        cls,
        output_contract: type | None,
    ) -> "McpDevOutputRendererBinding | None":
        if output_contract is None:
            return None
        for renderer_type in cls.__renderer_types__:
            binding = renderer_type.binding_for_output_contract(output_contract)
            if binding is not None:
                return binding
        return None

    @classmethod
    def binding_for_output_contract(
        cls,
        output_contract: type,
    ) -> "McpDevOutputRendererBinding | None":
        declared_output_contract = vars(cls).get("output_contract")
        if declared_output_contract is output_contract:
            return McpDevOutputRendererBinding(
                output_contract=output_contract,
                renderer_type=cls,
            )
        for binding in cls.render_bindings():
            if binding.output_contract is output_contract:
                return binding
        return None

    @classmethod
    def render_bindings(cls) -> tuple["McpDevOutputRendererBinding", ...]:
        """Return additional output-contract bindings owned by this renderer."""
        return ()

    @classmethod
    def render(cls, response: JsonObject) -> str:
        raise NotImplementedError

    @classmethod
    def render_with_options(
        cls,
        response: JsonObject,
        options: McpDevOutputRenderOptions,
    ) -> str:
        del options
        return cls.render(response)


@dataclass(frozen=True, slots=True)
class McpDevOutputRendererBinding:
    """Typed binding from one output DTO contract to its renderer behavior."""

    output_contract: type
    renderer_type: type[McpDevOutputRenderer]
    render_function: McpDevOutputRenderFunction | None = None

    def render_with_options(
        self,
        response: JsonObject,
        options: McpDevOutputRenderOptions,
    ) -> str:
        if self.render_function is not None:
            return self.render_function(response)
        return self.renderer_type.render_with_options(response, options)


class McpDevPayloadProjection:
    """Small read helpers for dev-client JSON envelopes."""

    @staticmethod
    def tool_result(
        payload: JsonObject,
        tool_name: str,
    ) -> Mapping[str, JsonValue] | None:
        results = payload.get("results")
        if not isinstance(results, list):
            return None
        for result in results:
            if isinstance(result, Mapping) and result.get("tool") == tool_name:
                return result
        return None

    @staticmethod
    def tool_response(
        payload: JsonObject,
        tool_name: str,
    ) -> JsonObject:
        result = McpDevPayloadProjection.tool_result(payload, tool_name)
        if result is None:
            return {
                "server": payload.get("server", {}),
                "errors": payload.get("errors", []),
                "results": [],
            }
        return {
            "server": payload.get("server", {}),
            "errors": payload.get("errors", []),
            "results": [dict(result)],
        }

    @staticmethod
    def first_tool_payload(payload: JsonObject) -> Mapping[str, JsonValue] | None:
        results = payload.get("results")
        if not isinstance(results, list) or not results:
            return None
        first_result = results[0]
        if not isinstance(first_result, Mapping):
            return None
        payloads = first_result.get("payloads")
        if not isinstance(payloads, list) or not payloads:
            return None
        first_payload = payloads[0]
        if not isinstance(first_payload, Mapping):
            return None
        return first_payload

    @staticmethod
    def tool_payload(
        payload: JsonObject,
        tool_name: str,
    ) -> Mapping[str, JsonValue] | None:
        result = McpDevPayloadProjection.tool_result(payload, tool_name)
        if result is None:
            return None
        payloads = result.get("payloads")
        if not isinstance(payloads, list) or not payloads:
            return None
        first_payload = payloads[0]
        if not isinstance(first_payload, Mapping):
            return None
        return first_payload

    @staticmethod
    def nested_mapping(
        payload: Mapping[str, JsonValue],
        key: str,
    ) -> Mapping[str, JsonValue]:
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
        return {}

    @staticmethod
    def sequence_of_mappings(value: JsonValue) -> tuple[Mapping[str, JsonValue], ...]:
        if not isinstance(value, list):
            return ()
        return tuple(item for item in value if isinstance(item, Mapping))

    @staticmethod
    def text(value: JsonValue) -> str:
        if value is None:
            return "<none>"
        return str(value)

    @staticmethod
    def quoted_text(value: JsonValue) -> str:
        if value is None:
            return "<none>"
        return json.dumps(str(value))


class McpDiagnosticRenderer:
    """Compact shared rendering for MCP error and warning payloads."""

    @staticmethod
    def error_lines(errors: tuple[Mapping[str, JsonValue], ...]) -> tuple[str, ...]:
        grouped_codes: dict[str, list[str]] = {}
        grouped_hints: dict[str, list[str]] = {}
        for error in errors:
            message = McpDevPayloadProjection.text(error.get("message"))
            hint = error.get("hint")
            hint_text = None if hint is None else McpDevPayloadProjection.quoted_text(hint)
            code = McpDevPayloadProjection.text(error.get("code"))
            codes = grouped_codes.setdefault(message, [])
            if code not in codes:
                codes.append(code)
            if hint_text is not None and hint_text not in grouped_hints.setdefault(
                message,
                [],
            ):
                grouped_hints[message].append(hint_text)

        lines: list[str] = []
        for message, codes in tuple(grouped_codes.items())[:3]:
            code_text = codes[0] if len(codes) == 1 else ", ".join(codes)
            line = (
                f"- {code_text}: "
                f"{message}"
            )
            hint_texts = grouped_hints.get(message, [])
            if len(hint_texts) == 1:
                line += f" hint={hint_texts[0]}"
            elif len(hint_texts) > 1:
                line += f" hints={len(hint_texts)} distinct; pass --json for details"
            lines.append(line)
        remaining_group_count = len(grouped_codes) - len(lines)
        if remaining_group_count > 0:
            lines.append(f"... {remaining_group_count} more diagnostics")
        return tuple(lines)
