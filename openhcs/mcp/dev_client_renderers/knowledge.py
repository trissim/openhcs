"""Knowledge, architecture, and function renderers for the MCP dev client."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import ClassVar

from openhcs.agent import dto as agent_dto
from openhcs.agent.dto.common import JsonObject, JsonValue
from openhcs.mcp.dev_client_rendering import (
    AuthoringContextRenderOptions,
    CatalogRenderOptions,
    McpDevOutputRenderer,
    McpDevPayloadProjection,
)
from openhcs.mcp.dev_client_renderers.object_state import ObjectStateScopeRenderer
from openhcs.mcp.dev_client_renderers.ui_bridge import CodeDocumentRenderer
from openhcs.mcp.dev_client_renderers.viewer import (
    RuntimeServerRenderer,
    ViewerValidationRenderer,
)

class ToolListRenderer:
    """Compact renderer for current MCP tool metadata."""

    @classmethod
    def render(
        cls,
        response: JsonObject,
        *,
        contains: str | None = None,
        limit: int = 80,
    ) -> str:
        errors = McpDevPayloadProjection.sequence_of_mappings(response.get("errors"))
        if errors:
            return "\n".join(
                ("Tools: failed", *RuntimeServerRenderer._error_lines(errors))
            )
        tools = McpDevPayloadProjection.sequence_of_mappings(response.get("tools"))
        if contains:
            needle = contains.casefold()
            tools = tuple(
                tool
                for tool in tools
                if needle in McpDevPayloadProjection.text(tool.get("name")).casefold()
                or needle
                in McpDevPayloadProjection.text(tool.get("description")).casefold()
            )
        bounded_limit = max(limit, 0)
        visible_tools = tools[:bounded_limit]
        lines = [
            (
                "Tools: "
                f"matched={len(tools)} total={McpDevPayloadProjection.text(response.get('tool_count'))} "
                f"shown={len(visible_tools)}"
            )
        ]
        if contains:
            lines.append(f"Filter: contains={contains}")
        if visible_tools:
            lines.append("Tool names:")
            lines.extend(cls._tool_lines(visible_tools))
        if len(visible_tools) < len(tools):
            lines.append(f"...<truncated {len(tools) - len(visible_tools)} tools>")
        return "\n".join(lines)

    @staticmethod
    def _tool_lines(tools: tuple[Mapping[str, JsonValue], ...]) -> list[str]:
        lines: list[str] = []
        for tool in tools:
            lines.append(
                "- "
                f"{McpDevPayloadProjection.text(tool.get('name'))}: "
                f"{McpDevPayloadProjection.text(tool.get('description'))}"
            )
        return lines

class KnowledgeCatalogRenderer(McpDevOutputRenderer):
    """Compact renderer for knowledge-base document catalogs."""

    output_contract = agent_dto.KnowledgeBaseCatalog

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
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)
        documents = McpDevPayloadProjection.sequence_of_mappings(
            payload.get("documents")
        )
        if contains:
            needle = contains.casefold()
            documents = tuple(
                document
                for document in documents
                if needle
                in McpDevPayloadProjection.text(document.get("document_id")).casefold()
                or needle
                in McpDevPayloadProjection.text(document.get("title")).casefold()
                or needle
                in McpDevPayloadProjection.text(document.get("summary")).casefold()
                or any(
                    needle in McpDevPayloadProjection.text(tag).casefold()
                    for tag in cls._tag_values(document)
                )
            )
        bounded_limit = max(limit, 0)
        visible_documents = documents[:bounded_limit]
        lines = [
            (
                "Knowledge documents: "
                f"matched={len(documents)} "
                f"shown={len(visible_documents)}"
            )
        ]
        if contains:
            lines.append(f"Filter: contains={contains}")
        ObjectStateScopeRenderer._append_messages(lines, payload)
        if visible_documents:
            lines.append("Documents:")
            for document in visible_documents:
                lines.append(cls._document_line(document))
        if len(visible_documents) < len(documents):
            lines.append(
                f"...<truncated {len(documents) - len(visible_documents)} documents>"
            )
        return "\n".join(lines)

    @classmethod
    def _document_line(cls, document: Mapping[str, JsonValue]) -> str:
        tag_values = cls._tag_values(document)
        tag_text = ",".join(McpDevPayloadProjection.text(tag) for tag in tag_values[:6])
        if len(tag_values) > 6:
            tag_text += f",+{len(tag_values) - 6}"
        return (
            "- "
            f"{McpDevPayloadProjection.text(document.get('document_id'))}: "
            f"title={McpDevPayloadProjection.quoted_text(document.get('title'))} "
            f"sections={McpDevPayloadProjection.text(document.get('section_count'))} "
            f"path={McpDevPayloadProjection.text(document.get('source_path'))} "
            f"tags={tag_text}"
        )

    @staticmethod
    def _tag_values(document: Mapping[str, JsonValue]) -> tuple[JsonValue, ...]:
        tags = document.get("tags")
        if not isinstance(tags, list):
            return ()
        return tuple(tags)

class KnowledgeSearchRenderer(McpDevOutputRenderer):
    """Compact renderer for knowledge search hits."""

    output_contract = agent_dto.KnowledgeBaseSearchResult

    @classmethod
    def render(
        cls,
        response: JsonObject,
    ) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)
        hits = McpDevPayloadProjection.sequence_of_mappings(payload.get("hits"))
        lines = [
            (
                "Knowledge search: "
                f"query={McpDevPayloadProjection.quoted_text(payload.get('query'))} "
                f"hits={len(hits)}"
            )
        ]
        ObjectStateScopeRenderer._append_messages(lines, payload)
        if hits:
            lines.append("Hits:")
            lines.extend(cls._hit_lines(hits))
        return "\n".join(lines)

    @staticmethod
    def _hit_lines(hits: tuple[Mapping[str, JsonValue], ...]) -> list[str]:
        lines: list[str] = []
        for hit in hits:
            document = McpDevPayloadProjection.nested_mapping(hit, "document")
            section = McpDevPayloadProjection.nested_mapping(hit, "section")
            lines.append(
                "- "
                f"{McpDevPayloadProjection.text(document.get('document_id'))}"
                f"#{McpDevPayloadProjection.text(section.get('section_id'))}: "
                f"score={McpDevPayloadProjection.text(hit.get('score'))} "
                f"line={McpDevPayloadProjection.text(hit.get('line_number'))} "
                f"title={McpDevPayloadProjection.quoted_text(section.get('title'))} "
                f"terms={ViewerValidationRenderer._sequence_text(hit.get('matched_terms'))}"
            )
            snippet = McpDevPayloadProjection.text(hit.get("snippet"))
            if snippet and snippet != "<none>":
                lines.append(f"  {snippet}")
        return lines

class KnowledgeDocumentRenderer(McpDevOutputRenderer):
    """Compact renderer for one knowledge-base document or section."""

    output_contract = agent_dto.KnowledgeBaseDocument

    MAX_SECTION_HINTS: ClassVar[int] = 12

    @classmethod
    def render(
        cls,
        response: JsonObject,
    ) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)
        document = McpDevPayloadProjection.nested_mapping(payload, "document")
        sections = McpDevPayloadProjection.sequence_of_mappings(payload.get("sections"))
        lines = [
            (
                "Knowledge document: "
                f"id={McpDevPayloadProjection.text(document.get('document_id'))} "
                f"title={McpDevPayloadProjection.quoted_text(document.get('title'))} "
                f"path={McpDevPayloadProjection.text(document.get('source_path'))} "
                f"sections={len(sections)} "
                f"max_chars={McpDevPayloadProjection.text(payload.get('max_chars'))}"
            )
        ]
        ObjectStateScopeRenderer._append_messages(lines, payload)
        selected_section_id = McpDevPayloadProjection.text(
            payload.get("selected_section_id")
        )
        if sections and selected_section_id == "<none>":
            lines.extend(cls._section_hint_lines(sections))
        elif selected_section_id != "<none>":
            lines.append(f"Selected section: {selected_section_id}")
        content = payload.get("content")
        if isinstance(content, str):
            lines.append("Content:")
            lines.append(content)
        if payload.get("truncated") is True:
            lines.append(
                "Content truncated; rerun with a larger --max-chars or a narrower "
                "--section-id."
            )
        return "\n".join(lines)

    @classmethod
    def _section_hint_lines(
        cls,
        sections: tuple[Mapping[str, JsonValue], ...],
    ) -> list[str]:
        lines = ["Sections:"]
        visible_sections = sections[: cls.MAX_SECTION_HINTS]
        for section in visible_sections:
            section_id = McpDevPayloadProjection.text(section.get("section_id"))
            title = McpDevPayloadProjection.text(section.get("title"))
            if title and title != "<none>" and title != section_id:
                lines.append(f"- {section_id}: {title}")
            else:
                lines.append(f"- {section_id}")
        omitted_count = len(sections) - len(visible_sections)
        if omitted_count > 0:
            lines.append(f"- ... {omitted_count} more sections")
        return lines

class ArchitectureCatalogRenderer(McpDevOutputRenderer):
    """Compact renderer for architecture topic catalogs."""

    output_contract = agent_dto.ArchitectureTopicPage

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
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)
        topics = McpDevPayloadProjection.sequence_of_mappings(payload.get("topics"))
        if contains:
            needle = contains.casefold()
            topics = tuple(
                topic
                for topic in topics
                if needle
                in McpDevPayloadProjection.text(topic.get("topic_id")).casefold()
                or needle
                in McpDevPayloadProjection.text(topic.get("title")).casefold()
                or needle
                in McpDevPayloadProjection.text(topic.get("summary")).casefold()
            )
        bounded_limit = max(limit, 0)
        visible_topics = topics[:bounded_limit]
        lines = [
            (
                "Architecture topics: "
                f"matched={len(topics)} shown={len(visible_topics)}"
            )
        ]
        if contains:
            lines.append(f"Filter: contains={contains}")
        ObjectStateScopeRenderer._append_messages(lines, payload)
        if visible_topics:
            lines.append("Topics:")
            lines.extend(cls._topic_lines(visible_topics))
        if len(visible_topics) < len(topics):
            lines.append(f"...<truncated {len(topics) - len(visible_topics)} topics>")
        return "\n".join(lines)

    @staticmethod
    def _topic_lines(
        topics: tuple[Mapping[str, JsonValue], ...],
    ) -> list[str]:
        lines: list[str] = []
        for topic in topics:
            lines.append(
                "- "
                f"{McpDevPayloadProjection.text(topic.get('topic_id'))}: "
                f"title={McpDevPayloadProjection.quoted_text(topic.get('title'))} "
                f"summary={McpDevPayloadProjection.quoted_text(topic.get('summary'))}"
            )
        return lines

class ArchitectureTopicRenderer(McpDevOutputRenderer):
    """Compact renderer for one source-backed architecture topic."""

    output_contract = agent_dto.ArchitectureTopic

    @classmethod
    def render(
        cls,
        response: JsonObject,
    ) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)
        concepts = cls._text_sequence(payload.get("concepts"))
        notes = cls._text_sequence(payload.get("cellprofiler_translation_notes"))
        symbols = McpDevPayloadProjection.sequence_of_mappings(
            payload.get("internal_symbols")
        )
        lines = [
            (
                "Architecture topic: "
                f"id={McpDevPayloadProjection.text(payload.get('topic_id'))} "
                f"title={McpDevPayloadProjection.quoted_text(payload.get('title'))} "
                f"concepts={len(concepts)} symbols={len(symbols)}"
            )
        ]
        summary = McpDevPayloadProjection.text(payload.get("summary"))
        if summary != "<none>":
            lines.append(f"Summary: {summary}")
        ObjectStateScopeRenderer._append_messages(lines, payload)
        if concepts:
            lines.append("Concepts:")
            lines.extend(f"- {concept}" for concept in concepts)
        if notes:
            lines.append("CellProfiler notes:")
            lines.extend(f"- {note}" for note in notes)
        if symbols:
            lines.append("Internal symbols:")
            lines.extend(cls._symbol_lines(symbols))
        return "\n".join(lines)

    @staticmethod
    def _symbol_lines(
        symbols: tuple[Mapping[str, JsonValue], ...],
    ) -> list[str]:
        lines: list[str] = []
        for symbol in symbols:
            source = McpDevPayloadProjection.text(symbol.get("source_path"))
            line_number = symbol.get("line_number")
            if line_number is not None:
                source = f"{source}:{McpDevPayloadProjection.text(line_number)}"
            lines.append(
                "- "
                f"{McpDevPayloadProjection.text(symbol.get('symbol_id'))}: "
                f"{McpDevPayloadProjection.text(symbol.get('title'))} "
                f"kind={McpDevPayloadProjection.text(symbol.get('symbol_kind'))} "
                f"import={McpDevPayloadProjection.text(symbol.get('import_path'))} "
                f"source={source}"
            )
            role = McpDevPayloadProjection.text(symbol.get("role"))
            if role != "<none>":
                lines.append(f"  role={role}")
        return lines

    @staticmethod
    def _text_sequence(value: JsonValue) -> tuple[str, ...]:
        if not isinstance(value, list):
            return ()
        return tuple(McpDevPayloadProjection.text(item) for item in value)

class InternalSymbolRenderer(McpDevOutputRenderer):
    """Compact renderer for one projected internal architecture symbol."""

    output_contract = agent_dto.InternalApiSymbol

    @classmethod
    def render(cls, response: JsonObject) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)
        source = McpDevPayloadProjection.text(payload.get("source_path"))
        line_number = payload.get("line_number")
        if line_number is not None:
            source = f"{source}:{McpDevPayloadProjection.text(line_number)}"
        lines = [
            (
                "Internal symbol: "
                f"id={McpDevPayloadProjection.text(payload.get('symbol_id'))} "
                f"title={McpDevPayloadProjection.quoted_text(payload.get('title'))} "
                f"kind={McpDevPayloadProjection.text(payload.get('symbol_kind'))}"
            ),
            f"Import: {McpDevPayloadProjection.text(payload.get('import_path'))}",
            f"Source: {source}",
        ]
        signature = payload.get("signature")
        if isinstance(signature, str) and signature:
            lines.append(f"Signature: {signature}")
        role = McpDevPayloadProjection.text(payload.get("role"))
        if role != "<none>":
            lines.append(f"Role: {role}")
        doc_summary = McpDevPayloadProjection.text(payload.get("doc_summary"))
        if doc_summary != "<none>":
            lines.append(f"Doc: {doc_summary}")
        return "\n".join(lines)

class FunctionSearchRenderer(McpDevOutputRenderer):
    """Compact renderer for processing-function search results."""

    output_contract = agent_dto.FunctionCatalogPage

    @classmethod
    def render(cls, response: JsonObject) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)
        items = McpDevPayloadProjection.sequence_of_mappings(payload.get("items"))
        lines = [
            (
                "Function search: "
                f"query={McpDevPayloadProjection.quoted_text(payload.get('query'))} "
                f"library={McpDevPayloadProjection.text(payload.get('library'))} "
                f"shown={len(items)} total={McpDevPayloadProjection.text(payload.get('total'))}"
            )
        ]
        ObjectStateScopeRenderer._append_messages(lines, payload)
        if items:
            lines.append("Functions:")
            lines.extend(cls._item_lines(items))
        return "\n".join(lines)

    @staticmethod
    def _item_lines(items: tuple[Mapping[str, JsonValue], ...]) -> list[str]:
        lines: list[str] = []
        for item in items:
            lines.append(
                "- "
                f"{McpDevPayloadProjection.text(item.get('function_id'))}: "
                f"{McpDevPayloadProjection.text(item.get('signature'))} "
                f"tags={ViewerValidationRenderer._sequence_text(item.get('backend_tags'))}"
            )
            summary = McpDevPayloadProjection.text(item.get("summary"))
            if summary and summary != "<none>":
                lines.append(f"  {summary}")
        return lines

class CustomFunctionRegistrationRenderer(McpDevOutputRenderer):
    """Compact renderer for custom-function registration results."""

    output_contract = agent_dto.CustomFunctionRegistrationResult

    @classmethod
    def render(cls, response: JsonObject) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)
        errors = McpDevPayloadProjection.sequence_of_mappings(payload.get("errors"))
        if errors:
            return "\n".join(
                (
                    "Custom function registration: failed",
                    *ViewerValidationRenderer._error_lines(errors),
                )
            )
        functions = McpDevPayloadProjection.sequence_of_mappings(
            payload.get("functions")
        )
        lines = [
            (
                "Custom function registration: "
                f"registered={McpDevPayloadProjection.text(payload.get('registered_count'))} "
                f"persisted={McpDevPayloadProjection.text(payload.get('persisted'))} "
                f"storage={McpDevPayloadProjection.text(payload.get('storage_dir'))}"
            )
        ]
        source_paths = payload.get("source_file_paths")
        if source_paths:
            lines.append(
                f"Files: {ViewerValidationRenderer._sequence_text(source_paths)}"
            )
        if functions:
            lines.append("Functions:")
            lines.extend(FunctionSearchRenderer._item_lines(functions))
            if payload.get("persisted") is False:
                lines.append(
                    "Lifetime: process-local only; follow-up dev_client commands "
                    "start a fresh MCP process. Omit --no-persist or reuse the "
                    "same MCP session before using these function ids."
                )
            lines.append("Next:")
            for function in functions:
                function_id = McpDevPayloadProjection.text(function.get("function_id"))
                lines.append(f"- function {function_id}")
                lines.append(f"- draft-pipeline-step {function_id} --name <step_name>")
        return "\n".join(lines)

class FunctionDetailRenderer(McpDevOutputRenderer):
    """Compact renderer for one processing-function detail payload."""

    output_contract = agent_dto.FunctionDetail

    @classmethod
    def render(cls, response: JsonObject) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)
        entry = McpDevPayloadProjection.nested_mapping(payload, "entry")
        runtime_contract = McpDevPayloadProjection.nested_mapping(
            payload,
            "runtime_contract",
        )
        parameters = McpDevPayloadProjection.sequence_of_mappings(
            payload.get("parameters")
        )
        agent_parameters = tuple(
            parameter
            for parameter in parameters
            if cls._parameter_supplier(parameter) == "agent"
        )
        runtime_parameters = tuple(
            parameter
            for parameter in parameters
            if cls._parameter_supplier(parameter) != "agent"
        )
        artifact_outputs = McpDevPayloadProjection.sequence_of_mappings(
            runtime_contract.get("artifact_outputs")
        )
        lines = [
            (
                "Function: "
                f"id={McpDevPayloadProjection.text(entry.get('function_id'))} "
                f"name={McpDevPayloadProjection.text(entry.get('name'))} "
                f"library={McpDevPayloadProjection.text(entry.get('library'))}"
            ),
            f"Signature: {McpDevPayloadProjection.text(entry.get('signature'))}",
        ]
        summary = McpDevPayloadProjection.text(entry.get("summary"))
        if summary and summary != "<none>":
            lines.append(f"Summary: {summary}")
        if agent_parameters:
            lines.append("Agent parameters:")
            lines.extend(cls._parameter_lines(agent_parameters))
        if runtime_parameters:
            lines.append("Runtime inputs:")
            lines.extend(cls._runtime_parameter_lines(runtime_parameters))
        if artifact_outputs:
            lines.append("Artifact outputs:")
            lines.extend(cls._artifact_lines(artifact_outputs))
        doc = payload.get("doc")
        if isinstance(doc, str) and doc:
            doc_chars = payload.get("doc_chars")
            lines.append(
                "Doc: "
                f"chars={McpDevPayloadProjection.text(doc_chars)} "
                f"truncated={McpDevPayloadProjection.text(payload.get('doc_truncated'))}"
            )
            lines.append(doc)
            if payload.get("doc_truncated") is True:
                lines.append(
                    "Doc truncated; rerun: "
                    f"function {McpDevPayloadProjection.text(entry.get('function_id'))} "
                    f"--max-doc-chars {McpDevPayloadProjection.text(doc_chars)}"
                )
        ObjectStateScopeRenderer._append_messages(lines, payload)
        return "\n".join(lines)

    @staticmethod
    def _parameter_lines(parameters: tuple[Mapping[str, JsonValue], ...]) -> list[str]:
        lines: list[str] = []
        for parameter in parameters:
            default = McpDevPayloadProjection.text(parameter.get("default_repr"))
            required = McpDevPayloadProjection.text(parameter.get("required"))
            annotation = McpDevPayloadProjection.text(parameter.get("annotation"))
            lines.append(
                "- "
                f"{McpDevPayloadProjection.text(parameter.get('name'))}: "
                f"required={required} type={annotation} default={default}"
            )
        return lines

    @classmethod
    def _runtime_parameter_lines(
        cls,
        parameters: tuple[Mapping[str, JsonValue], ...],
    ) -> list[str]:
        lines: list[str] = []
        for parameter in parameters:
            annotation = McpDevPayloadProjection.text(parameter.get("annotation"))
            description = McpDevPayloadProjection.text(parameter.get("description"))
            lines.append(
                "- "
                f"{McpDevPayloadProjection.text(parameter.get('name'))}: "
                f"supplied_by={cls._parameter_supplier(parameter)} "
                f"type={annotation} note={McpDevPayloadProjection.quoted_text(description)}"
            )
        return lines

    @staticmethod
    def _parameter_supplier(parameter: Mapping[str, JsonValue]) -> str:
        supplied_by = parameter.get("supplied_by")
        if isinstance(supplied_by, str) and supplied_by:
            return supplied_by
        return "agent"

    @staticmethod
    def _artifact_lines(artifacts: tuple[Mapping[str, JsonValue], ...]) -> list[str]:
        return [
            "- "
            f"{McpDevPayloadProjection.text(artifact.get('name'))}: "
            f"kind={McpDevPayloadProjection.text(artifact.get('kind'))} "
            f"required={McpDevPayloadProjection.text(artifact.get('required'))}"
            for artifact in artifacts
        ]

class AuthoringContextRenderer(McpDevOutputRenderer):
    """Compact renderer for authoring guidance."""

    output_contract = agent_dto.AuthoringContext

    @classmethod
    def render_with_options(
        cls,
        response: JsonObject,
        options: AuthoringContextRenderOptions,
    ) -> str:
        return cls.render(response, max_chars=options.max_chars)

    @classmethod
    def render(cls, response: JsonObject, *, max_chars: int = 2_000) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)
        content = payload.get("content")
        lines = [
            (
                "Authoring context: "
                f"kind={McpDevPayloadProjection.text(payload.get('kind'))}"
            )
        ]
        ObjectStateScopeRenderer._append_messages(lines, payload)
        if isinstance(content, str):
            lines.append("Content:")
            lines.append(CodeDocumentRenderer._source_text(content, max_source_chars=max_chars))
        return "\n".join(lines)
