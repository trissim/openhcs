"""Knowledge-base DTOs for OpenHCS agent integrations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Self

from openhcs.agent.dto.common import (
    AgentCliArgumentSpec,
    AgentCliRequest,
    AgentResultEnvelope,
    JsonObject,
)


@dataclass(frozen=True, slots=True)
class KnowledgeBaseDocumentSourceProjection:
    source_path: str
    section_count: int


@dataclass(frozen=True, slots=True)
class KnowledgeBaseDocumentSummary:
    document_id: str
    title: str
    summary: str
    source_path: str
    tags: tuple[str, ...]
    section_count: int

    def with_source_projection(
        self,
        projection: KnowledgeBaseDocumentSourceProjection,
    ) -> Self:
        return replace(
            self,
            source_path=projection.source_path,
            section_count=projection.section_count,
        )

    def matches_query(self, normalized_query: str) -> bool:
        return (
            normalized_query in self.document_id.casefold()
            or normalized_query in self.title.casefold()
            or normalized_query in self.summary.casefold()
            or any(normalized_query in tag.casefold() for tag in self.tags)
        )


@dataclass(frozen=True, slots=True)
class KnowledgeBaseSourceSpan:
    start_line: int
    end_line: int

    def line_slice(self, lines: tuple[str, ...]) -> tuple[str, ...]:
        return lines[self.start_line - 1 : self.end_line]

    def close_before(self, next_start_line: int) -> Self:
        return replace(
            self,
            end_line=max(self.start_line, next_start_line - 1),
        )


@dataclass(frozen=True, slots=True)
class KnowledgeBaseSectionSummary:
    section_id: str
    title: str
    level: int
    span: KnowledgeBaseSourceSpan

    @property
    def start_line(self) -> int:
        return self.span.start_line

    @property
    def end_line(self) -> int:
        return self.span.end_line

    def with_span(self, span: KnowledgeBaseSourceSpan) -> Self:
        return replace(
            self,
            span=span,
        )


@dataclass(frozen=True, slots=True)
class KnowledgeBaseDocumentTarget:
    document_id: str
    section_id: str | None = None

    def find_section(
        self,
        sections: tuple[KnowledgeBaseSectionSummary, ...],
    ) -> KnowledgeBaseSectionSummary | None:
        if self.section_id is None:
            return None
        for section in sections:
            if section.section_id == self.section_id:
                return section
        return None


@dataclass(frozen=True, slots=True)
class KnowledgeBaseContentBounds:
    max_chars: int = 12_000

    def effective_max_chars(self, maximum: int) -> int:
        return max(1, min(self.max_chars, maximum))

    def apply(self, content: str, maximum: int) -> tuple[str, bool, int]:
        effective_max_chars = self.effective_max_chars(maximum)
        if len(content) <= effective_max_chars:
            return content, False, effective_max_chars
        return content[:effective_max_chars], True, effective_max_chars


@dataclass(frozen=True, slots=True)
class KnowledgeBaseDocumentRequest(AgentCliRequest):
    target: KnowledgeBaseDocumentTarget
    bounds: KnowledgeBaseContentBounds = KnowledgeBaseContentBounds()

    @classmethod
    def agent_cli_factory(cls):
        return cls.from_cli_fields

    @classmethod
    def agent_cli_argument_specs(cls) -> tuple[AgentCliArgumentSpec, ...]:
        return (
            AgentCliArgumentSpec(
                field_name="document_id",
                positional=True,
                help="Document id from `knowledge` / openhcs_list_knowledge_documents.",
            ),
            AgentCliArgumentSpec(
                field_name="section_id",
                flags=("--section-id",),
                help="Optional section id from the document's section list.",
            ),
            AgentCliArgumentSpec(
                field_name="max_chars",
                flags=("--max-chars",),
                help="Maximum content characters returned from the selected document or section.",
            ),
        )

    @classmethod
    def from_cli_fields(
        cls,
        *,
        document_id: str,
        section_id: str | None = None,
        max_chars: int = 12_000,
    ) -> Self:
        resolved_document_id = document_id
        resolved_section_id = section_id
        if resolved_section_id is None and "#" in resolved_document_id:
            resolved_document_id, resolved_section_id = resolved_document_id.split("#", 1)
        return cls.from_fields(
            document_id=resolved_document_id,
            section_id=resolved_section_id,
            max_chars=max_chars,
        )

    @classmethod
    def from_fields(
        cls,
        *,
        document_id: str,
        section_id: str | None = None,
        max_chars: int = 12_000,
    ) -> Self:
        return cls(
            target=KnowledgeBaseDocumentTarget(
                document_id=document_id,
                section_id=section_id,
            ),
            bounds=KnowledgeBaseContentBounds(max_chars=max_chars),
        )

    def as_tool_arguments(self) -> JsonObject:
        return {
            "document_id": self.target.document_id,
            "section_id": self.target.section_id,
            "max_chars": self.bounds.max_chars,
        }


@dataclass(frozen=True, slots=True)
class KnowledgeBaseSearchRequest(AgentCliRequest):
    query: str
    limit: int = 10

    @classmethod
    def agent_cli_factory(cls):
        return cls.from_cli_fields

    @classmethod
    def agent_cli_argument_specs(cls) -> tuple[AgentCliArgumentSpec, ...]:
        return (
            AgentCliArgumentSpec(
                field_name="query",
                positional=True,
                nargs="*",
                help="Case-insensitive text searched across allowlisted knowledge docs.",
            ),
            AgentCliArgumentSpec(
                field_name="query_option",
                flags=("--query",),
                action="append",
                help="Case-insensitive text searched across allowlisted knowledge docs.",
            ),
        )

    @classmethod
    def from_cli_fields(
        cls,
        *,
        query: Sequence[str] = (),
        query_option: Sequence[str] | None = None,
        limit: int = 10,
    ) -> Self:
        query_parts = (*query, *(query_option or ()))
        if not query_parts:
            raise ValueError("knowledge-search requires a query.")
        return cls.from_fields(
            query=" ".join(query_parts),
            limit=limit,
        )

    @classmethod
    def from_fields(
        cls,
        *,
        query: str,
        limit: int = 10,
    ) -> Self:
        return cls(
            query=query,
            limit=limit,
        )

    def as_tool_arguments(self) -> JsonObject:
        return {
            "query": self.query,
            "limit": self.limit,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class KnowledgeBaseCatalog(AgentResultEnvelope):
    documents: tuple[KnowledgeBaseDocumentSummary, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class KnowledgeBaseDocument(AgentResultEnvelope):
    document: KnowledgeBaseDocumentSummary | None = None
    sections: tuple[KnowledgeBaseSectionSummary, ...] = ()
    content: str = ""
    selected_section_id: str | None = None
    truncated: bool = False
    max_chars: int = 12_000


@dataclass(frozen=True, slots=True)
class KnowledgeBaseSearchHit:
    document: KnowledgeBaseDocumentSummary
    section: KnowledgeBaseSectionSummary | None
    line_number: int | None
    snippet: str
    score: int = 0
    matched_terms: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class KnowledgeBaseSearchResult(AgentResultEnvelope):
    query: str
    hits: tuple[KnowledgeBaseSearchHit, ...] = ()
