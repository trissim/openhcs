"""Architecture projection DTOs for OpenHCS agent integrations."""

from __future__ import annotations

from dataclasses import dataclass

from openhcs.agent.dto.functions import ImportPathRef


@dataclass(frozen=True, slots=True)
class InternalApiSymbol(ImportPathRef):
    symbol_id: str
    title: str
    role: str
    symbol_kind: str
    signature: str | None
    doc_summary: str | None
    source_path: str | None
    line_number: int | None


@dataclass(frozen=True, slots=True)
class ArchitectureTopicSummary:
    topic_id: str
    title: str
    summary: str


@dataclass(frozen=True, slots=True)
class ArchitectureTopicPage:
    schema_version: str
    topics: tuple[ArchitectureTopicSummary, ...]


@dataclass(frozen=True, slots=True)
class ArchitectureTopic:
    schema_version: str
    topic_id: str
    title: str
    summary: str
    concepts: tuple[str, ...]
    cellprofiler_translation_notes: tuple[str, ...]
    internal_symbols: tuple[InternalApiSymbol, ...]
