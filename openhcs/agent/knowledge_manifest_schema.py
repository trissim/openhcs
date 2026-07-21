"""Dependency-free schema authority for the OpenHCS knowledge manifest."""

from __future__ import annotations

from enum import Enum
from pathlib import Path


DEFAULT_KNOWLEDGE_BASE_MANIFEST_PATH = Path(
    "docs/source/development/mcp_knowledge_base_manifest.json"
)
PACKAGED_KNOWLEDGE_BASE_ROOT = Path("resources/knowledge")


class KnowledgeBaseManifestField(str, Enum):
    """JSON field names for the source-backed knowledge-base manifest."""

    DOCUMENTS = "documents"
    DOCUMENT_ID = "document_id"
    TITLE = "title"
    SUMMARY = "summary"
    SOURCE_PATH = "source_path"
    TAGS = "tags"
    SECTION_COUNT = "section_count"
