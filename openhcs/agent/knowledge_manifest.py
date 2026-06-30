"""Knowledge-base manifest declarations shared by services and MCP."""

from __future__ import annotations

import json
from collections.abc import Mapping
from enum import Enum
from pathlib import Path


DEFAULT_KNOWLEDGE_BASE_MANIFEST_PATH = Path(
    "docs/source/development/mcp_knowledge_base_manifest.json"
)


class KnowledgeBaseManifestField(str, Enum):
    """JSON field names for the source-backed knowledge-base manifest."""

    DOCUMENTS = "documents"
    DOCUMENT_ID = "document_id"
    TITLE = "title"
    SUMMARY = "summary"
    SOURCE_PATH = "source_path"
    TAGS = "tags"
    SECTION_COUNT = "section_count"


def default_repo_root() -> Path:
    """Return the repository root containing the default knowledge manifest."""
    return Path(__file__).resolve().parents[2]


def default_knowledge_base_manifest_path() -> Path:
    """Return the default knowledge-base manifest source path."""
    return default_repo_root() / DEFAULT_KNOWLEDGE_BASE_MANIFEST_PATH


def knowledge_base_source_paths_from_manifest(
    manifest_path: Path | None = None,
) -> tuple[Path, ...]:
    """Return manifest and declared document paths without importing services."""
    selected_manifest_path = manifest_path or default_knowledge_base_manifest_path()
    repo_root = default_repo_root()
    try:
        manifest = json.loads(selected_manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return (selected_manifest_path,)
    if not isinstance(manifest, Mapping):
        return (selected_manifest_path,)
    documents = manifest.get(KnowledgeBaseManifestField.DOCUMENTS.value)
    if not isinstance(documents, list):
        return (selected_manifest_path,)
    source_paths: list[Path] = [selected_manifest_path]
    for document in documents:
        if not isinstance(document, Mapping):
            continue
        source_path = document.get(KnowledgeBaseManifestField.SOURCE_PATH.value)
        if isinstance(source_path, str):
            source_paths.append((repo_root / source_path).resolve())
    return tuple(source_paths)
