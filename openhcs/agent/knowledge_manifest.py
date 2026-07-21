"""Knowledge-base manifest declarations shared by services and MCP."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from openhcs.agent.knowledge_manifest_schema import (
    DEFAULT_KNOWLEDGE_BASE_MANIFEST_PATH,
    PACKAGED_KNOWLEDGE_BASE_ROOT,
    KnowledgeBaseManifestField,
)


def source_checkout_root() -> Path:
    """Return the repository root implied by this module's source location."""
    return Path(__file__).resolve().parents[2]


def packaged_knowledge_base_root() -> Path:
    """Return the package-resource root populated by the release build."""
    return Path(__file__).resolve().parent / PACKAGED_KNOWLEDGE_BASE_ROOT


def default_repo_root() -> Path:
    """Return the available root containing the canonical knowledge projection."""
    source_root = source_checkout_root()
    if (source_root / DEFAULT_KNOWLEDGE_BASE_MANIFEST_PATH).is_file():
        return source_root
    packaged_root = packaged_knowledge_base_root()
    if (packaged_root / DEFAULT_KNOWLEDGE_BASE_MANIFEST_PATH).is_file():
        return packaged_root
    return source_root


def default_knowledge_base_manifest_path() -> Path:
    """Return the default knowledge-base manifest source path."""
    return default_repo_root() / DEFAULT_KNOWLEDGE_BASE_MANIFEST_PATH


def knowledge_base_source_paths_from_manifest(
    manifest_path: Path | None = None,
) -> tuple[Path, ...]:
    """Return manifest and declared document paths without importing services."""
    selected_manifest_path = manifest_path or default_knowledge_base_manifest_path()
    repo_root = _knowledge_base_root_for_manifest(selected_manifest_path)
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


def _knowledge_base_root_for_manifest(manifest_path: Path) -> Path:
    """Recover the root that owns a canonical-relative manifest path."""
    resolved_manifest_path = manifest_path.resolve()
    relative_parts = DEFAULT_KNOWLEDGE_BASE_MANIFEST_PATH.parts
    if resolved_manifest_path.parts[-len(relative_parts) :] == relative_parts:
        root = resolved_manifest_path
        for _ in relative_parts:
            root = root.parent
        return root
    return default_repo_root()
