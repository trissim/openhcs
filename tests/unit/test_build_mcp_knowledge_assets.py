"""Tests for build-only MCP knowledge projection."""

import json

import pytest

from scripts.build_mcp_knowledge_assets import (
    KNOWLEDGE_MANIFEST_RELATIVE_PATH,
    PACKAGED_KNOWLEDGE_ROOT_RELATIVE_PATH,
    project_knowledge_assets,
)


def _project_with_document(tmp_path):
    project_root = tmp_path / "project"
    manifest_path = project_root / KNOWLEDGE_MANIFEST_RELATIVE_PATH
    document_path = project_root / "docs" / "guide.rst"
    manifest_path.parent.mkdir(parents=True)
    document_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "document_id": "guide",
                        "title": "Guide",
                        "summary": "Guide summary",
                        "source_path": "docs/guide.rst",
                        "tags": ["guide"],
                        "section_count": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    document_path.write_text("Guide\n=====\n", encoding="utf-8")
    return project_root, manifest_path, document_path


def test_projection_copies_only_manifest_declared_sources(tmp_path):
    project_root, manifest_path, document_path = _project_with_document(tmp_path)
    destination = tmp_path / "wheel" / "knowledge"

    projected = project_knowledge_assets(project_root, destination)

    assert projected == (
        destination / manifest_path.relative_to(project_root),
        destination / document_path.relative_to(project_root),
    )
    assert (destination / "docs" / "guide.rst").read_text(encoding="utf-8") == (
        "Guide\n=====\n"
    )


def test_projection_rejects_checked_in_mirror(tmp_path):
    project_root, _, _ = _project_with_document(tmp_path)

    with pytest.raises(ValueError, match="build output"):
        project_knowledge_assets(
            project_root,
            project_root / PACKAGED_KNOWLEDGE_ROOT_RELATIVE_PATH,
        )


def test_projection_rejects_project_ancestor(tmp_path):
    project_root, _, _ = _project_with_document(tmp_path)

    with pytest.raises(ValueError, match="must not own"):
        project_knowledge_assets(project_root, tmp_path)
