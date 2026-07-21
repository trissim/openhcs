"""Project canonical MCP knowledge documents into an installable package tree."""

from __future__ import annotations

import argparse
import json
import runpy
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path


_MANIFEST_SCHEMA = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "openhcs/agent/knowledge_manifest_schema.py"
    )
)
KnowledgeBaseManifestField = _MANIFEST_SCHEMA["KnowledgeBaseManifestField"]
KNOWLEDGE_MANIFEST_RELATIVE_PATH = _MANIFEST_SCHEMA[
    "DEFAULT_KNOWLEDGE_BASE_MANIFEST_PATH"
]
PACKAGED_KNOWLEDGE_ROOT_RELATIVE_PATH = (
    Path("openhcs/agent") / _MANIFEST_SCHEMA["PACKAGED_KNOWLEDGE_BASE_ROOT"]
)


def declared_knowledge_source_paths(project_root: Path) -> tuple[Path, ...]:
    """Return the canonical manifest and every uniquely declared source path."""
    root = project_root.resolve()
    manifest_path = root / KNOWLEDGE_MANIFEST_RELATIVE_PATH
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("MCP knowledge manifest root must be an object.")
    documents = manifest.get(KnowledgeBaseManifestField.DOCUMENTS.value)
    if not isinstance(documents, list) or not documents:
        raise ValueError("MCP knowledge manifest must declare documents.")

    relative_paths = [KNOWLEDGE_MANIFEST_RELATIVE_PATH]
    seen = {KNOWLEDGE_MANIFEST_RELATIVE_PATH}
    for document in documents:
        if not isinstance(document, Mapping):
            raise ValueError("MCP knowledge manifest documents must be objects.")
        raw_source_path = document.get(KnowledgeBaseManifestField.SOURCE_PATH.value)
        if not isinstance(raw_source_path, str) or not raw_source_path:
            raise ValueError("MCP knowledge document source_path must be a string.")
        relative_path = Path(raw_source_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"MCP knowledge source path must stay within the project: {raw_source_path}"
            )
        if relative_path in seen:
            raise ValueError(
                f"MCP knowledge source path is declared more than once: {raw_source_path}"
            )
        seen.add(relative_path)
        relative_paths.append(relative_path)

    source_paths = tuple(root / relative_path for relative_path in relative_paths)
    missing = tuple(path for path in source_paths if not path.is_file())
    if missing:
        formatted = ", ".join(path.relative_to(root).as_posix() for path in missing)
        raise FileNotFoundError(f"MCP knowledge sources are missing: {formatted}")
    return source_paths


def project_knowledge_assets(
    project_root: Path, destination_root: Path
) -> tuple[Path, ...]:
    """Copy the manifest-declared canonical sources into ``destination_root``."""
    root = project_root.resolve()
    destination = destination_root.resolve()
    checked_in_projection = (root / PACKAGED_KNOWLEDGE_ROOT_RELATIVE_PATH).resolve()
    if destination == checked_in_projection:
        raise ValueError(
            "MCP knowledge assets belong in build output, not the source package tree."
        )
    if destination == root or root.is_relative_to(destination):
        raise ValueError(
            f"MCP knowledge destination must not own the project root: {destination}"
        )
    source_paths = declared_knowledge_source_paths(root)
    if destination.exists():
        shutil.rmtree(destination)
    projected_paths: list[Path] = []
    for source_path in source_paths:
        relative_path = source_path.relative_to(root)
        destination_path = destination / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination_path)
        projected_paths.append(destination_path)
    return tuple(projected_paths)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--destination-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    project_root = args.project_root.resolve()
    destination_root = args.destination_root
    projected = project_knowledge_assets(project_root, destination_root)
    print(
        f"Projected {len(projected)} MCP knowledge assets into "
        f"{destination_root.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
