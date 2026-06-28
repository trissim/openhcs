"""Visible filesystem aliases for tools that reject hidden source paths."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from polystore.filemanager import FileManager

from openhcs.constants.constants import Backend


VISIBLE_SOURCE_ROOT_ENV = "OPENHCS_BENCHMARK_VISIBLE_SOURCE_ROOT"
DEFAULT_VISIBLE_SOURCE_ROOT = Path("/tmp") / "openhcs_benchmark_visible_sources"
VISIBLE_SOURCE_TARGET_MARKER = ".openhcs_visible_source_target.txt"


def resolve_visible_source_path(path: Path) -> Path:
    """
    Return a path without hidden components, preserving the original tree.

    CellProfiler example pipelines commonly include an Images rule excluding
    hidden directories. Acquired benchmark data lives under ``~/.cache`` by
    default, so a semantically valid image tree can be filtered out before any
    module executes. A stable directory of file symlinks keeps the cached image
    bytes unchanged while leaving the visible alias writable for OpenHCS
    metadata sidecars.
    """
    source_path = Path(path).expanduser().resolve()
    if _is_visible_path(source_path):
        return source_path

    filemanager = _visible_source_filemanager()
    backend = Backend.DISK.value
    alias_source_path = _alias_source_path(source_path)
    alias_root = Path(os.environ.get(VISIBLE_SOURCE_ROOT_ENV, DEFAULT_VISIBLE_SOURCE_ROOT))
    filemanager.ensure_directory(alias_root, backend)
    alias_path = alias_root / f"{alias_source_path.name}_{_path_digest(alias_source_path)}"
    if filemanager.exists(alias_path, backend) or filemanager.is_symlink(alias_path, backend):
        alias_is_symlink = filemanager.is_symlink(alias_path, backend)
        if (
            filemanager.is_dir(alias_path, backend)
            and not alias_is_symlink
            and _alias_matches_source(filemanager, alias_path, alias_source_path, backend)
            and not filemanager.is_symlink(
                alias_path / _openhcs_metadata_filename(),
                backend,
            )
        ):
            return alias_path / source_path.relative_to(alias_source_path)
        if alias_is_symlink:
            filemanager.delete(alias_path, backend)
        else:
            filemanager.delete_all(alias_path, backend)
    _create_visible_alias_tree(filemanager, alias_source_path, alias_path, backend)
    return alias_path / source_path.relative_to(alias_source_path)


def _is_visible_path(path: Path) -> bool:
    return all(part in {path.anchor, ""} or not part.startswith(".") for part in path.parts)


def _path_digest(path: Path) -> str:
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:12]


def _alias_source_path(source_path: Path) -> Path:
    if source_path.is_dir() and any(source_path.parent.glob("*.cppipe")):
        return source_path.parent
    return source_path


def _visible_source_filemanager() -> FileManager:
    from polystore.base import ensure_storage_registry, storage_registry

    ensure_storage_registry()
    return FileManager(storage_registry)


def _openhcs_metadata_filename() -> str:
    from openhcs.microscopes.openhcs import METADATA_CONFIG

    return METADATA_CONFIG.METADATA_FILENAME


def _alias_matches_source(
    filemanager: FileManager,
    alias_path: Path,
    source_path: Path,
    backend: str,
) -> bool:
    marker_path = alias_path / VISIBLE_SOURCE_TARGET_MARKER
    if not filemanager.is_file(marker_path, backend):
        return False
    return filemanager.load(marker_path, backend) == str(source_path)


def _create_visible_alias_tree(
    filemanager: FileManager,
    source_path: Path,
    alias_path: Path,
    backend: str,
) -> None:
    filemanager.ensure_directory(alias_path, backend)
    filemanager.save(
        str(source_path),
        alias_path / VISIBLE_SOURCE_TARGET_MARKER,
        backend,
    )
    if source_path.is_file():
        if source_path.name == _openhcs_metadata_filename():
            return
        filemanager.create_symlink(
            source_path,
            alias_path / source_path.name,
            backend,
        )
        return

    _, source_files = filemanager.collect_dirs_and_files(
        source_path,
        backend,
        recursive=True,
    )
    for source_file in source_files:
        source_file_path = Path(source_file)
        if source_file_path.name == _openhcs_metadata_filename():
            continue
        symlink_path = alias_path / source_file_path.relative_to(source_path)
        filemanager.ensure_directory(symlink_path.parent, backend)
        filemanager.create_symlink(
            source_file_path,
            symlink_path,
            backend,
            overwrite_symlinks_only=True,
        )
