"""Visible filesystem aliases for tools that reject hidden source paths."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path


VISIBLE_SOURCE_ROOT_ENV = "OPENHCS_BENCHMARK_VISIBLE_SOURCE_ROOT"
DEFAULT_VISIBLE_SOURCE_ROOT = Path("/tmp") / "openhcs_benchmark_visible_sources"


def resolve_visible_source_path(path: Path) -> Path:
    """
    Return a path without hidden components, preserving the original tree.

    CellProfiler example pipelines commonly include an Images rule excluding
    hidden directories. Acquired benchmark data lives under ``~/.cache`` by
    default, so a semantically valid image tree can be filtered out before any
    module executes. A stable symlink keeps the source tree unchanged while
    presenting it through a visible benchmark path.
    """
    source_path = Path(path).expanduser().resolve()
    if _is_visible_path(source_path):
        return source_path

    alias_root = Path(os.environ.get(VISIBLE_SOURCE_ROOT_ENV, DEFAULT_VISIBLE_SOURCE_ROOT))
    alias_root.mkdir(parents=True, exist_ok=True)
    alias_path = alias_root / f"{source_path.name}_{_path_digest(source_path)}"
    if alias_path.exists() or alias_path.is_symlink():
        if alias_path.resolve() == source_path:
            return alias_path
        if alias_path.is_dir() and not alias_path.is_symlink():
            shutil.rmtree(alias_path)
        else:
            alias_path.unlink()
    alias_path.symlink_to(source_path, target_is_directory=source_path.is_dir())
    return alias_path


def _is_visible_path(path: Path) -> bool:
    return all(part in {path.anchor, ""} or not part.startswith(".") for part in path.parts)


def _path_digest(path: Path) -> str:
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:12]
