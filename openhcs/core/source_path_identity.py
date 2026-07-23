"""Source path identity primitives shared by source-binding authorities."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=8192)
def source_path_identity_key(path: str) -> str:
    """Return the cached lexical identity used for source-binding path matches."""

    return str(Path(path))


def source_paths_equal(left: str, right: str) -> bool:
    """Return whether two source paths identify the same source-binding path."""

    return source_path_identity_key(left) == source_path_identity_key(right)
