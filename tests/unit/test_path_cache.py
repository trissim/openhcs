"""Regression tests for persisted UI path selection."""

from __future__ import annotations

import json

from openhcs.core.path_cache import PathCacheKey, UnifiedPathCache


def test_stale_pipeline_path_is_removed_and_fallback_is_used(tmp_path) -> None:
    """A moved pipeline cannot strand the next file dialog on a dead path."""

    cache_file = tmp_path / "path_cache.json"
    moved_path = tmp_path / "moved-away"
    fallback = tmp_path / "pipelines"
    fallback.mkdir()
    cache_file.write_text(
        json.dumps({PathCacheKey.PIPELINE_FILES.value: str(moved_path)}),
        encoding="utf-8",
    )

    cache = UnifiedPathCache(cache_file)

    assert cache.get_initial_path(PathCacheKey.PIPELINE_FILES, fallback) == fallback
    assert json.loads(cache_file.read_text(encoding="utf-8")) == {}
