"""Shared normalization for CellProfiler setting labels."""

from __future__ import annotations

from functools import lru_cache
import re


@lru_cache(maxsize=4096)
def normalize_cellprofiler_setting_name(name: str) -> str:
    """Normalize a CellProfiler setting label into a snake_case key."""
    without_parentheses = re.sub(r"\([^)]*\)", "", name)
    without_questions = without_parentheses.replace("?", "")
    words = re.sub(r"[^\w\s]", " ", without_questions).lower().split()
    return "_".join(words)
