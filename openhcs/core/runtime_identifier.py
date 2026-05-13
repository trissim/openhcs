"""Canonical runtime identifier normalization."""

from __future__ import annotations

import re
from functools import lru_cache

_IDENTIFIER_INITIALISM_BOUNDARY_RE = re.compile(r"([A-Z]+)([A-Z][a-z])")
_IDENTIFIER_LOWER_UPPER_BOUNDARY_RE = re.compile(r"([a-z0-9])([A-Z])")
_IDENTIFIER_ALPHA_NUMBER_BOUNDARY_RE = re.compile(r"([A-Za-z])([0-9])")
_IDENTIFIER_NUMBER_ALPHA_BOUNDARY_RE = re.compile(r"([0-9])([A-Za-z])")
_IDENTIFIER_NON_ALNUM_RE = re.compile(r"[^A-Za-z0-9]+")


def normalize_runtime_identifier(value: object) -> str:
    """Return OpenHCS' canonical identifier token for runtime comparison."""
    return _normalize_identifier_text(str(value).strip())


def normalize_runtime_source_name(source_name: str | None) -> str | None:
    """Return canonical source-image identity for runtime comparison."""
    if source_name is None:
        return None
    normalized = "__".join(
        part
        for part in (
            normalize_runtime_identifier(part)
            for part in str(source_name).split("__")
        )
        if part
    )
    return normalized or None


def runtime_source_name_tokens(source_name: str | None) -> tuple[str, ...]:
    """Return canonical source-name tokens used by measurement dialects."""
    normalized = normalize_runtime_source_name(source_name)
    if normalized is None:
        return ()
    return tuple(
        token
        for part in normalized.split("__")
        for token in part.split("_")
        if token
    )


@lru_cache(maxsize=32768)
def _normalize_identifier_text(text: str) -> str:
    text = _IDENTIFIER_INITIALISM_BOUNDARY_RE.sub(r"\1_\2", text)
    text = _IDENTIFIER_LOWER_UPPER_BOUNDARY_RE.sub(r"\1_\2", text)
    text = _IDENTIFIER_ALPHA_NUMBER_BOUNDARY_RE.sub(r"\1_\2", text)
    text = _IDENTIFIER_NUMBER_ALPHA_BOUNDARY_RE.sub(r"\1_\2", text)
    text = _IDENTIFIER_NON_ALNUM_RE.sub("_", text)
    return text.strip("_").lower()
