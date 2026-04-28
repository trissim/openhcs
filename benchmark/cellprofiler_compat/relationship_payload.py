"""Typed relationship payloads emitted by absorbed CellProfiler modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CellProfilerRelationshipPayload:
    """Parent-child relationship ids emitted by absorbed modules."""

    parent_ids: tuple[int, ...]
    child_ids: tuple[int, ...]
