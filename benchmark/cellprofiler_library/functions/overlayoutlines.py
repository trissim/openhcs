"""Compatibility aliases for CellProfiler OverlayOutlines backend semantics."""

from openhcs.processing.backends.cellprofiler.outlines import (
    LineMode,
    MaxType,
    OutlineDisplayMode,
    OutlineSourceKind,
    OverlayOutlineExecutionContext,
    OverlayOutlineRuntimeRow,
    overlay_outlines,
)

__all__ = [
    "LineMode",
    "MaxType",
    "OutlineDisplayMode",
    "OutlineSourceKind",
    "OverlayOutlineExecutionContext",
    "OverlayOutlineRuntimeRow",
    "overlay_outlines",
]
