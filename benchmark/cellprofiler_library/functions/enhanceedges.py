"""Benchmark-library facade for CellProfiler EnhanceEdges."""

from openhcs.processing.backends.cellprofiler.edge import (
    EdgeDirection,
    EdgeEnhancementRequest,
    EdgeEnhancementStrategy,
    EdgeMethod,
    enhance_edges,
)

__all__ = [
    "EdgeDirection",
    "EdgeEnhancementRequest",
    "EdgeEnhancementStrategy",
    "EdgeMethod",
    "enhance_edges",
]
