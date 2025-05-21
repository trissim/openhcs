"""
Cross-file analysis components for the Structural Intent Analysis system.
"""

from semantic_matrix_analyzer.cross_file.dependency_graph import (
    Node, Edge, DependencyGraph, DependencyExtractor
)

__all__ = [
    "Node",
    "Edge",
    "DependencyGraph",
    "DependencyExtractor"
]
