"""
Semantic Matrix Analyzer

A modular tool for analyzing Python codebases using AST (Abstract Syntax Tree) to create
semantically dense matrices that correlate intent with code structure.
"""

__version__ = "0.1.0"

from .semantic_matrix_analyzer import (
    Intent,
    IntentPattern,
    IntentRegistry,
    IntentDetector,
    IntentPlugin,
    ComponentAnalysis,
    SemanticMatrix,
    SemanticMatrixBuilder,
)
