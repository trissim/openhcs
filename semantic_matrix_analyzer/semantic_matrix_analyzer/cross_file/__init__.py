"""
Cross-file analysis module for Semantic Matrix Analyzer.

This module provides functionality for analyzing relationships and dependencies between files,
enabling the detection of architectural patterns, inconsistencies, and potential refactoring
opportunities across the codebase.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


# Import cross-file analysis-related classes for convenience
try:
    from semantic_matrix_analyzer.cross_file.dependency_graph import (
        Node, Edge, DependencyGraph, DependencyExtractor
    )
    from semantic_matrix_analyzer.cross_file.pattern_detection import (
        ArchitecturalPattern, ArchitecturalPatternDetector
    )
    from semantic_matrix_analyzer.cross_file.inconsistency import (
        Inconsistency, InconsistencyDetector
    )
    from semantic_matrix_analyzer.cross_file.refactoring import (
        RefactoringOpportunity, RefactoringOpportunityDetector
    )
except ImportError:
    logger.debug("Cross-file analysis modules not available.")
