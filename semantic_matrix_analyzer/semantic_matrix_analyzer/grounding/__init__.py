"""
Semantic grounding module for Semantic Matrix Analyzer.

This module provides functionality for grounding AI recommendations and findings in actual
code patterns found in the AST, preventing hallucination and ensuring that all insights
are directly tied to evidence from the codebase.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


# Import grounding-related classes for convenience
try:
    from semantic_matrix_analyzer.grounding.evidence import (
        Evidence, EvidenceCollector
    )
    from semantic_matrix_analyzer.grounding.recommendation import (
        Recommendation, RecommendationGrounder
    )
    from semantic_matrix_analyzer.grounding.pattern import (
        CodePattern, PatternMatcher
    )
    from semantic_matrix_analyzer.grounding.verification import (
        GroundingVerifier
    )
except ImportError:
    logger.debug("Grounding modules not available.")
