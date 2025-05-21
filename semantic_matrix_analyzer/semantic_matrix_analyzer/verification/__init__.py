"""
Verification module for Semantic Matrix Analyzer.

This module provides functionality for verifying code suggestions against the AST,
ensuring syntactic and semantic correctness, detecting potential side effects,
and providing confidence scores for each suggestion.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


# Import verification-related classes for convenience
try:
    from semantic_matrix_analyzer.verification.suggestion import (
        CodeSuggestion, VerificationResult, SuggestionVerifier
    )
    from semantic_matrix_analyzer.verification.simulation import (
        CodeChangeSimulator
    )
    from semantic_matrix_analyzer.verification.side_effects import (
        SideEffectDetector
    )
    from semantic_matrix_analyzer.verification.reporting import (
        VerificationReporter
    )
except ImportError:
    logger.debug("Verification modules not available.")
