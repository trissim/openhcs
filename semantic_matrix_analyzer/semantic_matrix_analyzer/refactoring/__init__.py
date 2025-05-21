"""
Interactive refactoring workflow module for Semantic Matrix Analyzer.

This module provides functionality for interactive refactoring workflows that break down
large changes into manageable steps, track progress, provide checkpoints for verification,
and support rollbacks if needed.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


# Import refactoring-related classes for convenience
try:
    from semantic_matrix_analyzer.refactoring.workflow import (
        RefactoringWorkflow, RefactoringStep, RefactoringCheckpoint,
        StepStatus, CheckpointStatus
    )
    from semantic_matrix_analyzer.refactoring.manager import (
        RefactoringManager
    )
    from semantic_matrix_analyzer.refactoring.executor import (
        RefactoringExecutor
    )
    from semantic_matrix_analyzer.refactoring.tracker import (
        ProgressTracker, ProgressEvent
    )
except ImportError:
    logger.debug("Refactoring modules not available.")
