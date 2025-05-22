"""
GPU-Accelerated Analyzers Module

This module provides GPU-accelerated implementations of code analyzers
for semantic analysis. It is designed to integrate with the Semantic Matrix Analyzer (SMA)
project and provides GPU-accelerated alternatives to the analyzer components.

The module includes:
- ComplexityAnalyzer: Analyzes code complexity
- DependencyAnalyzer: Analyzes code dependencies
- SemanticAnalyzer: Analyzes code semantics
- IntentExtractor: Extracts intent from code

Original code from the Brain project has been adapted for the Semantic Matrix Analyzer.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Set up logging
logger = logging.getLogger(__name__)

# Import analyzers
from .complexity_analyzer import ComplexityAnalyzer
from .dependency_analyzer import DependencyAnalyzer
from .semantic_analyzer import SemanticAnalyzer
from .intent_extractor import IntentExtractor

# Define exports
__all__ = [
    'ComplexityAnalyzer',
    'DependencyAnalyzer',
    'SemanticAnalyzer',
    'IntentExtractor'
]
