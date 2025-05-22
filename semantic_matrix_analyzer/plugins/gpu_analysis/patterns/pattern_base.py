"""
Pattern Base Classes for GPU Analysis Plugin.

This module provides the base classes for patterns and pattern matching.
These classes are used by the pattern matchers to represent patterns and matches.
"""

import logging
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Set up logging
logger = logging.getLogger(__name__)

# Define pattern type enum
class PatternType(Enum):
    """Types of patterns that can be detected."""
    TOKEN = auto()    # Token sequence matching
    REGEX = auto()    # Regular expression matching
    AST = auto()      # AST-based pattern matching
    SEMANTIC = auto() # Semantic pattern matching


class Pattern:
    """
    A pattern to detect in code.
    
    Patterns are used to identify specific code structures or behaviors.
    
    Attributes:
        name: Name of the pattern
        description: Description of the pattern
        pattern_type: Type of pattern
        pattern: The actual pattern (depends on pattern_type)
        weight: Weight of this pattern in the overall intent score
        is_negative: If True, presence of this pattern reduces the intent score
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        pattern_type: PatternType,
        pattern: Any,
        weight: float = 1.0,
        is_negative: bool = False
    ):
        """
        Initialize a pattern.
        
        Args:
            name: Name of the pattern
            description: Description of the pattern
            pattern_type: Type of pattern
            pattern: The actual pattern (depends on pattern_type)
            weight: Weight of this pattern in the overall intent score
            is_negative: If True, presence of this pattern reduces the intent score
        """
        self.name = name
        self.description = description
        self.pattern_type = pattern_type
        self.pattern = pattern
        self.weight = weight
        self.is_negative = is_negative


class PatternMatch:
    """
    A match of a pattern in code.
    
    Pattern matches provide evidence for findings.
    
    Attributes:
        pattern: The pattern that matched
        file_path: Path to the file where the match was found
        source_range: Range of lines where the match was found
        source_code: Source code that matched
        node: AST node that matched (for AST patterns)
        confidence: Confidence in the match
    """
    
    def __init__(
        self,
        pattern: Pattern,
        file_path: Path,
        source_range: Optional[Tuple[int, int]] = None,
        source_code: Optional[str] = None,
        node: Optional[Any] = None,
        confidence: float = 1.0
    ):
        """
        Initialize a pattern match.
        
        Args:
            pattern: The pattern that matched
            file_path: Path to the file where the match was found
            source_range: Range of lines where the match was found
            source_code: Source code that matched
            node: AST node that matched (for AST patterns)
            confidence: Confidence in the match
        """
        self.pattern = pattern
        self.file_path = file_path
        self.source_range = source_range
        self.source_code = source_code
        self.node = node
        self.confidence = confidence
