"""
GPU-Accelerated Pattern Matching Module

This module provides GPU-accelerated implementations of pattern matching algorithms
for code analysis. It is designed to integrate with the Semantic Matrix Analyzer (SMA)
project and provides GPU-accelerated alternatives to the pattern matching components.

The module includes:
- Pattern classes for different types of patterns (token, regex, AST)
- GPU-accelerated pattern matchers
- Factory functions for creating patterns

Original code from the Brain project has been adapted for the Semantic Matrix Analyzer.
"""

import logging
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch

# Set up logging
logger = logging.getLogger(__name__)

# Import pattern types
from .token_pattern import TokenPattern
from .regex_pattern import RegexPattern
from .ast_pattern import ASTPattern

# Import pattern base classes
from .pattern_base import PatternType, Pattern, PatternMatch


class GPUPatternMatcher:
    """
    GPU-accelerated pattern matcher.

    This class provides methods for matching patterns against code using GPU acceleration.
    It supports different types of patterns (token, regex, AST) and can match multiple
    patterns in parallel.

    Attributes:
        device: Device to use for matching ("cuda" or "cpu")
        config: Configuration for the matcher
    """

    def __init__(self, device: str = "cuda", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the pattern matcher.

        Args:
            device: Device to use for matching ("cuda" or "cpu")
            config: Configuration for the matcher
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.config = config or {}

        # Create pattern matchers
        self.token_matcher = TokenPattern(self.device)
        self.regex_matcher = RegexPattern(self.device)
        self.ast_matcher = ASTPattern(self.device)

    def match_pattern(
        self,
        pattern: Pattern,
        file_path: Path,
        file_content: str,
        ast_tensors: Optional[Dict[str, torch.Tensor]] = None
    ) -> List[PatternMatch]:
        """
        Match a pattern against a file.

        Args:
            pattern: Pattern to match
            file_path: Path to the file
            file_content: Content of the file
            ast_tensors: Optional AST tensors (for AST patterns)

        Returns:
            List of pattern matches
        """
        if pattern.pattern_type == PatternType.TOKEN:
            return self.token_matcher.match(pattern, file_path, file_content)
        elif pattern.pattern_type == PatternType.REGEX:
            return self.regex_matcher.match(pattern, file_path, file_content)
        elif pattern.pattern_type == PatternType.AST:
            if ast_tensors is None:
                from ..ast_tensor import ASTTensorizer
                tensorizer = ASTTensorizer(device=self.device)
                ast_tensors = tensorizer.tensorize(file_content)
            return self.ast_matcher.match(pattern, file_path, file_content, ast_tensors)
        else:
            logger.warning(f"Unsupported pattern type: {pattern.pattern_type}")
            return []

    def match_patterns(
        self,
        patterns: List[Pattern],
        file_path: Path,
        file_content: str,
        ast_tensors: Optional[Dict[str, torch.Tensor]] = None
    ) -> List[PatternMatch]:
        """
        Match multiple patterns against a file.

        Args:
            patterns: Patterns to match
            file_path: Path to the file
            file_content: Content of the file
            ast_tensors: Optional AST tensors (for AST patterns)

        Returns:
            List of pattern matches
        """
        matches = []

        # Group patterns by type for batch processing
        token_patterns = []
        regex_patterns = []
        ast_patterns = []

        for pattern in patterns:
            if pattern.pattern_type == PatternType.TOKEN:
                token_patterns.append(pattern)
            elif pattern.pattern_type == PatternType.REGEX:
                regex_patterns.append(pattern)
            elif pattern.pattern_type == PatternType.AST:
                ast_patterns.append(pattern)

        # Process token patterns
        if token_patterns:
            token_matches = self.token_matcher.match_batch(token_patterns, file_path, file_content)
            matches.extend(token_matches)

        # Process regex patterns
        if regex_patterns:
            regex_matches = self.regex_matcher.match_batch(regex_patterns, file_path, file_content)
            matches.extend(regex_matches)

        # Process AST patterns
        if ast_patterns:
            if ast_tensors is None:
                from ..ast_tensor import ASTTensorizer
                tensorizer = ASTTensorizer(device=self.device)
                ast_tensors = tensorizer.tensorize(file_content)
            ast_matches = self.ast_matcher.match_batch(ast_patterns, file_path, file_content, ast_tensors)
            matches.extend(ast_matches)

        return matches


# Factory functions for creating patterns

def create_token_pattern(
    name: str,
    description: str,
    sequence: Union[str, List[int]],
    weight: float = 1.0,
    is_negative: bool = False
) -> Pattern:
    """
    Create a token sequence pattern.

    Args:
        name: Name of the pattern
        description: Description of the pattern
        sequence: Token sequence to match
        weight: Weight of this pattern
        is_negative: If True, presence of this pattern reduces the intent score

    Returns:
        Token pattern
    """
    # Convert sequence to list if it's a string
    if isinstance(sequence, str):
        sequence_list = [ord(c) for c in sequence]
    else:
        sequence_list = sequence

    return Pattern(
        name=name,
        description=description,
        pattern_type=PatternType.TOKEN,
        pattern=sequence_list,
        weight=weight,
        is_negative=is_negative
    )


def create_regex_pattern(
    name: str,
    description: str,
    pattern: str,
    weight: float = 1.0,
    is_negative: bool = False
) -> Pattern:
    """
    Create a regex pattern.

    Args:
        name: Name of the pattern
        description: Description of the pattern
        pattern: Regex pattern to match
        weight: Weight of this pattern
        is_negative: If True, presence of this pattern reduces the intent score

    Returns:
        Regex pattern
    """
    import re
    return Pattern(
        name=name,
        description=description,
        pattern_type=PatternType.REGEX,
        pattern=re.compile(pattern),
        weight=weight,
        is_negative=is_negative
    )


def create_ast_pattern(
    name: str,
    description: str,
    node_type: str,
    condition: Optional[Dict[str, Any]] = None,
    weight: float = 1.0,
    is_negative: bool = False
) -> Pattern:
    """
    Create an AST pattern.

    Args:
        name: Name of the pattern
        description: Description of the pattern
        node_type: Type of AST node to match
        condition: Optional condition for the node
        weight: Weight of this pattern
        is_negative: If True, presence of this pattern reduces the intent score

    Returns:
        AST pattern
    """
    return Pattern(
        name=name,
        description=description,
        pattern_type=PatternType.AST,
        pattern=(node_type, condition),
        weight=weight,
        is_negative=is_negative
    )
