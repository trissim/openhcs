"""
GPU-Accelerated Token Pattern Matcher

This module provides a GPU-accelerated implementation of token pattern matching
for code analysis. It uses PyTorch to accelerate the matching process and can
run on both CPU and GPU.

Original code from the Brain project has been adapted for the Semantic Matrix Analyzer.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import from pattern_base to avoid circular imports
from .pattern_base import Pattern, PatternMatch, PatternType

# Set up logging
logger = logging.getLogger(__name__)


class TokenPattern:
    """
    GPU-accelerated token pattern matcher.

    This class provides methods for matching token patterns against code using GPU acceleration.
    It uses PyTorch to accelerate the matching process and can run on both CPU and GPU.

    Attributes:
        device: Device to use for matching ("cuda" or "cpu")
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize the token pattern matcher.

        Args:
            device: Device to use for matching ("cuda" or "cpu")
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"

    def match(
        self,
        pattern: Pattern,
        file_path: Path,
        file_content: str
    ) -> List[PatternMatch]:
        """
        Match a token pattern against a file.

        Args:
            pattern: Pattern to match
            file_path: Path to the file
            file_content: Content of the file

        Returns:
            List of pattern matches
        """
        if pattern.pattern_type != PatternType.TOKEN:
            logger.warning(f"Expected token pattern, got {pattern.pattern_type}")
            return []

        # Convert file content to token tensor
        tokens = torch.tensor([ord(c) for c in file_content], dtype=torch.int32, device=self.device)

        # Convert pattern to tensor
        sequence = pattern.pattern
        pattern_tensor = torch.tensor(sequence, dtype=torch.int32, device=self.device)

        # Find matches
        matches = []

        # Use sliding window to find matches
        if len(tokens) >= len(pattern_tensor):
            for i in range(len(tokens) - len(pattern_tensor) + 1):
                window = tokens[i:i+len(pattern_tensor)]
                if torch.all(window == pattern_tensor):
                    # Find line numbers
                    line_start = file_content[:i].count('\n') + 1
                    line_end = line_start + file_content[i:i+len(pattern_tensor)].count('\n')

                    # Extract matched source code
                    matched_text = file_content[i:i+len(pattern_tensor)]

                    # Create match
                    matches.append(PatternMatch(
                        pattern=pattern,
                        file_path=file_path,
                        source_range=(line_start, line_end),
                        source_code=matched_text,
                        confidence=1.0
                    ))

        return matches

    def match_batch(
        self,
        patterns: List[Pattern],
        file_path: Path,
        file_content: str
    ) -> List[PatternMatch]:
        """
        Match multiple token patterns against a file.

        Args:
            patterns: Patterns to match
            file_path: Path to the file
            file_content: Content of the file

        Returns:
            List of pattern matches
        """
        # Convert file content to token tensor
        tokens = torch.tensor([ord(c) for c in file_content], dtype=torch.int32, device=self.device)

        # Find matches for each pattern
        all_matches = []
        for pattern in patterns:
            if pattern.pattern_type != PatternType.TOKEN:
                continue

            # Convert pattern to tensor
            sequence = pattern.pattern
            pattern_tensor = torch.tensor(sequence, dtype=torch.int32, device=self.device)

            # Find matches
            matches = []

            # Use sliding window to find matches
            if len(tokens) >= len(pattern_tensor):
                for i in range(len(tokens) - len(pattern_tensor) + 1):
                    window = tokens[i:i+len(pattern_tensor)]
                    if torch.all(window == pattern_tensor):
                        # Find line numbers
                        line_start = file_content[:i].count('\n') + 1
                        line_end = line_start + file_content[i:i+len(pattern_tensor)].count('\n')

                        # Extract matched source code
                        matched_text = file_content[i:i+len(pattern_tensor)]

                        # Create match
                        matches.append(PatternMatch(
                            pattern=pattern,
                            file_path=file_path,
                            source_range=(line_start, line_end),
                            source_code=matched_text,
                            confidence=1.0
                        ))

            all_matches.extend(matches)

        return all_matches

    def compile(self, sequence: List[int]) -> torch.jit.ScriptModule:
        """
        Compile a token sequence pattern to a TorchScript module.

        Args:
            sequence: Token sequence to match

        Returns:
            Compiled TorchScript module
        """
        # Create tensor for sequence
        sequence_tensor = torch.tensor(sequence, dtype=torch.int32)

        @torch.jit.script
        def match_sequence(tokens: torch.Tensor) -> torch.Tensor:
            # Convert sequence to tensor on the same device
            pattern = torch.tensor(sequence, device=tokens.device, dtype=tokens.dtype)

            # Find matches using sliding window
            matches = torch.zeros(tokens.size(0) - pattern.size(0) + 1,
                                 dtype=torch.bool, device=tokens.device)

            for i in range(matches.size(0)):
                window = tokens[i:i+pattern.size(0)]
                matches[i] = torch.all(window == pattern)

            return matches

        return match_sequence
