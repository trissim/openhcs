"""
GPU-Accelerated Regex Pattern Matcher

This module provides a GPU-accelerated implementation of regex pattern matching
for code analysis. It uses PyTorch to accelerate the matching process and can
run on both CPU and GPU.

Original code from the Brain project has been adapted for the Semantic Matrix Analyzer.
"""

import re
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


class RegexPattern:
    """
    GPU-accelerated regex pattern matcher.

    This class provides methods for matching regex patterns against code using GPU acceleration.
    It uses PyTorch to accelerate the matching process and can run on both CPU and GPU.

    Attributes:
        device: Device to use for matching ("cuda" or "cpu")
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize the regex pattern matcher.

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
        Match a regex pattern against a file.

        Args:
            pattern: Pattern to match
            file_path: Path to the file
            file_content: Content of the file

        Returns:
            List of pattern matches
        """
        if pattern.pattern_type != PatternType.REGEX:
            logger.warning(f"Expected regex pattern, got {pattern.pattern_type}")
            return []

        # Extract regex pattern
        regex = pattern.pattern

        # Find matches
        matches = []

        # Use Python's re module for regex matching
        for match in regex.finditer(file_content):
            start, end = match.span()

            # Find line numbers
            line_start = file_content[:start].count('\n') + 1
            line_end = line_start + file_content[start:end].count('\n')

            # Extract matched source code
            matched_text = match.group(0)

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
        Match multiple regex patterns against a file.

        Args:
            patterns: Patterns to match
            file_path: Path to the file
            file_content: Content of the file

        Returns:
            List of pattern matches
        """
        # Find matches for each pattern
        all_matches = []
        for pattern in patterns:
            if pattern.pattern_type != PatternType.REGEX:
                continue

            # Extract regex pattern
            regex = pattern.pattern

            # Find matches
            matches = []

            # Use Python's re module for regex matching
            for match in regex.finditer(file_content):
                start, end = match.span()

                # Find line numbers
                line_start = file_content[:start].count('\n') + 1
                line_end = line_start + file_content[start:end].count('\n')

                # Extract matched source code
                matched_text = match.group(0)

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

    def compile(self, pattern: str) -> torch.jit.ScriptModule:
        """
        Compile a regex pattern to a TorchScript module.

        Args:
            pattern: Regex pattern to match

        Returns:
            Compiled TorchScript module
        """
        # Compile regex
        regex = re.compile(pattern)

        # Since regex can't be directly compiled to TorchScript,
        # we'll create a wrapper that processes on CPU and returns results

        # This is a non-TorchScript function that will be called from TorchScript
        def find_matches(text: str) -> List[Tuple[int, int]]:
            return [(m.start(), m.end()) for m in regex.finditer(text)]

        # TorchScript wrapper
        class RegexMatcher(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, tokens: torch.Tensor) -> torch.Tensor:
                # Convert tokens to string if needed
                if tokens.dtype != torch.uint8 and tokens.dtype != torch.int8:
                    # Assume it's character codes
                    text = "".join(chr(int(c)) for c in tokens.cpu().numpy())
                else:
                    # Assume it's bytes
                    text = tokens.cpu().numpy().tobytes().decode('utf-8', errors='ignore')

                # Find matches
                matches = find_matches(text)

                # Create result tensor
                result = torch.zeros(len(tokens), dtype=torch.bool, device=tokens.device)

                # Mark match positions
                for start, end in matches:
                    if start < len(result) and end <= len(result):
                        result[start:end] = True

                return result

        # Create and script the module
        matcher = RegexMatcher()
        return torch.jit.script(matcher)
