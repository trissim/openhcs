"""
Pattern Extraction System for GPU Analysis Plugin.

This module provides functionality for identifying and extracting patterns
from code, text, and user feedback with configurable weights and thresholds.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import re
import hashlib
from dataclasses import dataclass
import numpy as np

# Use relative imports to avoid circular imports
from .dynamic_config import ConfigObserver


@dataclass
class Pattern:
    """Represents a detected pattern."""

    id: str  # Unique identifier
    name: str  # Human-readable name
    pattern_type: str  # Type of pattern (e.g., "code", "text", "behavior")
    content: str  # The actual pattern content
    weight: float  # Importance weight (0.0 to 1.0)
    confidence: float  # Confidence in the pattern (0.0 to 1.0)
    metadata: Dict[str, Any]  # Additional metadata

    @staticmethod
    def create_id(content: str, pattern_type: str) -> str:
        """
        Create a unique ID for a pattern.

        Args:
            content: Pattern content
            pattern_type: Type of pattern

        Returns:
            Unique ID string
        """
        hash_input = f"{pattern_type}:{content}"
        return hashlib.md5(hash_input.encode()).hexdigest()


@dataclass
class PatternMatch:
    """Represents a match between a pattern and content."""

    pattern: Pattern  # The matched pattern
    content: str  # The content that matched
    score: float  # Match score (0.0 to 1.0)
    locations: List[Tuple[int, int]]  # Start and end positions of matches
    context: Dict[str, Any]  # Additional context


class PatternExtractor(ConfigObserver):
    """
    Extracts and matches patterns in code and text.

    This class provides:
    1. Pattern extraction from various sources
    2. Pattern matching with configurable thresholds
    3. Pattern similarity scoring
    4. Pattern evolution tracking
    """

    def __init__(self, config_manager):
        """
        Initialize the pattern extractor.

        Args:
            config_manager: Dynamic configuration manager
        """
        self.config_manager = config_manager
        self.patterns: Dict[str, Pattern] = {}

        # Register as observer for configuration changes
        self.config_manager.register_observer(self)

        # Initialize default pattern configuration if not present
        config = config_manager.get_config()
        if "pattern_config" not in config:
            pattern_config = {
                "pattern_config": {
                    "min_match_threshold": 0.7,
                    "max_patterns_per_type": 100,
                    "similarity_threshold": 0.8,
                    "pattern_types": {
                        "code": {
                            "enabled": True,
                            "weight": 1.0
                        },
                        "text": {
                            "enabled": True,
                            "weight": 0.8
                        },
                        "behavior": {
                            "enabled": True,
                            "weight": 0.9
                        }
                    }
                }
            }
            config_manager.update_config(pattern_config, "system")

    def on_config_changed(self, config: Dict[str, Any], changed_keys: Set[str], source: str) -> None:
        """
        Called when configuration changes.

        Args:
            config: The new configuration
            changed_keys: Set of keys that were changed
            source: Source of the change (e.g., "user", "system", "feedback")
        """
        # Check if pattern configuration changed
        pattern_keys = [key for key in changed_keys if key.startswith("pattern_config")]
        if pattern_keys:
            # Update pattern weights if needed
            for pattern_id, pattern in self.patterns.items():
                pattern_type = pattern.pattern_type
                if f"pattern_config.pattern_types.{pattern_type}.weight" in changed_keys:
                    type_weight = config["pattern_config"]["pattern_types"][pattern_type]["weight"]
                    pattern.weight = min(pattern.weight * type_weight, 1.0)

    def extract_patterns(self, content: str, pattern_type: str,
                         context: Optional[Dict[str, Any]] = None) -> List[Pattern]:
        """
        Extract patterns from content.

        Args:
            content: Content to extract patterns from
            pattern_type: Type of pattern to extract
            context: Optional context information

        Returns:
            List of extracted patterns
        """
        config = self.config_manager.get_config()
        patterns = []

        # Check if pattern type is enabled
        if "pattern_config" not in config or not config["pattern_config"]["pattern_types"].get(pattern_type, {}).get("enabled", False):
            return patterns

        # Extract patterns based on type
        if pattern_type == "code":
            patterns = self._extract_code_patterns(content, context)
        elif pattern_type == "text":
            patterns = self._extract_text_patterns(content, context)
        elif pattern_type == "behavior":
            patterns = self._extract_behavior_patterns(content, context)

        # Store new patterns
        for pattern in patterns:
            if pattern.id not in self.patterns:
                self.patterns[pattern.id] = pattern
            else:
                # Update existing pattern
                existing = self.patterns[pattern.id]
                existing.weight = (existing.weight + pattern.weight) / 2
                existing.confidence = (existing.confidence + pattern.confidence) / 2

        return patterns

    def _extract_code_patterns(self, code: str,
                              context: Optional[Dict[str, Any]] = None) -> List[Pattern]:
        """
        Extract patterns from code.

        Args:
            code: Code to extract patterns from
            context: Optional context information

        Returns:
            List of extracted patterns
        """
        patterns = []

        # Example pattern extraction logic for code
        # 1. Function definitions
        func_pattern = r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
        for match in re.finditer(func_pattern, code):
            func_name = match.group(1)
            pattern_content = f"function:{func_name}"
            pattern_id = Pattern.create_id(pattern_content, "code")

            patterns.append(Pattern(
                id=pattern_id,
                name=f"Function: {func_name}",
                pattern_type="code",
                content=pattern_content,
                weight=0.8,
                confidence=0.9,
                metadata={
                    "function_name": func_name,
                    "position": match.start()
                }
            ))

        # 2. Class definitions
        class_pattern = r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*"
        for match in re.finditer(class_pattern, code):
            class_name = match.group(1)
            pattern_content = f"class:{class_name}"
            pattern_id = Pattern.create_id(pattern_content, "code")

            patterns.append(Pattern(
                id=pattern_id,
                name=f"Class: {class_name}",
                pattern_type="code",
                content=pattern_content,
                weight=0.9,
                confidence=0.9,
                metadata={
                    "class_name": class_name,
                    "position": match.start()
                }
            ))

        # Additional pattern extraction logic would go here

        return patterns

    def _extract_text_patterns(self, text: str,
                              context: Optional[Dict[str, Any]] = None) -> List[Pattern]:
        """
        Extract patterns from text.

        Args:
            text: Text to extract patterns from
            context: Optional context information

        Returns:
            List of extracted patterns
        """
        patterns = []

        # Example pattern extraction logic for text
        # 1. Keywords and phrases
        keywords = ["improve", "fix", "update", "add", "remove", "change", "optimize"]
        for keyword in keywords:
            pattern = re.compile(r"\b" + re.escape(keyword) + r"\b", re.IGNORECASE)
            for match in pattern.finditer(text):
                pattern_content = f"keyword:{keyword.lower()}"
                pattern_id = Pattern.create_id(pattern_content, "text")

                patterns.append(Pattern(
                    id=pattern_id,
                    name=f"Keyword: {keyword}",
                    pattern_type="text",
                    content=pattern_content,
                    weight=0.7,
                    confidence=0.8,
                    metadata={
                        "keyword": keyword.lower(),
                        "position": match.start()
                    }
                ))

        # Additional pattern extraction logic would go here

        return patterns

    def _extract_behavior_patterns(self, behavior: str,
                                  context: Optional[Dict[str, Any]] = None) -> List[Pattern]:
        """
        Extract patterns from behavior descriptions.

        Args:
            behavior: Behavior description to extract patterns from
            context: Optional context information

        Returns:
            List of extracted patterns
        """
        # Implementation would depend on how behavior is represented
        # This is a placeholder
        return []

    def match_patterns(self, content: str, pattern_type: str,
                       threshold: Optional[float] = None) -> List[PatternMatch]:
        """
        Match content against known patterns.

        Args:
            content: Content to match
            pattern_type: Type of patterns to match against
            threshold: Minimum match score (0.0 to 1.0)

        Returns:
            List of pattern matches
        """
        config = self.config_manager.get_config()
        if threshold is None:
            threshold = config.get("pattern_config", {}).get("min_match_threshold", 0.7)

        matches = []

        # Get patterns of the specified type
        relevant_patterns = [p for p in self.patterns.values() if p.pattern_type == pattern_type]

        for pattern in relevant_patterns:
            # Match based on pattern type
            if pattern_type == "code":
                match_result = self._match_code_pattern(pattern, content)
            elif pattern_type == "text":
                match_result = self._match_text_pattern(pattern, content)
            elif pattern_type == "behavior":
                match_result = self._match_behavior_pattern(pattern, content)
            else:
                continue

            # Add match if score exceeds threshold
            if match_result and match_result.score >= threshold:
                matches.append(match_result)

        # Sort by score (highest first)
        matches.sort(key=lambda m: m.score, reverse=True)

        return matches

    def _match_code_pattern(self, pattern: Pattern, code: str) -> Optional[PatternMatch]:
        """
        Match a code pattern against code.

        Args:
            pattern: Pattern to match
            code: Code to match against

        Returns:
            PatternMatch if matched, None otherwise
        """
        # Extract pattern details
        if pattern.content.startswith("function:"):
            func_name = pattern.content.split(":", 1)[1]
            func_pattern = re.compile(r"def\s+" + re.escape(func_name) + r"\s*\(")

            # Find all matches
            locations = []
            for match in func_pattern.finditer(code):
                locations.append((match.start(), match.end()))

            if locations:
                return PatternMatch(
                    pattern=pattern,
                    content=code,
                    score=1.0,  # Exact match
                    locations=locations,
                    context={}
                )

        elif pattern.content.startswith("class:"):
            class_name = pattern.content.split(":", 1)[1]
            class_pattern = re.compile(r"class\s+" + re.escape(class_name) + r"\s*")

            # Find all matches
            locations = []
            for match in class_pattern.finditer(code):
                locations.append((match.start(), match.end()))

            if locations:
                return PatternMatch(
                    pattern=pattern,
                    content=code,
                    score=1.0,  # Exact match
                    locations=locations,
                    context={}
                )

        return None

    def _match_text_pattern(self, pattern: Pattern, text: str) -> Optional[PatternMatch]:
        """
        Match a text pattern against text.

        Args:
            pattern: Pattern to match
            text: Text to match against

        Returns:
            PatternMatch if matched, None otherwise
        """
        # Extract pattern details
        if pattern.content.startswith("keyword:"):
            keyword = pattern.content.split(":", 1)[1]
            keyword_pattern = re.compile(r"\b" + re.escape(keyword) + r"\b", re.IGNORECASE)

            # Find all matches
            locations = []
            for match in keyword_pattern.finditer(text):
                locations.append((match.start(), match.end()))

            if locations:
                return PatternMatch(
                    pattern=pattern,
                    content=text,
                    score=1.0,  # Exact match
                    locations=locations,
                    context={}
                )

        return None

    def _match_behavior_pattern(self, pattern: Pattern, behavior: str) -> Optional[PatternMatch]:
        """
        Match a behavior pattern against behavior.

        Args:
            pattern: Pattern to match
            behavior: Behavior to match against

        Returns:
            PatternMatch if matched, None otherwise
        """
        # Implementation would depend on how behavior is represented
        # This is a placeholder
        return None

    def calculate_pattern_similarity(self, pattern1: Pattern, pattern2: Pattern) -> float:
        """
        Calculate similarity between two patterns.

        Args:
            pattern1: First pattern
            pattern2: Second pattern

        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Only compare patterns of the same type
        if pattern1.pattern_type != pattern2.pattern_type:
            return 0.0

        # Extract content
        content1 = pattern1.content.split(":", 1)[1] if ":" in pattern1.content else pattern1.content
        content2 = pattern2.content.split(":", 1)[1] if ":" in pattern2.content else pattern2.content

        # Calculate similarity based on pattern type
        if pattern1.pattern_type == "code":
            return self._calculate_code_similarity(content1, content2)
        elif pattern1.pattern_type == "text":
            return self._calculate_text_similarity(content1, content2)
        elif pattern1.pattern_type == "behavior":
            return self._calculate_behavior_similarity(content1, content2)

        return 0.0

    def _calculate_code_similarity(self, content1: str, content2: str) -> float:
        """
        Calculate similarity between two code patterns.

        Args:
            content1: First pattern content
            content2: Second pattern content

        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Simple string similarity for now
        # Could be replaced with more sophisticated code similarity metrics
        return self._calculate_string_similarity(content1, content2)

    def _calculate_text_similarity(self, content1: str, content2: str) -> float:
        """
        Calculate similarity between two text patterns.

        Args:
            content1: First pattern content
            content2: Second pattern content

        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Simple string similarity for now
        # Could be replaced with more sophisticated text similarity metrics
        return self._calculate_string_similarity(content1, content2)

    def _calculate_behavior_similarity(self, content1: str, content2: str) -> float:
        """
        Calculate similarity between two behavior patterns.

        Args:
            content1: First pattern content
            content2: Second pattern content

        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Simple string similarity for now
        # Could be replaced with more sophisticated behavior similarity metrics
        return self._calculate_string_similarity(content1, content2)

    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate similarity between two strings.

        Args:
            str1: First string
            str2: Second string

        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Simple Levenshtein distance-based similarity
        if not str1 and not str2:
            return 1.0
        if not str1 or not str2:
            return 0.0

        # Calculate Levenshtein distance
        m, n = len(str1), len(str2)
        d = np.zeros((m + 1, n + 1), dtype=int)

        for i in range(m + 1):
            d[i, 0] = i
        for j in range(n + 1):
            d[0, j] = j

        for j in range(1, n + 1):
            for i in range(1, m + 1):
                if str1[i - 1] == str2[j - 1]:
                    d[i, j] = d[i - 1, j - 1]
                else:
                    d[i, j] = min(
                        d[i - 1, j] + 1,  # deletion
                        d[i, j - 1] + 1,  # insertion
                        d[i - 1, j - 1] + 1  # substitution
                    )

        # Convert distance to similarity
        max_len = max(m, n)
        return 1.0 - (d[m, n] / max_len if max_len > 0 else 0.0)