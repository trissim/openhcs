"""
Pattern detection module for Semantic Matrix Analyzer.

This module provides abstractions for defining and detecting patterns in code.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union


class PatternType(Enum):
    """Types of patterns that can be detected."""
    STRING = auto()  # Simple string matching
    REGEX = auto()   # Regular expression matching
    AST = auto()     # AST-based pattern matching
    SEMANTIC = auto() # Semantic pattern matching


@dataclass
class Pattern:
    """A pattern to detect in code.
    
    Patterns are used to identify specific code structures or behaviors.
    """
    name: str
    description: str
    pattern_type: PatternType
    pattern: Any  # The actual pattern (string, regex, AST node type, etc.)
    weight: float = 1.0  # Weight of this pattern in the overall intent score (0.0 to 1.0)
    is_negative: bool = False  # If True, presence of this pattern reduces the intent score


@dataclass
class PatternMatch:
    """A match of a pattern in code.
    
    Pattern matches provide evidence for findings.
    """
    pattern: Pattern
    file_path: Path
    source_range: Optional[Tuple[int, int]] = None  # (start_line, end_line)
    source_code: Optional[str] = None
    node: Optional[Any] = None  # The AST node that matched
    confidence: float = 1.0  # Confidence in the match (0.0 to 1.0)


@dataclass
class Intent:
    """An intent to detect in code.
    
    Intents represent high-level goals or principles that code should follow.
    """
    name: str
    description: str
    patterns: List[Pattern] = field(default_factory=list)
    
    def add_pattern(self, pattern: Pattern) -> None:
        """Add a pattern to this intent."""
        self.patterns.append(pattern)
    
    def add_string_pattern(
        self, 
        name: str, 
        description: str, 
        pattern: str, 
        weight: float = 1.0, 
        is_negative: bool = False
    ) -> None:
        """Add a string pattern to this intent."""
        self.patterns.append(Pattern(
            name=name,
            description=description,
            pattern_type=PatternType.STRING,
            pattern=pattern,
            weight=weight,
            is_negative=is_negative
        ))
    
    def add_regex_pattern(
        self, 
        name: str, 
        description: str, 
        pattern: str, 
        weight: float = 1.0, 
        is_negative: bool = False
    ) -> None:
        """Add a regex pattern to this intent."""
        import re
        self.patterns.append(Pattern(
            name=name,
            description=description,
            pattern_type=PatternType.REGEX,
            pattern=re.compile(pattern),
            weight=weight,
            is_negative=is_negative
        ))
    
    def add_ast_pattern(
        self, 
        name: str, 
        description: str, 
        node_type: str, 
        condition: Optional[Dict[str, Any]] = None,
        weight: float = 1.0, 
        is_negative: bool = False
    ) -> None:
        """Add an AST pattern to this intent."""
        self.patterns.append(Pattern(
            name=name,
            description=description,
            pattern_type=PatternType.AST,
            pattern=(node_type, condition),
            weight=weight,
            is_negative=is_negative
        ))
    
    def add_semantic_pattern(
        self, 
        name: str, 
        description: str, 
        pattern: Dict[str, Any], 
        weight: float = 1.0, 
        is_negative: bool = False
    ) -> None:
        """Add a semantic pattern to this intent."""
        self.patterns.append(Pattern(
            name=name,
            description=description,
            pattern_type=PatternType.SEMANTIC,
            pattern=pattern,
            weight=weight,
            is_negative=is_negative
        ))


class PatternMatcher(ABC):
    """Base class for pattern matchers.
    
    Pattern matchers are responsible for detecting patterns in code.
    """
    
    @abstractmethod
    def match_pattern(
        self, 
        pattern: Pattern, 
        file_path: Path, 
        file_content: str, 
        ast_node: Any
    ) -> List[PatternMatch]:
        """Match a pattern against a file.
        
        Args:
            pattern: The pattern to match.
            file_path: Path to the file.
            file_content: Content of the file.
            ast_node: AST representation of the file.
            
        Returns:
            A list of pattern matches.
        """
        pass


class StringPatternMatcher(PatternMatcher):
    """Matcher for string patterns."""
    
    def match_pattern(
        self, 
        pattern: Pattern, 
        file_path: Path, 
        file_content: str, 
        ast_node: Any
    ) -> List[PatternMatch]:
        """Match a string pattern against a file.
        
        Args:
            pattern: The pattern to match.
            file_path: Path to the file.
            file_content: Content of the file.
            ast_node: AST representation of the file.
            
        Returns:
            A list of pattern matches.
        """
        if pattern.pattern_type != PatternType.STRING:
            return []
        
        matches = []
        string_pattern = pattern.pattern
        
        # Simple string matching
        if string_pattern in file_content:
            # Find all occurrences
            start = 0
            while True:
                start = file_content.find(string_pattern, start)
                if start == -1:
                    break
                
                # Find line numbers
                line_start = file_content.count('\n', 0, start) + 1
                line_end = line_start + file_content[start:].count('\n', 0, len(string_pattern))
                
                # Extract matched source code
                lines = file_content.splitlines()
                source_code = '\n'.join(lines[line_start-1:line_end])
                
                matches.append(PatternMatch(
                    pattern=pattern,
                    file_path=file_path,
                    source_range=(line_start, line_end),
                    source_code=source_code
                ))
                
                start += len(string_pattern)
        
        return matches


class RegexPatternMatcher(PatternMatcher):
    """Matcher for regex patterns."""
    
    def match_pattern(
        self, 
        pattern: Pattern, 
        file_path: Path, 
        file_content: str, 
        ast_node: Any
    ) -> List[PatternMatch]:
        """Match a regex pattern against a file.
        
        Args:
            pattern: The pattern to match.
            file_path: Path to the file.
            file_content: Content of the file.
            ast_node: AST representation of the file.
            
        Returns:
            A list of pattern matches.
        """
        if pattern.pattern_type != PatternType.REGEX:
            return []
        
        matches = []
        regex_pattern = pattern.pattern
        
        # Find all matches
        for match in regex_pattern.finditer(file_content):
            start, end = match.span()
            
            # Find line numbers
            line_start = file_content.count('\n', 0, start) + 1
            line_end = line_start + file_content[start:end].count('\n')
            
            # Extract matched source code
            matched_text = match.group(0)
            
            matches.append(PatternMatch(
                pattern=pattern,
                file_path=file_path,
                source_range=(line_start, line_end),
                source_code=matched_text
            ))
        
        return matches
