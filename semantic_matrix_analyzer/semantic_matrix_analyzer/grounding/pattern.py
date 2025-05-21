"""
Pattern module for semantic grounding.

This module provides functionality for defining and matching code patterns in the codebase.
"""

import ast
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable

logger = logging.getLogger(__name__)


@dataclass
class CodePattern:
    """A pattern of code to match in the codebase."""
    
    id: str
    name: str
    description: str
    ast_pattern: Optional[Dict[str, Any]] = None
    regex_pattern: Optional[str] = None
    custom_matcher: Optional[Callable[[str], bool]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata
        }
        
        if self.ast_pattern:
            result["ast_pattern"] = self.ast_pattern
        
        if self.regex_pattern:
            result["regex_pattern"] = self.regex_pattern
        
        # Note: custom_matcher cannot be serialized
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodePattern':
        """Create from dictionary after deserialization."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            ast_pattern=data.get("ast_pattern"),
            regex_pattern=data.get("regex_pattern"),
            metadata=data.get("metadata", {})
        )


class PatternMatcher:
    """Matches code patterns in the codebase."""
    
    def __init__(self):
        """Initialize the pattern matcher."""
        self.patterns: Dict[str, CodePattern] = {}
    
    def add_pattern(self, pattern: CodePattern) -> None:
        """Add a pattern to the matcher.
        
        Args:
            pattern: The pattern to add.
        """
        self.patterns[pattern.id] = pattern
    
    def get_pattern(self, pattern_id: str) -> Optional[CodePattern]:
        """Get a pattern by ID.
        
        Args:
            pattern_id: The pattern ID.
            
        Returns:
            The pattern, or None if not found.
        """
        return self.patterns.get(pattern_id)
    
    def match_patterns(self, code: str, file_path: Optional[Path] = None) -> List[Tuple[CodePattern, List[Tuple[int, int]]]]:
        """Match patterns in code.
        
        Args:
            code: The code to match patterns in.
            file_path: The path to the file (optional).
            
        Returns:
            A list of tuples of (pattern, matches), where matches is a list of (start_line, end_line) tuples.
        """
        results = []
        
        for pattern in self.patterns.values():
            matches = []
            
            # Match using AST pattern
            if pattern.ast_pattern:
                ast_matches = self._match_ast_pattern(pattern, code, file_path)
                matches.extend(ast_matches)
            
            # Match using regex pattern
            if pattern.regex_pattern:
                regex_matches = self._match_regex_pattern(pattern, code)
                matches.extend(regex_matches)
            
            # Match using custom matcher
            if pattern.custom_matcher:
                custom_matches = self._match_custom_pattern(pattern, code)
                matches.extend(custom_matches)
            
            if matches:
                results.append((pattern, matches))
        
        return results
    
    def _match_ast_pattern(self, pattern: CodePattern, code: str, file_path: Optional[Path] = None) -> List[Tuple[int, int]]:
        """Match an AST pattern in code.
        
        Args:
            pattern: The pattern to match.
            code: The code to match in.
            file_path: The path to the file (optional).
            
        Returns:
            A list of (start_line, end_line) tuples.
        """
        matches = []
        
        try:
            # Parse the code
            tree = ast.parse(code, filename=str(file_path) if file_path else "<string>")
            
            # Match the pattern
            for node in ast.walk(tree):
                if self._node_matches_pattern(node, pattern.ast_pattern):
                    # Get the node's line range
                    if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                        matches.append((node.lineno, node.end_lineno))
        except Exception as e:
            logger.error(f"Error matching AST pattern: {e}")
        
        return matches
    
    def _node_matches_pattern(self, node: ast.AST, pattern: Dict[str, Any]) -> bool:
        """Check if a node matches a pattern.
        
        Args:
            node: The AST node.
            pattern: The pattern to match.
            
        Returns:
            True if the node matches the pattern, False otherwise.
        """
        # Check node type
        if "type" in pattern and type(node).__name__ != pattern["type"]:
            return False
        
        # Check node fields
        for field, value in pattern.items():
            if field == "type":
                continue
            
            if not hasattr(node, field):
                return False
            
            node_value = getattr(node, field)
            
            if isinstance(value, dict) and isinstance(node_value, ast.AST):
                # Recursively match nested nodes
                if not self._node_matches_pattern(node_value, value):
                    return False
            elif isinstance(value, list) and isinstance(node_value, list):
                # Match lists of nodes
                if len(value) != len(node_value):
                    return False
                
                for i, item in enumerate(value):
                    if isinstance(item, dict) and isinstance(node_value[i], ast.AST):
                        if not self._node_matches_pattern(node_value[i], item):
                            return False
                    elif item != node_value[i]:
                        return False
            elif value != node_value:
                return False
        
        return True
    
    def _match_regex_pattern(self, pattern: CodePattern, code: str) -> List[Tuple[int, int]]:
        """Match a regex pattern in code.
        
        Args:
            pattern: The pattern to match.
            code: The code to match in.
            
        Returns:
            A list of (start_line, end_line) tuples.
        """
        matches = []
        
        if not pattern.regex_pattern:
            return matches
        
        try:
            # Get the lines of code
            lines = code.splitlines()
            
            # Compile the regex
            regex = re.compile(pattern.regex_pattern)
            
            # Match the pattern
            for i, line in enumerate(lines):
                if regex.search(line):
                    # For simplicity, we're assuming the match is on a single line
                    matches.append((i + 1, i + 1))
        except Exception as e:
            logger.error(f"Error matching regex pattern: {e}")
        
        return matches
    
    def _match_custom_pattern(self, pattern: CodePattern, code: str) -> List[Tuple[int, int]]:
        """Match a custom pattern in code.
        
        Args:
            pattern: The pattern to match.
            code: The code to match in.
            
        Returns:
            A list of (start_line, end_line) tuples.
        """
        matches = []
        
        if not pattern.custom_matcher:
            return matches
        
        try:
            # Get the lines of code
            lines = code.splitlines()
            
            # Match the pattern
            for i, line in enumerate(lines):
                if pattern.custom_matcher(line):
                    # For simplicity, we're assuming the match is on a single line
                    matches.append((i + 1, i + 1))
        except Exception as e:
            logger.error(f"Error matching custom pattern: {e}")
        
        return matches
    
    def create_ast_pattern(self, node_type: str, **fields) -> Dict[str, Any]:
        """Create an AST pattern.
        
        Args:
            node_type: The AST node type.
            **fields: The fields to match.
            
        Returns:
            An AST pattern.
        """
        pattern = {"type": node_type}
        pattern.update(fields)
        return pattern
