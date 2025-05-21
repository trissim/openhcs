"""
AST-based pattern matcher for Semantic Matrix Analyzer.

This module provides a pattern matcher that uses AST to detect patterns in code.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from semantic_matrix_analyzer.language import LanguageParser
from semantic_matrix_analyzer.patterns import Pattern, PatternMatch, PatternMatcher, PatternType


class ASTPatternMatcher(PatternMatcher):
    """Matcher for AST patterns."""
    
    def __init__(self, language_parser: LanguageParser):
        """Initialize the matcher.
        
        Args:
            language_parser: The language parser to use.
        """
        self.language_parser = language_parser
    
    def match_pattern(
        self, 
        pattern: Pattern, 
        file_path: Path, 
        file_content: str, 
        ast_node: Any
    ) -> List[PatternMatch]:
        """Match an AST pattern against a file.
        
        Args:
            pattern: The pattern to match.
            file_path: Path to the file.
            file_content: Content of the file.
            ast_node: AST representation of the file.
            
        Returns:
            A list of pattern matches.
        """
        if pattern.pattern_type != PatternType.AST:
            return []
        
        matches = []
        node_type, condition = pattern.pattern
        
        # Find all matching nodes
        for node in self._find_nodes_of_type(ast_node, node_type):
            if self._node_matches_condition(node, condition):
                source_range = self.language_parser.get_node_source_range(node)
                source_code = self.language_parser.get_node_source(node, file_content)
                
                matches.append(PatternMatch(
                    pattern=pattern,
                    file_path=file_path,
                    source_range=source_range,
                    source_code=source_code,
                    node=node
                ))
        
        return matches
    
    def _find_nodes_of_type(self, root_node: Any, node_type: str) -> List[Any]:
        """Find all nodes of a specific type in the AST.
        
        Args:
            root_node: The root node to search from.
            node_type: The type of nodes to find.
            
        Returns:
            A list of matching nodes.
        """
        matching_nodes = []
        
        def visit(node):
            if self.language_parser.get_node_type(node) == node_type:
                matching_nodes.append(node)
            
            for child in self.language_parser.get_node_children(node):
                visit(child)
        
        visit(root_node)
        return matching_nodes
    
    def _node_matches_condition(self, node: Any, condition: Optional[Dict[str, Any]]) -> bool:
        """Check if a node matches a condition.
        
        Args:
            node: The node to check.
            condition: The condition to check against.
            
        Returns:
            True if the node matches the condition, False otherwise.
        """
        if condition is None:
            return True
        
        # Check name condition
        if "name" in condition:
            node_name = self.language_parser.get_node_name(node)
            if node_name != condition["name"]:
                return False
        
        # Check has_child condition
        if "has_child" in condition:
            child_type = condition["has_child"]
            child_found = False
            
            for child in self.language_parser.get_node_children(node):
                if self.language_parser.get_node_type(child) == child_type:
                    child_found = True
                    break
            
            if not child_found:
                return False
        
        # Check has_descendant condition
        if "has_descendant" in condition:
            descendant_type = condition["has_descendant"]
            descendant_found = False
            
            def visit(n):
                nonlocal descendant_found
                if self.language_parser.get_node_type(n) == descendant_type:
                    descendant_found = True
                    return True
                
                for child in self.language_parser.get_node_children(n):
                    if visit(child):
                        return True
                
                return False
            
            for child in self.language_parser.get_node_children(node):
                if visit(child):
                    break
            
            if not descendant_found:
                return False
        
        # Add more condition checks as needed
        
        return True
