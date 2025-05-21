"""
Python language parser for Semantic Matrix Analyzer.

This module provides a parser for Python code using the built-in ast module.
"""

import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from semantic_matrix_analyzer.language import LanguageParser, language_registry


class PythonParser(LanguageParser):
    """Parser for Python code.
    
    This parser uses the built-in ast module to parse Python code.
    """
    
    @classmethod
    def get_supported_extensions(cls) -> Set[str]:
        """Return file extensions supported by this parser.
        
        Returns:
            A set of file extensions (including the dot) that this parser supports.
        """
        return {".py", ".pyi"}
    
    def parse_file(self, file_path: Path) -> ast.AST:
        """Parse a Python file and return its AST.
        
        Args:
            file_path: Path to the file to parse.
            
        Returns:
            The AST representation of the file.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            SyntaxError: If the file contains syntax errors.
            ValueError: If the file is not a Python file.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() not in self.get_supported_extensions():
            raise ValueError(f"Not a Python file: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        return ast.parse(content, filename=str(file_path))
    
    def get_node_type(self, node: ast.AST) -> str:
        """Get the type of an AST node.
        
        Args:
            node: An AST node returned by parse_file.
            
        Returns:
            A string representing the type of the node.
        """
        return node.__class__.__name__
    
    def get_node_name(self, node: ast.AST) -> Optional[str]:
        """Get the name of an AST node, if applicable.
        
        Args:
            node: An AST node returned by parse_file.
            
        Returns:
            The name of the node, or None if the node does not have a name.
        """
        if hasattr(node, "name"):
            return node.name
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.ClassDef) or isinstance(node, ast.AsyncFunctionDef):
            return node.name
        elif isinstance(node, ast.ImportFrom):
            return node.module
        elif isinstance(node, ast.alias):
            return node.name
        
        return None
    
    def get_node_children(self, node: ast.AST) -> List[ast.AST]:
        """Get the children of an AST node.
        
        Args:
            node: An AST node returned by parse_file.
            
        Returns:
            A list of child nodes.
        """
        children = []
        
        for field, value in ast.iter_fields(node):
            if isinstance(value, ast.AST):
                children.append(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        children.append(item)
        
        return children
    
    def get_node_source_range(self, node: ast.AST) -> Optional[Tuple[int, int]]:
        """Get the source range of an AST node.
        
        Args:
            node: An AST node returned by parse_file.
            
        Returns:
            A tuple of (start_line, end_line), or None if not available.
            Line numbers are 1-based.
        """
        if hasattr(node, "lineno"):
            start_line = node.lineno
            end_line = getattr(node, "end_lineno", start_line)
            return (start_line, end_line)
        
        return None
    
    def get_node_source(self, node: ast.AST, file_content: str) -> Optional[str]:
        """Get the source code for an AST node.
        
        Args:
            node: An AST node returned by parse_file.
            file_content: The content of the file.
            
        Returns:
            The source code for the node, or None if not available.
        """
        source_range = self.get_node_source_range(node)
        if not source_range:
            return None
        
        start_line, end_line = source_range
        lines = file_content.splitlines()
        
        # Adjust for 0-based indexing
        start_idx = start_line - 1
        end_idx = end_line
        
        if start_idx < 0 or end_idx > len(lines):
            return None
        
        return "\n".join(lines[start_idx:end_idx])


# Register the Python parser
language_registry.register_parser(PythonParser)
