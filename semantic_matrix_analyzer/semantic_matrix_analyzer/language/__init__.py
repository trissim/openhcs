"""
Language parsing module for Semantic Matrix Analyzer.

This module provides abstractions for language-specific parsing and analysis.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Type


class LanguageParser(ABC):
    """Abstract base class for language-specific parsers.
    
    All language parsers must implement this interface to provide
    a consistent way to parse and analyze code in different languages.
    """
    
    @classmethod
    @abstractmethod
    def get_supported_extensions(cls) -> Set[str]:
        """Return file extensions supported by this parser.
        
        Returns:
            A set of file extensions (including the dot) that this parser supports.
            For example: {".py", ".pyi"}
        """
        pass
    
    @abstractmethod
    def parse_file(self, file_path: Path) -> Any:
        """Parse a file and return its AST.
        
        Args:
            file_path: Path to the file to parse.
            
        Returns:
            The AST representation of the file, which may be language-specific.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            SyntaxError: If the file contains syntax errors.
            ValueError: If the file is not supported by this parser.
        """
        pass
    
    @abstractmethod
    def get_node_type(self, node: Any) -> str:
        """Get the type of an AST node.
        
        Args:
            node: An AST node returned by parse_file.
            
        Returns:
            A string representing the type of the node.
        """
        pass
    
    @abstractmethod
    def get_node_name(self, node: Any) -> Optional[str]:
        """Get the name of an AST node, if applicable.
        
        Args:
            node: An AST node returned by parse_file.
            
        Returns:
            The name of the node, or None if the node does not have a name.
        """
        pass
    
    @abstractmethod
    def get_node_children(self, node: Any) -> List[Any]:
        """Get the children of an AST node.
        
        Args:
            node: An AST node returned by parse_file.
            
        Returns:
            A list of child nodes.
        """
        pass
    
    @abstractmethod
    def get_node_source_range(self, node: Any) -> Optional[Tuple[int, int]]:
        """Get the source range of an AST node.
        
        Args:
            node: An AST node returned by parse_file.
            
        Returns:
            A tuple of (start_line, end_line), or None if not available.
            Line numbers are 1-based.
        """
        pass
    
    @abstractmethod
    def get_node_source(self, node: Any, file_content: str) -> Optional[str]:
        """Get the source code for an AST node.
        
        Args:
            node: An AST node returned by parse_file.
            file_content: The content of the file.
            
        Returns:
            The source code for the node, or None if not available.
        """
        pass


class LanguageRegistry:
    """Registry for language parsers.
    
    This class maintains a registry of language parsers and provides
    methods to find the appropriate parser for a given file.
    """
    
    def __init__(self):
        """Initialize the registry."""
        self._parsers: Dict[str, Type[LanguageParser]] = {}
        self._extension_map: Dict[str, Type[LanguageParser]] = {}
    
    def register_parser(self, parser_class: Type[LanguageParser]) -> None:
        """Register a language parser.
        
        Args:
            parser_class: The language parser class to register.
        """
        parser_name = parser_class.__name__
        self._parsers[parser_name] = parser_class
        
        for ext in parser_class.get_supported_extensions():
            self._extension_map[ext] = parser_class
    
    def get_parser_for_file(self, file_path: Path) -> Optional[LanguageParser]:
        """Get the appropriate parser for a file.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            An instance of the appropriate language parser, or None if no parser is found.
        """
        ext = file_path.suffix.lower()
        parser_class = self._extension_map.get(ext)
        
        if parser_class:
            return parser_class()
        
        return None
    
    def get_parser_by_name(self, name: str) -> Optional[LanguageParser]:
        """Get a parser by name.
        
        Args:
            name: The name of the parser.
            
        Returns:
            An instance of the parser, or None if no parser with that name is found.
        """
        parser_class = self._parsers.get(name)
        
        if parser_class:
            return parser_class()
        
        return None
    
    def get_all_parsers(self) -> List[LanguageParser]:
        """Get all registered parsers.
        
        Returns:
            A list of instances of all registered parsers.
        """
        return [parser_class() for parser_class in self._parsers.values()]


# Global language registry
language_registry = LanguageRegistry()
