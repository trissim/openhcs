"""
Evidence module for semantic grounding.

This module provides functionality for collecting and managing evidence from the codebase
to support AI recommendations and findings.
"""

import ast
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class Evidence:
    """Evidence from the codebase to support a recommendation or finding."""
    
    id: str
    type: str  # "ast_node", "code_snippet", "pattern_match", etc.
    file_path: Path
    line_start: int
    line_end: int
    code_snippet: str
    ast_node_type: Optional[str] = None
    ast_node_dump: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "id": self.id,
            "type": self.type,
            "file_path": str(self.file_path),
            "line_start": self.line_start,
            "line_end": self.line_end,
            "code_snippet": self.code_snippet,
            "confidence": self.confidence,
            "metadata": self.metadata
        }
        
        if self.ast_node_type:
            result["ast_node_type"] = self.ast_node_type
        
        if self.ast_node_dump:
            result["ast_node_dump"] = self.ast_node_dump
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Evidence':
        """Create from dictionary after deserialization."""
        return cls(
            id=data["id"],
            type=data["type"],
            file_path=Path(data["file_path"]),
            line_start=data["line_start"],
            line_end=data["line_end"],
            code_snippet=data["code_snippet"],
            ast_node_type=data.get("ast_node_type"),
            ast_node_dump=data.get("ast_node_dump"),
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {})
        )
    
    def get_location_str(self) -> str:
        """Get a string representation of the evidence location."""
        return f"{self.file_path}:{self.line_start}-{self.line_end}"
    
    def get_summary(self) -> str:
        """Get a summary of the evidence."""
        return f"{self.type} evidence in {self.get_location_str()}: {self.code_snippet[:50]}..."


class EvidenceCollector:
    """Collects evidence from the codebase."""
    
    def __init__(self):
        """Initialize the evidence collector."""
        self.evidence_cache: Dict[str, Evidence] = {}
    
    def collect_evidence_for_file(self, file_path: Path) -> List[Evidence]:
        """Collect evidence from a file.
        
        Args:
            file_path: The path to the file.
            
        Returns:
            A list of evidence items.
        """
        evidence_list = []
        
        try:
            # Read the file
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()
            
            # Parse the file
            tree = ast.parse(code, filename=str(file_path))
            
            # Collect AST node evidence
            ast_evidence = self._collect_ast_node_evidence(tree, file_path, code)
            evidence_list.extend(ast_evidence)
            
            # Collect code pattern evidence
            pattern_evidence = self._collect_code_pattern_evidence(file_path, code)
            evidence_list.extend(pattern_evidence)
            
            # Cache the evidence
            for evidence in evidence_list:
                self.evidence_cache[evidence.id] = evidence
            
            return evidence_list
        except Exception as e:
            logger.error(f"Error collecting evidence from {file_path}: {e}")
            return []
    
    def _collect_ast_node_evidence(self, tree: ast.AST, file_path: Path, code: str) -> List[Evidence]:
        """Collect evidence from AST nodes.
        
        Args:
            tree: The AST.
            file_path: The path to the file.
            code: The file content.
            
        Returns:
            A list of evidence items.
        """
        evidence_list = []
        
        # Get the lines of code
        lines = code.splitlines()
        
        # Collect evidence for each node
        for node in ast.walk(tree):
            if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                # Get the node type
                node_type = type(node).__name__
                
                # Get the node's code snippet
                line_start = node.lineno
                line_end = getattr(node, "end_lineno", node.lineno)
                
                # Adjust for 0-based indexing
                snippet_lines = lines[line_start - 1:line_end]
                code_snippet = "\n".join(snippet_lines)
                
                # Create an evidence item
                evidence = Evidence(
                    id=f"ast:{file_path}:{line_start}-{line_end}:{node_type}",
                    type="ast_node",
                    file_path=file_path,
                    line_start=line_start,
                    line_end=line_end,
                    code_snippet=code_snippet,
                    ast_node_type=node_type,
                    ast_node_dump=ast.dump(node),
                    metadata={
                        "node_fields": {name: getattr(node, name) for name in node._fields if hasattr(node, name) and not isinstance(getattr(node, name), ast.AST)}
                    }
                )
                
                evidence_list.append(evidence)
        
        return evidence_list
    
    def _collect_code_pattern_evidence(self, file_path: Path, code: str) -> List[Evidence]:
        """Collect evidence from code patterns.
        
        Args:
            file_path: The path to the file.
            code: The file content.
            
        Returns:
            A list of evidence items.
        """
        evidence_list = []
        
        # Get the lines of code
        lines = code.splitlines()
        
        # Define patterns to look for
        patterns = [
            (r"def\s+\w+\s*\(", "function_definition"),
            (r"class\s+\w+\s*(\(.*\))?:", "class_definition"),
            (r"import\s+\w+", "import_statement"),
            (r"from\s+\w+\s+import", "from_import_statement"),
            (r"try\s*:", "try_except_block"),
            (r"if\s+.*:", "if_statement"),
            (r"for\s+.*:", "for_loop"),
            (r"while\s+.*:", "while_loop"),
            (r"with\s+.*:", "with_statement"),
            (r"raise\s+\w+", "raise_statement"),
            (r"return\s+", "return_statement"),
            (r"yield\s+", "yield_statement"),
            (r"assert\s+", "assert_statement"),
            (r"@\w+", "decorator"),
            (r"#.*", "comment"),
            (r"\"\"\".*\"\"\"", "docstring"),
            (r"\'\'\'.*\'\'\'", "docstring")
        ]
        
        # This is a placeholder implementation
        # In a real implementation, we would use more sophisticated pattern matching
        
        return evidence_list
    
    def get_evidence(self, evidence_id: str) -> Optional[Evidence]:
        """Get evidence by ID.
        
        Args:
            evidence_id: The evidence ID.
            
        Returns:
            The evidence, or None if not found.
        """
        return self.evidence_cache.get(evidence_id)
    
    def search_evidence(self, query: str) -> List[Evidence]:
        """Search for evidence.
        
        Args:
            query: The search query.
            
        Returns:
            A list of matching evidence items.
        """
        # This is a placeholder implementation
        # In a real implementation, we would use more sophisticated search
        
        results = []
        
        for evidence in self.evidence_cache.values():
            if query.lower() in evidence.code_snippet.lower():
                results.append(evidence)
        
        return results
