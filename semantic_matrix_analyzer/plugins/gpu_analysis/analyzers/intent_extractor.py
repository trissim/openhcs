"""
GPU-Accelerated Intent Extractor

This module provides a GPU-accelerated implementation of intent extraction from code.
It uses PyTorch to accelerate the extraction process and can run on both CPU and GPU.

Original code from the Brain project has been adapted for the Semantic Matrix Analyzer.
"""

import ast
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)


class IntentExtractor:
    """
    GPU-accelerated intent extractor.
    
    This class provides methods for extracting intent from code using GPU acceleration.
    It analyzes variable names, function signatures, comments, and documentation to
    understand what the code is meant to do.
    
    Attributes:
        device: Device to use for extraction ("cuda" or "cpu")
        config: Configuration for the extractor
    """
    
    def __init__(self, device: str = "cuda", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the intent extractor.
        
        Args:
            device: Device to use for extraction ("cuda" or "cpu")
            config: Configuration for the extractor
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.config = config or {}
        
        # Initialize pattern matcher
        from ..patterns import GPUPatternMatcher
        self.pattern_matcher = GPUPatternMatcher(device=self.device)
        
        # Initialize patterns
        self._init_patterns()
    
    def _init_patterns(self):
        """Initialize intent extraction patterns."""
        from ..patterns import create_regex_pattern, create_ast_pattern
        
        self.patterns = []
        
        # Docstring patterns
        self.patterns.append(create_regex_pattern(
            name="function_docstring",
            description="Function docstring",
            pattern=r'def\s+\w+\s*\([^)]*\)\s*:\s*\n\s*"""([^"]*)"""',
            weight=1.0
        ))
        
        self.patterns.append(create_regex_pattern(
            name="class_docstring",
            description="Class docstring",
            pattern=r'class\s+\w+\s*(?:\([^)]*\))?\s*:\s*\n\s*"""([^"]*)"""',
            weight=1.0
        ))
        
        # Comment patterns
        self.patterns.append(create_regex_pattern(
            name="line_comment",
            description="Line comment",
            pattern=r'#\s*(.*)',
            weight=0.5
        ))
        
        # Function name patterns
        self.patterns.append(create_regex_pattern(
            name="function_name",
            description="Function name",
            pattern=r'def\s+(\w+)\s*\(',
            weight=0.8
        ))
        
        # Variable name patterns
        self.patterns.append(create_regex_pattern(
            name="variable_assignment",
            description="Variable assignment",
            pattern=r'(\w+)\s*=\s*',
            weight=0.6
        ))
        
        # AST patterns
        self.patterns.append(create_ast_pattern(
            name="function_def",
            description="Function definition",
            node_type="FunctionDef",
            weight=0.8
        ))
        
        self.patterns.append(create_ast_pattern(
            name="class_def",
            description="Class definition",
            node_type="ClassDef",
            weight=0.8
        ))
    
    def extract_intent(
        self,
        code: str,
        file_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Extract intent from code.
        
        Args:
            code: Python code to analyze
            file_path: Optional path to the file
            
        Returns:
            Dictionary of intent information:
            - intent_score: Overall intent score
            - intent_sources: Sources of intent information
            - intent_summary: Summary of intent
        """
        # Convert file path to Path
        if file_path is not None:
            file_path = Path(file_path)
        else:
            file_path = Path("input.py")
        
        # Match patterns
        pattern_matches = self.pattern_matcher.match_patterns(
            self.patterns, file_path, code
        )
        
        # Extract intent from pattern matches
        intent_sources = {}
        
        for match in pattern_matches:
            pattern_name = match.pattern.name
            source_code = match.source_code
            
            if pattern_name not in intent_sources:
                intent_sources[pattern_name] = []
            
            intent_sources[pattern_name].append(source_code)
        
        # Extract docstrings using AST
        docstrings = self._extract_docstrings(code)
        if docstrings:
            intent_sources["docstrings"] = docstrings
        
        # Extract function and variable names using AST
        names = self._extract_names(code)
        if names:
            intent_sources["names"] = names
        
        # Compute intent score
        intent_score = self._compute_intent_score(intent_sources)
        
        # Generate intent summary
        intent_summary = self._generate_intent_summary(intent_sources)
        
        return {
            "intent_score": intent_score,
            "intent_sources": intent_sources,
            "intent_summary": intent_summary
        }
    
    def _extract_docstrings(self, code: str) -> List[str]:
        """
        Extract docstrings from code using AST.
        
        Args:
            code: Python code to analyze
            
        Returns:
            List of docstrings
        """
        docstrings = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                    docstring = ast.get_docstring(node)
                    if docstring:
                        docstrings.append(docstring)
        except Exception as e:
            logger.error(f"Error extracting docstrings: {e}")
        
        return docstrings
    
    def _extract_names(self, code: str) -> Dict[str, List[str]]:
        """
        Extract function and variable names from code using AST.
        
        Args:
            code: Python code to analyze
            
        Returns:
            Dictionary of name types to lists of names
        """
        names = {
            "function_names": [],
            "class_names": [],
            "variable_names": []
        }
        
        try:
            tree = ast.parse(code)
            
            # Extract function and class names
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    names["function_names"].append(node.name)
                elif isinstance(node, ast.ClassDef):
                    names["class_names"].append(node.name)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            names["variable_names"].append(target.id)
        except Exception as e:
            logger.error(f"Error extracting names: {e}")
        
        return names
    
    def _compute_intent_score(self, intent_sources: Dict[str, List[str]]) -> float:
        """
        Compute intent score from intent sources.
        
        Args:
            intent_sources: Sources of intent information
            
        Returns:
            Intent score (0.0 to 1.0)
        """
        # Simple scoring based on presence of intent sources
        score = 0.0
        
        if "function_docstring" in intent_sources:
            score += 0.3
        
        if "class_docstring" in intent_sources:
            score += 0.2
        
        if "line_comment" in intent_sources:
            score += 0.1
        
        if "docstrings" in intent_sources:
            score += 0.2
        
        if "names" in intent_sources:
            if intent_sources["names"].get("function_names"):
                score += 0.1
            if intent_sources["names"].get("class_names"):
                score += 0.05
            if intent_sources["names"].get("variable_names"):
                score += 0.05
        
        # Cap at 1.0
        return min(score, 1.0)
    
    def _generate_intent_summary(self, intent_sources: Dict[str, List[str]]) -> str:
        """
        Generate intent summary from intent sources.
        
        Args:
            intent_sources: Sources of intent information
            
        Returns:
            Intent summary
        """
        summary_parts = []
        
        # Extract from docstrings
        if "docstrings" in intent_sources:
            for docstring in intent_sources["docstrings"]:
                # Extract first line of docstring
                first_line = docstring.strip().split('\n')[0]
                if first_line:
                    summary_parts.append(first_line)
        
        # Extract from function docstrings
        if "function_docstring" in intent_sources:
            for docstring in intent_sources["function_docstring"]:
                # Extract first line of docstring
                match = re.search(r'"""([^"]*)"""', docstring)
                if match:
                    first_line = match.group(1).strip().split('\n')[0]
                    if first_line:
                        summary_parts.append(first_line)
        
        # Extract from class docstrings
        if "class_docstring" in intent_sources:
            for docstring in intent_sources["class_docstring"]:
                # Extract first line of docstring
                match = re.search(r'"""([^"]*)"""', docstring)
                if match:
                    first_line = match.group(1).strip().split('\n')[0]
                    if first_line:
                        summary_parts.append(first_line)
        
        # Extract from function names
        if "names" in intent_sources and "function_names" in intent_sources["names"]:
            for name in intent_sources["names"]["function_names"]:
                # Convert snake_case to words
                words = ' '.join(word.capitalize() for word in name.split('_'))
                if words:
                    summary_parts.append(f"Function to {words.lower()}")
        
        # Combine summary parts
        if summary_parts:
            return '. '.join(summary_parts)
        else:
            return "No clear intent found"
    
    def extract_intent_from_file(
        self,
        file_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Extract intent from a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary of intent information
        """
        # Convert file path to Path
        file_path = Path(file_path)
        
        # Read file
        with open(file_path, 'r') as f:
            code = f.read()
        
        # Extract intent
        return self.extract_intent(code, file_path)
    
    def batch_extract_intent(
        self,
        codes: List[str],
        file_paths: Optional[List[Union[str, Path]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract intent from multiple code snippets in batch.
        
        Args:
            codes: List of Python code snippets
            file_paths: Optional list of file paths
            
        Returns:
            List of intent information dictionaries
        """
        # Convert file paths to Paths
        if file_paths is not None:
            file_paths = [Path(path) for path in file_paths]
        else:
            file_paths = [Path(f"input_{i}.py") for i in range(len(codes))]
        
        results = []
        
        # Process each code snippet
        for code, path in zip(codes, file_paths):
            result = self.extract_intent(code, path)
            results.append(result)
        
        return results
