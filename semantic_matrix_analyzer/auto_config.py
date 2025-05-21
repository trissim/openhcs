"""
Automatic configuration generator for Semantic Matrix Analyzer.

This module provides functionality for automatically generating an initial
configuration based on codebase analysis, ensuring an unbiased starting point
for the Semantic Matrix Analyzer.
"""

import ast
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from semantic_matrix_analyzer.config_manager import ConfigManager


class AutoConfigGenerator:
    """
    Automatic configuration generator for Semantic Matrix Analyzer.
    
    This class analyzes a codebase to generate an initial configuration
    for the Semantic Matrix Analyzer, providing an unbiased starting point
    for analysis.
    """
    
    def __init__(self, project_dir: str):
        """
        Initialize the auto-config generator.
        
        Args:
            project_dir: Path to the project directory
        """
        self.project_dir = Path(project_dir)
        self.python_files = self._find_python_files()
        self.config_manager = ConfigManager()
        
        # Analysis results
        self.naming_patterns = {}
        self.common_tokens = {}
        self.intent_indicators = {}
        self.error_patterns = {}
        
    def _find_python_files(self) -> List[Path]:
        """
        Find all Python files in the project directory.
        
        Returns:
            List of paths to Python files
        """
        python_files = []
        
        for path in self.project_dir.rglob("*.py"):
            # Skip hidden directories and __pycache__
            if any(part.startswith(".") or part == "__pycache__" for part in path.parts):
                continue
            python_files.append(path)
            
        return python_files
    
    def generate_config(self) -> Dict[str, Any]:
        """
        Generate an initial configuration based on codebase analysis.
        
        Returns:
            Generated configuration dictionary
        """
        # Analyze the codebase
        self._analyze_naming_patterns()
        self._analyze_common_tokens()
        self._analyze_intent_indicators()
        self._analyze_error_patterns()
        
        # Get the default configuration
        config = self.config_manager.get_config()
        
        # Update weights based on analysis
        self._update_weights(config)
        
        # Update patterns based on analysis
        self._update_patterns(config)
        
        # Update tokens based on analysis
        self._update_tokens(config)
        
        # Update keys based on analysis
        self._update_keys(config)
        
        return config
    
    def _analyze_naming_patterns(self) -> None:
        """Analyze naming patterns in the codebase."""
        class_names = []
        function_names = []
        variable_names = []
        constant_names = []
        
        for file_path in self.python_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Parse the AST
                tree = ast.parse(content)
                
                # Extract names
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        class_names.append(node.name)
                    elif isinstance(node, ast.FunctionDef):
                        function_names.append(node.name)
                    elif isinstance(node, ast.Name):
                        if isinstance(node.ctx, ast.Store):
                            # Check if it's a constant (all uppercase)
                            if node.id.isupper():
                                constant_names.append(node.id)
                            else:
                                variable_names.append(node.id)
            except Exception:
                # Skip files that can't be parsed
                continue
        
        # Analyze naming patterns
        self.naming_patterns = {
            "class": self._extract_pattern(class_names),
            "function": self._extract_pattern(function_names),
            "variable": self._extract_pattern(variable_names),
            "constant": self._extract_pattern(constant_names)
        }
    
    def _extract_pattern(self, names: List[str]) -> str:
        """
        Extract a regex pattern from a list of names.
        
        Args:
            names: List of names to analyze
            
        Returns:
            Regex pattern as string
        """
        if not names:
            return ""
        
        # Check for common patterns
        starts_with_uppercase = sum(1 for name in names if name and name[0].isupper()) / len(names)
        starts_with_lowercase = sum(1 for name in names if name and name[0].islower()) / len(names)
        is_snake_case = sum(1 for name in names if "_" in name) / len(names)
        is_camel_case = sum(1 for name in names if not "_" in name and any(c.isupper() for c in name[1:])) / len(names)
        is_all_uppercase = sum(1 for name in names if name.isupper()) / len(names)
        
        # Determine the dominant pattern
        if is_all_uppercase > 0.7:
            return r"^[A-Z][A-Z0-9_]*$"
        elif starts_with_uppercase > 0.7:
            if is_camel_case > 0.7:
                return r"^[A-Z][a-zA-Z0-9]*$"
            else:
                return r"^[A-Z][a-z0-9]*(?:_[A-Z][a-z0-9]*)*$"
        elif starts_with_lowercase > 0.7:
            if is_snake_case > 0.7:
                return r"^[a-z][a-z0-9_]*$"
            elif is_camel_case > 0.7:
                return r"^[a-z][a-zA-Z0-9]*$"
            else:
                return r"^[a-z][a-z0-9]*$"
        
        # Default pattern
        return r"^[a-zA-Z][a-zA-Z0-9_]*$"
    
    def _analyze_common_tokens(self) -> None:
        """Analyze common tokens in the codebase."""
        # Tokens to look for
        docstring_markers = Counter()
        special_markers = Counter()
        
        for file_path in self.python_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Look for docstring markers
                docstring_pattern = r"@(\w+)"
                docstring_markers.update(re.findall(docstring_pattern, content))
                
                # Look for special markers
                special_pattern = r"(TODO|FIXME|NOTE|HACK|XXX|BUG|OPTIMIZE):"
                special_markers.update(re.findall(special_pattern, content))
            except Exception:
                # Skip files that can't be read
                continue
        
        # Store the most common tokens
        self.common_tokens = {
            "docstring_tags": [tag for tag, count in docstring_markers.most_common(10) if count > 1],
            "special_markers": [marker for marker, count in special_markers.most_common(10) if count > 1]
        }
    
    def _analyze_intent_indicators(self) -> None:
        """Analyze intent indicators in the codebase."""
        intent_words = Counter()
        
        for file_path in self.python_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Parse the AST
                tree = ast.parse(content)
                
                # Extract docstrings
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                        docstring = ast.get_docstring(node)
                        if docstring:
                            # Look for intent indicators
                            words = re.findall(r"\b(\w+)\b", docstring.lower())
                            intent_words.update(words)
            except Exception:
                # Skip files that can't be parsed
                continue
        
        # Filter common English words
        common_words = {"the", "a", "an", "and", "or", "but", "if", "then", "else", "when", "to", "of", "in", "on", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "from", "up", "down", "out", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can", "will", "just", "should", "now"}
        
        # Store the most common intent indicators
        self.intent_indicators = {
            "purpose_indicators": [word for word, count in intent_words.most_common(50) 
                                  if count > 2 and word not in common_words and len(word) > 3],
            "action_indicators": [word for word, count in intent_words.most_common(100) 
                                 if word.endswith(("ing", "ed", "es", "s")) and count > 2 and word not in common_words]
        }
    
    def _analyze_error_patterns(self) -> None:
        """Analyze error patterns in the codebase."""
        error_types = Counter()
        error_messages = Counter()
        
        for file_path in self.python_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Look for exception handling
                try_except_pattern = r"except\s+(\w+(?:\s*,\s*\w+)*)(?:\s+as\s+\w+)?:"
                for match in re.finditer(try_except_pattern, content):
                    exceptions = re.split(r"\s*,\s*", match.group(1))
                    error_types.update(exceptions)
                
                # Look for error messages
                error_msg_pattern = r"(?:raise|throw)\s+\w+\s*\(\s*[\"']([^\"']+)[\"']"
                error_messages.update(re.findall(error_msg_pattern, content))
            except Exception:
                # Skip files that can't be read
                continue
        
        # Store the most common error patterns
        self.error_patterns = {
            "error_types": [error for error, count in error_types.most_common(10) if count > 1],
            "error_messages": [msg for msg, count in error_messages.most_common(10) if count > 1]
        }
    
    def _update_weights(self, config: Dict[str, Any]) -> None:
        """
        Update weights in the configuration based on analysis.
        
        Args:
            config: Configuration dictionary to update
        """
        # Calculate weights based on codebase characteristics
        weights = config["analysis"]["weights"]
        
        # Adjust weights based on naming pattern consistency
        naming_consistency = sum(1 for pattern in self.naming_patterns.values() if pattern) / len(self.naming_patterns)
        weights["name_similarity"] = max(0.5, min(0.9, naming_consistency))
        
        # Adjust weights based on docstring presence
        weights["docstring_relevance"] = 0.7 if self.common_tokens.get("docstring_tags") else 0.4
        
        # Adjust weights based on error handling
        weights["error_detection"] = 0.8 if self.error_patterns.get("error_types") else 0.5
    
    def _update_patterns(self, config: Dict[str, Any]) -> None:
        """
        Update patterns in the configuration based on analysis.
        
        Args:
            config: Configuration dictionary to update
        """
        patterns = config["analysis"]["patterns"]["naming_conventions"]
        
        # Update naming patterns
        for name_type, pattern in self.naming_patterns.items():
            if pattern:
                patterns[name_type] = pattern
    
    def _update_tokens(self, config: Dict[str, Any]) -> None:
        """
        Update tokens in the configuration based on analysis.
        
        Args:
            config: Configuration dictionary to update
        """
        tokens = config["analysis"]["tokens"]
        
        # Update docstring tags
        if self.common_tokens.get("docstring_tags"):
            tokens["docstring_tags"] = self.common_tokens["docstring_tags"]
        
        # Update special markers
        if self.common_tokens.get("special_markers"):
            tokens["special_markers"] = self.common_tokens["special_markers"]
    
    def _update_keys(self, config: Dict[str, Any]) -> None:
        """
        Update keys in the configuration based on analysis.
        
        Args:
            config: Configuration dictionary to update
        """
        keys = config["analysis"]["keys"]
        
        # Update intent indicators
        if self.intent_indicators.get("purpose_indicators"):
            keys["intent_indicators"] = self.intent_indicators["purpose_indicators"][:10]
        
        # Update error indicators
        if self.error_patterns.get("error_types"):
            keys["error_indicators"] = self.error_patterns["error_types"]
    
    def save_config(self) -> None:
        """Generate and save the configuration."""
        config = self.generate_config()
        self.config_manager.save_config(config)


def generate_initial_config(project_dir: str) -> Dict[str, Any]:
    """
    Generate an initial configuration for a project if none exists.
    
    Args:
        project_dir: Path to the project directory
        
    Returns:
        Generated configuration dictionary
    """
    config_manager = ConfigManager()
    
    # Check if configuration already exists
    if Path(config_manager.config_path).exists():
        return config_manager.get_config()
    
    # Generate initial configuration
    auto_config = AutoConfigGenerator(project_dir)
    config = auto_config.generate_config()
    
    # Save the configuration
    config_manager.save_config(config)
    
    return config
