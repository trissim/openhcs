"""
Parsing Executor Module

This module provides the ParsingExecutor class for executing semantic parsing rules
in parallel. It is designed to integrate with the Semantic Matrix Analyzer (SMA)
project and provides GPU-accelerated alternatives to the semantic parsing components.

Original code from the Brain project has been adapted for the Semantic Matrix Analyzer.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .formalism import ParseFormalism
from .patterns import PatternMatch

# Set up logging
logger = logging.getLogger(__name__)


class ParsingExecutor:
    """
    Executor for semantic parsing rules.
    
    This class provides methods for executing semantic parsing rules in parallel.
    It supports different types of patterns (token, regex, AST) and can execute
    multiple rules in parallel.
    
    Attributes:
        device: Device to use for execution ("cuda" or "cpu")
        formalisms: Dictionary of formalisms by name
        config: Configuration for the executor
    """
    
    def __init__(self, device: str = "cuda", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the parsing executor.
        
        Args:
            device: Device to use for execution ("cuda" or "cpu")
            config: Configuration for the executor
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.formalisms = {}
        self.config = config or {}
    
    def add_formalism(self, formalism: ParseFormalism) -> None:
        """
        Add a formalism to the executor.
        
        Args:
            formalism: Formalism to add
        """
        self.formalisms[formalism.name] = formalism
    
    def remove_formalism(self, name: str) -> None:
        """
        Remove a formalism from the executor.
        
        Args:
            name: Name of the formalism to remove
        """
        if name in self.formalisms:
            del self.formalisms[name]
    
    def get_formalism(self, name: str) -> Optional[ParseFormalism]:
        """
        Get a formalism by name.
        
        Args:
            name: Name of the formalism
            
        Returns:
            Formalism or None if not found
        """
        return self.formalisms.get(name)
    
    def execute(
        self,
        input_data: Union[str, torch.Tensor, List],
        formalism_names: Optional[List[str]] = None,
        file_path: Optional[Path] = None
    ) -> Dict[str, List[PatternMatch]]:
        """
        Execute formalisms against input data.
        
        Args:
            input_data: Input data to execute against
            formalism_names: Names of formalisms to execute (if None, execute all)
            file_path: Optional path to the file
            
        Returns:
            Dictionary of formalism name to list of pattern matches
        """
        # Determine which formalisms to execute
        if formalism_names is None:
            formalisms = list(self.formalisms.values())
        else:
            formalisms = [self.formalisms[name] for name in formalism_names if name in self.formalisms]
        
        # Execute each formalism
        results = {}
        for formalism in formalisms:
            try:
                matches = formalism.match(input_data, file_path)
                results[formalism.name] = matches
            except Exception as e:
                logger.error(f"Error executing formalism {formalism.name}: {e}")
                results[formalism.name] = []
        
        return results
    
    def execute_parallel(
        self,
        input_data: Union[str, torch.Tensor, List],
        formalism_names: Optional[List[str]] = None,
        file_path: Optional[Path] = None
    ) -> Dict[str, List[PatternMatch]]:
        """
        Execute formalisms against input data in parallel.
        
        Args:
            input_data: Input data to execute against
            formalism_names: Names of formalisms to execute (if None, execute all)
            file_path: Optional path to the file
            
        Returns:
            Dictionary of formalism name to list of pattern matches
        """
        # Determine which formalisms to execute
        if formalism_names is None:
            formalisms = list(self.formalisms.values())
        else:
            formalisms = [self.formalisms[name] for name in formalism_names if name in self.formalisms]
        
        # Group formalisms by pattern type for batch processing
        token_formalisms = []
        regex_formalisms = []
        ast_formalisms = []
        
        for formalism in formalisms:
            if formalism.pattern_type == "token_sequence":
                token_formalisms.append(formalism)
            elif formalism.pattern_type == "regex":
                regex_formalisms.append(formalism)
            elif formalism.pattern_type == "ast_pattern":
                ast_formalisms.append(formalism)
        
        # Convert input data to appropriate format
        if isinstance(input_data, str):
            file_content = input_data
        elif isinstance(input_data, torch.Tensor):
            # Convert tensor to string
            if input_data.dtype == torch.uint8 or input_data.dtype == torch.int8:
                file_content = input_data.cpu().numpy().tobytes().decode('utf-8', errors='ignore')
            else:
                file_content = "".join(chr(int(c)) for c in input_data.cpu().numpy())
        elif isinstance(input_data, list):
            # Convert list to string
            file_content = "".join(chr(c) if isinstance(c, int) else c for c in input_data)
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
        
        # Use file path if provided, otherwise use a dummy path
        path = file_path or Path("input.txt")
        
        # Execute formalisms in parallel
        results = {}
        
        # Process token formalisms
        if token_formalisms:
            from .patterns.token_pattern import TokenPattern
            matcher = TokenPattern(device=self.device)
            
            # Extract patterns
            patterns = [formalism.pattern for formalism in token_formalisms]
            
            # Match patterns
            matches = matcher.match_batch(patterns, path, file_content)
            
            # Group matches by formalism
            for formalism, pattern in zip(token_formalisms, patterns):
                formalism_matches = [m for m in matches if m.pattern == pattern]
                results[formalism.name] = formalism_matches
        
        # Process regex formalisms
        if regex_formalisms:
            from .patterns.regex_pattern import RegexPattern
            matcher = RegexPattern(device=self.device)
            
            # Extract patterns
            patterns = [formalism.pattern for formalism in regex_formalisms]
            
            # Match patterns
            matches = matcher.match_batch(patterns, path, file_content)
            
            # Group matches by formalism
            for formalism, pattern in zip(regex_formalisms, patterns):
                formalism_matches = [m for m in matches if m.pattern == pattern]
                results[formalism.name] = formalism_matches
        
        # Process AST formalisms
        if ast_formalisms:
            from .patterns.ast_pattern import ASTPattern
            from .ast_tensor import ASTTensorizer
            
            # Tensorize AST
            tensorizer = ASTTensorizer(device=self.device)
            ast_tensors = tensorizer.tensorize(file_content)
            
            # Create matcher
            matcher = ASTPattern(device=self.device)
            
            # Extract patterns
            patterns = [formalism.pattern for formalism in ast_formalisms]
            
            # Match patterns
            matches = matcher.match_batch(patterns, path, file_content, ast_tensors)
            
            # Group matches by formalism
            for formalism, pattern in zip(ast_formalisms, patterns):
                formalism_matches = [m for m in matches if m.pattern == pattern]
                results[formalism.name] = formalism_matches
        
        return results
    
    def load_formalisms(self, directory: Union[str, Path]) -> None:
        """
        Load formalisms from a directory.
        
        Args:
            directory: Directory to load formalisms from
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Load all JSON files in the directory
        for file_path in directory.glob("*.json"):
            try:
                formalism = ParseFormalism.load(file_path)
                self.add_formalism(formalism)
                logger.info(f"Loaded formalism {formalism.name} from {file_path}")
            except Exception as e:
                logger.error(f"Error loading formalism from {file_path}: {e}")
    
    def save_formalisms(self, directory: Union[str, Path]) -> None:
        """
        Save formalisms to a directory.
        
        Args:
            directory: Directory to save formalisms to
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Save each formalism to a JSON file
        for name, formalism in self.formalisms.items():
            file_path = directory / f"{name}.json"
            try:
                formalism.save(file_path)
                logger.info(f"Saved formalism {name} to {file_path}")
            except Exception as e:
                logger.error(f"Error saving formalism {name} to {file_path}: {e}")
    
    def compile_formalisms(self) -> None:
        """Compile all formalisms to TorchScript modules."""
        for name, formalism in self.formalisms.items():
            try:
                formalism.compile()
                logger.info(f"Compiled formalism {name}")
            except Exception as e:
                logger.error(f"Error compiling formalism {name}: {e}")
    
    def clear_formalisms(self) -> None:
        """Clear all formalisms."""
        self.formalisms = {}
