"""
Semantic Parsing Formalism Module

This module provides the ParseFormalism class for defining, compiling, and executing
semantic parsing rules. It is designed to integrate with the Semantic Matrix Analyzer (SMA)
project and provides GPU-accelerated alternatives to the semantic parsing components.

Original code from the Brain project has been adapted for the Semantic Matrix Analyzer.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .patterns import (
    Pattern, PatternMatch, PatternType,
    create_token_pattern, create_regex_pattern, create_ast_pattern
)

# Set up logging
logger = logging.getLogger(__name__)


class ParseFormalism:
    """
    A formalized semantic parsing rule.
    
    Parse formalisms define rules for identifying semantic patterns in code.
    They can be compiled to GPU-executable functions for efficient processing.
    
    Attributes:
        name: Name of the formalism
        description: Description of the formalism
        pattern_type: Type of pattern (token_sequence, ast_pattern, regex)
        rule_definition: Definition of the rule
        weight: Weight of this formalism in the overall intent score
        is_negative: If True, presence of this pattern reduces the intent score
        compiled_module: Compiled TorchScript module
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        pattern_type: str,
        rule_definition: Dict[str, Any],
        weight: float = 1.0,
        is_negative: bool = False,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a parse formalism.
        
        Args:
            name: Name of the formalism
            description: Description of the formalism
            pattern_type: Type of pattern (token_sequence, ast_pattern, regex)
            rule_definition: Definition of the rule
            weight: Weight of this formalism in the overall intent score
            is_negative: If True, presence of this pattern reduces the intent score
            config: Optional configuration for the formalism
        """
        self.name = name
        self.description = description
        self.pattern_type = pattern_type
        self.rule_definition = rule_definition
        self.weight = weight
        self.is_negative = is_negative
        self.config = config or {}
        
        # Create pattern
        self.pattern = self._create_pattern()
        
        # Compiled module
        self.compiled_module = None
    
    def _create_pattern(self) -> Pattern:
        """
        Create a pattern from the rule definition.
        
        Returns:
            Pattern object
        """
        if self.pattern_type == "token_sequence":
            return create_token_pattern(
                name=self.name,
                description=self.description,
                sequence=self.rule_definition["sequence"],
                weight=self.weight,
                is_negative=self.is_negative
            )
        elif self.pattern_type == "regex":
            return create_regex_pattern(
                name=self.name,
                description=self.description,
                pattern=self.rule_definition["pattern"],
                weight=self.weight,
                is_negative=self.is_negative
            )
        elif self.pattern_type == "ast_pattern":
            return create_ast_pattern(
                name=self.name,
                description=self.description,
                node_type=self.rule_definition["node_type"],
                condition=self.rule_definition.get("condition"),
                weight=self.weight,
                is_negative=self.is_negative
            )
        else:
            raise ValueError(f"Unsupported pattern type: {self.pattern_type}")
    
    def compile(self) -> torch.jit.ScriptModule:
        """
        Compile the formalism to a TorchScript module.
        
        Returns:
            Compiled TorchScript module
        """
        if self.compiled_module is not None:
            return self.compiled_module
        
        if self.pattern_type == "token_sequence":
            from .patterns.token_pattern import TokenPattern
            matcher = TokenPattern(device="cuda" if torch.cuda.is_available() else "cpu")
            self.compiled_module = matcher.compile(self.rule_definition["sequence"])
        elif self.pattern_type == "regex":
            from .patterns.regex_pattern import RegexPattern
            matcher = RegexPattern(device="cuda" if torch.cuda.is_available() else "cpu")
            self.compiled_module = matcher.compile(self.rule_definition["pattern"])
        elif self.pattern_type == "ast_pattern":
            # AST patterns are more complex and require a custom compilation process
            # For now, we'll just return a dummy module
            self.compiled_module = torch.jit.script(lambda x: torch.zeros(1, dtype=torch.bool))
        else:
            raise ValueError(f"Unsupported pattern type: {self.pattern_type}")
        
        return self.compiled_module
    
    def match(
        self,
        input_data: Union[str, torch.Tensor, List],
        file_path: Optional[Path] = None
    ) -> List[PatternMatch]:
        """
        Match the formalism against input data.
        
        Args:
            input_data: Input data to match against
            file_path: Optional path to the file
            
        Returns:
            List of pattern matches
        """
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
        
        # Match pattern
        if self.pattern_type == "token_sequence":
            from .patterns.token_pattern import TokenPattern
            matcher = TokenPattern(device="cuda" if torch.cuda.is_available() else "cpu")
            return matcher.match(self.pattern, path, file_content)
        elif self.pattern_type == "regex":
            from .patterns.regex_pattern import RegexPattern
            matcher = RegexPattern(device="cuda" if torch.cuda.is_available() else "cpu")
            return matcher.match(self.pattern, path, file_content)
        elif self.pattern_type == "ast_pattern":
            from .patterns.ast_pattern import ASTPattern
            from .ast_tensor import ASTTensorizer
            
            # Tensorize AST
            tensorizer = ASTTensorizer(device="cuda" if torch.cuda.is_available() else "cpu")
            ast_tensors = tensorizer.tensorize(file_content)
            
            # Match pattern
            matcher = ASTPattern(device="cuda" if torch.cuda.is_available() else "cpu")
            return matcher.match(self.pattern, path, file_content, ast_tensors)
        else:
            raise ValueError(f"Unsupported pattern type: {self.pattern_type}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the formalism to a dictionary.
        
        Returns:
            Dictionary representation of the formalism
        """
        return {
            "name": self.name,
            "description": self.description,
            "pattern_type": self.pattern_type,
            "rule_definition": self.rule_definition,
            "weight": self.weight,
            "is_negative": self.is_negative,
            "config": self.config
        }
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the formalism to a file.
        
        Args:
            path: Path to save the formalism to
        """
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ParseFormalism':
        """
        Load a formalism from a file.
        
        Args:
            path: Path to load the formalism from
            
        Returns:
            Loaded formalism
        """
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)
        
        return cls(
            name=data["name"],
            description=data["description"],
            pattern_type=data["pattern_type"],
            rule_definition=data["rule_definition"],
            weight=data["weight"],
            is_negative=data["is_negative"],
            config=data.get("config")
        )
