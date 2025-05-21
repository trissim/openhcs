"""
Suggestion verification module for AST verification.

This module provides functionality for verifying code suggestions against the AST,
ensuring syntactic and semantic correctness.
"""

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class CodeSuggestion:
    """A suggestion for changing code."""
    
    file_path: Path
    start_line: int
    end_line: int
    original_code: str
    suggested_code: str
    description: str
    confidence: float = 0.0
    verification_result: Optional['VerificationResult'] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "file_path": str(self.file_path),
            "start_line": self.start_line,
            "end_line": self.end_line,
            "original_code": self.original_code,
            "suggested_code": self.suggested_code,
            "description": self.description,
            "confidence": self.confidence,
            "verification_result": self.verification_result.to_dict() if self.verification_result else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeSuggestion':
        """Create from dictionary after deserialization."""
        suggestion = cls(
            file_path=Path(data["file_path"]),
            start_line=data["start_line"],
            end_line=data["end_line"],
            original_code=data["original_code"],
            suggested_code=data["suggested_code"],
            description=data["description"],
            confidence=data["confidence"]
        )
        
        if data.get("verification_result"):
            suggestion.verification_result = VerificationResult.from_dict(data["verification_result"])
        
        return suggestion


@dataclass
class VerificationResult:
    """The result of verifying a suggestion."""
    
    is_valid: bool
    syntax_valid: bool
    semantic_valid: bool
    side_effects: List[str]
    confidence: float
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "is_valid": self.is_valid,
            "syntax_valid": self.syntax_valid,
            "semantic_valid": self.semantic_valid,
            "side_effects": self.side_effects,
            "confidence": self.confidence,
            "error_message": self.error_message
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VerificationResult':
        """Create from dictionary after deserialization."""
        return cls(
            is_valid=data["is_valid"],
            syntax_valid=data["syntax_valid"],
            semantic_valid=data["semantic_valid"],
            side_effects=data["side_effects"],
            confidence=data["confidence"],
            error_message=data.get("error_message")
        )


class SuggestionVerifier:
    """Verifies code suggestions against the AST."""
    
    def __init__(self, side_effect_detector=None):
        """Initialize the suggestion verifier.
        
        Args:
            side_effect_detector: The side effect detector to use (optional).
        """
        self.side_effect_detector = side_effect_detector
    
    def verify_suggestion(self, suggestion: CodeSuggestion) -> VerificationResult:
        """Verify a code suggestion against the AST.
        
        Args:
            suggestion: The code suggestion to verify.
            
        Returns:
            The verification result.
        """
        try:
            # Check syntax validity
            syntax_valid = self._check_syntax(suggestion.suggested_code)
            
            if not syntax_valid:
                return VerificationResult(
                    is_valid=False,
                    syntax_valid=False,
                    semantic_valid=False,
                    side_effects=[],
                    confidence=0.0,
                    error_message="Syntax error in suggested code"
                )
            
            # Check semantic validity
            semantic_valid, semantic_errors = self._check_semantics(suggestion)
            
            if not semantic_valid:
                return VerificationResult(
                    is_valid=False,
                    syntax_valid=True,
                    semantic_valid=False,
                    side_effects=[],
                    confidence=0.0,
                    error_message=f"Semantic error in suggested code: {', '.join(semantic_errors)}"
                )
            
            # Check for side effects
            side_effects = self._check_side_effects(suggestion)
            
            # Calculate confidence
            confidence = self._calculate_confidence(suggestion, syntax_valid, semantic_valid, side_effects)
            
            return VerificationResult(
                is_valid=True,
                syntax_valid=True,
                semantic_valid=True,
                side_effects=side_effects,
                confidence=confidence
            )
        except Exception as e:
            logger.error(f"Error verifying suggestion: {e}")
            return VerificationResult(
                is_valid=False,
                syntax_valid=False,
                semantic_valid=False,
                side_effects=[],
                confidence=0.0,
                error_message=f"Error verifying suggestion: {str(e)}"
            )
    
    def _check_syntax(self, code: str) -> bool:
        """Check if the code is syntactically valid.
        
        Args:
            code: The code to check.
            
        Returns:
            True if the code is syntactically valid, False otherwise.
        """
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    def _check_semantics(self, suggestion: CodeSuggestion) -> Tuple[bool, List[str]]:
        """Check if the code is semantically valid.
        
        Args:
            suggestion: The code suggestion to check.
            
        Returns:
            A tuple of (is_valid, errors), where is_valid is True if the code is
            semantically valid, and errors is a list of error messages.
        """
        # This is a placeholder implementation
        # In a real implementation, this would perform more sophisticated semantic checks
        
        errors = []
        
        # Parse the original and suggested code
        try:
            original_ast = ast.parse(suggestion.original_code)
            suggested_ast = ast.parse(suggestion.suggested_code)
            
            # Check for undefined variables
            original_names = self._extract_defined_names(original_ast)
            suggested_names = self._extract_used_names(suggested_ast)
            
            undefined_names = suggested_names - original_names
            if undefined_names:
                errors.append(f"Undefined names: {', '.join(undefined_names)}")
            
            return len(errors) == 0, errors
        except Exception as e:
            errors.append(f"Error checking semantics: {str(e)}")
            return False, errors
    
    def _extract_defined_names(self, tree: ast.AST) -> Set[str]:
        """Extract defined names from an AST.
        
        Args:
            tree: The AST to extract names from.
            
        Returns:
            A set of defined names.
        """
        names = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
            elif isinstance(node, ast.FunctionDef):
                names.add(node.name)
            elif isinstance(node, ast.ClassDef):
                names.add(node.name)
            elif isinstance(node, ast.Import):
                for name in node.names:
                    names.add(name.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                for name in node.names:
                    names.add(name.name)
        
        # Add built-in names
        names.update(dir(__builtins__))
        
        return names
    
    def _extract_used_names(self, tree: ast.AST) -> Set[str]:
        """Extract used names from an AST.
        
        Args:
            tree: The AST to extract names from.
            
        Returns:
            A set of used names.
        """
        names = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                names.add(node.id)
        
        return names
    
    def _check_side_effects(self, suggestion: CodeSuggestion) -> List[str]:
        """Check for potential side effects of the suggestion.
        
        Args:
            suggestion: The code suggestion to check.
            
        Returns:
            A list of potential side effects.
        """
        if self.side_effect_detector:
            return self.side_effect_detector.detect_side_effects(suggestion)
        
        # This is a placeholder implementation
        # In a real implementation, this would use the side effect detector
        
        side_effects = []
        
        # Parse the original and suggested code
        try:
            original_ast = ast.parse(suggestion.original_code)
            suggested_ast = ast.parse(suggestion.suggested_code)
            
            # Check for changes to function signatures
            original_functions = self._extract_function_signatures(original_ast)
            suggested_functions = self._extract_function_signatures(suggested_ast)
            
            for name, sig in suggested_functions.items():
                if name in original_functions and sig != original_functions[name]:
                    side_effects.append(f"Changed function signature: {name}")
            
            # Check for changes to class interfaces
            original_classes = self._extract_class_interfaces(original_ast)
            suggested_classes = self._extract_class_interfaces(suggested_ast)
            
            for name, interface in suggested_classes.items():
                if name in original_classes and interface != original_classes[name]:
                    side_effects.append(f"Changed class interface: {name}")
            
            return side_effects
        except Exception as e:
            logger.error(f"Error checking side effects: {e}")
            return []
    
    def _extract_function_signatures(self, tree: ast.AST) -> Dict[str, Tuple[List[str], List[str]]]:
        """Extract function signatures from an AST.
        
        Args:
            tree: The AST to extract function signatures from.
            
        Returns:
            A dictionary mapping function names to (args, defaults).
        """
        signatures = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                args = [arg.arg for arg in node.args.args]
                defaults = [ast.dump(default) for default in node.args.defaults]
                signatures[node.name] = (args, defaults)
        
        return signatures
    
    def _extract_class_interfaces(self, tree: ast.AST) -> Dict[str, Set[str]]:
        """Extract class interfaces from an AST.
        
        Args:
            tree: The AST to extract class interfaces from.
            
        Returns:
            A dictionary mapping class names to sets of method names.
        """
        interfaces = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = set()
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        methods.add(child.name)
                interfaces[node.name] = methods
        
        return interfaces
    
    def _calculate_confidence(
        self,
        suggestion: CodeSuggestion,
        syntax_valid: bool,
        semantic_valid: bool,
        side_effects: List[str]
    ) -> float:
        """Calculate the confidence score for the suggestion.
        
        Args:
            suggestion: The code suggestion.
            syntax_valid: Whether the suggestion is syntactically valid.
            semantic_valid: Whether the suggestion is semantically valid.
            side_effects: A list of potential side effects.
            
        Returns:
            The confidence score (0.0 to 1.0).
        """
        # Start with the suggestion's initial confidence
        confidence = suggestion.confidence
        
        # Adjust based on verification results
        if not syntax_valid:
            confidence *= 0.1
        if not semantic_valid:
            confidence *= 0.2
        
        # Reduce confidence based on side effects
        confidence *= max(0.1, 1.0 - (len(side_effects) * 0.1))
        
        return confidence
