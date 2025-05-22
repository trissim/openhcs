# plan_03_type_analysis.md
## Component: Type Hint Analysis

### Objective
Create components for analyzing type hints to extract intent. These components will extract type hints from AST, analyze parameter and return types, and infer intent from type information.

### Plan
1. Create a `TypeHintExtractor` class for extracting type hints from AST
2. Create a `TypeHintAnalyzer` class for analyzing parameter and return types
3. Implement type-to-intent mapping for common types
4. Create utilities for handling complex type annotations (Optional, Union, etc.)
5. Implement inference of intent from type relationships

### Findings
Type hints provide valuable information about the expected inputs and outputs of functions, as well as the relationships between different parts of the code. Different types of hints (basic types, Optional, Union, custom types, etc.) need to be handled differently.

Key patterns to recognize:
- Optional types indicate that a value might be missing
- Collection types indicate that multiple items are processed
- Custom types often represent domain concepts
- Return types indicate what a function promises to deliver
- Parameter types indicate what a function expects

### Implementation Draft

```python
"""
Type hint analysis components for extracting intent from type hints.
"""

import ast
import inspect
import logging
import re
import typing
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, get_type_hints

from semantic_matrix_analyzer.intent.models.intent import (
    Intent, TypeIntent, IntentType, CodeLocation
)

logger = logging.getLogger(__name__)


class TypeHintExtractor:
    """Extracts type hints from AST."""
    
    def __init__(self):
        """Initialize the type hint extractor."""
        pass
    
    def extract_type_hints_from_ast(self, node: ast.AST) -> Dict[str, str]:
        """Extract type hints from an AST node.
        
        Args:
            node: The AST node to extract type hints from.
            
        Returns:
            A dictionary mapping parameter names to type hint strings.
        """
        type_hints = {}
        
        if isinstance(node, ast.FunctionDef):
            # Extract parameter type hints
            for arg in node.args.args:
                if arg.annotation:
                    type_hints[arg.arg] = self._annotation_to_string(arg.annotation)
            
            # Extract return type hint
            if node.returns:
                type_hints["return"] = self._annotation_to_string(node.returns)
        
        return type_hints
    
    def _annotation_to_string(self, annotation: ast.AST) -> str:
        """Convert an annotation AST node to a string.
        
        Args:
            annotation: The annotation AST node.
            
        Returns:
            The annotation as a string.
        """
        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Attribute):
            return f"{self._annotation_to_string(annotation.value)}.{annotation.attr}"
        elif isinstance(annotation, ast.Subscript):
            value = self._annotation_to_string(annotation.value)
            slice_value = self._annotation_to_string(annotation.slice)
            return f"{value}[{slice_value}]"
        elif isinstance(annotation, ast.Index):
            return self._annotation_to_string(annotation.value)
        elif isinstance(annotation, ast.Tuple):
            elts = [self._annotation_to_string(elt) for elt in annotation.elts]
            return f"Tuple[{', '.join(elts)}]"
        elif isinstance(annotation, ast.List):
            elts = [self._annotation_to_string(elt) for elt in annotation.elts]
            return f"List[{', '.join(elts)}]"
        elif isinstance(annotation, ast.Dict):
            keys = [self._annotation_to_string(key) for key in annotation.keys]
            values = [self._annotation_to_string(value) for value in annotation.values]
            return f"Dict[{', '.join(keys)}, {', '.join(values)}]"
        elif isinstance(annotation, ast.Constant):
            return str(annotation.value)
        elif hasattr(ast, "Str") and isinstance(annotation, ast.Str):
            return annotation.s
        elif hasattr(ast, "Num") and isinstance(annotation, ast.Num):
            return str(annotation.n)
        elif hasattr(ast, "NameConstant") and isinstance(annotation, ast.NameConstant):
            return str(annotation.value)
        elif hasattr(ast, "Ellipsis") and isinstance(annotation, ast.Ellipsis):
            return "..."
        else:
            return ast.dump(annotation)


class TypeHintAnalyzer:
    """Analyzes type hints to extract intent."""
    
    def __init__(self):
        """Initialize the type hint analyzer."""
        # Common type mappings
        self.type_mappings = {
            "str": ("String", "Textual data", IntentType.ENTITY),
            "int": ("Integer", "Numeric data", IntentType.ENTITY),
            "float": ("Float", "Numeric data with decimal precision", IntentType.ENTITY),
            "bool": ("Boolean", "True/False condition", IntentType.STATE),
            "list": ("List", "Collection of items", IntentType.ENTITY),
            "dict": ("Dictionary", "Key-value mapping", IntentType.ENTITY),
            "set": ("Set", "Unique collection of items", IntentType.ENTITY),
            "tuple": ("Tuple", "Immutable collection of items", IntentType.ENTITY),
            "None": ("None", "No value", IntentType.STATE),
            "Any": ("Any", "Any type", IntentType.ENTITY),
            "Optional": ("Optional", "May be None", IntentType.STATE),
            "Union": ("Union", "One of several types", IntentType.ENTITY),
            "Callable": ("Callable", "Function or method", IntentType.ACTION),
            "Iterator": ("Iterator", "Sequence that can be iterated", IntentType.ENTITY),
            "Iterable": ("Iterable", "Can be iterated over", IntentType.ENTITY),
            "Generator": ("Generator", "Generates values on demand", IntentType.ACTION),
            "Type": ("Type", "Class or type object", IntentType.ENTITY),
            "Path": ("Path", "File system path", IntentType.ENTITY),
            "datetime": ("Datetime", "Date and time", IntentType.ENTITY),
            "date": ("Date", "Calendar date", IntentType.ENTITY),
            "time": ("Time", "Time of day", IntentType.ENTITY),
            "timedelta": ("Timedelta", "Duration", IntentType.ENTITY),
            "Exception": ("Exception", "Error condition", IntentType.STATE),
            "Pattern": ("Pattern", "Regular expression pattern", IntentType.ENTITY),
            "Match": ("Match", "Regular expression match", IntentType.ENTITY),
            # Add more as needed
        }
    
    def analyze_type_hint(
        self,
        type_hint: str,
        parameter_name: Optional[str] = None,
        is_return_type: bool = False,
        file_path: Optional[Path] = None,
        line_number: Optional[int] = None
    ) -> TypeIntent:
        """Analyze a type hint to extract intent.
        
        Args:
            type_hint: The type hint to analyze.
            parameter_name: The name of the parameter (optional).
            is_return_type: Whether this is a return type (optional).
            file_path: The path to the file containing the type hint (optional).
            line_number: The line number of the type hint (optional).
            
        Returns:
            A TypeIntent object.
        """
        # Parse the type hint
        is_optional, is_collection, is_custom_type, base_type = self._parse_type_hint(type_hint)
        
        # Get the intent for the base type
        name, description, intent_type = self._get_intent_for_type(base_type)
        
        # Adjust based on whether this is a parameter or return type
        if is_return_type:
            description = f"Returns {description.lower()}"
        elif parameter_name:
            description = f"Expects {description.lower()} for {parameter_name}"
        
        # Adjust based on optionality
        if is_optional:
            description += " (optional)"
        
        # Adjust based on collection
        if is_collection:
            description += " (collection)"
        
        # Create a location if file_path and line_number are provided
        location = None
        if file_path and line_number:
            location = CodeLocation(
                file_path=file_path,
                start_line=line_number,
                end_line=line_number
            )
        
        # Calculate confidence based on type quality
        confidence = self._calculate_confidence(type_hint, is_custom_type)
        
        # Create a TypeIntent
        intent = TypeIntent(
            name=name,
            description=description,
            type=intent_type,
            confidence=confidence,
            location=location,
            type_string=type_hint,
            is_optional=is_optional,
            is_collection=is_collection,
            is_custom_type=is_custom_type
        )
        
        return intent
    
    def _parse_type_hint(self, type_hint: str) -> Tuple[bool, bool, bool, str]:
        """Parse a type hint string.
        
        Args:
            type_hint: The type hint to parse.
            
        Returns:
            A tuple of (is_optional, is_collection, is_custom_type, base_type).
        """
        # Check for None or empty type hint
        if not type_hint or type_hint == "None":
            return False, False, False, "None"
        
        # Check for Optional
        is_optional = "Optional" in type_hint or "Union" in type_hint and "None" in type_hint
        
        # Check for collection types
        collection_types = ["List", "Dict", "Set", "Tuple", "Iterable", "Iterator", "Sequence"]
        is_collection = any(collection_type in type_hint for collection_type in collection_types)
        
        # Extract the base type
        base_type = type_hint
        
        # Handle Optional[Type]
        optional_match = re.match(r"Optional\[(.*)\]", type_hint)
        if optional_match:
            base_type = optional_match.group(1)
        
        # Handle Union[Type, None]
        union_match = re.match(r"Union\[(.*),\s*None\]", type_hint)
        if union_match:
            base_type = union_match.group(1)
        
        # Handle collection types
        for collection_type in collection_types:
            collection_match = re.match(f"{collection_type}\\[(.*?)\\]", type_hint)
            if collection_match:
                base_type = collection_match.group(1)
                break
        
        # Check if this is a custom type (not in our type mappings)
        is_custom_type = base_type not in self.type_mappings
        
        return is_optional, is_collection, is_custom_type, base_type
    
    def _get_intent_for_type(self, type_name: str) -> Tuple[str, str, IntentType]:
        """Get the intent for a type.
        
        Args:
            type_name: The type name.
            
        Returns:
            A tuple of (name, description, intent_type).
        """
        # Check if this is a known type
        if type_name in self.type_mappings:
            return self.type_mappings[type_name]
        
        # For custom types, use the type name as is
        return (type_name, f"Custom type {type_name}", IntentType.ENTITY)
    
    def _calculate_confidence(self, type_hint: str, is_custom_type: bool) -> float:
        """Calculate confidence based on type quality.
        
        Args:
            type_hint: The type hint.
            is_custom_type: Whether this is a custom type.
            
        Returns:
            The confidence score (0.0 to 1.0).
        """
        if not type_hint:
            return 0.0
        
        # Base confidence
        confidence = 0.6
        
        # Adjust based on type complexity
        if "Union" in type_hint or "Optional" in type_hint:
            confidence += 0.1  # More specific
        
        if any(collection in type_hint for collection in ["List", "Dict", "Set", "Tuple"]):
            confidence += 0.1  # More specific
        
        # Adjust based on custom type
        if is_custom_type:
            confidence += 0.2  # Custom types are often more meaningful
        
        # Cap confidence at 1.0
        return min(1.0, confidence)
    
    def analyze_parameter_type(
        self,
        parameter_name: str,
        type_hint: str,
        file_path: Optional[Path] = None,
        line_number: Optional[int] = None
    ) -> TypeIntent:
        """Analyze a parameter type hint to extract intent.
        
        Args:
            parameter_name: The name of the parameter.
            type_hint: The type hint to analyze.
            file_path: The path to the file containing the type hint (optional).
            line_number: The line number of the type hint (optional).
            
        Returns:
            A TypeIntent object.
        """
        return self.analyze_type_hint(type_hint, parameter_name, False, file_path, line_number)
    
    def analyze_return_type(
        self,
        type_hint: str,
        file_path: Optional[Path] = None,
        line_number: Optional[int] = None
    ) -> TypeIntent:
        """Analyze a return type hint to extract intent.
        
        Args:
            type_hint: The type hint to analyze.
            file_path: The path to the file containing the type hint (optional).
            line_number: The line number of the type hint (optional).
            
        Returns:
            A TypeIntent object.
        """
        return self.analyze_type_hint(type_hint, None, True, file_path, line_number)
```
