"""
Type hint analysis components for extracting intent from type hints.
"""

import ast
import inspect
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, get_type_hints

from semantic_matrix_analyzer.intent.models.intent import (
    Intent, TypeIntent, IntentType, CodeLocation
)
from semantic_matrix_analyzer.intent.config.configuration import (
    Configuration, ConfigurableAnalyzer
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

            # Handle different Python versions
            if hasattr(annotation, 'slice') and isinstance(annotation.slice, ast.Index):
                # Python 3.8 and earlier
                slice_value = self._annotation_to_string(annotation.slice.value)
            elif hasattr(annotation, 'slice') and not isinstance(annotation.slice, ast.Index):
                # Python 3.9+
                slice_value = self._annotation_to_string(annotation.slice)
            else:
                # Fallback
                slice_value = "Any"

            return f"{value}[{slice_value}]"
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


class ConfigurableTypeHintAnalyzer(ConfigurableAnalyzer):
    """Configurable analyzer for type hints."""

    def __init__(self, config: Optional[Configuration] = None):
        """Initialize the configurable type hint analyzer.

        Args:
            config: The configuration to use (optional).
        """
        super().__init__(config)
        self.type_mappings = self._load_type_mappings()

    def _get_config_section(self) -> str:
        """Get the configuration section for this analyzer.

        Returns:
            The configuration section name.
        """
        return "type_analysis"

    def is_enabled(self) -> bool:
        """Check if the analyzer is enabled.

        Returns:
            True if the analyzer is enabled, False otherwise.
        """
        return self.get_config_value("enabled", True)

    def _load_type_mappings(self) -> Dict[str, Tuple[str, str, IntentType]]:
        """Load type mappings from the configuration.

        Returns:
            A dictionary mapping type names to (name, description, intent_type) tuples.
        """
        mappings = {}

        # Get type mappings from configuration
        config_mappings = self.get_config_value("type_mappings", {})

        for type_name, mapping in config_mappings.items():
            if isinstance(mapping, list) and len(mapping) >= 3:
                name, description, intent_type_str = mapping[:3]
                try:
                    # Convert to lowercase to match enum values
                    intent_type_str = intent_type_str.lower()
                    intent_type = IntentType(intent_type_str)
                    mappings[type_name] = (name, description, intent_type)
                except ValueError:
                    logger.warning(f"Invalid intent type in type mapping: {intent_type_str}")

        return mappings

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
        # Check if the analyzer is enabled
        if not self.is_enabled():
            return TypeIntent(
                name=type_hint,
                description=f"Type hint: {type_hint}",
                type=IntentType.OTHER,
                confidence=0.0,
                type_string=type_hint,
                is_optional=False,
                is_collection=False,
                is_custom_type=True
            )

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
        confidence = self._calculate_confidence(type_hint, is_optional, is_collection, is_custom_type)

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

    def _calculate_confidence(self, type_hint: str, is_optional: bool, is_collection: bool, is_custom_type: bool) -> float:
        """Calculate confidence based on type quality.

        Args:
            type_hint: The type hint.
            is_optional: Whether the type is optional.
            is_collection: Whether the type is a collection.
            is_custom_type: Whether this is a custom type.

        Returns:
            The confidence score (0.0 to 1.0).
        """
        if not type_hint:
            return 0.0

        # Base confidence
        confidence = self.get_config_value("confidence.base_confidence", 0.6)

        # Adjust based on type complexity
        if is_optional:
            confidence += self.get_config_value("confidence.union_optional_bonus", 0.1)

        if is_collection:
            confidence += self.get_config_value("confidence.collection_bonus", 0.1)

        # Adjust based on custom type
        if is_custom_type:
            confidence += self.get_config_value("confidence.custom_type_bonus", 0.2)

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
