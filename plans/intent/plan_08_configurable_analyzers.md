# plan_08_configurable_analyzers.md
## Component: Configurable Analyzers

### Objective
Make all analyzers in the Structural Intent Analysis system configurable, allowing users to customize their behavior based on the configuration. This includes the name analyzer, type hint analyzer, and structural analyzer.

### Plan
1. Update the `NameAnalyzer` to use configuration for tokenization, semantic extraction, and confidence calculation
2. Update the `TypeHintAnalyzer` to use configuration for type mappings and confidence calculation
3. Update the `StructuralAnalyzer` to use configuration for pattern detection
4. Create factory methods for creating analyzers with configuration
5. Implement validation for analyzer-specific configuration

### Findings
Making analyzers configurable allows users to adapt the analysis to their specific codebase and preferences. Different projects may have different naming conventions, type hint usage, and architectural patterns, so the analyzers need to be flexible enough to accommodate these differences.

### Implementation Draft

```python
"""
Configurable analyzers for the Structural Intent Analysis system.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from semantic_matrix_analyzer.intent.models.intent import (
    Intent, NameIntent, TypeIntent, StructuralIntent, IntentType, CodeLocation
)
from semantic_matrix_analyzer.intent.config.configuration import (
    Configuration, ConfigurableAnalyzer
)

logger = logging.getLogger(__name__)


class ConfigurableNameAnalyzer(ConfigurableAnalyzer):
    """Configurable analyzer for names."""
    
    def __init__(self, config: Optional[Configuration] = None):
        """Initialize the configurable name analyzer.
        
        Args:
            config: The configuration to use (optional).
        """
        super().__init__(config)
        self.tokenizer = self._create_tokenizer()
        self.semantic_extractor = self._create_semantic_extractor()
    
    def _get_config_section(self) -> str:
        """Get the configuration section for this analyzer.
        
        Returns:
            The configuration section name.
        """
        return "name_analysis"
    
    def is_enabled(self) -> bool:
        """Check if the analyzer is enabled.
        
        Returns:
            True if the analyzer is enabled, False otherwise.
        """
        return self.get_config_value("enabled", True)
    
    def _create_tokenizer(self):
        """Create a tokenizer based on the configuration.
        
        Returns:
            A tokenizer object.
        """
        from semantic_matrix_analyzer.intent.analyzers.name_analyzer import NameTokenizer
        
        tokenizer = NameTokenizer()
        
        # Configure separators
        separators = self.get_config_value("tokenization.separators", ["_", "-", " "])
        tokenizer.separators = separators
        
        return tokenizer
    
    def _create_semantic_extractor(self):
        """Create a semantic extractor based on the configuration.
        
        Returns:
            A semantic extractor object.
        """
        from semantic_matrix_analyzer.intent.analyzers.name_analyzer import SemanticExtractor
        
        semantic_extractor = SemanticExtractor()
        
        # Configure action verbs
        action_verbs = self.get_config_value("semantic_extraction.action_verbs", {})
        if action_verbs:
            semantic_extractor.action_verbs = action_verbs
        
        # Configure design patterns
        design_patterns = self.get_config_value("semantic_extraction.design_patterns", {})
        if design_patterns:
            semantic_extractor.design_patterns = design_patterns
        
        # Configure domain objects
        domain_objects = self.get_config_value("semantic_extraction.domain_objects", {})
        if domain_objects:
            semantic_extractor.domain_objects = domain_objects
        
        return semantic_extractor
    
    def analyze_name(
        self,
        name: str,
        name_type: str,
        file_path: Optional[Path] = None,
        line_number: Optional[int] = None
    ) -> NameIntent:
        """Analyze a name to extract intent.
        
        Args:
            name: The name to analyze.
            name_type: The type of name ("class", "method", "variable", etc.).
            file_path: The path to the file containing the name (optional).
            line_number: The line number of the name (optional).
            
        Returns:
            A NameIntent object.
        """
        # Check if the analyzer is enabled
        if not self.is_enabled():
            return NameIntent(
                name=name,
                description=f"{name_type.capitalize()} name",
                type=IntentType.OTHER,
                confidence=0.0,
                original_name=name,
                tokens=[name],
                name_type=name_type
            )
        
        # Tokenize the name
        tokens = self.tokenizer.tokenize(name)
        
        # Extract intent from tokens
        intent_name, description, intent_type = self.semantic_extractor.extract_intent_from_tokens(tokens)
        
        # Create a location if file_path and line_number are provided
        location = None
        if file_path and line_number:
            location = CodeLocation(
                file_path=file_path,
                start_line=line_number,
                end_line=line_number
            )
        
        # Calculate confidence based on token quality
        confidence = self._calculate_confidence(tokens, name_type)
        
        # Create a NameIntent
        intent = NameIntent(
            name=intent_name,
            description=description,
            type=intent_type,
            confidence=confidence,
            location=location,
            original_name=name,
            tokens=tokens,
            name_type=name_type
        )
        
        return intent
    
    def _calculate_confidence(self, tokens: List[str], name_type: str) -> float:
        """Calculate confidence based on token quality.
        
        Args:
            tokens: The tokens to calculate confidence for.
            name_type: The type of name.
            
        Returns:
            The confidence score (0.0 to 1.0).
        """
        if not tokens:
            return 0.0
        
        # Base confidence
        confidence = self.get_config_value("confidence.base_confidence", 0.5)
        
        # Adjust based on token count
        if len(tokens) > 1:
            confidence += self.get_config_value("confidence.compound_name_bonus", 0.1)
        
        # Adjust based on token quality
        meaningful_tokens = 0
        for token in tokens:
            token_lower = token.lower()
            if (token_lower in self.semantic_extractor.action_verbs or
                token_lower in self.semantic_extractor.design_patterns or
                token_lower in self.semantic_extractor.domain_objects):
                meaningful_tokens += 1
        
        if tokens:
            confidence += self.get_config_value("confidence.meaningful_token_bonus", 0.2) * (meaningful_tokens / len(tokens))
        
        # Adjust based on name type
        if name_type == "class":
            confidence += self.get_config_value("confidence.class_name_bonus", 0.1)
        elif name_type == "method":
            confidence += self.get_config_value("confidence.method_name_bonus", 0.1)
        
        # Cap confidence at 1.0
        return min(1.0, confidence)
    
    def analyze_class_name(
        self,
        name: str,
        file_path: Optional[Path] = None,
        line_number: Optional[int] = None
    ) -> NameIntent:
        """Analyze a class name to extract intent.
        
        Args:
            name: The class name to analyze.
            file_path: The path to the file containing the class (optional).
            line_number: The line number of the class (optional).
            
        Returns:
            A NameIntent object.
        """
        return self.analyze_name(name, "class", file_path, line_number)
    
    def analyze_method_name(
        self,
        name: str,
        file_path: Optional[Path] = None,
        line_number: Optional[int] = None
    ) -> NameIntent:
        """Analyze a method name to extract intent.
        
        Args:
            name: The method name to analyze.
            file_path: The path to the file containing the method (optional).
            line_number: The line number of the method (optional).
            
        Returns:
            A NameIntent object.
        """
        return self.analyze_name(name, "method", file_path, line_number)
    
    def analyze_variable_name(
        self,
        name: str,
        file_path: Optional[Path] = None,
        line_number: Optional[int] = None
    ) -> NameIntent:
        """Analyze a variable name to extract intent.
        
        Args:
            name: The variable name to analyze.
            file_path: The path to the file containing the variable (optional).
            line_number: The line number of the variable (optional).
            
        Returns:
            A NameIntent object.
        """
        return self.analyze_name(name, "variable", file_path, line_number)


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
        # Implementation similar to the original TypeHintAnalyzer._parse_type_hint
        # but using configuration for collection types
        
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


class ConfigurableStructuralAnalyzer(ConfigurableAnalyzer):
    """Configurable analyzer for structural patterns."""
    
    def __init__(self, dependency_graph, config: Optional[Configuration] = None):
        """Initialize the configurable structural analyzer.
        
        Args:
            dependency_graph: The dependency graph to analyze.
            config: The configuration to use (optional).
        """
        super().__init__(config)
        self.dependency_graph = dependency_graph
    
    def _get_config_section(self) -> str:
        """Get the configuration section for this analyzer.
        
        Returns:
            The configuration section name.
        """
        return "structural_analysis"
    
    def is_enabled(self) -> bool:
        """Check if the analyzer is enabled.
        
        Returns:
            True if the analyzer is enabled, False otherwise.
        """
        return self.get_config_value("enabled", True)
    
    def analyze_dependency_graph(self) -> List[StructuralIntent]:
        """Analyze the dependency graph to extract intent.
        
        Returns:
            A list of StructuralIntent objects.
        """
        # Check if the analyzer is enabled
        if not self.is_enabled():
            return []
        
        intents = []
        
        # Detect layered architecture
        if self.get_config_value("patterns.layered_architecture.enabled", True):
            layered_intents = self._detect_layered_architecture()
            intents.extend(layered_intents)
        
        # Detect microservices architecture
        if self.get_config_value("patterns.microservices_architecture.enabled", True):
            microservices_intents = self._detect_microservices_architecture()
            intents.extend(microservices_intents)
        
        # Detect event-driven architecture
        if self.get_config_value("patterns.event_driven_architecture.enabled", True):
            event_driven_intents = self._detect_event_driven_architecture()
            intents.extend(event_driven_intents)
        
        # Detect MVC architecture
        if self.get_config_value("patterns.mvc_architecture.enabled", True):
            mvc_intents = self._detect_mvc_architecture()
            intents.extend(mvc_intents)
        
        # Detect repository pattern
        if self.get_config_value("patterns.repository_pattern.enabled", True):
            repository_intents = self._detect_repository_pattern()
            intents.extend(repository_intents)
        
        # Detect factory pattern
        if self.get_config_value("patterns.factory_pattern.enabled", True):
            factory_intents = self._detect_factory_pattern()
            intents.extend(factory_intents)
        
        # Detect singleton pattern
        if self.get_config_value("patterns.singleton_pattern.enabled", True):
            singleton_intents = self._detect_singleton_pattern()
            intents.extend(singleton_intents)
        
        return intents
    
    def _detect_layered_architecture(self) -> List[StructuralIntent]:
        """Detect layered architecture pattern.
        
        Returns:
            A list of StructuralIntent objects.
        """
        # Implementation similar to the original StructuralAnalyzer._detect_layered_architecture
        # but using configuration for layer names and confidence
        
        intents = []
        
        # Look for common layer names
        layer_names = self.get_config_value("patterns.layered_architecture.layer_names", [
            "presentation", "ui", "application", "service", "domain", "model", "data", "persistence", "infrastructure"
        ])
        
        # Get all file nodes
        file_nodes = self.dependency_graph.get_nodes_by_type("file")
        
        # Group files by layer
        layers: Dict[str, List[Node]] = {}
        for node in file_nodes:
            for layer_name in layer_names:
                if layer_name in str(node.file_path).lower():
                    if layer_name not in layers:
                        layers[layer_name] = []
                    layers[layer_name].append(node)
                    break
        
        # Check if we have at least 2 layers
        if len(layers) >= 2:
            # Check for dependencies between layers
            layer_dependencies: Dict[str, Set[str]] = {}
            for layer_name, layer_nodes in layers.items():
                layer_dependencies[layer_name] = set()
                for node in layer_nodes:
                    for edge in self.dependency_graph.get_outgoing_edges(node.id):
                        target_node = self.dependency_graph.get_node(edge.target_id)
                        if target_node and target_node.type == "file":
                            for other_layer_name, other_layer_nodes in layers.items():
                                if other_layer_name != layer_name and target_node in other_layer_nodes:
                                    layer_dependencies[layer_name].add(other_layer_name)
            
            # Check if we have dependencies between layers
            if any(deps for deps in layer_dependencies.values()):
                # Create a structural intent
                components = []
                for layer_name, layer_nodes in layers.items():
                    components.extend([node.id for node in layer_nodes])
                
                relationships = []
                for layer_name, deps in layer_dependencies.items():
                    for dep in deps:
                        relationships.append({
                            "source": layer_name,
                            "target": dep,
                            "type": "depends_on"
                        })
                
                confidence = self.get_config_value("patterns.layered_architecture.confidence", 0.7)
                
                intent = StructuralIntent(
                    name="Layered Architecture",
                    description="The codebase is organized in layers, with higher layers depending on lower layers.",
                    type=IntentType.PATTERN,
                    confidence=confidence,
                    pattern_name="Layered Architecture",
                    components=components,
                    relationships=relationships
                )
                
                intents.append(intent)
        
        return intents
    
    # Similar implementations for other pattern detection methods
    # (_detect_microservices_architecture, _detect_event_driven_architecture, etc.)
    # using configuration for pattern indicators and confidence
```
