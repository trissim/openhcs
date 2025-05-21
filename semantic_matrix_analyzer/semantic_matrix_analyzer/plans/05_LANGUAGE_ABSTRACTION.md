# Plan 05: Language Abstraction

## Objective

Create a common abstraction layer for language-specific features, enabling the Semantic Matrix Analyzer to work consistently across different programming languages while preserving language-specific insights.

## Rationale

Different programming languages have unique AST structures, idioms, and patterns. To make the SMA universally applicable, we need an abstraction layer that:

1. **Normalizes AST Structures**: Provides a consistent interface across language-specific ASTs
2. **Preserves Language Specificity**: Retains language-specific insights and patterns
3. **Enables Cross-Language Analysis**: Allows comparison of similar patterns across languages
4. **Simplifies Plugin Development**: Makes it easier to create language-agnostic plugins
5. **Facilitates Language Addition**: Reduces the effort required to add support for new languages

## Implementation Details

### 1. Universal AST (UAST)

Create a universal AST representation that normalizes across languages:

```python
@dataclass
class UASTNode:
    """A node in the Universal Abstract Syntax Tree (UAST)."""
    node_type: str
    name: Optional[str] = None
    value: Optional[Any] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    children: List['UASTNode'] = field(default_factory=list)
    parent: Optional['UASTNode'] = None
    source_range: Optional[Tuple[int, int]] = None  # (start, end) line numbers
    original_node: Optional[Any] = None  # The original language-specific node
    
    def add_child(self, child: 'UASTNode') -> None:
        """Add a child node."""
        self.children.append(child)
        child.parent = self
    
    def find_children(self, node_type: str) -> List['UASTNode']:
        """Find all children of a specific type."""
        return [child for child in self.children if child.node_type == node_type]
    
    def find_descendants(self, node_type: str) -> List['UASTNode']:
        """Find all descendants of a specific type."""
        result = []
        
        def visit(node):
            if node.node_type == node_type:
                result.append(node)
            for child in node.children:
                visit(child)
        
        for child in self.children:
            visit(child)
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the node to a dictionary."""
        return {
            "node_type": self.node_type,
            "name": self.name,
            "value": self.value,
            "attributes": self.attributes,
            "children": [child.to_dict() for child in self.children],
            "source_range": self.source_range
        }
```

### 2. Language-Specific UAST Converters

Create converters that transform language-specific ASTs into the UAST:

```python
class UASTConverter:
    """Base class for language-specific UAST converters."""
    
    def convert(self, node: Any) -> UASTNode:
        """Convert a language-specific AST node to a UAST node."""
        raise NotImplementedError

class PythonUASTConverter(UASTConverter):
    """Converts Python AST nodes to UAST nodes."""
    
    def convert(self, node: ast.AST) -> UASTNode:
        """Convert a Python AST node to a UAST node."""
        node_type = type(node).__name__
        name = None
        value = None
        attributes = {}
        source_range = None
        
        # Extract name if available
        if hasattr(node, "name"):
            name = node.name
        elif isinstance(node, ast.Name):
            name = node.id
        
        # Extract value if available
        if isinstance(node, ast.Constant):
            value = node.value
        
        # Extract source range if available
        if hasattr(node, "lineno"):
            start_line = node.lineno
            end_line = getattr(node, "end_lineno", start_line)
            source_range = (start_line, end_line)
        
        # Create UAST node
        uast_node = UASTNode(
            node_type=node_type,
            name=name,
            value=value,
            attributes=attributes,
            source_range=source_range,
            original_node=node
        )
        
        # Convert children
        for child_name, child_value in ast.iter_fields(node):
            if isinstance(child_value, ast.AST):
                child_uast = self.convert(child_value)
                uast_node.add_child(child_uast)
            elif isinstance(child_value, list):
                for item in child_value:
                    if isinstance(item, ast.AST):
                        child_uast = self.convert(item)
                        uast_node.add_child(child_uast)
        
        return uast_node

class JavaScriptUASTConverter(UASTConverter):
    """Converts JavaScript AST nodes to UAST nodes."""
    
    def convert(self, node: Any) -> UASTNode:
        """Convert a JavaScript AST node to a UAST node."""
        # Similar implementation for JavaScript
        pass

class JavaUASTConverter(UASTConverter):
    """Converts Java AST nodes to UAST nodes."""
    
    def convert(self, node: Any) -> UASTNode:
        """Convert a Java AST node to a UAST node."""
        # Similar implementation for Java
        pass
```

### 3. UAST-Based Pattern Matching

Adapt the pattern matching system to work with the UAST:

```python
class UASTPatternMatcher:
    """Matches patterns against UAST nodes."""
    
    def match(self, pattern: Dict[str, Any], node: UASTNode) -> bool:
        """Match a pattern against a UAST node."""
        pattern_type = pattern.get("type")
        
        if pattern_type == "node":
            return self._match_node(pattern, node)
        elif pattern_type == "sequence":
            return self._match_sequence(pattern, node)
        elif pattern_type == "alternative":
            return self._match_alternative(pattern, node)
        elif pattern_type == "optional":
            return self._match_optional(pattern, node)
        elif pattern_type == "repetition":
            return self._match_repetition(pattern, node)
        elif pattern_type == "has_child":
            return self._match_has_child(pattern, node)
        elif pattern_type == "has_descendant":
            return self._match_has_descendant(pattern, node)
        elif pattern_type == "follows":
            return self._match_follows(pattern, node)
        
        return False
    
    def _match_node(self, pattern: Dict[str, Any], node: UASTNode) -> bool:
        """Match a node pattern against a UAST node."""
        node_type = pattern.get("node_type")
        if node.node_type != node_type:
            return False
        
        attributes = pattern.get("attributes", {})
        for attr_name, attr_value in attributes.items():
            if attr_name == "name":
                if node.name != attr_value:
                    return False
            elif attr_name == "value":
                if node.value != attr_value:
                    return False
            elif attr_name not in node.attributes or node.attributes[attr_name] != attr_value:
                return False
        
        return True
    
    # Implement other pattern matching methods...
```

### 4. Language-Specific Pattern Adapters

Create adapters that translate language-specific patterns to UAST patterns:

```python
class PatternAdapter:
    """Base class for language-specific pattern adapters."""
    
    def adapt(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt a language-specific pattern to a UAST pattern."""
        raise NotImplementedError

class PythonPatternAdapter(PatternAdapter):
    """Adapts Python-specific patterns to UAST patterns."""
    
    def adapt(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt a Python-specific pattern to a UAST pattern."""
        pattern_type = pattern.get("type")
        
        if pattern_type == "node":
            return self._adapt_node_pattern(pattern)
        elif pattern_type == "has_child":
            return {
                "type": "has_child",
                "parent": self.adapt(pattern.get("parent")),
                "child": self.adapt(pattern.get("child"))
            }
        # Adapt other pattern types...
        
        return pattern
    
    def _adapt_node_pattern(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt a Python-specific node pattern to a UAST node pattern."""
        node_type = pattern.get("node_type")
        attributes = pattern.get("attributes", {})
        
        # Map Python-specific node types to UAST node types
        if node_type == "FunctionDef":
            node_type = "Function"
        elif node_type == "ClassDef":
            node_type = "Class"
        # Map other node types...
        
        return {
            "type": "node",
            "node_type": node_type,
            "attributes": attributes
        }
```

### 5. Language Feature Registry

Create a registry of language-specific features and their UAST equivalents:

```python
class LanguageFeature:
    """A language-specific feature and its UAST equivalent."""
    
    def __init__(self, name: str, description: str, languages: Dict[str, Dict[str, Any]]):
        self.name = name
        self.description = description
        self.languages = languages  # language_name -> {node_type, attributes, etc.}

class LanguageFeatureRegistry:
    """A registry of language features."""
    
    def __init__(self):
        self.features = {}
    
    def register_feature(self, feature: LanguageFeature) -> None:
        """Register a language feature."""
        self.features[feature.name] = feature
    
    def get_feature(self, name: str) -> Optional[LanguageFeature]:
        """Get a language feature by name."""
        return self.features.get(name)
    
    def get_language_specific_feature(self, name: str, language: str) -> Optional[Dict[str, Any]]:
        """Get a language-specific feature."""
        feature = self.get_feature(name)
        if feature:
            return feature.languages.get(language)
        return None
```

### 6. Cross-Language Pattern Library

Create a library of patterns that work across languages:

```python
class CrossLanguagePatternLibrary:
    """A library of patterns that work across languages."""
    
    def __init__(self, feature_registry: LanguageFeatureRegistry):
        self.feature_registry = feature_registry
        self.patterns = {}
    
    def add_pattern(self, name: str, pattern_template: Dict[str, Any], description: str) -> None:
        """Add a pattern to the library."""
        self.patterns[name] = {
            "template": pattern_template,
            "description": description
        }
    
    def get_pattern_for_language(self, name: str, language: str) -> Optional[Dict[str, Any]]:
        """Get a language-specific pattern based on the template."""
        pattern_info = self.patterns.get(name)
        if not pattern_info:
            return None
        
        template = pattern_info["template"]
        
        # Replace feature placeholders with language-specific features
        def replace_features(pattern_part):
            if isinstance(pattern_part, dict):
                if pattern_part.get("type") == "feature":
                    feature_name = pattern_part.get("feature_name")
                    feature = self.feature_registry.get_language_specific_feature(feature_name, language)
                    if feature:
                        return feature
                    return None
                
                result = {}
                for key, value in pattern_part.items():
                    result[key] = replace_features(value)
                return result
            elif isinstance(pattern_part, list):
                return [replace_features(item) for item in pattern_part]
            else:
                return pattern_part
        
        return replace_features(template)
```

## Success Criteria

1. UAST representation that works for at least 3 languages
2. Converters for Python, JavaScript, and Java
3. Pattern matching system that works with the UAST
4. Pattern adapters for language-specific patterns
5. Feature registry with at least 20 common features
6. Cross-language pattern library with at least 10 patterns

## Dependencies

- Plan 01: Multi-Language Support (for language-specific parsers)
- Plan 04: Pattern Detection (for pattern matching system)

## Timeline

- Research and design: 2 weeks
- UAST implementation: 2 weeks
- Language-specific converters: 3 weeks
- UAST-based pattern matching: 2 weeks
- Language-specific pattern adapters: 2 weeks
- Language feature registry: 1 week
- Cross-language pattern library: 2 weeks
- Testing and documentation: 2 weeks

Total: 16 weeks
