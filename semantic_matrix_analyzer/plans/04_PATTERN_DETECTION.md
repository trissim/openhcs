# Plan 04: Enhanced Pattern Detection

## Objective

Develop a more sophisticated pattern detection system that can identify complex code patterns across multiple languages, improving the accuracy and relevance of code quality analysis.

## Rationale

The current pattern detection system is limited to simple string and regex matching, with basic AST pattern support. This limits the tool's ability to detect complex patterns that span multiple lines or files, or that involve sophisticated relationships between code elements. Enhanced pattern detection will:

1. **Increase Accuracy**: Detect patterns that simple string/regex matching would miss
2. **Reduce False Positives**: Better distinguish between similar but distinct patterns
3. **Enable Cross-File Analysis**: Detect patterns that span multiple files
4. **Support Semantic Analysis**: Understand the meaning and intent of code, not just syntax
5. **Improve Language Agnosticism**: Work consistently across different programming languages

## Implementation Details

### 1. Pattern Language

Create a domain-specific language (DSL) for defining patterns:

```python
class PatternLanguage:
    """A DSL for defining code patterns."""
    
    @staticmethod
    def node(type_name: str, **attributes) -> Dict[str, Any]:
        """Define a node pattern."""
        return {
            "type": "node",
            "node_type": type_name,
            "attributes": attributes
        }
    
    @staticmethod
    def sequence(*patterns) -> Dict[str, Any]:
        """Define a sequence of patterns."""
        return {
            "type": "sequence",
            "patterns": patterns
        }
    
    @staticmethod
    def alternative(*patterns) -> Dict[str, Any]:
        """Define alternative patterns (OR)."""
        return {
            "type": "alternative",
            "patterns": patterns
        }
    
    @staticmethod
    def optional(pattern) -> Dict[str, Any]:
        """Define an optional pattern."""
        return {
            "type": "optional",
            "pattern": pattern
        }
    
    @staticmethod
    def repetition(pattern, min_count: int = 0, max_count: int = None) -> Dict[str, Any]:
        """Define a repeating pattern."""
        return {
            "type": "repetition",
            "pattern": pattern,
            "min_count": min_count,
            "max_count": max_count
        }
    
    @staticmethod
    def has_child(parent_pattern, child_pattern) -> Dict[str, Any]:
        """Define a pattern where a parent node has a specific child."""
        return {
            "type": "has_child",
            "parent": parent_pattern,
            "child": child_pattern
        }
    
    @staticmethod
    def has_descendant(ancestor_pattern, descendant_pattern) -> Dict[str, Any]:
        """Define a pattern where an ancestor node has a specific descendant."""
        return {
            "type": "has_descendant",
            "ancestor": ancestor_pattern,
            "descendant": descendant_pattern
        }
    
    @staticmethod
    def follows(first_pattern, second_pattern) -> Dict[str, Any]:
        """Define a pattern where one node follows another in the source code."""
        return {
            "type": "follows",
            "first": first_pattern,
            "second": second_pattern
        }
```

### 2. Pattern Compiler

Create a compiler that translates pattern definitions into executable matchers:

```python
class PatternCompiler:
    """Compiles pattern definitions into executable matchers."""
    
    def __init__(self):
        self.matchers = {
            "node": self._compile_node_pattern,
            "sequence": self._compile_sequence_pattern,
            "alternative": self._compile_alternative_pattern,
            "optional": self._compile_optional_pattern,
            "repetition": self._compile_repetition_pattern,
            "has_child": self._compile_has_child_pattern,
            "has_descendant": self._compile_has_descendant_pattern,
            "follows": self._compile_follows_pattern
        }
    
    def compile(self, pattern: Dict[str, Any]) -> Callable[[Any, Dict[str, Any]], bool]:
        """Compile a pattern definition into an executable matcher."""
        pattern_type = pattern.get("type")
        if pattern_type not in self.matchers:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
        
        return self.matchers[pattern_type](pattern)
    
    def _compile_node_pattern(self, pattern: Dict[str, Any]) -> Callable[[Any, Dict[str, Any]], bool]:
        """Compile a node pattern."""
        node_type = pattern.get("node_type")
        attributes = pattern.get("attributes", {})
        
        def matcher(node: Any, context: Dict[str, Any]) -> bool:
            # Check if the node is of the specified type
            if context["language_parser"].get_node_type(node) != node_type:
                return False
            
            # Check if the node has the specified attributes
            for attr_name, attr_value in attributes.items():
                if attr_name == "name":
                    node_name = context["language_parser"].get_node_name(node)
                    if node_name != attr_value:
                        return False
                # Add more attribute checks as needed
            
            return True
        
        return matcher
    
    # Implement other pattern compilers...
```

### 3. Pattern Matcher

Create a system for matching patterns against code:

```python
class PatternMatcher:
    """Matches patterns against code."""
    
    def __init__(self, pattern_compiler: PatternCompiler):
        self.pattern_compiler = pattern_compiler
    
    def match(self, pattern: Dict[str, Any], node: Any, language_parser: LanguageParser) -> bool:
        """Match a pattern against a node."""
        matcher = self.pattern_compiler.compile(pattern)
        context = {
            "language_parser": language_parser,
            "variables": {}
        }
        return matcher(node, context)
    
    def find_matches(self, pattern: Dict[str, Any], root_node: Any, language_parser: LanguageParser) -> List[Any]:
        """Find all nodes that match the pattern."""
        matches = []
        matcher = self.pattern_compiler.compile(pattern)
        context = {
            "language_parser": language_parser,
            "variables": {}
        }
        
        def visit(node):
            if matcher(node, context):
                matches.append(node)
            
            for child in language_parser.get_node_children(node):
                visit(child)
        
        visit(root_node)
        return matches
```

### 4. Cross-File Pattern Detection

Extend pattern matching to work across multiple files:

```python
class CrossFilePatternMatcher:
    """Matches patterns across multiple files."""
    
    def __init__(self, pattern_matcher: PatternMatcher):
        self.pattern_matcher = pattern_matcher
    
    def find_matches(self, pattern: Dict[str, Any], files: Dict[Path, Any], language_parser: LanguageParser) -> Dict[Path, List[Any]]:
        """Find all nodes that match the pattern across multiple files."""
        matches = {}
        
        for file_path, root_node in files.items():
            file_matches = self.pattern_matcher.find_matches(pattern, root_node, language_parser)
            if file_matches:
                matches[file_path] = file_matches
        
        return matches
    
    def find_cross_file_matches(self, pattern: Dict[str, Any], files: Dict[Path, Any], language_parser: LanguageParser) -> List[Dict[str, Any]]:
        """Find matches for patterns that span multiple files."""
        # This would require more sophisticated analysis
        # For example, tracking references between files
        pass
```

### 5. Semantic Pattern Detection

Implement semantic pattern detection that understands the meaning of code:

```python
class SemanticPatternMatcher:
    """Matches patterns based on semantic understanding of code."""
    
    def __init__(self, pattern_matcher: PatternMatcher):
        self.pattern_matcher = pattern_matcher
    
    def find_semantic_matches(self, semantic_pattern: Dict[str, Any], root_node: Any, language_parser: LanguageParser) -> List[Any]:
        """Find nodes that match a semantic pattern."""
        # Convert semantic pattern to syntactic patterns
        syntactic_patterns = self._semantic_to_syntactic(semantic_pattern, language_parser)
        
        # Find matches for each syntactic pattern
        all_matches = []
        for pattern in syntactic_patterns:
            matches = self.pattern_matcher.find_matches(pattern, root_node, language_parser)
            all_matches.extend(matches)
        
        return all_matches
    
    def _semantic_to_syntactic(self, semantic_pattern: Dict[str, Any], language_parser: LanguageParser) -> List[Dict[str, Any]]:
        """Convert a semantic pattern to syntactic patterns."""
        # This would implement semantic understanding
        # For example, converting "function with error handling" to specific try-catch patterns
        pass
```

### 6. Pattern Library

Create a library of common patterns for different languages and frameworks:

```python
class PatternLibrary:
    """A library of common patterns."""
    
    def __init__(self):
        self.patterns = {}
    
    def add_pattern(self, name: str, pattern: Dict[str, Any], description: str) -> None:
        """Add a pattern to the library."""
        self.patterns[name] = {
            "pattern": pattern,
            "description": description
        }
    
    def get_pattern(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a pattern from the library."""
        pattern_info = self.patterns.get(name)
        if pattern_info:
            return pattern_info["pattern"]
        return None
    
    def get_all_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Get all patterns in the library."""
        return self.patterns
```

## Success Criteria

1. Pattern language supporting complex pattern definitions
2. Pattern compiler and matcher with >90% accuracy
3. Cross-file pattern detection for common use cases
4. Semantic pattern detection for high-level concepts
5. Pattern library with at least 50 common patterns
6. Documentation and examples for creating custom patterns

## Dependencies

- Plan 01: Multi-Language Support (for language-agnostic pattern matching)
- Plan 02: Plugin System Enhancement (for pattern plugins)

## Timeline

- Research and design: 2 weeks
- Pattern language implementation: 2 weeks
- Pattern compiler and matcher: 2 weeks
- Cross-file pattern detection: 1 week
- Semantic pattern detection: 2 weeks
- Pattern library development: 2 weeks
- Testing and documentation: 1 week

Total: 12 weeks
