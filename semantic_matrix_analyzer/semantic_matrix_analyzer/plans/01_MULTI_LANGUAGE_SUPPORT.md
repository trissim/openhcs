# Plan 01: Multi-Language Support

## Objective

Extend the Semantic Matrix Analyzer to support multiple programming languages beyond Python, making it universally applicable to diverse codebases.

## Rationale

Currently, the SMA is limited to Python codebases, which significantly restricts its utility. Most real-world projects use multiple languages, and many teams work across different technology stacks. By supporting multiple languages, we can:

1. Increase the tool's applicability to more projects
2. Enable analysis of full-stack applications
3. Provide consistent code quality insights across an organization's entire codebase
4. Reduce the need for language-specific tools, decreasing cognitive load

## Implementation Details

### 1. Language Parser Abstraction

Create a common interface for language parsers:

```python
class LanguageParser:
    """Abstract base class for language-specific parsers."""
    
    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Return file extensions supported by this parser."""
        raise NotImplementedError
    
    def parse_file(self, file_path: Path) -> Any:
        """Parse a file and return its AST."""
        raise NotImplementedError
    
    def get_node_type(self, node: Any) -> str:
        """Get the type of an AST node."""
        raise NotImplementedError
    
    def get_node_name(self, node: Any) -> Optional[str]:
        """Get the name of an AST node, if applicable."""
        raise NotImplementedError
    
    def get_node_children(self, node: Any) -> List[Any]:
        """Get the children of an AST node."""
        raise NotImplementedError
    
    def get_node_source(self, node: Any) -> str:
        """Get the source code for an AST node."""
        raise NotImplementedError
```

### 2. Initial Language Implementations

Implement parsers for the following languages:

#### 2.1 JavaScript/TypeScript Parser

Use the `esprima` or `@typescript-eslint/parser` libraries to parse JavaScript and TypeScript files.

```python
class JavaScriptParser(LanguageParser):
    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        return [".js", ".jsx", ".ts", ".tsx"]
    
    def parse_file(self, file_path: Path) -> Any:
        # Use esprima or typescript-eslint to parse the file
        pass
```

#### 2.2 Java Parser

Use the `javalang` library to parse Java files.

```python
class JavaParser(LanguageParser):
    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        return [".java"]
    
    def parse_file(self, file_path: Path) -> Any:
        # Use javalang to parse the file
        pass
```

#### 2.3 C# Parser

Use the `Roslyn` API via Python.NET or a custom service to parse C# files.

```python
class CSharpParser(LanguageParser):
    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        return [".cs"]
    
    def parse_file(self, file_path: Path) -> Any:
        # Use Roslyn or a custom service to parse the file
        pass
```

### 3. Language Detection and Routing

Create a system to automatically detect the language of a file and route it to the appropriate parser:

```python
class LanguageRouter:
    """Routes files to the appropriate language parser."""
    
    def __init__(self):
        self.parsers = []
        self._extension_map = {}
        
    def register_parser(self, parser_class: Type[LanguageParser]):
        """Register a language parser."""
        parser = parser_class()
        self.parsers.append(parser)
        for ext in parser_class.get_supported_extensions():
            self._extension_map[ext] = parser
    
    def get_parser_for_file(self, file_path: Path) -> Optional[LanguageParser]:
        """Get the appropriate parser for a file."""
        ext = file_path.suffix.lower()
        return self._extension_map.get(ext)
    
    def parse_file(self, file_path: Path) -> Optional[Any]:
        """Parse a file using the appropriate parser."""
        parser = self.get_parser_for_file(file_path)
        if parser:
            return parser.parse_file(file_path)
        return None
```

### 4. Pattern Adaptation

Adapt existing pattern detection to work with the new language parsers:

1. Create language-specific pattern implementations
2. Develop a common pattern interface
3. Implement pattern factories for each language

### 5. Testing Infrastructure

Develop comprehensive testing for each language parser:

1. Unit tests for parser functionality
2. Integration tests with pattern detection
3. End-to-end tests with sample codebases

## Success Criteria

1. Support for at least 3 additional languages (JavaScript/TypeScript, Java, C#)
2. 90% accuracy in pattern detection across all supported languages
3. Comprehensive test suite with >80% code coverage
4. Documentation for each supported language
5. Example projects demonstrating multi-language analysis

## Dependencies

- None (this is a foundational plan)

## Timeline

- Research and design: 2 weeks
- Implementation of language parser abstraction: 1 week
- JavaScript/TypeScript parser: 2 weeks
- Java parser: 2 weeks
- C# parser: 2 weeks
- Language detection and routing: 1 week
- Pattern adaptation: 2 weeks
- Testing and documentation: 2 weeks

Total: 12 weeks (3 months)
