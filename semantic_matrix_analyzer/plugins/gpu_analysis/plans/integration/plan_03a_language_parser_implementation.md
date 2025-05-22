# Plan 03a: Language Parser Interface Alignment

## Objective

Refactor the GPU Language Parser to properly implement SMA's `LanguageParser` interface, ensuring seamless integration with SMA's language parsing system and enabling the implementation of unimplemented SMA methods.

## Background

SMA uses a language parsing system with a well-defined interface. All language parsers must implement the `LanguageParser` interface. The current GPU Language Parser doesn't fully implement this interface, which will cause integration issues. This plan focuses on the core interface alignment to enable the GPU-accelerated implementation of unimplemented SMA methods.

## Current State

The current GPU Language Parser is implemented in `brain/gpu_analysis/ast_adapter.py` as:

```python
class GPULanguageParser:
    """GPU-accelerated language parser."""

    def __init__(self, base_parser, device="cuda"):
        self.base_parser = base_parser
        self.device = device
        self.tensorizer = GPUASTTensorizer(device)
        self.adapter = ASTAdapter(base_parser)

    def parse_file(self, file_path):
        """Parse a file and return its AST."""
        # Use base parser to parse the file
        ast_node = self.base_parser.parse_file(file_path)

        # Convert to GPU-friendly format
        gpu_ast = self.tensorizer.tensorize(ast_node)

        # Return both representations for flexibility
        return {
            "ast": ast_node,
            "gpu_ast": gpu_ast
        }
```

SMA's `LanguageParser` interface in `semantic_matrix_analyzer/semantic_matrix_analyzer/language/__init__.py` requires:

```python
class LanguageParser(ABC):
    """Abstract base class for language-specific parsers.

    All language parsers must implement this interface to provide
    a consistent way to parse and analyze code in different languages.
    """

    @classmethod
    @abstractmethod
    def get_supported_extensions(cls) -> Set[str]:
        """Return file extensions supported by this parser.

        Returns:
            A set of file extensions (including the dot) that this parser supports.
            For example: {".py", ".pyi"}
        """
        pass

    @abstractmethod
    def parse_file(self, file_path: Path) -> Any:
        """Parse a file and return its AST.

        Args:
            file_path: Path to the file to parse.

        Returns:
            The AST representation of the file, which may be language-specific.

        Raises:
            FileNotFoundError: If the file does not exist.
            SyntaxError: If the file contains syntax errors.
            ValueError: If the file is not supported by this parser.
        """
        pass

    # Additional methods omitted for brevity
```

## Unimplemented SMA Methods

The following SMA methods are currently unimplemented and will be implemented using GPU acceleration:

1. `SemanticMatrixBuilder.analyze_component` in the core module:
   ```python
   # TODO: Extract dependencies
   # TODO: Analyze component
   # TODO: Detect patterns
   # TODO: Calculate intent alignments
   ```

2. CLI command handlers in `sma_cli.py`:
   ```python
   def handle_analyze_command(args: argparse.Namespace) -> None:
       """Handle the analyze command."""
       print_header("CODE ANALYSIS")
       print("Analyzing code for intent extraction...")
       # Implementation would go here
       print(color_text("Not yet implemented", "YELLOW"))
   ```

3. Semantic analysis placeholders in `generate_project_snapshot`:
   ```python
   # Placeholder for semantic analysis
   if focus in ["semantics", "all"] and depth >= 2:
       snapshot["semantics"] = {
           "status": "placeholder",
           "message": "Semantic analysis would analyze code patterns, naming conventions, and code quality."
       }
   ```

## Implementation Plan

### 1. Update GPU Language Parser Class Definition

Modify the `GPULanguageParser` class to inherit from `LanguageParser` and implement the basic required methods:

```python
from semantic_matrix_analyzer.semantic_matrix_analyzer.language import LanguageParser
from pathlib import Path
from typing import Any, List, Optional, Set, Tuple, Union
import torch
import logging

# Configure logging
logger = logging.getLogger(__name__)

class GPULanguageParser(LanguageParser):
    """
    GPU-accelerated language parser.

    This class implements SMA's LanguageParser interface and provides
    GPU acceleration for parsing and analyzing Python code.

    It serves as the foundation for implementing unimplemented SMA methods
    that require language parsing capabilities.
    """

    def __init__(self, device="cuda"):
        """
        Initialize the GPU language parser.

        Args:
            device: Device to use for GPU acceleration ("cuda" or "cpu")
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"

        # Create a base parser for Python
        from semantic_matrix_analyzer.semantic_matrix_analyzer.language.python_parser import PythonParser
        self.base_parser = PythonParser()

        # Initialize tensorizer
        from gpu_analysis.ast_tensor import GPUASTTensorizer
        self.tensorizer = GPUASTTensorizer(device=self.device)

        logger.info(f"GPU Language Parser initialized with device: {self.device}")

    @classmethod
    def get_supported_extensions(cls) -> Set[str]:
        """
        Return file extensions supported by this parser.

        Returns:
            A set of file extensions (including the dot) that this parser supports.
        """
        return {".py", ".pyi"}

    def parse_file(self, file_path: Path) -> Any:
        """
        Parse a file and return its AST.

        This method parses a file and returns both the standard AST representation
        and the GPU-friendly tensor representation, enabling GPU-accelerated analysis.

        Args:
            file_path: Path to the file to parse.

        Returns:
            The AST representation of the file, with both standard and GPU-friendly formats.

        Raises:
            FileNotFoundError: If the file does not exist.
            SyntaxError: If the file contains syntax errors.
            ValueError: If the file is not supported by this parser.
        """
        try:
            # Check if file exists
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Check if file extension is supported
            if file_path.suffix not in self.get_supported_extensions():
                raise ValueError(f"Unsupported file extension: {file_path.suffix}")

            # Use base parser to parse the file
            ast_node = self.base_parser.parse_file(file_path)

            # Convert to GPU-friendly format
            gpu_ast = self.tensorizer.tensorize(ast_node)

            # Return both representations for flexibility
            return {
                "ast": ast_node,
                "gpu_ast": gpu_ast,
                "file_path": str(file_path)
            }
        except SyntaxError as e:
            logger.error(f"Syntax error in file {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            raise
```

### 2. Implement Parse String Method

Add a method to parse code strings, which is useful for analyzing code snippets:

```python
def parse_string(self, code: str, file_path: Optional[Path] = None) -> Any:
    """
    Parse a string of code and return its AST.

    This method parses a string of code and returns both the standard AST
    representation and the GPU-friendly tensor representation, enabling
    GPU-accelerated analysis.

    Args:
        code: String of code to parse.
        file_path: Optional path to associate with the code.

    Returns:
        The AST representation of the code, with both standard and GPU-friendly formats.

    Raises:
        SyntaxError: If the code contains syntax errors.
    """
    try:
        # Use base parser to parse the string
        if hasattr(self.base_parser, 'parse_string'):
            ast_node = self.base_parser.parse_string(code)
        else:
            # Fall back to ast module if base parser doesn't support string parsing
            import ast
            ast_node = ast.parse(code)

        # Convert to GPU-friendly format
        gpu_ast = self.tensorizer.tensorize(ast_node)

        # Return both representations for flexibility
        return {
            "ast": ast_node,
            "gpu_ast": gpu_ast,
            "file_path": str(file_path) if file_path else None
        }
    except SyntaxError as e:
        logger.error(f"Syntax error in code: {e}")
        raise
    except Exception as e:
        logger.error(f"Error parsing code: {e}")
        raise
```

### 3. Implement File Type Detection

Add methods to detect file types and check if a file is supported:

```python
@classmethod
def is_supported_file(cls, file_path: Path) -> bool:
    """
    Check if a file is supported by this parser.

    Args:
        file_path: Path to the file to check.

    Returns:
        True if the file is supported, False otherwise.
    """
    return file_path.suffix in cls.get_supported_extensions()

@classmethod
def get_language_name(cls) -> str:
    """
    Get the name of the language supported by this parser.

    Returns:
        The name of the language.
    """
    return "Python-GPU"

def get_file_type(self, file_path: Path) -> str:
    """
    Get the type of a file.

    Args:
        file_path: Path to the file to check.

    Returns:
        The type of the file.
    """
    if not self.is_supported_file(file_path):
        raise ValueError(f"Unsupported file extension: {file_path.suffix}")

    # For Python files, we can distinguish between different types
    if file_path.suffix == ".py":
        return "python"
    elif file_path.suffix == ".pyi":
        return "python-interface"
    else:
        return "unknown"
```

### 4. Update Registration Function

Update the language parser registration function to work with SMA's language registry:

```python
def register_gpu_parser(language_registry):
    """
    Register GPU language parsers with SMA's language registry.

    This function registers the GPU language parser with SMA's language registry,
    enabling it to be used for parsing and analyzing code in SMA.

    Args:
        language_registry: SMA's language registry
    """
    try:
        # Register the GPU language parser
        language_registry.register_parser(GPULanguageParser)
        logger.info("GPU Language Parser registered with SMA")

        # Check if registration was successful
        registered_parsers = language_registry.get_registered_parsers()
        if GPULanguageParser in registered_parsers:
            logger.info("GPU Language Parser successfully registered")
        else:
            logger.warning("GPU Language Parser registration check failed")
    except Exception as e:
        logger.error(f"Error registering GPU Language Parser with SMA: {e}")
        raise
```

### 5. Add Integration with SMA's Unimplemented Methods

Add methods to integrate with SMA's unimplemented methods:

```python
def analyze_component_code(self, component_name: str, code: str) -> Dict[str, Any]:
    """
    Analyze component code using GPU acceleration.

    This method is designed to be used by SMA's `analyze_component` method
    to provide GPU-accelerated analysis of component code.

    Args:
        component_name: Name of the component
        code: Code to analyze

    Returns:
        Analysis results
    """
    try:
        # Parse the code
        parsed = self.parse_string(code)

        # Analyze the code
        from gpu_analysis.analyzers.semantic_analyzer import SemanticAnalyzer
        analyzer = SemanticAnalyzer(device=self.device)

        # Perform analysis
        results = analyzer.analyze(parsed["gpu_ast"])

        # Add component information
        results["component_name"] = component_name

        return results
    except Exception as e:
        logger.error(f"Error analyzing component {component_name}: {e}")
        raise

def analyze_file_for_cli(self, file_path: Path) -> Dict[str, Any]:
    """
    Analyze a file for CLI output using GPU acceleration.

    This method is designed to be used by SMA's CLI commands to provide
    GPU-accelerated analysis of files.

    Args:
        file_path: Path to the file to analyze

    Returns:
        Analysis results formatted for CLI output
    """
    try:
        # Parse the file
        parsed = self.parse_file(file_path)

        # Analyze the file
        from gpu_analysis.analyzers.semantic_analyzer import SemanticAnalyzer
        analyzer = SemanticAnalyzer(device=self.device)

        # Perform analysis
        results = analyzer.analyze(parsed["gpu_ast"])

        # Format results for CLI output
        cli_results = {
            "file_path": str(file_path),
            "complexity": results.get("complexity", {}),
            "patterns": results.get("pattern_matches", []),
            "dependencies": results.get("dependencies", {}),
            "metrics": results.get("metrics", {})
        }

        return cli_results
    except Exception as e:
        logger.error(f"Error analyzing file {file_path} for CLI: {e}")
        raise
```

## Implementation Focus

The implementation should focus on:

1. **Interface Compliance**: Implementing the basic methods required by SMA's `LanguageParser` interface.

2. **Registration Mechanism**: Implementing registration with SMA's language registry.

3. **Parsing Functionality**: Implementing parsing of Python files to return both standard and GPU-friendly AST representations.

4. **Integration Methods**: Implementing methods for integrating with SMA's unimplemented methods.

5. **Basic Error Handling**: Implementing essential error handling for architectural correctness.

## Success Criteria

1. The `GPULanguageParser` correctly implements the basic methods required by SMA's `LanguageParser` interface.

2. The parser can be registered with SMA's language registry.

3. The parser can correctly parse Python files and return both standard and GPU-friendly AST representations.

4. The parser provides methods for integrating with SMA's unimplemented methods.

Note: Testing will be deferred until the complete architecture is implemented and stable. The focus is on velocity and architectural completion rather than incremental validation.

## References

1. SMA Language Parser Interface: `semantic_matrix_analyzer/semantic_matrix_analyzer/language/__init__.py`

2. SMA Python Parser: `semantic_matrix_analyzer/semantic_matrix_analyzer/language/python_parser.py`

3. GPU AST Adapter: `brain/gpu_analysis/ast_adapter.py`

4. GPU AST Tensorizer: `brain/gpu_analysis/ast_tensor.py`

5. SMA Core Module: `semantic_matrix_analyzer/semantic_matrix_analyzer/core/__init__.py`

6. SMA CLI: `semantic_matrix_analyzer/sma_cli.py`
