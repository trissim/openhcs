# Plan 01: Plugin Interface Alignment

## Objective

Refactor the GPU Analysis Plugin to properly implement SMA's plugin interface, ensuring full compatibility with SMA's plugin system and enabling implementation of unimplemented SMA methods.

## Background

The Semantic Matrix Analyzer (SMA) uses a plugin architecture with a well-defined interface. All plugins must inherit from `SMAPlugin` and implement specific properties and methods. The current GPU Analysis Plugin doesn't follow this pattern, which will cause integration issues. Additionally, several key methods in SMA are currently unimplemented and will be implemented using GPU acceleration.

## Current State

The current GPU Analysis Plugin in `brain/gpu_analysis/plugin.py` is implemented as:

```python
class GPUAnalysisPlugin:
    """
    Plugin for GPU-accelerated analysis.

    This class provides a plugin interface for integrating GPU-accelerated analysis
    with the Semantic Matrix Analyzer. It follows proper separation of concerns,
    focusing only on GPU acceleration functionality.

    Attributes:
        device: Device to use for analysis ("cuda" or "cpu")
        config: Configuration dictionary
        semantic_analyzer: GPU-accelerated semantic analyzer
        ast_adapter: Adapter for converting between AST representations
    """

    def __init__(self, device: str = "cuda", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the GPU analysis plugin.

        Args:
            device: Device to use for analysis ("cuda" or "cpu")
            config: Configuration dictionary
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.config = config or {}

        # Initialize components
        self.semantic_analyzer = SemanticAnalyzer(device=self.device, config=self.config)
        self.ast_adapter = ASTAdapter(device=self.device, config=self.config)

        # Log initialization
        logger.info(f"GPU Analysis Plugin initialized with device: {self.device}")
        if self.device == "cuda":
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
```

SMA's plugin interface in `semantic_matrix_analyzer/semantic_matrix_analyzer/plugins/__init__.py` requires:

```python
class SMAPlugin(ABC):
    """Base class for all SMA plugins.

    All plugins must inherit from this class and implement its abstract methods.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the plugin."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Get the version of the plugin."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Get the description of the plugin."""
        pass

    def initialize(self, context: 'PluginContext') -> None:
        """Initialize the plugin with the given context.

        Args:
            context: The plugin context.
        """
        pass

    def shutdown(self) -> None:
        """Perform cleanup when the plugin is being unloaded."""
        pass
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

   def handle_error_trace_command(args: argparse.Namespace) -> None:
       """Handle the error-trace command."""
       print_header("ERROR TRACE PROCESSING")
       print("Processing error trace...")
       # Implementation would go here
       print(color_text("Not yet implemented", "YELLOW"))

   def handle_extract_intent_command(args: argparse.Namespace) -> None:
       """Handle the extract-intent command."""
       print_header("INTENT EXTRACTION")
       print("Extracting intent from conversation...")
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

### 1. Update Plugin Class Definition

Modify the `GPUAnalysisPlugin` class to inherit from `SMAPlugin` and implement the required properties and methods:

```python
from semantic_matrix_analyzer.semantic_matrix_analyzer.plugins import SMAPlugin, PluginContext

class GPUAnalysisPlugin(SMAPlugin):
    """
    Plugin for GPU-accelerated analysis.

    This class provides a plugin interface for integrating GPU-accelerated analysis
    with the Semantic Matrix Analyzer. It follows proper separation of concerns,
    focusing only on GPU acceleration functionality.

    Attributes:
        device: Device to use for analysis ("cuda" or "cpu")
        config: Configuration dictionary
        semantic_analyzer: GPU-accelerated semantic analyzer
        ast_adapter: Adapter for converting between AST representations
    """

    @property
    def name(self) -> str:
        """Get the name of the plugin."""
        return "gpu_analysis"

    @property
    def version(self) -> str:
        """Get the version of the plugin."""
        return "0.1.0"

    @property
    def description(self) -> str:
        """Get the description of the plugin."""
        return "GPU-accelerated semantic analysis for the Semantic Matrix Analyzer"

    def __init__(self, device: str = "cuda", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the GPU analysis plugin.

        Args:
            device: Device to use for analysis ("cuda" or "cpu")
            config: Configuration dictionary
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.config = config or {}
        self.context = None

        # Initialize components
        self.semantic_analyzer = SemanticAnalyzer(device=self.device, config=self.config)
        self.ast_adapter = ASTAdapter(device=self.device, config=self.config)

    def initialize(self, context: PluginContext) -> None:
        """Initialize the plugin with the given context.

        Args:
            context: The plugin context.
        """
        self.context = context

        # Log initialization using SMA's logging system
        context.log("info", f"GPU Analysis Plugin initialized with device: {self.device}")
        if self.device == "cuda":
            context.log("info", f"CUDA device: {torch.cuda.get_device_name(0)}")
            context.log("info", f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    def shutdown(self) -> None:
        """Perform cleanup when the plugin is being unloaded."""
        # Release GPU memory
        if hasattr(self, 'semantic_analyzer') and self.semantic_analyzer:
            self.semantic_analyzer.clear_cache()

        # Clear CUDA cache
        if self.device == "cuda":
            torch.cuda.empty_cache()

        if self.context:
            self.context.log("info", "GPU Analysis Plugin shutdown complete")
```

### 2. Add Methods for SMA Integration

Add methods specifically designed to implement the unimplemented SMA functionality:

```python
def analyze_component(self, component_name: str, file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Analyze a component using GPU acceleration.

    This method is designed to implement the unimplemented `analyze_component` method
    in SMA's `SemanticMatrixBuilder` class.

    Args:
        component_name: Name of the component
        file_path: Path to the file

    Returns:
        Dictionary of analysis results including:
        - dependencies: Component dependencies
        - metrics: Complexity metrics
        - pattern_matches: Pattern matches
        - intent_alignments: Intent alignment scores
    """
    try:
        if self.context:
            self.context.log("debug", f"Analyzing component: {component_name} in {file_path}")

        # Read the file
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        # Analyze the code
        results = self.semantic_analyzer.analyze(code, file_path)

        # Extract component-specific information
        component_results = {
            "name": component_name,
            "file_path": str(file_path),
            "dependencies": results.get("dependencies", {}),
            "metrics": results.get("complexity", {}),
            "pattern_matches": results.get("pattern_matches", []),
            "intent_alignments": results.get("intent_alignments", {})
        }

        return component_results
    except Exception as e:
        if self.context:
            self.context.log("error", f"Error analyzing component {component_name}: {e}")
        raise RuntimeError(f"GPU analysis failed for component {component_name}: {e}") from e

def analyze_error_trace(self, error_trace: str) -> Dict[str, Any]:
    """
    Analyze an error trace using GPU acceleration.

    This method is designed to implement the unimplemented `handle_error_trace_command`
    in SMA's CLI.

    Args:
        error_trace: Error trace text

    Returns:
        Dictionary of analysis results including:
        - error_type: Type of error
        - error_location: Location of the error
        - error_context: Context of the error
        - suggested_fixes: Suggested fixes
    """
    try:
        if self.context:
            self.context.log("debug", "Analyzing error trace")

        # Extract code snippets from error trace
        code_snippets = self.extract_code_from_error_trace(error_trace)

        # Analyze each code snippet
        snippet_results = []
        for file_path, code in code_snippets:
            result = self.semantic_analyzer.analyze(code, file_path)
            snippet_results.append((file_path, result))

        # Combine results
        return {
            "error_trace": error_trace,
            "snippets_analyzed": len(snippet_results),
            "snippet_results": snippet_results,
            "error_type": self.semantic_analyzer.classify_error(error_trace),
            "suggested_fixes": self.semantic_analyzer.suggest_fixes(error_trace, snippet_results)
        }
    except Exception as e:
        if self.context:
            self.context.log("error", f"Error analyzing error trace: {e}")
        raise RuntimeError(f"GPU analysis failed for error trace: {e}") from e

def extract_intents_from_conversation(self, conversation_text: str) -> List[Dict[str, Any]]:
    """
    Extract intents from conversation using GPU acceleration.

    This method is designed to implement the unimplemented `handle_extract_intent_command`
    in SMA's CLI.

    Args:
        conversation_text: Conversation text

    Returns:
        List of extracted intents
    """
    try:
        if self.context:
            self.context.log("debug", "Extracting intents from conversation")

        # Use GPU-accelerated NLP for intent extraction
        return self.semantic_analyzer.extract_intents(conversation_text)
    except Exception as e:
        if self.context:
            self.context.log("error", f"Error extracting intents: {e}")
        raise RuntimeError(f"GPU analysis failed for intent extraction: {e}") from e

def generate_semantic_snapshot(self, project_dir: str, depth: int = 3) -> Dict[str, Any]:
    """
    Generate a semantic snapshot of a project using GPU acceleration.

    This method is designed to implement the semantic analysis placeholders
    in SMA's `generate_project_snapshot` function.

    Args:
        project_dir: Path to project directory
        depth: Depth of analysis (1-5, where 5 is most detailed)

    Returns:
        Dictionary with semantic analysis results
    """
    try:
        if self.context:
            self.context.log("debug", f"Generating semantic snapshot for {project_dir}")

        # Find Python files
        python_files = []
        for root, _, files in os.walk(project_dir):
            for file in files:
                if file.endswith(".py"):
                    python_files.append(os.path.join(root, file))

        # Analyze a sample of files
        sample_size = min(10, len(python_files))
        sample_files = python_files[:sample_size]

        # Analyze files in batch
        sample_results = {}
        for file_path in sample_files:
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()
            result = self.semantic_analyzer.analyze(code, file_path)
            sample_results[file_path] = result

        # Extract semantic information
        return {
            "status": "analyzed",
            "sample_size": sample_size,
            "average_complexity": sum(r["complexity"]["cyclomatic_complexity"] for r in sample_results.values()) / sample_size if sample_size > 0 else 0,
            "common_patterns": self.extract_common_patterns(sample_results),
            "type_information": self.extract_type_information(sample_results)
        }
    except Exception as e:
        if self.context:
            self.context.log("error", f"Error generating semantic snapshot: {e}")
        raise RuntimeError(f"GPU analysis failed for semantic snapshot: {e}") from e
```

### 3. Add Helper Methods

Add helper methods to support the main functionality:

```python
def extract_code_from_error_trace(self, error_trace: str) -> List[Tuple[str, str]]:
    """
    Extract code snippets from an error trace.

    Args:
        error_trace: Error trace text

    Returns:
        List of (file_path, code) tuples
    """
    code_snippets = []

    # Extract file paths from error trace
    for line in error_trace.split("\n"):
        if line.strip().startswith("File ") and ", line " in line:
            # Extract file path
            file_path = line.split("File ")[1].split(", line ")[0].strip('"')

            # Check if file exists
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    code_snippets.append((file_path, f.read()))

    return code_snippets

def extract_common_patterns(self, analysis_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract common patterns from analysis results.

    Args:
        analysis_results: Dictionary of analysis results by file path

    Returns:
        List of common patterns
    """
    pattern_counts = {}

    for result in analysis_results.values():
        for match in result.get("pattern_matches", []):
            pattern_name = match["pattern"]["name"]
            pattern_counts[pattern_name] = pattern_counts.get(pattern_name, 0) + 1

    # Get top patterns
    top_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    return [{"name": name, "count": count} for name, count in top_patterns]

def extract_type_information(self, analysis_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract type information from analysis results.

    Args:
        analysis_results: Dictionary of analysis results by file path

    Returns:
        List of type information
    """
    type_counts = {}

    for result in analysis_results.values():
        if "types" in result:
            for type_name, count in result["types"].get("type_counts", {}).items():
                type_counts[type_name] = type_counts.get(type_name, 0) + count

    # Get top types
    top_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    return [{"name": name, "count": count} for name, count in top_types]
```

### 4. Add Standard Analysis Methods

Add standard analysis methods that will be used by the SMA integration methods:

```python
def analyze_code(self, code: str, file_path: Optional[Union[str, Path]] = None,
                analysis_types: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Analyze code using GPU acceleration.

    Args:
        code: Python code to analyze
        file_path: Optional path to the file
        analysis_types: Types of analysis to perform (if None, perform all)

    Returns:
        Dictionary of analysis results
    """
    if self.context:
        self.context.log("debug", f"Analyzing code from {file_path or 'string'}")
    return self.semantic_analyzer.analyze(code, file_path, analysis_types)

def analyze_file(self, file_path: Union[str, Path],
                analysis_types: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Analyze a file using GPU acceleration.

    Args:
        file_path: Path to the file
        analysis_types: Types of analysis to perform (if None, perform all)

    Returns:
        Dictionary of analysis results
    """
    try:
        if self.context:
            self.context.log("debug", f"Analyzing file: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        return self.analyze_code(code, file_path, analysis_types)
    except Exception as e:
        if self.context:
            self.context.log("error", f"Error analyzing file {file_path}: {e}")
        raise RuntimeError(f"GPU analysis failed for {file_path}: {e}") from e

def clear_cache(self) -> None:
    """Clear the GPU memory cache."""
    if hasattr(self, 'semantic_analyzer') and self.semantic_analyzer:
        self.semantic_analyzer.clear_cache()

    # Clear CUDA cache
    if self.device == "cuda":
        torch.cuda.empty_cache()

    if self.context:
        self.context.log("info", "GPU memory cache cleared")
```

## Implementation Focus

The implementation should focus on:

1. **Interface Compliance**: Ensuring the `GPUAnalysisPlugin` correctly implements SMA's plugin interface.

2. **Unimplemented SMA Methods**: Implementing all the unimplemented SMA functionality:
   - `analyze_component` for `SemanticMatrixBuilder.analyze_component`
   - `analyze_error_trace` for `handle_error_trace_command`
   - `extract_intents_from_conversation` for `handle_extract_intent_command`
   - `generate_semantic_snapshot` for semantic analysis in `generate_project_snapshot`

3. **Basic Integration**: Implementing essential integration with SMA's logging and error handling systems.

4. **GPU Acceleration**: Implementing GPU acceleration for all methods, with basic fallback to CPU when GPU is not available.

## Success Criteria

1. The GPU Analysis Plugin correctly implements SMA's plugin interface.

2. The plugin provides methods that implement all the unimplemented SMA functionality.

3. The plugin maintains architectural integrity through proper interface implementation.

4. The plugin provides GPU acceleration for all implemented methods.

Note: Testing will be deferred until the complete architecture is implemented and stable. The focus is on velocity and architectural completion rather than incremental validation.

## References

1. SMA Plugin Interface: `semantic_matrix_analyzer/semantic_matrix_analyzer/plugins/__init__.py`

2. GPU Analysis Plugin: `brain/gpu_analysis/plugin.py`

3. SMA Core Module: `semantic_matrix_analyzer/semantic_matrix_analyzer/core/__init__.py`

4. SMA CLI: `semantic_matrix_analyzer/sma_cli.py`
