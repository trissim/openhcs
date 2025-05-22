# Plan 05b: Error Handling Integration

## Objective

Implement a robust error handling system for the GPU Analysis Plugin that integrates with SMA's error handling system, ensuring consistent error reporting and graceful failure handling, especially for the implementation of unimplemented SMA methods.

## Background

SMA has its own error handling system for reporting and tracking issues. The GPU Analysis Plugin needs to integrate with this system to ensure consistent error reporting throughout the application, especially for the implementation of unimplemented SMA methods that will use GPU acceleration.

## Current State

The current GPU Analysis Plugin likely uses basic exception handling:

```python
# In brain/gpu_analysis/plugin.py:

try:
    # Do something
    pass
except Exception as e:
    logger.error(f"Error: {e}")
    raise
```

SMA's error handling system likely provides more structured error reporting and handling.

## Unimplemented SMA Methods

The following SMA methods are currently unimplemented and will be implemented using GPU acceleration, requiring proper error handling:

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

### 1. Create an Error Handling Module

Create a new module for error handling that integrates with SMA's error handling system:

```python
# Create a new file: brain/gpu_analysis/error_handling.py

from typing import Optional, Any, Dict, Type
import traceback
from gpu_analysis.logging_integration import get_logger

# Configure logging
logger = get_logger("error_handling")

class GPUAnalysisError(Exception):
    """
    Base class for all GPU Analysis Plugin errors.

    This class provides a consistent interface for error handling in the
    GPU Analysis Plugin, with integration with SMA's error handling system.
    """

    def __init__(self, message: str, cause: Optional[Exception] = None):
        """
        Initialize the error.

        Args:
            message: Error message
            cause: Cause of the error
        """
        self.message = message
        self.cause = cause
        super().__init__(message)

    def __str__(self) -> str:
        """
        Get string representation of the error.

        Returns:
            String representation
        """
        if self.cause:
            return f"{self.message} (caused by: {self.cause})"
        return self.message

class GPUNotAvailableError(GPUAnalysisError):
    """Error raised when GPU is not available."""
    pass

class GPUMemoryError(GPUAnalysisError):
    """Error raised when GPU memory is insufficient."""
    pass

class GPUAnalysisConfigError(GPUAnalysisError):
    """Error raised when configuration is invalid."""
    pass

class GPUAnalysisRuntimeError(GPUAnalysisError):
    """Error raised when a runtime error occurs."""
    pass

class GPUAnalysisParsingError(GPUAnalysisError):
    """Error raised when parsing fails."""
    pass

class GPUAnalysisComponentError(GPUAnalysisError):
    """Error raised when component analysis fails."""
    pass

class GPUAnalysisPatternError(GPUAnalysisError):
    """Error raised when pattern matching fails."""
    pass

class GPUAnalysisIntentError(GPUAnalysisError):
    """Error raised when intent extraction fails."""
    pass

def handle_error(error: Exception, context: Optional['PluginContext'] = None) -> GPUAnalysisError:
    """
    Handle an error and convert it to a GPU Analysis Plugin error.

    This function handles errors in a consistent way, logging them and
    converting them to GPU Analysis Plugin errors.

    Args:
        error: Error to handle
        context: SMA plugin context

    Returns:
        GPU Analysis Plugin error
    """
    # Get logger with context if available
    log = get_logger("error_handling")

    # Handle different types of errors
    if isinstance(error, GPUAnalysisError):
        # Already a GPU Analysis Plugin error
        log.error(str(error))
        return error

    elif isinstance(error, torch.cuda.CudaError):
        # CUDA error
        message = f"CUDA error: {error}"
        log.error(message)
        return GPUMemoryError(message, error)

    elif isinstance(error, ValueError):
        # Value error
        message = f"Invalid value: {error}"
        log.error(message)
        return GPUAnalysisConfigError(message, error)

    elif isinstance(error, ImportError):
        # Import error
        message = f"Import error: {error}"
        log.error(message)
        return GPUAnalysisRuntimeError(message, error)

    elif isinstance(error, SyntaxError):
        # Syntax error
        message = f"Syntax error: {error}"
        log.error(message)
        return GPUAnalysisParsingError(message, error)

    else:
        # Other error
        message = f"Error: {error}"
        log.error(message)
        log.debug(traceback.format_exc())
        return GPUAnalysisRuntimeError(message, error)

def check_gpu_available() -> None:
    """
    Check if GPU is available.

    Raises:
        GPUNotAvailableError: If GPU is not available
    """
    import torch
    if not torch.cuda.is_available():
        raise GPUNotAvailableError("CUDA is not available")
```

### 2. Add Error Handling for Unimplemented SMA Methods

Add error handling for the implementation of unimplemented SMA methods:

```python
# In the implementation of analyze_component:

def analyze_component(self, component: str, file_path: Path) -> Any:
    """
    Analyze a component using GPU acceleration.

    This method implements the unimplemented SMA method for analyzing components.

    Args:
        component: Component to analyze
        file_path: Path to the file containing the component

    Returns:
        Analysis results
    """
    try:
        # Parse the file
        try:
            parsed = self.language_parser.parse_file(file_path)
        except Exception as e:
            raise GPUAnalysisParsingError(f"Error parsing file {file_path}", e)

        # Extract the component
        try:
            component_node = self._extract_component(parsed, component)
        except Exception as e:
            raise GPUAnalysisComponentError(f"Error extracting component {component}", e)

        # Analyze the component
        try:
            results = self.semantic_analyzer.analyze(component_node)
        except Exception as e:
            raise GPUAnalysisRuntimeError(f"Error analyzing component {component}", e)

        return results

    except GPUAnalysisError:
        # Re-raise GPU Analysis Plugin errors
        raise

    except Exception as e:
        # Convert other errors to GPU Analysis Plugin errors
        error = handle_error(e)
        raise error

# In the implementation of handle_analyze_command:

def handle_analyze_command(args: argparse.Namespace) -> None:
    """
    Handle the analyze command.

    This method implements the unimplemented SMA CLI command for analyzing code.

    Args:
        args: Command arguments
    """
    try:
        # Extract arguments
        file_path = args.file_path if hasattr(args, 'file_path') else None
        code = args.code if hasattr(args, 'code') else None

        # Print header
        print_header("CODE ANALYSIS")
        print("Analyzing code for intent extraction using GPU acceleration...")

        # Analyze code or file
        try:
            if file_path:
                results = plugin.analyze_file(file_path)
            elif code:
                results = plugin.analyze_code(code)
            else:
                print("No code or file specified")
                return
        except GPUAnalysisError as e:
            print(f"Error: {e}")
            return

        # Print results
        print(f"Analysis completed using GPU acceleration")
        print(f"Complexity: {results.get('complexity', {})}")
        print(f"Patterns: {len(results.get('pattern_matches', []))}")

    except Exception as e:
        # Handle unexpected errors
        error = handle_error(e)
        print(f"Error: {error}")
```

### 3. Update Plugin to Use Integrated Error Handling

Update the GPU Analysis Plugin to use the integrated error handling:

```python
# In brain/gpu_analysis/gpu_analysis_plugin.py:

from gpu_analysis.error_handling import handle_error, GPUAnalysisError, check_gpu_available

class GPUAnalysisPlugin(SMAPlugin):
    """
    Plugin for GPU-accelerated analysis.

    This class provides a plugin interface for integrating GPU-accelerated analysis
    with the Semantic Matrix Analyzer. It follows proper separation of concerns,
    focusing only on GPU acceleration functionality.
    """

    # ... other methods ...

    def initialize(self, context: PluginContext) -> None:
        """Initialize the plugin with the given context.

        Args:
            context: The plugin context.
        """
        self.context = context

        # Set up logging
        set_context(context)

        try:
            # Get SMA's configuration
            sma_config = context.get_config()

            # Integrate GPU configuration with SMA's configuration
            from gpu_analysis.config_integration import get_gpu_config_from_sma
            self.config = get_gpu_config_from_sma(sma_config)

            # Update device based on configuration
            self.device = self.config["device"]
            if self.device == "cuda":
                try:
                    check_gpu_available()
                except GPUAnalysisError as e:
                    logger.warning(f"GPU not available: {e}")
                    self.device = "cpu"

            # Initialize components with updated configuration
            self.semantic_analyzer = SemanticAnalyzer(device=self.device, config=self.config)
            self.ast_adapter = ASTAdapter(device=self.device, config=self.config)

            # Log initialization
            logger.info(f"GPU Analysis Plugin initialized with device: {self.device}")
            if self.device == "cuda":
                logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
                logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

        except Exception as e:
            # Handle initialization error
            error = handle_error(e, context)
            logger.error(f"Error initializing GPU Analysis Plugin: {error}")
            raise error

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
        try:
            logger.debug(f"Analyzing code from {file_path or 'string'}")
            return self.semantic_analyzer.analyze(code, file_path, analysis_types)
        except Exception as e:
            # Handle analysis error
            error = handle_error(e, self.context)
            logger.error(f"Error analyzing code: {error}")
            raise error
```

### 4. Add Fallback Mechanisms for Graceful Degradation

Add fallback mechanisms to ensure graceful degradation when errors occur:

```python
# In the implementation of analyze_component:

def analyze_component(self, component: str, file_path: Path) -> Any:
    """
    Analyze a component using GPU acceleration.

    This method implements the unimplemented SMA method for analyzing components.

    Args:
        component: Component to analyze
        file_path: Path to the file containing the component

    Returns:
        Analysis results
    """
    try:
        # Try GPU-accelerated analysis
        return self._analyze_component_gpu(component, file_path)
    except GPUNotAvailableError as e:
        # Fall back to CPU analysis
        logger.warning(f"GPU not available, falling back to CPU: {e}")
        return self._analyze_component_cpu(component, file_path)
    except GPUMemoryError as e:
        # Fall back to CPU analysis
        logger.warning(f"GPU memory insufficient, falling back to CPU: {e}")
        return self._analyze_component_cpu(component, file_path)
    except Exception as e:
        # Handle other errors
        error = handle_error(e)
        logger.error(f"Error analyzing component {component}: {error}")
        raise error

# In the implementation of handle_analyze_command:

def handle_analyze_command(args: argparse.Namespace) -> None:
    """
    Handle the analyze command.

    This method implements the unimplemented SMA CLI command for analyzing code.

    Args:
        args: Command arguments
    """
    try:
        # Extract arguments
        file_path = args.file_path if hasattr(args, 'file_path') else None
        code = args.code if hasattr(args, 'code') else None

        # Print header
        print_header("CODE ANALYSIS")
        print("Analyzing code for intent extraction...")

        # Analyze code or file
        try:
            if file_path:
                results = plugin.analyze_file(file_path)
            elif code:
                results = plugin.analyze_code(code)
            else:
                print("No code or file specified")
                return
        except GPUNotAvailableError as e:
            print(f"Warning: {e}")
            print("Falling back to CPU analysis...")
            # Fall back to CPU analysis
            if file_path:
                results = plugin.analyze_file_cpu(file_path)
            elif code:
                results = plugin.analyze_code_cpu(code)
            else:
                print("No code or file specified")
                return
        except GPUAnalysisError as e:
            print(f"Error: {e}")
            return

        # Print results
        print(f"Analysis completed")
        print(f"Complexity: {results.get('complexity', {})}")
        print(f"Patterns: {len(results.get('pattern_matches', []))}")

    except Exception as e:
        # Handle unexpected errors
        error = handle_error(e)
        print(f"Error: {error}")
```

## Implementation Focus

The implementation should focus on:

1. **Error Handling Integration**: Implementing integration with SMA's error handling system.

2. **Component Error Handling**: Ensuring all components use the integrated error handling system.

3. **Method Error Handling**: Adding appropriate error handling to the implementation of unimplemented SMA methods.

4. **Error Categorization**: Implementing proper categorization and reporting of errors.

5. **Graceful Degradation**: Implementing graceful handling of GPU unavailability and other common errors.

## Success Criteria

1. The GPU Analysis Plugin's error handling is correctly integrated with SMA's error handling system.

2. All components of the GPU Analysis Plugin use the integrated error handling system.

3. The implementation of unimplemented SMA methods includes appropriate error handling.

4. The plugin gracefully handles GPU unavailability and other common errors.

Note: Testing will be deferred until the complete architecture is implemented and stable. The focus is on velocity and architectural completion rather than incremental validation.

## References

1. SMA Error Handling System: `semantic_matrix_analyzer/semantic_matrix_analyzer/errors/__init__.py`

2. SMA Plugin Context: `semantic_matrix_analyzer/semantic_matrix_analyzer/plugins/__init__.py`

3. Python Exception Handling: https://docs.python.org/3/tutorial/errors.html
