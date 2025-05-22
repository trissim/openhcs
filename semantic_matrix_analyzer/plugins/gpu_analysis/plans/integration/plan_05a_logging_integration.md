# Plan 05a: Logging Integration

## Objective

Refactor the GPU Analysis Plugin's logging to integrate with SMA's logging system, ensuring consistent logging throughout the application and proper logging for the implementation of unimplemented SMA methods.

## Background

SMA has its own logging system for tracking application events and issues. The GPU Analysis Plugin needs to integrate with this system to ensure consistent logging throughout the application, especially for the implementation of unimplemented SMA methods that will use GPU acceleration.

## Current State

The current GPU Analysis Plugin likely uses Python's standard logging module:

```python
# In brain/gpu_analysis/plugin.py:

import logging

# Configure logging
logger = logging.getLogger(__name__)

# Log initialization
logger.info(f"GPU Analysis Plugin initialized with device: {self.device}")
```

SMA's logging system in `semantic_matrix_analyzer/semantic_matrix_analyzer/logging/__init__.py` likely works differently:

```python
class Logger:
    """Logger for SMA.

    This class provides a consistent logging interface for SMA and its plugins.
    """

    def __init__(self, name: str, level: str = "info"):
        """Initialize the logger.

        Args:
            name: Name of the logger.
            level: Logging level.
        """
        self.name = name
        self.level = level

        # Set up Python logger
        self.logger = logging.getLogger(name)
        self.set_level(level)

    def set_level(self, level: str) -> None:
        """Set the logging level.

        Args:
            level: Logging level.
        """
        level_map = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL
        }

        self.level = level
        self.logger.setLevel(level_map.get(level.lower(), logging.INFO))

    def debug(self, message: str) -> None:
        """Log a debug message.

        Args:
            message: Message to log.
        """
        self.logger.debug(message)

    def info(self, message: str) -> None:
        """Log an info message.

        Args:
            message: Message to log.
        """
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log a warning message.

        Args:
            message: Message to log.
        """
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log an error message.

        Args:
            message: Message to log.
        """
        self.logger.error(message)

    def critical(self, message: str) -> None:
        """Log a critical message.

        Args:
            message: Message to log.
        """
        self.logger.critical(message)
```

SMA's plugin context in `semantic_matrix_analyzer/semantic_matrix_analyzer/plugins/__init__.py` likely provides a logging interface:

```python
class PluginContext:
    """Context provided to plugins.

    This class provides a context for plugins to interact with SMA.
    """

    def __init__(self, config: Dict[str, Any] = None, logger: Optional['Logger'] = None):
        """Initialize the plugin context.

        Args:
            config: Configuration dictionary.
            logger: Logger instance.
        """
        self.config = config or {}
        self.logger = logger or Logger("sma.plugin")

    def get_config(self) -> Dict[str, Any]:
        """Get the configuration.

        Returns:
            Configuration dictionary.
        """
        return self.config

    def log(self, level: str, message: str) -> None:
        """Log a message.

        Args:
            level: Logging level.
            message: Message to log.
        """
        if level == "debug":
            self.logger.debug(message)
        elif level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        elif level == "critical":
            self.logger.critical(message)
        else:
            self.logger.info(message)
```

## Unimplemented SMA Methods

The following SMA methods are currently unimplemented and will be implemented using GPU acceleration, requiring proper logging integration:

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

### 1. Create a Logging Integration Module

Create a new module to integrate the GPU Analysis Plugin's logging with SMA's logging system:

```python
# Create a new file: brain/gpu_analysis/logging_integration.py

from typing import Optional, Any, Dict
import logging

class SMALoggerAdapter:
    """
    Adapter for SMA's logging system.

    This class provides an adapter between Python's standard logging module
    and SMA's logging system, allowing GPU Analysis Plugin components to
    use SMA's logging system through a familiar interface.
    """

    def __init__(self, context: Optional['PluginContext'] = None, name: str = "gpu_analysis"):
        """
        Initialize the logger adapter.

        Args:
            context: SMA plugin context
            name: Logger name
        """
        self.context = context
        self.name = name

        # Set up fallback logger
        self.fallback_logger = logging.getLogger(name)

    def debug(self, message: str) -> None:
        """
        Log a debug message.

        Args:
            message: Message to log
        """
        if self.context:
            self.context.log("debug", f"[{self.name}] {message}")
        else:
            self.fallback_logger.debug(message)

    def info(self, message: str) -> None:
        """
        Log an info message.

        Args:
            message: Message to log
        """
        if self.context:
            self.context.log("info", f"[{self.name}] {message}")
        else:
            self.fallback_logger.info(message)

    def warning(self, message: str) -> None:
        """
        Log a warning message.

        Args:
            message: Message to log
        """
        if self.context:
            self.context.log("warning", f"[{self.name}] {message}")
        else:
            self.fallback_logger.warning(message)

    def error(self, message: str) -> None:
        """
        Log an error message.

        Args:
            message: Message to log
        """
        if self.context:
            self.context.log("error", f"[{self.name}] {message}")
        else:
            self.fallback_logger.error(message)

    def critical(self, message: str) -> None:
        """
        Log a critical message.

        Args:
            message: Message to log
        """
        if self.context:
            self.context.log("critical", f"[{self.name}] {message}")
        else:
            self.fallback_logger.critical(message)

    def exception(self, message: str) -> None:
        """
        Log an exception message.

        Args:
            message: Message to log
        """
        if self.context:
            self.context.log("error", f"[{self.name}] {message}")
        else:
            self.fallback_logger.exception(message)

# Global logger instance
_logger = SMALoggerAdapter()

def set_context(context: 'PluginContext') -> None:
    """
    Set the SMA plugin context for logging.

    Args:
        context: SMA plugin context
    """
    global _logger
    _logger = SMALoggerAdapter(context)

def get_logger(name: str = "gpu_analysis") -> SMALoggerAdapter:
    """
    Get a logger adapter for the given name.

    Args:
        name: Logger name

    Returns:
        Logger adapter
    """
    return SMALoggerAdapter(_logger.context, name)
```

### 2. Update Plugin to Use Integrated Logging

Update the GPU Analysis Plugin to use the integrated logging:

```python
# In brain/gpu_analysis/gpu_analysis_plugin.py:

from gpu_analysis.logging_integration import get_logger, set_context

# Configure logging
logger = get_logger("gpu_analysis_plugin")

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

            # Initialize components with updated configuration
            self.semantic_analyzer = SemanticAnalyzer(device=self.device, config=self.config)
            self.ast_adapter = ASTAdapter(device=self.device, config=self.config)

            # Log initialization
            logger.info(f"GPU Analysis Plugin initialized with device: {self.device}")
            if self.device == "cuda":
                logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
                logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

        except Exception as e:
            logger.error(f"Error initializing GPU Analysis Plugin: {e}")
            raise
```

### 3. Add Logging for Unimplemented SMA Methods

Add logging for the implementation of unimplemented SMA methods:

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
    logger.info(f"Analyzing component {component} in file {file_path}")

    try:
        # Parse the file
        logger.debug(f"Parsing file {file_path}")
        parsed = self.language_parser.parse_file(file_path)

        # Extract the component
        logger.debug(f"Extracting component {component}")
        component_node = self._extract_component(parsed, component)

        # Analyze the component
        logger.debug(f"Analyzing component {component}")
        results = self.semantic_analyzer.analyze(component_node)

        logger.info(f"Component analysis completed for {component}")
        return results

    except Exception as e:
        logger.error(f"Error analyzing component {component}: {e}")
        raise

# In the implementation of handle_analyze_command:

def handle_analyze_command(args: argparse.Namespace) -> None:
    """
    Handle the analyze command.

    This method implements the unimplemented SMA CLI command for analyzing code.

    Args:
        args: Command arguments
    """
    logger.info("Handling analyze command")

    try:
        # Extract arguments
        file_path = args.file_path if hasattr(args, 'file_path') else None
        code = args.code if hasattr(args, 'code') else None

        # Print header
        print_header("CODE ANALYSIS")
        print("Analyzing code for intent extraction using GPU acceleration...")

        # Analyze code or file
        if file_path:
            logger.debug(f"Analyzing file {file_path}")
            results = plugin.analyze_file(file_path)
        elif code:
            logger.debug("Analyzing code snippet")
            results = plugin.analyze_code(code)
        else:
            logger.warning("No code or file specified")
            print("No code or file specified")
            return

        # Print results
        print(f"Analysis completed using GPU acceleration")
        print(f"Complexity: {results.get('complexity', {})}")
        print(f"Patterns: {len(results.get('pattern_matches', []))}")

        logger.info("Analyze command completed")

    except Exception as e:
        logger.error(f"Error handling analyze command: {e}")
        print(f"Error: {e}")
```

### 4. Update Other Components to Use Integrated Logging

Update other components to use the integrated logging:

```python
# In brain/gpu_analysis/ast_adapter.py:

from gpu_analysis.logging_integration import get_logger

# Configure logging
logger = get_logger("ast_adapter")

class ASTAdapter:
    """
    Adapter between SMA's AST representation and GPU-friendly format.
    """

    # ... methods ...

    def convert_to_gpu_format(self, ast_node: Any) -> Dict[str, torch.Tensor]:
        """
        Convert SMA's AST representation to GPU-friendly format.

        Args:
            ast_node: AST node from SMA's parser

        Returns:
            Dictionary of tensors representing the AST
        """
        try:
            logger.debug("Converting AST to GPU format")
            return self.tensorizer.tensorize(ast_node)
        except Exception as e:
            logger.error(f"Error converting AST to GPU format: {e}")
            raise
```

## Implementation Focus

The implementation should focus on:

1. **Logging Integration**: Implementing integration with SMA's logging system.

2. **Component Logging**: Ensuring all components use the integrated logging system.

3. **Method Logging**: Adding appropriate logging to the implementation of unimplemented SMA methods.

4. **Consistency**: Maintaining consistent logging throughout the application.

5. **Information Quality**: Ensuring logs provide useful information for debugging and monitoring.

## Success Criteria

1. The GPU Analysis Plugin's logging is correctly integrated with SMA's logging system.

2. All components of the GPU Analysis Plugin use the integrated logging system.

3. The implementation of unimplemented SMA methods includes appropriate logging.

4. Logging is consistent throughout the application.

Note: Testing will be deferred until the complete architecture is implemented and stable. The focus is on velocity and architectural completion rather than incremental validation.

## References

1. SMA Logging System: `semantic_matrix_analyzer/semantic_matrix_analyzer/logging/__init__.py`

2. SMA Plugin Context: `semantic_matrix_analyzer/semantic_matrix_analyzer/plugins/__init__.py`

3. Python Logging Module: https://docs.python.org/3/library/logging.html
