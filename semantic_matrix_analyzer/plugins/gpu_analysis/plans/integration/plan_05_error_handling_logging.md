# Plan 05: Error Handling and Logging Integration

## Objective

Refactor the GPU Analysis Plugin's error handling and logging to integrate with SMA's error handling and logging systems, ensuring consistent error reporting and logging throughout the application.

## Background

SMA has its own error handling and logging systems for reporting and tracking issues. The GPU Analysis Plugin needs to integrate with these systems to ensure consistent error reporting and logging throughout the application.

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

### 2. Create an Error Handling Module

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
    if context:
        from gpu_analysis.logging_integration import set_context
        set_context(context)
    
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
    if not torch.cuda.is_available():
        raise GPUNotAvailableError("CUDA is not available")
```

### 3. Update Plugin to Use Integrated Logging and Error Handling

Update the GPU Analysis Plugin to use the integrated logging and error handling:

```python
# In brain/gpu_analysis/gpu_analysis_plugin.py:

from gpu_analysis.logging_integration import get_logger, set_context
from gpu_analysis.error_handling import handle_error, GPUAnalysisError, check_gpu_available

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

### 4. Update Other Components to Use Integrated Logging and Error Handling

Update other components to use the integrated logging and error handling:

```python
# In brain/gpu_analysis/ast_adapter.py:

from gpu_analysis.logging_integration import get_logger
from gpu_analysis.error_handling import handle_error, GPUAnalysisError

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
            # Handle conversion error
            error = handle_error(e)
            logger.error(f"Error converting AST to GPU format: {error}")
            raise error
```

### 5. Create Error Documentation

Create documentation for error handling:

```python
# Create a new file: brain/gpu_analysis/docs/error_handling.md

# GPU Analysis Plugin Error Handling

This document describes the error handling system for the GPU Analysis Plugin.

## Error Types

The GPU Analysis Plugin defines several error types:

- `GPUAnalysisError`: Base class for all GPU Analysis Plugin errors.
- `GPUNotAvailableError`: Error raised when GPU is not available.
- `GPUMemoryError`: Error raised when GPU memory is insufficient.
- `GPUAnalysisConfigError`: Error raised when configuration is invalid.
- `GPUAnalysisRuntimeError`: Error raised when a runtime error occurs.

## Error Handling

The GPU Analysis Plugin handles errors in a consistent way:

1. All errors are logged with appropriate severity.
2. Errors are converted to GPU Analysis Plugin errors with appropriate types.
3. Original errors are preserved as causes for debugging.
4. Errors are propagated to SMA for consistent handling.

## Error Messages

Error messages follow a consistent format:

- `GPUNotAvailableError`: "CUDA is not available"
- `GPUMemoryError`: "CUDA error: [error message]"
- `GPUAnalysisConfigError`: "Invalid value: [error message]"
- `GPUAnalysisRuntimeError`: "Error: [error message]"

## Handling GPU Unavailability

The GPU Analysis Plugin gracefully handles GPU unavailability:

1. If CUDA is not available at initialization, the plugin falls back to CPU.
2. If CUDA becomes unavailable during operation, the plugin raises a `GPUNotAvailableError`.
3. If GPU memory is insufficient, the plugin raises a `GPUMemoryError`.

## Integration with SMA

The GPU Analysis Plugin integrates with SMA's error handling system:

1. Errors are logged through SMA's logging system.
2. Errors are propagated to SMA for consistent handling.
3. Error messages include context for debugging.
```

## Testing Strategy

1. **Logging Integration**: Test that the GPU Analysis Plugin's logging is correctly integrated with SMA's logging system.

2. **Error Handling**: Test that the GPU Analysis Plugin's error handling is correctly integrated with SMA's error handling system.

3. **Error Types**: Test that the GPU Analysis Plugin's error types are correctly defined and used.

4. **Error Propagation**: Test that errors are correctly propagated to SMA for consistent handling.

5. **GPU Unavailability**: Test that the GPU Analysis Plugin gracefully handles GPU unavailability.

## Success Criteria

1. The GPU Analysis Plugin's logging is correctly integrated with SMA's logging system.

2. The GPU Analysis Plugin's error handling is correctly integrated with SMA's error handling system.

3. The GPU Analysis Plugin's error types are correctly defined and used.

4. Errors are correctly propagated to SMA for consistent handling.

5. The GPU Analysis Plugin gracefully handles GPU unavailability.

## References

1. SMA Logging System: `semantic_matrix_analyzer/semantic_matrix_analyzer/logging/__init__.py`

2. SMA Plugin Context: `semantic_matrix_analyzer/semantic_matrix_analyzer/plugins/__init__.py`

3. Python Logging Module: https://docs.python.org/3/library/logging.html
