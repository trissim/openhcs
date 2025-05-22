"""
Error Handling Module for GPU Analysis Plugin.

This module provides integration with SMA's error handling system, ensuring
consistent error reporting and graceful failure handling, especially for
the implementation of unimplemented SMA methods.
"""

from typing import Optional, Any, Dict, Type
import traceback
import torch

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


def with_error_handling(func):
    """
    Decorator for functions that need error handling.

    This decorator wraps a function with error handling, converting
    exceptions to GPU Analysis Plugin errors.

    Args:
        func: Function to wrap

    Returns:
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except GPUAnalysisError:
            # Re-raise GPU Analysis Plugin errors
            raise
        except Exception as e:
            # Convert other errors to GPU Analysis Plugin errors
            error = handle_error(e)
            raise error
    return wrapper
