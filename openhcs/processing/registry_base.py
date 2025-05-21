"""
Base registry module for function registration.

This module provides the base registry functionality without any imports
that could cause circular dependencies.
"""

import logging
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)

# Global registry of functions by backend type
FUNC_REGISTRY: Dict[str, List[Callable]] = {}

def initialize_registry(memory_types: List[str]) -> None:
    """
    Initialize the function registry with empty lists for each memory type.

    Args:
        memory_types: List of memory type strings
    """
    # Access the global registry directly without global statement
    # since we're only modifying its contents, not reassigning it
    FUNC_REGISTRY.clear()
    for memory_type in memory_types:
        FUNC_REGISTRY[memory_type] = []

def register_function(func: Callable, backend: str) -> None:
    """
    Register a function in the appropriate backend registry.

    Args:
        func: The function to register
        backend: The backend type (e.g., "numpy", "cupy")

    Raises:
        ValueError: If the backend is not supported
    """

    if backend not in FUNC_REGISTRY:
        raise ValueError(
            f"Cannot register function for unsupported backend '{backend}'. "
            f"Supported backends are: {', '.join(sorted(FUNC_REGISTRY.keys()))}"
        )

    # Skip if function is already registered
    if func in FUNC_REGISTRY[backend]:
        logger.debug("Function '%s' already registered for backend '%s'", func.__name__, backend)
        return

    # Add function to registry
    FUNC_REGISTRY[backend].append(func)

    # Add backend attribute for easier inspection
    setattr(func, "backend", backend)

def get_functions_by_backend(backend: str) -> List[Callable]:
    """
    Get all functions registered for a specific backend.

    Args:
        backend: The backend type (e.g., "numpy", "cupy")

    Returns:
        A list of functions registered for the specified backend

    Raises:
        ValueError: If the backend is not supported
    """
    if backend not in FUNC_REGISTRY:
        raise ValueError(
            f"Unsupported backend: {backend}. "
            f"Supported backends are: {', '.join(sorted(FUNC_REGISTRY.keys()))}"
        )
    return FUNC_REGISTRY[backend]

def get_function_info(func: Callable) -> Dict[str, Any]:
    """
    Get information about a registered function.

    Args:
        func: The function to get information about

    Returns:
        A dictionary containing information about the function

    Raises:
        ValueError: If the function is not decorated with a memory type
    """
    if not hasattr(func, 'input_memory_type'):
        raise ValueError(f"Function {func.__name__} is not decorated with a memory type")

    return {
        'name': func.__name__,
        'backend': getattr(func, "backend", func.input_memory_type),
        'input_memory_type': func.input_memory_type,
        'output_memory_type': func.output_memory_type,
        'doc': func.__doc__,
        'module': func.__module__
    }
