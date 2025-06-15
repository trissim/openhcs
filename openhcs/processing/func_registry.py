"""
Function registry for processing backends.

This module provides a registry for functions that can be executed by different
processing backends (numpy, cupy, torch, etc.). It automatically scans the
processing directory to register functions with matching input and output
memory types.

The function registry is a global singleton that is initialized during application
startup and shared across all components.

Valid memory types:
- numpy
- cupy
- torch
- tensorflow
- jax

Thread Safety:
    All functions in this module are thread-safe and use a lock to ensure
    consistent access to the global registry.
"""
from __future__ import annotations 

import importlib
import inspect
import logging
import os
import pkgutil
import sys
import threading
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Thread-safe lock for registry access
_registry_lock = threading.Lock()

# Global registry of functions by backend type
# Structure: {backend_name: [function1, function2, ...]}
FUNC_REGISTRY: Dict[str, List[Callable]] = {}

# Valid memory types
VALID_MEMORY_TYPES = {"numpy", "cupy", "torch", "tensorflow", "jax"}

# Flag to track if the registry has been initialized
_registry_initialized = False


def _auto_initialize_registry() -> None:
    """
    Auto-initialize the function registry on module import.

    This follows the same pattern as storage_registry in openhcs.io.base.
    """
    global _registry_initialized

    if _registry_initialized:
        return

    try:
        # Clear and initialize the registry with valid memory types
        FUNC_REGISTRY.clear()
        for memory_type in VALID_MEMORY_TYPES:
            FUNC_REGISTRY[memory_type] = []

        # Scan processing directory and register functions
        _scan_and_register_functions()

        logger.info(
            "Function registry auto-initialized with %d functions across %d backends",
            sum(len(funcs) for funcs in FUNC_REGISTRY.values()),
            len(VALID_MEMORY_TYPES)
        )

        # Mark registry as initialized
        _registry_initialized = True

    except Exception as e:
        logger.error(f"Failed to auto-initialize function registry: {e}")
        # Initialize empty registry as fallback
        FUNC_REGISTRY.clear()
        for memory_type in VALID_MEMORY_TYPES:
            FUNC_REGISTRY[memory_type] = []
        _registry_initialized = True


def initialize_registry() -> None:
    """
    Initialize the function registry and scan for functions to register.

    This function is now optional since the registry auto-initializes on import.
    It can be called to force re-initialization if needed.

    Thread-safe: Uses a lock to ensure consistent access to the global registry.

    Raises:
        RuntimeError: If the registry is already initialized and force=False
    """
    with _registry_lock:
        global _registry_initialized

        # Check if registry is already initialized
        if _registry_initialized:
            logger.info("Function registry already initialized, skipping manual initialization")
            return
        
        # Clear and initialize the registry with valid memory types
        FUNC_REGISTRY.clear()
        for memory_type in VALID_MEMORY_TYPES:
            FUNC_REGISTRY[memory_type] = []
        
        # Scan processing directory and register functions
        _scan_and_register_functions()
        
        logger.info(
            "Function registry initialized with %d functions across %d backends",
            sum(len(funcs) for funcs in FUNC_REGISTRY.values()),
            len(VALID_MEMORY_TYPES)
        )
        
        # Mark registry as initialized
        _registry_initialized = True


def _scan_and_register_functions() -> None:
    """
    Scan the processing directory for functions with matching input/output memory types.
    
    This function recursively imports all modules in the processing directory
    and registers functions that have matching input_memory_type and output_memory_type
    attributes that are in the set of valid memory types.
    
    This is an internal function called during initialization.
    """
    from openhcs import processing
    
    processing_path = os.path.dirname(processing.__file__)
    processing_package = "openhcs.processing"
    
    logger.info("Scanning for functions in %s", processing_path)
    
    # Walk through all modules in the processing package
    for _, module_name, is_pkg in pkgutil.walk_packages([processing_path], f"{processing_package}."):
        try:
            # Import the module
            logger.debug(f"Scanning module: {module_name}")
            module = importlib.import_module(module_name)

            # Skip packages (we'll process their modules separately)
            if is_pkg:
                logger.debug(f"Skipping package: {module_name}")
                continue

            # Find all functions in the module
            function_count = 0
            for name, obj in inspect.getmembers(module, inspect.isfunction):
                # Check if the function has the required attributes
                if hasattr(obj, "input_memory_type") and hasattr(obj, "output_memory_type"):
                    input_type = getattr(obj, "input_memory_type")
                    output_type = getattr(obj, "output_memory_type")

                    # Register if input and output types match and are valid
                    if input_type == output_type and input_type in VALID_MEMORY_TYPES:
                        _register_function(obj, input_type)
                        function_count += 1

            logger.debug(f"Module {module_name}: found {function_count} registerable functions")
        except Exception as e:
            logger.warning("Error importing module %s: %s", module_name, e)


def _register_function(func: Callable, memory_type: str) -> None:
    """
    Register a function for a specific memory type.
    
    This is an internal function used during automatic scanning.
    
    Args:
        func: The function to register
        memory_type: The memory type (e.g., "numpy", "cupy", "torch")
    """
    # Skip if function is already registered
    if func in FUNC_REGISTRY[memory_type]:
        logger.debug(
            "Function '%s' already registered for memory type '%s'",
            func.__name__, memory_type
        )
        return
    
    # Add function to registry
    FUNC_REGISTRY[memory_type].append(func)
    
    # Add memory_type attribute for easier inspection
    setattr(func, "backend", memory_type)
    
    logger.debug(
        "Registered function '%s' for memory type '%s'",
        func.__name__, memory_type
    )


def get_functions_by_memory_type(memory_type: str) -> List[Callable]:
    """
    Get all functions registered for a specific memory type.
    
    Thread-safe: Uses a lock to ensure consistent access to the global registry.
    
    Args:
        memory_type: The memory type (e.g., "numpy", "cupy", "torch")
        
    Returns:
        A list of functions registered for the specified memory type
        
    Raises:
        RuntimeError: If the registry is not initialized
        ValueError: If the memory type is not valid
    """
    with _registry_lock:
        # Check if registry is initialized (should be auto-initialized on import)
        if not _registry_initialized:
            logger.warning("Function registry not initialized, auto-initializing now")
            _auto_initialize_registry()
        
        # Check if memory type is valid
        if memory_type not in VALID_MEMORY_TYPES:
            raise ValueError(
                f"Invalid memory type: {memory_type}. "
                f"Valid types are: {', '.join(sorted(VALID_MEMORY_TYPES))}"
            )
        
        # Return a copy of the list to prevent external modification
        return list(FUNC_REGISTRY[memory_type])


def get_function_info(func: Callable) -> Dict[str, Any]:
    """
    Get information about a registered function.
    
    Args:
        func: The function to get information about
        
    Returns:
        A dictionary containing information about the function
        
    Raises:
        ValueError: If the function does not have memory type attributes
    """
    if not hasattr(func, "input_memory_type") or not hasattr(func, "output_memory_type"):
        raise ValueError(
            f"Function '{func.__name__}' does not have memory type attributes"
        )
    
    return {
        "name": func.__name__,
        "input_memory_type": func.input_memory_type,
        "output_memory_type": func.output_memory_type,
        "backend": getattr(func, "backend", func.input_memory_type),
        "doc": func.__doc__,
        "module": func.__module__
    }


def is_registry_initialized() -> bool:
    """
    Check if the function registry has been initialized.
    
    Thread-safe: Uses a lock to ensure consistent access to the initialization flag.
    
    Returns:
        True if the registry is initialized, False otherwise
    """
    with _registry_lock:
        return _registry_initialized


def get_valid_memory_types() -> Set[str]:
    """
    Get the set of valid memory types.

    Returns:
        A set of valid memory type names
    """
    return VALID_MEMORY_TYPES.copy()


def get_function_by_name(function_name: str, memory_type: str) -> Optional[Callable]:
    """
    Get a specific function by name and memory type from the registry.

    Args:
        function_name: Name of the function to find
        memory_type: The memory type (e.g., "numpy", "cupy", "torch")

    Returns:
        The function if found, None otherwise

    Raises:
        RuntimeError: If the registry is not initialized
        ValueError: If the memory type is not valid
    """
    functions = get_functions_by_memory_type(memory_type)

    for func in functions:
        if func.__name__ == function_name:
            return func

    return None


def get_all_function_names(memory_type: str) -> List[str]:
    """
    Get all function names registered for a specific memory type.

    Args:
        memory_type: The memory type (e.g., "numpy", "cupy", "torch")

    Returns:
        A list of function names

    Raises:
        RuntimeError: If the registry is not initialized
        ValueError: If the memory type is not valid
    """
    functions = get_functions_by_memory_type(memory_type)
    return [func.__name__ for func in functions]


# Auto-initialize the registry on module import (following storage_registry pattern)
_auto_initialize_registry()
