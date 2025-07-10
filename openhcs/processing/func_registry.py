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

# Import hook system for auto-decorating external libraries
_original_import = __builtins__['__import__']
_decoration_applied = set()
_import_hook_installed = False

# Global registry of functions by backend type
# Structure: {backend_name: [function1, function2, ...]}
FUNC_REGISTRY: Dict[str, List[Callable]] = {}

# Valid memory types
VALID_MEMORY_TYPES = {"numpy", "cupy", "torch", "tensorflow", "jax", "pyclesperanto"}

# Flag to track if the registry has been initialized
_registry_initialized = False

# Flag to track if we're currently in the initialization process (prevent recursion)
_registry_initializing = False


def _install_import_hook() -> None:
    """
    Install import hook to auto-decorate external library functions on import.

    This makes functions self-contained and eliminates the need for registry
    initialization in subprocess environments.
    """
    global _import_hook_installed

    if _import_hook_installed:
        return

    def _decorating_import(name, *args, **kwargs):
        """Import hook that auto-decorates external libraries."""
        module = _original_import(name, *args, **kwargs)

        # Only decorate once per module
        if name not in _decoration_applied:
            try:
                if name == 'pyclesperanto':
                    _decorate_pyclesperanto_on_import(module)
                    _decoration_applied.add(name)
                    logger.debug(f"Auto-decorated pyclesperanto functions on import")
                elif name.startswith('skimage'):
                    _decorate_skimage_on_import(module)
                    _decoration_applied.add(name)
                    logger.debug(f"Auto-decorated skimage functions on import: {name}")
            except Exception as e:
                logger.warning(f"Failed to auto-decorate {name}: {e}")

        return module

    # Install the hook
    __builtins__['__import__'] = _decorating_import
    _import_hook_installed = True
    logger.debug("Import hook installed for auto-decorating external libraries")


def _decorate_pyclesperanto_on_import(cle_module) -> None:
    """Auto-decorate pyclesperanto functions when module is imported."""
    try:
        # Use direct decoration to avoid circular imports
        # This is a simplified version that decorates common functions
        common_functions = [
            'sobel', 'gaussian_blur', 'threshold_otsu', 'erode_sphere', 'dilate_sphere',
            'opening_sphere', 'closing_sphere', 'subtract_images', 'add_images',
            'multiply_images', 'divide_images', 'maximum_images', 'minimum_images',
            'absolute_difference', 'mean_filter', 'median_filter', 'variance_filter',
            'standard_deviation_filter', 'entropy_filter', 'laplacian_filter'
        ]

        decorated_count = 0
        for func_name in common_functions:
            if hasattr(cle_module, func_name):
                func = getattr(cle_module, func_name)
                if callable(func) and not hasattr(func, 'input_memory_type'):
                    func.input_memory_type = "pyclesperanto"
                    func.output_memory_type = "pyclesperanto"
                    decorated_count += 1

        logger.debug(f"Auto-decorated {decorated_count} common pyclesperanto functions")

    except Exception as e:
        logger.warning(f"Failed to auto-decorate pyclesperanto: {e}")


def _decorate_skimage_on_import(skimage_module) -> None:
    """Auto-decorate scikit-image functions when module is imported."""
    try:
        # For now, just mark that we've seen this module
        # Full scikit-image decoration can be added later if needed
        logger.debug(f"Skimage module imported: {skimage_module.__name__}")

    except Exception as e:
        logger.warning(f"Failed to auto-decorate skimage: {e}")


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

        # Phase 1: Scan processing directory and register native OpenHCS functions
        _scan_and_register_functions()

        # Phase 2: Register external library functions
        _register_external_libraries()

        total_functions = sum(len(funcs) for funcs in FUNC_REGISTRY.values())
        logger.info(
            "Function registry auto-initialized with %d functions across %d backends",
            total_functions,
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
        
        # Phase 1: Scan processing directory and register native OpenHCS functions
        _scan_and_register_functions()

        # Phase 2: Register external library functions
        _register_external_libraries()
        
        logger.info(
            "Function registry initialized with %d functions across %d backends",
            sum(len(funcs) for funcs in FUNC_REGISTRY.values()),
            len(VALID_MEMORY_TYPES)
        )
        
        # Mark registry as initialized
        _registry_initialized = True


def load_prebuilt_registry(registry_data: Dict) -> None:
    """
    Load a pre-built function registry from serialized data.

    This allows subprocess workers to skip function discovery by loading
    a registry that was built in the main process.

    Args:
        registry_data: Dictionary containing the pre-built registry
    """
    with _registry_lock:
        global _registry_initialized

        FUNC_REGISTRY.clear()
        FUNC_REGISTRY.update(registry_data)
        _registry_initialized = True

        total_functions = sum(len(funcs) for funcs in FUNC_REGISTRY.values())
        logger.info(f"Loaded pre-built registry with {total_functions} functions")


def _scan_and_register_functions() -> None:
    """
    Scan the processing directory for native OpenHCS functions.

    This function recursively imports all modules in the processing directory
    and registers functions that have matching input_memory_type and output_memory_type
    attributes that are in the set of valid memory types.

    This is Phase 1 of initialization - only native OpenHCS functions.
    External library functions are registered in Phase 2.
    """
    from openhcs import processing

    processing_path = os.path.dirname(processing.__file__)
    processing_package = "openhcs.processing"

    logger.info("Phase 1: Scanning for native OpenHCS functions in %s", processing_path)

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


def _apply_unified_decoration(original_func, func_name, memory_type, create_wrapper=True):
    """
    Unified decoration pattern for all external library functions.

    This applies the same hybrid approach across all registries:
    1. Direct decoration (for subprocess compatibility)
    2. Optional enhanced wrapper (for dtype preservation)
    3. Module replacement (for best user experience)

    Args:
        original_func: The original external library function
        func_name: Function name for wrapper creation
        memory_type: MemoryType enum value (NUMPY, CUPY, PYCLESPERANTO)
        create_wrapper: Whether to create dtype-preserving wrapper

    Returns:
        The function to register (wrapper if created, original if not)
    """
    from openhcs.constants import MemoryType
    import sys

    # Step 1: Direct decoration (for subprocess compatibility)
    original_func.input_memory_type = memory_type.value
    original_func.output_memory_type = memory_type.value

    if not create_wrapper:
        return original_func

    # Step 2: Create enhanced wrapper (for dtype preservation)
    if memory_type == MemoryType.NUMPY:
        from openhcs.processing.backends.analysis.scikit_image_registry import _create_dtype_preserving_wrapper
        wrapper_func = _create_dtype_preserving_wrapper(original_func, func_name)
    elif memory_type == MemoryType.CUPY:
        from openhcs.processing.backends.analysis.cupy_registry import _create_cupy_dtype_preserving_wrapper
        wrapper_func = _create_cupy_dtype_preserving_wrapper(original_func, func_name)
    elif memory_type == MemoryType.PYCLESPERANTO:
        from openhcs.processing.backends.analysis.pyclesperanto_registry import _create_pyclesperanto_array_compliant_wrapper
        wrapper_func = _create_pyclesperanto_array_compliant_wrapper(original_func, func_name)
    else:
        # For other GPU libraries, use simpler wrapper (they generally preserve dtypes better)
        wrapper_func = original_func

    wrapper_func.input_memory_type = memory_type.value
    wrapper_func.output_memory_type = memory_type.value

    # Step 3: Module replacement (for best user experience)
    module_name = original_func.__module__
    if module_name in sys.modules:
        target_module = sys.modules[module_name]
        if hasattr(target_module, func_name):
            setattr(target_module, func_name, wrapper_func)
            logger.debug(f"Replaced {module_name}.{func_name} with enhanced function")

    return wrapper_func


def _register_external_libraries() -> None:
    """
    Phase 2: Register external library functions (pyclesperanto, scikit-image).

    This is separate from core scanning to avoid circular dependencies.
    External library registration should use direct registration, not trigger re-initialization.
    """
    logger.info("Phase 2: Registering external library functions...")

    try:
        from openhcs.processing.backends.analysis.pyclesperanto_registry import _register_pycle_ops_direct
        _register_pycle_ops_direct()
        logger.info("Successfully registered pyclesperanto functions")
    except ImportError as e:
        logger.warning(f"Could not register pyclesperanto functions: {e}")
    except Exception as e:
        logger.error(f"Error registering pyclesperanto functions: {e}")

    try:
        from openhcs.processing.backends.analysis.scikit_image_registry import _register_skimage_ops_direct
        _register_skimage_ops_direct()
        logger.info("Successfully registered scikit-image functions")
    except ImportError as e:
        logger.warning(f"Could not register scikit-image functions: {e}")
    except Exception as e:
        logger.error(f"Error registering scikit-image functions: {e}")

    try:
        from openhcs.processing.backends.analysis.cupy_registry import _register_cupy_ops_direct
        _register_cupy_ops_direct()
        logger.info("Successfully registered CuPy ndimage functions")
    except ImportError as e:
        logger.warning(f"Could not register CuPy functions: {e}")
    except Exception as e:
        logger.error(f"Error registering CuPy functions: {e}")


def register_function(func: Callable, backend: str = None, **kwargs) -> None:
    """
    Manually register a function with the function registry.

    This is the public API for registering functions that are not auto-discovered
    by the module scanner (e.g., dynamically decorated functions).

    Args:
        func: The function to register (must have input_memory_type and output_memory_type attributes)
        backend: Optional backend name (defaults to func.input_memory_type)
        **kwargs: Additional metadata (ignored for compatibility)

    Raises:
        ValueError: If function doesn't have required memory type attributes
        ValueError: If memory types are invalid
    """
    with _registry_lock:
        # Ensure registry is initialized
        if not _registry_initialized:
            _auto_initialize_registry()

        # Validate function has required attributes
        if not hasattr(func, "input_memory_type") or not hasattr(func, "output_memory_type"):
            raise ValueError(
                f"Function '{func.__name__}' must have input_memory_type and output_memory_type attributes"
            )

        input_type = func.input_memory_type
        output_type = func.output_memory_type

        # Validate memory types
        if input_type not in VALID_MEMORY_TYPES:
            raise ValueError(f"Invalid input memory type: {input_type}")
        if output_type not in VALID_MEMORY_TYPES:
            raise ValueError(f"Invalid output memory type: {output_type}")

        # Use input_memory_type as backend if not specified
        memory_type = backend or input_type
        if memory_type not in VALID_MEMORY_TYPES:
            raise ValueError(f"Invalid backend memory type: {memory_type}")

        # Register the function
        _register_function(func, memory_type)


def _register_function(func: Callable, memory_type: str) -> None:
    """
    Register a function for a specific memory type.

    This is an internal function used during automatic scanning and manual registration.

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


# Install import hook when this module is imported
# This ensures external library functions are auto-decorated everywhere
_install_import_hook()


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
# Skip initialization in subprocess workers for faster startup
import os
if not os.environ.get('OPENHCS_SUBPROCESS_MODE'):
    _auto_initialize_registry()
