"""
Image processing module for openhcs.

This module provides image processing functionality for openhcs,
including image normalization, sharpening, and other operations.

It also includes a function registry system that automatically registers
functions decorated with memory type decorators (@numpy, @cupy, etc.) for
runtime discovery and inspection.
"""


# Import function registry components (these don't import GPU libs)
from openhcs.processing.func_registry import (FUNC_REGISTRY,
                                                 get_function_info,
                                                 get_functions_by_memory_type,
                                                 get_function_by_name,
                                                 get_all_function_names,
                                                 get_valid_memory_types,
                                                 is_registry_initialized)
# Import decorators directly from core module (function_registry.py is deprecated)
from openhcs.core.memory import (cupy, jax, numpy,
                                           pyclesperanto, tensorflow, torch)


def __getattr__(name: str):
    """
    Lazy import of backend subpackages via reflection.

    Backend subpackages import GPU libraries (torch, cupy, jax, tensorflow)
    at module level, which takes 8+ seconds. We defer these imports until
    they're actually needed.

    Mathematical property: If name ∈ __all__ ∧ name ∉ globals(), then
    name must be a lazy backend subpackage that exists at the derived path.

    Args:
        name: Attribute name to resolve

    Returns:
        Imported backend module

    Raises:
        AttributeError: If attribute is not in __all__ or import fails
    """
    # Reflection: Check if name is declared in __all__ (our contract)
    if name not in __all__:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    # Mathematical derivation: Construct import path from module structure
    # __name__ = 'openhcs.processing'
    # name = 'enhance'
    # ∴ import_path = 'openhcs.processing.backends.enhance'
    import_path = f"{__name__}.backends.{name}"

    # Import and return the backend module
    import importlib
    return importlib.import_module(import_path)


__all__ = [
    # Image processor components

    # Function registry components
    "numpy", "cupy", "torch", "tensorflow", "jax",
    "FUNC_REGISTRY", "get_functions_by_memory_type", "get_function_info",
    "get_valid_memory_types", "is_registry_initialized",

    # Backend subpackages (lazy loaded via __getattr__)
    "processors",
    "enhance",
    "pos_gen",
    "assemblers",
    "analysis",
    "cellprofiler",
]
