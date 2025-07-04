"""
Image processing module for openhcs.

This module provides image processing functionality for openhcs,
including image normalization, sharpening, and other operations.

It also includes a function registry system that automatically registers
functions decorated with memory type decorators (@numpy, @cupy, etc.) for
runtime discovery and inspection.

Doctrinal Clauses:
- Clause 3 — Declarative Primacy: All functions are pure and stateless
- Clause 88 — No Inferred Capabilities: Explicit backend requirements
- Clause 106-A — Declared Memory Types: All methods specify memory types
- Clause 273 — Memory Backend Restrictions: GPU-only implementations are marked
"""


# Import backend subpackages
from openhcs.processing.backends import (analysis, assemblers, enhance,
                                            pos_gen, processors)
# Import function registry components
from openhcs.processing.func_registry import (FUNC_REGISTRY,
                                                 get_function_info,
                                                 get_functions_by_memory_type,
                                                 get_function_by_name,
                                                 get_all_function_names,
                                                 get_valid_memory_types,
                                                 is_registry_initialized)
# Import decorators directly from core module (function_registry.py is deprecated)
from openhcs.core.memory.decorators import (cupy, jax, numpy,
                                           pyclesperanto, tensorflow, torch)

__all__ = [
    # Image processor components

    # Function registry components
    "numpy", "cupy", "torch", "tensorflow", "jax",
    "FUNC_REGISTRY", "get_functions_by_memory_type", "get_function_info",
    "get_valid_memory_types", "is_registry_initialized",

    # Backend subpackages
    "processors",
    "enhance",
    "pos_gen",
    "assemblers",
    "analysis",
]
