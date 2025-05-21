"""
Function registry with automatic backend registration for OpenHCS.

This module provides decorators that both declare memory types and automatically register
functions in a central registry for runtime discovery. It extends the core memory decorators
from openhcs.core.memory.decorators with registration functionality.

The registry is organized by backend type, allowing the system to discover
all functions available for a particular backend.

Doctrinal Clauses:
- Clause 3 — Declarative Primacy: All functions are pure and stateless
- Clause 88 — No Inferred Capabilities: Explicit backend requirements
- Clause 106-A — Declared Memory Types: All methods specify memory types
- Clause 247 — All registered functions must be pure Python callables
- Clause 273 — Backend must be declared, not inferred
- Clause 304 — Registry must not mutate function behavior
"""

import logging
from typing import Any, Callable, Optional, TypeVar

from openhcs.constants.constants import MemoryType
# Import core memory decorators
from openhcs.core.memory.decorators import memory_types as core_memory_types
# Import base registry functionality
from openhcs.processing.func_registry import (_register_function,
                                                 get_function_info,
                                                 get_functions_by_memory_type,
                                                 initialize_registry)

# Define memory type constants for convenience
MEMORY_TYPE_NUMPY = MemoryType.NUMPY.value
MEMORY_TYPE_CUPY = MemoryType.CUPY.value
MEMORY_TYPE_TORCH = MemoryType.TORCH.value
MEMORY_TYPE_TENSORFLOW = MemoryType.TENSORFLOW.value
MEMORY_TYPE_JAX = MemoryType.JAX.value

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])

# Initialize the registry with valid memory types
initialize_registry()


def memory_types(*, input_type: str, output_type: str) -> Callable[[F], F]:
    """
    Decorator that explicitly declares memory types and registers the function.

    This decorator extends the core memory_types decorator by adding registration
    functionality. It enforces Clause 106-A (Declared Memory Types) by requiring explicit
    memory type declarations for both input and output, and registers the function
    in the appropriate backend registry.

    Args:
        input_type: The memory type for the function's input (e.g., "numpy", "cupy")
        output_type: The memory type for the function's output (e.g., "numpy", "cupy")

    Returns:
        A decorator function that sets memory type attributes and registers the function

    Raises:
        ValueError: If input_type or output_type is not a supported memory type
    """
    # Get the core decorator
    core_decorator = core_memory_types(input_type=input_type, output_type=output_type)

    def registry_decorator(func: F) -> F:
        """
        Decorator function that applies the core decorator and registers the function.

        Args:
            func: The function to decorate

        Returns:
            The decorated function with memory type attributes set and registered

        Raises:
            ValueError: If the function already has different memory type attributes
        """
        # Apply the core decorator first to set memory type attributes
        decorated_func = core_decorator(func)

        # Register the function based on input_type
        # 🔒 Clause 304: Registry must not mutate function behavior
        _register_function(decorated_func, input_type)

        # Return the decorated function
        return decorated_func

    return registry_decorator


def numpy(
    func: Optional[F] = None,
    *,
    input_type: str = MEMORY_TYPE_NUMPY,
    output_type: str = MEMORY_TYPE_NUMPY
) -> Any:
    """
    Decorator that declares a function as operating on numpy arrays and registers it.

    This is a convenience wrapper around memory_types with numpy defaults.

    Args:
        func: The function to decorate (optional)
        input_type: The memory type for the function's input (default: "numpy")
        output_type: The memory type for the function's output (default: "numpy")

    Returns:
        The decorated function with memory type attributes set

    Raises:
        ValueError: If input_type or output_type is not a supported memory type
    """
    decorator = memory_types(input_type=input_type, output_type=output_type)

    # Handle both @numpy and @numpy(input_type=..., output_type=...) forms
    if func is None:
        return decorator

    return decorator(func)


def cupy(
    func: Optional[F] = None,
    *,
    input_type: str = MEMORY_TYPE_CUPY,
    output_type: str = MEMORY_TYPE_CUPY
) -> Any:
    """
    Decorator that declares a function as operating on cupy arrays and registers it.

    This is a convenience wrapper around memory_types with cupy defaults.

    Args:
        func: The function to decorate (optional)
        input_type: The memory type for the function's input (default: "cupy")
        output_type: The memory type for the function's output (default: "cupy")

    Returns:
        The decorated function with memory type attributes set

    Raises:
        ValueError: If input_type or output_type is not a supported memory type
    """
    decorator = memory_types(input_type=input_type, output_type=output_type)

    # Handle both @cupy and @cupy(input_type=..., output_type=...) forms
    if func is None:
        return decorator

    return decorator(func)


def torch(
    func: Optional[F] = None,
    *,
    input_type: str = MEMORY_TYPE_TORCH,
    output_type: str = MEMORY_TYPE_TORCH
) -> Any:
    """
    Decorator that declares a function as operating on torch tensors and registers it.

    This is a convenience wrapper around memory_types with torch defaults.

    Args:
        func: The function to decorate (optional)
        input_type: The memory type for the function's input (default: "torch")
        output_type: The memory type for the function's output (default: "torch")

    Returns:
        The decorated function with memory type attributes set

    Raises:
        ValueError: If input_type or output_type is not a supported memory type
    """
    decorator = memory_types(input_type=input_type, output_type=output_type)

    # Handle both @torch and @torch(input_type=..., output_type=...) forms
    if func is None:
        return decorator

    return decorator(func)


def tensorflow(
    func: Optional[F] = None,
    *,
    input_type: str = MEMORY_TYPE_TENSORFLOW,
    output_type: str = MEMORY_TYPE_TENSORFLOW
) -> Any:
    """
    Decorator that declares a function as operating on tensorflow tensors and registers it.

    This is a convenience wrapper around memory_types with tensorflow defaults.

    Args:
        func: The function to decorate (optional)
        input_type: The memory type for the function's input (default: "tensorflow")
        output_type: The memory type for the function's output (default: "tensorflow")

    Returns:
        The decorated function with memory type attributes set

    Raises:
        ValueError: If input_type or output_type is not a supported memory type
    """
    decorator = memory_types(input_type=input_type, output_type=output_type)

    # Handle both @tensorflow and @tensorflow(input_type=..., output_type=...) forms
    if func is None:
        return decorator

    return decorator(func)


def jax(
    func: Optional[F] = None,
    *,
    input_type: str = MEMORY_TYPE_JAX,
    output_type: str = MEMORY_TYPE_JAX
) -> Any:
    """
    Decorator that declares a function as operating on JAX arrays and registers it.

    This is a convenience wrapper around memory_types with jax defaults.

    Args:
        func: The function to decorate (optional)
        input_type: The memory type for the function's input (default: "jax")
        output_type: The memory type for the function's output (default: "jax")

    Returns:
        The decorated function with memory type attributes set

    Raises:
        ValueError: If input_type or output_type is not a supported memory type
    """
    decorator = memory_types(input_type=input_type, output_type=output_type)

    # Handle both @jax and @jax(input_type=..., output_type=...) forms
    if func is None:
        return decorator

    return decorator(func)


# These functions are now imported from func_registry.py


# Example usage:
if __name__ == "__main__":
    from openhcs.processing.func_registry import FUNC_REGISTRY

    @torch
    def test_func(x):
        """Example function for testing the registry."""
        return x * 2

    assert test_func in FUNC_REGISTRY[MEMORY_TYPE_TORCH]
    assert hasattr(test_func, "backend")
    assert getattr(test_func, "backend") == MEMORY_TYPE_TORCH

    info = get_function_info(test_func)
    print(f"Function: {info['name']}")
    print(f"Backend: {info['backend']}")
    print(f"Documentation: {info['doc']}")
