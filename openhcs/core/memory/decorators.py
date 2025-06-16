"""
Memory type declaration decorators for OpenHCS.

This module provides decorators for explicitly declaring the memory interface
of pure functions, enforcing Clause 106-A (Declared Memory Types) and supporting
memory-type-aware dispatching and orchestration.

These decorators annotate functions with input_memory_type and output_memory_type
attributes without performing any runtime logic or type checking. They are used
by FuncStep and dispatchers to derive memory type compatibility.
"""

import logging
from typing import Any, Callable, Optional, TypeVar

from openhcs.constants.constants import VALID_MEMORY_TYPES

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


def memory_types(*, input_type: str, output_type: str) -> Callable[[F], F]:
    """
    Decorator that explicitly declares the memory types for a function's input and output.

    This decorator enforces Clause 106-A (Declared Memory Types) by requiring explicit
    memory type declarations for both input and output.

    Args:
        input_type: The memory type for the function's input (e.g., "numpy", "cupy")
        output_type: The memory type for the function's output (e.g., "numpy", "cupy")

    Returns:
        A decorator function that sets the memory type attributes

    Raises:
        ValueError: If input_type or output_type is not a supported memory type
    """
    # 🔒 Clause 88 — No Inferred Capabilities
    # Validate memory types at decoration time, not runtime
    if not input_type:
        raise ValueError(
            "Clause 106-A Violation: input_type must be explicitly declared. "
            "No default or inferred memory types are allowed."
        )

    if not output_type:
        raise ValueError(
            "Clause 106-A Violation: output_type must be explicitly declared. "
            "No default or inferred memory types are allowed."
        )

    # Validate that memory types are supported
    if input_type not in VALID_MEMORY_TYPES:
        raise ValueError(
            f"Clause 106-A Violation: input_type '{input_type}' is not supported. "
            f"Supported types are: {', '.join(sorted(VALID_MEMORY_TYPES))}"
        )

    if output_type not in VALID_MEMORY_TYPES:
        raise ValueError(
            f"Clause 106-A Violation: output_type '{output_type}' is not supported. "
            f"Supported types are: {', '.join(sorted(VALID_MEMORY_TYPES))}"
        )

    def decorator(func: F) -> F:
        """
        Decorator function that sets memory type attributes on the function.

        Args:
            func: The function to decorate

        Returns:
            The decorated function with memory type attributes set

        Raises:
            ValueError: If the function already has different memory type attributes
        """
        # 🔒 Clause 66 — Immutability
        # Check if memory type attributes already exist
        if hasattr(func, 'input_memory_type') and func.input_memory_type != input_type:
            raise ValueError(
                f"Clause 66 Violation: Function '{func.__name__}' already has input_memory_type "
                f"'{func.input_memory_type}', cannot change to '{input_type}'."
            )

        if hasattr(func, 'output_memory_type') and func.output_memory_type != output_type:
            raise ValueError(
                f"Clause 66 Violation: Function '{func.__name__}' already has output_memory_type "
                f"'{func.output_memory_type}', cannot change to '{output_type}'."
            )

        # Set memory type attributes using canonical names
        # 🔒 Clause 106-A.2 — Canonical Memory Type Attributes
        func.input_memory_type = input_type
        func.output_memory_type = output_type

        # Return the function unchanged (no wrapper)
        return func

    return decorator


def numpy(
    func: Optional[F] = None,
    *,
    input_type: str = "numpy",
    output_type: str = "numpy"
) -> Any:
    """
    Decorator that declares a function as operating on numpy arrays.

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


def cupy(func: Optional[F] = None, *, input_type: str = "cupy", output_type: str = "cupy") -> Any:
    """
    Decorator that declares a function as operating on cupy arrays.

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
    input_type: str = "torch",
    output_type: str = "torch"
) -> Any:
    """
    Decorator that declares a function as operating on torch tensors.

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
    input_type: str = "tensorflow",
    output_type: str = "tensorflow"
) -> Any:
    """
    Decorator that declares a function as operating on tensorflow tensors.

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
    input_type: str = "jax",
    output_type: str = "jax"
) -> Any:
    """
    Decorator that declares a function as operating on JAX arrays.

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


def pyclesperanto(
    func: Optional[F] = None,
    *,
    input_type: str = "pyclesperanto",
    output_type: str = "pyclesperanto"
) -> Any:
    """
    Decorator that declares a function as operating on pyclesperanto GPU arrays.

    This is a convenience wrapper around memory_types with pyclesperanto defaults.

    Args:
        func: The function to decorate (optional)
        input_type: The memory type for the function's input (default: "pyclesperanto")
        output_type: The memory type for the function's output (default: "pyclesperanto")

    Returns:
        The decorated function with memory type attributes set

    Raises:
        ValueError: If input_type or output_type is not a supported memory type
    """
    decorator = memory_types(input_type=input_type, output_type=output_type)

    # Handle both @pyclesperanto and @pyclesperanto(input_type=..., output_type=...) forms
    if func is None:
        return decorator

    return decorator(func)
